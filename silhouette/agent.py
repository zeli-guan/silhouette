"""Bounded generation loop. The caller owns the local executor and final result."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Any, Dict, Optional, Union

from . import entity_dp_publisher as edp

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "agent.json"
T_MAX = 6


def file_sha256(path: Union[Path, str]) -> str:
    """Return raw-byte SHA-256 hex digest of a file."""
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_profile_sha256(profile: Any) -> str:
    """Return canonical key-sorted SHA-256 hex digest of a frozen DP profile."""
    payload = json.dumps(profile, ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require_hex_digest(digest: Optional[str], label: str) -> str:
    if not isinstance(digest, str):
        raise ValueError(f"{label} must be a non-empty hex string")
    norm = digest.strip().lower()
    if not norm or not all(c in "0123456789abcdef" for c in norm):
        raise ValueError(f"{label} must be a non-empty hex string")
    return norm


def verify_profile_sha256(profile_or_path: Any, expected_sha256: str) -> str:
    """Verify a profile object or JSON file path against `expected_sha256`.

    Accepts canonical key-sorted JSON SHA-256, compact JSON SHA-256, or (when given a path)
    exact raw file bytes SHA-256. Returns the verified canonical SHA-256 digest.

    Note: SHA-256 verification pins artifact byte integrity against accidental drift;
    single-release receiver isolation (A9), dictionary independence (A1), and
    non-answer-conditioned querying (A4) remain operational deployment invariants.
    """
    expected = _require_hex_digest(expected_sha256, "expected_profile_sha256")
    raw_sha: Optional[str] = None
    if isinstance(profile_or_path, (str, Path)):
        raw_bytes = Path(profile_or_path).read_bytes()
        raw_sha = hashlib.sha256(raw_bytes).hexdigest()
        profile = json.loads(raw_bytes.decode("utf-8"))
    else:
        profile = profile_or_path
    canon_sorted = canonical_profile_sha256(profile)
    compact = json.dumps(profile, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    compact_sha = hashlib.sha256(compact.encode("utf-8")).hexdigest()
    compact_nl_sha = hashlib.sha256((compact + "\n").encode("utf-8")).hexdigest()
    valid = {canon_sorted, compact_sha, compact_nl_sha}
    if raw_sha is not None:
        valid.add(raw_sha)
    if expected not in valid:
        raise ValueError(
            f"Profile SHA-256 digest mismatch: expected={expected}, canonical={canon_sorted}"
            + (f", file={raw_sha}" if raw_sha is not None else "")
        )
    return canon_sorted


def _assert_dp_deployment_allowed(
    profile_obj: Any,
    *,
    path_str: Optional[str] = None,
    actual_canon_sha: Optional[str] = None,
    actual_file_sha: Optional[str] = None,
    allow_non_dp: bool = False,
) -> None:
    if allow_non_dp:
        return
    prohibited = isinstance(profile_obj, dict) and bool(profile_obj.get("deployment_prohibited"))
    if not prohibited and path_str is not None:
        norm_path = Path(path_str).as_posix().lower()
        if "noiseless" in norm_path or "mimic_demo94_patient_bins_deid" in norm_path:
            prohibited = True
    if not prohibited:
        manifests_dir = Path(__file__).resolve().parents[1] / "manifests"
        cov_path = manifests_dir / "custody_coverage_manifest_v10.json"
        if not cov_path.exists():
            cov_path = manifests_dir / "custody_coverage_manifest_v9.json"
        if cov_path.exists():
            cov = json.loads(cov_path.read_text(encoding="utf-8"))
            prohibited_shas = set(cov.get("prohibited_deployment_sha256s", ()))
            if (actual_canon_sha and actual_canon_sha in prohibited_shas) or (
                actual_file_sha and actual_file_sha in prohibited_shas
            ):
                prohibited = True
    if prohibited:
        raise ValueError(
            f"Refusing deployment_prohibited non-DP ablation profile ({path_str or actual_canon_sha}) "
            "unless allow_non_dp=True"
        )


class SingleReleaseReceiverLedger:
    """In-process (and optionally JSON-persisted) A9 single-release receiver binding ledger.

    Ensures a given `receiver_id` only ever observes queries derived from one canonical DP
    profile release (`canonical_sha256`), permitting unbounded post-processing reuse on that
    same release while failing closed on cross-release multi-arm exposure.
    """

    def __init__(self, ledger_path: Optional[Union[Path, str]] = None):
        self.ledger_path = Path(ledger_path) if ledger_path is not None else None
        self._lock = threading.Lock()
        self._bindings: Dict[str, str] = {}
        if self.ledger_path is not None and self.ledger_path.exists():
            raw = json.loads(self.ledger_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("Receiver ledger file must contain a JSON object")
            for k, v in raw.items():
                if not isinstance(k, str) or not k.strip():
                    raise ValueError("Receiver ledger keys must be non-empty strings")
                self._bindings[k] = _require_hex_digest(v, f"ledger[{k}]")

    def get_binding(self, receiver_id: str) -> Optional[str]:
        """Return the bound canonical SHA-256 digest for `receiver_id`, or None if unbound."""
        with self._lock:
            return self._bindings.get(receiver_id)

    def bind_or_verify(self, receiver_id: str, canonical_sha256: str) -> str:
        """Bind `receiver_id` to `canonical_sha256` or verify it matches the prior A9 binding."""
        if not isinstance(receiver_id, str) or not receiver_id.strip():
            raise ValueError("receiver_id must be a non-empty string")
        norm_sha = _require_hex_digest(canonical_sha256, "canonical_sha256")
        with self._lock:
            existing = self._bindings.get(receiver_id)
            if existing is not None:
                if existing != norm_sha:
                    raise ValueError(
                        f"Refusing cross-release exposure to receiver {receiver_id!r} under A9: "
                        f"bound={existing}, attempted={norm_sha}"
                    )
                return existing
            self._bindings[receiver_id] = norm_sha
            if self.ledger_path is not None:
                self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
                payload = json.dumps(self._bindings, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
                fd, tmp_name = tempfile.mkstemp(
                    prefix=f".{self.ledger_path.name}.",
                    suffix=".tmp",
                    dir=str(self.ledger_path.parent),
                )
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as fh:
                        fh.write(payload)
                    os.replace(tmp_name, self.ledger_path)
                except BaseException:
                    try:
                        os.unlink(tmp_name)
                    except OSError:
                        pass
                    raise
            return norm_sha


def verify_release_bundle(
    profile: Any,
    expected_profile_sha256: Optional[str] = None,
    dictionary_path: Optional[Union[Path, str]] = None,
    expected_dictionary_sha256: Optional[str] = None,
    *,
    expected_canonical_sha256: Optional[str] = None,
    expected_file_sha256: Optional[str] = None,
    allow_non_dp: bool = False,
) -> Dict[str, Any]:
    """Verify a frozen DP profile (and optional public dictionary) against pinned SHA-256 digests.

    SHA-256 verification pins artifact byte integrity against accidental drift, whereas
    single-release receiver isolation (A9), dictionary independence (A1), and
    non-answer-conditioned querying (A4) are operational deployment invariants.
    When `allow_non_dp=False` (default), rejects `deployment_prohibited` noiseless
    ablation profiles.
    """
    if expected_profile_sha256 is None and expected_canonical_sha256 is None and expected_file_sha256 is None:
        raise ValueError("expected_profile_sha256 must be a non-empty hex string")

    path_str: Optional[str] = None
    actual_file_sha: Optional[str] = None
    if isinstance(profile, (str, Path)):
        p = Path(profile)
        path_str = str(p)
        raw_bytes = p.read_bytes()
        actual_file_sha = hashlib.sha256(raw_bytes).hexdigest()
        profile_obj = json.loads(raw_bytes.decode("utf-8"))
    else:
        profile_obj = profile
    actual_canon_sha = canonical_profile_sha256(profile_obj)

    if expected_profile_sha256 is not None:
        verify_profile_sha256(profile, expected_profile_sha256)

    if expected_file_sha256 is not None:
        exp_f = _require_hex_digest(expected_file_sha256, "expected_file_sha256")
        if actual_file_sha is None or actual_file_sha != exp_f:
            raise ValueError(
                f"Release bundle file SHA-256 mismatch for {path_str or '<in-memory>'}: "
                f"expected={exp_f}, actual={actual_file_sha}"
            )
    if expected_canonical_sha256 is not None:
        exp_c = _require_hex_digest(expected_canonical_sha256, "expected_canonical_sha256")
        if actual_canon_sha != exp_c:
            raise ValueError(
                f"Release bundle canonical SHA-256 mismatch for {path_str or '<in-memory>'}: "
                f"expected={exp_c}, actual={actual_canon_sha}"
            )

    _assert_dp_deployment_allowed(
        profile_obj,
        path_str=path_str,
        actual_canon_sha=actual_canon_sha,
        actual_file_sha=actual_file_sha,
        allow_non_dp=allow_non_dp,
    )

    dict_sha: Optional[str] = None
    if dictionary_path is not None:
        exp_d = _require_hex_digest(expected_dictionary_sha256, "expected_dictionary_sha256")
        from .sandbox import PublicLookup

        lkp = PublicLookup(dictionary_path, expected_sha256=exp_d)
        try:
            dict_sha = lkp.file_sha256
        finally:
            lkp.connection.close()

    out: Dict[str, Any] = {
        "profile_sha256": actual_canon_sha,
        "canonical_sha256": actual_canon_sha,
        "file_sha256": actual_file_sha,
        "dictionary_sha256": dict_sha,
        "profile": profile_obj,
    }
    if path_str is not None:
        out["path"] = path_str
    return out


def generate_sql(
    model,
    question,
    schema,
    profile,
    sandbox=None,
    lookup=None,
    enable_sandbox: bool = True,
    expected_profile_sha256: Optional[str] = None,
    is_entity_profile: bool = False,
    require_pinned_digest: bool = True,
    allow_non_dp: bool = False,
    expected_dictionary_sha256: Optional[str] = None,
    receiver_id: Optional[str] = None,
    receiver_ledger: Optional[SingleReleaseReceiverLedger] = None,
):
    """Accept native chat-completion replies or a single {name, arguments} mock action.

    By default (`require_pinned_digest=True`), verifies release bundle integrity (requiring
    `expected_profile_sha256` and rejecting `deployment_prohibited` non-DP profiles unless
    `allow_non_dp=True`), checks `lookup` dictionary digest pinning when `lookup` is supplied,
    and enforces A9 single-release receiver isolation when `receiver_ledger` is provided before
    constructing prompts or calling `model`.
    """
    profile_obj: Any = None
    canon_sha: Optional[str] = None
    if profile is not None:
        if require_pinned_digest:
            if expected_profile_sha256 is None:
                raise ValueError("expected_profile_sha256 is required when require_pinned_digest=True")
            bundle = verify_release_bundle(
                profile,
                expected_profile_sha256=expected_profile_sha256,
                allow_non_dp=allow_non_dp,
            )
            profile_obj = bundle["profile"]
            canon_sha = bundle["canonical_sha256"]
        elif expected_profile_sha256 is not None:
            bundle = verify_release_bundle(
                profile,
                expected_profile_sha256=expected_profile_sha256,
                allow_non_dp=allow_non_dp,
            )
            profile_obj = bundle["profile"]
            canon_sha = bundle["canonical_sha256"]
        else:
            path_str: Optional[str] = None
            actual_file_sha: Optional[str] = None
            if isinstance(profile, (str, Path)):
                p = Path(profile)
                path_str = str(p)
                raw_bytes = p.read_bytes()
                actual_file_sha = hashlib.sha256(raw_bytes).hexdigest()
                profile_obj = json.loads(raw_bytes.decode("utf-8"))
            else:
                profile_obj = profile
            canon_sha = canonical_profile_sha256(profile_obj)
            _assert_dp_deployment_allowed(
                profile_obj,
                path_str=path_str,
                actual_canon_sha=canon_sha,
                actual_file_sha=actual_file_sha,
                allow_non_dp=allow_non_dp,
            )

    if lookup is not None:
        actual_dict_sha = getattr(lookup, "file_sha256", None)
        verified_dict_sha = getattr(lookup, "verified_sha256", None)
        if expected_dictionary_sha256 is not None:
            exp_d = _require_hex_digest(expected_dictionary_sha256, "expected_dictionary_sha256")
            if not isinstance(actual_dict_sha, str) or actual_dict_sha.lower() != exp_d:
                raise ValueError(
                    f"Public dictionary SHA-256 mismatch: expected={exp_d}, actual={actual_dict_sha}"
                )
            if verified_dict_sha is not None and str(verified_dict_sha).lower() != exp_d:
                raise ValueError(
                    f"Public dictionary verified SHA-256 mismatch: expected={exp_d}, verified={verified_dict_sha}"
                )
        elif require_pinned_digest:
            if (
                not isinstance(verified_dict_sha, str)
                or not isinstance(actual_dict_sha, str)
                or verified_dict_sha.lower() != actual_dict_sha.lower()
            ):
                raise ValueError(
                    "PublicLookup must be pinned with expected_sha256 or pass expected_dictionary_sha256 "
                    "to generate_sql when require_pinned_digest=True"
                )

    if receiver_ledger is not None:
        if canon_sha is None:
            raise ValueError("profile is required when receiver_ledger is provided")
        receiver_ledger.bind_or_verify(receiver_id or "", canon_sha)

    config = json.loads(CONFIG.read_text())
    if enable_sandbox and sandbox is None:
        raise ValueError("Schema sandbox required")
    tools = {t["function"]["name"]: t for t in config["tools"]}
    names = ["submit_sql"]
    if enable_sandbox:
        names.insert(0, "try_sql")
    if lookup is not None:
        names.append("lookup_reference")
    if is_entity_profile:
        profile_block = "\n" + edp.format_entity_profile_prompt(profile_obj)
    else:
        profile_block = (
            "\nPublic profile-format legend: " + config["profile_format"]
            + json.dumps(config["profile_legend"], ensure_ascii=False)
            + "\nQuestion-independent safe column profile:" + "\n"
            + json.dumps(profile_obj, ensure_ascii=False, separators=(",", ":"))
        )
    content = "Question: " + question + "\nSchema:\n" + schema + profile_block
    messages = [{"role": "system", "content": config["system" if enable_sandbox else "system_no_sandbox"]},
                {"role": "user", "content": content}]
    fallback = None
    for turn in range(T_MAX):
        outbound = list(messages)
        if turn == T_MAX - 1:
            outbound.append({"role":"user", "content":config["last_turn"]})
        request = dict(config["generation"], messages=outbound, tools=[tools[n] for n in names])
        response = model(request)
        if "choices" in response:
            message = response["choices"][0]["message"]
        else:
            message = {"role":"assistant", "tool_calls":[{
                "type":"function", "id":"call_" + str(turn), "function":{
                    "name":response.get("name"), "arguments":json.dumps(response.get("arguments"))}}]}
        messages.append({k:message[k] for k in ("role","content","tool_calls","reasoning_content") if k in message})
        calls = message.get("tool_calls") or []
        if not calls:
            messages.append({"role":"user", "content":"Use the native tools; submit the final SQL with submit_sql."})
        for call in calls:
            function = call.get("function", {})
            name = function.get("name")
            try:
                sql = json.loads(function.get("arguments", ""))["sql"]
                if not isinstance(sql, str) or not sql.strip():
                    raise ValueError("Expected non-empty SQL")
            except (KeyError, TypeError, ValueError):
                result = {"status":"error", "category":"invalid_tool_arguments"}
            else:
                if name not in names:
                    result = {"status":"error", "category":"unavailable_tool"}
                elif name == "submit_sql":
                    return sql
                elif name == "try_sql":
                    result = sandbox.try_sql(sql)
                    if result["status"] == "success":
                        fallback = sql
                else:
                    result = lookup.lookup_reference(sql)
            messages.append({"role":"tool", "tool_call_id":call.get("id","invalid"),
                             "name":name, "content":json.dumps(result, ensure_ascii=False)})
    if fallback is None:
        raise ValueError("No SQL submission or validated fallback")
    return fallback
