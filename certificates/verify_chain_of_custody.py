#!/usr/bin/env python3
"""Mandatory Chain-of-Custody Verifier (R1) for Silhouette artifact releases.

Verifies three cryptographic links end-to-end:
1. Manifest integrity: files listed in release manifests exist under `profiles/`
   and match their declared raw-byte SHA-256 digests.
2. Bundled profile & certificate consistency: every JSON profile under `profiles/`
   is structurally valid, has both raw-byte SHA-256 and canonical key-sorted
   SHA-256 computed via `silhouette.agent`, and agrees with `.dpmeta.json` sidecars
   and `certificates/*.json`.
3. Receipt-to-profile custody: designated primary/clean evaluation receipts in
   `stats/sql_and_audit_receipts.jsonl` bind via `release_sha256` (raw or canonical)
   to an actual bundled `profiles/**/*.json` file.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

ROOT_DEFAULT = Path(__file__).resolve().parents[1]
if str(ROOT_DEFAULT) not in sys.path:
    sys.path.insert(0, str(ROOT_DEFAULT))

from silhouette.agent import canonical_profile_sha256, file_sha256
from silhouette.jsonl_io import open_jsonl, resolve_jsonl_path

try:
    from certificates import verify_certificates as cert_verifier
except ImportError:
    cert_verifier = None


class ChainOfCustodyError(ValueError):
    """Raised when any manifest, profile, certificate, or receipt custody check fails."""


DEFAULT_PRIMARY_CLEAN_CONDITIONS: Tuple[str, ...] = (
    "BIRD_NOISELESS_V3/r0",
    "DG8_CLEAN_R1/r0",
    "DG8_CLEAN_R2/r0",
    "DG8_CLEAN_R3/r0",
    "DG8_FLATW_R2/r0",
    "DG8_FLATW_R3/r0",
    "DG8_NOVAL_POST_R1/r0",
    "DG8_NOVAL_POST_R2/r0",
    "DG8_NOVAL_POST_R3/r0",
    "DG8_OMEGA1_R1/r0",
    "DG8_OMEGA1_R2/r0",
    "DG8_OMEGA1_R3/r0",
    "DG8_UNPRUNED_V2_R1/r0",
    "DG8_UNPRUNED_V2_R2/r0",
    "DG8_UNPRUNED_V2_R3/r0",
    "EHR874_NOISELESS_NOLKP_V3/r0",
    "ENT_V2_DG8_R1/r0",
    "ENT_V2_DG8_R2/r0",
    "ENT_V2_DG8_R3/r0",
    "ENT_V2_IND8_PROMPT_R1/r0",
    "ENT_V2_IND8_PROMPT_R2/r0",
    "ENT_V2_IND8_PROMPT_R3/r0",
    "IND8_PROFILE_TOOL_R1/r0",
    "IND8_PROFILE_TOOL_R2/r0",
    "IND8_PROFILE_TOOL_R3/r0",
    "IND8_PROMPT_R2/r0",
    "IND8_PROMPT_R3/r0",
    "PUBCAP_FLATW_R1/r0",
    "PUBCAP_FLATW_R2/r0",
    "PUBCAP_FLATW_R3/r0",
    "PUBCAP_IND8_PROFILE_R1/r0",
    "PUBCAP_IND8_PROFILE_R2/r0",
    "PUBCAP_IND8_PROFILE_R3/r0",
    "PUBCAP_IND8_ROWS_R1/r0",
    "PUBCAP_IND8_ROWS_R2/r0",
    "PUBCAP_IND8_ROWS_R3/r0",
    "PUBCAP_OMEGA1_EQDELTA_R1/r0",
    "PUBCAP_OMEGA1_EQDELTA_R2/r0",
    "PUBCAP_OMEGA1_EQDELTA_R3/r0",
    "PUBCAP_SILHOUETTE_R1/r0",
    "PUBCAP_SILHOUETTE_R2/r0",
    "PUBCAP_SILHOUETTE_R3/r0",
    "PUBCAP_UNPRUNED_R1/r0",
    "PUBCAP_UNPRUNED_R2/r0",
    "PUBCAP_UNPRUNED_R3/r0",
    "SOL_DG8_CLEAN_R1/r0",
    "SOL_DG8_CLEAN_R2/r0",
    "SOL_DG8_CLEAN_R3/r0",
    "SOL_ENT_V2_DG8_R1/r0",
    "SOL_ENT_V2_IND8_PROMPT_R1/r0",
    "SOL_PUBCAP_IND8_ROWS_R1/r0",
    "TIGHTBPE_IND8_PROF_R1/r0",
    "TIGHTBPE_IND8_PROF_R2/r0",
    "TIGHTBPE_IND8_PROF_R3/r0",
    "TIGHTBPE_SILHOUETTE_R1/r0",
    "TIGHTBPE_SILHOUETTE_R2/r0",
    "TIGHTBPE_SILHOUETTE_R3/r0",
    "TIGHTBPE_STATIC_SYNTH_R1/r0",
    "TIGHTBPE_STATIC_SYNTH_R2/r0",
    "TIGHTBPE_STATIC_SYNTH_R3/r0",
    "TIGHTBPE_VALFIRST_SILH_R1/r0",
    "TIGHTBPE_VALFIRST_SILH_R2/r0",
    "TIGHTBPE_VALFIRST_SILH_R3/r0",
    "TIGHTBPE_VALFIRST_IND8_PROF_R1/r0",
    "TIGHTBPE_VALFIRST_IND8_PROF_R2/r0",
    "TIGHTBPE_VALFIRST_IND8_PROF_R3/r0",
    "FETCH1_PUBCAP_IND8_PROF_R1/r0",
    "FETCH1_PUBCAP_IND8_PROF_R2/r0",
    "FETCH1_PUBCAP_IND8_PROF_R3/r0",
)

STRICT_CLINICAL_EXTRA_CONDITIONS: Tuple[str, ...] = (
    "EHR_DG8P_V3_R1/r0",
    "EHR_DG8P_V3_R2/r0",
    "EHR_DG8P_V3_R3/r0",
    "EHR_DG8P_NOLKP_V3_R1/r0",
    "EHR_DG8P_NOLKP_V3_R2/r0",
    "EHR_DG8P_NOLKP_V3_R3/r0",
    "EHR_NOISELESS_LKP_V3/r0",
    "EHR_NOISELESS_NOLKP_V3/r0",
    "EHR874_DG8P_V3_R1/r0",
    "EHR874_DG8P_V3_R2/r0",
    "EHR874_DG8P_V3_R3/r0",
    "EHR874_DG8P_NOLKP_V3_R1/r0",
    "EHR874_DG8P_NOLKP_V3_R2/r0",
    "EHR874_DG8P_NOLKP_V3_R3/r0",
    "EHR874_NOISELESS_LKP_V3/r0",
    "EHR_ENT8_V3_R1/r0",
    "EHR_IND8_V3_R1/r0",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_file_sha(path: Path, expected_sha: str, label: str) -> str:
    if not path.exists() or not path.is_file():
        raise ChainOfCustodyError(f"Missing manifest target ({label}): {path}")
    actual = file_sha256(path)
    exp = str(expected_sha).strip().lower()
    if actual != exp:
        raise ChainOfCustodyError(
            f"Manifest SHA-256 mismatch for {label} ({path}): expected={exp}, actual={actual}"
        )
    return actual


def _iter_generic_manifest_entries(doc: Any) -> Iterable[Tuple[str, str]]:
    """Extract (relative_path, expected_sha256) pairs from standard manifest layouts."""
    if isinstance(doc, dict):
        if isinstance(doc.get("sha256"), dict):
            for rel, sha in doc["sha256"].items():
                yield str(rel), str(sha)
        if isinstance(doc.get("files"), dict):
            for rel, val in doc["files"].items():
                if isinstance(val, str):
                    yield str(rel), val
                elif isinstance(val, dict) and "sha256" in val:
                    yield str(rel), str(val["sha256"])
        if "profile_path" in doc and "sha256" in doc:
            yield str(doc["profile_path"]), str(doc["sha256"])
        if "rows_path" in doc and "sha256" in doc:
            yield str(doc["rows_path"]), str(doc["sha256"])
        for k, v in doc.items():
            if k not in ("sha256", "files") and isinstance(v, (dict, list)):
                yield from _iter_generic_manifest_entries(v)
    elif isinstance(doc, list):
        for item in doc:
            yield from _iter_generic_manifest_entries(item)


def verify_manifests(root: Path, *, require_all_manifests: bool = False) -> Dict[str, Any]:
    """Verify release manifests under `root/manifests/` against `root/profiles/`."""
    manifests_dir = root / "manifests"
    profiles_dir = root / "profiles"

    required_manifests = [
        "ehrsql_profiles_manifest.json",
        "pubcap_releases_manifest.json",
    ]
    optional_manifests = [
        "tightbpe_and_noval_manifest.json",
        "bird_profiles_manifest.json",
        "entity_profiles_manifest.json",
    ]
    if require_all_manifests:
        required_manifests.extend(optional_manifests)
        optional_manifests = []

    checked_files = 0
    manifests_verified: List[str] = []

    for mname in required_manifests + optional_manifests:
        mpath = manifests_dir / mname
        if not mpath.exists():
            if mname in required_manifests:
                raise ChainOfCustodyError(f"Required manifest missing: {mpath}")
            continue

        doc = _read_json(mpath)
        entries: List[Tuple[str, str]] = []

        if mname == "pubcap_releases_manifest.json":
            if "noiseless_ehrsql_sha256" in doc:
                entries.append(("ehrsql_noiseless_nolkp/release_01.json", doc["noiseless_ehrsql_sha256"]))
            if isinstance(doc.get("noiseless_bird_sha256"), dict):
                for db, sha in doc["noiseless_bird_sha256"].items():
                    entries.append((f"bird_noiseless_clean/release_01/{db}.json", sha))
            for _arm, reps in doc.get("arms", {}).items():
                for _rep, dbs in reps.items():
                    for _db, info in dbs.items():
                        rel = info.get("profile_path") or info.get("rows_path")
                        if not rel or "sha256" not in info:
                            raise ChainOfCustodyError(f"Incomplete pubcap manifest entry: {info!r}")
                        entries.append((rel, info["sha256"]))
        elif mname == "tightbpe_and_noval_manifest.json":
            for _arm, reps in doc.get("arms", {}).items():
                for _rep, dbs in reps.items():
                    for _db, info in dbs.items():
                        rel = info.get("path")
                        if not rel or "sha256" not in info:
                            raise ChainOfCustodyError(f"Incomplete tightbpe manifest entry: {info!r}")
                        entries.append((rel, info["sha256"]))
            noval_section = doc.get("dg8_noval_post") or doc.get("dg8_noval_post_r23") or {}
            for _rep, dbs in noval_section.items():
                for _db, info in dbs.items():
                    rel = info.get("path")
                    if not rel or "sha256" not in info:
                        raise ChainOfCustodyError(f"Incomplete noval manifest entry: {info!r}")
                    entries.append((rel, info["sha256"]))
        else:
            entries.extend(_iter_generic_manifest_entries(doc))

        if not entries:
            raise ChainOfCustodyError(f"Manifest contains no verifiable file entries: {mpath}")

        for rel, exp_sha in entries:
            target = (root / rel) if rel.startswith("profiles/") else (profiles_dir / rel)
            _verify_file_sha(target, exp_sha, f"{mname}:{rel}")
            checked_files += 1
        manifests_verified.append(mname)

    return {
        "manifests_verified": manifests_verified,
        "manifest_files_checked": checked_files,
    }


def _validate_five_field_profile_structure(obj: Any, rel_path: str) -> None:
    if not isinstance(obj, dict) or not obj:
        raise ChainOfCustodyError(f"Profile JSON must be a non-empty object: {rel_path}")
    allow_empty_table = "tightbpe_valfirst" in rel_path
    for tbl, cols in obj.items():
        if not isinstance(tbl, str) or not isinstance(cols, dict) or (not cols and not allow_empty_table):
            raise ChainOfCustodyError(f"Invalid table entry {tbl!r} in profile: {rel_path}")
        for col, entry in cols.items():
            if not isinstance(col, str) or not isinstance(entry, list) or len(entry) != 5:
                raise ChainOfCustodyError(
                    f"Column {tbl}.{col} must be a 5-field list in profile: {rel_path}"
                )


def verify_profiles_and_certificates(root: Path) -> Dict[str, Any]:
    """Verify all bundled profiles, compute raw + canonical SHA-256, and check certificates."""
    profiles_dir = root / "profiles"
    certs_dir = root / "certificates"

    # Load and verify certificate files if present
    cert_docs: Dict[str, Any] = {}
    if certs_dir.exists():
        for cp in sorted(certs_dir.glob("*.json")):
            cdoc = _read_json(cp)
            cert_docs[cp.name] = cdoc
            if cert_verifier is not None and isinstance(cdoc, dict) and "parameters" in cdoc:
                for q in cdoc["parameters"]:
                    try:
                        cert_verifier.check_parameter(q)
                    except Exception as exc:
                        raise ChainOfCustodyError(
                            f"Certificate parameter check failed in {cp.name}: {exc}"
                        ) from exc

    sha_index: Dict[str, List[str]] = defaultdict(list)
    profile_digests: Dict[str, Dict[str, str]] = {}
    profiles_checked = 0
    dpmeta_checked = 0

    for p in sorted(profiles_dir.rglob("*.json")):
        rel = str(p.relative_to(root)).replace("\\", "/")
        if p.name.endswith(".noval_meta.json"):
            meta = _read_json(p)
            prof_path = p.with_name(p.name[: -len(".noval_meta.json")] + ".json")
            if not prof_path.exists():
                raise ChainOfCustodyError(f"Missing profile JSON for sidecar {rel}")
            actual_raw = file_sha256(prof_path)
            if meta.get("sha256") != actual_raw:
                raise ChainOfCustodyError(
                    f"Sidecar SHA-256 mismatch in {rel}: expected={meta.get('sha256')}, actual={actual_raw}"
                )
            dpmeta_checked += 1
            continue
        if p.name.endswith(".dpmeta.json"):
            meta = _read_json(p)
            prof_path = p.with_name(p.name[: -len(".dpmeta.json")] + ".json")
            if not prof_path.exists():
                raise ChainOfCustodyError(f"Missing profile JSON for sidecar {rel}")
            actual_raw = file_sha256(prof_path)
            if meta.get("sha256") != actual_raw:
                raise ChainOfCustodyError(
                    f"Sidecar SHA-256 mismatch in {rel}: expected={meta.get('sha256')}, actual={actual_raw}"
                )
            prof_obj = _read_json(prof_path)
            if "tables" in meta and set(meta["tables"]) != set(prof_obj):
                raise ChainOfCustodyError(f"Sidecar table set mismatch in {rel}")

            # Cross-check certificate schedules when corresponding certificate bundle exists
            parts = p.relative_to(profiles_dir).parts
            group_dir = parts[0] if parts else ""
            if group_dir in ("bird_minidev_dg_v3", "pubcap_silhouette") and "bird_dg_v3_certificates.json" in cert_docs:
                cert = cert_docs["bird_dg_v3_certificates.json"]
                eps_key = str(meta["epsilon"])
                db_id = meta["db_id"]
                expected_tables = cert["schedules"][eps_key][db_id]
                for tname, tmeta in meta["tables"].items():
                    exp_t = expected_tables[tname]
                    if tmeta["rho"] != exp_t["rho"] or tmeta["delta_threshold"] != exp_t["delta_threshold"]:
                        raise ChainOfCustodyError(
                            f"Certificate ledger mismatch for {rel} table {tname}"
                        )
                    actual_params = [(s, c, meta["parameters"][i]) for s, c, i in tmeta["queries"]]
                    exp_params = [(s, c, cert["parameters"][i]) for s, c, i in exp_t["queries"]]
                    if actual_params != exp_params:
                        raise ChainOfCustodyError(
                            f"Certificate parameter mismatch for {rel} table {tname}"
                        )
            elif group_dir == "bird_entity_dp_eps8" and "bird_entity_dp_certificates.json" in cert_docs:
                cert = cert_docs["bird_entity_dp_certificates.json"]
                exp = cert["schedules"][str(meta["epsilon"])][meta["db_id"]]
                if meta["parameters"] != exp["parameters"] or meta["tables"] != exp["tables"]:
                    raise ChainOfCustodyError(f"Entity certificate mismatch for {rel}")
            elif group_dir == "entity_ind8_prompt" and "bird_entity_dp_certificates.json" in cert_docs:
                cert = cert_docs["bird_entity_dp_certificates.json"]
                exp = cert["ind8_prompt_schedules"][str(meta["epsilon"])][meta["db_id"]]
                if meta["parameters"] != exp["parameters"] or meta["tables"] != exp["tables"]:
                    raise ChainOfCustodyError(f"Entity IND8 certificate mismatch for {rel}")

            dpmeta_checked += 1
            continue

        raw_bytes = p.read_bytes()
        raw_sha = hashlib.sha256(raw_bytes).hexdigest()
        obj = json.loads(raw_bytes.decode("utf-8"))
        canon_sha = canonical_profile_sha256(obj)

        if p.name != "public_4_dictionaries.json":
            _validate_five_field_profile_structure(obj, rel)

        sha_index[raw_sha].append(rel)
        if canon_sha != raw_sha:
            sha_index[canon_sha].append(rel)
        # Also index compact non-key-sorted serialization used in some Turn-0 prompt hashes
        compact = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        for extra_bytes in (compact.encode("utf-8"), (compact + "\n").encode("utf-8")):
            esha = hashlib.sha256(extra_bytes).hexdigest()
            if rel not in sha_index[esha]:
                sha_index[esha].append(rel)

        profile_digests[rel] = {"raw_sha256": raw_sha, "canonical_sha256": canon_sha}
        profiles_checked += 1

    for p in sorted(profiles_dir.rglob("*.txt")):
        rel = str(p.relative_to(root)).replace("\\", "/")
        raw_sha = hashlib.sha256(p.read_bytes()).hexdigest()
        if rel not in sha_index[raw_sha]:
            sha_index[raw_sha].append(rel)
        profile_digests[rel] = {"raw_sha256": raw_sha, "canonical_sha256": raw_sha}
        profiles_checked += 1

    return {
        "certificates_checked": len(cert_docs),
        "profiles_checked": profiles_checked,
        "dpmeta_checked": dpmeta_checked,
        "sha_index": dict(sha_index),
        "profile_digests": profile_digests,
    }


def verify_receipts(
    root: Path,
    sha_index: Dict[str, List[str]],
    *,
    conditions: Optional[Sequence[str]] = None,
    strict_clinical: bool = False,
) -> Dict[str, Any]:
    """Verify semantic `(benchmark, condition, db_id)` custody and slot completeness on receipts."""
    receipts_path = resolve_jsonl_path(root / "stats" / "sql_and_audit_receipts.jsonl")
    if not receipts_path.exists():
        raise ChainOfCustodyError(f"Missing receipts file: {receipts_path}")

    cov_manifest_path = root / "manifests" / "custody_coverage_manifest_v10.json"
    if not cov_manifest_path.exists():
        cov_manifest_path = root / "manifests" / "custody_coverage_manifest_v9.json"
    cov_manifest: Optional[Dict[str, Any]] = None
    cov_series: Dict[str, Dict[str, Any]] = {}
    if cov_manifest_path.exists():
        cov_manifest = _read_json(cov_manifest_path)
        cov_series = cov_manifest.get("series", {})

    if conditions is not None:
        target_conditions: Set[str] = {c.strip() for c in conditions if c.strip()}
    elif strict_clinical:
        if cov_series:
            target_conditions = {
                info["condition"]
                for info in cov_series.values()
                if info.get("lineage_tier") == "canonical_custody_v3"
            }
        else:
            target_conditions = set(DEFAULT_PRIMARY_CLEAN_CONDITIONS) | set(STRICT_CLINICAL_EXTRA_CONDITIONS)
    else:
        target_conditions = set(DEFAULT_PRIMARY_CLEAN_CONDITIONS)

    checked_by_condition: Dict[str, int] = defaultdict(int)
    checked_by_series: Dict[str, int] = defaultdict(int)
    outbound_by_series: Dict[str, int] = defaultdict(int)
    all_series_counts: Dict[str, int] = defaultdict(int)
    all_outbound_total = 0
    total_all_receipts = 0
    resolved_profiles: Set[str] = set()

    with open_jsonl(receipts_path) as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            total_all_receipts += 1
            bm = row.get("benchmark", "")
            cond = row.get("condition", "")
            skey = f"{bm}::{cond}"
            all_series_counts[skey] += 1
            has_outbound = bool(row.get("outbound_messages_sha256"))
            if has_outbound:
                all_outbound_total += 1
            if cond not in target_conditions:
                continue

            sha = row.get("release_sha256")
            if not isinstance(sha, str) or not sha.strip():
                raise ChainOfCustodyError(
                    f"Receipt missing release_sha256 at line {line_no}: "
                    f"condition={cond!r}, benchmark={bm!r}, "
                    f"question_id={row.get('question_id')!r}, db_id={row.get('db_id')!r}"
                )
            sha_norm = sha.strip().lower()
            matches = sha_index.get(sha_norm)
            if not matches:
                raise ChainOfCustodyError(
                    f"Unverified release_sha256 at line {line_no}: "
                    f"condition={cond!r}, benchmark={bm!r}, "
                    f"question_id={row.get('question_id')!r}, db_id={row.get('db_id')!r}, "
                    f"release_sha256={sha_norm} does not resolve to any bundled profiles/**/*.json"
                )

            if cov_series:
                sentry = cov_series.get(skey)
                if sentry is None:
                    raise ChainOfCustodyError(f"Series {skey!r} missing from custody_coverage_manifest_v10.json")
                if sentry.get("lineage_tier") != "canonical_custody_v3":
                    raise ChainOfCustodyError(
                        f"Series {skey!r} has lineage_tier={sentry.get('lineage_tier')!r}, "
                        "not canonical_custody_v3"
                    )
                db_id = str(row.get("db_id", ""))
                exp_by_db = sentry.get("expected_release_sha256_by_db") or {}
                exp_sha = exp_by_db.get(db_id)
                if not exp_sha or sha_norm != exp_sha.strip().lower():
                    raise ChainOfCustodyError(
                        f"Semantic custody mismatch at line {line_no} for ({bm!r}, {cond!r}, {db_id!r}): "
                        f"expected release_sha256={exp_sha!r}, got={sha_norm!r}"
                    )
                exp_paths = sentry.get("expected_profile_paths_by_db", {}).get(db_id)
                if exp_paths and not any(m in exp_paths for m in matches):
                    raise ChainOfCustodyError(
                        f"Profile path binding mismatch at line {line_no} for ({bm!r}, {cond!r}, {db_id!r}): "
                        f"expected one of {exp_paths}, got {matches}"
                    )

            checked_by_condition[cond] += 1
            checked_by_series[skey] += 1
            if has_outbound:
                outbound_by_series[skey] += 1
            resolved_profiles.update(matches)

    total_checked = sum(checked_by_condition.values())
    if total_checked == 0:
        raise ChainOfCustodyError(
            f"No receipts matched target conditions {sorted(target_conditions)} in {receipts_path}"
        )

    if cov_manifest is not None:
        # Enforce per-condition expected_slots & expected_outbound_sha256_slots on every checked series
        for skey, sentry in cov_series.items():
            if sentry.get("condition") not in target_conditions:
                continue
            exp_n = int(sentry["expected_slots"])
            act_n = checked_by_series.get(skey, 0)
            if act_n != exp_n:
                raise ChainOfCustodyError(
                    f"Slot completeness mismatch for {skey!r}: expected={exp_n}, actual={act_n}"
                )
            exp_out = int(sentry["expected_outbound_sha256_slots"])
            act_out = outbound_by_series.get(skey, 0)
            if act_out != exp_out:
                raise ChainOfCustodyError(
                    f"Outbound SHA-256 completeness mismatch for {skey!r}: expected={exp_out}, actual={act_out}"
                )
        # On full strict-clinical run without condition filter, assert total receipt & outbound denominators
        if conditions is None and strict_clinical:
            exp_total = int(cov_manifest["total_aligned_slots"])
            if total_all_receipts != exp_total:
                raise ChainOfCustodyError(
                    f"Total aligned slots mismatch: expected={exp_total}, actual={total_all_receipts}"
                )
            exp_out_total = int(cov_manifest["outbound_messages_sha256_slots"])
            if all_outbound_total != exp_out_total:
                raise ChainOfCustodyError(
                    f"Total outbound_messages_sha256 slots mismatch: expected={exp_out_total}, actual={all_outbound_total}"
                )

    return {
        "conditions_checked": dict(sorted(checked_by_condition.items())),
        "receipts_verified": total_checked,
        "resolved_profile_files": len(resolved_profiles),
        "semantic_manifest_verified": cov_manifest is not None,
    }


def verify_chain_of_custody(
    root: Path = ROOT_DEFAULT,
    *,
    conditions: Optional[Sequence[str]] = None,
    strict_clinical: bool = False,
    require_all_manifests: bool = False,
) -> Dict[str, Any]:
    """Run all three chain-of-custody verification stages against `root`."""
    m_res = verify_manifests(root, require_all_manifests=require_all_manifests)
    p_res = verify_profiles_and_certificates(root)
    r_res = verify_receipts(
        root,
        p_res["sha_index"],
        conditions=conditions,
        strict_clinical=strict_clinical,
    )
    return {
        **m_res,
        "certificates_checked": p_res["certificates_checked"],
        "profiles_checked": p_res["profiles_checked"],
        "dpmeta_checked": p_res["dpmeta_checked"],
        **r_res,
    }


def run_self_test(root: Path = ROOT_DEFAULT) -> Dict[str, Any]:
    """Verify packaged artifact custody AND assert negative controls fail immediately."""
    live_summary = verify_chain_of_custody(root, strict_clinical=True)

    # Legacy pre-v3 clinical check on live receipts must immediately fail catching d8aaf0b7...
    caught_legacy = False
    try:
        verify_chain_of_custody(root, conditions=["DG8P_R1/r0"])
    except ChainOfCustodyError as exc:
        caught_legacy = "d8aaf0b7" in str(exc)
    assert caught_legacy, "Expected ChainOfCustodyError for legacy DG8P_R1/r0 (d8aaf0b7...)"

    with tempfile.TemporaryDirectory() as tmp:
        troot = Path(tmp)
        (troot / "manifests").mkdir()
        (troot / "profiles" / "demo").mkdir(parents=True)
        (troot / "certificates").mkdir()
        (troot / "stats").mkdir()

        prof = {"t": {"x": ["TEXT", 0.0, "10^0", ["AAAA"], ["ok"]]}}
        prof_path = troot / "profiles" / "demo" / "db.json"
        prof_path.write_text(json.dumps(prof, separators=(",", ":")) + "\n", encoding="utf-8")
        raw_sha = file_sha256(prof_path)
        canon_sha = canonical_profile_sha256(prof)

        (troot / "manifests" / "ehrsql_profiles_manifest.json").write_text(
            json.dumps({"tables_private": 1, "tables_public": [], "sha256": {"demo/db.json": raw_sha}}),
            encoding="utf-8",
        )
        (troot / "manifests" / "pubcap_releases_manifest.json").write_text(
            json.dumps({"arms": {"a": {"release_01": {"db": {"profile_path": "demo/db.json", "sha256": raw_sha}}}}}),
            encoding="utf-8",
        )
        (troot / "stats" / "sql_and_audit_receipts.jsonl").write_text(
            json.dumps({"benchmark": "ehrsql_600", "question_id": "q1", "db_id": "demo", "condition": "EHR_DG8P_V3_R1/r0", "release_sha256": canon_sha}) + "\n",
            encoding="utf-8",
        )
        ok_res = verify_chain_of_custody(troot, strict_clinical=True)
        assert ok_res["receipts_verified"] == 1

        # Negative 1: unverified clinical release_sha256 (d8aaf0b7...) caught immediately
        bad_sha = "d8aaf0b7" + "0" * 56
        (troot / "stats" / "sql_and_audit_receipts.jsonl").write_text(
            json.dumps({"benchmark": "ehrsql_600", "question_id": "q1", "db_id": "mimic_iv", "condition": "DG8P_R1/r0", "release_sha256": bad_sha}) + "\n",
            encoding="utf-8",
        )
        caught_receipt = False
        try:
            verify_chain_of_custody(troot, conditions=["DG8P_R1/r0"])
        except ChainOfCustodyError as exc:
            caught_receipt = bad_sha in str(exc)
        assert caught_receipt, "Expected ChainOfCustodyError for unverified clinical release_sha256"

        # Negative 2: tampered v3 clinical receipt also fails --strict-clinical immediately
        (troot / "stats" / "sql_and_audit_receipts.jsonl").write_text(
            json.dumps({"benchmark": "ehrsql_600", "question_id": "q1", "db_id": "mimic_iv", "condition": "EHR_DG8P_V3_R1/r0", "release_sha256": bad_sha}) + "\n",
            encoding="utf-8",
        )
        caught_v3_tamper = False
        try:
            verify_chain_of_custody(troot, strict_clinical=True)
        except ChainOfCustodyError as exc:
            caught_v3_tamper = bad_sha in str(exc)
        assert caught_v3_tamper, "Expected ChainOfCustodyError for tampered v3 clinical release_sha256"

        # Negative 3: manifest SHA-256 mismatch caught immediately
        (troot / "manifests" / "ehrsql_profiles_manifest.json").write_text(
            json.dumps({"tables_private": 1, "tables_public": [], "sha256": {"demo/db.json": "f" * 64}}),
            encoding="utf-8",
        )
        caught_manifest = False
        try:
            verify_manifests(troot)
        except ChainOfCustodyError:
            caught_manifest = True
        assert caught_manifest, "Expected ChainOfCustodyError for manifest SHA-256 mismatch"

        # Negative 4 & 5: Semantic R1/R2 swap (both hashes valid on disk!) and dropped 1 slot MUST FAIL
        prof2 = {"t": {"x": ["TEXT", 0.0, "10^0", ["BBBB"], ["ok2"]]}}
        prof2_path = troot / "profiles" / "demo" / "db_r2.json"
        prof2_path.write_text(json.dumps(prof2, separators=(",", ":")) + "\n", encoding="utf-8")
        raw2_sha = file_sha256(prof2_path)
        canon2_sha = canonical_profile_sha256(prof2)
        (troot / "manifests" / "ehrsql_profiles_manifest.json").write_text(
            json.dumps({"tables_private": 1, "tables_public": [], "sha256": {"demo/db.json": raw_sha, "demo/db_r2.json": raw2_sha}}),
            encoding="utf-8",
        )
        cov_doc = {
            "total_aligned_slots": 2,
            "outbound_messages_sha256_slots": 0,
            "series": {
                "ehrsql_600::EHR_DG8P_V3_R1/r0": {
                    "benchmark": "ehrsql_600",
                    "condition": "EHR_DG8P_V3_R1/r0",
                    "expected_slots": 1,
                    "lineage_tier": "canonical_custody_v3",
                    "expected_outbound_sha256_slots": 0,
                    "expected_release_sha256_by_db": {"mimic_iv": canon_sha},
                    "expected_profile_paths_by_db": {"mimic_iv": ["profiles/demo/db.json"]},
                },
                "ehrsql_600::EHR_DG8P_V3_R2/r0": {
                    "benchmark": "ehrsql_600",
                    "condition": "EHR_DG8P_V3_R2/r0",
                    "expected_slots": 1,
                    "lineage_tier": "canonical_custody_v3",
                    "expected_outbound_sha256_slots": 0,
                    "expected_release_sha256_by_db": {"mimic_iv": canon2_sha},
                    "expected_profile_paths_by_db": {"mimic_iv": ["profiles/demo/db_r2.json"]},
                },
            },
        }
        (troot / "manifests" / "custody_coverage_manifest_v10.json").write_text(json.dumps(cov_doc), encoding="utf-8")

        # Swap R1 and R2 release_sha256 -> both valid on disk, MUST FAIL semantic custody!
        (troot / "stats" / "sql_and_audit_receipts.jsonl").write_text(
            json.dumps({"benchmark": "ehrsql_600", "question_id": "q1", "db_id": "mimic_iv", "condition": "EHR_DG8P_V3_R1/r0", "release_sha256": canon2_sha}) + "\n"
            + json.dumps({"benchmark": "ehrsql_600", "question_id": "q1", "db_id": "mimic_iv", "condition": "EHR_DG8P_V3_R2/r0", "release_sha256": canon_sha}) + "\n",
            encoding="utf-8",
        )
        caught_swap = False
        try:
            verify_chain_of_custody(troot, strict_clinical=True)
        except ChainOfCustodyError as exc:
            caught_swap = "Semantic custody mismatch" in str(exc)
        assert caught_swap, "Expected ChainOfCustodyError when R1 and R2 release_sha256 are swapped"

        # Drop 1 slot from R2 (keep R1 intact) -> MUST FAIL slot completeness!
        cov_doc["total_aligned_slots"] = 1
        (troot / "manifests" / "custody_coverage_manifest_v10.json").write_text(json.dumps(cov_doc), encoding="utf-8")
        (troot / "stats" / "sql_and_audit_receipts.jsonl").write_text(
            json.dumps({"benchmark": "ehrsql_600", "question_id": "q1", "db_id": "mimic_iv", "condition": "EHR_DG8P_V3_R1/r0", "release_sha256": canon_sha}) + "\n",
            encoding="utf-8",
        )
        caught_drop = False
        try:
            verify_chain_of_custody(troot, strict_clinical=True)
        except ChainOfCustodyError as exc:
            caught_drop = "Slot completeness mismatch" in str(exc)
        assert caught_drop, "Expected ChainOfCustodyError when 1 slot is dropped"

    return {
        "live_summary": live_summary,
        "negative_controls_caught": True,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument(
        "--conditions",
        action="append",
        default=None,
        help="Comma-separated or repeated condition name(s) to verify in sql_and_audit_receipts.jsonl.",
    )
    parser.add_argument(
        "--strict-clinical",
        action="store_true",
        help="Require every clean BIRD & clean EHRSQL primary/entity/pubcap receipt to resolve to bundled profiles.",
    )
    parser.add_argument(
        "--require-all-manifests",
        action="store_true",
        help="Require bird_profiles_manifest.json and entity_profiles_manifest.json in addition to EHRSQL/pubcap.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run chain-of-custody verification plus isolated tampered/unverified negative-control tests.",
    )
    args = parser.parse_args(argv)

    parsed_conditions: Optional[List[str]] = None
    if args.conditions:
        parsed_conditions = [
            item.strip()
            for chunk in args.conditions
            for item in chunk.split(",")
            if item.strip()
        ]

    try:
        if args.self_test:
            res = run_self_test(args.root.resolve())
            live = res["live_summary"]
            print(
                "CHAIN OF CUSTODY SELF-TEST PASSED: "
                f"manifests={len(live['manifests_verified'])}, "
                f"profiles={live['profiles_checked']}, "
                f"dpmeta={live['dpmeta_checked']}, "
                f"receipts={live['receipts_verified']} "
                f"across {len(live['conditions_checked'])} conditions"
            )
            return 0

        res = verify_chain_of_custody(
            args.root.resolve(),
            conditions=parsed_conditions,
            strict_clinical=args.strict_clinical,
            require_all_manifests=args.require_all_manifests,
        )
        print(
            "CHAIN OF CUSTODY VERIFIED: "
            f"manifests={len(res['manifests_verified'])}, "
            f"profiles={res['profiles_checked']}, "
            f"dpmeta={res['dpmeta_checked']}, "
            f"receipts={res['receipts_verified']} "
            f"across {len(res['conditions_checked'])} conditions"
        )
        return 0
    except ChainOfCustodyError as exc:
        print(f"CHAIN OF CUSTODY ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
