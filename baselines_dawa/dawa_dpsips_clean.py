"""Clean Uniform 1D DP (DAWA + DP-SIPS) Profiler with OS CSPRNG & Exact Rational zCDP Certificates.

Fixes all Round 12 Codex audit findings:
  1. Uses OS CSPRNG (dg.BufferedOSCSPRNG backed by os.urandom) for all formal DP releases;
     optional `test_seed` is strictly restricted to non-privacy unit tests.
  2. Uses canonical per-row SQLite normalization (`silhouette.profile.column_statistics_exact`
     with `typeof(col)` + `canonical_val_key` and per-row `shape`/`bin_index`) BEFORE any
     counting, eliminating mixed int/float/str sensitivity collisions.
  3. Emits exact rational `Fraction` zCDP certificates (`.dpmeta.json`) compatible with
     `certificates/verify_certificates.py`, verifying `sum(rho) <= rho_T` and
     `sum(delta_tail) <= 5e-7` for every table.
  4. Uniform per-column 1D DP budget allocation (NO DDL role weights, NO key pruning)
     and deterministic TightBPE ColFirst (`drop_values_first`) & ValFirst (`preserve_values`)
     post-processing under `Cap_TightBPE(S_d) = 100 + 10*|T_d| + 12*|C_d|`.
"""
from __future__ import annotations

from collections import Counter
import copy
from fractions import Fraction
import hashlib
import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any, Callable, Collection, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from silhouette import dg_sampler as dg  # noqa: E402
from silhouette import profile as prof_mod  # noqa: E402
from silhouette import schema_roles as sr  # noqa: E402

TABLE_EPSILON = Fraction(8)
TABLE_DELTA = Fraction(1, 10**6)
TABLE_DELTA_HALF = TABLE_DELTA / 2  # 5e-7 for zCDP conversion + 5e-7 for selection tail
RHO_TABLE_8 = dg.rational_rho_budget(TABLE_EPSILON, TABLE_DELTA_HALF)
EHR_PUBLIC_TABLES = frozenset(("d_icd_diagnoses", "d_icd_procedures", "d_items", "d_labitems"))
MECHANISM_NAME = "DAWA+DP-SIPS"


class _UnitTestSeededRNG:
    """Deterministic SHAKE-256 counter RNG strictly for non-privacy unit tests."""

    def __init__(self, seed: int) -> None:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError(f"test_seed must be int, got {type(seed).__name__}")
        self._seed_bytes = f"unit_test_seed:{seed}".encode("utf-8")
        self._counter = 0
        self._buf = b""
        self._pos = 0

    def _refill(self) -> None:
        chunk = hashlib.shake_256(self._seed_bytes + self._counter.to_bytes(8, "big")).digest(65536)
        self._counter += 1
        self._buf = self._buf[self._pos :] + chunk
        self._pos = 0

    def _getrandbits(self, k: int) -> int:
        nbytes = (k + 7) >> 3
        while self._pos + nbytes > len(self._buf):
            self._refill()
        raw = self._buf[self._pos : self._pos + nbytes]
        self._pos += nbytes
        val = int.from_bytes(raw, "big")
        excess = (nbytes << 3) - k
        return val >> excess if excess else val

    def randrange(self, stop: int) -> int:
        if stop <= 0:
            raise ValueError(f"stop must be > 0, got {stop}")
        k = stop.bit_length()
        while True:
            r = self._getrandbits(k)
            if r < stop:
                return r


def _frac_str(f: Fraction) -> str:
    return f"{f.numerator}/{f.denominator}"


def ro_conn(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", errors="ignore")
    con.execute("PRAGMA query_only=ON")
    con.execute("PRAGMA temp_store=MEMORY")
    return con


def dawa_1d_l1_partition(noisy_counts: np.ndarray, sigma_est: float) -> List[Tuple[int, int]]:
    """DAWA 1D dynamic-programming L1 partition over ordered noisy histogram counts.

    Pure post-processing on DP noisy counts: partitions [0..m-1] into contiguous buckets [l, r]
    minimizing sum_bucket (L1 deviation from bucket mean + sigma_est * sqrt(2/pi)).
    """
    m = len(noisy_counts)
    if m == 0:
        return []
    if m == 1:
        return [(0, 0)]

    if m > 64:
        step = math.ceil(m / 64)
        chunks = [(i, min(m - 1, i + step - 1)) for i in range(0, m, step)]
        chunk_vals = np.array([noisy_counts[a : b + 1].sum() for a, b in chunks], dtype=np.float64)
        sub_parts = dawa_1d_l1_partition(chunk_vals, sigma_est)
        return [(chunks[a][0], chunks[b][1]) for a, b in sub_parts]

    noise_pen = max(1.0, sigma_est * math.sqrt(2.0 / math.pi))
    cost = np.full((m, m), np.inf, dtype=np.float64)
    for l in range(m):
        for r in range(l, m):
            seg = noisy_counts[l : r + 1]
            mean = float(seg.mean())
            dev = float(np.abs(seg - mean).sum())
            cost[l, r] = dev + noise_pen

    dp = np.full(m + 1, np.inf, dtype=np.float64)
    back = np.zeros(m + 1, dtype=int)
    dp[0] = 0.0
    for r in range(1, m + 1):
        for l in range(r):
            cand = dp[l] + cost[l, r - 1]
            if cand < dp[r]:
                dp[r] = cand
                back[r] = l

    parts: List[Tuple[int, int]] = []
    cur = m
    while cur > 0:
        prev = int(back[cur])
        parts.append((prev, cur - 1))
        cur = prev
    parts.reverse()
    return parts


def compute_column_histograms_normalized(
    con: sqlite3.Connection,
    table: str,
    col_meta: sr.ColumnMeta,
    nrows: int,
    *,
    need_values: bool = True,
) -> Dict[str, Any]:
    """Per-row normalized exact statistics reusing canonical `silhouette.profile.column_statistics_exact`.

    Scans `SELECT typeof(col), col FROM table WHERE col IS NOT NULL` row-by-row so mixed
    SQLite int/float/str/blob values NEVER collapse via Python's `1 == 1.0` dict equality
    before `shape()`, `bin_index()`, or `canonical_val_key()`.
    """
    st = prof_mod.column_statistics_exact(
        con, table, col_meta.name, col_meta.declared_type, nrows, need_values=need_values
    )
    st["distribution_sorted"] = [
        (k, st["distribution"][k]) for k in sorted(st["distribution"], key=repr) if st["distribution"][k] > 0
    ]
    st["values_sorted"] = [
        (k, st["values"][k]) for k in sorted(st["values"], key=repr) if st["values"][k] > 0
    ]
    return st


def plan_table_dawa_dpsips(
    table_name: str,
    col_metas: Tuple[sr.ColumnMeta, ...],
    *,
    epsilon: Fraction = TABLE_EPSILON,
    delta: Fraction = TABLE_DELTA,
) -> Dict[str, Any]:
    """Exact rational uniform 1D DP budget plan for one table (DAWA numeric + DP-SIPS cat/text)."""
    eps_f = Fraction(epsilon)
    delta_f = Fraction(delta)
    delta_half = delta_f / 2
    rho_table = dg.rational_rho_budget(eps_f, delta_half)

    # Logical queries: 1 row_count + 4 per column (uniform across ALL columns, no role weights, no key pruning)
    logical_queries: List[Tuple[str, Optional[str]]] = [("row_count", None)]
    for cm in col_metas:
        logical_queries.append(("null_count", cm.name))
        logical_queries.append(("distinct_count", cm.name))
        if sr.numeric(cm.declared_type):
            logical_queries.append(("bins", cm.name))
        else:
            logical_queries.append(("patterns", cm.name))
        logical_queries.append(("values", cm.name))

    n_q = len(logical_queries)
    n_hist = sum(1 for s, _ in logical_queries if s in prof_mod.HISTOGRAMS)
    assert n_hist == 2 * len(col_metas)
    r_total = 3 * n_q
    delta_per_hist = delta_half / n_hist

    items: List[Dict[str, Any]] = []
    for stat, col in logical_queries:
        if stat == "bins":
            # DAWA exact two-stage split: 1/3 for bin thresholding/partitioning + 2/3 for disjoint partition counts
            sub_specs = [("bins_stage1", 1, delta_per_hist), ("bins_stage2", 2, Fraction(0))]
        elif stat in ("patterns", "values"):
            sub_specs = [(stat, 3, delta_per_hist)]
        else:
            sub_specs = [(stat, 3, Fraction(0))]

        for sub_stat, r_j, dj in sub_specs:
            share = Fraction(r_j, r_total)
            rho_q = rho_table * share
            sigma2 = Fraction(1, 2) / rho_q
            item: Dict[str, Any] = {
                "stat": sub_stat,
                "column": col,
                "r_j": r_j,
                "r_total": r_total,
                "share": share,
                "rho": rho_q,
                "sigma2": sigma2,
                "delta": dj,
                "sensitivity2": 1,
            }
            if dj > 0:
                tau, tail_diag = dg.exact_dgauss_tau(sigma2, dj)
                item["tau"] = tau
                item["tail_diag"] = tail_diag
            items.append(item)

    rho_used = sum((it["rho"] for it in items), Fraction(0))
    delta_tail_used = sum((it["delta"] for it in items), Fraction(0))
    assert rho_used == rho_table and rho_used <= rho_table
    assert delta_tail_used == delta_half and delta_tail_used <= Fraction(1, 2_000_000)
    eps_upper = dg.conservative_eps_from_rho(rho_used, delta_half)
    assert Fraction(eps_upper) <= eps_f

    return {
        "table": table_name,
        "epsilon": eps_f,
        "delta": delta_f,
        "delta_conversion": delta_half,
        "delta_tail_budget": delta_half,
        "rho_table": rho_table,
        "rho_used": rho_used,
        "delta_tail_total": delta_tail_used,
        "epsilon_converted_upper": str(eps_upper),
        "n_logical_queries": n_q,
        "n_histograms": n_hist,
        "items": items,
    }


def _param_entry_dict(it: Dict[str, Any]) -> Dict[str, Any]:
    rho: Fraction = it["rho"]
    share: Fraction = it["share"]
    sigma2: Fraction = it["sigma2"]
    dj: Fraction = it["delta"]
    entry: Dict[str, Any] = {
        "r_j": f"{it['r_j']}/1",
        "share": _frac_str(share),
        "rho": _frac_str(rho),
        "rho_num": rho.numerator,
        "rho_den": rho.denominator,
        "sigma2": _frac_str(sigma2),
        "sigma2_num": sigma2.numerator,
        "sigma2_den": sigma2.denominator,
        "delta": _frac_str(dj),
        "delta_num": dj.numerator,
        "delta_den": dj.denominator,
        "sensitivity2": int(it["sensitivity2"]),
    }
    if dj > 0:
        entry["tau"] = int(it["tau"])
        entry.update(it["tail_diag"])
    return entry


def profile_table_dawa_dpsips(
    con: sqlite3.Connection,
    table: str,
    col_metas: Tuple[sr.ColumnMeta, ...],
    *,
    epsilon: Fraction = TABLE_EPSILON,
    delta: Fraction = TABLE_DELTA,
    test_seed: Optional[int] = None,
) -> Tuple[Dict[str, List[Any]], Dict[str, Any]]:
    """Profile one table using uniform-budget DAWA (numeric) + DP-SIPS (categorical/text) with OS CSPRNG."""
    rng: dg.IntegerRNG = dg.BufferedOSCSPRNG() if test_seed is None else _UnitTestSeededRNG(test_seed)
    plan = plan_table_dawa_dpsips(table, col_metas, epsilon=epsilon, delta=delta)
    qmap = {(it["stat"], it["column"]): it for it in plan["items"]}

    qt = sr.quote(table)
    nrows = int(con.execute(f"SELECT COUNT(*) FROM {qt}").fetchone()[0])
    n_noisy = max(0, nrows + dg.sample_dgauss(qmap[("row_count", None)]["sigma2"], rng))

    out: Dict[str, List[Any]] = {}
    for cm in col_metas:
        qc = sr.quote(cm.name)
        nonnull, distinct = con.execute(f"SELECT COUNT({qc}), COUNT(DISTINCT {qc}) FROM {qt}").fetchone()
        null_cnt = int(nrows - nonnull)
        distinct_cnt = int(distinct)
        nn = max(0, null_cnt + dg.sample_dgauss(qmap[("null_count", cm.name)]["sigma2"], rng))
        nd = max(0, distinct_cnt + dg.sample_dgauss(qmap[("distinct_count", cm.name)]["sigma2"], rng))
        null_frac = round(min(1.0, nn / max(1, n_noisy)), 3)
        order = prof_mod.order_of_magnitude(nd)

        # Post-processing gate on privatized nd: when nd > 50, common is unconditionally None
        need_values = nd <= 50
        st = compute_column_histograms_normalized(con, table, cm, nrows, need_values=need_values)

        if sr.numeric(cm.declared_type):
            q1 = qmap[("bins_stage1", cm.name)]
            q2 = qmap[("bins_stage2", cm.name)]
            sigma2_1, tau_1 = q1["sigma2"], q1["tau"]
            sigma2_2 = q2["sigma2"]

            surv_bins: List[Tuple[int, int]] = []
            exact_bins: Dict[int, int] = {}
            for b, cnt in st["distribution_sorted"]:
                if not isinstance(b, int) or b < 0:
                    continue
                exact_bins[b] = int(cnt)
                noisy = int(cnt) + dg.sample_dgauss(sigma2_1, rng)
                if noisy > tau_1:
                    surv_bins.append((b, noisy))
            surv_bins.sort(key=lambda x: x[0])

            if surv_bins:
                bin_ids = [b for b, _ in surv_bins]
                noisy_arr = np.array([c for _, c in surv_bins], dtype=np.float64)
                parts = dawa_1d_l1_partition(noisy_arr, math.sqrt(float(sigma2_2)))
                reconstructed: Dict[int, float] = {}
                for l_idx, r_idx in parts:
                    true_sum = sum(exact_bins[bin_ids[k]] for k in range(l_idx, r_idx + 1))
                    noisy_sum = max(0.0, float(true_sum + dg.sample_dgauss(sigma2_2, rng)))
                    width = r_idx - l_idx + 1
                    per_bin = noisy_sum / width
                    if per_bin > 0.0:
                        for k in range(l_idx, r_idx + 1):
                            reconstructed[bin_ids[k]] = per_bin
                dist = prof_mod.quantiles(reconstructed)
            else:
                dist = [None, None, None]
        else:
            qp = qmap[("patterns", cm.name)]
            sigma2_p, tau_p = qp["sigma2"], qp["tau"]
            pat_hist: Dict[Any, int] = {}
            for pat, cnt in st["distribution_sorted"]:
                noisy = int(cnt) + dg.sample_dgauss(sigma2_p, rng)
                if noisy > tau_p:
                    pat_hist[pat] = noisy
            dist = prof_mod.top_keys(pat_hist, 3)

        common = None
        if need_values:
            qv = qmap[("values", cm.name)]
            sigma2_v, tau_v = qv["sigma2"], qv["tau"]
            vh: Dict[Any, int] = {}
            for val_k, cnt in st["values_sorted"]:
                noisy = int(cnt) + dg.sample_dgauss(sigma2_v, rng)
                if noisy > tau_v:
                    vh[val_k] = noisy
            common = [prof_mod.decode_val_key(k) for k in prof_mod.top_keys(vh, 15)]

        out[cm.name] = [cm.declared_type, null_frac, order, dist, common]

    return out, plan


def build_dpmeta_document(
    db_id: str,
    profile_bytes: bytes,
    table_plans: Dict[str, Dict[str, Any]],
    *,
    rng_source: str,
    epsilon: Fraction = TABLE_EPSILON,
    delta: Fraction = TABLE_DELTA,
) -> Dict[str, Any]:
    """Assemble `.dpmeta.json` certificate sidecar compatible with `verify_certificates.py`."""
    delta_half = delta / 2
    rho_table = dg.rational_rho_budget(epsilon, delta_half)

    params: List[Dict[str, Any]] = []
    param_idx_by_key: Dict[Tuple[Any, ...], int] = {}
    tables_doc: Dict[str, Any] = {}

    for t, plan in table_plans.items():
        queries_idx: List[List[Any]] = []
        for it in plan["items"]:
            p_dict = _param_entry_dict(it)
            key = tuple(sorted(p_dict.items()))
            if key not in param_idx_by_key:
                param_idx_by_key[key] = len(params)
                params.append(p_dict)
            idx = param_idx_by_key[key]
            queries_idx.append([it["stat"], it["column"], idx])
        rho_t: Fraction = plan["rho_used"]
        dtail_t: Fraction = plan["delta_tail_total"]
        tables_doc[t] = {
            "rho": _frac_str(rho_t),
            "rho_num": rho_t.numerator,
            "rho_den": rho_t.denominator,
            "delta_threshold": _frac_str(dtail_t),
            "delta_tail_total": _frac_str(dtail_t),
            "delta_tail_num": dtail_t.numerator,
            "delta_tail_den": dtail_t.denominator,
            "epsilon_converted_upper": plan["epsilon_converted_upper"],
            "n_logical_queries": plan["n_logical_queries"],
            "n_histograms": plan["n_histograms"],
            "queries": queries_idx,
        }

    return {
        "db_id": db_id,
        "mechanism": MECHANISM_NAME,
        "epsilon": int(epsilon) if epsilon.denominator == 1 else float(epsilon),
        "epsilon_frac": _frac_str(epsilon),
        "delta": _frac_str(delta),
        "delta_conversion": _frac_str(delta_half),
        "delta_tail_budget": _frac_str(delta_half),
        "rho_table": _frac_str(rho_table),
        "rho_table_num": rho_table.numerator,
        "rho_table_den": rho_table.denominator,
        "rng_source": rng_source,
        "sha256": hashlib.sha256(profile_bytes).hexdigest(),
        "parameters": params,
        "tables": tables_doc,
    }


def profile_db_dawa_dpsips(
    src_db: Path,
    out_json: Path,
    *,
    is_ehr: bool = False,
    epsilon: Fraction = TABLE_EPSILON,
    delta: Fraction = TABLE_DELTA,
    test_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate 5-field DP profile JSON and exact `.dpmeta.json` certificate sidecar."""
    exclude = EHR_PUBLIC_TABLES if is_ehr else frozenset()
    pub_lower = {t.lower() for t in exclude}
    rng_source = "os_csprng" if test_seed is None else "unit_test_seed_non_private"

    with ro_conn(src_db) as con:
        own_tx = not con.in_transaction
        if own_tx:
            con.execute("BEGIN")
        try:
            schema = sr.extract_public_schema(con)
            profile: Dict[str, Any] = {}
            table_plans: Dict[str, Dict[str, Any]] = {}
            for table, cols in schema.tables.items():
                if table.lower() in pub_lower:
                    continue
                t_prof, t_plan = profile_table_dawa_dpsips(
                    con, table, cols, epsilon=epsilon, delta=delta, test_seed=test_seed
                )
                profile[table] = t_prof
                table_plans[table] = t_plan
        finally:
            if own_tx:
                con.rollback()

    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(profile, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    out_json.write_bytes(payload)

    dpmeta = build_dpmeta_document(
        src_db.stem,
        payload,
        table_plans,
        rng_source=rng_source,
        epsilon=epsilon,
        delta=delta,
    )
    dpmeta_path = out_json.with_suffix(".dpmeta.json")
    dpmeta_path.write_text(json.dumps(dpmeta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "mechanism": MECHANISM_NAME,
        "rng_source": rng_source,
        "profile_path": str(out_json),
        "dpmeta_path": str(dpmeta_path),
        "sha256": dpmeta["sha256"],
        "n_tables": len(table_plans),
    }


def verify_dpmeta_certificate(
    dpmeta_path: Path,
    *,
    require_os_csprng: bool = True,
    verify_cert_mod: Any = None,
) -> Dict[str, Any]:
    """Verify exact rational zCDP budget, tails, schema compatibility, and SHA-256 of `.dpmeta.json`."""
    meta = json.loads(dpmeta_path.read_text(encoding="utf-8"))
    db = meta["db_id"]
    jpath = dpmeta_path.with_name(f"{db}.json")
    if not jpath.exists():
        raise FileNotFoundError(f"Missing profile JSON for certificate: {jpath}")
    raw_bytes = jpath.read_bytes()
    actual_sha = hashlib.sha256(raw_bytes).hexdigest()
    if meta["sha256"] != actual_sha:
        raise ValueError(f"SHA-256 mismatch for {jpath}: expected {meta['sha256']}, got {actual_sha}")
    if require_os_csprng and meta.get("rng_source") != "os_csprng":
        raise ValueError(f"Formal certificate must use rng_source='os_csprng', got {meta.get('rng_source')!r}")

    eps = Fraction(meta["epsilon_frac"])
    delta = Fraction(meta["delta"])
    delta_half = delta / 2
    assert delta_half == Fraction(1, 2_000_000)
    rho_table_expected = dg.rational_rho_budget(eps, delta_half)
    assert Fraction(meta["rho_table"]) == rho_table_expected
    assert Fraction(meta["rho_table_num"], meta["rho_table_den"]) == rho_table_expected

    params = meta["parameters"]
    parsed_params: List[Tuple[Fraction, Fraction, Fraction, int, Fraction]] = []
    for q in params:
        rho = Fraction(q["rho"])
        share = Fraction(q["share"])
        sigma2 = Fraction(q["sigma2"])
        dj = Fraction(q["delta"])
        sens = int(q["sensitivity2"])
        assert Fraction(q["rho_num"], q["rho_den"]) == rho
        assert Fraction(q["sigma2_num"], q["sigma2_den"]) == sigma2
        assert Fraction(q["delta_num"], q["delta_den"]) == dj
        assert 0 < share <= 1 and rho > 0 and sigma2 * 2 * rho == sens
        if dj > 0:
            tau = q["tau"]
            assert isinstance(tau, int) and 1 <= tau <= 100000
            assert Fraction(q["tail_upper"]) <= dj / sens
            assert Fraction(q["prev_tail_lower"]) > dj / sens
        if verify_cert_mod is not None:
            verify_cert_mod.check_parameter(q)
        parsed_params.append((rho, share, dj, sens, Fraction(q["r_j"])))

    prof = json.loads(raw_bytes.decode("utf-8"))
    assert set(meta["tables"]) == set(prof)
    for t, tinfo in meta["tables"].items():
        rho_sum = sum((parsed_params[idx][0] for _, _, idx in tinfo["queries"]), Fraction(0))
        share_sum = sum((parsed_params[idx][1] for _, _, idx in tinfo["queries"]), Fraction(0))
        delta_sum = sum((parsed_params[idx][2] for _, _, idx in tinfo["queries"]), Fraction(0))
        assert rho_sum == Fraction(tinfo["rho"])
        assert Fraction(tinfo["rho_num"], tinfo["rho_den"]) == rho_sum
        assert rho_sum <= rho_table_expected
        assert share_sum == Fraction(1)
        assert delta_sum == Fraction(tinfo["delta_threshold"])
        assert delta_sum == Fraction(tinfo["delta_tail_total"])
        assert delta_sum <= delta_half
        assert Fraction(dg.conservative_eps_from_rho(rho_sum, delta_half)) <= eps

    return {"db_id": db, "tables": len(meta["tables"]), "parameters": len(params), "sha256": actual_sha}


def compute_tightbpe_cap(schema: sr.PublicSchema, public_tables: Optional[Collection[str]] = None) -> int:
    """Cap_TightBPE(S_d) = 100 + 10*|T_d| + 12*|C_d|."""
    pub_lower = {t.lower() for t in (public_tables or ())}
    n_t = 0
    n_c = 0
    for t, cols in schema.tables.items():
        if t.lower() in pub_lower:
            continue
        n_t += 1
        n_c += len(cols)
    return 100 + 10 * n_t + 12 * n_c


def truncate_profile_to_public_cap(
    profile_obj: Dict[str, Any],
    cap_tokens: int,
    schema: Optional[sr.PublicSchema] = None,
    *,
    profile_format: str = prof_mod.DEFAULT_PROFILE_FORMAT,
    profile_legend: Sequence[str] = prof_mod.DEFAULT_PROFILE_LEGEND,
    token_counter: Optional[Callable[[str], int]] = None,
    mode: str = "drop_values_first",
) -> Dict[str, Any]:
    """Canonical deterministic truncation matching `silhouette.profile.truncate_profile_to_public_cap`.

    Supports `mode="drop_values_first"` (ColFirst) and `mode="preserve_values"` (ValFirst).
    """
    if mode not in ("drop_values_first", "preserve_values"):
        raise ValueError(f"Unknown truncation mode: {mode!r}")

    out = copy.deepcopy(profile_obj)
    counter = prof_mod.estimate_prompt_tokens if token_counter is None else token_counter

    def _cur_tokens(obj: Optional[Dict[str, Any]] = None) -> int:
        target = out if obj is None else obj
        return counter(
            prof_mod.render_profile_prompt_block(
                target,
                profile_format=profile_format,
                profile_legend=profile_legend,
            )
        )

    if _cur_tokens() <= cap_tokens:
        return out

    col_order: List[Tuple[str, str]] = []
    essential_cols: Set[Tuple[str, str]] = set()
    seen_cols: Set[Tuple[str, str]] = set()
    if schema is not None:
        for table in sorted(schema.tables):
            if table not in out or not isinstance(out[table], dict):
                continue
            for cm in schema.tables[table]:
                if cm.name in out[table]:
                    key = (table, cm.name)
                    col_order.append(key)
                    seen_cols.add(key)
                    if cm.is_pk or cm.is_fk_src or cm.is_fk_tgt:
                        essential_cols.add(key)
    for table in sorted(out):
        if not isinstance(out[table], dict):
            continue
        for col in sorted(out[table]):
            key = (table, col)
            if key not in seen_cols:
                col_order.append(key)
                seen_cols.add(key)

    if mode == "preserve_values":
        val_cols: List[Tuple[str, str]] = []
        zero_cols: List[Tuple[str, str]] = []
        for t, c in col_order:
            vals = profile_obj[t][c][4]
            if isinstance(vals, list) and len(vals) > 0:
                val_cols.append((t, c))
            else:
                zero_cols.append((t, c))

        for k_val in (5, 3, 2, 1):
            cand = copy.deepcopy(profile_obj)
            for t, c in val_cols:
                cand[t][c][4] = cand[t][c][4][:k_val]
            if _cur_tokens(cand) <= cap_tokens:
                return cand

            for t, c in reversed(zero_cols):
                arr = cand[t][c]
                cand[t][c] = [arr[0], None, None, None, arr[4]]
                if _cur_tokens(cand) <= cap_tokens:
                    return cand

            for t, c in reversed(zero_cols):
                if (t, c) in essential_cols:
                    continue
                cand[t].pop(c, None)
                if _cur_tokens(cand) <= cap_tokens:
                    return cand

            for t, c in reversed(zero_cols):
                if c in cand[t]:
                    cand[t].pop(c, None)
                    if _cur_tokens(cand) <= cap_tokens:
                        return cand

        cand = copy.deepcopy(profile_obj)
        for t, c in val_cols:
            cand[t][c][4] = cand[t][c][4][:1]
        for t, c in zero_cols:
            cand[t].pop(c, None)

        for t, c in reversed(val_cols):
            arr = cand[t][c]
            cand[t][c] = [arr[0], None, None, None, arr[4]]
            if _cur_tokens(cand) <= cap_tokens:
                return cand

        val_trim_order = sorted(
            val_cols,
            key=lambda tc: (len(profile_obj[tc[0]][tc[1]][4]), -col_order.index(tc)),
        )
        for t, c in val_trim_order:
            cand[t].pop(c, None)
            if _cur_tokens(cand) <= cap_tokens:
                return cand

        for t in sorted(list(cand.keys()), reverse=True):
            if not cand[t]:
                cand.pop(t, None)
                if _cur_tokens(cand) <= cap_tokens:
                    return cand

        out = cand
    else:
        while _cur_tokens() > cap_tokens:
            changed = False
            for table, col in col_order:
                vals = out[table][col][4]
                if isinstance(vals, list) and vals:
                    long_idx = next((i for i, v in enumerate(vals) if isinstance(v, str) and len(v) > 80), None)
                    if long_idx is not None:
                        vals.pop(long_idx)
                    else:
                        vals.pop()
                    changed = True
                    if _cur_tokens() <= cap_tokens:
                        break
            if not changed:
                break

        while _cur_tokens() > cap_tokens:
            changed = False
            for table, col in col_order:
                pats = out[table][col][3]
                if isinstance(pats, list) and len(pats) > 1 and all(isinstance(x, str) for x in pats):
                    pats.pop()
                    changed = True
                    if _cur_tokens() <= cap_tokens:
                        break
            if not changed:
                break

        if _cur_tokens() <= cap_tokens:
            return out

        for table, col in reversed(col_order):
            if _cur_tokens() <= cap_tokens:
                break
            entry = out[table][col]
            if isinstance(entry, list) and len(entry) >= 4 and isinstance(entry[3], list) and entry[3]:
                entry[3] = []

        if _cur_tokens() <= cap_tokens:
            return out

        out["truncated_hard"] = True
        drop_order = [k for k in reversed(col_order) if k not in essential_cols] + [
            k for k in reversed(col_order) if k in essential_cols
        ]
        for table, col in drop_order:
            if _cur_tokens() <= cap_tokens:
                break
            if table in out and isinstance(out[table], dict) and col in out[table]:
                del out[table][col]
                if not out[table]:
                    del out[table]

        if _cur_tokens() <= cap_tokens:
            return out

    minimal = "\n" + json.dumps({"truncated_hard": True}, ensure_ascii=False, separators=(",", ":"))
    if counter(minimal) <= cap_tokens:
        out["_hard_rendered"] = minimal
        return out
    if cap_tokens <= 0 or counter("") > cap_tokens:
        out["_hard_rendered"] = ""
        return out
    lo, hi = 0, len(minimal)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if counter(minimal[:mid]) <= cap_tokens:
            lo = mid
        else:
            hi = mid - 1
    out["_hard_rendered"] = minimal[:lo]
    return out


def profile_content_stats(prof_obj: Dict[str, Any]) -> Dict[str, int]:
    """Count tables, columns, value-bearing columns, and retained frequent values."""
    n_tables = 0
    n_cols = 0
    n_val_cols = 0
    n_val_items = 0
    for tbl, cols in prof_obj.items():
        if not isinstance(cols, dict):
            continue
        n_tables += 1
        for arr in cols.values():
            n_cols += 1
            if isinstance(arr, list) and len(arr) >= 5:
                vals = arr[4]
                if isinstance(vals, list) and len(vals) > 0:
                    n_val_cols += 1
                    n_val_items += len(vals)
    return {
        "n_tables": n_tables,
        "n_cols": n_cols,
        "n_val_cols": n_val_cols,
        "n_val_items": n_val_items,
    }


def truncate_profile_tightbpe(
    src_json: Path,
    src_db: Path,
    out_json: Path,
    bpe_len_fn: Callable[[str], int],
    *,
    mode: str = "drop_values_first",
) -> Dict[str, Any]:
    """Deterministically truncate a clean DP profile to Cap_TightBPE(S_d) = 100 + 10*|T_d| + 12*|C_d|."""
    prof_obj = json.loads(src_json.read_text(encoding="utf-8"))
    with ro_conn(src_db) as con:
        schema = sr.extract_public_schema(con)
    cap = compute_tightbpe_cap(schema)
    orig_tok = bpe_len_fn(prof_mod.render_profile_prompt_block(prof_obj))
    capped = truncate_profile_to_public_cap(
        prof_obj,
        cap,
        schema,
        token_counter=bpe_len_fn,
        mode=mode,
    )
    final_tok = bpe_len_fn(prof_mod.render_profile_prompt_block(capped))
    assert final_tok <= cap, (final_tok, cap)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(capped, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    out_json.write_bytes(payload)
    stats = profile_content_stats(capped)
    return {
        "mode": mode,
        "cap_tightbpe": cap,
        "orig_bpe": orig_tok,
        "final_bpe": final_tok,
        "sha256": hashlib.sha256(payload).hexdigest(),
        **stats,
    }
