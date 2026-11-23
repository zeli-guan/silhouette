"""Baseline 4: DAWA (Li et al., PVLDB 2014) + DP-SIPS (Swanberg et al., PoPETs 2023).

Classic 1D DP histogram + partition selection baseline serialized into the identical 5-field
column JSON format as Silhouette, under exact (eps_T=8, delta=1e-6) per table:
  - Uniform budget allocation across row_count + all columns (NO schema-role KKT weights,
    NO key-column exclusion).
  - Numeric columns: DAWA 1D DP L1-partitioning (1/3 bins rho) + uniform-expansion
    partition counts (2/3 bins rho) over the fixed public EDGES grid.
  - Categorical / Text columns: DP-SIPS exact Discrete Gaussian partition selection
    with threshold tau calibrated to Pr[1 + Z > tau] <= delta_j.
  - Also supports TightBPE truncation (Cap_TightBPE = 100 + 10*|T_d| + 12*|C_d|) with Qwen2.5 BPE.
"""
from __future__ import annotations

from collections import Counter
import copy
from fractions import Fraction
import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

_SRC_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SRC_DIR.parents[1]
for _p in (str(_SRC_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dp_core import (
    EHR_PUBLIC_TABLES,
    RHO_TABLE_8,
    TABLE_DELTA,
    SHA256CounterRNG,
    exact_tau_for_sensitivity,
    ro_conn,
    sigma2_for_zcdp,
)
from silhouette import dg_sampler as dg
from silhouette import profile as prof_mod
from silhouette import schema_roles as sr


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

    # Restrict DP to m <= 64 by pre-grouping sparse regions if needed (still pure post-processing)
    # In practice, active surviving bins <= 64, or we group into <= 64 contiguous chunks first.
    if m > 64:
        step = math.ceil(m / 64)
        chunks = [(i, min(m - 1, i + step - 1)) for i in range(0, m, step)]
        chunk_vals = np.array([noisy_counts[a : b + 1].sum() for a, b in chunks], dtype=np.float64)
        sub_parts = dawa_1d_l1_partition(chunk_vals, sigma_est)
        return [(chunks[a][0], chunks[b][1]) for a, b in sub_parts]

    prefix = np.zeros(m + 1, dtype=np.float64)
    prefix[1:] = np.cumsum(noisy_counts)

    # Precompute bucket cost(l, r) = sum_{k=l..r} |c_k - mean| + sigma_est * sqrt(2/pi)
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


def profile_table_dawa_dpsips(
    con: sqlite3.Connection,
    table: str,
    col_metas: Tuple[sr.ColumnMeta, ...],
    rng: SHA256CounterRNG,
    *,
    epsilon: Fraction = Fraction(8),
    delta: Fraction = TABLE_DELTA,
) -> Tuple[Dict[str, List[Any]], Dict[str, Any]]:
    """Profile one table using uniform budget DAWA (numeric) + DP-SIPS (categorical/text)."""
    delta_half = delta / 2
    rho_table = dg.rational_rho_budget(epsilon, delta_half)

    # Uniform query list across row_count + ALL columns (no role weights, no key exclusion)
    queries: List[Tuple[str, Optional[str]]] = [("row_count", None)]
    for cm in col_metas:
        queries.append(("null_count", cm.name))
        queries.append(("distinct_count", cm.name))
        if sr.numeric(cm.declared_type):
            queries.append(("bins", cm.name))
        else:
            queries.append(("patterns", cm.name))
        queries.append(("values", cm.name))

    n_q = len(queries)
    hist_queries = [q for q in queries if q[0] in prof_mod.HISTOGRAMS]
    n_hist = len(hist_queries)
    rho_per_q = rho_table / n_q
    delta_per_hist = delta_half / n_hist

    # Exact rational budget check
    assert rho_per_q * n_q <= rho_table
    assert delta_per_hist * n_hist <= delta_half

    sigma2_scalar = sigma2_for_zcdp(rho_per_q, 1)
    # DAWA splits bin query into 1/3 for L1 partition selection + 2/3 for partition counts
    rho_dawa_1 = rho_per_q * Fraction(1, 3)
    rho_dawa_2 = rho_per_q - rho_dawa_1
    sigma2_dawa_1 = sigma2_for_zcdp(rho_dawa_1, 1)
    sigma2_dawa_2 = sigma2_for_zcdp(rho_dawa_2, 1)
    tau_dawa = exact_tau_for_sensitivity(sigma2_dawa_1, delta_per_hist, l1_max_step=1)

    # DP-SIPS uses full rho_per_q with threshold tau calibrated to delta_per_hist
    sigma2_sips = sigma2_for_zcdp(rho_per_q, 1)
    tau_sips = exact_tau_for_sensitivity(sigma2_sips, delta_per_hist, l1_max_step=1)

    sel_cols = ", ".join(sr.quote(cm.name) for cm in col_metas)
    all_rows = con.execute(f"SELECT {sel_cols} FROM {sr.quote(table)}").fetchall()
    nrows = len(all_rows)
    n_noisy = max(0, nrows + dg.sample_dgauss(sigma2_scalar, rng))
    out: Dict[str, List[Any]] = {}

    for col_i, cm in enumerate(col_metas):
        raw_cnts = Counter(r[col_i] for r in all_rows)
        null_cnt = raw_cnts.pop(None, 0)
        distinct_cnt = len(raw_cnts)
        nn = max(0, int(null_cnt) + dg.sample_dgauss(sigma2_scalar, rng))
        nd = max(0, int(distinct_cnt) + dg.sample_dgauss(sigma2_scalar, rng))
        dist_cnts: Counter = Counter()
        if sr.numeric(cm.declared_type):
            for v, c in raw_cnts.items():
                dist_cnts[prof_mod.bin_index(v)] += c
        else:
            for v, c in raw_cnts.items():
                dist_cnts[prof_mod.shape(v)] += c
        st = {
            "distribution_sorted": sorted(dist_cnts.items(), key=lambda x: repr(x[0])),
            "values_sorted": None,
        }
        if nd <= 50:
            v_cnts: Counter = Counter()
            for v, c in raw_cnts.items():
                tag = "integer" if isinstance(v, int) and not isinstance(v, bool) else ("real" if isinstance(v, float) else ("blob" if isinstance(v, bytes) else "text"))
                v_cnts[prof_mod.canonical_val_key(tag, v)] += c
            st["values_sorted"] = sorted(v_cnts.items(), key=lambda x: repr(x[0]))
        null_frac = round(min(1.0, nn / max(1, n_noisy)), 3)
        order = prof_mod.order_of_magnitude(nd)

        if sr.numeric(cm.declared_type):
            # Step 1 (DAWA): threshold & noisy counts on populated bins (using rho_dawa_1, delta_per_hist)
            surv_bins: List[Tuple[int, int]] = []
            for b, cnt in st["distribution_sorted"]:
                if not isinstance(b, int) or b < 0:
                    continue
                noisy = int(cnt) + dg.sample_dgauss(sigma2_dawa_1, rng)
                if noisy > tau_dawa:
                    surv_bins.append((b, noisy))
            surv_bins.sort(key=lambda x: x[0])
            if surv_bins:
                bin_ids = [b for b, _ in surv_bins]
                noisy_arr = np.array([c for _, c in surv_bins], dtype=np.float64)
                parts = dawa_1d_l1_partition(noisy_arr, math.sqrt(float(sigma2_dawa_2)))
                # Step 2 (DAWA): measure exact partition counts with sigma2_dawa_2 (disjoint partitions => sens=1)
                exact_arr = [dict(st["distribution_sorted"]).get(b, 0) for b in bin_ids]
                reconstructed: Dict[int, float] = {}
                for l_idx, r_idx in parts:
                    true_sum = sum(exact_arr[l_idx : r_idx + 1])
                    noisy_sum = max(0.0, float(true_sum + dg.sample_dgauss(sigma2_dawa_2, rng)))
                    width = r_idx - l_idx + 1
                    per_bin = noisy_sum / width
                    if per_bin > 0:
                        for k in range(l_idx, r_idx + 1):
                            reconstructed[bin_ids[k]] = per_bin
                dist = prof_mod.quantiles(reconstructed)
            else:
                dist = [None, None, None]
        else:
            # DP-SIPS pattern histogram
            pat_hist: Dict[Any, int] = {}
            for pat, cnt in st["distribution_sorted"]:
                noisy = int(cnt) + dg.sample_dgauss(sigma2_sips, rng)
                if noisy > tau_sips:
                    pat_hist[pat] = noisy
            dist = prof_mod.top_keys(pat_hist, 3)

        # DP-SIPS value partition selection (applied uniformly to ALL columns, capped at 15 if nd <= 50)
        common = None
        if nd <= 50:
            vh: Dict[Any, int] = {}
            for val_k, cnt in st["values_sorted"]:
                noisy = int(cnt) + dg.sample_dgauss(sigma2_sips, rng)
                if noisy > tau_sips:
                    vh[val_k] = noisy
            common = [prof_mod.decode_val_key(k) for k in prof_mod.top_keys(vh, 15)]

        out[cm.name] = [cm.declared_type, null_frac, order, dist, common]

    ledger = {
        "mechanism": "DAWA+DP-SIPS",
        "rho_used": float(rho_per_q * n_q),
        "rho_budget": float(rho_table),
        "delta_tail_used": float(delta_per_hist * n_hist),
        "epsilon_upper": str(dg.conservative_eps_from_rho(rho_per_q * n_q, delta_half)),
        "n_queries": n_q,
        "n_hist": n_hist,
    }
    return out, ledger


def profile_db_dawa_dpsips(
    src_db: Path,
    out_json: Path,
    seed: int,
    *,
    is_ehr: bool = False,
) -> Dict[str, Any]:
    """Generate 5-field DP profile JSON using DAWA + DP-SIPS under (eps_T=8, delta=1e-6)."""
    exclude = EHR_PUBLIC_TABLES if is_ehr else frozenset()
    with ro_conn(src_db) as con:
        schema = sr.extract_public_schema(con)
        profile: Dict[str, Any] = {}
        ledgers: Dict[str, Any] = {}
        for table, cols in schema.tables.items():
            if table in exclude:
                continue
            rng = SHA256CounterRNG(f"DAWA_DPSIPS|{seed}|{src_db.stem}|{table}")
            t_prof, t_led = profile_table_dawa_dpsips(con, table, cols, rng)
            profile[table] = t_prof
            ledgers[table] = t_led

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(profile, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    return {"baseline": "DAWA+DP-SIPS", "seed": seed, "tables": ledgers}


def truncate_profile_tightbpe(
    src_json: Path,
    src_db: Path,
    out_json: Path,
    bpe_len_fn: Callable[[str], int],
) -> Dict[str, Any]:
    """Truncate profile JSON to Stage 8/10 Cap_TightBPE(S_d) = 100 + 10*|T_d| + 12*|C_d|."""
    prof_obj = json.loads(src_json.read_text(encoding="utf-8"))
    with ro_conn(src_db) as con:
        schema = sr.extract_public_schema(con)
    T = len(schema.tables)
    C = sum(len(v) for v in schema.tables.values())
    cap = 100 + 10 * T + 12 * C

    orig_tok = bpe_len_fn(prof_mod.render_profile_prompt_block(prof_obj))
    capped = prof_mod.truncate_profile_to_public_cap(
        prof_obj, cap, schema, token_counter=bpe_len_fn
    )
    final_tok = bpe_len_fn(prof_mod.render_profile_prompt_block(capped))
    assert final_tok <= cap, (final_tok, cap)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(capped, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    return {"cap_tightbpe": cap, "orig_bpe": orig_tok, "final_bpe": final_tok}
