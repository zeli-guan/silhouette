"""Baseline 1: AIM (McKenna et al., PVLDB 2022) Workload-Adaptive Graphical Model Synthesis.

Implements exact zCDP (eps_T=8, delta=1e-6 -> rho_T=0.8748, delta_tail=0.5e-6):
  - 5% rho_T: noisy table row count n_tilde
  - 25% rho_T + delta/2: DP domain discovery via exact Discrete Gaussian + tau_j
  - 20% rho_T: DP 1-way marginals for all active-domain columns
  - 50% rho_T: AIM adaptive rounds (zCDP Exponential Mechanism marginal selection,
    Discrete Gaussian 2-way measurement, budget annealing, IPF graphical model fitting)
  - Optional post-processing: PubCap AIM static rows (zero extra privacy cost)
"""
from __future__ import annotations

from contextlib import closing
from fractions import Fraction
import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

_SRC_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SRC_DIR.parents[1]
for _p in (str(_SRC_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from baseline_privlava import _resolve_fks_and_constraints
from dp_core import (
    EHR_PUBLIC_TABLES,
    RHO_TABLE_8,
    TABLE_DELTA,
    TABLE_DELTA_HALF,
    TABLE_EPSILON,
    FactoredPairwisePGM,
    SHA256CounterRNG,
    decode_matrix_to_columns,
    dp_discover_table_domains,
    encode_rows_to_matrix,
    generate_sequential_pk,
    inspect_schema_and_dag,
    is_sequential_pk,
    materialize_sqlite_table,
    ro_conn,
    sigma2_for_zcdp,
)
from silhouette import dg_sampler as dg
from silhouette import profile as prof_mod
from silhouette import schema_roles as sr


def _noisy_1way(col_vals: np.ndarray, size: int, sigma2: Fraction, rng: SHA256CounterRNG) -> np.ndarray:
    counts = np.bincount(col_vals, minlength=size).astype(np.float64)
    for b in range(size):
        counts[b] += dg.sample_dgauss(sigma2, rng)
    return np.maximum(0.0, counts)


def _noisy_2way(
    ci: np.ndarray, cj: np.ndarray, si: int, sj: int, sigma2: Fraction, rng: SHA256CounterRNG
) -> np.ndarray:
    idx = ci.astype(np.int64) * sj + cj.astype(np.int64)
    counts = np.bincount(idx, minlength=si * sj).reshape(si, sj).astype(np.float64)
    for a in range(si):
        for b in range(sj):
            counts[a, b] += dg.sample_dgauss(sigma2, rng)
    return np.maximum(0.0, counts)


def _exact_2way(ci: np.ndarray, cj: np.ndarray, si: int, sj: int) -> np.ndarray:
    idx = ci.astype(np.int64) * sj + cj.astype(np.int64)
    return np.bincount(idx, minlength=si * sj).reshape(si, sj).astype(np.float64)


def synthesize_table_aim(
    rows: List[Tuple[Any, ...]],
    col_metas: Tuple[sr.ColumnMeta, ...],
    pk_cols: List[str],
    rng: SHA256CounterRNG,
    *,
    rho_table: Fraction = RHO_TABLE_8,
    delta_table: Fraction = TABLE_DELTA,
    max_rounds_cap: int = 24,
    max_pair_dom: int = 1024,
) -> Tuple[Dict[str, List[Any]], int, Dict[str, Any]]:
    """Synthesize a single table using AIM under exact zCDP (rho_table, delta_table/2)."""
    delta_half = delta_table / 2
    rho_cnt = rho_table * Fraction(5, 100)
    rho_dom = rho_table * Fraction(25, 100)
    rho_1way = rho_table * Fraction(20, 100)
    rho_aim = rho_table - rho_cnt - rho_dom - rho_1way
    assert rho_cnt + rho_dom + rho_1way + rho_aim <= rho_table

    # 1. Noisy row count
    sigma2_cnt = sigma2_for_zcdp(rho_cnt, 1)
    n_true = len(rows)
    n_syn = max(0, n_true + dg.sample_dgauss(sigma2_cnt, rng))

    seq_pks = {cm.name for cm in col_metas if is_sequential_pk(cm, pk_cols)}
    domains = dp_discover_table_domains(
        rows,
        col_metas,
        seq_pks,
        rho_dom,
        delta_half,
        rng,
        max_cat_bins=32,
        max_num_bins=16,
    )
    modeled_cols = [cm.name for cm in col_metas if cm.name in domains]
    col_idx = {cm.name: i for i, cm in enumerate(col_metas)}

    if not modeled_cols or n_syn == 0:
        col_data: Dict[str, List[Any]] = {}
        for cm in col_metas:
            if cm.name in seq_pks:
                col_data[cm.name] = generate_sequential_pk(cm.declared_type, n_syn)
            else:
                col_data[cm.name] = [None] * n_syn
        return col_data, n_syn, {
            "rho_used": float(rho_cnt + rho_dom),
            "rho_budget": float(rho_table),
            "rounds_run": 0,
            "n_true": n_true,
            "n_syn": n_syn,
        }

    mat = encode_rows_to_matrix(rows, modeled_cols, col_idx, domains)
    sizes = [domains[c]["size"] for c in modeled_cols]
    d = len(modeled_cols)
    pgm = FactoredPairwisePGM(sizes)

    # 2. Measure all 1-way marginals
    rho_1w_per = rho_1way / d
    sigma2_1w = sigma2_for_zcdp(rho_1w_per, 1)
    for k in range(d):
        noisy_1w = _noisy_1way(mat[:, k], sizes[k], sigma2_1w, rng)
        pgm.set_1way(k, noisy_1w)

    # 3. AIM iterative marginal selection + measurement + annealing
    np_rng = rng.derive_np_rng("aim_select")
    candidates: List[Tuple[int, int]] = [
        (i, j)
        for i in range(d)
        for j in range(i + 1, d)
        if sizes[i] * sizes[j] <= max_pair_dom
    ]
    rho_used_aim = Fraction(0)
    rounds_run = 0
    if candidates and d >= 2 and n_true > 0:
        max_rounds = min(max_rounds_cap, max(4, 2 * d))
        rho_round = rho_aim / max_rounds
        rho_rem = rho_aim
        measured_pairs: Set[Tuple[int, int]] = set()
        terminate = False

        while not terminate and len(measured_pairs) < len(candidates):
            rounds_run += 1
            if rho_rem < 2 * rho_round or rounds_run >= max_rounds:
                rho_round = rho_rem
                terminate = True
            rho_rem -= rho_round
            rho_used_aim += rho_round

            rho_sel = rho_round * Fraction(1, 10)
            rho_meas = rho_round - rho_sel
            sigma2_meas = sigma2_for_zcdp(rho_meas, 1)
            sigma_meas = math.sqrt(float(sigma2_meas))
            # Exact zCDP -> EM calibration: eps_em = sqrt(8 * rho_sel), sensitivity of ||M(D)-M(P)||_1 is 2 (add/remove 1 row changes L1 by <=1; replace 1 row <=2; use Delta=1 for add/remove)
            eps_em = math.sqrt(8.0 * float(rho_sel))

            # Score unmeasured candidates (cap scan to 120 pairs for very wide tables)
            unmeasured = [c for c in candidates if c not in measured_pairs]
            if not unmeasured:
                break
            if len(unmeasured) > 120:
                idx_sel = np_rng.choice(len(unmeasured), size=120, replace=False)
                pool = [unmeasured[int(k)] for k in idx_sel]
            else:
                pool = unmeasured

            scores = np.zeros(len(pool), dtype=np.float64)
            n_scale = max(1.0, float(n_syn))
            for idx_p, (i, j) in enumerate(pool):
                true_m = _exact_2way(mat[:, i], mat[:, j], sizes[i], sizes[j])
                est_m = pgm.project_2way(i, j) * n_scale
                l1_err = float(np.abs(true_m - est_m).sum())
                bias = sigma_meas * math.sqrt(2.0 / math.pi) * (sizes[i] * sizes[j])
                scores[idx_p] = l1_err - bias

            # Exponential Mechanism (sensitivity = 1 under add/remove)
            scaled = 0.5 * eps_em * (scores - scores.max())
            probs = np.exp(np.clip(scaled, -50.0, 0.0))
            probs /= probs.sum()
            chosen_idx = int(np_rng.choice(len(pool), p=probs))
            ci, cj = pool[chosen_idx]
            measured_pairs.add((ci, cj))

            old_est = pgm.project_2way(ci, cj) * n_scale
            noisy_m = _noisy_2way(mat[:, ci], mat[:, cj], sizes[ci], sizes[cj], sigma2_meas, rng)
            pgm.add_2way(ci, cj, noisy_m)
            pgm.fit_ipf(iters=10)
            new_est = pgm.project_2way(ci, cj) * n_scale

            # AIM budget annealing check
            thresh = sigma_meas * math.sqrt(2.0 / math.pi) * (sizes[ci] * sizes[cj])
            if float(np.abs(new_est - old_est).sum()) <= thresh and not terminate:
                rho_round = min(rho_rem, rho_round * 2)

    # 4. Sample synthetic rows
    syn_mat = pgm.sample(n_syn, np_rng)
    decoded = decode_matrix_to_columns(syn_mat, modeled_cols, domains, np_rng)
    col_data = {}
    for cm in col_metas:
        if cm.name in seq_pks:
            col_data[cm.name] = generate_sequential_pk(cm.declared_type, n_syn)
        else:
            col_data[cm.name] = decoded[cm.name]

    total_rho_used = rho_cnt + rho_dom + rho_1way + rho_used_aim
    assert total_rho_used <= rho_table + Fraction(1, 10**12)
    return col_data, n_syn, {
        "rho_used": float(total_rho_used),
        "rho_budget": float(rho_table),
        "epsilon_upper": str(dg.conservative_eps_from_rho(min(total_rho_used, rho_table), delta_half)),
        "rounds_run": rounds_run,
        "n_true": n_true,
        "n_syn": n_syn,
    }


def synthesize_db_aim(
    src_db: Path,
    out_sqlite: Path,
    seed: int,
    *,
    is_ehr: bool = False,
) -> Dict[str, Any]:
    """Synthesize a full SQLite database using AIM under per-table (eps=8, delta=1e-6)."""
    exclude = EHR_PUBLIC_TABLES if is_ehr else frozenset()
    meta = inspect_schema_and_dag(src_db, exclude_tables=exclude)

    out_sqlite.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_sqlite.with_name(f".tmp_{out_sqlite.name}")
    if tmp.exists():
        tmp.unlink()

    report: Dict[str, Any] = {"baseline": "AIM", "seed": seed, "tables": {}}
    with ro_conn(src_db) as src_con, closing(sqlite3.connect(str(tmp))) as dst_con:
        dst_con.execute("PRAGMA temp_store=MEMORY")
        for t in meta["tables"]:
            qt = sr.quote(t)
            cols_meta = meta["col_metas"][t]
            sel = ", ".join(sr.quote(cm.name) for cm in cols_meta)
            rows = src_con.execute(f"SELECT {sel} FROM {qt}").fetchall()
            rng = SHA256CounterRNG(f"AIM|{seed}|{src_db.stem}|{t}")
            col_data, n_syn, t_rep = synthesize_table_aim(rows, cols_meta, meta["pk_cols"][t], rng)
            col_data, n_syn = _resolve_fks_and_constraints(
                t, col_data, n_syn, meta["table_info"][t], meta["pk_cols"][t], set(meta["unique_cols"][t]),
                [], [], {}, None, None, rng.derive_np_rng("aim_post"),
            )
            materialize_sqlite_table(
                dst_con,
                t,
                meta["table_info"][t],
                col_data,
                n_syn,
                pk_cols=meta["pk_cols"][t],
                unique_cols=set(meta["unique_cols"][t]),
            )
            report["tables"][t] = t_rep

        if is_ehr:
            # Copy the 4 public dictionary tables verbatim
            for pt in sorted(EHR_PUBLIC_TABLES):
                qpt = sr.quote(pt)
                pinfo = src_con.execute(f"PRAGMA table_info({qpt})").fetchall()
                cnames = [c[1] for c in pinfo]
                cdefs = [
                    f"{sr.quote(c[1])} {c[2] or 'ANY'}" + (" PRIMARY KEY" if c[5] else "")
                    for c in pinfo
                ]
                dst_con.execute(f"CREATE TABLE {qpt} ({', '.join(cdefs)})")
                prows = src_con.execute(f"SELECT {', '.join(sr.quote(c) for c in cnames)} FROM {qpt}").fetchall()
                dst_con.executemany(
                    f"INSERT INTO {qpt} VALUES ({','.join('?' for _ in cnames)})",
                    prows,
                )
            dst_con.commit()
        dst_con.commit()
        assert dst_con.execute("PRAGMA quick_check").fetchone()[0] == "ok"

    tmp.replace(out_sqlite)
    return report


def render_pubcap_aim_rows(src_db: Path, aim_sqlite: Path, out_txt: Path) -> Dict[str, Any]:
    """Deterministically render PubCap static rows from D_AIM (zero extra privacy cost)."""
    with ro_conn(src_db) as con:
        schema = sr.extract_public_schema(con)
    cap = prof_mod.compute_public_profile_token_cap(schema)
    rows_txt = prof_mod.truncate_synth_rows_to_public_cap(aim_sqlite, cap, schema)
    tok = prof_mod.estimate_prompt_tokens(rows_txt)
    assert tok <= cap, (tok, cap)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(rows_txt, encoding="utf-8")
    return {"cap_tokens": cap, "actual_tokens": tok}
