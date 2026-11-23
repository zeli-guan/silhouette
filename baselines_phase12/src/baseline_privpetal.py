"""Baseline 3: PrivPetal (Cai, Xiao, Yang, SIGMOD / PACMMOD 2025).

Relational Data Synthesis via Permutation Relations under exact (eps_T=8, delta=1e-6) per table.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import closing
from fractions import Fraction
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

from baseline_privlava import (
    K_LATENT,
    _cluster_rows_deterministic,
    _noisy_1way,
    _noisy_2way,
    _parent_cluster_map_from_real,
    _resolve_fks_and_constraints,
    _synthesize_privlava_table,
)
from dp_core import (
    EHR_PUBLIC_TABLES,
    RHO_TABLE_8,
    TABLE_DELTA,
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
from silhouette import schema_roles as sr

S_MAX_PERM = 8


def _synthesize_privpetal_child(
    rows: List[Tuple[Any, ...]],
    col_metas: Tuple[sr.ColumnMeta, ...],
    pk_cols: List[str],
    primary_fk: Tuple[str, str, str],
    real_parent_cluster_map: Dict[Any, int],
    syn_parent_pks_by_cluster: Dict[int, List[Any]],
    rng: SHA256CounterRNG,
    *,
    rho_table: Fraction = RHO_TABLE_8,
    delta_table: Fraction = TABLE_DELTA,
) -> Tuple[Dict[str, List[Any]], int, List[Any], Dict[str, Any]]:
    delta_half = delta_table / 2
    fk_col = primary_fk[0]
    col_idx = {cm.name: i for i, cm in enumerate(col_metas)}
    fk_idx = col_idx[fk_col]
    seq_pks = {cm.name for cm in col_metas if is_sequential_pk(cm, pk_cols)}
    np_rng = rng.derive_np_rng("privpetal")

    rho_dom = rho_table * Fraction(20, 100)
    rho_gsize = rho_table * Fraction(20, 100)
    rho_cross = rho_table * Fraction(25, 100)
    rho_sibling = rho_table * Fraction(20, 100)
    rho_int = rho_table - rho_dom - rho_gsize - rho_cross - rho_sibling
    assert rho_dom + rho_gsize + rho_cross + rho_sibling + rho_int <= rho_table

    skip_cols = set(seq_pks) | {fk_col}
    domains = dp_discover_table_domains(rows, col_metas, skip_cols, rho_dom, delta_half, rng)
    modeled = [cm.name for cm in col_metas if cm.name in domains]

    groups_by_fk: Dict[Any, List[int]] = defaultdict(list)
    null_fk_indices: List[int] = []
    for i, r in enumerate(rows):
        fv = r[fk_idx]
        if fv is None:
            null_fk_indices.append(i)
        else:
            groups_by_fk[fv].append(i)

    # Split rho_gsize: 80% group-size hist (Delta_2=sqrt(2)), 10% total nonnull rows (Delta_2=1), 10% null-fk count (Delta_2=1)
    rho_gs_main = rho_gsize * Fraction(8, 10)
    rho_gs_tot = rho_gsize * Fraction(1, 10)
    rho_gs_null = rho_gsize - rho_gs_main - rho_gs_tot
    sigma2_gs_main = Fraction(1, 1) / rho_gs_main  # Delta_2^2 = 2 => sigma2 = 2 / (2 * rho) = 1 / rho
    sigma2_gs_tot = sigma2_for_zcdp(rho_gs_tot, 1)
    sigma2_gs_null = sigma2_for_zcdp(rho_gs_null, 1)

    gsize_hist = np.zeros((K_LATENT, S_MAX_PERM), dtype=np.float64)
    for fv, idxs in groups_by_fk.items():
        z = real_parent_cluster_map.get(fv, 0)
        sb = min(S_MAX_PERM, len(idxs)) - 1
        gsize_hist[z, sb] += 1.0
    for z in range(K_LATENT):
        for s in range(S_MAX_PERM):
            gsize_hist[z, s] = max(0.0, gsize_hist[z, s] + dg.sample_dgauss(sigma2_gs_main, rng))

    n_nonnull_target = max(0, (len(rows) - len(null_fk_indices)) + dg.sample_dgauss(sigma2_gs_tot, rng))
    n_null_fk_syn = max(0, len(null_fk_indices) + dg.sample_dgauss(sigma2_gs_null, rng))

    syn_fk_vals: List[Any] = []
    syn_z_list: List[int] = []
    group_slots: List[Tuple[int, int]] = []
    overflow_groups: List[Tuple[int, Any, int]] = []  # (gid, pk_val, z)
    gid = 0
    for z in range(K_LATENT):
        pks_z = syn_parent_pks_by_cluster.get(z, [])
        if not pks_z:
            continue
        n_active_est = min(len(pks_z), max(0, int(round(float(gsize_hist[z].sum())))))
        if n_active_est == 0:
            continue
        probs_s = gsize_hist[z]
        ps_sum = float(probs_s.sum())
        if ps_sum <= 0.0:
            continue
        probs_s = probs_s / ps_sum
        chosen_pks = np_rng.choice(len(pks_z), size=n_active_est, replace=False)
        sizes_drawn = np_rng.choice(np.arange(1, S_MAX_PERM + 1), size=n_active_est, p=probs_s)
        for pk_idx, s_len in zip(chosen_pks, sizes_drawn):
            pk_val = pks_z[int(pk_idx)]
            slen = int(s_len)
            for pos in range(slen):
                syn_fk_vals.append(pk_val)
                syn_z_list.append(z)
                group_slots.append((gid, pos))
            if slen == S_MAX_PERM:
                overflow_groups.append((gid, pk_val, z))
            gid += 1

    # Distribute heavy-tail surplus rows across overflow groups (or active groups) to match DP n_nonnull_target
    if n_nonnull_target > len(syn_fk_vals) and (overflow_groups or len(syn_fk_vals) > 0):
        surplus = n_nonnull_target - len(syn_fk_vals)
        if overflow_groups:
            targets = overflow_groups
        else:
            all_pks = [(i, syn_fk_vals[i], syn_z_list[i]) for i in range(len(syn_fk_vals))]
            targets = all_pks[: min(len(all_pks), 256)]
        if targets:
            picks = np_rng.integers(0, len(targets), size=surplus)
            slot_cnt: Counter = Counter()
            for p_i in picks:
                g_id, pk_val, z_val = targets[int(p_i)]
                pos = S_MAX_PERM + slot_cnt[g_id]
                slot_cnt[g_id] += 1
                syn_fk_vals.append(pk_val)
                syn_z_list.append(z_val)
                group_slots.append((g_id, pos))

    for _ in range(n_null_fk_syn):
        syn_fk_vals.append(None)
        syn_z_list.append(K_LATENT)
        group_slots.append((gid, 0))
        gid += 1

    n_syn = len(syn_fk_vals)
    if n_syn == 0 or not modeled:
        cdata = {}
        for cm in col_metas:
            if cm.name in seq_pks:
                cdata[cm.name] = generate_sequential_pk(cm.declared_type, n_syn)
            elif cm.name == fk_col:
                cdata[cm.name] = syn_fk_vals
            else:
                cdata[cm.name] = [None] * n_syn
        return cdata, n_syn, syn_fk_vals, {"rho_used": float(rho_dom + rho_gsize), "n_syn": n_syn}

    mat = encode_rows_to_matrix(rows, modeled, col_idx, domains)
    d = len(modeled)
    sizes = [K_LATENT + 1] + [domains[c]["size"] for c in modeled]
    pgm = FactoredPairwisePGM(sizes)

    z_arr = np.array(
        [K_LATENT if r[fk_idx] is None else real_parent_cluster_map.get(r[fk_idx], 0) for r in rows],
        dtype=np.int32,
    )
    z_counts_syn = np.bincount(np.array(syn_z_list, dtype=np.int32), minlength=K_LATENT + 1).astype(np.float64)
    pgm.set_1way(0, z_counts_syn)

    sigma2_cross = sigma2_for_zcdp(rho_cross / d, 1)
    for k in range(d):
        joint = _noisy_2way(z_arr, mat[:, k], K_LATENT + 1, sizes[k + 1], sigma2_cross, rng)
        pgm.add_2way(0, k + 1, joint)
        pgm.set_1way(k + 1, joint.sum(axis=0))

    cand_2w = [(i, j) for i in range(d) for j in range(i + 1, d) if sizes[i + 1] * sizes[j + 1] <= 512]
    if len(cand_2w) > 10:
        cand_2w = [cand_2w[int(k)] for k in np_rng.choice(len(cand_2w), size=10, replace=False)]
    n_int_q = d + len(cand_2w)
    sigma2_int = sigma2_for_zcdp(rho_int / max(1, n_int_q), 1)
    for k in range(d):
        pgm.set_1way(k + 1, _noisy_1way(mat[:, k], sizes[k + 1], sigma2_int, rng))
    for i, j in cand_2w:
        pgm.add_2way(i + 1, j + 1, _noisy_2way(mat[:, i], mat[:, j], sizes[i + 1], sizes[j + 1], sigma2_int, rng))

    pgm.fit_ipf(iters=10)

    # PrivPetal permutation-symmetric intra-group sibling transition matrix
    # Each parent group with >=2 rows contributes unit mass 1.0 across its ordered pairs (u != v)
    # => Delta_2 = sqrt(2) => sigma^2 = 1 / rho_sib_per
    sib_cols = [k for k in range(d) if sizes[k + 1] <= 64]
    sib_trans: Dict[int, np.ndarray] = {}
    if sib_cols and groups_by_fk:
        rho_sib_per = rho_sibling / len(sib_cols)
        sigma_sib = math.sqrt(float(Fraction(1, 1) / rho_sib_per))
        for k in sib_cols:
            sk = sizes[k + 1]
            acc = np.zeros((sk, sk), dtype=np.float64)
            for idxs in groups_by_fk.values():
                if len(idxs) < 2:
                    continue
                sub = mat[idxs[:S_MAX_PERM], k]
                m = len(sub)
                cnts = np.bincount(sub, minlength=sk).astype(np.float64)
                pair_mat = np.outer(cnts, cnts) - np.diag(cnts)
                acc += pair_mat / float(m * (m - 1))
            noisy_acc = np.maximum(0.0, acc + np_rng.normal(0.0, sigma_sib, size=(sk, sk)))
            noisy_acc = 0.5 * (noisy_acc + noisy_acc.T)
            row_sums = noisy_acc.sum(axis=1, keepdims=True)
            trans = np.where(row_sums > 0, noisy_acc / np.maximum(1e-12, row_sums), 1.0 / sk)
            sib_trans[k] = trans

    # Sequential auto-regressive substitution sampling per parent group
    full_out = np.zeros((n_syn, d), dtype=np.int32)
    syn_z_np = np.array(syn_z_list, dtype=np.int32)
    # Sort indices by (gid, slot_pos) so slot 0 is sampled before slot 1..
    order_by_slot = sorted(range(n_syn), key=lambda i: group_slots[i])

    # Precompute P(attr=k | z)
    cond_by_kz: Dict[Tuple[int, int], np.ndarray] = {}
    for k in range(d):
        joint = pgm.marginals_2d[(0, k + 1)]
        for z in range(K_LATENT + 1):
            p_cond = np.maximum(0.0, joint[z])
            cs = float(p_cond.sum())
            p_cond = (p_cond / cs) if cs > 0 else pgm.marginals_1d[k + 1]
            p_cond = np.maximum(0.0, p_cond)
            p_cond /= p_cond.sum()
            cond_by_kz[(k, z)] = p_cond

    for k in range(d):
        sk = sizes[k + 1]
        if k not in sib_trans:
            for z in range(K_LATENT + 1):
                mask = syn_z_np == z
                cnt = int(mask.sum())
                if cnt > 0:
                    full_out[mask, k] = np_rng.choice(sk, size=cnt, p=cond_by_kz[(k, z)])
        else:
            trans = sib_trans[k]
            prev_by_gid: Dict[int, int] = {}
            for r_i in order_by_slot:
                g_id, pos = group_slots[r_i]
                z = int(syn_z_np[r_i])
                p_cond = cond_by_kz[(k, z)]
                if pos == 0 or g_id not in prev_by_gid:
                    val = int(np_rng.choice(sk, p=p_cond))
                else:
                    prev_v = prev_by_gid[g_id]
                    blended = 0.6 * trans[prev_v] + 0.4 * p_cond
                    blended = np.maximum(0.0, blended)
                    blended /= blended.sum()
                    val = int(np_rng.choice(sk, p=blended))
                full_out[r_i, k] = val
                prev_by_gid[g_id] = val

    dec = decode_matrix_to_columns(full_out, modeled, domains, np_rng)
    cdata = {}
    for cm in col_metas:
        if cm.name in seq_pks:
            cdata[cm.name] = generate_sequential_pk(cm.declared_type, n_syn)
        elif cm.name == fk_col:
            cdata[cm.name] = syn_fk_vals
        else:
            cdata[cm.name] = dec[cm.name]
    return cdata, n_syn, syn_fk_vals, {"rho_used": float(rho_table), "n_syn": n_syn}


def synthesize_db_privpetal(
    src_db: Path,
    out_sqlite: Path,
    seed: int,
    *,
    is_ehr: bool = False,
) -> Dict[str, Any]:
    exclude = EHR_PUBLIC_TABLES if is_ehr else frozenset()
    meta = inspect_schema_and_dag(src_db, exclude_tables=exclude)

    edges_by_child: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    for st, sc, dt, dc in meta["dag_edges"]:
        edges_by_child[st].append((sc, dt, dc))
    self_by_table: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for st, sc, _dt, dc in meta["self_loops"]:
        self_by_table[st].append((sc, dc))

    out_sqlite.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_sqlite.with_name(f".tmp_{out_sqlite.name}")
    if tmp.exists():
        tmp.unlink()

    syn_cols_by_table: Dict[str, Dict[str, List[Any]]] = {}
    syn_clusters_by_table: Dict[str, List[int]] = {}
    report: Dict[str, Any] = {"baseline": "PrivPetal", "seed": seed, "tables": {}}

    with ro_conn(src_db) as src_con, closing(sqlite3.connect(str(tmp))) as dst_con:
        dst_con.execute("PRAGMA temp_store=MEMORY")
        dst_con.execute("PRAGMA foreign_keys=OFF")

        for t in meta["topo_order"]:
            qt = sr.quote(t)
            cols_meta = meta["col_metas"][t]
            pk_cols = meta["pk_cols"][t]
            seq_pks = {cm.name for cm in cols_meta if is_sequential_pk(cm, pk_cols)}
            np_rng = SHA256CounterRNG(f"PP_POST|{seed}|{src_db.stem}|{t}").derive_np_rng("post")

            sel = ", ".join(sr.quote(cm.name) for cm in cols_meta)
            rows = src_con.execute(f"SELECT {sel} FROM {qt}").fetchall()
            valid_fks = [
                (sc, dt, dc) for (sc, dt, dc) in edges_by_child[t]
                if dt in syn_cols_by_table and len(syn_cols_by_table[dt].get(dc, [])) > 0
            ]
            primary_fk = valid_fks[0] if valid_fks else None

            rng = SHA256CounterRNG(f"PRIVPETAL|{seed}|{src_db.stem}|{t}")
            if primary_fk is None:
                cdata, n_syn, syn_fk_vals, t_rep = _synthesize_privlava_table(
                    rows, cols_meta, pk_cols, None, None, None, rng,
                    rho_table=RHO_TABLE_8, delta_table=TABLE_DELTA,
                )
            else:
                sc, ptbl, pcol = primary_fk
                p_seq_pks = {cm.name for cm in meta["col_metas"][ptbl] if is_sequential_pk(cm, meta["pk_cols"][ptbl])}
                real_cluster_map = _parent_cluster_map_from_real(
                    src_con, ptbl, pcol, meta["col_metas"][ptbl], p_seq_pks, K_LATENT
                )
                syn_pks_by_cluster: Dict[int, List[Any]] = defaultdict(list)
                for pv, cz in zip(syn_cols_by_table[ptbl][pcol], syn_clusters_by_table[ptbl]):
                    if pv is not None:
                        syn_pks_by_cluster[cz].append(pv)

                cdata, n_syn, syn_fk_vals, t_rep = _synthesize_privpetal_child(
                    rows, cols_meta, pk_cols, primary_fk, real_cluster_map, syn_pks_by_cluster, rng,
                    rho_table=RHO_TABLE_8, delta_table=TABLE_DELTA,
                )

            cdata, n_syn = _resolve_fks_and_constraints(
                t, cdata, n_syn, meta["table_info"][t], pk_cols, set(meta["unique_cols"][t]),
                valid_fks, self_by_table[t], syn_cols_by_table,
                primary_fk[0] if primary_fk else None, syn_fk_vals, np_rng,
            )

            syn_cols_by_table[t] = cdata
            syn_clusters_by_table[t] = _cluster_rows_deterministic(cdata, n_syn, seq_pks, K_LATENT)

            fk_clauses: List[str] = []
            col_fk_cnt = Counter(sc for sc, _, _ in valid_fks)
            for sc, dt, dc in valid_fks:
                if col_fk_cnt[sc] == 1 and dc in meta["unique_cols"][dt]:
                    fk_clauses.append(
                        f"FOREIGN KEY ({sr.quote(sc)}) REFERENCES {sr.quote(dt)} ({sr.quote(dc)})"
                    )

            materialize_sqlite_table(
                dst_con, t, meta["table_info"][t], cdata, n_syn,
                pk_cols=pk_cols, unique_cols=set(meta["unique_cols"][t]),
                fk_clauses=fk_clauses,
            )
            t_rep["n_syn_final"] = n_syn
            report["tables"][t] = t_rep

        if is_ehr:
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
                dst_con.executemany(f"INSERT INTO {qpt} VALUES ({','.join('?' for _ in cnames)})", prows)
            dst_con.commit()

        dst_con.commit()
        fk_violations = dst_con.execute("PRAGMA foreign_key_check").fetchall()
        assert len(fk_violations) == 0, f"FK violations in {src_db.stem}: {fk_violations[:5]}"
        assert dst_con.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        report["fk_violations"] = len(fk_violations)

    tmp.replace(out_sqlite)
    return report
