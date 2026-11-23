"""Baseline 2: PrivLava (Cai, Xiao, Cormode, SIGMOD 2023) Multi-Table FK Relational Synthesis."""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from contextlib import closing
from fractions import Fraction
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
from silhouette import entity_dp_publisher as edp
from silhouette import schema_roles as sr

K_LATENT = 8


def _cluster_rows_deterministic(
    col_data: Dict[str, List[Any]],
    n_rows: int,
    exclude_cols: Set[str],
    k_clusters: int = K_LATENT,
) -> List[int]:
    if n_rows == 0:
        return []
    attr_cols = [c for c in sorted(col_data) if c not in exclude_cols]
    if not attr_cols:
        return [0] * n_rows
    use_cols = attr_cols[:3]
    clusters = []
    for i in range(n_rows):
        key_str = "|".join(repr(col_data[c][i]) for c in use_cols)
        h = 0
        for ch in key_str:
            h = (h * 131 + ord(ch)) & 0xFFFFFFFF
        clusters.append(h % k_clusters)
    return clusters


def _parent_cluster_map_from_real(
    real_con: sqlite3.Connection,
    ptbl: str,
    p_pk_col: str,
    p_cols_meta: Tuple[sr.ColumnMeta, ...],
    p_seq_pks: Set[str],
    k_clusters: int = K_LATENT,
) -> Dict[Any, int]:
    cnames = [cm.name for cm in p_cols_meta]
    sel = ", ".join(sr.quote(c) for c in cnames)
    prows = real_con.execute(f"SELECT {sel} FROM {sr.quote(ptbl)}").fetchall()
    col_data = {cn: [r[idx] for r in prows] for idx, cn in enumerate(cnames)}
    clusters = _cluster_rows_deterministic(col_data, len(prows), p_seq_pks, k_clusters)
    pk_vals = col_data[p_pk_col]
    return {pk_vals[i]: clusters[i] for i in range(len(prows)) if pk_vals[i] is not None}


def _noisy_1way(arr: np.ndarray, size: int, sigma2: Fraction, rng: SHA256CounterRNG) -> np.ndarray:
    c = np.bincount(arr, minlength=size).astype(np.float64)
    for b in range(size):
        c[b] += dg.sample_dgauss(sigma2, rng)
    return np.maximum(0.0, c)


def _noisy_2way(ci: np.ndarray, cj: np.ndarray, si: int, sj: int, sigma2: Fraction, rng: SHA256CounterRNG) -> np.ndarray:
    idx = ci.astype(np.int64) * sj + cj.astype(np.int64)
    c = np.bincount(idx, minlength=si * sj).reshape(si, sj).astype(np.float64)
    for a in range(si):
        for b in range(sj):
            c[a, b] += dg.sample_dgauss(sigma2, rng)
    return np.maximum(0.0, c)


def _resolve_fks_and_constraints(
    table: str,
    col_data: Dict[str, List[Any]],
    n_syn: int,
    t_info: List[Dict[str, Any]],
    pk_cols: List[str],
    unique_cols: Set[str],
    fk_edges_for_table: List[Tuple[str, str, str]],
    self_loops_for_table: List[Tuple[str, str]],
    syn_cols_by_table: Dict[str, Dict[str, List[Any]]],
    primary_fk_col: Optional[str],
    primary_fk_vals: Optional[List[Any]],
    np_rng: np.random.Generator,
) -> Tuple[Dict[str, List[Any]], int]:
    if n_syn == 0:
        return col_data, 0
    if primary_fk_col is not None and primary_fk_vals is not None:
        assert len(primary_fk_vals) == n_syn
        col_data[primary_fk_col] = list(primary_fk_vals)

    fks_by_col: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for sc, dt, dc in fk_edges_for_table:
        fks_by_col[sc].append((dt, dc))
    seq_pks = {pk_cols[0]} if len(pk_cols) == 1 else set()

    for sc, targets in fks_by_col.items():
        if sc == primary_fk_col:
            continue
        if sc in seq_pks:
            dt, dc = targets[0]
            pool = [v for v in syn_cols_by_table.get(dt, {}).get(dc, []) if v is not None]
            uniq_pool = list(dict.fromkeys(pool))
            if not uniq_pool:
                continue
            if n_syn > len(uniq_pool):
                n_syn = len(uniq_pool)
                for k in list(col_data):
                    col_data[k] = col_data[k][:n_syn]
            chosen = [uniq_pool[int(i)] for i in np_rng.choice(len(uniq_pool), size=n_syn, replace=False)]
            col_data[sc] = chosen
            continue
        pool = []
        for dt, dc in targets:
            pool.extend(v for v in syn_cols_by_table.get(dt, {}).get(dc, []) if v is not None)
        if not pool:
            continue
        cur = col_data.get(sc, [None] * n_syn)
        col_data[sc] = [None if v is None else pool[int(np_rng.integers(0, len(pool)))] for v in cur]

    if primary_fk_col is not None and (primary_fk_col in seq_pks or primary_fk_col in unique_cols):
        seen_u: Set[Any] = set()
        keep_idx: List[int] = []
        for idx, val in enumerate(col_data[primary_fk_col]):
            if val is not None and val not in seen_u:
                seen_u.add(val)
                keep_idx.append(idx)
        if len(keep_idx) < n_syn:
            n_syn = len(keep_idx)
            for k in list(col_data):
                col_data[k] = [col_data[k][i] for i in keep_idx]

    for c in t_info:
        if not c["notnull"] or n_syn == 0:
            continue
        cn = c["name"]
        vals = col_data[cn]
        nonnull = [v for v in vals if v is not None]
        fallback = (
            nonnull[0] if nonnull
            else (0 if sr.is_integer_type(c["type"]) else 0.0 if sr.numeric(c["type"]) else "")
        )
        col_data[cn] = [
            v if v is not None else (nonnull[int(np_rng.integers(0, len(nonnull)))] if nonnull else fallback)
            for v in vals
        ]

    type_by_col = {c["name"]: c["type"] for c in t_info}
    for uc in sorted(unique_cols):
        if uc in seq_pks or uc == primary_fk_col:
            continue
        seen_u = set()
        seq_i = 1
        is_int = sr.is_integer_type(type_by_col.get(uc, ""))
        fixed_u = []
        for val in col_data[uc]:
            if val is not None and val not in seen_u:
                seen_u.add(val)
                fixed_u.append(val)
            elif val is None:
                fixed_u.append(None)
            elif is_int:
                while seq_i in seen_u:
                    seq_i += 1
                seen_u.add(seq_i)
                fixed_u.append(seq_i)
                seq_i += 1
            else:
                cand = f"{uc}_{seq_i}"
                while cand in seen_u:
                    seq_i += 1
                    cand = f"{uc}_{seq_i}"
                seen_u.add(cand)
                fixed_u.append(cand)
                seq_i += 1
        col_data[uc] = fixed_u

    for sc, dc in self_loops_for_table:
        pool = [v for v in col_data.get(dc, []) if v is not None]
        if not pool:
            col_data[sc] = [None] * n_syn
        else:
            col_data[sc] = [
                None if v is None else pool[int(np_rng.integers(0, len(pool)))]
                for v in col_data.get(sc, [None] * n_syn)
            ]

    if len(pk_cols) > 1 and n_syn > 0:
        seen_pk: Set[Tuple[Any, ...]] = set()
        keep_idx = []
        for i in range(n_syn):
            for pkc in pk_cols:
                if col_data[pkc][i] is None:
                    if pkc in fks_by_col:
                        dt, dc = fks_by_col[pkc][0]
                        pool = [v for v in syn_cols_by_table.get(dt, {}).get(dc, []) if v is not None]
                        col_data[pkc][i] = pool[int(np_rng.integers(0, len(pool)))] if pool else f"PK_{i}"
                    else:
                        col_data[pkc][i] = f"PK_{i}"
            tup = tuple(col_data[pkc][i] for pkc in pk_cols)
            if tup in seen_pk:
                fk_pks = [pkc for pkc in pk_cols if pkc in fks_by_col]
                fixed = False
                if fk_pks:
                    pkc = fk_pks[-1]
                    dt, dc = fks_by_col[pkc][0]
                    pool = [v for v in syn_cols_by_table.get(dt, {}).get(dc, []) if v is not None]
                    if pool:
                        for _ in range(15):
                            col_data[pkc][i] = pool[int(np_rng.integers(0, len(pool)))]
                            tup2 = tuple(col_data[pkc][i] for pkc in pk_cols)
                            if tup2 not in seen_pk:
                                tup = tup2
                                fixed = True
                                break
                if not fixed:
                    continue
            seen_pk.add(tup)
            keep_idx.append(i)
        if len(keep_idx) < n_syn:
            n_syn = len(keep_idx)
            for k in list(col_data):
                col_data[k] = [col_data[k][i] for i in keep_idx]


    return col_data, n_syn


def _synthesize_privlava_table(
    rows: List[Tuple[Any, ...]],
    col_metas: Tuple[sr.ColumnMeta, ...],
    pk_cols: List[str],
    primary_parent_fk: Optional[Tuple[str, str, str]],
    real_parent_cluster_map: Optional[Dict[Any, int]],
    syn_parent_pks_by_cluster: Optional[Dict[int, List[Any]]],
    rng: SHA256CounterRNG,
    *,
    rho_table: Fraction,
    delta_table: Fraction,
    l2_sens: int = 1,
    l1_max_step: int = 1,
) -> Tuple[Dict[str, List[Any]], int, Optional[List[Any]], Dict[str, Any]]:
    delta_half = delta_table / 2
    seq_pks = {cm.name for cm in col_metas if is_sequential_pk(cm, pk_cols)}
    col_idx = {cm.name: i for i, cm in enumerate(col_metas)}
    np_rng = rng.derive_np_rng("privlava")

    has_parent = (
        primary_parent_fk is not None
        and real_parent_cluster_map is not None
        and syn_parent_pks_by_cluster is not None
        and any(len(v) > 0 for v in syn_parent_pks_by_cluster.values())
    )

    if not has_parent:
        rho_cnt = rho_table * Fraction(5, 100)
        rho_dom = rho_table * Fraction(25, 100)
        rho_1w = rho_table * Fraction(30, 100)
        rho_2w = rho_table - rho_cnt - rho_dom - rho_1w

        sigma2_cnt = sigma2_for_zcdp(rho_cnt, l2_sens)
        n_syn = max(0, len(rows) + dg.sample_dgauss(sigma2_cnt, rng))
        domains = dp_discover_table_domains(
            rows, col_metas, seq_pks, rho_dom, delta_half, rng,
            l2_sens=l2_sens, l1_max_step=l1_max_step,
        )
        modeled = [cm.name for cm in col_metas if cm.name in domains]
        if not modeled or n_syn == 0:
            cdata = {
                cm.name: (generate_sequential_pk(cm.declared_type, n_syn) if cm.name in seq_pks else [None] * n_syn)
                for cm in col_metas
            }
            return cdata, n_syn, None, {"rho_used": float(rho_cnt + rho_dom), "n_syn": n_syn}

        mat = encode_rows_to_matrix(rows, modeled, col_idx, domains)
        sizes = [domains[c]["size"] for c in modeled]
        pgm = FactoredPairwisePGM(sizes)
        d = len(modeled)
        sigma2_1w = sigma2_for_zcdp(rho_1w / d, l2_sens)
        for k in range(d):
            pgm.set_1way(k, _noisy_1way(mat[:, k], sizes[k], sigma2_1w, rng))
        cand_2w = [(i, j) for i in range(d) for j in range(i + 1, d) if sizes[i] * sizes[j] <= 512]
        if len(cand_2w) > 16:
            cand_2w = [cand_2w[int(k)] for k in np_rng.choice(len(cand_2w), size=16, replace=False)]
        if cand_2w:
            sigma2_2w = sigma2_for_zcdp(rho_2w / len(cand_2w), l2_sens)
            for i, j in cand_2w:
                pgm.add_2way(i, j, _noisy_2way(mat[:, i], mat[:, j], sizes[i], sizes[j], sigma2_2w, rng))
        syn_mat = pgm.sample(n_syn, np_rng)
        dec = decode_matrix_to_columns(syn_mat, modeled, domains, np_rng)
        cdata = {
            cm.name: (generate_sequential_pk(cm.declared_type, n_syn) if cm.name in seq_pks else dec[cm.name])
            for cm in col_metas
        }
        return cdata, n_syn, None, {"rho_used": float(rho_table), "n_syn": n_syn}

    fk_col = primary_parent_fk[0]
    fk_idx = col_idx[fk_col]
    rho_dom = rho_table * Fraction(20, 100)
    rho_deg = rho_table * Fraction(20, 100)
    rho_cross = rho_table * Fraction(30, 100)
    rho_int = rho_table - rho_dom - rho_deg - rho_cross

    skip_cols = set(seq_pks) | {fk_col}
    domains = dp_discover_table_domains(
        rows, col_metas, skip_cols, rho_dom, delta_half, rng,
        l2_sens=l2_sens, l1_max_step=l1_max_step,
    )
    modeled = [cm.name for cm in col_metas if cm.name in domains]

    z_arr = np.array(
        [K_LATENT if r[fk_idx] is None else real_parent_cluster_map.get(r[fk_idx], 0) for r in rows],
        dtype=np.int32,
    )
    sigma2_deg = sigma2_for_zcdp(rho_deg, l2_sens)
    z_counts_true = np.bincount(z_arr, minlength=K_LATENT + 1).astype(np.float64)
    z_counts_noisy = np.array(
        [max(0.0, float(z_counts_true[z]) + dg.sample_dgauss(sigma2_deg, rng)) for z in range(K_LATENT + 1)],
        dtype=np.float64,
    )
    for z in range(K_LATENT):
        if not syn_parent_pks_by_cluster.get(z):
            z_counts_noisy[z] = 0.0

    n_syn = int(round(float(z_counts_noisy.sum())))
    if n_syn <= 0:
        cdata = {
            cm.name: (generate_sequential_pk(cm.declared_type, 0) if cm.name in seq_pks else [])
            for cm in col_metas
        }
        return cdata, 0, [], {"rho_used": float(rho_dom + rho_deg), "n_syn": 0}

    z_probs = z_counts_noisy / float(z_counts_noisy.sum())
    syn_z = np_rng.choice(K_LATENT + 1, size=n_syn, p=z_probs)
    all_parent_pks = [pk for pks in syn_parent_pks_by_cluster.values() for pk in pks]
    syn_fk_vals: List[Any] = []
    for z in syn_z:
        iz = int(z)
        if iz == K_LATENT:
            syn_fk_vals.append(None)
        else:
            pool = syn_parent_pks_by_cluster.get(iz) or all_parent_pks
            syn_fk_vals.append(pool[int(np_rng.integers(0, len(pool)))] if pool else None)

    if not modeled:
        cdata = {}
        for cm in col_metas:
            if cm.name in seq_pks:
                cdata[cm.name] = generate_sequential_pk(cm.declared_type, n_syn)
            elif cm.name == fk_col:
                cdata[cm.name] = syn_fk_vals
            else:
                cdata[cm.name] = [None] * n_syn
        return cdata, n_syn, syn_fk_vals, {"rho_used": float(rho_dom + rho_deg), "n_syn": n_syn}

    mat = encode_rows_to_matrix(rows, modeled, col_idx, domains)
    d = len(modeled)
    sizes = [K_LATENT + 1] + [domains[c]["size"] for c in modeled]
    pgm = FactoredPairwisePGM(sizes)
    pgm.set_1way(0, z_counts_noisy)

    sigma2_cross = sigma2_for_zcdp(rho_cross / d, l2_sens)
    for k in range(d):
        joint = _noisy_2way(z_arr, mat[:, k], K_LATENT + 1, sizes[k + 1], sigma2_cross, rng)
        pgm.add_2way(0, k + 1, joint)
        pgm.set_1way(k + 1, joint.sum(axis=0))

    cand_2w = [(i, j) for i in range(d) for j in range(i + 1, d) if sizes[i + 1] * sizes[j + 1] <= 512]
    if len(cand_2w) > 12:
        cand_2w = [cand_2w[int(k)] for k in np_rng.choice(len(cand_2w), size=12, replace=False)]
    n_int_q = d + len(cand_2w)
    sigma2_int = sigma2_for_zcdp(rho_int / max(1, n_int_q), l2_sens)
    for k in range(d):
        pgm.set_1way(k + 1, _noisy_1way(mat[:, k], sizes[k + 1], sigma2_int, rng))
    for i, j in cand_2w:
        pgm.add_2way(i + 1, j + 1, _noisy_2way(mat[:, i], mat[:, j], sizes[i + 1], sizes[j + 1], sigma2_int, rng))

    pgm.fit_ipf(iters=10)
    _, _, tree_children = pgm._build_forest()
    full_out = np.zeros((n_syn, d + 1), dtype=np.int32)
    full_out[:, 0] = syn_z
    sampled_nodes: Set[int] = {0}
    adj: Dict[int, List[int]] = defaultdict(list)
    for u, chs in tree_children.items():
        for v in chs:
            adj[u].append(v)
            adj[v].append(u)
    q_bfs = deque([0])
    while q_bfs:
        u = q_bfs.popleft()
        for v in adj[u]:
            if v not in sampled_nodes:
                joint = pgm.marginals_2d[(u, v)] if u < v else pgm.marginals_2d[(v, u)].T
                pvals = full_out[:, u]
                for pv in range(sizes[u]):
                    mask = pvals == pv
                    cnt = int(mask.sum())
                    if cnt == 0:
                        continue
                    cond = np.maximum(0.0, joint[pv])
                    cs = float(cond.sum())
                    cond = (cond / cs) if cs > 0 else pgm.marginals_1d[v]
                    cond = np.maximum(0.0, cond)
                    cond /= cond.sum()
                    full_out[mask, v] = np_rng.choice(sizes[v], size=cnt, p=cond)
                sampled_nodes.add(v)
                q_bfs.append(v)
    for v in range(1, d + 1):
        if v not in sampled_nodes:
            joint = pgm.marginals_2d[(0, v)]
            for pv in range(sizes[0]):
                mask = full_out[:, 0] == pv
                cnt = int(mask.sum())
                if cnt == 0:
                    continue
                cond = np.maximum(0.0, joint[pv])
                cs = float(cond.sum())
                cond = (cond / cs) if cs > 0 else pgm.marginals_1d[v]
                cond = np.maximum(0.0, cond)
                cond /= cond.sum()
                full_out[mask, v] = np_rng.choice(sizes[v], size=cnt, p=cond)

    dec = decode_matrix_to_columns(full_out[:, 1:], modeled, domains, np_rng)
    cdata = {}
    for cm in col_metas:
        if cm.name in seq_pks:
            cdata[cm.name] = generate_sequential_pk(cm.declared_type, n_syn)
        elif cm.name == fk_col:
            cdata[cm.name] = syn_fk_vals
        else:
            cdata[cm.name] = dec[cm.name]
    return cdata, n_syn, syn_fk_vals, {"rho_used": float(rho_table), "n_syn": n_syn}


def synthesize_db_privlava(
    src_db: Path,
    out_sqlite: Path,
    seed: int,
    *,
    is_ehr: bool = False,
    entity_mode: bool = False,
    entity_cap: int = 5,
    entity_specs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    exclude = EHR_PUBLIC_TABLES if is_ehr else frozenset()
    meta = inspect_schema_and_dag(src_db, exclude_tables=exclude)

    if entity_mode:
        db_id = src_db.stem
        if entity_specs is not None:
            espec = entity_specs
        else:
            assert db_id in edp.BIRD_ENTITY_SPECS, f"Entity specs missing for {db_id}"
            espec = edp.BIRD_ENTITY_SPECS[db_id]
        direct_tables = [t for t in meta["topo_order"] if espec[t]["mode"] == "entity_direct"]
        n_direct = max(1, len(direct_tables))
        rho_per_table = RHO_TABLE_8 / n_direct
        delta_per_table = TABLE_DELTA / n_direct
        l2_sens = entity_cap
        l1_max_step = entity_cap
    else:
        espec = None
        rho_per_table = RHO_TABLE_8
        delta_per_table = TABLE_DELTA
        l2_sens = 1
        l1_max_step = 1

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
    report: Dict[str, Any] = {
        "baseline": "Entity-PrivLava" if entity_mode else "PrivLava",
        "seed": seed,
        "entity_mode": entity_mode,
        "entity_cap": entity_cap if entity_mode else None,
        "tables": {},
    }

    with ro_conn(src_db) as src_con, closing(sqlite3.connect(str(tmp))) as dst_con:
        dst_con.execute("PRAGMA temp_store=MEMORY")
        dst_con.execute("PRAGMA foreign_keys=OFF")

        for t in meta["topo_order"]:
            qt = sr.quote(t)
            cols_meta = meta["col_metas"][t]
            pk_cols = meta["pk_cols"][t]
            seq_pks = {cm.name for cm in cols_meta if is_sequential_pk(cm, pk_cols)}
            np_rng = SHA256CounterRNG(f"PL_POST|{seed}|{src_db.stem}|{t}").derive_np_rng("post")

            if entity_mode and espec[t]["mode"] == "schema_only":
                cdata = {cm.name: [] for cm in cols_meta}
                syn_cols_by_table[t] = cdata
                syn_clusters_by_table[t] = []
                materialize_sqlite_table(
                    dst_con, t, meta["table_info"][t], cdata, 0,
                    pk_cols=pk_cols, unique_cols=set(meta["unique_cols"][t]),
                )
                report["tables"][t] = {"mode": "schema_only", "n_syn": 0, "rho_used": 0.0}
                continue

            sel = ", ".join(sr.quote(cm.name) for cm in cols_meta)
            rows = src_con.execute(f"SELECT {sel} FROM {qt}").fetchall()
            if entity_mode:
                ec = espec[t]["entity_col"]
                ec_idx = [cm.name for cm in cols_meta].index(ec)
                ent_counts: Counter = Counter()
                capped_rows = []
                for r in rows:
                    ev = r[ec_idx]
                    if ev is None:
                        continue
                    if ent_counts[ev] < entity_cap:
                        ent_counts[ev] += 1
                        capped_rows.append(r)
                rows = capped_rows

            valid_fks = [
                (sc, dt, dc) for (sc, dt, dc) in edges_by_child[t]
                if dt in syn_cols_by_table and len(syn_cols_by_table[dt].get(dc, [])) > 0
            ]
            primary_fk = valid_fks[0] if valid_fks else None

            real_cluster_map = None
            syn_pks_by_cluster = None
            if primary_fk is not None:
                sc, ptbl, pcol = primary_fk
                p_seq_pks = {cm.name for cm in meta["col_metas"][ptbl] if is_sequential_pk(cm, meta["pk_cols"][ptbl])}
                real_cluster_map = _parent_cluster_map_from_real(
                    src_con, ptbl, pcol, meta["col_metas"][ptbl], p_seq_pks, K_LATENT
                )
                syn_pks_by_cluster = defaultdict(list)
                for pv, cz in zip(syn_cols_by_table[ptbl][pcol], syn_clusters_by_table[ptbl]):
                    if pv is not None:
                        syn_pks_by_cluster[cz].append(pv)

            rng = SHA256CounterRNG(f"PRIVLAVA|{seed}|{src_db.stem}|{t}")
            cdata, n_syn, syn_fk_vals, t_rep = _synthesize_privlava_table(
                rows, cols_meta, pk_cols, primary_fk, real_cluster_map, syn_pks_by_cluster, rng,
                rho_table=rho_per_table, delta_table=delta_per_table,
                l2_sens=l2_sens, l1_max_step=l1_max_step,
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
