"""Core exact Discrete Gaussian zCDP primitives, DP domain discovery, and Factored PGM (IPF)."""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from contextlib import contextmanager
from fractions import Fraction
import hashlib
import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from silhouette import dg_sampler as dg  # noqa: E402
from silhouette import profile as prof_mod  # noqa: E402
from silhouette import schema_roles as sr  # noqa: E402

TABLE_EPSILON = Fraction(8)
TABLE_DELTA = Fraction(1, 10**6)
TABLE_DELTA_HALF = TABLE_DELTA / 2
RHO_TABLE_8 = dg.rational_rho_budget(TABLE_EPSILON, TABLE_DELTA_HALF)
EHR_PUBLIC_TABLES = frozenset(("d_icd_diagnoses", "d_icd_procedures", "d_items", "d_labitems"))


class SHA256CounterRNG:
    """Deterministic seed-logged cryptographic counter PRNG implementing IntegerRNG."""

    def __init__(self, seed: int | str | bytes) -> None:
        if isinstance(seed, int):
            self.seed_bytes = f"seed:{seed}".encode("utf-8")
        elif isinstance(seed, str):
            self.seed_bytes = seed.encode("utf-8")
        else:
            self.seed_bytes = bytes(seed)
        self._counter = 0
        self._buf = b""
        self._pos = 0

    def _refill(self) -> None:
        chunk = hashlib.shake_256(self.seed_bytes + self._counter.to_bytes(8, "big")).digest(65536)
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

    def derive_np_rng(self, tag: str) -> np.random.Generator:
        digest = hashlib.sha256(self.seed_bytes + f"|np|{tag}".encode("utf-8")).digest()
        return np.random.default_rng(int.from_bytes(digest, "little"))


@contextmanager
def ro_conn(db_path: Path):
    con = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    try:
        con.text_factory = lambda b: b.decode("utf-8", errors="ignore")
        con.execute("PRAGMA query_only=ON")
        con.execute("PRAGMA temp_store=MEMORY")
        yield con
    finally:
        con.close()


def split_rational_budget(total_rho: Fraction, weights: Sequence[Fraction | int]) -> List[Fraction]:
    """Split exact rational rho across queries without exceeding total_rho."""
    if total_rho <= 0:
        raise ValueError("total_rho must be > 0")
    w_fracs = [Fraction(w) for w in weights]
    w_sum = sum(w_fracs, Fraction(0))
    if w_sum <= 0:
        raise ValueError("sum of weights must be > 0")
    shares = [total_rho * (w / w_sum) for w in w_fracs]
    assert sum(shares, Fraction(0)) <= total_rho
    return shares


def sigma2_for_zcdp(rho_q: Fraction, l2_sens: int | Fraction = 1) -> Fraction:
    """For L2 sensitivity Delta_2, N_Z(0, sigma2) gives rho = Delta_2^2 / (2 * sigma2)."""
    s = Fraction(l2_sens)
    if rho_q <= 0 or s <= 0:
        raise ValueError("rho_q and l2_sens must be positive")
    return (s * s) / (2 * rho_q)


def exact_tau_for_sensitivity(sigma2: Fraction, delta_j: Fraction, l1_max_step: int = 1) -> int:
    """Exact integer threshold tau such that Pr[l1_max_step + Z > tau] <= delta_j."""
    tau1, _ = dg.exact_dgauss_tau(sigma2, delta_j)
    return int(tau1) + max(0, int(l1_max_step) - 1)


def inspect_schema_and_dag(
    db_path: Path,
    *,
    exclude_tables: Set[str] | frozenset[str] = frozenset(),
) -> Dict[str, Any]:
    """Extract pure-DDL table info, PKs, UNIQUE cols, and acyclic topological order."""
    with ro_conn(db_path) as con:
        pub_schema = sr.extract_public_schema(con)
        all_tbls = [t for t in sorted(pub_schema.tables) if t not in exclude_tables]
        tbl_set = set(all_tbls)
        tbl_lower = {t.lower(): t for t in all_tbls}

        t_info: Dict[str, List[Dict[str, Any]]] = {}
        pk_cols_by_t: Dict[str, List[str]] = {}
        unique_cols_by_t: Dict[str, Set[str]] = {}
        col_metas_by_t: Dict[str, Tuple[sr.ColumnMeta, ...]] = {}

        for t in all_tbls:
            qt = sr.quote(t)
            cols = con.execute(f"PRAGMA table_info({qt})").fetchall()
            t_info[t] = [
                {"cid": c[0], "name": c[1], "type": c[2] or "ANY", "notnull": bool(c[3]), "pk": int(c[5])}
                for c in cols
            ]
            col_metas_by_t[t] = pub_schema.tables[t]
            pks = [c[1] for c in sorted((x for x in cols if x[5]), key=lambda x: x[5])]
            pk_cols_by_t[t] = pks
            ucols: Set[str] = set(pks) if len(pks) == 1 else set()
            for idx in con.execute(f"PRAGMA index_list({qt})"):
                if idx[2] and not idx[4]:
                    icols = con.execute(f"PRAGMA index_info({sr.quote(idx[1])})").fetchall()
                    if len(icols) == 1 and icols[0][2] is not None:
                        ucols.add(icols[0][2])
            unique_cols_by_t[t] = ucols

        raw_fk_groups: List[Tuple[str, int, List[Any]]] = []
        for t in all_tbls:
            groups = defaultdict(list)
            for fk in con.execute(f"PRAGMA foreign_key_list({sr.quote(t)})"):
                groups[fk[0]].append(fk)
            for fk_id, flist in sorted(groups.items()):
                flist.sort(key=lambda f: f[1])
                raw_fk_groups.append((t, fk_id, flist))

    dag_edges: List[Tuple[str, str, str, str]] = []
    public_fk_edges: List[Tuple[str, str, str, str]] = []
    self_loops: List[Tuple[str, str, str, str]] = []
    seen: Set[Tuple[str, str, str, str]] = set()

    for src_t, _fk_id, flist in raw_fk_groups:
        if len(flist) != 1:
            continue
        f = flist[0]
        dst_t_raw = f[2]
        dst_t = tbl_lower.get((dst_t_raw or "").lower(), dst_t_raw)
        src_c = f[3]
        dst_c = f[4]
        if dst_t in tbl_set:
            if dst_c is None:
                pks = pk_cols_by_t.get(dst_t, [])
                dst_c = pks[0] if len(pks) == 1 else src_c
            col_map = {c["name"].lower(): c["name"] for c in t_info[dst_t]}
            dst_c = col_map.get(dst_c.lower(), dst_c)
            edge = (src_t, src_c, dst_t, dst_c)
            if edge in seen:
                continue
            seen.add(edge)
            if src_t == dst_t:
                self_loops.append(edge)
            else:
                dag_edges.append(edge)
        elif dst_t in exclude_tables:
            public_fk_edges.append((src_t, src_c, dst_t, dst_c or src_c))

    # Kahn topological sort; if any cycle exists, break lexicographically last FK edge
    kept_edges = list(dag_edges)
    broken_edges: List[Tuple[str, str, str, str]] = []
    while True:
        children = defaultdict(set)
        indeg = {t: 0 for t in all_tbls}
        for src_t, _sc, dst_t, _dc in kept_edges:
            if src_t not in children[dst_t]:
                children[dst_t].add(src_t)
                indeg[src_t] += 1
        q = deque(sorted(t for t, d in indeg.items() if d == 0))
        topo: List[str] = []
        while q:
            u = q.popleft()
            topo.append(u)
            for v in sorted(children[u]):
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        if len(topo) == len(all_tbls):
            break
        cycle_nodes = set(all_tbls) - set(topo)
        cycle_edges = sorted(e for e in kept_edges if e[0] in cycle_nodes and e[2] in cycle_nodes)
        if not cycle_edges:
            break
        victim = cycle_edges[-1]
        kept_edges.remove(victim)
        broken_edges.append(victim)

    return {
        "tables": all_tbls,
        "table_info": t_info,
        "col_metas": col_metas_by_t,
        "pk_cols": pk_cols_by_t,
        "unique_cols": {t: sorted(v) for t, v in unique_cols_by_t.items()},
        "dag_edges": kept_edges,
        "broken_cycle_edges": broken_edges,
        "public_fk_edges": public_fk_edges,
        "self_loops": self_loops,
        "topo_order": topo,
        "pub_schema": pub_schema,
    }


def is_sequential_pk(cm: sr.ColumnMeta, pk_cols: List[str]) -> bool:
    return len(pk_cols) == 1 and cm.name == pk_cols[0]


def encode_cat(val: Any, max_len: int = 120) -> Tuple[str, Any]:
    if val is None:
        return ("null", None)
    if isinstance(val, bytes):
        return ("blob", val[:60].hex())
    if isinstance(val, bool):
        return ("int", int(val))
    if isinstance(val, int):
        return ("int", int(val))
    if isinstance(val, float):
        if not math.isfinite(val):
            return ("null", None)
        return ("real", round(val, 8))
    s = str(val)
    if len(s) > max_len:
        s = s[:max_len]
    return ("str", s)


def decode_cat(tok: Tuple[str, Any]) -> Any:
    tag, payload = tok
    if tag in ("null", "ood"):
        return None
    if tag == "blob":
        return bytes.fromhex(payload)
    return payload


def dp_discover_table_domains(
    rows: List[Tuple[Any, ...]],
    col_metas: Tuple[sr.ColumnMeta, ...],
    skip_cols: Set[str],
    rho_dom: Fraction,
    delta_dom: Fraction,
    rng: SHA256CounterRNG,
    *,
    max_cat_bins: int = 32,
    max_num_bins: int = 16,
    l2_sens: int = 1,
    l1_max_step: int = 1,
) -> Dict[str, Dict[str, Any]]:
    """Spend (rho_dom, delta_dom) via exact Discrete Gaussian thresholding to discover domains.

    Never reads raw categorical domains noiselessly.
    Special sentinel codes:
      - category code 0 = NULL
      - category code 1 = <OTHER> (reserved fallback / collapsed tail)
      - numeric code 0 = NULL
      - numeric bins 1..B = active surviving EDGES buckets
    """
    modeled = [cm for cm in col_metas if cm.name not in skip_cols]
    if not modeled:
        return {}

    n_cols = len(modeled)
    rho_per_col = rho_dom / n_cols
    delta_per_col = delta_dom / n_cols
    sigma2 = sigma2_for_zcdp(rho_per_col, l2_sens)
    tau = exact_tau_for_sensitivity(sigma2, delta_per_col, l1_max_step=l1_max_step)

    col_idx = {cm.name: i for i, cm in enumerate(col_metas)}
    domains: Dict[str, Dict[str, Any]] = {}

    for cm in modeled:
        j = col_idx[cm.name]
        is_num = sr.numeric(cm.declared_type) and not sr.is_integer_type(cm.declared_type)
        # Also treat pure numeric INTEGER columns that are not ID/FK/CODE as numeric if non-PK
        role = sr.classify_column(cm)
        if sr.numeric(cm.declared_type) and role == "NUMERIC_REAL":
            is_num = True
        elif sr.numeric(cm.declared_type) and role not in ("KEY_OR_FK", "IDENT_OR_CONTACT", "CODE", "CATEGORY"):
            is_num = True

        if is_num:
            raw_cnts = Counter(r[j] for r in rows)
            cnts: Counter = Counter()
            for v, c in raw_cnts.items():
                if v is None:
                    cnts[("null", None)] += c
                else:
                    cnts[("bin", prof_mod.bin_index(v))] += c
            surviving: List[Tuple[int, int]] = []
            null_surv = 0
            for key in sorted(cnts):
                noisy = int(cnts[key]) + dg.sample_dgauss(sigma2, rng)
                if noisy > tau:
                    if key[0] == "null":
                        null_surv = noisy
                    elif key[1] >= 0:
                        surviving.append((key[1], noisy))
            # Sort surviving bins by bin index, merge if > max_num_bins by keeping top-mass or stride
            surviving.sort(key=lambda x: x[0])
            if len(surviving) > max_num_bins:
                # Keep top max_num_bins by noisy count, then sort by bin index
                surviving = sorted(
                    sorted(surviving, key=lambda x: (-x[1], x[0]))[:max_num_bins],
                    key=lambda x: x[0],
                )
            active_bins = [b for b, _ in surviving]
            bin_to_code = {b: i + 1 for i, b in enumerate(active_bins)}
            domains[cm.name] = {
                "kind": "numeric",
                "is_int": sr.is_integer_type(cm.declared_type),
                "active_bins": active_bins,
                "bin_to_code": bin_to_code,
                "size": 1 + max(1, len(active_bins)),  # 0=NULL, 1..K=bins (or 1=fallback 0.0)
                "null_surv": null_surv,
                "tau": tau,
            }
        else:
            raw_cnts = Counter(r[j] for r in rows)
            cnts = Counter()
            for v, c in raw_cnts.items():
                cnts[encode_cat(v)] += c
            surv_cats: List[Tuple[Tuple[str, Any], int]] = []
            null_surv = 0
            for tok, c in cnts.items():
                noisy = int(c) + dg.sample_dgauss(sigma2, rng)
                if noisy > tau:
                    if tok[0] == "null":
                        null_surv = noisy
                    else:
                        surv_cats.append((tok, noisy))
            surv_cats.sort(key=lambda x: (-x[1], repr(x[0])))
            surv_cats = surv_cats[:max_cat_bins]
            active_cats = [tok for tok, _ in surv_cats]
            tok_to_code = {tok: i + 2 for i, tok in enumerate(active_cats)}
            domains[cm.name] = {
                "kind": "categorical",
                "active_cats": active_cats,
                "tok_to_code": tok_to_code,
                "size": 2 + len(active_cats),  # 0=NULL, 1=<OTHER>, 2..=active
                "null_surv": null_surv,
                "tau": tau,
            }
    return domains


def encode_rows_to_matrix(
    rows: List[Tuple[Any, ...]],
    col_names: List[str],
    col_idx: Dict[str, int],
    domains: Dict[str, Dict[str, Any]],
) -> np.ndarray:
    """Encode raw table rows into integer codes in [0, domain_size)."""
    n = len(rows)
    d = len(col_names)
    mat = np.zeros((n, d), dtype=np.int32)
    for c_pos, cname in enumerate(col_names):
        dom = domains[cname]
        j = col_idx[cname]
        if dom["kind"] == "numeric":
            bmap = dom["bin_to_code"]
            active = dom["active_bins"]
            if not active:
                for i, r in enumerate(rows):
                    mat[i, c_pos] = 0 if r[j] is None else 1
            else:
                import bisect
                for i, r in enumerate(rows):
                    v = r[j]
                    if v is None:
                        mat[i, c_pos] = 0
                    else:
                        b = prof_mod.bin_index(v)
                        if b in bmap:
                            mat[i, c_pos] = bmap[b]
                        elif b < 0:
                            mat[i, c_pos] = 0
                        else:
                            pos = bisect.bisect_left(active, b)
                            if pos == 0:
                                mat[i, c_pos] = 1
                            elif pos >= len(active):
                                mat[i, c_pos] = len(active)
                            else:
                                # Nearest active bin
                                dl = abs(b - active[pos - 1])
                                dr = abs(active[pos] - b)
                                mat[i, c_pos] = pos if dl <= dr else pos + 1
        else:
            tmap = dom["tok_to_code"]
            raw_cache: Dict[Any, int] = {}
            for i, r in enumerate(rows):
                v = r[j]
                if v is None:
                    mat[i, c_pos] = 0
                else:
                    code = raw_cache.get(v)
                    if code is None:
                        tok = encode_cat(v)
                        code = 0 if tok[0] == "null" else tmap.get(tok, 1)
                        try:
                            raw_cache[v] = code
                        except TypeError:
                            pass
                    mat[i, c_pos] = code
    return mat


def decode_matrix_to_columns(
    mat: np.ndarray,
    col_names: List[str],
    domains: Dict[str, Dict[str, Any]],
    np_rng: np.random.Generator,
) -> Dict[str, List[Any]]:
    """Decode synthetic integer matrix back into SQLite Python values."""
    n = mat.shape[0]
    out: Dict[str, List[Any]] = {}
    for c_pos, cname in enumerate(col_names):
        dom = domains[cname]
        codes = mat[:, c_pos]
        if dom["kind"] == "numeric":
            active = dom["active_bins"]
            is_int = dom["is_int"]
            vals: List[Any] = []
            if not active:
                for code in codes:
                    vals.append(None if code == 0 else (0 if is_int else 0.0))
            else:
                edges = prof_mod.EDGES
                for code in codes:
                    if code == 0:
                        vals.append(None)
                    else:
                        b = active[min(len(active) - 1, max(0, int(code) - 1))]
                        lo = edges[b]
                        hi = edges[b + 1] if b + 1 < len(edges) else lo * 1.5
                        if not (math.isfinite(lo) and math.isfinite(hi)):
                            v = 0.0
                        elif hi <= lo:
                            v = lo
                        else:
                            v = float(np_rng.uniform(lo, min(hi, lo + max(abs(lo), 1.0) * 2.0)))
                        vals.append(int(round(v)) if is_int else round(v, 4))
            out[cname] = vals
        else:
            cats = dom["active_cats"]
            vals = []
            for code in codes:
                ic = int(code)
                if ic == 0:
                    vals.append(None)
                elif ic == 1:
                    # <OTHER>: sample from surviving non-null categories if any, else synthetic token
                    if cats:
                        vals.append(decode_cat(cats[int(np_rng.integers(0, len(cats)))]))
                    else:
                        vals.append(f"OTHER_{cname}")
                else:
                    idx = min(len(cats) - 1, ic - 2)
                    vals.append(decode_cat(cats[idx]))
            out[cname] = vals
    return out


class FactoredPairwisePGM:
    """Fast pure-NumPy pairwise Graphical Model (Forest + IPF / Belief Propagation) & Sampler.

    Supports arbitrary 1-way marginals + acyclic or cyclic 2-way marginals with exact
    Maximum-Spanning-Tree / IPF calibration and ancestral tree sampling in O(N_cols * K^2).
    """

    def __init__(self, col_sizes: List[int]) -> None:
        self.sizes = list(col_sizes)
        self.d = len(self.sizes)
        self.marginals_1d: List[np.ndarray] = [
            np.full(k, 1.0 / max(1, k), dtype=np.float64) for k in self.sizes
        ]
        self.marginals_2d: Dict[Tuple[int, int], np.ndarray] = {}
        self.edges: List[Tuple[int, int]] = []

    def set_1way(self, col: int, noisy_counts: np.ndarray) -> None:
        arr = np.maximum(0.0, np.asarray(noisy_counts, dtype=np.float64))
        s = float(arr.sum())
        if s <= 0.0:
            arr = np.full(self.sizes[col], 1.0 / max(1, self.sizes[col]), dtype=np.float64)
        else:
            arr = arr / s
        self.marginals_1d[col] = arr

    def add_2way(self, i: int, j: int, noisy_joint: np.ndarray) -> None:
        if i > j:
            i, j = j, i
            noisy_joint = noisy_joint.T
        arr = np.maximum(1e-9, np.asarray(noisy_joint, dtype=np.float64))
        s = float(arr.sum())
        if s <= 0.0:
            arr = np.outer(self.marginals_1d[i], self.marginals_1d[j])
        else:
            arr = arr / s
        self.marginals_2d[(i, j)] = arr
        if (i, j) not in self.edges:
            self.edges.append((i, j))

    def _build_forest(self) -> Tuple[List[int], Dict[int, Optional[int]], Dict[int, List[int]]]:
        """Extract maximum-mutual-information spanning forest from measured 2-way edges."""
        scored_edges: List[Tuple[float, int, int]] = []
        for (i, j), p_ij in self.marginals_2d.items():
            pi = np.maximum(1e-12, p_ij.sum(axis=1))
            pj = np.maximum(1e-12, p_ij.sum(axis=0))
            indep = np.outer(pi, pj)
            mi = float(np.sum(p_ij * np.log(np.maximum(1e-12, p_ij) / np.maximum(1e-12, indep))))
            scored_edges.append((mi, i, j))
        scored_edges.sort(reverse=True)

        parent_uf = list(range(self.d))

        def find(x: int) -> int:
            while parent_uf[x] != x:
                parent_uf[x] = parent_uf[parent_uf[x]]
                x = parent_uf[x]
            return x

        adj: Dict[int, List[int]] = defaultdict(list)
        for _mi, i, j in scored_edges:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent_uf[ri] = rj
                adj[i].append(j)
                adj[j].append(i)

        visited = set()
        roots: List[int] = []
        tree_parent: Dict[int, Optional[int]] = {}
        tree_children: Dict[int, List[int]] = defaultdict(list)
        for start in range(self.d):
            if start in visited:
                continue
            roots.append(start)
            tree_parent[start] = None
            visited.add(start)
            q = deque([start])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        tree_parent[v] = u
                        tree_children[u].append(v)
                        q.append(v)
        return roots, tree_parent, tree_children

    def fit_ipf(self, iters: int = 12) -> None:
        """Run Iterative Proportional Fitting over all 1-way & 2-way marginals on the forest."""
        if not self.marginals_2d:
            return
        # Ensure each 2-way table agrees with 1-way marginals via Sinkhorn-Knopp IPF
        for _ in range(iters):
            for (i, j), mat in list(self.marginals_2d.items()):
                m = np.maximum(1e-12, mat)
                row_s = np.maximum(1e-12, m.sum(axis=1, keepdims=True))
                m *= self.marginals_1d[i][:, None] / row_s
                col_s = np.maximum(1e-12, m.sum(axis=0, keepdims=True))
                m *= self.marginals_1d[j][None, :] / col_s
                m /= max(1e-12, float(m.sum()))
                self.marginals_2d[(i, j)] = m

    def project_2way(self, i: int, j: int) -> np.ndarray:
        """Approximate or exact 2-way marginal under the current forest model."""
        if i == j:
            return np.diag(self.marginals_1d[i])
        key = (i, j) if i < j else (j, i)
        if key in self.marginals_2d:
            m = self.marginals_2d[key]
            return m if i < j else m.T
        # Compute along forest path if connected, else outer product
        roots, tree_parent, tree_children = self._build_forest()
        adj: Dict[int, List[int]] = defaultdict(list)
        for u, chs in tree_children.items():
            for v in chs:
                adj[u].append(v)
                adj[v].append(u)
        q = deque([(i, [i])])
        seen = {i}
        path: Optional[List[int]] = None
        while q:
            u, p = q.popleft()
            if u == j:
                path = p
                break
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    q.append((v, p + [v]))
        if path is None:
            return np.outer(self.marginals_1d[i], self.marginals_1d[j])
        cur = np.diag(self.marginals_1d[path[0]])
        for a, b in zip(path[:-1], path[1:]):
            edge_mat = self.marginals_2d[(a, b)] if a < b else self.marginals_2d[(b, a)].T
            pa = np.maximum(1e-12, self.marginals_1d[a])[:, None]
            trans = edge_mat / pa  # P(b | a)
            cur = cur @ trans
        return cur

    def sample(self, n_rows: int, np_rng: np.random.Generator) -> np.ndarray:
        if n_rows <= 0:
            return np.zeros((0, self.d), dtype=np.int32)
        self.fit_ipf()
        roots, tree_parent, tree_children = self._build_forest()
        out = np.zeros((n_rows, self.d), dtype=np.int32)

        def sample_subtree(u: int) -> None:
            p = tree_parent[u]
            if p is None:
                probs = self.marginals_1d[u]
                probs = np.maximum(0.0, probs)
                probs /= probs.sum()
                out[:, u] = np_rng.choice(self.sizes[u], size=n_rows, p=probs)
            else:
                joint = self.marginals_2d[(p, u)] if p < u else self.marginals_2d[(u, p)].T
                parent_vals = out[:, p]
                for pv in range(self.sizes[p]):
                    mask = parent_vals == pv
                    cnt = int(mask.sum())
                    if cnt == 0:
                        continue
                    cond = np.maximum(0.0, joint[pv])
                    cs = float(cond.sum())
                    if cs <= 0.0:
                        cond = self.marginals_1d[u]
                        cs = float(cond.sum())
                    cond = cond / max(1e-12, cs)
                    out[mask, u] = np_rng.choice(self.sizes[u], size=cnt, p=cond)
            for ch in tree_children[u]:
                sample_subtree(ch)

        for r in roots:
            sample_subtree(r)
        return out


def materialize_sqlite_table(
    dest: sqlite3.Connection,
    table: str,
    cols_meta: List[Dict[str, Any]],
    col_data: Dict[str, List[Any]],
    n_rows: int,
    *,
    pk_cols: Optional[List[str]] = None,
    unique_cols: Optional[Set[str]] = None,
    fk_clauses: Optional[List[str]] = None,
) -> None:
    """Create SQLite table and insert rows with clean type preservation."""
    qt = sr.quote(table)
    col_defs: List[str] = []
    pk_list = pk_cols or []
    u_set = unique_cols or set()
    for c in cols_meta:
        cname = c["name"]
        ctype = c["type"] or "ANY"
        cdef = f"{sr.quote(cname)} {ctype}"
        if len(pk_list) == 1 and cname == pk_list[0]:
            cdef += " PRIMARY KEY"
        elif cname in u_set and cname not in pk_list:
            cdef += " UNIQUE"
        col_defs.append(cdef)
    if len(pk_list) > 1:
        col_defs.append("PRIMARY KEY (" + ", ".join(sr.quote(p) for p in pk_list) + ")")
    if fk_clauses:
        col_defs.extend(fk_clauses)

    dest.execute(f"CREATE TABLE {qt} ({', '.join(col_defs)})")
    if n_rows > 0:
        cnames = [c["name"] for c in cols_meta]
        placeholders = ", ".join("?" for _ in cnames)
        qcols = ", ".join(sr.quote(cn) for cn in cnames)
        rows = [tuple(col_data[cn][i] for cn in cnames) for i in range(n_rows)]
        dest.executemany(f"INSERT INTO {qt} ({qcols}) VALUES ({placeholders})", rows)
    dest.commit()


def generate_sequential_pk(col_type: str, n_rows: int, start: int = 1) -> List[Any]:
    u = (col_type or "").upper()
    if any(x in u for x in ("CHAR", "TEXT", "CLOB")):
        return [str(i) for i in range(start, start + n_rows)]
    if "BLOB" in u:
        return [str(i).encode("ascii") for i in range(start, start + n_rows)]
    return list(range(start, start + n_rows))
