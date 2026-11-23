"""DG-v3 planning, canonical column statistics and five-field profile publication."""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter
import copy
from decimal import Decimal
from fractions import Fraction
import json
import math
from pathlib import Path
import re
import sqlite3
import statistics
from typing import Any, Callable, Collection, Dict, List, Optional, Sequence, Set, Tuple, Union

from . import dg_sampler as dg
from . import schema_roles as v3
from .schema_roles import numeric, quote

VERSION_DG_V3 = "c1-dg-v3"
TABLE_DELTA_FRAC = Fraction(1, 10**6)
HISTOGRAMS = ("values", "bins", "patterns")
COUNTS = ("row_count", "null_count", "distinct_count")
ND = statistics.NormalDist()

V3_ROLE_WEIGHTS_FRAC: Dict[str, Dict[str, Fraction]] = {
    role: {stat: Fraction(str(w)) for stat, w in stats.items()}
    for role, stats in v3.ROLE_WEIGHTS.items()
}
V3_ROW_COUNT_WEIGHT_FRAC = Fraction(str(v3.ROW_COUNT_WEIGHT))


def _frac_str(f: Fraction) -> str:
    return f"{f.numerator}/{f.denominator}"


# ------------------------------------------------------------------ A8 fix: deterministic f_val
def canonical_val_key(tag: str, value: Any) -> Tuple[str, Any]:
    """Deterministic row-order-independent grouping key f_val(typeof(val), val).

    Avoids SQLite GROUP BY CAST(c AS BLOB) (which truncates REAL via %.15g and picks
    first-row representative) and avoids COLLATE NOCASE / Python -0.0==0.0 / 1==1.0 collisions.
    """
    if tag == "integer":
        return ("integer", int(value))
    if tag == "real":
        v = float(value)
        return ("real", v.hex() if math.isfinite(v) else repr(v))
    if tag == "text":
        return ("text", str(value))
    if tag == "blob":
        return ("blob", bytes(value))
    raise ValueError(f"Unexpected SQLite typeof tag for non-NULL value: {tag!r}")


def decode_val_key(key: Tuple[str, Any]) -> Any:
    """Recover JSON-serializable scalar from canonical_val_key."""
    tag, payload = key
    if tag == "real":
        if isinstance(payload, str) and (payload.startswith("0x") or payload.startswith("-0x")):
            return float.fromhex(payload)
        return str(payload)
    if tag == "blob":
        return str(payload)
    return payload


def column_statistics_exact(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    typ: str,
    nrows: int,
    need_values: bool,
) -> Dict[str, Any]:
    """Scan column row-by-row with deterministic f_val(typeof(val), val) exact equality."""
    qc, qt = quote(column), quote(table)
    nonnull, distinct = conn.execute(
        f"SELECT COUNT({qc}), COUNT(DISTINCT {qc}) FROM {qt}"
    ).fetchone()
    values: Counter = Counter()
    distribution: Counter = Counter()
    numeric_col = numeric(typ)
    sql = f"SELECT typeof({qc}), {qc} FROM {qt} WHERE {qc} IS NOT NULL"
    for tag, val in conn.execute(sql):
        if need_values:
            values[canonical_val_key(tag, val)] += 1
        bucket = bin_index(val) if numeric_col else shape(val)
        distribution[bucket] += 1
    return {
        "null_count": int(nrows - nonnull),
        "distinct_count": int(distinct),
        "distribution": distribution,
        "values": values,
    }


def table_statistics_exact(
    conn: sqlite3.Connection,
    table: str,
    col_metas: Tuple[v3.ColumnMeta, ...],
    nrows: int,
    need_values_cols: Set[str],
) -> Dict[str, Dict[str, Any]]:
    """Compute exact per-column statistics for a table with sorted canonical bin/value items."""
    out: Dict[str, Dict[str, Any]] = {}
    for cm in col_metas:
        st = column_statistics_exact(conn, table, cm.name, cm.declared_type, nrows, cm.name in need_values_cols)
        st["distribution_sorted"] = [
            (k, st["distribution"][k]) for k in sorted(st["distribution"], key=repr) if st["distribution"][k] > 0
        ]
        st["values_sorted"] = [
            (k, st["values"][k]) for k in sorted(st["values"], key=repr) if st["values"][k] > 0
        ]
        out[cm.name] = st
    return out


# ------------------------------------------------------------------ Exact rational budget planning
def shape(value):
    # Same public, fixed rule as frozen C1; not a learned vocabulary.
    s = str(value)
    if re.fullmatch(r'\d{4}-\d{2}-\d{2}', s):
        return 'YYYY-MM-DD'
    sk = ''.join('A' if c.isalpha() else '9' if c.isdigit() else '_'
                 if c.isspace() else c if c in '-/:.()@+%' else '?' for c in s)
    return sk if len(sk) <= 80 else sk[:80] + '…'


def _fixed_edges():
    edges = {0.0}
    for exponent in range(-323, 309):
        for mantissa in (1, 2, 5):
            v = float(f'{mantissa}e{exponent}')
            if math.isfinite(v) and v > 0:
                edges.update((v, -v))
    return tuple(sorted(edges))


EDGES = _fixed_edges()


def bin_index(value):
    try:
        v = float(value)
    except (ValueError, TypeError, OverflowError):
        return -1
    if not math.isfinite(v):
        return -1
    return max(0, min(len(EDGES) - 1, bisect_right(EDGES, v) - 1))


def significant(value):
    return float(format(value, '.2g')) if math.isfinite(value) else None


def top_keys(histogram, cap):
    return sorted(histogram, key=lambda k: (-histogram[k], repr(k)))[:cap]


def quantiles(histogram):
    items = sorted((k, v) for k, v in histogram.items() if k >= 0)
    if not items:
        return [None, None, None]
    total = math.fsum(v for _, v in items)
    result = []
    for fraction in (.25, .5, .75):
        cumulative = 0.0
        for key, weight in items:
            cumulative += weight
            if cumulative >= fraction * total:
                result.append(significant(EDGES[key]))
                break
    return result


def order_of_magnitude(nd: int) -> Union[int, str]:
    """Exact integer order-of-magnitude bucket (avoids float log10(10**k - 1) rounding)."""
    n = int(nd)
    if n <= 0:
        return 0
    return f"10^{len(str(n)) - 1}"


def plan_table_dg_v3(
    public_schema: v3.PublicSchema,
    table_name: str,
    epsilon: Union[int, Fraction],
    delta: Fraction = TABLE_DELTA_FRAC,
    delta_mode: str = "proportional",
    prune_free_text: bool = False,
    scalar_kappa: float = v3.SCALAR_KAPPA,
    rational_scale: int = 10**12,
    equal_weights: bool = False,
    flat_weights: bool = False,
    unpruned_roles: bool = False,
    omega1_kkt: bool = False,
    column_policy: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Exact rational budget plan for DG-v3 (schema-graph convex allocation + Discrete Gaussian)."""
    if table_name not in public_schema.tables:
        raise KeyError(f"Unknown table in public schema: {table_name}")
    if sum((bool(equal_weights), bool(flat_weights), bool(unpruned_roles), bool(omega1_kkt))) > 1:
        raise ValueError("equal_weights, flat_weights, unpruned_roles, and omega1_kkt are mutually exclusive")
    eps_f = Fraction(epsilon)
    delta_f = Fraction(delta)
    delta_half = delta_f / 2
    rho_table = dg.rational_rho_budget(eps_f, delta_half)

    table_cols = public_schema.tables[table_name]
    raw_queries = v3.plan_queries_v3(
        table_cols,
        prune_free_text=prune_free_text,
        column_policy=column_policy,
        table_name=table_name,
    )

    if equal_weights:
        raw_queries = [("row_count", None, "TABLE", 1)]
        for cm in table_cols:
            role = v3.classify_column(cm)
            stats = ["null_count", "distinct_count", "bins" if numeric(cm.declared_type) else "patterns"]
            if (not cm.is_single_pk_or_unique and
                    (not numeric(cm.declared_type) or v3.is_integer_type(cm.declared_type))):
                stats.append("values")
            raw_queries.extend((s, cm.name, role, 1) for s in stats)
        delta_mode = "equal"
    elif flat_weights:
        raw_queries = [(s, c, role, 1) for s, c, role, _ in raw_queries]
        delta_mode = "equal"
    elif unpruned_roles:
        has_values = {c for s, c, _r, _w in raw_queries if s == "values" and c is not None}
        col_meta_map = {cm.name: cm for cm in table_cols}
        expanded = []
        for item in raw_queries:
            expanded.append(item)
            stat, col, role, _w = item
            if stat in ("bins", "patterns") and col is not None and col not in has_values:
                cm = col_meta_map[col]
                if not cm.is_single_pk_or_unique and (
                    not numeric(cm.declared_type) or v3.is_integer_type(cm.declared_type)
                ):
                    expanded.append(("values", col, role, 1.0))
        raw_queries = expanded
    elif omega1_kkt:
        raw_queries = [(s, c, role, 1) for s, c, role, _ in raw_queries]

    # Convert weights to exact rationals
    queries_rat: List[Tuple[str, Optional[str], str, Fraction]] = []
    for stat, col, role, omega in raw_queries:
        if equal_weights or flat_weights or omega1_kkt:
            w_f = Fraction(1)
        elif stat == "row_count":
            w_f = V3_ROW_COUNT_WEIGHT_FRAC
        elif stat in V3_ROLE_WEIGHTS_FRAC[role]:
            w_f = V3_ROLE_WEIGHTS_FRAC[role][stat]
        else:
            w_f = Fraction(str(omega))
        queries_rat.append((stat, col, role, w_f))

    hist_indices = [i for i, (s, _, _, _) in enumerate(queries_rat) if s in HISTOGRAMS]
    n_hist = len(hist_indices)
    if n_hist < len(table_cols) or n_hist < 1:
        raise ValueError("Invalid histogram count")

    if delta_mode == "equal":
        delta_js = {i: delta_half / n_hist for i in hist_indices}
    elif delta_mode == "proportional":
        hist_w_sum = sum((queries_rat[i][3] for i in hist_indices), Fraction(0))
        delta_js = {i: delta_half * (queries_rat[i][3] / hist_w_sum) for i in hist_indices}
    else:
        raise ValueError(f"Unknown delta_mode: {delta_mode}")

    # Closed-form (omega_j * kappa_j)^(2/3) discretized cleanly to positive integers r_j
    eff_scores: List[float] = []
    for i, (stat, _col, _role, w_f) in enumerate(queries_rat):
        if stat in HISTOGRAMS:
            dj = float(delta_js[i])
            kappa = -ND.inv_cdf(dj)
        else:
            kappa = scalar_kappa
        eff_scores.append((float(w_f) * kappa) ** (2.0 / 3.0))

    eff_sum = math.fsum(eff_scores)
    r_ints = [max(1, int(round(rational_scale * (s / eff_sum)))) for s in eff_scores]
    if equal_weights or flat_weights:
        r_ints = [1] * len(eff_scores)
    r_total = sum(r_ints)

    items: List[Dict[str, Any]] = []
    for i, (stat, col, role, w_f) in enumerate(queries_rat):
        rho_q = rho_table * Fraction(r_ints[i], r_total)
        sigma2 = Fraction(1, 2) / rho_q
        item: Dict[str, Any] = {
            "stat": stat,
            "column": col,
            "role": role,
            "weight": float(w_f),
            "weight_frac": _frac_str(w_f),
            "rational_share": r_ints[i],
            "rational_total": r_total,
            "rho_frac": _frac_str(rho_q),
            "rho": float(rho_q),
            "sigma2_frac": _frac_str(sigma2),
            "sigma": math.sqrt(float(sigma2)),
            "l2_sensitivity": 1,
            "_rho_obj": rho_q,
            "_sigma2_obj": sigma2,
        }
        if stat in HISTOGRAMS:
            dj_f = delta_js[i]
            tau, tail_diag = dg.exact_dgauss_tau(sigma2, dj_f)
            item["delta_frac"] = _frac_str(dj_f)
            item["delta"] = float(dj_f)
            item["tau"] = tau
            item["tau_tail_diag"] = tail_diag
            item["_delta_obj"] = dj_f
        else:
            item["delta_frac"] = "0/1"
            item["delta"] = 0.0
            item["_delta_obj"] = Fraction(0)
        items.append(item)

    mechanism = (
        "EQ" if equal_weights
        else "DG-FlatW" if flat_weights
        else "DG-Unpruned" if unpruned_roles
        else "DG-Omega1" if omega1_kkt
        else "DG-v3"
    )
    plan = {
        "version": VERSION_DG_V3,
        "mechanism": mechanism,
        "table": table_name,
        "epsilon": float(eps_f),
        "epsilon_frac": _frac_str(eps_f),
        "delta": float(delta_f),
        "delta_frac": _frac_str(delta_f),
        "delta_conversion_frac": _frac_str(delta_half),
        "delta_thresholds_total_frac": _frac_str(delta_half),
        "delta_mode": delta_mode,
        "prune_free_text": prune_free_text,
        "rho_table_frac": _frac_str(rho_table),
        "rho_table": float(rho_table),
        "n_histograms": n_hist,
        "n_value_histograms": sum(1 for s, _, _, _ in queries_rat if s == "values"),
        "queries": items,
        "_rho_table_obj": rho_table,
        "_delta_obj": delta_f,
        "_eps_obj": eps_f,
    }
    plan["ledger"] = assert_plan_dg(plan)
    return plan


def assert_plan_dg(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Exact rational ledger verification: zero float tolerance on rho and delta."""
    rho_table: Fraction = plan["_rho_table_obj"]
    delta_total_budget: Fraction = plan["_delta_obj"]
    eps_budget: Fraction = plan["_eps_obj"]
    delta_half = delta_total_budget / 2

    rho_used = sum((q["_rho_obj"] for q in plan["queries"]), Fraction(0))
    dth_used = sum((q["_delta_obj"] for q in plan["queries"]), Fraction(0))

    assert rho_used <= rho_table, (rho_used, rho_table)
    assert dth_used <= delta_half, (dth_used, delta_half)

    eps_upper = dg.conservative_eps_from_rho(rho_used, delta_half)
    assert Fraction(eps_upper) <= eps_budget
    assert dth_used + delta_half <= delta_total_budget

    for q in plan["queries"]:
        assert q["_rho_obj"] > 0 and q["l2_sensitivity"] == 1
        assert q["_sigma2_obj"] == Fraction(1, 2) / q["_rho_obj"]
        if q["stat"] in HISTOGRAMS:
            assert q["_delta_obj"] > 0
            assert isinstance(q["tau"], int) and q["tau"] >= 1
    stats = [q["stat"] for q in plan["queries"]]
    assert stats.count("row_count") == 1

    return {
        "rho_used_frac": _frac_str(rho_used),
        "rho_used": float(rho_used),
        "rho_budget_frac": _frac_str(rho_table),
        "delta_thresholds_used_frac": _frac_str(dth_used),
        "delta_total_frac": _frac_str(dth_used + delta_half),
        "epsilon_converted_upper": str(eps_upper),
        "exact_rational_no_overspend": True,
    }


def strip_private_plan_objs(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Return JSON-serializable public plan dictionary without Fraction internal keys."""
    out = {k: v for k, v in plan.items() if not k.startswith("_")}
    out["queries"] = [{k: v for k, v in q.items() if not k.startswith("_")} for q in plan["queries"]]
    return out


# ------------------------------------------------------------------ Discrete Gaussian Release
def noisy_count_dg(value: int, sigma2: Fraction, rng: Optional[dg.IntegerRNG] = None) -> int:
    """Add exact Discrete Gaussian Z ~ N_Z(0, sigma2) to integer count and clamp to >= 0."""
    return max(0, int(value) + dg.sample_dgauss(sigma2, rng))


def threshold_histogram_dg(
    sorted_items: List[Tuple[Any, int]],
    sigma2: Fraction,
    tau: int,
    rng: Optional[dg.IntegerRNG] = None,
) -> Dict[Any, int]:
    """Add exact Discrete Gaussian Z ~ N_Z(0, sigma2) per partition; release iff noisy > tau (strict)."""
    out: Dict[Any, int] = {}
    for key, cnt in sorted_items:
        noisy = int(cnt) + dg.sample_dgauss(sigma2, rng)
        if noisy > tau:
            out[key] = noisy
    return out


def release_table_dg(
    stats: Dict[str, Dict[str, Any]],
    col_metas: Tuple[v3.ColumnMeta, ...],
    nrows: int,
    plan: Dict[str, Any],
    rng: Optional[dg.IntegerRNG] = None,
) -> Dict[str, List[Any]]:
    """Release a DP table profile under exact Discrete Gaussian sampling (zero float RNG)."""
    r_rng = dg.BufferedOSCSPRNG() if rng is None else rng
    q = {(it["stat"], it["column"]): it for it in plan["queries"]}
    rc = q[("row_count", None)]
    n = noisy_count_dg(nrows, rc["_sigma2_obj"], r_rng)
    out: Dict[str, List[Any]] = {}
    for cm in col_metas:
        name, typ = cm.name, cm.declared_type
        need_values = ("values", name) in q
        st = stats[name]
        nn = noisy_count_dg(st["null_count"], q[("null_count", name)]["_sigma2_obj"], r_rng)
        nd = noisy_count_dg(st["distinct_count"], q[("distinct_count", name)]["_sigma2_obj"], r_rng)
        null_fraction = round(min(1.0, nn / max(1, n)), 3)
        order: Union[int, str] = order_of_magnitude(nd)
        if numeric(typ):
            it = q[("bins", name)]
            dist = threshold_histogram_dg(st["distribution_sorted"], it["_sigma2_obj"], it["tau"], r_rng)
            distribution = quantiles(dist)
        else:
            it = q[("patterns", name)]
            dist = threshold_histogram_dg(st["distribution_sorted"], it["_sigma2_obj"], it["tau"], r_rng)
            distribution = top_keys(dist, 3)
        common = None
        if need_values and nd <= 50:
            # Post-processing gate equivalence: when nd > 50, common is unconditionally None
            # independent of values histogram draws, so skipping unused draws preserves exact law.
            it = q[("values", name)]
            vh = threshold_histogram_dg(st["values_sorted"], it["_sigma2_obj"], it["tau"], r_rng)
            common = [decode_val_key(k) for k in top_keys(vh, 15)]
        out[name] = [typ, null_fraction, order, distribution, common]
    return out


def noiseless_table(stats, col_names_types, nrows: int, need_val_cols: Set[str]):
    out = {}
    for name, typ in col_names_types:
        st = stats[name]
        nn = int(st['null_count'])
        nd = int(st['distinct_count'])
        null_fraction = round(min(1.0, nn / max(1, nrows)), 3)
        order = order_of_magnitude(nd)
        dist = dict(st['distribution_sorted'])
        if numeric(typ):
            distribution = quantiles(dist)
        else:
            distribution = top_keys(dist, 3)
        common = None
        if (name in need_val_cols) and nd <= 50:
            vh = dict(st['values_sorted'])
            common = [decode_val_key(k) for k in top_keys(vh, 15)]
        out[name] = [typ, null_fraction, order, distribution, common]
    return out



def no_values(profile):
    return {t: {c: row[:4] + [None if row[4] is None else []]
                for c, row in cols.items()} for t, cols in profile.items()}


def publish(
    conn: sqlite3.Connection,
    epsilon: Union[int, Fraction] = 8,
    mode: str = "DG-v3",
    column_policy: Optional[Dict[str, str]] = None,
    public_tables: Optional[Collection[str]] = None,
    prune_free_text: bool = False,
):
    """Use a dedicated connection and a single read transaction for all statistics.

    Tables listed in ``public_tables`` are deterministically skipped during private
    profiling (zero private reads and omitted from the returned private profile dict).
    """
    if mode not in ("DG-v3", "EQ", "DG-FlatW", "DG-Unpruned", "DG-Omega1", "No-Values", "Noiseless"):
        raise ValueError("Unknown publication mode")
    pub_lower = {t.lower() for t in (public_tables or ())}
    own_transaction = not conn.in_transaction
    if own_transaction:
        conn.execute("BEGIN")
    try:
        schema = v3.extract_public_schema(conn)
        profile, ledgers = {}, {}
        for table, cols in schema.tables.items():
            if table.lower() in pub_lower:
                continue
            plan = plan_table_dg_v3(
                schema,
                table,
                epsilon,
                prune_free_text=prune_free_text,
                equal_weights=(mode == "EQ"),
                flat_weights=(mode == "DG-FlatW"),
                unpruned_roles=(mode == "DG-Unpruned"),
                omega1_kkt=(mode == "DG-Omega1"),
                column_policy=column_policy,
            )
            values = {q["column"] for q in plan["queries"] if q["stat"] == "values"}
            n = conn.execute("SELECT COUNT(*) FROM " + quote(table)).fetchone()[0]
            stats = table_statistics_exact(conn, table, cols, n, values)
            if mode == "Noiseless":
                profile[table] = noiseless_table(stats, [(c.name, c.declared_type) for c in cols], n, values)
            else:
                profile[table] = release_table_dg(stats, cols, n, plan)
                ledgers[table] = strip_private_plan_objs(plan)
        return (no_values(profile) if mode == "No-Values" else profile), ledgers
    finally:
        if own_transaction:
            conn.rollback()


def publish_noiseless(
    db_path: Union[sqlite3.Connection, str, Path],
    *,
    public_tables: Optional[Collection[str]] = None,
    column_policy: Optional[Dict[str, str]] = None,
    prune_free_text: bool = False,
) -> Dict[str, Dict[str, List[Any]]]:
    """Run the exact v3 five-field profile pipeline with Z=0 and tau=0 (noiseless baseline)."""
    if isinstance(db_path, sqlite3.Connection):
        prof, _ = publish(
            db_path,
            epsilon=8,
            mode="Noiseless",
            column_policy=column_policy,
            public_tables=public_tables,
            prune_free_text=prune_free_text,
        )
        return prof
    p_obj = Path(db_path).resolve()
    con = sqlite3.connect(p_obj.as_uri() + "?mode=ro&immutable=1", uri=True)
    try:
        con.execute("PRAGMA query_only=ON")
        prof, _ = publish(
            con,
            epsilon=8,
            mode="Noiseless",
            column_policy=column_policy,
            public_tables=public_tables,
            prune_free_text=prune_free_text,
        )
        return prof
    finally:
        con.close()


# ------------------------------------------------------------------ Public-DDL Token Cap (Cap_pub)
_PROMPT_TOKEN_RE = re.compile(r"[A-Za-z_]+|\d+|[^\sA-Za-z0-9_]+|\s+")
DEFAULT_PROFILE_FORMAT = (
    "Column profile arrays: [type, null_fraction, distinct_order, numeric q25/q50/q75 "
    "(2 significant digits) OR string/date shape patterns, common_values (up to 15 surviving "
    "values; [] if none survive; null if suppressed or distinct > 50)]. 10^n denotes "
    "distinct-count magnitude. "
)
DEFAULT_PROFILE_LEGEND = [
    "type",
    "null_fraction",
    "distinct_order",
    "q25_q50_q75_2sig",
    "shape_patterns",
    "common_values",
    "10^",
]
STATIC_ROWS_HEADER = "Static sample rows from synthetic 1-way DP tables (upfront view):\n"


_BPE_APPROX_RE = re.compile(r"\s+|[A-Za-z]{1,4}|\d{1,3}|[^\sA-Za-z0-9]")


def estimate_prompt_tokens(text: str) -> int:
    """Deterministic pure-stdlib regex token estimator matching Stage 6 accounting."""
    return len(_PROMPT_TOKEN_RE.findall(text))


def estimate_bpe_tokens_approx(text: str) -> int:
    """Deterministic pure-stdlib subword/BPE approximation token estimator."""
    if not text:
        return 0
    return len(_BPE_APPROX_RE.findall(text))


def compute_public_profile_token_cap(
    db_schema: v3.PublicSchema,
    *,
    public_tables: Optional[Collection[str]] = None,
) -> int:
    """Pre-specified public-DDL-only token cap Cap_pub(S_d) = 200 + 20*|T| + 30*|C|.

    Depends strictly on public CREATE TABLE table count |T| and column count |C|
    (zero private row access and zero dependency on any private DP release).
    """
    pub_lower = {t.lower() for t in (public_tables or ())}
    n_tables = 0
    n_cols = 0
    for table, cols in db_schema.tables.items():
        if table.lower() in pub_lower:
            continue
        n_tables += 1
        n_cols += len(cols)
    return 200 + 20 * n_tables + 30 * n_cols


def render_profile_prompt_block(
    profile_obj: Dict[str, Any],
    *,
    profile_format: str = DEFAULT_PROFILE_FORMAT,
    profile_legend: Sequence[str] = DEFAULT_PROFILE_LEGEND,
) -> str:
    """Render the exact Turn-0 profile prompt block serialized in agent.generate_sql."""
    if isinstance(profile_obj, dict) and "_hard_rendered" in profile_obj:
        return str(profile_obj["_hard_rendered"])
    compact = json.dumps(profile_obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return (
        "\nPublic profile-format legend: "
        + profile_format
        + json.dumps(list(profile_legend), ensure_ascii=False)
        + "\nQuestion-independent safe column profile:\n"
        + compact
    )


def truncate_profile_to_public_cap(
    profile_obj: Dict[str, Any],
    cap_tokens: int,
    schema: Optional[v3.PublicSchema] = None,
    *,
    profile_format: str = DEFAULT_PROFILE_FORMAT,
    profile_legend: Sequence[str] = DEFAULT_PROFILE_LEGEND,
    token_counter: Optional[Callable[[str], int]] = None,
    mode: str = "drop_values_first",
) -> Dict[str, Any]:
    """Deterministically trim per-column fields until rendered tokens <= cap_tokens.

    When ``mode="drop_values_first"`` (default), Phases 1-2 trim common_values and
    shape patterns round-robin, and Phase 2b/3 drop distributions/columns.
    When ``mode="preserve_values"`` (ValFirst), common_values on value-bearing columns
    are preserved (trying k_val in 5, 3, 2, 1) while compacting and dropping
    zero-value columns first in reverse table/column order before any value-bearing
    columns are compacted or trimmed.
    Falls back to a minimal hard-capped block if even the legend exceeds ``cap_tokens``
    so ``counter(render_profile_prompt_block(out)) <= cap_tokens`` always holds.
    """
    if mode not in ("drop_values_first", "preserve_values"):
        raise ValueError(f"Unknown truncation mode: {mode!r}")

    out = copy.deepcopy(profile_obj)
    counter = estimate_prompt_tokens if token_counter is None else token_counter

    def _cur_tokens(obj: Optional[Dict[str, Any]] = None) -> int:
        target = out if obj is None else obj
        return counter(
            render_profile_prompt_block(
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
        # Phase 1: round-robin trim common_values lists (long strings >80 chars first, else tail)
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

        # Phase 2: round-robin trim string shape patterns if still above cap_tokens
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

        # Phase 2b: drop remaining optional distributions (single patterns, numeric quantiles)
        for table, col in reversed(col_order):
            if _cur_tokens() <= cap_tokens:
                break
            entry = out[table][col]
            if isinstance(entry, list) and len(entry) >= 4 and isinstance(entry[3], list) and entry[3]:
                entry[3] = []

        if _cur_tokens() <= cap_tokens:
            return out

        # Phase 3: hard truncation of non-essential columns, then essential columns & empty tables
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

    # Phase 4: if even the verbose legend + {"truncated_hard":true} exceeds cap_tokens,
    # render minimal hard-capped payload so counter(rendered) <= cap_tokens ALWAYS holds.
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


truncate_profile_to_cap = truncate_profile_to_public_cap


def render_profile_capped(
    profile_obj: Dict[str, Any],
    cap_tokens: int,
    schema: Optional[v3.PublicSchema] = None,
    *,
    profile_format: str = DEFAULT_PROFILE_FORMAT,
    profile_legend: Sequence[str] = DEFAULT_PROFILE_LEGEND,
    token_counter: Optional[Callable[[str], int]] = None,
    mode: str = "drop_values_first",
) -> str:
    """Truncate ``profile_obj`` to ``cap_tokens`` under ``mode`` and render its prompt block."""
    capped = truncate_profile_to_public_cap(
        profile_obj,
        cap_tokens,
        schema,
        profile_format=profile_format,
        profile_legend=profile_legend,
        token_counter=token_counter,
        mode=mode,
    )
    return render_profile_prompt_block(
        capped,
        profile_format=profile_format,
        profile_legend=profile_legend,
    )


def _format_static_cell(val: Any, max_len: int = 160) -> Any:
    if isinstance(val, (bytes, bytearray)):
        return "<BLOB>"
    if isinstance(val, str) and len(val) > max_len:
        return val[:max_len] + "..."
    return val


def truncate_synth_rows_to_public_cap(
    synth_db: Union[sqlite3.Connection, str, Path],
    cap_tokens: int,
    schema: v3.PublicSchema,
    *,
    max_rows_per_table: int = 60,
    max_cell_len: int = 160,
) -> str:
    """Render deterministic strided synthetic rows capped at <= cap_tokens."""
    owns = False
    if isinstance(synth_db, sqlite3.Connection):
        con = synth_db
    else:
        p_obj = Path(synth_db).resolve()
        con = sqlite3.connect(p_obj.as_uri() + "?mode=ro&immutable=1", uri=True)
        con.execute("PRAGMA query_only=ON")
        owns = True

    try:
        tables = sorted(schema.tables)
        tbl_cols: Dict[str, List[str]] = {}
        tbl_rows: Dict[str, List[str]] = {}
        for t in tables:
            qt = quote(t)
            cols = [cm.name for cm in schema.tables[t]]
            tbl_cols[t] = cols
            nrows = int(con.execute(f"SELECT COUNT(*) FROM {qt}").fetchone()[0])
            if nrows <= 0:
                tbl_rows[t] = []
                continue
            want = min(max_rows_per_table, nrows)
            stride = max(1, nrows // want)
            sel_cols = ", ".join(quote(c) for c in cols)
            try:
                fetched = con.execute(
                    f"SELECT {sel_cols} FROM {qt} WHERE ((rowid - 1) % ?) = 0 ORDER BY rowid LIMIT ?",
                    (stride, want),
                ).fetchall()
            except sqlite3.OperationalError:
                fetched = con.execute(f"SELECT {sel_cols} FROM {qt} LIMIT ?", (want,)).fetchall()
            tbl_rows[t] = [
                json.dumps([_format_static_cell(v, max_cell_len) for v in row], ensure_ascii=False, separators=(",", ":"))
                for row in fetched
            ]
    finally:
        if owns:
            con.close()

    chosen: Dict[str, List[str]] = {t: [] for t in tables}

    def _render() -> str:
        parts = [STATIC_ROWS_HEADER]
        for t in tables:
            parts.append(f"Table {t}({','.join(tbl_cols[t])}):")
            if chosen[t]:
                parts.extend(chosen[t])
            else:
                parts.append("[]")
        return "\n".join(parts) + "\n"

    idx_by_t = {t: 0 for t in tables}
    while True:
        added_any = False
        for t in tables:
            if idx_by_t[t] >= len(tbl_rows[t]):
                continue
            chosen[t].append(tbl_rows[t][idx_by_t[t]])
            if estimate_prompt_tokens(_render()) > cap_tokens:
                chosen[t].pop()
                continue
            idx_by_t[t] += 1
            added_any = True
        if not added_any:
            break

    return _render()
