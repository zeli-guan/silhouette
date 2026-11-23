"""Database-wide direct-star entity-level Discrete Gaussian DP publisher (CAP=5 + schema_only)."""
from __future__ import annotations

from bisect import insort
from collections import Counter, defaultdict
from fractions import Fraction
import json
import math
import sqlite3
import statistics
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from . import dg_sampler as dg
from . import profile as p
from . import schema_roles as v3
from .schema_roles import is_integer_type, numeric, quote

CAP = 5
VERSION_ENTITY_DP = "c1-entity-dp-v2-direct-star"
VALID_ENTITY_MODES = ("entity_direct", "schema_only")
ND = statistics.NormalDist()

ENTITY_PROFILE_FORMAT = (
    "Entity-level DP column profile arrays (C <= 5 keys/entity): "
    "[type, entity null-presence fraction, distinct_order (capped entity-support), "
    "numeric q25/q50/q75 (2 significant digits, capped entity-support C <= 5 keys/entity) "
    "OR string/date shape patterns (capped entity-support C <= 5 keys/entity), "
    "common_values (capped entity-support C <= 5 keys/entity, up to 15 surviving values; "
    "[] if none survive; null if suppressed, schema_only, or distinct > 50)]. "
    "10^n denotes distinct-count magnitude. "
)
ENTITY_PROFILE_LEGEND = [
    "type",
    "entity_null_presence_fraction",
    "distinct_order_capped_entity_support_C5",
    "q25_q50_q75_2sig_capped_entity_support_C5",
    "shape_patterns_capped_entity_support_C5",
    "common_values_capped_entity_support_C5",
    "10^",
]


BIRD_ENTITY_SPECS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "thrombosis_prediction": {
        "Patient": {"mode": "entity_direct", "entity_col": "ID"},
        "Examination": {"mode": "entity_direct", "entity_col": "ID"},
        "Laboratory": {"mode": "entity_direct", "entity_col": "ID"},
    },
    "financial": {
        "account": {"mode": "entity_direct", "entity_col": "account_id"},
        "loan": {"mode": "entity_direct", "entity_col": "account_id"},
        "order": {"mode": "entity_direct", "entity_col": "account_id"},
        "trans": {"mode": "entity_direct", "entity_col": "account_id"},
        "disp": {"mode": "entity_direct", "entity_col": "account_id"},
        "client": {"mode": "schema_only"},
        "card": {"mode": "schema_only"},
        "district": {"mode": "schema_only"},
    },
    "codebase_community": {
        "users": {"mode": "entity_direct", "entity_col": "Id"},
        "badges": {"mode": "entity_direct", "entity_col": "UserId"},
        "comments": {"mode": "entity_direct", "entity_col": "UserId"},
        "postHistory": {"mode": "entity_direct", "entity_col": "UserId"},
        "votes": {"mode": "entity_direct", "entity_col": "UserId"},
        "posts": {"mode": "entity_direct", "entity_col": "OwnerUserId"},
        "postLinks": {"mode": "schema_only"},
        "tags": {"mode": "schema_only"},
    },
    "debit_card_specializing": {
        "customers": {"mode": "entity_direct", "entity_col": "CustomerID"},
        "yearmonth": {"mode": "entity_direct", "entity_col": "CustomerID"},
        "transactions_1k": {"mode": "entity_direct", "entity_col": "CustomerID"},
        "gasstations": {"mode": "schema_only"},
        "products": {"mode": "schema_only"},
    },
}

EHR_CLINICAL_TABLES = (
    "admissions",
    "chartevents",
    "cost",
    "diagnoses_icd",
    "icustays",
    "inputevents",
    "labevents",
    "microbiologyevents",
    "outputevents",
    "patients",
    "prescriptions",
    "procedures_icd",
    "transfers",
)

EHR_PUBLIC_TABLES = (
    "d_icd_diagnoses",
    "d_icd_procedures",
    "d_items",
    "d_labitems",
)

EHR_ENTITY_SPECS: Dict[str, Dict[str, Any]] = {
    **{t: {"mode": "entity_direct", "entity_col": "subject_id"} for t in EHR_CLINICAL_TABLES},
    **{t: {"mode": "schema_only"} for t in EHR_PUBLIC_TABLES},
}


def format_entity_profile_prompt(profile: Dict[str, Dict[str, List[Any]]], cap: int = CAP) -> str:
    """Render an entity-level DP profile block with explicit entity null-presence & C<=5 semantics."""
    header = (
        f"Public entity-profile legend (capped entity-support (C <= {int(cap)} keys/entity); "
        "null_frac = entity null-presence fraction; quantiles & common_values = capped entity-support "
        f"(C <= {int(cap)} keys/entity)): {ENTITY_PROFILE_FORMAT}"
        + json.dumps(ENTITY_PROFILE_LEGEND, ensure_ascii=False)
    )
    body = json.dumps(profile, ensure_ascii=False, separators=(",", ":"))
    return f"{header}\nQuestion-independent safe entity column profile (C <= {int(cap)} keys/entity):\n{body}"


def validate_entity_specs(
    public_schema: v3.PublicSchema,
    entity_specs: Dict[str, Dict[str, Any]],
) -> None:
    """Validate that entity_specs covers every schema table exactly with valid direct-star modes."""
    schema_tables = set(public_schema.tables)
    spec_tables = set(entity_specs)
    extra = spec_tables - schema_tables
    if extra:
        raise KeyError(f"Unknown tables in entity_specs: {sorted(extra)}")
    missing = schema_tables - spec_tables
    if missing:
        raise KeyError(f"Missing schema tables in entity_specs: {sorted(missing)}")

    for table, cols in public_schema.tables.items():
        spec = entity_specs[table]
        t_mode = spec.get("mode")
        if t_mode not in VALID_ENTITY_MODES:
            raise ValueError(
                f"Unsupported entity table mode for {table!r}: {t_mode!r}; "
                f"allowed modes are {VALID_ENTITY_MODES}"
            )
        if t_mode == "entity_direct":
            ec = spec.get("entity_col")
            col_names = {cm.name for cm in cols}
            if not isinstance(ec, str) or ec not in col_names:
                raise KeyError(
                    f"Entity column {ec!r} not found in table {table!r} columns {sorted(col_names)}"
                )


def _insert_bounded_top_c(top_c: List[Tuple[str, Any]], item: Any, cap: int = CAP) -> None:
    """Maintain at most `cap` lexicographically smallest (by repr) distinct items in O(C) space."""
    if cap <= 0:
        return
    rkey = (repr(item), item)
    for existing in top_c:
        if existing == rkey:
            return
    if len(top_c) < cap:
        insort(top_c, rkey)
    elif rkey < top_c[-1]:
        insort(top_c, rkey)
        top_c.pop()


def _aggregate_bounded(groups: Dict[Any, List[Tuple[str, Any]]]) -> Counter:
    hist: Counter = Counter()
    for group in groups.values():
        hist.update(item for _r, item in group)
    return hist


def _aggregate_capped(groups: Dict[Any, Any], cap: int = CAP) -> Counter:
    hist: Counter = Counter()
    for group in groups.values():
        if isinstance(group, list) and (not group or (isinstance(group[0], tuple) and len(group[0]) == 2 and isinstance(group[0][0], str))):
            # Already bounded (repr, item) list from _insert_bounded_top_c; still handle plain lists safely
            if all(isinstance(x, tuple) and len(x) == 2 and x[0] == repr(x[1]) for x in group):
                hist.update(item for _r, item in group[:cap])
                continue
        hist.update(sorted(group, key=repr)[:cap])
    return hist


def entity_table_statistics(
    conn: sqlite3.Connection,
    table: str,
    col_metas: Tuple[v3.ColumnMeta, ...],
    spec: Dict[str, Any],
    need_values_cols: Set[str],
    cap: int = CAP,
) -> Tuple[int, Dict[str, Dict[str, Any]]]:
    """Compute exact entity-capped statistics with O(C) bounded memory per (entity, stat)."""
    mode = spec["mode"]
    if mode == "schema_only":
        return 0, {}
    if mode != "entity_direct":
        raise ValueError(f"Unsupported entity table mode: {mode!r}")

    ec = quote(spec["entity_col"])
    qt = quote(table)
    ent_rows = conn.execute(f"SELECT typeof({ec}), {ec} FROM {qt} WHERE {ec} IS NOT NULL")
    nrows = len({p.canonical_val_key(et, ev) for et, ev in ent_rows})

    out: Dict[str, Dict[str, Any]] = {}
    for cm in col_metas:
        missing: Set[Tuple[str, Any]] = set()
        keys: Dict[Tuple[str, Any], List[Tuple[str, Any]]] = defaultdict(list)
        buckets: Dict[Tuple[str, Any], List[Tuple[str, Any]]] = defaultdict(list)
        numeric_col = numeric(cm.declared_type)
        qc = quote(cm.name)
        sql = f"SELECT typeof({ec}), {ec}, typeof({qc}), {qc} FROM {qt} WHERE {ec} IS NOT NULL"
        for et, ev, tag, value in conn.execute(sql):
            ent_key = p.canonical_val_key(et, ev)
            if value is None:
                missing.add(ent_key)
            else:
                _insert_bounded_top_c(keys[ent_key], p.canonical_val_key(tag, value), cap)
                _insert_bounded_top_c(
                    buckets[ent_key], p.bin_index(value) if numeric_col else p.shape(value), cap
                )
        values_hist = _aggregate_bounded(keys)
        dist_hist = _aggregate_bounded(buckets)
        out[cm.name] = {
            "null_count": len(missing),
            "distinct_count": len(values_hist),
            "distribution": dist_hist,
            "values": values_hist if cm.name in need_values_cols else Counter(),
            "distribution_sorted": [
                (k, dist_hist[k]) for k in sorted(dist_hist, key=repr) if dist_hist[k] > 0
            ],
            "values_sorted": (
                [(k, values_hist[k]) for k in sorted(values_hist, key=repr) if values_hist[k] > 0]
                if cm.name in need_values_cols
                else []
            ),
        }
    return nrows, out


def plan_entity_db(
    public_schema: v3.PublicSchema,
    entity_specs: Dict[str, Dict[str, Any]],
    epsilon: Union[int, Fraction] = 8,
    delta: Fraction = p.TABLE_DELTA_FRAC,
    mode: str = "Entity-DG-v3",
    prune_free_text: bool = False,
    scalar_kappa: float = v3.SCALAR_KAPPA,
    rational_scale: int = 10**12,
    cap: int = CAP,
    column_policy: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Global convex (or uniform 1:1:1 unpruned) budget plan across all entity_direct tables."""
    if mode not in ("Entity-DG-v3", "Entity-IND8-Prompt"):
        raise ValueError(f"Unknown entity publication mode: {mode!r}")
    validate_entity_specs(public_schema, entity_specs)

    eps_f = Fraction(epsilon)
    delta_f = Fraction(delta)
    delta_half = delta_f / 2
    rho_db = dg.rational_rho_budget(eps_f, delta_half)

    raw_items: List[Dict[str, Any]] = []
    table_caps: Dict[str, int] = {}
    for table, cols in public_schema.tables.items():
        spec = entity_specs[table]
        if spec["mode"] == "schema_only":
            table_caps[table] = 0
            continue
        sens_cap = cap
        table_caps[table] = sens_cap

        if mode == "Entity-DG-v3":
            raw_q = v3.plan_queries_v3(
                cols,
                prune_free_text=prune_free_text,
                column_policy=column_policy,
                table_name=table,
            )
            for stat, col, role, _omega in raw_q:
                w_f = (
                    p.V3_ROW_COUNT_WEIGHT_FRAC
                    if stat == "row_count"
                    else p.V3_ROLE_WEIGHTS_FRAC[role][stat]
                )
                hist = stat in p.HISTOGRAMS
                sens2 = sens_cap if hist else (sens_cap * sens_cap if stat == "distinct_count" else 1)
                raw_items.append({
                    "table": table,
                    "stat": stat,
                    "column": col,
                    "role": role,
                    "weight_frac": w_f,
                    "sens_cap": sens_cap,
                    "sensitivity2": sens2,
                    "hist": hist,
                })
        else:
            # Uniform 1:1:1 unpruned 1-way control under identical direct-star entity sensitivity
            raw_q = [("row_count", None, "TABLE")]
            for cm in cols:
                role = v3.classify_column(cm)
                stats = ["null_count", "distinct_count", "bins" if numeric(cm.declared_type) else "patterns"]
                if not cm.is_single_pk_or_unique and (
                    not numeric(cm.declared_type) or is_integer_type(cm.declared_type)
                ):
                    stats.append("values")
                raw_q.extend((s, cm.name, role) for s in stats)
            for stat, col, role in raw_q:
                hist = stat in p.HISTOGRAMS
                sens2 = sens_cap if hist else (sens_cap * sens_cap if stat == "distinct_count" else 1)
                raw_items.append({
                    "table": table,
                    "stat": stat,
                    "column": col,
                    "role": role,
                    "weight_frac": Fraction(1),
                    "sens_cap": sens_cap,
                    "sensitivity2": sens2,
                    "hist": hist,
                })

    hist_indices = [i for i, it in enumerate(raw_items) if it["hist"]]
    n_hist = len(hist_indices)
    if n_hist < 1:
        raise ValueError("Invalid histogram count: at least one entity_direct table is required")

    if mode == "Entity-IND8-Prompt":
        delta_mode = "equal"
        delta_js = {i: delta_half / n_hist for i in hist_indices}
        r_ints = [1] * len(raw_items)
    else:
        delta_mode = "proportional"
        hist_w_sum = sum((raw_items[i]["weight_frac"] for i in hist_indices), Fraction(0))
        delta_js = {i: delta_half * (raw_items[i]["weight_frac"] / hist_w_sum) for i in hist_indices}
        eff_scores: List[float] = []
        for i, it in enumerate(raw_items):
            if it["hist"]:
                dj_bin = float(delta_js[i] / it["sens_cap"])
                kappa = -ND.inv_cdf(dj_bin)
            else:
                kappa = scalar_kappa
            eff_scores.append((float(it["weight_frac"]) * kappa * math.sqrt(it["sensitivity2"])) ** (2.0 / 3.0))
        eff_sum = math.fsum(eff_scores)
        r_ints = [max(1, int(round(rational_scale * (s / eff_sum)))) for s in eff_scores]

    r_total = sum(r_ints)
    items: List[Dict[str, Any]] = []
    for i, it in enumerate(raw_items):
        rho_q = rho_db * Fraction(r_ints[i], r_total)
        sens2 = it["sensitivity2"]
        sigma2 = Fraction(sens2) / (2 * rho_q)
        share_q = Fraction(r_ints[i], r_total)
        entry: Dict[str, Any] = {
            "table": it["table"],
            "stat": it["stat"],
            "column": it["column"],
            "role": it["role"],
            "weight": float(it["weight_frac"]),
            "weight_frac": p._frac_str(it["weight_frac"]),
            "r_j": f"{r_ints[i]}/1",
            "rational_share": r_ints[i],
            "rational_total": r_total,
            "share": p._frac_str(share_q),
            "rho": p._frac_str(rho_q),
            "sigma2": p._frac_str(sigma2),
            "sigma": math.sqrt(float(sigma2)),
            "sensitivity2": sens2,
            "sens_cap": it["sens_cap"],
            "_rho_obj": rho_q,
            "_sigma2_obj": sigma2,
        }
        if it["hist"]:
            dj_f = delta_js[i]
            tau, tail_diag = dg.exact_dgauss_tau(sigma2, dj_f / it["sens_cap"])
            entry["delta"] = p._frac_str(dj_f)
            entry["tau"] = tau
            entry.update(tail_diag)
            entry["_delta_obj"] = dj_f
        else:
            entry["delta"] = "0/1"
            entry["_delta_obj"] = Fraction(0)
        items.append(entry)

    plan = {
        "version": VERSION_ENTITY_DP,
        "mechanism": mode,
        "epsilon": float(eps_f),
        "epsilon_frac": p._frac_str(eps_f),
        "delta": float(delta_f),
        "delta_frac": p._frac_str(delta_f),
        "delta_conversion_frac": p._frac_str(delta_half),
        "delta_thresholds_total_frac": p._frac_str(delta_half),
        "delta_mode": delta_mode,
        "cap": cap,
        "rho_db_frac": p._frac_str(rho_db),
        "rho_db": float(rho_db),
        "n_histograms": n_hist,
        "n_value_histograms": sum(1 for it in raw_items if it["stat"] == "values"),
        "table_sensitivity_caps": table_caps,
        "queries": items,
        "_rho_db_obj": rho_db,
        "_delta_obj": delta_f,
        "_eps_obj": eps_f,
    }
    plan["ledger"] = assert_plan_entity_db(plan)
    return plan


def assert_plan_entity_db(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Exact rational ledger verification for database-wide direct-star entity DP."""
    rho_db: Fraction = plan["_rho_db_obj"]
    delta_total: Fraction = plan["_delta_obj"]
    eps_budget: Fraction = plan["_eps_obj"]
    delta_half = delta_total / 2

    rho_used = sum((q["_rho_obj"] for q in plan["queries"]), Fraction(0))
    dth_used = sum((q["_delta_obj"] for q in plan["queries"]), Fraction(0))
    assert rho_used <= rho_db, (rho_used, rho_db)
    assert dth_used <= delta_half, (dth_used, delta_half)

    eps_upper = dg.conservative_eps_from_rho(rho_used, delta_half)
    assert Fraction(eps_upper) <= eps_budget, (eps_upper, eps_budget)
    assert dth_used + delta_half <= delta_total

    for q in plan["queries"]:
        assert q["_rho_obj"] > 0
        assert q["_sigma2_obj"] * 2 * q["_rho_obj"] == q["sensitivity2"]
        if q["stat"] in p.HISTOGRAMS:
            assert q["_delta_obj"] > 0
            assert isinstance(q["tau"], int) and q["tau"] >= 1

    return {
        "rho_used_frac": p._frac_str(rho_used),
        "rho_used": float(rho_used),
        "rho_budget_frac": p._frac_str(rho_db),
        "delta_thresholds_used_frac": p._frac_str(dth_used),
        "delta_total_frac": p._frac_str(dth_used + delta_half),
        "epsilon_converted_upper": str(eps_upper),
        "exact_rational_no_overspend": True,
    }


def _export_certificate_metadata(
    db_id: str,
    plan: Dict[str, Any],
    entity_specs: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Build .dpmeta.json certificate sidecar matching verify_certificates.py parameter schema."""
    param_keys = (
        "r_j", "share", "rho", "sigma2", "delta", "sensitivity2",
        "tau", "tail_lower", "tail_upper", "prev_tail_lower", "prev_tail_upper",
    )
    params: List[Dict[str, Any]] = []
    param_idx_by_sig: Dict[Tuple[Any, ...], int] = {}
    tables_meta: Dict[str, Dict[str, Any]] = {}

    by_table: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for q in plan["queries"]:
        by_table[q["table"]].append(q)

    for table, cap_val in plan["table_sensitivity_caps"].items():
        t_mode = entity_specs[table]["mode"]
        if t_mode == "schema_only":
            assert not by_table[table] and cap_val == 0
            tables_meta[table] = {
                "mode": "schema_only",
                "rho": "0/1",
                "delta_threshold": "0/1",
                "sensitivity_cap": 0,
                "queries": [],
            }
            continue

        t_queries = by_table[table]
        rho_t = sum((q["_rho_obj"] for q in t_queries), Fraction(0))
        dth_t = sum((q["_delta_obj"] for q in t_queries), Fraction(0))
        q_refs: List[List[Any]] = []
        for q in t_queries:
            prec = {k: q[k] for k in param_keys if k in q}
            sig = tuple(sorted(prec.items()))
            if sig not in param_idx_by_sig:
                param_idx_by_sig[sig] = len(params)
                params.append(prec)
            q_refs.append([q["stat"], q["column"], param_idx_by_sig[sig]])
        tables_meta[table] = {
            "mode": "entity_direct",
            "rho": p._frac_str(rho_t),
            "delta_threshold": p._frac_str(dth_t),
            "sensitivity_cap": cap_val,
            "queries": q_refs,
        }

    return {
        "db_id": db_id,
        "version": plan["version"],
        "mechanism": plan["mechanism"],
        "epsilon": int(plan["epsilon"]) if float(plan["epsilon"]).is_integer() else plan["epsilon"],
        "delta": plan["delta_frac"],
        "cap": plan["cap"],
        "delta_mode": plan["delta_mode"],
        "rho_db": plan["ledger"]["rho_used_frac"],
        "rho_budget": plan["rho_db_frac"],
        "delta_threshold_total": plan["ledger"]["delta_thresholds_used_frac"],
        "epsilon_converted_upper": plan["ledger"]["epsilon_converted_upper"],
        "table_sensitivity_caps": plan["table_sensitivity_caps"],
        "parameters": params,
        "tables": tables_meta,
    }


def publish_entity_db(
    conn: sqlite3.Connection,
    entity_specs: Dict[str, Dict[str, Any]],
    db_id: str = "database",
    epsilon: Union[int, Fraction] = 8,
    delta: Fraction = p.TABLE_DELTA_FRAC,
    mode: str = "Entity-DG-v3",
    prune_free_text: bool = False,
    column_policy: Optional[Dict[str, str]] = None,
    rng: Optional[dg.IntegerRNG] = None,
) -> Tuple[Dict[str, Dict[str, List[Any]]], Dict[str, Any]]:
    """Publish a database-wide direct-star entity-level DP 5-field profile and verified certificate metadata."""
    r_rng = dg.BufferedOSCSPRNG() if rng is None else rng
    own_transaction = not conn.in_transaction
    if own_transaction:
        conn.execute("BEGIN")
    try:
        schema = v3.extract_public_schema(conn)
        plan = plan_entity_db(
            schema,
            entity_specs,
            epsilon=epsilon,
            delta=delta,
            mode=mode,
            prune_free_text=prune_free_text,
            column_policy=column_policy,
        )
        by_table_q: Dict[str, Dict[Tuple[str, Optional[str]], Dict[str, Any]]] = defaultdict(dict)
        for q in plan["queries"]:
            by_table_q[q["table"]][(q["stat"], q["column"])] = q

        profile: Dict[str, Dict[str, List[Any]]] = {}
        for table, cols in schema.tables.items():
            spec = entity_specs[table]
            if spec["mode"] == "schema_only":
                profile[table] = {
                    cm.name: [cm.declared_type, None, None, None, None]
                    for cm in cols
                }
                continue
            tq = by_table_q[table]
            need_values = {c for (s, c) in tq if s == "values" and c is not None}
            nrows, stats = entity_table_statistics(
                conn, table, cols, spec, need_values, cap=plan["cap"]
            )
            sub_plan = {"queries": list(tq.values())}
            profile[table] = p.release_table_dg(stats, cols, nrows, sub_plan, rng=r_rng)

        meta = _export_certificate_metadata(db_id, plan, entity_specs)
        return profile, meta
    finally:
        if own_transaction:
            conn.rollback()
