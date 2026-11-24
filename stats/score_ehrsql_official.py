"""Standalone stdlib EHRSQL official top-100 & uncapped dual-comparator scorer."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sqlite3
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_TIMEOUT_S = 60.0

from silhouette.jsonl_io import iter_jsonl
from silhouette.sandbox import LocalExecutor, error_category

CURRENT_DATE = "2100-12-31"
CURRENT_TIME = "23:59:00"
NOW = f"{CURRENT_DATE} {CURRENT_TIME}"
PRECOMPUTED_DICT = {
    "temperature": (35.5, 38.1),
    "sao2": (95.0, 100.0),
    "heart rate": (60.0, 100.0),
    "respiration": (12.0, 18.0),
    "systolic bp": (90.0, 120.0),
    "diastolic bp": (60.0, 90.0),
    "mean bp": (60.0, 110.0),
}
_TIME_PATTERN = re.compile(r"(DATE_SUB|DATE_ADD)\((\w+\(\)|'[^']+')[, ]+ INTERVAL (\d+) (MONTH|YEAR|DAY)\)")

# Exact >100-row gold queries verified against mimic_iv.sqlite (see ehrsql_874_disagreement_audit.json & p7_core_summary.json)
GT100_GOLD_QIDS_600 = {"8970ce40dfe4e0498bf2959e", "a3703a6f4da15988521301ef"}
GT100_GOLD_QIDS_874 = {
    "06b14b6b3b8637569dd435d9",
    "06b4d2263fcb4017df542024",
    "2294e45331b6e84efb7132a1",
    "879dfc7ea11934d27bc66bc7",
    "949dc9ac2dfec702b23219cc",
    "efed4e108b270dda785398e4",
}


def _convert_date_function(match: re.Match[str]) -> str:
    function = match.group(1)
    date = match.group(2)
    number = match.group(3)
    unit = match.group(4).lower()
    if number == "1":
        unit = unit.rstrip("s")
    else:
        unit = unit + "s" if not unit.endswith("s") else unit
    sign = "-" if function == "DATE_SUB" else "+"
    return f"datetime({date}, '{sign}{number} {unit}')"


def post_process_sql(query: str) -> str:
    """Apply the official EHRSQL MySQL-to-SQLite & temporal/vital-range rewrite."""
    query = re.sub("[ ]+", " ", query.replace("\n", " ")).strip()
    query = query.replace("> =", ">=").replace("< =", "<=").replace("! =", "!=")
    query = _TIME_PATTERN.sub(_convert_date_function, query)

    if "current_time" in query:
        query = query.replace("current_time", f"'{NOW}'")
    if "current_date" in query:
        query = query.replace("current_date", f"'{CURRENT_DATE}'")
    if "'now'" in query:
        query = query.replace("'now'", f"'{NOW}'")
    if "NOW()" in query:
        query = query.replace("NOW()", f"'{NOW}'")
    if "CURDATE()" in query:
        query = query.replace("CURDATE()", f"'{CURRENT_DATE}'")
    if "CURTIME()" in query:
        query = query.replace("CURTIME()", f"'{CURRENT_TIME}'")

    lows = re.findall(r"[ \n]+([a-zA-Z0-9_]+_lower)", query)
    highs = re.findall(r"[ \n]+([a-zA-Z0-9_]+_upper)", query)
    if lows and highs:
        vital_lower_expr = lows[0]
        vital_upper_expr = highs[0]
        vital_name_list = list(
            set(
                re.findall(r"([a-zA-Z0-9_]+)_lower", vital_lower_expr)
                + re.findall(r"([a-zA-Z0-9_]+)_upper", vital_upper_expr)
            )
        )
        if len(vital_name_list) == 1:
            processed_vital_name = vital_name_list[0].replace("_", " ")
            if processed_vital_name in PRECOMPUTED_DICT:
                vital_range = PRECOMPUTED_DICT[processed_vital_name]
                query = query.replace(vital_lower_expr, f"{vital_range[0]}").replace(
                    vital_upper_expr, f"{vital_range[1]}"
                )

    return query.replace("%y", "%Y").replace("%j", "%J")


def _process_item(item: Any) -> str:
    if isinstance(item, bytes):
        item = item.decode("utf-8", "ignore")
    try:
        return str(round(float(item), 3))
    except (TypeError, ValueError, OverflowError):
        return str(item)


def normalize_rows(rows: list[tuple[Any, ...]] | None, *, cap_top100: bool = True) -> str:
    """Official EHRSQL sorted normalized row serialization (top-100 or uncapped)."""
    if rows is None:
        return "error_pred"
    normalized = sorted([[_process_item(c) for c in row] for row in rows])
    return str(normalized[:100] if cap_top100 else normalized)


def _categorize_sqlite_error(exc: Exception) -> str:
    low = str(exc).lower()
    if "interrupted" in low:
        return "timeout"
    if "no such column" in low:
        return "unknown_column"
    if "no such table" in low:
        return "unknown_table"
    if "syntax" in low:
        return "syntax_error"
    if "ambiguous" in low:
        return "ambiguous_column"
    if "no such function" in low:
        return "unknown_function"
    if "misuse of aggregate" in low or "misuse of window" in low:
        return "aggregate_misuse"
    return "other_sqlite_error"


def execute_on_connection(
    con: sqlite3.Connection,
    sql: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> tuple[list[tuple[Any, ...]] | None, str | None, str | None]:
    """Execute post-processed SQL on an open SQLite connection with progress timeout."""
    if not isinstance(sql, str) or not sql.strip():
        return None, "empty_sql", "empty_sql"
    psql = post_process_sql(sql)
    deadline = time.monotonic() + timeout_s
    con.set_progress_handler(lambda: int(time.monotonic() >= deadline), 5000)
    try:
        rows = con.execute(psql).fetchall()
        return rows, None, None
    except Exception as exc:
        return None, _categorize_sqlite_error(exc), f"{type(exc).__name__}: {exc}"
    finally:
        con.set_progress_handler(None, 0)


def execute_ro(
    db_path: Path,
    sql: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> tuple[list[tuple[Any, ...]] | None, str | None, str | None]:
    """Execute EHRSQL query read-only against mimic_iv.sqlite."""
    if not isinstance(sql, str) or not sql.strip():
        return None, "empty_sql", "empty_sql"
    con = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", "ignore")
    con.execute("PRAGMA query_only=ON")
    con.execute("PRAGMA temp_store=MEMORY")
    try:
        return execute_on_connection(con, sql, timeout_s=timeout_s)
    finally:
        con.close()


def score_rows(
    pred_rows: list[tuple[Any, ...]] | None,
    pred_err: str | None,
    gold_rows: list[tuple[Any, ...]],
) -> tuple[int, int]:
    """Compute (official_top100_ex, uncapped_ex) from fetched result rows."""
    if pred_err is not None or pred_rows is None:
        return 0, 0
    official_ex = int(normalize_rows(pred_rows, cap_top100=True) == normalize_rows(gold_rows, cap_top100=True))
    uncapped_ex = int(normalize_rows(pred_rows, cap_top100=False) == normalize_rows(gold_rows, cap_top100=False))
    return official_ex, uncapped_ex


def score_sql_on_connection(
    con: sqlite3.Connection,
    pred_sql: str,
    gold_sql: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Execute gold & predicted SQL on an open connection under both top-100 and uncapped comparators."""
    g_rows, g_err, g_msg = execute_on_connection(con, gold_sql, timeout_s=timeout_s)
    if g_err is not None or g_rows is None:
        raise RuntimeError(f"Gold SQL failed: {g_err}: {g_msg}")
    p_rows, p_err, p_msg = execute_on_connection(con, pred_sql, timeout_s=timeout_s)
    official_ex, uncapped_ex = score_rows(p_rows, p_err, g_rows)
    return {
        "official_set_ex": official_ex,
        "uncapped_ex": uncapped_ex,
        "gold_rows": len(g_rows),
        "pred_rows": len(p_rows) if p_rows is not None else None,
        "runtime_error": p_err,
        "runtime_error_msg": p_msg,
    }


def validate_ehrsql_receipts_against_outcomes(root: Path = ROOT) -> dict[str, Any]:
    """Verify 100% slot-by-slot agreement between EHRSQL outcomes and receipts + gt100 uncapped equality."""
    ehr_man = json.loads((root / "manifests/ehrsql_2024_manifest.json").read_text(encoding="utf-8"))
    ehr_600_ids = {x["ehrsql_id"] for x in ehr_man["confirmation"]}
    ehr_874_ids = set(ehr_man["official_test"]["held_out"])

    outcomes_by_bcq: dict[tuple[str, str, str], int] = {}
    outcomes_by_cond: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    for r in iter_jsonl(root / "stats/per_question_outcomes.jsonl"):
        qid = str(r["question_id"])
        if r["db_id"] != "mimic_iv":
            continue
        bm = r.get("benchmark")
        if bm is None:
            bm = "ehrsql_600" if qid in ehr_600_ids else ("ehrsql_874" if qid in ehr_874_ids else "ehrsql_dev")
        if bm not in ("ehrsql_600", "ehrsql_874"):
            continue
        key = (bm, r["condition"], qid)
        assert key not in outcomes_by_bcq, f"Duplicate outcome key: {key}"
        outcomes_by_bcq[key] = int(r["outcome"])
        outcomes_by_cond[(bm, r["condition"])][qid] = int(r["outcome"])

    receipts_by_bcq: dict[tuple[str, str, str], dict[str, Any]] = {}
    for r in iter_jsonl(root / "stats/sql_and_audit_receipts.jsonl"):
        bm = r.get("benchmark")
        if bm not in ("ehrsql_600", "ehrsql_874"):
            continue
        assert r["db_id"] == "mimic_iv", r
        key = (bm, r["condition"], str(r["question_id"]))
        assert key not in receipts_by_bcq, f"Duplicate receipt key: {key}"
        receipts_by_bcq[key] = r

    missing_keys = set(outcomes_by_bcq) - set(receipts_by_bcq)
    extra_keys = set(receipts_by_bcq) - set(outcomes_by_bcq)
    assert not missing_keys, f"EHRSQL outcome keys missing from receipts: {sorted(missing_keys)[:10]}"
    assert not extra_keys, f"EHRSQL receipt keys missing from outcomes: {sorted(extra_keys)[:10]}"

    gt100_checked = 0
    gt100_none = 0
    gt100_distinct_qids: set[str] = set()
    uncapped_checked = 0
    for key, out_ex in outcomes_by_bcq.items():
        bm, cond, qid = key
        rec = receipts_by_bcq[key]
        assert rec.get("predicted_sql") is not None, f"Null predicted_sql for {key}"
        assert rec.get("total_tokens") is not None, f"Null total_tokens for {key}"
        assert int(rec["official_set_ex"]) == out_ex, f"Outcome mismatch for {key}: rec={rec['official_set_ex']} out={out_ex}"
        is_gt100 = (
            qid in GT100_GOLD_QIDS_600
            or qid in GT100_GOLD_QIDS_874
            or (rec.get("gold_rows") or 0) > 100
        )
        if is_gt100:
            if rec.get("uncapped_ex") is None:
                gt100_none += 1
            else:
                assert int(rec["uncapped_ex"]) == int(rec["official_set_ex"]), (
                    f"gt100 uncapped != official_set_ex for {key}"
                )
                gt100_checked += 1
                gt100_distinct_qids.add(qid)
        if rec.get("uncapped_ex") is not None:
            uncapped_checked += 1

    assert gt100_none == 0, f"Expected 0 None uncapped_ex among >100-row slots, got {gt100_none}"
    assert gt100_checked == 132, f"Expected 132/132 >100-row slot instances verified, got {gt100_checked}"

    assert len(outcomes_by_cond) == 40, f"Expected 40 EHRSQL conditions, got {len(outcomes_by_cond)}"
    assert len(outcomes_by_bcq) == 27562, f"Expected 27562 EHRSQL slots, got {len(outcomes_by_bcq)}"
    assert gt100_distinct_qids == (GT100_GOLD_QIDS_600 | GT100_GOLD_QIDS_874), gt100_distinct_qids

    gold_doc = json.loads((root / "manifests/ehrsql_gold_queries.json").read_text(encoding="utf-8"))
    assert set(gold_doc["ehrsql_600"]) == ehr_600_ids and len(gold_doc["ehrsql_600"]) == 600
    assert set(gold_doc["ehrsql_874"]) == ehr_874_ids and len(gold_doc["ehrsql_874"]) == 874
    empty_ehrsql = sum(1 for rec in receipts_by_bcq.values() if rec.get("deliverable_status") == "empty_submission")
    denied_ehrsql = sum(1 for rec in receipts_by_bcq.values() if str(rec.get("deliverable_status", "")).startswith("authorizer_denied:"))
    assert empty_ehrsql == 31, empty_ehrsql
    assert denied_ehrsql == 0, denied_ehrsql
    assert sum(outcomes_by_cond[("ehrsql_600", "DAWA_CLEAN_EHR600_R1/r0")].values()) == 306

    # Assert canonical v3 clinical totals and EHR874_NOISELESS_NOLKP_V3/r0 == 429/874
    v3_nl = outcomes_by_cond[("ehrsql_874", "EHR874_NOISELESS_NOLKP_V3/r0")]
    assert len(v3_nl) == 874 and sum(v3_nl.values()) == 429
    ehr_v3_600_hits = sum(sum(outcomes_by_cond[("ehrsql_600", f"EHR_DG8P_V3_R{r}/r0")].values()) for r in (1, 2, 3))
    ehr_v3_600_nolkp_hits = sum(sum(outcomes_by_cond[("ehrsql_600", f"EHR_DG8P_NOLKP_V3_R{r}/r0")].values()) for r in (1, 2, 3))
    ehr_v3_874_hits = sum(sum(outcomes_by_cond[("ehrsql_874", f"EHR874_DG8P_V3_R{r}/r0")].values()) for r in (1, 2, 3))
    ehr_v3_874_nolkp_hits = sum(sum(outcomes_by_cond[("ehrsql_874", f"EHR874_DG8P_NOLKP_V3_R{r}/r0")].values()) for r in (1, 2, 3))
    assert ehr_v3_600_hits == 893, ehr_v3_600_hits
    assert ehr_v3_600_nolkp_hits == 811, ehr_v3_600_nolkp_hits
    assert ehr_v3_874_hits == 1368, ehr_v3_874_hits
    assert ehr_v3_874_nolkp_hits == 1240, ehr_v3_874_nolkp_hits
    assert sum(outcomes_by_cond[("ehrsql_600", "EHR_ENT8_V3_R1/r0")].values()) == 298
    assert sum(outcomes_by_cond[("ehrsql_600", "EHR_IND8_V3_R1/r0")].values()) == 296

    # Unseen-37 template (235 queries) breakdown for EHR874_DG8P_V3_R1..R3
    strata = json.loads((root / "manifests/question_strata_and_remap.json").read_text(encoding="utf-8"))
    unseen_235 = strata["ehrsql_template_clusters"]["ehr874_unseen_235_qids"]
    seen_639 = strata["ehrsql_template_clusters"]["ehr874_seen_639_qids"]
    assert len(strata["ehrsql_template_clusters"]["ehr874_unseen_37_templates"]) == 37
    assert len(unseen_235) == 235 and len(seen_639) == 639
    unseen_v3_hits = [sum(outcomes_by_cond[("ehrsql_874", f"EHR874_DG8P_V3_R{r}/r0")][q] for q in unseen_235) for r in (1, 2, 3)]
    seen_v3_hits = [sum(outcomes_by_cond[("ehrsql_874", f"EHR874_DG8P_V3_R{r}/r0")][q] for q in seen_639) for r in (1, 2, 3)]

    # Disagreement ruleset concordance vs Full access for EHR_DG8P_V3_R1..R3 (and EHR874_DG8P_V3_R1..R3)
    v3_dis = json.loads((root / "stats/ehrsql_v3_disagreement_audit.json").read_text(encoding="utf-8"))
    concordance_600 = {}
    for rep in (1, 2, 3):
        c = f"EHR_DG8P_V3_R{rep}/r0"
        drec = v3_dis["ehrsql_600_vs_C4"][c]
        w_obs = sum(1 for q in ehr_600_ids if outcomes_by_cond[("ehrsql_600", c)][q] == 1 and outcomes_by_cond[("ehrsql_600", "C4/r0")][q] == 0)
        l_obs = sum(1 for q in ehr_600_ids if outcomes_by_cond[("ehrsql_600", c)][q] == 0 and outcomes_by_cond[("ehrsql_600", "C4/r0")][q] == 1)
        assert drec["dg8p_only_wins"] == w_obs and drec["c4_only_wins"] == l_obs, (c, drec, w_obs, l_obs)
        concordance_600[c] = {"n_disagreements": drec["n_disagreements"], "agreement_pct": drec["agreement_pct"], "cohens_kappa": drec["cohens_kappa"]}

    return {
        "ehrsql_conditions_verified": len(outcomes_by_cond),
        "ehrsql_slots_verified": len(outcomes_by_bcq),
        "empty_submission_ehrsql": empty_ehrsql,
        "authorizer_denied_ehrsql": denied_ehrsql,
        "uncapped_slots_checked": uncapped_checked,
        "gt100_slots_verified": gt100_checked,
        "gt100_distinct_questions": len(gt100_distinct_qids),
        "ehr874_noiseless_v3_hits": sum(v3_nl.values()),
        "ehr_dg8p_v3_600_hits": ehr_v3_600_hits,
        "ehr_dg8p_nolkp_v3_600_hits": ehr_v3_600_nolkp_hits,
        "ehr874_dg8p_v3_hits": ehr_v3_874_hits,
        "ehr874_dg8p_nolkp_v3_hits": ehr_v3_874_nolkp_hits,
        "ehr874_v3_unseen235_hits": unseen_v3_hits,
        "ehr874_v3_seen639_hits": seen_v3_hits,
        "ehr600_v3_disagreement_concordance": concordance_600,
    }


def load_gold_queries(gold_labels_json: Path | None = None, root: Path = ROOT) -> dict[tuple[str, str], str]:
    """Load gold SQL queries keyed by (benchmark, qid) or ('*', qid) from manifests/ehrsql_gold_queries.json."""
    path = gold_labels_json if gold_labels_json is not None else (root / "manifests/ehrsql_gold_queries.json")
    if not path.exists():
        raise FileNotFoundError(f"Missing EHRSQL gold SQL manifest: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[tuple[str, str], str] = {}
    if isinstance(raw, dict) and ("ehrsql_600" in raw or "ehrsql_874" in raw):
        for bm in ("ehrsql_600", "ehrsql_874"):
            for qid, info in raw.get(bm, {}).items():
                sql = info["raw_gold_sql"] if isinstance(info, dict) else str(info)
                out[(bm, str(qid))] = str(sql)
    elif isinstance(raw, dict):
        for k, v in raw.items():
            out[("*", str(k))] = str(v["raw_gold_sql"] if isinstance(v, dict) else v)
    elif isinstance(raw, list):
        for item in raw:
            qid = str(item.get("id") or item.get("ehrsql_id") or item.get("question_id"))
            out[("*", qid)] = str(item.get("query") or item.get("SQL") or item.get("sql"))
    return out


def _execute_ehrsql_deliverable(
    ex: LocalExecutor,
    sql: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> tuple[list[tuple[Any, ...]] | None, str | None]:
    """Execute post-processed EHRSQL query via silhouette.sandbox.LocalExecutor(allow_cte=True)."""
    if not isinstance(sql, str) or not sql.strip():
        return None, "empty_submission"
    psql = post_process_sql(sql)
    try:
        rows = ex.execute(psql, timeout_s=timeout_s, allow_cte=True)
        return rows, None
    except Exception as exc:
        return None, error_category(exc)


def replay_ehrsql_db(
    sqlite_db: Path,
    gold_labels_json: Path | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    conditions: set[str] | None = None,
    *,
    compare_unrestricted: bool = True,
    receipts_path: Path | None = None,
) -> dict[str, Any]:
    """Fail-closed Tier-2 replay of all 26,962 EHRSQL predictions via LocalExecutor(allow_cte=True)."""
    if not sqlite_db.exists():
        raise FileNotFoundError(f"Missing SQLite DB: {sqlite_db}")
    gold_map = load_gold_queries(gold_labels_json, root=ROOT)

    con = sqlite3.connect(f"file:{sqlite_db}?mode=ro&immutable=1", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", "ignore")
    con.execute("PRAGMA query_only=ON")
    con.execute("PRAGMA temp_store=MEMORY")
    ex = LocalExecutor(sqlite_db)

    summary: dict[str, dict[str, Any]] = {}
    gold_cache: dict[tuple[str, str], list[tuple[Any, ...]]] = {}
    rec_file = receipts_path if receipts_path is not None else (ROOT / "stats/sql_and_audit_receipts.jsonl")
    try:
        for r in iter_jsonl(rec_file):
            bm = r.get("benchmark")
            if bm not in ("ehrsql_600", "ehrsql_874"):
                continue
            cond = r["condition"]
            if conditions and cond not in conditions:
                continue
            qid = str(r["question_id"])
            gold_sql = gold_map.get((bm, qid)) or gold_map.get(("*", qid))
            if gold_sql is None:
                raise RuntimeError(
                    f"Missing gold SQL for ({bm}, {qid}) in {gold_labels_json or 'manifests/ehrsql_gold_queries.json'} "
                    "(fail-closed: historical stored fallback removed)"
                )

            key = f"{bm}::{cond}"
            st = summary.setdefault(
                key,
                {
                    "n": 0,
                    "deliverable_set_ex": 0,
                    "deliverable_uncapped_ex": 0,
                    "official_set_ex": 0,
                    "uncapped_ex": 0,
                    "errors": 0,
                    "empty_submission": 0,
                    "authorizer_denied": 0,
                },
            )
            pred_sql = r.get("predicted_sql") or ""
            if not pred_sql.strip():
                st["n"] += 1
                st["empty_submission"] += 1
                continue

            gkey = (bm, qid)
            if gkey not in gold_cache:
                g_rows, g_err = _execute_ehrsql_deliverable(ex, gold_sql, timeout_s=timeout_s)
                if g_err is not None or g_rows is None:
                    raise RuntimeError(f"Gold SQL failed under LocalExecutor for ({bm}, {qid}): {g_err}")
                gold_cache[gkey] = g_rows

            d_rows, d_err = _execute_ehrsql_deliverable(ex, pred_sql, timeout_s=timeout_s)
            d_top100, d_unc = score_rows(d_rows, d_err, gold_cache[gkey])
            st["n"] += 1
            st["deliverable_set_ex"] += d_top100
            st["deliverable_uncapped_ex"] += d_unc
            st["errors"] += int(d_err is not None)
            if d_err == "read_only_denied":
                st["authorizer_denied"] += 1

            if compare_unrestricted:
                u_rows, u_err, _ = execute_on_connection(con, pred_sql, timeout_s=timeout_s)
                u_top100, u_unc = score_rows(u_rows, u_err, gold_cache[gkey])
                st["official_set_ex"] += u_top100
                st["uncapped_ex"] += u_unc
            else:
                st["official_set_ex"] += d_top100
                st["uncapped_ex"] += d_unc
    finally:
        con.close()

    for key, st in sorted(summary.items()):
        n = st["n"]
        st["deliverable_set_ex_pct"] = round(100.0 * st["deliverable_set_ex"] / n, 2) if n else None
        st["deliverable_uncapped_ex_pct"] = round(100.0 * st["deliverable_uncapped_ex"] / n, 2) if n else None
        st["top100_ex_pct"] = round(100.0 * st["official_set_ex"] / n, 2) if n else None
        st["uncapped_ex_pct"] = round(100.0 * st["uncapped_ex"] / n, 2) if n else None
        print(
            f"{key:44s}: n={n:3d} | Deliv={st['deliverable_set_ex']:3d} ({st['deliverable_set_ex_pct']:5.2f}%) "
            f"| Off={st['official_set_ex']:3d} ({st['top100_ex_pct']:5.2f}%) "
            f"| empty={st['empty_submission']} | denied={st['authorizer_denied']} | err={st['errors']}"
        )
    return summary


def run_self_test() -> dict[str, Any]:
    """Run synthetic EHRSQL top-100 vs uncapped + LocalExecutor DDL + fail-closed + offline receipt checks."""
    import tempfile

    con = sqlite3.connect(":memory:")
    con.executescript("""
        CREATE TABLE labevents(subject_id INTEGER, itemid INTEGER, valuenum REAL, charttime TEXT);
    """)
    rows = [(i, 50912, 1.0 + i * 0.0001, "2100-12-30 10:00:00") for i in range(1, 106)]
    con.executemany("INSERT INTO labevents VALUES (?, ?, ?, ?)", rows)
    con.commit()

    # 1. Top-100 vs Uncapped fixture (>100 rows, where row 105 differs)
    gold_sql = "SELECT subject_id, ROUND(valuenum, 4) FROM labevents WHERE charttime >= DATE_SUB(NOW(), INTERVAL 2 DAY)"
    pred_top100_match = (
        "SELECT subject_id, valuenum FROM labevents "
        "WHERE subject_id <= 104 AND charttime >= DATE_SUB(NOW(), INTERVAL 2 DAY)"
    )
    res_trunc = score_sql_on_connection(con, pred_top100_match, gold_sql)
    assert res_trunc["gold_rows"] == 105
    assert res_trunc["official_set_ex"] == 0
    con.execute("INSERT INTO labevents VALUES (999, 50912, 2.0, '2100-12-30 10:00:00')")
    con.commit()
    pred_gt100_tail = (
        "SELECT subject_id, valuenum FROM labevents "
        "WHERE subject_id != 999 AND charttime >= DATE_SUB(NOW(), INTERVAL 2 DAY)"
    )
    res_gt100 = score_sql_on_connection(con, pred_gt100_tail, gold_sql)
    assert res_gt100["official_set_ex"] == 1
    assert res_gt100["uncapped_ex"] == 0

    # 2. Exact match with 3-decimal float rounding & vital range expansion
    res_exact = score_sql_on_connection(con, gold_sql, gold_sql)
    assert res_exact["official_set_ex"] == 1 and res_exact["uncapped_ex"] == 1
    assert "60.0" in post_process_sql("SELECT * FROM chartevents WHERE valuenum BETWEEN mean_bp_lower AND mean_bp_upper")
    con.close()

    # 3. Live LocalExecutor on MIMIC-IV public DDL + CTE + patient_treatment_response view denial + fail-closed unknown qid
    with tempfile.TemporaryDirectory() as tmp:
        tdir = Path(tmp)
        db_p = tdir / "mimic_ddl.sqlite"
        cm = sqlite3.connect(str(db_p))
        ddl_text = (ROOT / "manifests/mimic_iv_public_ddl.sql").read_text(encoding="utf-8")
        cm.executescript(ddl_text)
        cm.execute("CREATE VIEW patient_treatment_response AS SELECT subject_id FROM patients")
        cm.execute("INSERT INTO patients(row_id, subject_id, gender, dob, dod) VALUES (1, 10001, 'F', '2050-01-01', NULL)")
        cm.commit()
        cm.close()

        ex = LocalExecutor(db_p)
        d_rows, d_err = _execute_ehrsql_deliverable(
            ex,
            "WITH p AS MATERIALIZED (SELECT subject_id AS sid FROM patients) SELECT p.sid FROM p",
        )
        assert d_err is None and d_rows == [(10001,)], (d_rows, d_err)
        v_rows, v_err = _execute_ehrsql_deliverable(ex, "SELECT subject_id FROM patient_treatment_response")
        assert v_rows is None and v_err == "read_only_denied", (v_rows, v_err)

        bad_rec = tdir / "bad_receipts.jsonl"
        bad_rec.write_text(
            json.dumps({
                "benchmark": "ehrsql_600",
                "condition": "EHR_DG8P_V3_R1/r0",
                "question_id": "unknown_qid_must_fail_closed",
                "predicted_sql": "SELECT 1",
            }) + "\n",
            encoding="utf-8",
        )
        caught_missing = False
        try:
            replay_ehrsql_db(db_p, receipts_path=bad_rec)
        except RuntimeError as exc:
            caught_missing = "fail-closed" in str(exc)
        assert caught_missing, "Expected fail-closed RuntimeError on missing gold SQL"

    # 4. Validate all 26,962 EHRSQL receipts vs per_question_outcomes.jsonl & manifests/ehrsql_gold_queries.json
    val = validate_ehrsql_receipts_against_outcomes()
    return {"gt100_fixture": res_gt100, **val}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Run synthetic fixtures and offline receipt cross-checks.")
    parser.add_argument("--sqlite-db", type=Path, default=None, help="Path to local mimic_iv.sqlite for full execution replay.")
    parser.add_argument("--gold-json", type=Path, default=None, help="Optional EHRSQL label JSON (defaults to manifests/ehrsql_gold_queries.json).")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="Per-query SQLite timeout in seconds (default 60s).")
    parser.add_argument("--conditions", nargs="*", default=None, help="Optional condition labels to replay.")
    parser.add_argument("--no-compare-unrestricted", action="store_true", help="Skip unrestricted read-only comparison pass.")
    args = parser.parse_args(argv)

    if args.sqlite_db is not None:
        replay_ehrsql_db(
            args.sqlite_db,
            gold_labels_json=args.gold_json,
            timeout_s=args.timeout,
            conditions=set(args.conditions) if args.conditions else None,
            compare_unrestricted=not args.no_compare_unrestricted,
        )
    else:
        res = run_self_test()
        print(
            f"EHRSQL OFFICIAL SCORER SELF-TEST PASSED "
            f"({res['ehrsql_conditions_verified']} conditions, {res['ehrsql_slots_verified']} slots, "
            f"empty={res['empty_submission_ehrsql']}, denied={res['authorizer_denied_ehrsql']}, "
            f"gt100_slots={res['gt100_slots_verified']} across {res['gt100_distinct_questions']} distinct questions [2+6], "
            f"EHR_V3_600={res['ehr_dg8p_v3_600_hits']}/1800, EHR_NOLKP_600={res['ehr_dg8p_nolkp_v3_600_hits']}/1800, "
            f"EHR874_V3={res['ehr874_dg8p_v3_hits']}/2622, EHR874_NOLKP={res['ehr874_dg8p_nolkp_v3_hits']}/2622, "
            f"unseen235={res['ehr874_v3_unseen235_hits']}, EHR874_NL_V3={res['ehr874_noiseless_v3_hits']}/874)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
