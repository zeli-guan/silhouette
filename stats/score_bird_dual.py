"""Standalone stdlib dual-comparator scorer (Official BIRD Set EX vs Strict Ordered/Multiset EX)."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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

from silhouette.jsonl_io import iter_jsonl, load_jsonl
from silhouette.sandbox import LocalExecutor, error_category


def has_top_level_order_by(sql: str) -> bool:
    """Return True if SQL has an ORDER BY clause at parenthesis depth 0 outside strings/comments."""
    if not isinstance(sql, str):
        return False
    n = len(sql)
    i = 0
    depth = 0
    top_chars: list[str] = []
    while i < n:
        ch = sql[i]
        if ch == "-" and i + 1 < n and sql[i + 1] == "-":
            j = sql.find("\n", i + 2)
            i = n if j == -1 else j + 1
            top_chars.append(" ")
            continue
        if ch == "/" and i + 1 < n and sql[i + 1] == "*":
            j = sql.find("*/", i + 2)
            i = n if j == -1 else j + 2
            top_chars.append(" ")
            continue
        if ch in ("'", '"', "`"):
            q = ch
            i += 1
            while i < n:
                if sql[i] == q:
                    if i + 1 < n and sql[i + 1] == q:
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            top_chars.append(" ")
            continue
        if ch == "[":
            j = sql.find("]", i + 1)
            i = n if j == -1 else j + 1
            top_chars.append(" ")
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif depth == 0:
            top_chars.append(ch)
        i += 1
    top_sql = "".join(top_chars)
    return re.search(r"\bORDER\s+BY\b", top_sql, flags=re.IGNORECASE) is not None


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


_NUM_PREFIX_RE = re.compile(r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")


def _sqlite_iif(cond: Any, t: Any, f: Any) -> Any:
    """Deterministic Python emulation of SQLite IIF(cond, t, f) for SQLite < 3.32.0."""
    if cond is None:
        return f
    if isinstance(cond, (int, float)):
        return t if cond != 0 else f
    if isinstance(cond, (bytes, bytearray)):
        cond = cond.decode("utf-8", "replace")
    s = str(cond)
    m = _NUM_PREFIX_RE.match(s)
    if m:
        try:
            return t if float(m.group(1)) != 0.0 else f
        except ValueError:
            return f
    return f


def register_compat_functions(con: sqlite3.Connection, force: bool = False) -> None:
    """Register IIF(cond, t, f) on SQLite connections when sqlite_version_info < (3, 32, 0)."""
    if force or sqlite3.sqlite_version_info < (3, 32, 0):
        con.create_function("iif", 3, _sqlite_iif)


def execute_ro(db_path: Path, sql: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> tuple[list[tuple[Any, ...]] | None, str | None, str | None]:
    """Execute SQL read-only against a SQLite file with a progress-handler timeout."""
    if not isinstance(sql, str) or not sql.strip():
        return None, "empty_sql", "empty_sql"
    con = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    con.execute("PRAGMA query_only=ON")
    con.execute("PRAGMA temp_store=MEMORY")
    register_compat_functions(con)
    return execute_on_connection(con, sql, timeout_s=timeout_s, close_after=True)


def execute_on_connection(
    con: sqlite3.Connection,
    sql: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    close_after: bool = False,
) -> tuple[list[tuple[Any, ...]] | None, str | None, str | None]:
    """Execute SQL on an open SQLite connection with a progress-handler timeout."""
    if not isinstance(sql, str) or not sql.strip():
        if close_after:
            con.close()
        return None, "empty_sql", "empty_sql"
    register_compat_functions(con)
    deadline = time.monotonic() + timeout_s
    con.set_progress_handler(lambda: int(time.monotonic() >= deadline), 5000)
    try:
        rows = con.execute(sql).fetchall()
        return rows, None, None
    except Exception as exc:
        return None, _categorize_sqlite_error(exc), f"{type(exc).__name__}: {exc}"
    finally:
        con.set_progress_handler(None, 0)
        if close_after:
            con.close()


def score_rows(
    pred_rows: list[tuple[Any, ...]] | None,
    pred_err: str | None,
    gold_rows: list[tuple[Any, ...]],
    has_top_order: bool,
) -> tuple[int, int, str]:
    """Compute (official_set_ex, strict_ex, strict_mode) from fetched result rows."""
    mode = "ordered" if has_top_order else "multiset"
    if pred_err is not None or pred_rows is None:
        return 0, 0, mode
    official_set_ex = int(set(pred_rows) == set(gold_rows))
    if has_top_order:
        strict_ex = int(list(pred_rows) == list(gold_rows))
    else:
        strict_ex = int(Counter(pred_rows) == Counter(gold_rows))
    return official_set_ex, strict_ex, mode


def score_sql_on_connection(
    con: sqlite3.Connection,
    pred_sql: str,
    gold_sql: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Execute gold & predicted SQL on an open connection and score under both comparators."""
    has_top_order = has_top_level_order_by(gold_sql)
    g_rows, g_err, g_msg = execute_on_connection(con, gold_sql, timeout_s=timeout_s)
    if g_err is not None or g_rows is None:
        raise RuntimeError(f"Gold SQL failed: {g_err}: {g_msg}")
    p_rows, p_err, p_msg = execute_on_connection(con, pred_sql, timeout_s=timeout_s)
    set_ex, strict_ex, mode = score_rows(p_rows, p_err, g_rows, has_top_order)
    return {
        "official_set_ex": set_ex,
        "strict_ex": strict_ex,
        "strict_mode": mode,
        "runtime_error": p_err,
        "runtime_error_msg": p_msg,
    }


def validate_summary_against_receipts(root: Path = ROOT) -> dict[str, Any]:
    """Verify stats/dual_comparator_summary.json agrees 100% with stats/sql_and_audit_receipts.jsonl & per_question_outcomes.jsonl."""
    summary = json.loads((root / "stats/dual_comparator_summary.json").read_text(encoding="utf-8"))
    outcomes_by_bcq = {
        (r["benchmark"], r["condition"], r["question_id"]): int(r["outcome"])
        for r in iter_jsonl(root / "stats/per_question_outcomes.jsonl")
    }
    receipts = load_jsonl(root / "stats/sql_and_audit_receipts.jsonl")
    by_cond: dict[str, list[dict[str, Any]]] = defaultdict(list)
    conflicts = 0
    for r in receipts:
        key = (r["benchmark"], r["condition"], r["question_id"])
        out_val = outcomes_by_bcq[key]
        if int(r["official_set_ex"]) != out_val:
            conflicts += 1
        if r.get("replay_60s_set_ex") is not None:
            if int(r["replay_60s_set_ex"]) != int(r["official_set_ex"]) or int(r["replay_60s_set_ex"]) != out_val:
                conflicts += 1
        if r.get("runtime_error") is not None and (int(r["official_set_ex"]) != 0 or out_val != 0):
            conflicts += 1
        if r.get("strict_ex") is not None:
            by_cond[r["condition"]].append(r)

    assert conflicts == 0, f"Found {conflicts} outcome/timeout conflicts across receipts!"
    assert set(summary["conditions"]) == set(by_cond), (
        set(summary["conditions"]) ^ set(by_cond)
    )
    for cond, items in by_cond.items():
        rec = summary["conditions"][cond]
        n = len(items)
        set_cnt = sum(x["replay_60s_set_ex"] for x in items)
        pub_cnt = sum(x["official_set_ex"] for x in items)
        strict_cnt = sum(x["strict_ex"] for x in items)
        err_cnt = Counter(x["runtime_error"] for x in items if x["runtime_error"] is not None)
        assert rec["n"] == n, (cond, rec["n"], n)
        assert rec["set_ex_cnt"] == set_cnt, (cond, rec["set_ex_cnt"], set_cnt)
        assert rec["published_set_ex_cnt"] == pub_cnt, (cond, rec["published_set_ex_cnt"], pub_cnt)
        assert rec["strict_ex_cnt"] == strict_cnt, (cond, rec["strict_ex_cnt"], strict_cnt)
        assert rec["runtime_errors_total"] == sum(err_cnt.values()), cond
        assert rec["runtime_error_breakdown"] == dict(sorted(err_cnt.items())), cond
        for js in ("0", "1", "2+"):
            sub = [x for x in items if x["join_stratum"] == js]
            if not sub:
                assert js not in rec["by_join_stratum"]
            else:
                jrec = rec["by_join_stratum"][js]
                assert jrec["n"] == len(sub)
                assert jrec["set_ex_cnt"] == sum(x["replay_60s_set_ex"] for x in sub)
                assert jrec["strict_ex_cnt"] == sum(x["strict_ex"] for x in sub)

    dg8 = summary["dg8_clean_uniform_1320_errors_60s"]
    assert dg8["total_slots"] == 1320
    assert dg8["runtime_errors_total"] == 17
    assert summary["conditions"]["DG8_CLEAN_R1/r0"]["published_set_ex_cnt"] == 205
    assert summary["conditions"]["DG8_CLEAN_R2/r0"]["published_set_ex_cnt"] == 207
    assert summary["conditions"]["DG8_CLEAN_R3/r0"]["published_set_ex_cnt"] == 208
    # Verify reconciled timeout rows & Stage 8 BIRD TightBPE / noVal / Remap (2,030 slots)
    assert summary["conditions"]["SOL_C0/r0"]["set_ex_cnt"] == 319
    assert summary["conditions"]["PUBCAP_SILHOUETTE_R1/r0"]["set_ex_cnt"] == 189
    assert summary["conditions"]["PUBCAP_UNPRUNED_R1/r0"]["set_ex_cnt"] == 192
    assert summary["conditions"]["PUBCAP_OMEGA1_EQDELTA_R3/r0"]["set_ex_cnt"] == 205
    assert summary["conditions"]["REMAP_S2_DG8_R1/r0"]["set_ex_cnt"] == 6
    remap_conds = [c for c in by_cond if c.startswith(("REMAP_", "SOL_REMAP_"))]
    remap_slots = sum(len(by_cond[c]) for c in remap_conds)
    assert len(remap_conds) == 19 and remap_slots == 2030, (len(remap_conds), remap_slots)
    bird_outcomes = sum(1 for (bm, _, _) in outcomes_by_bcq if bm == "bird")
    bird_receipts = [r for r in receipts if r.get("benchmark") == "bird"]
    assert bird_outcomes == len(bird_receipts) == 51493, (bird_outcomes, len(bird_receipts))
    assert len(by_cond) == 153, len(by_cond)
    empty_bird = sum(1 for r in bird_receipts if r.get("deliverable_status") == "empty_submission")
    denied_bird = sum(1 for r in bird_receipts if str(r.get("deliverable_status", "")).startswith("authorizer_denied:"))
    assert empty_bird == 68, empty_bird
    assert denied_bird == 73, denied_bird
    for rep in (1, 2, 3):
        assert f"TIGHTBPE_SILHOUETTE_R{rep}/r0" in by_cond
        assert f"TIGHTBPE_IND8_PROF_R{rep}/r0" in by_cond
        assert f"TIGHTBPE_STATIC_SYNTH_R{rep}/r0" in by_cond
        assert f"DG8_NOVAL_POST_R{rep}/r0" in by_cond
        assert f"TIGHTBPE_VALFIRST_SILH_R{rep}/r0" in by_cond
        assert f"TIGHTBPE_VALFIRST_IND8_PROF_R{rep}/r0" in by_cond
        assert f"FETCH1_PUBCAP_IND8_PROF_R{rep}/r0" in by_cond

    # Dual-audit 5-column reconciliation check against stats/executor_transition_summary_v10.json
    et_v10 = json.loads((root / "stats/executor_transition_summary_v10.json").read_text(encoding="utf-8"))
    rec_tbl = et_v10["reconciliation_5col_table"]
    for label, subset in (
        ("BIRD", bird_receipts),
        ("EHRSQL-600", [r for r in receipts if r.get("benchmark") == "ehrsql_600"]),
        ("EHRSQL-874", [r for r in receipts if r.get("benchmark") == "ehrsql_874"]),
    ):
        exp = rec_tbl[label]
        assert len(subset) == exp["n"], (label, len(subset), exp["n"])
        assert sum(int(r["historical_official_set_ex"]) for r in subset) == exp["historical_official_set_ex"], label
        assert sum(int(r["raw_official_set_ex"]) for r in subset) == exp["raw_official_set_ex"], label
        assert sum(int(r["protocol_audited_official_ex"]) for r in subset) == exp["protocol_audited_official_ex"], label
        assert sum(int(r["raw_deliverable_set_ex"]) for r in subset) == exp["raw_deliverable_set_ex"], label
        assert sum(int(r["protocol_audited_deliverable_ex"]) for r in subset) == exp["protocol_audited_deliverable_ex"], label
        for r in subset:
            inv_mult = 0 if r.get("status") == "invalid" else 1
            assert int(r["protocol_audited_official_ex"]) == int(r["raw_official_set_ex"]) * inv_mult
            assert int(r["protocol_audited_deliverable_ex"]) == int(r["raw_deliverable_set_ex"]) * inv_mult
            if r.get("status") == "invalid":
                assert isinstance(r.get("protocol_audit_reason"), str) and r["protocol_audit_reason"].strip()

    return {
        "validated_conditions": len(by_cond),
        "validated_receipts": sum(len(v) for v in by_cond.values()),
        "bird_and_remap_slots_verified": len(bird_receipts),
        "empty_submission_bird": empty_bird,
        "authorizer_denied_bird": denied_bird,
        "remap_slots_verified": remap_slots,
        "timeout_outcome_conflicts": conflicts,
    }


def _qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def build_remap_sqlite_from_manifest(
    source_bird_dir: Path,
    remap_manifest: dict[str, Any],
    out_dir: Path,
    seed_key: str = "bird_remap_130",
) -> dict[str, Path]:
    """Deterministically reconstruct remapped BIRD SQLite DBs from standard BIRD DBs + manifest bijections."""
    import shutil

    out_dir.mkdir(parents=True, exist_ok=True)
    bijections = remap_manifest[seed_key]["bijections"]
    built_paths: dict[str, Path] = {}
    for src_path in sorted(source_bird_dir.glob("*.sqlite")):
        if src_path.name.startswith("._"):
            continue
        db_id = src_path.stem
        if db_id not in bijections:
            built_paths[db_id] = src_path
            continue
        dst_path = out_dir / f"{db_id}.sqlite"
        shutil.copy2(src_path, dst_path)
        con = sqlite3.connect(str(dst_path))
        try:
            for tc, col_map in bijections[db_id].items():
                tbl, col = tc.split(".", 1)
                qt, qc = _qident(tbl), _qident(col)
                case_parts = " ".join("WHEN ? THEN ?" for _ in col_map)
                params = [x for pair in col_map.items() for x in pair]
                sql_upd = f"UPDATE {qt} SET {qc} = CASE {qc} {case_parts} ELSE {qc} END WHERE {qc} IS NOT NULL"
                con.execute(sql_upd, params)
            con.commit()
            if con.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise RuntimeError(f"Remap PRAGMA quick_check failed for {seed_key}:{db_id}")
            for tc, col_map in bijections[db_id].items():
                tbl, col = tc.split(".", 1)
                qt, qc = _qident(tbl), _qident(col)
                got = {r[0] for r in con.execute(f"SELECT DISTINCT {qc} FROM {qt} WHERE {qc} IS NOT NULL")}
                if got != set(col_map.values()):
                    raise RuntimeError(f"Remap bijection verification failed for {seed_key}:{db_id}.{tc}")
        finally:
            con.close()
        built_paths[db_id] = dst_path
    return built_paths


def _execute_deliverable(db_path: Path, sql: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> tuple[list[tuple[Any, ...]] | None, str | None, str]:
    """Execute SQL through silhouette.sandbox.LocalExecutor (allow_cte=True)."""
    if not isinstance(sql, str) or not sql.strip():
        return None, "empty_submission", "empty_submission"
    ex = LocalExecutor(db_path)
    try:
        rows = ex.execute(sql, timeout_s=timeout_s, allow_cte=True)
        return rows, None, "ok"
    except Exception as exc:
        cat = error_category(exc)
        return None, cat, cat


def run_self_test() -> dict[str, Any]:
    """Run synthetic Q218 (multiset duplicate rows) and Q199 (ORDER BY) fixtures + receipt validation."""
    con = sqlite3.connect(":memory:")
    con.executescript("""
        CREATE TABLE district(district_id INTEGER PRIMARY KEY, A2 TEXT, A4 INTEGER, A12 REAL, A13 REAL);
        CREATE TABLE account(account_id INTEGER PRIMARY KEY, district_id INTEGER);
        CREATE TABLE loan(loan_id INTEGER PRIMARY KEY, account_id INTEGER, status TEXT);
        CREATE TABLE client(client_id INTEGER PRIMARY KEY, district_id INTEGER);
        CREATE TABLE disp(disp_id INTEGER PRIMARY KEY, client_id INTEGER, account_id INTEGER);
        CREATE TABLE card(card_id INTEGER PRIMARY KEY, disp_id INTEGER, type TEXT);

        INSERT INTO district VALUES
            (1, 'Brno', 10000, 2.0, 3.0),
            (2, 'Prague', 50000, 1.0, 1.5),
            (3, 'Ostrava', 80000, 4.0, 3.6);
        INSERT INTO account VALUES (10, 2), (20, 1);
        INSERT INTO loan VALUES (100, 10, 'D'), (200, 20, 'D');
        INSERT INTO client VALUES (1, 1), (2, 1), (3, 2);
        INSERT INTO disp VALUES (1, 1, 20), (2, 2, 20), (3, 3, 10);
        INSERT INTO card VALUES (1, 1, 'classic'), (2, 2, 'classic'), (3, 3, 'classic');
    """)

    # 1. Q218 duplicate-row multiset fixture (Gold SQL has NO top-level ORDER BY)
    q218_gold = (
        "SELECT c.type, d.A2 AS district_name, d.A4 AS inhabitants "
        "FROM card c "
        "JOIN disp p ON c.disp_id = p.disp_id "
        "JOIN client cl ON p.client_id = cl.client_id "
        "JOIN district d ON cl.district_id = d.district_id "
        "WHERE d.A4 = (SELECT MIN(A4) FROM district)"
    )
    q218_distinct = (
        "SELECT DISTINCT c.type, d.A2 AS district_name, d.A4 AS inhabitants "
        "FROM card c "
        "JOIN disp p ON c.disp_id = p.disp_id "
        "JOIN client cl ON p.client_id = cl.client_id "
        "JOIN district d ON cl.district_id = d.district_id "
        "WHERE d.A4 = (SELECT MIN(A4) FROM district)"
    )
    res_218_dup = score_sql_on_connection(con, q218_distinct, q218_gold)
    assert res_218_dup["strict_mode"] == "multiset"
    assert res_218_dup["official_set_ex"] == 1
    assert res_218_dup["strict_ex"] == 0
    res_218_exact = score_sql_on_connection(con, q218_gold, q218_gold)
    assert res_218_exact["official_set_ex"] == 1 and res_218_exact["strict_ex"] == 1

    # 2. Q199 top-level ORDER BY fixture (ordered comparator)
    q199_gold = (
        "SELECT d.A2 AS district, ROUND((d.A13 - d.A12) * 100.0 / d.A12, 2) AS pct_change "
        "FROM district d "
        "WHERE EXISTS (SELECT 1 FROM account a JOIN loan l ON l.account_id = a.account_id "
        "WHERE a.district_id = d.district_id AND l.status = 'D') "
        "ORDER BY district ASC"
    )
    q199_unordered = (
        "SELECT d.A2 AS district, ROUND((d.A13 - d.A12) * 100.0 / d.A12, 2) AS pct_change "
        "FROM district d "
        "WHERE EXISTS (SELECT 1 FROM account a JOIN loan l ON l.account_id = a.account_id "
        "WHERE a.district_id = d.district_id AND l.status = 'D') "
        "ORDER BY district DESC"
    )
    res_199_ord = score_sql_on_connection(con, q199_unordered, q199_gold)
    assert res_199_ord["strict_mode"] == "ordered"
    assert res_199_ord["official_set_ex"] == 1
    assert res_199_ord["strict_ex"] == 0
    res_199_exact = score_sql_on_connection(con, q199_gold, q199_gold)
    assert res_199_exact["official_set_ex"] == 1 and res_199_exact["strict_ex"] == 1

    # 3. Subquery ORDER BY must NOT trigger top-level ordered mode
    assert not has_top_level_order_by("SELECT A2 FROM district WHERE district_id = (SELECT district_id FROM district ORDER BY A4 DESC LIMIT 1)")
    assert has_top_level_order_by("SELECT A2 FROM district ORDER BY A4")
    con.close()

    # 4. Live LocalExecutor execution on synthetic schema + materialized CTE + json_extract + json_each denial + Remap builder
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tdir = Path(tmp)
        src_dir = tdir / "src"
        src_dir.mkdir()
        db_p = src_dir / "schools.sqlite"
        cs = sqlite3.connect(str(db_p))
        cs.execute("CREATE TABLE schools(District TEXT, Status TEXT)")
        cs.execute("INSERT INTO schools VALUES ('Alameda', 'Open')")
        cs.commit()
        cs.close()

        ex = LocalExecutor(db_p)
        assert ex.execute("WITH q AS MATERIALIZED (SELECT District AS d FROM schools) SELECT q.d FROM q") == [("Alameda",)]
        assert ex.execute("SELECT json_extract('{\"a\":1}', '$.a')") == [(1,)]
        rows_je, err_je, _ = _execute_deliverable(db_p, "SELECT value FROM json_each('[1,2]')")
        assert rows_je is None and err_je in ("read_only_denied", "unknown_table"), (rows_je, err_je)

        fake_manifest = {
            "bird_remap_130": {
                "bijections": {
                    "schools": {"schools.Status": {"Open": "V_STATUS_001"}}
                }
            }
        }
        built = build_remap_sqlite_from_manifest(src_dir, fake_manifest, tdir / "remap_s1", "bird_remap_130")
        assert LocalExecutor(built["schools"]).execute("SELECT Status FROM schools") == [("V_STATUS_001",)]

    # 5. Validate dual_comparator_summary.json & all 47,533 BIRD + Remap receipts
    val = validate_summary_against_receipts()
    return {"q218_fixture": res_218_dup, "q199_fixture": res_199_ord, **val}


def _classify_bird_dataset(cond: str) -> str:
    if "REMAP_S2_" in cond:
        return "bird_remap_s2"
    if "REMAP_" in cond:
        return "bird_remap_s1"
    return "bird_std"


def replay_bird_dir(
    bird_dir: Path,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    conditions: set[str] | None = None,
    *,
    compare_unrestricted: bool = True,
) -> dict[str, Any]:
    """Replay ALL 51,493 BIRD & BIRD-Remap predictions via LocalExecutor(allow_cte=True)."""
    import tempfile

    jsonl_path = bird_dir / "mini_dev_sqlite.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Expected {jsonl_path}")
    gold_by_q = {
        r["question_id"]: r
        for r in map(json.loads, jsonl_path.read_text(encoding="utf-8").splitlines())
    }
    receipts = [
        r
        for r in iter_jsonl(ROOT / "stats/sql_and_audit_receipts.jsonl")
        if r.get("benchmark", "bird") == "bird"
    ]
    if conditions:
        receipts = [r for r in receipts if r["condition"] in conditions]

    needed_kinds = {_classify_bird_dataset(r["condition"]) for r in receipts}
    remap_manifest = (
        json.loads((ROOT / "manifests/question_strata_and_remap.json").read_text(encoding="utf-8"))
        if ("bird_remap_s1" in needed_kinds or "bird_remap_s2" in needed_kinds)
        else {}
    )
    source_db_dir = bird_dir / "databases" if (bird_dir / "databases").is_dir() else bird_dir
    gold_cache: dict[tuple[str, int], tuple[list[tuple[Any, ...]], bool]] = {}
    summary: dict[str, dict[str, Any]] = {}

    with tempfile.TemporaryDirectory() as tmp_remap:
        troot = Path(tmp_remap)
        s1_dir = (
            build_remap_sqlite_from_manifest(source_db_dir, remap_manifest, troot / "s1", "bird_remap_130")
            if "bird_remap_s1" in needed_kinds
            else {}
        )
        s2_dir = (
            build_remap_sqlite_from_manifest(source_db_dir, remap_manifest, troot / "s2", "bird_remap_seed2")
            if "bird_remap_s2" in needed_kinds
            else {}
        )

        for r in receipts:
            cond = r["condition"]
            dk = _classify_bird_dataset(cond)
            st = summary.setdefault(
                cond,
                {
                    "n": 0,
                    "deliverable_set_ex": 0,
                    "deliverable_strict_ex": 0,
                    "official_set_ex": 0,
                    "set_ex": 0,
                    "strict_ex": 0,
                    "errors": 0,
                    "empty_submission": 0,
                    "no_submission": 0,
                    "unreplayable": 0,
                    "authorizer_denied": 0,
                },
            )
            pred_sql = r.get("predicted_sql")
            has_sql = isinstance(pred_sql, str) and bool(pred_sql.strip())
            if not has_sql:
                if pred_sql is None and r.get("strict_ex") is None:
                    st["unreplayable"] += 1
                    continue
                st["n"] += 1
                st["empty_submission"] += 1
                st["no_submission"] += 1
                continue

            q = int(r["question_id"])
            db_id = r["db_id"]
            if dk == "bird_remap_s1":
                db_path = s1_dir[db_id]
            elif dk == "bird_remap_s2":
                db_path = s2_dir[db_id]
            else:
                db_path = source_db_dir / f"{db_id}.sqlite"
            if not db_path.exists():
                raise FileNotFoundError(f"Missing SQLite DB for {db_id} under {bird_dir}")

            gkey = (dk, q)
            if gkey not in gold_cache:
                if dk == "bird_remap_s1":
                    g_sql = remap_manifest["bird_remap_130"]["gold_sqls"][str(q)]["SQL_rewritten"]
                elif dk == "bird_remap_s2":
                    g_sql = remap_manifest["bird_remap_seed2"]["gold_sqls_75"][str(q)]["SQL_rewritten"]
                else:
                    g_sql = gold_by_q[q]["SQL"]
                g_rows, g_err, g_msg = execute_ro(db_path, g_sql, timeout_s=timeout_s)
                if g_err is not None or g_rows is None:
                    raise RuntimeError(f"Gold SQL failed for {dk} Q{q}: {g_err}: {g_msg}")
                gold_cache[gkey] = (g_rows, has_top_level_order_by(g_sql))
            g_rows, has_ord = gold_cache[gkey]

            d_rows, d_err, _d_cat = _execute_deliverable(db_path, pred_sql, timeout_s=timeout_s)
            d_set_ex, d_str_ex, _ = score_rows(d_rows, d_err, g_rows, has_ord)
            st["n"] += 1
            st["deliverable_set_ex"] += d_set_ex
            st["deliverable_strict_ex"] += d_str_ex
            st["errors"] += int(d_err is not None)
            if d_err == "read_only_denied":
                st["authorizer_denied"] += 1

            if compare_unrestricted:
                u_rows, u_err, _ = execute_ro(db_path, pred_sql, timeout_s=timeout_s)
                u_set_ex, u_str_ex, _ = score_rows(u_rows, u_err, g_rows, has_ord)
                st["official_set_ex"] += u_set_ex
                st["set_ex"] += u_set_ex
                st["strict_ex"] += u_str_ex
            else:
                st["official_set_ex"] += d_set_ex
                st["set_ex"] += d_set_ex
                st["strict_ex"] += d_str_ex

    for cond, st in sorted(summary.items()):
        n = st["n"]
        st["deliverable_set_ex_pct"] = round(100.0 * st["deliverable_set_ex"] / n, 2) if n else None
        st["deliverable_strict_ex_pct"] = round(100.0 * st["deliverable_strict_ex"] / n, 2) if n else None
        st["set_ex_pct"] = round(100.0 * st["official_set_ex"] / n, 2) if n else None
        st["strict_ex_pct"] = round(100.0 * st["strict_ex"] / n, 2) if n else None
        d_pct = f"{st['deliverable_set_ex_pct']:5.2f}%" if st["deliverable_set_ex_pct"] is not None else "  N/A "
        o_pct = f"{st['set_ex_pct']:5.2f}%" if st["set_ex_pct"] is not None else "  N/A "
        print(
            f"{cond:30s}: n={n:3d} | DelivSet={st['deliverable_set_ex']:3d} ({d_pct}) "
            f"| OffSet={st['official_set_ex']:3d} ({o_pct}) "
            f"| empty={st['empty_submission']} | denied={st['authorizer_denied']} | err={st['errors']}"
        )
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Standalone BIRD dual-comparator (Set vs Strict, LocalExecutor allow_cte=True) scorer.")
    parser.add_argument("--self-test", action="store_true", help="Run synthetic fixtures and receipt cross-checks (default when no args).")
    parser.add_argument("--bird-dir", type=Path, default=None, help="Directory containing mini_dev_sqlite.jsonl and BIRD SQLite DBs.")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="Per-query SQLite timeout in seconds (default 60s).")
    parser.add_argument("--conditions", nargs="*", default=None, help="Optional condition labels to replay.")
    parser.add_argument("--no-compare-unrestricted", action="store_true", help="Skip unrestricted read-only comparison pass.")
    args = parser.parse_args(argv)

    if args.bird_dir is not None:
        replay_bird_dir(
            args.bird_dir,
            timeout_s=args.timeout,
            conditions=set(args.conditions) if args.conditions else None,
            compare_unrestricted=not args.no_compare_unrestricted,
        )
    else:
        res = run_self_test()
        print(
            f"DUAL SCORER SELF-TEST PASSED ({res['validated_conditions']} dual conditions, "
            f"{res['bird_and_remap_slots_verified']} BIRD+Remap slots, "
            f"empty={res['empty_submission_bird']}, denied={res['authorizer_denied_bird']} verified)"
        )


if __name__ == "__main__":
    main()
