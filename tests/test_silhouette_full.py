"""Synthetic boundary tests plus integrity checks of the packaged benchmark evidence."""
from __future__ import annotations

import ast
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random
import sqlite3
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from certificates import verify_certificates as verifier
from silhouette import dg_sampler as dg, patient_dp, profile as p, schema_roles as roles
from silhouette.agent import canonical_profile_sha256, file_sha256, generate_sql
from silhouette.audit import TransmissionLedger
from silhouette.sandbox import K, PUBLIC_TABLES, LocalExecutor, PublicLookup, SchemaSandbox
from stats import score_bird_dual, score_ehrsql_official
from stats.reproduce_stats import exact_rademacher_p, holm, paired_trend, sign_test


def read(path):
    return json.loads(path.read_text())


def memory(ddl):
    c = sqlite3.connect(":memory:")
    c.executescript(ddl)
    return c


def test_canonical_storage_and_collation():
    values = [1, 1.0, "1", b"1", -0.0, 0.0, float("inf"), float("-inf"), float("nan"),
              "Owner", "OWNER", "Owner", None]
    def stats(seq):
        c = memory("CREATE TABLE t(v COLLATE NOCASE)")
        c.executemany("INSERT INTO t VALUES (?)", [(v,) for v in seq])
        return p.column_statistics_exact(c, "t", "v", "ANY", len(values), True)
    before = stats(values)
    assert before == stats(reversed(values))
    keys = before["values"]
    assert all(keys[k] == 1 for k in (("integer", 1), ("real", 1.0.hex()), ("text", "1"),
                                       ("blob", b"1"), ("text", "OWNER"), ("real", "inf"),
                                       ("real", "-inf"), ("real", (-0.0).hex()), ("real", 0.0.hex())))
    assert keys["text", "Owner"] == 2
    assert before["null_count"] == 2  # SQLite converts a bound IEEE NaN to NULL.
    assert p.canonical_val_key("real", float("nan")) == ("real", "nan")


@pytest.mark.parametrize("value", [None, "owner", "OWNER", 1, 1.0, -0.0, b"1", float("inf")])
def test_neighbor_histogram_sensitivity(value):
    c = memory("CREATE TABLE t(v COLLATE NOCASE)")
    c.executemany("INSERT INTO t VALUES (?)", [(None,), (1,), ("OWNER",)])
    for typ in ("TEXT", "NUMERIC"):
        pre = p.column_statistics_exact(c, "t", "v", typ, 3, True)
        c.execute("INSERT INTO t VALUES (?)", (value,))
        post = p.column_statistics_exact(c, "t", "v", typ, 4, True)
        for key in ("distribution", "values"):
            counts = pre[key].keys() | post[key].keys()
            assert sum((post[key][k]-pre[key][k])**2 for k in counts) == int(value is not None)
        for key in ("distinct_count", "null_count"):
            assert 0 <= post[key]-pre[key] <= 1
        c.execute("DELETE FROM t WHERE rowid=(SELECT max(rowid) FROM t)")


def test_schema_only_schedule():
    c = memory("""CREATE TABLE ref(id INTEGER PRIMARY KEY);
        CREATE TABLE t(id INTEGER REFERENCES ref, phone TEXT, notes TEXT, is_active INTEGER,
        qty INTEGER, amount REAL, created DATE, colour TEXT, code TEXT UNIQUE);
        INSERT INTO t(colour) VALUES ('violet');""")
    reads = []
    sql = []
    def forbid(action, first, second, db, origin):
        if action == sqlite3.SQLITE_READ:
            reads.append(first)
            if first != "sqlite_master":
                return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    c.set_authorizer(forbid)
    c.set_trace_callback(sql.append)
    schema = roles.extract_public_schema(c)
    n = len(sql)
    plan = p.plan_table_dg_v3(schema, "t", 2)
    assert len(sql) == n and set(reads) == {"sqlite_master"}
    assert [roles.classify_column(v) for v in schema.tables["t"]] == [
        "KEY_OR_FK", "IDENT_OR_CONTACT", "FREE_TEXT", "ENUM_INT", "NUMERIC_INT", "NUMERIC_REAL",
        "TEMPORAL", "CATEGORICAL_TEXT", "KEY_OR_FK"]
    assert {q["weight"] for q in plan["queries"]} == {.25, .5, 1, 1.5, 2, 4, 6}
    assert plan["n_histograms"] >= len(schema.tables["t"])
    body = ast.parse((ROOT/"silhouette/profile.py").read_text())
    fn = next(x for x in body.body if getattr(x, "name", "") == "plan_table_dg_v3")
    assert not any(isinstance(x, ast.Attribute) and x.attr in {"execute", "connect", "cursor"}
                   for x in ast.walk(fn))


@pytest.mark.parametrize("size", [0, 3])
def test_empty_and_all_null(size, monkeypatch):
    c = memory("CREATE TABLE t(x TEXT, y INT)")
    c.executemany("INSERT INTO t VALUES (NULL,NULL)", [()] * size)
    monkeypatch.setattr(dg, "sample_dgauss", lambda *args: 0)
    profile, ledger = p.publish(c, 1)
    assert profile["t"]["x"] == ["TEXT", float(bool(size)), 0, [], []]
    assert profile["t"]["y"][3:] == [[None, None, None], []]
    assert ledger["t"]["delta_frac"] == "1/1000000"


def test_gate_and_strict_threshold(monkeypatch):
    monkeypatch.setattr(dg, "sample_dgauss", lambda *args: 0)
    assert p.threshold_histogram_dg([("left", 8), ("right", 9)], Fraction(1), 8) == {"right": 9}
    c = memory("CREATE TABLE t(x TEXT)")
    c.executemany("INSERT INTO t VALUES (?)", [(str(i),) for i in range(51)])
    assert p.publish(c)[0]["t"]["x"][4] is None
    c.execute("DELETE FROM t WHERE x='50'")
    assert p.publish(c)[0]["t"]["x"][4] == []
    assert p.quantiles({}) == [None, None, None]
    assert p.bin_index(float("nan")) == p.bin_index(float("-inf")) == -1
    assert all(x in p.EDGES for x in (-2.0, 0.0, 2e-22, 5e295))
    assert p.shape("q"*81) == "A"*80 + "\u2026"


def test_publication_modes():
    c = memory("CREATE TABLE t(owner_id INTEGER, colour TEXT)")
    c.executemany("INSERT INTO t VALUES (1,'ochre')", [()] * 100)
    exact, ledger = p.publish(c, mode="Noiseless")
    assert exact["t"]["colour"][4] == ["ochre"] and ledger == {}
    assert p.no_values(exact)["t"]["colour"] == exact["t"]["colour"][:4] + [[]]
    assert p.no_values(exact)["t"]["owner_id"][4] is None
    public, plan = p.publish(c, mode="EQ")
    assert {q["rational_share"] for q in plan["t"]["queries"]} == {1}
    assert any(q["column"] == "owner_id" and q["stat"] == "values" for q in plan["t"]["queries"])
    assert all(v[4] in ([], None) for v in p.publish(c, mode="No-Values")[0]["t"].values())


@pytest.mark.parametrize("s2,d", [(Fraction(1, 4), Fraction(1, 5)), (Fraction(7, 3), Fraction(1, 2000000)),
                                   (Fraction(100), Fraction(1, 10**8))])
def test_tail_and_minimality(s2, d):
    tau, q = dg.exact_dgauss_tau(s2, d)
    lo, hi, plo, phi = verifier.tails(s2, tau)
    assert Fraction(q["tail_lower"]) <= lo <= hi <= Fraction(q["tail_upper"]) <= d
    assert d < Fraction(q["prev_tail_lower"]) <= plo <= phi <= Fraction(q["prev_tail_upper"])


def test_cks_sampler():
    assert isinstance(dg._DEFAULT_CSPRNG, random.SystemRandom)
    rng = random.Random(91)  # Test-only reproducible distribution check.
    counts = Counter(dg.sample_dgauss(Fraction(2), rng) for _ in range(16000))
    expected = dg.dgauss_pmf_table(Fraction(2), 12)
    assert sum(abs(counts[k]/16000 - mass) for k, mass in expected.items()) < .04
    with pytest.raises(TypeError):
        dg.sample_dgauss(2.0)
    assert all(0 <= dg.BufferedOSCSPRNG(1).randrange(97) < 97 for _ in range(10))


def test_rho_conversion():
    for eps in (1, 2, 4, 8):
        rho = dg.rational_rho_budget(eps, verifier.HALF)
        assert Fraction(dg.conservative_eps_from_rho(rho, verifier.HALF)) <= eps


def test_rowless_schema_projection():
    source = memory("""CREATE TABLE ref(a INT,b INT,PRIMARY KEY(b,a));
        CREATE TABLE t(x TEXT DEFAULT 'fixture-default' CHECK(length(x)>4), a INT,b INT,
                         FOREIGN KEY(b,a) REFERENCES ref(b,a));
        CREATE INDEX extra ON t(x);
        INSERT INTO t(x) VALUES ('fixture-row'); ANALYZE;""")
    box = SchemaSandbox(source)
    assert box.try_sql("SELECT x FROM t") == {"status": "success"}
    assert box.connection.execute("SELECT COUNT(*) FROM t").fetchone() == (0,)
    # Inspect a local backup; the validation role cannot read its catalog.
    dst = sqlite3.connect(":memory:")
    box.connection.backup(dst)
    ddl = str(dst.execute("SELECT name,sql FROM sqlite_master").fetchall())
    for term in ("fixture", "CHECK", "DEFAULT", "TRIGGER", "extra", "sqlite_stat"):
        assert term not in ddl
    assert len(dst.execute("PRAGMA foreign_key_list(t)").fetchall()) == 2
    assert box.try_sql("SELECT missing FROM t") == {"status":"error","category":"unknown_column"}
    assert box.try_sql("SELECT abs(-9223372036854775808) FROM (SELECT 1)") == {"status":"error","category":"execution_error"}
    assert box.try_sql("DELETE FROM t")["status"] == "error"


def test_public_lookup(tmp_path):
    path = tmp_path/"dictionary.sqlite"
    c = sqlite3.connect(str(path))
    for t in PUBLIC_TABLES:
        c.execute('CREATE TABLE "'+t+'" (x INTEGER)')
    c.executemany("INSERT INTO d_items VALUES (?)", [(i,) for i in range(53)])
    c.commit()
    with pytest.raises(ValueError, match="expected_sha256 is required"):
        PublicLookup(path)
    api = PublicLookup(path, allow_unpinned=True)
    r = api.lookup_reference("SELECT x FROM d_items ORDER BY x")
    assert len(r["rows"]) == K and r["truncated"] and r["rows"][-1] == [49]
    for sql in ("SELECT * FROM private_table","SELECT sql FROM sqlite_master","DELETE FROM d_items",
                "PRAGMA table_info(d_items)"):
        assert api.lookup_reference(sql)["status"] == "error"
    api.connection.close()
    c.execute("CREATE TABLE private_table(x)")
    with pytest.raises(ValueError):
        PublicLookup(path, allow_unpinned=True)


@pytest.mark.parametrize("enabled", [True, False])
def test_agent_boundary(tmp_path, enabled):
    path = tmp_path/"private.sqlite"
    c = sqlite3.connect(str(path))
    c.execute("CREATE TABLE t(x TEXT)")
    c.execute("INSERT INTO t VALUES ('local-fixture-marker')")
    c.commit()
    empty = SchemaSandbox(c)
    requests = []
    def model(req):
        requests.append(req)
        names = [t["function"]["name"] for t in req["tools"]]
        assert ("try_sql" in names) == enabled
        assert "local-fixture-marker" not in json.dumps(req)
        return {"name":"try_sql" if len(requests)==1 and enabled else "submit_sql",
                "arguments":{"sql":"SELECT x FROM t"}}
    final = generate_sql(
        model,
        "Return x",
        "t(x TEXT)",
        {},
        empty,
        enable_sandbox=enabled,
        expected_profile_sha256=canonical_profile_sha256({}),
    )
    assert LocalExecutor(path).execute(final) == [("local-fixture-marker",)]
    assert len(requests) == 1 + int(enabled)
    with pytest.raises(sqlite3.Error):
        LocalExecutor(path).execute("DELETE FROM t")


def test_materialized_cte_and_json_scalars(tmp_path):
    path = tmp_path / "schools.sqlite"
    c = sqlite3.connect(str(path))
    c.execute("CREATE TABLE schools(District TEXT)")
    c.execute("INSERT INTO schools VALUES ('Alameda')")
    c.execute("CREATE VIEW secret_view AS SELECT District FROM schools")
    c.commit()
    box = SchemaSandbox(c)
    cte_sql = "WITH q AS MATERIALIZED (SELECT District AS d FROM schools) SELECT q.d FROM q"
    assert box.try_sql(cte_sql) == {"status": "success"}
    le = LocalExecutor(path)
    assert le.execute(cte_sql) == [("Alameda",)]
    json_sql = """SELECT json_extract('{"a":1}', '$.a')"""
    assert box.try_sql(json_sql) == {"status": "success"}
    assert le.execute(json_sql) == [(1,)]
    assert box.try_sql("SELECT json_valid('{\"a\":1}'), json_type('{\"a\":1}', '$.a'), json_array_length('[1,2]'), json_quote('x')") == {"status": "success"}
    je_res = box.try_sql("SELECT value FROM json_each('[1,2]')")
    if sqlite3.sqlite_version_info >= (3, 38, 0):
        assert je_res == {"status": "error", "category": "read_only_denied"}
    else:
        assert je_res["status"] == "error" and je_res["category"] in ("read_only_denied", "unknown_table")
    with pytest.raises(sqlite3.Error):
        le.execute("SELECT value FROM json_each('[1,2]')")
    assert box.try_sql("SELECT District FROM secret_view")["status"] == "error"
    with pytest.raises(sqlite3.Error):
        le.execute("SELECT District FROM secret_view")
    c.close()


def test_agent_turn_cap_and_fallback():
    c = memory("CREATE TABLE t(x TEXT)")
    seen = []
    def model(r):
        seen.append(r)
        return {"name":"try_sql","arguments":{"sql":"SELECT x FROM t"}}
    empty_sha = canonical_profile_sha256({})
    assert generate_sql(model, "List x", "t(x)", {}, SchemaSandbox(c), expected_profile_sha256=empty_sha) == "SELECT x FROM t"
    assert len(seen)==6 and "last model call" in seen[-1]["messages"][-1]["content"]
    with pytest.raises(ValueError):
        generate_sql(model, "List x", "t(x)", {}, enable_sandbox=False, expected_profile_sha256=empty_sha)


def test_decomposed_transmission():
    rows = [("fixture-one","fixture-two"),("fixture-two","fixture-three")]
    l = TransmissionLedger({"t":rows}, ["fixture-one"])
    l.record({"description":"fixture-one / fixture-two"})
    for _ in range(2):
        l.record({"rows":[rows[1]]},copied=[("t",1,0)])
    assert l.summary() == {
        "copied_private_tuples": [1, 2],
        "outside_profile_serialized_tuples": [1, 1],
        "full_payload_coincidences": [1, 1],
        "out_of_profile_literals": [2, 5],
        "multi_column_matches": [1, 1],
        "serialized_tuple_matches": [1, 1],
    }
    with pytest.raises(ValueError):
        l.record({"rows":[rows[0]]},copied=[("t",1,0)])


def test_serialized_tuple_matches_distinction():
    rows = [("alpha_val", "beta_val"), ("gamma_val", "delta_val")]
    ledger = TransmissionLedger({"t": rows}, [])
    # Split across different lines and dict keys: multi_column_matches=1, serialized_tuple_matches=0
    ledger.record({"msg": "alpha_val\nother text\nbeta_val"})
    assert ledger.summary()["multi_column_matches"] == [1, 1]
    assert ledger.summary()["serialized_tuple_matches"] == [0, 0]
    # Contiguous JSON array for row 1 without copied provenance: both increment
    ledger.record({"rows": [["gamma_val", "delta_val"]]})
    assert ledger.summary()["multi_column_matches"] == [2, 2]
    assert ledger.summary()["serialized_tuple_matches"] == [1, 1]


def test_schema_role_upgrades_and_zero_benchmark_words():
    src = (ROOT / "silhouette" / "schema_roles.py").read_text(encoding="utf-8")
    for banned in (
        "multiverseid", "streetabr", "mailstreet", "mailstrabr", "mailzip",
        "printings", "variations", "booster", "charter", "magnet",
    ):
        assert banned not in src.lower()

    c = memory("""CREATE TABLE u(
        id INTEGER PRIMARY KEY,
        md5 TEXT, sha1 TEXT, sha256 TEXT, sha512 TEXT,
        DisplayName TEXT, first_name TEXT, last_name TEXT, player_name TEXT,
        AboutMe TEXT, bio TEXT, "Charter School (Y/N)" INTEGER, status TEXT
    )""")
    schema = roles.extract_public_schema(c)
    role_by_col = {cm.name: roles.classify_column(cm) for cm in schema.tables["u"]}
    for col in ("md5", "sha1", "sha256", "sha512", "DisplayName", "first_name", "last_name", "player_name"):
        assert role_by_col[col] == "IDENT_OR_CONTACT", (col, role_by_col[col])
    assert role_by_col["AboutMe"] == "FREE_TEXT"
    assert role_by_col["bio"] == "FREE_TEXT"
    assert role_by_col["Charter School (Y/N)"] == "ENUM_INT"
    assert role_by_col["status"] == "CATEGORICAL_TEXT"

    # column_policy override
    q_pol = roles.plan_queries_v3(schema.tables["u"], column_policy={"u.status": "IDENT_OR_CONTACT"}, table_name="u")
    assert not any(s == "values" and col == "status" for s, col, _r, _w in q_pol)


def test_order_of_magnitude_and_deconfounded_ablations(monkeypatch):
    assert p.order_of_magnitude(0) == 0
    assert p.order_of_magnitude(9) == "10^0"
    assert p.order_of_magnitude(10**15 - 1) == "10^14"
    assert p.order_of_magnitude(10**15) == "10^15"

    c = memory("""CREATE TABLE ref(id INTEGER PRIMARY KEY);
        CREATE TABLE t(id INTEGER PRIMARY KEY, fk_id INTEGER REFERENCES ref, DisplayName TEXT, colour TEXT, val REAL);
        INSERT INTO ref VALUES (1);
        INSERT INTO t VALUES (1, 1, 'Alice', 'ochre', 1.5);""")
    schema = roles.extract_public_schema(c)
    dg_plan = p.plan_table_dg_v3(schema, "t", 8)
    flat_plan = p.plan_table_dg_v3(schema, "t", 8, flat_weights=True)
    unpruned_plan = p.plan_table_dg_v3(schema, "t", 8, unpruned_roles=True)

    # DG-FlatW: identical J_T query set, flat weights=1, equal delta_mode, r_ints=1
    assert flat_plan["mechanism"] == "DG-FlatW"
    assert flat_plan["delta_mode"] == "equal"
    assert [(q["stat"], q["column"]) for q in flat_plan["queries"]] == [(q["stat"], q["column"]) for q in dg_plan["queries"]]
    assert {q["rational_share"] for q in flat_plan["queries"]} == {1}
    assert {q["weight"] for q in flat_plan["queries"]} == {1.0}

    # DG-Unpruned: adds values with weight 1.0 to non-unique discrete/integer columns only (never REAL)
    assert unpruned_plan["mechanism"] == "DG-Unpruned"
    assert unpruned_plan["delta_mode"] == "proportional"
    unpruned_vals = {q["column"]: q["weight"] for q in unpruned_plan["queries"] if q["stat"] == "values"}
    assert "id" not in unpruned_vals  # single PK omitted
    assert "val" not in unpruned_vals  # REAL columns never receive values histograms
    assert unpruned_vals["fk_id"] == 1.0
    assert unpruned_vals["DisplayName"] == 1.0
    assert unpruned_vals["colour"] == 6.0

    # DG-Omega1: keeps DG-v3 pruned query set, sets w_f=1, keeps proportional delta and KKT r_ints ~ kappa^(2/3)
    omega1_plan = p.plan_table_dg_v3(schema, "t", 8, omega1_kkt=True)
    assert omega1_plan["mechanism"] == "DG-Omega1"
    assert omega1_plan["delta_mode"] == "proportional"
    assert [(q["stat"], q["column"]) for q in omega1_plan["queries"]] == [(q["stat"], q["column"]) for q in dg_plan["queries"]]
    assert {q["weight"] for q in omega1_plan["queries"]} == {1.0}
    hist_deltas = {q["delta_frac"] for q in omega1_plan["queries"] if q["stat"] in p.HISTOGRAMS}
    assert len(hist_deltas) == 1
    assert len({q["rational_share"] for q in omega1_plan["queries"]}) > 1

    monkeypatch.setattr(dg, "sample_dgauss", lambda *args: 0)
    _, ledgers_fw = p.publish(c, 8, mode="DG-FlatW")
    _, ledgers_up = p.publish(c, 8, mode="DG-Unpruned")
    _, ledgers_o1 = p.publish(c, 8, mode="DG-Omega1")
    assert ledgers_fw["t"]["mechanism"] == "DG-FlatW"
    assert ledgers_up["t"]["mechanism"] == "DG-Unpruned"
    assert ledgers_o1["t"]["mechanism"] == "DG-Omega1"


def test_entity_dp_v2_direct_star_and_d1_financial_counterexample(monkeypatch):
    from silhouette import entity_dp_publisher as edp
    assert edp.VERSION_ENTITY_DP == "c1-entity-dp-v2-direct-star"

    def make_financial_db():
        c = memory("""
            CREATE TABLE district(district_id INTEGER PRIMARY KEY, A2 TEXT, A3 TEXT);
            CREATE TABLE account(account_id INTEGER PRIMARY KEY, district_id INTEGER, frequency TEXT, date TEXT);
            CREATE TABLE client(client_id INTEGER PRIMARY KEY, gender TEXT, birth_date TEXT, district_id INTEGER);
            CREATE TABLE disp(disp_id INTEGER PRIMARY KEY, client_id INTEGER, account_id INTEGER, type TEXT);
            CREATE TABLE card(card_id INTEGER PRIMARY KEY, disp_id INTEGER, type TEXT, issued TEXT);
            CREATE TABLE loan(loan_id INTEGER PRIMARY KEY, account_id INTEGER, date TEXT, amount INTEGER, status TEXT);
            CREATE TABLE "order"(order_id INTEGER PRIMARY KEY, account_id INTEGER, bank_to TEXT, amount REAL);
            CREATE TABLE trans(trans_id INTEGER PRIMARY KEY, account_id INTEGER, date TEXT, type TEXT, amount INTEGER);
        """)
        c.execute("INSERT INTO district VALUES (1, 'Prague', 'central')")
        c.executemany("INSERT INTO account VALUES (?, 1, 'POPLATEK MESICNE', '1995-01-01')", [(i,) for i in range(2, 202)])
        c.executemany("INSERT INTO client VALUES (?, 'F', '1970-01-01', 1)", [(i,) for i in range(1, 201)])
        c.executemany("INSERT INTO disp VALUES (?, ?, ?, 'OWNER')", [(i, i - 1, i) for i in range(2, 202)])
        c.execute("INSERT INTO card VALUES (1, 2, 'classic', '1996-01-01')")
        c.execute("INSERT INTO loan VALUES (2, 2, '1997-01-01', 10000, 'A')")
        c.execute('INSERT INTO "order" VALUES (2, 2, \'AB\', 500.0)')
        c.execute("INSERT INTO trans VALUES (2, 2, '1996-02-01', 'PRIJEM', 200)")
        c.commit()
        return c

    c_a = make_financial_db()
    c_b = make_financial_db()
    # Neighboring entity edit for account_id=1: add account 1 + 200 disp links to all 200 clients + loan/order/trans + arbitrary client/card/district edits
    c_b.execute("INSERT INTO account VALUES (1, 1, 'POPLATEK TYDNE', '1995-02-02')")
    c_b.executemany("INSERT INTO disp VALUES (?, ?, 1, 'DISPONENT')", [(1000 + i, i) for i in range(1, 201)])
    c_b.executemany("INSERT INTO loan VALUES (?, 1, '1997-02-02', ?, 'B')", [(100 + i, 1000 + i) for i in range(8)])
    c_b.executemany('INSERT INTO "order" VALUES (?, 1, ?, ?)', [(100 + i, f"B{i}", 10.0 + i) for i in range(8)])
    c_b.executemany("INSERT INTO trans VALUES (?, 1, '1997-03-03', ?, ?)", [(100 + i, f"T{i}", i) for i in range(8)])
    c_b.executemany("INSERT INTO client VALUES (?, 'M', '1980-01-01', 1)", [(500 + i,) for i in range(50)])
    c_b.executemany("INSERT INTO card VALUES (?, 2, 'gold', '1998-01-01')", [(500 + i,) for i in range(50)])
    c_b.executemany("INSERT INTO district VALUES (?, 'Brno', 'south')", [(500 + i,) for i in range(50)])
    c_b.commit()

    specs = edp.BIRD_ENTITY_SPECS["financial"]
    schema = roles.extract_public_schema(c_a)

    # Reject missing/extra tables and invalid entity_col
    with pytest.raises(KeyError):
        edp.validate_entity_specs(schema, {k: v for k, v in specs.items() if k != "district"})
    with pytest.raises(KeyError):
        edp.validate_entity_specs(schema, {**specs, "extra_tbl": {"mode": "schema_only"}})
    with pytest.raises(KeyError):
        edp.validate_entity_specs(schema, {**specs, "account": {"mode": "entity_direct", "entity_col": "nope"}})

    # Assert 0 SQL reads on schema_only tables (client, card, district) during publish_entity_db
    for mode in ("Entity-DG-v3", "Entity-IND8-Prompt"):
        reads_a, reads_b = [], []
        def make_auth(log):
            def auth(action, first, second, db, origin):
                if action == sqlite3.SQLITE_READ:
                    log.append(first)
                    if first in ("client", "card", "district"):
                        return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK
            return auth
        monkeypatch.setattr(dg, "sample_dgauss", lambda *args: 0)
        c_a.set_authorizer(make_auth(reads_a))
        prof_a, meta_a = edp.publish_entity_db(c_a, specs, db_id="financial", epsilon=8, mode=mode)
        c_a.set_authorizer(lambda *_args: sqlite3.SQLITE_OK)
        c_b.set_authorizer(make_auth(reads_b))
        prof_b, meta_b = edp.publish_entity_db(c_b, specs, db_id="financial", epsilon=8, mode=mode)
        c_b.set_authorizer(lambda *_args: sqlite3.SQLITE_OK)

        assert not (set(reads_a) & {"client", "card", "district"})
        assert not (set(reads_b) & {"client", "card", "district"})
        assert meta_a["version"] == meta_b["version"] == edp.VERSION_ENTITY_DP
        assert meta_a["table_sensitivity_caps"] == {
            "account": 5, "disp": 5, "loan": 5, "order": 5, "trans": 5,
            "client": 0, "card": 0, "district": 0,
        }
        for t in ("client", "card", "district"):
            assert prof_a[t] == prof_b[t]
            assert all(row[1:] == [None, None, None, None] for row in prof_a[t].values())
            assert meta_a["tables"][t] == {
                "mode": "schema_only", "rho": "0/1", "delta_threshold": "0/1", "sensitivity_cap": 0, "queries": []
            }
        for q in meta_a["parameters"]:
            verifier.check_parameter(q)

    # Exact direct-star entity sensitivity bounds across all 5 entity_direct tables
    for t in ("account", "disp", "loan", "order", "trans"):
        cols = schema.tables[t]
        need_v = {cm.name for cm in cols}
        n_a, st_a = edp.entity_table_statistics(c_a, t, cols, specs[t], need_v)
        n_b, st_b = edp.entity_table_statistics(c_b, t, cols, specs[t], need_v)
        assert 0 <= n_b - n_a <= 1
        for cm in cols:
            sa, sb = st_a[cm.name], st_b[cm.name]
            assert 0 <= sb["null_count"] - sa["null_count"] <= 1
            assert 0 <= sb["distinct_count"] - sa["distinct_count"] <= 5
            for hkey in ("values", "distribution"):
                diff = sb[hkey] - sa[hkey]
                assert not (sa[hkey] - sb[hkey])
                assert sum(v * v for v in diff.values()) <= 5


def test_dp_query_row_and_entity_engines(monkeypatch):
    from silhouette.dp_query import DPQueryEntityEngine, DPQueryRowDBWideEngine, DPQueryRowEngine
    from silhouette import entity_dp_publisher as edp

    c = memory("""
        CREATE TABLE account(account_id INTEGER PRIMARY KEY, district_id INTEGER, frequency TEXT);
        CREATE TABLE disp(disp_id INTEGER PRIMARY KEY, client_id INTEGER, account_id INTEGER, type TEXT);
        CREATE TABLE district(district_id INTEGER PRIMARY KEY, name TEXT);
        INSERT INTO account VALUES (1, 10, 'monthly'), (2, 10, 'monthly'), (3, 20, 'weekly');
        INSERT INTO disp VALUES (1, 1, 1, 'OWNER'), (2, 2, 1, 'OWNER'), (3, 3, 1, 'OWNER'),
                               (4, 4, 1, 'USER'), (5, 5, 1, 'USER'), (6, 6, 1, 'DISPONENT'),
                               (7, 7, 1, 'ADMIN'), (8, 8, 2, 'OWNER');
        INSERT INTO district VALUES (10, 'Prague'), (20, 'Brno');
    """)
    c.executemany("INSERT INTO account VALUES (?, 30, 'yearly')", [(10 + i,) for i in range(150)])
    c.executemany("INSERT INTO disp VALUES (?, ?, ?, 'OWNER')", [(100 + i, 100 + i, 10 + i) for i in range(150)])
    c.commit()
    sql_trace = []
    c.set_trace_callback(sql_trace.append)
    monkeypatch.setattr(dg, "sample_dgauss", lambda *args: 0)

    row_eng = DPQueryRowEngine(c, q_db=4, max_probes_per_q=2)
    assert row_eng.rho_table == dg.rational_rho_budget(8, verifier.HALF)
    assert row_eng.sigma2 == Fraction(1, 2) / (row_eng.rho_table / 8)
    lo, hi, plo, phi = verifier.tails(row_eng.sigma2, row_eng.tau)
    assert hi <= row_eng.delta_probe < plo

    # Public rejections consume 0 data access and 0 slots
    n_before = len(sql_trace)
    r_join = row_eng.query("SELECT COUNT(*) FROM account JOIN disp ON account.account_id = disp.account_id")
    r_raw = row_eng.query("SELECT * FROM account")
    r_sub = row_eng.query("SELECT COUNT(*) FROM account WHERE account_id IN (SELECT account_id FROM disp)")
    r_rand = row_eng.query("SELECT COUNT(*) FROM account WHERE random() > 0")
    r_const_ovf = row_eng.query("SELECT COUNT(*) FROM account WHERE abs(-9223372036854775808) > 0")
    r_bad_col = row_eng.query("SELECT COUNT(*) FROM account WHERE nope = 1")
    assert all(
        r["status"] == "rejected_public_check"
        for r in (r_join, r_raw, r_sub, r_rand, r_const_ovf, r_bad_col)
    )
    assert len(sql_trace) == n_before
    assert row_eng.table_probes_used["account"] == 0

    # Valid count + exact normalized-SQL intra-stream cache (0 extra slot, 0 SQL reads)
    r1 = row_eng.query("SELECT count(*) FROM account WHERE frequency = 'monthly'", question_id=1)
    assert r1["status"] == "ok" and r1["noisy_count"] == 2 and not r1["cached"]
    assert row_eng.table_probes_used["account"] == 1
    n_after_r1 = len(sql_trace)
    r1_cached = row_eng.query('select COUNT(*) from "account" where ("frequency"=\'monthly\')', question_id=2)
    assert r1_cached["cached"] is True and r1_cached["noisy_count"] == 2
    assert row_eng.table_probes_used["account"] == 1 and len(sql_trace) == n_after_r1

    # Group-by with WHERE: count=2 <= tau suppressed; count=150 > tau survives; per-question budget exhaustion
    r2_sup = row_eng.probe("account", column="frequency", filter_sql="district_id = 10", question_id=1)
    assert r2_sup["status"] == "ok" and r2_sup["mode"] == "group_count" and r2_sup["rows"] == []
    r_q_exhausted = row_eng.probe("account", question_id=1)
    assert r_q_exhausted["status"] == "budget_exhausted" and r_q_exhausted["category"] == "question_probe_budget_exhausted"
    r2_surv = row_eng.probe("account", column="frequency", filter_sql="district_id = 30", question_id=3)
    assert r2_surv["status"] == "ok" and r2_surv["rows"] == [{"value": "yearly", "noisy_count": 150}]

    # SUM and AVG probes on Row engine + q_plan dilution
    row_q50 = DPQueryRowEngine(c, q_plan=50, max_probes_per_q=2, clip_bounds=(-100, 100))
    assert row_q50.budget_slots == 100
    r_sum = row_q50.query("SELECT SUM(district_id) FROM account WHERE frequency = 'monthly'")
    assert r_sum["status"] == "ok" and r_sum["mode"] == "sum" and r_sum["noisy_sum"] == 20.0
    r_avg = row_q50.query("SELECT AVG(district_id) FROM account WHERE frequency = 'monthly'")
    assert r_avg["status"] == "ok" and r_avg["mode"] == "avg" and r_avg["noisy_avg"] == 10.0

    # DB-Wide Row engine
    dbw = DPQueryRowDBWideEngine(c, q_db=2, max_probes_per_q=2)
    assert dbw.budget_slots == 4
    assert dbw.rho_probe == dbw.rho_db / 4
    d1 = dbw.probe("account", question_id=1)
    assert d1["status"] == "ok" and d1["noisy_count"] == 153 and d1["probes_used_db"] == 1
    d2 = dbw.probe("disp", question_id=2)
    assert d2["status"] == "ok" and d2["noisy_count"] == 158 and d2["probes_used_db"] == 2

    # Entity engine
    specs = {
        "account": {"mode": "entity_direct", "entity_col": "account_id"},
        "disp": {"mode": "entity_direct", "entity_col": "account_id"},
        "district": {"mode": "schema_only"},
    }
    ent_eng = DPQueryEntityEngine(c, specs, q_db=4, max_probes_per_q=2)
    assert ent_eng.sigma2_count == Fraction(1) / (2 * ent_eng.rho_probe)
    assert ent_eng.sigma2_group == Fraction(5) / (2 * ent_eng.rho_probe)
    elo, ehi, eplo, ephi = verifier.tails(ent_eng.sigma2_group, ent_eng.tau_group)
    assert ehi <= ent_eng.delta_probe / 5 < eplo

    n_ent = len(sql_trace)
    r_so1 = ent_eng.query("SELECT COUNT(*) FROM district")
    r_so2 = ent_eng.probe("district")
    r_ent_req = ent_eng.query("SELECT COUNT(*) FROM disp")
    assert all(r["status"] == "rejected_public_check" for r in (r_so1, r_so2, r_ent_req))
    assert r_so1["category"] == "schema_only_table"
    assert r_so2["category"] == "schema_only_table"
    assert r_ent_req["category"] == "entity_count_required"
    assert len(sql_trace) == n_ent

    e1 = ent_eng.probe("disp", question_id=1)
    assert e1["status"] == "ok" and e1["noisy_count"] == 152 and e1["sensitivity2"] == 1
    e1_dup = ent_eng.query('SELECT COUNT(DISTINCT "account_id") FROM "disp"', question_id=2)
    assert e1_dup["cached"] is True and ent_eng.probes_used == 1

    # 5-capped per-entity GROUP BY: OWNER has 152 distinct entities (> tau_group), while rare types <= tau_group are suppressed
    e2 = ent_eng.probe("disp", column="type", question_id=1)
    assert e2["status"] == "ok" and e2["sensitivity2"] == 5 and e2["tau"] == ent_eng.tau_group
    assert e2["rows"] == [{"value": "OWNER", "noisy_count": 152}]
    # Budget exhaustion on entity engine
    ent_small = DPQueryEntityEngine(c, specs, q_db=1, max_probes_per_q=1)
    assert ent_small.probe("disp")["status"] == "ok"
    assert ent_small.probe("account")["status"] == "budget_exhausted"



def test_patient_cap_and_neighbor():
    c = memory("CREATE TABLE events(subject_id INTEGER, token TEXT)")
    c.executemany("INSERT INTO events VALUES (1,?)", [(str(i),) for i in reversed(range(8))] + [("1",),(None,)])
    c.execute("INSERT INTO events VALUES(NULL,'unlinked')")
    a = patient_dp.patient_statistics(c,"events","token","TEXT")
    assert a["row_count"] == a["null_count"] == 1 and a["distinct_count"] == 5
    assert set(a["values"]) == {("text",str(i)) for i in range(5)}
    c.executemany("INSERT INTO events VALUES (2,?)", [(str(i),) for i in range(3,10)])
    b = patient_dp.patient_statistics(c,"events","token","TEXT")
    for name in ("values","distribution"):
        diff=b[name]-a[name]
        assert sum(v*v for v in diff.values())<=5 and not (a[name]-b[name])
    assert b["row_count"] == 2 and 0 <= b["distinct_count"]-a["distinct_count"] <= 5


def test_patient_composition():
    c=memory((ROOT/"manifests/mimic_iv_public_ddl.sql").read_text())
    schema=roles.extract_public_schema(c)
    for kind in ("per_table","db_uniform","db_global"):
        plan=patient_dp.plan_patient(schema,8,kind)
        n=13 if kind=="per_table" else 1
        assert len(plan["queries"])==346
        assert plan["rho"]==n*dg.rational_rho_budget(8,verifier.HALF)
        assert plan["delta"]==n*2*verifier.HALF


def test_offline_certificates():
    out=subprocess.check_output([sys.executable,"-B",str(ROOT/"certificates/verify_certificates.py")],text=True)
    assert out.strip()=="ALL CERTIFICATES VERIFIED (0 ERRORS)"
    data=read(ROOT/"certificates/ehrsql_dg_v3_certificates.json")
    q=next(q for q in data["parameters"] if "tau" in q)
    for k,v in (("tau",q["tau"]+1),("tail_upper","0/1"),("sigma2","1/2")):
        with pytest.raises(ValueError):
            verifier.check_parameter(dict(q,**{k:v}))


def test_frozen_release_hashes():
    row_releases = list((ROOT/"profiles/bird_minidev_dg_v3").rglob("*.dpmeta.json"))
    ent_dg_releases = list((ROOT/"profiles/bird_entity_dp_eps8").rglob("*.dpmeta.json"))
    ent_ind_releases = list((ROOT/"profiles/entity_ind8_prompt").rglob("*.dpmeta.json"))
    assert (len(row_releases), len(ent_dg_releases), len(ent_ind_releases)) == (66, 12, 12)
    for pth in row_releases + ent_dg_releases + ent_ind_releases:
        meta = read(pth)
        path = pth.with_name(meta["db_id"]+".json")
        assert meta["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert set(meta["tables"]) == set(read(path))
        if pth in row_releases:
            txt = path.read_text(encoding="utf-8")
            assert ("user" + "10525") not in txt


def test_paired_inference():
    assert sign_test([0, 0]) == 1 and sign_test([1] * 4) == 0.125
    assert holm([0.05, 0.001, 0.03]) == [0.06, 0.003, 0.06]
    assert paired_trend([[0] * 4, [1] * 4], (1, 2, 4, 8)) == (0, 1)
    z, pvalue = paired_trend([[0, 0, 1, 1]] * 20, (1, 2, 4, 8))
    assert z > 0 and pvalue < 0.001

    # exact_rademacher_p reduces bit-for-bit to exact McNemar for 1-vs-1
    rng = random.Random(20261121)
    for _ in range(15):
        a = [rng.randint(0, 1) for _ in range(40)]
        b = [rng.randint(0, 1) for _ in range(40)]
        assert exact_rademacher_p(a, b, 1, 1) == sign_test([x - y for x, y in zip(a, b)])

    # exact_rademacher_p for 2-vs-2 and 3-vs-3: symmetric, exact null behavior, rejects unbalanced R_A != R_B
    a3 = [1.0, 2 / 3, 1 / 3, 0.0, 1.0, 2 / 3]
    b3 = [2 / 3, 1.0, 0.0, 1 / 3, 1 / 3, 0.0]
    p3 = exact_rademacher_p(a3, b3, 3, 3)
    assert p3 == exact_rademacher_p(b3, a3, 3, 3)
    assert 0.0 < p3 <= 1.0
    assert exact_rademacher_p([1.0, 0.0], [0.0, 1.0], 2, 2) == 1.0
    with pytest.raises(ValueError):
        exact_rademacher_p([1.0, 0.0], [0.0, 1.0], 3, 1)

    # Zero-bias check under symmetric balanced R=3 null vs null draws
    null_exceed = 0
    for trial in range(200):
        trng = random.Random(1000 + trial)
        x = [sum(trng.randint(0, 1) for _ in range(3)) / 3.0 for _ in range(30)]
        y = [sum(trng.randint(0, 1) for _ in range(3)) / 3.0 for _ in range(30)]
        if exact_rademacher_p(x, y, 3, 3) <= 0.05:
            null_exceed += 1
    assert null_exceed <= 20  # Nominal 5% (expected ~10/200, zero inflation)


def test_dual_scorer_self_test():
    res = score_bird_dual.run_self_test()
    assert res["q218_fixture"]["official_set_ex"] == 1
    assert res["q218_fixture"]["strict_ex"] == 0
    assert res["q218_fixture"]["strict_mode"] == "multiset"
    assert res["q199_fixture"]["official_set_ex"] == 1
    assert res["q199_fixture"]["strict_ex"] == 0
    assert res["q199_fixture"]["strict_mode"] == "ordered"
    assert res["validated_conditions"] == 162
    assert res["remap_slots_verified"] == 2030
    assert res["timeout_outcome_conflicts"] == 0


def test_ehrsql_official_scorer_self_test():
    res = score_ehrsql_official.run_self_test()
    assert res["ehrsql_conditions_verified"] == 40
    assert res["ehrsql_slots_verified"] == 27562
    assert res["ehr874_noiseless_v3_hits"] == 429
    assert res["ehr_dg8p_v3_600_hits"] == 893
    assert res["ehr874_dg8p_v3_hits"] == 1368
    assert res["ehr874_v3_unseen235_hits"] == [115, 128, 129]
    assert res["gt100_distinct_questions"] == 8
    assert res["gt100_slots_verified"] == 132
    assert res["gt100_slots_verified"] > 0


def test_frozen_view_across_private_neighbors(tmp_path):
    captured, answers = [], []
    fixed_profile = {"t":{"x":["TEXT",0,"10^0",["AAAA"],None]}}
    for index, values in enumerate((["left"], ["left","right"])):
        path = tmp_path / str(index)
        conn = sqlite3.connect(str(path))
        conn.execute("CREATE TABLE t(x TEXT)")
        conn.executemany("INSERT INTO t VALUES (?)", [(v,) for v in values])
        conn.commit()
        box = SchemaSandbox(conn)
        events = []
        def model(req):
            events.append(json.dumps(req, sort_keys=True))
            name = "try_sql" if len(events) == 1 else "submit_sql"
            return {"name":name, "arguments":{"sql":"SELECT count(*) FROM t"}}
        final = generate_sql(
            model,
            "Count the tiles",
            "t(x)",
            fixed_profile,
            box,
            expected_profile_sha256=canonical_profile_sha256(fixed_profile),
        )
        captured.append(events)
        answers.append(LocalExecutor(path).execute(final))
    assert captured[0] == captured[1]
    assert answers == [[(1,)], [(2,)]]


def test_opt_in_lookup_and_local_error(tmp_path):
    path = tmp_path / "public.sqlite"
    c = sqlite3.connect(str(path))
    for t in PUBLIC_TABLES:
        c.execute('CREATE TABLE "' + t + '" (x TEXT)')
    c.execute("INSERT INTO d_items VALUES ('public-fixture')")
    c.commit()
    calls = []
    def model(req):
        calls.append(req)
        if len(calls) == 1:
            return {"name":"lookup_reference","arguments":{"sql":"SELECT x FROM d_items"}}
        assert "public-fixture" in req["messages"][-1]["content"]
        return {"name":"submit_sql","arguments":{"sql":"SELECT missing FROM items"}}
    sql = generate_sql(
        model,
        "List items",
        "items(x)",
        {},
        lookup=PublicLookup(path, expected_sha256=file_sha256(path)),
        enable_sandbox=False,
        expected_profile_sha256=canonical_profile_sha256({}),
    )
    with pytest.raises(sqlite3.OperationalError):
        LocalExecutor(path).execute(sql)
    assert len(calls) == 2


def test_metadata_agrees_with_frozen_ledger():
    group = read(ROOT / "certificates/bird_dg_v3_certificates.json")
    for path in (ROOT/"profiles/bird_minidev_dg_v3").rglob("*.dpmeta.json"):
        meta = read(path)
        tables = group["schedules"][str(meta["epsilon"])][meta["db_id"]]
        for name, t in meta["tables"].items():
            expected = tables[name]
            assert t["rho"] == expected["rho"] and t["delta_threshold"] == expected["delta_threshold"]
            assert [(s,c,meta["parameters"][i]) for s,c,i in t["queries"]] == [
                (s,c,group["parameters"][i]) for s,c,i in expected["queries"]]
    ent_group = read(ROOT / "certificates/bird_entity_dp_certificates.json")
    for path in (ROOT/"profiles/bird_entity_dp_eps8").rglob("*.dpmeta.json"):
        meta = read(path)
        expected = ent_group["schedules"][str(meta["epsilon"])][meta["db_id"]]
        assert meta["parameters"] == expected["parameters"] and meta["tables"] == expected["tables"]
    for path in (ROOT/"profiles/entity_ind8_prompt").rglob("*.dpmeta.json"):
        meta = read(path)
        expected = ent_group["ind8_prompt_schedules"][str(meta["epsilon"])][meta["db_id"]]
        assert meta["parameters"] == expected["parameters"] and meta["tables"] == expected["tables"]


def test_patient_scalar_and_histogram_identity():
    plan = patient_dp.allocate([("t","row_count",None,"TABLE",1),
                                ("t","null_count","x","CATEGORICAL_TEXT",1),
                                ("t","distinct_count","x","CATEGORICAL_TEXT",1.5),
                                ("t","values","x","CATEGORICAL_TEXT",6)],
                               Fraction(1,4), verifier.HALF)
    assert [q["sensitivity2"] for q in plan] == [1,1,25,5]
    assert sum(q["delta"] for q in plan) == verifier.HALF
    for q in plan:
        assert q["sigma2"]*2*q["rho"] == q["sensitivity2"]


def test_statistical_manifest_alignment():
    bird = read(ROOT/"manifests/bird_minidev_2026_manifest.json")
    ehr = read(ROOT/"manifests/ehrsql_2024_manifest.json")
    test = ehr["official_test"]
    confirmed = [q["ehrsql_id"] for q in ehr["confirmation"]]
    assert len(set(bird["dev"] + bird["evaluation"])) == 500
    assert (len(bird["dev"]),len(bird["evaluation"])) == (60,440)
    assert (len(test["dev"]),len(test["held_out"]),len(confirmed)) == (60,874,600)
    assert len(set(test["dev"]+test["held_out"]+confirmed)) == 1534
    known = set(bird["dev"]+bird["evaluation"]+confirmed+test["dev"]+test["held_out"])
    cond_sums = Counter()
    cond_counts = Counter()
    bm_cond_counts = Counter()
    allowed_outcome_keys = {
        "benchmark", "question_id", "db_id", "condition", "outcome",
        "official_set_ex", "strict_ex", "uncapped_ex", "pre_replay_ex", "status", "release_sha256",
        "deliverable_set_ex", "deliverable_strict_ex", "deliverable_uncapped_ex", "deliverable_status",
        "legacy_nocte_denied", "legacy_nocte_reason",
        "sandbox_telemetry_available", "telemetry_source_format",
        "sandbox_calls_turn_ordered", "sandbox_error_calls_turn_ordered",
        "slot_has_sandbox_error", "slot_error_then_rewrite", "slot_error_then_exec_ok",
        "slot_error_then_ex1", "slot_any_sql_revision", "slot_zero_error_revision",
    }
    slot_tel_keys = (
        "slot_has_sandbox_error", "slot_error_then_rewrite", "slot_error_then_exec_ok",
        "slot_error_then_ex1", "slot_any_sql_revision", "slot_zero_error_revision",
    )
    for line in (ROOT/"stats/per_question_outcomes.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        assert {"benchmark", "question_id", "db_id", "condition", "outcome", "deliverable_set_ex", "sandbox_telemetry_available"}.issubset(row)
        assert row["benchmark"] in {"bird", "ehrsql_600", "ehrsql_874"}
        if row["benchmark"] in {"ehrsql_600", "ehrsql_874"}:
            assert row["db_id"] == "mimic_iv"
        assert set(row) <= allowed_outcome_keys
        assert row["question_id"] in known and row["outcome"] in (0, 1) and row["deliverable_set_ex"] in (0, 1)
        if row["sandbox_telemetry_available"]:
            for k in slot_tel_keys:
                assert row[k] in (0, 1), (row["condition"], k, row[k])
        else:
            for k in slot_tel_keys:
                assert row[k] is None, (row["condition"], k, row[k])
        assert not row["condition"].startswith(("ENT8_", "SOL_ENT8_", "DG8_CLEAN_S4HYBRID_"))
        cond_counts[row["condition"]] += 1
        bm_cond_counts[(row["benchmark"], row["condition"])] += 1
        cond_sums[row["condition"]] += row["outcome"]
    assert sum(cond_counts.values()) == 83015
    assert len(cond_counts) == 195
    assert len(bm_cond_counts) == 202
    assert (cond_sums["DG8_CLEAN_R1/r0"], cond_sums["DG8_CLEAN_R2/r0"], cond_sums["DG8_CLEAN_R3/r0"]) == (205, 207, 208)
    assert sum(cond_sums[f"DAWA_CLEAN_BIRD_R{r}/r0"] for r in (1, 2, 3)) == 600
    assert sum(cond_sums[f"TIGHTBPE_COLFIRST_DAWA_CLEAN_BIRD_R{r}/r0"] for r in (1, 2, 3)) == 455
    assert sum(cond_sums[f"TIGHTBPE_VALFIRST_DAWA_CLEAN_BIRD_R{r}/r0"] for r in (1, 2, 3)) == 538
    assert cond_sums["DAWA_CLEAN_EHR600_R1/r0"] == 306
    assert cond_sums["EHR874_NOISELESS_NOLKP_V3/r0"] == 429
    assert cond_sums["EHR874_NOISELESS_LKP_V3/r0"] == 457
    assert cond_sums["EHR874_NOISELESS_NOLKP/r0"] == 377
    assert cond_sums["EHR_NOISELESS_LKP_V3/r0"] == 324
    assert cond_sums["EHR_NOISELESS_NOLKP_V3/r0"] == 269
    assert sum(cond_sums[f"EHR_DG8P_NOLKP_V3_R{r}/r0"] for r in (1, 2, 3)) == 811
    assert sum(cond_sums[f"EHR874_DG8P_NOLKP_V3_R{r}/r0"] for r in (1, 2, 3)) == 1240
    assert cond_sums["SOL_REMAP_C0/r0"] == 57
    assert cond_sums["SOL_C0/r0"] == 319
    assert cond_sums["PUBCAP_SILHOUETTE_R1/r0"] == 189
    assert cond_sums["EHR_DG8P_V3_R1/r0"] == 302
    assert cond_sums["TIGHTBPE_SILHOUETTE_R1/r0"] == 141
    assert cond_sums["DG8_NOVAL_POST_R3/r0"] == 188
    assert sum(cond_sums[f"TIGHTBPE_VALFIRST_SILH_R{r}/r0"] for r in (1, 2, 3)) == 517
    assert sum(cond_sums[f"TIGHTBPE_VALFIRST_IND8_PROF_R{r}/r0"] for r in (1, 2, 3)) == 575
    assert sum(cond_sums[f"FETCH1_PUBCAP_IND8_PROF_R{r}/r0"] for r in (1, 2, 3)) == 584


def test_strata_remap_manifest_and_receipts():
    strata = read(ROOT/"manifests/question_strata_and_remap.json")
    s1 = strata["bird_eval_440_strata"]["S1_130"]
    s0 = strata["bird_eval_440_strata"]["S0_310"]
    assert len(s1) == 130 and len(s0) == 310 and set(s1).isdisjoint(s0)
    assert (len(strata["gold_hidden_literal_106"]["hidden_106"]), len(strata["gold_hidden_literal_106"]["no_hidden_334"])) == (106, 334)
    assert (len(strata["parsed_base_tables"]["1_base_table"]), len(strata["parsed_base_tables"]["2_base_tables"]), len(strata["parsed_base_tables"]["3+_base_tables"])) == (50, 242, 148)
    assert len(strata["remap_pred_48"]) == 48
    assert len(strata["fixed_55_qids"]) == 55
    assert strata["ehrsql_template_clusters"]["ehr600_n_templates"] == 120
    assert strata["ehrsql_template_clusters"]["ehr874_n_templates"] == 134
    assert len(strata["ehrsql_template_clusters"]["ehr874_unseen_37_templates"]) == 37
    assert len(strata["ehrsql_template_clusters"]["ehr874_unseen_235_qids"]) == 235
    rm = strata["bird_remap_130"]
    assert len(rm["remap_fail_50"]) == 50 and len(rm["remap_pass_80"]) == 80
    assert len(rm["remapped_columns"]) == 19
    assert len(rm["gold_sqls"]) == 130
    assert len(strata["bird_remap_seed2"]["remap_col_slice_75"]) == 75
    assert len(strata["bird_remap_seed2"]["gold_sqls_75"]) == 75
    assert strata["bird_eval_440_join_strata"]["counts"] == {"0_joins": 99, "1_join": 183, "2plus_joins": 158}
    assert len(strata["bird_eval_440_join_strata"]["top_level_order_by_qids"]) == 147
    assert (len(strata["dp_query_subsets"]["q15_163"]), len(strata["dp_query_subsets"]["q5_55"])) == (163, 55)
    assert len(strata["entity_159"]["question_ids"]) == 159
    assert len(strata["sol_denominators"]["table7_428"]) == 428
    assert len(strata["sol_denominators"]["clean_425"]) == 425
    assert len(strata["ehrsql_confirm_literal_strata"]["patient_literal_136"]) == 136
    assert len(strata["ehrsql_confirm_literal_strata"]["non_patient_literal_464"]) == 464
    assert len(strata["bird_dev_exposure_strata"]["low_dev_le4_134"]) == 134
    assert len(strata["bird_dev_exposure_strata"]["mid_dev_5_6_127"]) == 127
    assert len(strata["bird_dev_exposure_strata"]["high_dev_ge7_179"]) == 179
    digests = read(ROOT/"manifests/release_digests.json")
    assert "ENT_V2_DG8_R1" in digests["conditions"] and "REMAP_S2_DG8_R1" in digests["conditions"]
    assert "SYN8_FK_DAG_R1" in digests["conditions"] and "IND8_PROFILE_PRUNED_TM_R1" in digests["conditions"]
    from silhouette.jsonl_io import open_jsonl

    with open_jsonl(ROOT / "stats/sql_and_audit_receipts.jsonl") as fh:
        receipts = [line for line in fh if line.strip()]
    assert len(receipts) == 83015
    parsed_r = [json.loads(line) for line in receipts]
    reconciled_audit_rows = [r for r in parsed_r if r.get("audit_condition")]
    assert len(reconciled_audit_rows) == 4840
    assert len({r["audit_condition"] for r in reconciled_audit_rows}) == 11
    outbound_sha_rows = [r for r in parsed_r if r.get("outbound_messages_sha256")]
    assert len(outbound_sha_rows) == 78089
    p7_aud = read(ROOT/"stats/p7_audit_summary.json")
    assert p7_aud["total_conditions"] == 115
    first = parsed_r[0]
    assert {"benchmark", "question_id", "db_id", "condition", "predicted_sql", "tool_calls", "total_tokens", "status", "official_set_ex"}.issubset(first)
    # 100% of LLM receipts have non-null predicted_sql and total_tokens
    for r in parsed_r:
        assert r["predicted_sql"] is not None
        assert r["total_tokens"] is not None


def test_clean_profiles_pii_absent():
    pii = {
        ("codebase_community", "comments", "UserDisplayName"),
        ("codebase_community", "postHistory", "UserDisplayName"),
        ("codebase_community", "posts", "OwnerDisplayName"),
        ("codebase_community", "posts", "LastEditorDisplayName"),
        ("codebase_community", "users", "DisplayName"),
        ("european_football_2", "Player", "player_name"),
        ("formula_1", "drivers", "forename"),
        ("formula_1", "drivers", "surname"),
        ("student_club", "member", "first_name"),
        ("student_club", "member", "last_name"),
        ("superhero", "superhero", "full_name"),
    }
    for pth in (ROOT/"profiles/bird_minidev_dg_v3").rglob("*.json"):
        if pth.name.endswith(".dpmeta.json"):
            continue
        prof = read(pth)
        for db, tbl, col in pii:
            if db == pth.stem:
                assert prof[tbl][col][4] is None


def test_dp_query_zero_error_side_channel_adversarial_rows(monkeypatch):
    from silhouette.dp_query import DPQueryEntityEngine, DPQueryRowDBWideEngine, DPQueryRowEngine

    INT64_MIN = -(1 << 63)
    def make_db(include_adversarial: bool):
        c = memory("""
            CREATE TABLE items(id INTEGER PRIMARY KEY, eid INTEGER, val INTEGER, tag TEXT);
            INSERT INTO items VALUES (1, 10, 5, 'ok'), (2, 11, 7, 'ok'), (3, 12, 0, 'ok');
        """)
        if include_adversarial:
            c.execute("INSERT INTO items VALUES (4, 13, ?, '%bad%')", (INT64_MIN,))
        c.commit()
        return c

    monkeypatch.setattr(dg, "sample_dgauss", lambda *args: 0)
    c_clean = make_db(False)
    c_adv = make_db(True)

    # Constant abs(INT64_MIN) fails public SchemaSandbox at 0 budget / 0 private reads
    eng_clean = DPQueryRowEngine(c_clean, q_plan=5, max_probes_per_q=2)
    r_pub = eng_clean.query(f"SELECT COUNT(*) FROM items WHERE abs({INT64_MIN}) > 0")
    assert r_pub["status"] == "rejected_public_check"
    assert eng_clean.table_probes_used["items"] == 0

    # Data-dependent abs(val) / printf(tag, 1) / division by zero NEVER leak errors or skip budget
    for cls in (DPQueryRowEngine, DPQueryRowDBWideEngine):
        e_clean = cls(c_clean, q_plan=5, max_probes_per_q=2)
        e_adv = cls(c_adv, q_plan=5, max_probes_per_q=2)
        for qsql in (
            "SELECT COUNT(*) FROM items WHERE abs(val) > 0",
            "SELECT COUNT(*) FROM items WHERE printf(tag, 1) IS NOT NULL",
            "SELECT COUNT(*) FROM items WHERE 10 / val > 0",
        ):
            rc = e_clean.query(qsql)
            ra = e_adv.query(qsql)
            assert rc["status"] == "ok" and ra["status"] == "ok"
            assert abs(ra["noisy_count"] - rc["noisy_count"]) <= 1

        # Verify budget incremented unconditionally
        used = e_adv.table_probes_used["items"] if isinstance(e_adv, DPQueryRowEngine) else e_adv.probes_used
        assert used == 3

    # Entity engine on adversarial rows too
    ent = DPQueryEntityEngine(c_adv, {"items": {"mode": "entity_direct", "entity_col": "eid"}}, q_plan=5)
    re_adv = ent.query("SELECT COUNT(DISTINCT eid) FROM items WHERE abs(val) > 0")
    assert re_adv["status"] == "ok" and re_adv["noisy_count"] == 3 and ent.probes_used == 1


def test_ehr_17_table_entity_specs_and_prompt_formatter(monkeypatch):
    from silhouette import entity_dp_publisher as edp

    c = memory((ROOT / "manifests/mimic_iv_public_ddl.sql").read_text())
    schema = roles.extract_public_schema(c)
    assert len(schema.tables) == 17
    assert len(edp.EHR_ENTITY_SPECS) == 17
    edp.validate_entity_specs(schema, edp.EHR_ENTITY_SPECS)

    reads = []
    def auth(action, first, second, db, origin):
        if action == sqlite3.SQLITE_READ:
            reads.append(first)
            if first in edp.EHR_PUBLIC_TABLES:
                return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    c.set_authorizer(auth)
    monkeypatch.setattr(dg, "sample_dgauss", lambda *args: 0)
    prof, meta = edp.publish_entity_db(c, edp.EHR_ENTITY_SPECS, db_id="mimic_iv", epsilon=8)
    c.set_authorizer(lambda *_args: sqlite3.SQLITE_OK)

    assert not (set(reads) & set(edp.EHR_PUBLIC_TABLES))
    for pub_tbl in edp.EHR_PUBLIC_TABLES:
        assert meta["tables"][pub_tbl]["mode"] == "schema_only"
        assert all(row[1:] == [None, None, None, None] for row in prof[pub_tbl].values())

    prompt_txt = edp.format_entity_profile_prompt(prof)
    assert "entity null-presence fraction" in prompt_txt
    assert "capped entity-support (C <= 5 keys/entity)" in prompt_txt

    # Verify generate_sql(..., is_entity_profile=True) emits the entity-profile semantics
    captured = []
    def model(req):
        captured.append(req)
        return {"name": "submit_sql", "arguments": {"sql": "SELECT COUNT(*) FROM patients"}}
    sql = generate_sql(
        model, "Count patients", "patients(subject_id INT)", prof,
        sandbox=SchemaSandbox(c), is_entity_profile=True,
        expected_profile_sha256=canonical_profile_sha256(prof),
    )
    assert sql == "SELECT COUNT(*) FROM patients"
    assert "entity null-presence fraction" in captured[0]["messages"][1]["content"]
    assert "capped entity-support (C <= 5 keys/entity)" in captured[0]["messages"][1]["content"]


def test_bounded_o_c_entity_streaming_equivalence():
    from silhouette import entity_dp_publisher as edp

    # Include bin_index values 2 and 10 (repr('10') < repr('2')) and unsorted mixed canonical keys
    rng = random.Random(20261121)
    c = memory("CREATE TABLE ev(eid INTEGER, num REAL, txt TEXT)")
    rows = []
    for eid in range(1, 12):
        vals = [float(x) for x in (1, 2, 10, 100, 1000, 0.0, -0.0, 1e-5, 500.0, 42.0)]
        rng.shuffle(vals)
        for idx, v in enumerate(vals):
            rows.append((eid, v if idx % 7 != 0 else None, f"tok_{rng.randint(1, 30)}" if idx % 5 != 0 else None))
    c.executemany("INSERT INTO ev VALUES (?, ?, ?)", rows)
    c.commit()

    # Check _insert_bounded_top_c invariant len <= C at every insertion
    top_c = []
    all_seen = set()
    for item in [10, 2, 5, 1, 9, 3, 2, 0, -1, 7, 4]:
        all_seen.add(item)
        edp._insert_bounded_top_c(top_c, item, cap=5)
        assert len(top_c) <= 5
    assert [x for _r, x in top_c] == sorted(all_seen, key=repr)[:5]

    # Exact bit-for-bit equivalence vs brute-force sorted(set)[:C] across full table
    schema = roles.extract_public_schema(c)
    cols = schema.tables["ev"]
    nrows, stats = edp.entity_table_statistics(c, "ev", cols, {"mode": "entity_direct", "entity_col": "eid"}, {"eid", "num", "txt"}, cap=5)

    for cm in cols:
        qt, qc = roles.quote("ev"), roles.quote(cm.name)
        by_ent_keys = {}
        by_ent_buckets = {}
        missing = set()
        numeric_col = roles.numeric(cm.declared_type)
        for eid, tag, val in c.execute(f"SELECT eid, typeof({qc}), {qc} FROM {qt} WHERE eid IS NOT NULL"):
            ek = p.canonical_val_key("integer", eid)
            if val is None:
                missing.add(ek)
            else:
                by_ent_keys.setdefault(ek, set()).add(p.canonical_val_key(tag, val))
                by_ent_buckets.setdefault(ek, set()).add(p.bin_index(val) if numeric_col else p.shape(val))
        ref_vals = Counter()
        for g in by_ent_keys.values():
            ref_vals.update(sorted(g, key=repr)[:5])
        ref_dist = Counter()
        for g in by_ent_buckets.values():
            ref_dist.update(sorted(g, key=repr)[:5])
        assert stats[cm.name]["null_count"] == len(missing)
        assert stats[cm.name]["distinct_count"] == len(ref_vals)
        assert stats[cm.name]["values"] == ref_vals
        assert stats[cm.name]["distribution"] == ref_dist


def test_agent_profile_sha256_binding():
    from silhouette.agent import canonical_profile_sha256

    c = memory("CREATE TABLE t(x TEXT)")
    box = SchemaSandbox(c)
    prof = {"t": {"x": ["TEXT", 0.0, "10^0", ["AAAA"], ["alpha"]]}}
    digest = canonical_profile_sha256(prof)

    # Valid digest passes
    sql = generate_sql(
        lambda _req: {"name": "submit_sql", "arguments": {"sql": "SELECT x FROM t"}},
        "List x", "t(x TEXT)", prof, sandbox=box, expected_profile_sha256=digest,
    )
    assert sql == "SELECT x FROM t"

    # Tampered profile raises ValueError before calling model
    tampered = {"t": {"x": ["TEXT", 0.0, "10^0", ["AAAA"], ["beta"]]}}
    called = []
    with pytest.raises(ValueError, match="digest mismatch"):
        generate_sql(
            lambda req: called.append(req) or {"name": "submit_sql", "arguments": {"sql": "SELECT x FROM t"}},
            "List x", "t(x TEXT)", tampered, sandbox=box, expected_profile_sha256=digest,
        )
    assert called == []


def test_transmission_ledger_segment_aware():
    rows = [("ochre", 3), ("azure", 4)]
    prof = {"tiles": {"colour": ["TEXT", 0.0, "10^0", ["AAAAA"], ["azure", "ochre"]], "sides": ["INT", 0.0, "10^0", [3, 3, 4], [3, 4]]}}
    prof_txt = json.dumps(prof, ensure_ascii=False, separators=(",", ":"))
    ledger = TransmissionLedger({"tiles": rows}, prof, public=["CREATE TABLE tiles(colour TEXT, sides INT)"])

    # 1. Segment-aware profile_1d triggers full_payload_coincidences ONLY (0 outside_profile_serialized_tuples, 0 out_of_profile_literals)
    s1 = ledger.record(
        segments=[
            {"kind": "prompt", "text": "Question: How many tiles?"},
            {"kind": "profile_1d", "text": prof_txt},
            {"kind": "tool_reply", "text": '{"status": "success"}'},
        ]
    )
    assert s1["copied_private_tuples"] == [0, 0]
    assert s1["outside_profile_serialized_tuples"] == [0, 0]
    assert s1["full_payload_coincidences"] == [2, 2]
    assert s1["out_of_profile_literals"] == [0, 0]

    # 2. profile_1d_text inside an outbound request dict also isolates profile_1d cleanly
    ledger2 = TransmissionLedger({"tiles": rows}, prof, public=["How many?"])
    s2 = ledger2.record({"messages": [{"role": "user", "content": f"How many?\n{prof_txt}"}]}, profile_1d_text=prof_txt)
    assert s2["outside_profile_serialized_tuples"] == [0, 0]
    assert s2["full_payload_coincidences"] == [2, 2]

    # 3. Non-profile serialized tuple in tool_reply + copied_row_ids provenance
    s3 = ledger.record(
        segments=[{"kind": "tool_reply", "text": '["ochre", 3]'}],
        copied_row_ids=[("tiles", 1)],
    )
    assert s3["copied_private_tuples"] == [1, 1]
    assert s3["outside_profile_serialized_tuples"] == [1, 1]


def test_score_bird_dual_iif_compat(monkeypatch):
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 31, 1))
    con = sqlite3.connect(":memory:")
    con.executescript("""
        CREATE TABLE schools(id INTEGER PRIMARY KEY, score INTEGER, flag TEXT);
        INSERT INTO schools VALUES (1, 10, '1'), (2, 0, '0'), (3, NULL, 'abc'), (4, -2, '1abc');
    """)
    q_iif = "SELECT id, IIF(score > 0, 'pos', 'nonpos'), IIF(flag, 't', 'f') FROM schools ORDER BY id"
    q_case = "SELECT id, CASE WHEN score > 0 THEN 'pos' ELSE 'nonpos' END, CASE WHEN flag THEN 't' ELSE 'f' END FROM schools ORDER BY id"
    score_bird_dual.register_compat_functions(con, force=True)
    rows_udf = con.execute(q_iif).fetchall()
    rows_case = con.execute(q_case).fetchall()
    assert rows_udf == rows_case == [(1, "pos", "t"), (2, "nonpos", "f"), (3, "nonpos", "f"), (4, "nonpos", "t")]
    res = score_bird_dual.score_sql_on_connection(con, q_iif, q_case)
    assert res["official_set_ex"] == 1 and res["strict_ex"] == 1
    con.close()


def test_schema_sandbox_cross_thread():
    import threading

    src = memory("CREATE TABLE items(id INTEGER PRIMARY KEY, label TEXT); INSERT INTO items VALUES (1, 'x');")
    box = SchemaSandbox(src)
    errors = []
    results = []
    lock = threading.Lock()

    def worker(tid: int):
        try:
            for i in range(15):
                r_ok = box.try_sql("SELECT id, label FROM items WHERE id > 0")
                r_bad = box.try_sql("SELECT nope FROM items")
                with lock:
                    results.append((r_ok["status"], r_bad["category"]))
        except Exception as exc:
            with lock:
                errors.append(repr(exc))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len(results) == 120
    assert all(st == "success" and cat == "unknown_column" for st, cat in results)
    box.connection.close()


def test_profile_lookup_tool_and_ehrsql_manifest():
    from silhouette.profile_tool import IND8ProfileTool, PROFILE_TOOLS_SCHEMA, ProfileLookupTool

    assert [t["function"]["name"] for t in PROFILE_TOOLS_SCHEMA] == ["get_table_profile", "get_column_profile"]
    assert IND8ProfileTool is ProfileLookupTool

    manifest = read(ROOT / "manifests/ehrsql_profiles_manifest.json")
    assert manifest["tables_private"] == 13
    assert set(manifest["tables_public"]) == PUBLIC_TABLES
    for rel_sub, exp_sha in manifest["sha256"].items():
        fpath = ROOT / "profiles" / rel_sub
        assert fpath.exists(), fpath
        assert hashlib.sha256(fpath.read_bytes()).hexdigest() == exp_sha, rel_sub

    ind_path = ROOT / "profiles/ehrsql_ind8_prompt/release_01.json"
    ind_sha = manifest["sha256"]["ehrsql_ind8_prompt/release_01.json"]
    tool = ProfileLookupTool(ind_path, expected_sha256=ind_sha)
    assert len(tool.list_tables()) == 13

    t_res = tool.get_table_profile("PATIENTS")
    assert t_res["status"] == "ok" and t_res["table"] == "patients" and "subject_id" in t_res["columns"]
    c_res = tool.get_column_profile("patients", "GENDER")
    assert c_res["status"] == "ok" and c_res["column"] == "gender" and len(c_res["profile"]) == 5
    assert tool.get_table_profile("missing_tbl")["category"] == "unknown_table"
    assert tool.get_column_profile("patients", "missing_col")["category"] == "unknown_column"

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        ProfileLookupTool(ind_path, expected_sha256="0" * 64)

    # Dict-mode ProfileLookupTool & frozen public 4-dictionary lookup
    dict_tool = ProfileLookupTool({"t": {"x": ["TEXT", 0.0, "10^0", ["AAA"], ["a"]]}})
    assert dict_tool.get_column_profile("t", "x")["profile"][4] == ["a"]
    pub_lkp = PublicLookup(
        ROOT / "profiles/public_4_dictionaries.sqlite",
        expected_sha256=manifest["sha256"]["public_4_dictionaries.sqlite"],
    )
    pub_res = pub_lkp.lookup_reference("SELECT COUNT(*) FROM d_items")
    assert pub_res["status"] == "success" and pub_res["rows"][0][0] > 0
    pub_lkp.connection.close()


def test_dp_query_total_udfs_and_dbwide_exhaustion(monkeypatch):
    from silhouette.dp_query import DPQueryRowDBWideEngine, INT64_MAX, INT64_MIN, _dp_safe_abs, _dp_safe_printf

    assert _dp_safe_abs(INT64_MIN) == INT64_MAX
    assert _dp_safe_abs(None) is None
    assert _dp_safe_printf("%s-%d", "ok", 7) == "ok-7"
    assert _dp_safe_printf("%d", "not-a-number") is None

    c = memory("""
        CREATE TABLE a(id INTEGER PRIMARY KEY, v INTEGER);
        CREATE TABLE b(id INTEGER PRIMARY KEY, w TEXT);
        INSERT INTO a VALUES (1, 10), (2, 20);
        INSERT INTO b VALUES (1, 'x');
    """)
    monkeypatch.setattr(dg, "sample_dgauss", lambda *args: 0)
    dbw = DPQueryRowDBWideEngine(c, q_plan=1, max_probes_per_q=2, max_db_probes=2)
    assert dbw.query("SELECT * FROM a")["status"] == "rejected_public_check"
    assert dbw.probes_used == 0

    r1 = dbw.query("SELECT COUNT(*) FROM a", question_id=1)
    assert r1["status"] == "ok" and r1["noisy_count"] == 2
    r1_cached = dbw.query('select count(*) from "a"', question_id=2)
    assert r1_cached["cached"] is True and dbw.probes_used == 1

    r2 = dbw.query("SELECT COUNT(*) FROM b", question_id=2)
    assert r2["status"] == "ok" and r2["noisy_count"] == 1 and dbw.probes_used == 2
    r_ex = dbw.query("SELECT COUNT(*) FROM a WHERE v > 0", question_id=3)
    assert r_ex["status"] == "budget_exhausted" and r_ex["category"] == "db_probe_budget_exhausted"
    dbw.close()
    c.close()


def test_dp_query_total_function_neighbor_sensitivity_and_docstrings(monkeypatch):
    from silhouette import dp_query as dq

    # 1. Verify no progress-handler wall-clock timeout or table-level catch-all in dp_query.py
    src = (ROOT / "silhouette/dp_query.py").read_text(encoding="utf-8")
    assert "set_progress_handler" not in src
    assert "_fetch_rows_total" not in src
    assert "db-1row" in dq.DPQueryRowDBWideEngine.__doc__
    assert "all-row" in dq.DPQueryRowDBWideEngine.__doc__

    # 2. Neighboring databases D ~ D' where D' appends a row causing div-by-zero / INT64_MIN / bad printf
    INT64_MIN = -(1 << 63)
    def build_pair():
        ca = memory("""
            CREATE TABLE t(id INTEGER PRIMARY KEY, eid INTEGER, grp TEXT, val INTEGER, tag TEXT);
            INSERT INTO t VALUES
                (1, 100, 'A', 5, '%s'),
                (2, 101, 'A', 10, '%s'),
                (3, 102, 'B', 2, '%s'),
                (4, 103, 'B', 4, '%s');
        """)
        cb = memory("""
            CREATE TABLE t(id INTEGER PRIMARY KEY, eid INTEGER, grp TEXT, val INTEGER, tag TEXT);
            INSERT INTO t VALUES
                (1, 100, 'A', 5, '%s'),
                (2, 101, 'A', 10, '%s'),
                (3, 102, 'B', 2, '%s'),
                (4, 103, 'B', 4, '%s');
        """)
        return ca, cb

    monkeypatch.setattr(dg, "sample_dgauss", lambda *args: 0)
    adversarial_rows = [
        (5, 104, "A", 0, "%d"),               # division by zero + invalid %d printf
        (5, 104, "A", INT64_MIN, "%bad%"),    # INT64_MIN overflow + malformed printf
        (5, 104, "B", -99999999, "%s %s %f"), # extreme out-of-range + missing printf args
    ]
    for adv in adversarial_rows:
        ca, cb = build_pair()
        cb.execute("INSERT INTO t VALUES (?, ?, ?, ?, ?)", adv)
        cb.commit()
        for cls in (dq.DPQueryRowEngine, dq.DPQueryRowDBWideEngine):
            ea = cls(ca, q_plan=20, max_probes_per_q=2, clip_bounds=(-100, 100), sum_scale=10)
            eb = cls(cb, q_plan=20, max_probes_per_q=2, clip_bounds=(-100, 100), sum_scale=10)
            # COUNT(*) with div-by-zero, abs(INT64_MIN), printf(tag, 1) -> never resets other 4 rows to 0!
            for sql in (
                "SELECT COUNT(*) FROM t WHERE 10 / val > 0",
                "SELECT COUNT(*) FROM t WHERE NOT (10 / val > 0)",
                "SELECT COUNT(*) FROM t WHERE abs(val) > 0",
                "SELECT COUNT(*) FROM t WHERE printf(tag, 1) IS NOT NULL",
                "SELECT COUNT(DISTINCT eid) FROM t WHERE 10 / val > 0",
            ):
                ra = ea.query(sql)
                rb = eb.query(sql)
                assert ra["status"] == "ok" and rb["status"] == "ok"
                assert ra["noisy_count"] == 4 or "NOT" in sql  # 4 clean rows never reset to 0
                assert abs(rb["noisy_count"] - ra["noisy_count"]) <= 1, (sql, adv, ra, rb)

            # SUM with [L, U] clamp sensitivity <= U - L
            sa = ea.query("SELECT SUM(val) FROM t WHERE 10 / val > 0")
            sb = eb.query("SELECT SUM(val) FROM t WHERE 10 / val > 0")
            assert sa["noisy_sum"] == 21.0
            assert abs(sb["noisy_sum"] - sa["noisy_sum"]) <= 200.0

            # GROUP BY sensitivity <= 1 per bin
            ga = ea.query("SELECT grp, COUNT(*) FROM t WHERE abs(val) > 0 GROUP BY grp")
            gb = eb.query("SELECT grp, COUNT(*) FROM t WHERE abs(val) > 0 GROUP BY grp")
            assert ga["status"] == "ok" and gb["status"] == "ok"
            ea.close(); eb.close()

        # Entity engine direct-star sensitivity <= 1
        specs = {"t": {"mode": "entity_direct", "entity_col": "eid"}}
        enta = dq.DPQueryEntityEngine(ca, specs, q_plan=10, max_probes_per_q=2)
        entb = dq.DPQueryEntityEngine(cb, specs, q_plan=10, max_probes_per_q=2)
        rea = enta.query("SELECT COUNT(DISTINCT eid) FROM t WHERE printf(tag, 1) IS NOT NULL")
        reb = entb.query("SELECT COUNT(DISTINCT eid) FROM t WHERE printf(tag, 1) IS NOT NULL")
        assert rea["noisy_count"] == 4 and abs(reb["noisy_count"] - rea["noisy_count"]) <= 1
        enta.close(); entb.close()
        ca.close(); cb.close()

    # 3. Direct eval_unit_total exception isolation
    c = memory("CREATE TABLE t(x INTEGER);")
    parsed = dq.parse_and_normalize_probe("SELECT COUNT(*) FROM t WHERE x > 0", roles.extract_public_schema(c))
    assert dq.eval_unit_total({}, parsed) == 0  # missing key caught strictly per-unit -> 0
    c.close()


def test_public_tables_skip_publish_noiseless_and_pubcap(tmp_path):
    c = memory("""
        CREATE TABLE d_items(itemid INTEGER PRIMARY KEY, label TEXT);
        CREATE TABLE patients(subject_id INTEGER PRIMARY KEY, gender TEXT, age INTEGER);
        INSERT INTO d_items VALUES (1, 'secret-dict-token');
        INSERT INTO patients VALUES (10, 'F', 42), (11, 'M', 65);
    """)
    reads = []
    def auth(action, first, second, db, origin):
        if action == sqlite3.SQLITE_READ:
            reads.append(first)
            if first == "d_items":
                return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    c.set_authorizer(auth)
    prof_nl = p.publish_noiseless(c, public_tables={"d_items"})
    c.set_authorizer(lambda *_args: sqlite3.SQLITE_OK)

    assert "d_items" not in reads
    assert list(prof_nl.keys()) == ["patients"]
    assert prof_nl["patients"]["gender"][4] == ["F", "M"]
    assert len(prof_nl["patients"]["age"][3]) == 3
    assert all(x is not None for x in prof_nl["patients"]["age"][3])

    # Public DDL cap & truncation
    c.set_authorizer(auth)
    schema = roles.extract_public_schema(c)
    c.set_authorizer(lambda *_args: sqlite3.SQLITE_OK)
    cap_all = p.compute_public_profile_token_cap(schema)
    cap_priv = p.compute_public_profile_token_cap(schema, public_tables={"d_items"})
    assert cap_all > cap_priv > 0

    big_prof = {
        "patients": {
            "subject_id": ["INTEGER", 0.0, "10^0", [10.0, 10.0, 10.0], None],
            "gender": ["TEXT", 0.0, "10^0", ["A"], ["F_" + ("x" * 100), "M_" + ("y" * 100)]],
            "age": ["INTEGER", 0.0, "10^0", [20.0, 50.0, 50.0], [42, 65, 70, 80]],
        }
    }
    trimmed = p.truncate_profile_to_public_cap(big_prof, 240, roles.PublicSchema({"patients": schema.tables["patients"]}, ()))
    assert p.estimate_prompt_tokens(p.render_profile_prompt_block(trimmed)) <= 240

    # Synthetic rows truncation
    db_file = tmp_path / "synth.sqlite"
    cs = sqlite3.connect(str(db_file))
    cs.execute("CREATE TABLE patients(subject_id INTEGER PRIMARY KEY, gender TEXT, age INTEGER)")
    cs.executemany("INSERT INTO patients VALUES (?, ?, ?)", [(i, "F", 30 + i) for i in range(1, 50)])
    cs.commit()
    cs.close()
    rows_txt = p.truncate_synth_rows_to_public_cap(
        db_file, 150, roles.PublicSchema({"patients": schema.tables["patients"]}, ())
    )
    assert p.estimate_prompt_tokens(rows_txt) <= 150


def test_score_bird_dual_benchmark_filter_and_submission_split(tmp_path, monkeypatch):
    # Build synthetic bird_dir with 1 DB and 1 question
    bdir = tmp_path / "bird"
    bdir.mkdir()
    db_path = bdir / "toy.sqlite"
    con = sqlite3.connect(str(db_path))
    con.execute("CREATE TABLE t(id INTEGER PRIMARY KEY, v INTEGER)")
    con.execute("INSERT INTO t VALUES (1, 10)")
    con.commit()
    con.close()
    (bdir / "mini_dev_sqlite.jsonl").write_text(
        json.dumps({"question_id": 1, "db_id": "toy", "SQL": "SELECT v FROM t"}) + "\n",
        encoding="utf-8",
    )
    fake_root = tmp_path / "root"
    (fake_root / "stats").mkdir(parents=True)
    receipts = [
        {"benchmark": "ehrsql_600", "question_id": "q1", "db_id": "mimic_iv", "condition": "EHR/r0", "predicted_sql": "SELECT 1", "strict_ex": 1},
        {"benchmark": "bird", "question_id": 1, "db_id": "toy", "condition": "COND_A/r0", "predicted_sql": "SELECT v FROM t", "strict_ex": 1},
        {"benchmark": "bird", "question_id": 1, "db_id": "toy", "condition": "COND_A/r0", "predicted_sql": "", "strict_ex": 0},
        {"benchmark": "bird", "question_id": 1, "db_id": "toy", "condition": "COND_UNREP/r0", "predicted_sql": None, "strict_ex": None},
    ]
    (fake_root / "stats/sql_and_audit_receipts.jsonl").write_text(
        "\n".join(json.dumps(r) for r in receipts) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(score_bird_dual, "ROOT", fake_root)
    summary = score_bird_dual.replay_bird_dir(bdir)
    assert "EHR/r0" not in summary
    assert summary["COND_A/r0"]["n"] == 2
    assert summary["COND_A/r0"]["set_ex"] == 1
    assert summary["COND_A/r0"]["no_submission"] == 1
    assert summary["COND_A/r0"]["unreplayable"] == 0
    assert summary["COND_A/r0"]["errors"] == 0
    assert summary["COND_UNREP/r0"]["n"] == 0
    assert summary["COND_UNREP/r0"]["unreplayable"] == 1


def test_public_lookup_sha256_and_local_executor_preview(tmp_path):
    dict_path = tmp_path / "public_dict.sqlite"
    con = sqlite3.connect(str(dict_path))
    for t in PUBLIC_TABLES:
        con.execute('CREATE TABLE "' + t + '" (x INTEGER)')
    con.executemany("INSERT INTO d_items VALUES (?)", [(i,) for i in range(1005)])
    con.commit()
    con.close()

    good_sha = hashlib.sha256(dict_path.read_bytes()).hexdigest()
    api = PublicLookup(dict_path, expected_sha256=good_sha.upper())
    assert api.file_sha256 == good_sha
    api.connection.close()

    with pytest.raises(ValueError, match="Public dictionary SHA-256 mismatch"):
        PublicLookup(dict_path, expected_sha256="0" * 64)

    ex = LocalExecutor(dict_path)
    prev_full = ex.execute_preview("SELECT x FROM d_items WHERE x < 10 ORDER BY x")
    assert prev_full["columns"] == ["x"]
    assert len(prev_full["rows"]) == 10
    assert prev_full["truncated"] is False

    prev_capped = ex.execute_preview("SELECT x FROM d_items ORDER BY x")
    assert prev_capped["columns"] == ["x"]
    assert len(prev_capped["rows"]) == 1000
    assert prev_capped["rows"][0] == (0,) and prev_capped["rows"][-1] == (999,)
    assert prev_capped["truncated"] is True


def test_agent_release_bundle_and_profile_hard_cap_feasibility(tmp_path):
    from silhouette.agent import canonical_profile_sha256, file_sha256, verify_profile_sha256, verify_release_bundle

    prof = {
        "patients": {
            "subject_id": ["INTEGER", 0.0, "10^1", [1.0, 5.0, 9.0], None],
            "gender": ["TEXT", 0.0, "10^0", ["A", "B"], ["F_" + ("x" * 90), "M_" + ("y" * 90)]],
            "notes": ["TEXT", 0.1, "10^1", ["AAAA_BBBB"], ["note_alpha", "note_beta"]],
        },
        "admissions": {
            "hadm_id": ["INTEGER", 0.0, "10^1", [10.0, 20.0, 30.0], None],
            "subject_id": ["INTEGER", 0.0, "10^1", [1.0, 5.0, 9.0], None],
            "admittime": ["TEXT", 0.0, "10^1", ["YYYY-MM-DD"], ["2020-01-01"]],
        },
    }
    pfile = tmp_path / "bundle.json"
    pfile.write_text(json.dumps(prof, indent=2) + "\n", encoding="utf-8")
    raw_sha = file_sha256(pfile)
    canon_sha = canonical_profile_sha256(prof)
    assert raw_sha != canon_sha

    assert verify_profile_sha256(pfile, canon_sha) == canon_sha
    assert verify_profile_sha256(pfile, raw_sha) == canon_sha
    res = verify_release_bundle(pfile, expected_canonical_sha256=canon_sha, expected_file_sha256=raw_sha)
    assert res["file_sha256"] == raw_sha and res["canonical_sha256"] == canon_sha
    with pytest.raises(ValueError, match="file SHA-256 mismatch"):
        verify_release_bundle(pfile, expected_canonical_sha256=canon_sha, expected_file_sha256="a" * 64)
    with pytest.raises(ValueError, match="canonical SHA-256 mismatch"):
        verify_release_bundle(pfile, expected_canonical_sha256="b" * 64, expected_file_sha256=raw_sha)

    # Custom token counter + hard-cap feasibility (D7)
    con = memory("""
        CREATE TABLE patients(subject_id INTEGER PRIMARY KEY, gender TEXT, notes TEXT);
        CREATE TABLE admissions(hadm_id INTEGER PRIMARY KEY, subject_id INTEGER REFERENCES patients(subject_id), admittime TEXT);
    """)
    schema = roles.extract_public_schema(con)
    con.close()

    for cap, counter in (
        (220, p.estimate_bpe_tokens_approx),
        (195, p.estimate_prompt_tokens),
        (40, p.estimate_prompt_tokens),
        (12, len),
    ):
        trimmed = p.truncate_profile_to_cap(prof, cap, schema, token_counter=counter)
        rendered = p.render_profile_prompt_block(trimmed)
        assert counter(rendered) <= cap, (cap, counter(rendered))
        assert trimmed.get("truncated_hard") is True


def test_truncate_preserve_values_mode():
    from silhouette import profile as p, schema_roles as roles

    con = memory("""
        CREATE TABLE patients(subject_id INTEGER PRIMARY KEY, gender TEXT, notes TEXT);
        CREATE TABLE admissions(hadm_id INTEGER PRIMARY KEY, subject_id INTEGER REFERENCES patients(subject_id), admittime TEXT);
    """)
    schema = roles.extract_public_schema(con)
    con.close()
    prof = {
        "patients": {
            "subject_id": ["INTEGER", 0.0, "10^3", [1.0, 2.0, 3.0], None],
            "gender": ["TEXT", 0.0, "10^0", ["A"], ["F", "M"]],
            "notes": ["TEXT", 0.1, "10^2", ["AAA", "BBB", "CCC"], []],
        },
        "admissions": {
            "hadm_id": ["INTEGER", 0.0, "10^3", [10.0, 20.0, 30.0], None],
            "subject_id": ["INTEGER", 0.0, "10^3", [1.0, 5.0, 9.0], None],
            "admittime": ["TEXT", 0.0, "10^1", ["YYYY-MM-DD"], ["2020-01-01", "2020-01-02", "2020-01-03"]],
        },
    }
    cap = 225
    drop_first = p.truncate_profile_to_public_cap(prof, cap, schema, mode="drop_values_first")
    preserve = p.truncate_profile_to_public_cap(prof, cap, schema, mode="preserve_values")
    rendered = p.render_profile_capped(prof, cap, schema, mode="preserve_values")
    assert p.estimate_prompt_tokens(p.render_profile_prompt_block(drop_first)) <= cap
    assert p.estimate_prompt_tokens(rendered) <= cap
    drop_val_count = sum(
        len(entry[4])
        for cols in drop_first.values()
        if isinstance(cols, dict)
        for entry in cols.values()
        if isinstance(entry[4], list)
    )
    assert drop_val_count == 0
    assert preserve["patients"]["gender"][4] == ["F", "M"]
    assert preserve["admissions"]["admittime"][4] == ["2020-01-01", "2020-01-02", "2020-01-03"]
    assert "notes" not in preserve["patients"] or preserve["patients"]["notes"][1:4] == [None, None, None]
    with pytest.raises(ValueError, match="Unknown truncation mode"):
        p.truncate_profile_to_public_cap(prof, cap, schema, mode="bad_mode")


def test_verify_chain_of_custody_cli_and_tamper_detection(tmp_path):
    from certificates import verify_chain_of_custody as coc
    from silhouette.agent import canonical_profile_sha256, file_sha256

    out = subprocess.check_output(
        [sys.executable, "-B", str(ROOT / "certificates/verify_chain_of_custody.py"), "--self-test"],
        text=True,
    )
    assert "CHAIN OF CUSTODY SELF-TEST PASSED" in out

    # Synthetic repo where receipt carries a tampered/unverified release_sha256 (e.g. d8aaf0b7...)
    (tmp_path / "manifests").mkdir()
    (tmp_path / "profiles" / "ehrsql_dg8p").mkdir(parents=True)
    (tmp_path / "certificates").mkdir()
    (tmp_path / "stats").mkdir()

    prof = {"patients": {"subject_id": ["INTEGER", 0.0, "10^0", [1.0, 1.0, 1.0], None]}}
    prof_path = tmp_path / "profiles" / "ehrsql_dg8p" / "release_01.json"
    prof_path.write_text(json.dumps(prof, separators=(",", ":")) + "\n", encoding="utf-8")
    raw_sha = file_sha256(prof_path)
    canon_sha = canonical_profile_sha256(prof)

    (tmp_path / "manifests" / "ehrsql_profiles_manifest.json").write_text(
        json.dumps({"tables_private": 1, "tables_public": [], "sha256": {"ehrsql_dg8p/release_01.json": raw_sha}}),
        encoding="utf-8",
    )
    (tmp_path / "manifests" / "pubcap_releases_manifest.json").write_text(
        json.dumps({"arms": {"a": {"release_01": {"db": {"profile_path": "ehrsql_dg8p/release_01.json", "sha256": raw_sha}}}}}),
        encoding="utf-8",
    )
    (tmp_path / "stats" / "sql_and_audit_receipts.jsonl").write_text(
        json.dumps({
            "benchmark": "ehrsql_600",
            "question_id": "q1",
            "db_id": "mimic_iv",
            "condition": "EHR_DG8P_V3_R1/r0",
            "release_sha256": canon_sha,
        }) + "\n",
        encoding="utf-8",
    )
    ok = coc.verify_chain_of_custody(tmp_path, strict_clinical=True)
    assert ok["receipts_verified"] == 1

    # Tamper receipt to unverified clinical release_sha256
    tampered_sha = "d8aaf0b7" + "1" * 56
    (tmp_path / "stats" / "sql_and_audit_receipts.jsonl").write_text(
        json.dumps({
            "benchmark": "ehrsql_600",
            "question_id": "q1",
            "db_id": "mimic_iv",
            "condition": "EHR_DG8P_V3_R1/r0",
            "release_sha256": tampered_sha,
        }) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(coc.ChainOfCustodyError, match=tampered_sha):
        coc.verify_chain_of_custody(tmp_path, strict_clinical=True)
    proc = subprocess.run(
        [sys.executable, "-B", str(ROOT / "certificates/verify_chain_of_custody.py"), "--root", str(tmp_path), "--strict-clinical"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1
    assert tampered_sha in proc.stderr

    # Canonical repository strict-clinical must pass (exit 0) while legacy DG8P_R1/r0 fails (exit 1, d8aaf0b7...)
    out_strict = subprocess.run(
        [sys.executable, "-B", str(ROOT / "certificates/verify_chain_of_custody.py"), "--strict-clinical"],
        capture_output=True,
        text=True,
    )
    assert out_strict.returncode == 0, out_strict.stderr
    assert "96 conditions" in out_strict.stdout
    assert "45224" in out_strict.stdout
    out_legacy = subprocess.run(
        [sys.executable, "-B", str(ROOT / "certificates/verify_chain_of_custody.py"), "--conditions", "DG8P_R1/r0"],
        capture_output=True,
        text=True,
    )
    assert out_legacy.returncode == 1 and "d8aaf0b7" in out_legacy.stderr

    surv = read(ROOT / "stats/clinical_survival_3stage_verified.json")
    assert surv["resampling_method"] == "whole_patient_bootstrap_with_replacement"
    assert surv["schedules"]["db_global_eps8"]["cohorts"]["100000"]["stage3_eq10_top15_published"]["all_125"]["mean_pct"] == 60.18


def test_sandbox_cte_and_legacy_parity(tmp_path):
    from silhouette.sandbox import FUNCTIONS, ordinary_tables, restrict, restrict_legacy_no_cte

    db_path = tmp_path / "bird_cte.sqlite"
    con = sqlite3.connect(str(db_path))
    con.executescript(
        """
        CREATE TABLE schools(CDSCode TEXT PRIMARY KEY, District TEXT, Charter INTEGER, Enrollment INTEGER);
        INSERT INTO schools VALUES ('1', 'Unified A', 1, 100), ('2', 'Unified A', 0, 250), ('3', 'Unified B', 1, 400);
        """
    )
    con.commit()

    q20_sql = "WITH adm AS (SELECT District, Charter FROM schools WHERE Charter = 1) SELECT District FROM adm ORDER BY District"
    chained_sql = (
        "WITH base AS (SELECT District, Enrollment FROM Schools), "
        "agg AS (SELECT District, SUM(Enrollment) AS total_enr FROM base GROUP BY District) "
        "SELECT District, total_enr FROM agg ORDER BY total_enr DESC"
    )
    mat_count_sql = (
        "WITH grouped AS (SELECT District, COUNT(*) AS cnt FROM schools GROUP BY District) "
        "SELECT COUNT(*) FROM grouped"
    )
    rec_sql = (
        "WITH RECURSIVE cnt(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM cnt WHERE x < 4) "
        "SELECT x FROM cnt"
    )
    win_sql = "SELECT CDSCode, CUME_DIST() OVER (ORDER BY Enrollment) FROM schools ORDER BY CDSCode LIMIT 1"

    box = SchemaSandbox(con)
    ex = LocalExecutor(db_path)
    for sql in (q20_sql, chained_sql, mat_count_sql, rec_sql, win_sql):
        assert box.try_sql(sql) == {"status": "success"}, sql

    assert ex.execute(q20_sql) == [("Unified A",), ("Unified B",)]
    assert ex.execute(chained_sql) == [("Unified B", 400), ("Unified A", 350)]
    assert ex.execute(mat_count_sql) == [(2,)]
    assert ex.execute(rec_sql) == [(1,), (2,), (3,), (4,)]
    assert ex.execute(win_sql) == [("1", pytest.approx(1.0 / 3.0))]

    # Expanded deterministic function whitelist & strict exclusions
    expected_added = {
        "concat", "concat_ws", "timediff", "floor", "ceil", "ceiling", "power", "pow",
        "sqrt", "mod", "ln", "log", "log10", "log2", "exp", "pi", "acos", "asin", "atan",
        "atan2", "cos", "sin", "tan", "degrees", "radians", "cume_dist", "format",
        "octet_length", "unistr", "unistr_quote",
    }
    assert expected_added <= FUNCTIONS
    for forbidden in (
        "load_extension", "random", "randomblob", "changes", "total_changes",
        "last_insert_rowid", "sqlite_version", "sqlite_source_id",
        "sqlite_compileoption_used", "sqlite_compileoption_get", "fts3_tokenizer",
    ):
        assert forbidden not in FUNCTIONS

    # Any newly whitelisted function supported by the host SQLite build passes sandbox & LocalExecutor
    probe_exprs = [
        "FLOOR(1.9)", "CEIL(1.1)", "CEILING(1.1)", "POWER(2, 3)", "POW(2, 4)",
        "SQRT(9)", "MOD(7, 3)", "OCTET_LENGTH('abc')", "FORMAT('%s-%d', 'ok', 2)",
        "CONCAT('a', 'b')", "CONCAT_WS('-', 'a', 'b')", "TIMEDIFF('2020-01-02', '2020-01-01')",
    ]
    for expr in probe_exprs:
        try:
            con.execute(f"SELECT {expr}").fetchall()
        except sqlite3.OperationalError:
            continue
        q = f"SELECT {expr} FROM schools LIMIT 1"
        assert box.try_sql(q) == {"status": "success"}, q
        assert len(ex.execute(q)) == 1

    # Authorizer allows newly whitelisted function names even on custom-registered deterministic functions
    con_fn = sqlite3.connect(":memory:")
    con_fn.execute("CREATE TABLE t(x REAL)")
    con_fn.create_function("floor", 1, lambda v: float(int(v)))
    restrict(con_fn, ["t"])
    assert con_fn.execute("SELECT floor(1.9)").fetchone() == (1.0,)
    con_fn.close()

    # Legacy allow_cte=False reproduces Round-8 D1 read_only_denied on CTEs
    legacy_box = SchemaSandbox(con, allow_cte=False)
    assert legacy_box.try_sql(q20_sql) == {"status": "error", "category": "read_only_denied"}
    with pytest.raises(sqlite3.DatabaseError, match="prohibited|authorized"):
        ex.execute(q20_sql, allow_cte=False)
    with pytest.raises(sqlite3.DatabaseError, match="prohibited|authorized"):
        ex.execute_preview(q20_sql, allow_cte=False)

    con_leg = sqlite3.connect(str(db_path))
    restrict_legacy_no_cte(con_leg, ordinary_tables(con_leg))
    with pytest.raises(sqlite3.DatabaseError, match="prohibited|authorized"):
        con_leg.execute(q20_sql).fetchall()
    con_leg.close()
    box.connection.close()
    legacy_box.connection.close()
    con.close()


def test_sandbox_rejects_views_triggers_and_unsafe_ops(tmp_path):
    from silhouette.sandbox import ordinary_tables

    # Real-DB compatibility (e.g., mimic_iv.sqlite with unused persistent views/triggers):
    # opening SchemaSandbox / LocalExecutor succeeds for ordinary tables & CTEs while
    # strictly excluding and denying reads against persistent views or triggers.
    view_db = tmp_path / "with_view_and_trigger.sqlite"
    cv = sqlite3.connect(str(view_db))
    cv.executescript(
        """
        CREATE TABLE patients(subject_id INT PRIMARY KEY, gender TEXT);
        INSERT INTO patients VALUES (1, 'F');
        CREATE VIEW patient_treatment_response AS SELECT subject_id FROM patients;
        CREATE TRIGGER tr AFTER INSERT ON patients BEGIN DELETE FROM patients; END;
        """
    )
    cv.commit()

    assert ordinary_tables(cv) == ["patients"]
    vbox = SchemaSandbox(cv)
    assert vbox.try_sql("WITH c AS (SELECT * FROM patients) SELECT count(*) FROM c") == {"status": "success"}
    assert vbox.try_sql("SELECT * FROM patient_treatment_response")["status"] == "error"
    vbox.connection.close()
    cv.close()

    vex = LocalExecutor(view_db)
    assert vex.execute("WITH c AS (SELECT * FROM patients) SELECT count(*) FROM c") == [(1,)]
    for denied_view_sql in (
        "SELECT * FROM patient_treatment_response",
        "SELECT COUNT(*) FROM patient_treatment_response",
    ):
        with pytest.raises(sqlite3.DatabaseError, match="prohibited|authorized"):
            vex.execute(denied_view_sql)
        with pytest.raises(sqlite3.DatabaseError, match="prohibited|authorized"):
            vex.execute_preview(denied_view_sql)

    safe_db = tmp_path / "safe.sqlite"
    cs = sqlite3.connect(str(safe_db))
    cs.execute("CREATE TABLE t(x INT)")
    cs.execute("INSERT INTO t VALUES (1)")
    cs.commit()
    box = SchemaSandbox(cs)
    ex = LocalExecutor(safe_db)
    for bad_sql in (
        "SELECT sql FROM sqlite_master",
        "SELECT COUNT(*) FROM sqlite_master",
        "PRAGMA table_info(t)",
        "SELECT * FROM pragma_table_info('t')",
        "ATTACH DATABASE ':memory:' AS aux",
        "INSERT INTO t VALUES (2)",
        "UPDATE t SET x = 3",
        "DELETE FROM t",
        "DROP TABLE t",
        "SELECT random() FROM t",
    ):
        r = box.try_sql(bad_sql)
        assert r["status"] == "error", (bad_sql, r)
        with pytest.raises((sqlite3.Error, ValueError)):
            ex.execute(bad_sql)
    box.connection.close()
    cs.close()


def test_pinning_enforcement_release_bundle_and_public_lookup(tmp_path):
    from silhouette.agent import canonical_profile_sha256, verify_release_bundle

    prof = {"t": {"x": ["TEXT", 0.0, "10^0", ["AAAA"], ["alpha"]]}}
    prof_path = tmp_path / "release.json"
    prof_path.write_text(json.dumps(prof, separators=(",", ":")) + "\n", encoding="utf-8")
    canon_sha = canonical_profile_sha256(prof)

    dict_path = tmp_path / "dict.sqlite"
    cd = sqlite3.connect(str(dict_path))
    for tbl in PUBLIC_TABLES:
        cd.execute(f'CREATE TABLE "{tbl}" (x INTEGER)')
    cd.execute("INSERT INTO d_items VALUES (7)")
    cd.commit()
    cd.close()
    dict_sha = hashlib.sha256(dict_path.read_bytes()).hexdigest()

    # PublicLookup requires expected_sha256 unless allow_unpinned=True
    with pytest.raises(ValueError, match="expected_sha256 is required"):
        PublicLookup(dict_path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        PublicLookup(dict_path, expected_sha256="0" * 64)
    lkp = PublicLookup(dict_path, expected_sha256=dict_sha)
    lkp.connection.close()
    lkp_unpinned = PublicLookup(dict_path, allow_unpinned=True)
    lkp_unpinned.connection.close()

    # verify_release_bundle requires non-empty hex profile digest and dictionary digest when dictionary_path given
    for bad_digest in (None, "", "   ", "not-hex-xyz"):
        with pytest.raises(ValueError):
            verify_release_bundle(prof_path, bad_digest)
    with pytest.raises(ValueError):
        verify_release_bundle(prof_path, canon_sha, dictionary_path=dict_path)
    with pytest.raises(ValueError):
        verify_release_bundle(prof_path, canon_sha, dictionary_path=dict_path, expected_dictionary_sha256="")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_release_bundle(prof_path, canon_sha, dictionary_path=dict_path, expected_dictionary_sha256="f" * 64)

    verified = verify_release_bundle(
        prof_path,
        expected_profile_sha256=canon_sha,
        dictionary_path=dict_path,
        expected_dictionary_sha256=dict_sha,
    )
    assert verified["profile_sha256"] == canon_sha
    assert verified["dictionary_sha256"] == dict_sha

    # generate_sql(..., require_pinned_digest=True) enforces expected_profile_sha256
    c = memory("CREATE TABLE t(x TEXT)")
    box = SchemaSandbox(c)
    with pytest.raises(ValueError, match="expected_profile_sha256 is required"):
        generate_sql(
            lambda _r: {"name": "submit_sql", "arguments": {"sql": "SELECT x FROM t"}},
            "List x",
            "t(x TEXT)",
            prof,
            sandbox=box,
            require_pinned_digest=True,
        )
    sql = generate_sql(
        lambda _r: {"name": "submit_sql", "arguments": {"sql": "SELECT x FROM t"}},
        "List x",
        "t(x TEXT)",
        prof,
        sandbox=box,
        expected_profile_sha256=canon_sha,
        require_pinned_digest=True,
    )
    assert sql == "SELECT x FROM t"
    box.connection.close()
    c.close()


def test_chain_of_custody_tamper_and_deployment_prohibited(tmp_path):
    from certificates import verify_chain_of_custody as coc
    from silhouette.agent import canonical_profile_sha256, file_sha256, verify_release_bundle

    # 1. verify_release_bundle(..., allow_non_dp=False) rejects offline-only noiseless profiles & deployment_prohibited dicts
    ehr_nl = ROOT / "profiles/ehrsql_noiseless_nolkp/release_01.json"
    ehr_nl_sha = file_sha256(ehr_nl)
    with pytest.raises(ValueError, match="deployment_prohibited|noiseless"):
        verify_release_bundle(ehr_nl, expected_profile_sha256=ehr_nl_sha)
    ok_ehr_nl = verify_release_bundle(ehr_nl, expected_profile_sha256=ehr_nl_sha, allow_non_dp=True)
    assert ok_ehr_nl["file_sha256"] == ehr_nl_sha

    bird_nl = ROOT / "profiles/bird_noiseless_clean/release_01/california_schools.json"
    bird_nl_sha = file_sha256(bird_nl)
    with pytest.raises(ValueError, match="deployment_prohibited|noiseless"):
        verify_release_bundle(bird_nl, expected_profile_sha256=bird_nl_sha)
    assert verify_release_bundle(bird_nl, expected_profile_sha256=bird_nl_sha, allow_non_dp=True)["file_sha256"] == bird_nl_sha

    deid = read(ROOT / "stats/mimic_demo94_patient_bins_deid.json")
    assert deid.get("deployment_prohibited") is True

    cov = read(ROOT / "manifests/custody_coverage_manifest_v10.json")
    assert cov["total_aligned_slots"] == 83015
    assert cov["unique_benchmark_condition_series"] == 202
    assert cov["canonical_release_custody_slots"] == 45224
    assert cov["canonical_release_custody_conditions"] == 96
    assert cov["outbound_messages_sha256_slots"] == 78089
    assert cov["profile_inventory_breakdown"]["total_custody_pinned_profile_files"] == 736
    assert cov["profile_inventory_breakdown"]["sidecar_metadata_files"]["total_sidecars"] == 388

    # 2. Semantic (benchmark, condition, db_id) cross-release swap & dropped-slot tamper tests
    (tmp_path / "manifests").mkdir()
    (tmp_path / "profiles" / "ehrsql_dg8p").mkdir(parents=True)
    (tmp_path / "certificates").mkdir()
    (tmp_path / "stats").mkdir()

    p1 = {"patients": {"subject_id": ["INTEGER", 0.0, "10^0", [1.0, 1.0, 1.0], None]}}
    p2 = {"patients": {"subject_id": ["INTEGER", 0.0, "10^0", [2.0, 2.0, 2.0], None]}}
    f1 = tmp_path / "profiles/ehrsql_dg8p/release_01.json"
    f2 = tmp_path / "profiles/ehrsql_dg8p/release_02.json"
    f1.write_text(json.dumps(p1, separators=(",", ":")) + "\n", encoding="utf-8")
    f2.write_text(json.dumps(p2, separators=(",", ":")) + "\n", encoding="utf-8")
    c1, c2 = canonical_profile_sha256(p1), canonical_profile_sha256(p2)
    r1, r2 = file_sha256(f1), file_sha256(f2)

    (tmp_path / "manifests/ehrsql_profiles_manifest.json").write_text(
        json.dumps({"tables_private": 1, "tables_public": [], "sha256": {"ehrsql_dg8p/release_01.json": r1, "ehrsql_dg8p/release_02.json": r2}}),
        encoding="utf-8",
    )
    (tmp_path / "manifests/pubcap_releases_manifest.json").write_text(
        json.dumps({"arms": {"a": {"release_01": {"db": {"profile_path": "ehrsql_dg8p/release_01.json", "sha256": r1}}, "release_02": {"db": {"profile_path": "ehrsql_dg8p/release_02.json", "sha256": r2}}}}}),
        encoding="utf-8",
    )
    (tmp_path / "manifests/custody_coverage_manifest_v10.json").write_text(
        json.dumps({
            "series": {
                "ehrsql_600::EHR_DG8P_V3_R1/r0": {
                    "benchmark": "ehrsql_600",
                    "condition": "EHR_DG8P_V3_R1/r0",
                    "lineage_tier": "canonical_custody_v3",
                    "expected_slots": 1,
                    "expected_release_sha256_by_db": {"mimic_iv": c1},
                    "expected_outbound_sha256_slots": 0,
                }
            }
        }),
        encoding="utf-8",
    )
    # Swap R2 digest into R1 receipt -> semantic mismatch must raise ChainOfCustodyError!
    (tmp_path / "stats/sql_and_audit_receipts.jsonl").write_text(
        json.dumps({"benchmark": "ehrsql_600", "question_id": "q1", "db_id": "mimic_iv", "condition": "EHR_DG8P_V3_R1/r0", "release_sha256": c2}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(coc.ChainOfCustodyError):
        coc.verify_chain_of_custody(tmp_path, strict_clinical=True)

    # Dropped-slot tamper -> expected_slots mismatch must raise ChainOfCustodyError!
    (tmp_path / "stats/sql_and_audit_receipts.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(coc.ChainOfCustodyError):
        coc.verify_chain_of_custody(tmp_path, strict_clinical=True)


def test_unbalanced_exact_rademacher_and_manifest_doc_parity():
    import re
    from stats import reproduce_stats as rs

    with pytest.raises(ValueError, match="balanced replicates"):
        rs.exact_rademacher_p([1.0, 0.0], [0.5, 0.0], 3, 2, allow_unbalanced=False)

    # All-sign-flips exact check on n_nz=2 unbalanced 3-vs-2: diffs [+1, +1/2] -> scale 6 -> [6, 3] -> mags [2, 1], obs=3 -> only (+,+) gives |sum| >= 3 => 2/4 = 0.5
    p_unb = rs.exact_rademacher_p([1.0, 0.5], [0.0, 0.0], 3, 2, allow_unbalanced=True)
    assert p_unb == pytest.approx(0.5)

    # compare_conditions emits both all-release primary p and matched_minR / matched_r1 sensitivities when R_a != R_b
    data = {
        "A1": {1: 1, 2: 1, 3: 1, 4: 0},
        "A2": {1: 1, 2: 1, 3: 0, 4: 0},
        "A3": {1: 1, 2: 0, 3: 0, 4: 0},
        "B1": {1: 0, 2: 0, 3: 0, 4: 0},
        "B2": {1: 0, 2: 0, 3: 0, 4: 1},
    }
    cmp_res = rs.compare_conditions(data, ["A1", "A2", "A3"], ["B1", "B2"], [1, 2, 3, 4])
    assert "all-release exact Rademacher" in cmp_res["p_method"]
    assert "matched_minR_question_p" in cmp_res and "matched_r1_question_p" in cmp_res

    # Stage-9 documentation parity check across README and docs/
    docs = [
        ROOT / "README.md",
        ROOT / "docs/TIER1_TIER2_REPRODUCIBILITY.md",
        ROOT / "docs/EXPERIMENTAL_LINEAGE.md",
        ROOT / "docs/A1_A10_COMPLIANCE.md",
    ]
    stale_tokens = ["67,999", "67999", "57,537", "57537", "17,062", "17062", "15,490", "15490", "28,448", "28448", "tr_table11_ehrhout", "3.53.2"]
    for doc_path in docs:
        txt = doc_path.read_text(encoding="utf-8")
        for tok in stale_tokens:
            assert tok not in txt, (doc_path.name, tok)
        assert not re.search(r"(?<![./])\./release-local\.json", txt), doc_path.name

    readme_txt = (ROOT / "README.md").read_text(encoding="utf-8")
    repro_txt = (ROOT / "docs/TIER1_TIER2_REPRODUCIBILITY.md").read_text(encoding="utf-8")
    for required in ("83,015", "78,089", "115", "96", "202", "195", "45,224"):
        assert required in readme_txt and required in repro_txt, required


def test_remap_builder_spec_and_ehrsql_gold_manifest():
    spec = read(ROOT / "manifests/remap_deterministic_builder_spec.json")
    assert spec["seed1"]["gold_matches"] == 130 and spec["seed1"]["bit_exact"] is True
    assert spec["seed2"]["gold_matches"] == 75 and spec["seed2"]["bit_exact"] is True

    gold = read(ROOT / "manifests/ehrsql_gold_queries.json")
    assert "CC BY 4.0" in gold["metadata"]["license"]
    assert len(gold["ehrsql_600"]) == 600
    assert len(gold["ehrsql_874"]) == 874

    fig5 = read(ROOT / "stats/fig5_exact_tau_prefix_v9.json")
    s11 = fig5["summary_11_tables"]
    keys = ["4", "8", "16", "32", "all"]
    assert [s11[k]["tau_hist_median_exact"]["median"] for k in keys] == [16, 21, 29, 38, 40]
    assert [s11[k]["tau_all_stats_pooled_exact"]["median"] for k in keys] == [15, 21, 29, 38, 49]
    assert [s11[k]["tokens_exact"]["median"] for k in keys] == [162, 330, 603, 1068, 1247]






