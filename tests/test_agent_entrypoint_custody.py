"""Fail-closed custody, non-DP rejection, dictionary pinning, and A9 receiver isolation tests for generate_sql."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from silhouette.agent import (
    SingleReleaseReceiverLedger,
    canonical_profile_sha256,
    file_sha256,
    generate_sql,
)
from silhouette.sandbox import PUBLIC_TABLES, PublicLookup, SchemaSandbox


def _make_sandbox():
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE t(x TEXT)")
    con.commit()
    return con, SchemaSandbox(con)


def test_generate_sql_defaults_fail_closed_without_pinned_digest():
    con, box = _make_sandbox()
    prof = {"t": {"x": ["TEXT", 0.0, "10^0", ["AAAA"], ["alpha"]]}}
    model_calls = 0

    def model(_req):
        nonlocal model_calls
        model_calls += 1
        return {"name": "submit_sql", "arguments": {"sql": "SELECT x FROM t"}}

    with pytest.raises(ValueError, match="expected_profile_sha256 is required"):
        generate_sql(model, "List x", "t(x TEXT)", prof, sandbox=box)
    assert model_calls == 0

    with pytest.raises(ValueError, match="digest mismatch"):
        generate_sql(
            model,
            "List x",
            "t(x TEXT)",
            prof,
            sandbox=box,
            expected_profile_sha256="0" * 64,
        )
    assert model_calls == 0

    box.connection.close()
    con.close()


def test_generate_sql_rejects_deployment_prohibited_even_with_exact_sha(tmp_path):
    con, box = _make_sandbox()
    model_calls = 0

    def model(_req):
        nonlocal model_calls
        model_calls += 1
        return {"name": "submit_sql", "arguments": {"sql": "SELECT x FROM t"}}

    # 1. Synthetic dict with deployment_prohibited=True
    prohibited_dict = {"deployment_prohibited": True, "t": {"x": ["TEXT", 0.0, "10^0", ["AAAA"], ["alpha"]]}}
    prohibited_sha = canonical_profile_sha256(prohibited_dict)
    with pytest.raises(ValueError, match="deployment_prohibited"):
        generate_sql(
            model,
            "List x",
            "t(x TEXT)",
            prohibited_dict,
            sandbox=box,
            expected_profile_sha256=prohibited_sha,
        )
    assert model_calls == 0

    # Even with require_pinned_digest=False, deployment_prohibited is still rejected unless allow_non_dp=True
    with pytest.raises(ValueError, match="deployment_prohibited"):
        generate_sql(
            model,
            "List x",
            "t(x TEXT)",
            prohibited_dict,
            sandbox=box,
            require_pinned_digest=False,
        )
    assert model_calls == 0

    # Opt-in allow_non_dp=True succeeds
    sql = generate_sql(
        model,
        "List x",
        "t(x TEXT)",
        prohibited_dict,
        sandbox=box,
        expected_profile_sha256=prohibited_sha,
        allow_non_dp=True,
    )
    assert sql == "SELECT x FROM t"
    assert model_calls == 1

    # 2. Real packaged noiseless profiles fail closed unless allow_non_dp=True
    for rel_path in (
        ROOT / "profiles" / "bird_noiseless_clean" / "release_01" / "financial.json",
        ROOT / "profiles" / "ehrsql_noiseless_nolkp" / "release_01.json",
    ):
        model_calls = 0
        sha = file_sha256(rel_path)
        with pytest.raises(ValueError, match="deployment_prohibited|noiseless"):
            generate_sql(
                model,
                "List x",
                "t(x TEXT)",
                rel_path,
                sandbox=box,
                expected_profile_sha256=sha,
            )
        assert model_calls == 0

        ok_sql = generate_sql(
            model,
            "List x",
            "t(x TEXT)",
            rel_path,
            sandbox=box,
            expected_profile_sha256=sha,
            allow_non_dp=True,
        )
        assert ok_sql == "SELECT x FROM t"

    # 3. Clean Path profile succeeds by default when pinned SHA-256 matches
    clean_path = tmp_path / "clean_dp.json"
    clean_prof = {"t": {"x": ["TEXT", 0.0, "10^0", ["AAAA"], ["alpha"]]}}
    clean_path.write_text(json.dumps(clean_prof), encoding="utf-8")
    clean_sha = file_sha256(clean_path)
    assert generate_sql(
        model,
        "List x",
        "t(x TEXT)",
        clean_path,
        sandbox=box,
        expected_profile_sha256=clean_sha,
    ) == "SELECT x FROM t"

    box.connection.close()
    con.close()


def test_single_release_receiver_ledger_a9_enforcement(tmp_path):
    con, box = _make_sandbox()
    r1 = {"t": {"x": ["TEXT", 0.0, "10^0", ["AAAA"], ["alpha"]]}}
    r2 = {"t": {"x": ["TEXT", 0.0, "10^0", ["AAAA"], ["beta"]]}}
    s1 = canonical_profile_sha256(r1)
    s2 = canonical_profile_sha256(r2)

    ledger_file = tmp_path / "a9_ledger.json"
    ledger = SingleReleaseReceiverLedger(ledger_file)

    calls = 0

    def model(_req):
        nonlocal calls
        calls += 1
        return {"name": "submit_sql", "arguments": {"sql": "SELECT x FROM t"}}

    # Unbounded reuse on same release R1 for receiver w1 is allowed under A9
    for q in ("First question", "Second question", "Third question"):
        assert generate_sql(
            model,
            q,
            "t(x TEXT)",
            r1,
            sandbox=box,
            expected_profile_sha256=s1,
            receiver_id="w1",
            receiver_ledger=ledger,
        ) == "SELECT x FROM t"
    assert calls == 3

    # Reloaded ledger from disk preserves A9 binding and rejects cross-release R2 for w1 before calling model
    reloaded = SingleReleaseReceiverLedger(ledger_file)
    assert reloaded.get_binding("w1") == s1
    with pytest.raises(ValueError, match="A9"):
        generate_sql(
            model,
            "Cross-release probe",
            "t(x TEXT)",
            r2,
            sandbox=box,
            expected_profile_sha256=s2,
            receiver_id="w1",
            receiver_ledger=reloaded,
        )
    assert calls == 3

    # Missing receiver_id fails closed
    with pytest.raises(ValueError, match="receiver_id"):
        generate_sql(
            model,
            "Missing receiver",
            "t(x TEXT)",
            r1,
            sandbox=box,
            expected_profile_sha256=s1,
            receiver_ledger=reloaded,
        )
    assert calls == 3

    # Distinct receiver w2 can bind to R2 independently
    assert generate_sql(
        model,
        "Independent receiver",
        "t(x TEXT)",
        r2,
        sandbox=box,
        expected_profile_sha256=s2,
        receiver_id="w2",
        receiver_ledger=reloaded,
    ) == "SELECT x FROM t"
    assert calls == 4

    box.connection.close()
    con.close()


def test_generate_sql_public_lookup_digest_pinning(tmp_path):
    con, box = _make_sandbox()
    dict_path = tmp_path / "dict.sqlite"
    cd = sqlite3.connect(str(dict_path))
    for tbl in PUBLIC_TABLES:
        cd.execute(f'CREATE TABLE "{tbl}" (x TEXT)')
    cd.execute("INSERT INTO d_items VALUES ('ok')")
    cd.commit()
    cd.close()
    dict_sha = file_sha256(dict_path)

    prof = {"t": {"x": ["TEXT", 0.0, "10^0", ["AAAA"], ["alpha"]]}}
    prof_sha = canonical_profile_sha256(prof)
    calls = 0

    def model(_req):
        nonlocal calls
        calls += 1
        return {"name": "submit_sql", "arguments": {"sql": "SELECT x FROM t"}}

    unpinned_lkp = PublicLookup(dict_path, allow_unpinned=True)
    try:
        with pytest.raises(ValueError, match="PublicLookup must be pinned"):
            generate_sql(
                model,
                "List x",
                "t(x TEXT)",
                prof,
                sandbox=box,
                lookup=unpinned_lkp,
                expected_profile_sha256=prof_sha,
            )
        assert calls == 0

        with pytest.raises(ValueError, match="dictionary SHA-256 mismatch"):
            generate_sql(
                model,
                "List x",
                "t(x TEXT)",
                prof,
                sandbox=box,
                lookup=unpinned_lkp,
                expected_profile_sha256=prof_sha,
                expected_dictionary_sha256="f" * 64,
            )
        assert calls == 0

        assert generate_sql(
            model,
            "List x",
            "t(x TEXT)",
            prof,
            sandbox=box,
            lookup=unpinned_lkp,
            expected_profile_sha256=prof_sha,
            expected_dictionary_sha256=dict_sha,
        ) == "SELECT x FROM t"
        assert calls == 1
    finally:
        unpinned_lkp.connection.close()

    pinned_lkp = PublicLookup(dict_path, expected_sha256=dict_sha)
    try:
        assert generate_sql(
            model,
            "List x",
            "t(x TEXT)",
            prof,
            sandbox=box,
            lookup=pinned_lkp,
            expected_profile_sha256=prof_sha,
        ) == "SELECT x FROM t"
        assert calls == 2
    finally:
        pinned_lkp.connection.close()
        box.connection.close()
        con.close()


def test_verify_1way_synth_factorial_packaged():
    import importlib.util

    script_path = ROOT / "scripts" / "verify_1way_synth_factorial.py"
    spec = importlib.util.spec_from_file_location("verify_1way_synth_factorial", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    res = mod.verify_packaged(ROOT)
    assert res["ind8_profiles_verified"] == 33
    assert res["pubcap_ind8_profile_exact_sha_matched"] == 33
    assert res["pubcap_ind8_rows_exact_sha_matched"] == 33
    assert res["deterministic_1way_synth_roundtrip_verified"] == 33
    assert res["tightbpe_1way_files_verified"] == 66


def test_schema_sandbox_preserves_without_rowid(tmp_path):
    from silhouette.sandbox import LocalExecutor

    db_path = tmp_path / "without_rowid.sqlite"
    con = sqlite3.connect(str(db_path))
    con.executescript(
        """
        CREATE TABLE ordinary_tbl (
            id TEXT PRIMARY KEY,
            note TEXT DEFAULT 'WITHOUT ROWID' -- WITHOUT ROWID
        );
        CREATE TABLE "quoted(paren)" (
            k TEXT PRIMARY KEY,
            "WITHOUT ROWID" INTEGER
        ) /* WITHOUT ROWID */;
        CREATE TABLE kv_wor (
            k TEXT PRIMARY KEY,
            v INTEGER
        ) WITHOUT ROWID;
        CREATE TABLE kv_wor_comment (
            k TEXT PRIMARY KEY,
            v INTEGER CHECK (length(k) > 0)
        ) WITHOUT /* inline */ ROWID;
        """
    )
    con.commit()

    box = SchemaSandbox(con)
    exe = LocalExecutor(db_path)
    try:
        # Ordinary tables keep rowid in both SchemaSandbox and LocalExecutor
        for tbl in ("ordinary_tbl", "quoted(paren)"):
            qname = '"' + tbl.replace('"', '""') + '"'
            assert box.try_sql(f"SELECT rowid FROM {qname}") == {"status": "success"}
            assert exe.execute(f"SELECT rowid FROM {qname}") == []

        # WITHOUT ROWID tables reject SELECT rowid identically in both SchemaSandbox and LocalExecutor
        for tbl in ("kv_wor", "kv_wor_comment"):
            res = box.try_sql(f"SELECT rowid FROM {tbl}")
            assert res["status"] == "error" and res["category"] == "unknown_column", (tbl, res)
            with pytest.raises(sqlite3.OperationalError, match="no such column: rowid"):
                exe.execute(f"SELECT rowid FROM {tbl}")
            # Normal primary-key query still succeeds
            assert box.try_sql(f"SELECT k FROM {tbl}") == {"status": "success"}
            assert exe.execute(f"SELECT k FROM {tbl}") == []
    finally:
        box.connection.close()
        con.close()


def test_jsonl_io_transparent_gz_fallback(tmp_path):
    import gzip
    from silhouette.jsonl_io import iter_jsonl, load_jsonl, resolve_jsonl_path

    plain = tmp_path / "sample.jsonl"
    gz = tmp_path / "sample.jsonl.gz"
    payload = '{"a": 1}\n{"b": 2}\n'
    with gzip.GzipFile(filename="", mode="wb", compresslevel=9, fileobj=gz.open("wb"), mtime=0) as fh:
        fh.write(payload.encode("utf-8"))

    # When only .jsonl.gz exists, passing .jsonl resolves automatically to .jsonl.gz
    assert resolve_jsonl_path(plain) == gz
    assert load_jsonl(plain) == [{"a": 1}, {"b": 2}]
    assert list(iter_jsonl(gz)) == [{"a": 1}, {"b": 2}]

    # When .jsonl also exists, plain takes precedence
    plain.write_text('{"a": 99}\n', encoding="utf-8")
    assert resolve_jsonl_path(plain) == plain
    assert load_jsonl(plain) == [{"a": 99}]
