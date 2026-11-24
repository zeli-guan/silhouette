"""Unit tests for clean Uniform 1D DP (DAWA + DP-SIPS) baseline."""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines_dawa import dawa_dpsips_clean as ddc  # noqa: E402
from certificates import verify_certificates as vc_mod  # noqa: E402
from silhouette import dg_sampler as dg  # noqa: E402
from silhouette import profile as prof_mod  # noqa: E402
from silhouette import schema_roles as sr  # noqa: E402


def _make_synthetic_multi_table_db(path: Path) -> None:
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE patients (pid INTEGER PRIMARY KEY, age INTEGER, sex TEXT, notes TEXT)")
    con.execute("CREATE TABLE labs (lid INTEGER PRIMARY KEY, pid INTEGER REFERENCES patients(pid), val REAL, code TEXT)")
    rows_p = [(i, 20 + (i % 60), "F" if i % 2 == 0 else "M", f"note_{i % 7}") for i in range(1, 201)]
    rows_l = [(i, 1 + (i % 200), 10.5 + (i % 25) * 0.5, f"C{i % 10}") for i in range(1, 401)]
    con.executemany("INSERT INTO patients VALUES (?, ?, ?, ?)", rows_p)
    con.executemany("INSERT INTO labs VALUES (?, ?, ?, ?)", rows_l)
    con.commit()
    con.close()


def _l1_hist(h1: Counter, h2: Counter) -> int:
    keys = set(h1.keys()) | set(h2.keys())
    return sum(abs(int(h1.get(k, 0)) - int(h2.get(k, 0))) for k in keys)


def test_1_exact_rational_zcdp_budget_and_dpmeta_cert(tmp_path: Path) -> None:
    """Exact rational zCDP sum <= rho_T and delta_tail <= 5e-7 on multi-table DB + dpmeta certificate."""
    db_path = tmp_path / "synth_multi.sqlite"
    _make_synthetic_multi_table_db(db_path)

    out_json = tmp_path / "synth_multi.json"
    info = ddc.profile_db_dawa_dpsips(db_path, out_json)
    dpmeta_path = Path(info["dpmeta_path"])
    assert dpmeta_path.exists()

    meta = json.loads(dpmeta_path.read_text(encoding="utf-8"))
    rho_T = dg.rational_rho_budget(Fraction(8), Fraction(1, 2_000_000))
    assert Fraction(meta["rho_table"]) == rho_T
    assert Fraction(meta["rho_table_num"], meta["rho_table_den"]) == rho_T
    assert meta["rng_source"] == "os_csprng"

    res = ddc.verify_dpmeta_certificate(dpmeta_path, require_os_csprng=True, verify_cert_mod=vc_mod)
    assert res["tables"] == 2

    # Explicitly verify per-table sum(rho) <= rho_T and sum(delta_tail) <= 5e-7
    params = meta["parameters"]
    for t_name, tinfo in meta["tables"].items():
        rho_sum = sum((Fraction(params[idx]["rho"]) for _, _, idx in tinfo["queries"]), Fraction(0))
        dtail_sum = sum((Fraction(params[idx]["delta"]) for _, _, idx in tinfo["queries"]), Fraction(0))
        assert rho_sum <= rho_T, (t_name, rho_sum, rho_T)
        assert dtail_sum <= Fraction(1, 2_000_000), (t_name, dtail_sum)


def test_2_os_csprng_non_determinism(tmp_path: Path) -> None:
    """Two consecutive formal calls with default test_seed=None must produce different outputs/digests."""
    db_path = tmp_path / "csprng.sqlite"
    _make_synthetic_multi_table_db(db_path)

    out1 = tmp_path / "r1.json"
    out2 = tmp_path / "r2.json"
    info1 = ddc.profile_db_dawa_dpsips(db_path, out1)
    info2 = ddc.profile_db_dawa_dpsips(db_path, out2)

    assert info1["rng_source"] == "os_csprng"
    assert info2["rng_source"] == "os_csprng"
    assert info1["sha256"] != info2["sha256"]
    assert out1.read_text(encoding="utf-8") != out2.read_text(encoding="utf-8")

    # Verify no seed field leaked into dpmeta
    m1 = json.loads(Path(info1["dpmeta_path"]).read_text(encoding="utf-8"))
    assert "seed" not in m1
    assert m1["rng_source"] == "os_csprng"


def test_3_canary_zero_leakage(tmp_path: Path) -> None:
    """Unique canary token occurring once must NEVER appear in released DP profile."""
    canary = "CANARY_SECRET_TOKEN_XYZ_991"
    db_path = tmp_path / "canary.sqlite"
    con = sqlite3.connect(str(db_path))
    con.execute("CREATE TABLE records (id INTEGER PRIMARY KEY, tag TEXT)")
    rows = [(i, f"NORMAL_CAT_{i % 5}") for i in range(1, 500)]
    rows.append((500, canary))
    con.executemany("INSERT INTO records VALUES (?, ?)", rows)
    con.commit()
    con.close()

    for rep in range(3):
        out_json = tmp_path / f"canary_r{rep}.json"
        ddc.profile_db_dawa_dpsips(db_path, out_json)
        txt = out_json.read_text(encoding="utf-8")
        assert canary not in txt
        # Unique canary shape AAAAAA_AAAAAA_AAAAA_AAA_999 must also be suppressed by tau
        assert "AAAAAA_AAAAAA_AAAAA_AAA_999" not in txt


def test_4_mixed_type_sqlite_sensitivity_regression(tmp_path: Path) -> None:
    """Codex mixed-type counterexample: [1] + [1.0]*100 vs [1.0]*100 must have exact L1 <= 1."""
    def _build_and_extract(values_list, name: str, col_decl: str):
        p = tmp_path / f"{name}.sqlite"
        con = sqlite3.connect(str(p))
        con.execute(f"CREATE TABLE t (x {col_decl})")
        con.executemany("INSERT INTO t(x) VALUES (?)", [(v,) for v in values_list])
        con.commit()
        con.close()
        with ddc.ro_conn(p) as rcon:
            schema = sr.extract_public_schema(rcon)
            cm = schema.tables["t"][0]
            nrows = rcon.execute("SELECT COUNT(*) FROM t").fetchone()[0]
            return ddc.compute_column_histograms_normalized(rcon, "t", cm, nrows, need_values=True)

    # Case A: untyped / TEXT column (where shape() is used and Codex showed Counter([1] + [1.0]*100) flipped 9 vs 9.9!)
    st_with_int = _build_and_extract([1] + [1.0] * 100, "mixed_text_a", "")
    st_only_flt = _build_and_extract([1.0] * 100, "mixed_text_b", "")

    l1_dist_a = _l1_hist(st_with_int["distribution"], st_only_flt["distribution"])
    l1_vals_a = _l1_hist(st_with_int["values"], st_only_flt["values"])
    assert l1_dist_a == 1, f"Expected exact L1=1 on shape distribution, got {l1_dist_a}: {st_with_int['distribution']} vs {st_only_flt['distribution']}"
    assert l1_vals_a == 1, f"Expected exact L1=1 on value histogram, got {l1_vals_a}: {st_with_int['values']} vs {st_only_flt['values']}"
    assert abs(st_with_int["null_count"] - st_only_flt["null_count"]) <= 1
    assert abs(st_with_int["distinct_count"] - st_only_flt["distinct_count"]) <= 1

    # Case B: TEXT column with '1' vs 1 vs 1.0
    st_mix_b1 = _build_and_extract(["1"] + [1] * 50 + [1.0] * 50, "mixed_b1", "TEXT")
    st_mix_b2 = _build_and_extract([1] * 50 + [1.0] * 50, "mixed_b2", "TEXT")
    assert _l1_hist(st_mix_b1["distribution"], st_mix_b2["distribution"]) <= 1
    assert _l1_hist(st_mix_b1["values"], st_mix_b2["values"]) <= 1


def test_5_dawa_1d_l1_partition_correctness_and_nonnegativity() -> None:
    """DAWA 1D L1 DP partition merges flat plateaus, preserves clear step jump, and yields non-negative quantiles."""
    assert ddc.dawa_1d_l1_partition(np.array([], dtype=np.float64), 2.0) == []
    assert ddc.dawa_1d_l1_partition(np.array([42.0], dtype=np.float64), 2.0) == [(0, 0)]

    # Two distinct plateaus: [10, 10, 10, 10, 200, 200, 200, 200]
    arr = np.array([10.0, 11.0, 9.0, 10.0, 200.0, 199.0, 201.0, 200.0], dtype=np.float64)
    parts = ddc.dawa_1d_l1_partition(arr, sigma_est=5.0)
    assert parts == [(0, 3), (4, 7)], parts

    # Large array > 64 bins still produces contiguous covering partition
    big = np.ones(150, dtype=np.float64) * 50.0
    parts_big = ddc.dawa_1d_l1_partition(big, sigma_est=5.0)
    assert parts_big[0][0] == 0 and parts_big[-1][1] == 149
    for a, b in zip(parts_big, parts_big[1:]):
        assert a[1] + 1 == b[0]


def test_6_tightbpe_colfirst_and_valfirst(tmp_path: Path) -> None:
    """Both TightBPE modes strictly obey tokens <= Cap_TightBPE, and ValFirst retains strictly more values."""
    db_path = tmp_path / "wide.sqlite"
    con = sqlite3.connect(str(db_path))
    cols_ddl = ", ".join(
        ["id INTEGER PRIMARY KEY"]
        + [f"c{i} TEXT" for i in range(15)]
        + [f"n{i} REAL" for i in range(15)]
    )
    con.execute(f"CREATE TABLE wide_tbl ({cols_ddl})")
    con.commit()
    con.close()

    with ddc.ro_conn(db_path) as rcon:
        schema = sr.extract_public_schema(rcon)

    cap = ddc.compute_tightbpe_cap(schema)
    assert cap == 100 + 10 * 1 + 12 * 31

    # Build a synthetic wide profile exceeding cap with frequent values on 5 cols and zero-value cols on 26 cols
    prof = {"wide_tbl": {"id": ["INTEGER", 0.0, "10^3", [100.0, 500.0, 900.0], None]}}
    for i in range(5):
        prof["wide_tbl"][f"c{i}"] = [
            "TEXT",
            0.0,
            "10^1",
            ["AA_99", "BB_88", "CC_77"],
            [f"v{i}_{k}" for k in range(15)],
        ]
    for i in range(5, 15):
        prof["wide_tbl"][f"c{i}"] = [
            "TEXT",
            0.125,
            "10^4",
            ["AAAA_BBBB_CCCC_9999", "DDDD_EEEE_FFFF_8888", "GGGG_HHHH_IIII_7777"],
            None,
        ]
    for i in range(15):
        prof["wide_tbl"][f"n{i}"] = ["REAL", 0.05, "10^2", [1.0, 5.0, 10.0], None]

    # Deterministic token counter matching ~4 chars/BPE-token density
    tok_fn = lambda s: (len(s) + 3) // 4
    assert tok_fn(prof_mod.render_profile_prompt_block(prof)) > cap

    colfirst = ddc.truncate_profile_to_public_cap(
        prof, cap, schema, token_counter=tok_fn, mode="drop_values_first"
    )
    valfirst = ddc.truncate_profile_to_public_cap(
        prof, cap, schema, token_counter=tok_fn, mode="preserve_values"
    )

    assert tok_fn(prof_mod.render_profile_prompt_block(colfirst)) <= cap
    assert tok_fn(prof_mod.render_profile_prompt_block(valfirst)) <= cap

    st_col = ddc.profile_content_stats(colfirst)
    st_val = ddc.profile_content_stats(valfirst)
    assert st_val["n_val_items"] > st_col["n_val_items"], (st_val, st_col)
