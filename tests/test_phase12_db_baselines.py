"""Automated privacy, sensitivity, zCDP budget, FK integrity, and zero-leakage tests."""
from __future__ import annotations

from fractions import Fraction
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "baselines_phase12" / "src"
for _p in (str(SRC), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dp_core
import baseline_aim
import baseline_privlava
import baseline_privpetal
import baseline_dawa_dpsips
from silhouette import dg_sampler as dg


def _make_canary_relational_db(db_path: Path, canary: str = "CANARY_SECRET_TOKEN_XYZ_991") -> None:
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(str(db_path))
    con.execute("CREATE TABLE parents (pid INTEGER PRIMARY KEY, region TEXT, score REAL)")
    con.execute(
        "CREATE TABLE children (cid INTEGER PRIMARY KEY, pid INTEGER, category TEXT, amount REAL, "
        "FOREIGN KEY(pid) REFERENCES parents(pid))"
    )
    prows = []
    for i in range(1, 201):
        reg = "NORTH" if i <= 100 else "SOUTH"
        prows.append((i, reg, 10.0 + (i % 5)))
    # Inject 1 unique canary parent and 1 unique canary child
    prows.append((201, canary, 999999999.0))
    con.executemany("INSERT INTO parents VALUES (?, ?, ?)", prows)

    crows = []
    cid = 1
    for pid in range(1, 201):
        for k in range(3):
            crows.append((cid, pid, f"CAT_{k % 2}", 5.0 + k))
            cid += 1
    crows.append((cid, 201, canary, 999999999.0))
    con.executemany("INSERT INTO children VALUES (?, ?, ?, ?)", crows)
    con.commit()
    con.close()


def test_exact_zcdp_budget_constants():
    eps_up = dg.conservative_eps_from_rho(dp_core.RHO_TABLE_8, dp_core.TABLE_DELTA_HALF)
    assert Fraction(eps_up) <= Fraction(8)
    assert float(dp_core.RHO_TABLE_8) <= 0.8748

    # Check sensitivity=C=5 tau shift: Pr[5 + Z > tau_5] <= delta_j
    s2 = dp_core.sigma2_for_zcdp(Fraction(1, 10), l2_sens=5)
    assert s2 == Fraction(25, 2 * Fraction(1, 10))
    tau1, _ = dg.exact_dgauss_tau(s2, Fraction(1, 10**7))
    tau5 = dp_core.exact_tau_for_sensitivity(s2, Fraction(1, 10**7), l1_max_step=5)
    assert tau5 == tau1 + 4


def test_canary_zero_leakage_and_budget_aim(tmp_path: Path):
    canary = "CANARY_SECRET_TOKEN_XYZ_991"
    src = tmp_path / "canary.sqlite"
    _make_canary_relational_db(src, canary=canary)
    out = tmp_path / "aim.sqlite"
    rep = baseline_aim.synthesize_db_aim(src, out, seed=20261123)
    for t, trep in rep["tables"].items():
        assert trep["rho_used"] <= float(dp_core.RHO_TABLE_8) + 1e-12
        assert float(trep["epsilon_upper"]) <= 8.0

    con = sqlite3.connect(str(out))
    for t in ("parents", "children"):
        dump = repr(con.execute(f"SELECT * FROM {t}").fetchall())
        assert canary not in dump, f"Canary leaked in AIM table {t}!"
    con.close()


def test_canary_zero_leakage_and_fk_integrity_privlava_and_privpetal(tmp_path: Path):
    canary = "CANARY_SECRET_TOKEN_XYZ_991"
    src = tmp_path / "canary.sqlite"
    _make_canary_relational_db(src, canary=canary)

    for name, fn in [("PrivLava", baseline_privlava.synthesize_db_privlava),
                     ("PrivPetal", baseline_privpetal.synthesize_db_privpetal)]:
        out = tmp_path / f"{name}.sqlite"
        rep = fn(src, out, seed=20261123)
        assert rep["fk_violations"] == 0
        for t, trep in rep["tables"].items():
            assert trep["rho_used"] <= float(dp_core.RHO_TABLE_8) + 1e-12
        con = sqlite3.connect(str(out))
        assert con.execute("PRAGMA foreign_key_check").fetchall() == []
        for t in ("parents", "children"):
            rows = con.execute(f"SELECT * FROM {t}").fetchall()
            assert canary not in repr(rows), f"Canary leaked in {name} table {t}!"
        # Verify 100% non-dangling FKs
        dangling = con.execute(
            "SELECT COUNT(*) FROM children c LEFT JOIN parents p ON c.pid = p.pid WHERE c.pid IS NOT NULL AND p.pid IS NULL"
        ).fetchone()[0]
        assert dangling == 0
        con.close()


def test_canary_zero_leakage_dawa_dpsips(tmp_path: Path):
    canary = "CANARY_SECRET_TOKEN_XYZ_991"
    src = tmp_path / "canary.sqlite"
    _make_canary_relational_db(src, canary=canary)
    out_json = tmp_path / "dawa_dpsips.json"
    rep = baseline_dawa_dpsips.profile_db_dawa_dpsips(src, out_json, seed=20261123)
    raw_txt = out_json.read_text(encoding="utf-8")
    assert canary not in raw_txt
    for t, led in rep["tables"].items():
        assert led["rho_used"] <= float(dp_core.RHO_TABLE_8) + 1e-12
        assert float(led["epsilon_upper"]) <= 8.0
        assert led["delta_tail_used"] <= float(dp_core.TABLE_DELTA_HALF) + 1e-15

    out_tb = tmp_path / "dawa_dpsips_tightbpe.json"
    tb_info = baseline_dawa_dpsips.truncate_profile_tightbpe(
        out_json, src, out_tb, bpe_len_fn=lambda s: max(1, len(s) // 4)
    )
    assert tb_info["final_bpe"] <= tb_info["cap_tightbpe"]
    assert canary not in out_tb.read_text(encoding="utf-8")


def _make_synthetic_thrombosis_db(db_path: Path, canary: str = "CANARY_SECRET_TOKEN_XYZ_991") -> None:
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(str(db_path))
    con.execute("CREATE TABLE Patient (ID INTEGER PRIMARY KEY, SEX TEXT, Diagnosis TEXT)")
    con.execute(
        "CREATE TABLE Examination (ID INTEGER, ExamDate TEXT, Thrombosis INTEGER, ANA TEXT, "
        "FOREIGN KEY(ID) REFERENCES Patient(ID))"
    )
    con.execute(
        "CREATE TABLE Laboratory (ID INTEGER, Date TEXT, GPT REAL, GOT REAL, "
        "FOREIGN KEY(ID) REFERENCES Patient(ID))"
    )
    patients = [(i, "F" if i % 2 == 0 else "M", "SLE" if i % 3 == 0 else "RA") for i in range(1, 181)]
    # Unique canary patient with >5 child rows to exercise entity_cap=5 capping
    patients.append((999, canary, canary))
    con.executemany("INSERT INTO Patient VALUES (?, ?, ?)", patients)

    exams = []
    labs = []
    for pid in range(1, 181):
        for k in range(7):
            exams.append((pid, f"2020-01-{10 + k:02d}", 1 if k % 2 == 0 else 0, "POS" if pid % 2 == 0 else "NEG"))
            labs.append((pid, f"2020-02-{10 + k:02d}", 20.0 + k, 25.0 + k))
    for k in range(8):
        exams.append((999, f"2021-01-{10 + k:02d}", 2, canary))
        labs.append((999, f"2021-02-{10 + k:02d}", 999999.0, 999999.0))
    con.executemany("INSERT INTO Examination VALUES (?, ?, ?, ?)", exams)
    con.executemany("INSERT INTO Laboratory VALUES (?, ?, ?, ?)", labs)
    con.commit()
    con.close()


def test_entity_privlava_on_thrombosis(tmp_path: Path):
    canary = "CANARY_SECRET_TOKEN_XYZ_991"
    src = tmp_path / "thrombosis_prediction.sqlite"
    _make_synthetic_thrombosis_db(src, canary=canary)
    out = tmp_path / "thrombosis_ent.sqlite"
    rep = baseline_privlava.synthesize_db_privlava(src, out, seed=101, entity_mode=True, entity_cap=5)
    assert rep["fk_violations"] == 0
    total_rho = sum(t["rho_used"] for t in rep["tables"].values())
    assert total_rho <= float(dp_core.RHO_TABLE_8) + 1e-9
    eps_up = dg.conservative_eps_from_rho(dp_core.RHO_TABLE_8, dp_core.TABLE_DELTA_HALF)
    assert float(eps_up) <= 8.0

    con = sqlite3.connect(str(out))
    assert con.execute("PRAGMA foreign_key_check").fetchall() == []
    for tbl in ("Patient", "Examination", "Laboratory"):
        rows = con.execute(f"SELECT * FROM {tbl}").fetchall()
        assert canary not in repr(rows), f"Canary leaked in Entity-PrivLava table {tbl}!"
    for child in ("Examination", "Laboratory"):
        dangling = con.execute(
            f"SELECT COUNT(*) FROM {child} c LEFT JOIN Patient p ON c.ID = p.ID WHERE c.ID IS NOT NULL AND p.ID IS NULL"
        ).fetchone()[0]
        assert dangling == 0
    con.close()


def test_dawa_1d_l1_partition_postprocessing():
    arr = np.array([10.0, 10.0, 10.0, 100.0, 100.0, 100.0])
    parts = baseline_dawa_dpsips.dawa_1d_l1_partition(arr, sigma_est=1.0)
    assert len(parts) == 2
    assert parts == [(0, 2), (3, 5)]
