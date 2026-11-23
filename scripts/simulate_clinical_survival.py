#!/usr/bin/env python3
"""Stage 8 portable clinical survival & scaling simulator (zero external repo imports).

Tier-1 (`--self-test`):
  Reproduces the full Whole-Patient Bootstrap AND Marginal Binomial 3-stage clinical
  survival (M=200 seeded cohorts across N in {94, 1k, 10k, 100k}, 192 cells total)
  directly from the bundled de-identified patient-bin matrix
  `stats/mimic_demo94_patient_bins_deid.json`, `manifests/mimic_iv_public_ddl.sql`,
  and `silhouette.dg_sampler.sample_dgauss`, asserting exact parity with
  `stats/clinical_survival_3stage_verified.json`.

Tier-2 (`--bird-db-dir ...` / `--mimic-db ...`):
  Optional local regeneration against locally supplied SQLite databases using
  only `silhouette.*` and the Python standard library + NumPy.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import concurrent.futures
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import re
import sqlite3
import statistics
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from silhouette import dg_sampler as dg
from silhouette import patient_dp as pdp
from silhouette import profile as prof_mod
from silhouette import schema_roles as sr


class SHA256IntegerRNG:
    """Deterministic counter-mode SHA-256 integer RNG implementing `dg.IntegerRNG`."""

    def __init__(self, seed_bytes: bytes) -> None:
        self._seed = seed_bytes
        self._counter = 0
        self._buf = b""
        self._pos = 0

    def _fill(self, need_bytes: int) -> None:
        chunks = [self._buf[self._pos :]]
        cur_len = len(chunks[0])
        while cur_len < need_bytes:
            h = hashlib.sha256(self._seed + self._counter.to_bytes(8, "big")).digest()
            self._counter += 1
            chunks.append(h)
            cur_len += len(h)
        self._buf = b"".join(chunks)
        self._pos = 0

    def _getrandbits(self, k: int) -> int:
        if k <= 0:
            return 0
        nbytes = (k + 7) >> 3
        if self._pos + nbytes > len(self._buf):
            self._fill(max(4096, nbytes))
        raw = self._buf[self._pos : self._pos + nbytes]
        self._pos += nbytes
        val = int.from_bytes(raw, "big")
        excess = (nbytes << 3) - k
        return val >> excess if excess else val

    def randrange(self, stop: int) -> int:
        k = stop.bit_length()
        while True:
            r = self._getrandbits(k)
            if r < stop:
                return r


def _sha256_seed_int(*parts: Any) -> int:
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _extract_public_schema_from_ddl(ddl_path: Path) -> sr.PublicSchema:
    con = sqlite3.connect(":memory:")
    con.executescript(ddl_path.read_text(encoding="utf-8"))
    try:
        return sr.extract_public_schema(con)
    finally:
        con.close()


def _summarize_survival_rates(arr: List[float]) -> Dict[str, Any]:
    mean_v = 100.0 * float(np.mean(arr))
    lo_v, hi_v = np.percentile(arr, [2.5, 97.5])
    lo_p = 100.0 * float(lo_v)
    hi_p = 100.0 * float(hi_v)
    return {
        "mean_pct": round(mean_v, 2),
        "ci95_pct": [round(lo_p, 2), round(hi_p, 2)],
        "mean_pct_exact": round(mean_v, 6),
        "ci95_pct_exact": [round(lo_p, 6), round(hi_p, 6)],
    }


def _eval_survival_task(
    args: Tuple[str, str, int, str, str, int, int, str, str],
) -> Tuple[str, int, Dict[str, Any]]:
    mode, sname, eps, sched, _label, N, m_cohorts, deid_path_s, ddl_path_s = args
    deid_doc = json.loads(Path(deid_path_s).read_text(encoding="utf-8"))
    pub = _extract_public_schema_from_ddl(Path(ddl_path_s))

    cols_meta = deid_doc["columns"]
    n_patients = int(deid_doc["n_patients"])
    n_reachable = int(deid_doc["c1_reachable_cap5"])
    c1_total = int(deid_doc["c1_total"])

    plan = pdp.plan_patient(pub, eps, schedule=sched)
    val_qs = {(q["table"], q["column"]): q for q in plan["queries"] if q["stat"] == "values"}
    dc_qs = {(q["table"], q["column"]): q for q in plan["queries"] if q["stat"] == "distinct_count"}

    col_specs: List[Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, int, Fraction, Fraction]] = []
    for cm in cols_meta:
        c_idx = cm["col_idx"]
        nb = int(cm["n_bins"])
        mat = np.zeros((n_patients, nb), dtype=np.int64)
        for p_rec in deid_doc["patient_matrix"]:
            p_i = int(p_rec["patient_idx"])
            for b_i in p_rec["capped_bins_by_col"].get(str(c_idx), ()):
                mat[p_i, int(b_i)] = 1
        key = (cm["table"], cm["column"])
        p_vec = (np.sum(mat, axis=0) / float(n_patients)) if mode == "marginal_binomial" else None
        col_specs.append((
            mat if mode == "whole_patient" else None,
            p_vec,
            np.array(cm["is_c1"], dtype=bool),
            int(val_qs[key]["tau"]),
            val_qs[key]["sigma2"],
            dc_qs[key]["sigma2"],
        ))

    probs = np.full(n_patients, 1.0 / n_patients, dtype=np.float64)
    seed_prefix = "stage8_whole_patient_v8" if mode == "whole_patient" else "stage8_marginal_binomial_v8"
    s1_r: List[float] = []
    s2_r: List[float] = []
    s3_r: List[float] = []
    s1_a: List[float] = []
    s2_a: List[float] = []
    s3_a: List[float] = []

    for m in range(m_cohorts):
        seed_str = f"{seed_prefix}:{sname}:{N}:{m}"
        digest = hashlib.sha256(seed_str.encode("utf-8")).digest()
        np_rng = np.random.default_rng(int.from_bytes(digest, "big"))
        dg_rng = SHA256IntegerRNG(digest + b":cks_dg")
        p_counts = np_rng.multinomial(N, probs) if mode == "whole_patient" else None

        tot_s1 = tot_s2 = tot_s3 = 0
        for mat, p_vec, is_c1, tau, v_s2, d_s2 in col_specs:
            if mode == "whole_patient":
                assert mat is not None and p_counts is not None
                if mat.shape[1] == 0:
                    continue
                x_b = p_counts @ mat
            else:
                assert p_vec is not None
                if p_vec.shape[0] == 0:
                    continue
                x_b = np_rng.binomial(N, p_vec)
            occ_idx = np.flatnonzero(x_b > 0)
            d_occ = len(occ_idx)
            if d_occ == 0:
                continue

            noisy_occ = np.array(
                [int(x_b[b]) + dg.sample_dgauss(v_s2, dg_rng) for b in occ_idx],
                dtype=np.int64,
            )
            pass_thr = noisy_occ > tau
            occ_c1 = is_c1[occ_idx]
            s1_hits = int(np.sum(pass_thr & occ_c1))
            tot_s1 += s1_hits

            d_hat = max(0, d_occ + dg.sample_dgauss(d_s2, dg_rng))
            if d_hat <= 50:
                tot_s2 += s1_hits
                surv_pos = np.flatnonzero(pass_thr)
                if len(surv_pos) > 0:
                    if len(surv_pos) > 15:
                        surv_scores = noisy_occ[surv_pos]
                        order = np.argsort(-surv_scores, kind="mergesort")[:15]
                        kept_pos = surv_pos[order]
                    else:
                        kept_pos = surv_pos
                    tot_s3 += int(np.sum(occ_c1[kept_pos]))

        assert tot_s3 <= tot_s2 <= tot_s1
        s1_r.append(tot_s1 / float(n_reachable))
        s2_r.append(tot_s2 / float(n_reachable))
        s3_r.append(tot_s3 / float(n_reachable))
        s1_a.append(tot_s1 / float(c1_total))
        s2_a.append(tot_s2 / float(c1_total))
        s3_a.append(tot_s3 / float(c1_total))

    return sname, N, {
        "stage1_occupied_gt_tau": {
            "reachable_120": _summarize_survival_rates(s1_r),
            "all_125": _summarize_survival_rates(s1_a),
        },
        "stage2_dhat_le_50_gate": {
            "reachable_120": _summarize_survival_rates(s2_r),
            "all_125": _summarize_survival_rates(s2_a),
        },
        "stage3_eq10_top15_published": {
            "reachable_120": _summarize_survival_rates(s3_r),
            "all_125": _summarize_survival_rates(s3_a),
        },
    }


def _run_survival_from_deid(
    mode: str,
    deid_path: Path = ROOT / "stats/mimic_demo94_patient_bins_deid.json",
    ddl_path: Path = ROOT / "manifests/mimic_iv_public_ddl.sql",
    *,
    m_cohorts: int = 200,
    max_workers: Optional[int] = None,
) -> Dict[str, Any]:
    import os

    if mode not in ("whole_patient", "marginal_binomial"):
        raise ValueError(f"Unsupported mode: {mode!r}")

    deid_bytes = deid_path.read_bytes()
    deid_sha = hashlib.sha256(deid_bytes).hexdigest()
    deid_doc = json.loads(deid_bytes.decode("utf-8"))
    n_patients = int(deid_doc["n_patients"])
    n_reachable = int(deid_doc["c1_reachable_cap5"])
    c1_total = int(deid_doc["c1_total"])
    assert n_patients == 94 and c1_total == 125 and n_reachable == 120

    schedules = [
        ("db_global_eps8", 8, "db_global", "DB-wide convex (eps=8)"),
        ("db_global_eps1", 1, "db_global", "DB-wide convex (eps=1)"),
        ("db_uniform_eps8", 8, "db_uniform", "DB-wide uniform (eps=8)"),
        ("per_table_eps8", 8, "per_table", "Per-table (eps_T=8)"),
    ]
    cohorts = (94, 1000, 10000, 100000)
    tasks = [
        (mode, sname, eps, sched, label, N, m_cohorts, str(deid_path), str(ddl_path))
        for sname, eps, sched, label in schedules
        for N in cohorts
    ]

    out_schedules: Dict[str, Any] = {
        sname: {"label": label, "epsilon": eps, "schedule": sched, "cohorts": {}}
        for sname, eps, sched, label in schedules
    }
    workers = max_workers if max_workers is not None else min(16, os.cpu_count() or 4)
    if workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            for sname, N, res in ex.map(_eval_survival_task, tasks):
                out_schedules[sname]["cohorts"][str(N)] = res
    else:
        for t in tasks:
            sname, N, res = _eval_survival_task(t)
            out_schedules[sname]["cohorts"][str(N)] = res

    is_wp = mode == "whole_patient"
    return {
        "schema_version": "clinical-survival-whole-patient-v8" if is_wp else "clinical-survival-marginal-binomial-v8",
        "resampling_method": "whole_patient_bootstrap_with_replacement" if is_wp else "marginal_per_bin_binomial",
        "noise_sampler": "exact_cks_integer_dgauss (silhouette.dg_sampler.sample_dgauss)",
        "seed_rule": (
            "sha256('stage8_whole_patient_v8:{schedule}:{N}:{m}')"
            if is_wp
            else "sha256('stage8_marginal_binomial_v8:{schedule}:{N}:{m}')"
        ),
        "deid_fixture_sha256": deid_sha,
        "m_cohorts": m_cohorts,
        "n_patients_source": 94,
        "cap": 5,
        "n_tables": int(deid_doc["n_tables"]),
        "n_stats": 346,
        "n_histograms": 139,
        "c1_total": c1_total,
        "c1_reachable_cap5": n_reachable,
        "schedules": out_schedules,
    }


def run_whole_patient_bootstrap_from_deid(
    deid_path: Path = ROOT / "stats/mimic_demo94_patient_bins_deid.json",
    ddl_path: Path = ROOT / "manifests/mimic_iv_public_ddl.sql",
    *,
    m_cohorts: int = 200,
    max_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """Reproduce Whole-Patient Bootstrap 3-stage survival directly from de-identified patient bins."""
    return _run_survival_from_deid(
        "whole_patient", deid_path, ddl_path, m_cohorts=m_cohorts, max_workers=max_workers
    )


def run_marginal_binomial_from_deid(
    deid_path: Path = ROOT / "stats/mimic_demo94_patient_bins_deid.json",
    ddl_path: Path = ROOT / "manifests/mimic_iv_public_ddl.sql",
    *,
    m_cohorts: int = 200,
    max_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """Reproduce Marginal-Binomial 3-stage survival directly from de-identified patient bins."""
    return _run_survival_from_deid(
        "marginal_binomial", deid_path, ddl_path, m_cohorts=m_cohorts, max_workers=max_workers
    )


def _iter_profile_leaves(obj: Any) -> Iterable[str]:
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_profile_leaves(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_profile_leaves(v)
    elif obj is not None:
        yield str(obj)


_SQL_STR_RE = re.compile(r"'((?:[^']|'')*)'")
_SQL_NUM_RE = re.compile(r"(?<![A-Za-z0-9_])([+-]?\d+(?:\.\d+)?)(?![A-Za-z0-9_])")


def _extract_sql_literals(sql: str) -> Set[str]:
    lits: Set[str] = set()
    for m in _SQL_STR_RE.finditer(sql):
        val = m.group(1).replace("''", "'")
        if val:
            lits.add(val)
    stripped = _SQL_STR_RE.sub(" ", sql)
    for m in _SQL_NUM_RE.finditer(stripped):
        lits.add(m.group(1))
    return lits


def _norm_token(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip()).casefold()


def compute_unified_fig4b_fig4c(
    bird_db_dir: Path,
    c1_dir: Path,
    table_sizes_json: Path,
    *,
    m_releases: int = 20,
    seed_base: int = 20261121,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Compute Fig 4b (eps in {1,2,4,8}) and Fig 4c (size bins at eps=8) from the EXACT SAME 20 releases."""
    sizes = json.loads(table_sizes_json.read_text(encoding="utf-8"))

    def size_bin(n: int) -> str:
        return "<1k" if n < 1000 else ("1k-100k" if n <= 100000 else ">100k")

    table_meta: Dict[Tuple[str, str], Tuple[int, str, Dict[str, Set[str]]]] = {}
    for db_id in sorted(sizes):
        ref = json.loads((c1_dir / f"{db_id}.json").read_text(encoding="utf-8"))
        for t, nrows in sorted(sizes[db_id].items()):
            c1_vals: Dict[str, Set[str]] = {}
            for col, entry in ref[t].items():
                if entry[4]:
                    c1_vals[col] = {str(v) for v in entry[4]}
            table_meta[(db_id, t)] = (nrows, size_bin(nrows), c1_vals)

    by_bin_counts = Counter(v[1] for v in table_meta.values())
    by_bin_target_vals = Counter()
    for _, sb, c1_vals in table_meta.values():
        by_bin_target_vals[sb] += sum(len(s) for s in c1_vals.values())
    total_targets = sum(by_bin_target_vals.values())
    assert total_targets == 1112, total_targets

    eps_list = (1, 2, 4, 8)
    modes = ("DG-v3", "DG-FlatW")
    bins = ("<1k", "1k-100k", ">100k")
    hits: Dict[int, Dict[str, List[Dict[str, int]]]] = {
        e: {m: [{sb: 0 for sb in bins} for _ in range(m_releases)] for m in modes}
        for e in eps_list
    }
    tots_by_bin: Dict[str, int] = {sb: int(by_bin_target_vals[sb]) for sb in bins}

    for db_id in sorted(sizes):
        db_path = bird_db_dir / f"{db_id}.sqlite"
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        schema = sr.extract_public_schema(con)
        for table, cols in schema.tables.items():
            _nrows, sb, c1_vals = table_meta[(db_id, table)]
            if not c1_vals:
                continue
            n = int(con.execute("SELECT COUNT(*) FROM " + prof_mod.quote(table)).fetchone()[0])
            target_cols = set(c1_vals.keys())
            col_metas = tuple(cm for cm in cols if cm.name in target_cols)
            stats = prof_mod.table_statistics_exact(con, table, col_metas, n, target_cols)

            for eps in eps_list:
                plan_v3 = prof_mod.plan_table_dg_v3(schema, table, eps, flat_weights=False)
                plan_fw = prof_mod.plan_table_dg_v3(schema, table, eps, flat_weights=True)
                qmaps = {
                    "DG-v3": {(it["stat"], it["column"]): it for it in plan_v3["queries"]},
                    "DG-FlatW": {(it["stat"], it["column"]): it for it in plan_fw["queries"]},
                }
                for m_idx in range(m_releases):
                    rng_seed = _sha256_seed_int(seed_base, eps, m_idx, db_id, table)
                    r_rng = random.Random(rng_seed)
                    for col, target_set in c1_vals.items():
                        st = stats[col]
                        for m_name in modes:
                            qmap = qmaps[m_name]
                            if ("values", col) not in qmap:
                                continue
                            nd = prof_mod.noisy_count_dg(
                                st["distinct_count"], qmap[("distinct_count", col)]["_sigma2_obj"], r_rng
                            )
                            if nd <= 50:
                                it = qmap[("values", col)]
                                vh = prof_mod.threshold_histogram_dg(
                                    st["values_sorted"], it["_sigma2_obj"], it["tau"], r_rng
                                )
                                kept = {str(prof_mod.decode_val_key(k)) for k in prof_mod.top_keys(vh, 15)}
                                hits[eps][m_name][m_idx][sb] += len(target_set & kept)
        con.close()

    fig4b: Dict[str, Any] = {"n_target_c1_values": total_targets, "M_releases": m_releases, "by_epsilon": {}}
    for eps in eps_list:
        fig4b["by_epsilon"][str(eps)] = {}
        for m_name in modes:
            rates = [sum(hits[eps][m_name][m_idx][sb] for sb in bins) / float(total_targets) for m_idx in range(m_releases)]
            fig4b["by_epsilon"][str(eps)][m_name] = {
                "mean_pct": round(100.0 * statistics.mean(rates), 2),
                "sd_pp": round(100.0 * statistics.stdev(rates), 2),
                "min_pct": round(100.0 * min(rates), 2),
                "max_pct": round(100.0 * max(rates), 2),
                "mean_pct_exact": round(100.0 * statistics.mean(rates), 6),
                "sd_pp_exact": round(100.0 * statistics.stdev(rates), 6),
            }

    res_modes: Dict[str, Any] = {}
    for m_name in modes:
        res_modes[m_name] = {}
        for sb in bins:
            rates = [hits[8][m_name][m_idx][sb] / float(tots_by_bin[sb]) for m_idx in range(m_releases)]
            res_modes[m_name][sb] = {
                "n_tables": int(by_bin_counts[sb]),
                "n_target_c1_values": int(tots_by_bin[sb]),
                "retention_mean_pct": round(100.0 * statistics.mean(rates), 2),
                "retention_sd_pp": round(100.0 * statistics.stdev(rates), 2),
                "retention_min_pct": round(100.0 * min(rates), 2),
                "retention_max_pct": round(100.0 * max(rates), 2),
                "retention_mean_pct_exact": round(100.0 * statistics.mean(rates), 6),
                "retention_sd_pp_exact": round(100.0 * statistics.stdev(rates), 6),
            }
        weighted = sum(
            res_modes[m_name][sb]["n_target_c1_values"] * res_modes[m_name][sb]["retention_mean_pct_exact"]
            for sb in bins
        ) / float(total_targets)
        assert abs(weighted - fig4b["by_epsilon"]["8"][m_name]["mean_pct_exact"]) < 1e-5

    fig4c = {
        "total_tables": len(table_meta),
        "M_releases": m_releases,
        "by_table_size_bin": res_modes["DG-v3"],
        "by_mode_and_table_size_bin": res_modes,
    }
    return fig4b, fig4c


def _verify_schedule_cells(reproduced: Dict[str, Any], stored: Dict[str, Any]) -> int:
    assert reproduced["schema_version"] == stored["schema_version"]
    assert reproduced["deid_fixture_sha256"] == stored["deid_fixture_sha256"]
    assert set(reproduced["schedules"]) == set(stored["schedules"])
    checked = 0
    for sname, srec in reproduced["schedules"].items():
        for cohort, crec in srec["cohorts"].items():
            for stg, drec in crec.items():
                for denom in ("reachable_120", "all_125"):
                    got = drec[denom]
                    exp = stored["schedules"][sname]["cohorts"][cohort][stg][denom]
                    assert got == exp, (sname, cohort, stg, denom, got, exp)
                    checked += 1
    return checked


def run_self_test() -> Dict[str, Any]:
    """Verify portable 192-cell Whole-Patient Bootstrap + Marginal Binomial reproduction."""
    t0 = time.monotonic()
    reproduced_wp = run_whole_patient_bootstrap_from_deid()
    reproduced_mb = run_marginal_binomial_from_deid()
    stored_doc = json.loads(
        (ROOT / "stats/clinical_survival_3stage_verified.json").read_text(encoding="utf-8")
    )
    stored_wp = stored_doc.get("whole_patient_bootstrap", stored_doc)
    stored_mb = stored_doc["marginal_binomial"]

    checked_wp = _verify_schedule_cells(reproduced_wp, stored_wp)
    checked_mb = _verify_schedule_cells(reproduced_mb, stored_mb)
    checked_cells = checked_wp + checked_mb
    assert checked_wp == 96 and checked_mb == 96 and checked_cells == 192

    s3_100k_all = reproduced_wp["schedules"]["db_global_eps8"]["cohorts"]["100000"]["stage3_eq10_top15_published"]["all_125"]
    s3_100k_pt = reproduced_wp["schedules"]["per_table_eps8"]["cohorts"]["100000"]["stage3_eq10_top15_published"]["reachable_120"]
    mb_100k_all = reproduced_mb["schedules"]["db_global_eps8"]["cohorts"]["100000"]["stage3_eq10_top15_published"]["all_125"]
    return {
        "elapsed_s": round(time.monotonic() - t0, 2),
        "cells_verified_bit_exact": checked_cells,
        "whole_patient_cells": checked_wp,
        "marginal_binomial_cells": checked_mb,
        "db_global_eps8_100k_all125": s3_100k_all,
        "per_table_eps8_100k_reach120": s3_100k_pt,
        "marginal_db_global_eps8_100k_all125": mb_100k_all,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Reproduce whole-patient bootstrap & marginal binomial (192 cells) from bundled de-identified fixture and verify bit-exact parity.",
    )
    parser.add_argument(
        "--out-stats-dir",
        type=Path,
        default=ROOT / "stats",
        help="Destination stats directory for `--write` or Tier-2 regeneration.",
    )
    parser.add_argument("--m-cohorts", type=int, default=200)
    parser.add_argument("--write", action="store_true", help="Write reproduced survival JSON into --out-stats-dir.")
    args = parser.parse_args(argv)

    if args.self_test or not args.write:
        res = run_self_test()
        print(
            f"CLINICAL SURVIVAL SELF-TEST PASSED ({res['cells_verified_bit_exact']} cells bit-exact "
            f"[{res['whole_patient_cells']} whole_patient + {res['marginal_binomial_cells']} marginal_binomial] "
            f"in {res['elapsed_s']}s; db_global@100k={res['db_global_eps8_100k_all125']['mean_pct']}%, "
            f"per_table_reach@100k={res['per_table_eps8_100k_reach120']['mean_pct']}%, "
            f"marginal_db_global@100k={res['marginal_db_global_eps8_100k_all125']['mean_pct']}%)"
        )
        return 0

    wp = run_whole_patient_bootstrap_from_deid(m_cohorts=args.m_cohorts)
    mb = run_marginal_binomial_from_deid(m_cohorts=args.m_cohorts)
    out_path = args.out_stats_dir / "clinical_survival_3stage_verified.json"
    combined = dict(wp)
    combined["whole_patient_bootstrap"] = wp
    combined["marginal_binomial"] = mb
    out_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
