#!/usr/bin/env python3
"""Unified Paper Table 2-9 & Technical Report Table 13-21 builder from artifact_release/stats/ and manifests/.

Computes all Stage-8 authoritative summary rates, paired 95% question & cluster bootstrap CIs,
balanced matched-replicate exact Rademacher p-values, exact 2^11 BIRD DB-cluster sign-flip
p-values, Monte Carlo 120/134-template EHRSQL cluster sign-flip p-values, Holm multiple-testing
families (question-level & cluster-level), Equal-Public-Cap (PubCap) & TightBPE BPE-cap analyses,
Post-hoc No-Values (noVal R1..R3) analyses, leak-free 2x2 factorial decomposition, canonical
custody-verified EHRSQL v3 (600 confirmation + 874 held-out) results, disagreement rulesets,
dual-comparator (Set vs Strict) reconciliation with zero timeout conflicts, sandboxed SQL
validation telemetry, and whole-patient bootstrap 3-stage clinical survival metrics.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
from statistics import mean, stdev
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from silhouette.jsonl_io import load_jsonl, open_jsonl
from stats.reproduce_stats import (
    B,
    SEED,
    cluster_bootstrap,
    compare_conditions,
    exact_rademacher_p,
    holm,
    interval,
    paired_bootstrap,
    paired_trend,
    rate_summary,
    ratio_bootstrap,
    vec,
)


def exact_cluster_sign_flip_2powK(
    a_vals: list[float],
    b_vals: list[float],
    cluster_ids: list[Any],
) -> float:
    """Exact 2^K Rademacher cluster sign-flip p-value (for K <= 15 clusters, e.g. 11 BIRD DBs)."""
    by_c: dict[Any, float] = defaultdict(float)
    for c, a, b in zip(cluster_ids, a_vals, b_vals):
        by_c[c] += a - b
    clusters = [by_c[k] for k in sorted(by_c)]
    k = len(clusters)
    if k == 0:
        return 1.0
    if k > 15:
        raise ValueError(f"Exact 2^K enumeration only supported for K <= 15, got {k}")
    obs = abs(sum(clusters))
    if obs <= 1e-12:
        return 1.0
    ge = 0
    total = 1 << k
    for mask in range(total):
        s = 0.0
        for i in range(k):
            s += clusters[i] if (mask >> i) & 1 else -clusters[i]
        if abs(s) >= obs - 1e-12:
            ge += 1
    return ge / total


def mc_cluster_sign_flip(
    a_vals: list[float],
    b_vals: list[float],
    cluster_ids: list[Any],
    draws: int = 100000,
    seed: int = SEED,
) -> float:
    """Monte Carlo Rademacher cluster sign-flip p-value for large K (120 / 134 templates)."""
    by_c: dict[Any, float] = defaultdict(float)
    for c, a, b in zip(cluster_ids, a_vals, b_vals):
        by_c[c] += a - b
    clusters = [by_c[k] for k in sorted(by_c)]
    obs = abs(sum(clusters))
    if obs <= 1e-12:
        return 1.0
    k = len(clusters)
    chunks = [clusters[i : i + 8] for i in range(0, k, 8)]
    luts: list[list[float]] = []
    for ch in chunks:
        m = len(ch)
        lut = [sum(ch[j] if (mask >> j) & 1 else -ch[j] for j in range(m)) for mask in range(1 << m)]
        luts.append(lut)
    rng = random.Random(seed)
    ge = 0
    thresh = obs - 1e-12
    if len(luts) == 1:
        lut0 = luts[0]
        m0 = 1 << len(chunks[0])
        for _ in range(draws):
            if abs(lut0[rng.randrange(m0)]) >= thresh:
                ge += 1
    else:
        for _ in range(draws):
            mask = rng.getrandbits(k)
            s = 0.0
            for lut in luts:
                s += lut[mask & 255]
                mask >>= 8
            if abs(s) >= thresh:
                ge += 1
    return (ge + 1) / (draws + 1)


def full_compare(
    data: dict[str, dict[Any, int]],
    conds_a: str | list[str],
    conds_b: str | list[str],
    ids: list[Any],
    cluster_ids: list[Any] | None = None,
    *,
    exact_2powk: bool = False,
    mc_cluster: bool = False,
) -> dict[str, Any]:
    """Extend compare_conditions with exact 2^K / MC cluster p-values and matched-replicate CIs."""
    res = compare_conditions(data, conds_a, conds_b, ids, cluster_ids)
    a_list = [conds_a] if isinstance(conds_a, str) else list(conds_a)
    b_list = [conds_b] if isinstance(conds_b, str) else list(conds_b)
    has_clusters = cluster_ids is not None and len(set(cluster_ids)) >= 2
    if has_clusters:
        a_vals = vec(data, a_list, ids)
        b_vals = vec(data, b_list, ids)
        if exact_2powk:
            res["exact_2powK_cluster_p"] = exact_cluster_sign_flip_2powK(a_vals, b_vals, cluster_ids)
        if mc_cluster:
            res["mc_cluster_p"] = mc_cluster_sign_flip(a_vals, b_vals, cluster_ids)
    if len(a_list) != len(b_list):
        k = min(len(a_list), len(b_list))
        ma_vals = vec(data, a_list[:k], ids)
        mb_vals = vec(data, b_list[:k], ids)
        mdiffs = [a - b for a, b in zip(ma_vals, mb_vals)]
        mlo, mhi = paired_bootstrap(mdiffs)
        res["matched_ci95_pp"] = [round(100.0 * mlo, 2), round(100.0 * mhi, 2)]
        r1a_vals = vec(data, [a_list[0]], ids)
        r1b_vals = vec(data, [b_list[0]], ids)
        if has_clusters and cluster_ids is not None:
            mclo, mchi = cluster_bootstrap(mdiffs, cluster_ids)
            res["matched_cluster_ci95_pp"] = [round(100.0 * mclo, 2), round(100.0 * mchi, 2)]
            if exact_2powk:
                cp_min = exact_cluster_sign_flip_2powK(ma_vals, mb_vals, cluster_ids)
                cp_r1 = exact_cluster_sign_flip_2powK(r1a_vals, r1b_vals, cluster_ids)
                res["matched_exact_2powK_cluster_p"] = cp_min
                res["matched_minR_cluster_p"] = cp_min
                res["matched_r1_cluster_p"] = cp_r1
            if mc_cluster:
                mp_min = mc_cluster_sign_flip(ma_vals, mb_vals, cluster_ids)
                mp_r1 = mc_cluster_sign_flip(r1a_vals, r1b_vals, cluster_ids)
                res["matched_mc_cluster_p"] = mp_min
                res["matched_minR_cluster_p"] = mp_min
                res["matched_r1_cluster_p"] = mp_r1
    return res


def _telemetry_summary(
    rec_by_bcq: dict[tuple[str, str, Any], dict[str, Any]],
    benchmark: str,
    conds: list[str],
    ids: list[Any],
) -> dict[str, Any]:
    rows = [rec_by_bcq[(benchmark, c, q)] for c in conds for q in ids if (benchmark, c, q) in rec_by_bcq]
    if not rows:
        return {}
    ttoks = [r["total_tokens"] for r in rows if r.get("total_tokens") is not None]
    ptoks = [r["prompt_tokens"] for r in rows if r.get("prompt_tokens") is not None]
    ctoks = [r["completion_tokens"] for r in rows if r.get("completion_tokens") is not None]
    cached = [r["cached_input_tokens"] for r in rows if r.get("cached_input_tokens") is not None]
    sb = [r["sandbox_calls"] for r in rows if r.get("sandbox_calls") is not None]
    lk = [r["lookup_calls"] for r in rows if r.get("lookup_calls") is not None]
    tc = [r["tool_calls"] for r in rows if r.get("tool_calls") is not None]
    lats = [r["latency_s"] for r in rows if r.get("latency_s") is not None]
    out: dict[str, Any] = {
        "slots": len(rows),
        "mean_total_tokens": round(mean(ttoks), 2) if ttoks else None,
        "mean_prompt_tokens": round(mean(ptoks), 2) if ptoks else None,
        "mean_completion_tokens": round(mean(ctoks), 2) if ctoks else None,
        "mean_sandbox_calls": round(mean(sb), 3) if sb else (round(mean(tc), 3) if tc else None),
        "mean_lookup_calls": round(mean(lk), 3) if lk else 0.0,
        "mean_latency_s": round(mean(lats), 3) if lats else None,
    }
    if cached:
        out["mean_cached_input_tokens"] = round(mean(cached), 2)
        if ptoks and sum(ptoks) > 0:
            out["cache_hit_rate_pct"] = round(100.0 * sum(cached) / sum(ptoks), 2)
    return out


def rate_with_cluster(
    data: dict[str, dict[Any, int]],
    conds: str | list[str],
    ids: list[Any],
    cluster_ids: list[Any] | None = None,
    strict_data: dict[str, dict[Any, int]] | None = None,
    *,
    rec_by_bcq: dict[tuple[str, str, Any], dict[str, Any]] | None = None,
    benchmark: str | None = None,
    per_rep_cluster: bool = False,
) -> dict[str, Any]:
    """Rate summary with cluster bootstrap 95% CI, strict EX, and receipt telemetry."""
    out = rate_summary(data, conds, ids)
    cond_list = [conds] if isinstance(conds, str) else list(conds)
    vals = vec(data, cond_list, ids)
    out["hits_total"] = int(sum(data[c][q] for c in cond_list for q in ids))
    out["slots_total"] = len(cond_list) * len(ids)
    if cluster_ids is not None and len(set(cluster_ids)) >= 2:
        clo, chi = cluster_bootstrap(vals, cluster_ids)
        out["cluster_ci95_pp"] = [round(100.0 * clo, 2), round(100.0 * chi, 2)]
        if per_rep_cluster:
            for rep_row in out.get("per_replicate", []):
                if len(cond_list) == 1:
                    rep_row["cluster_ci95_pp"] = list(out["cluster_ci95_pp"])
                else:
                    rvals = vec(data, rep_row["condition"], ids)
                    rlo, rhi = cluster_bootstrap(rvals, cluster_ids)
                    rep_row["cluster_ci95_pp"] = [round(100.0 * rlo, 2), round(100.0 * rhi, 2)]
    if strict_data is not None and all(c in strict_data and all(q in strict_data[c] for q in ids) for c in cond_list):
        svals = vec(strict_data, cond_list, ids)
        out["strict_ex_mean_pct"] = round(100.0 * mean(svals), 2)
        out["strict_ex_hits_total"] = int(sum(strict_data[c][q] for c in cond_list for q in ids))
    if rec_by_bcq is not None and benchmark is not None:
        out["telemetry"] = _telemetry_summary(rec_by_bcq, benchmark, cond_list, ids)
    return out


def _load_outcomes_and_receipts() -> tuple[
    dict[str, dict[str, dict[Any, int]]],
    dict[str, dict[str, dict[Any, int]]],
    dict[str, dict[str, dict[Any, int]]],
    dict[Any, str],
    list[dict[str, Any]],
    dict[tuple[str, str, Any], dict[str, Any]],
]:
    """Load outcomes & receipts partitioned by benchmark so EHRSQL-600/874 & BIRD never collide."""
    data_bm: dict[str, dict[str, dict[Any, int]]] = defaultdict(lambda: defaultdict(dict))
    strict_bm: dict[str, dict[str, dict[Any, int]]] = defaultdict(lambda: defaultdict(dict))
    deliv_bm: dict[str, dict[str, dict[Any, int]]] = defaultdict(lambda: defaultdict(dict))
    dbs: dict[Any, str] = {}
    with open_jsonl(ROOT / "stats/per_question_outcomes.jsonl") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            bm = r["benchmark"]
            q = r["question_id"]
            db = r["db_id"]
            cond = r["condition"]
            if q in data_bm[bm][cond] or dbs.setdefault(q, db) != db or r["outcome"] not in (0, 1):
                raise ValueError(f"Invalid or duplicate outcome: {bm} {cond} {q}")
            data_bm[bm][cond][q] = int(r["outcome"])
            if r.get("strict_ex") is not None:
                strict_bm[bm][cond][q] = int(r["strict_ex"])
            if r.get("deliverable_set_ex") is not None:
                deliv_bm[bm][cond][q] = int(r["deliverable_set_ex"])

    receipts = load_jsonl(ROOT / "stats/sql_and_audit_receipts.jsonl")
    rec_by_bcq: dict[tuple[str, str, Any], dict[str, Any]] = {}
    for r in receipts:
        bm = r["benchmark"]
        cond = r["condition"]
        q = r["question_id"]
        rec_by_bcq[(bm, cond, q)] = r
        if r.get("strict_ex") is not None:
            strict_bm[bm][cond][q] = int(r["strict_ex"])
        if r.get("deliverable_set_ex") is not None:
            deliv_bm[bm][cond][q] = int(r["deliverable_set_ex"])
    return data_bm, strict_bm, deliv_bm, dbs, receipts, rec_by_bcq


def _compute_table3_audit(receipts: list[dict[str, Any]]) -> dict[str, Any]:
    """Read uniform DB-deduplicated [unique_across_dbs, cumulative_across_slots] metrics from p7_audit_summary.json."""
    p7_aud = json.loads((ROOT / "stats/p7_audit_summary.json").read_text(encoding="utf-8"))
    conds = p7_aud["conditions"]

    def pool_r3(keys: list[str]) -> dict[str, Any]:
        sub = [conds[k] for k in keys]
        return {
            "replicates": keys,
            "slots_per_rep": sub[0]["slots"],
            "pooled_slots": sum(x["slots"] for x in sub),
            "copied": [round(mean(x["copied"][0] for x in sub), 2), round(mean(x["copied"][1] for x in sub), 2)],
            "ser_out": [round(mean(x["ser_out"][0] for x in sub), 2), round(mean(x["ser_out"][1] for x in sub), 2)],
            "full_payload": [round(mean(x["full_payload"][0] for x in sub)), round(mean(x["full_payload"][1] for x in sub))],
            "oob": [round(mean(x["oob"][0] for x in sub), 2), round(mean(x["oob"][1] for x in sub), 2)],
            "rows_transmitted": round(mean(x.get("rows_transmitted", 0) for x in sub), 2),
        }

    table3_rows = {
        "BIRD_Schema_only_r0": conds["BIRD_C0_r0"],
        "BIRD_Schema_only_r2": conds["BIRD_C0_r2"],
        "BIRD_Full_access_r0": conds["BIRD_C4_r0"],
        "BIRD_Full_access_r2": conds["BIRD_C4_r2"],
        "BIRD_Select_3": conds["BIRD_S3"],
        "BIRD_1way_Synth_Probe_R3": pool_r3(["BIRD_IND8", "IND8_PROBE_R2/r0", "IND8_PROBE_R3/r0"]),
        "BIRD_PubCap_Static_Synth_R3": pool_r3(["PUBCAP_IND8_ROWS_R1/r0", "PUBCAP_IND8_ROWS_R2/r0", "PUBCAP_IND8_ROWS_R3/r0"]),
        "BIRD_2way_Synth": conds["SYN8/r0"],
        "BIRD_FK_DAG_Synth_R1": conds["SYN8_FK_DAG_R1/r0"],
        "BIRD_DP_Query_v2_TP8_R3": pool_r3(["DP_QUERY_ROW_TP8_V2_R1/r0", "DP_QUERY_ROW_TP8_V2_R2/r0", "DP_QUERY_ROW_TP8_V2_R3/r0"]),
        "BIRD_DP_Query_v2_DB8_R1": conds["DP_QUERY_ROW_DB8_V2_R1/r0"],
        "BIRD_On_Demand_Profile_Tool_R3": pool_r3(["IND8_PROFILE_TOOL_R1/r0", "IND8_PROFILE_TOOL_R2/r0", "IND8_PROFILE_TOOL_R3/r0"]),
        "BIRD_PubCap_1way_Profile_R3": pool_r3(["PUBCAP_IND8_PROFILE_R1/r0", "PUBCAP_IND8_PROFILE_R2/r0", "PUBCAP_IND8_PROFILE_R3/r0"]),
        "BIRD_Unpruned_1way_Profile_R3": pool_r3(["BIRD_IND8_PROMPT", "BIRD_IND8_PROMPT_R2", "BIRD_IND8_PROMPT_R3"]),
        "BIRD_PubCap_Silhouette_R3": pool_r3(["PUBCAP_SILHOUETTE_R1/r0", "PUBCAP_SILHOUETTE_R2/r0", "PUBCAP_SILHOUETTE_R3/r0"]),
        "BIRD_Silhouette_R3": pool_r3(["BIRD_DG8_CLEAN_R1_full440", "BIRD_DG8_CLEAN_R2_full440", "BIRD_DG8_CLEAN_R3_full440"]),
        "BIRD_Noiseless_V3": conds["BIRD_NOISELESS_V3/r0"],
        "BIRD_TightBPE_Silhouette_R3": pool_r3(["TIGHTBPE_SILHOUETTE_R1/r0", "TIGHTBPE_SILHOUETTE_R2/r0", "TIGHTBPE_SILHOUETTE_R3/r0"]),
        "BIRD_TightBPE_1way_Profile_R3": pool_r3(["TIGHTBPE_IND8_PROF_R1/r0", "TIGHTBPE_IND8_PROF_R2/r0", "TIGHTBPE_IND8_PROF_R3/r0"]),
        "BIRD_TightBPE_Static_Synth_R3": pool_r3(["TIGHTBPE_STATIC_SYNTH_R1/r0", "TIGHTBPE_STATIC_SYNTH_R2/r0", "TIGHTBPE_STATIC_SYNTH_R3/r0"]),
        "BIRD_TightBPE_ValFirst_Silhouette_R3": pool_r3(["TIGHTBPE_VALFIRST_SILH_R1/r0", "TIGHTBPE_VALFIRST_SILH_R2/r0", "TIGHTBPE_VALFIRST_SILH_R3/r0"]),
        "BIRD_TightBPE_ValFirst_1way_Profile_R3": pool_r3(["TIGHTBPE_VALFIRST_IND8_PROF_R1/r0", "TIGHTBPE_VALFIRST_IND8_PROF_R2/r0", "TIGHTBPE_VALFIRST_IND8_PROF_R3/r0"]),
        "BIRD_Fetch1_PubCap_1way_Profile_R3": pool_r3(["FETCH1_PUBCAP_IND8_PROF_R1/r0", "FETCH1_PUBCAP_IND8_PROF_R2/r0", "FETCH1_PUBCAP_IND8_PROF_R3/r0"]),
        "BIRD_noVal_Post_R23": pool_r3(["DG8_NOVAL_POST_R2/r0", "DG8_NOVAL_POST_R3/r0"]),
        "SOL_PubCap_Static_Synth_R1": conds["SOL_PUBCAP_IND8_ROWS_R1/r0"],
        "SOL_Silhouette_R3": pool_r3(["SOL_DG8_CLEAN_R1/r0", "SOL_DG8_CLEAN_R2/r0", "SOL_DG8_CLEAN_R3/r0"]),
        "EHR600_Schema_only": conds["EHR600_Schema_only"],
        "EHR600_Silhouette_V3_R3": pool_r3(["EHR_DG8P_V3_R1/r0", "EHR_DG8P_V3_R2/r0", "EHR_DG8P_V3_R3/r0"]),
        "EHR600_Silhouette_noLookup_V3_R3": pool_r3(["EHR_DG8P_NOLKP_V3_R1/r0", "EHR_DG8P_NOLKP_V3_R2/r0", "EHR_DG8P_NOLKP_V3_R3/r0"]),
        "EHR600_Noiseless_LKP_V3": conds["EHR_NOISELESS_LKP_V3/r0"],
        "EHR600_Noiseless_noLookup_V3": conds["EHR_NOISELESS_NOLKP_V3/r0"],
        "EHR600_1way_Profile_V3": conds["EHR_IND8_V3_R1/r0"],
        "EHR600_Entity_Demo94_V3": conds["EHR_ENT8_V3_R1/r0"],
        "EHR600_Synthetic_DB_1way": conds["EHR600_Synthetic_DB_1way"],
        "EHR600_Select_3": conds["EHR600_Select_3"],
        "EHR600_Full_access": conds["EHR600_Full_access"],
        "EHR874_Schema_only": conds["EHR874_Schema_only"],
        "EHR874_Silhouette_V3_R3": pool_r3(["EHR874_DG8P_V3_R1/r0", "EHR874_DG8P_V3_R2/r0", "EHR874_DG8P_V3_R3/r0"]),
        "EHR874_Silhouette_noLookup_V3_R3": pool_r3(["EHR874_DG8P_NOLKP_V3_R1/r0", "EHR874_DG8P_NOLKP_V3_R2/r0", "EHR874_DG8P_NOLKP_V3_R3/r0"]),
        "EHR874_Noiseless_LKP_V3": conds["EHR874_NOISELESS_LKP_V3/r0"],
        "EHR874_Noiseless_noLookup_V3": conds["EHR874_NOISELESS_NOLKP_V3/r0"],
        "EHR874_Full_access": conds["EHR874_Full_access"],
    }
    legacy_superseded_audit = {
        "EHR600_Silhouette_legacy_R3": pool_r3(["EHR600_Silhouette_R1", "EHR600_Silhouette_R2", "EHR600_Silhouette_R3"]),
        "EHR600_1way_Profile_legacy": conds["EHR600_1way_Profile"],
        "EHR600_Demo94_Entity_legacy": conds["EHR600_Demo94_Entity"],
        "EHR874_Silhouette_legacy": conds["EHR874_Silhouette"],
        "EHR874_Silhouette_noLookup_legacy": conds["EHR874_Silhouette_noLookup"],
        "EHR874_Noiseless_noLookup_legacy": conds["EHR874_Noiseless_noLookup"],
    }

    q201_fixture = json.loads((ROOT / "stats/bird_financial_q201_audit_fixture.json").read_text(encoding="utf-8"))
    q201_audit = {
        rel_name: q201_fixture["releases"][rel_name]["reference_transmission_ledger_full"]
        for rel_name in ("DG8_CLEAN_R1", "DG8_CLEAN_R2", "DG8_CLEAN_R3", "P2_DG8_R1_ref")
    }
    outbound_sha_counts = Counter(
        (r["benchmark"], r["condition"]) for r in receipts if r.get("outbound_messages_sha256")
    )
    return {
        "definition": p7_aud["definition"],
        "total_conditions": len(conds),
        "outbound_messages_sha256_total_slots": sum(outbound_sha_counts.values()),
        "outbound_messages_sha256_by_condition": {
            f"{bm}:{cond}": cnt for (bm, cond), cnt in sorted(outbound_sha_counts.items())
        },
        "table3_rows": table3_rows,
        "clinical_v3_profile_slots_split": p7_aud["clinical_v3_profile_slots_split"],
        "legacy_superseded_archive": legacy_superseded_audit,
        "all_db_dedup_conditions": conds,
        "financial_q201_fixture_audit": q201_audit,
    }


def _compute_factorial_2x2(
    bird_data: dict[str, dict[Any, int]],
    bird_strict: dict[str, dict[Any, int]],
    bird_440: list[int],
    bird_dbs: list[str],
    rec_by_bcq: dict[tuple[str, str, Any], dict[str, Any]],
) -> dict[str, Any]:
    """Leak-Free 2x2 Representation x Delivery Factorial on identical IND8 source (R=3)."""
    prof_tool = ["IND8_PROFILE_TOOL_R1/r0", "IND8_PROFILE_TOOL_R2/r0", "IND8_PROFILE_TOOL_R3/r0"]
    prof_prompt = ["PUBCAP_IND8_PROFILE_R1/r0", "PUBCAP_IND8_PROFILE_R2/r0", "PUBCAP_IND8_PROFILE_R3/r0"]
    rows_tool = ["IND8/r0", "IND8_PROBE_R2/r0", "IND8_PROBE_R3/r0"]
    rows_prompt = ["PUBCAP_IND8_ROWS_R1/r0", "PUBCAP_IND8_ROWS_R2/r0", "PUBCAP_IND8_ROWS_R3/r0"]
    unpruned_prof = ["IND8_PROMPT/r0", "IND8_PROMPT_R2/r0", "IND8_PROMPT_R3/r0"]
    pubcap_silh = ["PUBCAP_SILHOUETTE_R1/r0", "PUBCAP_SILHOUETTE_R2/r0", "PUBCAP_SILHOUETTE_R3/r0"]
    silh_nat = ["DG8_CLEAN_R1/r0", "DG8_CLEAN_R2/r0", "DG8_CLEAN_R3/r0"]

    cells = {
        "Profile_x_Tool": rate_with_cluster(bird_data, prof_tool, bird_440, bird_dbs, bird_strict, rec_by_bcq=rec_by_bcq, benchmark="bird"),
        "Profile_x_Prompt_PubCap": rate_with_cluster(bird_data, prof_prompt, bird_440, bird_dbs, bird_strict, rec_by_bcq=rec_by_bcq, benchmark="bird"),
        "SynthRows_x_Tool": rate_with_cluster(bird_data, rows_tool, bird_440, bird_dbs, bird_strict, rec_by_bcq=rec_by_bcq, benchmark="bird"),
        "SynthRows_x_Prompt_PubCap": rate_with_cluster(bird_data, rows_prompt, bird_440, bird_dbs, bird_strict, rec_by_bcq=rec_by_bcq, benchmark="bird"),
        "Coverage_Unpruned_1way_Profile": rate_with_cluster(bird_data, unpruned_prof, bird_440, bird_dbs, bird_strict, rec_by_bcq=rec_by_bcq, benchmark="bird"),
        "EqualCap_PubCap_Silhouette": rate_with_cluster(bird_data, pubcap_silh, bird_440, bird_dbs, bird_strict, rec_by_bcq=rec_by_bcq, benchmark="bird"),
        "Natural_Silhouette": rate_with_cluster(bird_data, silh_nat, bird_440, bird_dbs, bird_strict, rec_by_bcq=rec_by_bcq, benchmark="bird"),
    }

    v_pt = vec(bird_data, prof_tool, bird_440)
    v_pp = vec(bird_data, prof_prompt, bird_440)
    v_rt = vec(bird_data, rows_tool, bird_440)
    v_rp = vec(bird_data, rows_prompt, bird_440)

    def effect_stats(lhs: list[float], rhs: list[float], name: str) -> dict[str, Any]:
        diffs = [a - b for a, b in zip(lhs, rhs)]
        lo, hi = paired_bootstrap(diffs)
        clo, chi = cluster_bootstrap(diffs, bird_dbs)
        p_2pow11 = exact_cluster_sign_flip_2powK(lhs, rhs, bird_dbs)
        ms = [round(12 * d) for d in diffs]
        p_rad = exact_rademacher_p([m / 6.0 for m in ms], [0.0] * len(ms), 3, 3) if all(abs(d * 6 - round(d * 6)) < 1e-9 for d in diffs) else mc_cluster_sign_flip(lhs, rhs, list(range(len(diffs))), draws=100000)
        return {
            "effect": name,
            "diff_pp": round(100.0 * mean(diffs), 2),
            "ci95_pp": [round(100.0 * lo, 2), round(100.0 * hi, 2)],
            "cluster_ci95_pp": [round(100.0 * clo, 2), round(100.0 * chi, 2)],
            "p": p_rad,
            "exact_2pow11_cluster_p": p_2pow11,
        }

    del_lhs = [0.5 * (pp + rp) for pp, rp in zip(v_pp, v_rp)]
    del_rhs = [0.5 * (pt + rt) for pt, rt in zip(v_pt, v_rt)]
    rep_lhs = [0.5 * (pp + pt) for pp, pt in zip(v_pp, v_pt)]
    rep_rhs = [0.5 * (rp + rt) for rp, rt in zip(v_rp, v_rt)]
    int_lhs = [pp - pt for pp, pt in zip(v_pp, v_pt)]
    int_rhs = [rp - rt for rp, rt in zip(v_rp, v_rt)]

    return {
        "cells": cells,
        "simple_effects": {
            "Delivery_within_Profile (PubCap_1wProf - ProfileTool)": full_compare(bird_data, prof_prompt, prof_tool, bird_440, bird_dbs, exact_2powk=True),
            "Delivery_within_SynthRows (PubCap_Rows - Probe)": full_compare(bird_data, rows_prompt, rows_tool, bird_440, bird_dbs, exact_2powk=True),
            "Representation_within_Prompt (PubCap_1wProf - PubCap_Rows)": full_compare(bird_data, prof_prompt, rows_prompt, bird_440, bird_dbs, exact_2powk=True),
            "Representation_within_Tool (ProfileTool - Probe)": full_compare(bird_data, prof_tool, rows_tool, bird_440, bird_dbs, exact_2powk=True),
        },
        "main_effects_and_interaction": {
            "Main_Delivery (Prompt - Tool)": effect_stats(del_lhs, del_rhs, "Main Delivery (Prompt - Tool)"),
            "Main_Representation (Profile - SynthRows)": effect_stats(rep_lhs, rep_rhs, "Main Representation (Profile - SynthRows)"),
            "Interaction ((ProfilePrompt - ProfileTool) - (RowsPrompt - RowsTool))": effect_stats(int_lhs, int_rhs, "Interaction"),
        },
        "coverage_and_allocation_steps": {
            "Coverage_step (Unpruned_1wProf - PubCap_1wProf)": full_compare(bird_data, unpruned_prof, prof_prompt, bird_440, bird_dbs, exact_2powk=True),
            "EqualCap_Allocation_vs_Profile (PubCap_Silh - PubCap_1wProf)": full_compare(bird_data, pubcap_silh, prof_prompt, bird_440, bird_dbs, exact_2powk=True),
            "EqualCap_Allocation_vs_Rows (PubCap_Silh - PubCap_Rows)": full_compare(bird_data, pubcap_silh, rows_prompt, bird_440, bird_dbs, exact_2powk=True),
            "Natural_Allocation (Silhouette - Unpruned_1wProf)": full_compare(bird_data, silh_nat, unpruned_prof, bird_440, bird_dbs, exact_2powk=True),
            "Natural_vs_PubCap_Silhouette (Silhouette - PubCap_Silh)": full_compare(bird_data, silh_nat, pubcap_silh, bird_440, bird_dbs, exact_2powk=True),
        },
    }


def _turn_ordered_sandbox_row(
    rec_by_bcq: dict[tuple[str, str, Any], dict[str, Any]],
    benchmark: str,
    conds: list[str],
    qids: list[Any],
    setting_label: str,
) -> dict[str, Any]:
    """Compute strictly turn-ordered sandbox telemetry and deliverable/official EX from enriched receipts."""
    items = [rec_by_bcq[(benchmark, c, q)] for c in conds for q in qids if (benchmark, c, q) in rec_by_bcq]
    n = len(items)
    avail_items = [r for r in items if r.get("sandbox_telemetry_available")]
    na = len(avail_items)
    full_telem = (na == n and n > 0)

    deliv_ex = sum(int(r.get("deliverable_set_ex", r.get("official_set_ex", 0))) for r in items)
    off_ex = sum(int(r.get("official_set_ex", 0)) for r in items)
    empty_sub = sum(
        1 for r in items
        if r.get("deliverable_status") == "empty_submission"
        or r.get("no_submission")
        or r.get("runtime_error") == "empty_sql"
        or r.get("status") == "no_submission"
        or not (r.get("predicted_sql") or "").strip()
    )
    deliv_non_ok = Counter(
        r["deliverable_status"] for r in items
        if r.get("deliverable_status") and r["deliverable_status"] not in ("ok", "empty_submission")
    )
    sub_err_counter = Counter(
        r["runtime_error"] for r in items
        if r.get("runtime_error")
        and r["runtime_error"] != "empty_sql"
        and not r.get("no_submission")
        and (r.get("predicted_sql") or "").strip()
    )

    if not full_telem:
        # Never coerce missing telemetry into 0; emit explicit "NA" / None
        return {
            "benchmark": benchmark,
            "setting": setting_label,
            "conditions": list(conds),
            "slots": n,
            "telemetry_available": False,
            "n_telemetry_slots": na,
            "runs_with_sandbox_call": "NA",
            "runs_with_sandbox_call_pct": "NA",
            "sandbox_calls_sum": "NA",
            "calls_per_slot": "NA",
            "error_calls_sum": "NA",
            "error_calls_pct": "NA",
            "error_slots": "NA",
            "error_slots_pct": "NA",
            "error_then_rewrite_slots": "NA",
            "error_then_rewrite_pct": "NA",
            "error_then_exec_ok": "NA",
            "error_then_ex1": "NA",
            "any_sql_revision_slots": "NA",
            "any_sql_revision_pct": "NA",
            "zero_error_revision_slots": "NA",
            "deliverable_set_ex": deliv_ex,
            "deliverable_set_ex_pct": round(100.0 * deliv_ex / n, 2) if n else 0.0,
            "official_set_ex": off_ex,
            "official_set_ex_pct": round(100.0 * off_ex / n, 2) if n else 0.0,
            "empty_submission_count": empty_sub,
            "deliverable_error_slots": sum(deliv_non_ok.values()),
            "deliverable_error_breakdown": dict(sorted(deliv_non_ok.items())),
            "submitted_sql_errors_count": sum(sub_err_counter.values()),
            "submitted_sql_error_breakdown": dict(sorted(sub_err_counter.items())),
        }

    runs_with_call = sum(1 for r in avail_items if (r.get("sandbox_calls_turn_ordered") or 0) > 0)
    calls_sum = sum(int(r.get("sandbox_calls_turn_ordered") or 0) for r in avail_items)
    err_calls = sum(int(r.get("sandbox_error_calls_turn_ordered") or 0) for r in avail_items)
    err_slots = sum(int(r["slot_has_sandbox_error"]) for r in avail_items)
    err_rw = sum(int(r["slot_error_then_rewrite"]) for r in avail_items)
    err_exec_ok = sum(int(r["slot_error_then_exec_ok"]) for r in avail_items)
    err_ex1 = sum(int(r["slot_error_then_ex1"]) for r in avail_items)
    any_rev = sum(int(r["slot_any_sql_revision"]) for r in avail_items)
    zero_err_rev = sum(int(r["slot_zero_error_revision"]) for r in avail_items)

    return {
        "benchmark": benchmark,
        "setting": setting_label,
        "conditions": list(conds),
        "slots": n,
        "telemetry_available": True,
        "n_telemetry_slots": na,
        "runs_with_sandbox_call": runs_with_call,
        "runs_with_sandbox_call_pct": round(100.0 * runs_with_call / n, 1),
        "sandbox_calls_sum": calls_sum,
        "calls_per_slot": round(calls_sum / n, 2),
        "error_calls_sum": err_calls,
        "error_calls_pct": round(100.0 * err_calls / calls_sum, 1) if calls_sum else 0.0,
        "error_slots": err_slots,
        "error_slots_pct": round(100.0 * err_slots / n, 2),
        "error_then_rewrite_slots": err_rw,
        "error_then_rewrite_pct": round(100.0 * err_rw / n, 2),
        "error_then_exec_ok": err_exec_ok,
        "error_then_exec_ok_ratio": f"{err_exec_ok}/{err_rw}",
        "error_then_ex1": err_ex1,
        "error_then_ex1_ratio": f"{err_ex1}/{err_rw}",
        "any_sql_revision_slots": any_rev,
        "any_sql_revision_pct": round(100.0 * any_rev / n, 2),
        "zero_error_revision_slots": zero_err_rev,
        "deliverable_set_ex": deliv_ex,
        "deliverable_set_ex_pct": round(100.0 * deliv_ex / n, 2),
        "official_set_ex": off_ex,
        "official_set_ex_pct": round(100.0 * off_ex / n, 2),
        "empty_submission_count": empty_sub,
        "deliverable_error_slots": sum(deliv_non_ok.values()),
        "deliverable_error_breakdown": dict(sorted(deliv_non_ok.items())),
        "submitted_sql_errors_count": sum(sub_err_counter.values()),
        "submitted_sql_error_breakdown": dict(sorted(sub_err_counter.items())),
    }


def _compute_tightbpe_and_noval(
    bird_data: dict[str, dict[Any, int]],
    bird_strict: dict[str, dict[Any, int]],
    bird_440: list[int],
    bird_dbs: list[str],
    dev_strata: dict[str, Any],
    rec_by_bcq: dict[tuple[str, str, Any], dict[str, Any]],
) -> dict[str, Any]:
    """Compute TightBPE (Qwen2.5 BPE cap 100+10|T|+12|C|) and Post-hoc No-Values (R1..R3) full stats."""
    tman = json.loads((ROOT / "manifests/tightbpe_and_noval_manifest.json").read_text(encoding="utf-8"))

    tb_silh = ["TIGHTBPE_SILHOUETTE_R1/r0", "TIGHTBPE_SILHOUETTE_R2/r0", "TIGHTBPE_SILHOUETTE_R3/r0"]
    tb_prof = ["TIGHTBPE_IND8_PROF_R1/r0", "TIGHTBPE_IND8_PROF_R2/r0", "TIGHTBPE_IND8_PROF_R3/r0"]
    tb_synth = ["TIGHTBPE_STATIC_SYNTH_R1/r0", "TIGHTBPE_STATIC_SYNTH_R2/r0", "TIGHTBPE_STATIC_SYNTH_R3/r0"]
    vf_silh = ["TIGHTBPE_VALFIRST_SILH_R1/r0", "TIGHTBPE_VALFIRST_SILH_R2/r0", "TIGHTBPE_VALFIRST_SILH_R3/r0"]
    vf_prof = ["TIGHTBPE_VALFIRST_IND8_PROF_R1/r0", "TIGHTBPE_VALFIRST_IND8_PROF_R2/r0", "TIGHTBPE_VALFIRST_IND8_PROF_R3/r0"]
    noval_r3 = ["DG8_NOVAL_POST_R1/r0", "DG8_NOVAL_POST_R2/r0", "DG8_NOVAL_POST_R3/r0"]
    noval_r23 = ["DG8_NOVAL_POST_R2/r0", "DG8_NOVAL_POST_R3/r0"]
    dg8_clean = ["DG8_CLEAN_R1/r0", "DG8_CLEAN_R2/r0", "DG8_CLEAN_R3/r0"]
    pubcap_silh = ["PUBCAP_SILHOUETTE_R1/r0", "PUBCAP_SILHOUETTE_R2/r0", "PUBCAP_SILHOUETTE_R3/r0"]
    pubcap_prof = ["PUBCAP_IND8_PROFILE_R1/r0", "PUBCAP_IND8_PROFILE_R2/r0", "PUBCAP_IND8_PROFILE_R3/r0"]
    pubcap_rows = ["PUBCAP_IND8_ROWS_R1/r0", "PUBCAP_IND8_ROWS_R2/r0", "PUBCAP_IND8_ROWS_R3/r0"]

    bpe_token_stats: dict[str, Any] = {}
    for arm_name, rels in tman["arms"].items():
        per_rep = {}
        all_orig, all_fin = [], []
        for rel_name, db_map in sorted(rels.items()):
            orig_list = [v["orig_bpe_tokens"] for v in db_map.values()]
            fin_list = [v["final_bpe_tokens"] for v in db_map.values()]
            cap_list = [v["cap_tokens"] for v in db_map.values()]
            trunc_n = sum(1 for v in db_map.values() if v.get("truncated"))
            per_rep[rel_name] = {
                "n_dbs": len(db_map),
                "truncated_dbs": trunc_n,
                "cap_sum": sum(cap_list),
                "orig_bpe_sum": sum(orig_list),
                "final_bpe_sum": sum(fin_list),
                "orig_bpe_mean": round(mean(orig_list), 1),
                "final_bpe_mean": round(mean(fin_list), 1),
            }
            all_orig.extend(orig_list)
            all_fin.extend(fin_list)
        bpe_token_stats[arm_name] = {
            "tokenizer": tman["tokenizer"],
            "cap_formula": tman["cap_formula"],
            "pooled_orig_bpe_mean": round(mean(all_orig), 1),
            "pooled_final_bpe_mean": round(mean(all_fin), 1),
            "per_release": per_rep,
        }

    dev_exposure_rates: dict[str, Any] = {}
    for sname in ("low_dev_le4_134", "mid_dev_5_6_127", "high_dev_ge7_179"):
        sqids = dev_strata[sname]
        sdbs = [bird_dbs[bird_440.index(q)] for q in sqids]
        dev_exposure_rates[sname] = {
            "n": len(sqids),
            "TIGHTBPE_SILHOUETTE_R3": rate_with_cluster(bird_data, tb_silh, sqids, sdbs, bird_strict),
            "TIGHTBPE_IND8_PROF_R3": rate_with_cluster(bird_data, tb_prof, sqids, sdbs, bird_strict),
            "TIGHTBPE_STATIC_SYNTH_R3": rate_with_cluster(bird_data, tb_synth, sqids, sdbs, bird_strict),
            "TIGHTBPE_VALFIRST_SILH_R3": rate_with_cluster(bird_data, vf_silh, sqids, sdbs, bird_strict),
            "TIGHTBPE_VALFIRST_IND8_PROF_R3": rate_with_cluster(bird_data, vf_prof, sqids, sdbs, bird_strict),
            "PUBCAP_SILHOUETTE_R3": rate_with_cluster(bird_data, pubcap_silh, sqids, sdbs, bird_strict),
            "PUBCAP_IND8_PROFILE_R3": rate_with_cluster(bird_data, pubcap_prof, sqids, sdbs, bird_strict),
            "PUBCAP_IND8_ROWS_R3": rate_with_cluster(bird_data, pubcap_rows, sqids, sdbs, bird_strict),
            "DG8_CLEAN_R3": rate_with_cluster(bird_data, dg8_clean, sqids, sdbs, bird_strict),
        }

    return {
        "tightbpe": {
            "cap_formula": tman["cap_formula"],
            "tokenizer": tman["tokenizer"],
            "bpe_profile_block_stats": bpe_token_stats,
            "rates": {
                "TIGHTBPE_SILHOUETTE_R3": rate_with_cluster(bird_data, tb_silh, bird_440, bird_dbs, bird_strict, rec_by_bcq=rec_by_bcq, benchmark="bird"),
                "TIGHTBPE_IND8_PROF_R3": rate_with_cluster(bird_data, tb_prof, bird_440, bird_dbs, bird_strict, rec_by_bcq=rec_by_bcq, benchmark="bird"),
                "TIGHTBPE_STATIC_SYNTH_R3": rate_with_cluster(bird_data, tb_synth, bird_440, bird_dbs, bird_strict, rec_by_bcq=rec_by_bcq, benchmark="bird"),
                "TIGHTBPE_VALFIRST_SILH_R3": rate_with_cluster(bird_data, vf_silh, bird_440, bird_dbs, bird_strict, rec_by_bcq=rec_by_bcq, benchmark="bird"),
                "TIGHTBPE_VALFIRST_IND8_PROF_R3": rate_with_cluster(bird_data, vf_prof, bird_440, bird_dbs, bird_strict, rec_by_bcq=rec_by_bcq, benchmark="bird"),
            },
            "comparisons": {
                "TIGHTBPE_STATIC_SYNTH_minus_SILHOUETTE_3v3": full_compare(bird_data, tb_synth, tb_silh, bird_440, bird_dbs, exact_2powk=True),
                "TIGHTBPE_STATIC_SYNTH_minus_IND8_PROF_3v3": full_compare(bird_data, tb_synth, tb_prof, bird_440, bird_dbs, exact_2powk=True),
                "TIGHTBPE_IND8_PROF_minus_SILHOUETTE_3v3": full_compare(bird_data, tb_prof, tb_silh, bird_440, bird_dbs, exact_2powk=True),
                "TIGHTBPE_SILHOUETTE_minus_PUBCAP_SILHOUETTE_3v3": full_compare(bird_data, tb_silh, pubcap_silh, bird_440, bird_dbs, exact_2powk=True),
                "TIGHTBPE_STATIC_SYNTH_minus_PUBCAP_ROWS_3v3": full_compare(bird_data, tb_synth, pubcap_rows, bird_440, bird_dbs, exact_2powk=True),
                "VALFIRST_SILH_minus_OLD_TIGHTBPE_SILH_3v3": full_compare(bird_data, vf_silh, tb_silh, bird_440, bird_dbs, exact_2powk=True),
                "VALFIRST_IND8_PROF_minus_OLD_TIGHTBPE_IND8_PROF_3v3": full_compare(bird_data, vf_prof, tb_prof, bird_440, bird_dbs, exact_2powk=True),
                "VALFIRST_IND8_PROF_minus_TIGHTBPE_STATIC_SYNTH_3v3": full_compare(bird_data, vf_prof, tb_synth, bird_440, bird_dbs, exact_2powk=True),
                "VALFIRST_SILH_minus_TIGHTBPE_STATIC_SYNTH_3v3": full_compare(bird_data, vf_silh, tb_synth, bird_440, bird_dbs, exact_2powk=True),
            },
            "by_dev_exposure": dev_exposure_rates,
        },
        "dg8_noval_post": {
            "rates": {
                "DG8_NOVAL_POST_R1": rate_with_cluster(bird_data, "DG8_NOVAL_POST_R1/r0", bird_440, bird_dbs, bird_strict, rec_by_bcq=rec_by_bcq, benchmark="bird"),
                "DG8_NOVAL_POST_R2": rate_with_cluster(bird_data, "DG8_NOVAL_POST_R2/r0", bird_440, bird_dbs, bird_strict, rec_by_bcq=rec_by_bcq, benchmark="bird"),
                "DG8_NOVAL_POST_R3": rate_with_cluster(bird_data, "DG8_NOVAL_POST_R3/r0", bird_440, bird_dbs, bird_strict, rec_by_bcq=rec_by_bcq, benchmark="bird"),
                "DG8_NOVAL_POST_R23": rate_with_cluster(bird_data, noval_r23, bird_440, bird_dbs, bird_strict, rec_by_bcq=rec_by_bcq, benchmark="bird"),
                "DG8_NOVAL_POST_R1_R3": rate_with_cluster(bird_data, noval_r3, bird_440, bird_dbs, bird_strict, rec_by_bcq=rec_by_bcq, benchmark="bird"),
            },
            "comparisons": {
                "DG8_NOVAL_POST_R1_minus_DG8_CLEAN_R1_1v1": full_compare(bird_data, ["DG8_NOVAL_POST_R1/r0"], ["DG8_CLEAN_R1/r0"], bird_440, bird_dbs, exact_2powk=True),
                "DG8_NOVAL_POST_R1_R3_minus_DG8_CLEAN_R3_3v3": full_compare(bird_data, noval_r3, dg8_clean, bird_440, bird_dbs, exact_2powk=True),
            },
        },
    }


def _compute_ehrsql_v3_and_dual(
    e600_data: dict[str, dict[Any, int]],
    e874_data: dict[str, dict[Any, int]],
    ehr_600: list[str],
    ehr_874: list[str],
    ehr_tmpl_clusters: dict[str, Any],
    ehr_confirm_lit: dict[str, Any],
    rec_by_bcq: dict[tuple[str, str, Any], dict[str, Any]],
) -> dict[str, Any]:
    """Compute canonical v3 EHRSQL-600 and EHRSQL-874 stats, unseen templates, official vs uncapped parity, and disagreement concordance."""
    ehr600_tmpls = [ehr_tmpl_clusters["ehr600_by_id"][q] for q in ehr_600]
    ehr874_tmpls = [ehr_tmpl_clusters["ehr874_by_id"][q] for q in ehr_874]
    ehr874_unseen_235 = ehr_tmpl_clusters["ehr874_unseen_235_qids"]
    ehr874_seen_639 = ehr_tmpl_clusters["ehr874_seen_639_qids"]

    ehr600_v3_r3 = ["EHR_DG8P_V3_R1/r0", "EHR_DG8P_V3_R2/r0", "EHR_DG8P_V3_R3/r0"]
    ehr600_nolkp_v3_r3 = ["EHR_DG8P_NOLKP_V3_R1/r0", "EHR_DG8P_NOLKP_V3_R2/r0", "EHR_DG8P_NOLKP_V3_R3/r0"]
    ehr600_legacy_r3 = ["DG8P_R1/r0", "DG8P_R2/r0", "DG8P_R3/r0"]
    ehr874_v3_r3 = ["EHR874_DG8P_V3_R1/r0", "EHR874_DG8P_V3_R2/r0", "EHR874_DG8P_V3_R3/r0"]
    ehr874_nolkp_v3_r3 = ["EHR874_DG8P_NOLKP_V3_R1/r0", "EHR874_DG8P_NOLKP_V3_R2/r0", "EHR874_DG8P_NOLKP_V3_R3/r0"]

    def run_holm_ehr600(target_conds: list[str], specs: list[tuple[str, list[str]]]) -> list[dict[str, Any]]:
        cmps = [full_compare(e600_data, target_conds, conds, ehr_600, ehr600_tmpls, mc_cluster=True) for _lbl, conds in specs]
        for (lbl, _), r, hp_q, hp_c in zip(specs, cmps, holm([x["p"] for x in cmps]), holm([x["mc_cluster_p"] for x in cmps])):
            r["baseline"] = lbl
            r["holm_p"] = hp_q
            r["holm_cluster_p"] = hp_c
        return cmps

    # Canonical EHRSQL-600 (v3 custody-verified clinical arms primary; legacy pre-v3 quarantined in legacy_superseded_archive)
    ehr600_rates = {
        lbl: rate_with_cluster(e600_data, conds, ehr_600, ehr600_tmpls, rec_by_bcq=rec_by_bcq, benchmark="ehrsql_600")
        for lbl, conds in [
            ("Schema_only_C0", "C0/r0"),
            ("LookupOnly_C0P", "C0P/r0"),
            ("Local_32B", "Local-32B"),
            ("OmniSQL_7B", "OmniSQL-7B"),
            ("OmniSQL_14B", "OmniSQL-14B"),
            ("Silhouette_noLookup_V3_R3", ehr600_nolkp_v3_r3),
            ("Noiseless_noLookup_V3", "EHR_NOISELESS_NOLKP_V3/r0"),
            ("Noiseless_LKP_V3", "EHR_NOISELESS_LKP_V3/r0"),
            ("Synth_1way_IND8P", "IND8P/r0"),
            ("Synth_2way_SYN8P", "SYN8P/r0"),
            ("Surrogate_3", "SUR3/r0"),
            ("Select_3", "S3/r0"),
            ("1way_Profile_V3", "EHR_IND8_V3_R1/r0"),
            ("Silhouette_DG8P_V3_R3", ehr600_v3_r3),
            ("Entity_Demo94_V3", "EHR_ENT8_V3_R1/r0"),
            ("Full_access_C4", "C4/r0"),
        ]
    }
    ehr600_legacy_rates = {
        lbl: rate_with_cluster(e600_data, conds, ehr_600, ehr600_tmpls, rec_by_bcq=rec_by_bcq, benchmark="ehrsql_600")
        for lbl, conds in [
            ("Silhouette_DG8P_legacy_R3", ehr600_legacy_r3),
            ("Silhouette_noLookup_legacy", "DG8_NOLKP/r0"),
            ("1way_Profile_legacy", "EHR_IND8P_PROMPT/r0"),
            ("Entity_Demo94_legacy", "EHR_ENT8_DEMO94/r0"),
        ]
    }
    ehr600_8_specs = [
        ("Schema only (C0)", ["C0/r0"]),
        ("LookupOnly (C0P)", ["C0P/r0"]),
        ("Silhouette w/o Lookup v3 (EHR_DG8P_NOLKP_V3_R1..R3)", ehr600_nolkp_v3_r3),
        ("Synth. 1-way (IND8P)", ["IND8P/r0"]),
        ("Synth. 2-way (SYN8P)", ["SYN8P/r0"]),
        ("Select 3 (S3)", ["S3/r0"]),
        ("Surrogate 3 (SUR3)", ["SUR3/r0"]),
        ("1-way Profile v3 (EHR_IND8_V3_R1)", ["EHR_IND8_V3_R1/r0"]),
    ]
    ehr600_9_specs = ehr600_8_specs + [("Local 32B", ["Local-32B"])]
    ehr600_8_cmps = run_holm_ehr600(ehr600_v3_r3, ehr600_8_specs)
    ehr600_9_cmps = run_holm_ehr600(ehr600_v3_r3, ehr600_9_specs)
    ehr600_vs_c4 = full_compare(e600_data, ehr600_v3_r3, ["C4/r0"], ehr_600, ehr600_tmpls, mc_cluster=True)

    ehr600_clinical_2x2 = {
        "DP_Lookup_ON": ehr600_rates["Silhouette_DG8P_V3_R3"],
        "DP_Lookup_OFF": ehr600_rates["Silhouette_noLookup_V3_R3"],
        "Noiseless_Lookup_ON": ehr600_rates["Noiseless_LKP_V3"],
        "Noiseless_Lookup_OFF": ehr600_rates["Noiseless_noLookup_V3"],
        "Lookup_effect_under_DP": full_compare(e600_data, ehr600_v3_r3, ehr600_nolkp_v3_r3, ehr_600, ehr600_tmpls, mc_cluster=True),
        "DP_noise_delta_Lookup_ON": full_compare(e600_data, ehr600_v3_r3, ["EHR_NOISELESS_LKP_V3/r0"], ehr_600, ehr600_tmpls, mc_cluster=True),
        "DP_noise_delta_Lookup_OFF": full_compare(e600_data, ehr600_nolkp_v3_r3, ["EHR_NOISELESS_NOLKP_V3/r0"], ehr_600, ehr600_tmpls, mc_cluster=True),
        "Lookup_effect_under_Noiseless": full_compare(e600_data, ["EHR_NOISELESS_LKP_V3/r0"], ["EHR_NOISELESS_NOLKP_V3/r0"], ehr_600, ehr600_tmpls, mc_cluster=True),
    }

    # Confirm literal strata (136 patient-literal vs 464 non-patient-literal)
    lit_strata_600: dict[str, Any] = {}
    for sname in ("patient_literal_136", "non_patient_literal_464", "patient_literal_only_122", "dict_literal_268", "both_14"):
        qids = ehr_confirm_lit[sname]
        tmps = [ehr_tmpl_clusters["ehr600_by_id"][q] for q in qids]
        lit_strata_600[sname] = {
            "n": len(qids),
            "C0": rate_with_cluster(e600_data, "C0/r0", qids, tmps),
            "C0P": rate_with_cluster(e600_data, "C0P/r0", qids, tmps),
            "DG8P_NOLKP_V3_R3": rate_with_cluster(e600_data, ehr600_nolkp_v3_r3, qids, tmps),
            "EHR_IND8_V3_R1": rate_with_cluster(e600_data, "EHR_IND8_V3_R1/r0", qids, tmps),
            "EHR_ENT8_V3_R1": rate_with_cluster(e600_data, "EHR_ENT8_V3_R1/r0", qids, tmps),
            "EHR_DG8P_V3_R3": rate_with_cluster(e600_data, ehr600_v3_r3, qids, tmps),
            "C4": rate_with_cluster(e600_data, "C4/r0", qids, tmps),
            "DG8P_V3_minus_C0": full_compare(e600_data, ehr600_v3_r3, ["C0/r0"], qids, tmps, mc_cluster=True),
            "DG8P_V3_minus_DG8P_NOLKP_V3": full_compare(e600_data, ehr600_v3_r3, ehr600_nolkp_v3_r3, qids, tmps, mc_cluster=True),
            "DG8P_V3_minus_C4": full_compare(e600_data, ehr600_v3_r3, ["C4/r0"], qids, tmps, mc_cluster=True),
        }

    # Canonical EHRSQL-874 v3
    ehr874_rates = {
        lbl: rate_with_cluster(
            e874_data, cond, ehr_874, ehr874_tmpls,
            rec_by_bcq=rec_by_bcq, benchmark="ehrsql_874", per_rep_cluster=True,
        )
        for lbl, cond in [
            ("EHR874_C0", "EHR874_C0/r0"),
            ("EHR874_DG8P_NOLKP_V3_R3", ehr874_nolkp_v3_r3),
            ("EHR874_NOISELESS_NOLKP_V3", "EHR874_NOISELESS_NOLKP_V3/r0"),
            ("EHR874_NOISELESS_LKP_V3", "EHR874_NOISELESS_LKP_V3/r0"),
            ("EHR874_DG8P_V3_R3", ehr874_v3_r3),
            ("EHR874_DG8P_V3_R1", "EHR874_DG8P_V3_R1/r0"),
            ("EHR874_DG8P_V3_R2", "EHR874_DG8P_V3_R2/r0"),
            ("EHR874_DG8P_V3_R3_rep3", "EHR874_DG8P_V3_R3/r0"),
            ("EHR874_C4", "EHR874_C4/r0"),
        ]
    }
    ehr874_legacy_rates = {
        lbl: rate_with_cluster(e874_data, cond, ehr_874, ehr874_tmpls, rec_by_bcq=rec_by_bcq, benchmark="ehrsql_874")
        for lbl, cond in [
            ("EHR874_DG8P_legacy_R1", "EHR874_DG8P_R1/r0"),
            ("EHR874_DG8_NOLKP_legacy_R1", "EHR874_DG8_NOLKP_R1/r0"),
            ("EHR874_NOISELESS_NOLKP_legacy", "EHR874_NOISELESS_NOLKP/r0"),
        ]
    }
    ehr874_cmps = {
        key: full_compare(e874_data, ca, cb, ehr_874, ehr874_tmpls, mc_cluster=True)
        for key, ca, cb in [
            ("DG8P_V3_R3_minus_C0", ehr874_v3_r3, ["EHR874_C0/r0"]),
            ("DG8P_V3_R3_minus_C4", ehr874_v3_r3, ["EHR874_C4/r0"]),
            ("DG8P_V3_R3_minus_DG8P_NOLKP_V3_R3", ehr874_v3_r3, ehr874_nolkp_v3_r3),
            ("DG8P_V3_R3_minus_NOISELESS_LKP_V3", ehr874_v3_r3, ["EHR874_NOISELESS_LKP_V3/r0"]),
            ("DG8P_NOLKP_V3_R3_minus_NOISELESS_NOLKP_V3", ehr874_nolkp_v3_r3, ["EHR874_NOISELESS_NOLKP_V3/r0"]),
            ("NOISELESS_LKP_V3_minus_NOISELESS_NOLKP_V3", ["EHR874_NOISELESS_LKP_V3/r0"], ["EHR874_NOISELESS_NOLKP_V3/r0"]),
            ("DG8P_V3_R3_minus_NOISELESS_NOLKP_V3", ehr874_v3_r3, ["EHR874_NOISELESS_NOLKP_V3/r0"]),
            ("NOISELESS_NOLKP_V3_minus_DG8P_NOLKP_V3_R3", ["EHR874_NOISELESS_NOLKP_V3/r0"], ehr874_nolkp_v3_r3),
            ("DG8P_NOLKP_V3_R3_minus_C0", ehr874_nolkp_v3_r3, ["EHR874_C0/r0"]),
        ]
    }
    ehr874_clinical_2x2 = {
        "DP_Lookup_ON": ehr874_rates["EHR874_DG8P_V3_R3"],
        "DP_Lookup_OFF": ehr874_rates["EHR874_DG8P_NOLKP_V3_R3"],
        "Noiseless_Lookup_ON": ehr874_rates["EHR874_NOISELESS_LKP_V3"],
        "Noiseless_Lookup_OFF": ehr874_rates["EHR874_NOISELESS_NOLKP_V3"],
        "Lookup_effect_under_DP": ehr874_cmps["DG8P_V3_R3_minus_DG8P_NOLKP_V3_R3"],
        "DP_noise_delta_Lookup_ON": ehr874_cmps["DG8P_V3_R3_minus_NOISELESS_LKP_V3"],
        "DP_noise_delta_Lookup_OFF": ehr874_cmps["DG8P_NOLKP_V3_R3_minus_NOISELESS_NOLKP_V3"],
        "Lookup_effect_under_Noiseless": ehr874_cmps["NOISELESS_LKP_V3_minus_NOISELESS_NOLKP_V3"],
    }
    ehr874_unseen_seen: dict[str, Any] = {}
    for sname, sqids in [("all_874", ehr_874), ("unseen_37_templates", ehr874_unseen_235), ("seen_97_templates", ehr874_seen_639)]:
        stmpls = [ehr_tmpl_clusters["ehr874_by_id"][q] for q in sqids]
        ehr874_unseen_seen[sname] = {
            "n_queries": len(sqids),
            "n_templates": len(set(stmpls)),
            "rates": {
                lbl: rate_with_cluster(e874_data, cond, sqids, stmpls, rec_by_bcq=rec_by_bcq, benchmark="ehrsql_874")
                for lbl, cond in [
                    ("C0", "EHR874_C0/r0"),
                    ("DG8P_NOLKP_V3_R3", ehr874_nolkp_v3_r3),
                    ("NOISELESS_NOLKP_V3", "EHR874_NOISELESS_NOLKP_V3/r0"),
                    ("NOISELESS_LKP_V3", "EHR874_NOISELESS_LKP_V3/r0"),
                    ("DG8P_V3_R3", ehr874_v3_r3),
                    ("DG8P_V3_R1", "EHR874_DG8P_V3_R1/r0"),
                    ("DG8P_V3_R2", "EHR874_DG8P_V3_R2/r0"),
                    ("DG8P_V3_R3_rep3", "EHR874_DG8P_V3_R3/r0"),
                    ("C4", "EHR874_C4/r0"),
                ]
            },
            "DG8P_V3_R3_minus_C0": full_compare(e874_data, ehr874_v3_r3, ["EHR874_C0/r0"], sqids, stmpls, mc_cluster=True),
            "DG8P_V3_R3_minus_C4": full_compare(e874_data, ehr874_v3_r3, ["EHR874_C4/r0"], sqids, stmpls, mc_cluster=True),
            "DG8P_V3_R3_minus_DG8P_NOLKP_V3_R3": full_compare(e874_data, ehr874_v3_r3, ehr874_nolkp_v3_r3, sqids, stmpls, mc_cluster=True),
            "NOISELESS_V3_minus_DG8P_NOLKP_V3_R3": full_compare(e874_data, ["EHR874_NOISELESS_NOLKP_V3/r0"], ehr874_nolkp_v3_r3, sqids, stmpls, mc_cluster=True),
        }

    # Check official_ex vs uncapped_ex parity across >100-row gold questions
    ehr_v3_disagree = json.loads((ROOT / "stats/ehrsql_v3_disagreement_audit.json").read_text(encoding="utf-8"))
    ehr874_legacy_audit = json.loads((ROOT / "stats/ehrsql_874_disagreement_audit.json").read_text(encoding="utf-8"))

    # Distinct >100-row gold questions and total slots verified from receipts
    gt100_qids_600 = set()
    gt100_qids_874 = set()
    uncapped_mismatches = 0
    for (bm, _cond, qid), r in rec_by_bcq.items():
        if bm in ("ehrsql_600", "ehrsql_874"):
            if r.get("uncapped_ex") is not None and r.get("official_ex") is not None:
                if int(r["uncapped_ex"]) != int(r["official_ex"]):
                    uncapped_mismatches += 1
            if r.get("gold_rows_gt_100") or r.get("gold_len_gt_100"):
                if bm == "ehrsql_600":
                    gt100_qids_600.add(qid)
                else:
                    gt100_qids_874.add(qid)
    # Count distinct >100-row gold questions directly via manifests + existing score_ehrsql_official logic if flag omitted
    from stats import score_ehrsql_official as seo
    seo_res = seo.run_self_test()

    return {
        "ehrsql_600": {
            "n_templates": ehr_tmpl_clusters["ehr600_n_templates"],
            "rates": ehr600_rates,
            "holm_8_baseline_family": ehr600_8_cmps,
            "holm_9_baseline_family": ehr600_9_cmps,
            "vs_C4": ehr600_vs_c4,
            "clinical_2x2_lookup_x_dp": ehr600_clinical_2x2,
            "confirm_literal_strata": lit_strata_600,
            "legacy_superseded_archive": ehr600_legacy_rates,
        },
        "ehrsql_874": {
            "n_templates": ehr_tmpl_clusters["ehr874_n_templates"],
            "rates": ehr874_rates,
            "comparisons": ehr874_cmps,
            "clinical_2x2_lookup_x_dp": ehr874_clinical_2x2,
            "unseen_vs_seen_templates": ehr874_unseen_seen,
            "legacy_superseded_archive": ehr874_legacy_rates,
        },
        "disagreement_rulesets": {
            "v3_canonical_vs_C4": ehr_v3_disagree,
            "legacy_874_R1_vs_C4": {
                k: ehr874_legacy_audit[k]
                for k in (
                    "total_874", "both_win", "both_lose", "dg8p_only_wins", "c4_only_wins",
                    "gold_empty_count", "gold_gt100_count", "gold_eq100_count",
                    "dg8p_gt100_count_all874", "c4_gt100_count_all874",
                    "annotator1_summary", "annotator2_summary", "inter_annotator_agreement",
                )
            },
        },
        "official_vs_uncapped_ex_parity": {
            "ehrsql_conditions_verified": seo_res["ehrsql_conditions_verified"],
            "ehrsql_slots_verified": seo_res["ehrsql_slots_verified"],
            "official_vs_uncapped_mismatches": uncapped_mismatches,
            "gt100_distinct_questions_total": seo_res["gt100_distinct_questions"],
            "gt100_confirmation_distinct_questions": len(seo.GT100_GOLD_QIDS_600),
            "gt100_heldout_distinct_questions": len(seo.GT100_GOLD_QIDS_874),
            "gt100_slots_verified": seo_res["gt100_slots_verified"],
        },
    }


def _compute_stage10_valfirst_and_fetch1(
    bird_data: dict[str, dict[Any, int]],
    bird_deliv: dict[str, dict[Any, int]],
    bird_strict: dict[str, dict[Any, int]],
    bird_440: list[int],
    bird_dbs: list[str],
    rec_by_bcq: dict[tuple[str, str, Any], dict[str, Any]],
) -> dict[str, Any]:
    """Compute Stage-10 TightBPE-ValFirst & Fetch-All-Once (Fetch1) pooled rates and all 8 paired comparisons."""
    vf_silh = ["TIGHTBPE_VALFIRST_SILH_R1/r0", "TIGHTBPE_VALFIRST_SILH_R2/r0", "TIGHTBPE_VALFIRST_SILH_R3/r0"]
    vf_prof = ["TIGHTBPE_VALFIRST_IND8_PROF_R1/r0", "TIGHTBPE_VALFIRST_IND8_PROF_R2/r0", "TIGHTBPE_VALFIRST_IND8_PROF_R3/r0"]
    fetch1 = ["FETCH1_PUBCAP_IND8_PROF_R1/r0", "FETCH1_PUBCAP_IND8_PROF_R2/r0", "FETCH1_PUBCAP_IND8_PROF_R3/r0"]
    tb_silh = ["TIGHTBPE_SILHOUETTE_R1/r0", "TIGHTBPE_SILHOUETTE_R2/r0", "TIGHTBPE_SILHOUETTE_R3/r0"]
    tb_prof = ["TIGHTBPE_IND8_PROF_R1/r0", "TIGHTBPE_IND8_PROF_R2/r0", "TIGHTBPE_IND8_PROF_R3/r0"]
    tb_synth = ["TIGHTBPE_STATIC_SYNTH_R1/r0", "TIGHTBPE_STATIC_SYNTH_R2/r0", "TIGHTBPE_STATIC_SYNTH_R3/r0"]
    pubcap_prof = ["PUBCAP_IND8_PROFILE_R1/r0", "PUBCAP_IND8_PROFILE_R2/r0", "PUBCAP_IND8_PROFILE_R3/r0"]
    prof_tool = ["IND8_PROFILE_TOOL_R1/r0", "IND8_PROFILE_TOOL_R2/r0", "IND8_PROFILE_TOOL_R3/r0"]
    pubcap_rows = ["PUBCAP_IND8_ROWS_R1/r0", "PUBCAP_IND8_ROWS_R2/r0", "PUBCAP_IND8_ROWS_R3/r0"]

    arm_map = {
        "TIGHTBPE_VALFIRST_SILH_R1..R3": vf_silh,
        "TIGHTBPE_VALFIRST_IND8_PROF_R1..R3": vf_prof,
        "FETCH1_PUBCAP_IND8_PROF_R1..R3": fetch1,
        "TIGHTBPE_SILHOUETTE_R1..R3": tb_silh,
        "TIGHTBPE_IND8_PROF_R1..R3": tb_prof,
        "TIGHTBPE_STATIC_SYNTH_R1..R3": tb_synth,
        "PUBCAP_IND8_PROFILE_R1..R3": pubcap_prof,
        "IND8_PROFILE_TOOL_R1..R3": prof_tool,
        "PUBCAP_IND8_ROWS_R1..R3": pubcap_rows,
    }
    specs_8 = [
        ("ValFirst_Silh_vs_Old_TightBPE_Silh", vf_silh, tb_silh),
        ("ValFirst_Ind8Prof_vs_Old_TightBPE_Ind8Prof", vf_prof, tb_prof),
        ("ValFirst_Silh_vs_TightBPE_StaticSynth", vf_silh, tb_synth),
        ("ValFirst_Ind8Prof_vs_TightBPE_StaticSynth", vf_prof, tb_synth),
        ("ValFirst_Silh_vs_ValFirst_Ind8Prof", vf_silh, vf_prof),
        ("Fetch1_PubCap_Ind8Prof_vs_Inline_PubCap_Ind8Prof", fetch1, pubcap_prof),
        ("Fetch1_PubCap_Ind8Prof_vs_Ind8ProfileTool", fetch1, prof_tool),
        ("Fetch1_PubCap_Ind8Prof_vs_PubCap_Ind8Rows", fetch1, pubcap_rows),
    ]

    rates_out: dict[str, Any] = {}
    cmps_out: dict[str, Any] = {}
    ref_out: dict[str, Any] = {}
    for metric_name, metric_data in (("official_set_ex", bird_data), ("deliverable_set_ex", bird_deliv)):
        rates_out[metric_name] = {
            lbl: rate_with_cluster(
                metric_data, conds, bird_440, bird_dbs,
                bird_strict if metric_name == "official_set_ex" else None,
                rec_by_bcq=rec_by_bcq, benchmark="bird",
            )
            for lbl, conds in arm_map.items()
        }
        for lbl in ("FETCH1_PUBCAP_IND8_PROF_R1..R3",):
            conds = arm_map[lbl]
            f_calls = sum(int(rec_by_bcq[("bird", c, q)].get("fetch_calls") or 0) for c in conds for q in bird_440)
            rates_out[metric_name][lbl]["telemetry"]["fetch_calls_total"] = f_calls
            rates_out[metric_name][lbl]["telemetry"]["slots_with_fetch_call"] = sum(
                1 for c in conds for q in bird_440 if (rec_by_bcq[("bird", c, q)].get("fetch_calls") or 0) > 0
            )

        cmps_list = [full_compare(metric_data, ca, cb, bird_440, bird_dbs, exact_2powk=True) for _lbl, ca, cb in specs_8]
        hp8_q = holm([x["p"] for x in cmps_list])
        hp8_c = holm([x["exact_2powK_cluster_p"] for x in cmps_list])
        hp5_q = holm([x["p"] for x in cmps_list[:5]])
        hp5_c = holm([x["exact_2powK_cluster_p"] for x in cmps_list[:5]])
        hp3_q = holm([x["p"] for x in cmps_list[5:]])
        hp3_c = holm([x["exact_2powK_cluster_p"] for x in cmps_list[5:]])

        metric_cmps: dict[str, Any] = {}
        for idx, ((lbl, _, _), r, hq, hc) in enumerate(zip(specs_8, cmps_list, hp8_q, hp8_c)):
            r["comparison"] = lbl
            r["holm_8_p"] = hq
            r["holm_8_cluster_p"] = hc
            if idx < 5:
                r["holm_family"] = "valfirst_5"
                r["holm_family_p"] = hp5_q[idx]
                r["holm_family_cluster_p"] = hp5_c[idx]
            else:
                r["holm_family"] = "fetch1_3"
                r["holm_family_p"] = hp3_q[idx - 5]
                r["holm_family_cluster_p"] = hp3_c[idx - 5]
            metric_cmps[lbl] = r
        cmps_out[metric_name] = metric_cmps
        ref_out[metric_name] = {
            "Inline_PubCap_Ind8Prof_vs_Ind8ProfileTool": full_compare(
                metric_data, pubcap_prof, prof_tool, bird_440, bird_dbs, exact_2powk=True
            )
        }

    return {
        "pooled_arm_rates": rates_out,
        "paired_comparisons_8": cmps_out,
        "reference_comparisons": ref_out,
    }


def _compute_leave_one_db_out_sensitivity(
    bird_data: dict[str, dict[Any, int]],
    bird_440: list[int],
    bird_dbs: list[str],
) -> dict[str, Any]:
    """Compute leave-one-DB-out diff_pp min/max/range across all 11 BIRD DBs for primary comparisons."""
    unique_dbs = sorted(set(bird_dbs))
    assert len(unique_dbs) == 11, unique_dbs
    specs = [
        ("Silhouette_vs_Schema_only", ["DG8_CLEAN_R1/r0", "DG8_CLEAN_R2/r0", "DG8_CLEAN_R3/r0"], ["C0/r0", "C0/r2"]),
        ("Silhouette_vs_DP_Query", ["DG8_CLEAN_R1/r0", "DG8_CLEAN_R2/r0", "DG8_CLEAN_R3/r0"], ["DP_QUERY_ROW_TP8_V2_R1/r0", "DP_QUERY_ROW_TP8_V2_R2/r0", "DP_QUERY_ROW_TP8_V2_R3/r0"]),
        ("Silhouette_vs_Full_access", ["DG8_CLEAN_R1/r0", "DG8_CLEAN_R2/r0", "DG8_CLEAN_R3/r0"], ["C4/r0", "C4/r2"]),
        ("PubCap_Silh_vs_PubCap_Static_Synth", ["PUBCAP_SILHOUETTE_R1/r0", "PUBCAP_SILHOUETTE_R2/r0", "PUBCAP_SILHOUETTE_R3/r0"], ["PUBCAP_IND8_ROWS_R1/r0", "PUBCAP_IND8_ROWS_R2/r0", "PUBCAP_IND8_ROWS_R3/r0"]),
        ("ValFirst_Silh_vs_Old_TightBPE_Silh", ["TIGHTBPE_VALFIRST_SILH_R1/r0", "TIGHTBPE_VALFIRST_SILH_R2/r0", "TIGHTBPE_VALFIRST_SILH_R3/r0"], ["TIGHTBPE_SILHOUETTE_R1/r0", "TIGHTBPE_SILHOUETTE_R2/r0", "TIGHTBPE_SILHOUETTE_R3/r0"]),
        ("ValFirst_Ind8Prof_vs_TightBPE_StaticSynth", ["TIGHTBPE_VALFIRST_IND8_PROF_R1/r0", "TIGHTBPE_VALFIRST_IND8_PROF_R2/r0", "TIGHTBPE_VALFIRST_IND8_PROF_R3/r0"], ["TIGHTBPE_STATIC_SYNTH_R1/r0", "TIGHTBPE_STATIC_SYNTH_R2/r0", "TIGHTBPE_STATIC_SYNTH_R3/r0"]),
        ("Fetch1_vs_Ind8ProfileTool", ["FETCH1_PUBCAP_IND8_PROF_R1/r0", "FETCH1_PUBCAP_IND8_PROF_R2/r0", "FETCH1_PUBCAP_IND8_PROF_R3/r0"], ["IND8_PROFILE_TOOL_R1/r0", "IND8_PROFILE_TOOL_R2/r0", "IND8_PROFILE_TOOL_R3/r0"]),
        ("SOL_Silhouette_vs_SOL_Schema_only", ["SOL_DG8_CLEAN_R1/r0", "SOL_DG8_CLEAN_R2/r0", "SOL_DG8_CLEAN_R3/r0"], ["SOL_C0/r0"]),
    ]

    out: dict[str, Any] = {}
    for label, conds_a, conds_b in specs:
        full_a = vec(bird_data, conds_a, bird_440)
        full_b = vec(bird_data, conds_b, bird_440)
        full_diff = round(100.0 * mean(a - b for a, b in zip(full_a, full_b)), 2)
        per_db: dict[str, Any] = {}
        for drop_db in unique_dbs:
            sub_qids = [q for q, d in zip(bird_440, bird_dbs) if d != drop_db]
            va = vec(bird_data, conds_a, sub_qids)
            vb = vec(bird_data, conds_b, sub_qids)
            a_pct = round(100.0 * mean(va), 2)
            b_pct = round(100.0 * mean(vb), 2)
            d_pp = round(100.0 * mean(a - b for a, b in zip(va, vb)), 2)
            per_db[drop_db] = {
                "n_remaining": len(sub_qids),
                "a_pct": a_pct,
                "b_pct": b_pct,
                "diff_pp": d_pp,
            }
        min_db, min_rec = min(per_db.items(), key=lambda kv: kv[1]["diff_pp"])
        max_db, max_rec = max(per_db.items(), key=lambda kv: kv[1]["diff_pp"])
        out[label] = {
            "conds_a": conds_a,
            "conds_b": conds_b,
            "full_diff_pp": full_diff,
            "min_diff_pp": min_rec["diff_pp"],
            "min_dropped_db": min_db,
            "max_diff_pp": max_rec["diff_pp"],
            "max_dropped_db": max_db,
            "range_pp": round(max_rec["diff_pp"] - min_rec["diff_pp"], 2),
            "per_dropped_db": per_db,
        }
    return out


def build_all_tables() -> dict[str, Any]:
    data_bm, strict_bm, deliv_bm, dbs, receipts, rec_by_bcq = _load_outcomes_and_receipts()
    bird_data = data_bm["bird"]
    bird_deliv = deliv_bm["bird"]
    bird_strict = strict_bm["bird"]
    e600_data = data_bm["ehrsql_600"]
    e874_data = data_bm["ehrsql_874"]

    bird_man = json.loads((ROOT / "manifests/bird_minidev_2026_manifest.json").read_text(encoding="utf-8"))
    bird_440: list[int] = bird_man["evaluation"]
    bird_dbs = [dbs[q] for q in bird_440]

    ehr_man = json.loads((ROOT / "manifests/ehrsql_2024_manifest.json").read_text(encoding="utf-8"))
    ehr_600: list[str] = [x["ehrsql_id"] for x in ehr_man["confirmation"]]
    ehr_874: list[str] = ehr_man["official_test"]["held_out"]

    strata = json.loads((ROOT / "manifests/question_strata_and_remap.json").read_text(encoding="utf-8"))
    ehr_tmpl_clusters = strata["ehrsql_template_clusters"]
    ehr_confirm_lit = strata["ehrsql_confirm_literal_strata"]
    dev_strata = strata["bird_dev_exposure_strata"]

    s1_130 = strata["bird_eval_440_strata"]["S1_130"]
    s0_310 = strata["bird_eval_440_strata"]["S0_310"]
    hidden_106 = strata["gold_hidden_literal_106"]["hidden_106"]
    no_hidden_334 = strata["gold_hidden_literal_106"]["no_hidden_334"]
    parsed_base = strata["parsed_base_tables"]
    join_strata = strata["bird_eval_440_join_strata"]

    fail_50 = strata["bird_remap_130"]["remap_fail_50"]
    pred_48 = strata["remap_pred_48"]
    pass_80 = strata["bird_remap_130"]["remap_pass_80"]
    all_130 = strata["bird_remap_130"]["question_ids"]
    remap_s2_75 = strata["bird_remap_seed2"]["remap_col_slice_75"]
    remap_s2_s1_49 = strata["bird_remap_seed2"]["s1_in_remap_col_slice_49"]
    remap_s2_s0_26 = strata["bird_remap_seed2"]["s0_in_remap_col_slice_26"]

    ent_159 = strata["entity_159"]["question_ids"]
    ent_by_db = strata["entity_159"]["by_db"]
    ent_dbs = [dbs[q] for q in ent_159]

    q15_163 = strata["dp_query_subsets"]["q15_163"]
    fixed_55 = strata["fixed_55_qids"]
    sol_428 = strata["sol_denominators"]["table7_428"]
    sol_425 = strata["sol_denominators"]["clean_425"]

    dg8_clean = ["DG8_CLEAN_R1/r0", "DG8_CLEAN_R2/r0", "DG8_CLEAN_R3/r0"]
    ind8_prompt = ["IND8_PROMPT/r0", "IND8_PROMPT_R2/r0", "IND8_PROMPT_R3/r0"]
    ind8_probe_r3 = ["IND8/r0", "IND8_PROBE_R2/r0", "IND8_PROBE_R3/r0"]
    ind8_prof_tool_r3 = ["IND8_PROFILE_TOOL_R1/r0", "IND8_PROFILE_TOOL_R2/r0", "IND8_PROFILE_TOOL_R3/r0"]
    tp8_v2 = ["DP_QUERY_ROW_TP8_V2_R1/r0", "DP_QUERY_ROW_TP8_V2_R2/r0", "DP_QUERY_ROW_TP8_V2_R3/r0"]

    pubcap_silh = ["PUBCAP_SILHOUETTE_R1/r0", "PUBCAP_SILHOUETTE_R2/r0", "PUBCAP_SILHOUETTE_R3/r0"]
    pubcap_flatw = ["PUBCAP_FLATW_R1/r0", "PUBCAP_FLATW_R2/r0", "PUBCAP_FLATW_R3/r0"]
    pubcap_omega1 = ["PUBCAP_OMEGA1_EQDELTA_R1/r0", "PUBCAP_OMEGA1_EQDELTA_R2/r0", "PUBCAP_OMEGA1_EQDELTA_R3/r0"]
    pubcap_unpruned = ["PUBCAP_UNPRUNED_R1/r0", "PUBCAP_UNPRUNED_R2/r0", "PUBCAP_UNPRUNED_R3/r0"]
    pubcap_ind8_prof = ["PUBCAP_IND8_PROFILE_R1/r0", "PUBCAP_IND8_PROFILE_R2/r0", "PUBCAP_IND8_PROFILE_R3/r0"]
    pubcap_ind8_rows = ["PUBCAP_IND8_ROWS_R1/r0", "PUBCAP_IND8_ROWS_R2/r0", "PUBCAP_IND8_ROWS_R3/r0"]
    noval_r3 = ["DG8_NOVAL_POST_R1/r0", "DG8_NOVAL_POST_R2/r0", "DG8_NOVAL_POST_R3/r0"]

    bird_c0 = ["C0/r0", "C0/r2"]
    bird_c4 = ["C4/r0", "C4/r2"]

    ehr_bundle = _compute_ehrsql_v3_and_dual(
        e600_data, e874_data, ehr_600, ehr_874, ehr_tmpl_clusters, ehr_confirm_lit, rec_by_bcq
    )
    tightbpe_noval = _compute_tightbpe_and_noval(bird_data, bird_strict, bird_440, bird_dbs, dev_strata, rec_by_bcq)
    factorial_2x2 = _compute_factorial_2x2(bird_data, bird_strict, bird_440, bird_dbs, rec_by_bcq)

    # =========================================================================
    # PAPER TABLE 2: Main Benchmarks (BIRD 440, EHRSQL 600 v3, Holm Families, 2x2 Factorial)
    # =========================================================================
    dg8_vec = vec(bird_data, dg8_clean, bird_440)
    c0_vec = vec(bird_data, bird_c0, bird_440)
    c4_vec = vec(bird_data, bird_c4, bird_440)
    rlo, rhi = ratio_bootstrap(dg8_vec, c0_vec, c4_vec)
    bird_rec = (mean(dg8_vec) - mean(c0_vec)) / (mean(c4_vec) - mean(c0_vec))

    pubcap_vec = vec(bird_data, pubcap_silh, bird_440)
    prlo, prhi = ratio_bootstrap(pubcap_vec, c0_vec, c4_vec)
    pubcap_rec = (mean(pubcap_vec) - mean(c0_vec)) / (mean(c4_vec) - mean(c0_vec))

    bird_rates = {
        lbl: rate_with_cluster(bird_data, conds, bird_440, bird_dbs, bird_strict, rec_by_bcq=rec_by_bcq, benchmark="bird")
        for lbl, conds in [
            ("Schema_only_C0", bird_c0),
            ("MaskSQL_Full", "MASKSQL_FULL/r0"),
            ("MaskSQL_legacy", "MaskSQL"),
            ("Local_32B", "Local-32B"),
            ("OmniSQL_7B", "OmniSQL-7B"),
            ("OmniSQL_14B", "OmniSQL-14B"),
            ("Surrogate_3", "SUR3/r0"),
            ("DP_Query_TP8_v1", "DP_QUERY_ROW_TP8_FULL/r0"),
            ("DP_Query_TP8_v2_R3", tp8_v2),
            ("DP_Query_DB8_v1", "DP_QUERY_DB8/r0"),
            ("DP_Query_DB8_v2_R1", "DP_QUERY_ROW_DB8_V2_R1/r0"),
            ("On_Demand_Profile_Tool_R3", ind8_prof_tool_r3),
            ("Synth_1way_IND8_R1", "IND8/r0"),
            ("Synth_1way_IND8_PROBE_R3", ind8_probe_r3),
            ("Synth_2way_SYN8", "SYN8/r0"),
            ("Synth_FK_DAG_R1", "SYN8_FK_DAG_R1/r0"),
            ("Select_3", "S3/r0"),
            ("PubCap_Static_Synth_R3", pubcap_ind8_rows),
            ("PubCap_1way_Profile_R3", pubcap_ind8_prof),
            ("PubCap_Unpruned_R3", pubcap_unpruned),
            ("PubCap_Omega1_EqDelta_R3", pubcap_omega1),
            ("PubCap_FlatW_R3", pubcap_flatw),
            ("PubCap_Silhouette_R3", pubcap_silh),
            ("1way_Profile_R3", ind8_prompt),
            ("Silhouette_DG8_CLEAN_R3", dg8_clean),
            ("BIRD_Noiseless_V3", "BIRD_NOISELESS_V3/r0"),
            ("Full_access_C4", bird_c4),
            ("TightBPE_Silhouette_R3", tightbpe_noval["tightbpe"]["rates"]["TIGHTBPE_SILHOUETTE_R3"]),
            ("TightBPE_1way_Profile_R3", tightbpe_noval["tightbpe"]["rates"]["TIGHTBPE_IND8_PROF_R3"]),
            ("TightBPE_Static_Synth_R3", tightbpe_noval["tightbpe"]["rates"]["TIGHTBPE_STATIC_SYNTH_R3"]),
        ]
        if isinstance(conds, (str, list))
    }
    bird_rates["TightBPE_Silhouette_R3"] = tightbpe_noval["tightbpe"]["rates"]["TIGHTBPE_SILHOUETTE_R3"]
    bird_rates["TightBPE_1way_Profile_R3"] = tightbpe_noval["tightbpe"]["rates"]["TIGHTBPE_IND8_PROF_R3"]
    bird_rates["TightBPE_Static_Synth_R3"] = tightbpe_noval["tightbpe"]["rates"]["TIGHTBPE_STATIC_SYNTH_R3"]
    bird_rates["TightBPE_ValFirst_Silhouette_R3"] = tightbpe_noval["tightbpe"]["rates"]["TIGHTBPE_VALFIRST_SILH_R3"]
    bird_rates["TightBPE_ValFirst_1way_Profile_R3"] = tightbpe_noval["tightbpe"]["rates"]["TIGHTBPE_VALFIRST_IND8_PROF_R3"]
    bird_rates["Fetch1_PubCap_1way_Profile_R3"] = rate_with_cluster(
        bird_data,
        ["FETCH1_PUBCAP_IND8_PROF_R1/r0", "FETCH1_PUBCAP_IND8_PROF_R2/r0", "FETCH1_PUBCAP_IND8_PROF_R3/r0"],
        bird_440,
        bird_dbs,
        bird_strict,
        rec_by_bcq=rec_by_bcq,
        benchmark="bird",
    )
    bird_rates["DG8_noVal_Post_R3"] = tightbpe_noval["dg8_noval_post"]["rates"]["DG8_NOVAL_POST_R1_R3"]

    def run_holm_bird(target_conds: list[str], specs: list[tuple[str, list[str]]]) -> list[dict[str, Any]]:
        cmps = [full_compare(bird_data, target_conds, conds, bird_440, bird_dbs, exact_2powk=True) for _lbl, conds in specs]
        for (lbl, _), r, hp_q, hp_c in zip(specs, cmps, holm([x["p"] for x in cmps]), holm([x["exact_2powK_cluster_p"] for x in cmps])):
            r["baseline"] = lbl
            r["holm_p"] = hp_q
            r["holm_cluster_p"] = hp_c
        return cmps

    bird_9_dp_specs = [
        ("MASKSQL_FULL", ["MASKSQL_FULL/r0"]),
        ("SUR3", ["SUR3/r0"]),
        ("DP_QUERY_ROW_TP8_V2 (R1..R3)", tp8_v2),
        ("DP_QUERY_ROW_DB8_V2_R1", ["DP_QUERY_ROW_DB8_V2_R1/r0"]),
        ("IND8_PROFILE_TOOL (R1..R3)", ind8_prof_tool_r3),
        ("IND8_PROBE (R1..R3)", ind8_probe_r3),
        ("SYN8_FK_DAG_R1", ["SYN8_FK_DAG_R1/r0"]),
        ("PUBCAP_IND8_ROWS (R1..R3)", pubcap_ind8_rows),
        ("PUBCAP_IND8_PROFILE (R1..R3)", pubcap_ind8_prof),
    ]
    bird_10_incl_c0_specs = [("C0 (r0, r2)", bird_c0)] + bird_9_dp_specs
    bird_all_specs = bird_10_incl_c0_specs + [
        ("SYN8", ["SYN8/r0"]),
        ("S3", ["S3/r0"]),
        ("IND8_PROMPT (R1..R3)", ind8_prompt),
    ]

    paper_table2_main = {
        "bird_440": {
            "rates": bird_rates,
            "recovery_pct": round(100.0 * bird_rec, 2),
            "recovery_ci95_pct": [round(100.0 * rlo, 2), round(100.0 * rhi, 2)],
            "pubcap_silhouette_recovery_pct": round(100.0 * pubcap_rec, 2),
            "pubcap_silhouette_recovery_ci95_pct": [round(100.0 * prlo, 2), round(100.0 * prhi, 2)],
            "holm_9_dp_baselines_vs_silhouette": run_holm_bird(dg8_clean, bird_9_dp_specs),
            "holm_10_incl_c0_vs_silhouette": run_holm_bird(dg8_clean, bird_10_incl_c0_specs),
            "holm_13_all_baselines_vs_silhouette": run_holm_bird(dg8_clean, bird_all_specs),
            "holm_9_dp_baselines_vs_pubcap_silhouette": run_holm_bird(pubcap_silh, bird_9_dp_specs),
            "holm_10_incl_c0_vs_pubcap_silhouette": run_holm_bird(pubcap_silh, bird_10_incl_c0_specs),
            "vs_C4": full_compare(bird_data, dg8_clean, bird_c4, bird_440, bird_dbs, exact_2powk=True),
            "pubcap_silhouette_vs_C4": full_compare(bird_data, pubcap_silh, bird_c4, bird_440, bird_dbs, exact_2powk=True),
            "leak_free_2x2_factorial": factorial_2x2,
        },
        "ehrsql_600": ehr_bundle["ehrsql_600"],
    }

    # =========================================================================
    # PAPER TABLE 3: Uniform DB-Deduplicated TransmissionLedger Audit
    # =========================================================================
    paper_table3_audit = _compute_table3_audit(receipts)

    # =========================================================================
    # PAPER TABLE 4: Budget Sweep, Equal Public Cap (R=3), TightBPE, & Uncapped Ablations (incl. noVal R=3)
    # =========================================================================
    pubcap_arms = {
        "PUBCAP_SILHOUETTE_R3": bird_rates["PubCap_Silhouette_R3"],
        "PUBCAP_FLATW_R3": bird_rates["PubCap_FlatW_R3"],
        "PUBCAP_OMEGA1_EQDELTA_R3": bird_rates["PubCap_Omega1_EqDelta_R3"],
        "PUBCAP_UNPRUNED_R3": bird_rates["PubCap_Unpruned_R3"],
        "PUBCAP_IND8_PROFILE_R3": bird_rates["PubCap_1way_Profile_R3"],
        "PUBCAP_IND8_ROWS_R3": bird_rates["PubCap_Static_Synth_R3"],
    }
    pubcap_specs = [
        ("PUBCAP_FLATW_R3", pubcap_flatw),
        ("PUBCAP_OMEGA1_EQDELTA_R3", pubcap_omega1),
        ("PUBCAP_UNPRUNED_R3", pubcap_unpruned),
        ("PUBCAP_IND8_PROFILE_R3", pubcap_ind8_prof),
        ("PUBCAP_IND8_ROWS_R3", pubcap_ind8_rows),
    ]
    pubcap_cmps = run_holm_bird(pubcap_silh, pubcap_specs)

    budget_curve = {
        str(eps): rate_with_cluster(bird_data, cond, bird_440, bird_dbs, bird_strict, rec_by_bcq=rec_by_bcq, benchmark="bird")
        for eps, cond in [(1, "DG1_CLEAN_R1/r0"), (2, "DG2_CLEAN_R1/r0"), (4, "DG4_CLEAN_R1/r0"), (8, "DG8_CLEAN_R1/r0")]
    }
    arms_clean = ("DG1_CLEAN_R1/r0", "DG2_CLEAN_R1/r0", "DG4_CLEAN_R1/r0", "DG8_CLEAN_R1/r0")
    z_tr, p_tr = paired_trend([[bird_data[a][q] for a in arms_clean] for q in bird_440], (1, 2, 4, 8))

    def _holm_abl_family(specs: list[tuple[str, list[str], list[str]]]) -> list[dict[str, Any]]:
        cmps = [full_compare(bird_data, ca, cb, bird_440, bird_dbs, exact_2powk=True) for _lbl, ca, cb in specs]
        for (lbl, _, _), r, hp_q, hp_c in zip(specs, cmps, holm([x["p"] for x in cmps]), holm([x["exact_2powK_cluster_p"] for x in cmps])):
            r["ablation"] = lbl
            r["holm_p"] = hp_q
            r["holm_2pow11_cluster_p"] = hp_c
        return cmps

    abl_specs_v7 = [
        ("DG8_NOVAL_POST_R1 (1-vs-1)", ["DG8_NOVAL_POST_R1/r0"], ["DG8_CLEAN_R1/r0"]),
        ("DG8_FLATW (R1..R3, 3-vs-3)", ["DG8_FLATW/r0", "DG8_FLATW_R2/r0", "DG8_FLATW_R3/r0"], dg8_clean),
        ("DG8_OMEGA1 (R1..R3, 3-vs-3)", ["DG8_OMEGA1_R1/r0", "DG8_OMEGA1_R2/r0", "DG8_OMEGA1_R3/r0"], dg8_clean),
        ("DG8_UNPRUNED_V2 (R1..R3, 3-vs-3)", ["DG8_UNPRUNED_V2_R1/r0", "DG8_UNPRUNED_V2_R2/r0", "DG8_UNPRUNED_V2_R3/r0"], dg8_clean),
        ("DG8_NOSANDBOX (1-vs-1)", ["DG8_NOSANDBOX/r0"], ["DG8_CLEAN_R1/r0"]),
    ]
    abl_specs_noval_r3 = [
        ("DG8_NOVAL_POST (R1..R3, 3-vs-3)", noval_r3, dg8_clean),
        ("DG8_FLATW (R1..R3, 3-vs-3)", ["DG8_FLATW/r0", "DG8_FLATW_R2/r0", "DG8_FLATW_R3/r0"], dg8_clean),
        ("DG8_OMEGA1 (R1..R3, 3-vs-3)", ["DG8_OMEGA1_R1/r0", "DG8_OMEGA1_R2/r0", "DG8_OMEGA1_R3/r0"], dg8_clean),
        ("DG8_UNPRUNED_V2 (R1..R3, 3-vs-3)", ["DG8_UNPRUNED_V2_R1/r0", "DG8_UNPRUNED_V2_R2/r0", "DG8_UNPRUNED_V2_R3/r0"], dg8_clean),
        ("DG8_NOSANDBOX (1-vs-1)", ["DG8_NOSANDBOX/r0"], ["DG8_CLEAN_R1/r0"]),
    ]

    ehr_v3_r3_list = ["EHR_DG8P_V3_R1/r0", "EHR_DG8P_V3_R2/r0", "EHR_DG8P_V3_R3/r0"]
    ehr_nolkp_v3_r3_list = ["EHR_DG8P_NOLKP_V3_R1/r0", "EHR_DG8P_NOLKP_V3_R2/r0", "EHR_DG8P_NOLKP_V3_R3/r0"]
    ehr874_v3_r3_list = ["EHR874_DG8P_V3_R1/r0", "EHR874_DG8P_V3_R2/r0", "EHR874_DG8P_V3_R3/r0"]
    ehr874_nolkp_v3_r3_list = ["EHR874_DG8P_NOLKP_V3_R1/r0", "EHR874_DG8P_NOLKP_V3_R2/r0", "EHR874_DG8P_NOLKP_V3_R3/r0"]
    paper_table4_ablation = {
        "block_a_equal_public_cap_R3": {
            "cap_formula": "200 + 20 * |T| + 30 * |C|",
            "rates": pubcap_arms,
            "comparisons_vs_PUBCAP_SILHOUETTE_R3": pubcap_cmps,
        },
        "block_b_uncapped_ablations": {
            "budget_curve_R1": budget_curve,
            "paired_trend_1_2_4_8": {"z": round(z_tr, 6), "p": p_tr},
            "natural_rates": {
                "DG8_CLEAN_R3": bird_rates["Silhouette_DG8_CLEAN_R3"],
                "DG8_UNPRUNED_V2_R3": rate_with_cluster(bird_data, ["DG8_UNPRUNED_V2_R1/r0", "DG8_UNPRUNED_V2_R2/r0", "DG8_UNPRUNED_V2_R3/r0"], bird_440, bird_dbs, bird_strict, rec_by_bcq=rec_by_bcq, benchmark="bird"),
                "IND8_PROMPT_R3": bird_rates["1way_Profile_R3"],
                "DG8_OMEGA1_R3": rate_with_cluster(bird_data, ["DG8_OMEGA1_R1/r0", "DG8_OMEGA1_R2/r0", "DG8_OMEGA1_R3/r0"], bird_440, bird_dbs, bird_strict, rec_by_bcq=rec_by_bcq, benchmark="bird"),
                "DG8_FLATW_R3": rate_with_cluster(bird_data, ["DG8_FLATW/r0", "DG8_FLATW_R2/r0", "DG8_FLATW_R3/r0"], bird_440, bird_dbs, bird_strict, rec_by_bcq=rec_by_bcq, benchmark="bird"),
                "DG8_NOVAL_POST_R1": tightbpe_noval["dg8_noval_post"]["rates"]["DG8_NOVAL_POST_R1"],
                "DG8_NOVAL_POST_R1_R3": tightbpe_noval["dg8_noval_post"]["rates"]["DG8_NOVAL_POST_R1_R3"],
                "DG8_NOSANDBOX_R1": rate_with_cluster(bird_data, "DG8_NOSANDBOX/r0", bird_440, bird_dbs, bird_strict, rec_by_bcq=rec_by_bcq, benchmark="bird"),
                "BIRD_NOISELESS_V3": bird_rates["BIRD_Noiseless_V3"],
                "DG8_NOISELESS_legacy": rate_with_cluster(bird_data, "DG8_NOISELESS/r0", bird_440, bird_dbs, bird_strict, rec_by_bcq=rec_by_bcq, benchmark="bird"),
            },
            "holm_5_ablation_family_noval_r3": _holm_abl_family(abl_specs_noval_r3),
            "holm_5_ablation_family_noval_r1": _holm_abl_family(abl_specs_v7),
            "supplementary_ablations": {
                "BIRD_NOISELESS_V3_minus_DG8_CLEAN_R1": full_compare(bird_data, ["BIRD_NOISELESS_V3/r0"], ["DG8_CLEAN_R1/r0"], bird_440, bird_dbs, exact_2powk=True),
                "BIRD_NOISELESS_V3_minus_DG8_CLEAN_R3": full_compare(bird_data, ["BIRD_NOISELESS_V3/r0"], dg8_clean, bird_440, bird_dbs, exact_2powk=True),
                "DG8_OMEGA1_minus_DG8_FLATW_3v3": full_compare(bird_data, ["DG8_OMEGA1_R1/r0", "DG8_OMEGA1_R2/r0", "DG8_OMEGA1_R3/r0"], ["DG8_FLATW/r0", "DG8_FLATW_R2/r0", "DG8_FLATW_R3/r0"], bird_440, bird_dbs, exact_2powk=True),
            },
        },
        "block_c_tightbpe_qwen25_cap": tightbpe_noval["tightbpe"],
        "block_d_ehrsql600_v3_ablations": {
            "Schema_only_C0_minus_Silh_V3": full_compare(e600_data, ["C0/r0"], ehr_v3_r3_list, ehr_600, [ehr_tmpl_clusters["ehr600_by_id"][q] for q in ehr_600], mc_cluster=True),
            "Silhouette_noLookup_V3_minus_Silh_V3": full_compare(e600_data, ehr_nolkp_v3_r3_list, ehr_v3_r3_list, ehr_600, [ehr_tmpl_clusters["ehr600_by_id"][q] for q in ehr_600], mc_cluster=True),
            "Noiseless_LKP_V3_minus_Silh_V3": full_compare(e600_data, ["EHR_NOISELESS_LKP_V3/r0"], ehr_v3_r3_list, ehr_600, [ehr_tmpl_clusters["ehr600_by_id"][q] for q in ehr_600], mc_cluster=True),
            "Noiseless_noLookup_V3_minus_Silh_noLookup_V3": full_compare(e600_data, ["EHR_NOISELESS_NOLKP_V3/r0"], ehr_nolkp_v3_r3_list, ehr_600, [ehr_tmpl_clusters["ehr600_by_id"][q] for q in ehr_600], mc_cluster=True),
            "LookupOnly_C0P_minus_Silh_V3": full_compare(e600_data, ["C0P/r0"], ehr_v3_r3_list, ehr_600, [ehr_tmpl_clusters["ehr600_by_id"][q] for q in ehr_600], mc_cluster=True),
            "1way_Profile_V3_minus_Silh_V3": full_compare(e600_data, ["EHR_IND8_V3_R1/r0"], ehr_v3_r3_list, ehr_600, [ehr_tmpl_clusters["ehr600_by_id"][q] for q in ehr_600], mc_cluster=True),
        },
        "same_source_1way_dp_2x2_factorial": factorial_2x2,
    }

    # =========================================================================
    # PAPER TABLE 5 & TR TABLE 17: Turn-Ordered Sandboxed SQL Validation & Error Reconciliation
    # =========================================================================
    t5_authoritative = json.loads((ROOT / "stats/table5_table13_authoritative.json").read_text(encoding="utf-8"))
    turn_ordered_rows = {
        # BIRD (glm-5.3)
        "BIRD_Schema_only_C0": _turn_ordered_sandbox_row(rec_by_bcq, "bird", bird_c0, bird_440, "BIRD Schema only (C0 r0+r2)"),
        "BIRD_Silhouette_Clean_R3": _turn_ordered_sandbox_row(rec_by_bcq, "bird", dg8_clean, bird_440, "BIRD Silhouette Clean, R=3 (DG8_CLEAN_R1..R3)"),
        "BIRD_DG1_CLEAN_R1": _turn_ordered_sandbox_row(rec_by_bcq, "bird", ["DG1_CLEAN_R1/r0"], bird_440, "BIRD Silhouette eps=1 (DG1_CLEAN_R1)"),
        "BIRD_DG2_CLEAN_R1": _turn_ordered_sandbox_row(rec_by_bcq, "bird", ["DG2_CLEAN_R1/r0"], bird_440, "BIRD Silhouette eps=2 (DG2_CLEAN_R1)"),
        "BIRD_DG4_CLEAN_R1": _turn_ordered_sandbox_row(rec_by_bcq, "bird", ["DG4_CLEAN_R1/r0"], bird_440, "BIRD Silhouette eps=4 (DG4_CLEAN_R1)"),
        "BIRD_Silhouette_noSandbox": _turn_ordered_sandbox_row(rec_by_bcq, "bird", ["DG8_NOSANDBOX/r0"], bird_440, "BIRD Silhouette-noSandbox"),
        "BIRD_Select_3": _turn_ordered_sandbox_row(rec_by_bcq, "bird", ["S3/r0"], bird_440, "BIRD Select 3 (S3)"),
        "BIRD_Full_access_C4": _turn_ordered_sandbox_row(rec_by_bcq, "bird", bird_c4, bird_440, "BIRD Full access (C4 r0+r2)"),
        "BIRD_Noiseless_V3": _turn_ordered_sandbox_row(rec_by_bcq, "bird", ["BIRD_NOISELESS_V3/r0"], bird_440, "BIRD Noiseless v3"),
        "BIRD_PubCap_Silhouette_R3": _turn_ordered_sandbox_row(rec_by_bcq, "bird", pubcap_silh, bird_440, "BIRD PubCap Silhouette R=3"),
        "BIRD_noVal_Post_R1": _turn_ordered_sandbox_row(rec_by_bcq, "bird", ["DG8_NOVAL_POST_R1/r0"], bird_440, "BIRD Silhouette-noVal R1"),
        "BIRD_noVal_Post_R23": _turn_ordered_sandbox_row(rec_by_bcq, "bird", ["DG8_NOVAL_POST_R2/r0", "DG8_NOVAL_POST_R3/r0"], bird_440, "BIRD Silhouette-noVal R2..R3"),
        "BIRD_TightBPE_Silhouette_R3": _turn_ordered_sandbox_row(rec_by_bcq, "bird", ["TIGHTBPE_SILHOUETTE_R1/r0", "TIGHTBPE_SILHOUETTE_R2/r0", "TIGHTBPE_SILHOUETTE_R3/r0"], bird_440, "BIRD TightBPE Silhouette R=3"),
        "BIRD_TightBPE_1way_Profile_R3": _turn_ordered_sandbox_row(rec_by_bcq, "bird", ["TIGHTBPE_IND8_PROF_R1/r0", "TIGHTBPE_IND8_PROF_R2/r0", "TIGHTBPE_IND8_PROF_R3/r0"], bird_440, "BIRD TightBPE 1-way Profile R=3"),
        "BIRD_TightBPE_Static_Synth_R3": _turn_ordered_sandbox_row(rec_by_bcq, "bird", ["TIGHTBPE_STATIC_SYNTH_R1/r0", "TIGHTBPE_STATIC_SYNTH_R2/r0", "TIGHTBPE_STATIC_SYNTH_R3/r0"], bird_440, "BIRD TightBPE Static Synth R=3"),
        "BIRD_TightBPE_ValFirst_Silhouette_R3": _turn_ordered_sandbox_row(rec_by_bcq, "bird", ["TIGHTBPE_VALFIRST_SILH_R1/r0", "TIGHTBPE_VALFIRST_SILH_R2/r0", "TIGHTBPE_VALFIRST_SILH_R3/r0"], bird_440, "BIRD TightBPE ValFirst Silhouette R=3"),
        "BIRD_TightBPE_ValFirst_1way_Profile_R3": _turn_ordered_sandbox_row(rec_by_bcq, "bird", ["TIGHTBPE_VALFIRST_IND8_PROF_R1/r0", "TIGHTBPE_VALFIRST_IND8_PROF_R2/r0", "TIGHTBPE_VALFIRST_IND8_PROF_R3/r0"], bird_440, "BIRD TightBPE ValFirst 1-way Profile R=3"),
        "BIRD_Fetch1_PubCap_1way_Profile_R3": _turn_ordered_sandbox_row(rec_by_bcq, "bird", ["FETCH1_PUBCAP_IND8_PROF_R1/r0", "FETCH1_PUBCAP_IND8_PROF_R2/r0", "FETCH1_PUBCAP_IND8_PROF_R3/r0"], bird_440, "BIRD Fetch-All-Once PubCap 1-way Profile R=3"),
        "BIRD_SOL_DG8_CLEAN_R3": _turn_ordered_sandbox_row(rec_by_bcq, "bird", ["SOL_DG8_CLEAN_R1/r0", "SOL_DG8_CLEAN_R2/r0", "SOL_DG8_CLEAN_R3/r0"], bird_440, "BIRD gpt-6.1-sol Silhouette Clean R=3"),
        # EHRSQL-600 Canonical v3
        "EHR600_Schema_only_C0": _turn_ordered_sandbox_row(rec_by_bcq, "ehrsql_600", ["C0/r0"], ehr_600, "EHR-600 Schema only (C0)"),
        "EHR600_LookupOnly_C0P": _turn_ordered_sandbox_row(rec_by_bcq, "ehrsql_600", ["C0P/r0"], ehr_600, "EHR-600 LookupOnly (C0P)"),
        "EHR600_Silhouette_noLookup_V3_R3": _turn_ordered_sandbox_row(rec_by_bcq, "ehrsql_600", ehr_nolkp_v3_r3_list, ehr_600, "EHR-600 Silhouette w/o Lookup v3, R=3"),
        "EHR600_Silhouette_V3_R3": _turn_ordered_sandbox_row(rec_by_bcq, "ehrsql_600", ehr_v3_r3_list, ehr_600, "EHR-600 Silhouette v3, R=3"),
        "EHR600_Noiseless_LKP_V3": _turn_ordered_sandbox_row(rec_by_bcq, "ehrsql_600", ["EHR_NOISELESS_LKP_V3/r0"], ehr_600, "EHR-600 Noiseless Lookup ON v3"),
        "EHR600_Noiseless_noLookup_V3": _turn_ordered_sandbox_row(rec_by_bcq, "ehrsql_600", ["EHR_NOISELESS_NOLKP_V3/r0"], ehr_600, "EHR-600 Noiseless Lookup OFF v3"),
        "EHR600_1way_Profile_V3": _turn_ordered_sandbox_row(rec_by_bcq, "ehrsql_600", ["EHR_IND8_V3_R1/r0"], ehr_600, "EHR-600 1-way Profile v3"),
        "EHR600_Entity_Demo94_V3": _turn_ordered_sandbox_row(rec_by_bcq, "ehrsql_600", ["EHR_ENT8_V3_R1/r0"], ehr_600, "EHR-600 Entity Demo94 v3"),
        "EHR600_Full_access_C4": _turn_ordered_sandbox_row(rec_by_bcq, "ehrsql_600", ["C4/r0"], ehr_600, "EHR-600 Full access (C4)"),
        # EHRSQL-874 Held-Out Canonical v3
        "EHR874_Schema_only_C0": _turn_ordered_sandbox_row(rec_by_bcq, "ehrsql_874", ["EHR874_C0/r0"], ehr_874, "EHR-874 Schema only (C0)"),
        "EHR874_Silhouette_noLookup_V3_R3": _turn_ordered_sandbox_row(rec_by_bcq, "ehrsql_874", ehr874_nolkp_v3_r3_list, ehr_874, "EHR-874 Silhouette w/o Lookup v3, R=3"),
        "EHR874_Silhouette_V3_R3": _turn_ordered_sandbox_row(rec_by_bcq, "ehrsql_874", ehr874_v3_r3_list, ehr_874, "EHR-874 Silhouette v3, R=3"),
        "EHR874_Noiseless_LKP_V3": _turn_ordered_sandbox_row(rec_by_bcq, "ehrsql_874", ["EHR874_NOISELESS_LKP_V3/r0"], ehr_874, "EHR-874 Noiseless Lookup ON v3"),
        "EHR874_Noiseless_noLookup_V3": _turn_ordered_sandbox_row(rec_by_bcq, "ehrsql_874", ["EHR874_NOISELESS_NOLKP_V3/r0"], ehr_874, "EHR-874 Noiseless Lookup OFF v3"),
        "EHR874_Full_access_C4": _turn_ordered_sandbox_row(rec_by_bcq, "ehrsql_874", ["EHR874_C4/r0"], ehr_874, "EHR-874 Full access (C4)"),
    }
    legacy_sandbox_rows = {
        "EHR600_Silhouette_noLookup_legacy": _turn_ordered_sandbox_row(rec_by_bcq, "ehrsql_600", ["DG8_NOLKP/r0"], ehr_600, "Legacy EHR-600 noLookup (DG8_NOLKP)"),
        "EHR600_Silhouette_legacy_R3": _turn_ordered_sandbox_row(rec_by_bcq, "ehrsql_600", ["DG8P_R1/r0", "DG8P_R2/r0", "DG8P_R3/r0"], ehr_600, "Legacy EHR-600 Silhouette R=3 (DG8P_R1..R3)"),
        "EHR874_Silhouette_noLookup_legacy": _turn_ordered_sandbox_row(rec_by_bcq, "ehrsql_874", ["EHR874_DG8_NOLKP_R1/r0"], ehr_874, "Legacy EHR-874 noLookup (EHR874_DG8_NOLKP_R1)"),
    }
    paper_table5_sandbox = {
        "bird": {
            "Schema_only": turn_ordered_rows["BIRD_Schema_only_C0"],
            "Silhouette_R3": turn_ordered_rows["BIRD_Silhouette_Clean_R3"],
            "Silhouette_noSandbox": turn_ordered_rows["BIRD_Silhouette_noSandbox"],
            "Select_3": turn_ordered_rows["BIRD_Select_3"],
            "Full_access": turn_ordered_rows["BIRD_Full_access_C4"],
        },
        "ehrsql_600_v3_canonical": {
            "Schema_only": turn_ordered_rows["EHR600_Schema_only_C0"],
            "Silhouette_noLookup_V3_R3": turn_ordered_rows["EHR600_Silhouette_noLookup_V3_R3"],
            "Silhouette_V3_R3": turn_ordered_rows["EHR600_Silhouette_V3_R3"],
            "Noiseless_LKP_V3": turn_ordered_rows["EHR600_Noiseless_LKP_V3"],
            "Noiseless_noLookup_V3": turn_ordered_rows["EHR600_Noiseless_noLookup_V3"],
            "1way_Profile_V3": turn_ordered_rows["EHR600_1way_Profile_V3"],
            "Entity_Demo94_V3": turn_ordered_rows["EHR600_Entity_Demo94_V3"],
            "Full_access": turn_ordered_rows["EHR600_Full_access_C4"],
        },
        "ehrsql_874_v3_canonical": {
            "Schema_only": turn_ordered_rows["EHR874_Schema_only_C0"],
            "Silhouette_noLookup_V3_R3": turn_ordered_rows["EHR874_Silhouette_noLookup_V3_R3"],
            "Silhouette_V3_R3": turn_ordered_rows["EHR874_Silhouette_V3_R3"],
            "Noiseless_LKP_V3": turn_ordered_rows["EHR874_Noiseless_LKP_V3"],
            "Noiseless_noLookup_V3": turn_ordered_rows["EHR874_Noiseless_noLookup_V3"],
            "Full_access": turn_ordered_rows["EHR874_Full_access_C4"],
        },
        "legacy_superseded_archive": legacy_sandbox_rows,
    }

    # =========================================================================
    # PAPER TABLE 6: Adversarial Remapping (Seed 1 n=130, Seed 2 n=75)
    # =========================================================================
    remap_dg8_r3 = ["REMAP_DG8_R1/r0", "REMAP_DG8_R2/r0", "REMAP_DG8_R3/r0"]
    remap_s1_slices: dict[str, Any] = {}
    for sname, qids in [("Pred_48", pred_48), ("Fail_50", fail_50), ("Pass_80", pass_80), ("All_130", all_130)]:
        sdbs = [dbs[q] for q in qids]
        sol_c0_tok = mean(rec_by_bcq[("bird", "SOL_REMAP_C0/r0", q)]["total_tokens"] for q in qids)
        sol_i8_tok = mean(rec_by_bcq[("bird", "SOL_REMAP_IND8_PROMPT/r0", q)]["total_tokens"] for q in qids)
        sol_dg_tok = mean(rec_by_bcq[("bird", "SOL_REMAP_DG8_R1/r0", q)]["total_tokens"] for q in qids)
        sol_c4_tok = mean(rec_by_bcq[("bird", "SOL_REMAP_C4/r0", q)]["total_tokens"] for q in qids)
        remap_s1_slices[sname] = {
            "n": len(qids),
            "glm53": {
                "Orig_C0": rate_with_cluster(bird_data, bird_c0, qids, sdbs, bird_strict),
                "Orig_IND8_PROMPT_R3": rate_with_cluster(bird_data, ind8_prompt, qids, sdbs, bird_strict),
                "Orig_DG8_CLEAN_R3": rate_with_cluster(bird_data, dg8_clean, qids, sdbs, bird_strict),
                "Orig_C4": rate_with_cluster(bird_data, bird_c4, qids, sdbs, bird_strict),
                "REMAP_C0": rate_with_cluster(bird_data, "REMAP_C0/r0", qids, sdbs, bird_strict),
                "REMAP_IND8": rate_with_cluster(bird_data, "REMAP_IND8/r0", qids, sdbs, bird_strict),
                "REMAP_IND8_PROMPT": rate_with_cluster(bird_data, "REMAP_IND8_PROMPT/r0", qids, sdbs, bird_strict),
                "REMAP_DG8_R1": rate_with_cluster(bird_data, "REMAP_DG8_R1/r0", qids, sdbs, bird_strict),
                "REMAP_DG8_R3": rate_with_cluster(bird_data, remap_dg8_r3, qids, sdbs, bird_strict),
                "REMAP_C4": rate_with_cluster(bird_data, "REMAP_C4/r0", qids, sdbs, bird_strict),
                "REMAP_DG8_R3_minus_C0": full_compare(bird_data, remap_dg8_r3, ["REMAP_C0/r0"], qids, sdbs),
                "REMAP_DG8_R3_minus_REMAP_IND8_PROMPT": full_compare(bird_data, remap_dg8_r3, ["REMAP_IND8_PROMPT/r0"], qids, sdbs),
                "REMAP_DG8_R3_minus_Orig_DG8_CLEAN_R3": full_compare(bird_data, remap_dg8_r3, dg8_clean, qids, sdbs),
                "REMAP_C4_minus_Orig_C4": full_compare(bird_data, ["REMAP_C4/r0"], bird_c4, qids, sdbs),
            },
            "gpt61_sol": {
                "Orig_C0": rate_with_cluster(bird_data, "SOL_C0/r0", qids, sdbs, bird_strict),
                "Orig_IND8_PROMPT": rate_with_cluster(bird_data, "SOL_IND8_PROMPT/r0", qids, sdbs, bird_strict),
                "Orig_DG8_CLEAN_R1": rate_with_cluster(bird_data, "SOL_DG8_CLEAN_R1/r0", qids, sdbs, bird_strict),
                "Orig_C4": rate_with_cluster(bird_data, "SOL_C4/r0", qids, sdbs, bird_strict),
                "SOL_REMAP_C0": rate_with_cluster(bird_data, "SOL_REMAP_C0/r0", qids, sdbs, bird_strict),
                "SOL_REMAP_IND8_PROMPT": rate_with_cluster(bird_data, "SOL_REMAP_IND8_PROMPT/r0", qids, sdbs, bird_strict),
                "SOL_REMAP_DG8_R1": rate_with_cluster(bird_data, "SOL_REMAP_DG8_R1/r0", qids, sdbs, bird_strict),
                "SOL_REMAP_C4": rate_with_cluster(bird_data, "SOL_REMAP_C4/r0", qids, sdbs, bird_strict),
                "SOL_REMAP_DG8_minus_C0": full_compare(bird_data, "SOL_REMAP_DG8_R1/r0", "SOL_REMAP_C0/r0", qids, sdbs),
                "SOL_REMAP_C4_minus_DG8": full_compare(bird_data, "SOL_REMAP_C4/r0", "SOL_REMAP_DG8_R1/r0", qids, sdbs),
                "SOL_REMAP_DG8_minus_Orig_DG8_CLEAN_R1": full_compare(bird_data, "SOL_REMAP_DG8_R1/r0", "SOL_DG8_CLEAN_R1/r0", qids, sdbs),
                "mean_total_tokens": {
                    "SOL_REMAP_C0": round(sol_c0_tok, 1),
                    "SOL_REMAP_IND8_PROMPT": round(sol_i8_tok, 1),
                    "SOL_REMAP_DG8_R1": round(sol_dg_tok, 1),
                    "SOL_REMAP_C4": round(sol_c4_tok, 1),
                    "Silh_token_reduction_vs_Full_pct": round(100.0 * (1.0 - sol_dg_tok / sol_c4_tok), 2),
                },
            },
        }

    s2_dbs = [dbs[q] for q in remap_s2_75]

    def s2_model_block(prefix: str, qids: list[int], qdbs: list[str]) -> dict[str, Any]:
        c0_c = f"{prefix}C0/r0"
        i8p_c = f"{prefix}IND8_PROMPT_R1/r0"
        dg8_c = f"{prefix}DG8_R1/r0"
        c4_c = f"{prefix}C4/r0"
        v_c0 = vec(bird_data, c0_c, qids)
        v_dg = vec(bird_data, dg8_c, qids)
        v_c4 = vec(bird_data, c4_c, qids)
        rec_r = (mean(v_dg) - mean(v_c0)) / (mean(v_c4) - mean(v_c0)) if mean(v_c4) > mean(v_c0) else 0.0
        rlo_s2, rhi_s2 = ratio_bootstrap(v_dg, v_c0, v_c4)
        return {
            "n": len(qids),
            "C0": rate_with_cluster(bird_data, c0_c, qids, qdbs, bird_strict),
            "IND8_PROMPT_R1": rate_with_cluster(bird_data, i8p_c, qids, qdbs, bird_strict),
            "DG8_R1": rate_with_cluster(bird_data, dg8_c, qids, qdbs, bird_strict),
            "C4": rate_with_cluster(bird_data, c4_c, qids, qdbs, bird_strict),
            "recovery_ratio": round(rec_r, 4),
            "recovery_ci95": [round(rlo_s2, 4), round(rhi_s2, 4)],
            "DG8_minus_C0": full_compare(bird_data, dg8_c, c0_c, qids, qdbs),
            "IND8_PROMPT_minus_C0": full_compare(bird_data, i8p_c, c0_c, qids, qdbs),
            "C4_minus_DG8": full_compare(bird_data, c4_c, dg8_c, qids, qdbs),
        }

    paper_table6_remap = {
        "bird_remap_seed1": remap_s1_slices,
        "bird_remap_seed2_75": {
            "glm53": {
                "all_75": s2_model_block("REMAP_S2_", remap_s2_75, s2_dbs),
                "s1_49": s2_model_block("REMAP_S2_", remap_s2_s1_49, [dbs[q] for q in remap_s2_s1_49]),
                "s0_26": s2_model_block("REMAP_S2_", remap_s2_s0_26, [dbs[q] for q in remap_s2_s0_26]),
            },
            "gpt61_sol": {
                "all_75": s2_model_block("SOL_REMAP_S2_", remap_s2_75, s2_dbs),
                "s1_49": s2_model_block("SOL_REMAP_S2_", remap_s2_s1_49, [dbs[q] for q in remap_s2_s1_49]),
                "s0_26": s2_model_block("SOL_REMAP_S2_", remap_s2_s0_26, [dbs[q] for q in remap_s2_s0_26]),
            },
        },
    }

    # =========================================================================
    # PAPER TABLE 7: gpt-6.1-sol Backbone (n=440, n=428, n=425, reconciled SOL_C0 = 319/440 = 72.50%)
    # =========================================================================
    sol_dg8_clean_r3 = ["SOL_DG8_CLEAN_R1/r0", "SOL_DG8_CLEAN_R2/r0", "SOL_DG8_CLEAN_R3/r0"]
    paper_table7_sol: dict[str, Any] = {}
    for dlabel, qset in [("full_440", bird_440), ("table7_428", sol_428), ("clean_425", sol_425)]:
        sdbs = [dbs[q] for q in qset]
        sc0 = vec(bird_data, "SOL_C0/r0", qset)
        sc4 = vec(bird_data, "SOL_C4/r0", qset)
        sdg_cl3 = vec(bird_data, sol_dg8_clean_r3, qset)
        rec_cl3 = (mean(sdg_cl3) - mean(sc0)) / (mean(sc4) - mean(sc0))
        slo, shi = ratio_bootstrap(sdg_cl3, sc0, sc4)
        paper_table7_sol[dlabel] = {
            "n": len(qset),
            "SOL_C0": rate_with_cluster(bird_data, "SOL_C0/r0", qset, sdbs, bird_strict, rec_by_bcq=rec_by_bcq, benchmark="bird"),
            "SOL_PUBCAP_IND8_ROWS_R1": rate_with_cluster(bird_data, "SOL_PUBCAP_IND8_ROWS_R1/r0", qset, sdbs, bird_strict, rec_by_bcq=rec_by_bcq, benchmark="bird"),
            "SOL_IND8_STATIC_ROWS_TM_R1": rate_with_cluster(bird_data, "SOL_IND8_STATIC_ROWS_TM_R1/r0", qset, sdbs, bird_strict, rec_by_bcq=rec_by_bcq, benchmark="bird"),
            "SOL_IND8_PROBE_R1": rate_with_cluster(bird_data, "SOL_IND8_PROBE_R1/r0", qset, sdbs, bird_strict, rec_by_bcq=rec_by_bcq, benchmark="bird"),
            "SOL_IND8_PROMPT": rate_with_cluster(bird_data, "SOL_IND8_PROMPT/r0", qset, sdbs, bird_strict, rec_by_bcq=rec_by_bcq, benchmark="bird"),
            "SOL_DG8_CLEAN_R1": rate_with_cluster(bird_data, "SOL_DG8_CLEAN_R1/r0", qset, sdbs, bird_strict, rec_by_bcq=rec_by_bcq, benchmark="bird"),
            "SOL_DG8_CLEAN_R2": rate_with_cluster(bird_data, "SOL_DG8_CLEAN_R2/r0", qset, sdbs, bird_strict, rec_by_bcq=rec_by_bcq, benchmark="bird"),
            "SOL_DG8_CLEAN_R3": rate_with_cluster(bird_data, "SOL_DG8_CLEAN_R3/r0", qset, sdbs, bird_strict, rec_by_bcq=rec_by_bcq, benchmark="bird"),
            "SOL_DG8_CLEAN_R3_pooled": rate_with_cluster(bird_data, sol_dg8_clean_r3, qset, sdbs, bird_strict, rec_by_bcq=rec_by_bcq, benchmark="bird"),
            "SOL_C4": rate_with_cluster(bird_data, "SOL_C4/r0", qset, sdbs, bird_strict, rec_by_bcq=rec_by_bcq, benchmark="bird"),
            "recovery_clean_r3": round(rec_cl3, 4),
            "recovery_clean_r3_ci95": [round(slo, 4), round(shi, 4)],
            "comparisons": {
                "SOL_DG8_CLEAN_R3_minus_C0": full_compare(bird_data, sol_dg8_clean_r3, ["SOL_C0/r0"], qset, sdbs, exact_2powk=True),
                "SOL_DG8_CLEAN_R3_minus_PUBCAP_IND8_ROWS_R1": full_compare(bird_data, sol_dg8_clean_r3, ["SOL_PUBCAP_IND8_ROWS_R1/r0"], qset, sdbs, exact_2powk=True),
                "SOL_DG8_CLEAN_R3_minus_IND8_PROBE_R1": full_compare(bird_data, sol_dg8_clean_r3, ["SOL_IND8_PROBE_R1/r0"], qset, sdbs, exact_2powk=True),
                "SOL_DG8_CLEAN_R3_minus_IND8_PROMPT": full_compare(bird_data, sol_dg8_clean_r3, ["SOL_IND8_PROMPT/r0"], qset, sdbs, exact_2powk=True),
                "SOL_C4_minus_SOL_DG8_CLEAN_R3": full_compare(bird_data, ["SOL_C4/r0"], sol_dg8_clean_r3, qset, sdbs, exact_2powk=True),
                "SOL_PUBCAP_IND8_ROWS_R1_minus_SOL_C0": full_compare(bird_data, ["SOL_PUBCAP_IND8_ROWS_R1/r0"], ["SOL_C0/r0"], qset, sdbs, exact_2powk=True),
                "SOL_IND8_PROBE_R1_minus_SOL_PUBCAP_IND8_ROWS_R1": full_compare(bird_data, ["SOL_IND8_PROBE_R1/r0"], ["SOL_PUBCAP_IND8_ROWS_R1/r0"], qset, sdbs, exact_2powk=True),
            },
        }

    # =========================================================================
    # PAPER TABLE 8: financial Case Studies (Q201, Q234, Q215, Q229)
    # =========================================================================
    case_qids = [201, 234, 215, 229]
    table8_cases: dict[str, Any] = {}
    for qid in case_qids:
        entry: dict[str, Any] = {"question_id": qid, "db_id": dbs[qid], "arms": {}}
        for cond in (
            "C0/r0", "C0/r2", "SYN8/r0", "IND8/r0",
            "IND8_PROMPT/r0", "IND8_PROMPT_R2/r0", "IND8_PROMPT_R3/r0",
            "DG8_CLEAN_R1/r0", "DG8_CLEAN_R2/r0", "DG8_CLEAN_R3/r0",
            "C4/r0", "C4/r2",
            "SOL_C0/r0", "SOL_IND8_PROMPT/r0", "SOL_DG8_CLEAN_R1/r0", "SOL_C4/r0",
        ):
            rec_r = rec_by_bcq.get(("bird", cond, qid), {})
            entry["arms"][cond] = {
                "official_set_ex": bird_data[cond][qid],
                "strict_ex": bird_strict[cond].get(qid) if cond in bird_strict else None,
                "release_sha256": rec_r.get("release_sha256"),
                "predicted_sql": rec_r.get("predicted_sql"),
            }
        table8_cases[f"Q{qid}"] = entry
    fin_qids = sorted(q for q in bird_440 if dbs[q] == "financial")
    paper_table8_casestudy = {
        "financial_42_rates": {
            cond: rate_summary(bird_data, cond, fin_qids)
            for cond in ("C0/r0", "C0/r2", "SYN8/r0", "IND8/r0", "IND8_PROMPT/r0", "IND8_PROMPT_R2/r0", "IND8_PROMPT_R3/r0", "DG8_CLEAN_R1/r0", "DG8_CLEAN_R2/r0", "DG8_CLEAN_R3/r0", "C4/r0", "C4/r2", "SOL_C0/r0", "SOL_IND8_PROMPT/r0", "SOL_DG8_CLEAN_R1/r0", "SOL_C4/r0")
        },
        "DG8_CLEAN_R3_financial_42": rate_with_cluster(bird_data, dg8_clean, fin_qids, None, bird_strict),
        "case_studies": table8_cases,
    }

    # =========================================================================
    # PAPER TABLE 9 & TR TABLE 18: Direct-Star Entity-Level DP (BIRD n=159 & EHRSQL Demo94 v3 n=600)
    # =========================================================================
    ent_dg8 = ["ENT_V2_DG8_R1/r0", "ENT_V2_DG8_R2/r0", "ENT_V2_DG8_R3/r0"]
    ent_ind = ["ENT_V2_IND8_PROMPT_R1/r0", "ENT_V2_IND8_PROMPT_R2/r0", "ENT_V2_IND8_PROMPT_R3/r0"]
    ent_dpq_v1 = ["ENT_V2_DP_QUERY_R1/r0", "ENT_V2_DP_QUERY_R2/r0", "ENT_V2_DP_QUERY_R3/r0"]
    ent_dpq_v2 = ["ENT_V2_DP_QUERY_CLEAN_R1/r0", "ENT_V2_DP_QUERY_CLEAN_R2/r0", "ENT_V2_DP_QUERY_CLEAN_R3/r0"]

    tr_table18_entity_perdb: dict[str, Any] = {}
    for db_name, qids in sorted(ent_by_db.items()):
        tr_table18_entity_perdb[db_name] = {
            "n": len(qids),
            "glm_C0": rate_summary(bird_data, bird_c0, qids)["mean_pct"],
            "glm_ENT_V2_DG8": rate_summary(bird_data, ent_dg8, qids)["mean_pct"],
            "glm_ENT_V2_IND8_PROMPT": rate_summary(bird_data, ent_ind, qids)["mean_pct"],
            "glm_ENT_V2_DP_QUERY_v1": rate_summary(bird_data, ent_dpq_v1, qids)["mean_pct"],
            "glm_ENT_V2_DP_QUERY_CLEAN_v2": rate_summary(bird_data, ent_dpq_v2, qids)["mean_pct"],
            "glm_C4": rate_summary(bird_data, bird_c4, qids)["mean_pct"],
            "sol_C0": rate_summary(bird_data, "SOL_C0/r0", qids)["mean_pct"],
            "sol_ENT_V2_DG8_R1": rate_summary(bird_data, "SOL_ENT_V2_DG8_R1/r0", qids)["mean_pct"],
            "sol_ENT_V2_IND8_PROMPT_R1": rate_summary(bird_data, "SOL_ENT_V2_IND8_PROMPT_R1/r0", qids)["mean_pct"],
            "sol_ENT_V2_DP_QUERY_R1": rate_summary(bird_data, "SOL_ENT_V2_DP_QUERY_R1/r0", qids)["mean_pct"],
            "sol_C4": rate_summary(bird_data, "SOL_C4/r0", qids)["mean_pct"],
        }

    ehr600_tmpls = [ehr_tmpl_clusters["ehr600_by_id"][q] for q in ehr_600]
    paper_table9_entity = {
        "bird_entity_dp_159": {
            "glm_159": {
                "rates": {
                    "C0": rate_with_cluster(bird_data, bird_c0, ent_159, ent_dbs, bird_strict),
                    "ENT_V2_DG8_R3": rate_with_cluster(bird_data, ent_dg8, ent_159, ent_dbs, bird_strict, rec_by_bcq=rec_by_bcq, benchmark="bird"),
                    "ENT_V2_IND8_PROMPT_R3": rate_with_cluster(bird_data, ent_ind, ent_159, ent_dbs, bird_strict, rec_by_bcq=rec_by_bcq, benchmark="bird"),
                    "ENT_V2_DP_QUERY_v1_R3": rate_with_cluster(bird_data, ent_dpq_v1, ent_159, ent_dbs, bird_strict),
                    "ENT_V2_DP_QUERY_CLEAN_v2_R3": rate_with_cluster(bird_data, ent_dpq_v2, ent_159, ent_dbs, bird_strict, rec_by_bcq=rec_by_bcq, benchmark="bird"),
                    "Row_DG8_CLEAN_R3": rate_with_cluster(bird_data, dg8_clean, ent_159, ent_dbs, bird_strict),
                    "C4": rate_with_cluster(bird_data, bird_c4, ent_159, ent_dbs, bird_strict),
                },
                "comparisons": {
                    "ENT_V2_DG8_minus_ENT_V2_DP_QUERY_CLEAN_v2": full_compare(bird_data, ent_dg8, ent_dpq_v2, ent_159, ent_dbs),
                    "ENT_V2_DG8_minus_ENT_V2_IND8_PROMPT": full_compare(bird_data, ent_dg8, ent_ind, ent_159, ent_dbs),
                    "ENT_V2_DP_QUERY_CLEAN_v2_minus_C0": full_compare(bird_data, ent_dpq_v2, bird_c0, ent_159, ent_dbs),
                    "ENT_V2_DG8_minus_C0": full_compare(bird_data, ent_dg8, bird_c0, ent_159, ent_dbs),
                    "ENT_V2_DG8_minus_Row_DG8_CLEAN": full_compare(bird_data, ent_dg8, dg8_clean, ent_159, ent_dbs),
                },
            },
            "sol_159": {
                "rates": {
                    "SOL_C0": rate_with_cluster(bird_data, "SOL_C0/r0", ent_159, ent_dbs, bird_strict),
                    "SOL_ENT_V2_IND8_PROMPT_R1": rate_with_cluster(bird_data, "SOL_ENT_V2_IND8_PROMPT_R1/r0", ent_159, ent_dbs, bird_strict, rec_by_bcq=rec_by_bcq, benchmark="bird"),
                    "SOL_ENT_V2_DG8_R1": rate_with_cluster(bird_data, "SOL_ENT_V2_DG8_R1/r0", ent_159, ent_dbs, bird_strict, rec_by_bcq=rec_by_bcq, benchmark="bird"),
                    "SOL_ENT_V2_DP_QUERY_R1": rate_with_cluster(bird_data, "SOL_ENT_V2_DP_QUERY_R1/r0", ent_159, ent_dbs, bird_strict, rec_by_bcq=rec_by_bcq, benchmark="bird"),
                    "SOL_C4": rate_with_cluster(bird_data, "SOL_C4/r0", ent_159, ent_dbs, bird_strict),
                },
                "comparisons": {
                    "SOL_ENT_V2_DP_QUERY_R1_minus_C0": full_compare(bird_data, "SOL_ENT_V2_DP_QUERY_R1/r0", "SOL_C0/r0", ent_159, ent_dbs),
                    "SOL_ENT_V2_DP_QUERY_R1_minus_ENT_V2_DG8_R1": full_compare(bird_data, "SOL_ENT_V2_DP_QUERY_R1/r0", "SOL_ENT_V2_DG8_R1/r0", ent_159, ent_dbs),
                    "SOL_ENT_V2_DP_QUERY_R1_minus_IND8_PROMPT_R1": full_compare(bird_data, "SOL_ENT_V2_DP_QUERY_R1/r0", "SOL_ENT_V2_IND8_PROMPT_R1/r0", ent_159, ent_dbs),
                    "SOL_C4_minus_SOL_ENT_V2_DP_QUERY_R1": full_compare(bird_data, "SOL_C4/r0", "SOL_ENT_V2_DP_QUERY_R1/r0", ent_159, ent_dbs),
                },
            },
        },
        "ehrsql_demo94_v3_600": {
            "rate": rate_with_cluster(e600_data, "EHR_ENT8_V3_R1/r0", ehr_600, ehr600_tmpls, rec_by_bcq=rec_by_bcq, benchmark="ehrsql_600"),
            "minus_C0": full_compare(e600_data, "EHR_ENT8_V3_R1/r0", "C0/r0", ehr_600, ehr600_tmpls, mc_cluster=True),
            "minus_C0P": full_compare(e600_data, "EHR_ENT8_V3_R1/r0", "C0P/r0", ehr_600, ehr600_tmpls, mc_cluster=True),
            "minus_DG8P_V3_R3": full_compare(e600_data, ["EHR_ENT8_V3_R1/r0"], ehr_v3_r3_list, ehr_600, ehr600_tmpls, mc_cluster=True),
        },
        "ehrsql_demo94_legacy_pre_v3_600": {
            "rate": rate_with_cluster(e600_data, "EHR_ENT8_DEMO94/r0", ehr_600, ehr600_tmpls, rec_by_bcq=rec_by_bcq, benchmark="ehrsql_600"),
        },
    }

    # =========================================================================
    # TR TABLE 14: Dual Set vs Strict EX + Failed-Slot Attribution
    # =========================================================================
    dual_summary = json.loads((ROOT / "stats/dual_comparator_summary.json").read_text(encoding="utf-8"))
    failed_slots_breakdown: dict[str, Any] = {}
    bird_rec_by_cond: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in receipts:
        if r["benchmark"] == "bird":
            bird_rec_by_cond[r["condition"]].append(r)
    for cond, items in sorted(bird_rec_by_cond.items()):
        sub_err = Counter(
            r["runtime_error"] for r in items
            if r.get("runtime_error") and r["runtime_error"] != "empty_sql"
        )
        no_sub = sum(
            1 for r in items
            if r.get("runtime_error") == "empty_sql" or r.get("status") == "no_submission" or not (r.get("predicted_sql") or "").strip()
        )
        audit_inv = sum(1 for r in items if r.get("status") == "invalid")
        failed_slots_breakdown[cond] = {
            "n": len(items),
            "submitted_sql_runtime_errors": sum(sub_err.values()),
            "submitted_sql_error_breakdown": dict(sorted(sub_err.items())),
            "no_submission_count": no_sub,
            "audit_invalid_count": audit_inv,
        }

    exec_trans = json.loads((ROOT / "stats/executor_transition_summary_v10.json").read_text(encoding="utf-8"))
    dual_audit_reconciliation = {
        "schema_version": exec_trans["schema_version"],
        "total_slots": exec_trans["total_slots"],
        "reconciliation_5col_table": exec_trans["reconciliation_5col_table"],
        "sol_dg8_clean_r1_r3": exec_trans["sol_dg8_clean_r1_r3"],
        "audit_ledger_path": "manifests/audit_invalid_and_flip_ledger_v10.json",
    }
    tr_table14_dual = {
        "total_dual_conditions": len(dual_summary["conditions"]),
        "dual_comparator_conditions": dual_summary["conditions"],
        "failed_slots_breakdown": failed_slots_breakdown,
        "dg8_clean_uniform_1320_errors_60s": dual_summary["dg8_clean_uniform_1320_errors_60s"],
        "unified_local_executor_transitions": exec_trans,
    }

    # =========================================================================
    # TR TABLE 15: Complete Strata (Literals, Parsed Base Tables, Joins, Dev-Set Exposure, EHRSQL Patient-Literal)
    # =========================================================================
    def strata_block(qids: list[int]) -> dict[str, Any]:
        sdbs = [dbs[q] for q in qids]
        return {
            "n": len(qids),
            "glm53": {
                "Schema_only_C0": rate_with_cluster(bird_data, bird_c0, qids, sdbs, bird_strict),
                "IND8": rate_with_cluster(bird_data, "IND8/r0", qids, sdbs, bird_strict),
                "IND8_PROBE_R3": rate_with_cluster(bird_data, ind8_probe_r3, qids, sdbs, bird_strict),
                "SYN8": rate_with_cluster(bird_data, "SYN8/r0", qids, sdbs, bird_strict),
                "S3": rate_with_cluster(bird_data, "S3/r0", qids, sdbs, bird_strict),
                "PUBCAP_IND8_ROWS_R3": rate_with_cluster(bird_data, pubcap_ind8_rows, qids, sdbs, bird_strict),
                "PUBCAP_IND8_PROFILE_R3": rate_with_cluster(bird_data, pubcap_ind8_prof, qids, sdbs, bird_strict),
                "PUBCAP_SILHOUETTE_R3": rate_with_cluster(bird_data, pubcap_silh, qids, sdbs, bird_strict),
                "TIGHTBPE_SILHOUETTE_R3": rate_with_cluster(bird_data, ["TIGHTBPE_SILHOUETTE_R1/r0", "TIGHTBPE_SILHOUETTE_R2/r0", "TIGHTBPE_SILHOUETTE_R3/r0"], qids, sdbs, bird_strict),
                "TIGHTBPE_IND8_PROF_R3": rate_with_cluster(bird_data, ["TIGHTBPE_IND8_PROF_R1/r0", "TIGHTBPE_IND8_PROF_R2/r0", "TIGHTBPE_IND8_PROF_R3/r0"], qids, sdbs, bird_strict),
                "TIGHTBPE_STATIC_SYNTH_R3": rate_with_cluster(bird_data, ["TIGHTBPE_STATIC_SYNTH_R1/r0", "TIGHTBPE_STATIC_SYNTH_R2/r0", "TIGHTBPE_STATIC_SYNTH_R3/r0"], qids, sdbs, bird_strict),
                "1way_Profile_R3": rate_with_cluster(bird_data, ind8_prompt, qids, sdbs, bird_strict),
                "Silhouette_R3": rate_with_cluster(bird_data, dg8_clean, qids, sdbs, bird_strict),
                "Full_access_C4": rate_with_cluster(bird_data, bird_c4, qids, sdbs, bird_strict),
                "Silh_minus_C0": full_compare(bird_data, dg8_clean, bird_c0, qids, sdbs, exact_2powk=True),
                "Silh_minus_IND8_PROBE_R3": full_compare(bird_data, dg8_clean, ind8_probe_r3, qids, sdbs, exact_2powk=True),
                "Silh_minus_1wayProf": full_compare(bird_data, dg8_clean, ind8_prompt, qids, sdbs, exact_2powk=True),
                "Silh_minus_C4": full_compare(bird_data, dg8_clean, bird_c4, qids, sdbs, exact_2powk=True),
            },
            "gpt61_sol": {
                "SOL_C0": rate_with_cluster(bird_data, "SOL_C0/r0", qids, sdbs, bird_strict),
                "SOL_PUBCAP_IND8_ROWS_R1": rate_with_cluster(bird_data, "SOL_PUBCAP_IND8_ROWS_R1/r0", qids, sdbs, bird_strict),
                "SOL_IND8_PROBE_R1": rate_with_cluster(bird_data, "SOL_IND8_PROBE_R1/r0", qids, sdbs, bird_strict),
                "SOL_IND8_PROMPT": rate_with_cluster(bird_data, "SOL_IND8_PROMPT/r0", qids, sdbs, bird_strict),
                "SOL_DG8_CLEAN_R1": rate_with_cluster(bird_data, "SOL_DG8_CLEAN_R1/r0", qids, sdbs, bird_strict),
                "SOL_DG8_CLEAN_R3": rate_with_cluster(bird_data, sol_dg8_clean_r3, qids, sdbs, bird_strict),
                "SOL_C4": rate_with_cluster(bird_data, "SOL_C4/r0", qids, sdbs, bird_strict),
                "SOL_Silh_R3_minus_C0": full_compare(bird_data, sol_dg8_clean_r3, ["SOL_C0/r0"], qids, sdbs, exact_2powk=True),
            },
        }

    tr_table15_strata = {
        "literal_strata": {
            "S1_130": strata_block(s1_130),
            "S0_310": strata_block(s0_310),
            "Gold_Hidden_Literal_106": strata_block(hidden_106),
            "Gold_No_Hidden_Literal_334": strata_block(no_hidden_334),
        },
        "parsed_base_tables": {
            "1_base_table": strata_block(parsed_base["1_base_table"]),
            "2_base_tables": strata_block(parsed_base["2_base_tables"]),
            "3+_base_tables": strata_block(parsed_base["3+_base_tables"]),
            "2+_multi_table": strata_block(parsed_base["2+_multi_table"]),
        },
        "explicit_joins": {
            "0_joins": strata_block(join_strata["0_joins"]),
            "1_join": strata_block(join_strata["1_join"]),
            "2+_joins": strata_block(join_strata["2plus_joins"]),
        },
        "bird_dev_exposure_strata": {
            "definition": dev_strata["definition"],
            "dev_counts_by_db": dev_strata["dev_counts_by_db"],
            "low_dev_le4_134": strata_block(dev_strata["low_dev_le4_134"]),
            "mid_dev_5_6_127": strata_block(dev_strata["mid_dev_5_6_127"]),
            "high_dev_ge7_179": strata_block(dev_strata["high_dev_ge7_179"]),
        },
        "ehrsql_confirm_literal_strata": ehr_bundle["ehrsql_600"]["confirm_literal_strata"],
    }

    # =========================================================================
    # TR TABLE 16: Holm Multiple-Testing Families
    # =========================================================================
    tr_table16_holm = {
        "bird_9_dp_baselines_vs_silhouette": paper_table2_main["bird_440"]["holm_9_dp_baselines_vs_silhouette"],
        "bird_10_incl_c0_vs_silhouette": paper_table2_main["bird_440"]["holm_10_incl_c0_vs_silhouette"],
        "bird_13_all_baselines_vs_silhouette": paper_table2_main["bird_440"]["holm_13_all_baselines_vs_silhouette"],
        "bird_9_dp_baselines_vs_pubcap_silhouette": paper_table2_main["bird_440"]["holm_9_dp_baselines_vs_pubcap_silhouette"],
        "bird_10_incl_c0_vs_pubcap_silhouette": paper_table2_main["bird_440"]["holm_10_incl_c0_vs_pubcap_silhouette"],
        "ehrsql600_v3_8_baseline_family": ehr_bundle["ehrsql_600"]["holm_8_baseline_family"],
        "ehrsql600_v3_9_baseline_family": ehr_bundle["ehrsql_600"]["holm_9_baseline_family"],
        "bird_5_ablation_family_noval_r3": paper_table4_ablation["block_b_uncapped_ablations"]["holm_5_ablation_family_noval_r3"],
        "bird_5_ablation_family_noval_r1": paper_table4_ablation["block_b_uncapped_ablations"]["holm_5_ablation_family_noval_r1"],
        "bird_5_equal_pubcap_family": paper_table4_ablation["block_a_equal_public_cap_R3"]["comparisons_vs_PUBCAP_SILHOUETTE_R3"],
    }

    # =========================================================================
    # TR TABLE 17: Complete Turn-Ordered Sandboxed SQL Validation Reconciliation
    # =========================================================================
    tr_table17_sandbox_full = {
        "turn_ordered_stage9_arms": turn_ordered_rows,
        "legacy_superseded_archive": legacy_sandbox_rows,
        "legacy_adjacent_reference_archive": t5_authoritative,
    }

    # =========================================================================
    # TR TABLE 13: EHRSQL-874 Held-Out + Unseen Templates + Disagreement Rulesets + Parity
    # (Note: TR Table 11 is Configuration; TR Table 13 is Held-Out EHRSQL-874)
    # =========================================================================
    tr_table13_ehrhout = {
        "n_queries": len(ehr_874),
        "n_templates": ehr_bundle["ehrsql_874"]["n_templates"],
        "rates": ehr_bundle["ehrsql_874"]["rates"],
        "comparisons": ehr_bundle["ehrsql_874"]["comparisons"],
        "clinical_2x2_lookup_x_dp": ehr_bundle["ehrsql_874"]["clinical_2x2_lookup_x_dp"],
    }
    tr_table13_ehrfull = {
        "ehrsql_874": ehr_bundle["ehrsql_874"],
        "disagreement_rulesets": ehr_bundle["disagreement_rulesets"],
        "official_vs_uncapped_ex_parity": ehr_bundle["official_vs_uncapped_ex_parity"],
    }

    # =========================================================================
    # TR TABLE 19 & SCALING / DP-QUERY V2
    # =========================================================================
    tr_table19_survival = json.loads((ROOT / "stats/clinical_survival_3stage_verified.json").read_text(encoding="utf-8"))
    scaling_raw = json.loads((ROOT / "stats/stage6_scaling_and_survival.json").read_text(encoding="utf-8"))

    dpq_v2_nested = {}
    for lbl, qset in [("n55", fixed_55), ("n163", q15_163), ("n440", bird_440)]:
        qdbs = [dbs[q] for q in qset]
        dpq_v2_nested[lbl] = {
            "n": len(qset),
            "DP_QUERY_ROW_TP8_V2_R3": rate_with_cluster(bird_data, tp8_v2, qset, qdbs, bird_strict),
            "DP_QUERY_ROW_DB8_V2_R1": rate_with_cluster(bird_data, "DP_QUERY_ROW_DB8_V2_R1/r0", qset, qdbs, bird_strict),
            "DG8_CLEAN_R3": rate_with_cluster(bird_data, dg8_clean, qset, qdbs, bird_strict),
            "DG8_CLEAN_minus_TP8_V2": full_compare(bird_data, dg8_clean, tp8_v2, qset, qdbs, exact_2powk=True),
        }
    fixed55_sweep: dict[str, Any] = {}
    f55_dbs = [dbs[q] for q in fixed_55]
    for qp in (5, 15, 50, 200):
        qconds = [f"DPQ_FIXED55_QP{qp}_R{r}/r0" for r in (1, 2, 3)]
        fixed55_sweep[f"qp_{qp}"] = {
            "rate": rate_with_cluster(bird_data, qconds, fixed_55, f55_dbs, bird_strict),
            "vs_DG8_CLEAN_R3": full_compare(bird_data, qconds, dg8_clean, fixed_55, f55_dbs, exact_2powk=True),
        }
    fixed55_sweep["qp200_minus_qp5"] = full_compare(
        bird_data,
        [f"DPQ_FIXED55_QP200_R{r}/r0" for r in (1, 2, 3)],
        [f"DPQ_FIXED55_QP5_R{r}/r0" for r in (1, 2, 3)],
        fixed_55,
        f55_dbs,
        exact_2powk=True,
    )

    scaling_and_survival = {
        **scaling_raw,
        "dp_query_v2_batch_a": {
            "TP8_V2_R3_440": bird_rates["DP_Query_TP8_v2_R3"],
            "DB8_V2_R1_440": bird_rates["DP_Query_DB8_v2_R1"],
            "DG8_CLEAN_minus_TP8_V2_R3": full_compare(bird_data, dg8_clean, tp8_v2, bird_440, bird_dbs, exact_2powk=True),
            "DG8_CLEAN_R1_minus_DB8_V2_R1": full_compare(bird_data, ["DG8_CLEAN_R1/r0"], ["DP_QUERY_ROW_DB8_V2_R1/r0"], bird_440, bird_dbs, exact_2powk=True),
            "nested_prefixes": dpq_v2_nested,
        },
        "fixed55_dilution_batch_b": {
            "DG8_CLEAN_R3_on_55": rate_with_cluster(bird_data, dg8_clean, fixed_55, f55_dbs, bird_strict),
            "sweep": fixed55_sweep,
        },
    }

    table_manifest = {
        "paper_table2_main": {
            "paper_table_number": 2,
            "latex_label": "tab:main",
            "title": "End-to-end EX with 95% CIs and mean total cloud tokens per question on 440 BIRD and 600 EHRSQL confirmation queries under glm-5.3",
            "input_files": ["stats/per_question_outcomes.jsonl", "stats/sql_and_audit_receipts.jsonl.gz", "manifests/bird_minidev_2026_manifest.json", "manifests/ehrsql_2024_manifest.json", "manifests/question_strata_and_remap.json"],
            "builder_function": "build_all_tables",
        },
        "paper_table3_audit": {
            "paper_table_number": 3,
            "latex_label": "tab:audit",
            "title": "Database-deduplicated TransmissionLedger audit and outbound_messages_sha256 slot-level verification across BIRD and EHRSQL",
            "input_files": ["stats/p7_audit_summary.json", "stats/sql_and_audit_receipts.jsonl.gz", "stats/bird_financial_q201_audit_fixture.json"],
            "builder_function": "_compute_table3_audit",
        },
        "paper_table4_ablation": {
            "paper_table_number": 4,
            "latex_label": "tab:ablation",
            "title": "Budget schedule, equal-public-cap variants vs. PubCap Silhouette, TightBPE fixed-cap crossover, and single-factor uncapped ablations",
            "input_files": ["stats/per_question_outcomes.jsonl", "stats/sql_and_audit_receipts.jsonl.gz", "manifests/tightbpe_and_noval_manifest.json", "manifests/question_strata_and_remap.json"],
            "builder_function": "build_all_tables",
        },
        "paper_table5_sandbox": {
            "paper_table_number": 5,
            "latex_label": "tab:sandbox",
            "title": "Turn-ordered Sandboxed SQL Validation usage and deliverable/official execution errors on BIRD and canonical v3 EHRSQL under uniform 60 s SQLite 3.53 execution",
            "input_files": ["stats/sql_and_audit_receipts.jsonl.gz", "stats/sandbox_turn_order_telemetry_v10.json"],
            "builder_function": "_turn_ordered_sandbox_row",
        },
        "paper_table6_remap": {
            "paper_table_number": 6,
            "latex_label": "tab:remap",
            "title": "Code-recoded BIRD-Remap under Seed 1 (n=130 / 48 / 50 / 80) and Seed 2 (n=75 / 49 / 26)",
            "input_files": ["stats/per_question_outcomes.jsonl", "stats/sql_and_audit_receipts.jsonl.gz", "manifests/question_strata_and_remap.json"],
            "builder_function": "build_all_tables",
        },
        "paper_table7_sol": {
            "paper_table_number": 7,
            "latex_label": "tab:sol",
            "title": "Cross-model evaluation on 440 BIRD questions under gpt-6.1-sol with 95% CIs, mean total tokens and recovery",
            "input_files": ["stats/per_question_outcomes.jsonl", "stats/sql_and_audit_receipts.jsonl.gz", "manifests/question_strata_and_remap.json"],
            "builder_function": "build_all_tables",
        },
        "paper_table8_casestudy": {
            "paper_table_number": 8,
            "latex_label": "tab:casestudy",
            "title": "Recorded SQL predicates and whole-query EX across 3 clean Silhouette runs vs. both Schema only runs on BIRD financial",
            "input_files": ["stats/per_question_outcomes.jsonl", "stats/sql_and_audit_receipts.jsonl.gz"],
            "builder_function": "build_all_tables",
        },
        "paper_table9_entity": {
            "paper_table_number": 9,
            "latex_label": "tab:entity",
            "title": "Direct-star database-wide entity-level DP at eps_db=8, delta_db=1e-6, C=5 on 4 entity BIRD DBs (n=159) and canonical v3 EHRSQL Demo94 (n=600)",
            "input_files": ["stats/per_question_outcomes.jsonl", "stats/sql_and_audit_receipts.jsonl.gz", "manifests/question_strata_and_remap.json"],
            "builder_function": "build_all_tables",
        },
        "tr_table13_ehrhout": {
            "paper_table_number": 13,
            "latex_label": "tab:ehrsql-test",
            "title": "Held-out EHRSQL 2024 test set (n=874) canonical v3 Silhouette vs baselines and comparisons",
            "input_files": ["stats/per_question_outcomes.jsonl", "stats/sql_and_audit_receipts.jsonl.gz", "manifests/ehrsql_2024_manifest.json", "manifests/question_strata_and_remap.json"],
            "builder_function": "_compute_ehrsql_v3_and_dual",
        },
        "tr_table13_ehrfull": {
            "paper_table_number": 13,
            "latex_label": "tab:ehrsql-test",
            "title": "Unified EX with question & 134-template cluster CIs on all 874 EHRSQL test queries, 235 unseen-template queries, 639 seen-template queries, disagreement rulesets, and official vs uncapped EX parity",
            "input_files": ["stats/per_question_outcomes.jsonl", "stats/sql_and_audit_receipts.jsonl.gz", "stats/ehrsql_v3_disagreement_audit.json", "stats/ehrsql_874_disagreement_audit.json", "manifests/question_strata_and_remap.json"],
            "builder_function": "_compute_ehrsql_v3_and_dual",
        },
        "dual_audit_reconciliation": {
            "paper_table_number": 14,
            "latex_label": "tab:dual-audit-reconcile",
            "title": "Exact five-column dual-audit execution reconciliation across all 78,455 receipts separating raw 60 s SQLite 3.53.1 execution from harness status='invalid' gating",
            "input_files": ["stats/executor_transition_summary_v10.json", "manifests/audit_invalid_and_flip_ledger_v10.json", "stats/sql_and_audit_receipts.jsonl.gz"],
            "builder_function": "_compute_ehrsql_v3_and_dual",
        },
        "stage10_valfirst_and_fetch1": {
            "paper_table_number": 15,
            "latex_label": "tab:stage10-valfirst",
            "title": "Stage-10 paired value-preserving TightBPE-ValFirst and single-call Fetch-All-Once controls alongside 11-DB leave-one-DB-out sensitivity",
            "input_files": ["stats/per_question_outcomes.jsonl", "stats/sql_and_audit_receipts.jsonl.gz", "stats/sandbox_turn_order_telemetry_v10.json"],
            "builder_function": "_compute_stage10_valfirst_and_fetch1",
        },
        "tr_table14_dual": {
            "paper_table_number": 16,
            "latex_label": "tab:dual-comparator",
            "title": "BIRD 60 s SQLite replay comparing official Set EX against Strict multiset/ordered tuple EX and failed-slot attribution across 153 conditions",
            "input_files": ["stats/dual_comparator_summary.json", "stats/executor_transition_summary_v10.json", "stats/sql_and_audit_receipts.jsonl.gz"],
            "builder_function": "build_all_tables",
        },
        "tr_table15_strata": {
            "paper_table_number": 17,
            "latex_label": "tab:strata",
            "title": "Stratified BIRD execution accuracy across hidden literals, parsed base tables, explicit JOIN depth, dev-set exposure, plus EHRSQL-600 patient-literal vs non-patient-literal strata",
            "input_files": ["stats/per_question_outcomes.jsonl", "manifests/question_strata_and_remap.json"],
            "builder_function": "build_all_tables",
        },
        "tr_table16_holm": {
            "paper_table_number": 18,
            "latex_label": "tab:holm-families",
            "title": "Expanded Holm multiple-testing families with 95% question & cluster CIs, question-level Holm p-values, and cluster-level p-values",
            "input_files": ["stats/per_question_outcomes.jsonl", "manifests/question_strata_and_remap.json"],
            "builder_function": "build_all_tables",
        },
        "tr_table17_sandbox_full": {
            "paper_table_number": 19,
            "latex_label": "tab:sandbox-full",
            "title": "Complete turn-ordered Sandboxed SQL Validation and deliverable/official error statistics across canonical v3 and BIRD benchmarks",
            "input_files": ["stats/sql_and_audit_receipts.jsonl.gz", "stats/sandbox_turn_order_telemetry_v10.json"],
            "builder_function": "_turn_ordered_sandbox_row",
        },
        "tr_table18_entity_perdb": {
            "paper_table_number": 20,
            "latex_label": "tab:entity-perdb",
            "title": "Per-database EX across the four entity-linked BIRD databases under strict direct-star entity-level DP",
            "input_files": ["stats/per_question_outcomes.jsonl", "manifests/question_strata_and_remap.json"],
            "builder_function": "build_all_tables",
        },
        "tr_table19_survival": {
            "paper_table_number": 21,
            "latex_label": "tab:patient-scaling",
            "title": "Whole-patient bootstrap (primary, M=200, C=5) and marginal binomial 13-table clinical value survival across Stage-1 / Stage-2 / Stage-3",
            "input_files": ["stats/clinical_survival_3stage_verified.json", "stats/mimic_demo94_patient_bins_deid.json"],
            "builder_function": "build_all_tables",
        },
    }

    stage10_valfirst_and_fetch1 = _compute_stage10_valfirst_and_fetch1(
        bird_data, bird_deliv, bird_strict, bird_440, bird_dbs, rec_by_bcq
    )
    leave_one_db_out_sensitivity = _compute_leave_one_db_out_sensitivity(
        bird_data, bird_440, bird_dbs
    )

    return {
        "schema_version": "authoritative-paper-stats-v10",
        "meta": {
            "total_outcomes": sum(len(v) for b in data_bm.values() for v in b.values()),
            "unique_conditions": len({c for b in data_bm.values() for c in b}),
            "unique_benchmark_condition_series": sum(len(b) for b in data_bm.values()),
            "total_receipts": len(receipts),
            "outbound_messages_sha256_slots": paper_table3_audit["outbound_messages_sha256_total_slots"],
            "bootstrap_draws": B,
            "bootstrap_seed": SEED,
            "sqlite_timeout_s": 60.0,
        },
        "table_manifest": table_manifest,
        "paper_table2_main": paper_table2_main,
        "paper_table3_audit": paper_table3_audit,
        "paper_table4_ablation": paper_table4_ablation,
        "paper_table5_sandbox": paper_table5_sandbox,
        "paper_table6_remap": paper_table6_remap,
        "paper_table7_sol": paper_table7_sol,
        "paper_table8_casestudy": paper_table8_casestudy,
        "paper_table9_entity": paper_table9_entity,
        "tr_table13_ehrhout": tr_table13_ehrhout,
        "tr_table13_ehrfull": tr_table13_ehrfull,
        "tr_table14_dual": tr_table14_dual,
        "tr_table15_strata": tr_table15_strata,
        "tr_table16_holm": tr_table16_holm,
        "tr_table17_sandbox_full": tr_table17_sandbox_full,
        "tr_table18_entity_perdb": tr_table18_entity_perdb,
        "tr_table19_survival": tr_table19_survival,
        "tightbpe_and_noval": tightbpe_noval,
        "scaling_and_survival": scaling_and_survival,
        "stage10_valfirst_and_fetch1": stage10_valfirst_and_fetch1,
        "leave_one_db_out_sensitivity": leave_one_db_out_sensitivity,
        "dual_audit_reconciliation": dual_audit_reconciliation,
    }


def run_self_test(stats: dict[str, Any]) -> None:
    expected_keys = {
        "schema_version", "meta", "table_manifest",
        "paper_table2_main", "paper_table3_audit", "paper_table4_ablation",
        "paper_table5_sandbox", "paper_table6_remap", "paper_table7_sol",
        "paper_table8_casestudy", "paper_table9_entity",
        "tr_table13_ehrhout", "tr_table13_ehrfull", "tr_table14_dual",
        "tr_table15_strata", "tr_table16_holm", "tr_table17_sandbox_full",
        "tr_table18_entity_perdb", "tr_table19_survival",
        "tightbpe_and_noval", "scaling_and_survival",
        "stage10_valfirst_and_fetch1", "leave_one_db_out_sensitivity", "dual_audit_reconciliation",
    }
    assert expected_keys <= set(stats), expected_keys - set(stats)
    assert stats["schema_version"] == "authoritative-paper-stats-v10"
    assert stats["meta"]["total_outcomes"] == 78455
    assert stats["meta"]["total_receipts"] == 78455
    assert stats["meta"]["unique_conditions"] == 185
    assert stats["meta"]["unique_benchmark_condition_series"] == 192
    assert stats["meta"]["outbound_messages_sha256_slots"] == 73529

    t2 = stats["paper_table2_main"]
    br = t2["bird_440"]["rates"]
    assert br["Silhouette_DG8_CLEAN_R3"]["mean_pct"] == 46.97
    assert br["Silhouette_DG8_CLEAN_R3"]["strict_ex_mean_pct"] == 45.23
    assert br["Schema_only_C0"]["mean_pct"] == 30.11
    assert br["Full_access_C4"]["mean_pct"] == 46.48
    assert br["BIRD_Noiseless_V3"]["mean_pct"] == 50.68
    # Reconciled PubCap timeout fixes (Q174)
    assert br["PubCap_Silhouette_R3"]["hits_total"] == 617
    assert br["PubCap_Silhouette_R3"]["mean_pct"] == 46.74
    assert br["PubCap_FlatW_R3"]["mean_pct"] == 45.68
    assert br["PubCap_Omega1_EqDelta_R3"]["hits_total"] == 593
    assert br["PubCap_Omega1_EqDelta_R3"]["mean_pct"] == 44.92
    assert br["PubCap_Unpruned_R3"]["hits_total"] == 588
    assert br["PubCap_Unpruned_R3"]["mean_pct"] == 44.55
    assert br["PubCap_1way_Profile_R3"]["mean_pct"] == 44.32
    assert br["PubCap_Static_Synth_R3"]["mean_pct"] == 41.29
    # TightBPE & noVal R1..R3
    assert br["TightBPE_Silhouette_R3"]["mean_pct"] == 31.97
    assert br["TightBPE_1way_Profile_R3"]["mean_pct"] == 32.42
    assert br["TightBPE_Static_Synth_R3"]["mean_pct"] == 39.85
    assert br["DG8_noVal_Post_R3"]["hits_total"] == 534
    assert br["DG8_noVal_Post_R3"]["mean_pct"] == 40.45

    fact = t2["bird_440"]["leak_free_2x2_factorial"]
    assert fact["main_effects_and_interaction"]["Interaction ((ProfilePrompt - ProfileTool) - (RowsPrompt - RowsTool))"]["diff_pp"] == 4.24

    # Canonical custody-verified EHRSQL-600 & EHRSQL-874 v3
    e6 = t2["ehrsql_600"]["rates"]
    assert e6["Silhouette_DG8P_V3_R3"]["hits_total"] == 893
    assert e6["Silhouette_DG8P_V3_R3"]["mean_pct"] == 49.61
    assert e6["Silhouette_noLookup_V3_R3"]["hits_total"] == 811
    assert e6["Silhouette_noLookup_V3_R3"]["mean_pct"] == 45.06
    assert e6["Noiseless_LKP_V3"]["hits"] == 324
    assert e6["Noiseless_noLookup_V3"]["hits"] == 269
    assert e6["Entity_Demo94_V3"]["hits"] == 298
    assert e6["1way_Profile_V3"]["hits"] == 296

    e8 = stats["tr_table13_ehrhout"]["rates"]
    assert e8["EHR874_NOISELESS_NOLKP_V3"]["hits"] == 429
    assert e8["EHR874_NOISELESS_NOLKP_V3"]["mean_pct"] == 49.08
    assert e8["EHR874_NOISELESS_LKP_V3"]["hits"] == 457
    assert e8["EHR874_NOISELESS_LKP_V3"]["mean_pct"] == 52.29
    assert e8["EHR874_DG8P_V3_R3"]["hits_total"] == 1368
    assert e8["EHR874_DG8P_V3_R3"]["mean_pct"] == 52.17
    assert e8["EHR874_DG8P_NOLKP_V3_R3"]["hits_total"] == 1240
    assert e8["EHR874_DG8P_NOLKP_V3_R3"]["mean_pct"] == 47.29
    assert stats["tr_table13_ehrfull"]["ehrsql_874"]["unseen_vs_seen_templates"]["unseen_37_templates"]["n_queries"] == 235
    assert stats["tr_table13_ehrfull"]["official_vs_uncapped_ex_parity"]["official_vs_uncapped_mismatches"] == 0
    assert stats["tr_table13_ehrfull"]["official_vs_uncapped_ex_parity"]["gt100_distinct_questions_total"] == 8

    t3 = stats["paper_table3_audit"]
    assert t3["total_conditions"] == 105
    assert t3["clinical_v3_profile_slots_split"]["total_clinical_v3_profile_slots"] == 12992
    assert t3["clinical_v3_profile_slots_split"]["dp_clinical_profile_slots"] == 10044
    assert t3["clinical_v3_profile_slots_split"]["noiseless_clinical_control_slots"] == 2948
    t3r = t3["table3_rows"]
    assert t3r["BIRD_Silhouette_R3"]["full_payload"] == [2076465, 91289760]
    assert t3r["BIRD_PubCap_Silhouette_R3"]["copied"] == [0.0, 0.0]
    assert t3r["BIRD_PubCap_Silhouette_R3"]["ser_out"] == [0.0, 0.0]
    assert t3r["BIRD_PubCap_Silhouette_R3"]["oob"] == [0.0, 0.0]
    assert t3r["EHR600_Silhouette_V3_R3"]["copied"] == [0.0, 0.0]
    assert t3r["EHR600_Silhouette_noLookup_V3_R3"]["copied"] == [0.0, 0.0]
    assert t3r["EHR874_Silhouette_V3_R3"]["copied"] == [0.0, 0.0]
    assert t3r["EHR874_Silhouette_noLookup_V3_R3"]["copied"] == [0.0, 0.0]
    assert t3r["EHR874_Noiseless_noLookup_V3"]["copied"] == [0, 0]
    assert t3["financial_q201_fixture_audit"]["DG8_CLEAN_R1"]["full_payload_coincidences"] == [1063628, 1063628]

    t5 = stats["paper_table5_sandbox"]
    assert t5["bird"]["Silhouette_R3"]["telemetry_available"] is True
    assert t5["bird"]["Silhouette_R3"]["n_telemetry_slots"] == 1320
    assert t5["bird"]["Silhouette_noSandbox"]["runs_with_sandbox_call"] == 0
    assert t5["ehrsql_600_v3_canonical"]["Silhouette_V3_R3"]["submitted_sql_errors_count"] == 9
    assert t5["ehrsql_600_v3_canonical"]["Silhouette_V3_R3"]["empty_submission_count"] == 0
    assert t5["ehrsql_600_v3_canonical"]["Silhouette_V3_R3"]["slots"] == 1800
    assert t5["ehrsql_600_v3_canonical"]["Silhouette_V3_R3"]["error_then_rewrite_slots"] == 50

    t6 = stats["paper_table6_remap"]
    assert t6["bird_remap_seed1"]["All_130"]["gpt61_sol"]["SOL_REMAP_C0"]["hits"] == 57
    assert t6["bird_remap_seed1"]["All_130"]["gpt61_sol"]["SOL_REMAP_C0"]["mean_pct"] == 43.85
    assert t6["bird_remap_seed2_75"]["glm53"]["all_75"]["DG8_R1"]["hits"] == 6
    assert t6["bird_remap_seed2_75"]["glm53"]["all_75"]["DG8_R1"]["mean_pct"] == 8.0

    # Reconciled SOL_C0 Q132 timeout fix -> 319/440 = 72.50%
    t7 = stats["paper_table7_sol"]["full_440"]
    assert t7["SOL_C0"]["hits"] == 319
    assert t7["SOL_C0"]["mean_pct"] == 72.5
    assert t7["SOL_DG8_CLEAN_R3_pooled"]["mean_pct"] == 80.08
    assert t7["SOL_PUBCAP_IND8_ROWS_R1"]["mean_pct"] == 77.5
    assert t7["SOL_PUBCAP_IND8_ROWS_R1"]["strict_ex_mean_pct"] == 75.45
    assert t7["SOL_IND8_PROBE_R1"]["mean_pct"] == 78.64

    t8 = stats["paper_table8_casestudy"]["case_studies"]
    assert [t8["Q201"]["arms"][f"DG8_CLEAN_R{r}/r0"]["official_set_ex"] for r in (1, 2, 3)] == [0, 0, 0]
    assert [t8["Q234"]["arms"][f"DG8_CLEAN_R{r}/r0"]["official_set_ex"] for r in (1, 2, 3)] == [1, 1, 0]
    assert [t8["Q215"]["arms"][f"DG8_CLEAN_R{r}/r0"]["official_set_ex"] for r in (1, 2, 3)] == [1, 1, 1]
    assert [t8["Q229"]["arms"][f"DG8_CLEAN_R{r}/r0"]["official_set_ex"] for r in (1, 2, 3)] == [1, 1, 1]

    t9 = stats["paper_table9_entity"]
    assert t9["bird_entity_dp_159"]["glm_159"]["rates"]["ENT_V2_DP_QUERY_CLEAN_v2_R3"]["mean_pct"] == 38.57
    assert t9["bird_entity_dp_159"]["sol_159"]["rates"]["SOL_ENT_V2_DP_QUERY_R1"]["mean_pct"] == 72.96
    assert t9["ehrsql_demo94_v3_600"]["rate"]["hits"] == 298

    t19 = stats["tr_table19_survival"]
    assert t19["c1_total"] == 125
    assert t19["resampling_method"] == "whole_patient_bootstrap_with_replacement"
    assert t19["schedules"]["db_global_eps8"]["cohorts"]["100000"]["stage3_eq10_top15_published"]["all_125"]["mean_pct"] == 60.18
    assert t19["schedules"]["per_table_eps8"]["cohorts"]["100000"]["stage3_eq10_top15_published"]["reachable_120"]["mean_pct"] == 84.1

    ss = stats["scaling_and_survival"]
    assert ss["dp_query_v2_batch_a"]["TP8_V2_R3_440"]["mean_pct"] == 33.11
    assert ss["fixed55_dilution_batch_b"]["sweep"]["qp_5"]["rate"]["mean_pct"] == 56.97

    # Stage-10 assertions: tr_table13_ehrhout cluster CIs, stage10_valfirst_and_fetch1, leave_one_db_out, dual_audit
    for rname, rval in stats["tr_table13_ehrhout"]["rates"].items():
        assert len(rval.get("cluster_ci95_pp", [])) == 2, f"Missing cluster_ci95_pp on tr_table13_ehrhout rate {rname}"
        for rep_row in rval.get("per_replicate", []):
            assert len(rep_row.get("cluster_ci95_pp", [])) == 2, f"Missing cluster_ci95_pp on per_replicate {rep_row}"
    for cname, cval in stats["tr_table13_ehrhout"]["comparisons"].items():
        assert len(cval.get("cluster_ci95_pp", [])) == 2, f"Missing cluster_ci95_pp on tr_table13_ehrhout cmp {cname}"

    s10b = stats["stage10_valfirst_and_fetch1"]
    assert s10b["pooled_arm_rates"]["official_set_ex"]["TIGHTBPE_VALFIRST_SILH_R1..R3"]["hits_total"] == 517
    assert s10b["pooled_arm_rates"]["official_set_ex"]["TIGHTBPE_VALFIRST_IND8_PROF_R1..R3"]["hits_total"] == 575
    assert s10b["pooled_arm_rates"]["official_set_ex"]["FETCH1_PUBCAP_IND8_PROF_R1..R3"]["hits_total"] == 584
    assert s10b["pooled_arm_rates"]["deliverable_set_ex"]["TIGHTBPE_VALFIRST_SILH_R1..R3"]["hits_total"] == 515
    assert s10b["pooled_arm_rates"]["deliverable_set_ex"]["TIGHTBPE_VALFIRST_IND8_PROF_R1..R3"]["hits_total"] == 573
    assert s10b["pooled_arm_rates"]["deliverable_set_ex"]["FETCH1_PUBCAP_IND8_PROF_R1..R3"]["hits_total"] == 581
    assert s10b["pooled_arm_rates"]["official_set_ex"]["FETCH1_PUBCAP_IND8_PROF_R1..R3"]["telemetry"]["slots_with_fetch_call"] == 1320
    assert len(s10b["paired_comparisons_8"]["official_set_ex"]) == 8
    assert len(s10b["paired_comparisons_8"]["deliverable_set_ex"]) == 8

    lodo = stats["leave_one_db_out_sensitivity"]
    assert len(lodo) == 8
    for lname, lval in lodo.items():
        assert len(lval["per_dropped_db"]) == 11, lname
        assert lval["min_diff_pp"] <= lval["full_diff_pp"] <= lval["max_diff_pp"], lname

    da_rec = stats["dual_audit_reconciliation"]["reconciliation_5col_table"]
    assert da_rec["BIRD"]["n"] == 51493
    assert da_rec["EHRSQL-600"]["n"] == 15600
    assert da_rec["EHRSQL-874"]["n"] == 11362
    assert da_rec["total"]["n"] == 78455
    assert da_rec["total"]["status_invalid_total"] == 44
    assert da_rec["total"]["status_invalid_with_raw_hit"] == 21

    tm = stats["table_manifest"]
    expected_tr_numbers = {
        "tr_table13_ehrhout": (13, "tab:ehrsql-test"),
        "tr_table13_ehrfull": (13, "tab:ehrsql-test"),
        "dual_audit_reconciliation": (14, "tab:dual-audit-reconcile"),
        "stage10_valfirst_and_fetch1": (15, "tab:stage10-valfirst"),
        "tr_table14_dual": (16, "tab:dual-comparator"),
        "tr_table15_strata": (17, "tab:strata"),
        "tr_table16_holm": (18, "tab:holm-families"),
        "tr_table17_sandbox_full": (19, "tab:sandbox-full"),
        "tr_table18_entity_perdb": (20, "tab:entity-perdb"),
        "tr_table19_survival": (21, "tab:patient-scaling"),
    }
    for k, (num, lbl) in expected_tr_numbers.items():
        assert tm[k]["paper_table_number"] == num, (k, tm[k])
        assert tm[k]["latex_label"] == lbl, (k, tm[k])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Verify all Table 2-19 & Stage-8 invariants.")
    parser.add_argument("--export-json", type=Path, default=None, help="Export single source-of-truth JSON.")
    args = parser.parse_args(argv)

    stats = build_all_tables()
    if args.self_test:
        run_self_test(stats)
        print(
            f"REBUILD ALL TABLES SELF-TEST PASSED "
            f"({stats['meta']['unique_conditions']} conditions, "
            f"{stats['meta']['total_outcomes']} outcomes, "
            f"{stats['meta']['total_receipts']} receipts)"
        )
    if args.export_json is not None:
        args.export_json.parent.mkdir(parents=True, exist_ok=True)
        args.export_json.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Exported authoritative stats JSON to {args.export_json}")
    if not args.self_test and args.export_json is None:
        print(json.dumps({"meta": stats["meta"], "bird_dg8_clean": stats["paper_table2_main"]["bird_440"]["rates"]["Silhouette_DG8_CLEAN_R3"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
