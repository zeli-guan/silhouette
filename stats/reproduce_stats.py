"""Paired execution-accuracy comparisons and unbiased matched-replicate inference from ID-only receipts."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
from functools import lru_cache, reduce
import json
import math
from pathlib import Path
import random
from statistics import mean, stdev
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from silhouette.jsonl_io import open_jsonl
B = 10000
SEED = 20261114


def interval(samples: list[float]) -> tuple[float, float]:
    s = sorted(samples)
    return s[int(0.025 * len(s))], s[math.ceil(0.975 * len(s)) - 1]


@lru_cache(maxsize=64)
def _boot_draws(n: int, draws: int, seed: int) -> tuple[tuple[int, ...], ...]:
    rng = random.Random(seed)
    return tuple(tuple(rng.randrange(n) for _ in range(n)) for _ in range(draws))


def paired_bootstrap(differences: list[float], draws: int = B, seed: int = SEED) -> tuple[float, float]:
    n = len(differences)
    draws_idx = _boot_draws(n, draws, seed)
    getitem = differences.__getitem__
    return interval([sum(map(getitem, idx)) / n for idx in draws_idx])


def ratio_bootstrap(
    a_vals: list[float],
    c0_vals: list[float],
    c4_vals: list[float],
    draws: int = B,
    seed: int = SEED,
) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(a_vals)
    ratios = []
    for _ in range(draws):
        sa = sc0 = sc4 = 0.0
        for _ in range(n):
            i = rng.randrange(n)
            sa += a_vals[i]
            sc0 += c0_vals[i]
            sc4 += c4_vals[i]
        if sc4 > sc0:
            ratios.append((sa - sc0) / (sc4 - sc0))
    return interval(ratios)


@lru_cache(maxsize=32)
def _cluster_boot_draws(k: int, draws: int, seed: int) -> tuple[tuple[int, ...], ...]:
    rng = random.Random(seed)
    return tuple(tuple(rng.randrange(k) for _ in range(k)) for _ in range(draws))


def cluster_bootstrap(
    differences: list[float],
    db_ids: list[str],
    draws: int = B,
    seed: int = SEED,
) -> tuple[float, float]:
    by_db: dict[str, list[float]] = defaultdict(list)
    for db, d in zip(db_ids, differences):
        by_db[db].append(d)
    keys = sorted(by_db)
    sums = tuple(sum(by_db[k]) for k in keys)
    counts = tuple(len(by_db[k]) for k in keys)
    if all(n > 0 for n in counts):
        draws_idx = _cluster_boot_draws(len(keys), draws, seed)
        get_sum = sums.__getitem__
        get_cnt = counts.__getitem__
        return interval([sum(map(get_sum, idx)) / sum(map(get_cnt, idx)) for idx in draws_idx])
    groups = list(zip(sums, counts))
    rng = random.Random(seed)
    reps = []
    for _ in range(draws):
        while True:
            total = 0.0
            count = 0
            for _ in groups:
                s, n = groups[rng.randrange(len(groups))]
                total += s
                count += n
            if count > 0:
                reps.append(total / count)
                break
    return interval(reps)


def sign_test(d: list[float]) -> float:
    """Exact two-sided binomial sign test for 1-vs-1 binary differences."""
    wins, losses = sum(x > 0 for x in d), sum(x < 0 for x in d)
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1))
    return min(1.0, float(Fraction(2 * tail, 1 << n)))


def exact_rademacher_p(
    a_vals: list[float],
    b_vals: list[float],
    r_a: int,
    r_b: int,
    *,
    allow_unbalanced: bool = False,
) -> float:
    """Exact two-sided Rademacher sign-flip p-value on per-question arm means.

    Computes m_i = round(lcm(r_a, r_b) * (a_i - b_i)) (exact integer multiples) and
    evaluates Pr_{s in {+-1}^{n_nz}}(|sum s_i |m_i|| >= |sum m_i|) via exact integer DP
    over non-zero |m_i|. Reduces bit-for-bit to exact two-sided McNemar when r_a == r_b == 1.
    """
    if r_a != r_b and not allow_unbalanced:
        raise ValueError(f"exact_rademacher_p requires balanced replicates (r_a == r_b), got {r_a} vs {r_b}")
    if r_a <= 0 or r_b <= 0:
        raise ValueError("Replicate count must be positive")
    if len(a_vals) != len(b_vals):
        raise ValueError("a_vals and b_vals must have equal length")

    scale = (r_a * r_b) // math.gcd(r_a, r_b)
    ms = [round(scale * (a - b)) for a, b in zip(a_vals, b_vals)]
    mags = [abs(m) for m in ms if m != 0]
    n_nz = len(mags)
    if n_nz == 0:
        return 1.0
    obs = abs(sum(ms))
    if obs == 0:
        return 1.0

    g = reduce(math.gcd, mags)
    if g > 1:
        mags = [m // g for m in mags]
        obs //= g
    total_sum = sum(mags)
    k_max = (total_sum - obs) // 2
    if k_max < 0:
        return 0.0

    dp = [0] * (k_max + 1)
    dp[0] = 1
    for m in mags:
        if m <= k_max:
            for t in range(k_max, m - 1, -1):
                dp[t] += dp[t - m]
    tail_count = 2 * sum(dp)
    return min(1.0, float(Fraction(tail_count, 1 << n_nz)))


def wlt(d: list[float]) -> tuple[int, int, int]:
    return sum(x > 0 for x in d), sum(x < 0 for x in d), sum(x == 0 for x in d)


def holm(ps: list[float]) -> list[float]:
    adjusted = [0.0] * len(ps)
    bound = 0.0
    for rank, idx in enumerate(sorted(range(len(ps)), key=ps.__getitem__)):
        bound = max(bound, min(1.0, ps[idx] * (len(ps) - rank)))
        adjusted[idx] = bound
    return adjusted


def paired_trend(panel: list[list[int]], epsilons: tuple[int, ...]) -> tuple[float, float]:
    """Stratified Cochran-Armitage score conditional on each question's success total."""
    weights = [math.log(x) for x in epsilons]
    centered = [w - mean(weights) for w in weights]
    k = len(weights)
    ss = sum(w * w for w in centered)
    score = var = 0.0
    for row in panel:
        if len(row) != k or any(y not in (0, 1) for y in row):
            raise ValueError("Expected complete binary responses for every dose")
        m = sum(row)
        score += sum(w * y for w, y in zip(centered, row))
        var += ss * m * (k - m) / (k * (k - 1))
    if var == 0:
        return 0.0, 1.0
    z = score / math.sqrt(var)
    return z, math.erfc(abs(z) / math.sqrt(2))


def vec(data: dict[str, dict[Any, int]], conds: str | list[str], ids: list[Any]) -> list[float]:
    if isinstance(conds, str):
        conds = [conds]
    return [mean(data[c][q] for c in conds) for q in ids]


def compare_conditions(
    data: dict[str, dict[Any, int]],
    conds_a: str | list[str],
    conds_b: str | list[str],
    ids: list[Any],
    db_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Compare two condition replicate sets with balanced matched-replicate exact p-values."""
    if isinstance(conds_a, str):
        conds_a = [conds_a]
    if isinstance(conds_b, str):
        conds_b = [conds_b]
    r_a, r_b = len(conds_a), len(conds_b)
    a_vals = vec(data, conds_a, ids)
    b_vals = vec(data, conds_b, ids)
    diffs = [a - b for a, b in zip(a_vals, b_vals)]
    lo, hi = paired_bootstrap(diffs)
    clo, chi = (cluster_bootstrap(diffs, db_ids) if db_ids is not None and len(set(db_ids)) >= 2 else (None, None))

    out: dict[str, Any] = {
        "n": len(ids),
        "r_a": r_a,
        "r_b": r_b,
        "conds_a": list(conds_a),
        "conds_b": list(conds_b),
        "a_pct": round(100.0 * mean(a_vals), 2),
        "b_pct": round(100.0 * mean(b_vals), 2),
        "diff_pp": round(100.0 * mean(diffs), 2),
        "ci95_pp": [round(100.0 * lo, 2), round(100.0 * hi, 2)],
    }
    if clo is not None and chi is not None:
        out["cluster_ci95_pp"] = [round(100.0 * clo, 2), round(100.0 * chi, 2)]

    if r_a == r_b:
        p = exact_rademacher_p(a_vals, b_vals, r_a, r_b)
        w, l, t = wlt(diffs)
        out["p"] = p
        out["p_method"] = "1-vs-1 McNemar" if r_a == 1 else f"{r_a}-vs-{r_b} exact Rademacher"
        out["wlt"] = [w, l, t]
        if r_a > 1:
            indiv = {}
            for ca, cb in zip(conds_a, conds_b):
                va = vec(data, ca, ids)
                vb = vec(data, cb, ids)
                indiv[f"{ca} vs {cb}"] = {
                    "diff_pp": round(100.0 * mean(x - y for x, y in zip(va, vb)), 2),
                    "wlt": list(wlt([x - y for x, y in zip(va, vb)])),
                    "p": exact_rademacher_p(va, vb, 1, 1),
                }
            out["paired_replicate_1v1"] = indiv
    else:
        # Primary estimand: exact sign-flip on ALL releases (same estimand as diff_pp & bootstrap CIs)
        p_all = exact_rademacher_p(a_vals, b_vals, r_a, r_b, allow_unbalanced=True)
        w_all, l_all, t_all = wlt(diffs)
        out["p"] = p_all
        out["p_method"] = f"{r_a}-vs-{r_b} all-release exact Rademacher"
        out["wlt"] = [w_all, l_all, t_all]

        # Explicit sensitivity: matched min(R_a, R_b) AND matched R1-vs-R1
        k = min(r_a, r_b)
        ma_conds, mb_conds = conds_a[:k], conds_b[:k]
        ma_vals = vec(data, ma_conds, ids)
        mb_vals = vec(data, mb_conds, ids)
        mdiffs = [a - b for a, b in zip(ma_vals, mb_vals)]
        p_minr = exact_rademacher_p(ma_vals, mb_vals, k, k)
        mw, ml, mt = wlt(mdiffs)
        out["matched_wlt"] = [mw, ml, mt]
        out["matched_diff_pp"] = round(100.0 * mean(mdiffs), 2)
        out["matched_minR_diff_pp"] = out["matched_diff_pp"]
        out["matched_minR_question_p"] = p_minr
        out["matched_minR_wlt"] = [mw, ml, mt]

        r1a_vals = vec(data, [conds_a[0]], ids)
        r1b_vals = vec(data, [conds_b[0]], ids)
        r1_diffs = [a - b for a, b in zip(r1a_vals, r1b_vals)]
        out["matched_r1_diff_pp"] = round(100.0 * mean(r1_diffs), 2)
        out["matched_r1_question_p"] = exact_rademacher_p(r1a_vals, r1b_vals, 1, 1)
        out["matched_r1_wlt"] = list(wlt(r1_diffs))

        indiv = {}
        for ca in conds_a:
            for cb in conds_b:
                va = vec(data, ca, ids)
                vb = vec(data, cb, ids)
                indiv[f"{ca} vs {cb}"] = {
                    "diff_pp": round(100.0 * mean(x - y for x, y in zip(va, vb)), 2),
                    "wlt": list(wlt([x - y for x, y in zip(va, vb)])),
                    "p": exact_rademacher_p(va, vb, 1, 1),
                }
        out["individual_1v1"] = indiv
    return out


def print_cmp(label: str, res: dict[str, Any]) -> float:
    extra = f" | cluster [{res['cluster_ci95_pp'][0]:+.2f}, {res['cluster_ci95_pp'][1]:+.2f}]" if "cluster_ci95_pp" in res else ""
    wlt_vals = res.get("wlt") or res.get("matched_wlt")
    wlt_str = f"{wlt_vals[0]}/{wlt_vals[1]}/{wlt_vals[2]}"
    print(
        f"  {label:46s}: diff {res['diff_pp']:+6.2f} pp [{res['ci95_pp'][0]:+.2f}, {res['ci95_pp'][1]:+.2f}]"
        f"{extra} | {wlt_str} ({res['p_method']}) | p={res['p']:.4g}"
    )
    return float(res["p"])


def rate_summary(data: dict[str, dict[Any, int]], conds: str | list[str], ids: list[Any]) -> dict[str, Any]:
    if isinstance(conds, str):
        conds = [conds]
    reps = []
    for c in conds:
        v = vec(data, c, ids)
        reps.append({"condition": c, "hits": int(sum(v)), "n": len(ids), "pct": round(100.0 * mean(v), 2)})
    vals = vec(data, conds, ids)
    lo, hi = paired_bootstrap(vals)
    out = {
        "n": len(ids),
        "replicates": len(conds),
        "mean_pct": round(100.0 * mean(vals), 2),
        "ci95_pp": [round(100.0 * lo, 2), round(100.0 * hi, 2)],
        "per_replicate": reps,
    }
    if len(conds) > 1:
        out["sd_pp"] = round(stdev([r["pct"] for r in reps]), 2)
    else:
        out["hits"] = reps[0]["hits"]
    return out


def summarize_receipt_telemetry(receipts_path: Path) -> dict[str, Any]:
    by_cond: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with open_jsonl(receipts_path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                by_cond[r["condition"]].append(r)
    out = {}
    for c, items in sorted(by_cond.items()):
        n = len(items)
        ptoks = [x["prompt_tokens"] for x in items if x.get("prompt_tokens") is not None]
        ctoks = [x["completion_tokens"] for x in items if x.get("completion_tokens") is not None]
        ttoks = [x["total_tokens"] for x in items if x.get("total_tokens") is not None]
        lats = [x["latency_s"] for x in items if x.get("latency_s") is not None]
        calls = [x["tool_calls"] for x in items if x.get("tool_calls") is not None]
        errs = Counter(x["runtime_error"] for x in items if x.get("runtime_error") is not None)
        rec: dict[str, Any] = {
            "n": n,
            "official_set_ex_hits": sum(int(x["official_set_ex"]) for x in items if x.get("official_set_ex") is not None),
            "official_set_ex_pct": round(100.0 * mean(x["official_set_ex"] for x in items if x.get("official_set_ex") is not None), 2),
            "tool_calls_mean": round(mean(calls), 2) if calls else None,
            "total_tokens_mean": round(mean(ttoks), 1) if ttoks else None,
            "prompt_tokens_mean": round(mean(ptoks), 1) if ptoks else None,
            "completion_tokens_mean": round(mean(ctoks), 1) if ctoks else None,
            "latency_s_mean": round(mean(lats), 2) if lats else None,
            "runtime_errors_total": sum(errs.values()),
            "runtime_error_breakdown": dict(sorted(errs.items())),
        }
        if any(x.get("strict_ex") is not None for x in items):
            sitems = [x["strict_ex"] for x in items if x.get("strict_ex") is not None]
            rec["strict_ex_hits"] = sum(sitems)
            rec["strict_ex_pct"] = round(100.0 * mean(sitems), 2)
        if any(x.get("audit") is not None for x in items):
            first_a = next(x["audit"] for x in items if x.get("audit") is not None)
            if "copied_private_tuples" in first_a:
                rec["audit_totals"] = {
                    k: [
                        sum(x["audit"][k][0] for x in items if x.get("audit")),
                        sum(x["audit"][k][1] for x in items if x.get("audit")),
                    ]
                    for k in ("copied_private_tuples", "serialized_tuple_matches", "out_of_profile_literals", "multi_column_matches")
                }
            elif "true_invalid" in first_a:
                rec["audit_totals"] = {
                    "audit_flagged": sum(int(x["audit"]["audit_flagged"]) for x in items if x.get("audit")),
                    "true_invalid": sum(int(x["audit"]["true_invalid"]) for x in items if x.get("audit")),
                }
        if any(x.get("dp_probe_stats") is not None for x in items):
            dp_keys = sorted({k for x in items if x.get("dp_probe_stats") for k in x["dp_probe_stats"]})
            rec["dp_probe_totals"] = {
                k: sum(x["dp_probe_stats"].get(k, 0) for x in items if x.get("dp_probe_stats"))
                for k in dp_keys
            }
        out[c] = rec
    return out


def compute_all_stats(table5_json_path: Path | None = None) -> dict[str, Any]:
    data: dict[str, dict[Any, int]] = defaultdict(dict)
    dbs: dict[Any, str] = {}
    with open_jsonl(ROOT / "stats/per_question_outcomes.jsonl") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            q = r["question_id"]
            db = r["db_id"]
            label = r["condition"]
            if q in data[label] or dbs.setdefault(q, db) != db or r["outcome"] not in (0, 1):
                raise ValueError(f"Invalid or duplicate paired observation: {label} {q}")
            data[label][q] = r["outcome"]

    bird = json.loads((ROOT / "manifests/bird_minidev_2026_manifest.json").read_text(encoding="utf-8"))["evaluation"]
    ehr_m = json.loads((ROOT / "manifests/ehrsql_2024_manifest.json").read_text(encoding="utf-8"))
    ehr = [q["ehrsql_id"] for q in ehr_m["confirmation"]]
    ehr874 = ehr_m["official_test"]["held_out"]
    strata = json.loads((ROOT / "manifests/question_strata_and_remap.json").read_text(encoding="utf-8"))
    s1_130 = strata["bird_eval_440_strata"]["S1_130"]
    s0_310 = strata["bird_eval_440_strata"]["S0_310"]
    fail_50 = strata["bird_remap_130"]["remap_fail_50"]
    pass_80 = strata["bird_remap_130"]["remap_pass_80"]
    remap_s2_75 = strata["bird_remap_seed2"]["remap_col_slice_75"]
    remap_s2_s1_49 = strata["bird_remap_seed2"]["s1_in_remap_col_slice_49"]
    remap_s2_s0_26 = strata["bird_remap_seed2"]["s0_in_remap_col_slice_26"]
    ent_159 = strata["entity_159"]["question_ids"]
    ent_by_db = strata["entity_159"]["by_db"]
    sol_428 = strata["sol_denominators"]["table7_428"]
    sol_425 = strata["sol_denominators"]["clean_425"]
    join_strata = strata["bird_eval_440_join_strata"]
    q15_163 = strata["dp_query_subsets"]["q15_163"]
    q5_55 = strata["dp_query_subsets"]["q5_55"]
    bird_dbs = [dbs[q] for q in bird]

    dual_summary = json.loads((ROOT / "stats/dual_comparator_summary.json").read_text(encoding="utf-8"))

    print(f"Outcomes {sum(map(len, data.values()))} | conditions {len(data)} | resamples {B} | seed {SEED}")
    stats_out: dict[str, Any] = {
        "meta": {
            "total_outcomes": sum(map(len, data.values())),
            "unique_conditions": len(data),
            "bootstrap_draws": B,
            "bootstrap_seed": SEED,
            "primary_inference_rule": "exact_rademacher_p on balanced matched replicates (1v1=McNemar, 2v2, 3v3); no 3-vs-1 sign test",
        }
    }

    # 1. Table 2 BIRD (n=440) + 9-baseline Holm family
    print("\n=== Table 2: BIRD Mini-Dev (n=440, glm-5.3, Uniform Stage-4/5 DG8_CLEAN) ===")
    dg8_conds = ["DG8_CLEAN_R1/r0", "DG8_CLEAN_R2/r0", "DG8_CLEAN_R3/r0"]
    dg8_clean = vec(data, dg8_conds, bird)
    bird_c0 = vec(data, ["C0/r0", "C0/r2"], bird)
    bird_c4 = vec(data, ["C4/r0", "C4/r2"], bird)
    rlo, rhi = ratio_bootstrap(dg8_clean, bird_c0, bird_c4)
    rec = (mean(dg8_clean) - mean(bird_c0)) / (mean(bird_c4) - mean(bird_c0))
    print(
        f"DG8_CLEAN (R1..R3 mean): {100*mean(dg8_clean):.2f}% | "
        f"C0: {100*mean(bird_c0):.2f}% | C4: {100*mean(bird_c4):.2f}% | "
        f"recovery {100*rec:.2f}% [{100*rlo:.2f}%, {100*rhi:.2f}%]"
    )
    bird_family = [
        ("C0 (r0, r2)", ["C0/r0", "C0/r2"]),
        ("IND8", ["IND8/r0"]),
        ("SYN8", ["SYN8/r0"]),
        ("DP_QUERY_DB8", ["DP_QUERY_DB8/r0"]),
        ("S3", ["S3/r0"]),
        ("SUR3", ["SUR3/r0"]),
        ("MASKSQL_FULL", ["MASKSQL_FULL/r0"]),
        ("IND8_PROMPT (R1..R3)", ["IND8_PROMPT/r0", "IND8_PROMPT_R2/r0", "IND8_PROMPT_R3/r0"]),
        ("DP_QUERY_ROW_TP8_FULL", ["DP_QUERY_ROW_TP8_FULL/r0"]),
    ]
    bird_cmp_rows = []
    raw_ps = []
    for name, conds in bird_family:
        res = compare_conditions(data, dg8_conds, conds, bird, bird_dbs)
        raw_ps.append(print_cmp(f"DG8_CLEAN - {name} ({res['b_pct']:.2f}%)", res))
        bird_cmp_rows.append((name, res))
    adj_ps = holm(raw_ps)
    print("  BIRD 9-baseline Holm family (matched balanced exact p):")
    holm_bird = []
    for (name, res), rp, ap in zip(bird_cmp_rows, raw_ps, adj_ps):
        res["holm_p"] = ap
        holm_bird.append({"baseline": name, **res})
        print(f"    {name:24s}: diff {res['diff_pp']:+6.2f} pp | raw p={rp:.6g} | Holm p={ap:.6g} ({res['p_method']})")
    c4_cmp = compare_conditions(data, dg8_conds, ["C4/r0", "C4/r2"], bird, bird_dbs)
    print_cmp(f"DG8_CLEAN - C4 (r0, r2; {c4_cmp['b_pct']:.2f}%)", c4_cmp)

    stats_out["bird_440"] = {
        "DG8_CLEAN": rate_summary(data, dg8_conds, bird),
        "C0": rate_summary(data, ["C0/r0", "C0/r2"], bird),
        "C4": rate_summary(data, ["C4/r0", "C4/r2"], bird),
        "IND8_PROMPT": rate_summary(data, ["IND8_PROMPT/r0", "IND8_PROMPT_R2/r0", "IND8_PROMPT_R3/r0"], bird),
        "DP_QUERY_ROW_TP8_FULL": rate_summary(data, "DP_QUERY_ROW_TP8_FULL/r0", bird),
        "recovery_pct": round(100.0 * rec, 2),
        "recovery_ci95_pct": [round(100.0 * rlo, 2), round(100.0 * rhi, 2)],
        "holm_9_baseline_family": holm_bird,
        "vs_C4": c4_cmp,
    }

    # 2. Table 2 EHRSQL Confirmation (n=600) + 8-baseline Holm family
    print("\n=== Table 2: EHRSQL Confirmation (n=600, glm-5.3) ===")
    dg8p_conds = ["DG8P_R1/r0", "DG8P_R2/r0", "DG8P_R3/r0"]
    dg8p = vec(data, dg8p_conds, ehr)
    ehr_c0 = vec(data, ["C0/r0"], ehr)
    ehr_c4 = vec(data, ["C4/r0"], ehr)
    print(f"DG8P (R1..R3 mean): {100*mean(dg8p):.2f}% | C0: {100*mean(ehr_c0):.2f}% | C4: {100*mean(ehr_c4):.2f}%")
    ehr_family = [
        ("C0", ["C0/r0"]),
        ("C0P", ["C0P/r0"]),
        ("DG8_NOLKP", ["DG8_NOLKP/r0"]),
        ("IND8P", ["IND8P/r0"]),
        ("SYN8P", ["SYN8P/r0"]),
        ("S3", ["S3/r0"]),
        ("SUR3", ["SUR3/r0"]),
        ("EHR_IND8P_PROMPT", ["EHR_IND8P_PROMPT/r0"]),
    ]
    ehr_cmp_rows = []
    eraw_ps = []
    for name, conds in ehr_family:
        res = compare_conditions(data, dg8p_conds, conds, ehr)
        eraw_ps.append(print_cmp(f"DG8P - {name} ({res['b_pct']:.2f}%)", res))
        ehr_cmp_rows.append((name, res))
    eadj_ps = holm(eraw_ps)
    print("  EHRSQL 8-baseline Holm family (matched 1-vs-1 R1 exact McNemar):")
    holm_ehr = []
    for (name, res), rp, ap in zip(ehr_cmp_rows, eraw_ps, eadj_ps):
        res["holm_p"] = ap
        holm_ehr.append({"baseline": name, **res})
        print(f"    {name:20s}: diff {res['diff_pp']:+6.2f} pp | raw p={rp:.6g} | Holm p={ap:.6g}")
    ehr_c4_cmp = compare_conditions(data, dg8p_conds, ["C4/r0"], ehr)
    print_cmp(f"DG8P - C4 ({ehr_c4_cmp['b_pct']:.2f}%)", ehr_c4_cmp)

    stats_out["ehrsql_600"] = {
        "DG8P": rate_summary(data, dg8p_conds, ehr),
        "C0": rate_summary(data, "C0/r0", ehr),
        "C4": rate_summary(data, "C4/r0", ehr),
        "holm_8_baseline_family": holm_ehr,
        "vs_C4": ehr_c4_cmp,
    }

    # 3. EHRSQL Held-Out (n=874) incl. EHR874_NOISELESS_NOLKP
    print("\n=== EHRSQL Held-Out Split (n=874, sealed_test_ids) ===")
    e874_rates = {}
    for lbl, cond in [
        ("EHR874_C0", "EHR874_C0/r0"),
        ("EHR874_DG8_NOLKP_R1", "EHR874_DG8_NOLKP_R1/r0"),
        ("EHR874_NOISELESS_NOLKP", "EHR874_NOISELESS_NOLKP/r0"),
        ("EHR874_DG8P_R1", "EHR874_DG8P_R1/r0"),
        ("EHR874_C4", "EHR874_C4/r0"),
    ]:
        rs = rate_summary(data, cond, ehr874)
        e874_rates[lbl] = rs
        print(f"  {lbl:24s}: {rs['hits']}/874 ({rs['mean_pct']:.2f}%) | 95% CI [{rs['ci95_pp'][0]:.2f}, {rs['ci95_pp'][1]:.2f}]")
    e874_cmps = {}
    for key, ca, cb in [
        ("DG8P_R1_minus_C0", "EHR874_DG8P_R1/r0", "EHR874_C0/r0"),
        ("DG8P_R1_minus_C4", "EHR874_DG8P_R1/r0", "EHR874_C4/r0"),
        ("DG8P_R1_minus_DG8_NOLKP_R1", "EHR874_DG8P_R1/r0", "EHR874_DG8_NOLKP_R1/r0"),
        ("DG8_NOLKP_R1_minus_C0", "EHR874_DG8_NOLKP_R1/r0", "EHR874_C0/r0"),
        ("NOISELESS_NOLKP_minus_DG8_NOLKP_R1", "EHR874_NOISELESS_NOLKP/r0", "EHR874_DG8_NOLKP_R1/r0"),
        ("DG8P_R1_minus_NOISELESS_NOLKP", "EHR874_DG8P_R1/r0", "EHR874_NOISELESS_NOLKP/r0"),
        ("NOISELESS_NOLKP_minus_C0", "EHR874_NOISELESS_NOLKP/r0", "EHR874_C0/r0"),
    ]:
        res = compare_conditions(data, ca, cb, ehr874)
        e874_cmps[key] = res
        print_cmp(key, res)
    stats_out["ehrsql_874"] = {"rates": e874_rates, "comparisons": e874_cmps}

    # 4. Table 4 Budget Trend & BIRD 5-Ablation Holm Family (n=440)
    print("\n=== Table 4: Budget Trend & BIRD 5-Ablation Holm Family (n=440) ===")
    budget_curve = {}
    for eps, cond in [(1, "DG1_CLEAN_R1/r0"), (2, "DG2_CLEAN_R1/r0"), (4, "DG4_CLEAN_R1/r0"), (8, "DG8_CLEAN_R1/r0")]:
        rs = rate_summary(data, cond, bird)
        budget_curve[str(eps)] = rs
        print(f"  eps={eps} ({cond:15s}): {rs['hits']:3d}/440 ({rs['mean_pct']:.2f}%) | 95% CI [{rs['ci95_pp'][0]:.2f}, {rs['ci95_pp'][1]:.2f}]")
    arms_clean = ("DG1_CLEAN_R1/r0", "DG2_CLEAN_R1/r0", "DG4_CLEAN_R1/r0", "DG8_CLEAN_R1/r0")
    z, trend_p = paired_trend([[data[a][q] for a in arms_clean] for q in bird], (1, 2, 4, 8))
    print(f"  BIRD Clean DG-v3 paired trend (R1 endpoint): z={z:.6f} | two-sided p={trend_p:.6g}")

    ablation_family = [
        ("DG8_NOVAL_POST_R1 (1-vs-1)", ["DG8_NOVAL_POST_R1/r0"], ["DG8_CLEAN_R1/r0"]),
        ("DG8_FLATW (R1..R3, 3-vs-3)", ["DG8_FLATW/r0", "DG8_FLATW_R2/r0", "DG8_FLATW_R3/r0"], dg8_conds),
        ("DG8_OMEGA1 (R1..R3, 3-vs-3)", ["DG8_OMEGA1_R1/r0", "DG8_OMEGA1_R2/r0", "DG8_OMEGA1_R3/r0"], dg8_conds),
        ("DG8_UNPRUNED_V2 (R1..R3, 3-vs-3)", ["DG8_UNPRUNED_V2_R1/r0", "DG8_UNPRUNED_V2_R2/r0", "DG8_UNPRUNED_V2_R3/r0"], dg8_conds),
        ("DG8_NOSANDBOX (1-vs-1)", ["DG8_NOSANDBOX/r0"], ["DG8_CLEAN_R1/r0"]),
    ]
    abl_rows = []
    abl_raw_ps = []
    for label, a_conds, b_conds in ablation_family:
        res = compare_conditions(data, a_conds, b_conds, bird, bird_dbs)
        abl_raw_ps.append(print_cmp(f"{label} ({res['a_pct']:.2f}%) - DG8_CLEAN", res))
        abl_rows.append((label, res))
    abl_adj_ps = holm(abl_raw_ps)
    print("  BIRD 5-ablation Holm family:")
    holm_abl = []
    for (label, res), rp, ap in zip(abl_rows, abl_raw_ps, abl_adj_ps):
        res["holm_p"] = ap
        holm_abl.append({"ablation": label, **res})
        print(f"    {label:34s}: diff {res['diff_pp']:+6.2f} pp | raw p={rp:.6g} | Holm p={ap:.6g}")

    supp_abl = {}
    for key, ca, cb, scope, sdbs in [
        ("DG8_NOISELESS_minus_DG8_CLEAN_R1", ["DG8_NOISELESS/r0"], ["DG8_CLEAN_R1/r0"], bird, bird_dbs),
        ("DG8_NOVAL_POST_R1_minus_DG8_CLEAN_mean", ["DG8_NOVAL_POST_R1/r0"], dg8_conds, bird, bird_dbs),
        ("DG8_OMEGA1_minus_DG8_FLATW_3v3", ["DG8_OMEGA1_R1/r0", "DG8_OMEGA1_R2/r0", "DG8_OMEGA1_R3/r0"], ["DG8_FLATW/r0", "DG8_FLATW_R2/r0", "DG8_FLATW_R3/r0"], bird, bird_dbs),
        ("IND8_PROMPT_R1_minus_IND8_BIRD", ["IND8_PROMPT/r0"], ["IND8/r0"], bird, bird_dbs),
        ("IND8_PROMPT_R3_minus_IND8_BIRD", ["IND8_PROMPT/r0", "IND8_PROMPT_R2/r0", "IND8_PROMPT_R3/r0"], ["IND8/r0"], bird, bird_dbs),
        ("EHR_IND8P_PROMPT_minus_IND8P", ["EHR_IND8P_PROMPT/r0"], ["IND8P/r0"], ehr, None),
    ]:
        res = compare_conditions(data, ca, cb, scope, sdbs)
        supp_abl[key] = res
        print_cmp(key, res)

    stats_out["bird_440_ablations"] = {
        "budget_curve": budget_curve,
        "paired_trend_1_2_4_8": {"z": round(z, 6), "p": trend_p},
        "holm_5_ablation_family": holm_abl,
        "supplementary": supp_abl,
    }

    # 5. Direct-Star Entity-DP v2 (BIRD n=159 + EHRSQL Demo94 n=600)
    print("\n=== Direct-Star Entity-DP v2 (BIRD n=159 & EHRSQL Demo94 n=600) ===")
    ent_dbs = [dbs[q] for q in ent_159]
    ent_dg8 = ["ENT_V2_DG8_R1/r0", "ENT_V2_DG8_R2/r0", "ENT_V2_DG8_R3/r0"]
    ent_ind = ["ENT_V2_IND8_PROMPT_R1/r0", "ENT_V2_IND8_PROMPT_R2/r0", "ENT_V2_IND8_PROMPT_R3/r0"]
    ent_dpq = ["ENT_V2_DP_QUERY_R1/r0", "ENT_V2_DP_QUERY_R2/r0", "ENT_V2_DP_QUERY_R3/r0"]
    ent_c0 = ["C0/r0", "C0/r2"]
    ent_c4 = ["C4/r0", "C4/r2"]

    ent_rates_glm = {
        "ENT_V2_DG8": rate_summary(data, ent_dg8, ent_159),
        "ENT_V2_IND8_PROMPT": rate_summary(data, ent_ind, ent_159),
        "ENT_V2_DP_QUERY": rate_summary(data, ent_dpq, ent_159),
        "Row_DG8_CLEAN": rate_summary(data, dg8_conds, ent_159),
        "C0": rate_summary(data, ent_c0, ent_159),
        "C4": rate_summary(data, ent_c4, ent_159),
    }
    print(
        f"  glm-5.3 BIRD n=159: ENT_V2_DG8={ent_rates_glm['ENT_V2_DG8']['mean_pct']:.2f}% | "
        f"ENT_V2_IND8_PROMPT={ent_rates_glm['ENT_V2_IND8_PROMPT']['mean_pct']:.2f}% | "
        f"ENT_V2_DP_QUERY={ent_rates_glm['ENT_V2_DP_QUERY']['mean_pct']:.2f}% | "
        f"Row_DG8_CLEAN={ent_rates_glm['Row_DG8_CLEAN']['mean_pct']:.2f}% | "
        f"C0={ent_rates_glm['C0']['mean_pct']:.2f}% | C4={ent_rates_glm['C4']['mean_pct']:.2f}%"
    )
    ent_cmps_glm = {}
    for key, ca, cb in [
        ("ENT_V2_DG8_minus_ENT_V2_DP_QUERY", ent_dg8, ent_dpq),
        ("ENT_V2_DG8_minus_ENT_V2_IND8_PROMPT", ent_dg8, ent_ind),
        ("ENT_V2_IND8_PROMPT_minus_ENT_V2_DP_QUERY", ent_ind, ent_dpq),
        ("ENT_V2_DG8_minus_Row_DG8_CLEAN", ent_dg8, dg8_conds),
        ("ENT_V2_DG8_minus_C0", ent_dg8, ent_c0),
        ("ENT_V2_DG8_minus_C4", ent_dg8, ent_c4),
    ]:
        res = compare_conditions(data, ca, cb, ent_159, ent_dbs)
        ent_cmps_glm[key] = res
        print_cmp(f"glm159 {key}", res)

    ent_per_db = {}
    for db_name, qids in sorted(ent_by_db.items()):
        ent_per_db[db_name] = {
            "n": len(qids),
            "glm_ENT_V2_DG8": rate_summary(data, ent_dg8, qids)["mean_pct"],
            "glm_ENT_V2_IND8_PROMPT": rate_summary(data, ent_ind, qids)["mean_pct"],
            "glm_ENT_V2_DP_QUERY": rate_summary(data, ent_dpq, qids)["mean_pct"],
            "sol_C0": rate_summary(data, "SOL_C0/r0", qids)["mean_pct"],
            "sol_ENT_V2_DG8_R1": rate_summary(data, "SOL_ENT_V2_DG8_R1/r0", qids)["mean_pct"],
            "sol_ENT_V2_IND8_PROMPT_R1": rate_summary(data, "SOL_ENT_V2_IND8_PROMPT_R1/r0", qids)["mean_pct"],
            "sol_C4": rate_summary(data, "SOL_C4/r0", qids)["mean_pct"],
        }

    ent_rates_sol = {
        "SOL_C0": rate_summary(data, "SOL_C0/r0", ent_159),
        "SOL_ENT_V2_DG8_R1": rate_summary(data, "SOL_ENT_V2_DG8_R1/r0", ent_159),
        "SOL_ENT_V2_IND8_PROMPT_R1": rate_summary(data, "SOL_ENT_V2_IND8_PROMPT_R1/r0", ent_159),
        "SOL_DG8_CLEAN_R1": rate_summary(data, "SOL_DG8_CLEAN_R1/r0", ent_159),
        "SOL_C4": rate_summary(data, "SOL_C4/r0", ent_159),
    }
    print(
        f"  gpt-6.1-sol BIRD n=159: ENT_V2_DG8_R1={ent_rates_sol['SOL_ENT_V2_DG8_R1']['mean_pct']:.2f}% | "
        f"ENT_V2_IND8_PROMPT_R1={ent_rates_sol['SOL_ENT_V2_IND8_PROMPT_R1']['mean_pct']:.2f}% | "
        f"SOL_C0={ent_rates_sol['SOL_C0']['mean_pct']:.2f}% | SOL_C4={ent_rates_sol['SOL_C4']['mean_pct']:.2f}%"
    )
    ent_cmps_sol = {}
    for key, ca, cb in [
        ("SOL_ENT_V2_DG8_R1_minus_C0", "SOL_ENT_V2_DG8_R1/r0", "SOL_C0/r0"),
        ("SOL_ENT_V2_DG8_R1_minus_IND8_PROMPT_R1", "SOL_ENT_V2_DG8_R1/r0", "SOL_ENT_V2_IND8_PROMPT_R1/r0"),
        ("SOL_C4_minus_SOL_ENT_V2_DG8_R1", "SOL_C4/r0", "SOL_ENT_V2_DG8_R1/r0"),
        ("SOL_C4_minus_SOL_ENT_V2_IND8_PROMPT_R1", "SOL_C4/r0", "SOL_ENT_V2_IND8_PROMPT_R1/r0"),
    ]:
        res = compare_conditions(data, ca, cb, ent_159, ent_dbs)
        ent_cmps_sol[key] = res
        print_cmp(f"sol159 {key}", res)

    ehr_ent_rs = rate_summary(data, "EHR_ENT8_DEMO94/r0", ehr)
    print(f"  glm-5.3 EHRSQL Demo94 n=600: EHR_ENT8_DEMO94={ehr_ent_rs['hits']}/600 ({ehr_ent_rs['mean_pct']:.2f}%)")
    ehr_ent_cmps = {
        "EHR_ENT8_DEMO94_minus_C0": compare_conditions(data, "EHR_ENT8_DEMO94/r0", "C0/r0", ehr),
        "EHR_ENT8_DEMO94_minus_C0P": compare_conditions(data, "EHR_ENT8_DEMO94/r0", "C0P/r0", ehr),
        "EHR_ENT8_DEMO94_minus_DG8P": compare_conditions(data, "EHR_ENT8_DEMO94/r0", dg8p_conds, ehr),
    }
    for k, r in ehr_ent_cmps.items():
        print_cmp(k, r)

    stats_out["entity_dp_v2"] = {
        "glm_159": {"rates": ent_rates_glm, "comparisons": ent_cmps_glm},
        "sol_159": {"rates": ent_rates_sol, "comparisons": ent_cmps_sol},
        "per_db": ent_per_db,
        "ehrsql_demo94_600": {"rate": ehr_ent_rs, "comparisons": ehr_ent_cmps},
    }

    # 6. De-Memorized BIRD-Remap Seed 1 (130 / fail_50 / pass_80) & Seed 2 (75)
    print("\n=== De-Memorized BIRD-Remap Seed 1 (130/50/80) & Seed 2 (75) ===")
    remap_dbs = [dbs[q] for q in s1_130]
    remap_dg8 = ["REMAP_DG8_R1/r0", "REMAP_DG8_R2/r0", "REMAP_DG8_R3/r0"]
    s1_rates_glm = {
        "REMAP_C0": rate_summary(data, "REMAP_C0/r0", s1_130),
        "REMAP_IND8": rate_summary(data, "REMAP_IND8/r0", s1_130),
        "REMAP_IND8_PROMPT": rate_summary(data, "REMAP_IND8_PROMPT/r0", s1_130),
        "REMAP_DG8": rate_summary(data, remap_dg8, s1_130),
        "REMAP_C4": rate_summary(data, "REMAP_C4/r0", s1_130),
    }
    print(
        f"  glm-5.3 Seed1 (130): C0={s1_rates_glm['REMAP_C0']['mean_pct']:.2f}% | "
        f"IND8={s1_rates_glm['REMAP_IND8']['mean_pct']:.2f}% | "
        f"IND8_PROMPT={s1_rates_glm['REMAP_IND8_PROMPT']['mean_pct']:.2f}% | "
        f"DG8={s1_rates_glm['REMAP_DG8']['mean_pct']:.2f}% | C4={s1_rates_glm['REMAP_C4']['mean_pct']:.2f}%"
    )
    s1_cmps_glm = {}
    for key, ca, cb in [
        ("REMAP_DG8_minus_C0", remap_dg8, ["REMAP_C0/r0"]),
        ("REMAP_DG8_minus_IND8", remap_dg8, ["REMAP_IND8/r0"]),
        ("REMAP_DG8_minus_IND8_PROMPT", remap_dg8, ["REMAP_IND8_PROMPT/r0"]),
        ("REMAP_DG8_minus_C4", remap_dg8, ["REMAP_C4/r0"]),
    ]:
        res = compare_conditions(data, ca, cb, s1_130, remap_dbs)
        s1_cmps_glm[key] = res
        print_cmp(f"glm130 {key}", res)

    f50_cmps_glm = {
        "REMAP_DG8_minus_C0_fail50": compare_conditions(data, remap_dg8, ["REMAP_C0/r0"], fail_50),
        "REMAP_C4_minus_DG8_fail50": compare_conditions(data, ["REMAP_C4/r0"], remap_dg8, fail_50),
    }
    print(
        f"  glm-5.3 fail_50: C0={rate_summary(data, 'REMAP_C0/r0', fail_50)['mean_pct']:.2f}% | "
        f"DG8={rate_summary(data, remap_dg8, fail_50)['mean_pct']:.2f}% | "
        f"C4={rate_summary(data, 'REMAP_C4/r0', fail_50)['mean_pct']:.2f}%"
    )
    print_cmp("glm fail_50 REMAP_DG8 - REMAP_C0", f50_cmps_glm["REMAP_DG8_minus_C0_fail50"])

    # Exact Original-DB vs Remap-DB deltas on pass_80
    print("  Original-DB vs Remap-DB deltas on pass_80 (n=80):")
    pass80_dbs = [dbs[q] for q in pass_80]
    p80_deltas = {}
    for key, orig_c, rem_c in [
        ("glm_DG8_CLEAN_vs_REMAP_DG8_pass80", dg8_conds, remap_dg8),
        ("glm_IND8_PROMPT_vs_REMAP_pass80", ["IND8_PROMPT/r0"], ["REMAP_IND8_PROMPT/r0"]),
        ("glm_C0_vs_REMAP_C0_pass80", ["C0/r0", "C0/r2"], ["REMAP_C0/r0"]),
        ("glm_C4_vs_REMAP_C4_pass80", ["C4/r0", "C4/r2"], ["REMAP_C4/r0"]),
        ("sol_DG8_CLEAN_R1_vs_REMAP_DG8_R1_pass80", ["SOL_DG8_CLEAN_R1/r0"], ["SOL_REMAP_DG8_R1/r0"]),
        ("sol_IND8_PROMPT_vs_REMAP_pass80", ["SOL_IND8_PROMPT/r0"], ["SOL_REMAP_IND8_PROMPT/r0"]),
        ("sol_C0_vs_REMAP_C0_pass80", ["SOL_C0/r0"], ["SOL_REMAP_C0/r0"]),
        ("sol_C4_vs_REMAP_C4_pass80", ["SOL_C4/r0"], ["SOL_REMAP_C4/r0"]),
    ]:
        res = compare_conditions(data, rem_c, orig_c, pass_80, pass80_dbs)
        p80_deltas[key] = res
        print_cmp(f"  {key}", res)

    s1_rates_sol = {
        "SOL_REMAP_C0": rate_summary(data, "SOL_REMAP_C0/r0", s1_130),
        "SOL_REMAP_IND8_PROMPT": rate_summary(data, "SOL_REMAP_IND8_PROMPT/r0", s1_130),
        "SOL_REMAP_DG8_R1": rate_summary(data, "SOL_REMAP_DG8_R1/r0", s1_130),
        "SOL_REMAP_C4": rate_summary(data, "SOL_REMAP_C4/r0", s1_130),
    }
    print(
        f"  gpt-6.1-sol Seed1 (130): C0={s1_rates_sol['SOL_REMAP_C0']['mean_pct']:.2f}% | "
        f"IND8_PROMPT={s1_rates_sol['SOL_REMAP_IND8_PROMPT']['mean_pct']:.2f}% | "
        f"DG8_R1={s1_rates_sol['SOL_REMAP_DG8_R1']['mean_pct']:.2f}% | C4={s1_rates_sol['SOL_REMAP_C4']['mean_pct']:.2f}%"
    )
    sol_s1_cmp = compare_conditions(data, "SOL_REMAP_DG8_R1/r0", "SOL_REMAP_C0/r0", s1_130, remap_dbs)
    sol_f50_cmp = compare_conditions(data, "SOL_REMAP_DG8_R1/r0", "SOL_REMAP_C0/r0", fail_50)
    print_cmp("sol130 REMAP_DG8_R1 - REMAP_C0", sol_s1_cmp)
    print_cmp("sol_fail50 REMAP_DG8_R1 - REMAP_C0", sol_f50_cmp)

    # Seed 2 (n=75 remap_col_slice)
    s2_dbs = [dbs[q] for q in remap_s2_75]
    s2_rates = {
        "REMAP_S2_C0": rate_summary(data, "REMAP_S2_C0/r0", remap_s2_75),
        "REMAP_S2_IND8_PROMPT_R1": rate_summary(data, "REMAP_S2_IND8_PROMPT_R1/r0", remap_s2_75),
        "REMAP_S2_DG8_R1": rate_summary(data, "REMAP_S2_DG8_R1/r0", remap_s2_75),
        "REMAP_S2_C4": rate_summary(data, "REMAP_S2_C4/r0", remap_s2_75),
    }
    print(
        f"  glm-5.3 Seed2 (n=75 remap_col_slice): C0={s2_rates['REMAP_S2_C0']['mean_pct']:.2f}% | "
        f"IND8_PROMPT={s2_rates['REMAP_S2_IND8_PROMPT_R1']['mean_pct']:.2f}% | "
        f"DG8_R1={s2_rates['REMAP_S2_DG8_R1']['mean_pct']:.2f}% | C4={s2_rates['REMAP_S2_C4']['mean_pct']:.2f}%"
    )
    s2_cmps = {}
    for key, ca, cb in [
        ("REMAP_S2_DG8_R1_minus_C0", "REMAP_S2_DG8_R1/r0", "REMAP_S2_C0/r0"),
        ("REMAP_S2_IND8_PROMPT_R1_minus_C0", "REMAP_S2_IND8_PROMPT_R1/r0", "REMAP_S2_C0/r0"),
        ("REMAP_S2_C4_minus_DG8_R1", "REMAP_S2_C4/r0", "REMAP_S2_DG8_R1/r0"),
        ("REMAP_S2_DG8_R1_minus_orig_DG8_R1", "REMAP_S2_DG8_R1/r0", "DG8_CLEAN_R1/r0"),
        ("REMAP_S2_C4_minus_orig_C4_r0", "REMAP_S2_C4/r0", "C4/r0"),
    ]:
        res = compare_conditions(data, ca, cb, remap_s2_75, s2_dbs)
        s2_cmps[key] = res
        print_cmp(f"Seed2(75) {key}", res)

    stats_out["bird_remap"] = {
        "seed1_130": {
            "glm_rates": s1_rates_glm,
            "glm_comparisons": s1_cmps_glm,
            "glm_fail_50": {
                "C0": rate_summary(data, "REMAP_C0/r0", fail_50),
                "DG8": rate_summary(data, remap_dg8, fail_50),
                "C4": rate_summary(data, "REMAP_C4/r0", fail_50),
                "comparisons": f50_cmps_glm,
            },
            "pass_80_orig_vs_remap": p80_deltas,
            "sol_rates": s1_rates_sol,
            "sol_130_cmp": sol_s1_cmp,
            "sol_fail_50_cmp": sol_f50_cmp,
        },
        "seed2_75": {
            "rates": s2_rates,
            "s1_49_rates": {c: rate_summary(data, f"{c}/r0", remap_s2_s1_49) for c in ("REMAP_S2_C0", "REMAP_S2_IND8_PROMPT_R1", "REMAP_S2_DG8_R1", "REMAP_S2_C4")},
            "s0_26_rates": {c: rate_summary(data, f"{c}/r0", remap_s2_s0_26) for c in ("REMAP_S2_C0", "REMAP_S2_IND8_PROMPT_R1", "REMAP_S2_DG8_R1", "REMAP_S2_C4")},
            "comparisons": s2_cmps,
        },
    }

    # 7. DP-Query Table-Parallel Budget Dilution (Q in {5, 15, Q_full})
    print("\n=== DP-Query Table-Parallel Dilution (Q in {5, 15, Q_full}) ===")
    dpq_rates = {
        "TP8_FULL_440": rate_summary(data, "DP_QUERY_ROW_TP8_FULL/r0", bird),
        "TP8_Q15_163": rate_summary(data, "DP_QUERY_ROW_TP8_Q15/r0", q15_163),
        "TP8_Q5_55": rate_summary(data, "DP_QUERY_ROW_TP8_Q5/r0", q5_55),
        "TP8_FULL_on_Q15_163": rate_summary(data, "DP_QUERY_ROW_TP8_FULL/r0", q15_163),
        "TP8_FULL_on_Q5_55": rate_summary(data, "DP_QUERY_ROW_TP8_FULL/r0", q5_55),
    }
    for k, rs in dpq_rates.items():
        print(f"  {k:22s}: {rs['hits']}/{rs['n']} ({rs['mean_pct']:.2f}%) | 95% CI [{rs['ci95_pp'][0]:.2f}, {rs['ci95_pp'][1]:.2f}]")
    dpq_cmps = {
        "DG8_CLEAN_minus_TP8_FULL_440": compare_conditions(data, dg8_conds, ["DP_QUERY_ROW_TP8_FULL/r0"], bird, bird_dbs),
        "IND8_PROMPT_minus_TP8_FULL_440": compare_conditions(data, ["IND8_PROMPT/r0", "IND8_PROMPT_R2/r0", "IND8_PROMPT_R3/r0"], ["DP_QUERY_ROW_TP8_FULL/r0"], bird, bird_dbs),
        "TP8_Q15_minus_TP8_FULL_on_163": compare_conditions(data, "DP_QUERY_ROW_TP8_Q15/r0", "DP_QUERY_ROW_TP8_FULL/r0", q15_163, [dbs[q] for q in q15_163]),
        "TP8_Q5_minus_TP8_FULL_on_55": compare_conditions(data, "DP_QUERY_ROW_TP8_Q5/r0", "DP_QUERY_ROW_TP8_FULL/r0", q5_55, [dbs[q] for q in q5_55]),
        "DG8_CLEAN_minus_TP8_Q15_on_163": compare_conditions(data, dg8_conds, ["DP_QUERY_ROW_TP8_Q15/r0"], q15_163, [dbs[q] for q in q15_163]),
        "DG8_CLEAN_minus_TP8_Q5_on_55": compare_conditions(data, dg8_conds, ["DP_QUERY_ROW_TP8_Q5/r0"], q5_55, [dbs[q] for q in q5_55]),
    }
    for k, res in dpq_cmps.items():
        print_cmp(k, res)
    stats_out["dp_query_dilution"] = {"rates": dpq_rates, "comparisons": dpq_cmps}

    # 8. Dual Comparator (Set vs Strict) & Gold JOIN-Count Stratification
    print("\n=== Dual Comparator (Set vs Strict) & Gold JOIN-Count Strata ===")
    join_groups = [
        ("0_joins", join_strata["0_joins"]),
        ("1_join", join_strata["1_join"]),
        ("2plus_joins", join_strata["2plus_joins"]),
        ("1plus_multitable", sorted(join_strata["1_join"] + join_strata["2plus_joins"])),
    ]
    join_out = {}
    for jname, jids in join_groups:
        jdbs = [dbs[q] for q in jids]
        g_dg = rate_summary(data, dg8_conds, jids)
        g_c0 = rate_summary(data, ["C0/r0", "C0/r2"], jids)
        g_c4 = rate_summary(data, ["C4/r0", "C4/r2"], jids)
        s_dg = rate_summary(data, ["SOL_DG8_R1/r0", "SOL_DG8_R2/r0", "SOL_DG8_R3/r0"], jids)
        s_c0 = rate_summary(data, "SOL_C0/r0", jids)
        s_c4 = rate_summary(data, "SOL_C4/r0", jids)
        cmp_g_c0 = compare_conditions(data, dg8_conds, ["C0/r0", "C0/r2"], jids, jdbs)
        cmp_g_ind = compare_conditions(data, dg8_conds, ["IND8/r0"], jids, jdbs)
        cmp_s_c0 = compare_conditions(data, ["SOL_DG8_R1/r0", "SOL_DG8_R2/r0", "SOL_DG8_R3/r0"], ["SOL_C0/r0"], jids, jdbs)
        print(
            f"  [{jname:16s} n={len(jids):3d}] glm DG8={g_dg['mean_pct']:.2f}% C0={g_c0['mean_pct']:.2f}% C4={g_c4['mean_pct']:.2f}% | "
            f"sol DG8={s_dg['mean_pct']:.2f}% C0={s_c0['mean_pct']:.2f}% C4={s_c4['mean_pct']:.2f}%"
        )
        print_cmp(f"    glm {jname} DG8 - C0", cmp_g_c0)
        join_out[jname] = {
            "n": len(jids),
            "glm_DG8_CLEAN": g_dg,
            "glm_C0": g_c0,
            "glm_C4": g_c4,
            "glm_DG8_minus_C0": cmp_g_c0,
            "glm_DG8_minus_IND8": cmp_g_ind,
            "sol_DG8": s_dg,
            "sol_C0": s_c0,
            "sol_C4": s_c4,
            "sol_DG8_minus_C0": cmp_s_c0,
        }
    print(
        f"  Dual Comparator uniform DG8_CLEAN (R1..R3, 1,320 slots): "
        f"Set(pub)={dual_summary['dg8_clean_uniform_1320_errors_60s']['published_set_ex_mean_pct']:.2f}%, "
        f"Set(60s)={dual_summary['dg8_clean_uniform_1320_errors_60s']['set_ex_60s_mean_pct']:.2f}%, "
        f"Strict={dual_summary['dg8_clean_uniform_1320_errors_60s']['strict_ex_mean_pct']:.2f}%, "
        f"Errors={dual_summary['dg8_clean_uniform_1320_errors_60s']['runtime_errors_total']}/1320"
    )
    stats_out["join_stratification"] = join_out
    stats_out["dual_comparator"] = {
        "dg8_clean_uniform_1320": dual_summary["dg8_clean_uniform_1320_errors_60s"],
        "conditions": {
            c: {
                k: v[k]
                for k in ("n", "set_ex_cnt", "set_ex_pct", "published_set_ex_cnt", "published_set_ex_pct", "strict_ex_cnt", "strict_ex_pct", "set_minus_strict_pp", "runtime_errors_total", "runtime_error_breakdown", "by_join_stratum")
            }
            for c, v in dual_summary["conditions"].items()
        },
    }

    # 9. gpt-6.1-sol Full n=440 + Table 6 Stratified (S1=130 / S0=310)
    print("\n=== gpt-6.1-sol Full BIRD (n=440, 428, 425) & Stratified Table 6 ===")
    sol_full = {}
    for dlabel, qset in [("full_440", bird), ("table7_428", sol_428), ("clean_425", sol_425)]:
        sc0 = vec(data, "SOL_C0/r0", qset)
        sindp = vec(data, "SOL_IND8_PROMPT/r0", qset)
        sdg_cl = vec(data, "SOL_DG8_CLEAN_R1/r0", qset)
        sdg = vec(data, ["SOL_DG8_R1/r0", "SOL_DG8_R2/r0", "SOL_DG8_R3/r0"], qset)
        sc4 = vec(data, "SOL_C4/r0", qset)
        rec_cl = (mean(sdg_cl) - mean(sc0)) / (mean(sc4) - mean(sc0))
        rec_dg = (mean(sdg) - mean(sc0)) / (mean(sc4) - mean(sc0))
        print(
            f"  [{dlabel}] C0={100*mean(sc0):.2f}% | IND8_PROMPT={100*mean(sindp):.2f}% | "
            f"DG8_CLEAN_R1={100*mean(sdg_cl):.2f}% (rec {rec_cl:.3f}) | "
            f"DG8_R3={100*mean(sdg):.2f}% (rec {rec_dg:.3f}) | C4={100*mean(sc4):.2f}%"
        )
        sol_full[dlabel] = {
            "SOL_C0": rate_summary(data, "SOL_C0/r0", qset),
            "SOL_IND8_PROMPT": rate_summary(data, "SOL_IND8_PROMPT/r0", qset),
            "SOL_DG8_CLEAN_R1": rate_summary(data, "SOL_DG8_CLEAN_R1/r0", qset),
            "SOL_DG8_R3": rate_summary(data, ["SOL_DG8_R1/r0", "SOL_DG8_R2/r0", "SOL_DG8_R3/r0"], qset),
            "SOL_C4": rate_summary(data, "SOL_C4/r0", qset),
            "recovery_clean_r1": round(rec_cl, 4),
            "recovery_dg8_r3": round(rec_dg, 4),
        }
    sol_cmps = {
        "SOL_DG8_CLEAN_R1_minus_C0": compare_conditions(data, "SOL_DG8_CLEAN_R1/r0", "SOL_C0/r0", bird, bird_dbs),
        "SOL_DG8_R3_minus_C0": compare_conditions(data, ["SOL_DG8_R1/r0", "SOL_DG8_R2/r0", "SOL_DG8_R3/r0"], ["SOL_C0/r0"], bird, bird_dbs),
        "SOL_IND8_PROMPT_minus_C0": compare_conditions(data, "SOL_IND8_PROMPT/r0", "SOL_C0/r0", bird, bird_dbs),
        "SOL_DG8_CLEAN_R1_minus_C4": compare_conditions(data, "SOL_DG8_CLEAN_R1/r0", "SOL_C4/r0", bird, bird_dbs),
    }
    for k, res in sol_cmps.items():
        print_cmp(k, res)

    print("\n=== Table 6 Stratified S1 (n=130) vs S0 (n=310) ===")
    t6_out = {}
    for sname, sids in [("S1_130", s1_130), ("S0_310", s0_310)]:
        sdbs = [dbs[q] for q in sids]
        g_dg = rate_summary(data, dg8_conds, sids)
        g_c0 = rate_summary(data, ["C0/r0", "C0/r2"], sids)
        g_c4 = rate_summary(data, ["C4/r0", "C4/r2"], sids)
        s_dg = rate_summary(data, ["SOL_DG8_R1/r0", "SOL_DG8_R2/r0", "SOL_DG8_R3/r0"], sids)
        s_dg_cl = rate_summary(data, "SOL_DG8_CLEAN_R1/r0", sids)
        s_c0 = rate_summary(data, "SOL_C0/r0", sids)
        s_c4 = rate_summary(data, "SOL_C4/r0", sids)
        cmp_g = compare_conditions(data, dg8_conds, ["C0/r0", "C0/r2"], sids, sdbs)
        cmp_s = compare_conditions(data, ["SOL_DG8_R1/r0", "SOL_DG8_R2/r0", "SOL_DG8_R3/r0"], ["SOL_C0/r0"], sids, sdbs)
        print(f"  {sname} glm-5.3: DG8_CLEAN={g_dg['mean_pct']:.2f}% | C0={g_c0['mean_pct']:.2f}% | C4={g_c4['mean_pct']:.2f}%")
        print_cmp(f"    glm {sname} DG8_CLEAN - C0", cmp_g)
        print(f"  {sname} gpt-6.1-sol: DG8_R3={s_dg['mean_pct']:.2f}% | DG8_CLEAN_R1={s_dg_cl['mean_pct']:.2f}% | C0={s_c0['mean_pct']:.2f}% | C4={s_c4['mean_pct']:.2f}%")
        print_cmp(f"    sol {sname} DG8_R3 - C0", cmp_s)
        t6_out[sname] = {
            "glm": {"DG8_CLEAN": g_dg, "C0": g_c0, "C4": g_c4, "DG8_CLEAN_minus_C0": cmp_g},
            "sol": {"DG8_R3": s_dg, "DG8_CLEAN_R1": s_dg_cl, "C0": s_c0, "C4": s_c4, "DG8_R3_minus_C0": cmp_s},
        }
    stats_out["gpt61_sol"] = {"denominators": sol_full, "comparisons": sol_cmps, "table6_stratified": t6_out}

    # 10. Receipt telemetry & Table 5 / Table 13 reconciliation
    stats_out["receipt_telemetry"] = summarize_receipt_telemetry(ROOT / "stats/sql_and_audit_receipts.jsonl")
    if table5_json_path is not None and table5_json_path.exists():
        stats_out["table5_table13_authoritative"] = json.loads(table5_json_path.read_text(encoding="utf-8"))

    return stats_out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Reproduce all Silhouette paired statistical analyses.")
    parser.add_argument("--dump-json", type=Path, default=None, help="Optional path to dump structured stats JSON.")
    parser.add_argument("--table5-json", type=Path, default=None, help="Optional path to table5_table13_authoritative.json.")
    args = parser.parse_args(argv)

    table5_path = args.table5_json
    if table5_path is None and args.dump_json is not None:
        candidate = args.dump_json.parent / "table5_table13_authoritative.json"
        if candidate.exists():
            table5_path = candidate

    stats_out = compute_all_stats(table5_json_path=table5_path)
    if args.dump_json is not None:
        args.dump_json.parent.mkdir(parents=True, exist_ok=True)
        args.dump_json.write_text(json.dumps(stats_out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nDumped authoritative stats JSON to {args.dump_json}")


if __name__ == "__main__":
    main()
