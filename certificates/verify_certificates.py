"""Offline proof checker for every row and patient ledger. Run from any directory."""
from __future__ import annotations

from collections import Counter
from decimal import Decimal, ROUND_FLOOR, localcontext
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from silhouette.dg_sampler import rational_rho_budget

HERE = Path(__file__).resolve().parent
HALF = Fraction(1, 2000000)
HIST = {"values", "bins", "patterns"}
F = lru_cache(maxsize=None)(Fraction)


def require(truth):
    if not truth:
        raise ValueError("Certificate identity or coverage check failed")


def load(path):
    return json.loads(path.read_text())


@lru_cache(maxsize=None)
def tails(sigma2, tau):
    """Enclose both tails independently with outward-rounded fixed-point integers.

    w[k+1]/w[k] starts at exp(-1/(2v)) and shrinks by its square each step.
    After truncation, the next ratio majorizes every subsequent ratio, so the
    infinite remainder is at most w[k]*ratio/(1-ratio).
    """
    bits = 144
    unit = 1 << bits
    mask = unit - 1
    with localcontext() as ctx:
        ctx.prec = 75
        ctx.rounding = ROUND_FLOOR
        x = -Decimal(sigma2.denominator) / Decimal(2 * sigma2.numerator)
        lo = int(x.exp().next_minus() * unit)
        hi = int(x.next_plus().exp().next_plus() * unit) + 1
    ratio_lo, ratio_hi = lo * lo >> bits, (hi * hi + mask) >> bits
    wlo = whi = unit
    total_lo = total_hi = tail_lo = tail_hi = 0
    previous_lo = previous_hi = unit
    k = 0
    while True:
        k += 1
        wlo, whi = wlo * lo >> bits, (whi * hi + mask) >> bits
        lo, hi = lo * ratio_lo >> bits, (hi * ratio_hi + mask) >> bits
        total_lo += wlo
        total_hi += whi
        if k == tau - 1:
            previous_lo, previous_hi = wlo, whi
        if k >= tau:
            tail_lo += wlo
            tail_hi += whi
            if whi <= 8 and hi < unit:
                remainder = (whi * hi + unit - hi - 1) // (unit - hi)
                break
    den_lo = unit + 2 * total_lo
    den_hi = unit + 2 * (total_hi + remainder)
    return (Fraction(tail_lo, den_hi), Fraction(tail_hi + remainder, den_lo),
            Fraction(tail_lo + previous_lo, den_hi),
            Fraction(tail_hi + previous_hi + remainder, den_lo))


def check_parameter(q):
    rho, share, v, delta = (F(q[k]) for k in ("rho", "share", "sigma2", "delta"))
    sens = q["sensitivity2"]
    require(0 < share <= 1 and rho > 0 and v * 2 * rho == sens)
    if delta:
        threshold = delta / sens
        tau = q["tau"]
        require(type(tau) is int and 1 <= tau <= 100000 and 0 < threshold < Fraction(1, 2))
        a, b, c, d = tails(v, tau)
        require(F(q["tail_lower"]) <= a <= b <= F(q["tail_upper"]) <= threshold)
        require(threshold < F(q["prev_tail_lower"]) <= c <= d <= F(q["prev_tail_upper"]))
    return rho, share, delta, sens, F(q["r_j"])


def schedule(columns):
    yield "row_count", None, 1
    for col, info in columns.items():
        for stat, weight in info["weights"].items():
            yield stat, col, weight


def check_group(tables, params, schema, budget, delta, patient, global_split=False, table_caps=None):
    require(set(tables) == set(schema))
    runs = []
    for table, group in tables.items():
        items = list(schedule(schema[table]))
        require(len(group["queries"]) == len(items))
        cap = table_caps[table] if table_caps is not None else (5 if patient else 1)
        is_ent = cap > 1
        current = Counter()
        for (stat, col, index), (expected, name, weight) in zip(group["queries"], items):
            require((stat, col) == (expected, name) and type(index) is int and 0 <= index < len(params))
            current[index, stat, weight] += 1
        contributions = []
        for (index, stat, weight), count in current.items():
            rho, share, dj, sens, r_j = params[index]
            hist = stat in HIST
            expected_sens = (cap if hist else cap * cap if stat == "distinct_count" else 1) if is_ent else 1
            require(sens == expected_sens and (dj > 0) == hist)
            contributions.append((rho, share, dj, sens, hist, weight, r_j, count))
        require(sum(x[0]*x[-1] for x in contributions) == F(group["rho"]))
        require(sum(x[2]*x[-1] for x in contributions) == F(group["delta_threshold"]))
        runs.append(contributions)
    if global_split:
        runs = [sum(runs, [])]
    for rows in runs:
        hist_weight = sum(F(x[5])*x[-1] for x in rows if x[4])
        scores = []
        for rho, share, dj, sens, hist, weight, _, count in rows:
            require(rho == budget * share)
            require(dj == (delta * F(weight) / hist_weight if hist else 0))
            kappa = -NormalDist().inv_cdf(float(dj / sens)) if hist else 2.5
            scores.append((weight * kappa * math.sqrt(sens)) ** (2.0 / 3.0))
        total = math.fsum(s for s, row in zip(scores, rows) for _ in range(row[-1]))
        shares = [max(1, round(10**12 * (s / total))) for s in scores]
        denom = sum(n*row[-1] for n, row in zip(shares, rows))
        require(all(n == row[6] and Fraction(n, denom) == row[1] for n, row in zip(shares, rows)))


def verify_entity_db(doc, role_map):
    require(doc.get("version") == "c1-entity-dp-v2-direct-star")
    require(set(doc["schedules"]) == {"8"})
    budget = rational_rho_budget(8, HALF)
    expected_dbs = {"codebase_community", "debit_card_specializing", "financial", "thrombosis_prediction"}
    require(set(doc["schedules"]["8"]) == expected_dbs)
    for db, entry in doc["schedules"]["8"].items():
        require(entry.get("version") == "c1-entity-dp-v2-direct-star")
        params = [check_parameter(q) for q in entry["parameters"]]
        require(F(entry["rho_db"]) == budget and F(entry["delta_threshold_total"]) == HALF)
        require(F(entry["epsilon_converted_upper"]) <= 8)
        caps = entry["table_sensitivity_caps"]
        require(set(caps) == set(entry["tables"]) == set(role_map[db]) and all(v in (0, 5) for v in caps.values()))
        entity_tables = {}
        for t, group in entry["tables"].items():
            require(group.get("sensitivity_cap") == caps[t])
            if caps[t] == 0:
                require(group.get("mode") == "schema_only")
                require(F(group["rho"]) == 0 and F(group["delta_threshold"]) == 0 and group["queries"] == [])
            else:
                require(group.get("mode") == "entity_direct")
                entity_tables[t] = group
        require(len(entity_tables) >= 1)
        db_schema = {t: role_map[db][t] for t in entity_tables}
        check_group(entity_tables, params, db_schema, budget, HALF, True, global_split=True, table_caps=caps)

    require("ind8_prompt_schedules" in doc and set(doc["ind8_prompt_schedules"]) == {"8"})
    require(set(doc["ind8_prompt_schedules"]["8"]) == expected_dbs)
    for db, entry in doc["ind8_prompt_schedules"]["8"].items():
        require(entry.get("version") == "c1-entity-dp-v2-direct-star")
        params = [check_parameter(q) for q in entry["parameters"]]
        require(F(entry["rho_db"]) == budget and F(entry["delta_threshold_total"]) == HALF)
        require(F(entry["epsilon_converted_upper"]) <= 8)
        caps = entry["table_sensitivity_caps"]
        require(set(caps) == set(entry["tables"]) == set(role_map[db]) and all(v in (0, 5) for v in caps.values()))
        require(all(p[4] == 1 for p in params))
        n_q = sum(len(t["queries"]) for t in entry["tables"].values())
        require(n_q > 0 and all(p[0] == budget / n_q and p[1] == F(1, n_q) for p in params))
        n_hist = sum(1 for t in entry["tables"].values() for s, _, _ in t["queries"] if s in HIST)
        require(n_hist > 0)
        for p in params:
            if p[2] > 0:
                require(p[2] == HALF / n_hist)
        total_rho = Fraction(0)
        total_delta = Fraction(0)
        for t, group in entry["tables"].items():
            require(group.get("sensitivity_cap") == caps[t])
            if caps[t] == 0:
                require(group.get("mode") == "schema_only")
                require(F(group["rho"]) == 0 and F(group["delta_threshold"]) == 0 and group["queries"] == [])
                continue
            require(group.get("mode") == "entity_direct")
            rho_t = Fraction(0)
            del_t = Fraction(0)
            for stat, col, idx in group["queries"]:
                require(type(idx) is int and 0 <= idx < len(params))
                rho, share, dj, sens, _r_j = params[idx]
                hist = stat in HIST
                expected_sens = 5 if hist else (25 if stat == "distinct_count" else 1)
                require(sens == expected_sens and (dj > 0) == hist)
                rho_t += rho
                del_t += dj
            require(rho_t == F(group["rho"]) and del_t == F(group["delta_threshold"]))
            total_rho += rho_t
            total_delta += del_t
        require(total_rho == budget and total_delta == HALF)

    # Verify frozen entity profiles on disk match schema_only stubs and 0 queries
    for rel_root in (ROOT / "profiles/bird_entity_dp_eps8", ROOT / "profiles/entity_ind8_prompt"):
        dpmetas = sorted(rel_root.rglob("*.dpmeta.json"))
        require(len(dpmetas) >= 4)
        for mpath in dpmetas:
            meta = load(mpath)
            db = meta["db_id"]
            prof = load(mpath.with_name(db + ".json"))
            require(meta.get("version") == "c1-entity-dp-v2-direct-star")
            caps = meta["table_sensitivity_caps"]
            require(set(caps) == set(meta["tables"]) == set(prof) == set(role_map[db]))
            for t, tinfo in meta["tables"].items():
                if caps[t] == 0:
                    require(tinfo.get("mode") == "schema_only")
                    require(tinfo["queries"] == [] and tinfo["sensitivity_cap"] == 0)
                    require(F(tinfo["rho"]) == 0 and F(tinfo["delta_threshold"]) == 0)
                    for col, cinfo in role_map[db][t].items():
                        require(prof[t][col] == [cinfo["type"], None, None, None, None])
                else:
                    require(tinfo.get("mode") == "entity_direct" and tinfo["sensitivity_cap"] == 5)


def verify():
    role_map = load(ROOT / "manifests/schema_role_map.json")
    ehr = load(ROOT / "manifests/ehrsql_2024_manifest.json")
    ledger = load(HERE / "dg_v3_ledgers_ehrsql.json")
    clinical = {t: role_map["mimic_iv"][t] for t in ehr["private_tables"]}
    require(len(clinical) == 13 and sum(len(list(schedule(c))) for c in clinical.values()) == 346)
    bird = {name for name in load(ROOT / "manifests/bird_minidev_2026_manifest.json")["sha256"]
            if name.endswith(".sqlite")}
    require(len(bird) == 11 and set(ledger) == {"1", "2", "4", "8"})
    for name in ("bird_dg_v3", "ehrsql_dg_v3", "patient_dp"):
        doc = load(HERE / (name + "_certificates.json"))
        params = [check_parameter(q) for q in doc["parameters"]]
        require(set(doc["schedules"]) == {"1", "2", "4", "8"})
        for eps, dbs in doc["schedules"].items():
            budget = rational_rho_budget(int(eps), HALF)
            if name == "patient_dp":
                require(set(dbs) == {"per_table", "db_uniform", "db_global"})
                for kind, group in dbs.items():
                    n = len(clinical)
                    tables = group["tables"]
                    copies = n if kind == "per_table" else 1
                    divisor = n if kind == "db_uniform" else 1
                    require(F(group["rho"]) == copies * budget and F(group["delta"]) == copies * 2 * HALF)
                    check_group(tables, params, clinical, budget / divisor, HALF / divisor,
                                True, kind == "db_global")
            else:
                require(set(dbs) == ({n[:-7] for n in bird} if name.startswith("bird") else {"mimic_iv"}))
                for db, tables in dbs.items():
                    check_group(tables, params, role_map[db], budget, HALF, False)
                    if db == "mimic_iv":
                        for t, g in tables.items():
                            expanded = [dict(stat=s, column=c, **doc["parameters"][i])
                                        for s, c, i in g["queries"]]
                            require(ledger[eps][t] == dict(g, queries=expanded))
                        require(set(ledger[eps]) == set(tables))
    verify_entity_db(load(HERE / "bird_entity_dp_certificates.json"), role_map)
    verify_pubcap_and_ablation_dpmetas(role_map)
    verify_dawa_dpsips_clean_dpmetas(role_map)


def verify_pubcap_and_ablation_dpmetas(role_map):
    """Verify exact rational certificates & SHA-256 digests for all pubcap_* and ablation profiles."""
    from silhouette.profile import estimate_prompt_tokens, render_profile_prompt_block

    expected_dirs = (
        ("pubcap_silhouette", "DG-v3", True),
        ("pubcap_flatw", "DG-FlatW", True),
        ("pubcap_omega1_eqdelta", "DG-Omega1", True),
        ("pubcap_unpruned", "DG-Unpruned", True),
        ("bird_dg8_flatw", "DG-FlatW", False),
        ("bird_dg8_omega1", "DG-Omega1", False),
        ("bird_dg8_unpruned_v2", "DG-Unpruned", False),
    )
    for rel_name, expected_mech, is_pubcap in expected_dirs:
        rel_dir = ROOT / "profiles" / rel_name
        if not rel_dir.exists():
            continue
        dpmetas = sorted(rel_dir.rglob("*.dpmeta.json"))
        require(len(dpmetas) == 33)  # 11 BIRD DBs x 3 releases
        for mpath in dpmetas:
            meta = load(mpath)
            db = meta["db_id"]
            jpath = mpath.with_name(db + ".json")
            require(jpath.exists())
            require(meta["mechanism"] == expected_mech)
            require(meta["sha256"] == hashlib.sha256(jpath.read_bytes()).hexdigest())
            budget = rational_rho_budget(int(meta["epsilon"]), HALF)
            params = [check_parameter(q) for q in meta["parameters"]]
            prof = load(jpath)
            require(set(meta["tables"]) == set(prof) == set(role_map[db]))
            for t, tinfo in meta["tables"].items():
                rho_sum = sum((params[idx][0] for _, _, idx in tinfo["queries"]), Fraction(0))
                delta_sum = sum((params[idx][2] for _, _, idx in tinfo["queries"]), Fraction(0))
                require(rho_sum == F(tinfo["rho"]) and rho_sum <= budget)
                require(delta_sum == F(tinfo["delta_threshold"]) and delta_sum == HALF)
            if is_pubcap:
                cap = int(meta["pubcap_tokens"])
                actual = estimate_prompt_tokens(render_profile_prompt_block(prof))
                require(actual == int(meta["actual_block_tokens"]) and actual <= cap)


def verify_dawa_dpsips_clean_dpmetas(role_map):
    """Verify exact rational certificates, OS-CSPRNG provenance, & SHA-256 digests for all 34 DAWA+DP-SIPS clean releases."""
    from silhouette.sandbox import PUBLIC_TABLES

    dawa_root = ROOT / "profiles/dawa_dpsips_clean"
    dpmetas = sorted(dawa_root.rglob("*.dpmeta.json"))
    require(len(dpmetas) == 34)  # 11 BIRD DBs x 3 releases + 1 MIMIC-IV release
    for mpath in dpmetas:
        meta = load(mpath)
        db = meta["db_id"]
        jpath = mpath.with_name(db + ".json")
        require(jpath.exists())
        require(meta["mechanism"] == "DAWA+DP-SIPS")
        require(meta.get("rng_source") == "os_csprng" and "seed" not in meta)
        require(meta["sha256"] == hashlib.sha256(jpath.read_bytes()).hexdigest())
        budget = rational_rho_budget(int(meta["epsilon"]), HALF)
        require(F(meta["rho_table"]) == budget)
        params = [check_parameter(q) for q in meta["parameters"]]
        prof = load(jpath)
        exp_tables = (set(role_map["mimic_iv"]) - PUBLIC_TABLES) if db == "mimic_iv" else set(role_map[db])
        require(set(meta["tables"]) == set(prof) == exp_tables)
        for t, tinfo in meta["tables"].items():
            rho_sum = sum((params[idx][0] for _, _, idx in tinfo["queries"]), Fraction(0))
            delta_sum = sum((params[idx][2] for _, _, idx in tinfo["queries"]), Fraction(0))
            require(rho_sum == F(tinfo["rho"]) and rho_sum <= budget)
            require(delta_sum == F(tinfo["delta_threshold"]) and delta_sum <= HALF)


if __name__ == "__main__":
    verify()
    print("ALL CERTIFICATES VERIFIED (0 ERRORS)")
