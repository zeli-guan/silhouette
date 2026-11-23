"""Patient-add/remove adjacency. All visits of one subject form one unit."""
from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction
import math
from statistics import NormalDist

from .dg_sampler import rational_rho_budget, exact_dgauss_tau, conservative_eps_from_rho
from .entity_dp_publisher import _aggregate_bounded, _insert_bounded_top_c
from .profile import HISTOGRAMS, TABLE_DELTA_FRAC, canonical_val_key, bin_index, shape
from .schema_roles import plan_queries_v3, numeric, quote

CAP = 5


def patient_statistics(conn, table, column, declared_type):
    """Local exact contributions for one column with O(C) bounded memory per subject."""
    people = set()
    missing = set()
    keys, buckets = defaultdict(list), defaultdict(list)
    qc = quote(column)
    sql = ("SELECT subject_id, typeof(" + qc + "), " + qc + " FROM " + quote(table)
           + " WHERE subject_id IS NOT NULL")
    for subject, tag, value in conn.execute(sql):
        people.add(subject)
        if value is None:
            missing.add(subject)
        else:
            _insert_bounded_top_c(keys[subject], canonical_val_key(tag, value), CAP)
            _insert_bounded_top_c(
                buckets[subject], bin_index(value) if numeric(declared_type) else shape(value), CAP
            )

    values = _aggregate_bounded(keys)
    return dict(row_count=len(people), null_count=len(missing), distinct_count=len(values),
                values=values, distribution=_aggregate_bounded(buckets))


def allocate(queries, rho_budget, delta_budget):
    hist_weight = sum(Fraction(w) for _, s, _, _, w in queries if s in HISTOGRAMS)
    allocated = []
    scores = []
    for table, stat, column, role, weight in queries:
        hist = stat in HISTOGRAMS
        delta = delta_budget * Fraction(weight) / hist_weight if hist else Fraction(0)
        sensitivity2 = CAP if hist else CAP ** 2 if stat == "distinct_count" else 1
        kappa = -NormalDist().inv_cdf(float(delta / CAP)) if hist else 2.5
        scores.append((float(weight) * kappa * math.sqrt(sensitivity2)) ** (2.0 / 3.0))
        allocated.append(dict(table=table, stat=stat, column=column, role=role,
                              weight_frac=Fraction(weight), delta=delta, sensitivity2=sensitivity2))
    total_score = math.fsum(scores)
    shares = [max(1, round(10**12 * x / total_score)) for x in scores]
    total = sum(shares)
    for q, share in zip(allocated, shares):
        q["r_j"] = share
        q["share"] = Fraction(share, total)
        q["rho"] = rho_budget * q["share"]
        q["sigma2"] = q["sensitivity2"] / (2 * q["rho"])
        if q["delta"]:
            q["tau"], q["bounds"] = exact_dgauss_tau(q["sigma2"], q["delta"] / CAP)
    return allocated


def plan_patient(schema, epsilon, schedule="per_table"):
    """Plan independent table budgets, a uniform DB split, or one global convex allocation."""
    rho = rational_rho_budget(epsilon, TABLE_DELTA_FRAC / 2)
    groups = [[(table, s, c, role, w) for s, c, role, w in plan_queries_v3(cols)]
              for table, cols in schema.tables.items() if any(c.name == "subject_id" for c in cols)]
    if not groups or schedule not in ("per_table", "db_uniform", "db_global"):
        raise ValueError("Expected patient-linked tables and a known composition schedule")
    if schedule == "db_global":
        groups = [sum(groups, [])]
    divisor = len(groups) if schedule == "db_uniform" else 1
    queries = [q for group in groups for q in allocate(group, rho / divisor, TABLE_DELTA_FRAC / (2 * divisor))]
    composed_rho = sum(q["rho"] for q in queries)
    threshold = sum(q["delta"] for q in queries)
    return dict(queries=queries, rho=composed_rho, delta=2 * threshold,
                epsilon_upper=conservative_eps_from_rho(composed_rho, threshold))
