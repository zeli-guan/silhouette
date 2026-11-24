# Tier-1 & Tier-2 Reproducibility Guide

This guide documents all offline (**Tier-1**) verification suites, external-database / plotting (**Tier-2**) commands, the 1:1 mapping from Paper & Technical Report Tables 1–21 to `rebuild_all_tables.py` JSON keys, and the per-condition `TransmissionLedger` audit coverage.

---

## 1. Tier-1 — Offline Verification Suites (Zero External Inputs)

All Tier-1 scripts run strictly from the `artifact_release/` directory in a clean Python 3.8+ environment (`SQLite >= 3.35.0` for `AS MATERIALIZED` CTE support; paper evaluated on `SQLite 3.53.1`) with **`numpy`** and **`pytest`** only (zero references to parent/external directories or remote databases), and support both `.jsonl` and `.jsonl.gz` receipts (`stats/sql_and_audit_receipts.jsonl.gz`) transparently:

```sh
python certificates/verify_certificates.py
python certificates/verify_chain_of_custody.py --self-test
python certificates/verify_chain_of_custody.py --strict-clinical
python stats/rebuild_all_tables.py --self-test --export-json ../authoritative_paper_stats_v10.json
python stats/score_bird_dual.py --self-test
python stats/score_ehrsql_official.py --self-test
python stats/verify_audit_and_sandbox.py --self-test
python scripts/simulate_clinical_survival.py --self-test
python scripts/verify_1way_synth_factorial.py
python -m pytest -q
```

Negative custody-tamper sanity check (exits with status `1` catching legacy pre-v3 digest `d8aaf0b7...`):

```sh
python certificates/verify_chain_of_custody.py --conditions DG8P_R1/r0
```

### Summary of What Each Tier-1 Suite Verifies

1. **`certificates/verify_certificates.py`**: Checks all rational `rho_T`, discrete Gaussian variance parameters `sigma2`, exact tail bounds `tau`, `Direct-Star Entity-DP v2`, `patient_dp` composition schedules, and all `34` `DAWA+DP-SIPS` OS-CSPRNG `.dpmeta.json` sidecar certificates (`0` errors).
2. **`certificates/verify_chain_of_custody.py --strict-clinical`**: Verifies semantic `(benchmark, condition, db_id, release)` SHA-256 chain-of-custody against `manifests/custody_coverage_manifest_v10.json` across `4` profile manifests (`ehrsql_profiles_manifest`, `pubcap_releases`, `tightbpe_and_noval_manifest`, `phase13_dawa_clean_manifest`), `736` custody-pinned profile files (`261` scheduled DP profiles + `396` post-processed capped/noVal profiles + `66` static synth `.txt` files + `12` deployment-prohibited noiseless controls + `1` public dictionary JSON), `388` sidecar ledgers (`223` scheduled DP certificates + `165` post-processed sidecars), and **`45,224`** canonical receipt slots across **`96`** conditions (`EHR_DG8P_V3_R1..R3`, `EHR_DG8P_NOLKP_V3_R1..R3`, `EHR874_DG8P_V3_R1..R3`, `EHR874_DG8P_NOLKP_V3_R1..R3`, `EHR_ENT8_V3_R1`, `EHR_IND8_V3_R1`, `EHR_NOISELESS_LKP_V3`, `EHR_NOISELESS_NOLKP_V3`, `EHR874_NOISELESS_LKP_V3`, `EHR874_NOISELESS_NOLKP_V3`, `DG8_CLEAN_R1..R3`, `SOL_DG8_CLEAN_R1..R3`, `PUBCAP_*`, `TIGHTBPE_*`, `TIGHTBPE_VALFIRST_*`, `FETCH1_PUBCAP_IND8_PROF_*`, `DG8_NOVAL_POST_*`, `ENT_V2_*`, `DAWA_CLEAN_*`, `TIGHTBPE_COLFIRST_DAWA_CLEAN_*`, `TIGHTBPE_VALFIRST_DAWA_CLEAN_*`); `--self-test` additionally verifies negative cross-release swap (`R1` vs `R2`) and dropped-slot tamper rejection.
3. **`stats/rebuild_all_tables.py --self-test`**: Rebuilds every paper Table 2–9 and TR Table 13–21 directly from `stats/per_question_outcomes.jsonl` and `stats/sql_and_audit_receipts.jsonl.gz` (automatic `.jsonl`/`.jsonl.gz` support; **`83,015`** 1:1 aligned rows across **`195`** unique conditions / **`202`** `(benchmark, condition)` series; **`82,916`** non-empty / **`99`** empty submissions; `0` null `predicted_sql`/`total_tokens`; 10,000-draw paired & cluster bootstraps `seed=20261114`; 134-template `cluster_ci95_pp` on every row of `tr_table13_ehrhout`; `stage10_valfirst_and_fetch1` 8 paired comparisons + Holm families; `uniform_1d_dp_dawa_clean` 4,560-slot Clean DAWA+DP-SIPS suite; `leave_one_db_out_sensitivity` across 11 BIRD DBs; `dual_audit_reconciliation`; `exact_rademacher_p` supporting all-release unequal-$R$ sign-flip with `matched_minR_*`/`matched_r1_*` sensitivity fields; exact $2^{11}$ BIRD DB-cluster sign-flip $p$-values; MC 120/134-template cluster sign-flip $p$-values; Holm families; turn-ordered sandbox telemetry).
4. **`stats/score_bird_dual.py --self-test`**: Validates Set EX vs Strict EX comparators under unified `LocalExecutor(..., allow_cte=True)` (`SQLite 3.53` parity) across `162` conditions (**`55,453`** receipts incl. `3,960` Phase-13 DAWA+DP-SIPS, `3,960` Stage-10 TightBPE-ValFirst/Fetch1, and `2,030` BIRD-Remap Seed 1 & Seed 2 replays; `68` empty submissions, `78` `authorizer_denied:*` slots, `0` timeout/outcome conflicts; `SOL_C0/r0=319/440`, `PUBCAP_SILHOUETTE_R1/r0=189/440`, `PUBCAP_UNPRUNED_R1/r0=192/440`, `PUBCAP_OMEGA1_EQDELTA_R3/r0=205/440`), plus the 5-column `raw` vs. `protocol_audited` execution reconciliation (`stats/executor_transition_summary_v10.json` & `manifests/audit_invalid_and_flip_ledger_v10.json`).
5. **`stats/score_ehrsql_official.py --self-test`**: Validates all **`40`** EHRSQL conditions (**`27,562`** slots: `16,200` confirmation + `11,362` held-out; `31` empty, `0` denied), fail-closed gold-SQL loading via `manifests/ehrsql_gold_queries.json` (CC BY 4.0, zero patient rows), canonical v3 clinical `2x2` totals (`EHR_DG8P_V3_R1..R3=893/1,800`, `EHR_DG8P_NOLKP_V3_R1..R3=811/1,800`, `EHR874_DG8P_V3_R1..R3=1,368/2,622`, `EHR874_DG8P_NOLKP_V3_R1..R3=1,240/2,622`, `EHR_ENT8_V3_R1=298/600`, `EHR_IND8_V3_R1=296/600`, `DAWA_CLEAN_EHR600_R1=306/600`, `EHR_NOISELESS_LKP_V3=324/600`, `EHR_NOISELESS_NOLKP_V3=269/600`, `EHR874_NOISELESS_LKP_V3=457/874`, `EHR874_NOISELESS_NOLKP_V3=429/874`), unseen-37 templates (`235` queries: `115/128/129`), `132/132` `>100`-row slot parity (`0` `None`, `100%` `official_set_ex == uncapped_ex`) across all `8` distinct `>100`-row gold questions (`2` confirmation + `6` held-out), and disagreement ruleset concordance against `stats/ehrsql_v3_disagreement_audit.json`.
6. **`stats/verify_audit_and_sandbox.py --self-test`**: Verifies the `financial` `Q201` `TransmissionLedger` fixture, **`115`** DB-deduplicated audit conditions (`stats/p7_audit_summary.json`, incl. `12,992 = 10,044 + 2,948` clinical v3 profile slot split), **`78,089 / 83,015`** (`94.07%`) slot-level `outbound_messages_sha256` receipts, zero private-tuple copying on profile conditions, and turn-ordered `SchemaSandbox` telemetry (`stats/sandbox_turn_order_telemetry_v10.json`, incl. `440/440` `fetch_calls` on `FETCH1_PUBCAP_IND8_PROF_*`).
7. **`scripts/simulate_clinical_survival.py --self-test`**: Reproduces all `192` cells (`96` `whole_patient_bootstrap` + `96` `marginal_binomial`) bit-for-bit from offline-only `stats/mimic_demo94_patient_bins_deid.json` (`deployment_prohibited: true`, `db_global@100k=60.18%` / `62.68%` reachable; `per_table_reach@100k=84.10%`).
8. **`scripts/verify_1way_synth_factorial.py`**: Verifies manifest SHA-256 pinning of all `33` `pubcap_ind8_profile`, `33` evaluated `pubcap_ind8_rows`, and `66` TightBPE 1-way files, plus deterministic repeatability of `truncate_profile_to_public_cap` (exact SHA-256 identity) and the offline 1-way synthesizer/truncator (`txt_a == txt_b <= cap_tokens`) from frozen 1-way profiles.
9. **`python -m pytest -q`**: Runs the full offline unit, custody (`generate_sql` default fail-closed `verify_release_bundle` + `SingleReleaseReceiverLedger`), doc-parity, and artifact integrity test suite.

---

## 2. Tier-2 — Database Replay, Profile Regeneration, & Figure Rendering

### 2.1 Publication Figure Rendering (`matplotlib` + `numpy`)

```sh
python scripts/plot_stage6_figures.py --pdf-dir ../vldb_2027/figures --png-dir ./figures_png --dpi 150
```

### 2.2 Local SQLite Replay & Profile Regeneration (Requires Local BIRD / MIMIC-IV SQLite DBs)

1. Verify input checksums against `manifests/bird_minidev_2026_manifest.json` (`sha256`) and `manifests/ehrsql_2024_manifest.json`.
2. Replay all stored `predicted_sql` locally under 60 s `SQLite 3.53+` timeouts (`LocalExecutor(..., allow_cte=True)`):
   - `score_bird_dual.py --bird-dir` automatically builds deterministic `BIRD-Remap` Seed 1 (`S1_130`) and Seed 2 (`75`) SQLite databases in memory/tempdir using `manifests/remap_deterministic_builder_spec.json` (`build_remap_sqlite_from_manifest`), covering all `47,533` BIRD + Remap slots (`--compare-unrestricted` enabled by default).
   - `score_ehrsql_official.py --sqlite-db` automatically loads `manifests/ehrsql_gold_queries.json` (CC BY 4.0) and executes both gold and predicted SQL through `LocalExecutor(..., allow_cte=True)` (**fail-closed**: raises `RuntimeError` if any slot `question_id` is missing from the gold manifest).

```sh
python stats/score_bird_dual.py --bird-dir /path/to/bird-inputs
python stats/score_ehrsql_official.py --sqlite-db /path/to/mimic_iv.sqlite
```

3. Regenerate fresh MIMIC-IV DP profiles locally from SQLite or PhysioNet ZIP (writes outside `artifact_release/`):

```sh
python scripts/regenerate_mimic_iv_profile.py /path/to/mimic_iv.sqlite ../release-local.json --epsilon 8
```

---

## 3. Paper & Technical Report Table 1–21 to Script / JSON Key Mapping

Running `python stats/rebuild_all_tables.py --self-test --export-json ../authoritative_paper_stats_v10.json` emits a `"table_manifest"` dictionary aligned 1:1 with `Technical_Report.pdf` alongside these top-level keys (`1:1` mapped to `vldb_2027` Tables), including `dual_audit_reconciliation`, `stage10_valfirst_and_fetch1`, and `leave_one_db_out_sensitivity`:

| Paper / TR Table | LaTeX Label | JSON Top-Level Key | Primary Input Sources |
| --- | --- | --- | --- |
| Table 1 | `tab:paradigms` | *(qualitative taxonomy — non-numeric)* | Conceptual comparison in `sections/sec2_related_work.tex` |
| Table 2 | `tab:main` | `paper_table2_main` | `per_question_outcomes.jsonl`, `sql_and_audit_receipts.jsonl.gz`, `question_strata_and_remap.json` |
| Table 3 | `tab:audit` | `paper_table3_audit` | `p7_audit_summary.json`, `sql_and_audit_receipts.jsonl.gz`, `bird_financial_q201_audit_fixture.json` |
| Table 4 | `tab:ablation` | `paper_table4_ablation` (+ `tightbpe_and_noval`) | `per_question_outcomes.jsonl`, `sql_and_audit_receipts.jsonl.gz`, `tightbpe_and_noval_manifest.json` |
| Table 5 | `tab:sandbox` | `paper_table5_sandbox` | `per_question_outcomes.jsonl`, `sql_and_audit_receipts.jsonl.gz`, `sandbox_turn_order_telemetry_v10.json` |
| Table 6 | `tab:remap` | `paper_table6_remap` | `per_question_outcomes.jsonl`, `sql_and_audit_receipts.jsonl.gz`, `question_strata_and_remap.json` |
| Table 7 | `tab:sol` | `paper_table7_sol` | `per_question_outcomes.jsonl`, `sql_and_audit_receipts.jsonl.gz`, `question_strata_and_remap.json` |
| Table 8 | `tab:casestudy` | `paper_table8_casestudy` | `per_question_outcomes.jsonl`, `sql_and_audit_receipts.jsonl.gz` |
| Table 9 | `tab:entity` | `paper_table9_entity` | `per_question_outcomes.jsonl`, `sql_and_audit_receipts.jsonl.gz`, `question_strata_and_remap.json` |
| Table 10 | `tab:a1-a10` | *(compliance matrix)* | `docs/A1_A10_COMPLIANCE.md` |
| Table 11 | `tab:protocol-config` | *(protocol configuration)* | `configs/`, `docs/EXPERIMENTAL_LINEAGE.md` |
| Table 12 | `tab:role-rules` | *(deterministic DDL cascade)* | `silhouette/schema_roles.py` |
| Table 13 | `tab:ehrsql-test` | `tr_table13_ehrfull` (`+ tr_table13_ehrhout` w/ 134-tmpl `cluster_ci95_pp`) | `per_question_outcomes.jsonl`, `sql_and_audit_receipts.jsonl.gz`, `ehrsql_v3_disagreement_audit.json`, `ehrsql_874_disagreement_audit.json` |
| Table 14 | `tab:dual-audit-reconcile` | `dual_audit_reconciliation` | `executor_transition_summary_v10.json`, `audit_invalid_and_flip_ledger_v10.json`, `sql_and_audit_receipts.jsonl.gz` |
| Table 15 | `tab:stage10-valfirst` | `stage10_valfirst_and_fetch1` (+ `leave_one_db_out_sensitivity`) | `per_question_outcomes.jsonl`, `sql_and_audit_receipts.jsonl.gz`, `sandbox_turn_order_telemetry_v10.json` |
| Table 16 | `tab:dual-comparator` | `tr_table14_dual` | `dual_comparator_summary.json`, `executor_transition_summary_v10.json`, `sql_and_audit_receipts.jsonl.gz` |
| Table 17 | `tab:strata` | `tr_table15_strata` | `per_question_outcomes.jsonl`, `question_strata_and_remap.json` |
| Table 18 | `tab:holm-families` | `tr_table16_holm` | `per_question_outcomes.jsonl`, `question_strata_and_remap.json` |
| Table 19 | `tab:sandbox-full` | `tr_table17_sandbox_full` | `per_question_outcomes.jsonl`, `sql_and_audit_receipts.jsonl.gz`, `sandbox_turn_order_telemetry_v10.json` |
| Table 20 | `tab:entity-perdb` | `tr_table18_entity_perdb` | `per_question_outcomes.jsonl`, `question_strata_and_remap.json` |
| Table 21 | `tab:patient-scaling` | `tr_table19_survival` | `clinical_survival_3stage_verified.json`, `mimic_demo94_patient_bins_deid.json` |

---

## 4. `TransmissionLedger` Audit Coverage & 5-Column `raw` vs. `protocol_audited` Reconciliation

- **DB-deduplicated summary (`stats/p7_audit_summary.json`)**: Covers **`115`** conditions uniformly deduplicating unique transmitted strings/rows across the `11` BIRD DBs and `mimic_iv` (incl. `12,992` clinical v3 profile slots = `10,044` DP clinical profile slots + `2,948` noiseless clinical control slots, plus `4,560` Clean `DAWA + DP-SIPS` slots).
- **Slot-level per-request audit with `outbound_messages_sha256` (`stats/sql_and_audit_receipts.jsonl.gz`)**: Covers **`78,089 / 83,015`** (**`94.07%`**) aligned slots (`23,558` prior v9 + `46,011` Stage-10 harvested from remote run logs + `3,960` Stage-10 TightBPE-ValFirst & Fetch1 slots + `4,560` Phase-13 Clean DAWA+DP-SIPS slots; remaining `4,926` unresolved slots are `local_gpu_models_single_pass`, `MaskSQL` / `MASKSQL_FULL`, `DP_QUERY_DB8/r0`, and `DG{1,2,4}_CLEAN_R1` unchanged 4-DB carryover).
- **5-Column `raw` vs. `protocol_audited` Execution Reconciliation (`stats/executor_transition_summary_v10.json` & `manifests/audit_invalid_and_flip_ledger_v10.json`)**:

| Benchmark | Slots (`n`) | `historical_official_set_ex` | `raw_official_set_ex` | `protocol_audited_official_ex` | `raw_deliverable_set_ex` | `protocol_audited_deliverable_ex` | `status=="invalid"` (`raw=1`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `BIRD` (`47,533` orig + `3,960` S10 + `3,960` P13) | `55,453` | `24,154` | `24,169` | `24,148` | `24,141` | `24,120` | `44` (`21`) |
| `EHRSQL-600` (`15,600` + `600` P13) | `16,200` | `7,474` | `7,479` | `7,479` | `7,479` | `7,479` | `0` (`0`) |
| `EHRSQL-874` | `11,362` | `5,459` | `5,461` | `5,461` | `5,461` | `5,461` | `0` (`0`) |
| **Total** | **`83,015`** | **`37,087`** | **`37,109`** | **`37,088`** | **`37,081`** | **`37,060`** | **`44` (`21`)** |

Zeroing all `21` raw-hit `status=="invalid"` slots (including `SOL_DG8_CLEAN_R3/r0 Q412` `"abs-path:/tmp"`, reducing `SOL_DG8_CLEAN_R1..R3` from `1,061` raw hits to `1,057` audited hits) exactly reconciles `raw_*` to `protocol_audited_*`, with the remaining `1` net difference vs. `historical_official_set_ex` explained by `15` documented 60 s timeout-jitter slots.
