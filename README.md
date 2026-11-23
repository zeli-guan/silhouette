# Silhouette SQL

**Repository:** https://github.com/zeli-guan/silhouette

Reproducibility and supplementary artifact for PVLDB Vol. 20. This repository contains the `DG-v3` one-shot profile publisher, frozen BIRD and EHRSQL releases, public privacy certificates, benchmark splits, BIRD dual-comparator (with deterministic `BIRD-Remap` builder) and EHRSQL official CC-BY-4.0 gold-SQL scorers, semantic `(benchmark, condition, db_id, release)` chain-of-custody verification, fail-closed `generate_sql` release-bundle and A9 `SingleReleaseReceiverLedger` enforcement, uniform DB-deduplicated and slot-level `TransmissionLedger` audit summaries, turn-ordered `SchemaSandbox` telemetry, and reconciled per-question execution outcomes (**`78,455`** 1:1 aligned slots across **`185`** unique conditions / **`192`** `(benchmark, condition)` series).

All Tier-1 verification suites run locally on Python 3.8+ with `SQLite >= 3.35.0` (required for `AS MATERIALIZED` CTE support; paper evaluated on `SQLite 3.53.1`) and `numpy` (`requirements.txt`; plus `pytest` for the regression test suite; zero external directory imports, zero network calls, and zero non-deterministic `hash()` seeds). All JSONL readers transparently support both uncompressed `.jsonl` and gzip-compressed `.jsonl.gz` (`sql_and_audit_receipts.jsonl.gz`). See [docs/TIER1_TIER2_REPRODUCIBILITY.md](docs/TIER1_TIER2_REPRODUCIBILITY.md) for the full Table 1–21 mapping, 5-column `raw` vs. `protocol_audited` execution reconciliation, and `outbound_messages_sha256` audit coverage table.

## Verification Tiers

### Tier 1 — Offline Artifact & Certificate Verification (Zero External Inputs)

From the repository root (`artifact_release/`), run:

```sh
python certificates/verify_certificates.py && python certificates/verify_chain_of_custody.py --self-test && python certificates/verify_chain_of_custody.py --strict-clinical && python stats/rebuild_all_tables.py --self-test && python stats/score_bird_dual.py --self-test && python stats/score_ehrsql_official.py --self-test && python stats/verify_audit_and_sandbox.py --self-test && python scripts/simulate_clinical_survival.py --self-test && python scripts/verify_1way_synth_factorial.py && python baselines_phase12/src/verify_1way_provenance.py && python baselines_phase12/src/consolidate_phase12.py
```

For the full reviewer regression test suite, install `pytest` (`pytest<8.4` on Python 3.8) and run:

```sh
python -m pytest -q
```

- `certificates/verify_certificates.py`: Verifies every rational budget allocation, discrete Gaussian variance parameter, sparse-vector threshold tail enclosure, and `Direct-Star Entity-DP v2` schedule (`0` errors).
- `certificates/verify_chain_of_custody.py --strict-clinical`: Enforces 100% semantic `(benchmark, condition, db_id, release)` SHA-256 chain-of-custody verification against `manifests/custody_coverage_manifest_v10.json` across **`40,664`** canonical receipt slots in **`86`** conditions (`EHR_DG8P_V3_R1..R3`, `EHR_DG8P_NOLKP_V3_R1..R3`, `EHR874_DG8P_V3_R1..R3`, `EHR874_DG8P_NOLKP_V3_R1..R3`, `EHR_ENT8_V3_R1`, `EHR_IND8_V3_R1`, `EHR_NOISELESS_LKP_V3`, `EHR_NOISELESS_NOLKP_V3`, `EHR874_NOISELESS_LKP_V3`, `EHR874_NOISELESS_NOLKP_V3`, `DG8_CLEAN_*`, `SOL_DG8_CLEAN_*`, `PUBCAP_*`, `TIGHTBPE_*`, `TIGHTBPE_VALFIRST_*`, `FETCH1_PUBCAP_IND8_PROF_*`, `DG8_NOVAL_POST_*`, `ENT_V2_*`), all **`636`** custody-pinned profile files, and **`354`** `.dpmeta.json`/`.noval_meta.json` sidecars, plus `--self-test` negative cross-release swap and dropped-slot tamper checks. Legacy pre-v3 clinical conditions (`DG8P_R1/r0`) are tagged `"legacy_pre_v3_clinical"` and fail `--conditions DG8P_R1/r0` (`exit 1`) as expected.
- `silhouette/agent.py` (`generate_sql`, `verify_release_bundle`, `SingleReleaseReceiverLedger`): Fails closed by default (`require_pinned_digest=True`), rejecting unpinned profiles/dictionaries and `deployment_prohibited` non-DP ablation artifacts (`allow_non_dp=False`) while enforcing A9 single-release receiver isolation via `SingleReleaseReceiverLedger` before prompt construction or model calls.
- `stats/rebuild_all_tables.py --self-test --export-json path/to/authoritative_paper_stats_v10.json`: Rebuilds Paper Tables 2–9 and TR Tables 13–21 (`paper_table2_main` through `tr_table19_survival`, including `tr_table13_ehrhout` with 134-template `cluster_ci95_pp` on every row, `dual_audit_reconciliation` [TR Table 14], `stage10_valfirst_and_fetch1` [TR Table 15], `leave_one_db_out_sensitivity` across all 11 BIRD DBs, `table_manifest` aligned 1:1 with `Technical_Report.pdf`, `tightbpe_and_noval`, `scaling_and_survival`, all-release unequal-$R$ `exact_rademacher_p` with `matched_minR_*`/`matched_r1_*` sensitivities, and turn-ordered sandbox telemetry) directly from `stats/per_question_outcomes.jsonl` and `stats/sql_and_audit_receipts.jsonl.gz` (automatic `.jsonl`/`.jsonl.gz` reader support).
- `stats/score_bird_dual.py --self-test` & `stats/score_ehrsql_official.py --self-test`: Validate comparator edge cases under unified `LocalExecutor(..., allow_cte=True)` (`SQLite 3.53` parity), `153` BIRD/Remap dual conditions (`51,493` slots, `0` timeout/outcome conflicts, `68` empty submissions, `73` `authorizer_denied:*` slots, and 5-column `raw` vs. `protocol_audited` execution reconciliation against `stats/executor_transition_summary_v10.json` & `manifests/audit_invalid_and_flip_ledger_v10.json`), and all `39` EHRSQL conditions (`26,962` slots, `31` empty, `0` denied, canonical v3 `2x2` totals, `ehrsql_gold_queries.json` fail-closed validation, unseen-37 templates, `130/130` `>100`-row `uncapped_ex` parity across `8` `>100`-row gold questions with `0` `None`, and disagreement rulesets).
- `stats/verify_audit_and_sandbox.py --self-test`: Checks the `financial` `Q201` `TransmissionLedger` fixture, **`105`** DB-deduplicated conditions in `stats/p7_audit_summary.json` (plus `12,992 = 10,044 + 2,948` clinical v3 slot split), **`73,529 / 78,455`** (**`93.72%`**) slot-level `outbound_messages_sha256` receipts, and turn-ordered `SchemaSandbox` telemetry (`stats/sandbox_turn_order_telemetry_v10.json`, incl. `440/440` `fetch_calls` on `FETCH1_PUBCAP_IND8_PROF_*`).
- `scripts/simulate_clinical_survival.py --self-test`: Reproduces all `192` cells (`96` `whole_patient_bootstrap` + `96` `marginal_binomial`) of `stats/clinical_survival_3stage_verified.json` directly from offline ablation artifact `stats/mimic_demo94_patient_bins_deid.json` (`deployment_prohibited: true`) + `silhouette.dg_sampler`.
- `scripts/verify_1way_synth_factorial.py` & `baselines_phase12/src/verify_1way_provenance.py`: Verify manifest SHA-256 pinning of all `33` `pubcap_ind8_profile`, `33` evaluated `pubcap_ind8_rows`, and `66` TightBPE 1-way files, deterministic repeatability of `truncate_profile_to_public_cap` (exact SHA-256 match) and the offline 1-way synthesizer/truncator (`txt_a == txt_b <= cap_tokens`), and **33/33 bit-for-bit SHA-256 provenance** of `pubcap_ind8_rows` recorded in `manifests/verify_1way_provenance_report.json` (`matched == 33`, `total == 33`).
- `baselines_phase12/` (`dp_core.py`, `baseline_aim.py`, `baseline_privlava.py`, `baseline_privpetal.py`, `baseline_dawa_dpsips.py`, `build_all_releases.py`, `verify_1way_provenance.py`, `consolidate_phase12.py`): Exact Discrete Gaussian zCDP (`eps_T = 8`, `delta = 1e-6`, `rho_T = 0.8748`) implementations of **`AIM`** (PVLDB'22), **`PrivLava` / `Entity-PrivLava`** (SIGMOD'23), **`PrivPetal`** (SIGMOD'25), and **`DAWA + DP-SIPS`** (PVLDB'14 + PoPETs'23), together with `manifests/phase12_releases_manifest.json`, `stats/phase12_authoritative_summary.json` (**`9,917`** aligned slots across `23` conditions, `0` `fw_errors`, `0` copied private tuples), and standalone offline unit tests in `tests/test_phase12_db_baselines.py`.

### Tier 2 — Figure Regeneration & Local SQLite Benchmark Replay

To regenerate the publication vector PDFs (`fig3_frontier.pdf` through `fig6_interactive.pdf`) into `../vldb_2027/figures/` (plus optional 150-DPI PNG previews via `--png-dir`):

```sh
python scripts/plot_stage6_figures.py --pdf-dir ../vldb_2027/figures --png-dir ./figures_png --dpi 150
```

Supply locally downloaded, checksum-verified BIRD Mini-Dev or MIMIC-IV SQLite databases to replay every stored `predicted_sql` under exact 60 s SQLite comparators or regenerate fresh DP releases:

```sh
python stats/score_bird_dual.py --bird-dir /path/to/bird-inputs
python stats/score_ehrsql_official.py --sqlite-db /path/to/mimic_iv.sqlite
python scripts/regenerate_mimic_iv_profile.py /path/to/mimic_iv.sqlite ../release-local.json --epsilon 8
```

| Directory | Contents |
| --- | --- |
| `silhouette/` | Publisher (`DG-v3`, `DG-FlatW`, `DG-Omega1`, `DG-Unpruned`, `EQ`, `publish_noiseless` with `public_tables` skip, `Cap_pub` truncation), structural schema classifier, CKS20 sampler, `Direct-Star Entity-DP v2`, `ProfileLookupTool`, deterministic DP query engines (`DPQueryRowEngine`, `DPQueryRowDBWideEngine`, `DPQueryEntityEngine`), patient planner, `SchemaSandbox`, and `TransmissionLedger` audit |
| `baselines_phase12/` | Phase-12 native database DP baselines (`AIM`, `PrivLava`, `Entity-PrivLava`, `PrivPetal`, `DAWA + DP-SIPS`), exact Discrete Gaussian zCDP core (`dp_core.py`), release builder, 33/33 1-way provenance verifier (`verify_1way_provenance.py`), and 9,917-slot consolidation & paired inference script (`consolidate_phase12.py`) |
| `manifests/` | Split IDs, source hashes, question strata (`S1_130`, `gold_hidden_literal_106`, `parsed_base_tables`, `BIRD-Remap` Seed 1 & Seed 2, `entity_159`, `bird_dev_exposure_strata` `134/127/179`, `ehrsql_confirm_literal_strata` `136/464`, `ehrsql_template_clusters` `120/134/37`), `custody_coverage_manifest_v10.json` (`192` series / `78,455` slots, `636` pinned profile breakdown), `audit_invalid_and_flip_ledger_v10.json` (`44` `status=="invalid"` rows), `ehrsql_gold_queries.json` (CC BY 4.0 gold SQL templates, zero patient data), `remap_deterministic_builder_spec.json`, `pubcap_releases_manifest.json`, `verify_1way_provenance_report.json` (`33/33` SHA-256 provenance), `phase12_releases_manifest.json`, `tightbpe_and_noval_manifest.json` (`Qwen2.5` BPE token budgets incl. `tightbpe_valfirst_*`), `ehrsql_profiles_manifest.json`, `release_digests.json` |
| `certificates/` | Privacy schedules (`bird_dg_v3`, `ehrsql_dg_v3`, `patient_dp`, `bird_entity_dp`), `verify_certificates.py`, and `verify_chain_of_custody.py` |
| `profiles/` | Frozen `bird_minidev_dg_v3/` (`66`), `bird_entity_dp_eps8/`, `entity_ind8_prompt/`, `tightbpe_silhouette/` (`33`), `tightbpe_ind8_prof/` (`33`), `tightbpe_static_synth/` (`33`), `tightbpe_valfirst_silhouette/` (`33`), `tightbpe_valfirst_ind8_prof/` (`33`), `bird_dg8_noval_post/` (`33` + `33` sidecars), frozen EHRSQL releases (`ehrsql_dg8p/`, `ehrsql_ind8_prompt/`, `ehrsql_entity_demo94/`, offline-only `ehrsql_noiseless_nolkp/`), offline-only `bird_noiseless_clean/`, and `public_4_dictionaries.{sqlite,json}` |
| `scripts/` | Local MIMIC-IV regeneration (`regenerate_mimic_iv_profile.py`), portable survival simulator (`simulate_clinical_survival.py`), 1-way synth/profile factorial verifier (`verify_1way_synth_factorial.py`), and publication PDF/PNG figure generator (`plot_stage6_figures.py`) |
| `configs/`, `docs/` | Frozen prompts, tool schemas, baseline configs, `TIER1_TIER2_REPRODUCIBILITY.md`, `EXPERIMENTAL_LINEAGE.md`, and `A1_A10_COMPLIANCE.md` |
| `stats/`, `examples/`, `tests/` | Reconciled Stage-10 outcomes & receipts (`per_question_outcomes.jsonl` & `sql_and_audit_receipts.jsonl.gz` with automatic `.jsonl`/`.jsonl.gz` loading: **`78,455`** 1:1 aligned rows across **`185`** unique conditions / **`192`** series, **`73,529`** `outbound_messages_sha256` slots [`93.72%`], **`78,356`** non-empty / **`99`** empty submissions, `0` null `predicted_sql`/`total_tokens`), `phase12_authoritative_summary.json` (`23` Phase-12 baseline conditions / `9,917` slots, `0` `fw_errors`, `0` copied private tuples), `p7_audit_summary.json` (`105` DB-dedup conditions), `executor_transition_summary_v10.json` (5-column `raw` vs. `protocol_audited` reconciliation), `sandbox_turn_order_telemetry_v10.json`, `empty_probes_and_forensics_v10.json`, `fig5_exact_tau_prefix_v9.json`, `dual_comparator_summary.json` (`153` conditions), `clinical_survival_3stage_verified.json` (`whole_patient_bootstrap` + `marginal_binomial`), offline `mimic_demo94_patient_bins_deid.json`, `ehrsql_v3_disagreement_audit.json`, rebuild/scorer/audit scripts, demo, and regression test suite (`test_agent_entrypoint_custody.py`, `test_phase12_db_baselines.py`, `test_silhouette_full.py`) |

## Obtain and verify benchmark inputs

**BIRD.** Use the SQLite JSONL and SQLite databases from BIRD Mini-Dev matching SHA-256 checksums in `manifests/bird_minidev_2026_manifest.json` (`60` development + `440` evaluation IDs).

**EHRSQL 2024.** Obtain MIMIC-IV Clinical Database Demo v2.2 under its published PhysioNet terms (`manifests/ehrsql_2024_manifest.json`: `60` dev + `874` held-out test + `600` confirmation).

MIT terms cover artifact code. Use upstream benchmark data, derived releases, and model weights under their respective terms.
