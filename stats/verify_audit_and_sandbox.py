#!/usr/bin/env python3
"""Offline verifier for Table 3 zero-leakage invariants, Stage-7 DB-dedup audits, and Table 5 / Table 13 error counts."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
STATS = ROOT / "stats"

from silhouette.audit import TransmissionLedger
from silhouette.jsonl_io import load_jsonl


def verify_q201_fixture() -> dict:
    fix = json.loads((STATS / "bird_financial_q201_audit_fixture.json").read_text(encoding="utf-8"))
    concrete = fix["concrete_coincident_rows"]
    tables = {
        "account": [tuple(concrete["account_rowid3"])],
        "client": [tuple(concrete["client_rowid5"])],
        "disp": [tuple(concrete["disp_rowid3"])],
        "trans": [tuple(concrete["trans_rowid87"])],
    }
    res = {}
    for rel_name in ("DG8_CLEAN_R1", "DG8_CLEAN_R2", "DG8_CLEAN_R3", "P2_DG8_R1_ref"):
        rel = fix["releases"][rel_name]
        public = [rel["system_prompt"], *rel["header_lines"], rel["tools"]]
        prof_txt = json.dumps(rel["profile_obj"], ensure_ascii=False, separators=(",", ":"))
        ledger = TransmissionLedger(tables, rel["profile_obj"], public=public)
        segs = [
            {"kind": "prompt", "text": rel["system_prompt"] + "\n" + "\n".join(rel["header_lines"])},
            {"kind": "profile_1d", "text": prof_txt},
        ]
        for tr in rel["tool_responses"]:
            segs.append({"kind": "tool_reply", "text": json.dumps(tr["response"], ensure_ascii=False)})
        summary = ledger.record(segments=segs)
        assert summary["copied_private_tuples"] == [0, 0], rel_name
        assert summary["outside_profile_serialized_tuples"] == [0, 0], rel_name
        assert summary["full_payload_coincidences"] == [4, 4], (rel_name, summary)
        assert summary["out_of_profile_literals"] == [0, 0], rel_name
        res[rel_name] = rel["reference_transmission_ledger_full"]["full_payload_coincidences"]
    assert res["DG8_CLEAN_R1"] == res["P2_DG8_R1_ref"] == [1063628, 1063628]
    assert res["DG8_CLEAN_R2"] == [1063609, 1063609]
    assert res["DG8_CLEAN_R3"] == [1063730, 1063730]
    return res


def verify_audit_and_sandbox() -> dict:
    audit_path = STATS / "audit_summary_verified.json"
    p7_audit_path = STATS / "p7_audit_summary.json"
    dpq_audit_path = STATS / "dpq_v2_db_dedup_audit.json"
    t5_path = STATS / "table5_table13_authoritative.json"
    receipts_path = STATS / "sql_and_audit_receipts.jsonl"

    audit_doc = json.loads(audit_path.read_text(encoding="utf-8"))
    p7_audit_doc = json.loads(p7_audit_path.read_text(encoding="utf-8"))
    dpq_audit_doc = json.loads(dpq_audit_path.read_text(encoding="utf-8"))
    t5_doc = json.loads(t5_path.read_text(encoding="utf-8"))
    receipts = load_jsonl(receipts_path)

    summary = audit_doc["summary"]
    zero_leak_conds = [
        "BIRD_C0_r0",
        "BIRD_C0_r2",
        "BIRD_IND8_PROMPT",
        "BIRD_DG8_CLEAN_R1_full440",
        "BIRD_DG8_CLEAN_R2_full440",
        "BIRD_DG8_CLEAN_R3_full440",
    ]
    for cond in zero_leak_conds:
        row = summary[cond]
        assert row["slots"] == 440, f"{cond}: expected 440 slots"
        assert row["copied_unique_sum_across_dbs"] == 0, f"{cond}: copied_unique != 0"
        assert row["copied_cum"] == 0, f"{cond}: copied_cum != 0"
        assert row["ser_out_unique_sum_across_dbs"] == 0, f"{cond}: ser_out_unique != 0"
        assert row["ser_out_cum"] == 0, f"{cond}: ser_out_cum != 0"
        assert row["oob_unique_sum_across_dbs"] == 0, f"{cond}: oob_unique != 0"
        assert row["oob_cum"] == 0, f"{cond}: oob_cum != 0"
        assert row["rows_transmitted_in_sql_or_sample"] == 0, f"{cond}: rows != 0"

    assert summary["BIRD_S3"]["copied_unique_sum_across_dbs"] == 210
    assert summary["BIRD_S3"]["copied_cum"] == 8196
    assert summary["BIRD_C4_r0"]["copied_unique_sum_across_dbs"] == 198
    assert summary["BIRD_C4_r0"]["copied_cum"] == 213
    assert summary["BIRD_C4_r2"]["copied_unique_sum_across_dbs"] == 121
    assert summary["BIRD_C4_r2"]["copied_cum"] == 126

    # Verify Stage-6 & Stage-7 DB-dedup audit invariants
    p7_conds = p7_audit_doc["conditions"]
    for prefix in (
        "PUBCAP_SILHOUETTE",
        "PUBCAP_FLATW",
        "PUBCAP_OMEGA1_EQDELTA",
        "PUBCAP_UNPRUNED",
        "PUBCAP_IND8_PROFILE",
        "IND8_PROFILE_TOOL",
        "SOL_DG8_CLEAN",
    ):
        for rep in (1, 2, 3):
            c = f"{prefix}_R{rep}/r0"
            row = p7_conds[c]
            assert row["slots"] == 440, c
            assert row["copied"] == [0, 0], (c, row["copied"])
            assert row["ser_out"] == [0, 0], (c, row["ser_out"])
            assert row["oob"] == [0, 0], (c, row["oob"])
    for c, exp_slots in (
        ("BIRD_NOISELESS_V3/r0", 440),
        ("EHR874_NOISELESS_NOLKP_V3/r0", 874),
        ("EHR874_Silhouette", 874),
        ("EHR874_Silhouette_noLookup", 874),
    ):
        row = p7_conds[c]
        assert row["slots"] == exp_slots, c
        assert row["copied"] == [0, 0], c
        assert row["ser_out"] == [0, 0], c
        assert row["oob"] == [0, 0], c
    for c in (
        "PUBCAP_IND8_ROWS_R1/r0",
        "PUBCAP_IND8_ROWS_R2/r0",
        "PUBCAP_IND8_ROWS_R3/r0",
        "SOL_PUBCAP_IND8_ROWS_R1/r0",
    ):
        row = p7_conds[c]
        assert row["slots"] == 440, c
        assert row["copied"] == [0, 0], (c, row["copied"])
        assert row["ser_out"][0] > 0 and row["oob"][0] > 0, (c, row)
    for c in (
        "DP_QUERY_ROW_TP8_V2_R1/r0",
        "DP_QUERY_ROW_TP8_V2_R2/r0",
        "DP_QUERY_ROW_TP8_V2_R3/r0",
        "DP_QUERY_ROW_DB8_V2_R1/r0",
    ):
        assert c in dpq_audit_doc["conditions"] and c in p7_conds, c
        assert p7_conds[c]["copied"] == [0, 0], c

    # Verify Stage-9/Stage-10/Phase-13 DB-dedup audit invariants (105 + 10 Phase-13 -> 115 total in p7_audit_summary.json)
    assert p7_audit_doc["total_conditions"] == len(p7_conds) == 115, p7_audit_doc["total_conditions"]
    v3_split = p7_audit_doc["clinical_v3_profile_slots_split"]
    assert v3_split["total_clinical_v3_profile_slots"] == 12992
    assert v3_split["dp_clinical_profile_slots"] == 10044
    assert v3_split["noiseless_clinical_control_slots"] == 2948
    for c, exp_slots in (
        ("EHR_DG8P_V3_R1/r0", 600),
        ("EHR_DG8P_V3_R2/r0", 600),
        ("EHR_DG8P_V3_R3/r0", 600),
        ("EHR_DG8P_NOLKP_V3_R1/r0", 600),
        ("EHR_DG8P_NOLKP_V3_R2/r0", 600),
        ("EHR_DG8P_NOLKP_V3_R3/r0", 600),
        ("EHR_NOISELESS_LKP_V3/r0", 600),
        ("EHR_NOISELESS_NOLKP_V3/r0", 600),
        ("EHR_ENT8_V3_R1/r0", 600),
        ("EHR_IND8_V3_R1/r0", 600),
        ("EHR874_DG8P_V3_R1/r0", 874),
        ("EHR874_DG8P_V3_R2/r0", 874),
        ("EHR874_DG8P_V3_R3/r0", 874),
        ("EHR874_DG8P_NOLKP_V3_R1/r0", 874),
        ("EHR874_DG8P_NOLKP_V3_R2/r0", 874),
        ("EHR874_DG8P_NOLKP_V3_R3/r0", 874),
        ("EHR874_NOISELESS_LKP_V3/r0", 874),
        ("TIGHTBPE_SILHOUETTE_R1/r0", 440),
        ("TIGHTBPE_SILHOUETTE_R2/r0", 440),
        ("TIGHTBPE_SILHOUETTE_R3/r0", 440),
        ("TIGHTBPE_IND8_PROF_R1/r0", 440),
        ("TIGHTBPE_IND8_PROF_R2/r0", 440),
        ("TIGHTBPE_IND8_PROF_R3/r0", 440),
        ("DG8_NOVAL_POST_R2/r0", 440),
        ("DG8_NOVAL_POST_R3/r0", 440),
        ("TIGHTBPE_VALFIRST_SILH_R1/r0", 440),
        ("TIGHTBPE_VALFIRST_SILH_R2/r0", 440),
        ("TIGHTBPE_VALFIRST_SILH_R3/r0", 440),
        ("TIGHTBPE_VALFIRST_IND8_PROF_R1/r0", 440),
        ("TIGHTBPE_VALFIRST_IND8_PROF_R2/r0", 440),
        ("TIGHTBPE_VALFIRST_IND8_PROF_R3/r0", 440),
        ("FETCH1_PUBCAP_IND8_PROF_R1/r0", 440),
        ("FETCH1_PUBCAP_IND8_PROF_R2/r0", 440),
        ("FETCH1_PUBCAP_IND8_PROF_R3/r0", 440),
    ):
        row = p7_conds[c]
        assert row["slots"] == exp_slots, c
        assert row["copied"] == [0, 0], (c, row["copied"])
        assert row["ser_out"] == [0, 0], (c, row["ser_out"])
    for rep in (1, 2, 3):
        c = f"TIGHTBPE_STATIC_SYNTH_R{rep}/r0"
        row = p7_conds[c]
        assert row["slots"] == 440, c
        assert row["copied"] == [0, 0], (c, row["copied"])
        assert row["ser_out"][0] > 0 and row["oob"][0] > 0, (c, row)

    by_cond_bm = defaultdict(list)
    by_cond = defaultdict(list)
    for r in receipts:
        assert r.get("benchmark") in ("bird", "ehrsql_600", "ehrsql_874"), r
        assert r.get("predicted_sql") is not None, (r["condition"], r["question_id"])
        assert r.get("total_tokens") is not None, (r["condition"], r["question_id"])
        by_cond_bm[(r["benchmark"], r["condition"])].append(r)
        by_cond[r["condition"]].append(r)

    profile_receipt_conds = [
        c for c in by_cond
        if any(
            tag in c
            for tag in (
                "DG8_",
                "IND8_PROMPT",
                "IND8_PROFILE_PRUNED_TM",
                "IND8_PROFILE_TOOL",
                "ENT_V2_DG8",
                "ENT_V2_IND8_PROMPT",
                "REMAP_S2_DG8",
                "REMAP_S2_IND8_PROMPT",
                "PUBCAP_SILHOUETTE",
                "PUBCAP_FLATW",
                "PUBCAP_OMEGA1_EQDELTA",
                "PUBCAP_UNPRUNED",
                "PUBCAP_IND8_PROFILE",
                "BIRD_NOISELESS_V3",
                "EHR874_NOISELESS_NOLKP_V3",
                "EHR874_NOISELESS_LKP_V3",
                "EHR_NOISELESS_NOLKP_V3",
                "EHR_NOISELESS_LKP_V3",
                "EHR_DG8P_V3",
                "EHR_DG8P_NOLKP_V3",
                "EHR874_DG8P_V3",
                "EHR874_DG8P_NOLKP_V3",
                "EHR_ENT8_V3",
                "EHR_IND8_V3",
                "TIGHTBPE_SILHOUETTE",
                "TIGHTBPE_IND8_PROF",
                "TIGHTBPE_VALFIRST_SILH",
                "TIGHTBPE_VALFIRST_IND8_PROF",
                "FETCH1_PUBCAP_IND8_PROF",
                "DG8_NOVAL_POST",
            )
        )
    ]
    incidental_oob_slots = {
        ("TIGHTBPE_SILHOUETTE_R1/r0", 73),
        ("TIGHTBPE_SILHOUETTE_R1/r0", 129),
        ("TIGHTBPE_SILHOUETTE_R2/r0", 129),
        ("TIGHTBPE_IND8_PROF_R2/r0", 146),
        ("FETCH1_PUBCAP_IND8_PROF_R1/r0", 73),
        ("TIGHTBPE_VALFIRST_IND8_PROF_R2/r0", 119),
    }
    checked_receipt_slots = 0
    outbound_sha_slots = 0
    for r in receipts:
        if r.get("outbound_messages_sha256"):
            outbound_sha_slots += 1
    assert outbound_sha_slots == 78089, f"Expected 78089 outbound_messages_sha256 slots, got {outbound_sha_slots}"
    for rep in (1, 2, 3):
        for c in (
            f"DAWA_CLEAN_BIRD_R{rep}/r0",
            f"TIGHTBPE_VALFIRST_DAWA_CLEAN_BIRD_R{rep}/r0",
            f"TIGHTBPE_COLFIRST_DAWA_CLEAN_BIRD_R{rep}/r0",
        ):
            assert p7_conds[c]["slots"] == 440 and p7_conds[c]["copied"] == [0, 0], c
            for r in by_cond[c]:
                assert r["copied_private_tuples"] == [0, 0], (c, r["question_id"])
                assert r.get("raw_tools_exposed", 0) == 0 and r.get("raw_tools_invoked", 0) == 0
                checked_receipt_slots += 1
    assert p7_conds["DAWA_CLEAN_EHR600_R1/r0"]["slots"] == 600 and p7_conds["DAWA_CLEAN_EHR600_R1/r0"]["copied"] == [0, 0]
    for r in by_cond["DAWA_CLEAN_EHR600_R1/r0"]:
        assert r["copied_private_tuples"] == [0, 0] and r["outside_profile_serialized_tuples"] == [0, 0] and r["out_of_profile_literals"] == [0, 0]
        assert r.get("raw_tools_exposed", 0) == 0 and r.get("raw_tools_invoked", 0) == 0
        checked_receipt_slots += 1
    for cond in profile_receipt_conds:
        for r in by_cond[cond]:
            aud = r.get("audit")
            if isinstance(aud, dict) and "copied_private_tuples" in aud:
                assert aud["copied_private_tuples"] == [0, 0], f"{cond} q={r['question_id']}"
                assert aud["serialized_tuple_matches"] == [0, 0], f"{cond} q={r['question_id']}"
                if cond == "IND8_PROMPT_R3/r0" and r["question_id"] == 73:
                    assert aud["out_of_profile_literals"] == [1, 1], f"{cond} q={r['question_id']}"
                else:
                    assert aud["out_of_profile_literals"] == [0, 0], f"{cond} q={r['question_id']}"
                checked_receipt_slots += 1
            elif "copied_private_tuples" in r:
                assert r["copied_private_tuples"] == [0, 0], f"{cond} q={r['question_id']}"
                assert r["outside_profile_serialized_tuples"] == [0, 0], f"{cond} q={r['question_id']}"
                if (cond, r["question_id"]) in incidental_oob_slots:
                    assert r["out_of_profile_literals"] == [1, 1], f"{cond} q={r['question_id']}"
                else:
                    assert r["out_of_profile_literals"] == [0, 0], f"{cond} q={r['question_id']}"
                checked_receipt_slots += 1

    # Reconcile Q201 receipts against bird_financial_q201_audit_fixture.json
    q201_expected = verify_q201_fixture()
    for rep, rkey in ((1, "DG8_CLEAN_R1"), (2, "DG8_CLEAN_R2"), (3, "DG8_CLEAN_R3")):
        q201_row = next(r for r in by_cond[f"DG8_CLEAN_R{rep}/r0"] if r["question_id"] == 201)
        assert q201_row["audit"]["full_payload_coincidences"] == q201_expected[rkey], (rep, q201_row["audit"])

    # Verify SOL_REMAP_C0/r0 Q441 reconciliation
    q441_rec = next(r for r in by_cond["SOL_REMAP_C0/r0"] if r["question_id"] == 441)
    assert q441_rec["status"] == "invalid" and q441_rec["official_set_ex"] == 0 and q441_rec["strict_ex"] == 0

    # Reconcile runtime error counts in sql_and_audit_receipts.jsonl against Table 5 / Table 13
    t5_by_setting = {(row["benchmark"], row["setting"]): row for row in t5_doc}
    assert len(t5_doc) == 19, f"Expected 19 Table 5/13 rows, got {len(t5_doc)}"

    dg_clean_err = sum(
        1 for c in ("DG8_CLEAN_R1/r0", "DG8_CLEAN_R2/r0", "DG8_CLEAN_R3/r0")
        for r in by_cond[c] if r.get("runtime_error")
    )
    dg_row = t5_by_setting[("BIRD", "Silhouette Clean, R=3 (Uniform 440, 60s)")]
    assert dg_clean_err == dg_row["submitted_sql_errors_count"] == 17

    c0_err = sum(1 for c in ("C0/r0", "C0/r2") for r in by_cond_bm[("bird", c)] if r.get("runtime_error"))
    c0_row = t5_by_setting[("BIRD", "Schema only (C0 r0+r2)")]
    assert c0_err == c0_row["submitted_sql_errors_count"] == 11

    c4_err = sum(1 for c in ("C4/r0", "C4/r2") for r in by_cond_bm[("bird", c)] if r.get("runtime_error"))
    c4_row = t5_by_setting[("BIRD", "Full access (C4 r0+r2)")]
    assert c4_err == 17 and c4_row["submitted_sql_errors_count"] == 16 and c4_row["no_submission_count"] == 1

    assert t5_by_setting[("EHRSQL", "Silhouette-noLookup (DG8_NOLKP_EHR)")]["submitted_sql_errors_count"] == 5
    assert t5_by_setting[("EHRSQL", "Silhouette-noLookup (DG8_NOLKP_EHR)")]["no_submission_count"] == 1
    assert t5_by_setting[("EHRSQL", "Silhouette, R=3 (DG8P_R1..R3)")]["submitted_sql_errors_count"] == 7
    assert t5_by_setting[("EHRSQL", "Silhouette, R=3 (DG8P_R1..R3)")]["no_submission_count"] == 1
    assert t5_by_setting[("EHRSQL", "Full access (C4)")]["submitted_sql_errors_count"] == 6
    assert t5_by_setting[("EHRSQL", "Full access (C4)")]["no_submission_count"] == 3

    ehr_v3_err = sum(
        1 for c in ("EHR_DG8P_V3_R1/r0", "EHR_DG8P_V3_R2/r0", "EHR_DG8P_V3_R3/r0")
        for r in by_cond[c] if r.get("runtime_error")
    )
    ehr_v3_nosub = sum(
        1 for c in ("EHR_DG8P_V3_R1/r0", "EHR_DG8P_V3_R2/r0", "EHR_DG8P_V3_R3/r0")
        for r in by_cond[c] if r.get("no_submission") or r.get("status") == "no_submission"
    )
    assert ehr_v3_err == 9 and ehr_v3_nosub == 0, (ehr_v3_err, ehr_v3_nosub)

    # Verify turn-ordered sandbox telemetry invariants (83,015 total slots)
    te_doc = json.loads((STATS / "sandbox_turn_order_telemetry_v10.json").read_text(encoding="utf-8"))
    assert te_doc["total_slots"] == 83015, te_doc["total_slots"]
    for c in (
        "DG8_CLEAN_R1/r0", "DG8_CLEAN_R2/r0", "DG8_CLEAN_R3/r0",
        "DG1_CLEAN_R1/r0", "DG2_CLEAN_R1/r0", "DG4_CLEAN_R1/r0",
        "TIGHTBPE_VALFIRST_SILH_R1/r0", "TIGHTBPE_VALFIRST_SILH_R2/r0", "TIGHTBPE_VALFIRST_SILH_R3/r0",
        "TIGHTBPE_VALFIRST_IND8_PROF_R1/r0", "TIGHTBPE_VALFIRST_IND8_PROF_R2/r0", "TIGHTBPE_VALFIRST_IND8_PROF_R3/r0",
        "FETCH1_PUBCAP_IND8_PROF_R1/r0", "FETCH1_PUBCAP_IND8_PROF_R2/r0", "FETCH1_PUBCAP_IND8_PROF_R3/r0",
    ):
        assert te_doc["by_condition"][f"bird::{c}"]["n_telemetry_slots"] == 440
        assert te_doc["by_condition"][f"bird::{c}"]["telemetry_available"] is True
    for rep in (1, 2, 3):
        f1_te = te_doc["by_condition"][f"bird::FETCH1_PUBCAP_IND8_PROF_R{rep}/r0"]
        assert f1_te["slots_with_fetch_call"] == 440 and f1_te["fetch_calls_sum"] == 440
    assert te_doc["ehr_dg8p_v3_stage8_reconciliation"]["slots_error_then_rewrite"] == 50
    for r in receipts:
        if not r["sandbox_telemetry_available"]:
            assert r["slot_has_sandbox_error"] is None
            assert r["slot_error_then_rewrite"] is None
            assert r["slot_any_sql_revision"] is None

    return {
        "audit_bird_conditions_verified": len(summary),
        "p7_audit_conditions_verified": len(p7_conds),
        "receipt_profile_slots_verified": checked_receipt_slots,
        "outbound_sha_slots_verified": outbound_sha_slots,
        "table5_13_rows_verified": len(t5_doc),
        "bird_dg8_clean_60s_errors": dg_clean_err,
        "ehr_dg8p_v3_errors": ehr_v3_err,
        "turn_ordered_total_slots": te_doc["total_slots"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Run full offline audit & Table 5/13 checks")
    _ = parser.parse_args(argv)
    res = verify_audit_and_sandbox()
    print(
        f"AUDIT & SANDBOX VERIFIED: {res['audit_bird_conditions_verified']} BIRD S5 + "
        f"{res['p7_audit_conditions_verified']} S7 DB-dedup audit conds, "
        f"{res['receipt_profile_slots_verified']} receipt slots, "
        f"{res['table5_13_rows_verified']} Table 5/13 rows (DG8_CLEAN 60s errors={res['bird_dg8_clean_60s_errors']}/1320)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
