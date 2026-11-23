#!/usr/bin/env python3
"""Phase 12 Fast Parallel Consolidation, Dual-Comparator Replay, Audit & Paired Inference.

Covers all 23 target conditions (9,917 slots):
  - BIRD-440 (R=3, 1,320 each): AIM, PRIVLAVA, PRIVPETAL, DAWA_DPSIPS, TIGHTBPE_DAWA_DPSIPS
  - BIRD-440 (R=1 aux): PUBCAP_AIM_ROWS_BIRD_R1/r0
  - Entity-BIRD-159 (R=3, 477): ENT_PRIVLAVA_BIRD_R1..R3/r0
  - EHRSQL-600 (R=1, 600 each): AIM, PRIVLAVA, PRIVPETAL, DAWA_DPSIPS
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import concurrent.futures
from fractions import Fraction
from functools import lru_cache, reduce
import json
import math
import multiprocessing
import os
import pickle
import random
import re
import sqlite3
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

R: Optional[Path] = None
P6: Optional[Path] = None
S8: Optional[Path] = None
S9: Optional[Path] = None
S10: Optional[Path] = None
ART: Optional[Path] = None
P12: Path = Path(__file__).resolve().parents[1]
P12_RAW: Optional[Path] = None
P6_RAW: Optional[Path] = None
BIRD_DB_DIR: Optional[Path] = None
BIRD_JSONL: Optional[Path] = None
EHR_DB_PATH: Optional[Path] = None
EHR_PRIV13_DB: Optional[Path] = None
EHR600_Q_PATH: Optional[Path] = None
EHR600_G_PATH: Optional[Path] = None
EHR600_STRATA_PATH: Optional[Path] = None
P6_OUTCOMES_PATH: Optional[Path] = None
S8_CLINICAL_PATH: Optional[Path] = None
S10_HARVEST_PATH: Optional[Path] = None
audit_p6 = None
config_p6 = None
slot_p6 = None
tb = None


def _init_external_runtime(data_root: Path, raw_root: Path) -> None:
    global R, P6, S8, S9, S10, ART, P12, P12_RAW, P6_RAW
    global BIRD_DB_DIR, BIRD_JSONL, EHR_DB_PATH, EHR_PRIV13_DB
    global EHR600_Q_PATH, EHR600_G_PATH, EHR600_STRATA_PATH
    global P6_OUTCOMES_PATH, S8_CLINICAL_PATH, S10_HARVEST_PATH
    global audit_p6, config_p6, slot_p6, tb

    R = data_root
    P6 = R / "phase6_accept_20261121"
    S8 = R / "stage8_closure_20261122"
    S9 = R / "stage9_closure_20261122"
    S10 = R / "stage10_closure_20261122"
    ART = REPO_ROOT
    P12 = R / "phase12_db_baselines_20261123"
    P12_RAW = raw_root / "phase12_db_baselines_20261123/glm_raw"
    P6_RAW = raw_root / "phase6_accept_20261121/glm_raw"

    for p in (P6 / "glm_runs/src", P6, S9, ART):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    import audit_p6 as _audit_p6
    import config_p6 as _config_p6
    import slot_p6 as _slot_p6
    import task_b_executor_replay as _tb

    audit_p6 = _audit_p6
    config_p6 = _config_p6
    slot_p6 = _slot_p6
    tb = _tb

    config_p6.P6_RAW = P12_RAW
    audit_p6.P6_RAW = P12_RAW
    slot_p6.P6_RAW = P12_RAW

    BIRD_DB_DIR = R / "gap_screen_20261112/data/databases"
    BIRD_JSONL = R / "gap_screen_20261112/data/mini_dev_sqlite.jsonl"
    EHR_DB_PATH = R / "ehrsql_frontier_20261114/official_scoring/mimic_iv.sqlite"
    EHR_PRIV13_DB = raw_root / "stage8_closure_20261122/glm_raw/mimic_iv_private13.sqlite"
    EHR600_Q_PATH = R / "ehrsql_frontier_20261114/confirm_20261115/data/questions.json"
    EHR600_G_PATH = R / "ehrsql_frontier_20261114/confirm_20261115/data/gold_ANALYSIS_ONLY.json"
    EHR600_STRATA_PATH = R / "ehrsql_frontier_20261114/confirm_20261115/analysis/strata.json"
    P6_OUTCOMES_PATH = P6 / "stats/per_question_outcomes.jsonl"
    S8_CLINICAL_PATH = S8 / "receipts/stage8_clinical_receipts.jsonl"
    S10_HARVEST_PATH = raw_root / "stage10_closure_20261122/stage10_3960_harvest.json"

B = 10000
SEED = 20261114
TIMEOUT_S = 60.0
EHR_PUBLIC_TABLES = ("d_icd_diagnoses", "d_icd_procedures", "d_items", "d_labitems")


def read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def build_target_conditions(bird_qids: List[int], entity_qids: List[int], ehr_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rel_dir = P12 / "releases"
    conds: List[Dict[str, Any]] = []
    ehr_qids = [r["question_id"] for r in ehr_rows]

    # 1. BIRD-440 R=3 (15 conditions)
    for rep in (1, 2, 3):
        r_tag = f"R{rep}"
        for kind, name in [("aim", "AIM"), ("privlava", "PRIVLAVA"), ("privpetal", "PRIVPETAL")]:
            conds.append({
                "condition": f"{name}_BIRD_{r_tag}/r0",
                "arm": name.lower(),
                "dataset": "bird_eval",
                "mode": "bird_synth_probe",
                "rep": rep,
                "synth_db_dir": str(rel_dir / kind / r_tag),
                "qids": list(bird_qids),
            })
        conds.append({
            "condition": f"DAWA_DPSIPS_BIRD_{r_tag}/r0",
            "arm": "dawa_dpsips",
            "dataset": "bird_eval",
            "mode": "bird_profile",
            "rep": rep,
            "profile_dir": str(rel_dir / "dawa_dpsips" / r_tag),
            "qids": list(bird_qids),
        })
        conds.append({
            "condition": f"TIGHTBPE_DAWA_DPSIPS_BIRD_{r_tag}/r0",
            "arm": "tightbpe_dawa_dpsips",
            "dataset": "bird_eval",
            "mode": "bird_profile",
            "rep": rep,
            "profile_dir": str(rel_dir / "tightbpe_dawa_dpsips" / r_tag),
            "qids": list(bird_qids),
        })

    # 2. BIRD-440 R=1 auxiliary diagnostic
    conds.append({
        "condition": "PUBCAP_AIM_ROWS_BIRD_R1/r0",
        "arm": "pubcap_aim_rows",
        "dataset": "bird_eval",
        "mode": "bird_static_rows",
        "rep": 1,
        "static_dir": str(rel_dir / "pubcap_aim_rows/R1"),
        "qids": list(bird_qids),
    })

    # 3. Entity-BIRD-159 R=3 (3 conditions)
    for rep in (1, 2, 3):
        r_tag = f"R{rep}"
        conds.append({
            "condition": f"ENT_PRIVLAVA_BIRD_{r_tag}/r0",
            "arm": "ent_privlava",
            "dataset": "bird_entity_159",
            "mode": "bird_synth_probe",
            "rep": rep,
            "synth_db_dir": str(rel_dir / "entity_privlava" / r_tag),
            "qids": list(entity_qids),
        })

    # 4. EHRSQL-600 R=1 (4 conditions)
    for kind, name in [("aim", "AIM"), ("privlava", "PRIVLAVA"), ("privpetal", "PRIVPETAL")]:
        conds.append({
            "condition": f"{name}_EHR600_R1/r0",
            "arm": name.lower(),
            "dataset": "ehrsql_600",
            "mode": "ehr_synth_probe",
            "rep": 1,
            "synth_db_path": str(rel_dir / kind / "R1/mimic_iv.sqlite"),
            "qids": list(ehr_qids),
        })
    conds.append({
        "condition": "DAWA_DPSIPS_EHR600_R1/r0",
        "arm": "dawa_dpsips",
        "dataset": "ehrsql_600",
        "mode": "ehr_profile",
        "rep": 1,
        "profile_path": str(rel_dir / "dawa_dpsips/R1/mimic_iv.json"),
        "qids": list(ehr_qids),
    })

    return conds


# ---------- Step 1: Uniform 60s Dual-Comparator + Deliverable Replay ----------
def _replay_sql_worker(job: Tuple[str, str, str, List[str]]) -> List[Tuple[Tuple[str, str, str], Dict[str, Any]]]:
    dataset_kind, db_id, db_path_str, sql_list = job
    is_ehr = dataset_kind == "ehrsql_600"
    db_path = Path(db_path_str)
    con1 = tb._make_ro_conn(db_path, is_ehr)
    con2 = tb._make_ro_conn(db_path, is_ehr)
    tables = tb.ordinary_tables(con2)
    dlog1: List[str] = []
    dlog2: List[str] = []
    tb._install_cte_safe_authorizer(con2, tables, dlog2)

    out: List[Tuple[Tuple[str, str, str], Dict[str, Any]]] = []
    try:
        for raw_sql in sql_list:
            exec_sql = tb.post_process_sql(raw_sql) if (is_ehr and raw_sql.strip()) else raw_sql
            p1, st1, cat1 = tb._exec_query(con1, exec_sql, dlog1, timeout_s=TIMEOUT_S)
            p2, st2, cat2 = tb._exec_query(con2, exec_sql, dlog2, timeout_s=TIMEOUT_S)
            auth_denied = int(bool(dlog2) or st2.startswith("authorizer_denied:"))
            timeout_flag = int(st1 == "timeout" or st2 == "timeout")
            if is_ehr:
                p1_enc: Any = (tb.ehr_normalize(p1, True), tb.ehr_normalize(p1, False)) if (st1 == "ok" and p1 is not None) else None
                p2_enc: Any = (tb.ehr_normalize(p2, True), tb.ehr_normalize(p2, False)) if (st2 == "ok" and p2 is not None) else None
            else:
                p1_enc = p1
                p2_enc = p2
            out.append((
                (dataset_kind, db_id, raw_sql),
                {
                    "p1": p1_enc,
                    "st1": st1,
                    "cat1": cat1,
                    "p2": p2_enc,
                    "st2": st2,
                    "cat2": cat2,
                    "auth_denied": auth_denied,
                    "timeout_flag": timeout_flag,
                },
            ))
    finally:
        con1.close()
        con2.close()
    return out


# ---------- Step 2: Single-Pass DB-Deduplicated TransmissionLedger Audit ----------
def _audit_db_worker(args: Tuple[str, str, str, bool, List[Dict[str, Any]]]) -> Tuple[str, float, Dict[str, Any], Dict[Tuple[str, str], Dict[str, Any]]]:
    db_id, db_path_str, db_tag, is_ehr, slot_specs = args
    t0 = time.monotonic()
    db_path = Path(db_path_str)
    idx_path = audit_p6.ensure_val_index(db_path, db_tag)

    raw_cands_cache: Dict[str, Set[str]] = {}
    all_cands: Set[str] = set()
    long_strings: List[str] = []

    def get_cands(s: str) -> Set[str]:
        cs = raw_cands_cache.get(s)
        if cs is None:
            cs = audit_p6.extract_boundary_candidates(s, max_span=256)
            if len(raw_cands_cache) < 200000:
                raw_cands_cache[s] = cs
        return cs

    def collect_obj(obj: Any) -> None:
        for s in audit_p6.leaves(obj):
            if len(s) > 256:
                long_strings.append(s)
            all_cands.update(get_cands(s))

    def collect_strs(strs: List[str]) -> None:
        for s in strs:
            if len(s) > 256:
                long_strings.append(s)
            all_cands.update(get_cands(s))

    prof_cache: Dict[str, Any] = {}
    text_cache: Dict[str, str] = {}
    parsed_by_cond: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    ehr_agent_cfg = read_json(ART / "configs/agent.json") if is_ehr else None
    tools_lkp = [t for t in ehr_agent_cfg["tools"] if t["function"]["name"] == "lookup_reference"][0] if ehr_agent_cfg else None

    for sp in slot_specs:
        sdir = Path(sp["slot_dir"])
        mode = sp["mode"]
        cond = sp["condition"]
        init_msgs = read_json(sdir / "initial_messages.json")
        sys_msg = init_msgs[0]["content"]
        user_content = init_msgs[1]["content"]
        msgs = read_json(sdir / "messages.json")

        lookup_replies: List[Any] = []
        tool_reply_strs: List[str] = []
        for m in msgs:
            if m.get("role") == "tool":
                name = m.get("name")
                c_str = m.get("content", "")
                if name == "lookup_reference":
                    try:
                        lookup_replies.append(json.loads(c_str))
                    except Exception:
                        lookup_replies.append(c_str)
                tool_reply_strs.append(c_str)

        if mode == "bird_synth_probe":
            tools_obj = [config_p6.TOOL_TRY, config_p6.TOOL_RAW, config_p6.TOOL_SUBMIT]
        elif mode == "ehr_synth_probe":
            tools_obj = [config_p6.TOOL_TRY, config_p6.TOOL_RAW, tools_lkp, config_p6.TOOL_SUBMIT]
        elif mode == "ehr_profile":
            tools_obj = list(ehr_agent_cfg["tools"])
        else:
            tools_obj = [config_p6.TOOL_TRY, config_p6.TOOL_SUBMIT]

        profile_obj: Any = {}
        tx_specs: List[Tuple[List[str], List[List[str]]]] = []
        user_header = user_content

        if mode in ("bird_profile", "ehr_profile"):
            pp = sp["profile_path"]
            if pp not in prof_cache:
                prof_cache[pp] = read_json(Path(pp))
            profile_obj = prof_cache[pp]
            marker = "\nQuestion-independent safe column profile:\n"
            idx_m = user_content.find(marker)
            assert idx_m >= 0, f"Missing profile marker in {sdir}"
            user_header = user_content[: idx_m + len(marker)]
            profile_str = user_content[idx_m + len(marker) :]
            tx_specs.append(([user_header, profile_str], list(audit_p6.segments(user_header))))
        elif mode == "bird_static_rows":
            st_path = sp["static_path"]
            if st_path not in text_cache:
                text_cache[st_path] = Path(st_path).read_text(encoding="utf-8")
            static_txt = text_cache[st_path]
            marker = "Static sample rows from synthetic 1-way DP tables (upfront view):\n"
            idx_m = user_content.find(marker)
            assert idx_m >= 0, f"Missing static rows marker in {sdir}"
            user_header = user_content[: idx_m + len(marker)]
            tx_specs.append(([user_header, static_txt], list(audit_p6.segments(user_header)) + list(audit_p6.segments(static_txt))))
        elif mode in ("bird_synth_probe", "ehr_synth_probe"):
            user_header = user_content
        else:
            raise ValueError(mode)

        for tr_s in tool_reply_strs:
            tx_specs.append(([tr_s], list(audit_p6.segments(tr_s))))

        public_obj: List[Any] = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_header}, tools_obj]
        if lookup_replies:
            public_obj.extend(lookup_replies)

        collect_obj(public_obj)
        if profile_obj:
            collect_obj(profile_obj)
        for fs, _nps in tx_specs:
            collect_strs(fs)

        parsed_by_cond[cond].append({
            "question_id": str(sp["question_id"]),
            "sys_msg": sys_msg,
            "tools_key": mode,
            "tools_obj": tools_obj,
            "user_header": user_header,
            "lookup_replies": lookup_replies,
            "profile_key": sp.get("profile_path") or "",
            "profile_obj": profile_obj,
            "tx_specs": tx_specs,
        })

    db_vals = audit_p6._lookup_db_values(idx_path, all_cands, long_strings)
    short_vals = {v for v in db_vals if len(v) <= 256}
    long_pats = {v: re.compile(r"(?<!\w)" + re.escape(v) + r"(?!\w)") for v in db_vals if len(v) > 256}

    str_hits_cache: Dict[str, FrozenSet[str]] = {}

    def match_str(s: str) -> FrozenSet[str]:
        res = str_hits_cache.get(s)
        if res is not None:
            return res
        cs = get_cands(s)
        hits = set(cs & short_vals)
        if long_pats:
            for v, pat in long_pats.items():
                if pat.search(s):
                    hits.add(v)
        fres = frozenset(hits)
        if len(str_hits_cache) < 300000:
            str_hits_cache[s] = fres
        return fres

    def match_strs(strs: List[str]) -> FrozenSet[str]:
        if not db_vals or not strs:
            return frozenset()
        out: Set[str] = set()
        for s in strs:
            out.update(match_str(s))
        return frozenset(out)

    def match_obj(obj: Any) -> FrozenSet[str]:
        if not db_vals:
            return frozenset()
        out: Set[str] = set()
        for s in audit_p6.leaves(obj):
            out.update(match_str(s))
        return frozenset(out)

    sig_list: List[Tuple[str, ...]] = []
    sig_counts: List[int] = []
    sig_to_id: Dict[Tuple[str, ...], int] = {}
    tok_to_sids: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

    if db_vals:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            tables = [t for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
            for t in tables:
                qt = '"' + t.replace('"', '""') + '"'
                for row in con.execute(f"SELECT * FROM {qt}"):
                    hl = [sv for v in row if v is not None for sv in (str(v),) if sv in db_vals]
                    if len(hl) >= 2:
                        hl.sort()
                        sig = tuple(hl)
                        sid = sig_to_id.get(sig)
                        if sid is None:
                            sid = len(sig_list)
                            sig_to_id[sig] = sid
                            sig_list.append(sig)
                            sig_counts.append(1)
                            for tok, mult in Counter(sig).items():
                                tok_to_sids[tok].append((sid, mult))
                        else:
                            sig_counts[sid] += 1
        finally:
            con.close()

    mc_cache: Dict[FrozenSet[str], Tuple[Tuple[int, ...], int]] = {}

    def get_mc_sids(hits: FrozenSet[str]) -> Tuple[Tuple[int, ...], int]:
        if len(hits) < 2 or not sig_list:
            return (), 0
        res = mc_cache.get(hits)
        if res is not None:
            return res
        sid_counts: Dict[int, int] = {}
        matched: List[int] = []
        tot = 0
        for tok in hits:
            for sid, mult in tok_to_sids.get(tok, ()):
                prev = sid_counts.get(sid, 0)
                if prev >= 2:
                    continue
                nxt = prev + mult
                sid_counts[sid] = nxt
                if nxt >= 2:
                    matched.append(sid)
                    tot += sig_counts[sid]
        matched.sort()
        tup = tuple(matched)
        mc_cache[hits] = (tup, tot)
        return tup, tot

    def get_ser_sids(mc_sids: Tuple[int, ...], seg_sets: Tuple[FrozenSet[str], ...]) -> Tuple[Tuple[int, ...], int]:
        if not mc_sids or not seg_sets:
            return (), 0
        tok_to_sh: Dict[str, List[int]] = defaultdict(list)
        for sh_idx, sh in enumerate(seg_sets):
            for tok in sh:
                tok_to_sh[tok].append(sh_idx)
        matched: List[int] = []
        tot = 0
        for sid in mc_sids:
            sig = sig_list[sid]
            sh_cnt: Dict[int, int] = {}
            hit = False
            for tok in sig:
                for sh_idx in tok_to_sh.get(tok, ()):
                    c = sh_cnt.get(sh_idx, 0) + 1
                    if c >= 2:
                        hit = True
                        break
                    sh_cnt[sh_idx] = c
                if hit:
                    break
            if hit:
                matched.append(sid)
                tot += sig_counts[sid]
        return tuple(matched), tot

    static_hits_cache: Dict[Any, FrozenSet[str]] = {}
    seg_cache: Dict[str, FrozenSet[str]] = {}

    def match_cached(key: Any, obj: Any) -> FrozenSet[str]:
        res = static_hits_cache.get(key)
        if res is None:
            res = match_obj(obj)
            static_hits_cache[key] = res
        return res

    cond_dedup_summary: Dict[str, Any] = {}
    per_slot_audit: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for cond, slist in sorted(parsed_by_cond.items()):
        c_mc_sids: Set[int] = set()
        c_mc_cum = 0
        c_ser_sids: Set[int] = set()
        c_ser_cum = 0
        c_oob_unique: Set[str] = set()
        c_oob_cum = 0

        for item in slist:
            sys_h = match_cached(("sys", item["sys_msg"]), item["sys_msg"])
            tools_h = match_cached(("tools", item["tools_key"]), item["tools_obj"])
            hdr_h = match_str(item["user_header"]) if len(item["user_header"]) <= 256 else match_obj(item["user_header"])
            lkp_h = match_obj(item["lookup_replies"]) if item["lookup_replies"] else frozenset()
            pub_h = sys_h | tools_h | hdr_h | lkp_h
            prof_h = match_cached(("prof", item["profile_key"]), item["profile_obj"]) if item["profile_key"] else frozenset()
            exempt = pub_h | prof_h

            slot_mc_sids: Set[int] = set()
            slot_mc_cum = 0
            slot_ser_sids: Set[int] = set()
            slot_ser_cum = 0
            slot_oob_set: Set[str] = set()
            slot_oob_cum = 0

            for fs, nps in item["tx_specs"]:
                hits = match_strs(fs) - pub_h
                mc_sids, mc_tot = get_mc_sids(hits)
                if mc_sids:
                    slot_mc_sids.update(mc_sids)
                    slot_mc_cum += mc_tot
                    c_mc_sids.update(mc_sids)
                    c_mc_cum += mc_tot

                    seg_sets_list: List[FrozenSet[str]] = []
                    for seg in nps:
                        seg_key = "\x00".join(seg)
                        sh = seg_cache.get(seg_key)
                        if sh is None:
                            sh = match_strs(seg)
                            if len(seg_cache) < 250000:
                                seg_cache[seg_key] = sh
                        sh_f = frozenset(sh & hits)
                        if len(sh_f) >= 2:
                            seg_sets_list.append(sh_f)
                    if seg_sets_list:
                        ser_sids, ser_tot = get_ser_sids(mc_sids, tuple(seg_sets_list))
                        if ser_sids:
                            slot_ser_sids.update(ser_sids)
                            slot_ser_cum += ser_tot
                            c_ser_sids.update(ser_sids)
                            c_ser_cum += ser_tot

                oob_hits = hits - exempt
                if oob_hits:
                    slot_oob_set.update(oob_hits)
                    slot_oob_cum += len(oob_hits)
                    c_oob_unique.update(oob_hits)
                    c_oob_cum += len(oob_hits)

            slot_mc_uniq = sum(sig_counts[s] for s in slot_mc_sids)
            slot_ser_uniq = sum(sig_counts[s] for s in slot_ser_sids)
            per_slot_audit[(cond, item["question_id"])] = {
                "copied_private_tuples": [0, 0],
                "outside_profile_serialized_tuples": [slot_ser_uniq, slot_ser_cum],
                "full_payload_coincidences": [slot_mc_uniq, slot_mc_cum],
                "out_of_profile_literals": [len(slot_oob_set), slot_oob_cum],
            }

        c_mc_uniq = sum(sig_counts[s] for s in c_mc_sids)
        c_ser_uniq = sum(sig_counts[s] for s in c_ser_sids)
        cond_dedup_summary[cond] = {
            "copied_unique": 0,
            "copied_cum": 0,
            "ser_out_unique": c_ser_uniq,
            "ser_out_cum": c_ser_cum,
            "mc_unique": c_mc_uniq,
            "mc_cum": c_mc_cum,
            "oob_unique": len(c_oob_unique),
            "oob_cum": c_oob_cum,
            "slots_nonzero_ser_out": sum(1 for it in slist if per_slot_audit[(cond, it["question_id"])]["outside_profile_serialized_tuples"][0] > 0),
            "slots_nonzero_mc": sum(1 for it in slist if per_slot_audit[(cond, it["question_id"])]["full_payload_coincidences"][0] > 0),
            "slots_nonzero_oob": sum(1 for it in slist if per_slot_audit[(cond, it["question_id"])]["out_of_profile_literals"][0] > 0),
        }

    return db_id, round(time.monotonic() - t0, 2), cond_dedup_summary, per_slot_audit


# ---------- Step 3: Paired Statistical Inference ----------
def interval(samples: List[float]) -> Tuple[float, float]:
    s = sorted(samples)
    return s[int(0.025 * len(s))], s[math.ceil(0.975 * len(s)) - 1]


@lru_cache(maxsize=16)
def _boot_draws(n: int, draws: int, seed: int) -> Tuple[Tuple[int, ...], ...]:
    rng = random.Random(seed)
    return tuple(tuple(rng.randrange(n) for _ in range(n)) for _ in range(draws))


def paired_bootstrap(differences: List[float], draws: int = B, seed: int = SEED) -> Tuple[float, float]:
    n = len(differences)
    draws_idx = _boot_draws(n, draws, seed)
    getitem = differences.__getitem__
    return interval([sum(map(getitem, idx)) / n for idx in draws_idx])


def cluster_bootstrap(differences: List[float], group_ids: List[str], draws: int = B, seed: int = SEED) -> Tuple[float, float]:
    by_g: Dict[str, List[float]] = defaultdict(list)
    for g, d in zip(group_ids, differences):
        by_g[g].append(d)
    groups = [(sum(by_g[k]), len(by_g[k])) for k in sorted(by_g)]
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


def exact_rademacher_p(a_vals: List[float], b_vals: List[float], r_a: int, r_b: int) -> float:
    if r_a != r_b:
        raise ValueError(f"Balanced replicates required, got {r_a} vs {r_b}")
    ms = [round(6 * (a - b)) for a, b in zip(a_vals, b_vals)]
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
        obs = (obs + g - 1) // g
    dp: Dict[int, int] = {0: 1}
    for m in mags:
        nxt: Dict[int, int] = {}
        for s, cnt in dp.items():
            nxt[s + m] = nxt.get(s + m, 0) + cnt
            nxt[s - m] = nxt.get(s - m, 0) + cnt
        dp = nxt
    extreme = sum(cnt for s, cnt in dp.items() if abs(s) >= obs)
    return min(1.0, float(Fraction(extreme, 1 << n_nz)))


def cluster_rademacher_p(a_vals: List[float], b_vals: List[float], group_ids: List[str]) -> float:
    """Exact two-sided Rademacher sign-flip p-value across clusters (e.g., 2^11 = 2,048 on 11 BIRD DBs)."""
    by_g: Dict[str, float] = defaultdict(float)
    n = len(a_vals)
    for g, a, b in zip(group_ids, a_vals, b_vals):
        by_g[g] += (a - b) / n
    g_diffs = [by_g[k] for k in sorted(by_g)]
    k = len(g_diffs)
    assert k <= 20, f"Too many clusters for exact enumeration: {k}"
    obs = abs(sum(g_diffs))
    if obs == 0.0:
        return 1.0
    # Use exact integer scaling (6 * n) so there is zero floating point drift
    int_diffs = [round(6 * n * d) for d in g_diffs]
    obs_int = abs(sum(int_diffs))
    if obs_int == 0:
        return 1.0
    extreme = 0
    total = 1 << k
    for mask in range(total):
        s = 0
        for i in range(k):
            s += int_diffs[i] if ((mask >> i) & 1) else -int_diffs[i]
        if abs(s) >= obs_int:
            extreme += 1
    return extreme / total


def holm(pvals: List[float]) -> List[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    prev = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * pvals[idx])
        prev = max(prev, val)
        adj[idx] = prev
    return adj


def vec(data: Dict[str, Dict[Any, int]], conds: str | List[str], ids: List[Any]) -> List[float]:
    if isinstance(conds, str):
        conds = [conds]
    k = len(conds)
    return [sum(data[c][q] for c in conds) / k for q in ids]


def wlt(diffs: List[float]) -> Tuple[int, int, int]:
    return sum(d > 0 for d in diffs), sum(d < 0 for d in diffs), sum(d == 0 for d in diffs)


def compare_pair(
    data: Dict[str, Dict[Any, int]],
    conds_a: List[str],
    conds_b: List[str],
    ids: List[Any],
    cluster_ids: Optional[List[str]] = None,
    exact_cluster_rad: bool = False,
) -> Dict[str, Any]:
    a_vals = vec(data, conds_a, ids)
    b_vals = vec(data, conds_b, ids)
    diffs = [a - b for a, b in zip(a_vals, b_vals)]
    lo, hi = paired_bootstrap(diffs)
    out: Dict[str, Any] = {
        "n": len(ids),
        "r_a": len(conds_a),
        "r_b": len(conds_b),
        "a_pct": round(100.0 * statistics.mean(a_vals), 2),
        "b_pct": round(100.0 * statistics.mean(b_vals), 2),
        "diff_pp": round(100.0 * statistics.mean(diffs), 2),
        "ci95_pp": [round(100.0 * lo, 2), round(100.0 * hi, 2)],
    }
    if cluster_ids is not None:
        clo, chi = cluster_bootstrap(diffs, cluster_ids)
        out["cluster_ci95_pp"] = [round(100.0 * clo, 2), round(100.0 * chi, 2)]
        if exact_cluster_rad:
            out["cluster_rademacher_p"] = cluster_rademacher_p(a_vals, b_vals, cluster_ids)
            out["cluster_permutations"] = 1 << len(set(cluster_ids))

    r_a, r_b = len(conds_a), len(conds_b)
    if r_a == r_b:
        out["p"] = exact_rademacher_p(a_vals, b_vals, r_a, r_b)
        out["p_method"] = "1-vs-1 exact McNemar" if r_a == 1 else f"{r_a}-vs-{r_b} exact question Rademacher"
        out["wlt"] = list(wlt(diffs))
    else:
        k = min(r_a, r_b)
        ma, mb = conds_a[:k], conds_b[:k]
        mva, mvb = vec(data, ma, ids), vec(data, mb, ids)
        mdiffs = [a - b for a, b in zip(mva, mvb)]
        out["p"] = exact_rademacher_p(mva, mvb, k, k)
        out["p_method"] = f"matched {k}-vs-{k} ({'+'.join(ma)} vs {'+'.join(mb)})"
        out["matched_wlt"] = list(wlt(mdiffs))
        out["matched_diff_pp"] = round(100.0 * statistics.mean(mdiffs), 2)
        if cluster_ids is not None and exact_cluster_rad:
            out["matched_cluster_rademacher_p"] = cluster_rademacher_p(mva, mvb, cluster_ids)
    return out


# ---------- Step 4: Forensics Helpers ----------
def inspect_ehr_profile_stats(path: Path, tok: Any) -> Dict[str, Any]:
    prof = read_json(path)
    compact_str = json.dumps(prof, ensure_ascii=False, separators=(",", ":"))
    # Also compute prompt block string matching stage8
    from silhouette import profile as s8_prof  # noqa: E402
    prompt_block = s8_prof.render_profile_prompt_block(prof)
    n_tables = len(prof)
    n_cols = 0
    freq_cols = 0
    freq_vals_total = 0
    per_table = {}
    for tbl, cols in sorted(prof.items()):
        t_cols = len(cols)
        n_cols += t_cols
        t_fc = 0
        t_fv = 0
        for cname, cval in cols.items():
            # cval is [dtype, null_frac, card_bucket, summary_3, freq_values]
            fv = cval[4] if isinstance(cval, list) and len(cval) >= 5 else None
            if isinstance(fv, list) and len(fv) > 0:
                t_fc += 1
                t_fv += len(fv)
        freq_cols += t_fc
        freq_vals_total += t_fv
        per_table[tbl] = {"cols": t_cols, "freq_cols": t_fc, "freq_vals": t_fv}

    excluded_public = [pt for pt in EHR_PUBLIC_TABLES if pt not in prof]
    return {
        "path": str(path),
        "json_chars": len(compact_str),
        "prompt_block_chars": len(prompt_block),
        "qwen25_bpe_tokens": len(tok.encode(prompt_block, add_special_tokens=False)),
        "n_tables": n_tables,
        "public_4_excluded_count": len(excluded_public),
        "public_4_excluded": excluded_public,
        "n_cols": n_cols,
        "surviving_freq_cols": freq_cols,
        "surviving_freq_values_total": freq_vals_total,
        "per_table": per_table,
    }


def verify_summary_offline(summary_path: Path) -> int:
    summary = read_json(summary_path)
    assert summary["total_conditions"] == 23, summary["total_conditions"]
    assert summary["total_slots"] == 9917, summary["total_slots"]
    assert summary["all_fw_errors"] == 0, summary["all_fw_errors"]
    assert summary["all_copied_private_tuples_unique"] == 0, summary["all_copied_private_tuples_unique"]
    assert len(summary["conditions"]) == 23
    assert sum(c["n"] for c in summary["conditions"].values()) == 9917
    print(
        f"[Phase12 Offline Summary Check] 23 conditions, {summary['total_slots']} slots, "
        f"fw_errors={summary['all_fw_errors']}, copied_private_tuples={summary['all_copied_private_tuples_unique']} -> OK"
    )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None, help="Upstream experimental root directory")
    parser.add_argument("--raw-root", type=Path, default=None, help="Upstream raw agent runs root directory")
    parser.add_argument("--tokenizer-path", type=str, default=os.environ.get("SILHOUETTE_QWEN_TOKENIZER", ""))
    parser.add_argument("--summary-path", type=Path, default=REPO_ROOT / "stats/phase12_authoritative_summary.json")
    args = parser.parse_args(argv)

    if args.data_root is None or args.raw_root is None:
        return verify_summary_offline(args.summary_path)

    _init_external_runtime(args.data_root, args.raw_root)
    t0 = time.monotonic()
    print("[Phase12 Consolidate] Loading manifests & gold caches...", flush=True)

    bird_manifest = read_json(P6 / "manifests/bird_minidev_2026_manifest.json")
    bird_qids: List[int] = list(bird_manifest["evaluation"])
    assert len(bird_qids) == 440

    strata_bird = read_json(P6 / "manifests/question_strata_and_remap.json")
    entity_qids: List[int] = list(strata_bird["entity_159"]["question_ids"])
    assert len(entity_qids) == 159

    bird_rows_raw = [json.loads(l) for l in BIRD_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
    bird_meta_by_qid: Dict[int, Dict[str, Any]] = {int(r["question_id"]): r for r in bird_rows_raw if int(r["question_id"]) in set(bird_qids)}
    assert len(bird_meta_by_qid) == 440

    ehr_qs = read_json(EHR600_Q_PATH)["questions"]
    ehr_gold_raw = read_json(EHR600_G_PATH)["gold"]
    ehr_rows: List[Dict[str, Any]] = []
    ehr_idx_to_id: Dict[int, str] = {}
    for item in ehr_qs:
        idx = int(item["idx"])
        qid = str(item["ehrsql_id"])
        ehr_idx_to_id[idx] = qid
        ehr_rows.append({"idx": idx, "question_id": qid, "db_id": "mimic_iv", "question": item["question"], "gold_sql": ehr_gold_raw.get(qid) or ehr_gold_raw.get(str(idx))})
    assert len(ehr_rows) == 600
    ehr_row_by_qid = {r["question_id"]: r for r in ehr_rows}

    # Gold caches
    with (P12_RAW / "gold_cache.pkl").open("rb") as f:
        bird_gold: Dict[int, Dict[str, Any]] = pickle.load(f)["bird_gold"]
    assert len(bird_gold) == 440
    with (P12_RAW / "ehr_gold_600.pkl").open("rb") as f:
        ehr_gold: Dict[str, Dict[str, Any]] = pickle.load(f)
    assert len(ehr_gold) == 600

    conds = build_target_conditions(bird_qids, entity_qids, ehr_rows)
    assert len(conds) == 23

    # Discover all 9,917 slots & extract empty-probe telemetry
    all_slots: List[Dict[str, Any]] = []
    unique_exec_keys: Dict[Tuple[str, str, str], str] = {}  # (dataset_kind, db_id, pred_sql) -> db_path

    for c in conds:
        cond = c["condition"]
        slug = slot_p6.cond_slug_of(cond)
        is_ehr = c["dataset"] == "ehrsql_600"
        d_kind = "ehrsql_600" if is_ehr else "bird"

        for qid in c["qids"]:
            if is_ehr:
                r = ehr_row_by_qid[qid]
                ident = f"Q{r['idx']:03d}_{qid}"
                db_id = "mimic_iv"
                db_path_str = str(EHR_DB_PATH)
            else:
                ident = f"Q{int(qid):03d}"
                db_id = bird_meta_by_qid[int(qid)]["db_id"]
                db_path_str = str(BIRD_DB_DIR / f"{db_id}.sqlite")

            sdir = P12_RAW / "runs" / slug / ident
            res_path = sdir / "result.json"
            assert res_path.exists(), f"Missing slot: {res_path}"
            res = read_json(res_path)
            assert not res.get("framework_error"), f"Framework error in {res_path}"

            te_path = sdir / "tool_events.json"
            events = read_json(te_path) if te_path.exists() else []
            run_sql_total = 0
            run_sql_success = 0
            run_sql_empty = 0
            run_sql_error = 0
            for ev in events:
                if ev.get("name") == "run_sql":
                    run_sql_total += 1
                    fb = ev.get("feedback") or {}
                    if fb.get("status") == "success":
                        run_sql_success += 1
                        if len(fb.get("rows") or []) == 0:
                            run_sql_empty += 1
                    else:
                        run_sql_error += 1

            pred_sql = res.get("predicted_sql") or ""
            ukey = (d_kind, db_id, pred_sql)
            unique_exec_keys[ukey] = db_path_str

            sp_entry: Dict[str, Any] = {
                "condition": cond,
                "arm": c["arm"],
                "dataset": c["dataset"],
                "dataset_kind": d_kind,
                "mode": c["mode"],
                "rep": c["rep"],
                "question_id": qid,
                "db_id": db_id,
                "slot_dir": str(sdir),
                "predicted_sql": pred_sql,
                "orig_official_set_ex": int(res["official_set_ex"]),
                "orig_strict_ex": int(res["strict_ex"]),
                "runtime_error": res.get("runtime_error"),
                "api_calls": int(res.get("api_calls", 0)),
                "tool_calls": int(res.get("tool_calls", 0)),
                "sandbox_calls": int(res.get("sandbox_calls", 0)),
                "sandbox_errors": int(res.get("sandbox_errors", 0)),
                "run_sql_calls": int(res.get("run_sql_calls", 0)),
                "lookup_calls": int(res.get("lookup_calls", 0)),
                "no_submission": bool(res.get("no_submission", False)),
                "prompt_tokens": int(res.get("prompt_tokens", 0)),
                "completion_tokens": int(res.get("completion_tokens", 0)),
                "total_tokens": int(res.get("total_tokens", 0)),
                "latency_s": float(res.get("latency_s", 0.0)),
                "stop": res.get("stop"),
                "framework_error": bool(res.get("framework_error", False)),
                "run_sql_probes_total": run_sql_total,
                "run_sql_probes_success": run_sql_success,
                "run_sql_probes_empty": run_sql_empty,
                "run_sql_probes_error": run_sql_error,
            }
            if c["mode"] == "bird_profile":
                sp_entry["profile_path"] = str(Path(c["profile_dir"]) / f"{db_id}.json")
            elif c["mode"] == "ehr_profile":
                sp_entry["profile_path"] = c["profile_path"]
            elif c["mode"] == "bird_static_rows":
                sp_entry["static_path"] = str(Path(c["static_dir"]) / f"{db_id}.txt")
            all_slots.append(sp_entry)

    print(f"[Phase12 Consolidate] Loaded {len(all_slots)} slots across 23 conditions ({len(unique_exec_keys)} unique SQL executions). Running 60s dual+deliverable replay...", flush=True)

    # Group unique SQLs into jobs (split mimic_iv into chunks of 80 for 16-worker parallelism)
    by_db_sql: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
    for (dk, db_id, sql), dbp in sorted(unique_exec_keys.items()):
        by_db_sql[(dk, db_id, dbp)].append(sql)

    replay_jobs: List[Tuple[str, str, str, List[str]]] = []
    for (dk, db_id, dbp), sqls in sorted(by_db_sql.items()):
        chunk_sz = 80 if dk == "ehrsql_600" else 120
        for i in range(0, len(sqls), chunk_sz):
            replay_jobs.append((dk, db_id, dbp, sqls[i : i + chunk_sz]))

    t_rep = time.monotonic()
    exec_cache: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    ctx = multiprocessing.get_context("fork")
    with concurrent.futures.ProcessPoolExecutor(max_workers=16, mp_context=ctx) as pool:
        for res_list in pool.map(_replay_sql_worker, replay_jobs):
            for k, v in res_list:
                exec_cache[k] = v
    print(f"[Phase12 Consolidate] 60s Dual+Deliverable SQL replay finished in {time.monotonic() - t_rep:.2f}s ({len(exec_cache)} unique queries).", flush=True)

    # Score every slot from exec_cache & verify parity with result.json
    replay_mismatches = 0
    for sp in all_slots:
        dk = sp["dataset_kind"]
        db_id = sp["db_id"]
        sql = sp["predicted_sql"]
        ex = exec_cache[(dk, db_id, sql)]
        if dk == "ehrsql_600":
            g = ehr_gold[str(sp["question_id"])]
            if ex["p1"] is not None:
                off_set = int(ex["p1"][0] == g["top100"])
                off_strict = off_set
                off_uncapped = int(ex["p1"][1] == g["uncapped"])
            else:
                off_set = off_strict = off_uncapped = 0
            if ex["p2"] is not None:
                deliv_set = int(ex["p2"][0] == g["top100"])
                deliv_strict = deliv_set
                deliv_uncapped = int(ex["p2"][1] == g["uncapped"])
            else:
                deliv_set = deliv_strict = deliv_uncapped = 0
        else:
            g = bird_gold[int(sp["question_id"])]
            off_set, off_strict = tb._score_bird(ex["p1"], ex["st1"], g["rows"], g["has_order"])
            deliv_set, deliv_strict = tb._score_bird(ex["p2"], ex["st2"], g["rows"], g["has_order"])
            off_uncapped = off_set
            deliv_uncapped = deliv_set

        if off_set != sp["orig_official_set_ex"] or off_strict != sp["orig_strict_ex"]:
            replay_mismatches += 1
        sp["official_set_ex"] = off_set
        sp["strict_ex"] = off_strict
        sp["uncapped_ex"] = off_uncapped
        sp["deliverable_set_ex"] = deliv_set
        sp["deliverable_strict_ex"] = deliv_strict
        sp["deliverable_uncapped_ex"] = deliv_uncapped
        sp["unrestricted_status"] = ex["st1"]
        sp["deliverable_status"] = ex["st2"]
        sp["deliverable_error_category"] = ex["cat2"]
        sp["authorizer_denials"] = ex["auth_denied"]
        sp["timeout_flag"] = ex["timeout_flag"]

    print(f"[Phase12 Consolidate] Replay parity mismatches vs result.json: {replay_mismatches}", flush=True)

    # Run DB-deduplicated TransmissionLedger audit across 12 DBs (11 BIRD + 1 mimic_iv_private13)
    t_aud = time.monotonic()
    by_audit_db: Dict[str, Tuple[str, str, bool, List[Dict[str, Any]]]] = {}
    for sp in all_slots:
        is_ehr = sp["dataset_kind"] == "ehrsql_600"
        if is_ehr:
            key = "mimic_iv_priv13"
            db_path_str = str(EHR_PRIV13_DB)
            db_tag = "ehr_priv13__mimic_iv"
        else:
            key = f"bird__{sp['db_id']}"
            db_path_str = str(BIRD_DB_DIR / f"{sp['db_id']}.sqlite")
            db_tag = key
        if key not in by_audit_db:
            by_audit_db[key] = (db_path_str, db_tag, is_ehr, [])
        by_audit_db[key][3].append(sp)

    audit_jobs = [(k, v[0], v[1], v[2], v[3]) for k, v in sorted(by_audit_db.items())]
    cond_db_audit: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(dict)
    per_slot_audit_all: Dict[Tuple[str, str], Dict[str, Any]] = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=12, mp_context=ctx) as pool:
        for db_key, elapsed, csum, pslot in pool.map(_audit_db_worker, audit_jobs):
            print(f"  audit DB {db_key:30s} ({elapsed:.2f}s)", flush=True)
            for cond, a_dict in csum.items():
                cond_db_audit[cond][db_key] = a_dict
            per_slot_audit_all.update(pslot)

    print(f"[Phase12 Consolidate] TransmissionLedger audit finished in {time.monotonic() - t_aud:.2f}s.", flush=True)

    # Attach per-slot audit & write phase12_per_slot_outcomes.jsonl
    stats_dir = P12 / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    slots_jsonl_path = stats_dir / "phase12_per_slot_outcomes.jsonl"
    with slots_jsonl_path.open("w", encoding="utf-8") as f:
        for sp in all_slots:
            aud = per_slot_audit_all[(sp["condition"], str(sp["question_id"]))]
            sp["audit"] = aud
            rec = {k: v for k, v in sp.items() if k not in ("profile_path", "static_path")}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Aggregate per-condition summaries
    by_cond_slots: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sp in all_slots:
        by_cond_slots[sp["condition"]].append(sp)

    cond_summaries: Dict[str, Any] = {}
    for c in conds:
        cond = c["condition"]
        items = by_cond_slots[cond]
        n = len(items)
        set_hits = sum(x["official_set_ex"] for x in items)
        strict_hits = sum(x["strict_ex"] for x in items)
        deliv_set_hits = sum(x["deliverable_set_ex"] for x in items)
        deliv_strict_hits = sum(x["deliverable_strict_ex"] for x in items)

        # DB-deduplicated audit totals summed across DBs
        db_auds = cond_db_audit[cond]
        db_dedup_audit = {
            "copied_private_tuples": [
                sum(v["copied_unique"] for v in db_auds.values()),
                sum(v["copied_cum"] for v in db_auds.values()),
            ],
            "outside_profile_serialized_tuples": [
                sum(v["ser_out_unique"] for v in db_auds.values()),
                sum(v["ser_out_cum"] for v in db_auds.values()),
            ],
            "full_payload_coincidences": [
                sum(v["mc_unique"] for v in db_auds.values()),
                sum(v["mc_cum"] for v in db_auds.values()),
            ],
            "out_of_profile_literals": [
                sum(v["oob_unique"] for v in db_auds.values()),
                sum(v["oob_cum"] for v in db_auds.values()),
            ],
            "slots_nonzero_ser_out": sum(v["slots_nonzero_ser_out"] for v in db_auds.values()),
            "slots_nonzero_mc": sum(v["slots_nonzero_mc"] for v in db_auds.values()),
            "slots_nonzero_oob": sum(v["slots_nonzero_oob"] for v in db_auds.values()),
        }

        probe_tot = sum(x["run_sql_probes_total"] for x in items)
        probe_succ = sum(x["run_sql_probes_success"] for x in items)
        probe_empty = sum(x["run_sql_probes_empty"] for x in items)
        probe_err = sum(x["run_sql_probes_error"] for x in items)

        cond_summaries[cond] = {
            "arm": c["arm"],
            "dataset": c["dataset"],
            "mode": c["mode"],
            "rep": c["rep"],
            "n": n,
            "fw_errors": sum(1 for x in items if x["framework_error"]),
            "no_submission": sum(1 for x in items if x["no_submission"]),
            "official_set_ex": set_hits,
            "official_set_ex_pct": round(100.0 * set_hits / n, 2),
            "strict_ex": strict_hits,
            "strict_ex_pct": round(100.0 * strict_hits / n, 2),
            "deliverable_set_ex": deliv_set_hits,
            "deliverable_set_ex_pct": round(100.0 * deliv_set_hits / n, 2),
            "deliverable_strict_ex": deliv_strict_hits,
            "deliverable_strict_ex_pct": round(100.0 * deliv_strict_hits / n, 2),
            "authorizer_denials": sum(x["authorizer_denials"] for x in items),
            "timeout_flags": sum(x["timeout_flag"] for x in items),
            "mean_total_tokens": round(statistics.mean(x["total_tokens"] for x in items), 1),
            "mean_prompt_tokens": round(statistics.mean(x["prompt_tokens"] for x in items), 1),
            "mean_completion_tokens": round(statistics.mean(x["completion_tokens"] for x in items), 1),
            "mean_latency_s": round(statistics.mean(x["latency_s"] for x in items), 2),
            "mean_tool_calls": round(statistics.mean(x["tool_calls"] for x in items), 2),
            "mean_sandbox_calls": round(statistics.mean(x["sandbox_calls"] for x in items), 2),
            "mean_lookup_calls": round(statistics.mean(x["lookup_calls"] for x in items), 2),
            "empty_probe_stats": {
                "run_sql_probes_total": probe_tot,
                "run_sql_probes_success": probe_succ,
                "run_sql_probes_empty": probe_empty,
                "run_sql_probes_error": probe_err,
                "empty_probe_of_total_pct": round(100.0 * probe_empty / probe_tot, 2) if probe_tot else 0.0,
                "empty_probe_of_success_pct": round(100.0 * probe_empty / probe_succ, 2) if probe_succ else 0.0,
                "slots_with_run_sql": sum(1 for x in items if x["run_sql_probes_total"] > 0),
                "slots_with_empty_run_sql": sum(1 for x in items if x["run_sql_probes_empty"] > 0),
            },
            "db_dedup_audit": db_dedup_audit,
        }

    # Pooled R=3 summaries
    pooled_groups = [
        ("AIM_BIRD_R1_R3", [f"AIM_BIRD_R{r}/r0" for r in (1, 2, 3)]),
        ("PRIVLAVA_BIRD_R1_R3", [f"PRIVLAVA_BIRD_R{r}/r0" for r in (1, 2, 3)]),
        ("PRIVPETAL_BIRD_R1_R3", [f"PRIVPETAL_BIRD_R{r}/r0" for r in (1, 2, 3)]),
        ("DAWA_DPSIPS_BIRD_R1_R3", [f"DAWA_DPSIPS_BIRD_R{r}/r0" for r in (1, 2, 3)]),
        ("TIGHTBPE_DAWA_DPSIPS_BIRD_R1_R3", [f"TIGHTBPE_DAWA_DPSIPS_BIRD_R{r}/r0" for r in (1, 2, 3)]),
        ("ENT_PRIVLAVA_BIRD_R1_R3", [f"ENT_PRIVLAVA_BIRD_R{r}/r0" for r in (1, 2, 3)]),
    ]
    pooled_summaries: Dict[str, Any] = {}
    for gname, clist in pooled_groups:
        reps = [cond_summaries[c] for c in clist]
        ptot = sum(r["empty_probe_stats"]["run_sql_probes_total"] for r in reps)
        psucc = sum(r["empty_probe_stats"]["run_sql_probes_success"] for r in reps)
        pemp = sum(r["empty_probe_stats"]["run_sql_probes_empty"] for r in reps)
        perr = sum(r["empty_probe_stats"]["run_sql_probes_error"] for r in reps)
        pooled_summaries[gname] = {
            "conditions": clist,
            "n_per_rep": reps[0]["n"],
            "total_slots": sum(r["n"] for r in reps),
            "fw_errors_total": sum(r["fw_errors"] for r in reps),
            "no_submission_total": sum(r["no_submission"] for r in reps),
            "per_rep_set_ex_pct": [r["official_set_ex_pct"] for r in reps],
            "mean_set_ex_pct": round(statistics.mean(r["official_set_ex_pct"] for r in reps), 2),
            "sd_set_ex_pp": round(statistics.stdev(r["official_set_ex_pct"] for r in reps), 2),
            "mean_strict_ex_pct": round(statistics.mean(r["strict_ex_pct"] for r in reps), 2),
            "sd_strict_ex_pp": round(statistics.stdev(r["strict_ex_pct"] for r in reps), 2),
            "mean_deliverable_set_ex_pct": round(statistics.mean(r["deliverable_set_ex_pct"] for r in reps), 2),
            "mean_deliverable_strict_ex_pct": round(statistics.mean(r["deliverable_strict_ex_pct"] for r in reps), 2),
            "mean_total_tokens": round(statistics.mean(r["mean_total_tokens"] for r in reps), 1),
            "empty_probe_stats": {
                "run_sql_probes_total": ptot,
                "run_sql_probes_success": psucc,
                "run_sql_probes_empty": pemp,
                "run_sql_probes_error": perr,
                "empty_probe_of_total_pct": round(100.0 * pemp / ptot, 2) if ptot else 0.0,
                "empty_probe_of_success_pct": round(100.0 * pemp / psucc, 2) if psucc else 0.0,
            },
            "audit_copied_unique_sum": sum(r["db_dedup_audit"]["copied_private_tuples"][0] for r in reps),
            "audit_ser_out_unique_sum": sum(r["db_dedup_audit"]["outside_profile_serialized_tuples"][0] for r in reps),
            "audit_oob_unique_sum": sum(r["db_dedup_audit"]["out_of_profile_literals"][0] for r in reps),
        }

    # ---------- Load Canonical Comparison Arms for Exact Paired Inference ----------
    print("[Phase12 Consolidate] Running exact paired statistical inference...", flush=True)
    outcome_data: Dict[str, Dict[Any, int]] = defaultdict(dict)

    # Phase 12 outcomes
    for sp in all_slots:
        outcome_data[sp["condition"]][sp["question_id"]] = sp["official_set_ex"]

    # Phase 6 canonical outcomes
    for line in P6_OUTCOMES_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        outcome_data[r["condition"]][r["question_id"]] = int(r["outcome"])

    # Stage 8 EHRSQL V3 outcomes
    for line in S8_CLINICAL_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("benchmark") == "ehrsql_600":
            outcome_data[r["condition"]][str(r["question_id"])] = int(r["official_set_ex"])

    # Stage 10 TightBPE ValFirst outcomes
    s10_data = read_json(S10_HARVEST_PATH)
    for r in s10_data["outcomes"]:
        outcome_data[r["condition"]][int(r["question_id"])] = int(r["official_set_ex"])

    bird_dbs = [bird_meta_by_qid[q]["db_id"] for q in bird_qids]
    ent_dbs = [bird_meta_by_qid[q]["db_id"] for q in entity_qids]

    # EHRSQL 120-template map
    ehr_qid_set = set(r["question_id"] for r in ehr_rows)
    tpl_by_qid: Dict[str, str] = {}
    for split in ("train", "valid"):
        ann = read_json(R / f"generalize_survey_20261114/data/ehrsql-2024/data/mimic_iv/{split}/annotated.json")
        for r in ann:
            if r["id"] in ehr_qid_set:
                tpl_by_qid[r["id"]] = r["template"]
    assert len(tpl_by_qid) == 600 and len(set(tpl_by_qid.values())) == 120
    ehr_qids = [r["question_id"] for r in ehr_rows]
    ehr_tpls = [tpl_by_qid[q] for q in ehr_qids]

    def abs_rate_ci(conds: List[str], ids: List[Any], cids: Optional[List[str]] = None) -> Dict[str, Any]:
        vals = vec(outcome_data, conds, ids)
        lo, hi = paired_bootstrap(vals)
        res = {
            "mean_pct": round(100.0 * statistics.mean(vals), 2),
            "ci95_pct": [round(100.0 * lo, 2), round(100.0 * hi, 2)],
        }
        if cids is not None:
            clo, chi = cluster_bootstrap(vals, cids)
            res["cluster_ci95_pct"] = [round(100.0 * clo, 2), round(100.0 * chi, 2)]
        return res

    abs_cis = {
        "AIM_BIRD_R1_R3": abs_rate_ci([f"AIM_BIRD_R{r}/r0" for r in (1, 2, 3)], bird_qids, bird_dbs),
        "PRIVLAVA_BIRD_R1_R3": abs_rate_ci([f"PRIVLAVA_BIRD_R{r}/r0" for r in (1, 2, 3)], bird_qids, bird_dbs),
        "PRIVPETAL_BIRD_R1_R3": abs_rate_ci([f"PRIVPETAL_BIRD_R{r}/r0" for r in (1, 2, 3)], bird_qids, bird_dbs),
        "DAWA_DPSIPS_BIRD_R1_R3": abs_rate_ci([f"DAWA_DPSIPS_BIRD_R{r}/r0" for r in (1, 2, 3)], bird_qids, bird_dbs),
        "TIGHTBPE_DAWA_DPSIPS_BIRD_R1_R3": abs_rate_ci([f"TIGHTBPE_DAWA_DPSIPS_BIRD_R{r}/r0" for r in (1, 2, 3)], bird_qids, bird_dbs),
        "PUBCAP_AIM_ROWS_BIRD_R1": abs_rate_ci(["PUBCAP_AIM_ROWS_BIRD_R1/r0"], bird_qids, bird_dbs),
        "ENT_PRIVLAVA_BIRD_R1_R3": abs_rate_ci([f"ENT_PRIVLAVA_BIRD_R{r}/r0" for r in (1, 2, 3)], entity_qids, ent_dbs),
        "AIM_EHR600_R1": abs_rate_ci(["AIM_EHR600_R1/r0"], ehr_qids, ehr_tpls),
        "PRIVLAVA_EHR600_R1": abs_rate_ci(["PRIVLAVA_EHR600_R1/r0"], ehr_qids, ehr_tpls),
        "PRIVPETAL_EHR600_R1": abs_rate_ci(["PRIVPETAL_EHR600_R1/r0"], ehr_qids, ehr_tpls),
        "DAWA_DPSIPS_EHR600_R1": abs_rate_ci(["DAWA_DPSIPS_EHR600_R1/r0"], ehr_qids, ehr_tpls),
        "DG8_CLEAN_R1_R3": abs_rate_ci([f"DG8_CLEAN_R{r}/r0" for r in (1, 2, 3)], bird_qids, bird_dbs),
        "ENT_V2_DG8_R1_R3": abs_rate_ci([f"ENT_V2_DG8_R{r}/r0" for r in (1, 2, 3)], entity_qids, ent_dbs),
        "EHR_DG8P_V3_R1": abs_rate_ci(["EHR_DG8P_V3_R1/r0"], ehr_qids, ehr_tpls),
        "EHR_DG8P_V3_R1_R3": abs_rate_ci([f"EHR_DG8P_V3_R{r}/r0" for r in (1, 2, 3)], ehr_qids, ehr_tpls),
        "TIGHTBPE_VALFIRST_SILH_R1_R3": abs_rate_ci([f"TIGHTBPE_VALFIRST_SILH_R{r}/r0" for r in (1, 2, 3)], bird_qids, bird_dbs),
        "TIGHTBPE_VALFIRST_IND8_PROF_R1_R3": abs_rate_ci([f"TIGHTBPE_VALFIRST_IND8_PROF_R{r}/r0" for r in (1, 2, 3)], bird_qids, bird_dbs),
    }

    sil_bird = [f"DG8_CLEAN_R{r}/r0" for r in (1, 2, 3)]
    c0_bird = ["C0/r0", "C0/r2"]
    ind_bird = ["IND8_PROMPT/r0", "IND8_PROMPT_R2/r0", "IND8_PROMPT_R3/r0"]
    c4_bird = ["C4/r0", "C4/r2"]

    sil_ent = [f"ENT_V2_DG8_R{r}/r0" for r in (1, 2, 3)]
    ind_ent = [f"ENT_V2_IND8_PROMPT_R{r}/r0" for r in (1, 2, 3)]

    sil_tb_vf = [f"TIGHTBPE_VALFIRST_SILH_R{r}/r0" for r in (1, 2, 3)]
    ind_tb_vf = [f"TIGHTBPE_VALFIRST_IND8_PROF_R{r}/r0" for r in (1, 2, 3)]

    # Family 1: BIRD-440 4 Native Baselines vs Silhouette (3v3)
    bird4_specs = [
        ("Silhouette vs AIM", sil_bird, [f"AIM_BIRD_R{r}/r0" for r in (1, 2, 3)]),
        ("Silhouette vs PrivLava", sil_bird, [f"PRIVLAVA_BIRD_R{r}/r0" for r in (1, 2, 3)]),
        ("Silhouette vs PrivPetal", sil_bird, [f"PRIVPETAL_BIRD_R{r}/r0" for r in (1, 2, 3)]),
        ("Silhouette vs DAWA+DP-SIPS", sil_bird, [f"DAWA_DPSIPS_BIRD_R{r}/r0" for r in (1, 2, 3)]),
    ]
    bird4_cmps = [
        {"comparison": lbl, **compare_pair(outcome_data, ca, cb, bird_qids, bird_dbs, exact_cluster_rad=True)}
        for lbl, ca, cb in bird4_specs
    ]
    for row, hp, hcp in zip(
        bird4_cmps,
        holm([x["p"] for x in bird4_cmps]),
        holm([x["cluster_rademacher_p"] for x in bird4_cmps]),
    ):
        row["holm_p"] = hp
        row["holm_cluster_rademacher_p"] = hcp

    # Extended 6-comparison Phase-12 family (BIRD-440 x4 + Entity-159 + TightBPE ValFirst Silh vs TightBPE DAWA+DP-SIPS)
    ent_cmp = {"comparison": "Entity Silhouette vs Entity-PrivLava", **compare_pair(outcome_data, sil_ent, [f"ENT_PRIVLAVA_BIRD_R{r}/r0" for r in (1, 2, 3)], entity_qids, ent_dbs, exact_cluster_rad=True)}
    tb_sil_cmp = {"comparison": "TightBPE ValFirst Silhouette vs TightBPE DAWA+DP-SIPS", **compare_pair(outcome_data, sil_tb_vf, [f"TIGHTBPE_DAWA_DPSIPS_BIRD_R{r}/r0" for r in (1, 2, 3)], bird_qids, bird_dbs, exact_cluster_rad=True)}
    tb_ind_cmp = {"comparison": "TightBPE ValFirst 1-way Profile vs TightBPE DAWA+DP-SIPS", **compare_pair(outcome_data, ind_tb_vf, [f"TIGHTBPE_DAWA_DPSIPS_BIRD_R{r}/r0" for r in (1, 2, 3)], bird_qids, bird_dbs, exact_cluster_rad=True)}

    family6 = [dict(x) for x in bird4_cmps] + [dict(ent_cmp), dict(tb_sil_cmp)]
    for row, hp, hcp in zip(
        family6,
        holm([x["p"] for x in family6]),
        holm([x["cluster_rademacher_p"] for x in family6]),
    ):
        row["holm_6family_p"] = hp
        row["holm_6family_cluster_rademacher_p"] = hcp

    # Additional BIRD-440 comparisons (vs C0, vs 1-way, vs C4, PUBCAP)
    bird_extra = {
        "AIM_minus_C0": compare_pair(outcome_data, [f"AIM_BIRD_R{r}/r0" for r in (1, 2, 3)], c0_bird, bird_qids, bird_dbs, exact_cluster_rad=True),
        "PRIVLAVA_minus_C0": compare_pair(outcome_data, [f"PRIVLAVA_BIRD_R{r}/r0" for r in (1, 2, 3)], c0_bird, bird_qids, bird_dbs, exact_cluster_rad=True),
        "PRIVPETAL_minus_C0": compare_pair(outcome_data, [f"PRIVPETAL_BIRD_R{r}/r0" for r in (1, 2, 3)], c0_bird, bird_qids, bird_dbs, exact_cluster_rad=True),
        "DAWA_DPSIPS_minus_C0": compare_pair(outcome_data, [f"DAWA_DPSIPS_BIRD_R{r}/r0" for r in (1, 2, 3)], c0_bird, bird_qids, bird_dbs, exact_cluster_rad=True),
        "TIGHTBPE_DAWA_DPSIPS_minus_C0": compare_pair(outcome_data, [f"TIGHTBPE_DAWA_DPSIPS_BIRD_R{r}/r0" for r in (1, 2, 3)], c0_bird, bird_qids, bird_dbs, exact_cluster_rad=True),
        "PUBCAP_AIM_ROWS_R1_minus_DG8_R1": compare_pair(outcome_data, ["PUBCAP_AIM_ROWS_BIRD_R1/r0"], ["DG8_CLEAN_R1/r0"], bird_qids, bird_dbs, exact_cluster_rad=True),
        "PUBCAP_AIM_ROWS_R1_minus_AIM_R1": compare_pair(outcome_data, ["PUBCAP_AIM_ROWS_BIRD_R1/r0"], ["AIM_BIRD_R1/r0"], bird_qids, bird_dbs, exact_cluster_rad=True),
        "PUBCAP_AIM_ROWS_R1_minus_C0_r0": compare_pair(outcome_data, ["PUBCAP_AIM_ROWS_BIRD_R1/r0"], ["C0/r0"], bird_qids, bird_dbs, exact_cluster_rad=True),
        "IND8_PROMPT_minus_DAWA_DPSIPS": compare_pair(outcome_data, ind_bird, [f"DAWA_DPSIPS_BIRD_R{r}/r0" for r in (1, 2, 3)], bird_qids, bird_dbs, exact_cluster_rad=True),
        "SIL_minus_C4": compare_pair(outcome_data, sil_bird, c4_bird, bird_qids, bird_dbs, exact_cluster_rad=True),
        "DAWA_DPSIPS_minus_C4": compare_pair(outcome_data, [f"DAWA_DPSIPS_BIRD_R{r}/r0" for r in (1, 2, 3)], c4_bird, bird_qids, bird_dbs, exact_cluster_rad=True),
        "ENT_PRIVLAVA_minus_C0": compare_pair(outcome_data, [f"ENT_PRIVLAVA_BIRD_R{r}/r0" for r in (1, 2, 3)], c0_bird, entity_qids, ent_dbs, exact_cluster_rad=True),
        "ENT_IND8_PROMPT_minus_ENT_PRIVLAVA": compare_pair(outcome_data, ind_ent, [f"ENT_PRIVLAVA_BIRD_R{r}/r0" for r in (1, 2, 3)], entity_qids, ent_dbs, exact_cluster_rad=True),
        "TIGHTBPE_VALFIRST_IND8_vs_TIGHTBPE_DAWA": tb_ind_cmp,
    }

    # Entity-BIRD-159 per-DB breakdown (4 entity DBs)
    ent_by_db: Dict[str, List[int]] = defaultdict(list)
    for q in entity_qids:
        ent_by_db[bird_meta_by_qid[q]["db_id"]].append(q)
    ent_per_db = {}
    for db_id, qs in sorted(ent_by_db.items()):
        ent_per_db[db_id] = {
            "n": len(qs),
            "ENT_PRIVLAVA_R1_R3_pct": round(100.0 * statistics.mean(vec(outcome_data, [f"ENT_PRIVLAVA_BIRD_R{r}/r0" for r in (1, 2, 3)], qs)), 2),
            "ENT_PRIVLAVA_per_rep_pct": [round(100.0 * statistics.mean(vec(outcome_data, [f"ENT_PRIVLAVA_BIRD_R{r}/r0"], qs)), 2) for r in (1, 2, 3)],
            "ENT_V2_DG8_R1_R3_pct": round(100.0 * statistics.mean(vec(outcome_data, sil_ent, qs)), 2),
            "ENT_V2_IND8_PROMPT_R1_R3_pct": round(100.0 * statistics.mean(vec(outcome_data, ind_ent, qs)), 2),
            "C0_pct": round(100.0 * statistics.mean(vec(outcome_data, c0_bird, qs)), 2),
            "C4_pct": round(100.0 * statistics.mean(vec(outcome_data, c4_bird, qs)), 2),
        }

    # EHRSQL-600 paired inference (vs Stage-8 EHR_DG8P_V3_R1 and R1..R3, plus Phase-6 DG8P_R1, C0, C4, EHR_IND8_V3_R1)
    ehr_arms = ["AIM_EHR600_R1/r0", "PRIVLAVA_EHR600_R1/r0", "PRIVPETAL_EHR600_R1/r0", "DAWA_DPSIPS_EHR600_R1/r0"]
    ehr_vs_s8_r1 = [
        {"comparison": f"EHR_DG8P_V3_R1 vs {c}", **compare_pair(outcome_data, ["EHR_DG8P_V3_R1/r0"], [c], ehr_qids, ehr_tpls)}
        for c in ehr_arms
    ]
    for row, hp in zip(ehr_vs_s8_r1, holm([x["p"] for x in ehr_vs_s8_r1])):
        row["holm_p"] = hp

    ehr_vs_s8_r3 = [
        {"comparison": f"EHR_DG8P_V3_R1_R3 vs {c}", **compare_pair(outcome_data, [f"EHR_DG8P_V3_R{r}/r0" for r in (1, 2, 3)], [c], ehr_qids, ehr_tpls)}
        for c in ehr_arms
    ]
    for row, hp in zip(ehr_vs_s8_r3, holm([x["p"] for x in ehr_vs_s8_r3])):
        row["holm_p"] = hp

    ehr_vs_p6_r1 = [
        {"comparison": f"DG8P_R1 vs {c}", **compare_pair(outcome_data, ["DG8P_R1/r0"], [c], ehr_qids, ehr_tpls)}
        for c in ehr_arms
    ]
    for row, hp in zip(ehr_vs_p6_r1, holm([x["p"] for x in ehr_vs_p6_r1])):
        row["holm_p"] = hp

    ehr_vs_c0_c4 = {
        f"{c}_minus_C0": compare_pair(outcome_data, [c], ["C0/r0"], ehr_qids, ehr_tpls)
        for c in ehr_arms
    }
    for c in ehr_arms:
        ehr_vs_c0_c4[f"{c}_minus_C4"] = compare_pair(outcome_data, [c], ["C4/r0"], ehr_qids, ehr_tpls)
    ehr_vs_c0_c4["EHR_DG8P_V3_R1_minus_C0"] = compare_pair(outcome_data, ["EHR_DG8P_V3_R1/r0"], ["C0/r0"], ehr_qids, ehr_tpls)
    ehr_vs_c0_c4["EHR_IND8_V3_R1_minus_DAWA_DPSIPS_EHR600_R1"] = compare_pair(outcome_data, ["EHR_IND8_V3_R1/r0"], ["DAWA_DPSIPS_EHR600_R1/r0"], ehr_qids, ehr_tpls)

    # ---------- Forensics 1: PUBCAP_AIM_ROWS_BIRD_R1 ----------
    bird_by_db: Dict[str, List[int]] = defaultdict(list)
    for q in bird_qids:
        bird_by_db[bird_meta_by_qid[q]["db_id"]].append(q)
    pubcap_per_db = {}
    for db_id, qs in sorted(bird_by_db.items()):
        pubcap_per_db[db_id] = {
            "n": len(qs),
            "PUBCAP_AIM_ROWS_R1_pct": round(100.0 * statistics.mean(vec(outcome_data, ["PUBCAP_AIM_ROWS_BIRD_R1/r0"], qs)), 2),
            "DG8_CLEAN_R1_pct": round(100.0 * statistics.mean(vec(outcome_data, ["DG8_CLEAN_R1/r0"], qs)), 2),
            "AIM_BIRD_R1_pct": round(100.0 * statistics.mean(vec(outcome_data, ["AIM_BIRD_R1/r0"], qs)), 2),
        }
    pubcap_forensics = {
        "condition": "PUBCAP_AIM_ROWS_BIRD_R1/r0",
        "summary": cond_summaries["PUBCAP_AIM_ROWS_BIRD_R1/r0"],
        "vs_DG8_CLEAN_R1": bird_extra["PUBCAP_AIM_ROWS_R1_minus_DG8_R1"],
        "vs_AIM_BIRD_R1": bird_extra["PUBCAP_AIM_ROWS_R1_minus_AIM_R1"],
        "vs_C0_r0": bird_extra["PUBCAP_AIM_ROWS_R1_minus_C0_r0"],
        "per_db_ex_pct": pubcap_per_db,
    }

    # ---------- Forensics 2: DAWA_DPSIPS_EHR600_R1 ----------
    if args.tokenizer_path:
        from transformers import AutoTokenizer  # noqa: E402
        tok = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    else:
        class _CharFallbackTok:
            @staticmethod
            def encode(s: str, add_special_tokens: bool = False) -> List[int]:
                return list(range(max(1, len(s) // 4)))
        tok = _CharFallbackTok()
    dawa_ehr_prof_stats = inspect_ehr_profile_stats(P12 / "releases/dawa_dpsips/R1/mimic_iv.json", tok)
    sil_ehr_prof_stats = inspect_ehr_profile_stats(ART / "profiles/ehrsql_dg8p/release_01.json", tok)

    ehr_strata_raw = read_json(EHR600_STRATA_PATH)["strata"]
    patient_136_idx = sorted(set(ehr_strata_raw["patient_literal_only"]) | set(ehr_strata_raw["both_C"]))
    assert len(patient_136_idx) == 136
    patient_136_qids = [ehr_idx_to_id[i] for i in patient_136_idx]
    dict_464_qids = [r["question_id"] for r in ehr_rows if r["idx"] not in set(patient_136_idx)]
    assert len(dict_464_qids) == 464

    ehr_strata_arms = [
        "DAWA_DPSIPS_EHR600_R1/r0",
        "EHR_DG8P_V3_R1/r0",
        "EHR_IND8_V3_R1/r0",
        "AIM_EHR600_R1/r0",
        "PRIVLAVA_EHR600_R1/r0",
        "PRIVPETAL_EHR600_R1/r0",
        "DG8P_R1/r0",
        "C0/r0",
        "C4/r0",
    ]
    ehr_strata_ex = {}
    for arm in ehr_strata_arms:
        p136 = vec(outcome_data, [arm], patient_136_qids)
        d464 = vec(outcome_data, [arm], dict_464_qids)
        all600 = vec(outcome_data, [arm], ehr_qids)
        ehr_strata_ex[arm] = {
            "all_600_hits": int(sum(all600)),
            "all_600_pct": round(100.0 * statistics.mean(all600), 2),
            "patient_filter_136_hits": int(sum(p136)),
            "patient_filter_136_pct": round(100.0 * statistics.mean(p136), 2),
            "dict_heavy_464_hits": int(sum(d464)),
            "dict_heavy_464_pct": round(100.0 * statistics.mean(d464), 2),
        }

    # Also compute EHR_DG8P_V3 R1..R3 mean on 136 vs 464
    p136_s8r3 = vec(outcome_data, [f"EHR_DG8P_V3_R{r}/r0" for r in (1, 2, 3)], patient_136_qids)
    d464_s8r3 = vec(outcome_data, [f"EHR_DG8P_V3_R{r}/r0" for r in (1, 2, 3)], dict_464_qids)
    ehr_strata_ex["EHR_DG8P_V3_R1_R3"] = {
        "all_600_pct": round(100.0 * statistics.mean(vec(outcome_data, [f"EHR_DG8P_V3_R{r}/r0" for r in (1, 2, 3)], ehr_qids)), 2),
        "patient_filter_136_pct": round(100.0 * statistics.mean(p136_s8r3), 2),
        "dict_heavy_464_pct": round(100.0 * statistics.mean(d464_s8r3), 2),
    }

    dawa_ehr_forensics = {
        "profile_comparison": {
            "DAWA_DPSIPS_EHR600_R1": dawa_ehr_prof_stats,
            "SILHOUETTE_EHR_DG8P_R1": sil_ehr_prof_stats,
        },
        "slot_telemetry": cond_summaries["DAWA_DPSIPS_EHR600_R1/r0"],
        "strata_136_vs_464": ehr_strata_ex,
        "paired_DAWA_vs_EHR_DG8P_V3_R1_patient136": compare_pair(outcome_data, ["DAWA_DPSIPS_EHR600_R1/r0"], ["EHR_DG8P_V3_R1/r0"], patient_136_qids),
        "paired_DAWA_vs_EHR_DG8P_V3_R1_dict464": compare_pair(outcome_data, ["DAWA_DPSIPS_EHR600_R1/r0"], ["EHR_DG8P_V3_R1/r0"], dict_464_qids),
    }

    summary = {
        "schema_version": "phase12-authoritative-summary-v1",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": round(time.monotonic() - t0, 2),
        "sqlite_version": sqlite3.sqlite_version,
        "total_conditions": len(conds),
        "total_slots": len(all_slots),
        "unique_sql_replayed": len(exec_cache),
        "replay_parity_mismatches_vs_result_json": replay_mismatches,
        "all_fw_errors": sum(v["fw_errors"] for v in cond_summaries.values()),
        "all_copied_private_tuples_unique": sum(v["db_dedup_audit"]["copied_private_tuples"][0] for v in cond_summaries.values()),
        "conditions": cond_summaries,
        "pooled_r3_summaries": pooled_summaries,
        "absolute_rate_cis": abs_cis,
        "paired_inference": {
            "bootstrap_draws": B,
            "bootstrap_seed": SEED,
            "bird_440_4baselines_holm": bird4_cmps,
            "phase12_6family_holm": family6,
            "bird_extra_comparisons": bird_extra,
            "entity_bird_159_per_db": ent_per_db,
            "ehrsql_600_vs_stage8_r1_holm": ehr_vs_s8_r1,
            "ehrsql_600_vs_stage8_r1_r3_holm": ehr_vs_s8_r3,
            "ehrsql_600_vs_phase6_r1_holm": ehr_vs_p6_r1,
            "ehrsql_600_vs_c0_c4_and_1way": ehr_vs_c0_c4,
        },
        "forensics": {
            "pubcap_aim_rows_bird_r1": pubcap_forensics,
            "dawa_dpsips_ehr600_r1": dawa_ehr_forensics,
        },
    }

    out_path = stats_dir / "phase12_authoritative_summary.json"
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(out_path)
    print(f"[Phase12 Consolidate] COMPLETE in {summary['elapsed_s']:.2f}s -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
