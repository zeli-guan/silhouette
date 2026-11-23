"""Build all Phase 12 DP baseline releases across 11 BIRD DBs + MIMIC-IV x R=3 replicates."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = BASE / "src"
for _p in (str(SRC), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import baseline_aim  # noqa: E402
import baseline_dawa_dpsips  # noqa: E402
import baseline_privlava  # noqa: E402
import baseline_privpetal  # noqa: E402

ENTITY_4 = ("codebase_community", "debit_card_specializing", "financial", "thrombosis_prediction")
BASE_SEED = 20261123


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def _run_one_task(task: Tuple[str, int, str, str, str, str]) -> Dict[str, Any]:
    kind, rep, db_id, bird_db_dir_str, ehr_db_str, rel_dir_str = task
    bird_db_dir = Path(bird_db_dir_str)
    ehr_db = Path(ehr_db_str)
    rel_dir = Path(rel_dir_str)
    t0 = time.monotonic()
    is_ehr = db_id == "mimic_iv"
    src_db = ehr_db if is_ehr else (bird_db_dir / f"{db_id}.sqlite")
    seed = BASE_SEED + rep * 1000 + sum(ord(c) for c in (kind + db_id))
    r_tag = f"R{rep}"

    if kind == "aim":
        out = rel_dir / "aim" / r_tag / f"{db_id}.sqlite"
        txt_out = rel_dir / "pubcap_aim_rows" / r_tag / f"{db_id}.txt"
        if out.exists() and out.stat().st_size > 0 and (is_ehr or (txt_out.exists() and txt_out.stat().st_size > 0)):
            extra = {"pubcap": {"path": str(txt_out), "sha256": sha256_file(txt_out)}} if not is_ehr else {}
            return {"kind": kind, "rep": rep, "db_id": db_id, "path": str(out), "sha256": sha256_file(out), "info": {"cached": True}, "extra": extra, "sec": 0.0}
        rep_info = baseline_aim.synthesize_db_aim(src_db, out, seed=seed, is_ehr=is_ehr)
        extra = {}
        if not is_ehr:
            txt_out = rel_dir / "pubcap_aim_rows" / r_tag / f"{db_id}.txt"
            extra["pubcap"] = baseline_aim.render_pubcap_aim_rows(src_db, out, txt_out)
            extra["pubcap"]["sha256"] = sha256_file(txt_out)
            extra["pubcap"]["path"] = str(txt_out)
        return {"kind": kind, "rep": rep, "db_id": db_id, "path": str(out), "sha256": sha256_file(out), "info": rep_info, "extra": extra, "sec": round(time.monotonic() - t0, 2)}

    if kind == "privlava":
        out = rel_dir / "privlava" / r_tag / f"{db_id}.sqlite"
        if out.exists() and out.stat().st_size > 0:
            return {"kind": kind, "rep": rep, "db_id": db_id, "path": str(out), "sha256": sha256_file(out), "info": {"cached": True}, "sec": 0.0}
        rep_info = baseline_privlava.synthesize_db_privlava(src_db, out, seed=seed, is_ehr=is_ehr)
        return {"kind": kind, "rep": rep, "db_id": db_id, "path": str(out), "sha256": sha256_file(out), "info": rep_info, "sec": round(time.monotonic() - t0, 2)}

    if kind == "entity_privlava":
        out = rel_dir / "entity_privlava" / r_tag / f"{db_id}.sqlite"
        if out.exists() and out.stat().st_size > 0:
            return {"kind": kind, "rep": rep, "db_id": db_id, "path": str(out), "sha256": sha256_file(out), "info": {"cached": True}, "sec": 0.0}
        rep_info = baseline_privlava.synthesize_db_privlava(src_db, out, seed=seed, is_ehr=False, entity_mode=True, entity_cap=5)
        return {"kind": kind, "rep": rep, "db_id": db_id, "path": str(out), "sha256": sha256_file(out), "info": rep_info, "sec": round(time.monotonic() - t0, 2)}

    if kind == "privpetal":
        out = rel_dir / "privpetal" / r_tag / f"{db_id}.sqlite"
        if out.exists() and out.stat().st_size > 0:
            return {"kind": kind, "rep": rep, "db_id": db_id, "path": str(out), "sha256": sha256_file(out), "info": {"cached": True}, "sec": 0.0}
        rep_info = baseline_privpetal.synthesize_db_privpetal(src_db, out, seed=seed, is_ehr=is_ehr)
        return {"kind": kind, "rep": rep, "db_id": db_id, "path": str(out), "sha256": sha256_file(out), "info": rep_info, "sec": round(time.monotonic() - t0, 2)}

    if kind == "dawa_dpsips":
        out = rel_dir / "dawa_dpsips" / r_tag / f"{db_id}.json"
        if out.exists() and out.stat().st_size > 0:
            return {"kind": kind, "rep": rep, "db_id": db_id, "path": str(out), "sha256": sha256_file(out), "info": {"cached": True}, "sec": 0.0}
        rep_info = baseline_dawa_dpsips.profile_db_dawa_dpsips(src_db, out, seed=seed, is_ehr=is_ehr)
        return {"kind": kind, "rep": rep, "db_id": db_id, "path": str(out), "sha256": sha256_file(out), "info": rep_info, "sec": round(time.monotonic() - t0, 2)}

    raise ValueError(kind)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bird-db-dir", type=Path, default=Path(os.environ.get("SILHOUETTE_BIRD_DB_DIR", "")))
    parser.add_argument("--ehr-db", type=Path, default=Path(os.environ.get("SILHOUETTE_EHR_DB", "")))
    parser.add_argument("--rel-dir", type=Path, default=BASE / "releases")
    parser.add_argument("--manifest-out", type=Path, default=REPO_ROOT / "manifests/phase12_releases_manifest.json")
    parser.add_argument("--tokenizer-path", type=str, default=os.environ.get("SILHOUETTE_QWEN_TOKENIZER", ""))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    if not str(args.bird_db_dir) or not args.bird_db_dir.is_dir():
        raise FileNotFoundError("Pass --bird-db-dir DIR (or set SILHOUETTE_BIRD_DB_DIR) pointing to BIRD SQLite databases")
    if not str(args.ehr_db) or not args.ehr_db.is_file():
        raise FileNotFoundError("Pass --ehr-db FILE (or set SILHOUETTE_EHR_DB) pointing to mimic_iv.sqlite")

    t0 = time.monotonic()
    bird_dbs = sorted(p.stem for p in args.bird_db_dir.glob("*.sqlite") if not p.name.startswith("._"))
    assert len(bird_dbs) == 11, f"Expected 11 BIRD DBs in {args.bird_db_dir}, got {len(bird_dbs)}"
    all_dbs = bird_dbs + ["mimic_iv"]

    tasks: List[Tuple[str, int, str, str, str, str]] = []
    for rep in (1, 2, 3):
        for db in all_dbs:
            for kind in ("aim", "privlava", "privpetal", "dawa_dpsips"):
                tasks.append((kind, rep, db, str(args.bird_db_dir), str(args.ehr_db), str(args.rel_dir)))
        for db in ENTITY_4:
            tasks.append(("entity_privlava", rep, db, str(args.bird_db_dir), str(args.ehr_db), str(args.rel_dir)))

    print(f"[Build Releases] Dispatching {len(tasks)} release tasks across {args.workers} workers...", flush=True)
    manifest: Dict[str, Any] = {"arms": {}, "tightbpe_dawa_dpsips": {}}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run_one_task, t): t for t in tasks}
        for fut in as_completed(futs):
            res = fut.result()
            k, r, db = res["kind"], res["rep"], res["db_id"]
            print(f"  done {k:15s} R{r} {db:25s} ({res['sec']:.2f}s)", flush=True)
            manifest["arms"].setdefault(k, {}).setdefault(f"R{r}", {})[db] = {
                "path": res["path"],
                "sha256": res["sha256"],
            }
            if res.get("extra", {}).get("pubcap"):
                manifest["arms"].setdefault("pubcap_aim_rows", {}).setdefault(f"R{r}", {})[db] = res["extra"]["pubcap"]

    if args.tokenizer_path:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
        bpe_len = lambda s: len(tok.encode(s, add_special_tokens=False))
        for rep in (1, 2, 3):
            r_tag = f"R{rep}"
            for db in bird_dbs:
                src_json = args.rel_dir / "dawa_dpsips" / r_tag / f"{db}.json"
                src_db = args.bird_db_dir / f"{db}.sqlite"
                out_json = args.rel_dir / "tightbpe_dawa_dpsips" / r_tag / f"{db}.json"
                tinfo = baseline_dawa_dpsips.truncate_profile_tightbpe(src_json, src_db, out_json, bpe_len)
                manifest["tightbpe_dawa_dpsips"].setdefault(r_tag, {})[db] = {
                    "path": str(out_json),
                    "sha256": sha256_file(out_json),
                    **tinfo,
                }
        print("[Build Releases] TightBPE DAWA+DP-SIPS truncation complete.", flush=True)

    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[Build Releases] ALL COMPLETE in {time.monotonic()-t0:.2f}s -> {args.manifest_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
