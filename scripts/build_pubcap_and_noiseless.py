#!/usr/bin/env python3
"""Stage 7 Builder for Tasks 2 & 3:
1. Run publish_noiseless on MIMIC-IV demo (17->13 public_tables projection) and 11 BIRD DBs.
   Verify 0 off-grid numeric quartiles, 0 partial-null triples, 0 null items in patterns/values.
2. Build pre-specified public-DDL token-cap (Cap_pub(S_d)) releases across R=3 replicates:
   - pubcap_silhouette/release_01..03/
   - pubcap_flatw/release_01..03/
   - pubcap_omega1_eqdelta/release_01..03/
   - pubcap_unpruned/release_01..03/
   - pubcap_ind8_profile/release_01..03/
   - pubcap_ind8_rows/release_01..03/
   plus full unpruned ind8_profile, bird_dg8_flatw, bird_dg8_omega1, bird_dg8_unpruned_v2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from silhouette import profile as prof_mod
from silhouette import schema_roles as sr
from silhouette.sandbox import PUBLIC_TABLES

VALID_GRID_SIG2 = {prof_mod.significant(e) for e in prof_mod.EDGES}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump_json_compact(path: Path, obj: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    return sha256_file(path)


def dump_json_pretty(path: Path, obj: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return sha256_file(path)


def verify_clean_noiseless_profile(profile_obj: Dict[str, Any], label: str) -> None:
    """Assert 0 off-grid numeric quartiles, 0 partial-null triples, 0 null items in patterns/values."""
    for tbl, cols in profile_obj.items():
        for col, entry in cols.items():
            assert isinstance(entry, list) and len(entry) == 5, (label, tbl, col, entry)
            typ, _nf, _ord, dist, vals = entry
            if sr.numeric(typ):
                assert isinstance(dist, list) and len(dist) == 3, (label, tbl, col, dist)
                all_none = all(x is None for x in dist)
                all_grid = all(isinstance(x, (int, float)) and float(x) in VALID_GRID_SIG2 for x in dist)
                assert all_none or all_grid, f"Invalid/partial-null or off-grid quartiles in {label}:{tbl}.{col}: {dist}"
            else:
                assert isinstance(dist, list), (label, tbl, col, dist)
                assert all(x is not None and isinstance(x, str) for x in dist), (
                    f"Null/non-str pattern in {label}:{tbl}.{col}: {dist}"
                )
            if vals is not None:
                assert isinstance(vals, list), (label, tbl, col, vals)
                assert all(v is not None for v in vals), f"Null in common_values {label}:{tbl}.{col}: {vals}"


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=ROOT,
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT,
    )
    parser.add_argument(
        "--mimic-db",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--bird-db-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--dp-profiles-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--ind8-probe-dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args(argv)

    R = args.project_root.resolve()
    OUT = args.out_root.resolve()
    OUT_PROF = OUT / "profiles"
    OUT_MAN = OUT / "manifests"
    OUT_PROF.mkdir(parents=True, exist_ok=True)
    OUT_MAN.mkdir(parents=True, exist_ok=True)

    mimic_db = (args.mimic_db or (R / "data" / "mimic_iv.sqlite")).resolve()
    bird_db_dir = (args.bird_db_dir or (R / "data" / "bird_databases")).resolve()
    dp_prof_dir = (args.dp_profiles_dir or (R / "profiles")).resolve()
    ind8_probe_dir = (args.ind8_probe_dir or (R / "data" / "ind8_probe_sqlite")).resolve()
    bird_dbs = sorted(p.stem for p in bird_db_dir.glob("*.sqlite") if not p.name.startswith("._"))
    assert len(bird_dbs) == 11, bird_dbs

    # 1. Task 2: publish_noiseless on MIMIC-IV demo (17->13) and 11 BIRD DBs
    ehr_nl = prof_mod.publish_noiseless(mimic_db, public_tables=PUBLIC_TABLES)
    assert len(ehr_nl) == 13 and not (set(ehr_nl) & set(PUBLIC_TABLES)), set(ehr_nl)
    verify_clean_noiseless_profile(ehr_nl, "mimic_iv")
    ehr_nl_path = OUT_PROF / "ehrsql_noiseless_nolkp/release_01.json"
    ehr_nl_sha = dump_json_pretty(ehr_nl_path, ehr_nl)
    print(f"[Noiseless EHRSQL] 13 tables verified clean -> sha256={ehr_nl_sha}")

    bird_nl_shas: Dict[str, str] = {}
    for db in bird_dbs:
        nl_prof = prof_mod.publish_noiseless(bird_db_dir / f"{db}.sqlite")
        verify_clean_noiseless_profile(nl_prof, db)
        p_out = OUT_PROF / f"bird_noiseless_clean/release_01/{db}.json"
        bird_nl_shas[db] = dump_json_compact(p_out, nl_prof)
    print(f"[Noiseless BIRD] All 11 DBs verified clean.")

    # 2. Task 3: Compute pure public-DDL Cap_pub(S_d) for each of the 11 BIRD DBs
    schemas_by_db: Dict[str, sr.PublicSchema] = {}
    caps_by_db: Dict[str, Dict[str, int]] = {}
    for db in bird_dbs:
        db_uri = (bird_db_dir / f"{db}.sqlite").resolve().as_uri() + "?mode=ro"
        con = sqlite3.connect(db_uri, uri=True)
        sch = sr.extract_public_schema(con)
        con.close()
        schemas_by_db[db] = sch
        cap = prof_mod.compute_public_profile_token_cap(sch)
        caps_by_db[db] = {
            "n_tables": len(sch.tables),
            "n_columns": sum(len(v) for v in sch.tables.values()),
            "cap_pub_tokens": cap,
        }
        print(f"  Cap_pub({db:25s}): |T|={caps_by_db[db]['n_tables']:2d}, |C|={caps_by_db[db]['n_columns']:3d} -> {cap} tokens")

    # 3. Copy full unpruned uncapped profiles into OUT_PROF (if source != destination)
    uncapped_copies = {
        "bird_dg8_flatw": dp_prof_dir / "bird_dg8_flatw",
        "bird_dg8_omega1": dp_prof_dir / "bird_dg8_omega1",
        "bird_dg8_unpruned_v2": dp_prof_dir / "bird_dg8_unpruned_v2",
    }
    for name, src_dir in uncapped_copies.items():
        for rep in (1, 2, 3):
            dst_r = OUT_PROF / f"{name}/release_0{rep}"
            dst_r.mkdir(parents=True, exist_ok=True)
            for db in bird_dbs:
                for ext in (".json", ".dpmeta.json"):
                    src_f = src_dir / f"release_0{rep}/{db}{ext}"
                    dst_f = dst_r / f"{db}{ext}"
                    if src_f.resolve() != dst_f.resolve():
                        shutil.copy2(src_f, dst_f)

    for rep in (1, 2, 3):
        dst_r = OUT_PROF / f"ind8_profile/release_0{rep}"
        dst_r.mkdir(parents=True, exist_ok=True)
        for db in bird_dbs:
            src_f = dp_prof_dir / f"ind8_profile/release_0{rep}/{db}.json"
            dst_f = dst_r / f"{db}.json"
            if src_f.resolve() != dst_f.resolve():
                shutil.copy2(src_f, dst_f)

    # 4. Build all 6 pubcap_* arms across release_01..03
    dp_sources = {
        "pubcap_silhouette": dp_prof_dir / "bird_minidev_dg_v3/eps_8",
        "pubcap_flatw": dp_prof_dir / "bird_dg8_flatw",
        "pubcap_omega1_eqdelta": dp_prof_dir / "bird_dg8_omega1",
        "pubcap_unpruned": dp_prof_dir / "bird_dg8_unpruned_v2",
    }

    manifest: Dict[str, Any] = {
        "schema_version": "pubcap-releases-v1",
        "cap_formula": "200 + 20 * |T| + 30 * |C|",
        "token_estimator": "re.compile(r'[A-Za-z_]+|\\d+|[^\\sA-Za-z0-9_]+|\\s+')",
        "noiseless_ehrsql_sha256": ehr_nl_sha,
        "noiseless_bird_sha256": bird_nl_shas,
        "dbs": caps_by_db,
        "arms": {},
    }

    for arm_name, src_base in dp_sources.items():
        arm_doc: Dict[str, Any] = {}
        for rep in (1, 2, 3):
            r_tag = f"release_0{rep}"
            arm_doc[r_tag] = {}
            out_r = OUT_PROF / arm_name / r_tag
            out_r.mkdir(parents=True, exist_ok=True)
            for db in bird_dbs:
                cap = caps_by_db[db]["cap_pub_tokens"]
                src_json = src_base / r_tag / f"{db}.json"
                src_meta = src_base / r_tag / f"{db}.dpmeta.json"
                prof_obj = json.loads(src_json.read_text(encoding="utf-8"))
                capped_obj = prof_mod.truncate_profile_to_public_cap(prof_obj, cap, schemas_by_db[db])
                tok_cnt = prof_mod.estimate_prompt_tokens(prof_mod.render_profile_prompt_block(capped_obj))
                assert tok_cnt <= cap, (arm_name, r_tag, db, tok_cnt, cap)

                dst_json = out_r / f"{db}.json"
                sha = dump_json_compact(dst_json, capped_obj)
                meta = json.loads(src_meta.read_text(encoding="utf-8"))
                meta["sha256"] = sha
                meta["pubcap_tokens"] = cap
                meta["actual_block_tokens"] = tok_cnt
                meta["truncated_by_pubcap"] = (capped_obj != prof_obj)
                dump_json_pretty(out_r / f"{db}.dpmeta.json", meta)
                arm_doc[r_tag][db] = {
                    "profile_path": f"{arm_name}/{r_tag}/{db}.json",
                    "dpmeta_path": f"{arm_name}/{r_tag}/{db}.dpmeta.json",
                    "sha256": sha,
                    "cap_tokens": cap,
                    "actual_block_tokens": tok_cnt,
                    "truncated": capped_obj != prof_obj,
                }
        manifest["arms"][arm_name] = arm_doc

    # pubcap_ind8_profile
    ind_prof_doc: Dict[str, Any] = {}
    for rep in (1, 2, 3):
        r_tag = f"release_0{rep}"
        ind_prof_doc[r_tag] = {}
        out_r = OUT_PROF / "pubcap_ind8_profile" / r_tag
        out_r.mkdir(parents=True, exist_ok=True)
        for db in bird_dbs:
            cap = caps_by_db[db]["cap_pub_tokens"]
            src_json = dp_prof_dir / f"ind8_profile/release_0{rep}/{db}.json"
            prof_obj = json.loads(src_json.read_text(encoding="utf-8"))
            capped_obj = prof_mod.truncate_profile_to_public_cap(prof_obj, cap, schemas_by_db[db])
            tok_cnt = prof_mod.estimate_prompt_tokens(prof_mod.render_profile_prompt_block(capped_obj))
            assert tok_cnt <= cap, ("pubcap_ind8_profile", r_tag, db, tok_cnt, cap)
            sha = dump_json_compact(out_r / f"{db}.json", capped_obj)
            ind_prof_doc[r_tag][db] = {
                "profile_path": f"pubcap_ind8_profile/{r_tag}/{db}.json",
                "sha256": sha,
                "cap_tokens": cap,
                "actual_block_tokens": tok_cnt,
                "truncated": capped_obj != prof_obj,
            }
    manifest["arms"]["pubcap_ind8_profile"] = ind_prof_doc

    # pubcap_ind8_rows
    ind_rows_doc: Dict[str, Any] = {}
    for rep in (1, 2, 3):
        r_tag = f"release_0{rep}"
        ind_rows_doc[r_tag] = {}
        out_r = OUT_PROF / "pubcap_ind8_rows" / r_tag
        out_r.mkdir(parents=True, exist_ok=True)
        for db in bird_dbs:
            cap = caps_by_db[db]["cap_pub_tokens"]
            src_sqlite = ind8_probe_dir / f"R{rep}/{db}.sqlite"
            rows_txt = prof_mod.truncate_synth_rows_to_public_cap(src_sqlite, cap, schemas_by_db[db])
            tok_cnt = prof_mod.estimate_prompt_tokens(rows_txt)
            assert tok_cnt <= cap, ("pubcap_ind8_rows", r_tag, db, tok_cnt, cap)
            dst_txt = out_r / f"{db}.txt"
            dst_txt.write_text(rows_txt, encoding="utf-8")
            sha = sha256_file(dst_txt)
            ind_rows_doc[r_tag][db] = {
                "rows_path": f"pubcap_ind8_rows/{r_tag}/{db}.txt",
                "sha256": sha,
                "cap_tokens": cap,
                "actual_block_tokens": tok_cnt,
            }
    manifest["arms"]["pubcap_ind8_rows"] = ind_rows_doc

    man_path = OUT_MAN / "pubcap_releases_manifest.json"
    dump_json_pretty(man_path, manifest)
    print(f"[PubCap Releases] All 6 arms x 3 releases x 11 DBs staged under {OUT_PROF}")
    print(f"Manifest written -> {man_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
