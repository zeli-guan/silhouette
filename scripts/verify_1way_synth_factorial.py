#!/usr/bin/env python3
"""Deterministic verifier and builder for the 2x2 1-way synthetic vs profile factorial.

Supports two self-contained modes (stdlib + numpy only, zero external paths):
1. Default (`--verify-packaged`):
   - (a) Manifest SHA-256 pinning: verifies all 33 frozen `profiles/ind8_profile/release_01..03/{db}.json`
     releases, all 33 evaluated `profiles/pubcap_ind8_profile/release_01..03/{db}.json` Cap_pub-truncated
     profiles, and all 33 frozen evaluated `profiles/pubcap_ind8_rows/release_01..03/{db}.txt` row blocks
     against `manifests/pubcap_releases_manifest.json`, plus all 66 `tightbpe_ind8_prof` &
     `tightbpe_static_synth` files against `manifests/tightbpe_and_noval_manifest.json`.
   - (b) Reference post-processor determinism from frozen 1-way profiles: re-runs
     `truncate_profile_to_public_cap` from the frozen `ind8_profile` JSONs (verifying bit-for-bit
     SHA-256 identity with `pubcap_ind8_profile`), and deterministically synthesizes 1-way SQLite
     tables from the frozen `ind8_profile` JSONs (`build_deterministic_1way_synth_db`) to verify
     repeatable `truncate_synth_rows_to_public_cap` output (`txt_a == txt_b`) within the
     `Cap_pub(S_d)` token budget (`<= cap_tokens`). Because `build_deterministic_1way_synth_db`
     uses an offline reference sampler rather than the historical upstream probe `.sqlite`
     sampling parameters, `txt_a` is checked for deterministic repeatability and budget compliance
     rather than byte-for-byte identity against `pubcap_ind8_rows`.
2. Optional (`--ind8-probe-dir DIR --bird-db-dir DIR`):
   - Given the original upstream 1-way synthetic `.sqlite` probe databases (`R1..R3/{db}.sqlite`)
     and public BIRD schema DDLs, verifies bit-for-bit SHA-256 identity of all 33 packaged
     `profiles/pubcap_ind8_rows/release_01..03/{db}.txt` files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import sys
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from silhouette import profile as prof_mod  # noqa: E402
from silhouette import schema_roles as sr  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _schema_from_profile(profile_obj: Dict[str, Any]) -> sr.PublicSchema:
    """Reconstruct column-ordered PublicSchema purely from a frozen profile JSON."""
    tables: Dict[str, tuple[sr.ColumnMeta, ...]] = {}
    for table in sorted(profile_obj):
        cols_dict = profile_obj[table]
        if not isinstance(cols_dict, dict):
            continue
        col_metas: List[sr.ColumnMeta] = []
        for idx, (col_name, entry) in enumerate(cols_dict.items()):
            decl_type = str(entry[0]) if isinstance(entry, list) and entry else "TEXT"
            col_metas.append(
                sr.ColumnMeta(
                    name=col_name,
                    declared_type=decl_type,
                    is_pk=(idx == 0),
                    is_single_pk_or_unique=(idx == 0),
                    is_composite_pk=False,
                    is_fk_src=False,
                    is_fk_tgt=False,
                )
            )
        tables[table] = tuple(col_metas)
    return sr.PublicSchema(tables=tables, fk_edges=())


def _distinct_budget(distinct_order: Any, max_distinct: int) -> int:
    if distinct_order == 0 or distinct_order is None:
        return 0
    if isinstance(distinct_order, str) and distinct_order.startswith("10^"):
        try:
            exp = int(distinct_order.split("^", 1)[1])
            return max(1, min(max_distinct, 10 ** min(exp, 3)))
        except ValueError:
            pass
    return min(max_distinct, 16)


def _synth_value_from_shape(shape: str, idx: int, rng: random.Random) -> str:
    chars: List[str] = []
    for ch in shape:
        if ch == "\u2026":
            break
        if ch == "A":
            chars.append(chr(ord("A") + (idx + rng.randrange(26)) % 26))
        elif ch == "a":
            chars.append(chr(ord("a") + (idx + rng.randrange(26)) % 26))
        elif ch == "0":
            chars.append(str((idx + rng.randrange(10)) % 10))
        else:
            chars.append(ch)
    return "".join(chars) or f"v_{idx}"


def build_deterministic_1way_synth_db(
    ind8_profile: Dict[str, Any],
    *,
    seed: int = 20261121,
    rows_per_table: int = 64,
    out_sqlite: Optional[Path] = None,
) -> sqlite3.Connection:
    """Deterministically materialize a 1-way synthetic SQLite DB purely from a frozen IND8_PROMPT profile."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    if out_sqlite is not None:
        out_sqlite.parent.mkdir(parents=True, exist_ok=True)
        if out_sqlite.exists():
            out_sqlite.unlink()
        con = sqlite3.connect(str(out_sqlite))
    else:
        con = sqlite3.connect(":memory:")

    for table in sorted(ind8_profile):
        cols = ind8_profile[table]
        if not isinstance(cols, dict) or not cols:
            continue
        col_defs = []
        for cname, entry in cols.items():
            decl_type = str(entry[0]) if isinstance(entry, list) and entry else "TEXT"
            col_defs.append(f"{sr.quote(cname)} {sr.quote(decl_type)}")
        con.execute(f"CREATE TABLE {sr.quote(table)} ({', '.join(col_defs)})")

        col_names = list(cols.keys())
        col_values: Dict[str, List[Any]] = {}
        for cname in col_names:
            entry = cols[cname]
            decl_type = str(entry[0])
            null_frac = float(entry[1]) if entry[1] is not None else 0.0
            null_frac = min(1.0, max(0.0, null_frac))
            distinct_order = entry[2]
            dist = entry[3]
            common_vals = entry[4]

            n_null = int(round(rows_per_table * null_frac))
            n_nonnull = max(0, rows_per_table - n_null)
            is_num = sr.numeric(decl_type)
            is_int = "INT" in decl_type.upper()

            pool: List[Any] = []
            if isinstance(common_vals, list) and common_vals:
                pool.extend(common_vals)

            target_k = _distinct_budget(distinct_order, max(len(pool), 16))
            if is_num:
                qs = [x for x in (dist if isinstance(dist, list) else []) if isinstance(x, (int, float))]
                if qs:
                    lo, med, hi = float(min(qs)), float(qs[len(qs) // 2]), float(max(qs))
                    span = max(abs(hi - lo), 1.0)
                    samples = np_rng.uniform(lo - 0.1 * span, hi + 0.1 * span, size=max(target_k, 8))
                    for s in samples:
                        val = int(round(float(s))) if is_int else round(float(s), 4)
                        pool.append(val)
                    if not pool:
                        pool.append(int(round(med)) if is_int else med)
                elif not pool:
                    pool.extend(range(1, max(2, target_k) + 1) if is_int else [float(i) for i in range(1, max(2, target_k) + 1)])
            else:
                shapes = [s for s in (dist if isinstance(dist, list) else []) if isinstance(s, str) and s]
                idx = 0
                while len(pool) < max(1, target_k):
                    if shapes:
                        sh = shapes[idx % len(shapes)]
                        pool.append(_synth_value_from_shape(sh, idx, rng))
                    else:
                        pool.append(f"syn_{cname}_{idx}")
                    idx += 1

            if n_nonnull == 0:
                vals = [None] * rows_per_table
            else:
                nonnull_vals = [pool[i % len(pool)] for i in range(n_nonnull)]
                vals = nonnull_vals + [None] * n_null
                rng.shuffle(vals)
            col_values[cname] = vals

        rows = [tuple(col_values[c][r] for c in col_names) for r in range(rows_per_table)]
        placeholders = ", ".join("?" for _ in col_names)
        quoted_cols = ", ".join(sr.quote(c) for c in col_names)
        con.executemany(f"INSERT INTO {sr.quote(table)} ({quoted_cols}) VALUES ({placeholders})", rows)

    con.commit()
    return con


def verify_packaged(root: Path) -> Dict[str, Any]:
    pub_manifest = json.loads((root / "manifests/pubcap_releases_manifest.json").read_text(encoding="utf-8"))
    tight_manifest = json.loads((root / "manifests/tightbpe_and_noval_manifest.json").read_text(encoding="utf-8"))

    dbs = sorted(pub_manifest["dbs"].keys())
    assert len(dbs) == 11, dbs

    ind8_verified = 0
    pubcap_prof_sha_matched = 0
    pubcap_rows_sha_matched = 0
    synth_roundtrip_verified = 0

    for rep in (1, 2, 3):
        r_tag = f"release_0{rep}"
        for db in dbs:
            cap = int(pub_manifest["dbs"][db]["cap_pub_tokens"])
            ind8_path = root / "profiles" / "ind8_profile" / r_tag / f"{db}.json"
            assert ind8_path.exists(), ind8_path
            ind8_obj = json.loads(ind8_path.read_text(encoding="utf-8"))
            ind8_verified += 1

            # 1. Verify Cap_pub-truncated 1-way profile SHA-256 and deterministic truncation
            prof_entry = pub_manifest["arms"]["pubcap_ind8_profile"][r_tag][db]
            prof_path = root / "profiles" / prof_entry["profile_path"]
            assert sha256_file(prof_path) == prof_entry["sha256"], (r_tag, db)
            schema = _schema_from_profile(ind8_obj)
            recomputed = prof_mod.truncate_profile_to_public_cap(ind8_obj, cap, schema)
            recomputed_sha = hashlib.sha256(canonical_json_bytes(recomputed)).hexdigest()
            assert recomputed_sha == prof_entry["sha256"], (
                f"Deterministic Cap_pub profile truncation mismatch for {r_tag}/{db}: "
                f"{recomputed_sha} != {prof_entry['sha256']}"
            )
            pubcap_prof_sha_matched += 1

            # 2. (a) Manifest SHA-256 pinning & Cap_pub budget check for frozen evaluated pubcap_ind8_rows
            rows_entry = pub_manifest["arms"]["pubcap_ind8_rows"][r_tag][db]
            rows_path = root / "profiles" / rows_entry["rows_path"]
            assert sha256_file(rows_path) == rows_entry["sha256"], (r_tag, db)
            rows_txt = rows_path.read_text(encoding="utf-8")
            assert prof_mod.estimate_prompt_tokens(rows_txt) <= cap, (r_tag, db)
            pubcap_rows_sha_matched += 1

            # 3. (b) Reference post-processor determinism from frozen IND8_PROMPT JSON:
            #    synthesize 1-way SQLite twice with identical seed and verify repeatable Cap_pub
            #    row rendering (txt_a == txt_b <= cap_tokens). Byte identity against the
            #    packaged pubcap_ind8_rows files uses historical upstream probe SQLite params
            #    and is checked separately via --ind8-probe-dir.
            con_a = build_deterministic_1way_synth_db(ind8_obj, seed=20261121 + rep)
            con_b = build_deterministic_1way_synth_db(ind8_obj, seed=20261121 + rep)
            try:
                txt_a = prof_mod.truncate_synth_rows_to_public_cap(con_a, cap, schema)
                txt_b = prof_mod.truncate_synth_rows_to_public_cap(con_b, cap, schema)
                assert txt_a == txt_b, (r_tag, db)
                assert prof_mod.estimate_prompt_tokens(txt_a) <= cap, (r_tag, db)
            finally:
                con_a.close()
                con_b.close()
            synth_roundtrip_verified += 1

    # 4. (a) Manifest SHA-256 pinning for 66 TightBPE 1-way profile & static synth files
    tight_verified = 0
    for arm_name in ("tightbpe_ind8_prof", "tightbpe_static_synth"):
        for r_tag, db_map in tight_manifest["arms"][arm_name].items():
            for db, entry in db_map.items():
                fpath = root / entry["path"]
                assert sha256_file(fpath) == entry["sha256"], (arm_name, r_tag, db)
                tight_verified += 1

    return {
        "ind8_profiles_verified": ind8_verified,
        "pubcap_ind8_profile_exact_sha_matched": pubcap_prof_sha_matched,
        "pubcap_ind8_rows_exact_sha_matched": pubcap_rows_sha_matched,
        "deterministic_1way_synth_roundtrip_verified": synth_roundtrip_verified,
        "tightbpe_1way_files_verified": tight_verified,
        "verification_scope": {
            "manifest_sha256_pinned": (
                "All 33 frozen evaluated pubcap_ind8_rows files, 33 pubcap_ind8_profile files, "
                "and 66 tightbpe_ind8_prof/tightbpe_static_synth files match manifest SHA-256 digests."
            ),
            "reference_postprocessor_determinism": (
                "Re-running truncate_profile_to_public_cap reproduces all 33 pubcap_ind8_profile "
                "SHA-256 digests bit-for-bit; build_deterministic_1way_synth_db + "
                "truncate_synth_rows_to_public_cap is deterministic (txt_a == txt_b <= cap_tokens) "
                "from frozen 1-way profiles without assuming historical probe SQLite sampling parameters."
            ),
        },
    }


def verify_external_probe_sqlite(root: Path, ind8_probe_dir: Path, bird_db_dir: Path) -> Dict[str, Any]:
    pub_manifest = json.loads((root / "manifests/pubcap_releases_manifest.json").read_text(encoding="utf-8"))
    matched = 0
    for rep in (1, 2, 3):
        r_tag = f"release_0{rep}"
        for db, cap_info in sorted(pub_manifest["dbs"].items()):
            cap = int(cap_info["cap_pub_tokens"])
            bird_db = bird_db_dir / f"{db}.sqlite"
            probe_db = ind8_probe_dir / f"R{rep}" / f"{db}.sqlite"
            con_schema = sqlite3.connect(bird_db.resolve().as_uri() + "?mode=ro", uri=True)
            try:
                schema = sr.extract_public_schema(con_schema)
            finally:
                con_schema.close()
            rendered = prof_mod.truncate_synth_rows_to_public_cap(probe_db, cap, schema)
            rendered_sha = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
            expected_sha = pub_manifest["arms"]["pubcap_ind8_rows"][r_tag][db]["sha256"]
            assert rendered_sha == expected_sha, (
                f"External probe SQLite row rendering mismatch for {r_tag}/{db}: {rendered_sha} != {expected_sha}"
            )
            matched += 1
    return {"external_probe_rows_exact_sha_matched": matched}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--ind8-probe-dir", type=Path, default=None)
    parser.add_argument("--bird-db-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    res = verify_packaged(args.root.resolve())
    if args.ind8_probe_dir is not None or args.bird_db_dir is not None:
        if args.ind8_probe_dir is None or args.bird_db_dir is None:
            raise ValueError("Pass both --ind8-probe-dir and --bird-db-dir together")
        res.update(verify_external_probe_sqlite(args.root.resolve(), args.ind8_probe_dir, args.bird_db_dir))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
