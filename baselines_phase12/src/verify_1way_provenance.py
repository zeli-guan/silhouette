"""Step 3: Verify exact 33/33 provenance of pubcap_ind8_rows from phase6 ind8_releases probe SQLite DBs."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from silhouette import profile as prof_mod  # noqa: E402
from silhouette import schema_roles as sr  # noqa: E402


def verify_packaged_report(manifest_path: Path, report_path: Path) -> int:
    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    rep_obj = json.loads(report_path.read_text(encoding="utf-8"))
    assert rep_obj.get("matched") == 33 and rep_obj.get("total") == 33, rep_obj
    entries = rep_obj.get("entries", [])
    assert len(entries) == 33
    for ent in entries:
        r_tag = ent["rep"]
        db = ent["db"]
        exp_sha = man["arms"]["pubcap_ind8_rows"][r_tag][db]["sha256"]
        assert ent["sha256"] == exp_sha, (r_tag, db, ent["sha256"], exp_sha)
        rows_path = REPO_ROOT / "profiles" / man["arms"]["pubcap_ind8_rows"][r_tag][db]["rows_path"]
        assert rows_path.exists(), rows_path
        actual_sha = hashlib.sha256(rows_path.read_bytes()).hexdigest()
        assert actual_sha == exp_sha, (r_tag, db, actual_sha, exp_sha)
    print(f"[Step 3 Provenance] Offline report & packaged files verified 33/33 -> {report_path}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "manifests/pubcap_releases_manifest.json")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "manifests/verify_1way_provenance_report.json")
    parser.add_argument("--ind8-probe-dir", type=Path, default=None)
    parser.add_argument("--bird-db-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.ind8_probe_dir is None or args.bird_db_dir is None:
        return verify_packaged_report(args.manifest, args.out)

    man = json.loads(args.manifest.read_text(encoding="utf-8"))
    matched = 0
    details = []
    for rep in (1, 2, 3):
        r_tag = f"release_0{rep}"
        for db, cap_info in sorted(man["dbs"].items()):
            cap = int(cap_info["cap_pub_tokens"])
            probe_db = args.ind8_probe_dir / f"R{rep}" / f"{db}.sqlite"
            bird_db = args.bird_db_dir / f"{db}.sqlite"
            con_s = sqlite3.connect(bird_db.resolve().as_uri() + "?mode=ro", uri=True)
            try:
                schema = sr.extract_public_schema(con_s)
            finally:
                con_s.close()
            rendered = prof_mod.truncate_synth_rows_to_public_cap(probe_db, cap, schema)
            r_sha = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
            exp_sha = man["arms"]["pubcap_ind8_rows"][r_tag][db]["sha256"]
            assert r_sha == exp_sha, (r_tag, db, r_sha, exp_sha)
            matched += 1
            details.append({"rep": r_tag, "db": db, "sha256": r_sha, "source_sqlite": str(probe_db)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "matched": matched,
                "total": 33,
                "note": (
                    "All 33/33 packaged pubcap_ind8_rows files match bit-for-bit SHA-256 when "
                    "truncate_synth_rows_to_public_cap is invoked on the 33 upstream ind8_releases "
                    "probe SQLite databases."
                ),
                "entries": details,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[Step 3 Provenance] Matched {matched}/33 bit-for-bit -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
