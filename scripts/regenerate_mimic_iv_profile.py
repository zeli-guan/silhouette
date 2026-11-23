"""Regenerate one local release from the pinned SQLite snapshot or PhysioNet demo ZIP."""
from __future__ import annotations

import argparse
from contextlib import closing
import gzip
import hashlib
import json
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from silhouette.profile import publish
from silhouette.sandbox import PUBLIC_TABLES, readonly

REVISION = "f9e1aa02160d39e3f8df52bf5c69c5cf2e472499"
PREPROCESS = {
    "preprocess_db.py": "b5f37001d679740f2c07852a30b20cb165155a1f5d90a4f19aca8bba9aa2898f",
    "preprocess_db_mimic_iv.py": "0a735e4cd0001230afea0d1110fd9c47ea9cd6621f1e29a80150d77e432803b8",
    "preprocess_utils.py": "3837ee5e76e3fa6bed5bfc93c591664c8bd2c348b5093827c618e0581341e1c3",
}


def check(path, expected):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    if h.hexdigest() != expected:
        raise ValueError("Input checksum mismatch for " + path.name)


def preprocess(archive, work, checkout, interpreter):
    with zipfile.ZipFile(archive) as z:
        for member in z.namelist():
            parts = Path(member).parts
            if len(parts) == 3 and parts[1] in ("hosp", "icu") and parts[2].endswith(".csv.gz"):
                dest = work / parts[1] / parts[2][:-3]
                dest.parent.mkdir(exist_ok=True)
                with z.open(member) as raw, gzip.GzipFile(fileobj=raw) as csv, dest.open("wb") as f:
                    shutil.copyfileobj(csv, f)
    for name, digest in PREPROCESS.items():
        dest = work / name
        if checkout:
            shutil.copyfile(checkout / "preprocess" / name, dest)
        else:
            url = "https://raw.githubusercontent.com/glee4810/ehrsql-2024/" + REVISION + "/preprocess/" + name
            with urllib.request.urlopen(url, timeout=60) as src, dest.open("wb") as f:
                shutil.copyfileobj(src, f)
        check(dest, digest)
    dbdir = work / "mimic_iv"
    dbdir.mkdir()
    shutil.copyfile(ROOT / "manifests/mimic_iv_public_ddl.sql", dbdir / "mimic_iv.sql")
    cmd = [interpreter, "-B", str(work / "preprocess_db.py"), "--data_dir", str(work),
           "--db_name", "mimic_iv", "--num_patient", "100", "--timeshift",
           "--current_time", "2100-12-31 23:59:00", "--start_year", "2100", "--time_span", "0",
           "--out_dir", str(work), "--cur_patient_ratio", "0.1"]
    subprocess.run(cmd, cwd=str(work), check=True)
    return dbdir / "mimic_iv.sqlite"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="EHRSQL SQLite or original demo ZIP")
    parser.add_argument("output", type=Path, help="New local JSON file outside the artifact")
    parser.add_argument("--epsilon", type=int, choices=[1, 2, 4, 8], default=8)
    parser.add_argument("--checkout", type=Path, help="Local pinned EHRSQL checkout for offline ZIP preprocessing")
    parser.add_argument("--preprocess-python", default=sys.executable, help="Python with upstream preprocessing dependencies")
    args = parser.parse_args()
    target = args.output.resolve()
    if target == ROOT or ROOT in target.parents or target.exists():
        parser.error("Choose a new local output outside the distributed artifact")
    manifest = json.loads((ROOT / "manifests/ehrsql_2024_manifest.json").read_text())
    zipped = args.input.suffix == ".zip"
    expected = "mimic-iv-clinical-database-demo-2.2.zip" if zipped else "mimic_iv.sqlite"
    check(args.input, manifest["sha256"][expected])
    with tempfile.TemporaryDirectory(dir=str(target.parent)) as tmp:
        work = Path(tmp)
        db = preprocess(args.input, work, args.checkout, args.preprocess_python) if zipped else args.input
        with closing(readonly(db)) as conn:
            profile, _ = publish(conn, args.epsilon, public_tables=PUBLIC_TABLES)
        staged = work / "release.json"
        staged.write_text(json.dumps(profile, separators=(",", ":"), allow_nan=False) + "\n")
        staged.chmod(0o600)
        os.link(str(staged), str(target))
    print("Local DG-v3 release written.")


if __name__ == "__main__":
    main()
