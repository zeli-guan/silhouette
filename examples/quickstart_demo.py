"""Offline, synthetic end-to-end example."""
from __future__ import annotations

from contextlib import closing
from fractions import Fraction
import json
from pathlib import Path
import sqlite3
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from certificates.verify_certificates import check_parameter
from silhouette.agent import (
    SingleReleaseReceiverLedger,
    canonical_profile_sha256,
    generate_sql,
    verify_release_bundle,
)
from silhouette.audit import TransmissionLedger
from silhouette.profile import publish
from silhouette.sandbox import LocalExecutor, SchemaSandbox


def main():
    if sqlite3.sqlite_version_info < (3, 35, 0):
        raise RuntimeError(
            f"SQLite >= 3.35.0 is required for AS MATERIALIZED CTE support "
            f"(found {sqlite3.sqlite_version}; paper evaluated on SQLite 3.53.1)."
        )
    with tempfile.TemporaryDirectory(prefix=".synthetic-demo-", dir=str(ROOT)) as tmp:
        path = Path(tmp) / "demo.sqlite"
        ddl = "CREATE TABLE tiles(colour TEXT, sides INTEGER)"
        private = [("ochre", 3)] * 80 + [("azure", 4)] * 80
        with closing(sqlite3.connect(str(path))) as con:
            con.execute(ddl)
            con.executemany("INSERT INTO tiles VALUES (?,?)", private)
            con.commit()
            release, plans = publish(con)
            empty = SchemaSandbox(con)
        plan = plans["tiles"]
        rho = Fraction(0)
        delta = Fraction(0)
        for q in plan["queries"]:
            r, _, d, _, _ = check_parameter(dict(r_j=q["rational_share"], rho=q["rho_frac"], sigma2=q["sigma2_frac"],
                share=Fraction(q["rational_share"], q["rational_total"]), delta=q["delta_frac"],
                sensitivity2=1, tau=q.get("tau"), **q.get("tau_tail_diag", {})))
            rho += r
            delta += d
        if rho != Fraction(plan["rho_table_frac"]) or delta != Fraction(1, 2000000):
            raise ValueError("Synthetic ledger mismatch")
        sql = "SELECT COUNT(*) FROM tiles"
        profile_text = json.dumps(release, ensure_ascii=False, separators=(",", ":"))
        profile_sha = canonical_profile_sha256(release)
        verified_bundle = verify_release_bundle(release, expected_profile_sha256=profile_sha)
        receiver_ledger = SingleReleaseReceiverLedger(Path(tmp) / "receiver_ledger.json")
        ledger = TransmissionLedger({"tiles": private}, release, public=[ddl, "How many tiles?"])
        calls = iter(("try_sql", "submit_sql"))

        def model(request):
            ledger.record(request, profile_1d_text=profile_text)
            return {"name": next(calls), "arguments": {"sql": sql}}

        final = generate_sql(
            model,
            "How many tiles?",
            ddl,
            release,
            sandbox=empty,
            expected_profile_sha256=verified_bundle["profile_sha256"],
            receiver_id="cloud-prod-v1",
            receiver_ledger=receiver_ledger,
        )
        empty.connection.close()
        result = LocalExecutor(path).execute(final)
        summary = ledger.summary()
        assert result == [(160,)]
        assert summary["copied_private_tuples"] == [0, 0]
        assert summary["outside_profile_serialized_tuples"] == [0, 0]
        assert summary["out_of_profile_literals"] == [0, 0]
        assert summary["full_payload_coincidences"][0] > 0
        print("Verified synthetic DG-v3 release ({} statistics).".format(len(plan["queries"])))
        print("Sandbox validation succeeded. Local answer: {}.".format(result[0][0]))
        print("Outbound audit (unique, cumulative):", summary)


if __name__ == "__main__":
    main()
