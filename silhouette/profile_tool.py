"""On-demand read-only tool wrapper over frozen 1-way DP 5-field column profiles."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

PROFILE_TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_table_profile",
            "description": "Retrieve the frozen 1-way DP 5-field column profile for all columns of a table.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {"type": "string", "description": "Table name from the public schema."},
                },
                "required": ["table_name"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_column_profile",
            "description": "Retrieve the frozen 1-way DP 5-field profile for a single (table_name, column_name).",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {"type": "string", "description": "Table name from the public schema."},
                    "column_name": {"type": "string", "description": "Column name in table_name."},
                },
                "required": ["table_name", "column_name"],
                "additionalProperties": False,
            },
        },
    },
]


class ProfileLookupTool:
    """Read-only on-demand tool accessor over a frozen 1-way DP JSON profile."""

    def __init__(
        self,
        profile_or_path: Union[Path, str, Dict[str, Dict[str, List[Any]]]],
        expected_sha256: Optional[str] = None,
    ) -> None:
        if isinstance(profile_or_path, dict):
            self.profile_path: Optional[Path] = None
            raw = json.dumps(
                profile_or_path,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            self.sha256 = hashlib.sha256(raw).hexdigest()
            self.profile: Dict[str, Dict[str, List[Any]]] = profile_or_path
        else:
            self.profile_path = Path(profile_or_path)
            raw = self.profile_path.read_bytes()
            self.sha256 = hashlib.sha256(raw).hexdigest()
            self.profile = json.loads(raw.decode("utf-8"))

        if expected_sha256 is not None and self.sha256 != expected_sha256.lower():
            raise ValueError(
                f"Profile SHA-256 mismatch for {self.profile_path or '<dict>'}: "
                f"expected={expected_sha256}, actual={self.sha256}"
            )
        self._table_map = {
            t.lower(): t for t, cols in self.profile.items() if isinstance(cols, dict)
        }
        self._col_maps = {
            t: {c.lower(): c for c in cols}
            for t, cols in self.profile.items()
            if isinstance(cols, dict)
        }

    def list_tables(self) -> List[str]:
        return sorted(self.profile)

    def get_table_profile(self, table_name: str) -> Dict[str, Any]:
        canon_t = self._table_map.get((table_name or "").strip().lower())
        if canon_t is None:
            return {
                "status": "error",
                "category": "unknown_table",
                "message": f"Unknown table {table_name!r}. Available tables: {self.list_tables()}.",
            }
        return {
            "status": "ok",
            "table": canon_t,
            "columns": self.profile[canon_t],
            "source_sha256": self.sha256,
        }

    def get_column_profile(self, table_name: str, column_name: str) -> Dict[str, Any]:
        canon_t = self._table_map.get((table_name or "").strip().lower())
        if canon_t is None:
            return {
                "status": "error",
                "category": "unknown_table",
                "message": f"Unknown table {table_name!r}. Available tables: {self.list_tables()}.",
            }
        canon_c = self._col_maps[canon_t].get((column_name or "").strip().lower())
        if canon_c is None:
            return {
                "status": "error",
                "category": "unknown_column",
                "message": f"Unknown column {column_name!r} in table {canon_t!r}.",
            }
        return {
            "status": "ok",
            "table": canon_t,
            "column": canon_c,
            "profile": self.profile[canon_t][canon_c],
            "source_sha256": self.sha256,
        }


IND8ProfileTool = ProfileLookupTool
