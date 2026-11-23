"""Local disclosure measurements for actually transmitted payloads, including resends."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def leaves(value):
    if isinstance(value, dict):
        for k, v in value.items():
            yield str(k)
            yield from leaves(v)
    elif isinstance(value, (tuple, list)):
        for v in value:
            yield from leaves(v)
    elif value is not None:
        yield str(value)


def segments(value):
    """Yield contiguous line, JSON array, or tuple segments from an outbound payload."""
    if isinstance(value, str):
        for line in value.splitlines():
            if line:
                yield [line]
    elif isinstance(value, (tuple, list)):
        scalar_items = [str(v) for v in value if v is not None and not isinstance(v, (dict, list, tuple))]
        if scalar_items:
            yield scalar_items
        for v in value:
            if isinstance(v, (dict, list, tuple, str)):
                yield from segments(v)
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from segments(str(k))
            yield from segments(v)


def typed(row):
    return tuple((type(v).__name__, repr(v)) for v in row)


def _strip_text(value: Any, target: str) -> Any:
    """Recursively remove occurrences of `target` from string values without mutating `value`."""
    if not target:
        return value
    if isinstance(value, str):
        return value.replace(target, "")
    if isinstance(value, list):
        return [_strip_text(v, target) for v in value]
    if isinstance(value, tuple):
        return tuple(_strip_text(v, target) for v in value)
    if isinstance(value, dict):
        return {k: _strip_text(v, target) for k, v in value.items()}
    return value


class TransmissionLedger:
    """Segment-aware outbound disclosure ledger.

    Rows are indexed by source table and local row ID. Supply raw-copy provenance explicitly
    via `copied` or `copied_row_ids`. Supports segment-aware auditing (`profile_1d` vs `prompt`
    / `tool_reply`) so 1D DP profile tokens are distinguished from non-profile serialized tuples.
    """

    def __init__(self, tables, profile, public=()):
        self.rows = {(table, i): tuple(row) for table, rows in tables.items()
                     for i, row in enumerate(rows)}
        cells = {str(v) for row in self.rows.values() for v in row if v is not None}
        self.patterns = {v: re.compile(r"(?<!\w)" + re.escape(v) + r"(?!\w)") for v in cells if v}
        self.public = self.matches(public)
        self.profile_literals = self.matches(profile)
        self.exempt = self.public | self.profile_literals
        self._auto_profile_texts: Tuple[str, ...] = ()
        if isinstance(profile, dict) and profile:
            try:
                c1 = json.dumps(profile, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                c2 = json.dumps(profile, ensure_ascii=False)
                self._auto_profile_texts = tuple(dict.fromkeys([c1, c2]))
            except (TypeError, ValueError):
                self._auto_profile_texts = ()

        self.copied = set()
        self.coincidences = set()
        self.outside_serialized_tuples = set()
        self.outside = set()
        self.copy_count = 0
        self.match_count = 0
        self.outside_serialized_count = 0
        self.cell_count = 0

    def matches(self, payload):
        strings = list(leaves(payload))
        return {v for v, pattern in self.patterns.items()
                if any(pattern.search(s) for s in strings)}

    def record(
        self,
        payload: Any = None,
        copied: Sequence[Tuple[str, int, int]] = (),
        *,
        profile_1d_text: Optional[str] = None,
        segments: Optional[Sequence[Dict[str, Any]]] = None,
        copied_row_ids: Sequence[Any] = (),
    ) -> Dict[str, List[int]]:
        """Record an outbound transmission and return updated Table 3 metrics.

        Args:
            payload: Optional arbitrary outbound JSON/dict/list/str payload.
            copied: Sequence of `(table, row_id, payload_row_idx)` provenance tuples.
            profile_1d_text: Optional serialized 1D DP profile substring to classify as
                `profile_1d` when scanning `payload`.
            segments: Optional list of `{"kind": "profile_1d"|"prompt"|"tool_reply", "text": ...}`.
            copied_row_ids: Optional sequence of `(table, row_id)` provenance pairs for
                segment-based calls without `payload["rows"]`.
        """
        extract_segs = globals()["segments"]

        direct = set()
        for table, index, position in copied:
            origin = (table, index)
            if typed(self.rows[origin]) != typed(payload["rows"][position]):
                raise ValueError("Provenance does not match transmitted row")
            direct.add(origin)
            self.copied.add((table, typed(self.rows[origin])))
            self.copy_count += 1

        for item in copied_row_ids:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                origin = (item[0], item[1])
            else:
                raise ValueError("copied_row_ids entries must be (table, row_id) pairs")
            if origin not in self.rows:
                raise KeyError(f"Unknown copied row provenance: {origin!r}")
            direct.add(origin)
            self.copied.add((origin[0], typed(self.rows[origin])))
            self.copy_count += 1

        full_strings: List[str] = []
        if payload is not None:
            full_strings.extend(leaves(payload))

        non_profile_segs: List[List[str]] = []
        if segments is not None:
            for seg in segments:
                txt = seg.get("text")
                if txt is None:
                    continue
                s_txt = str(txt)
                full_strings.append(s_txt)
                if seg.get("kind") != "profile_1d":
                    non_profile_segs.extend(extract_segs(s_txt))
            if payload is not None:
                stripped = _strip_text(payload, profile_1d_text) if profile_1d_text else payload
                non_profile_segs.extend(extract_segs(stripped))
        elif payload is not None:
            stripped = payload
            if profile_1d_text is not None:
                stripped = _strip_text(stripped, profile_1d_text)
            else:
                for ptxt in self._auto_profile_texts:
                    stripped = _strip_text(stripped, ptxt)
            non_profile_segs.extend(extract_segs(stripped))

        hits = {
            v for v, pattern in self.patterns.items()
            if any(pattern.search(s) for s in full_strings)
        } - self.public

        non_profile_hits = {
            v for v, pattern in self.patterns.items()
            if any(pattern.search(s) for seg in non_profile_segs for s in seg)
        } - self.public

        if len(hits) >= 2:
            for key, row in self.rows.items():
                if key not in direct:
                    if sum(str(v) in hits for v in row if v is not None) >= 2:
                        self.coincidences.add(key)
                        self.match_count += 1

        if len(non_profile_hits) >= 2:
            segment_hit_sets = []
            for seg in non_profile_segs:
                seg_hits = {v for v in non_profile_hits if any(self.patterns[v].search(s) for s in seg)}
                if len(seg_hits) >= 2:
                    segment_hit_sets.append(seg_hits)
            if segment_hit_sets:
                for key, row in self.rows.items():
                    if key not in direct:
                        if any(sum(str(v) in sh for v in row if v is not None) >= 2 for sh in segment_hit_sets):
                            self.outside_serialized_tuples.add(key)
                            self.outside_serialized_count += 1

        outside = hits - self.exempt
        self.outside.update(outside)
        self.cell_count += len(outside)
        return self.summary()

    def summary(self) -> Dict[str, List[int]]:
        copied_stat = [len(self.copied), self.copy_count]
        outside_ser_stat = [len(self.outside_serialized_tuples), self.outside_serialized_count]
        full_coinc_stat = [len(self.coincidences), self.match_count]
        oob_stat = [len(self.outside), self.cell_count]
        return {
            "copied_private_tuples": copied_stat,
            "outside_profile_serialized_tuples": outside_ser_stat,
            "full_payload_coincidences": full_coinc_stat,
            "out_of_profile_literals": oob_stat,
            # Backward-compatible aliases:
            "multi_column_matches": list(full_coinc_stat),
            "serialized_tuple_matches": list(outside_ser_stat),
        }
