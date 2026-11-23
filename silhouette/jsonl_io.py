"""Transparent reader utilities for plain `.jsonl` and gzip-compressed `.jsonl.gz` artifacts."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterator, List, TextIO, Union


def resolve_jsonl_path(path: Union[str, Path]) -> Path:
    """Return `path` if present, otherwise fall back to `.jsonl.gz` (or `.jsonl`)."""
    p = Path(path)
    if p.exists():
        return p
    gz = Path(f"{p}.gz")
    if gz.exists():
        return gz
    if p.suffix == ".gz":
        plain = p.with_suffix("")
        if plain.exists():
            return plain
    return p


def open_jsonl(path: Union[str, Path]) -> TextIO:
    """Open a `.jsonl` or `.jsonl.gz` file as UTF-8 text (`path` first, `.gz` fallback)."""
    resolved = resolve_jsonl_path(path)
    if resolved.suffix == ".gz":
        return gzip.open(resolved, "rt", encoding="utf-8")
    return resolved.open("r", encoding="utf-8")


def iter_jsonl(path: Union[str, Path]) -> Iterator[dict[str, Any]]:
    """Yield parsed JSON objects from a `.jsonl` or `.jsonl.gz` file, skipping blank lines."""
    with open_jsonl(path) as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def load_jsonl(path: Union[str, Path]) -> List[dict[str, Any]]:
    """Load all parsed JSON objects from a `.jsonl` or `.jsonl.gz` file."""
    return list(iter_jsonl(path))
