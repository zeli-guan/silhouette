"""SQLite capabilities for schema validation, fixed public lookups and local execution."""
from __future__ import annotations

from collections import defaultdict
from contextlib import closing
import hashlib
from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Optional

from .schema_roles import quote

PUBLIC_TABLES = frozenset(("d_icd_diagnoses", "d_icd_procedures", "d_items", "d_labitems"))
K = 50
FUNCTIONS = frozenset("""
    abs acos asin atan atan2 avg ceil ceiling char coalesce concat concat_ws cos count
    cume_dist date datetime degrees dense_rank exp first_value floor format glob
    group_concat hex ifnull iif instr json_array_length json_extract json_quote
    json_type json_valid julianday lag last_value lead length like
    likelihood likely ln log log10 log2 lower ltrim max min mod nullif ntile
    octet_length percent_rank pi pow power printf quote radians rank replace round
    row_number rtrim sign sin sqrt strftime substr substring sum tan time timediff
    total trim typeof unicode unistr unistr_quote unixepoch unlikely upper zeroblob
""".split())


def readonly(path):
    con = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    con.execute("PRAGMA query_only=ON")
    con.execute("PRAGMA temp_store=MEMORY")
    return con


def _ordinary_table_ddls(con):
    rows = con.execute("SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = []
    for name, ddl in rows:
        if name.startswith("sqlite_"):
            continue
        if not ddl or not ddl.lstrip().upper().startswith("CREATE TABLE"):
            raise ValueError("Only ordinary SQLite tables are supported")
        tables.append((name, ddl))
    return tables


def ordinary_tables(con):
    return [name for name, _ in _ordinary_table_ddls(con)]


def _strip_sql_comments_and_quotes(sql: str) -> str:
    """Replace SQL comments, string literals, and quoted identifiers with spaces."""
    out = []
    i = 0
    n = len(sql)
    while i < n:
        ch = sql[i]
        if ch == "-" and i + 1 < n and sql[i + 1] == "-":
            j = sql.find("\n", i + 2)
            i = n if j == -1 else j + 1
            out.append(" ")
            continue
        if ch == "/" and i + 1 < n and sql[i + 1] == "*":
            j = sql.find("*/", i + 2)
            i = n if j == -1 else j + 2
            out.append(" ")
            continue
        if ch in ("'", '"', "`"):
            q = ch
            i += 1
            while i < n:
                if sql[i] == q:
                    if i + 1 < n and sql[i + 1] == q:
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            out.append(" ")
            continue
        if ch == "[":
            j = sql.find("]", i + 1)
            i = n if j == -1 else j + 1
            out.append(" ")
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _ddl_without_rowid(ddl: str) -> bool:
    """Return True iff a CREATE TABLE DDL specifies WITHOUT ROWID in its table-options tail."""
    cleaned = _strip_sql_comments_and_quotes(ddl)
    start = cleaned.find("(")
    if start == -1:
        return False
    depth = 0
    end = -1
    for idx in range(start, len(cleaned)):
        ch = cleaned[idx]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                end = idx
                break
    if end == -1:
        return False
    tail = cleaned[end + 1 :]
    return bool(re.search(r"\bWITHOUT\s+ROWID\b", tail, re.IGNORECASE))


_EPONYMOUS_VTABS = frozenset((
    "bytecode", "completion", "dbpage", "dbstat", "fts3_tokenizer", "fts5vocab",
    "generate_series", "geopoly", "json_each", "json_tree", "rtree", "tables_used",
))


def restrict(con, allowed, allow_cte: bool = True):
    persistent = con.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('view', 'trigger')"
    ).fetchall()
    persistent_objects = frozenset(r[0].lower() for r in persistent)
    names_lower = frozenset(n.lower() for n in allowed)

    def authorize(action, first, second, db, origin):
        if allow_cte:
            origin_ok = (origin is None) or (
                isinstance(origin, str) and origin.lower() not in persistent_objects
            )
            # Materialized CTEs (e.g. WITH q AS MATERIALIZED (...) SELECT q.col / COUNT(*) FROM q)
            # trigger SQLITE_READ(first='q', second='' or 'col', db=None, origin=None) in SQLite.
            ephemeral_cte_read = (
                action == sqlite3.SQLITE_READ
                and isinstance(first, str)
                and db is None
                and not first.lower().startswith(("sqlite_", "pragma_"))
                and first.lower() not in persistent_objects
                and first.lower() not in _EPONYMOUS_VTABS
            )
        else:
            origin_ok = origin is None
            ephemeral_cte_read = False
        ok = (
            action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_RECURSIVE)
            or (
                action == sqlite3.SQLITE_READ
                and isinstance(first, str)
                and first.lower() in names_lower
                and db in ("main", None)
            )
            or ephemeral_cte_read
            or (
                action == sqlite3.SQLITE_FUNCTION
                and isinstance(second, str)
                and second.lower() in FUNCTIONS
            )
        )
        return sqlite3.SQLITE_OK if (ok and origin_ok) else sqlite3.SQLITE_DENY

    con.set_authorizer(authorize)


def restrict_legacy_no_cte(con, allowed):
    """Install legacy authorizer rejecting non-None CTE origins (`origin is None`)."""
    return restrict(con, allowed, allow_cte=False)


def error_category(error):
    msg = str(error).lower()
    for text, category in (("no such table", "unknown_table"), ("no such column", "unknown_column"),
                           ("ambiguous", "ambiguous_column"), ("syntax", "syntax_error"),
                           ("interrupted", "timeout"), ("no such function", "unknown_function"),
                           ("authorized", "read_only_denied"), ("prohibited", "read_only_denied")):
        if text in msg:
            return category
    return "execution_error"


def run(con, sql, preview=False):
    deadline = time.monotonic() + 30
    con.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
    try:
        cur = con.execute(sql)
        if cur.description is None:
            return {"status": "error", "category": "select_required"}
        if not preview:
            cur.fetchmany(1)
            return {"status": "success"}
        rows = cur.fetchmany(K + 1)
        return {"status": "success", "columns": [c[0] for c in cur.description],
                "rows": [[{"blob_hex": v.hex()} if isinstance(v, bytes) else v for v in row]
                         for row in rows[:K]], "truncated": len(rows) > K}
    except (sqlite3.Error, sqlite3.Warning, ValueError, OverflowError) as exc:
        return {"status": "error", "category": error_category(exc)}
    finally:
        con.set_progress_handler(None, 0)


class SchemaSandbox:
    def __init__(self, source, *, allow_cte: bool = True):
        self.connection = sqlite3.connect(":memory:", check_same_thread=False)
        self._lock = threading.Lock()
        table_ddls = _ordinary_table_ddls(source)
        tables = [t for t, _ in table_ddls]
        for t, ddl in table_ddls:
            cols = source.execute("PRAGMA table_info(" + quote(t) + ")").fetchall()
            parts = [quote(c[1]) + (" " + quote(c[2]) if c[2] else "") for c in cols]
            pk = sorted((c for c in cols if c[5]), key=lambda c: c[5])
            if pk:
                parts.append("PRIMARY KEY (" + ",".join(quote(c[1]) for c in pk) + ")")
            groups = defaultdict(list)
            for f in source.execute("PRAGMA foreign_key_list(" + quote(t) + ")"):
                groups[f[0]].append(f)
            for group in groups.values():
                fs = sorted(group, key=lambda f: f[1])
                clause = ("FOREIGN KEY (" + ",".join(quote(f[3]) for f in fs)
                          + ") REFERENCES " + quote(fs[0][2]))
                if all(f[4] is not None for f in fs):
                    clause += " (" + ",".join(quote(f[4]) for f in fs) + ")"
                parts.append(clause)
            suffix = " WITHOUT ROWID" if (pk and _ddl_without_rowid(ddl)) else ""
            self.connection.execute(
                "CREATE TABLE " + quote(t) + " (" + ",".join(parts) + ")" + suffix
            )
            if self.connection.execute("SELECT count(*) FROM " + quote(t)).fetchone() != (0,):
                raise RuntimeError("Schema projection contains rows")
        self.connection.execute("PRAGMA query_only=ON")
        restrict(self.connection, tables, allow_cte=allow_cte)

    def try_sql(self, sql):
        with self._lock:
            return run(self.connection, sql)


class PublicLookup:
    """Read-only lookup handle over an isolated 4-table public dictionary SQLite snapshot.

    SHA-256 verification pins dictionary artifact byte integrity against accidental drift;
    dictionary independence (A1), non-answer-conditioned querying (A4), and single-release
    receiver isolation (A9) are operational deployment invariants.
    """

    def __init__(
        self,
        dictionary_path,
        expected_sha256: Optional[str] = None,
        *,
        allow_unpinned: bool = False,
    ):
        dict_path = Path(dictionary_path)
        self.dictionary_path = dict_path
        self.file_sha256 = hashlib.sha256(dict_path.read_bytes()).hexdigest()
        self.verified_sha256: Optional[str] = None
        if expected_sha256 is None:
            if not allow_unpinned:
                raise ValueError("expected_sha256 is required unless allow_unpinned=True")
        else:
            if (
                not isinstance(expected_sha256, str)
                or not expected_sha256.strip()
                or self.file_sha256 != expected_sha256.strip().lower()
            ):
                raise ValueError(
                    f"Public dictionary SHA-256 mismatch: expected={expected_sha256}, actual={self.file_sha256}"
                )
            self.verified_sha256 = self.file_sha256
        self.connection = readonly(dict_path)
        objects = self.connection.execute(
            "SELECT name, type FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'").fetchall()
        tables = set(ordinary_tables(self.connection))
        if tables != PUBLIC_TABLES or any(kind not in ("table", "index") for _, kind in objects):
            self.connection.close()
            raise ValueError("Expected an isolated four-table public dictionary snapshot")
        restrict(self.connection, PUBLIC_TABLES)

    def lookup_reference(self, sql):
        return run(self.connection, sql, preview=True)


class LocalExecutor:
    """Caller receives final rows locally after the last model call."""
    def __init__(self, path):
        self.path = path

    def execute(self, sql, timeout_s: float = 60.0, allow_cte: bool = True):
        with closing(readonly(self.path)) as conn:
            restrict(conn, ordinary_tables(conn), allow_cte=allow_cte)
            if timeout_s is not None and timeout_s > 0:
                deadline = time.monotonic() + float(timeout_s)
                conn.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
            try:
                cur = conn.execute(sql)
                if cur.description is None:
                    raise ValueError("Expected a read-only result")
                return cur.fetchall()
            finally:
                conn.set_progress_handler(None, 0)

    def execute_preview(self, sql, max_rows: int = 1000, timeout_s: float = 60.0, allow_cte: bool = True) -> dict:
        if int(max_rows) < 0:
            raise ValueError("max_rows must be >= 0")
        limit = int(max_rows)
        with closing(readonly(self.path)) as conn:
            restrict(conn, ordinary_tables(conn), allow_cte=allow_cte)
            if timeout_s is not None and timeout_s > 0:
                deadline = time.monotonic() + float(timeout_s)
                conn.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
            try:
                cur = conn.execute(sql)
                if cur.description is None:
                    raise ValueError("Expected a read-only result")
                rows = cur.fetchmany(limit + 1)
                return {
                    "columns": [c[0] for c in cur.description],
                    "rows": rows[:limit],
                    "truncated": len(rows) > limit,
                }
            finally:
                conn.set_progress_handler(None, 0)
