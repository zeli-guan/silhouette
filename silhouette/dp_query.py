"""Interactive Differentially Private single-table aggregate query engines (Row-DP & Direct-Star Entity-DP).

Self-contained pure-stdlib AST validator, public SchemaSandbox pre-validator, normalized-SQL
intra-stream cache, pure-Python mathematically total per-unit evaluator (zero wall-clock
timeouts, zero table-level catch-all fallbacks, zero SQLite UDF side-effects), and exact
Discrete Gaussian (CKS20) response engines:
  1. DPQueryRowEngine: Table-parallel composition across tables under single-row adjacency
     (~_db-1row or ~_all-row), with per-table (epsilon_T=8, delta_T=1e-6) split across
     budget_slots = max_probes_per_q * q_plan.
  2. DPQueryRowDBWideEngine: Database-wide sequential composition across all tables under
     single-row database adjacency ~_db-1row (adding/removing 1 row in 1 table of D;
     distinguished from ~_all-row where m tables each change by 1 row), with DB-wide
     (epsilon_db=8, delta_db=1e-6) split across max_db_probes = max_probes_per_q * q_plan.
  3. DPQueryEntityEngine: Database-wide composition across entity_direct tables under
     direct-star entity adjacency (CAP=5), rejecting schema_only tables at 0 data access.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
import copy
from fractions import Fraction
import math
from pathlib import Path
import re
import sqlite3
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from . import dg_sampler as dg
from . import entity_dp_publisher as edp
from . import profile as p
from . import sandbox as sb
from . import schema_roles as v3
from .schema_roles import quote

INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1

ALLOWED_SQLITE_FUNCTIONS = frozenset({
    "abs", "coalesce", "glob", "hex", "ifnull", "iif", "instr", "length",
    "like", "lower", "ltrim", "nullif", "printf", "replace", "round",
    "rtrim", "sign", "substr", "substring", "trim", "typeof", "unicode", "upper",
})

BLOCKED_SQL_KEYWORDS = frozenset({
    "ATTACH", "ALTER", "BEGIN", "COMMIT", "CREATE", "CROSS", "DELETE", "DETACH",
    "DROP", "EXCEPT", "EXPLAIN", "HAVING", "INNER", "INSERT", "INTERSECT",
    "JOIN", "LEFT", "NATURAL", "OUTER", "OVER", "PRAGMA", "REINDEX",
    "RELEASE", "REPLACE", "RIGHT", "ROLLBACK", "SAVEPOINT", "UNION",
    "UPDATE", "VACUUM", "VALUES", "WINDOW", "WITH",
})

_EXEC_FUNC_REMAP = {
    "abs": "DP_ABS",
    "printf": "DP_PRINTF",
}

_TOKEN_RE = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<comment>--[^\n]*|/\*[\s\S]*?\*/)
  | (?P<squote>'(?:''|[^'])*')
  | (?P<dquote>"(?:""|[^"])*")
  | (?P<bquote>`(?:``|[^`])*`)
  | (?P<bracket>\[[^\]\n]*\])
  | (?P<op><>|!=|<=|>=|==|\|\||[=<>(),.;*+\-/%])
  | (?P<num>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)
  | (?P<word>[A-Za-z_][A-Za-z0-9_$]*)
    """,
    re.VERBOSE,
)

_SQLITE_NUM_PREFIX_RE = re.compile(r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")


def _dp_safe_abs(val: Any) -> Any:
    """Total exception-free SQLite ABS() clamping INT64_MIN to INT64_MAX."""
    if val is None:
        return None
    if isinstance(val, int):
        if val <= INT64_MIN:
            return INT64_MAX
        return abs(val)
    if isinstance(val, float):
        return abs(val)
    if isinstance(val, bytes):
        val = val.decode("utf-8", "replace")
    if isinstance(val, str):
        m = _SQLITE_NUM_PREFIX_RE.match(val)
        if not m:
            return 0.0
        try:
            f = float(m.group(1))
            return abs(f) if math.isfinite(f) else 0.0
        except (ValueError, OverflowError):
            return 0.0
    return 0.0


def _dp_safe_printf(*args: Any) -> Optional[str]:
    """Total exception-free SQLite printf() wrapper returning None on malformed format/args."""
    if not args or args[0] is None:
        return None
    fmt = str(args[0])
    # Convert C-style %s/%d/%f formatting safely on best-effort basis; return None on any error.
    try:
        tokens = list(re.finditer(r"%(%|[-+ #0]*\d*(?:\.\d+)?[diuoxXfFeEgGcs])", fmt))
        out: List[str] = []
        pos = 0
        arg_idx = 1
        for m in tokens:
            out.append(fmt[pos : m.start()])
            pos = m.end()
            spec = m.group(1)
            if spec == "%":
                out.append("%")
                continue
            if arg_idx >= len(args):
                val = 0 if spec[-1] in "diuoxXfFeEgG" else ""
            else:
                val = args[arg_idx]
                arg_idx += 1
            if val is None:
                val = 0 if spec[-1] in "diuoxXfFeEgG" else ""
            conv = spec[-1]
            if conv in "diuoxX":
                out.append(format(int(float(val)), "d" if conv in "diu" else conv))
            elif conv in "fFeEgG":
                out.append(format(float(val), conv))
            elif conv == "c":
                s_val = str(val)
                out.append(s_val[:1])
            else:
                out.append(str(val))
        out.append(fmt[pos:])
        return "".join(out)
    except Exception:
        return None


def _coerce_numeric(val: Any) -> Optional[float]:
    """Convert SQLite scalar to finite float under SQLite numeric coercion, or None."""
    if val is None or isinstance(val, bytes):
        return None
    if isinstance(val, (int, float)):
        f = float(val)
        return f if math.isfinite(f) else None
    if isinstance(val, str):
        m = _SQLITE_NUM_PREFIX_RE.match(val)
        if not m:
            return 0.0
        try:
            f = float(m.group(1))
            return f if math.isfinite(f) else None
        except (ValueError, OverflowError):
            return None
    return None


def _clamp_scaled_int(val: Any, clip_lo: int, clip_hi: int, scale: int) -> Optional[int]:
    """Clamp numeric cell to [clip_lo, clip_hi] and round to integer * scale."""
    num = _coerce_numeric(val)
    if num is None:
        return None
    if num < clip_lo:
        num = float(clip_lo)
    elif num > clip_hi:
        num = float(clip_hi)
    return int(round(num * scale))


class DPQueryValidationError(ValueError):
    """Raised when an interactive SQL probe violates single-table DP AST constraints."""

    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category
        self.message = message


def _tokenize(sql: str) -> List[Tuple[str, str]]:
    if not isinstance(sql, str) or not sql.strip():
        raise DPQueryValidationError("empty_sql", "SQL query must be a non-empty string.")
    tokens: List[Tuple[str, str]] = []
    pos = 0
    n = len(sql)
    while pos < n:
        m = _TOKEN_RE.match(sql, pos)
        if m is None:
            raise DPQueryValidationError(
                "syntax_error",
                f"Unsupported syntax near character {pos}: {sql[pos:pos+20]!r}.",
            )
        kind = m.lastgroup
        val = m.group(0)
        if kind == "comment":
            raise DPQueryValidationError("comments_disallowed", "SQL comments are not permitted in DP probes.")
        if kind != "ws":
            tokens.append((kind, val))
        pos = m.end()
    if tokens and tokens[-1] == ("op", ";"):
        tokens.pop()
    for kind, val in tokens:
        if kind == "op" and val == ";":
            raise DPQueryValidationError("multi_statement", "Multiple SQL statements are not permitted.")
        if kind == "word" and val.upper() in BLOCKED_SQL_KEYWORDS:
            up = val.upper()
            if up in ("JOIN", "INNER", "LEFT", "RIGHT", "OUTER", "CROSS", "NATURAL"):
                raise DPQueryValidationError(
                    "join_disallowed",
                    "Multi-table JOINs are not allowed in DP aggregate probes; query one table at a time.",
                )
            raise DPQueryValidationError(
                "disallowed_keyword",
                f"Keyword {up} is not permitted in single-table DP aggregate probes.",
            )
    return tokens


def _unquote_ident(kind: str, val: str) -> str:
    if kind == "dquote":
        return val[1:-1].replace('""', '"')
    if kind == "bquote":
        return val[1:-1].replace("``", "`")
    if kind == "bracket":
        return val[1:-1]
    return val


def _is_ident(tok: Tuple[str, str]) -> bool:
    return tok[0] in ("word", "dquote", "bquote", "bracket")


def _kw(tok: Optional[Tuple[str, str]], word: str) -> bool:
    return tok is not None and tok[0] == "word" and tok[1].upper() == word


class _WhereParser:
    """Recursive-descent parser, canonical normalizer, and pure-Python AST builder for WHERE predicates."""

    def __init__(
        self,
        tokens: List[Tuple[str, str]],
        table_name: str,
        col_map_lower: Dict[str, str],
        func_remap: Optional[Dict[str, str]] = None,
    ) -> None:
        self.tokens = tokens
        self.pos = 0
        self.table_name = table_name
        self.table_lower = table_name.lower()
        self.col_map = col_map_lower
        self.func_remap = func_remap or {}
        self.referenced_cols: Set[str] = set()

    def peek(self, offset: int = 0) -> Optional[Tuple[str, str]]:
        idx = self.pos + offset
        return self.tokens[idx] if 0 <= idx < len(self.tokens) else None

    def advance(self) -> Tuple[str, str]:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def match_kw(self, word: str) -> bool:
        if _kw(self.peek(), word):
            self.pos += 1
            return True
        return False

    def match_op(self, op: str) -> bool:
        t = self.peek()
        if t is not None and t == ("op", op):
            self.pos += 1
            return True
        return False

    def parse(self) -> str:
        sql, _ast = self.parse_with_ast()
        return sql

    def parse_with_ast(self) -> Tuple[str, Tuple[Any, ...]]:
        sql, ast_node = self.parse_or()
        if self.pos != len(self.tokens):
            raise DPQueryValidationError(
                "invalid_where",
                f"Unexpected token {self.tokens[self.pos][1]!r} in WHERE clause.",
            )
        return sql, ast_node

    def parse_or(self) -> Tuple[str, Tuple[Any, ...]]:
        sql0, ast0 = self.parse_and()
        sql_parts = [sql0]
        ast_parts = [ast0]
        while self.match_kw("OR"):
            s, a = self.parse_and()
            sql_parts.append(s)
            ast_parts.append(a)
        if len(sql_parts) == 1:
            return sql0, ast0
        return "(" + " OR ".join(sql_parts) + ")", ("or", tuple(ast_parts))

    def parse_and(self) -> Tuple[str, Tuple[Any, ...]]:
        sql0, ast0 = self.parse_not()
        sql_parts = [sql0]
        ast_parts = [ast0]
        while self.match_kw("AND"):
            s, a = self.parse_not()
            sql_parts.append(s)
            ast_parts.append(a)
        if len(sql_parts) == 1:
            return sql0, ast0
        return "(" + " AND ".join(sql_parts) + ")", ("and", tuple(ast_parts))

    def parse_not(self) -> Tuple[str, Tuple[Any, ...]]:
        if self.match_kw("NOT"):
            s, a = self.parse_not()
            return f"(NOT {s})", ("not", a)
        return self.parse_comparison()

    def parse_comparison(self) -> Tuple[str, Tuple[Any, ...]]:
        lhs_sql, lhs_ast = self.parse_concat()
        t = self.peek()
        if t is None:
            return lhs_sql, lhs_ast

        # IS [NOT] NULL / IS [NOT] expr
        if self.match_kw("IS"):
            neg = self.match_kw("NOT")
            if self.match_kw("NULL"):
                return f"({lhs_sql} IS {'NOT ' if neg else ''}NULL)", ("is_null", lhs_ast, neg)
            rhs_sql, rhs_ast = self.parse_concat()
            return f"({lhs_sql} IS {'NOT ' if neg else ''}{rhs_sql})", ("is_expr", lhs_ast, rhs_ast, neg)

        # [NOT] IN (...) / BETWEEN ... AND ... / LIKE ... / GLOB ...
        neg = False
        if _kw(t, "NOT"):
            nxt = self.peek(1)
            if nxt is not None and nxt[0] == "word" and nxt[1].upper() in ("IN", "BETWEEN", "LIKE", "GLOB"):
                self.advance()
                neg = True

        if self.match_kw("IN"):
            if not self.match_op("("):
                raise DPQueryValidationError("invalid_where", "IN operator requires a parenthesized literal list.")
            if _kw(self.peek(), "SELECT"):
                raise DPQueryValidationError("subquery_disallowed", "Subqueries are not allowed in DP WHERE predicates.")
            sql_items: List[str] = []
            val_items: List[Any] = []
            if not self.match_op(")"):
                s0, v0 = self._parse_literal_or_signed_num()
                sql_items.append(s0)
                val_items.append(v0)
                while self.match_op(","):
                    si, vi = self._parse_literal_or_signed_num()
                    sql_items.append(si)
                    val_items.append(vi)
                if not self.match_op(")"):
                    raise DPQueryValidationError("invalid_where", "Expected ')' closing IN list.")
            return (
                f"({lhs_sql} {'NOT ' if neg else ''}IN ({', '.join(sql_items)}))",
                ("in", lhs_ast, tuple(val_items), neg),
            )

        if self.match_kw("BETWEEN"):
            lo_sql, lo_ast = self.parse_concat()
            if not self.match_kw("AND"):
                raise DPQueryValidationError("invalid_where", "BETWEEN requires AND.")
            hi_sql, hi_ast = self.parse_concat()
            return (
                f"({lhs_sql} {'NOT ' if neg else ''}BETWEEN {lo_sql} AND {hi_sql})",
                ("between", lhs_ast, lo_ast, hi_ast, neg),
            )

        if self.match_kw("LIKE"):
            pat_sql, pat_ast = self.parse_concat()
            return f"({lhs_sql} {'NOT ' if neg else ''}LIKE {pat_sql})", ("like", lhs_ast, pat_ast, neg)

        if self.match_kw("GLOB"):
            pat_sql, pat_ast = self.parse_concat()
            return f"({lhs_sql} {'NOT ' if neg else ''}GLOB {pat_sql})", ("glob", lhs_ast, pat_ast, neg)

        if neg:
            raise DPQueryValidationError("invalid_where", "Unexpected NOT in comparison.")

        t = self.peek()
        if t is not None and t[0] == "op" and t[1] in ("=", "==", "!=", "<>", "<", "<=", ">", ">="):
            op = self.advance()[1]
            if op == "==":
                op = "="
            elif op == "!=":
                op = "<>"
            rhs_sql, rhs_ast = self.parse_concat()
            return f"({lhs_sql} {op} {rhs_sql})", ("cmp", op, lhs_ast, rhs_ast)
        return lhs_sql, lhs_ast

    def _parse_literal_or_signed_num(self) -> Tuple[str, Any]:
        t = self.peek()
        if t is None:
            raise DPQueryValidationError("invalid_where", "Unexpected end of IN list.")
        if t == ("op", "-") or t == ("op", "+"):
            sign = self.advance()[1]
            nt = self.peek()
            if nt is None or nt[0] != "num":
                raise DPQueryValidationError("invalid_where", "Expected numeric literal after sign in IN list.")
            val = self.advance()[1]
            num_val = _parse_num_literal(f"{sign}{val}")
            return (f"{sign}{val}" if sign == "-" else val), num_val
        if t[0] == "squote":
            tok = self.advance()[1]
            return tok, tok[1:-1].replace("''", "'")
        if t[0] == "num":
            tok = self.advance()[1]
            return tok, _parse_num_literal(tok)
        if _kw(t, "NULL"):
            return self.advance()[1].upper(), None
        if _kw(t, "TRUE"):
            return self.advance()[1].upper(), 1
        if _kw(t, "FALSE"):
            return self.advance()[1].upper(), 0
        raise DPQueryValidationError("invalid_where", "IN list may only contain scalar literals.")

    def parse_concat(self) -> Tuple[str, Tuple[Any, ...]]:
        s0, a0 = self.parse_add()
        sql_parts = [s0]
        ast_parts = [a0]
        while self.match_op("||"):
            s, a = self.parse_add()
            sql_parts.append(s)
            ast_parts.append(a)
        if len(sql_parts) == 1:
            return s0, a0
        return "(" + " || ".join(sql_parts) + ")", ("concat", tuple(ast_parts))

    def parse_add(self) -> Tuple[str, Tuple[Any, ...]]:
        sql_expr, ast_expr = self.parse_mul()
        while True:
            t = self.peek()
            if t is not None and t[0] == "op" and t[1] in ("+", "-"):
                op = self.advance()[1]
                rs, ra = self.parse_mul()
                sql_expr = f"({sql_expr} {op} {rs})"
                ast_expr = ("binop", op, ast_expr, ra)
            else:
                break
        return sql_expr, ast_expr

    def parse_mul(self) -> Tuple[str, Tuple[Any, ...]]:
        sql_expr, ast_expr = self.parse_unary()
        while True:
            t = self.peek()
            if t is not None and t[0] == "op" and t[1] in ("*", "/", "%"):
                op = self.advance()[1]
                rs, ra = self.parse_unary()
                sql_expr = f"({sql_expr} {op} {rs})"
                ast_expr = ("binop", op, ast_expr, ra)
            else:
                break
        return sql_expr, ast_expr

    def parse_unary(self) -> Tuple[str, Tuple[Any, ...]]:
        t = self.peek()
        if t is not None and t[0] == "op" and t[1] in ("+", "-"):
            op = self.advance()[1]
            inner_sql, inner_ast = self.parse_unary()
            if op == "+":
                return inner_sql, ("unary", "+", inner_ast)
            return f"(-{inner_sql})", ("unary", "-", inner_ast)
        return self.parse_primary()

    def parse_primary(self) -> Tuple[str, Tuple[Any, ...]]:
        t = self.peek()
        if t is None:
            raise DPQueryValidationError("invalid_where", "Unexpected end of WHERE expression.")

        if self.match_op("("):
            if _kw(self.peek(), "SELECT"):
                raise DPQueryValidationError("subquery_disallowed", "Subqueries are not allowed in DP WHERE predicates.")
            inner_sql, inner_ast = self.parse_or()
            if not self.match_op(")"):
                raise DPQueryValidationError("invalid_where", "Unclosed '(' in WHERE predicate.")
            return inner_sql, inner_ast

        if t[0] == "squote":
            tok = self.advance()[1]
            return tok, ("lit", tok[1:-1].replace("''", "'"))
        if t[0] == "num":
            tok = self.advance()[1]
            return tok, ("lit", _parse_num_literal(tok))
        if _kw(t, "NULL"):
            return self.advance()[1].upper(), ("lit", None)
        if _kw(t, "TRUE"):
            return self.advance()[1].upper(), ("lit", 1)
        if _kw(t, "FALSE"):
            return self.advance()[1].upper(), ("lit", 0)
        if _kw(t, "SELECT"):
            raise DPQueryValidationError("subquery_disallowed", "Subqueries are not allowed in DP WHERE predicates.")

        # Function call or column reference
        if _is_ident(t):
            first = self.advance()
            # Function call: word followed by '('
            if first[0] == "word" and self.match_op("("):
                fname = first[1].lower()
                if fname == "cast":
                    expr_sql, expr_ast = self.parse_concat()
                    if not self.match_kw("AS"):
                        raise DPQueryValidationError("invalid_where", "CAST requires AS <type>.")
                    type_tok = self.peek()
                    if type_tok is None or type_tok[0] != "word":
                        raise DPQueryValidationError("invalid_where", "CAST requires valid target type name.")
                    tname = self.advance()[1].upper()
                    if not self.match_op(")"):
                        raise DPQueryValidationError("invalid_where", "Expected ')' closing CAST.")
                    return f"CAST({expr_sql} AS {tname})", ("cast", expr_ast, tname)
                if fname not in ALLOWED_SQLITE_FUNCTIONS:
                    raise DPQueryValidationError(
                        "disallowed_function",
                        f"Function {fname}() is not in the deterministic allowlist for DP probes.",
                    )
                arg_sqls: List[str] = []
                arg_asts: List[Tuple[Any, ...]] = []
                if not self.match_op(")"):
                    s0, a0 = self.parse_concat()
                    arg_sqls.append(s0)
                    arg_asts.append(a0)
                    while self.match_op(","):
                        si, ai = self.parse_concat()
                        arg_sqls.append(si)
                        arg_asts.append(ai)
                    if not self.match_op(")"):
                        raise DPQueryValidationError("invalid_where", f"Expected ')' closing {fname}().")
                out_name = self.func_remap.get(fname, fname.upper())
                return f"{out_name}({', '.join(arg_sqls)})", ("func", fname, tuple(arg_asts))

            # Table-qualified column: table.col
            if self.match_op("."):
                t_ref = _unquote_ident(first[0], first[1])
                if t_ref.lower() != self.table_lower:
                    raise DPQueryValidationError(
                        "cross_table_reference",
                        f"Column qualifier {t_ref!r} does not match target table {self.table_name!r}.",
                    )
                col_tok = self.peek()
                if col_tok is None or not _is_ident(col_tok):
                    raise DPQueryValidationError("invalid_where", f"Expected column name after {t_ref}.")
                self.advance()
                col_raw = _unquote_ident(col_tok[0], col_tok[1])
            else:
                col_raw = _unquote_ident(first[0], first[1])

            col_canon = self.col_map.get(col_raw.lower())
            if col_canon is None:
                raise DPQueryValidationError(
                    "unknown_column",
                    f"Column {col_raw!r} does not exist on table {self.table_name!r}.",
                )
            self.referenced_cols.add(col_canon)
            return quote(col_canon), ("col", col_canon)

        raise DPQueryValidationError("invalid_where", f"Unexpected token {t[1]!r} in WHERE predicate.")


def _strip_optional_alias(tokens: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    if len(tokens) >= 2 and _kw(tokens[-2], "AS") and _is_ident(tokens[-1]):
        return tokens[:-2]
    return tokens


def _parse_single_col_ref(
    tokens: List[Tuple[str, str]],
    table_name: str,
    col_map: Dict[str, str],
) -> Optional[str]:
    if len(tokens) == 1 and _is_ident(tokens[0]):
        cname = _unquote_ident(tokens[0][0], tokens[0][1])
        return col_map.get(cname.lower())
    if len(tokens) == 3 and _is_ident(tokens[0]) and tokens[1] == ("op", ".") and _is_ident(tokens[2]):
        tname = _unquote_ident(tokens[0][0], tokens[0][1])
        if tname.lower() != table_name.lower():
            return None
        cname = _unquote_ident(tokens[2][0], tokens[2][1])
        return col_map.get(cname.lower())
    return None


def _parse_agg_expr(
    tokens: List[Tuple[str, str]],
    table_name: str,
    col_map: Dict[str, str],
) -> Tuple[str, str, Optional[str]]:
    """Parse COUNT(*)/COUNT(col)/COUNT(DISTINCT col)/SUM(col)/AVG(col).

    Returns (agg_func, agg_kind, target_col) where:
      - ('count', 'star', None)
      - ('count', 'distinct', col)
      - ('count', 'nonnull', col)
      - ('sum', 'col', col)
      - ('avg', 'col', col)
    """
    toks = _strip_optional_alias(tokens)
    if (
        len(toks) == 4
        and _kw(toks[0], "COUNT")
        and toks[1] == ("op", "(")
        and toks[2] == ("op", "*")
        and toks[3] == ("op", ")")
    ):
        return "count", "star", None
    if (
        len(toks) >= 5
        and _kw(toks[0], "COUNT")
        and toks[1] == ("op", "(")
        and _kw(toks[2], "DISTINCT")
        and toks[-1] == ("op", ")")
    ):
        col = _parse_single_col_ref(toks[3:-1], table_name, col_map)
        if col is None:
            raise DPQueryValidationError(
                "invalid_aggregate",
                "COUNT(DISTINCT ...) must reference a single column of the target table.",
            )
        return "count", "distinct", col
    if (
        len(toks) >= 4
        and toks[0][0] == "word"
        and toks[0][1].upper() in ("COUNT", "SUM", "AVG")
        and toks[1] == ("op", "(")
        and toks[-1] == ("op", ")")
    ):
        fname = toks[0][1].lower()
        col = _parse_single_col_ref(toks[2:-1], table_name, col_map)
        if col is None:
            raise DPQueryValidationError(
                "invalid_aggregate",
                f"{fname.upper()}(...) must reference a single column of the target table.",
            )
        kind = "nonnull" if fname == "count" else "col"
        return fname, kind, col
    raise DPQueryValidationError(
        "non_aggregate_select",
        "Only aggregate probes SELECT COUNT/SUM/AVG(...) or SELECT col, COUNT/SUM/AVG(...) ... GROUP BY col are permitted.",
    )


def _split_top_level_commas(tokens: List[Tuple[str, str]]) -> List[List[Tuple[str, str]]]:
    parts: List[List[Tuple[str, str]]] = []
    cur: List[Tuple[str, str]] = []
    depth = 0
    for tok in tokens:
        if tok == ("op", "("):
            depth += 1
        elif tok == ("op", ")"):
            depth -= 1
            if depth < 0:
                raise DPQueryValidationError("syntax_error", "Unmatched ')' in SELECT projection.")
        if tok == ("op", ",") and depth == 0:
            parts.append(cur)
            cur = []
        else:
            cur.append(tok)
    if depth != 0:
        raise DPQueryValidationError("syntax_error", "Unclosed '(' in SELECT projection.")
    parts.append(cur)
    return parts


def parse_and_normalize_probe(
    sql: str,
    public_schema: v3.PublicSchema,
    require_distinct_col_by_table: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Validate single-table aggregate SQL probe and return canonical normalized descriptor."""
    tokens = _tokenize(sql)
    if not _kw(tokens[0], "SELECT"):
        raise DPQueryValidationError("select_required", "DP probes must be SELECT aggregate statements.")
    if any(_kw(tok, "SELECT") for tok in tokens[1:]):
        raise DPQueryValidationError("subquery_disallowed", "Subqueries are not allowed in DP aggregate probes.")

    # Locate top-level FROM / WHERE / GROUP BY / ORDER BY / LIMIT at paren depth 0
    depth = 0
    kw_pos: Dict[str, int] = {}
    i = 1
    while i < len(tokens):
        tok = tokens[i]
        if tok == ("op", "("):
            depth += 1
        elif tok == ("op", ")"):
            depth -= 1
        elif depth == 0 and tok[0] == "word":
            up = tok[1].upper()
            if up in ("FROM", "WHERE", "LIMIT") and up not in kw_pos:
                kw_pos[up] = i
            elif up == "GROUP" and i + 1 < len(tokens) and _kw(tokens[i + 1], "BY") and "GROUP_BY" not in kw_pos:
                kw_pos["GROUP_BY"] = i
            elif up == "ORDER" and i + 1 < len(tokens) and _kw(tokens[i + 1], "BY") and "ORDER_BY" not in kw_pos:
                kw_pos["ORDER_BY"] = i
        i += 1

    if "FROM" not in kw_pos:
        raise DPQueryValidationError("invalid_sql", "SELECT probe must include a FROM <table> clause.")
    from_idx = kw_pos["FROM"]
    from_end = min(
        [kw_pos[k] for k in ("WHERE", "GROUP_BY", "ORDER_BY", "LIMIT") if k in kw_pos] + [len(tokens)]
    )
    from_toks = tokens[from_idx + 1 : from_end]
    if len(from_toks) >= 3 and _is_ident(from_toks[0]) and from_toks[1] == ("op", ".") and _is_ident(from_toks[2]):
        db_alias = _unquote_ident(from_toks[0][0], from_toks[0][1])
        if db_alias.lower() != "main":
            raise DPQueryValidationError("invalid_table", f"Unsupported schema qualifier {db_alias!r}.")
        rest_from = from_toks[3:]
        t_tok = from_toks[2]
    elif from_toks and _is_ident(from_toks[0]):
        t_tok = from_toks[0]
        rest_from = from_toks[1:]
    else:
        raise DPQueryValidationError("invalid_table", "Expected a single table name in FROM clause.")

    if rest_from:
        if any(tok == ("op", ",") for tok in rest_from):
            raise DPQueryValidationError(
                "join_disallowed",
                "Comma-joins across multiple tables are not permitted in DP aggregate probes.",
            )
        raise DPQueryValidationError(
            "invalid_from",
            "FROM clause must reference exactly one unaliased table from the public schema.",
        )

    raw_table = _unquote_ident(t_tok[0], t_tok[1])
    table_map = {t.lower(): t for t in public_schema.tables}
    canon_table = table_map.get(raw_table.lower())
    if canon_table is None:
        raise DPQueryValidationError(
            "unknown_table",
            f"Unknown table {raw_table!r}. Available tables: {sorted(public_schema.tables)}.",
        )
    col_map = {cm.name.lower(): cm.name for cm in public_schema.tables[canon_table]}

    # Parse projection
    proj_toks = tokens[1:from_idx]
    if _kw(proj_toks[0] if proj_toks else None, "DISTINCT"):
        raise DPQueryValidationError(
            "non_aggregate_select",
            "Non-aggregate SELECT DISTINCT is not allowed; use COUNT/SUM/AVG(...) or GROUP BY.",
        )
    proj_items = _split_top_level_commas(proj_toks)
    if len(proj_items) == 1:
        agg_func, count_kind, agg_col = _parse_agg_expr(proj_items[0], canon_table, col_map)
        mode = agg_func  # "count", "sum", or "avg"
        group_col: Optional[str] = None
        if "GROUP_BY" in kw_pos:
            raise DPQueryValidationError(
                "invalid_group_by",
                "GROUP BY probes must include the grouping column in SELECT: SELECT col, COUNT/SUM/AVG(...) ... GROUP BY col.",
            )
    elif len(proj_items) == 2:
        sel_col = _parse_single_col_ref(_strip_optional_alias(proj_items[0]), canon_table, col_map)
        if sel_col is None:
            raise DPQueryValidationError(
                "non_aggregate_select",
                "GROUP BY probes must be formatted as: SELECT <col>, COUNT/SUM/AVG(...) FROM <table> ... GROUP BY <col>.",
            )
        agg_func, count_kind, agg_col = _parse_agg_expr(proj_items[1], canon_table, col_map)
        if "GROUP_BY" not in kw_pos:
            raise DPQueryValidationError(
                "non_aggregate_select",
                "Non-aggregate projection without GROUP BY is not permitted.",
            )
        gb_start = kw_pos["GROUP_BY"] + 2
        gb_end = min([kw_pos[k] for k in ("ORDER_BY", "LIMIT") if k in kw_pos] + [len(tokens)])
        gb_col = _parse_single_col_ref(tokens[gb_start:gb_end], canon_table, col_map)
        if gb_col != sel_col:
            raise DPQueryValidationError(
                "invalid_group_by",
                f"GROUP BY column must match selected column {sel_col!r} (single-column GROUP BY only).",
            )
        mode = "group_count" if agg_func == "count" else f"group_{agg_func}"
        group_col = sel_col
    else:
        raise DPQueryValidationError(
            "non_aggregate_select",
            "Only 1-column aggregate or 2-column (col, aggregate) probes are permitted.",
        )

    distinct_col = agg_col if (agg_func == "count" and count_kind == "distinct") else None

    # Entity engine enforcement: require COUNT(DISTINCT entity_col) when require_distinct_col_by_table is given
    if require_distinct_col_by_table is not None:
        req_ec = require_distinct_col_by_table.get(canon_table)
        if req_ec is not None:
            if agg_func != "count" or count_kind != "distinct" or distinct_col != req_ec:
                raise DPQueryValidationError(
                    "entity_count_required",
                    f"Entity-DP probes on {canon_table!r} must use COUNT(DISTINCT {quote(req_ec)}), "
                    f"because raw row {agg_func.upper()}(...) has unbounded entity sensitivity.",
                )

    # Parse and normalize WHERE if present into canonical SQL + pure-Python AST
    canon_where: Optional[str] = None
    where_ast: Optional[Tuple[Any, ...]] = None
    where_cols: Set[str] = set()
    if "WHERE" in kw_pos:
        w_start = kw_pos["WHERE"] + 1
        w_end = min([kw_pos[k] for k in ("GROUP_BY", "ORDER_BY", "LIMIT") if k in kw_pos] + [len(tokens)])
        w_toks = tokens[w_start:w_end]
        if not w_toks:
            raise DPQueryValidationError("invalid_where", "Empty WHERE clause.")
        wp = _WhereParser(w_toks, canon_table, col_map)
        canon_where, where_ast = wp.parse_with_ast()
        where_cols = set(wp.referenced_cols)

    ref_cols_set: Set[str] = set(where_cols)
    if agg_col is not None:
        ref_cols_set.add(agg_col)
    if group_col is not None:
        ref_cols_set.add(group_col)
    referenced_cols = sorted(ref_cols_set)

    if agg_func == "count":
        if count_kind == "star":
            agg_sql = "COUNT(*)"
        elif count_kind == "distinct":
            assert distinct_col is not None
            agg_sql = f"COUNT(DISTINCT {quote(distinct_col)})"
        else:
            assert agg_col is not None
            agg_sql = f"COUNT({quote(agg_col)})"
    else:
        assert agg_col is not None
        agg_sql = f"{agg_func.upper()}({quote(agg_col)})"

    if group_col is None:
        canon_sql = f"SELECT {agg_sql} FROM {quote(canon_table)}"
        if canon_where:
            canon_sql += f" WHERE {canon_where}"
    else:
        canon_sql = f"SELECT {quote(group_col)}, {agg_sql} FROM {quote(canon_table)}"
        if canon_where:
            canon_sql += f" WHERE {canon_where}"
        canon_sql += f" GROUP BY {quote(group_col)}"

    return {
        "mode": mode,
        "agg_func": agg_func,
        "agg_col": agg_col,
        "table": canon_table,
        "column": group_col,
        "count_kind": count_kind,
        "distinct_col": distinct_col,
        "where_sql": canon_where,
        "where_ast": where_ast,
        "referenced_cols": referenced_cols,
        "canonical_sql": canon_sql,
    }


def _parse_num_literal(tok: str) -> Union[int, float]:
    if "." not in tok and "e" not in tok.lower():
        try:
            v = int(tok)
            if INT64_MIN <= v <= INT64_MAX:
                return v
        except ValueError:
            pass
    return float(tok)


def _python_typeof(val: Any) -> str:
    if val is None:
        return "null"
    if isinstance(val, bool) or isinstance(val, int):
        return "integer"
    if isinstance(val, float):
        return "real"
    if isinstance(val, str):
        return "text"
    if isinstance(val, (bytes, bytearray)):
        return "blob"
    return "text"


def _sqlite_arith_num(val: Any) -> Optional[Union[int, float]]:
    if val is None:
        return None
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return val if math.isfinite(val) else None
    if isinstance(val, (bytes, bytearray)):
        val = bytes(val).decode("utf-8", "replace")
    if isinstance(val, str):
        m = _SQLITE_NUM_PREFIX_RE.match(val)
        if not m:
            return 0
        s = m.group(1)
        if "." not in s and "e" not in s.lower():
            try:
                return int(s)
            except ValueError:
                pass
        f = float(s)
        return f if math.isfinite(f) else None
    return 0


def _sqlite_truth(val: Any) -> Optional[bool]:
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return val != 0
    if isinstance(val, (bytes, bytearray)):
        val = bytes(val).decode("utf-8", "replace")
    if isinstance(val, str):
        m = _SQLITE_NUM_PREFIX_RE.match(val)
        if not m:
            return False
        try:
            return float(m.group(1)) != 0.0
        except (ValueError, OverflowError):
            return False
    return False


def _sqlite_cmp(a: Any, b: Any) -> Optional[int]:
    if a is None or b is None:
        return None
    def _rank(x: Any) -> int:
        if isinstance(x, (int, float)):
            return 1
        if isinstance(x, str):
            return 2
        if isinstance(x, (bytes, bytearray)):
            return 3
        return 2
    ra, rb = _rank(a), _rank(b)
    if ra != rb:
        return -1 if ra < rb else 1
    if ra == 3:
        ba, bb = bytes(a), bytes(b)
        return -1 if ba < bb else (1 if ba > bb else 0)
    return -1 if a < b else (1 if a > b else 0)


def _sqlite_like(pat: Any, val: Any, esc: Any = None) -> Optional[bool]:
    if pat is None or val is None:
        return None
    p_str = str(pat)
    v_str = str(val)
    esc_ch = str(esc)[0] if esc is not None and len(str(esc)) > 0 else None
    rx_parts = ["^"]
    i = 0
    while i < len(p_str):
        ch = p_str[i]
        if esc_ch is not None and ch == esc_ch and i + 1 < len(p_str):
            rx_parts.append(re.escape(p_str[i + 1]))
            i += 2
            continue
        if ch == "%":
            rx_parts.append(".*")
        elif ch == "_":
            rx_parts.append(".")
        else:
            rx_parts.append(re.escape(ch))
        i += 1
    rx_parts.append("$")
    return re.match("".join(rx_parts), v_str, flags=re.DOTALL | re.IGNORECASE) is not None


def _sqlite_glob(pat: Any, val: Any) -> Optional[bool]:
    if pat is None or val is None:
        return None
    p_str = str(pat)
    v_str = str(val)
    rx_parts = ["^"]
    i = 0
    n = len(p_str)
    while i < n:
        ch = p_str[i]
        if ch == "*":
            rx_parts.append(".*")
            i += 1
        elif ch == "?":
            rx_parts.append(".")
            i += 1
        elif ch == "[":
            j = i + 1
            if j < n and p_str[j] in ("!", "^"):
                j += 1
            if j < n and p_str[j] == "]":
                j += 1
            while j < n and p_str[j] != "]":
                j += 1
            if j >= n:
                rx_parts.append(re.escape(ch))
                i += 1
            else:
                body = p_str[i + 1 : j]
                if body.startswith("!"):
                    body = "^" + body[1:]
                rx_parts.append("[" + body + "]")
                i = j + 1
        else:
            rx_parts.append(re.escape(ch))
            i += 1
    rx_parts.append("$")
    return re.match("".join(rx_parts), v_str, flags=re.DOTALL) is not None


def _sqlite_substr(val: Any, start_v: Any, len_v: Any = None) -> Any:
    if val is None or start_v is None or (len_v is not None and len_v is None):
        return None
    is_blob = isinstance(val, (bytes, bytearray))
    seq = bytes(val) if is_blob else str(val)
    n = len(seq)
    st = int(_sqlite_arith_num(start_v) or 0)
    ln = int(_sqlite_arith_num(len_v) or 0) if len_v is not None else None
    if st > 0:
        idx0 = st - 1
    elif st < 0:
        idx0 = n + st
        if idx0 < 0:
            if ln is not None:
                ln = max(0, ln + idx0)
            idx0 = 0
    else:
        if ln is not None:
            ln = max(0, ln - 1)
        idx0 = 0
    if ln is None:
        return seq[idx0:]
    if ln <= 0:
        return b"" if is_blob else ""
    return seq[idx0 : idx0 + ln]


def _sqlite_cast(val: Any, tname: str) -> Any:
    if val is None:
        return None
    up = tname.upper()
    if "INT" in up:
        num = _sqlite_arith_num(val)
        return int(num) if num is not None else 0
    if any(k in up for k in ("REAL", "FLOA", "DOUB")):
        c = _coerce_numeric(val)
        return c if c is not None else 0.0
    if any(k in up for k in ("NUM", "DEC")):
        num = _sqlite_arith_num(val)
        return num if num is not None else 0
    if any(k in up for k in ("TEXT", "CHAR", "CLOB", "STR")):
        return val.decode("utf-8", "replace") if isinstance(val, (bytes, bytearray)) else str(val)
    if "BLOB" in up:
        return bytes(val) if isinstance(val, (bytes, bytearray)) else str(val).encode("utf-8")
    return val


def _eval_where_ast(node: Tuple[Any, ...], row: Dict[str, Any]) -> Any:
    """Pure-Python evaluator for validated single-table SQLite WHERE AST nodes."""
    kind = node[0]
    if kind == "lit":
        return node[1]
    if kind == "col":
        return row[node[1]]
    if kind == "or":
        has_none = False
        for ch in node[1]:
            tv = _sqlite_truth(_eval_where_ast(ch, row))
            if tv is True:
                return 1
            if tv is None:
                has_none = True
        return None if has_none else 0
    if kind == "and":
        has_none = False
        for ch in node[1]:
            tv = _sqlite_truth(_eval_where_ast(ch, row))
            if tv is False:
                return 0
            if tv is None:
                has_none = True
        return None if has_none else 1
    if kind == "not":
        tv = _sqlite_truth(_eval_where_ast(node[1], row))
        if tv is None:
            return None
        return 0 if tv else 1
    if kind == "is_null":
        v = _eval_where_ast(node[1], row)
        neg = node[2]
        return int(v is not None) if neg else int(v is None)
    if kind == "is_expr":
        a = _eval_where_ast(node[1], row)
        b = _eval_where_ast(node[2], row)
        neg = node[3]
        if a is None and b is None:
            eq = True
        elif a is None or b is None:
            eq = False
        else:
            eq = _sqlite_cmp(a, b) == 0
        return int(not eq) if neg else int(eq)
    if kind == "in":
        lhs = _eval_where_ast(node[1], row)
        items = node[2]
        neg = node[3]
        if lhs is None:
            return None
        has_none = False
        for it in items:
            if it is None:
                has_none = True
            elif _sqlite_cmp(lhs, it) == 0:
                return 0 if neg else 1
        if has_none:
            return None
        return 1 if neg else 0
    if kind == "between":
        lhs = _eval_where_ast(node[1], row)
        lo = _eval_where_ast(node[2], row)
        hi = _eval_where_ast(node[3], row)
        neg = node[4]
        c_lo = _sqlite_cmp(lhs, lo)
        c_hi = _sqlite_cmp(lhs, hi)
        tv_lo = None if c_lo is None else (c_lo >= 0)
        tv_hi = None if c_hi is None else (c_hi <= 0)
        if tv_lo is False or tv_hi is False:
            inside: Optional[bool] = False
        elif tv_lo is None or tv_hi is None:
            inside = None
        else:
            inside = True
        if inside is None:
            return None
        return int(not inside) if neg else int(inside)
    if kind == "like":
        lhs = _eval_where_ast(node[1], row)
        pat = _eval_where_ast(node[2], row)
        res = _sqlite_like(pat, lhs)
        if res is None:
            return None
        return int(not res) if node[3] else int(res)
    if kind == "glob":
        lhs = _eval_where_ast(node[1], row)
        pat = _eval_where_ast(node[2], row)
        res = _sqlite_glob(pat, lhs)
        if res is None:
            return None
        return int(not res) if node[3] else int(res)
    if kind == "cmp":
        op = node[1]
        c = _sqlite_cmp(_eval_where_ast(node[2], row), _eval_where_ast(node[3], row))
        if c is None:
            return None
        if op == "=":
            return int(c == 0)
        if op == "<>":
            return int(c != 0)
        if op == "<":
            return int(c < 0)
        if op == "<=":
            return int(c <= 0)
        if op == ">":
            return int(c > 0)
        if op == ">=":
            return int(c >= 0)
        return None
    if kind == "concat":
        parts = []
        for ch in node[1]:
            v = _eval_where_ast(ch, row)
            if v is None:
                return None
            parts.append(v.decode("utf-8", "replace") if isinstance(v, (bytes, bytearray)) else str(v))
        return "".join(parts)
    if kind == "unary":
        v = _eval_where_ast(node[2], row)
        if v is None:
            return None
        if node[1] == "+":
            return v
        n = _sqlite_arith_num(v)
        return None if n is None else -n
    if kind == "binop":
        op = node[1]
        a = _sqlite_arith_num(_eval_where_ast(node[2], row))
        b = _sqlite_arith_num(_eval_where_ast(node[3], row))
        if a is None or b is None:
            return None
        if op in ("/", "%") and b == 0:
            return None
        if isinstance(a, int) and isinstance(b, int):
            if op == "+":
                return a + b
            if op == "-":
                return a - b
            if op == "*":
                return a * b
            if op == "/":
                q = abs(a) // abs(b)
                return -q if (a < 0) ^ (b < 0) else q
            if op == "%":
                q = abs(a) // abs(b)
                q = -q if (a < 0) ^ (b < 0) else q
                return a - q * b
        fa, fb = float(a), float(b)
        if op == "+":
            res = fa + fb
        elif op == "-":
            res = fa - fb
        elif op == "*":
            res = fa * fb
        elif op == "/":
            res = fa / fb
        else:
            res = math.fmod(fa, fb)
        return res if math.isfinite(res) else None
    if kind == "cast":
        return _sqlite_cast(_eval_where_ast(node[1], row), node[2])
    if kind == "func":
        fname = node[1]
        args = [_eval_where_ast(a, row) for a in node[2]]
        if fname == "abs":
            return _dp_safe_abs(args[0])
        if fname == "coalesce":
            for a in args:
                if a is not None:
                    return a
            return None
        if fname == "ifnull":
            return args[0] if args[0] is not None else args[1]
        if fname == "nullif":
            if args[0] is None:
                return None
            if args[1] is not None and _sqlite_cmp(args[0], args[1]) == 0:
                return None
            return args[0]
        if fname == "iif":
            return args[1] if _sqlite_truth(args[0]) is True else args[2]
        if fname == "glob":
            r = _sqlite_glob(args[0], args[1])
            return None if r is None else int(r)
        if fname == "like":
            r = _sqlite_like(args[0], args[1], args[2] if len(args) > 2 else None)
            return None if r is None else int(r)
        if fname == "hex":
            if args[0] is None:
                return None
            b = bytes(args[0]) if isinstance(args[0], (bytes, bytearray)) else str(args[0]).encode("utf-8")
            return b.hex().upper()
        if fname == "instr":
            if args[0] is None or args[1] is None:
                return None
            return str(args[0]).find(str(args[1])) + 1
        if fname == "length":
            if args[0] is None:
                return None
            return len(args[0]) if isinstance(args[0], (bytes, bytearray)) else len(str(args[0]))
        if fname == "lower":
            return None if args[0] is None else str(args[0]).lower()
        if fname == "upper":
            return None if args[0] is None else str(args[0]).upper()
        if fname in ("trim", "ltrim", "rtrim"):
            if args[0] is None or (len(args) > 1 and args[1] is None):
                return None
            chars = str(args[1]) if len(args) > 1 else " "
            s = str(args[0])
            return s.strip(chars) if fname == "trim" else (s.lstrip(chars) if fname == "ltrim" else s.rstrip(chars))
        if fname == "printf":
            return _dp_safe_printf(*args)
        if fname == "replace":
            if args[0] is None or args[1] is None or args[2] is None:
                return None
            src, old, new = str(args[0]), str(args[1]), str(args[2])
            return src if old == "" else src.replace(old, new)
        if fname == "round":
            if args[0] is None:
                return None
            num = _sqlite_arith_num(args[0])
            if num is None:
                return None
            ndig = int(_sqlite_arith_num(args[1]) or 0) if len(args) > 1 and args[1] is not None else 0
            return round(float(num), ndig)
        if fname == "sign":
            if args[0] is None:
                return None
            num = _sqlite_arith_num(args[0])
            if num is None:
                return None
            return 1 if num > 0 else (-1 if num < 0 else 0)
        if fname in ("substr", "substring"):
            if args[0] is None or args[1] is None or (len(args) > 2 and args[2] is None):
                return None
            return _sqlite_substr(args[0], args[1], args[2] if len(args) > 2 else None)
        if fname == "typeof":
            return _python_typeof(args[0])
        if fname == "unicode":
            if args[0] is None:
                return None
            s = str(args[0])
            return ord(s[0]) if s else None
    return None


def eval_unit_total(
    unit_record: Dict[str, Any],
    parsed: Dict[str, Any],
    *,
    clip_lo: int = -1000,
    clip_hi: int = 1000,
    sum_scale: int = 100,
) -> Any:
    """Mathematically total per-row evaluator mapping one row to a bounded contribution.

    Every exception (cast/type/overflow/regex/printf/zero-division) is caught strictly
    per-row so a pathological cell maps ONLY that single row to zero contribution
    (preserving exact l2 sensitivity <= Delta_{2,j}, never aborting or zeroing other rows).
    """
    mode = parsed["mode"]
    where_ast = parsed["where_ast"]
    agg_col = parsed["agg_col"]
    group_col = parsed["column"]
    count_kind = parsed["count_kind"]

    if mode in ("sum", "avg"):
        zero: Any = (0, 0)
    elif mode in ("group_sum", "group_avg"):
        zero = None
    elif mode == "count":
        zero = None if count_kind == "distinct" else 0
    else:
        zero = None

    try:
        if where_ast is not None:
            if _sqlite_truth(_eval_where_ast(where_ast, unit_record)) is not True:
                return zero

        if mode == "count":
            if count_kind == "star":
                return 1
            assert agg_col is not None
            aval = unit_record[agg_col]
            if aval is None:
                return zero
            if count_kind == "nonnull":
                return 1
            return p.canonical_val_key(_python_typeof(aval), aval)

        if mode in ("sum", "avg"):
            assert agg_col is not None
            aval = unit_record[agg_col]
            if aval is None:
                return zero
            sv = _clamp_scaled_int(aval, clip_lo, clip_hi, sum_scale)
            return (1, sv) if sv is not None else zero

        assert group_col is not None
        gval = unit_record[group_col]
        if gval is None:
            return None
        gk = p.canonical_val_key(_python_typeof(gval), gval)

        if mode == "group_count":
            if count_kind == "star":
                return (gk, 1)
            assert agg_col is not None
            aval = unit_record[agg_col]
            if aval is None:
                return None
            if count_kind == "nonnull":
                return (gk, 1)
            return (gk, p.canonical_val_key(_python_typeof(aval), aval))

        # group_sum or group_avg
        assert agg_col is not None
        aval = unit_record[agg_col]
        if aval is None:
            return None
        sv = _clamp_scaled_int(aval, clip_lo, clip_hi, sum_scale)
        return (gk, 1, sv) if sv is not None else None
    except Exception:
        return zero


def eval_entity_total(
    u_rows: Sequence[Dict[str, Any]],
    parsed: Dict[str, Any],
    *,
    cap: int = edp.CAP,
) -> Any:
    """Mathematically total per-entity evaluator under direct-star adjacency (C <= cap).

    Evaluates each row of entity u independently in pure Python, catching any predicate or
    key-extraction error strictly per-row so malformed data in one row/entity never affects
    other rows or entities.
    """
    where_ast = parsed["where_ast"]
    mode = parsed["mode"]
    if mode == "count":
        for rec in u_rows:
            try:
                if where_ast is None or _sqlite_truth(_eval_where_ast(where_ast, rec)) is True:
                    return 1
            except Exception:
                continue
        return 0

    group_col = parsed["column"]
    assert group_col is not None
    top_c: List[Tuple[str, Any]] = []
    for rec in u_rows:
        try:
            if where_ast is not None and _sqlite_truth(_eval_where_ast(where_ast, rec)) is not True:
                continue
            gval = rec[group_col]
            if gval is None:
                continue
            edp._insert_bounded_top_c(
                top_c,
                p.canonical_val_key(_python_typeof(gval), gval),
                cap,
            )
        except Exception:
            continue
    return [item for _r, item in top_c]


def _open_conn(conn_or_path: Union[sqlite3.Connection, str, Path]) -> Tuple[sqlite3.Connection, bool]:
    if isinstance(conn_or_path, sqlite3.Connection):
        return conn_or_path, False
    p_obj = Path(conn_or_path).resolve()
    con = sqlite3.connect(p_obj.as_uri() + "?mode=ro", uri=True, timeout=30.0)
    con.execute("PRAGMA query_only=ON")
    con.execute("PRAGMA temp_store=MEMORY")
    return con, True


@contextmanager
def _restricted_read(conn: sqlite3.Connection, allowed_table: str):
    """Restrict SQLite execution to plain single-table raw row scanning (no UDFs, no timeout)."""

    def authorize(action, first, second, db, origin):
        if origin is not None:
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_SELECT:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ and first == allowed_table and db in ("main", None):
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    conn.set_authorizer(authorize)
    try:
        yield
    finally:
        if sys.version_info >= (3, 11):
            conn.set_authorizer(None)
        else:
            conn.set_authorizer(lambda *_args: sqlite3.SQLITE_OK)


def _scan_table_deterministic(
    conn: sqlite3.Connection,
    table: str,
    cols: Sequence[str],
    has_rowid: bool = True,
) -> List[Dict[str, Any]]:
    """Deterministically scan referenced raw columns of a single table without SQL expressions."""
    qt = quote(table)
    col_list = list(cols)
    if col_list:
        cols_sql = ", ".join(quote(c) for c in col_list)
        if has_rowid:
            sql = f"SELECT rowid, {cols_sql} FROM {qt} ORDER BY rowid"
            cur = conn.execute(sql)
            return [dict(zip(col_list, rec[1:])) for rec in cur]
        sql = f"SELECT {cols_sql} FROM {qt}"
        cur = conn.execute(sql)
        return [dict(zip(col_list, rec)) for rec in cur]
    if has_rowid:
        cur = conn.execute(f"SELECT rowid FROM {qt} ORDER BY rowid")
        return [{} for _ in cur]
    cur = conn.execute(f"SELECT 1 FROM {qt}")
    return [{} for _ in cur]


class _BaseRowDPEngine:
    """Shared public SchemaSandbox pre-validation, cache, and total row-DP evaluation."""

    def __init__(
        self,
        conn_or_path: Union[sqlite3.Connection, str, Path],
        *,
        max_probes_per_q: int = 2,
        epsilon: Union[int, Fraction] = 8,
        delta: Fraction = p.TABLE_DELTA_FRAC,
        top_k: int = 15,
        rng: Optional[dg.IntegerRNG] = None,
        clip_bounds: Tuple[int, int] = (-1000, 1000),
        sum_scale: int = 100,
    ) -> None:
        if max_probes_per_q < 1:
            raise ValueError("max_probes_per_q must be >= 1")
        clip_lo, clip_hi = int(clip_bounds[0]), int(clip_bounds[1])
        if clip_lo > clip_hi:
            raise ValueError("clip_bounds must satisfy clip_lo <= clip_hi")
        if sum_scale < 1:
            raise ValueError("sum_scale must be >= 1")

        self._conn, self._owns_conn = _open_conn(conn_or_path)
        self.schema = v3.extract_public_schema(self._conn)
        self._sandbox = sb.SchemaSandbox(self._conn)

        self.max_probes_per_q = int(max_probes_per_q)
        self.epsilon = Fraction(epsilon)
        self.delta = Fraction(delta)
        self.delta_conv = self.delta / 2
        self.delta_th = self.delta / 2
        self.top_k = int(top_k)
        self.clip_lo = clip_lo
        self.clip_hi = clip_hi
        self.sum_scale = int(sum_scale)
        self.clip_max_abs = max(abs(clip_lo), abs(clip_hi))
        self.clip_span = max(self.clip_max_abs, clip_hi - clip_lo)
        self._rng: dg.IntegerRNG = dg.BufferedOSCSPRNG() if rng is None else rng
        self.question_probes_used: Dict[Any, int] = defaultdict(int)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._has_rowid_cache: Dict[str, bool] = {}

    def close(self) -> None:
        self._sandbox.connection.close()
        if self._owns_conn:
            self._conn.close()

    def _table_has_rowid(self, table: str) -> bool:
        if table not in self._has_rowid_cache:
            res = self._sandbox.try_sql(f"SELECT rowid FROM {quote(table)} LIMIT 0")
            self._has_rowid_cache[table] = (res.get("status") == "success")
        return self._has_rowid_cache[table]

    def _init_noise_params(self, rho_budget: Fraction, budget_slots: int) -> None:
        self.budget_slots = int(budget_slots)
        self.rho_probe = rho_budget / self.budget_slots
        self.delta_probe = self.delta_th / self.budget_slots
        # Exact Discrete Gaussian parameters for sensitivity-1 counts
        self.sigma2 = Fraction(1, 2) / self.rho_probe
        self.tau, self.tail_diag = dg.exact_dgauss_tau(self.sigma2, self.delta_probe)
        # Half-rho parameters for SUM/AVG (count gating + clamped sum normalized to [L, U] span)
        self.rho_half = self.rho_probe / 2
        self.sigma2_half_cnt = Fraction(1, 2) / self.rho_half
        self.tau_half_cnt, _ = dg.exact_dgauss_tau(self.sigma2_half_cnt, self.delta_probe)
        m_scaled = self.clip_span * self.sum_scale
        sens2_sum = max(1, m_scaled * m_scaled)
        self.sigma2_sum = Fraction(sens2_sum, 2) / self.rho_probe
        self.sigma2_sum_half = Fraction(sens2_sum, 2) / self.rho_half

    def _public_precheck(self, sql: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Run public AST validation + empty SchemaSandbox check (0 private reads, 0 budget)."""
        try:
            parsed = parse_and_normalize_probe(sql, self.schema)
        except DPQueryValidationError as exc:
            return None, {
                "status": "rejected_public_check",
                "category": exc.category,
                "message": exc.message,
                "cached": False,
            }
        sand_res = self._sandbox.try_sql(sql)
        if sand_res.get("status") != "success":
            return None, {
                "status": "rejected_public_check",
                "category": sand_res.get("category", "public_sandbox_error"),
                "message": f"Public empty-schema pre-check failed ({sand_res.get('category', 'error')}).",
                "cached": False,
            }
        return parsed, None

    def probe(
        self,
        table: str,
        column: Optional[str] = None,
        filter_sql: Optional[str] = None,
        question_id: Any = None,
    ) -> Dict[str, Any]:
        """Structured convenience wrapper delegating to AST-validated query()."""
        qt = quote(table)
        if column is None:
            sql = f"SELECT COUNT(*) FROM {qt}"
            if filter_sql:
                sql += f" WHERE ({filter_sql})"
        else:
            qc = quote(column)
            sql = f"SELECT {qc}, COUNT(*) FROM {qt}"
            if filter_sql:
                sql += f" WHERE ({filter_sql})"
            sql += f" GROUP BY {qc}"
        return self.query(sql, question_id=question_id)

    def _eval_and_noise(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Pure-Python total per-row evaluation over deterministic scan + Discrete Gaussian."""
        table = parsed["table"]
        canon_sql = parsed["canonical_sql"]
        mode = parsed["mode"]
        agg_func = parsed["agg_func"]
        agg_col = parsed["agg_col"]
        group_col = parsed["column"]
        count_kind = parsed["count_kind"]
        ref_cols = parsed["referenced_cols"]
        has_rowid = self._table_has_rowid(table)

        with _restricted_read(self._conn, table):
            row_records = _scan_table_deterministic(self._conn, table, ref_cols, has_rowid=has_rowid)

        if mode == "count":
            if count_kind == "distinct":
                distinct_vals: Set[Tuple[str, Any]] = set()
                for rec in row_records:
                    contrib = eval_unit_total(
                        rec, parsed, clip_lo=self.clip_lo, clip_hi=self.clip_hi, sum_scale=self.sum_scale
                    )
                    if contrib is not None:
                        distinct_vals.add(contrib)
                true_count = len(distinct_vals)
            else:
                true_count = 0
                for rec in row_records:
                    true_count += int(
                        eval_unit_total(
                            rec, parsed, clip_lo=self.clip_lo, clip_hi=self.clip_hi, sum_scale=self.sum_scale
                        )
                    )
            noisy_count = max(0, true_count + int(dg.sample_dgauss(self.sigma2, self._rng)))
            return {
                "status": "ok",
                "mode": "count",
                "agg": "count",
                "table": table,
                "column": None,
                "canonical_sql": canon_sql,
                "noisy_count": noisy_count,
                "sigma2": p._frac_str(self.sigma2),
                "tau": None,
                "cached": False,
            }

        if mode in ("sum", "avg"):
            assert agg_col is not None
            scaled_sum = 0
            nonnull_cnt = 0
            for rec in row_records:
                c_cnt, c_sv = eval_unit_total(
                    rec, parsed, clip_lo=self.clip_lo, clip_hi=self.clip_hi, sum_scale=self.sum_scale
                )
                nonnull_cnt += c_cnt
                scaled_sum += c_sv
            if mode == "sum":
                noisy_scaled = scaled_sum + int(dg.sample_dgauss(self.sigma2_sum, self._rng))
                return {
                    "status": "ok",
                    "mode": "sum",
                    "agg": "sum",
                    "table": table,
                    "column": agg_col,
                    "canonical_sql": canon_sql,
                    "noisy_sum": noisy_scaled / self.sum_scale,
                    "clip_bounds": [self.clip_lo, self.clip_hi],
                    "sigma2": p._frac_str(self.sigma2_sum),
                    "tau": None,
                    "cached": False,
                }
            noisy_scaled = scaled_sum + int(dg.sample_dgauss(self.sigma2_sum_half, self._rng))
            noisy_sum = noisy_scaled / self.sum_scale
            noisy_cnt = max(0, nonnull_cnt + int(dg.sample_dgauss(self.sigma2_half_cnt, self._rng)))
            noisy_avg: Optional[float] = None
            if noisy_cnt > 0:
                noisy_avg = max(float(self.clip_lo), min(float(self.clip_hi), noisy_sum / noisy_cnt))
            return {
                "status": "ok",
                "mode": "avg",
                "agg": "avg",
                "table": table,
                "column": agg_col,
                "canonical_sql": canon_sql,
                "noisy_avg": noisy_avg,
                "noisy_sum": noisy_sum,
                "noisy_count": noisy_cnt,
                "clip_bounds": [self.clip_lo, self.clip_hi],
                "sigma2": p._frac_str(self.sigma2_sum_half),
                "tau": None,
                "cached": False,
            }

        # GROUP BY modes: group_count, group_sum, group_avg
        assert group_col is not None
        if mode == "group_count":
            if count_kind == "distinct":
                by_grp: Dict[Tuple[str, Any], Set[Tuple[str, Any]]] = defaultdict(set)
                for rec in row_records:
                    contrib = eval_unit_total(
                        rec, parsed, clip_lo=self.clip_lo, clip_hi=self.clip_hi, sum_scale=self.sum_scale
                    )
                    if contrib is not None:
                        gk, vk = contrib
                        by_grp[gk].add(vk)
                counts: Counter = Counter({gk: len(vs) for gk, vs in by_grp.items()})
            else:
                counts = Counter()
                for rec in row_records:
                    contrib = eval_unit_total(
                        rec, parsed, clip_lo=self.clip_lo, clip_hi=self.clip_hi, sum_scale=self.sum_scale
                    )
                    if contrib is not None:
                        gk, inc = contrib
                        counts[gk] += inc

            surviving: List[Tuple[Any, int, Tuple[str, Any]]] = []
            for key in sorted(counts, key=repr):
                noisy = int(counts[key]) + int(dg.sample_dgauss(self.sigma2, self._rng))
                if noisy > self.tau:
                    surviving.append((p.decode_val_key(key), noisy, key))
            surviving.sort(key=lambda item: (-item[1], repr(item[2])))
            out_rows = [{"value": val, "noisy_count": cnt} for val, cnt, _k in surviving[: self.top_k]]
            return {
                "status": "ok",
                "mode": "group_count",
                "agg": "count",
                "table": table,
                "column": group_col,
                "canonical_sql": canon_sql,
                "rows": out_rows,
                "truncated": len(surviving) > self.top_k,
                "sigma2": p._frac_str(self.sigma2),
                "tau": self.tau,
                "cached": False,
            }

        # group_sum or group_avg
        assert agg_col is not None
        grp_cnts: Counter = Counter()
        grp_sums: Counter = Counter()
        for rec in row_records:
            contrib = eval_unit_total(
                rec, parsed, clip_lo=self.clip_lo, clip_hi=self.clip_hi, sum_scale=self.sum_scale
            )
            if contrib is not None:
                gk, inc, sv = contrib
                grp_cnts[gk] += inc
                grp_sums[gk] += sv

        surv_agg: List[Tuple[Any, int, float, Optional[float], Tuple[str, Any]]] = []
        for key in sorted(grp_cnts, key=repr):
            noisy_c = int(grp_cnts[key]) + int(dg.sample_dgauss(self.sigma2_half_cnt, self._rng))
            if noisy_c > self.tau_half_cnt:
                noisy_s = (int(grp_sums[key]) + int(dg.sample_dgauss(self.sigma2_sum_half, self._rng))) / self.sum_scale
                noisy_a = max(float(self.clip_lo), min(float(self.clip_hi), noisy_s / noisy_c))
                surv_agg.append((p.decode_val_key(key), noisy_c, noisy_s, noisy_a, key))
        surv_agg.sort(key=lambda item: (-item[1], repr(item[4])))
        out_rows = []
        for val, nc, ns, na, _k in surv_agg[: self.top_k]:
            rec_out: Dict[str, Any] = {"value": val, "noisy_count": nc, "noisy_sum": ns}
            if agg_func == "avg":
                rec_out["noisy_avg"] = na
            out_rows.append(rec_out)
        return {
            "status": "ok",
            "mode": mode,
            "agg": agg_func,
            "table": table,
            "column": group_col,
            "agg_column": agg_col,
            "canonical_sql": canon_sql,
            "rows": out_rows,
            "truncated": len(surv_agg) > self.top_k,
            "clip_bounds": [self.clip_lo, self.clip_hi],
            "sigma2": p._frac_str(self.sigma2_sum_half),
            "tau": self.tau_half_cnt,
            "cached": False,
        }


class DPQueryRowEngine(_BaseRowDPEngine):
    """Interactive DP aggregate engine under single-row adjacency D ~_row D' (Table-Parallel).

    Parallel composition gives each table T its own (epsilon, delta) budget
    (default epsilon=8, delta=1e-6), divided across max_probes_per_table = max_probes_per_q * q_plan slots.
    Repeated identical normalized SQL probes on T are served from an intra-stream cache
    at 0 extra privacy/slot cost.
    """

    def __init__(
        self,
        conn_or_path: Union[sqlite3.Connection, str, Path],
        q_db: Optional[int] = None,
        max_probes_per_q: int = 2,
        epsilon: Union[int, Fraction] = 8,
        delta: Fraction = p.TABLE_DELTA_FRAC,
        top_k: int = 15,
        rng: Optional[dg.IntegerRNG] = None,
        *,
        q_plan: Optional[int] = None,
        max_probes_per_table: Optional[int] = None,
        clip_bounds: Tuple[int, int] = (-1000, 1000),
        sum_scale: int = 100,
    ) -> None:
        super().__init__(
            conn_or_path,
            max_probes_per_q=max_probes_per_q,
            epsilon=epsilon,
            delta=delta,
            top_k=top_k,
            rng=rng,
            clip_bounds=clip_bounds,
            sum_scale=sum_scale,
        )
        q_eff = q_plan if q_plan is not None else (q_db if q_db is not None else 1)
        if q_eff < 1:
            raise ValueError("q_plan / q_db must be >= 1")
        self.q_plan = int(q_eff)
        self.q_db = self.q_plan
        slots = int(max_probes_per_table) if max_probes_per_table is not None else self.max_probes_per_q * self.q_plan
        if slots < 1:
            raise ValueError("max_probes_per_table must be >= 1")
        self.max_probes_per_table = slots
        self.rho_table = dg.rational_rho_budget(self.epsilon, self.delta_conv)
        self._init_noise_params(self.rho_table, slots)
        self.table_probes_used: Dict[str, int] = {t: 0 for t in self.schema.tables}

    def query(self, sql: str, question_id: Any = None) -> Dict[str, Any]:
        """Public-precheck, cache-check, unconditionally account budget, and run total DP probe."""
        parsed, err = self._public_precheck(sql)
        if err is not None:
            return err
        assert parsed is not None

        canon_sql = parsed["canonical_sql"]
        if canon_sql in self._cache:
            cached_resp = copy.deepcopy(self._cache[canon_sql])
            cached_resp["cached"] = True
            return cached_resp

        table = parsed["table"]
        if question_id is not None and self.question_probes_used[question_id] >= self.max_probes_per_q:
            return {
                "status": "budget_exhausted",
                "category": "question_probe_budget_exhausted",
                "message": f"At most {self.max_probes_per_q} unique DP probes are allowed per question.",
                "cached": False,
            }
        if self.table_probes_used[table] >= self.budget_slots:
            return {
                "status": "budget_exhausted",
                "category": "table_probe_budget_exhausted",
                "message": f"Table {table!r} has exhausted all {self.budget_slots} DP probe slots.",
                "cached": False,
            }

        # Unconditional budget increment BEFORE touching private D (zero budget-refund side-channel)
        self.table_probes_used[table] += 1
        if question_id is not None:
            self.question_probes_used[question_id] += 1

        resp = self._eval_and_noise(parsed)
        resp["probes_used_table"] = self.table_probes_used[table]
        resp["probes_remaining_table"] = self.budget_slots - self.table_probes_used[table]
        self._cache[canon_sql] = copy.deepcopy(resp)
        return resp


class DPQueryRowDBWideEngine(_BaseRowDPEngine):
    """Interactive DP aggregate engine under single-row database adjacency D ~_{db-1row} D'.

    Adjacency definition: D ~_{db-1row} D' iff D' is obtained from D by adding or removing
    at most 1 row in at most 1 table of the database D. This is strictly weaker than
    all-table single-row adjacency ~_{all-row} (where m tables each change by 1 row); under
    ~_{db-1row}, sequential composition across all max_db_probes = max_probes_per_q * q_plan
    probes against the shared DB-wide (epsilon_db=8, delta_db=1e-6) budget holds directly
    (rho_probe = rho_db / max_db_probes, delta_probe = (delta_db/2) / max_db_probes).
    """

    def __init__(
        self,
        conn_or_path: Union[sqlite3.Connection, str, Path],
        q_db: Optional[int] = None,
        max_probes_per_q: int = 2,
        epsilon: Union[int, Fraction] = 8,
        delta: Fraction = p.TABLE_DELTA_FRAC,
        top_k: int = 15,
        rng: Optional[dg.IntegerRNG] = None,
        *,
        q_plan: Optional[int] = None,
        max_db_probes: Optional[int] = None,
        clip_bounds: Tuple[int, int] = (-1000, 1000),
        sum_scale: int = 100,
    ) -> None:
        super().__init__(
            conn_or_path,
            max_probes_per_q=max_probes_per_q,
            epsilon=epsilon,
            delta=delta,
            top_k=top_k,
            rng=rng,
            clip_bounds=clip_bounds,
            sum_scale=sum_scale,
        )
        q_eff = q_plan if q_plan is not None else (q_db if q_db is not None else 1)
        if q_eff < 1:
            raise ValueError("q_plan / q_db must be >= 1")
        self.q_plan = int(q_eff)
        self.q_db = self.q_plan
        slots = int(max_db_probes) if max_db_probes is not None else self.max_probes_per_q * self.q_plan
        if slots < 1:
            raise ValueError("max_db_probes must be >= 1")
        self.max_db_probes = slots
        self.rho_db = dg.rational_rho_budget(self.epsilon, self.delta_conv)
        self._init_noise_params(self.rho_db, slots)
        self.probes_used: int = 0

    def query(self, sql: str, question_id: Any = None) -> Dict[str, Any]:
        """Public-precheck, cache-check, unconditionally account DB-wide budget, and run total DP probe."""
        parsed, err = self._public_precheck(sql)
        if err is not None:
            return err
        assert parsed is not None

        canon_sql = parsed["canonical_sql"]
        if canon_sql in self._cache:
            cached_resp = copy.deepcopy(self._cache[canon_sql])
            cached_resp["cached"] = True
            return cached_resp

        if question_id is not None and self.question_probes_used[question_id] >= self.max_probes_per_q:
            return {
                "status": "budget_exhausted",
                "category": "question_probe_budget_exhausted",
                "message": f"At most {self.max_probes_per_q} unique DP probes are allowed per question.",
                "cached": False,
            }
        if self.probes_used >= self.budget_slots:
            return {
                "status": "budget_exhausted",
                "category": "db_probe_budget_exhausted",
                "message": f"Database has exhausted all {self.budget_slots} DP probe slots.",
                "cached": False,
            }

        # Unconditional budget increment BEFORE touching private D
        self.probes_used += 1
        if question_id is not None:
            self.question_probes_used[question_id] += 1

        resp = self._eval_and_noise(parsed)
        resp["probes_used_db"] = self.probes_used
        resp["probes_remaining_db"] = self.budget_slots - self.probes_used
        self._cache[canon_sql] = copy.deepcopy(resp)
        return resp


class DPQueryEntityEngine:
    """Interactive DP aggregate engine under direct-star entity adjacency D ~_ent D'.

    Splits database-wide (epsilon_db=8, delta_db=1e-6) budget across max_probes_per_q * q_plan
    probes on entity_direct tables (CAP=5), rejecting schema_only tables at 0 data access,
    with exact normalized-SQL intra-stream cache and total exception-free private evaluation.
    """

    def __init__(
        self,
        conn_or_path: Union[sqlite3.Connection, str, Path],
        entity_specs: Dict[str, Dict[str, Any]],
        q_db: Optional[int] = None,
        max_probes_per_q: int = 2,
        epsilon: Union[int, Fraction] = 8,
        delta: Fraction = p.TABLE_DELTA_FRAC,
        cap: int = edp.CAP,
        top_k: int = 15,
        rng: Optional[dg.IntegerRNG] = None,
        *,
        q_plan: Optional[int] = None,
        max_db_probes: Optional[int] = None,
    ) -> None:
        if max_probes_per_q < 1:
            raise ValueError("max_probes_per_q must be >= 1")
        q_eff = q_plan if q_plan is not None else (q_db if q_db is not None else 1)
        if q_eff < 1:
            raise ValueError("q_plan / q_db must be >= 1")

        self._conn, self._owns_conn = _open_conn(conn_or_path)
        self.schema = v3.extract_public_schema(self._conn)
        edp.validate_entity_specs(self.schema, entity_specs)
        self._sandbox = sb.SchemaSandbox(self._conn)
        self._has_rowid_cache: Dict[str, bool] = {}

        self.entity_specs = copy.deepcopy(entity_specs)
        self.entity_cols: Dict[str, str] = {
            t: spec["entity_col"]
            for t, spec in self.entity_specs.items()
            if spec["mode"] == "entity_direct"
        }
        self.q_plan = int(q_eff)
        self.q_db = self.q_plan
        self.max_probes_per_q = int(max_probes_per_q)
        slots = int(max_db_probes) if max_db_probes is not None else self.max_probes_per_q * self.q_plan
        if slots < 1:
            raise ValueError("max_db_probes must be >= 1")
        self.budget_slots = slots
        self.cap = int(cap)
        self.epsilon = Fraction(epsilon)
        self.delta = Fraction(delta)
        self.delta_conv = self.delta / 2
        self.delta_th_db = self.delta / 2
        self.rho_db = dg.rational_rho_budget(self.epsilon, self.delta_conv)
        self.rho_probe = self.rho_db / self.budget_slots
        self.delta_probe = self.delta_th_db / self.budget_slots
        self.sigma2_count = Fraction(1) / (2 * self.rho_probe)
        self.sigma2_group = Fraction(self.cap) / (2 * self.rho_probe)
        self.tau_group, self.tail_diag = dg.exact_dgauss_tau(self.sigma2_group, self.delta_probe / self.cap)
        self.top_k = int(top_k)
        self._rng: dg.IntegerRNG = dg.BufferedOSCSPRNG() if rng is None else rng
        self.probes_used: int = 0
        self.question_probes_used: Dict[Any, int] = defaultdict(int)
        self._cache: Dict[str, Dict[str, Any]] = {}

    def close(self) -> None:
        self._sandbox.connection.close()
        if self._owns_conn:
            self._conn.close()

    def probe(
        self,
        table: str,
        column: Optional[str] = None,
        filter_sql: Optional[str] = None,
        question_id: Any = None,
    ) -> Dict[str, Any]:
        """Structured convenience wrapper building direct-star COUNT(DISTINCT entity_col) probes."""
        table_map = {t.lower(): t for t in self.schema.tables}
        canon_table = table_map.get((table or "").lower())
        if canon_table is None:
            return {
                "status": "rejected_public_check",
                "category": "unknown_table",
                "message": f"Unknown table {table!r}. Available tables: {sorted(self.schema.tables)}.",
                "cached": False,
            }
        if self.entity_specs[canon_table]["mode"] == "schema_only":
            return {
                "status": "rejected_public_check",
                "category": "schema_only_table",
                "message": (
                    f"Table {canon_table!r} is schema_only under direct-star entity adjacency "
                    "and cannot be queried (0 private data access)."
                ),
                "cached": False,
            }
        ec = quote(self.entity_cols[canon_table])
        qt = quote(canon_table)
        if column is None:
            sql = f"SELECT COUNT(DISTINCT {ec}) FROM {qt}"
            if filter_sql:
                sql += f" WHERE ({filter_sql})"
        else:
            qc = quote(column)
            sql = f"SELECT {qc}, COUNT(DISTINCT {ec}) FROM {qt}"
            if filter_sql:
                sql += f" WHERE ({filter_sql})"
            sql += f" GROUP BY {qc}"
        return self.query(sql, question_id=question_id)

    def query(self, sql: str, question_id: Any = None) -> Dict[str, Any]:
        """Public-precheck, cache-check, unconditionally account budget, and run total entity-DP probe."""
        try:
            parsed = parse_and_normalize_probe(
                sql,
                self.schema,
                require_distinct_col_by_table=self.entity_cols,
            )
        except DPQueryValidationError as exc:
            return {
                "status": "rejected_public_check",
                "category": exc.category,
                "message": exc.message,
                "cached": False,
            }

        table = parsed["table"]
        if self.entity_specs[table]["mode"] == "schema_only":
            return {
                "status": "rejected_public_check",
                "category": "schema_only_table",
                "message": (
                    f"Table {table!r} is schema_only under direct-star entity adjacency "
                    "and cannot be queried (0 private data access)."
                ),
                "cached": False,
            }

        sand_res = self._sandbox.try_sql(sql)
        if sand_res.get("status") != "success":
            return {
                "status": "rejected_public_check",
                "category": sand_res.get("category", "public_sandbox_error"),
                "message": f"Public empty-schema pre-check failed ({sand_res.get('category', 'error')}).",
                "cached": False,
            }

        canon_sql = parsed["canonical_sql"]
        if canon_sql in self._cache:
            cached_resp = copy.deepcopy(self._cache[canon_sql])
            cached_resp["cached"] = True
            return cached_resp

        if question_id is not None and self.question_probes_used[question_id] >= self.max_probes_per_q:
            return {
                "status": "budget_exhausted",
                "category": "question_probe_budget_exhausted",
                "message": f"At most {self.max_probes_per_q} unique DP probes are allowed per question.",
                "cached": False,
            }
        if self.probes_used >= self.budget_slots:
            return {
                "status": "budget_exhausted",
                "category": "db_probe_budget_exhausted",
                "message": f"Database has exhausted all {self.budget_slots} entity-DP probe slots.",
                "cached": False,
            }

        # Unconditional budget increment BEFORE touching private D
        self.probes_used += 1
        if question_id is not None:
            self.question_probes_used[question_id] += 1

        entity_col = self.entity_cols[table]
        scan_cols = sorted(set(parsed["referenced_cols"]) | {entity_col})
        if table not in self._has_rowid_cache:
            res = self._sandbox.try_sql(f"SELECT rowid FROM {quote(table)} LIMIT 0")
            self._has_rowid_cache[table] = (res.get("status") == "success")
        has_rowid = self._has_rowid_cache[table]

        with _restricted_read(self._conn, table):
            row_records = _scan_table_deterministic(self._conn, table, scan_cols, has_rowid=has_rowid)

        by_entity: Dict[Tuple[str, Any], List[Dict[str, Any]]] = defaultdict(list)
        for rec in row_records:
            try:
                ev = rec[entity_col]
                if ev is None:
                    continue
                ek = p.canonical_val_key(_python_typeof(ev), ev)
                by_entity[ek].append(rec)
            except Exception:
                continue

        if parsed["mode"] == "count":
            true_count = sum(
                int(eval_entity_total(u_rows, parsed, cap=self.cap))
                for u_rows in by_entity.values()
            )
            noisy_count = max(0, true_count + int(dg.sample_dgauss(self.sigma2_count, self._rng)))
            resp: Dict[str, Any] = {
                "status": "ok",
                "mode": "count",
                "agg": "count",
                "table": table,
                "column": None,
                "entity_col": entity_col,
                "canonical_sql": canon_sql,
                "noisy_count": noisy_count,
                "sensitivity2": 1,
                "sigma2": p._frac_str(self.sigma2_count),
                "tau": None,
                "cached": False,
            }
        else:
            col = parsed["column"]
            assert col is not None
            counts: Counter = Counter()
            for u_rows in by_entity.values():
                kept_keys = eval_entity_total(u_rows, parsed, cap=self.cap)
                for vk in kept_keys:
                    counts[vk] += 1
            surviving: List[Tuple[Any, int, Tuple[str, Any]]] = []
            for key in sorted(counts, key=repr):
                noisy = int(counts[key]) + int(dg.sample_dgauss(self.sigma2_group, self._rng))
                if noisy > self.tau_group:
                    surviving.append((p.decode_val_key(key), noisy, key))
            surviving.sort(key=lambda item: (-item[1], repr(item[2])))
            out_rows = [{"value": val, "noisy_count": cnt} for val, cnt, _k in surviving[: self.top_k]]
            resp = {
                "status": "ok",
                "mode": "group_count",
                "agg": "count",
                "table": table,
                "column": col,
                "entity_col": entity_col,
                "canonical_sql": canon_sql,
                "rows": out_rows,
                "truncated": len(surviving) > self.top_k,
                "sensitivity2": self.cap,
                "sigma2": p._frac_str(self.sigma2_group),
                "tau": self.tau_group,
                "cached": False,
            }

        resp["probes_used_db"] = self.probes_used
        resp["probes_remaining_db"] = self.budget_slots - self.probes_used
        self._cache[canon_sql] = copy.deepcopy(resp)
        return resp
