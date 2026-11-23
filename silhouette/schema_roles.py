"""Public SQLite metadata, role precedence and schema-only query scheduling."""
from __future__ import annotations

from dataclasses import dataclass
import re
import sqlite3

def quote(name):
    return '"' + name.replace('"', '""') + '"'


def numeric(typ):
    return bool(re.search(r'INT|REAL|FLOA|DOUB|NUM|DEC', typ, re.I))


def is_integer_type(typ):
    return bool(re.search(r'INT', typ or '', re.I))


ROLE_WEIGHTS = {
    'KEY_OR_FK': {
        'null_count': 0.5,
        'distinct_count': 0.5,
        'bins': 0.5,
        'patterns': 0.5,
    },
    'IDENT_OR_CONTACT': {
        'null_count': 0.5,
        'distinct_count': 0.5,
        'bins': 0.5,
        'patterns': 0.5,
    },
    'FREE_TEXT': {
        'null_count': 0.5,
        'distinct_count': 0.5,
        'patterns': 0.5,
        'values': 0.25,
    },
    'TEMPORAL': {
        'null_count': 1.0,
        'distinct_count': 1.5,
        'patterns': 2.0,
        'values': 1.0,
    },
    'NUMERIC_REAL': {
        'null_count': 1.0,
        'distinct_count': 1.0,
        'bins': 2.0,
    },
    'NUMERIC_INT': {
        'null_count': 1.0,
        'distinct_count': 1.5,
        'bins': 2.0,
        'values': 1.0,
    },
    'ENUM_INT': {
        'null_count': 1.0,
        'distinct_count': 1.5,
        'bins': 1.0,
        'values': 4.0,
    },
    'CATEGORICAL_TEXT': {
        'null_count': 1.0,
        'distinct_count': 1.5,
        'patterns': 2.0,
        'values': 6.0,
    },
}
ROW_COUNT_WEIGHT = 1.0
SCALAR_KAPPA = 2.5  # Calibrated 2.5-sigma confidence scale for scalar counts (distinct gate reliability) vs Phi^-1(1-delta_j) for histograms.


@dataclass(frozen=True)
class ColumnMeta:
    name: str
    declared_type: str
    is_pk: bool
    is_single_pk_or_unique: bool
    is_composite_pk: bool
    is_fk_src: bool
    is_fk_tgt: bool


@dataclass(frozen=True)
class PublicSchema:
    tables: dict      # {table_name: tuple[ColumnMeta, ...]}
    fk_edges: tuple   # tuple of (src_table, src_col, dst_table, dst_col)


def tokenize_name(name):
    """Split SQL column name on delimiters and CamelCase boundaries."""
    parts = re.split(r'[^A-Za-z0-9]+', name or '')
    tokens = []
    for p in parts:
        if not p:
            continue
        sub = re.findall(r'[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|\d+', p)
        tokens.extend(sub if sub else [p])
    return tokens


def is_id_name(name):
    low = [t.lower() for t in tokenize_name(name)]
    if not low:
        return False
    if low[-1] in ('id', 'ids', 'uuid', 'guid'):
        return True
    if len(low) >= 3 and low[0] == 'link' and low[1] == 'to':
        return True
    return False


_IDENT_RAW_WORDS = frozenset({'uuid', 'guid', 'hash', 'md5', 'sha1', 'sha256', 'sha512'})
_IDENT_TOKENS = frozenset({
    'uuid', 'guid', 'hash', 'md5', 'sha1', 'sha256', 'sha512',
    'url', 'urls', 'website', 'email', 'phone', 'fax', 'mobile',
    'street', 'ext', 'fname', 'lname', 'firstname', 'lastname', 'zip',
    'username', 'nickname', 'handle', 'login', 'ssn', 'passport',
    'iban', 'avatar', 'homepage', 'surname', 'forename',
})
_IDENT_BIGRAMS = frozenset({
    ('display', 'name'), ('screen', 'name'), ('user', 'name'),
    ('first', 'name'), ('last', 'name'), ('full', 'name'),
    ('real', 'name'), ('middle', 'name'), ('family', 'name'),
    ('given', 'name'), ('player', 'name'), ('owner', 'name'),
    ('author', 'name'), ('editor', 'name'),
})


def is_ident_or_contact_name(name):
    if is_id_name(name):
        return True
    raw_words = set(re.findall(r'[a-z0-9]+', (name or '').lower()))
    if raw_words & _IDENT_RAW_WORDS:
        return True
    low = [t.lower() for t in tokenize_name(name)]
    if set(low) & _IDENT_TOKENS:
        return True
    for i in range(len(low) - 1):
        pair = (low[i], low[i + 1])
        if pair in _IDENT_BIGRAMS:
            return True
        if low[i] in ('f', 'l') and low[i + 1] == 'name':
            return True
    return False


def is_free_text_name(name):
    low = [t.lower() for t in tokenize_name(name)]
    if not low:
        return False
    if low == ['text']:
        return True
    if set(low) & {
        'body', 'description', 'comment', 'comments', 'notes', 'summary',
        'biography', 'abstract', 'aboutme', 'bio',
    }:
        return True
    if any((low[i], low[i + 1]) == ('about', 'me') for i in range(len(low) - 1)):
        return True
    if len(low) >= 2 and low[-1] == 'text' and low[0] in ('flavor', 'original', 'full', 'raw', 'long', 'short'):
        return True
    if low[-1] in ('title', 'urls'):
        return True
    return False


def is_temporal_type(typ):
    return bool(re.search(r'DATE|TIME', typ or '', re.I))


def is_flag_or_enum_int_name(name):
    low = [t.lower() for t in tokenize_name(name)]
    if set(low) & {
        'is', 'has', 'flag', 'status', 'type', 'kind', 'class',
        'gender', 'approved', 'active', 'enabled', 'yn',
    }:
        return True
    return any((low[i], low[i + 1]) == ('y', 'n') for i in range(len(low) - 1))


def _extract_public_schema(con):
    """Read ONLY public SQLite schema PRAGMAs (zero table-row access)."""
    names = [
        r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT GLOB 'sqlite_*' ORDER BY name"
        )
    ]
    raw_info = {}
    raw_unique = {}
    raw_fk_src = {}
    fk_edges = []
    fk_targets = set()

    for t in names:
        qt = quote(t)
        info = con.execute('PRAGMA table_info(' + qt + ')').fetchall()
        raw_info[t] = info
        pk_cols = [r[1] for r in info if r[5]]
        unique = set(pk_cols) if len(pk_cols) == 1 else set()
        for idx in con.execute('PRAGMA index_list(' + qt + ')').fetchall():
            if idx[2] and not idx[4]:  # unique and non-partial
                cols = con.execute('PRAGMA index_info(' + quote(idx[1]) + ')').fetchall()
                if len(cols) == 1 and cols[0][2] is not None:
                    unique.add(cols[0][2])
        raw_unique[t] = unique

        fks = con.execute('PRAGMA foreign_key_list(' + qt + ')').fetchall()
        src_cols = set()
        for fk in fks:
            dst_t, src_c, dst_c = fk[2], fk[3], fk[4]
            if src_c is not None:
                src_cols.add(src_c)
                fk_edges.append((t, src_c, dst_t, dst_c))
                if dst_t is not None and dst_c is not None:
                    fk_targets.add((dst_t, dst_c))
        raw_fk_src[t] = src_cols

    tables = {}
    for t in names:
        info = raw_info[t]
        pk_cols = {r[1] for r in info if r[5]}
        is_comp_pk = len(pk_cols) > 1
        cols = []
        for r in info:
            cname = r[1]
            ctyp = r[2] or 'ANY'
            cols.append(ColumnMeta(
                name=cname,
                declared_type=ctyp,
                is_pk=(cname in pk_cols),
                is_single_pk_or_unique=(cname in raw_unique[t]),
                is_composite_pk=(is_comp_pk and cname in pk_cols),
                is_fk_src=(cname in raw_fk_src[t]),
                is_fk_tgt=((t, cname) in fk_targets),
            ))
        tables[t] = tuple(cols)
    return PublicSchema(tables=tables, fk_edges=tuple(fk_edges))


def classify_column(col: ColumnMeta, prune_free_text: bool = False):
    """Deterministic structural role from public ColumnMeta alone."""
    typ = col.declared_type
    if (
        col.is_single_pk_or_unique
        or col.is_fk_src
        or (col.is_composite_pk and (is_integer_type(typ) or is_id_name(col.name)))
    ):
        return 'KEY_OR_FK'
    if is_ident_or_contact_name(col.name):
        return 'IDENT_OR_CONTACT'
    if not numeric(typ) and is_free_text_name(col.name):
        return 'FREE_TEXT'
    if numeric(typ):
        if is_integer_type(typ):
            return 'ENUM_INT' if is_flag_or_enum_int_name(col.name) else 'NUMERIC_INT'
        return 'NUMERIC_REAL'
    if is_temporal_type(typ):
        return 'TEMPORAL'
    return 'CATEGORICAL_TEXT'


def plan_queries_v3(table_cols, prune_free_text=False, column_policy=None, table_name=None):
    """Schema-Graph-Aware query set J_T and prior relevance weights omega_j.

    Guarantees at least one histogram per column (bins or patterns), so H_T >= |C_T| >= 1.
    Returns list of (stat, column_name_or_None, role, omega).
    Optional column_policy maps col_name (or "table.col") to an explicit role in ROLE_WEIGHTS.
    """
    queries = [('row_count', None, 'TABLE', ROW_COUNT_WEIGHT)]
    for col in table_cols:
        role = None
        if column_policy is not None:
            if table_name is not None and f"{table_name}.{col.name}" in column_policy:
                role = column_policy[f"{table_name}.{col.name}"]
            elif col.name in column_policy:
                role = column_policy[col.name]
            if role is not None and role not in ROLE_WEIGHTS:
                raise ValueError(f"Unknown role in column_policy: {role!r}")
        if role is None:
            role = classify_column(col, prune_free_text=prune_free_text)
        rw = ROLE_WEIGHTS[role]
        queries.append(('null_count', col.name, role, rw['null_count']))
        queries.append(('distinct_count', col.name, role, rw['distinct_count']))
        if numeric(col.declared_type):
            queries.append(('bins', col.name, role, rw.get('bins', rw.get('patterns', 1.0))))
            if is_integer_type(col.declared_type) and role not in ('KEY_OR_FK', 'IDENT_OR_CONTACT') and 'values' in rw:
                queries.append(('values', col.name, role, rw['values']))
        else:
            queries.append(('patterns', col.name, role, rw.get('patterns', rw.get('bins', 1.0))))
            skip_vals = role in ('KEY_OR_FK', 'IDENT_OR_CONTACT') or (prune_free_text and role == 'FREE_TEXT')
            if not skip_vals and 'values' in rw:
                queries.append(('values', col.name, role, rw['values']))
    return queries



def extract_public_schema(con):
    """Expose only catalog enumeration and four quoted metadata PRAGMAs to classification."""
    class Catalog:
        def execute(self, statement):
            catalog = ("SELECT name FROM sqlite_master WHERE type='table' "
                       "AND name NOT GLOB 'sqlite_*' ORDER BY name")
            metadata = r'PRAGMA (table_info|index_list|index_info|foreign_key_list)\("(?:[^"]|"")*"\)'
            if statement != catalog and re.fullmatch(metadata, statement) is None:
                raise RuntimeError("Planning attempted a non-metadata query")
            return con.execute(statement)

    return _extract_public_schema(Catalog())
