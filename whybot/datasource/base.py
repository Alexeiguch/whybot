"""DataSource adapter contract. Every implementation runs read-only SQL and introspects its schema."""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import Any


class DataSourceError(RuntimeError):
    pass


_LIT = re.compile(r"'(?:[^']|'')*'")


def _outside_literals(sql: str, fn) -> str:
    """Apply fn to the text outside single-quoted literals only."""
    out, pos = [], 0
    for m in _LIT.finditer(sql):
        out.append(fn(sql[pos:m.start()]))
        out.append(m.group(0))
        pos = m.end()
    out.append(fn(sql[pos:]))
    return "".join(out)


def normalize_sql(sql: str) -> str:
    sql = _outside_literals(sql, lambda t: re.sub(r"--[^\n]*", " ", t))
    sql = _outside_literals(sql, lambda t: re.sub(r"\s+", " ", t))
    return sql.strip().rstrip(";").strip()


def sql_hash(sql: str) -> str:
    """Case-insensitive for keywords and identifiers, case-sensitive inside string literals."""
    return "sha256:" + hashlib.sha256(_outside_literals(normalize_sql(sql), str.lower).encode()).hexdigest()[:16]


_FORBIDDEN = re.compile(r"\b(insert|update|delete|merge|drop|create|alter|truncate|grant|revoke|cache|uncache|refresh|call|exec|copy|attach|install|load|pragma|set)\b", re.I)
_FORBIDDEN_FN = re.compile(r"\b(reflect|try_reflect|java_method|input_file_name|current_user|session_user|user|getenv|read_text|read_csv|read_parquet|read_json|system)\s*\(", re.I)
MAX_ROWS = 500
_TRAILING_LIMIT = re.compile(r"(?is)\blimit\s+(\d+)\s*$")


def referenced_tables(sql: str) -> set[str]:
    """Every table the statement reads, from a real parse (Spark SQL, which Zetaris Lightning speaks).

    Quotes and case are normalised; CTE names are not tables. Unqualified names, table functions and anything
    the parser cannot read raise, so a query the check cannot understand is refused rather than waved through.
    """
    import sqlglot
    from sqlglot import exp
    from sqlglot.errors import SqlglotError

    try:
        trees = [t for t in sqlglot.parse(sql, read="spark") if t is not None]
    except SqlglotError as e:
        raise DataSourceError(f"could not parse the query to check which tables it reads: {str(e).splitlines()[0][:200]}")
    if len(trees) != 1:
        raise DataSourceError("multiple statements are not allowed")
    tree = trees[0]
    ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    out: set[str] = set()
    for t in tree.find_all(exp.Table):
        if not isinstance(t.this, exp.Identifier):
            raise DataSourceError("table functions are not allowed; read a catalogued table by its full two-part name")
        name = t.name
        if not t.db:
            if name.lower() in ctes:
                continue
            raise DataSourceError(f"unqualified table '{name}': use the full two-part name, e.g. whybot.yellow_daily")
        out.add(".".join(p for p in (t.catalog, t.db, name) if p).lower())
    return out


def guard_sql(sql: str, default_limit: int = MAX_ROWS, allowed_tables: set[str] | None = None) -> str:
    """Allow a single read-only SELECT/WITH statement over allowlisted tables; add or clamp the LIMIT to MAX_ROWS.

    allowed_tables=None skips only the table check (used to pre-normalise SQL); DataSource.run always passes the
    catalog, and DataSource.allowed_tables raises when the catalog is unavailable, so execution fails closed.
    """
    s = normalize_sql(sql)
    stripped = _LIT.sub("''", s)
    if ";" in stripped:
        raise DataSourceError("multiple statements are not allowed")
    if not re.match(r"(?is)^\s*(select|with)\b", s):
        raise DataSourceError("only SELECT/WITH queries are allowed")
    m = _FORBIDDEN.search(stripped)
    if m:
        raise DataSourceError(f"forbidden keyword in query: {m.group(1)}")
    m = _FORBIDDEN_FN.search(stripped)
    if m:
        raise DataSourceError(f"forbidden function in query: {m.group(1)}")
    if allowed_tables is not None:
        refs = referenced_tables(s)
        bad = sorted(refs - {t.lower() for t in allowed_tables})
        if bad:
            raise DataSourceError(f"table(s) not in the evidence allowlist: {', '.join(bad)} (allowed: {', '.join(sorted(allowed_tables))})")
    cap = min(default_limit, MAX_ROWS)
    m = _TRAILING_LIMIT.search(s)
    if not m:
        s = f"{s} LIMIT {cap}"
    elif int(m.group(1)) > cap:
        s = s[:m.start(1)] + str(cap)
    return s


def _jsonable(v: Any) -> Any:
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if hasattr(v, "isoformat"):
        return v.isoformat()
    try:
        return float(v)
    except Exception:
        return str(v)


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[list[Any]]
    elapsed_s: float = 0.0
    source_label: str = ""
    recorded_at: str | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.rows = [[_jsonable(v) for v in r] for r in self.rows]

    @property
    def truncated(self) -> bool:
        return bool(self.meta.get("rows_total", 0) > len(self.rows))

    def preview(self, n: int = 12) -> str:
        head = " | ".join(self.columns)
        body = "\n".join(" | ".join(str(v) for v in r) for r in self.rows[:n])
        more = f"\n... ({len(self.rows) - n} more rows)" if len(self.rows) > n else ""
        return f"{head}\n{body}{more}"


class DataSource:
    """name: stable id; label: what the UI and traces show (LIVE / RECORDED / DEV-ONLY); dialect hints for the SQL agent."""
    name = "base"
    label = "base"
    kind = "live"  # live | recorded | dev
    dialect = "ANSI SQL"
    dialect_notes = ""

    def catalog(self) -> list[dict]:
        """[{table, columns:[[name,type],...], note}]"""
        raise NotImplementedError

    def describe(self, table: str) -> list[list[str]]:
        for t in self.catalog():
            if t["table"].lower() == table.lower():
                return t["columns"]
        raise DataSourceError(f"unknown table {table}")

    def _execute(self, sql: str) -> QueryResult:
        raise NotImplementedError

    def allowed_tables(self) -> set[str]:
        """The catalogue is the allowlist. If it cannot be read, no query runs (fail closed)."""
        try:
            tables = {t["table"].lower() for t in self.catalog()}
        except Exception as e:
            raise DataSourceError(f"catalog unavailable, so no query can be checked against the allowlist: {str(e)[:200]}")
        if not tables:
            raise DataSourceError("catalog is empty, so no query can be checked against the allowlist")
        return tables

    def run(self, sql: str, timeout_s: float = 60.0) -> QueryResult:
        sql = guard_sql(sql, allowed_tables=self.allowed_tables())
        t0 = time.time()
        ex = cf.ThreadPoolExecutor(max_workers=1)
        fut = ex.submit(self._execute, sql)
        try:
            res = fut.result(timeout=timeout_s)
        except cf.TimeoutError:
            self.cancel()
            raise DataSourceError(f"query timed out after {timeout_s:.0f}s")
        except DataSourceError:
            raise
        except Exception as e:
            raise DataSourceError(f"{type(e).__name__}: {str(e)[:500]}") from e
        finally:
            ex.shutdown(wait=False)
        res.elapsed_s = round(time.time() - t0, 3)
        res.source_label = res.source_label or self.label
        return res

    def cancel(self) -> None:
        pass

    def close(self) -> None:
        pass
