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
_TABLE_REF = re.compile(r"\b(?:from|join)\s+([A-Za-z_][\w]*\.[A-Za-z_][\w]*)", re.I)


def guard_sql(sql: str, default_limit: int = 500, allowed_tables: set[str] | None = None) -> str:
    """Allow a single read-only SELECT/WITH statement over allowlisted tables; append a LIMIT when missing."""
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
        refs = {r.lower() for r in _TABLE_REF.findall(stripped)}
        bad = sorted(refs - {t.lower() for t in allowed_tables})
        if bad:
            raise DataSourceError(f"table(s) not in the evidence allowlist: {', '.join(bad)} (allowed: {', '.join(sorted(allowed_tables))})")
    if not re.search(r"(?is)\blimit\s+\d+\s*$", s):
        s = f"{s} LIMIT {default_limit}"
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

    def allowed_tables(self) -> set[str] | None:
        try:
            return {t["table"].lower() for t in self.catalog()}
        except Exception:
            return None

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
