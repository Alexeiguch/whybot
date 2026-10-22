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


def normalize_sql(sql: str) -> str:
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"\s+", " ", sql).strip().rstrip(";").strip()
    return sql


def sql_hash(sql: str) -> str:
    return "sha256:" + hashlib.sha256(normalize_sql(sql).lower().encode()).hexdigest()[:16]


_FORBIDDEN = re.compile(r"\b(insert|update|delete|merge|drop|create|alter|truncate|grant|revoke|cache|uncache|refresh|call|exec|copy|attach|install|load|pragma|set)\b", re.I)


def guard_sql(sql: str, default_limit: int = 500) -> str:
    """Allow a single read-only SELECT/WITH statement; append a LIMIT when missing."""
    s = normalize_sql(sql)
    if ";" in s:
        raise DataSourceError("multiple statements are not allowed")
    if not re.match(r"(?is)^\s*(select|with)\b", s):
        raise DataSourceError("only SELECT/WITH queries are allowed")
    stripped = re.sub(r"'[^']*'", "''", s)
    m = _FORBIDDEN.search(stripped)
    if m:
        raise DataSourceError(f"forbidden keyword in query: {m.group(1)}")
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

    def run(self, sql: str, timeout_s: float = 60.0) -> QueryResult:
        sql = guard_sql(sql)
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
