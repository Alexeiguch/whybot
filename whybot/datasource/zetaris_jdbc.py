"""Zetaris Lightning over JDBC (jaydebeapi + the Zetaris driver JAR). The live, primary data source."""
from __future__ import annotations

import os
import re
import threading
from pathlib import Path

from .base import DataSource, DataSourceError, QueryResult

DRIVER_CLASS = "com.zetaris.lightning.jdbc.LightningDriver"
DEFAULT_URL = "jdbc:zetaris:lightning@localhost:10000"


def to_python(v):
    """Turn JPype boxed Java values into plain Python ones.

    jaydebeapi returns BOOLEAN and BIGINT columns as java.lang.Boolean / java.lang.Long. Those
    subclass int, so str() gives "True" live but json.dumps() writes 1, and a replayed prompt
    would differ from the recorded one. Normalising here keeps live, recorded and replayed rows identical.
    """
    if v is None or type(v) in (bool, int, float, str):
        return v
    name = type(v).__name__
    if name.endswith("Boolean"):
        return bool(v)
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float):
        return float(v)
    if name.endswith(("Long", "Integer", "Short", "Byte")):
        return int(v)
    if name.endswith(("Double", "Float")):
        return float(v)
    return str(v)


def _find_java_home() -> str | None:
    if os.environ.get("JAVA_HOME") and Path(os.environ["JAVA_HOME"]).is_dir():
        return os.environ["JAVA_HOME"]
    for cand in sorted(Path("/usr/lib/jvm").glob("*"), reverse=True) if Path("/usr/lib/jvm").is_dir() else []:
        if (cand / "bin" / "java").exists():
            return str(cand)
    return None


class ZetarisJDBCSource(DataSource):
    name = "zetaris"
    label = "LIVE Zetaris Lightning (JDBC)"
    kind = "live"
    dialect = "Zetaris Lightning SQL (Spark SQL dialect)"
    dialect_notes = (
        "Spark SQL functions: to_date(ts), date_format(d,'EEEE') for weekday name, dayofweek(d) returns 1=Sunday..7=Saturday, "
        "to_date(CAST(x AS STRING),'yyyyMMdd') for yyyymmdd integers, date_add/datediff, CASE WHEN. "
        "Reference tables by their full two-part name exactly as listed under TABLES (e.g. whybot.yellow_daily). No semicolons. Prefer aggregated outputs (<200 rows). "
        "ZETARIS QUIRKS (verified): (1) never read the same whybot.* view twice in one query (no self-joins, no two CTEs over the same view): "
        "it fails with MISSING_ATTRIBUTES; compare target vs baseline with conditional aggregation instead, e.g. "
        "MAX(CASE WHEN trip_date = DATE '2024-11-28' THEN trips END) AS target_trips, AVG(CASE WHEN trip_date <> DATE '2024-11-28' THEN trips END) AS baseline_avg_trips, "
        "or simply return one row per date. Joining two DIFFERENT views on the date key is fine. "
        "(2) Compare dates with DATE 'yyyy-mm-dd' literals, never integers. date_add(d, n) takes 2 arguments. "
        "(3) An error like \"mismatched input ... expecting {'ADD', 'ANALYZE', ...}\" or \"expecting 'METADATA'\" means the statement failed to parse or resolve: "
        "simplify it, use only listed column names, and avoid short table aliases at the start of the select list."
    )

    def __init__(self, databases: list[str], tables: list[str] | None = None):
        self.url = os.environ.get("ZETARIS_JDBC_URL", DEFAULT_URL)
        self.jar = os.environ.get("ZETARIS_JDBC_JAR", "")
        self.driver = os.environ.get("ZETARIS_JDBC_CLASS", DRIVER_CLASS)
        user, pwd = os.environ.get("ZETARIS_USERNAME"), os.environ.get("ZETARIS_PASSWORD")
        if not (user and pwd):
            raise DataSourceError("ZETARIS_USERNAME / ZETARIS_PASSWORD not set (put them in .env.local)")
        if not self.jar or not Path(self.jar).is_file():
            raise DataSourceError("ZETARIS_JDBC_JAR is not set or the driver JAR does not exist")
        home = _find_java_home()
        if not home:
            raise DataSourceError("no JDK/JRE found; set JAVA_HOME")
        os.environ["JAVA_HOME"] = home
        try:
            import jaydebeapi
        except ImportError as e:
            raise DataSourceError("jaydebeapi is not installed") from e
        try:
            self._conn = jaydebeapi.connect(self.driver, self.url, [user, pwd], self.jar)
        except Exception as e:
            raise DataSourceError(f"JDBC connect failed: {type(e).__name__}: {str(e)[:300]}") from e
        self._lock = threading.Lock()
        self.databases = databases
        self.tables = list(tables or [])
        self.tables_allow = {t.lower() for t in self.tables}
        self._active = None
        self._catalog: list[dict] | None = None

    LOCK_TIMEOUT_S = 120.0

    def _acquire(self):
        if not self._lock.acquire(timeout=self.LOCK_TIMEOUT_S):
            raise DataSourceError("Zetaris session busy: a previous query is still running (lock timeout)")

    def _raw(self, sql: str) -> QueryResult:
        self._acquire()
        try:
            cur = self._conn.cursor()
            self._active = cur
            try:
                cur.execute(sql)
                cols = [d[0] for d in (cur.description or [])]
                rows = [[to_python(v) for v in r] for r in cur.fetchall()] if cols else []
            finally:
                self._active = None
                try:
                    cur.close()
                except Exception:
                    pass
        finally:
            self._lock.release()
        return QueryResult(columns=cols, rows=rows[:500], source_label=self.label, meta={"rows_total": len(rows)})

    def _execute(self, sql: str) -> QueryResult:
        try:
            return self._raw(sql)
        except Exception as e:
            raise DataSourceError(f"Zetaris error: {str(e)[:600]}") from e

    def catalog(self) -> list[dict]:
        if self._catalog is not None:
            return self._catalog
        out = []
        if self.tables:  # explicit allowlist (incl. SchemaStore views such as whybot.yellow_daily)
            for t in self.tables:
                out.append({"table": t, "columns": self._columns(t), "note": ""})
            self._catalog = out
            return out
        for db in self.databases:
            try:
                res = self._raw(f"SHOW LIGHTNING TABLES {db}")
            except Exception as e:
                out.append({"table": f"{db}.*", "columns": [], "note": f"introspection failed: {str(e)[:200]}"})
                continue
            name_idx = 0
            for i, c in enumerate(res.columns):
                if c.lower() in ("tablename", "table_name", "name", "table"):
                    name_idx = i
            for row in res.rows:
                t = f"{db}.{row[name_idx]}"
                if self.tables_allow and t.lower() not in self.tables_allow:
                    continue
                out.append({"table": t, "columns": self._columns(t), "note": ""})
        self._catalog = out
        return out

    def _columns(self, table: str) -> list[list[str]]:
        if not re.match(r"^[A-Za-z_]\w*\.[A-Za-z_]\w*$", table):
            return []
        try:
            res = self._raw(f"DESCRIBE DATASOURCE TABLE {table}")
            cols = [[str(r[0]), (str(r[1]) if len(r) > 1 else "").strip('"')] for r in res.rows if r and r[0] and not str(r[0]).startswith("#")]
            if cols:
                return cols
        except Exception:
            pass
        try:
            res = self._raw(f"SELECT * FROM {table} LIMIT 1")
            return [[c, ""] for c in res.columns]
        except Exception:
            return []

    def describe(self, table: str) -> list[list[str]]:
        return self._columns(table)

    def cancel(self) -> None:
        """Best effort: cancel the running JDBC statement so the session lock is released after a timeout."""
        cur = self._active
        try:
            if cur is not None and getattr(cur, "_prep", None) is not None:
                cur._prep.cancel()
        except Exception:
            pass

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
