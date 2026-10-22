"""Zetaris Lightning over JDBC (jaydebeapi + the Zetaris driver JAR). The live, primary data source."""
from __future__ import annotations

import os
import threading
from pathlib import Path

from .base import DataSource, DataSourceError, QueryResult

DRIVER_CLASS = "com.zetaris.lightning.jdbc.LightningDriver"
DEFAULT_URL = "jdbc:zetaris:lightning@localhost:10000"


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
        "Reference tables by their full two-part name exactly as listed under TABLES (e.g. NY_TAXI.taxi_zones). No semicolons. Prefer aggregated outputs (<200 rows)."
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
        self.tables_allow = {t.lower() for t in (tables or [])}
        self._catalog: list[dict] | None = None

    def _raw(self, sql: str) -> QueryResult:
        with self._lock:
            cur = self._conn.cursor()
            try:
                cur.execute(sql)
                cols = [d[0] for d in (cur.description or [])]
                rows = [list(r) for r in cur.fetchall()] if cols else []
            finally:
                try:
                    cur.close()
                except Exception:
                    pass
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
        try:
            res = self._raw(f"DESCRIBE DATASOURCE TABLE {table}")
            cols = [[str(r[0]), str(r[1]) if len(r) > 1 else ""] for r in res.rows if r and r[0] and not str(r[0]).startswith("#")]
            if cols:
                return cols
        except Exception:
            pass
        try:
            with self._lock:
                cur = self._conn.cursor()
                cur.execute(f"SELECT * FROM {table} LIMIT 1")
                cols = [[d[0], str(d[1])] for d in cur.description]
                cur.close()
            return cols
        except Exception:
            return []

    def describe(self, table: str) -> list[list[str]]:
        return self._columns(table)

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
