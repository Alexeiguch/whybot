"""DEV-ONLY data source: DuckDB over a small public NYC TLC / NOAA GHCN / Nager.Date sample.

This is NOT Zetaris. It exists so the agent loop can be developed before the Zetaris instance is up.
It is never used by SAMPLE_MODE or the demo, and every result it returns is labelled DEV-ONLY.
Fetch the files with: python scripts/fetch_dev_data.py
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from .base import DataSource, DataSourceError, QueryResult

DEV_DIR = Path(__file__).resolve().parents[2] / "data" / "dev"
STATIONS = ("USW00094728", "USW00014732", "USW00094789")  # Central Park, LaGuardia, JFK


class DevDuckDBSource(DataSource):
    name = "dev"
    label = "DEV-ONLY DuckDB sample (not Zetaris)"
    kind = "dev"
    dialect = "DuckDB SQL (dev stand-in for Zetaris Lightning)"
    dialect_notes = (
        "DuckDB functions: CAST(ts AS DATE), dayname(d) for weekday name, dayofweek(d) returns 0=Sunday..6=Saturday, "
        "strptime(CAST(x AS VARCHAR),'%Y%m%d')::DATE for yyyymmdd integers, date_diff('day',a,b), CASE WHEN. "
        "Reference tables by their full two-part name exactly as listed under TABLES (e.g. NY_TAXI.yellow_tripdata). No semicolons. Prefer aggregated outputs (<200 rows)."
    )

    def __init__(self, data_dir: Path = DEV_DIR):
        import duckdb
        d = Path(data_dir)
        parquets = sorted(d.glob("yellow_tripdata_*.parquet"))
        if not parquets:
            raise DataSourceError(f"dev data missing in {d}; run python scripts/fetch_dev_data.py")
        self.con = duckdb.connect()
        self._lock = threading.Lock()
        c = self.con
        for s in ("NY_TAXI", "NOAA_GHCN_S3", "HOLIDAYS"):
            c.execute(f"CREATE SCHEMA {s}")
        files = ", ".join(f"'{p}'" for p in parquets)
        c.execute(f"CREATE VIEW NY_TAXI.yellow_tripdata AS SELECT * FROM read_parquet([{files}])")
        c.execute(f"CREATE TABLE NY_TAXI.taxi_zones AS SELECT * FROM read_csv('{d / 'taxi_zone_lookup.csv'}')")
        st = [str(d / f"{s}.csv") for s in STATIONS if (d / f"{s}.csv").exists()]
        c.execute(f"""CREATE TABLE NOAA_GHCN_S3.noaa_ghcn_daily_2024 AS
            SELECT ID, DATE, ELEMENT, DATA_VALUE, M_FLAG, Q_FLAG, S_FLAG, OBS_TIME
            FROM read_csv({st!r}, types={{'DATE':'INTEGER','DATA_VALUE':'INTEGER','OBS_TIME':'VARCHAR','M_FLAG':'VARCHAR','Q_FLAG':'VARCHAR','S_FLAG':'VARCHAR'}})
            WHERE DATE BETWEEN 20231201 AND 20240331""")
        hol = json.loads((d / "holidays_2024.json").read_text())
        c.execute("CREATE TABLE HOLIDAYS.us_public_holidays_2024 (holiday_date DATE, local_name VARCHAR, name VARCHAR, is_nationwide BOOLEAN, counties VARCHAR, types VARCHAR)")
        c.executemany("INSERT INTO HOLIDAYS.us_public_holidays_2024 VALUES (?,?,?,?,?,?)",
                      [[h["date"], h["localName"], h["name"], bool(h["global"]), ",".join(h["counties"] or []), ",".join(h["types"] or [])] for h in hol])
        self._catalog = None

    def catalog(self):
        if self._catalog is None:
            out = []
            rows = self.con.execute("SELECT table_schema, table_name FROM information_schema.tables WHERE table_schema IN ('NY_TAXI','NOAA_GHCN_S3','HOLIDAYS') ORDER BY 1,2").fetchall()
            for s, t in rows:
                cols = self.con.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_schema=? AND table_name=? ORDER BY ordinal_position", [s, t]).fetchall()
                out.append({"table": f"{s}.{t}", "columns": [[a, b] for a, b in cols], "note": ""})
            self._catalog = out
        return self._catalog

    def _execute(self, sql):
        cur = self.con.cursor()
        try:
            cur.execute(sql)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchmany(500)
        except Exception as e:
            raise DataSourceError(f"DuckDB error: {str(e)[:600]}") from e
        finally:
            cur.close()
        return QueryResult(columns=cols, rows=[list(r) for r in rows], source_label=self.label)

    def cancel(self):
        try:
            self.con.interrupt()
        except Exception:
            pass
