from __future__ import annotations

from ..config import Settings
from .base import DataSource, DataSourceError


def make_datasource(s: Settings) -> DataSource:
    if s.data_source == "replay":
        from .replay import ReplayDataSource
        return ReplayDataSource(s.fixtures_dir)
    if s.data_source == "zetaris":
        from .zetaris_jdbc import ZetarisJDBCSource
        src = ZetarisJDBCSource(s.zetaris_databases, s.zetaris_tables)
        if s.record_fixtures:
            from .recording import RecordingDataSource
            src = RecordingDataSource(src, s.fixtures_dir)
        return src
    if s.data_source == "dev":
        from .dev_duckdb import DevDuckDBSource
        return DevDuckDBSource()
    raise DataSourceError(f"unknown DATA_SOURCE {s.data_source!r} (zetaris | replay | dev)")
