"""Records real Zetaris results to fixtures/recorded so SAMPLE_MODE can replay them (labelled 'recorded')."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .base import DataSource, DataSourceError, QueryResult, guard_sql, sql_hash


class RecordingDataSource(DataSource):
    def __init__(self, inner: DataSource, fixtures_dir: Path):
        if inner.kind != "live":
            raise ValueError("only live Zetaris results may be recorded as fixtures")
        self.inner = inner
        self.dir = Path(fixtures_dir) / "queries"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.name, self.label, self.kind = inner.name, inner.label, inner.kind
        self.dialect, self.dialect_notes = inner.dialect, inner.dialect_notes
        self._catalog_written = False

    def catalog(self):
        cat = self.inner.catalog()
        if not self._catalog_written:
            (self.dir.parent / "catalog.json").write_text(json.dumps(
                {"source": "zetaris", "recorded_at": _now(), "dialect": self.dialect, "dialect_notes": self.dialect_notes,
                 "catalog": cat}, indent=1))
            self._catalog_written = True
        return cat

    def describe(self, table):
        return self.inner.describe(table)

    def run(self, sql, timeout_s=60.0) -> QueryResult:
        g = guard_sql(sql, allowed_tables=self.inner.allowed_tables())  # guard errors are local and re-raise identically on replay
        h = sql_hash(g)
        try:
            res = self.inner.run(sql, timeout_s)
        except DataSourceError as e:
            # A real Zetaris error is part of the investigation (it drives schema introspection and query repair), so it
            # is recorded too: replay raises the same message and the repair prompts match the live run.
            (self.dir / f"{h.split(':')[1]}.json").write_text(json.dumps({
                "source": "zetaris", "sql": g, "sql_hash": h, "recorded_at": _now(), "error": str(e), "label": self.inner.label}, indent=1))
            raise
        (self.dir / f"{h.split(':')[1]}.json").write_text(json.dumps({
            "source": "zetaris", "sql": g, "sql_hash": h, "recorded_at": _now(), "columns": res.columns,
            "rows": res.rows, "elapsed_s": res.elapsed_s, "label": self.inner.label, "rows_total": res.meta.get("rows_total")}, indent=1, default=str))
        return res

    def close(self):
        self.inner.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
