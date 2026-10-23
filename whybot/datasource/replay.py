"""ReplayDataSource: serves RECORDED real Zetaris query results from fixtures (SAMPLE_MODE=true). Never invents data."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .base import DataSource, DataSourceError, QueryResult, guard_sql, sql_hash


class ReplayDataSource(DataSource):
    name = "replay"
    kind = "recorded"

    def __init__(self, fixtures_dir: Path, allow_sources: tuple[str, ...] = ("zetaris",)):
        self.dir = Path(fixtures_dir)
        cat_path = self.dir / "catalog.json"
        if not cat_path.is_file():
            raise DataSourceError(f"no recorded catalog at {cat_path}; record fixtures from live Zetaris first (scripts/record_fixtures.py)")
        cat = json.loads(cat_path.read_text())
        if cat.get("source") not in allow_sources:
            raise DataSourceError(f"fixtures source '{cat.get('source')}' is not allowed for replay")
        self.src = cat["source"]
        self.label = f"RECORDED {self.src} results (replay, recorded {cat.get('recorded_at', '?')})"
        self.dialect = cat.get("dialect", "Zetaris Lightning SQL (Spark SQL dialect)")
        # The SQL agent sees the same notes as in the live run, so replayed prompts hash identically (prompt_match).
        # Only when a live LLM writes SQL against recorded results (SAMPLE_LLM=live) does it need the replay hint.
        self.dialect_notes = cat.get("dialect_notes", "")
        if os.environ.get("SAMPLE_LLM", "replay") == "live":
            self.dialect_notes += " REPLAY MODE: only previously recorded queries can be answered."
        self.table_notes = cat.get("table_notes")  # the catalog notes the recorded run was given (None: use the current file)
        self._catalog = cat["catalog"]
        self._fx: dict[str, dict] = {}
        for p in sorted((self.dir / "queries").glob("*.json")):
            fx = json.loads(p.read_text())
            if fx.get("source") in allow_sources:
                self._fx[fx["sql_hash"]] = fx

    def catalog(self):
        return self._catalog

    def recorded_sql(self, tables: list[str] | None = None) -> list[str]:
        out = []
        for fx in self._fx.values():
            if fx.get("error"):
                continue
            if not tables or any(t.lower() in fx["sql"].lower() for t in tables):
                out.append(fx["sql"])
        return out

    def run(self, sql, timeout_s=60.0) -> QueryResult:
        g = guard_sql(sql, allowed_tables=self.allowed_tables())
        fx = self._fx.get(sql_hash(g))
        if not fx:
            options = self.recorded_sql()[:8]
            raise DataSourceError("REPLAY MISS: this exact SQL was not recorded from Zetaris. Re-issue one of the recorded "
                                  "queries verbatim if it fits the purpose:\n" + "\n".join(f"- {s}" for s in options))
        if fx.get("error"):  # a recorded live Zetaris error: replay it verbatim
            raise DataSourceError(fx["error"])
        return QueryResult(columns=fx["columns"], rows=fx["rows"], elapsed_s=fx.get("elapsed_s", 0.0),
                           source_label=self.label, recorded_at=fx.get("recorded_at"), meta={"replayed": True})
