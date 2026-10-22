import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from whybot.config import Settings
from whybot.datasource.base import DataSource, DataSourceError, QueryResult
from whybot.llm import BaseLLM


class FakeLLM(BaseLLM):
    """Scripted LLM: per role, a list of responses (str or dict, or callable(messages)->str) consumed in order."""

    def __init__(self, script: dict):
        super().__init__()
        self.script = {k: list(v) for k, v in script.items()}
        self.calls: list[tuple[str, list]] = []

    def chat(self, role, messages, *, max_tokens=4096, temperature=0.2):
        self.calls.append((role, messages))
        queue = self.script.get(role)
        if not queue:
            raise AssertionError(f"FakeLLM: no scripted response left for role {role}")
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if callable(item):
            item = item(messages)
        return (json.dumps(item) if isinstance(item, dict) else item), {"model": f"fake-{role}", "latency_s": 0}


class FakeSource(DataSource):
    name, label, kind, dialect = "fake", "FAKE test source", "dev", "DuckDB SQL"

    def __init__(self, routes):
        """routes: list of (substring, QueryResult | Exception)."""
        self.routes = routes
        self.executed: list[str] = []

    def catalog(self):
        return [
            {"table": "NY_TAXI.yellow_tripdata", "columns": [["tpep_pickup_datetime", "TIMESTAMP"], ["fare_amount", "DOUBLE"]], "note": ""},
            {"table": "NOAA_GHCN_S3.noaa_ghcn_daily_2024", "columns": [["ID", "VARCHAR"], ["DATE", "INTEGER"], ["ELEMENT", "VARCHAR"], ["DATA_VALUE", "INTEGER"]], "note": ""},
        ]

    def _execute(self, sql):
        self.executed.append(sql)
        for sub, res in self.routes:
            if sub.lower() in sql.lower():
                if isinstance(res, Exception):
                    raise res
                return res
        raise DataSourceError(f"no route for {sql[:80]}")


@pytest.fixture
def settings(tmp_path):
    s = Settings()
    s.logs_dir = tmp_path / "logs"
    s.fixtures_dir = tmp_path / "fixtures"
    s.meterless_enabled = False
    s.max_iterations = 3
    s.run_timeout_s = 60
    return s


def QR(cols, rows):
    return QueryResult(columns=cols, rows=rows)
