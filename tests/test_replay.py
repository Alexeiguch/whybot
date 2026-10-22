import json

import pytest

from whybot.datasource.base import DataSourceError, guard_sql, sql_hash
from whybot.datasource.replay import ReplayDataSource


def write_fixtures(d, source="zetaris"):
    (d / "queries").mkdir(parents=True)
    (d / "catalog.json").write_text(json.dumps({"source": source, "recorded_at": "2026-10-23T10:00:00+00:00", "catalog": [
        {"table": "NY_TAXI.taxi_zones", "columns": [["LocationID", "int"]], "note": ""}]}))
    sql = guard_sql("SELECT count(*) AS n FROM NY_TAXI.taxi_zones")
    (d / "queries" / "a.json").write_text(json.dumps({"source": source, "sql": sql, "sql_hash": sql_hash(sql),
                                                      "recorded_at": "2026-10-23T10:00:00+00:00", "columns": ["n"], "rows": [[265]]}))


def test_replay_hit_is_labelled_recorded(tmp_path):
    write_fixtures(tmp_path)
    ds = ReplayDataSource(tmp_path)
    res = ds.run("select   count(*) as n from NY_TAXI.taxi_zones")
    assert res.rows == [[265]] and res.source_label.startswith("RECORDED zetaris") and res.recorded_at
    assert ds.kind == "recorded"


def test_replay_miss_lists_recorded_queries(tmp_path):
    write_fixtures(tmp_path)
    with pytest.raises(DataSourceError, match="REPLAY MISS"):
        ReplayDataSource(tmp_path).run("SELECT 1 FROM NY_TAXI.taxi_zones")


def test_replay_refuses_non_zetaris_fixtures(tmp_path):
    write_fixtures(tmp_path, source="dev")
    with pytest.raises(DataSourceError):
        ReplayDataSource(tmp_path)
