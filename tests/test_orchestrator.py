from pathlib import Path
"""Branching behaviour of the full loop with a scripted fake LLM and a fake data source (no network, no Meterless)."""
import json

from conftest import QR, FakeLLM, FakeSource

from whybot.datasource.base import DataSourceError, sql_hash
from whybot.orchestrator import WhyBot

PLAN = {"metric": "daily trips", "target_window": "2024-02-13", "baseline_window": "other Feb Tuesdays",
        "hypotheses": [
            {"id": "H0", "kind": "metric", "statement": "trips fell", "support_if": "lower", "refute_if": "same"},
            {"id": "H1", "kind": "weather", "statement": "snow", "support_if": "snow>0", "refute_if": "no snow"},
            {"id": "H2", "kind": "holiday", "statement": "holiday", "support_if": "holiday", "refute_if": "none"},
            {"id": "H3", "kind": "data_quality", "statement": "missing data", "support_if": "gaps", "refute_if": "no gaps"}],
        "queries": [{"hypothesis_id": "H0", "purpose": "trips per Tuesday", "tables": ["NY_TAXI.yellow_tripdata"]},
                    {"hypothesis_id": "H1", "purpose": "snow per Tuesday", "tables": ["NOAA_GHCN_S3.noaa_ghcn_daily_2024"]}]}
SQL_TRIPS = {"sql": "SELECT trip_date, trips FROM NY_TAXI.yellow_tripdata", "tables_used": ["NY_TAXI.yellow_tripdata"]}
SQL_SNOW_BAD = {"sql": "SELECT dt, snow FROM NOAA_GHCN_S3.noaa_ghcn_daily_2024", "tables_used": ["NOAA_GHCN_S3.noaa_ghcn_daily_2024"]}
SQL_SNOW_OK = {"sql": "SELECT DATE, DATA_VALUE AS snow_mm FROM NOAA_GHCN_S3.noaa_ghcn_daily_2024 WHERE ELEMENT='SNOW'", "tables_used": ["NOAA_GHCN_S3.noaa_ghcn_daily_2024"]}
CLAIMS = {"claims": [
    {"cid": "C1", "hypothesis_id": "H0", "statement": "13 Feb had 69067 trips vs 102152", "direction": "supports",
     "cited_values": [{"qid": "Q1", "column": "trips", "value": 69067, "row_hint": "trip_date=2024-02-13"}]},
    {"cid": "C2", "hypothesis_id": "H1", "statement": "81 mm snow on 13 Feb", "direction": "supports",
     "cited_values": [{"qid": "Q2", "column": "snow_mm", "value": 81, "row_hint": "DATE=20240213"}]}]}


def critic(decision, verdict="supported", hyp="supported", follow=False):
    return {"verdicts": [{"cid": "C1", "verdict": "supported", "strength": 0.9, "reason": "ok"},
                         {"cid": "C2", "verdict": verdict, "strength": 0.8, "reason": "r"}],
            "hypothesis_status": [{"hypothesis_id": "H0", "status": "supported", "reason": "r"},
                                  {"hypothesis_id": "H1", "status": hyp, "reason": "r"}],
            "decision": decision, "feedback": "check snow on baseline days too",
            "follow_up_queries": [{"hypothesis_id": "H1", "purpose": "snow on all Feb Tuesdays", "tables": ["NOAA_GHCN_S3.noaa_ghcn_daily_2024"]}] if follow else []}


def routes():
    return [("select trip_date", QR(["trip_date", "trips"], [["2024-02-06", 102152], ["2024-02-13", 69067]])),
            ("select dt, snow", DataSourceError('Binder Error: column "dt" not found')),
            ("element='snow'", QR(["DATE", "snow_mm"], [[20240206, 0], [20240213, 81]]))]


def writer_for(qids):
    def w(messages):
        return json.dumps({"headline": "Snowstorm cut trips", "answer": "a", "recommendation": "r", "confidence": 0.8,
                           "claims": [{"text": "t", "citations": [{"qid": q, "sql_hash": h, "table": t} for q, h, t in qids]}],
                           "caveats": [], "lessons": ["filter one GHCN station"]})
    return w


def events(res, settings):
    return [json.loads(l) for l in (settings.logs_dir / f"{res['run_id']}.jsonl").read_text().splitlines()]


def test_sql_repair_then_critic_retry_then_accept(settings):
    h1, h2 = sql_hash(SQL_TRIPS["sql"] + " LIMIT 500"), sql_hash(SQL_SNOW_OK["sql"] + " LIMIT 500")
    llm = FakeLLM({
        "planner": [PLAN, {**PLAN, "queries": [PLAN["queries"][1]]}],
        "sql": [SQL_TRIPS, SQL_SNOW_BAD, SQL_SNOW_OK, SQL_SNOW_OK],
        "analyst": [CLAIMS],
        "critic": [critic("retry", verdict="weak", hyp="inconclusive", follow=True), critic("accept")],
        "writer": [writer_for([("Q1", h1, "NY_TAXI.yellow_tripdata"), ("Q2", h2, "NOAA_GHCN_S3.noaa_ghcn_daily_2024")])],
    })
    res = WhyBot(settings, llm=llm, datasource=FakeSource(routes()), use_meterless=False, echo=False).run("why?")
    assert res["status"] == "answered", res
    assert res["iterations"] == 2
    acts = [(e["agent_name"], e["action"]) for e in events(res, settings)]
    assert ("Executor", "query_failed") in acts and ("Executor", "schema_introspect_for_repair") in acts
    branches = [e["output_summary"] for e in events(res, settings) if e["action"] == "branch"]
    assert branches[0].startswith("retry") and branches[-1].startswith("write")
    q2 = next(e for e in res["evidence"] if e["qid"] == "Q2")
    assert q2["status"] == "ok" and len(q2["attempts"]) == 2
    assert all(c["citations"] for c in res["output"]["claims"])


def test_escalates_to_insufficient_evidence_after_three_iterations(settings):
    llm = FakeLLM({
        "planner": [PLAN, {**PLAN, "queries": [PLAN["queries"][0]]}, {**PLAN, "queries": [PLAN["queries"][1]]}],
        "sql": [SQL_TRIPS, SQL_SNOW_OK],
        "analyst": [CLAIMS],
        "critic": [critic("retry", verdict="weak", hyp="inconclusive", follow=True)],
        "writer": [writer_for([])],
    })
    llm.script["writer"] = [lambda m: json.dumps({"headline": "Insufficient evidence", "answer": "a", "recommendation": "r",
                                                  "confidence": 0.2, "claims": [], "caveats": ["x"]})]
    res = WhyBot(settings, llm=llm, datasource=FakeSource(routes()), use_meterless=False, echo=False).run("why?")
    assert res["status"] == "insufficient_evidence"
    assert res["iterations"] == 3
    acts = [e["action"] for e in events(res, settings)]
    assert "escalate" in acts and acts.count("plan") == 3
    # iteration 2's first plan ignored the Critic's follow-up for H1 and was sent back to the Planner
    assert any(e["action"] == "json_retry" and "follow-up test for H1" in e["output_summary"] for e in events(res, settings))


def test_writer_bad_citation_is_retried_with_feedback(settings):
    h1, h2 = sql_hash(SQL_TRIPS["sql"] + " LIMIT 500"), sql_hash(SQL_SNOW_OK["sql"] + " LIMIT 500")
    llm = FakeLLM({
        "planner": [PLAN], "sql": [SQL_TRIPS, SQL_SNOW_OK], "analyst": [CLAIMS], "critic": [critic("accept")],
        "writer": [writer_for([("Q1", "sha256:made-up", "NY_TAXI.yellow_tripdata")]), writer_for([("Q1", h1, "NY_TAXI.yellow_tripdata")])],
    })
    res = WhyBot(settings, llm=llm, datasource=FakeSource(routes()), use_meterless=False, echo=False).run("why?")
    assert res["status"] == "answered"
    assert res["output"]["claims"][0]["citations"][0]["sql_hash"] == h1
    retry = [e for e in events(res, settings) if e["agent_name"] == "Writer" and e["action"] == "json_retry"]
    assert retry and "sql_hash" in retry[0]["output_summary"]


def test_failing_query_falls_back_and_run_continues(settings):
    settings.max_iterations = 1
    llm = FakeLLM({"planner": [PLAN], "sql": [SQL_TRIPS, SQL_SNOW_BAD], "analyst": [{"claims": [CLAIMS["claims"][0]]}],
                   "critic": [{**critic("accept"), "verdicts": [{"cid": "C1", "verdict": "supported", "strength": 0.9, "reason": "ok"}],
                               "hypothesis_status": [{"hypothesis_id": "H0", "status": "supported", "reason": "r"}]}],
                   "writer": [writer_for([])]})
    res = WhyBot(settings, llm=llm, datasource=FakeSource(routes()), use_meterless=False, echo=False).run("why?")
    q2 = next(e for e in res["evidence"] if e["qid"] == "Q2")
    assert q2["status"] == "failed"  # repair re-proposed the identical SQL, which the SQL validator refuses
    assert "fallback" in [e["action"] for e in events(res, settings)]
    assert res["status"] == "insufficient_evidence"  # only H0 quantified, no cause supported


def test_planner_invalid_tables_are_rejected_and_retried(settings):
    bad = {**PLAN, "queries": [{"hypothesis_id": "H0", "purpose": "x", "tables": ["NY_TAXI.does_not_exist"]}]}
    llm = FakeLLM({"planner": [bad, PLAN], "sql": [SQL_TRIPS, SQL_SNOW_OK], "analyst": [CLAIMS], "critic": [critic("accept")],
                   "writer": [writer_for([("Q1", sql_hash(SQL_TRIPS["sql"] + " LIMIT 500"), "NY_TAXI.yellow_tripdata")])]})
    res = WhyBot(settings, llm=llm, datasource=FakeSource(routes()), use_meterless=False, echo=False).run("why?")
    assert res["status"] == "answered"
    assert any(e["action"] == "json_retry" and "not in the catalog" in e["output_summary"] for e in events(res, settings))


def test_no_secrets_in_trace(settings, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-SUPERSECRETVALUE123")
    llm = FakeLLM({"planner": ["not json nvapi-SUPERSECRETVALUE123", PLAN], "sql": [SQL_TRIPS, SQL_SNOW_OK], "analyst": [CLAIMS],
                   "critic": [critic("accept")], "writer": [writer_for([("Q1", sql_hash(SQL_TRIPS["sql"] + " LIMIT 500"), "NY_TAXI.yellow_tripdata")])]})
    res = WhyBot(settings, llm=llm, datasource=FakeSource(routes()), use_meterless=False, echo=False).run("why?")
    assert "SUPERSECRETVALUE" not in (settings.logs_dir / f"{res['run_id']}.jsonl").read_text()


def test_structured_error_when_source_unavailable(settings):
    settings.data_source = "replay"  # no fixtures in tmp dir
    llm = FakeLLM({"planner": [PLAN]})
    res = WhyBot(settings, llm=llm, use_meterless=False, echo=False).run("why?")
    assert res["status"] == "error" and res["error"]["recoverable"] is True


def test_holiday_data_quality_check_flags_disagreement_as_note(settings):
    from conftest import QR as _QR
    from whybot.orchestrator import WhyBot as _WB
    from whybot.schemas import Plan
    from whybot.trace import Tracer

    class DQSource(FakeSource):
        def catalog(self):
            return super().catalog() + [{"table": "TPCDS_DB.date_dim", "columns": [], "note": ""},
                                        {"table": "whybot.us_holidays", "columns": [], "note": ""}]
    ds = DQSource([("tpcds_db.date_dim d left join", _QR(["cal_date", "tpcds_d_holiday", "nager_holiday"],
                                                          [["2024-11-28", "N", "Thanksgiving Day"], ["2024-12-25", "Y", "Christmas Day"], ["2024-12-31", "Y", None]]))])
    ev = {}
    tr = Tracer("dq", settings.logs_dir, echo=False)
    p = Plan(metric="m", target_window="2024-11-28", baseline_window="b", hypotheses=[], queries=[])
    notes = _WB(settings, datasource=ds, use_meterless=False)._holiday_dq_check("why 28 Nov 2024", p, [c["table"] for c in ds.catalog()], ds, tr, ev)
    assert len(notes) == 2 and "Thanksgiving" in notes[0] and "2024-12-31" in notes[1]
    assert ev["DQ1"].hypothesis_id == "data_quality"


def test_invalid_run_id_is_rejected(settings):
    import pytest
    with pytest.raises(ValueError):
        WhyBot(settings, llm=FakeLLM({}), datasource=FakeSource([]), use_meterless=False, echo=False).run("q", run_id="../escape")


def test_reused_run_id_never_overwrites_an_existing_trace(settings):
    import pytest
    from whybot.runid import RunIdError

    def bot():
        llm = FakeLLM({"planner": [PLAN], "sql": [SQL_TRIPS, SQL_SNOW_OK], "analyst": [CLAIMS], "critic": [critic("accept")],
                       "writer": [writer_for([("Q1", sql_hash(SQL_TRIPS["sql"] + " LIMIT 500"), "NY_TAXI.yellow_tripdata")])]})
        return WhyBot(settings, llm=llm, datasource=FakeSource(routes()), use_meterless=False, echo=False)
    res = bot().run("why?", run_id="same")
    before = (Path(settings.logs_dir) / "same.jsonl").read_bytes()
    with pytest.raises(RunIdError, match="already exists"):
        bot().run("why?", run_id="same")
    assert (Path(settings.logs_dir) / "same.jsonl").read_bytes() == before
    assert sum(1 for e in events(res, settings) if e["action"] == "start") == 1


def test_reserved_and_malformed_run_ids_are_refused(settings):
    import pytest
    from whybot.runid import RunIdError

    for bad in ("recorded-example-1-thanksgiving", "Recorded-x", "../x", "a/b", "x..y", ".hidden", "a" * 90):
        with pytest.raises(RunIdError):
            WhyBot(settings, llm=FakeLLM({}), datasource=FakeSource(routes()), use_meterless=False, echo=False).run("why?", run_id=bad)
    assert not list(Path(settings.logs_dir).glob("recorded-*"))


def test_only_a_live_recording_may_replace_a_recorded_trace(tmp_path):
    import pytest
    from whybot.runid import RunIdError
    from whybot.trace import Tracer

    (tmp_path / "recorded-x.jsonl").write_text("live trace\n")
    with pytest.raises(RunIdError):
        Tracer("recorded-x", tmp_path, echo=False)
    assert (tmp_path / "recorded-x.jsonl").read_text() == "live trace\n"
    Tracer("recorded-x", tmp_path, echo=False, recording=True)  # scripts/record_fixtures.py re-recording
    assert (tmp_path / "recorded-x.jsonl").read_text() == ""


def test_failed_follow_up_plan_finishes_with_verified_claims(settings):
    llm = FakeLLM({"planner": [PLAN, "not json"], "sql": [SQL_TRIPS, SQL_SNOW_OK], "analyst": [CLAIMS],
                   "critic": [critic("retry")],
                   "writer": [writer_for([("Q1", sql_hash(SQL_TRIPS["sql"] + " LIMIT 500"), "NY_TAXI.yellow_tripdata")])]})
    res = WhyBot(settings, llm=llm, datasource=FakeSource(routes()), use_meterless=False, echo=False).run("why?")
    assert res["status"] != "error"
    assert any(e["action"] == "failed" and e["agent_name"] == "Planner" for e in events(res, settings))
