"""Checks v7 and the faster run. Fixture rows, SQL and claim text are copied from the live Zetaris run
run-20261028T184140Z-b95249 (Thanksgiving, Thu 28 Nov 2024): correct facts, but the Writer was rejected three times and
fell back (confidence 0.30), Q1 failed three times on an unclosed ROUND(, and the run took 379 s."""
import json
import time

from conftest import QR, FakeLLM, FakeSource

from whybot.checks import (CHECKS_VERSION, blank_day_lists, check_aggregate_labels, check_claim_numbers,
                           check_measure_wording, sql_parse_problems)
from whybot.datasource.base import sql_hash
from whybot.evidence import Evidence
from whybot.orchestrator import WhyBot
from whybot.schemas import Claim


def _ev(qid, columns, rows, tables=("whybot.yellow_daily",)):
    return Evidence(qid=qid, hypothesis_id="H", purpose="p", iteration=1, status="ok", sql_hash=f"sha256:{qid}",
                    columns=columns, rows=rows, row_count=len(rows), tables=list(tables))


Q2 = _ev("Q2", ["trip_date", "trips", "d_day_name"],
         [["2024-11-07", 130018, "Thursday"], ["2024-11-14", 141642, "Thursday"], ["2024-11-21", 153149, "Thursday"],
          ["2024-11-28", 77244, "Thursday"]])
Q3 = _ev("Q3", ["trip_date", "prcp_mm", "tmax_c", "tmin_c"],
         [["2024-11-07", 0.0, 22.8, 12.2], ["2024-11-14", 0.0, 8.3, 3.9], ["2024-11-21", 39.9, 12.2, 5.0],
          ["2024-11-28", 21.8, 9.4, 5.6]], tables=("whybot.nyc_weather_daily",))
Q5 = _ev("Q5", ["trip_date", "target_baseline_avg_trip_distance", "target_baseline_avg_total_amount", "day_name"],
         [["2024-11-07", 3.249533679952007, 28.344273792857695, "Thursday"],
          ["2024-11-14", 5.088759831123541, 28.70524837265613, "Thursday"],
          ["2024-11-21", 4.552283854285692, 28.761140849758707, "Thursday"],
          ["2024-11-28", 9.471360752938693, 25.322095955671823, "Thursday"]])
Q6 = _ev("Q6", ["trip_date", "trips"], [["2024-11-28", 77244]])
EV = {q.qid: q for q in (Q2, Q3, Q5, Q6)}
DQ = ["2024-11-28: TPCDS date_dim.d_holiday=N but Nager.Date says Thanksgiving Day (national holiday) [DQ1 sha256:ff7eddb0f4eb3255]"]


def agg(text, qids=None, writer=True):
    return check_aggregate_labels(text, EV, set(qids or EV), version=7, for_writer=writer)


def test_version_is_7():
    assert CHECKS_VERSION >= 7


# ---- Writer: the b95249 drafts that were rejected although every number was a real cell

def test_per_day_values_listed_after_baseline_label_pass():
    assert agg("baseline averages of 3.25 (Nov 7), 5.09 (Nov 14), 4.55 (Nov 21)") == []
    assert agg("avg total amount was 25.32 vs baseline Thursdays of 28.34, 28.71, 28.76") == []
    assert agg("avg distance 9.47 mi vs 3.25\u20135.09 mi baseline") == []


def test_named_baseline_day_and_extreme_pass():
    assert agg("Weather was not uniquely adverse; baseline Nov 21 had 39.9 mm precipitation") == []
    assert agg("lower than on 21 Nov (39.9 mm), the highest baseline trips of 153,149") == []


def test_single_day_still_rejected_as_an_average():
    # 141,642 is 14 Nov's count; the mean of the three Thursdays is 141,603
    assert agg("Trips fell to 77,244 vs baseline average of 141,642")
    assert agg("from a baseline of 28.71 to 25.32")
    # a dated cell called an average is still an average claim
    assert agg("the baseline average of 21 Nov was 39.9 mm")


def test_writer_message_names_no_uncitable_number():
    (p,) = agg("Trips fell to 77,244 vs baseline average of 141,642")
    assert "nearest aggregate" not in p and "141,603" not in p
    # the analyst still gets the hint (it can declare a derived mean)
    assert "nearest aggregate" in agg("Trips fell to 77,244 vs baseline average of 141,642", writer=False)[0]


def test_number_from_an_unsupported_query_is_named():
    (p, *_) = agg("avg distance 9.47 mi vs 3.25\u20135.09 mi baseline", qids={"Q2", "Q3"})
    assert "comes from Q5" in p


def test_slash_day_list_is_a_date_list():
    assert blank_day_lists("cooler than Nov 7/14", {(11, 7), (11, 14)}, slash=True) == "cooler than Nov 7   "
    assert "99" in blank_day_lists("on 2024-11-07/14/99", {(11, 7), (11, 14)}, slash=True)


# ---- Analyst wording and DQ consistency at claim level

C5 = ("On 2024-11-28, avg_trip_distance=9.47 and avg_total_amount=25.32 vs baseline Thursdays avg_trip_distance of 3.25 "
      "(2024-11-07), 5.09 (2024-11-14), 4.55 (2024-11-21) and avg_total_amount of 28.34, 28.71, 28.76 respectively; the "
      "target day has a notably longer avg_trip_distance and shorter avg_total_amount compared to all three baseline Thursdays")
C6 = ("yellow_daily has a row for 2024-11-28 with trips=77244 (non-null, non-zero) and no data-quality flags; the row "
      "exists with a plausible trip count, refuting missing or incomplete data as a cause")


def test_shorter_amount_is_flagged_longer_distance_is_not():
    (p,) = check_measure_wording(C5)
    assert "shorter avg_total_amount" in p
    assert check_measure_wording("a longer average trip distance and a lower average total amount") == []


def test_no_dq_flags_claim_rejected_when_notes_exist():
    c = Claim.model_validate({"cid": "c6", "hypothesis_id": "H3", "statement": C6, "direction": "refutes",
                              "cited_values": [{"qid": "Q6", "column": "trips", "value": 77244, "row_hint": "trip_date=2024-11-28"}]})
    with_notes = check_claim_numbers(c.model_copy(deep=True), EV, version=7, dq_notes=DQ).problems
    assert any("data-quality" in p.lower() or "data quality" in p.lower() for p in with_notes)
    assert not check_claim_numbers(c.model_copy(deep=True), EV, version=7).problems


# ---- SQL: the b95249 Q1 statement (unclosed ROUND) is caught before it runs

Q1_SQL = """SELECT y.trip_date, y.trips AS trips,
       ROUND(100.0 * (y.trips - AVG(CASE WHEN y.trip_date <> DATE '2024-11-28' THEN y.trips END)) / NULLIF(AVG(CASE WHEN y.trip_date <> DATE '2024-11-28' THEN y.trips END), 0) AS pct_change_vs_baseline
FROM whybot.yellow_daily y
GROUP BY y.trip_date, y.trips"""


def test_unclosed_round_is_found_with_its_location():
    probs = sql_parse_problems(Q1_SQL)
    assert probs and "(" in probs[0]
    assert sql_parse_problems(Q1_SQL.replace(", 0) AS", ", 0), 1) AS")) == []


# ---- Fallback rendering

def test_fallback_is_sentences_without_repeated_headline():
    o = WhyBot.__new__(WhyBot)
    sup = [{"cid": "c2", "hypothesis_id": "H2", "statement": "us_holidays shows Thanksgiving Day on 2024-11-28"},
           {"cid": "c1", "hypothesis_id": "H0", "statement": "Trips on Thursday 2024-11-28 were -45.5% vs the other Thursdays"},
           {"cid": "c3", "hypothesis_id": "H1", "statement": "On 2024-11-28, prcp_mm=21.8."}]
    a = o._fallback_answer("why?", "answered", sup, [])
    assert a.headline == "Trips on Thursday 2024-11-28 were -45.5% vs the other Thursdays"
    assert a.headline not in a.answer
    assert a.answer == "Us_holidays shows Thanksgiving Day on 2024-11-28. On 2024-11-28, prcp_mm=21.8."
    assert a.confidence == 0.3 and "verbatim" in a.caveats[0]


# ---- Speed: parallel execution is gated, keeps qid order, and a guard refusal skips the re-describe

class _Live:
    kind, concurrency = "live", 4


def test_parallelism_only_for_live_llm_and_live_source(monkeypatch):
    from whybot.llm import NvidiaLLM
    o = WhyBot.__new__(WhyBot)
    o._checks_version, o._cassette = 7, None
    nv = NvidiaLLM.__new__(NvidiaLLM)
    assert o._parallelism(nv, _Live()) == 4
    assert o._parallelism(FakeLLM({}), _Live()) == 1
    assert o._parallelism(nv, FakeSource([])) == 1
    o._cassette = object()
    assert o._parallelism(nv, _Live()) == 1
    o._cassette = None
    monkeypatch.setenv("WHYBOT_PARALLEL", "0")
    assert o._parallelism(nv, _Live()) == 1
    monkeypatch.delenv("WHYBOT_PARALLEL")
    o._checks_version = 6
    assert o._parallelism(nv, _Live()) == 1


PLAN = {"metric": "daily trips", "target_window": "2024-02-13", "baseline_window": "other Feb Tuesdays",
        "hypotheses": [{"id": "H0", "kind": "metric", "statement": "trips fell", "support_if": "lower", "refute_if": "same"},
                       {"id": "H1", "kind": "weather", "statement": "snow", "support_if": "snow>0", "refute_if": "no snow"},
                       {"id": "H2", "kind": "holiday", "statement": "holiday", "support_if": "holiday", "refute_if": "none"},
                       {"id": "H3", "kind": "data_quality", "statement": "missing data", "support_if": "gaps", "refute_if": "no gaps"}],
        "queries": [{"hypothesis_id": "H0", "purpose": "trips per Tuesday", "tables": ["NY_TAXI.yellow_tripdata"]},
                    {"hypothesis_id": "H1", "purpose": "snow per Tuesday", "tables": ["NOAA_GHCN_S3.noaa_ghcn_daily_2024"]}]}
SQL_TRIPS = "SELECT trip_date, trips FROM NY_TAXI.yellow_tripdata"
SQL_SNOW = "SELECT DATE, DATA_VALUE AS snow_mm FROM NOAA_GHCN_S3.noaa_ghcn_daily_2024 WHERE ELEMENT='SNOW'"


class _SlowSource(FakeSource):
    concurrency = 2

    def _execute(self, sql):
        time.sleep(0.3 if "trip_date" in sql else 0.05)  # Q1 finishes last
        return super()._execute(sql)


def test_parallel_queries_keep_plan_order(settings, monkeypatch):
    monkeypatch.setattr(WhyBot, "_parallelism", lambda self, llm, ds: 2)

    def sql(messages):
        text = messages[-1]["content"]
        s = SQL_TRIPS if "trips per Tuesday" in text else SQL_SNOW
        return json.dumps({"sql": s, "tables_used": [t for t in PLAN["queries"][0]["tables"] + PLAN["queries"][1]["tables"] if t in s]})

    h1, h2 = sql_hash(SQL_TRIPS + " LIMIT 500"), sql_hash(SQL_SNOW + " LIMIT 500")
    claims = {"claims": [
        {"cid": "C1", "hypothesis_id": "H0", "statement": "13 Feb had 69067 trips vs 102152", "direction": "supports",
         "cited_values": [{"qid": "Q1", "column": "trips", "value": 69067, "row_hint": "trip_date=2024-02-13"}]},
        {"cid": "C2", "hypothesis_id": "H1", "statement": "81 mm snow on 13 Feb", "direction": "supports",
         "cited_values": [{"qid": "Q2", "column": "snow_mm", "value": 81, "row_hint": "DATE=20240213"}]}]}
    critic = {"verdicts": [{"cid": "C1", "verdict": "supported", "strength": 0.9, "reason": "ok"},
                           {"cid": "C2", "verdict": "supported", "strength": 0.8, "reason": "ok"}],
              "hypothesis_status": [{"hypothesis_id": "H0", "status": "supported", "reason": "r"},
                                    {"hypothesis_id": "H1", "status": "supported", "reason": "r"}],
              "decision": "accept", "feedback": "", "follow_up_queries": []}
    writer = {"headline": "Snowstorm cut trips", "answer": "a", "recommendation": "r", "confidence": 0.8,
              "claims": [{"text": "t", "citations": [{"qid": "Q1", "sql_hash": h1, "table": "NY_TAXI.yellow_tripdata"},
                                                     {"qid": "Q2", "sql_hash": h2, "table": "NOAA_GHCN_S3.noaa_ghcn_daily_2024"}]}],
              "caveats": [], "lessons": []}
    llm = FakeLLM({"planner": [PLAN], "sql": [sql], "analyst": [claims], "critic": [critic], "writer": [writer]})
    src = _SlowSource([("select trip_date", QR(["trip_date", "trips"], [["2024-02-06", 102152], ["2024-02-13", 69067]])),
                       ("element='snow'", QR(["DATE", "snow_mm"], [[20240206, 0], [20240213, 81]]))])
    res = WhyBot(settings, llm=llm, datasource=src, use_meterless=False, echo=False).run("why?")
    assert res["status"] == "answered", res
    ev = {e["qid"]: e for e in res["evidence"]}
    assert "trips" in ev["Q1"]["columns"] and "snow_mm" in ev["Q2"]["columns"]
    acts = [json.loads(l)["action"] for l in (settings.logs_dir / f"{res['run_id']}.jsonl").read_text().splitlines()]
    assert "parallel_execution" in acts


def test_guard_refusal_repairs_without_redescribing(settings, monkeypatch):
    # what slips past the SQL agent's own parse check is refused by the guard; that refusal says nothing about the schema
    monkeypatch.setattr("whybot.agents.sql_parse_problems", lambda sql: [])
    bad = {"sql": "SELECT trip_date, ROUND(trips FROM NY_TAXI.yellow_tripdata", "tables_used": ["NY_TAXI.yellow_tripdata"]}
    good = {"sql": SQL_TRIPS, "tables_used": ["NY_TAXI.yellow_tripdata"]}
    plan = {**PLAN, "queries": [PLAN["queries"][0]]}
    claims = {"claims": [{"cid": "C1", "hypothesis_id": "H0", "statement": "13 Feb had 69067 trips vs 102152", "direction": "supports",
                          "cited_values": [{"qid": "Q1", "column": "trips", "value": 69067, "row_hint": "trip_date=2024-02-13"}]}]}
    critic = {"verdicts": [{"cid": "C1", "verdict": "supported", "strength": 0.9, "reason": "ok"}],
              "hypothesis_status": [{"hypothesis_id": "H0", "status": "supported", "reason": "r"}],
              "decision": "accept", "feedback": "", "follow_up_queries": []}
    h1 = sql_hash(SQL_TRIPS + " LIMIT 500")
    writer = {"headline": "Trips fell", "answer": "a", "recommendation": "r", "confidence": 0.8,
              "claims": [{"text": "t", "citations": [{"qid": "Q1", "sql_hash": h1, "table": "NY_TAXI.yellow_tripdata"}]}],
              "caveats": [], "lessons": []}
    llm = FakeLLM({"planner": [plan], "sql": [bad, good, good], "analyst": [claims], "critic": [critic], "writer": [writer]})
    src = FakeSource([("select trip_date", QR(["trip_date", "trips"], [["2024-02-06", 102152], ["2024-02-13", 69067]]))])
    res = WhyBot(settings, llm=llm, datasource=src, use_meterless=False, echo=False).run("why?")
    q1 = next(e for e in res["evidence"] if e["qid"] == "Q1")
    assert q1["status"] == "ok" and len(q1["attempts"]) == 2
    acts = [json.loads(l)["action"] for l in (settings.logs_dir / f"{res['run_id']}.jsonl").read_text().splitlines()]
    assert "schema_introspect_for_repair" not in acts
    assert not any("ROUND(trips" in s for s in src.executed)


def test_schema_cache_reuses_column_lists_within_ttl(tmp_path, monkeypatch):
    from whybot.datasource.zetaris_jdbc import ZetarisJDBCSource
    monkeypatch.setenv("WHYBOT_CACHE_DIR", str(tmp_path))
    ZetarisJDBCSource._MEM_CACHE.clear()

    def src():
        z = ZetarisJDBCSource.__new__(ZetarisJDBCSource)
        z.url, z.tables, z.databases = "jdbc:zetaris:x", ["whybot.yellow_daily"], ["TPCDS_DB"]
        return z
    cat = [{"table": "whybot.yellow_daily", "columns": [["trip_date", "DATE"], ["trips", "BIGINT"]], "note": ""}]
    a = src()
    assert a._cached_catalog() is None and a.catalog_info["source"] == "live"
    a._store_catalog(cat, 17.2)
    ZetarisJDBCSource._MEM_CACHE.clear()  # a new process reads the file
    b = src()
    assert b._cached_catalog() == cat and b.catalog_info["source"] == "cache" and b.catalog_info["live_cost_s"] == 17.2
    monkeypatch.setenv("ZETARIS_SCHEMA_TTL_S", "0")
    assert src()._cached_catalog() is None
    monkeypatch.setenv("ZETARIS_SCHEMA_TTL_S", "1800")
    c = src()
    c.tables = ["whybot.yellow_daily", "whybot.nyc_weather_daily"]  # another allowlist is another key
    assert c._cached_catalog() is None
    # an empty column list (a failed describe) is never cached
    ZetarisJDBCSource._MEM_CACHE.clear()
    (tmp_path / "zetaris_schema.json").unlink()
    src()._store_catalog([{"table": "whybot.yellow_daily", "columns": [], "note": ""}], 1.0)
    assert not (tmp_path / "zetaris_schema.json").exists()


def test_undefined_formula_variable_names_the_cited_values():
    # mac-live-thanksgiving-v7a: three Analyst attempts on "unsupported expression: Name(id='v4')" with three cited values
    from whybot.checks import check_derived_values
    c = Claim.model_validate({"cid": "c0", "hypothesis_id": "H0", "direction": "supports",
                              "statement": "Trips on 2024-11-28 were {p}% vs the mean of the other three Thursdays",
                              "cited_values": [{"qid": "Q1", "column": "trips", "value": 77244, "row_hint": "trip_date=2024-11-28"},
                                               {"qid": "Q1", "column": "baseline_avg_trips", "value": 141603.0, "row_hint": "avg"},
                                               {"qid": "Q1", "column": "target_trips", "value": 77244, "row_hint": "trip_date=2024-11-28"}],
                              "derived": [{"name": "p", "formula": "(v1 - avg(v2,v3,v4)) / avg(v2,v3,v4) * 100"}]})
    p = check_derived_values(c, version=7)[0]
    assert "v4 not defined" in p and "v1..v3" in p and "v2 = Q1.baseline_avg_trips 141603.0" in p
    assert "unsupported expression" in check_derived_values(c.model_copy(deep=True), version=6)[0]


# ---- run mac-live-thanksgiving-v7b-20261028T195353Z: answered (0.92) in 251 s, but ~110 s went on avoidable retries

V7B_C2 = ("Precipitation on 2024-11-28 (21.8 mm) was higher than baseline Thursday 2024-11-07 (0.0 mm) and 2024-11-14 "
          "(0.0 mm), but lower than 2024-11-21 (39.9 mm)")


def _claim(statement, cited):
    return Claim.model_validate({"cid": "c2", "hypothesis_id": "H2", "statement": statement, "direction": "supports",
                                 "cited_values": cited})


def test_comparison_holds_for_the_days_its_own_clause_names():
    # the Analyst named the days correctly and was rejected twice (33 s) because 21 Nov from the 'but lower' clause
    # was counted as one of the 'higher than' days; the weather fact then never reached the answer
    cited = [{"qid": "Q3", "column": "prcp_mm", "value": v, "row_hint": f"trip_date={d}"}
             for d, v in (("2024-11-28", 21.8), ("2024-11-07", 0.0), ("2024-11-14", 0.0), ("2024-11-21", 39.9))]
    assert check_claim_numbers(_claim(V7B_C2, cited), EV, version=7).problems == []
    # still rejected when the clause itself is wrong
    wrong = V7B_C2.replace("2024-11-14 (0.0 mm), but lower than 2024-11-21 (39.9 mm)", "2024-11-21 (39.9 mm)")
    assert any("does not hold" in p for p in check_claim_numbers(_claim(wrong, cited), EV, version=7).problems)
    assert any("does not hold" in p for p in check_claim_numbers(_claim("21.8 mm on 2024-11-28 was higher than the "
                                                                         "baseline Thursdays", cited), EV, version=7).problems)


def test_plural_weekday_in_a_claim_that_cites_no_rows():
    c = _claim("Fare and distance on 2024-11-28 versus baseline Thursdays were not queried", [])
    assert not any("weekday" in p for p in check_claim_numbers(c, EV, version=7).problems)


def test_a_list_of_days_is_not_an_average():
    probs = check_measure_wording("Trips fell to 77,244 on 2024-11-28 from an average of 130,018, 141,642 and 153,149 "
                                  "on the other three Thursdays")
    assert probs and "130,018, 141,642 and 153,149" in probs[0]
    assert check_measure_wording("Trips fell to 77,244 from 130,018, 141,642 and 153,149 on the other three Thursdays, "
                                 "an average of 141,603") == []


V7B_Q5 = ("SELECT y.trip_date, y.avg_total_amount AS target_avg_total_amount, AVG(CASE WHEN d.d_day_name = 'Thursday' AND "
          "y.trip_date <> DATE '2024-11-28' THEN y.avg_total_amount END) AS baseline_avg_total_amount FROM whybot.yellow_daily y "
          "JOIN TPCDS_DB.date_dim d ON d.d_date = y.trip_date WHERE d.d_day_name = 'Thursday' GROUP BY y.trip_date ORDER BY y.trip_date")


def test_missing_aggregation_is_caught_before_it_runs():
    probs = sql_parse_problems(V7B_Q5)
    assert probs and "y.avg_total_amount" in probs[0] and "GROUP BY" in probs[0]
    ok = [V7B_Q5.replace("y.avg_total_amount AS target", "MAX(y.avg_total_amount) AS target"),
          "SELECT y.trip_date, d.d_day_name, MAX(y.trips) AS t FROM whybot.yellow_daily y JOIN TPCDS_DB.date_dim d "
          "ON d.d_date = y.trip_date GROUP BY y.trip_date, d.d_day_name",
          "SELECT d.d_day_name AS dn, COUNT(*) AS n FROM TPCDS_DB.date_dim d GROUP BY dn",
          "SELECT y.trip_date, y.trips FROM whybot.yellow_daily y ORDER BY 1",
          "SELECT y.trip_date, y.trips, AVG(y.trips) OVER () AS a FROM whybot.yellow_daily y"]
    for sql in ok:
        assert sql_parse_problems(sql) == [], sql


def test_non_schema_zetaris_error_repairs_without_redescribing(settings, monkeypatch):
    from whybot.datasource.base import DataSourceError
    monkeypatch.setattr("whybot.agents.sql_parse_problems", lambda sql: [])
    bad = {"sql": "SELECT trip_date, COUNT(*) AS trips FROM NY_TAXI.yellow_tripdata GROUP BY 1 /* bad */",
           "tables_used": ["NY_TAXI.yellow_tripdata"]}
    good = {"sql": SQL_TRIPS, "tables_used": ["NY_TAXI.yellow_tripdata"]}
    plan = {**PLAN, "queries": [PLAN["queries"][0]]}
    claims = {"claims": [{"cid": "C1", "hypothesis_id": "H0", "statement": "13 Feb had 69067 trips vs 102152", "direction": "supports",
                          "cited_values": [{"qid": "Q1", "column": "trips", "value": 69067, "row_hint": "trip_date=2024-02-13"}]}]}
    critic = {"verdicts": [{"cid": "C1", "verdict": "supported", "strength": 0.9, "reason": "ok"}],
              "hypothesis_status": [{"hypothesis_id": "H0", "status": "supported", "reason": "r"}],
              "decision": "accept", "feedback": "", "follow_up_queries": []}
    h1 = sql_hash(SQL_TRIPS + " LIMIT 500")
    writer = {"headline": "Trips fell", "answer": "a", "recommendation": "r", "confidence": 0.8,
              "claims": [{"text": "t", "citations": [{"qid": "Q1", "sql_hash": h1, "table": "NY_TAXI.yellow_tripdata"}]}],
              "caveats": [], "lessons": []}
    llm = FakeLLM({"planner": [plan], "sql": [bad, good, good], "analyst": [claims], "critic": [critic], "writer": [writer]})
    err = DataSourceError("[MISSING_AGGREGATION] The non-aggregating expression \"trips\" is based on columns which are not "
                          "participating in the GROUP BY clause.")
    src = FakeSource([("/* bad */", err), ("select trip_date", QR(["trip_date", "trips"], [["2024-02-06", 102152], ["2024-02-13", 69067]]))])
    res = WhyBot(settings, llm=llm, datasource=src, use_meterless=False, echo=False).run("why?")
    q1 = next(e for e in res["evidence"] if e["qid"] == "Q1")
    assert q1["status"] == "ok" and len(q1["attempts"]) == 2
    acts = [json.loads(l)["action"] for l in (settings.logs_dir / f"{res['run_id']}.jsonl").read_text().splitlines()]
    assert "schema_introspect_for_repair" not in acts and "repair_without_reintrospection" in acts


def test_analyst_stops_retrying_when_a_retry_makes_no_progress():
    from whybot.agents import Analyst
    bad = {"claims": [{"cid": "c2", "hypothesis_id": "H2", "statement": "21.8 mm on 2024-11-28 was higher than the baseline Thursdays",
                       "direction": "supports",
                       "cited_values": [{"qid": "Q3", "column": "prcp_mm", "value": v, "row_hint": f"trip_date={d}"}
                                        for d, v in (("2024-11-28", 21.8), ("2024-11-07", 0.0), ("2024-11-21", 39.9))]}]}
    llm = FakeLLM({"analyst": [bad, bad, bad]})
    cs, _ = Analyst(llm, version=7).draft("why?", "H2 weather", list(EV.values()), "")
    assert len(llm.calls) == 2  # v6 asked a third time for the same rejected sentence
    llm6 = FakeLLM({"analyst": [bad, bad, bad]})
    Analyst(llm6, version=6).draft("why?", "H2 weather", list(EV.values()), "")
    assert len(llm6.calls) == 3


def test_writer_sees_no_numbers_from_rejected_claims():
    from whybot.orchestrator import mask_numbers
    assert mask_numbers("Fare dropped 11.5% (under 15% threshold) on 2024-11-28 (Q5, H4)") == \
        "Fare dropped [n]% (under [n]% threshold) on 2024-11-28 (Q5, H4)"
    assert mask_numbers("baseline 21 Nov had 39.9 mm") == "baseline 21 Nov had [n] mm"


def test_iso_date_with_two_slash_days_is_a_date_list():
    # run v7d: the Writer copied 'Mon-Wed trips 2024-11-25/26/27 were 104082/114424/107719' from a supported claim and
    # was rejected three times for '26' and '27' (fallback, confidence 0.30)
    from whybot.checks import known_days
    q = _ev("Q4", ["trip_date", "trips"], [["2024-11-25", 104082], ["2024-11-26", 114424], ["2024-11-27", 107719]])
    text = "Mon-Wed trips 2024-11-25/26/27 were 104082/114424/107719"
    out = blank_day_lists(text, known_days({"Q4": q}), slash=True)
    assert "26" not in out.replace("2024-11-25", "") and "104082/114424/107719" in out
    assert blank_day_lists("2024-11-25/26/29 were", known_days({"Q4": q}), slash=True) == "2024-11-25/26/29 were"


def test_a_citation_by_sql_hash_is_the_same_query():
    # run v7e: 'c0 cites unknown sha256:c4d48f5c297e8a78' three times (the hash of a real query sat in the qid field)
    from whybot.agents import Analyst
    claim = {"claims": [{"cid": "c0", "hypothesis_id": "H0", "statement": "Trips on 2024-11-28 were 77244", "direction": "supports",
                         "cited_values": [{"qid": "sha256:Q2", "column": "trips", "value": 77244, "row_hint": "trip_date=2024-11-28"}]}]}
    llm = FakeLLM({"analyst": [claim, claim, claim]})
    cs, _ = Analyst(llm, version=7).draft("why?", "H0", list(EV.values()), "")
    assert len(llm.calls) == 1 and cs.claims[0].cited_values[0].qid == "Q2"
    wrong = {"claims": [{**claim["claims"][0], "cited_values": [{**claim["claims"][0]["cited_values"][0], "qid": "sha256:nope"}]}]}
    llm2 = FakeLLM({"analyst": [wrong, wrong, wrong]})
    try:
        Analyst(llm2, version=7).draft("why?", "H0", list(EV.values()), "")
    except Exception:
        pass
    assert len(llm2.calls) == 3  # an unknown hash is still rejected


# run v7f's real Writer text (answered at 0.90, but not fit to show)
V7F_ANSWER = ("On 2024-11-28, yellow taxi trips dropped to 77,244 from 130,018 on Nov 7, 141,642 on Nov 14, and 153,149 on Nov 21. "
              "The -45.5% change versus the mean of those three Thursdays aligns with Thanksgiving Day falling on 2024-11-28, while "
              "the other November Thursdays showed normal volume. No critic-supported claim in the evidence list validates "
              "Thanksgiving as the cause; the holiday association is noted as contextual alignment only.")
V7F_CLAIM2 = ("The mean baseline across the other three Thursdays is not quoted as a single number; per-day values are listed "
              "instead")


def test_writer_never_writes_about_its_own_process():
    from whybot.checks import check_meta_text
    assert any("critic" in p for p in check_meta_text([V7F_ANSWER]))
    assert len(check_meta_text([V7F_CLAIM2])) == 2  # both halves of the sentence
    assert check_meta_text(["The 45.5% drop is consistent with other holidays, as confirmed by the hypothesis H1 status."])
    assert check_meta_text(["Trips fell 45.5% on Thanksgiving Day, 2024-11-28, driven by the national holiday.",
                            "Data quality (not evidence): the TPCDS date_dim.d_holiday flag disagrees with Nager.Date on 3 date(s)."]) == []


def test_hypothesis_ids_in_brackets_are_removed_not_rejected():
    from whybot.checks import _drop_hypothesis_ids
    from whybot.schemas import FinalAnswer
    a = FinalAnswer(headline="h", answer="a", recommendation="r", confidence=0.9, claims=[],
                    caveats=["Weather hypothesis (H2) is inconclusive: one baseline Thursday had more rain"])
    _drop_hypothesis_ids(a)
    assert a.caveats == ["Weather hypothesis is inconclusive: one baseline Thursday had more rain"]


def test_a_supported_holiday_cause_is_stated_as_the_cause():
    from whybot.checks import check_supported_holiday_named
    sup = ["holiday: Thanksgiving (national holiday) on 2024-11-28 caused the drop in trips"]
    assert check_supported_holiday_named(V7F_ANSWER, sup)
    assert check_supported_holiday_named("Trips fell 45.5%, driven by Thanksgiving Day.", sup) == []
    assert check_supported_holiday_named("Thanksgiving, a national holiday, drove the drop.", sup) == []
    assert check_supported_holiday_named(V7F_ANSWER, ["other: Trips on 2024-11-28 were -45.5% vs the mean"]) == []


def test_a_derived_value_over_text_cells_is_dropped_not_rejected():
    # run v7f: c2 'Thanksgiving ... caused the drop' cited holiday_name cells with 'case when v1 is not null then 1 else 0
    # end' and was rejected, so the supported holiday cause never reached the Writer
    from whybot.checks import check_derived_values
    from whybot.schemas import Claim
    c = Claim(cid="c2", hypothesis_id="H1", statement="Thanksgiving (national holiday) on 2024-11-28 caused the drop in trips",
              direction="supports",
              cited_values=[{"qid": "Q6", "column": "holiday_name", "value": "Thanksgiving Day", "row_hint": "trip_date=2024-11-28"},
                            {"qid": "Q6", "column": "holiday_name", "value": None, "row_hint": "trip_date=2024-11-07"}],
              derived=[{"name": "holiday_present_target", "formula": "case when v1 is not null then 1 else 0 end"},
                       {"name": "holiday_present_baseline", "formula": "avg(v2)"}])
    assert check_derived_values(c, version=7) == [] and c.derived == []
    base = c.model_dump()
    assert check_derived_values(Claim(**{**base, "derived": [{"name": "x", "formula": "case when v1 is not null then 1 else 0 end"}]}),
                                version=6)
    bad = Claim(**{**base, "statement": "Trips fell 45.5% on Thanksgiving", "derived": []})
    assert check_derived_values(bad, version=7)  # an unbacked percentage is still rejected


def test_a_list_labelled_as_average_is_rejected_for_the_writer():
    from whybot.checks import check_measure_wording
    assert check_measure_wording("Trips fell to 77,244 on 2024-11-28 from an average of 130,018, 141,642 and 153,149 on the other Thursdays.")
    assert check_measure_wording("Trips fell to 77,244 on 2024-11-28, 45.5% below an average of 141,603 on the other Thursdays.") == []


V7F_Q2_A1 = ("WITH target AS (SELECT trip_date, trips FROM whybot.yellow_daily WHERE trip_date = DATE '2024-11-28'), baseline AS "
             "(SELECT trip_date, trips FROM whybot.yellow_daily WHERE trip_date BETWEEN DATE '2024-11-01' AND DATE '2024-11-30' AND "
             "trip_date <> DATE '2024-11-28'), holidays AS (SELECT holiday_date, holiday_name FROM whybot.us_holidays WHERE "
             "holiday_date = DATE '2024-11-28') SELECT y.trip_date, y.trips AS target_trips, h.holiday_name, AVG(b.trips) AS "
             "baseline_avg_trips, (y.trips - AVG(b.trips)) / AVG(b.trips) * 100 AS pct_change FROM target y CROSS JOIN holidays h "
             "LEFT JOIN baseline b ON TRUE GROUP BY y.trip_date, y.trips, h.holiday_name ORDER BY y.trip_date LIMIT 1")
V7F_Q2_A3 = ("SELECT y.trip_date, y.trips AS target_trips, h.holiday_name, AVG(CASE WHEN b.trip_date <> DATE '2024-11-28' THEN "
             "b.trips END) AS baseline_avg_trips FROM whybot.yellow_daily y LEFT JOIN whybot.us_holidays h ON h.holiday_date = "
             "y.trip_date JOIN TPCDS_DB.date_dim d ON d.d_date = y.trip_date WHERE y.trip_date BETWEEN DATE '2024-11-01' AND "
             "DATE '2024-11-30' GROUP BY y.trip_date, y.trips, h.holiday_name ORDER BY y.trip_date LIMIT 500")


def test_self_join_of_a_cached_table_is_caught_before_it_runs():
    # run v7f Q2 (and v7e Q7/Q8): [MISSING_ATTRIBUTES] tpep_pickup_datetime, every time
    probs = sql_parse_problems(V7F_Q2_A1)
    assert probs and "whybot.yellow_daily is read 2 times" in probs[0]
    sub = ("SELECT y.trip_date, AVG(b.trips) AS a FROM whybot.yellow_daily y CROSS JOIN (SELECT trip_date, trips FROM "
           "whybot.yellow_daily) b GROUP BY y.trip_date")
    assert any("read 2 times" in p for p in sql_parse_problems(sub))
    ok = ("SELECT y.trip_date, y.trips, h.holiday_name, d.d_day_name FROM whybot.yellow_daily y LEFT JOIN whybot.us_holidays h "
          "ON h.holiday_date = y.trip_date JOIN TPCDS_DB.date_dim d ON d.d_date = y.trip_date ORDER BY y.trip_date")
    assert sql_parse_problems(ok) == []
    cte = "WITH t AS (SELECT trip_date, trips FROM whybot.yellow_daily) SELECT t.trip_date, t.trips FROM t ORDER BY 1"
    assert sql_parse_problems(cte) == []


def test_undefined_alias_is_caught_before_it_runs():
    # run v7f Q2 attempt 3: UNRESOLVED_COLUMN `b`.`trip_date`
    probs = sql_parse_problems(V7F_Q2_A3)
    assert any("alias b" in p for p in probs)
    assert sql_parse_problems("SELECT x.trip_date FROM (SELECT trip_date FROM whybot.yellow_daily) x") == []


def test_the_baseline_average_is_written_into_the_change_claim():
    # run v8a: 'Trips on 2024-11-28 were -45.5% vs the mean of the other Thursdays (...)' gave the Writer no single
    # average, so it listed 130,018, 141,642 and 153,149 instead of 'an average of 141,603'
    from whybot.checks import check_derived_values
    from whybot.schemas import Claim
    cells = [("2024-11-28", 77244), ("2024-11-07", 130018), ("2024-11-14", 141642), ("2024-11-21", 153149)]
    c = Claim(cid="c1", hypothesis_id="H0", direction="supports",
              statement="Trips on 2024-11-28 were {pct_change_vs_baseline}% vs the mean of the other Thursdays (2024-11-07, 2024-11-14, 2024-11-21)",
              cited_values=[{"qid": "Q2", "column": "trips", "value": v, "row_hint": f"trip_date={d}"} for d, v in cells],
              derived=[{"name": "pct_change_vs_baseline", "formula": "(v1 - avg(v2, v3, v4)) / avg(v2, v3, v4) * 100"}])
    assert check_derived_values(c, version=7) == []
    assert c.statement == ("Trips on 2024-11-28 were -45.5% vs the mean of the other Thursdays (an average of 141,603) "
                           "(2024-11-07, 2024-11-14, 2024-11-21)")
    assert agg(c.statement, ["Q2"]) == []


def test_citations_are_listed_once_per_query():
    from whybot.checks import _drop_hypothesis_ids
    from whybot.schemas import Citation, FinalAnswer, FinalClaim
    cit = Citation(qid="Q1", sql_hash="sha256:Q1", table="whybot.yellow_daily")
    a = FinalAnswer(headline="h", answer="a", recommendation="r", confidence=0.9, caveats=[],
                    claims=[FinalClaim(text="t", citations=[cit, cit, cit, cit])])
    _drop_hypothesis_ids(a)
    assert len(a.claims[0].citations) == 1


def test_a_supported_holiday_hypothesis_is_named_as_the_cause():
    from whybot.checks import check_supported_holiday_named
    sup = ["holiday: 2024-11-28 was Thanksgiving Day; the other cited Thursdays (2024-11-07, 2024-11-14, 2024-11-21) were not holidays"]
    v8a = "Trips fell to 77,244 on 2024-11-28. The target day was Thanksgiving Day, while the baseline Thursdays had no holiday."
    assert check_supported_holiday_named(v8a, sup)
    assert check_supported_holiday_named(v8a + " The drop was driven by Thanksgiving Day.", sup) == []


# --- v8c: the baseline average and % change come from a query row, every run ---
from whybot.agents import add_baseline_summary_query, baseline_summary_problems  # noqa: E402
from whybot.checks import check_claim_numbers  # noqa: E402
from whybot.schemas import ClaimSet, Plan  # noqa: E402

QS = _ev("Q9", ["target_date", "target_value", "baseline_avg", "baseline_days", "pct_change_vs_baseline"],
         [["2024-12-21", 90000, 120000.0, 3, -25.0]])


def _plan(queries):
    return Plan.model_validate({"metric": "daily trips", "target_window": "2024-12-21",
                                "baseline_window": "2024-12-07, 2024-12-14, 2024-12-28", "hypotheses": [],
                                "queries": queries})


def test_the_plan_always_gets_a_baseline_summary_query_for_any_target():
    p = _plan([{"hypothesis_id": "H0", "purpose": "daily trips per Saturday", "tables": ["whybot.yellow_daily"]},
               {"hypothesis_id": "H1", "purpose": "weather", "tables": ["whybot.weather_daily"]}])
    add_baseline_summary_query(p)
    q = p.queries[1]
    assert q.hypothesis_id == "H0" and q.tables == ["whybot.yellow_daily"]
    assert "baseline_avg" in q.purpose and "pct_change_vs_baseline" in q.purpose and "2024-12-21" in q.purpose
    assert "Thanksgiving" not in q.purpose
    add_baseline_summary_query(p)
    assert len(p.queries) == 3  # added once


def test_the_analyst_must_cite_the_baseline_summary_row():
    other = ClaimSet.model_validate({"claims": [{"cid": "c1", "hypothesis_id": "H0", "statement": "x", "direction": "supports",
                                                 "cited_values": [{"qid": "Q2", "column": "trips", "value": 77244,
                                                                   "row_hint": "trip_date=2024-11-28"}]}]})
    assert baseline_summary_problems(other, [QS, Q2])
    good = ClaimSet.model_validate({"claims": [{"cid": "c1", "hypothesis_id": "H0", "direction": "supports",
        "statement": "Trips on 2024-12-21 were 90,000, -25.0% vs an average of 120,000 on the other Saturdays",
        "cited_values": [{"qid": "Q9", "column": c, "value": v} for c, v in
                         [("target_value", 90000), ("pct_change_vs_baseline", -25.0), ("baseline_avg", 120000.0)]]}]})
    assert baseline_summary_problems(good, [QS, Q2]) == []
    assert check_claim_numbers(good.claims[0], {"Q9": QS, "Q2": Q2}).problems == []


def test_a_row_hint_on_a_one_row_result_is_moot_and_placeholders_take_the_cited_cell():
    c = Claim.model_validate({"cid": "c1", "hypothesis_id": "H0", "direction": "supports",
        "statement": "Trips on 2024-12-21 were {pct_change_vs_baseline}% vs an average of {baseline_avg} on the other Saturdays",
        "cited_values": [{"qid": "Q9", "column": "pct_change_vs_baseline", "value": -25.0, "row_hint": "trip_date=2024-12-21"},
                         {"qid": "Q9", "column": "baseline_avg", "value": 120000.0, "row_hint": "trip_date=2024-12-21"}]})
    ch = check_claim_numbers(c, {"Q9": QS})
    assert ch.problems == []
    assert c.statement == "Trips on 2024-12-21 were -25.0% vs an average of 120,000 on the other Saturdays"


def test_identical_sql_runs_once_per_run():
    import threading
    from whybot.orchestrator import WhyBot as Orchestrator
    calls = []

    class DS:
        def run(self, sql, timeout_s=None):
            calls.append(sql)
            threading.Event().wait(0.05)
            return "rows"

    o = Orchestrator.__new__(Orchestrator)
    o.s = type("S", (), {"sql_timeout_s": 5})()
    o._shared_sql = {}
    ts = [threading.Thread(target=lambda: o._run_once(DS(), "h1", "select 1")) for _ in range(5)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert calls == ["select 1"] and o._run_once(DS(), "h2", "select 2") == "rows" and len(calls) == 2
