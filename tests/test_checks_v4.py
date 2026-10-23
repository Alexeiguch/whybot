"""Checks v4: the root causes found when re-recording the samples under v3, each pinned by a test."""
import json

import pytest
from conftest import QR, FakeLLM, FakeSource

from whybot.agents import Critic, SQLAgent
from whybot.checks import (CHECKS_VERSION, blank_day_lists, check_calendar_sql, check_critic_verdicts, check_text_numbers,
                           known_days, repair_table_names)
from whybot.datasource.base import sql_hash
from whybot.evidence import Evidence
from whybot.rerun import rerun_reason
from whybot.schemas import Claim, CitedValue, ClaimSet, ClaimVerdict, CriticReport, HypothesisStatus

THURSDAYS = Evidence(qid="Q1", hypothesis_id="H0", purpose="p", iteration=1, status="ok", columns=["trip_date", "trips"],
                     rows=[["2024-11-07", 130018], ["2024-11-14", 141642], ["2024-11-21", 153149], ["2024-11-28", 77244]],
                     row_count=4, tables=["whybot.yellow_daily"])
EV = {"Q1": THURSDAYS}


def test_checks_version_is_4():
    assert CHECKS_VERSION == 4


# ---- 1. bare day-of-month numbers in a date list ----------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "Trips on 2024-11-28 were 77,244 vs the three prior Thursdays (2024-11-07, 14, 21).",   # take-1 #1 Writer draft
    "Thursdays Nov 7, 14 and 21 averaged about 141.6k trips.",
    "Baseline: 7, 14 and 21 Nov; target 28 Nov.",
    "the other Thursdays (Nov 7th, 14th & 21st)",
])
def test_day_lists_of_known_dates_are_dates_not_figures(text):
    assert check_text_numbers(text, EV, {"Q1"}, day_lists=known_days(EV)) == []
    assert check_text_numbers(text, EV, {"Q1"}) != [] or "7th" in text  # v3 rejected the bare 14 / 21


@pytest.mark.parametrize("text, bad", [
    ("Trips on 2024-11-07, 14, 99 thousand", "99"),            # 99 is no day of the month
    ("Thursdays Nov 7, 14 and 15 were busy", "15"),            # Nov 15 is not in the evidence: still checked as a figure
    ("On Nov 21, 153149 trips; on Nov 28, 70 thousand", "70"),  # a real figure after a date is still checked
    ("2024-11-21, 35% more than usual", "35%"),
])
def test_real_figures_next_to_dates_are_still_checked(text, bad):
    probs = check_text_numbers(text, EV, {"Q1"}, day_lists=known_days(EV))
    assert any(f"'{bad}'" in p for p in probs), probs


def test_a_list_with_one_unknown_date_is_not_blanked():
    assert blank_day_lists("Nov 7, 14 and 15", {(11, 7), (11, 14)}) == "Nov 7, 14 and 15"
    assert blank_day_lists("Nov 7, 14", {(11, 7), (11, 14)}).strip() == "Nov 7"


def test_writer_check_uses_day_lists_only_from_v4():
    from whybot.checks import validate_final
    from whybot.schemas import Citation, FinalAnswer, FinalClaim
    ans = FinalAnswer(headline="Thanksgiving (2024-11-28) cut trips", answer="77,244 trips vs Thursdays 2024-11-07, 14, 21.",
                      recommendation="r", confidence=0.9, caveats=[], lessons=[],
                      claims=[FinalClaim(text="t", citations=[Citation(qid="Q1", sql_hash=THURSDAYS.sql_hash, table="whybot.yellow_daily")])])
    v3 = validate_final(ans, EV, {"Q1"}, [], context="", version=3)
    v4 = validate_final(ans, EV, {"Q1"}, [], context="", version=4)
    assert any("'14'" in p for p in v3) and not any("'14'" in p or "'21'" in p for p in v4)


# ---- 2. d_dow is 0=Sunday..6=Saturday ----------------------------------------------------------------------------------
@pytest.mark.parametrize("sql, flagged", [
    ("SELECT d_date FROM TPCDS_DB.date_dim WHERE d_dow = 7", True),
    ("SELECT d_date FROM TPCDS_DB.date_dim WHERE d_dow IN (1, 7)", True),
    ("SELECT d_date FROM TPCDS_DB.date_dim WHERE d_dow = '7'", True),
    ("SELECT d_date FROM TPCDS_DB.date_dim WHERE d_dow = 6", False),
    ("SELECT d_date FROM TPCDS_DB.date_dim WHERE d_dow IN (0, 6)", False),
    ("SELECT d_date FROM TPCDS_DB.date_dim WHERE d_dow >= 6", False),
    ("SELECT trip_date FROM whybot.yellow_daily WHERE dayofweek(trip_date) = 7", False),
])
def test_d_dow_7_is_flagged(sql, flagged):
    assert bool(check_calendar_sql(sql)) is flagged


CAT = [{"table": "whybot.yellow_daily", "columns": [["trip_date", "DATE"], ["trips", "BIGINT"]]},
       {"table": "TPCDS_DB.date_dim", "columns": [["d_date", "DATE"], ["d_dow", "INT"]]}]


def _sql_agent(version, responses):
    llm = FakeLLM({"sql": responses})
    return llm, SQLAgent(llm, version)


def test_sql_agent_sees_the_calendar_convention_and_exact_table_names_from_v4():
    for v, expect in ((3, False), (4, True)):
        llm, agent = _sql_agent(v, [{"sql": "SELECT d_date FROM TPCDS_DB.date_dim WHERE d_dow = 6", "tables_used": []}])
        agent.write("Saturdays", [], CAT, {}, "Spark", "", "", "q")
        user = llm.calls[0][1][-1]["content"]
        assert ("ALLOWED TABLES" in user and "whybot.yellow_daily, TPCDS_DB.date_dim" in user and "0=Sunday" in user) is expect


def test_sql_agent_is_sent_back_for_d_dow_7_from_v4():
    bad = {"sql": "SELECT d_date FROM TPCDS_DB.date_dim WHERE d_dow = 7", "tables_used": []}
    good = {"sql": "SELECT d_date FROM TPCDS_DB.date_dim WHERE d_dow = 6", "tables_used": []}
    llm, agent = _sql_agent(4, [bad, good])
    d, _ = agent.write("Saturdays", [], CAT, {}, "Spark", "", "", "q")
    assert d.sql.endswith("d_dow = 6") and "Saturday is 6" in llm.calls[1][1][-1]["content"]
    llm, agent = _sql_agent(3, [bad, good])
    assert agent.write("Saturdays", [], CAT, {}, "Spark", "", "", "q")[0].sql.endswith("d_dow = 7")


def test_catalog_notes_state_the_d_dow_range():
    from whybot.notes import load_notes
    note = load_notes()["TPCDS_DB.date_dim"]
    assert "6=Saturday" in note and "never 7" in note


# ---- 3. allowlist close misses -----------------------------------------------------------------------------------------
ALLOWED = ["whybot.yellow_daily", "whybot.us_holidays", "TPCDS_DB.date_dim", "whybot.nyc_weather_daily"]


@pytest.mark.parametrize("sql, fixed, repairs", [
    ("SELECT d.d_date FROM whybot.date_dim d JOIN whybot.yellow_daily y ON y.trip_date = d.d_date",
     "SELECT d.d_date FROM TPCDS_DB.date_dim d JOIN whybot.yellow_daily y ON y.trip_date = d.d_date", ["whybot.date_dim -> TPCDS_DB.date_dim"]),
    ("SELECT 1 FROM whybot.tpcds_db.date_dim", "SELECT 1 FROM TPCDS_DB.date_dim", ["whybot.tpcds_db.date_dim -> TPCDS_DB.date_dim"]),
    ("SELECT 1 FROM tpcds.date_dim", "SELECT 1 FROM TPCDS_DB.date_dim", ["tpcds.date_dim -> TPCDS_DB.date_dim"]),
    ("SELECT 1 FROM whybot.weather", "SELECT 1 FROM whybot.weather", []),         # no table of that name: still refused
    ("SELECT 1 FROM tpcds_db.date_dim", "SELECT 1 FROM tpcds_db.date_dim", []),   # already allowed (case-insensitive)
])
def test_close_table_misses_are_repaired_to_the_one_table_they_can_mean(sql, fixed, repairs):
    assert repair_table_names(sql, ALLOWED) == (fixed, repairs)


def test_ambiguous_table_names_are_not_guessed():
    assert repair_table_names("SELECT 1 FROM x.date_dim", ["a.date_dim", "b.date_dim"]) == ("SELECT 1 FROM x.date_dim", [])


def test_executor_repairs_the_name_before_the_allowlist_fails_closed(settings):
    from test_orchestrator import CLAIMS, PLAN, critic, writer_for
    from whybot.orchestrator import WhyBot
    # one listed table plus a close miss passes the SQL agent's own check and reaches the allowlist (take-1 #3)
    wrong = {"sql": "SELECT trip_date, trips FROM NY_TAXI.yellow_tripdata WHERE trip_date IN (SELECT DATE FROM whybot.noaa_ghcn_daily_2024)",
             "tables_used": []}
    snow = {"sql": "SELECT DATE, DATA_VALUE AS snow_mm FROM NOAA_GHCN_S3.noaa_ghcn_daily_2024 WHERE ELEMENT='SNOW'", "tables_used": []}
    h1 = sql_hash("SELECT trip_date, trips FROM NY_TAXI.yellow_tripdata WHERE trip_date IN "
                  "(SELECT DATE FROM NOAA_GHCN_S3.noaa_ghcn_daily_2024) LIMIT 500")
    h2 = sql_hash(snow["sql"] + " LIMIT 500")
    llm = FakeLLM({"planner": [PLAN], "sql": [wrong, snow], "analyst": [CLAIMS], "critic": [critic("accept")],
                   "writer": [writer_for([("Q1", h1, "NY_TAXI.yellow_tripdata"), ("Q2", h2, "NOAA_GHCN_S3.noaa_ghcn_daily_2024")])]})
    routes = [("select trip_date", QR(["trip_date", "trips"], [["2024-02-06", 102152], ["2024-02-13", 69067]])),
              ("element='snow'", QR(["DATE", "snow_mm"], [[20240206, 0], [20240213, 81]]))]
    src = FakeSource(routes)
    res = WhyBot(settings, llm=llm, datasource=src, use_meterless=False, echo=False).run("why?")
    ev = [json.loads(l) for l in (settings.logs_dir / f"{res['run_id']}.jsonl").read_text().splitlines()]
    rep = [e for e in ev if e["action"] == "sql_table_name_repaired"]
    assert rep and rep[0]["data"]["repairs"] == ["whybot.noaa_ghcn_daily_2024 -> NOAA_GHCN_S3.noaa_ghcn_daily_2024"]
    q1 = next(e for e in res["evidence"] if e["qid"] == "Q1")
    assert q1["status"] == "ok" and q1["sql_hash"] == h1 and len(q1["attempts"]) == 1


# ---- 4. the Critic judges claims against rows, not the hypothesis direction --------------------------------------------
def _report(reason, verdict="unsupported"):
    return CriticReport(verdicts=[ClaimVerdict(cid="c1", verdict=verdict, strength=0.2, reason=reason)],
                        hypothesis_status=[HypothesisStatus(hypothesis_id="H0", status="inconclusive", reason="r")],
                        decision="retry", feedback="f")


class _Ok:
    def __init__(self, ok):
        self.ok, self.problems = ok, [] if ok else ["number missing"]


@pytest.mark.parametrize("reason, flagged", [
    ("The claim states +0.15% vs the mean, but the hypothesis H0 expects a reduction.", True),   # take-1 #2
    ("Trips were not lower, so this does not support the hypothesis.", True),
    ("Baseline includes a holiday Saturday, not like-for-like.", False),
    ("The cited number 131279 is from the wrong row.", False),
])
def test_critic_verdict_resting_on_hypothesis_direction_is_flagged(reason, flagged):
    assert bool(check_critic_verdicts(_report(reason), {"c1": _Ok(True)})) is flagged


def test_failed_number_checks_and_supported_verdicts_are_left_alone():
    r = "but the hypothesis expects a reduction"
    assert check_critic_verdicts(_report(r), {"c1": _Ok(False)}) == []
    assert check_critic_verdicts(_report(r, "supported"), {"c1": _Ok(True)}) == []


def test_critic_is_sent_back_once_from_v4_and_told_refutation_is_evidence():
    claims = ClaimSet(claims=[Claim(cid="c1", hypothesis_id="H0", statement="Trips on 2024-12-21 were +0.15% vs the other Saturdays",
                                    direction="refutes", cited_values=[CitedValue(qid="Q1", column="trips", value=131279, row_hint="d=2024-12-21")])])
    first = _report("Numbers are right but the hypothesis H0 expects a reduction.").model_dump()
    second = {**first, "verdicts": [{"cid": "c1", "verdict": "supported", "strength": 0.9, "reason": "matches the rows"}],
              "hypothesis_status": [{"hypothesis_id": "H0", "status": "refuted", "reason": "no drop"}], "decision": "accept"}
    for v, n_calls, verdict in ((4, 2, "supported"), (3, 1, "unsupported")):
        llm = FakeLLM({"critic": [first, second]})
        rep, _ = Critic(llm, v).review("q", "H0", claims, [], {"c1": _Ok(True)}, [], 1, 3)
        assert len(llm.calls) == n_calls and rep.verdicts[0].verdict == verdict
        assert ("never mark a claim unsupported because of the direction" in llm.calls[0][1][0]["content"]) is (v >= 4)


# ---- 5. NVIDIA timeouts re-run like 5xx ----------------------------------------------------------------------------------
@pytest.mark.parametrize("result, events, rerun", [
    ({"status": "error", "error": {"message": "planner: 502 Bad Gateway"}}, [], True),
    ({"status": "error", "error": {"message": "planner: APITimeoutError: Request timed out."}}, [], True),
    ({"status": "error", "error": {"message": "planner: watchdog timeout after 185s"}}, [], True),
    ({"status": "error", "error": {"message": "DataSourceError: table not allowed"}}, [], False),
    ({"status": "insufficient_evidence"},
     [{"agent_name": "Critic", "action": "failed", "status": "error", "output_summary": "critic: APITimeoutError: Request timed out."},
      {"agent_name": "Critic", "action": "fallback_verdicts", "status": "fallback", "output_summary": "LLM critic unavailable"}], True),
    ({"status": "answered"},  # a timeout the per-call retry absorbed: keep the run
     [{"agent_name": "Critic", "action": "llm_error_retry", "status": "error", "output_summary": "critic: Request timed out."}], False),
    ({"status": "insufficient_evidence"},  # a Meterless bridge timeout is not an NVIDIA failure
     [{"agent_name": "Memory", "action": "meterless_unavailable", "status": "error", "output_summary": "bridge timed out or exited"},
      {"agent_name": "Writer", "action": "fallback_answer", "status": "fallback", "output_summary": "x"}], False),
])
def test_recorder_reruns_on_nvidia_timeouts_like_5xx(result, events, rerun):
    assert bool(rerun_reason(result, events)) is rerun


# ---- old recordings keep replaying exactly ---------------------------------------------------------------------------------
def test_cassettes_store_and_replay_their_own_notes(tmp_path):
    from whybot.llm import BaseLLM, CassetteLLM

    class Inner(BaseLLM):
        def model_for(self, role):
            return "fake"

        def chat(self, role, messages, *, max_tokens=4096, temperature=0.2):
            return "{}", {"model": "fake", "latency_s": 0.0}

    cas = CassetteLLM(tmp_path / "c.json", inner=Inner(), run_id="recorded-x")
    cas.header.update(table_notes={"t": "n"}, dialect_notes="d")
    cas.chat("planner", [{"role": "user", "content": "q"}])
    cas.save("q")
    h = CassetteLLM(tmp_path / "c.json").header
    assert h["table_notes"] == {"t": "n"} and h["dialect_notes"] == "d" and h["checks_version"] == CHECKS_VERSION


def test_recording_does_not_overwrite_the_committed_catalog_notes(tmp_path):
    from whybot.datasource.recording import RecordingDataSource
    (tmp_path / "catalog.json").write_text(json.dumps({"source": "zetaris", "table_notes": {"old": "note"}, "catalog": []}))

    class Live(FakeSource):
        kind = "live"
    RecordingDataSource(Live([]), tmp_path).catalog()
    assert json.loads((tmp_path / "catalog.json").read_text())["table_notes"] == {"old": "note"}
