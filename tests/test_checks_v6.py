"""Checks v6: dates and weekdays. Every fixture row and claim below is copied from the live Zetaris run
mac-live-thanksgiving-20261028T180049Z (Thanksgiving, Thu 28 Nov 2024), whose answer quoted Friday 29 Nov weather as
28 Nov's and called the Thursday baseline 'the other Saturdays' while every check passed."""
import pytest

from whybot.agents import Analyst
from whybot.checks import (CHECKS_VERSION, check_calendar_sql, check_claim_dates, check_claim_numbers, check_text_dates,
                           month_days, weekdays_text)
from whybot.evidence import Evidence
from whybot.schemas import CitedValue, Claim, DerivedValue

QUESTION = "Why did NYC yellow taxi trips drop on Thursday 28 Nov 2024 compared with the surrounding weeks?"


def _ev(qid, columns, rows, tables=("whybot.yellow_daily",)):
    return Evidence(qid=qid, hypothesis_id="H", purpose="p", iteration=1, status="ok", sql_hash=f"sha256:{qid}",
                    columns=columns, rows=rows, row_count=len(rows), tables=list(tables))


Q1 = _ev("Q1", ["trip_date", "target_trips", "baseline_avg_trips"],
         [["2024-11-07", None, 130018.0], ["2024-11-14", None, 141642.0], ["2024-11-21", None, 153149.0],
          ["2024-11-28", 77244, None]])
Q2 = _ev("Q2", ["trip_date", "holiday_name", "is_national", "d_day_name", "trip_count"],
         [["2024-11-07", None, None, "Thursday", 130018], ["2024-11-14", None, None, "Thursday", 141642],
          ["2024-11-21", None, None, "Thursday", 153149], ["2024-11-28", "Thanksgiving Day", True, "Thursday", 77244]])
# Q3 filtered d_dow = 5, which is Friday in TPCDS: no 2024-11-28 row at all
Q3 = _ev("Q3", ["trip_date", "prcp_mm", "snow_mm", "tmax_c", "tmin_c", "is_holiday"],
         [["2024-11-01", 0.0, 0, 26.1, 13.9, "No"], ["2024-11-08", 0.0, 0, 20.6, 10.0, "No"],
          ["2024-11-15", 0.0, 0, 14.4, 5.0, "No"], ["2024-11-22", 13.5, 0, 4.4, 2.8, "No"],
          ["2024-11-29", 0.0, 0, 8.3, 1.1, "No"], ["2024-12-06", 0.0, 0, 1.1, -1.0, "No"],
          ["2024-12-13", 0.0, 0, 0.6, -2.7, "No"], ["2024-12-20", 1.0, 0, 2.8, 0.6, "No"],
          ["2024-12-27", 0.0, 0, 6.7, -3.2, "No"]], tables=("whybot.nyc_weather_daily",))
Q4 = _ev("Q4", ["weekday", "avg_trips"], [["Thursday", 141603.0]])
Q5 = _ev("Q5", ["trip_date", "target_avg_dist", "baseline_avg_dist"],
         [["2024-11-01", 6.659148013653906, 6.659148013653906], ["2024-11-08", 4.015269273501627, 4.015269273501627],
          ["2024-11-15", 9.815712506174702, 9.815712506174702], ["2024-11-22", 4.444827485068259, 4.444827485068259],
          ["2024-11-29", 4.01485039497579, 4.01485039497579]])
Q6 = _ev("Q6", ["trip_date", "trip_count"], [["2024-11-28", 77244]])
EV = {q.qid: q for q in (Q1, Q2, Q3, Q4, Q5, Q6)}


def _cv(qid, col, val, hint):
    return CitedValue(qid=qid, column=col, value=val, row_hint=hint)


def _claim(statement, cited, cid="c", hyp="H0", derived=()):
    return Claim(cid=cid, hypothesis_id=hyp, statement=statement, direction="supports", cited_values=list(cited), derived=list(derived))


C1_CITES = [_cv("Q1", "target_trips", 77244, "trip_date=2024-11-28"),
            _cv("Q1", "baseline_avg_trips", 130018.0, "trip_date=2024-11-07"),
            _cv("Q1", "baseline_avg_trips", 141642.0, "trip_date=2024-11-14"),
            _cv("Q1", "baseline_avg_trips", 153149.0, "trip_date=2024-11-21")]
C1_DERIVED = [DerivedValue(name="pct_change_vs_baseline", value=None, formula="(v1 - avg(v2,v3,v4)) / avg(v2,v3,v4) * 100")]
# the run's c3, verbatim: values from Fri 29 Nov, Fri 22 Nov and Fri 15 Nov presented as 28 Nov and 'baseline Thursdays'
C3 = _claim("Weather on 2024-11-28 had prcp_mm=0 and tmax_c=8.3, similar to baseline Thursdays with no extreme precipitation or cold",
            [_cv("Q3", "prcp_mm", 0.0, "trip_date=2024-11-29"), _cv("Q3", "tmax_c", 8.3, "trip_date=2024-11-29"),
             _cv("Q3", "prcp_mm", 13.5, "trip_date=2024-11-22"), _cv("Q3", "tmax_c", 4.4, "trip_date=2024-11-22"),
             _cv("Q3", "prcp_mm", 0.0, "trip_date=2024-11-15"), _cv("Q3", "tmax_c", 14.4, "trip_date=2024-11-15")], "c3", "H2")


def test_checks_version_is_6_or_later():
    assert CHECKS_VERSION >= 6


# ---- a claim about date D needs a row dated D --------------------------------------------------------------------------
def test_friday_weather_quoted_as_thursday_is_rejected():
    probs = check_claim_dates(C3, EV)
    assert any("2024-11-28" in p and "row dated" in p for p in probs), probs
    assert any("Thursdays" in p and "Friday" in p for p in probs), probs
    ch = check_claim_numbers(C3.model_copy(deep=True), EV, QUESTION)
    assert not ch.ok  # so the Critic marks it unsupported whatever the LLM verdict says


def test_value_check_alone_still_passes_the_wrong_day_claim_at_v5():
    # documents why the run passed: both numbers exist in Q3, just on 2024-11-29
    assert check_claim_numbers(C3.model_copy(deep=True), EV, QUESTION, version=5).ok


def test_the_real_28_nov_weather_row_backs_the_claim():
    q = _ev("Q3", ["trip_date", "d_day_name", "prcp_mm", "snow_mm", "tmax_c", "tmin_c"],
            [["2024-11-28", "Thursday", 21.8, 0, 9.4, 5.6]], tables=("whybot.nyc_weather_daily",))
    c = _claim("Weather on 2024-11-28 (Thursday) had prcp_mm=21.8 and tmax_c=9.4",
               [_cv("Q3", "prcp_mm", 21.8, "trip_date=2024-11-28"), _cv("Q3", "tmax_c", 9.4, "trip_date=2024-11-28")])
    assert check_claim_dates(c, {"Q3": q}) == []


def test_distance_claim_about_28_nov_citing_only_fridays_is_rejected():
    c5 = _claim("avg_trip_distance on 2024-11-28 is not available; baseline Thursdays ranged from 4.02 to 9.82 miles",
                [_cv("Q5", "baseline_avg_dist", v, f"trip_date={d}") for d, v, _ in Q5.rows], "c5", "H4")
    probs = check_claim_dates(c5, EV)
    assert any("Thursdays" in p for p in probs), probs


def test_a_date_named_only_to_exclude_it_needs_no_row():
    c = _claim("Thursdays in November 2024 excluding 2024-11-28 averaged 141603.0 trips",
               [_cv("Q1", "baseline_avg_trips", 130018.0, "trip_date=2024-11-07")])
    assert not [p for p in check_claim_dates(c, EV) if "row dated" in p]


# ---- weekday labels must be the weekday of the cited rows ------------------------------------------------------------
def test_thursday_baseline_called_saturdays_is_rejected():
    c1 = _claim("Trips on 2024-11-28 were {pct_change_vs_baseline}% vs the mean of the other Saturdays in November 2024",
                C1_CITES, "c1", derived=C1_DERIVED)
    probs = check_claim_dates(c1, EV)
    assert probs and "'Saturdays'" in probs[0] and "Thursday" in probs[0], probs
    ch = check_claim_numbers(c1.model_copy(deep=True), EV, QUESTION)
    assert not ch.ok and any("Saturdays" in p for p in ch.problems)


def test_thursday_baseline_called_thursdays_passes():
    c1 = _claim("Trips on 2024-11-28 were {pct_change_vs_baseline}% vs the mean of the other Thursdays in November 2024",
                C1_CITES, "c1", derived=C1_DERIVED)
    c = c1.model_copy(deep=True)
    ch = check_claim_numbers(c, EV, QUESTION)
    assert ch.ok, ch.problems
    assert "-45.5%" in c.statement, c.statement  # (77244 - 141603) / 141603


@pytest.mark.parametrize("cid, statement, cited", [
    ("c2", "2024-11-28 is Thanksgiving Day (is_national=true) while baseline Thursdays are not holidays",
     [_cv("Q2", "is_national", 1.0, "trip_date=2024-11-28"), _cv("Q2", "holiday_name", "Thanksgiving Day", "trip_date=2024-11-28"),
      _cv("Q2", "trip_count", 130018, "trip_date=2024-11-07"), _cv("Q2", "trip_count", 141642, "trip_date=2024-11-14"),
      _cv("Q2", "trip_count", 153149, "trip_date=2024-11-21")]),
    ("c4", "Average Thursday trips in November 2024 (excluding 28th) were 141603.0, indicating 28 Nov is below the normal "
           "Thursday level", [_cv("Q4", "avg_trips", 141603.0, "weekday=Thursday")]),
    ("c6", "Row exists in yellow_daily for 2024-11-28 with trip_count=77244 (>0), so data is present and plausible",
     [_cv("Q6", "trip_count", 77244, "trip_date=2024-11-28")]),
])
def test_the_runs_correct_claims_still_pass(cid, statement, cited):
    assert check_claim_dates(_claim(statement, cited, cid), EV) == []


# ---- the SQL that picked Fridays ---------------------------------------------------------------------------------------
Q3_SQL = ("SELECT y.trip_date, w.prcp_mm, w.snow_mm, w.tmax_c, w.tmin_c, CASE WHEN h.holiday_name IS NOT NULL THEN 'Yes' ELSE 'No' END "
          "AS is_holiday FROM whybot.yellow_daily y JOIN whybot.nyc_weather_daily w ON w.obs_date = y.trip_date LEFT JOIN "
          "whybot.us_holidays h ON h.holiday_date = y.trip_date JOIN TPCDS_DB.date_dim d ON d.d_date = y.trip_date WHERE "
          "y.trip_date BETWEEN DATE '2024-11-01' AND DATE '2024-12-31' AND d.d_dow = 5 ORDER BY y.trip_date LIMIT 500")
Q3_PURPOSE = "Compare weather (prcp_mm, snow_mm, tmax_c, tmin_c) on 2024-11-28 vs baseline Thursdays from nyc_weather_daily."
Q4_SQL = ("SELECT d.d_day_name AS weekday, AVG(y.trips) AS avg_trips FROM whybot.yellow_daily y JOIN TPCDS_DB.date_dim d ON "
          "d.d_date = y.trip_date WHERE d.d_dow = 4 AND y.trip_date BETWEEN DATE '2024-11-01' AND DATE '2024-11-30' AND "
          "y.trip_date <> DATE '2024-11-28' GROUP BY d.d_day_name ORDER BY d.d_day_name LIMIT 500")
Q5_SQL = ("SELECT y.trip_date, y.avg_trip_distance AS target_avg_dist, AVG(CASE WHEN d.d_dow IN (1,2,3,4,5) AND y.trip_date <> "
          "DATE '2024-11-28' THEN y.avg_trip_distance END) AS baseline_avg_dist FROM whybot.yellow_daily y JOIN TPCDS_DB.date_dim d "
          "ON d.d_date = y.trip_date WHERE y.trip_date BETWEEN DATE '2024-11-01' AND DATE '2024-11-30' AND d.d_dow = 5 "
          "GROUP BY y.trip_date, y.avg_trip_distance ORDER BY y.trip_date LIMIT 500")


def test_spark_numbered_thursday_filter_is_refused():
    probs = check_calendar_sql(Q3_SQL, QUESTION + "\n" + Q3_PURPOSE)
    assert probs and "Friday" in probs[0] and "Thursday is 4" in probs[0] and "d_day_name" in probs[0], probs
    assert check_calendar_sql(Q5_SQL, QUESTION + "\nCompare avg_trip_distance on 2024-11-28 vs baseline Thursdays")


def test_correct_thursday_filters_pass():
    assert check_calendar_sql(Q4_SQL, QUESTION + "\nCompute average trips for all Thursdays in November 2024") == []
    assert check_calendar_sql(Q3_SQL.replace("d.d_dow = 5", "d.d_day_name = 'Thursday'"), QUESTION + "\n" + Q3_PURPOSE) == []
    assert check_calendar_sql(Q3_SQL) == []  # without intent (checks < 6) only d_dow = 7 is refused


def test_friday_filter_is_fine_when_the_purpose_asks_for_fridays():
    assert check_calendar_sql(Q3_SQL, QUESTION + "\nWeather on Friday 2024-11-29, the day after Thanksgiving") == []


# ---- the Writer's false date that forced the deterministic fallback ---------------------------------------------------
@pytest.mark.parametrize("text", ["Trips on Thursday 28 marked a -45.5% drop", "a drop of 28 marked by the holiday",
                                  "trips 28 declined", "markedly 28 fewer"])
def test_words_starting_with_a_month_abbreviation_are_not_dates(text):
    assert month_days(text) == set()
    assert check_text_dates(text, EV, QUESTION) == []


@pytest.mark.parametrize("text, md", [("28 Nov", (11, 28)), ("Nov 28", (11, 28)), ("28 November 2024", (11, 28)),
                                      ("Sept 3", (9, 3)), ("2024-11-28", (11, 28)), ("Dec. 25", (12, 25)), ("21 March", (3, 21))])
def test_real_month_names_are_still_dates(text, md):
    assert month_days(text) == {md}


def test_an_invented_date_is_still_flagged():
    assert check_text_dates("rain on 28 Mar", EV, QUESTION)


# ---- prompts ---------------------------------------------------------------------------------------------------------
def test_weekday_table_lists_the_weekday_of_every_result_date():
    t = weekdays_text(list(EV.values()))
    assert "2024-11-28 Thursday" in t and "2024-11-29 Friday" in t


def test_analyst_prompt_no_longer_shows_a_saturday_example_at_v6():
    seen = {}

    class Spy:
        def complete_json(self, role, system, user, schema, validator=None, max_tokens=0):
            seen["system"], seen["user"] = system, user
            raise RuntimeError("stop")

    with pytest.raises(RuntimeError):
        Analyst(Spy()).draft(QUESTION, "H0", list(EV.values()), "")
    assert "other Saturdays" not in seen["system"] and "2024-11-28 Thursday" in seen["user"]
    with pytest.raises(RuntimeError):
        Analyst(Spy(), version=5).draft(QUESTION, "H0", list(EV.values()), "")
    assert "other Saturdays" in seen["system"]  # replays of v5 cassettes keep their exact prompt
