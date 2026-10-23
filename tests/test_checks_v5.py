"""Checks v5: the remaining causes found re-recording under v4 (units, aggregate labels, comparison scope, plan
thresholds), each pinned by a test."""
import json

import pytest
from conftest import FakeLLM

from whybot.agents import Analyst, Writer
from whybot.checks import (CHECKS_VERSION, check_claim_numbers, check_units, column_unit, units_text, validate_final)
from whybot.evidence import Evidence
from whybot.schemas import Citation, CitedValue, Claim, FinalAnswer, FinalClaim

MONDAYS = Evidence(qid="Q1", hypothesis_id="H0", purpose="p", iteration=1, status="ok", sql_hash="sha256:q1",
                   columns=["trip_date", "trips", "avg_trip_distance", "avg_total_amount"], tables=["whybot.yellow_daily"],
                   rows=[["2025-01-06", 80127, 4.40, 26.11], ["2025-01-13", 100492, 5.53, 27.20], ["2025-01-20", 90094, 4.79, 36.48],
                         ["2025-01-27", 95000, 4.65, 26.85]], row_count=4)
WEATHER = Evidence(qid="Q2", hypothesis_id="H1", purpose="p", iteration=1, status="ok", sql_hash="sha256:q2",
                   columns=["obs_date", "prcp_mm", "snow_mm", "tmax_c"], tables=["whybot.nyc_weather_daily"],
                   rows=[["2025-01-20", 0.0, 0.0, -6.1], ["2025-01-13", 3.3, 0.0, 2.8]], row_count=2)
EV = {"Q1": MONDAYS, "Q2": WEATHER}


def _answer(text, qids=("Q1",)):
    return FinalAnswer(headline="h", answer=text, recommendation="r", confidence=0.9, caveats=[], lessons=[],
                       claims=[FinalClaim(text="t", citations=[Citation(qid=q, sql_hash=EV[q].sql_hash, table=EV[q].tables[0]) for q in qids])])


def test_checks_version_is_5():
    assert CHECKS_VERSION == 5


# ---- (a) units -------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("col, unit", [("avg_trip_distance", "mi"), ("trip_distance", "mi"), ("prcp_mm", "mm"), ("snow_mm", "mm"),
                                       ("tmax_c", "C"), ("tmin_c", "C"), ("avg_total_amount", "USD"), ("fare_amount", "USD"),
                                       ("trips", None), ("trip_date", None), ("distance_km", "km")])
def test_column_units_come_from_the_column_name(col, unit):
    assert column_unit(col) == unit


@pytest.mark.parametrize("text", [
    "2025-01-20 avg_trip_distance of 4.79 km is within the baseline range (4.40–5.53 km)",   # v4 take-1 #3 claim
    "Average trip distance rose from ~4.3 km baseline to 4.79 km",                         # v4 take-1 #1 answer pattern
    "a 4.79-mile trip, about 7.7 kilometres",
])
def test_km_for_a_distance_in_miles_is_flagged(text):
    probs = check_units(text, EV, {"Q1"})
    assert probs and all("miles" in p for p in probs), probs


@pytest.mark.parametrize("text, bad", [
    ("High of -6.1 mm on MLK Day", "deg C"),       # a temperature written as a depth
    ("Average fare 36.48 EUR", "USD"),
    ("It rained 3.3 inches on 13 Jan", "mm"),
    ("tmax was -6.1 °F", "deg C"),
])
def test_other_unit_mismatches_are_flagged(text, bad):
    probs = check_units(text, EV, {"Q1", "Q2"})
    assert any(bad in p for p in probs), probs


@pytest.mark.parametrize("text", [
    "avg_trip_distance of 4.79 miles (4.40–5.53 mi); fare $36.48 (36.48 USD); 3.3 mm rain; high of -6.1 °C",
    "90,094 trips, within 80,127–100,492",
    "4.79 miles vs a 4.40 mile low",
])
def test_correct_units_pass(text):
    assert check_units(text, EV, {"Q1", "Q2"}) == []


def test_units_are_checked_in_claims_and_the_answer_only_from_v5():
    c = Claim(cid="C1", hypothesis_id="H2", statement="avg_trip_distance on 2025-01-20 was 4.79 km", direction="refutes",
              cited_values=[CitedValue(qid="Q1", column="avg_trip_distance", value=4.79, row_hint="trip_date=2025-01-20")])
    assert check_claim_numbers(c.model_copy(deep=True), EV, version=4).ok
    v5 = check_claim_numbers(c.model_copy(deep=True), EV, version=5)
    assert not v5.ok and any("miles" in p for p in v5.problems)
    ans = _answer("Distance was 4.79 km.")
    assert not any("miles" in p for p in validate_final(ans, EV, {"Q1"}, [], version=4))
    assert any("miles" in p for p in validate_final(ans, EV, {"Q1"}, [], version=5))


def test_units_text_lists_the_column_units():
    t = units_text(EV)
    assert "avg_trip_distance = miles" in t and "never write km" in t and "tmax_c = deg C" in t and "prcp_mm, snow_mm = mm" in t


def _prompts(llm):
    return [m[-1]["content"] for _, m in llm.calls]


def test_analyst_and_writer_prompts_get_units_only_from_v5():
    claim = {"claims": [{"cid": "C1", "hypothesis_id": "H0", "statement": "Trips on 2025-01-20 were 90094", "direction": "supports",
                         "cited_values": [{"qid": "Q1", "column": "trips", "value": 90094, "row_hint": "trip_date=2025-01-20"}]}]}
    for v, expect in ((4, False), (5, True)):
        llm = FakeLLM({"analyst": [claim]})
        Analyst(llm, v).draft("q", "H0", [MONDAYS], "")
        assert ("UNITS:" in _prompts(llm)[0]) is expect
        ans = _answer("Trips were 90,094.").model_dump()
        llm = FakeLLM({"writer": [ans]})
        Writer(llm, v).write("q", "answered", [{"cid": "C1", "cited_values": [{"qid": "Q1"}]}], [], [MONDAYS], [], lambda a: [])
        assert ("avg_trip_distance = miles" in _prompts(llm)[0]) is expect


def test_catalog_note_says_distances_are_miles():
    notes = json.loads(open("whybot/catalog_notes.json").read())
    assert "MILES" in notes["whybot.yellow_daily"] and "never write km" in notes["whybot.yellow_daily"]


# ---- (b) a value labelled average / mean / baseline is the aggregate of the cited rows ----------------------------------------
from whybot.checks import check_aggregate_labels  # noqa: E402

THURSDAYS = Evidence(qid="Q1", hypothesis_id="H0", purpose="p", iteration=1, status="ok", sql_hash="sha256:t",
                     columns=["trip_date", "trips", "avg_trip_distance", "avg_total_amount"], tables=["whybot.yellow_daily"],
                     rows=[["2024-11-07", 130018, 3.25, 28.34], ["2024-11-14", 141642, 5.09, 28.71], ["2024-11-21", 153149, 4.55, 28.76],
                           ["2024-11-28", 77244, 9.47, 25.32]], row_count=4)
# v4 take-1 #1: the SQL agent named per-day counts baseline_avg_trips
MISNAMED = Evidence(qid="Q2", hypothesis_id="H0", purpose="p", iteration=1, status="ok", sql_hash="sha256:m",
                    columns=["trip_date", "target_trips", "baseline_avg_trips"], tables=["whybot.yellow_daily"],
                    rows=[["2024-11-07", None, 130018.0], ["2024-11-14", None, 141642.0], ["2024-11-21", None, 153149.0], ["2024-11-28", 77244, None]])
ONE_VALUE = Evidence(qid="Q3", hypothesis_id="H0", purpose="p", iteration=1, status="ok", sql_hash="sha256:o",
                     columns=["target_trips", "baseline_avg_trips"], tables=["whybot.yellow_daily"], rows=[[77244, 141603.0]])
TEV = {"Q1": THURSDAYS, "Q2": MISNAMED, "Q3": ONE_VALUE}


@pytest.mark.parametrize("text", [
    "Trips fell 45.5% versus the mean of the three prior Thursdays (77,244 trips vs. baseline average of 141,642).",  # v4 take-1 #1
    "baseline Thursdays averaged 141,642 trips",
    "the mean of 141,642 trips",
    "fares fell from a baseline of 28.71 to 25.32",     # one Thursday's fare, not the mean (28.60)
    "trips were 77,244 vs a 141,700 baseline",
])
def test_a_single_rows_value_labelled_as_the_average_is_flagged(text):
    probs = check_aggregate_labels(text, TEV, {"Q1", "Q2"})
    assert probs and "not the average" in probs[0], probs


def test_the_flag_names_the_true_aggregate():
    p = check_aggregate_labels("vs. baseline average of 141,642", TEV, {"Q1"})[0]
    assert "141,603" in p


@pytest.mark.parametrize("text, qids", [
    ("77,244 trips vs. baseline average of 141,603", {"Q1"}),                  # mean of the three prior Thursdays
    ("the baseline averaged 141.6k trips", {"Q1"}),                             # stated precision 0.1k
    ("about 141,600 trips on a baseline average", {"Q1"}),
    ("distance rose (from ~4.3 miles baseline to 9.47 miles)", {"Q1"}),        # 4.30 = mean of the others; 9.47 is not labelled
    ("average fare 25.32 on 2024-11-28", {"Q1"}),                               # a per-row avg_ column, plain 'average'
    ("the baseline average was 141,603", {"Q3"}),                               # a one-value aggregate column
    ("baseline trip counts: 130,018, 141,642, 153,149", {"Q1"}),               # a list of the baseline rows
    ("the baseline Thursday 2024-11-14 had 141,642 trips", {"Q1"}),            # one named baseline day
    ("Trips were -45.5% vs the mean of the three prior Thursdays", {"Q1"}),    # a %, checked by the derived-value check
])
def test_true_aggregates_and_unlabelled_values_pass(text, qids):
    assert check_aggregate_labels(text, TEV, qids) == []


def test_a_verified_derived_value_backs_an_average():
    assert check_aggregate_labels("mean of the other Thursdays was 141,603.3", TEV, {"Q2"}, derived=[141603.33]) == []


def test_aggregate_labels_are_checked_in_claims_and_the_answer_only_from_v5():
    c = Claim(cid="C1", hypothesis_id="H0", statement="77,244 trips vs. baseline average of 141,642", direction="supports",
              cited_values=[CitedValue(qid="Q2", column="target_trips", value=77244, row_hint="trip_date=2024-11-28"),
                            CitedValue(qid="Q2", column="baseline_avg_trips", value=141642.0, row_hint="trip_date=2024-11-14")])
    assert check_claim_numbers(c.model_copy(deep=True), TEV, version=4).ok
    assert any("not the average" in p for p in check_claim_numbers(c.model_copy(deep=True), TEV, version=5).problems)
    ans = FinalAnswer(headline="h", answer="77,244 trips vs. baseline average of 141,642.", recommendation="r", confidence=0.9,
                      claims=[FinalClaim(text="t", citations=[Citation(qid="Q1", sql_hash="sha256:t", table="whybot.yellow_daily")])])
    assert not any("not the average" in p for p in validate_final(ans, TEV, {"Q1"}, [], version=4))
    assert any("not the average" in p for p in validate_final(ans, TEV, {"Q1"}, [], version=5))


# ---- (c) comparison scope ------------------------------------------------------------------------------------------------
from whybot.checks import check_comparisons, check_comparisons_v5  # noqa: E402

SATURDAYS = Evidence(qid="Q5", hypothesis_id="H2", purpose="p", iteration=1, status="ok", sql_hash="sha256:s",
                     columns=["trip_date", "trips", "avg_dist"], tables=["whybot.yellow_daily"],
                     rows=[["2024-12-07", 88213, 3.93], ["2024-12-14", 145707, 4.27], ["2024-12-21", 131279, 4.79],
                           ["2024-12-28", 159150, 6.33], ["2024-12-23", 120000, 7.10]])  # a Monday: not a compared Saturday
SEV = {"Q5": SATURDAYS}
SNOW = ("No US holiday fell on 2024-12-21 or adjacent days, and average trip distance was higher on the snow day (4.79) than "
        "on comparison Saturdays, contradicting expectations.")   # v4 take-1 #2 answer: 28 Dec was 6.33


@pytest.mark.parametrize("text", [
    SNOW,
    "Distance on 2024-12-21 (4.79) was higher than the three prior Saturdays.",
    "Distance on 2024-12-21 (4.79) was longer than baseline Saturdays.",
    "Trips on 2024-12-21 (131,279) were lower than on the other Saturdays.",   # 7 Dec had 88,213
])
def test_a_comparison_that_fails_for_one_compared_day_is_flagged(text):
    probs = check_comparisons_v5(text, SEV, {"Q5"})
    assert probs and "every compared day" in probs[0] and "on average" in probs[0], probs


def test_v4_missed_the_split_comparison():
    assert check_comparisons(SNOW, SEV, {"Q5"}) == []


@pytest.mark.parametrize("text", [
    "avg_dist on 2024-12-21 (4.79) was higher than on 2024-12-07 (3.93) and 2024-12-14 (4.27).",   # names the days it holds for
    "Trips on 2024-12-28 (159,150) were higher than on the other three Saturdays.",
    "On average, distance on 2024-12-21 (4.79) was lower than on the other Saturdays.",            # mean of the Saturdays 4.84
    "Distance on 2024-12-21 (4.79) was not higher than the other Saturdays.",
    "Trips on 2024-12-21 (131,279) were within 5% of the other Saturdays.",
])
def test_comparisons_that_hold_pass(text):
    assert check_comparisons_v5(text, SEV, {"Q5"}) == []


def test_on_average_wording_is_checked_against_the_mean():
    probs = check_comparisons_v5("On average, distance on 2024-12-21 (4.79) was higher than on the other Saturdays.", SEV, {"Q5"})
    assert probs and "mean of the others" in probs[0]


def test_only_the_same_weekday_rows_are_compared():
    # the Monday 23 Dec (7.10) is in the result but is not a compared Saturday
    assert check_comparisons_v5("Distance on 2024-12-28 (6.33) was longer than the other Saturdays.", SEV, {"Q5"}) == []


def test_comparison_scope_is_checked_in_claims_and_the_answer_only_from_v5():
    c = Claim(cid="C1", hypothesis_id="H2", statement=SNOW, direction="refutes",
              cited_values=[CitedValue(qid="Q5", column="avg_dist", value=4.79, row_hint="trip_date=2024-12-21")])
    assert not any("every compared day" in p for p in check_claim_numbers(c.model_copy(deep=True), SEV, version=4).problems)
    assert any("every compared day" in p for p in check_claim_numbers(c.model_copy(deep=True), SEV, version=5).problems)
    ans = FinalAnswer(headline="h", answer=SNOW, recommendation="r", confidence=0.9,
                      claims=[FinalClaim(text="t", citations=[Citation(qid="Q5", sql_hash="sha256:s", table="whybot.yellow_daily")])])
    assert not any("every compared day" in p for p in validate_final(ans, SEV, {"Q5"}, [], version=4))
    assert any("every compared day" in p for p in validate_final(ans, SEV, {"Q5"}, [], version=5))


def test_writer_and_analyst_get_the_v5_wording_rules_only_from_v5():
    for v, expect in ((4, False), (5, True)):
        llm = FakeLLM({"writer": [_answer("Trips were 90,094.").model_dump()]})
        Writer(llm, v).write("q", "answered", [{"cid": "C1", "cited_values": [{"qid": "Q1"}]}], [], [MONDAYS], [], lambda a: [])
        assert ("must hold for every compared day" in llm.calls[0][1][0]["content"]) is expect


# ---- (d) thresholds from the plan ------------------------------------------------------------------------------------------
from whybot.checks import plan_threshold_mentions, plan_thresholds  # noqa: E402

PLAN = ("H0 [cause] Snow suppressed demand | support if: trips on 2024-12-21 are more than 5% below the other Saturdays' mean | "
        "refute if: within 5% of the other Saturdays' mean")
FLAT = Evidence(qid="Q2", hypothesis_id="H0", purpose="p", iteration=1, status="ok", sql_hash="sha256:q2",
                columns=["trip_date", "trips", "pct_change_vs_baseline"], tables=["whybot.yellow_daily"],
                rows=[["2024-12-21", 131279, 0.15]])
FEV = {"Q2": FLAT}


def _fans(text):
    return FinalAnswer(headline="h", answer=text, recommendation="r", confidence=0.9, caveats=[], lessons=[],
                       claims=[FinalClaim(text="t", citations=[Citation(qid="Q2", sql_hash=FLAT.sql_hash, table=FLAT.tables[0])])])


def test_plan_thresholds_are_read_from_the_hypotheses():
    assert plan_thresholds(PLAN) == {5.0}


@pytest.mark.parametrize("text", [
    "Trips were +0.15% vs the other Saturdays, well within the 5% threshold.",
    "Trips were +0.15% vs the other Saturdays, inside the ±5% band, so H0 is refuted.",
    "Trips were +0.15% vs the other Saturdays, not a meaningful reduction (>5%).",
    "The +0.15% change is small; a reduction of more than 5% was not observed.",
    "The 5% threshold for a material change is analyst-defined.",              # named as a parameter
])
def test_a_plan_threshold_beside_the_measured_change_it_is_compared_with_passes(text):
    assert plan_threshold_mentions(text, {5.0}, [0.15])[1] == []
    assert validate_final(_fans(text), FEV, {"Q2"}, [], version=5, plan_text=PLAN) == []


@pytest.mark.parametrize("text", [
    "Trips on 2024-12-21 were within 5% of the other Saturdays.",             # no measured change to compare with
    "Trips were +0.15% vs the other Saturdays, more than the 5% threshold.",  # the comparison does not hold
    "Trips fell more than 5% on the snow day.",
])
def test_a_plan_threshold_used_as_a_measurement_is_flagged_with_the_reason(text):
    probs = validate_final(_fans(text), FEV, {"Q2"}, [], version=5, plan_text=PLAN)
    assert any("threshold from the plan" in p for p in probs), probs
    assert not any("not backed" in p for p in probs), probs   # one explanation, not two


def test_a_percentage_the_plan_did_not_set_is_still_number_checked():
    probs = validate_final(_fans("Trips were +0.15%, within the 7% threshold."), FEV, {"Q2"}, [], version=5, plan_text=PLAN)
    assert any("7%" in p for p in probs)


def test_plan_thresholds_are_accepted_only_from_v5():
    text = "Trips were +0.15% vs the other Saturdays, well within the 5% threshold."
    assert any("5%" in p for p in validate_final(_fans(text), FEV, {"Q2"}, [], version=4, plan_text=PLAN))
    c = Claim(cid="C1", hypothesis_id="H0", statement=text, direction="refutes",
              cited_values=[CitedValue(qid="Q2", column="pct_change_vs_baseline", value=0.15, row_hint="trip_date=2024-12-21")])
    assert any("5%" in p for p in check_claim_numbers(c.model_copy(deep=True), FEV, version=4, plan_text=PLAN).problems)
    assert not any("5%" in p for p in check_claim_numbers(c.model_copy(deep=True), FEV, version=5, plan_text=PLAN).problems)
    bad = c.model_copy(update={"statement": "Trips on 2024-12-21 were within 5% of the other Saturdays."}, deep=True)
    assert any("threshold from the plan" in p for p in check_claim_numbers(bad, FEV, version=5, plan_text=PLAN).problems)


def test_writer_is_told_how_to_quote_plan_thresholds_only_from_v5():
    for v, expect in ((4, False), (5, True)):
        llm = FakeLLM({"writer": [_answer("Trips were 90,094.").model_dump()]})
        Writer(llm, v).write("q", "answered", [{"cid": "C1", "cited_values": [{"qid": "Q1"}]}], [], [MONDAYS], [], lambda a: [])
        assert ("thresholds (e.g. 'within 5%') are plan parameters" in llm.calls[0][1][0]["content"]) is expect
