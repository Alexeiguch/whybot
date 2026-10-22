import pytest

from whybot.checks import check_claim_numbers, decide_next, drop_uncited, validate_final, value_matches
from whybot.datasource.base import DataSourceError, guard_sql, sql_hash
from whybot.evidence import Evidence
from whybot.llm import extract_json, strip_reasoning
from whybot.schemas import Claim, CitedValue, ClaimVerdict, CriticReport, FinalAnswer, FinalClaim, Citation, FollowUp, HypothesisStatus


def ev(qid="Q1", status="ok"):
    return Evidence(qid=qid, hypothesis_id="H0", purpose="p", iteration=1, sql="select 1", sql_hash=sql_hash("select 1"),
                    tables=["NY_TAXI.yellow_tripdata"], status=status, columns=["trip_date", "trips"],
                    rows=[["2024-02-06", 102152], ["2024-02-13", 69067]], row_count=2)


def claim(value, column="trips", qid="Q1", hint="trip_date=2024-02-13"):
    return Claim(cid="C1", hypothesis_id="H1", statement="s", direction="supports",
                 cited_values=[CitedValue(qid=qid, column=column, value=value, row_hint=hint)])


def test_value_matches_rounding_and_tolerance():
    assert value_matches(69067, 69067.0)
    assert value_matches("34.8", 34.83)
    assert value_matches("69,067", 69067)
    assert not value_matches(70000, 69067)


@pytest.mark.parametrize("value,column,qid,ok", [
    (69067, "trips", "Q1", True),
    (102152, "trips", "Q1", False),   # exists in data but not in the hinted row
    (12345, "trips", "Q1", False),    # invented number
    (69067, "nope", "Q1", False),     # wrong column
    (69067, "trips", "Q9", False),    # unknown query
])
def test_number_check(value, column, qid, ok):
    assert check_claim_numbers(claim(value, column, qid), {"Q1": ev()}).ok is ok


def test_number_check_rejects_failed_query():
    assert not check_claim_numbers(claim(69067), {"Q1": ev(status="failed")}).ok


def report(decision, verdict="supported", hyp="supported", follow=False):
    return CriticReport(verdicts=[ClaimVerdict(cid="C1", verdict=verdict, strength=0.9, reason="r")],
                        hypothesis_status=[HypothesisStatus(hypothesis_id="H1", status=hyp, reason="r")],
                        decision=decision, feedback="fb",
                        follow_up_queries=[FollowUp(hypothesis_id="H1", purpose="x", tables=["NY_TAXI.yellow_tripdata"])] if follow else [])


def ok_checks():
    return {"C1": check_claim_numbers(claim(69067), {"Q1": ev()})}


def test_accept_writes():
    d = decide_next(1, 3, report("accept"), ok_checks(), {"Q1": ev()})
    assert d.action == "write" and d.supported_cids == ["C1"]


def test_numeric_failure_overrides_llm_supported():
    bad = {"C1": check_claim_numbers(claim(1), {"Q1": ev()})}
    d = decide_next(1, 3, report("accept"), bad, {"Q1": ev()})
    assert d.action == "retry" and d.supported_cids == []


def test_retry_until_cap_then_escalate():
    r = report("retry", verdict="weak", hyp="inconclusive", follow=True)
    assert decide_next(1, 3, r, ok_checks(), {"Q1": ev()}).action == "retry"
    assert decide_next(2, 3, r, ok_checks(), {"Q1": ev()}).action == "retry"
    assert decide_next(3, 3, r, ok_checks(), {"Q1": ev()}).action == "escalate"


def test_cap_reached_with_support_writes():
    assert decide_next(3, 3, report("retry", follow=True), ok_checks(), {"Q1": ev()}).action == "write"


def test_insufficient_without_followups_escalates_early():
    assert decide_next(1, 3, report("insufficient_evidence", verdict="weak", hyp="inconclusive"), ok_checks(), {"Q1": ev()}).action == "escalate"


def test_timeout_escalates():
    r = report("retry", verdict="weak", hyp="inconclusive", follow=True)
    assert decide_next(1, 3, r, ok_checks(), {"Q1": ev()}, out_of_time=True).action == "escalate"


def test_writer_citation_validation_and_drop():
    e = ev()
    good = Citation(qid="Q1", sql_hash=e.sql_hash, table="NY_TAXI.yellow_tripdata")
    bad = Citation(qid="Q1", sql_hash="sha256:wrong", table="NY_TAXI.yellow_tripdata")
    ans = FinalAnswer(headline="h", answer="a", recommendation="r", confidence=0.8,
                      claims=[FinalClaim(text="ok", citations=[good]), FinalClaim(text="bad", citations=[bad]), FinalClaim(text="none", citations=[])])
    probs = validate_final(ans, {"Q1": e}, {"Q1"})
    assert any("sql_hash" in p for p in probs) and any("no citation" in p for p in probs)
    fixed = drop_uncited(ans, {"Q1": e}, {"Q1"})
    assert [c.text for c in fixed.claims] == ["ok"] and validate_final(fixed, {"Q1": e}, {"Q1"}) == []
    assert any("not used by any critic-supported" in p for p in validate_final(ans.model_copy(update={"claims": [FinalClaim(text="x", citations=[good])]}), {"Q1": e}, set()))


@pytest.mark.parametrize("sql", ["DROP TABLE x", "select 1; select 2", "insert into t values (1)", "CACHE TABLE NY_TAXI.x", "with a as (select 1) delete from t"])
def test_guard_sql_rejects(sql):
    with pytest.raises(DataSourceError):
        guard_sql(sql)


def test_guard_sql_adds_limit_and_allows_keywords_in_literals():
    assert guard_sql("select 'drop' as x").endswith("LIMIT 500")
    assert guard_sql("select 1 limit 5") == "select 1 limit 5"


def test_reasoning_is_stripped_and_json_extracted():
    raw = "<think>secret chain of thought {\"no\": 1}</think>Sure:\n```json\n{\"a\": {\"b\": \"}\"}}\n```"
    assert "secret" not in strip_reasoning(raw)
    assert extract_json(raw) == {"a": {"b": "}"}}


def test_result_sanity_flags_cross_join_and_empty():
    from whybot.checks import result_sanity
    same = [["2024-02-06", 196, 81], ["2024-02-13", 196, 81], ["2024-02-20", 196, 81]]
    assert result_sanity(["dt", "prcp", "snow"], same)
    assert result_sanity(["dt", "n"], [])
    assert result_sanity(["dt", "prcp"], [["a", 1], ["b", 0], ["c", 1]]) == []


def _claim_txt(statement, vals, derived=()):
    from whybot.schemas import DerivedValue
    return Claim(cid="C1", hypothesis_id="H0", statement=statement, direction="supports",
                 cited_values=[CitedValue(qid="Q1", column="trips", value=v) for v in vals],
                 derived=[DerivedValue(name=n, value=v, formula=f) for n, v, f in derived])


BASE = "(v1 - avg(v2,v3,v4)) / avg(v2,v3,v4) * 100"
VALS = [77244, 130018, 141642, 153149]          # target + three baseline Thursdays (real recorded rows)


def test_derived_percentage_must_match_cited_cells():
    from whybot.checks import check_derived_values
    assert check_derived_values(_claim_txt("Trips fell 45.5% vs the baseline average", VALS, [("pct", -45.45, BASE)])) == []
    bad = check_derived_values(_claim_txt("Trips fell 49.5% vs the baseline average", VALS, [("pct", -49.5, BASE)]))
    assert bad and "45.45" in bad[0]                                    # the reviewed bug, now rejected
    assert check_derived_values(_claim_txt("Trips fell 49.5%", VALS))    # undeclared percentage
    assert check_derived_values(_claim_txt("Baseline was 1.83x the target", VALS, [("r", 1.83, "avg(v2,v3,v4) / v1")])) == []
    assert check_derived_values(_claim_txt("x", VALS, [("r", 1, "__import__('os')")]))


def test_derived_check_rejects_through_number_check():
    from whybot.schemas import DerivedValue
    e = Evidence(qid="Q1", hypothesis_id="H0", purpose="p", iteration=1, status="ok", columns=["d", "trips"],
                 rows=[["a", 77244], ["b", 130018], ["c", 141642], ["e", 153149]], row_count=4, tables=["t"])
    c = Claim(cid="C1", hypothesis_id="H0", statement="down 49.5%", direction="supports",
              cited_values=[CitedValue(qid="Q1", column="trips", value=v) for v in VALS],
              derived=[DerivedValue(name="pct", value=-49.5, formula=BASE)])
    ch = check_claim_numbers(c, {"Q1": e})
    assert not ch.ok and "49.5" in ch.problems[0]


def test_value_matches_has_no_prefix_or_loose_rounding():
    assert not value_matches("2", "2024-02-13")
    assert not value_matches("Thanks", "Thanksgiving Day")
    assert not value_matches(5, 5.4) and not value_matches(0, 0.49)
    assert value_matches("5.4", 5.43) and value_matches("Thanksgiving Day", "thanksgiving day")


def test_row_hint_that_matches_no_row_fails():
    c = claim(69067, hint="trip_date=2024-02-29")
    ch = check_claim_numbers(c, {"Q1": ev()})
    assert not ch.ok and "no row matches" in ch.problems[0]


def test_numbers_in_final_text_must_be_backed():
    from whybot.checks import check_text_numbers
    e = Evidence(qid="Q1", hypothesis_id="H0", purpose="p", iteration=1, status="ok", columns=["d", "trips"],
                 rows=[["2024-11-07", 130018], ["2024-11-14", 141642], ["2024-11-21", 153149], ["2024-11-28", 77244]], row_count=4, tables=["t"])
    ok = "On 28 Nov 2024 trips fell 45.5% to 77,244 versus about 141.6k on the other 3 Thursdays of 2024."
    assert check_text_numbers(ok, {"Q1": e}, {"Q1"}, [-45.45]) == []
    bad = "Trips fell 90% to 70,000."
    probs = check_text_numbers(bad, {"Q1": e}, {"Q1"}, [-45.45])
    assert len(probs) == 2


def test_guard_sql_blocks_functions_tables_and_keeps_literals():
    import pytest
    from whybot.datasource.base import DataSourceError, guard_sql, sql_hash
    allowed = {"whybot.yellow_daily"}
    for bad in ["SELECT reflect('java.lang.System','getenv') FROM whybot.yellow_daily",
                "SELECT java_method('x','y')", "SELECT * FROM NY_TAXI.trips",
                "SELECT * FROM whybot.yellow_daily a JOIN NY_TAXI.weather_data b ON 1=1",
                "SELECT 1; DROP TABLE x", "DELETE FROM whybot.yellow_daily"]:
        with pytest.raises(DataSourceError):
            guard_sql(bad, allowed_tables=allowed)
    ok = guard_sql("SELECT 'a -- not a comment;' AS s, trips FROM whybot.yellow_daily -- real comment", allowed_tables=allowed)
    assert "'a -- not a comment;'" in ok and "real comment" not in ok and ok.endswith("LIMIT 500")
    assert sql_hash("select 'Thanksgiving'") != sql_hash("select 'thanksgiving'")
    assert sql_hash("SELECT 'X'") == sql_hash("select 'X'")


def test_small_llm_arithmetic_slips_are_recomputed_not_trusted():
    from whybot.checks import check_derived_values
    c = _claim_txt("Trips fell 45.9% vs the baseline average", VALS, [("pct", -45.9, BASE)])
    fixes = []
    assert check_derived_values(c, fixes) == []
    assert fixes and "45.5" in c.statement and "45.9" not in c.statement


def test_dates_in_final_text_must_appear_in_results_or_notes():
    from whybot.checks import check_text_dates
    e = Evidence(qid="Q1", hypothesis_id="H0", purpose="p", iteration=1, status="ok", columns=["d", "trips"],
                 rows=[["2024-11-21", 153149], ["2024-11-28", 77244]], row_count=2, tables=["t"])
    note = "2024-11-11: d_holiday=N but Nager.Date says Veterans Day"
    assert check_text_dates("Thanksgiving, 28 Nov; rain on Nov 21st; Veterans Day (11 Nov) mislabelled", {"Q1": e}, note) == []
    assert check_text_dates("same issue for 14 Nov (Veterans Day)", {"Q1": e}, note)


def test_placeholders_are_filled_by_the_evaluator():
    from whybot.checks import check_derived_values
    from whybot.schemas import DerivedValue
    c = _claim_txt("Trips fell {pct}% vs the baseline average", VALS)
    c.derived = [DerivedValue(name="pct", formula=BASE)]
    assert check_derived_values(c) == [] and c.statement == "Trips fell 45.5% vs the baseline average"
    c2 = _claim_txt("Trips fell {nope}%", VALS)
    assert check_derived_values(c2)


def test_boolean_flags_match_across_encodings():
    assert value_matches("True", 1) and value_matches("False", 0) and value_matches(True, "true") and value_matches("Y", 1)
    assert not value_matches("True", 0) and not value_matches("2", "2024-02-13")


def test_direction_of_change_must_match_the_sign():
    from whybot.checks import check_derived_values
    from whybot.schemas import DerivedValue
    snow = [131279, 145899, 159150, 88213]   # real Saturdays Dec 2024: 21 Dec vs 7/14/28 Dec -> +0.15%
    c = _claim_txt("Trips on 21 Dec were lower than the other Saturdays", snow)
    c.derived = [DerivedValue(name="pct_change_vs_baseline", formula=BASE)]
    probs = check_derived_values(c)
    assert probs and "decrease" in probs[0]
    c2 = _claim_txt("Trips fell {pct}% vs the baseline", VALS)
    c2.derived = [DerivedValue(name="pct", formula=BASE)]
    assert check_derived_values(c2) == []


def test_text_percentages_need_a_verified_value():
    from whybot.checks import check_text_numbers
    e = Evidence(qid="Q1", hypothesis_id="H0", purpose="p", iteration=1, status="ok", columns=["d", "trips"],
                 rows=[["2024-12-07", 145899], ["2024-12-14", 159150], ["2024-12-21", 131279], ["2024-12-28", 88213]], row_count=4, tables=["t"])
    assert check_text_numbers("trips were 15% below the baseline", {"Q1": e}, {"Q1"}, [0.15])
    assert check_text_numbers("trips were 0.15% above the baseline", {"Q1": e}, {"Q1"}, [0.15]) == []


def test_neutral_placeholder_keeps_its_sign():
    from whybot.checks import check_derived_values
    from whybot.schemas import DerivedValue
    c = _claim_txt("Trips on 28 Nov were {pct}% vs the baseline mean", VALS)
    c.derived = [DerivedValue(name="pct", formula=BASE)]
    assert check_derived_values(c) == [] and "-45.5%" in c.statement


# ---- consistency: wording that the evidence contradicts (recorded sample #1, Thanksgiving 2024) ----
from whybot.checks import check_consistency, check_dq_consistency, check_weather_wording  # noqa: E402

THANKSGIVING_Q = "Why did NYC yellow taxi trips drop on Thanksgiving, 2024-11-28?"
DQ_NOTES = ["2024-11-28: TPCDS date_dim.d_holiday=N but Nager.Date says Thanksgiving Day (national holiday) [DQ1 sha256:x]"]


def wx(prcp_28=21.8, snow_28=0.0):
    sql = "select obs_date, prcp_mm, snow_mm from whybot.nyc_weather_daily"
    return {"Q3": Evidence(qid="Q3", hypothesis_id="H2", purpose="weather", iteration=1, sql=sql, sql_hash=sql_hash(sql),
                           tables=["whybot.nyc_weather_daily"], status="ok", columns=["obs_date", "prcp_mm", "snow_mm", "tmax_c"],
                           rows=[["2024-11-21", 0.0, 0.0, 9.4], ["2024-11-28", prcp_28, snow_28, 8.3]], row_count=2)}


@pytest.mark.parametrize("text,bad", [
    ("Data for 2024-11-28 is complete; weather conditions were mild.", True),   # the recorded contradiction
    ("Mild weather on Thanksgiving did not explain the drop.", True),          # date comes from the question
    ("It was a dry day on 2024-11-28.", True),
    ("There was no rain on 2024-11-28.", True),
    ("Clear skies on 2024-11-21 the week before.", False),                     # that day really was dry
    ("Heavy rain (21.8 mm) fell on 2024-11-28.", False),
    ("The weather was not mild: 21.8 mm of rain.", False),                     # negated
    ("The evidence is clear and the drop was mild relative to 2023.", False),  # not weather words
    ("Trips fell versus other Thursdays with no rain.", False),               # describes the comparison days
])
def test_benign_weather_words_must_match_recorded_precipitation(text, bad):
    assert bool(check_weather_wording(text, wx(), THANKSGIVING_Q)) is bad


def test_snow_makes_mild_wrong_and_threshold_is_respected():
    assert check_weather_wording("Weather was mild on 2024-11-28.", wx(prcp_28=0.0, snow_28=12.0))
    assert check_weather_wording("There was no snow on 2024-11-28.", wx(prcp_28=0.0, snow_28=12.0))
    assert not check_weather_wording("Weather was mild on 2024-11-28.", wx(prcp_28=0.3))  # a trace is still mild
    assert not check_weather_wording("Weather was mild on 2024-11-28.", {})               # no weather evidence, no verdict


def test_weather_contradiction_rejects_the_claim_deterministically():
    ev3 = wx()
    c = Claim(cid="C5", hypothesis_id="H2", statement="Weather on 2024-11-28 was mild and dry.", direction="refutes",
              cited_values=[CitedValue(qid="Q3", column="prcp_mm", value=21.8, row_hint="obs_date=2024-11-28")])
    assert not check_claim_numbers(c, ev3, THANKSGIVING_Q).ok
    c2 = c.model_copy(update={"statement": "On 2024-11-28 21.8 mm of rain fell."})
    assert check_claim_numbers(c2, ev3, THANKSGIVING_Q).ok


@pytest.mark.parametrize("text", [
    "Data for 2024-11-28 exists and appears complete; no data-quality flags.",
    "Data for 2024-11-28 is complete with no quality flags.",
    "The data is free of data quality issues.",
    "Data quality was clean.",
])
def test_no_dq_flags_cannot_coexist_with_dq_findings(text):
    assert check_dq_consistency(text, DQ_NOTES)
    assert check_dq_consistency(text, (), caveats=["Data quality: date_dim.d_holiday disagrees with Nager.Date"])
    assert not check_dq_consistency(text, ())                       # nothing was flagged: the phrase is fine
    assert not check_dq_consistency("Data for 2024-11-28 is complete.", DQ_NOTES)


def test_final_answer_with_both_sample1_contradictions_is_rejected():
    ev3 = wx()
    ans = FinalAnswer(headline="Yellow taxi trips fell 45.5% on Thanksgiving 2024",
                      answer="Data for 2024-11-28 is complete with no quality flags; weather conditions were mild.",
                      recommendation="r", claims=[FinalClaim(text="Rain on 2024-11-28 was 21.8 mm.",
                                                             citations=[Citation(qid="Q3", sql_hash=ev3["Q3"].sql_hash, table="t")])],
                      caveats=[], confidence=0.8)
    probs = validate_final(ans, ev3, {"Q3"}, context=THANKSGIVING_Q, dq_notes=DQ_NOTES)
    assert any("mild" in p for p in probs) and any("quality flags" in p for p in probs)
    fixed = ans.model_copy(update={"answer": "21.8 mm of rain fell on 2024-11-28; the TPCDS holiday flag was wrong for this date."})
    assert not check_consistency(fixed.answer, ev3, THANKSGIVING_Q, DQ_NOTES)


def test_zetaris_rows_normalise_boxed_java_values():
    import json
    from whybot.datasource.zetaris_jdbc import to_python

    class Boolean(int):  # stands in for JPype's java.lang.Boolean, an int subclass
        def __str__(self):
            return "True" if self else "False"

    class Long(int):
        pass

    row = [to_python(v) for v in (Boolean(1), Long(77244), 2.5, "x", None)]
    assert row == [True, 77244, 2.5, "x", None]
    assert [type(v) for v in row[:2]] == [bool, int]
    assert [str(v) for v in json.loads(json.dumps(row))] == [str(v) for v in row]
