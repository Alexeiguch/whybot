"""VPS live runs 1b and 2 (Thanksgiving, Thu 28 Nov 2024): the baseline summary query returned target_value and
baseline_avg but no pct_change_vs_baseline column, so WhyBot wrote no C0, the Analyst's own H0 claim failed on an
undefined v4, and the Writer could not quote the average (141,603) or the change (-45.5%). Run 1b also passed a claim
that the baseline Thursdays had no rain although 2024-11-21 had 39.9 mm."""
from test_checks_v7 import Q2, Q3, _ev

from whybot.agents import baseline_summary_claim, summary_row_qids
from whybot.checks import (_c0_is_summary, check_baseline_weather, check_claim_numbers, check_text_numbers, known_days,
                           prune_to_valid, validate_final)
from whybot.schemas import Claim, FinalAnswer

ROW = _ev("Q2S", ["target_date", "target_value", "baseline_avg", "baseline_days"], [["2024-11-28", 77244, 141603.0, 3]])
EV = {"Q2": Q2, "Q3": Q3, "Q2S": ROW}


def test_c0_is_written_when_the_summary_row_has_no_pct_column():
    c = baseline_summary_claim([Q2, ROW], "trips", "2024-11-07, 2024-11-14, 2024-11-21")
    assert c is not None and c.cid == "C0"
    assert c.statement == "Trips on 2024-11-28 were 77,244, -45.5% vs an average of 141,603 over the comparable baseline days"
    assert [d.formula for d in c.derived] == ["(v1 - v2) / v2 * 100"]
    assert check_claim_numbers(c, EV).problems == []
    assert _c0_is_summary(EV) and summary_row_qids(EV) == {"Q2S"}


def test_c0_uses_a_null_pct_cell_as_missing():
    row = _ev("Q9", ["target_date", "target_value", "baseline_avg", "pct_change_vs_baseline"],
              [["2024-11-28", 77244, 141603.0, None]])
    c = baseline_summary_claim([row], "trips", "x")
    assert "-45.5%" in c.statement and check_claim_numbers(c, {"Q9": row}).problems == []


def test_the_writer_may_quote_c0s_average_and_rounded_change():
    c = baseline_summary_claim([ROW], "trips", "x")
    check_claim_numbers(c, EV)
    derived = [d.value for d in c.derived]
    for text in ("Trips fell 45% to 77,244 vs an average of 141,603.", "Trips were -45.5% vs 141,603 on average."):
        assert check_text_numbers(text, EV, {"Q2S"}, derived, slash_days=True) == []
    assert check_text_numbers("Trips fell 47% vs 141,603.", EV, {"Q2S"}, derived, slash_days=True)


def test_a_pct_change_between_two_stated_backed_numbers_is_accepted():
    text = "Trips were 77,244 against 141,642 a week earlier, a 45% drop."
    assert check_text_numbers(text, EV, {"Q2"}, (), slash_days=True) == []
    assert check_text_numbers("Trips were 77,244, a 45% drop.", EV, {"Q2"}, (), slash_days=True)


def test_month_day_shorthand_after_an_iso_date_is_a_date_list():
    days = known_days(EV)
    text = "Baseline Thursdays (2024-11-07, 11-14, 11-21) averaged 141,603 trips."
    assert check_text_numbers(text, EV, {"Q2S"}, (), day_lists=days, slash_days=True) == []
    assert check_text_numbers("On 2024-11-07, 11-19 trips fell.", EV, {"Q2S"}, (), day_lists=days, slash_days=True)


def _rain_claim(statement):
    return Claim.model_validate({"cid": "c3", "hypothesis_id": "H2", "direction": "supports", "statement": statement,
        "cited_values": [{"qid": "Q3", "column": "prcp_mm", "value": v, "row_hint": f"trip_date={d}"} for d, v in
                         [("2024-11-28", 21.8), ("2024-11-07", 0.0), ("2024-11-14", 0.0), ("2024-11-21", 39.9)]]})


def test_no_rain_on_the_baseline_days_must_hold_for_every_cited_day():
    bad = _rain_claim("2024-11-28 had rain (prcp_mm=21.8) while baseline Thursdays had no rain (prcp_mm=0.0)")
    probs = check_baseline_weather(bad, EV)
    assert probs and "11-21" in probs[0]
    assert any("baseline days" in p for p in check_claim_numbers(bad, EV).problems)
    good = _rain_claim("2024-11-28 had 21.8 mm of rain; the other Thursdays had 0.0, 0.0 and 39.9 mm")
    assert check_baseline_weather(good, EV) == []
    dry = Claim.model_validate({**_rain_claim("x").model_dump(), "statement": "2024-11-28 had rain while the other Thursdays had no rain",
                                "cited_values": [cv for cv in _rain_claim("x").model_dump()["cited_values"] if cv["value"] != 39.9]})
    assert check_baseline_weather(dry, EV) == []


def _answer(answer, claims=("Trips on 2024-11-28 were 77,244 vs an average of 141,603.",)):
    from whybot.schemas import Citation, FinalClaim
    cit = [Citation(qid="Q2S", sql_hash="sha256:Q2S", table="whybot.yellow_daily")]
    return FinalAnswer(headline="Thanksgiving drove a 45.5% drop in trips", answer=answer, recommendation="Plan for holidays.",
                       claims=[FinalClaim(text=t, citations=cit) for t in claims], caveats=[], confidence=0.9)


def test_the_writer_draft_is_kept_without_its_one_unbacked_sentence():
    c = baseline_summary_claim([ROW], "trips", "x")
    check_claim_numbers(c, EV)
    derived = [d.value for d in c.derived]

    def check(a):
        return validate_final(a, EV, {"Q2S"}, derived)
    ans = _answer("Trips fell to 77,244 vs an average of 141,603 (-45.5%). Fares averaged 4.61 more.")
    assert check(ans.model_copy(deep=True))
    dropped = prune_to_valid(ans, check)
    assert dropped == ["Fares averaged 4.61 more."]
    assert check(ans) == [] and "141,603" in ans.answer
    head_bad = _answer("Trips fell to 77,244.").model_copy(update={"headline": "Trips fell 47%"})
    assert prune_to_valid(head_bad, check) is None and head_bad.headline == "Trips fell 47%"


def test_the_analyst_is_not_retried_for_an_h0_claim_that_c0_replaces():
    from conftest import FakeLLM
    from whybot.agents import Analyst
    claims = {"claims": [
        {"cid": "c1", "hypothesis_id": "H0", "direction": "supports",
         "statement": "Trips on 2024-11-28 were {pct_change_vs_baseline}% vs an average of {baseline_avg}",
         "cited_values": [{"qid": "Q2S", "column": "target_value", "value": 77244},
                          {"qid": "Q2S", "column": "baseline_avg", "value": 141603.0},
                          {"qid": "Q2S", "column": "baseline_days", "value": 3}],
         "derived": [{"name": "pct_change_vs_baseline", "formula": "(v1 - avg(v2,v3,v4)) / avg(v2,v3,v4) * 100"}]},
        {"cid": "c2", "hypothesis_id": "H2", "direction": "supports", "statement": "2024-11-28 had 21.8 mm of rain",
         "cited_values": [{"qid": "Q3", "column": "prcp_mm", "value": 21.8, "row_hint": "trip_date=2024-11-28"}]}]}
    llm = FakeLLM({"analyst": [claims]})
    cs, meta = Analyst(llm).draft("why?", "H0", [Q2, Q3, ROW], "")
    assert [c.cid for c in cs.claims] == ["c1", "c2"] and meta.get("attempt", 1) == 1


def test_one_analyst_draft_is_enough_when_c0_and_a_clean_cause_claim_exist():
    from conftest import FakeLLM
    from whybot.agents import Analyst
    claims = {"claims": [
        {"cid": "c1", "hypothesis_id": "H1", "direction": "supports", "statement": "2024-11-28 had 21.8 mm of rain",
         "cited_values": [{"qid": "Q3", "column": "prcp_mm", "value": 21.8, "row_hint": "trip_date=2024-11-28"}]},
        {"cid": "c2", "hypothesis_id": "H3", "direction": "supports", "statement": "Fares averaged 4.61",
         "cited_values": [{"qid": "Q3", "column": "avg_total_amount", "value": 4.61, "row_hint": "trip_date=2024-11-28"}]}]}
    cs, meta = Analyst(FakeLLM({"analyst": [claims]})).draft("why?", "H0", [Q2, Q3, ROW], "")
    assert [c.cid for c in cs.claims] == ["c1", "c2"] and meta.get("attempt", 1) == 1
    # without the summary row (no C0) the side claim's problem still earns a repair round
    fixed = {"claims": [claims["claims"][0]]}
    cs, meta = Analyst(FakeLLM({"analyst": [claims, fixed]})).draft("why?", "H0", [Q2, Q3], "")
    assert [c.cid for c in cs.claims] == ["c1"] and meta.get("attempt") == 2


def test_null_citation_on_a_missing_row_backs_a_not_a_holiday_claim():
    from whybot.checks import check_claim_numbers
    from whybot.schemas import CitedValue
    hol = _ev("Q9", ["holiday_date", "holiday_name"], [["2024-11-28", "Thanksgiving Day"]])
    c = Claim(cid="c2", hypothesis_id="H1", direction="supports",
              statement="2024-11-28 was Thanksgiving Day; the other Thursdays were not holidays",
              cited_values=[CitedValue(qid="Q9", column="holiday_name", value="Thanksgiving Day", row_hint="holiday_date=2024-11-28"),
                            CitedValue(qid="Q9", column="holiday_name", value="None", row_hint="holiday_date=2024-11-07")])
    assert check_claim_numbers(c, {"Q9": hol}, version=7).problems == []
    c.cited_values[1].value = "Veterans Day"  # a non-null value on a missing row is still a problem
    assert check_claim_numbers(c, {"Q9": hol}, version=7).problems


def test_failing_holiday_claim_still_gets_a_repair_round():
    from conftest import FakeLLM
    from whybot.agents import Analyst
    bad = {"cid": "c1", "hypothesis_id": "H1", "direction": "supports", "statement": "2024-11-28 had 99.9 mm of rain",
           "cited_values": [{"qid": "Q3", "column": "prcp_mm", "value": 99.9, "row_hint": "trip_date=2024-11-28"}]}
    ok = {"cid": "c2", "hypothesis_id": "H2", "direction": "supports", "statement": "2024-11-28 had 21.8 mm of rain",
          "cited_values": [{"qid": "Q3", "column": "prcp_mm", "value": 21.8, "row_hint": "trip_date=2024-11-28"}]}
    plan = "H0 [metric] drop\nH1 [holiday] Thanksgiving\nH2 [weather] rain"
    cs, meta = Analyst(FakeLLM({"analyst": [{"claims": [bad, ok]}, {"claims": [ok]}]})).draft("why?", plan, [Q2, Q3, ROW], "")
    assert meta.get("attempt") == 2 and [c.cid for c in cs.claims] == ["c2"]


def test_baseline_keeps_only_days_before_the_target():
    from whybot.agents import prior_baseline_only
    from whybot.schemas import Plan
    p = Plan.model_construct(target_window="2024-11-28", baseline_window="2024-11-07,2024-11-14,2024-11-21,2024-12-05,2024-12-12")
    prior_baseline_only(p)
    assert p.baseline_window == "2024-11-07,2024-11-14,2024-11-21"
    p = Plan.model_construct(target_window="2024-11-28", baseline_window="2024-11-21,2024-12-05,2024-12-12")
    prior_baseline_only(p)  # fewer than two days before: left alone
    assert p.baseline_window == "2024-11-21,2024-12-05,2024-12-12"
    p = Plan.model_construct(target_window="2024-11-01..2024-11-30", baseline_window="2024-10-01..2024-10-31")
    prior_baseline_only(p)  # ranges are not day lists
    assert p.baseline_window == "2024-10-01..2024-10-31"
