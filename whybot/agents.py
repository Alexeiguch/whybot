"""The five agent roles. Each is a prompt + a validated JSON contract; the Orchestrator owns control flow."""
from __future__ import annotations

import json
import re

from .checks import (CHECKS_VERSION, check_calendar_sql, check_critic_verdicts, sql_parse_problems, tables_in_sql, units_text,
                     weekdays_text)
from .evidence import Evidence
from .llm import BaseLLM
from .schemas import ClaimSet, CriticReport, FinalAnswer, Plan, QuerySpec, SQLDraft

KINDS = ["metric", "weather", "holiday", "day_of_week", "fare_mix", "zone_mix", "data_quality", "other"]


def catalog_text(catalog: list[dict], notes: dict, tables: list[str] | None = None) -> str:
    out = []
    for t in catalog:
        if tables and t["table"].lower() not in [x.lower() for x in tables]:
            continue
        cols = ", ".join(f"{c[0]} {c[1]}" for c in t["columns"][:40])
        note = next((v for k, v in notes.items() if k.lower() == t["table"].lower()
                     or k.split(".")[-1].lower() == t["table"].split(".")[-1].lower()), "")
        out.append(f"- {t['table']}({cols})" + (f"\n  note: {note}" if note else ""))
    return "\n".join(out)


class Planner:
    SYSTEM = (
        "You are the Planner in WhyBot, a team of agents that explains why a business metric changed using SQL evidence. "
        "Turn the analyst's bounded question into testable, falsifiable hypotheses and a minimal SQL plan. "
        "Always include H0 (kind=metric) that quantifies the change: the target window vs a like-for-like baseline (same weekday where relevant). "
        "On the first iteration propose at least 4 competing hypotheses from: weather, holiday, day_of_week, fare_mix, zone_mix, data_quality. "
        "The data covers 2024-04-01..2025-02-28 only (weather to 2025-02-05): never plan a query outside that window, so there are no earlier years to compare. "
        "Each query purpose must be concrete (columns, grouping, filters, exact date windows) and use ONLY tables listed in the catalog. "
        "Keep queries few (one per hypothesis is ideal, max 7 total) and aggregated. Use memory lessons when relevant. "
        "On later iterations, only add queries that address the critic feedback; do not repeat queries that already succeeded."
    )

    # v7: the Planner (Ultra, ~25 tokens/s) spent 59 s writing 1,468 tokens; the plan's content needs far fewer
    SYSTEM_V7 = (" Be brief: one short sentence (at most 25 words) for each statement, support_if, refute_if and purpose, "
                 "and reply with compact JSON on one line (no indentation). Plan at most 4 queries: WhyBot adds the baseline "
                 "summary (target value, baseline average, % change) itself, and one daily query over the target and "
                 "baseline days that joins trips, weather and holidays can test several hypotheses at once.")

    def __init__(self, llm: BaseLLM, version: int = CHECKS_VERSION):
        self.llm, self.version = llm, version

    def plan(self, question: str, catalog: list[dict], notes: dict, memory: str, carryover: str,
             feedback: str, done_queries: list[str], iteration: int, required: list[str] | None = None) -> tuple[Plan, dict]:
        known = [t["table"].lower() for t in catalog]
        user = (f"QUESTION: {question}\nITERATION: {iteration}\n\nCATALOG:\n{catalog_text(catalog, notes)}\n\n"
                f"MEMORY (H-MEM recall of past investigations and lessons):\n{memory or '(none)'}\n\n"
                f"CARRYOVER (Markovian state from previous iterations):\n{carryover or '(first iteration)'}\n\n"
                f"CRITIC FEEDBACK:\n{feedback or '(none)'}\n\nQUERIES ALREADY RUN:\n" + ("\n".join(done_queries) or "(none)"))

        def validate(p: Plan) -> list[str]:
            errs = []
            ids = {h.id for h in p.hypotheses}
            if iteration == 1:
                if "H0" not in ids:
                    errs.append("H0 (kind=metric) is required")
                if len([h for h in p.hypotheses if h.id != "H0"]) < 3:
                    errs.append("propose at least 3 competing hypotheses besides H0")
            if not p.queries:
                errs.append("at least one query is required")
            if len(p.queries) > 8:
                errs.append("at most 8 queries")
            planned = {q.hypothesis_id for q in p.queries}
            for rid in required or []:
                if rid not in planned:
                    errs.append(f"the Critic requested a follow-up test for {rid}; add a query for it")
            for q in p.queries:
                if iteration == 1 and q.hypothesis_id not in ids:
                    errs.append(f"query references unknown hypothesis {q.hypothesis_id}")
                for t in q.tables:
                    if t.lower() not in known:
                        errs.append(f"table {t} is not in the catalog")
            return errs

        system = self.SYSTEM + (self.SYSTEM_V7 if self.version >= 7 else "")
        plan, meta = self.llm.complete_json("planner", system, user, Plan, validator=validate, max_tokens=3000)
        if self.version >= 7 and iteration == 1:
            add_baseline_summary_query(plan)
        return plan, meta


BASELINE_COLS = ("target_value", "baseline_avg", "pct_change_vs_baseline")


def add_baseline_summary_query(plan: Plan) -> None:
    """Run v8c: the Analyst listed the three baseline Thursdays but never declared their average or the % change, so
    the Writer had to drop -45.5% as unbacked. The change itself is the one number every answer needs, so WhyBot
    always plans a query whose result row IS the target value, the baseline average and the % change: those become
    cited cells, not arithmetic the Analyst may forget. General: any metric, target window and baseline days."""
    h0 = [q for q in plan.queries if q.hypothesis_id == "H0"]
    if any("baseline_avg" in q.purpose for q in h0):
        return
    tables = (h0[0].tables if h0 else plan.queries[0].tables if plan.queries else [])
    if not tables:
        return
    plan.queries.insert(len(h0), QuerySpec(hypothesis_id="H0", tables=list(tables), purpose=(
        f"Baseline summary for {plan.metric}: return ONE row with target_date (the target {plan.target_window}), "
        f"target_value (the {plan.metric} on the target), baseline_avg (the average daily {plan.metric} over the "
        f"comparable baseline days: {plan.baseline_window}; never include the target), baseline_days (how many baseline "
        f"days), and pct_change_vs_baseline = (target_value - baseline_avg) / baseline_avg * 100. Read the table once "
        f"with conditional aggregation (MAX(CASE WHEN <date> = <target> THEN ... END), AVG(CASE WHEN <date> IN "
        f"(<baseline days>) THEN ... END)) over WHERE <date> IN (target and baseline days), the target date as a literal (DATE '<target>' AS target_date); no GROUP BY, no join and no subquery.")))


class SQLAgent:
    SYSTEM = (
        "You are the SQL Executor agent in WhyBot. Write ONE read-only SELECT (or WITH) query for the stated purpose. "
        "Use only the given tables and columns, alias every computed column with a short snake_case name, aggregate, "
        "order the output, and keep it under 200 rows. Join only with explicit JOIN ... ON keys (never comma joins). "
        "For existence checks (e.g. is a date a holiday) return COUNT(*) or a LEFT JOIN so the result is never empty. "
        "Never use semicolons, DDL or DML."
    )

    def __init__(self, llm: BaseLLM, version: int = CHECKS_VERSION):
        self.llm, self.version = llm, version  # version: the checks version the run's prompts and checks follow

    def write(self, purpose: str, tables: list[str], catalog: list[dict], notes: dict, dialect: str,
              dialect_notes: str, memory: str, question: str, failed: list[dict] | None = None) -> tuple[SQLDraft, dict]:
        known = [t["table"] for t in catalog]
        user = (f"DIALECT: {dialect}\n{dialect_notes}\n\nTABLES:\n{catalog_text(catalog, notes, tables or None)}\n\n"
                f"INVESTIGATION QUESTION: {question}\nQUERY PURPOSE: {purpose}\n\nLESSONS FROM MEMORY:\n{memory or '(none)'}")
        if self.version >= 4:
            user += ("\n\nALLOWED TABLES (the evidence allowlist; write these fully-qualified names exactly, any other name is refused): "
                     + ", ".join(known) + "\nCALENDAR: TPCDS_DB.date_dim.d_dow is 0=Sunday, 1=Monday .. 6=Saturday (never 7).")
        if self.version >= 6:
            user += (" To filter a weekday use d_day_name (d_day_name = 'Thursday'), not a d_dow number: d_dow Thursday is 4 "
                     "and Friday is 5. Keep the target date itself in every comparison result (do not filter it out), and "
                     "return the date column so each row says which day it is.")
        if self.version >= 7:
            user += (" Read each whybot.* table at most once per query (no self-join, and no subquery or CTE over the same "
                     "table joined back to it: Zetaris refuses those). For a baseline, return one row per day for the target "
                     "and the baseline days together; the Analyst computes the average.")
        if failed:
            user += "\n\nPREVIOUS ATTEMPTS FAILED (repair the query):\n" + "\n".join(
                f"SQL: {f['sql']}\nERROR: {f['error'][:700]}" for f in failed[-2:])

        def validate(d: SQLDraft) -> list[str]:
            errs = []
            s = d.sql.strip().rstrip(";")
            if not re.match(r"(?is)^\s*(select|with)\b", s):
                errs.append("must start with SELECT or WITH")
            if ";" in s:
                errs.append("no semicolons")
            def _n(x):
                return re.sub(r"\s+limit\s+\d+\s*$", "", " ".join(x.lower().split()))
            for f in failed or []:
                if _n(f["sql"]) == _n(s):
                    errs.append("this SQL is identical to an attempt that already failed; change the approach")
                    break
            if not tables_in_sql(s, known):
                errs.append(f"query must reference the catalog tables by full name: {', '.join(tables or known[:3])}")
            if self.version >= 4:
                errs += check_calendar_sql(s, f"{question}\n{purpose}" if self.version >= 6 else "")
            if self.version >= 7 and not errs:
                # parse before running: an unclosed ROUND( used to fail in the guard, cost a schema re-describe and a whole
                # repair round (b95249 Q1, 3 attempts, ~40 s); now the model fixes it inside the same call
                errs += sql_parse_problems(s)
            if self.version >= 7 and "baseline_avg" in purpose and re.search(r"(?i)\bgroup\s+by\b", s):
                # run v11a: GROUP BY trip_date split the baseline summary into one row per day, so baseline_avg held each
                # day's own count and the % change was NULL; the summary is one aggregate row over all the days
                errs.append("the baseline summary must return ONE row: remove GROUP BY (aggregate over all target and "
                            "baseline days with MAX(CASE ...) and AVG(CASE ...)) and select the target date as a literal, "
                            "e.g. DATE '<target>' AS target_date")
            return errs

        d, meta = self.llm.complete_json("sql", self.SYSTEM, user, SQLDraft, validator=validate, max_tokens=1500)
        d.sql = d.sql.strip().rstrip(";")
        return d, meta


def baseline_summary_problems(cs: ClaimSet, evidence: list[Evidence]) -> list[str]:
    """The baseline-summary row must reach the answer as an H0 claim citing its average and its % change."""
    errs = []
    for e in evidence:
        cols = {c.lower() for c in e.columns}
        if e.status != "ok" or not e.rows or not {"baseline_avg", "pct_change_vs_baseline"} <= cols:
            continue
        cited = {cv.column.lower() for c in cs.claims for cv in c.cited_values if cv.qid == e.qid}
        if not {"baseline_avg", "pct_change_vs_baseline"} <= cited:
            errs.append(f"{e.qid} gives the baseline average and the % change: draft the H0 claim from it, citing "
                        f"{e.qid}.target_value, {e.qid}.baseline_avg and {e.qid}.pct_change_vs_baseline as cells (no derived), "
                        f"e.g. 'Trips on <target date> were <target_value>, <pct_change_vs_baseline>% vs an average of "
                        f"<baseline_avg> on the other <weekday>s'")
    return errs


def baseline_summary_claim(evidence: list[Evidence], metric: str, baseline_window: str):
    """Runs v10a-v12a: the Analyst kept wrapping the summary row's cells in a formula over a v4 it never cited, so the
    change claim failed its number check and the run looped. The row already holds the three numbers, so WhyBot
    writes the H0 change claim from it directly: no model arithmetic, every number a cited cell."""
    from .schemas import Claim
    for e in evidence:
        cols = {c.lower(): i for i, c in enumerate(e.columns)}
        if e.status != "ok" or len(e.rows) != 1 or not {"target_value", "baseline_avg", "pct_change_vs_baseline"} <= set(cols):
            continue
        row = e.rows[0]
        vals = {k: row[cols[k]] for k in ("target_value", "baseline_avg", "pct_change_vs_baseline")}
        try:
            t, b, p = (float(vals[k]) for k in ("target_value", "baseline_avg", "pct_change_vs_baseline"))
        except (TypeError, ValueError):
            continue
        day = str(row[cols["target_date"]])[:10] if "target_date" in cols else "the target day"
        # the baseline dates stay out of the sentence: they name rows this one-row result does not have
        stmt = (f"{metric[:1].upper() + metric[1:]} on {day} were {t:,.0f}, {p:+.1f}% vs an average of {b:,.0f} over the "
                f"comparable baseline days")
        return Claim.model_validate({"cid": "C0", "hypothesis_id": "H0", "statement": stmt,
                                     "direction": "supports",
                                     "cited_values": [{"qid": e.qid, "column": e.columns[cols[k]], "value": vals[k]}
                                                      for k in ("target_value", "pct_change_vs_baseline", "baseline_avg")]})
    return None


class Analyst:
    SYSTEM = (
        "You are the Analyst in WhyBot. From the query results only, draft precise claims for or against each hypothesis. "
        "Every claim must cite the exact numbers it relies on as cited_values {qid, column, value, row_hint} copied verbatim "
        "from the result rows (row_hint = key column=value identifying the row). Compare target vs baseline with effect sizes "
        "(differences or % changes you can show from cited values). Do not speculate beyond the numbers; draft refuting claims too. "
        "At most 8 short claims. Treat suspicious results (identical values on every row) as unreliable. "
        "Every percentage, ratio or average written in a statement must be declared in derived[] with a formula over the claim's "
        "cited values v1..vn (1-based, in the order of cited_values; cite EVERY operand: the target cell FIRST, then each baseline cell), e.g. "
        "{name: pct_change_vs_baseline, formula: (v1 - avg(v2,v3,v4)) / avg(v2,v3,v4) * 100}. Write the number in the statement as a "
        "placeholder {pct_change_vs_baseline}: WhyBot evaluates the formula and fills it in, so never do the arithmetic yourself. "
        "Rules: cited_values are raw cells copied from a returned row (never an average or total you computed: put those in derived); "
        "one metric per claim (a second metric needs its own claim and its own v-numbering); to claim something is ABSENT cite a "
        "COUNT cell equal to 0 or say it in words without a row_hint, never a row_hint for a row that is not in the result; "
        "if a query returned no rows or NULLs, do not draft a claim from it. The data only covers 2024-04-01..2025-02-28. "
        "The question's premise may be wrong: ALWAYS draft one H0 claim that states the measured change neutrally with its sign, "
        "e.g. 'Trips on 2024-12-21 were {pct_change_vs_baseline}% vs the mean of the other Saturdays' (WhyBot fills in a signed value), "
        "and if the change is small (under about 5%) or opposite to the question, add a claim saying the premise is not supported by the data."
    )

    SYSTEM_V5 = (" Write each value in its column's unit (see UNITS; trip distances are miles). Call a number an average, mean "
                 "or baseline only when it is one (declare it in derived), never a single day's value. 'Higher/lower than the other "
                 "days' must hold for every compared day; otherwise say 'on average' or name the days. A threshold from the "
                 "hypotheses (e.g. 5%) is not a measurement: write it only beside the measured change it is compared with.")

    SYSTEM_V6 = (" A claim about a date must cite values from the row for that date (row_hint = that date); a value from "
                 "another day's row never describes it, and if no returned row has the date, say the data has no row for it. "
                 "Name a weekday only as the weekday of the rows you cite (see WEEKDAYS): a baseline of Thursdays is 'the other "
                 "Thursdays'.")

    def __init__(self, llm: BaseLLM, version: int = CHECKS_VERSION):
        self.llm, self.version = llm, version

    SYSTEM_V7 = (" Amounts of money (avg_total_amount, fares) are higher or lower, never longer or shorter. Never write 'no "
                 "data-quality flags' (or issues) when DATA-QUALITY NOTES are listed; leave data quality to the caveats. "
                 "State a baseline as ONE average that WhyBot computes: declare derived baseline_avg = avg(v2, v3, v4) over the "
                 "cited baseline cells and write '{baseline_avg}' in the statement, e.g. 'Trips on <target date> were "
                 "{pct_change_vs_baseline}% vs an average of {baseline_avg} on the other <weekday>s'. Never call a list of "
                 "separate values an average. Derived values are arithmetic over numeric cells only: a holiday name is cited "
                 "and stated in words ('2024-11-28 was Thanksgiving Day; the other Thursdays were not holidays'), with no "
                 "derived value. When a result has the columns target_value, baseline_avg and pct_change_vs_baseline, "
                 "WhyBot writes the H0 change claim from that row itself: do not draft a claim from it.")

    def draft(self, question: str, plan_text: str, evidence: list[Evidence], feedback: str, dq_notes=()) -> tuple[ClaimSet, dict]:
        ev_ids = {e.qid for e in evidence}
        user = (f"QUESTION: {question}\n\nHYPOTHESES:\n{plan_text}\n\nEVIDENCE:\n" +
                "\n\n".join(e.for_prompt() for e in evidence) + f"\n\nCRITIC FEEDBACK ON EARLIER DRAFT:\n{feedback or '(none)'}")
        if self.version >= 5:
            user += "\n\n" + units_text(evidence)
        if self.version >= 6:
            user += "\n\n" + weekdays_text(evidence)
        if self.version >= 7 and dq_notes:
            user += "\n\nDATA-QUALITY NOTES (from the Critic's check; caveats, not evidence):\n" + "\n".join(f"- {n}" for n in dq_notes)

        ev_map = {e.qid: e for e in evidence}
        rounds = {"n": 0, "prev": None}

        def validate(cs: ClaimSet) -> list[str]:
            from .checks import check_claim_numbers
            rounds["n"] += 1
            errs = []
            if not cs.claims:
                errs.append("draft at least one claim")
            by_hash = {e.sql_hash.lower(): e.qid for e in evidence}
            for c in cs.claims:
                for cv in c.cited_values:
                    if cv.qid not in ev_ids and self.version >= 7:
                        # run v7e: the Analyst wrote Q3's sql_hash where its qid belongs, three rounds running (54 s). The
                        # hash names exactly one query, so it is the same citation; the value is still checked against it.
                        cv.qid = by_hash.get(cv.qid.strip().lower(), cv.qid)
                    if cv.qid not in ev_ids:
                        errs.append(f"{c.cid} cites unknown {cv.qid}")
            if errs or rounds["n"] >= 3:
                return errs  # two self-repair rounds; after that the Critic's own number check decides
            for c in cs.claims:
                ch = check_claim_numbers(c.model_copy(deep=True), ev_map, version=self.version, plan_text=plan_text, dq_notes=dq_notes)
                errs += [f"{c.cid}: {p}" for p in ch.problems]
            if self.version >= 7:
                # a retry that comes back with nothing but the problems it was already told about will not fix them on a
                # third try either (run v7b: c2 rejected identically twice, 33 s); stop here and let the Critic's own
                # number check reject those claims. Nothing is waived: the same checks run again there.
                prev, rounds["prev"] = rounds["prev"], set(errs)
                if errs and prev is not None and set(errs) <= prev:
                    return []
            return errs

        system = self.SYSTEM
        if self.version >= 6:  # the example's 'other Saturdays' was copied verbatim onto a Thursday baseline (v5 runs)
            system = system.replace("'Trips on 2024-12-21 were {pct_change_vs_baseline}% vs the mean of the other Saturdays'",
                                    "'Trips on <target date> were {pct_change_vs_baseline}% vs the mean of the other <weekday of the "
                                    "baseline rows>s'")
        system += (self.SYSTEM_V5 if self.version >= 5 else "") + (self.SYSTEM_V6 if self.version >= 6 else "") + \
            (self.SYSTEM_V7 if self.version >= 7 else "")
        return self.llm.complete_json("analyst", system, user, ClaimSet, validator=validate, max_tokens=5000)


class Critic:
    SYSTEM = (
        "You are the Critic/Evaluator in WhyBot. Be skeptical. For each claim decide supported / weak / unsupported against "
        "the returned numbers: is the effect large relative to the baseline, is the baseline like-for-like (same weekday), "
        "are confounders (holiday + weather on the same day, data-quality artefacts) addressed, does the cited number really "
        "say what the claim says? AUTOMATED CHECK failures mean the number is not in the data: mark those unsupported. "
        "Set each hypothesis to supported/refuted/inconclusive/untested. decision=accept only if the change is quantified (H0) "
        "and at least one cause is supported with adequate strength; decision=retry with concrete follow_up_queries (tables from the "
        "catalog) when a specific extra test would settle it; decision=insufficient_evidence if more SQL cannot help. "
        "A fare or zone mix shift describes HOW a change happened, not WHY: never accept it as the root cause while an external "
        "driver (weather, holiday) is untested or its query failed; request that test instead. When two drivers coincide "
        "(e.g. holiday and rain on the same day), say so and weigh them against baselines that isolate each one. "
        "If H0 shows no material change (under about 5%) or a change opposite to the question, the correct answer is that the "
        "premise is false: mark the neutral H0 claim supported, mark causal hypotheses refuted or inconclusive, and decide accept. "
        "Derived values named *pct* are already percentages: 0.15 means +0.15 percent (no change), not 15%. Read the sign and "
        "size of every derived value and check the statement's wording matches it; a change under about 5% is not a drop."
    )

    SYSTEM_V4 = (" Judge each claim against the returned rows, not against its hypothesis: a claim whose numbers match the rows "
                 "is evidence even when it refutes the hypothesis (e.g. the hypothesis expected a drop and the rows show none). Mark "
                 "such a claim supported and set the hypothesis refuted; never mark a claim unsupported because of the direction its "
                 "hypothesis expected.")

    SYSTEM_V7 = (" Be brief: each verdict reason is one sentence of at most 25 words, and reply with compact JSON on one line "
                 "(no indentation). Never compute new statistics in a reason (no CVs or percentages that no claim states).")

    def __init__(self, llm: BaseLLM, version: int = CHECKS_VERSION):
        self.llm, self.version = llm, version

    def review(self, question: str, plan_text: str, claims: ClaimSet, evidence: list[Evidence], auto_checks: dict,
               catalog_tables: list[str], iteration: int, max_iterations: int) -> tuple[CriticReport, dict]:
        cids = {c.cid for c in claims.claims}
        checks_txt = "\n".join(f"{cid}: {'PASS' if ch.ok else 'FAIL - ' + '; '.join(ch.problems)}" for cid, ch in auto_checks.items())
        user = (f"QUESTION: {question}\nITERATION {iteration} of {max_iterations}\n\nHYPOTHESES:\n{plan_text}\n\n"
                f"CLAIMS:\n{json.dumps([c.model_dump() for c in claims.claims], indent=None if self.version >= 7 else 1)}\n\nAUTOMATED NUMBER CHECKS:\n{checks_txt}\n\n"
                f"EVIDENCE:\n" + "\n\n".join(e.for_prompt() for e in evidence) + f"\n\nCATALOG TABLES: {', '.join(catalog_tables)}")

        def validate(r: CriticReport) -> list[str]:
            errs = [f"verdict for unknown claim {v.cid}" for v in r.verdicts if v.cid not in cids]
            missing = cids - {v.cid for v in r.verdicts}
            if missing:
                errs.append(f"missing verdicts for {sorted(missing)}")
            low = [t.lower() for t in catalog_tables]
            for f in r.follow_up_queries:
                errs += [f"follow-up table {t} not in catalog" for t in f.tables if t.lower() not in low]
            if self.version >= 4:
                errs += check_critic_verdicts(r, auto_checks)
            return errs

        system = self.SYSTEM + (self.SYSTEM_V4 if self.version >= 4 else "") + (self.SYSTEM_V7 if self.version >= 7 else "")
        return self.llm.complete_json("critic", system, user, CriticReport, validator=validate, max_tokens=3000)


class Writer:
    SYSTEM = (
        "You are the Writer in WhyBot. Produce the final answer for a research analyst using ONLY critic-supported claims. "
        "Every claim must carry citations {qid, sql_hash, table} copied exactly from the evidence list. "
        "Give a one-line headline, a short answer, an actionable recommendation, honest caveats (rejected or untested hypotheses, "
        "data limitations) and a calibrated confidence. Add 0-3 reusable lessons for future investigations (schema quirks, filters, joins). "
        "Cite ONLY the queries in the CITABLE EVIDENCE list. If the supported claims show the question's premise is false (no "
        "material change, or the opposite change), say so plainly in the headline. Every number you write must appear in a "
        "supported claim; do not compute new percentages."
    )

    SYSTEM_V5 = (" Write each value in its column's unit (see UNITS; trip distances are miles, never km). Call a number an "
                 "average, mean or baseline only when a supported claim gives it as one, never a single day's value. 'Higher/lower "
                 "than the other (comparison, baseline) days' must hold for every compared day; otherwise write 'on average' or "
                 "name the days it holds for. The hypotheses' thresholds (e.g. 'within 5%') are plan parameters, not "
                 "measurements: quote one only in the same sentence as the measured change it is compared with "
                 "('+0.15%, within the 5% threshold'), or leave it out.")

    SYSTEM_V7 = (" Write in plain words for a reader, not SQL: say 'average trip distance', not avg_trip_distance, and full "
                 "sentences, not col=value lists. To compare the target day with the baseline days, either quote the baseline "
                 "average exactly as a supported claim states it, or list the per-day values ('the other Thursdays: 3.25, 5.09 "
                 "and 4.55 mi'); never call a single day's value, or a number you worked out, the baseline or average. Money "
                 "amounts are higher or lower, never longer or shorter. Never write 'no data-quality flags/issues' when "
                 "DATA-QUALITY NOTES are listed: mention them in the caveats instead. The headline states the finding in one "
                 "sentence; the answer adds the explanation without repeating the headline word for word. Write about the data, "
                 "never about WhyBot's process: do not mention the critic, checks, claims, hypotheses (H1) or what is or is "
                 "not quoted. When a supported claim of kind holiday says the holiday caused the change, state it as the "
                 "cause ('driven by Thanksgiving Day'). Quote a baseline as one average when a supported claim gives it ('an "
                 "average of 141,603'), never as a list of values called an average.")

    def __init__(self, llm: BaseLLM, version: int = CHECKS_VERSION):
        self.llm, self.version = llm, version

    def write(self, question: str, status: str, supported_claims: list[dict], rejected: list[dict], evidence: list[Evidence],
              hypothesis_status: list[dict], validator, dq_notes=()) -> tuple[FinalAnswer, dict]:
        citable = {cv["qid"] for c in supported_claims for cv in c.get("cited_values", [])}
        ev_list = "\n".join(f"{e.qid}: sql_hash={e.sql_hash} tables={e.tables} purpose={e.purpose[:120]}"
                            for e in evidence if e.status == "ok" and e.qid in citable) or "(none)"
        user = (f"QUESTION: {question}\nRUN STATUS: {status}\n\nSUPPORTED CLAIMS (critic-approved, numbers verified):\n"
                f"{json.dumps(supported_claims, indent=1)}\n\nREJECTED OR WEAK CLAIMS:\n{json.dumps(rejected, indent=1)}\n\n"
                f"HYPOTHESIS STATUS:\n{json.dumps(hypothesis_status, indent=1)}\n\nCITABLE EVIDENCE (cite only from here):\n{ev_list}")
        if status == "insufficient_evidence":
            user += "\n\nThe investigation was escalated: say clearly that the evidence is insufficient, what was ruled out, and what data would settle it."
        if self.version >= 5:
            user += "\n\n" + units_text([e for e in evidence if e.qid in citable])
        if self.version >= 6:
            user += "\n\n" + weekdays_text([e for e in evidence if e.qid in citable])
        if self.version >= 7 and dq_notes:
            user += "\n\nDATA-QUALITY NOTES (caveats, not evidence):\n" + "\n".join(f"- {n}" for n in dq_notes)
        system = self.SYSTEM + (self.SYSTEM_V5 if self.version >= 5 else "") + (self.SYSTEM_V7 if self.version >= 7 else "")
        return self.llm.complete_json("writer", system, user, FinalAnswer, validator=validator, max_tokens=2500)
