"""The five agent roles. Each is a prompt + a validated JSON contract; the Orchestrator owns control flow."""
from __future__ import annotations

import json
import re

from .checks import tables_in_sql
from .evidence import Evidence
from .llm import BaseLLM
from .schemas import ClaimSet, CriticReport, FinalAnswer, Plan, SQLDraft

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

    def __init__(self, llm: BaseLLM):
        self.llm = llm

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

        return self.llm.complete_json("planner", self.SYSTEM, user, Plan, validator=validate, max_tokens=3000)


class SQLAgent:
    SYSTEM = (
        "You are the SQL Executor agent in WhyBot. Write ONE read-only SELECT (or WITH) query for the stated purpose. "
        "Use only the given tables and columns, alias every computed column with a short snake_case name, aggregate, "
        "order the output, and keep it under 200 rows. Join only with explicit JOIN ... ON keys (never comma joins). "
        "For existence checks (e.g. is a date a holiday) return COUNT(*) or a LEFT JOIN so the result is never empty. "
        "Never use semicolons, DDL or DML."
    )

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def write(self, purpose: str, tables: list[str], catalog: list[dict], notes: dict, dialect: str,
              dialect_notes: str, memory: str, question: str, failed: list[dict] | None = None) -> tuple[SQLDraft, dict]:
        known = [t["table"] for t in catalog]
        user = (f"DIALECT: {dialect}\n{dialect_notes}\n\nTABLES:\n{catalog_text(catalog, notes, tables or None)}\n\n"
                f"INVESTIGATION QUESTION: {question}\nQUERY PURPOSE: {purpose}\n\nLESSONS FROM MEMORY:\n{memory or '(none)'}")
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
            return errs

        d, meta = self.llm.complete_json("sql", self.SYSTEM, user, SQLDraft, validator=validate, max_tokens=1500)
        d.sql = d.sql.strip().rstrip(";")
        return d, meta


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
        "if a query returned no rows or NULLs, do not draft a claim from it. The data only covers 2024-04-01..2025-02-28."
    )

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def draft(self, question: str, plan_text: str, evidence: list[Evidence], feedback: str) -> tuple[ClaimSet, dict]:
        ev_ids = {e.qid for e in evidence}
        user = (f"QUESTION: {question}\n\nHYPOTHESES:\n{plan_text}\n\nEVIDENCE:\n" +
                "\n\n".join(e.for_prompt() for e in evidence) + f"\n\nCRITIC FEEDBACK ON EARLIER DRAFT:\n{feedback or '(none)'}")

        ev_map = {e.qid: e for e in evidence}
        rounds = {"n": 0}

        def validate(cs: ClaimSet) -> list[str]:
            from .checks import check_claim_numbers
            rounds["n"] += 1
            errs = []
            if not cs.claims:
                errs.append("draft at least one claim")
            for c in cs.claims:
                for cv in c.cited_values:
                    if cv.qid not in ev_ids:
                        errs.append(f"{c.cid} cites unknown {cv.qid}")
            if errs or rounds["n"] >= 3:
                return errs  # two self-repair rounds; after that the Critic's own number check decides
            for c in cs.claims:
                ch = check_claim_numbers(c.model_copy(deep=True), ev_map)
                errs += [f"{c.cid}: {p}" for p in ch.problems]
            return errs

        return self.llm.complete_json("analyst", self.SYSTEM, user, ClaimSet, validator=validate, max_tokens=5000)


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
        "Derived values named *pct* are already percentages: 0.15 means +0.15 percent (no change), not 15%. Read the sign and "
        "size of every derived value and check the statement's wording matches it; a change under about 5% is not a drop."
    )

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def review(self, question: str, plan_text: str, claims: ClaimSet, evidence: list[Evidence], auto_checks: dict,
               catalog_tables: list[str], iteration: int, max_iterations: int) -> tuple[CriticReport, dict]:
        cids = {c.cid for c in claims.claims}
        checks_txt = "\n".join(f"{cid}: {'PASS' if ch.ok else 'FAIL - ' + '; '.join(ch.problems)}" for cid, ch in auto_checks.items())
        user = (f"QUESTION: {question}\nITERATION {iteration} of {max_iterations}\n\nHYPOTHESES:\n{plan_text}\n\n"
                f"CLAIMS:\n{json.dumps([c.model_dump() for c in claims.claims], indent=1)}\n\nAUTOMATED NUMBER CHECKS:\n{checks_txt}\n\n"
                f"EVIDENCE:\n" + "\n\n".join(e.for_prompt() for e in evidence) + f"\n\nCATALOG TABLES: {', '.join(catalog_tables)}")

        def validate(r: CriticReport) -> list[str]:
            errs = [f"verdict for unknown claim {v.cid}" for v in r.verdicts if v.cid not in cids]
            missing = cids - {v.cid for v in r.verdicts}
            if missing:
                errs.append(f"missing verdicts for {sorted(missing)}")
            low = [t.lower() for t in catalog_tables]
            for f in r.follow_up_queries:
                errs += [f"follow-up table {t} not in catalog" for t in f.tables if t.lower() not in low]
            return errs

        return self.llm.complete_json("critic", self.SYSTEM, user, CriticReport, validator=validate, max_tokens=3000)


class Writer:
    SYSTEM = (
        "You are the Writer in WhyBot. Produce the final answer for a research analyst using ONLY critic-supported claims. "
        "Every claim must carry citations {qid, sql_hash, table} copied exactly from the evidence list. "
        "Give a one-line headline, a short answer, an actionable recommendation, honest caveats (rejected or untested hypotheses, "
        "data limitations) and a calibrated confidence. Add 0-3 reusable lessons for future investigations (schema quirks, filters, joins)."
    )

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def write(self, question: str, status: str, supported_claims: list[dict], rejected: list[dict], evidence: list[Evidence],
              hypothesis_status: list[dict], validator) -> tuple[FinalAnswer, dict]:
        ev_list = "\n".join(f"{e.qid}: sql_hash={e.sql_hash} tables={e.tables} purpose={e.purpose[:120]}" for e in evidence if e.status == "ok")
        user = (f"QUESTION: {question}\nRUN STATUS: {status}\n\nSUPPORTED CLAIMS (critic-approved, numbers verified):\n"
                f"{json.dumps(supported_claims, indent=1)}\n\nREJECTED OR WEAK CLAIMS:\n{json.dumps(rejected, indent=1)}\n\n"
                f"HYPOTHESIS STATUS:\n{json.dumps(hypothesis_status, indent=1)}\n\nEVIDENCE (cite from here):\n{ev_list}")
        if status == "insufficient_evidence":
            user += "\n\nThe investigation was escalated: say clearly that the evidence is insufficient, what was ruled out, and what data would settle it."
        return self.llm.complete_json("writer", self.SYSTEM, user, FinalAnswer, validator=validator, max_tokens=2500)
