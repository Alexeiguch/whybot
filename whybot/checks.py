"""Deterministic validation used by the Critic and Writer, and the branching policy. Pure functions (unit-tested)."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .evidence import Evidence
from .schemas import Claim, CriticReport, FinalAnswer


def _num(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip().replace(",", "").rstrip("%")
        try:
            return float(s)
        except ValueError:
            return None
    return None


def value_matches(claimed, actual, rel_tol: float = 0.01) -> bool:
    a, b = _num(claimed), _num(actual)
    if a is not None and b is not None:
        if a == b:
            return True
        # accept rounding to the precision the claim was written with
        s = str(claimed)
        decimals = len(s.split(".")[1]) if "." in s and _num(s) is not None else 0
        if round(b, decimals) == round(a, decimals):
            return True
        return abs(a - b) <= rel_tol * max(abs(a), abs(b), 1e-9)
    return str(claimed).strip().lower() == str(actual).strip().lower()[: len(str(claimed).strip())] and str(claimed).strip() != ""


@dataclass
class NumericCheck:
    cid: str
    ok: bool
    problems: list[str]


def check_claim_numbers(claim: Claim, evidence: dict[str, Evidence]) -> NumericCheck:
    """Every cited value must appear in the named column of the named, successful query result."""
    problems = []
    if not claim.cited_values:
        problems.append("claim cites no numbers")
    for cv in claim.cited_values:
        ev = evidence.get(cv.qid)
        if ev is None:
            problems.append(f"{cv.qid} does not exist")
            continue
        if ev.status != "ok":
            problems.append(f"{cv.qid} has no data ({ev.status})")
            continue
        cols = {c.lower(): i for i, c in enumerate(ev.columns)}
        idx = cols.get(cv.column.lower())
        if idx is None:
            problems.append(f"{cv.qid} has no column {cv.column!r} (has {', '.join(ev.columns)})")
            continue
        rows = ev.rows
        if cv.row_hint and "=" in cv.row_hint:
            k, _, v = cv.row_hint.partition("=")
            ki = cols.get(k.strip().lower())
            if ki is not None:
                sel = [r for r in rows if str(r[ki]).startswith(v.strip()) or value_matches(v.strip(), r[ki])]
                rows = sel or rows
        if not any(value_matches(cv.value, r[idx]) for r in rows):
            problems.append(f"{cv.qid}.{cv.column}={cv.value} not found in returned rows")
    return NumericCheck(claim.cid, not problems, problems)


@dataclass
class Decision:
    action: str          # write | retry | escalate
    reason: str
    supported_cids: list[str]


def decide_next(iteration: int, max_iterations: int, report: CriticReport, checks: dict[str, NumericCheck],
                evidence: dict[str, Evidence], out_of_time: bool = False) -> Decision:
    """Branching policy. Numeric checks override the LLM critic: a claim with a bad number is never 'supported'."""
    supported = [v.cid for v in report.verdicts
                 if v.verdict == "supported" and checks.get(v.cid, NumericCheck(v.cid, False, ["unchecked"])).ok]
    hyp_supported = {h.hypothesis_id for h in report.hypothesis_status if h.status == "supported" and h.hypothesis_id != "H0"}
    failed_queries = [e.qid for e in evidence.values() if e.status == "failed" and e.iteration == iteration]
    can_retry = iteration < max_iterations and not out_of_time
    explained = bool(supported) and bool(hyp_supported)

    if report.decision == "accept" and explained:
        return Decision("write", f"critic accepted; {len(supported)} supported claims", supported)
    if report.decision == "insufficient_evidence" and not report.follow_up_queries:
        return Decision("escalate", "critic judged the evidence insufficient and proposed no further tests", supported)
    if can_retry and (report.decision == "retry" or report.follow_up_queries or failed_queries or not explained):
        why = []
        if report.follow_up_queries:
            why.append(f"{len(report.follow_up_queries)} follow-up tests requested")
        if failed_queries:
            why.append(f"queries failed: {', '.join(failed_queries)}")
        if not explained:
            why.append("no hypothesis supported yet")
        return Decision("retry", "; ".join(why) or "critic requested retry", supported)
    if explained:
        return Decision("write", "iteration cap reached; writing with supported claims and caveats", supported)
    return Decision("escalate", "iteration cap or time budget reached without a supported explanation", supported)


def validate_final(ans: FinalAnswer, evidence: dict[str, Evidence], allowed_qids: set[str]) -> list[str]:
    """Writer contract: every claim has a citation whose qid, sql_hash and table match real, critic-approved evidence."""
    problems = []
    if not ans.claims:
        problems.append("no claims")
    for i, c in enumerate(ans.claims):
        if not c.citations:
            problems.append(f"claim {i + 1} has no citation")
        for cit in c.citations:
            ev = evidence.get(cit.qid)
            if not ev or ev.status != "ok":
                problems.append(f"claim {i + 1}: {cit.qid} is not a successful query")
                continue
            if cit.qid not in allowed_qids:
                problems.append(f"claim {i + 1}: {cit.qid} was not used by any critic-supported claim")
            if cit.sql_hash != ev.sql_hash:
                problems.append(f"claim {i + 1}: sql_hash for {cit.qid} must be {ev.sql_hash}")
            if cit.table.lower() not in [t.lower() for t in ev.tables]:
                problems.append(f"claim {i + 1}: table for {cit.qid} must be one of {ev.tables}")
    return problems


def drop_uncited(ans: FinalAnswer, evidence: dict[str, Evidence], allowed_qids: set[str]) -> FinalAnswer:
    """Last-resort repair: keep only valid citations, drop claims left with none."""
    keep = []
    for c in ans.claims:
        cits = [x for x in c.citations if x.qid in evidence and evidence[x.qid].status == "ok" and x.qid in allowed_qids
                and x.sql_hash == evidence[x.qid].sql_hash and x.table.lower() in [t.lower() for t in evidence[x.qid].tables]]
        if cits:
            keep.append(c.model_copy(update={"citations": cits}))
    return ans.model_copy(update={"claims": keep})


def tables_in_sql(sql: str, known: list[str]) -> list[str]:
    low = sql.lower()
    return [t for t in known if re.search(r"(?<![\w.])" + re.escape(t.lower()) + r"(?![\w])", low)]


def result_sanity(columns: list[str], rows: list[list]) -> list[str]:
    """Cheap smell tests on a query result, run before any agent reasons over it."""
    problems = []
    if not rows:
        problems.append("query returned 0 rows (check filters and date formats)")
        return problems
    if len(rows) >= 3 and len(columns) >= 2:
        keys = {str(r[0]) for r in rows}
        rest = {tuple(str(v) for v in r[1:]) for r in rows}
        if len(keys) == len(rows) and len(rest) == 1:
            problems.append("every row has identical values in all non-key columns although the keys differ: "
                            "likely a cross join or a missing join/group key; join explicitly on the date key")
    return problems
