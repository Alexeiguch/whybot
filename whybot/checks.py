"""Deterministic validation used by the Critic and Writer, and the branching policy. Pure functions (unit-tested)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .evidence import Evidence
from .schemas import Claim, CriticReport, FinalAnswer


# Version of the deterministic checks. A SAMPLE_MODE replay runs the checks at the version its cassette was recorded
# with (cassettes without a version are 1), so the replayed prompts stay identical, and reports what the current
# checks would add as a shadow_checks event. Live runs always use CHECKS_VERSION.
#   1: checks as of the recorded samples
#   2: a % or Nx must match a derived value or a rate column of a citable query; comparisons such as "higher than the
#      other three Mondays" are checked against the cited rows
#   3: every cause the headline or answer names (holiday, weather, trip mix, ...) needs a critic-supported claim
#      about that cause, or the wording must hedge it
#   4: bare day-of-month numbers that continue a date list ("2024-11-07, 14, 21", "7, 14 and 21 Dec") are dates when each
#      such date appears in the evidence or the question; the SQL agent sees the exact allowlisted table names, a close
#      miss (whybot.date_dim) is repaired to the one allowlisted table it can only mean, and d_dow = 7 is refused
#      (TPCDS d_dow is 0=Sunday..6=Saturday); the Critic may not mark a claim that matches its rows unsupported only
#      because it refutes its hypothesis
#   5: units: a number or unit word must match the unit of its column (TLC distances are miles, never km); the Analyst
#      and Writer prompts list the column units; a value labelled average/mean/baseline must equal an aggregate of the
#      cited rows (to its stated precision), not just appear somewhere in them; 'higher than the comparison days' must
#      hold for every compared day (or be worded 'on average' and hold against their mean); a threshold % from the plan's
#      hypotheses ('within 5%') may be quoted only beside the backed, measured change it is compared with
CHECKS_VERSION = 5


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


_BOOL = {"true": 1.0, "false": 0.0, "yes": 1.0, "no": 0.0, "y": 1.0, "n": 0.0}


def _boolish(v):
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, str) and v.strip().lower() in _BOOL:
        return _BOOL[v.strip().lower()]
    if isinstance(v, (int, float)) and v in (0, 1):
        return float(v)
    return None


def value_matches(claimed, actual, rel_tol: float = 0.005) -> bool:
    ba, bb = _boolish(claimed), _boolish(actual)
    if ba is not None and bb is not None and (isinstance(claimed, (bool, str)) or isinstance(actual, (bool, str))):
        return ba == bb  # True / 'true' / 'Y' / 1 are the same flag
    """Numbers match within rounding of the claimed precision or 0.5%; strings match exactly (case-insensitive).
    A number never matches a string (no prefix tricks such as '2' vs '2024-02-13')."""
    a, b = _num(claimed), _num(actual)
    if a is not None and b is not None:
        if a == b:
            return True
        s_ = str(claimed).replace(",", "").rstrip("%")
        decimals = len(s_.split(".")[1]) if "." in s_ else None
        if decimals is not None and round(b, decimals) == round(a, decimals):
            return True
        return abs(a - b) <= rel_tol * max(abs(a), abs(b), 1e-9)
    if a is not None or b is not None:
        return False
    return str(claimed).strip().lower() == str(actual).strip().lower()


@dataclass
class NumericCheck:
    cid: str
    ok: bool
    problems: list[str]
    corrections: list[str] = field(default_factory=list)


def check_claim_numbers(claim: Claim, evidence: dict[str, Evidence], context: str = "", version: int = CHECKS_VERSION,
                        plan_text: str = "") -> NumericCheck:
    """Every cited value must appear in the named column of the named, successful query result, and benign weather
    wording must not describe a day the weather evidence shows as wet or snowy. ('No data-quality flags' is checked on
    the final answer, where the run's data-quality caveat is rendered.)"""
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
        hint = (cv.row_hint or "").strip()
        if not hint and len(ev.rows) > 1 and len({str(r[idx]) for r in ev.rows}) > 1:
            problems.append(f"{cv.qid}: row_hint is required when the result has several rows")
            continue
        if hint:
            k, eq, v = hint.partition("=")
            ki = cols.get(k.strip().lower()) if eq else None
            if ki is None:  # never fall back to "any row": a bad hint must not widen the match
                problems.append(f"{cv.qid}: row_hint {hint!r} must be <column>=<value> using a returned column ({', '.join(ev.columns)})")
                continue
            v = v.strip().strip("'\"")
            rows = [r for r in rows if value_matches(v, r[ki]) or str(r[ki])[:10] == v[:10] and len(v) >= 8]
            if not rows:
                problems.append(f"{cv.qid}: no row matches {cv.row_hint}")
                continue
        if not any(value_matches(cv.value, r[idx]) for r in rows):
            problems.append(f"{cv.qid}.{cv.column}={cv.value} not found in returned rows")
    corrections: list[str] = []
    problems += check_derived_values(claim, corrections, plan_thresholds(plan_text) if version >= 5 else None)
    problems += check_weather_wording(claim.statement, evidence, context)
    if version >= 5:
        problems += check_comparisons_v5(claim.statement, evidence, {cv.qid for cv in claim.cited_values})
    elif version >= 2:
        problems += check_comparisons(claim.statement, evidence, {cv.qid for cv in claim.cited_values})
    if version >= 5:
        qids = {cv.qid for cv in claim.cited_values}
        problems += check_units(claim.statement, evidence, qids)
        problems += check_aggregate_labels(claim.statement, evidence, qids, [d.value for d in claim.derived])
    return NumericCheck(claim.cid, not problems, problems, corrections)


_PCT = re.compile(r"(?<![\w.])[-+]?(\d+(?:\.\d+)?)\s?%")
_RATIO = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s?(?:x|×)(?![\w])", re.I)


def eval_formula(formula: str, values: list[float]) -> float:
    """Safely evaluate arithmetic over v1..vn with avg/sum/min/max/abs. Raises ValueError on anything else."""
    import ast
    import operator as op
    env = {f"v{i + 1}": v for i, v in enumerate(values)}
    funcs = {"avg": lambda *a: sum(a) / len(a), "sum": lambda *a: sum(a), "min": min, "max": max, "abs": abs}
    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.USub: op.neg, ast.UAdd: op.pos}

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.Name) and n.id in env:
            if env[n.id] is None:
                raise ValueError(f"{n.id} is not numeric")
            return env[n.id]
        if isinstance(n, ast.BinOp) and type(n.op) in ops:
            return ops[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in ops:
            return ops[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in funcs and not n.keywords:
            return float(funcs[n.func.id](*[ev(a) for a in n.args]))
        raise ValueError(f"unsupported expression: {ast.dump(n)[:80]}")
    try:
        return ev(ast.parse(formula.replace("×", "*"), mode="eval"))
    except ZeroDivisionError as e:
        raise ValueError("division by zero") from e
    except SyntaxError as e:
        raise ValueError(f"bad formula syntax: {formula}") from e


def _close(stated: float, actual: float) -> bool:
    return abs(abs(stated) - abs(actual)) <= 0.051 + 0.002 * abs(stated)


def _within_rounding_band(stated: float, actual: float) -> bool:
    """LLM arithmetic slips (e.g. a mean of 144,878 for 145,713, or -46.7% for -47.0%) are corrected, not trusted;
    anything further off (e.g. 49.5% for 45.5%) is a wrong claim and is rejected."""
    return abs(abs(stated) - abs(actual)) <= max(0.6, 0.01 * abs(actual))


def _fmt_variants(v: float) -> list[str]:
    a = abs(v)
    return list(dict.fromkeys([f"{a:,.2f}", f"{a:.2f}", f"{a:,.1f}", f"{a:.1f}", f"{a:,.0f}", f"{a:.0f}"]))


def _rewrite_number(text: str, old: float, new: float) -> str:
    for o in _fmt_variants(old):
        decimals = len(o.split(".")[1]) if "." in o else 0
        n = f"{abs(new):,.{decimals}f}" if "," in o else f"{abs(new):.{decimals}f}"
        pat = re.compile(r"(?<![\d.,])" + re.escape(o) + r"(?![\d])")
        if pat.search(text):
            return pat.sub(n, text)
    return text


_DOWN = re.compile(r"\b(lower|fell|fall|falls|drop|dropped|drops|declin\w*|decreas\w*|below|fewer|less|cut|reduc\w*|plung\w*|down)\b", re.I)
_UP = re.compile(r"\b(higher|rose|rise|rises|increas\w*|above|more|jump\w*|surg\w*|up|grew|gain\w*)\b", re.I)


def check_direction(claim: Claim) -> list[str]:
    """A change claimed as a drop must have a negative change value, a rise a positive one."""
    changes = [d for d in claim.derived if d.value is not None and re.search(r"pct|change|diff|delta", d.name, re.I)]
    if len(changes) != 1:
        return []
    v = changes[0].value
    down, up = bool(_DOWN.search(claim.statement)), bool(_UP.search(claim.statement))
    if down and not up and v > 0:
        return [f"the statement describes a decrease but {changes[0].name} = {v:+g} (an increase)"]
    if up and not down and v < 0:
        return [f"the statement describes an increase but {changes[0].name} = {v:+g} (a decrease)"]
    return []


def check_derived_values(claim: Claim, corrections: list[str] | None = None, plan_pcts: set[float] | None = None) -> list[str]:
    """Every derived value is re-evaluated from the cited cells with its formula. Values within rounding of the
    recomputed number are replaced by it (and the statement rewritten), so the arithmetic is the evaluator's, not
    the model's. Values further off are rejected. Every % or Nx ratio in the statement must then be one of the
    recomputed derived values (or a cited cell). Catches 'fell 49.5%' when the cells give 45.5%."""
    nums = [_num(cv.value) for cv in claim.cited_values]
    problems, ok_values = [], [abs(n) for n in nums if n is not None]
    for d in claim.derived:
        try:
            actual = eval_formula(d.formula, nums)
        except ValueError as e:
            problems.append(f"derived {d.name}: {e}")
            continue
        ph = "{" + d.name + "}"
        if d.value is None or ph in claim.statement:
            # placeholder style: the evaluator, not the model, supplies the number
            fixed = round(actual, 2)
            txt = f"{abs(fixed):,.0f}" if abs(fixed) >= 1000 else (f"{abs(fixed):.1f}" if abs(fixed) >= 10 else f"{abs(fixed):.2f}")
            rest = claim.statement.replace(ph, "")
            if fixed != 0 and not (_DOWN.search(rest) or _UP.search(rest)):
                txt = ("+" if fixed > 0 else "-") + txt  # neutral wording keeps the sign: "were -45.5% vs ..."
            claim.statement = claim.statement.replace(ph, txt)
            d.value = fixed
            ok_values.append(abs(actual))
            continue
        if _close(d.value, actual):
            ok_values.append(abs(actual))
        elif _within_rounding_band(d.value, actual):
            fixed = round(actual, 2)
            claim.statement = _rewrite_number(claim.statement, d.value, fixed)
            if corrections is not None:
                corrections.append(f"derived {d.name}: {d.value:g} -> {fixed:g} (recomputed {d.formula})")
            d.value = fixed
            ok_values.append(abs(actual))
        else:
            problems.append(f"derived {d.name}={d.value:g} but {d.formula} = {actual:.2f} from the cited cells")
    problems += check_direction(claim)
    for left in re.findall(r"\{[A-Za-z_]\w*\}", claim.statement):
        problems.append(f"placeholder {left} has no derived value with that name")
    statement = claim.statement
    if plan_pcts:  # v5: a plan threshold next to the measured change it is compared with is not a measurement
        statement, thr = plan_threshold_mentions(statement, plan_pcts, ok_values)
        problems += thr
    for p in [float(m) for m in _PCT.findall(statement)]:
        if not any(_close(p, v) for v in ok_values):
            problems.append(f"{p:g}% in the statement is not recomputable from cited cells (declare it in derived with a formula)")
    for r in [float(m) for m in _RATIO.findall(statement)]:
        if not any(_close(r, v) for v in ok_values):
            problems.append(f"{r:g}x in the statement is not recomputable from cited cells (declare it in derived with a formula)")
    return problems


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


# Causes the final text may name. A cause is "named" when it appears in the causal part of a sentence ("driven by the
# holiday", "rain caused the drop"). Each needs a critic-supported claim that mentions the same cause.
CAUSES = {
    "holiday": r"holiday|thanksgiving|christmas|new year|mlk|martin luther king|memorial day|labou?r day|independence day|veterans day|columbus day|presidents'? day|juneteenth",
    "weather": r"weather|rain\w*|precipitation|snow\w*|storm\w*|wind\w*|cold|temperature\w*|heat|freez\w*|icy|ice",
    "trip mix": r"trip mix|fare mix|(?:trip )?distance|longer trips|shorter trips|airport|surcharge\w*|tolls?",
    "zone mix": r"zone\w*|borough\w*|location mix",
    "day of week": r"day[- ]of[- ]week|weekday effect|weekend effect",
    "data quality": r"data[- ]quality|missing data|data gap|incomplete data|outage",
}
_CAUSE_RE = {k: re.compile(rf"\b(?:{v})\b", re.I) for k, v in CAUSES.items()}
# trigger after which the cause is named ("driven by X"), and trigger before which it is named ("X drove ...")
_CAUSE_AFTER = re.compile(r"\b(?:driven by|due to|caused by|because(?: of)?|owing to|attribut(?:able|ed) to|explained by|"
                          r"(?:a |the )?result of|resulting from|thanks to|on account of|(?:the |a )?drivers? (?:was|were|is|are)|reflect(?:s|ing|ed)?)\b", re.I)
_CAUSE_BEFORE = re.compile(r"\b(?:drove|drives|caused|causes|led to|leads to|explains?|explained|accounts? for|accounted for|"
                           r"(?:is|was) (?:the |a )?(?:main |primary |key )?(?:driver|cause))\b", re.I)
_HEDGE = re.compile(r"\b(?:may|might|could|possibly|perhaps|potentially|unclear|uncertain|untested|unconfirmed|not (?:confirmed|tested|"
                    r"established|isolated|supported|proven)|cannot be (?:confirmed|isolated|tested|ruled out|assessed)|"
                    r"not|no|rather than|instead of|ruled out|suggest\w*|consistent with|possible|plausibl\w*)\b", re.I)
_SENT_SPLIT = re.compile(r"(?<=[.;!?])\s+")


def named_causes(text: str) -> list[tuple[str, str]]:
    """(cause, sentence) for every cause stated as a cause without a hedge. Clause after "driven by"/"due to"/...;
    clause before "drove"/"caused"/...; a hedge or negation anywhere in that clause, or just before the trigger, lifts it."""
    out = []
    for sent in _SENT_SPLIT.split(text or ""):
        spans = []
        for m in _CAUSE_AFTER.finditer(sent):
            end = _CLAUSE_BREAK.search(sent, m.end())
            spans.append((sent[max(0, m.start() - 40):m.start()], sent[m.end(): end.start() if end else len(sent)]))
        for m in _CAUSE_BEFORE.finditer(sent):
            starts = [b.end() for b in _CLAUSE_BREAK.finditer(sent, 0, m.start())]
            spans.append((sent[max(0, m.start() - 40):m.start()], sent[starts[-1] if starts else 0: m.start()]))
        for pre, span in spans:
            if _HEDGE.search(span) or _HEDGE.search(pre.split(",")[-1]):
                continue
            for k, rx in _CAUSE_RE.items():
                if rx.search(span) and (k, sent) not in out:
                    out.append((k, sent))
    return out


_KIND_CAUSE = {"holiday": "holiday", "weather": "weather", "fare_mix": "trip mix", "zone_mix": "zone mix",
               "day_of_week": "day of week", "data_quality": "data quality"}


def check_causes(text: str, supported_statements: list[str]) -> list[str]:
    """Every cause named in the headline or answer must be backed by a critic-supported claim that supports a
    hypothesis of that kind; otherwise the Writer must hedge it ("may have contributed") or drop it.
    supported_statements are "<hypothesis kind>: <statement>" for the supporting claims the Critic accepted. A claim on
    the metric itself ("trips fell 45.5% on Thanksgiving Thursday") backs no cause; a claim of kind "other" backs the
    causes its statement names."""
    backed = set()
    for st in supported_statements:
        kind, _, body = (st or "").partition(": ")
        if kind in _KIND_CAUSE:
            backed.add(_KIND_CAUSE[kind])
        elif kind == "other":
            backed |= {k for k, rx in _CAUSE_RE.items() if rx.search(body)}
    problems = []
    for k, sent in named_causes(text):
        if k not in backed:
            problems.append(f"the text names {k} as a cause (\"{sent.strip()[:140]}\") but no critic-supported claim supports a {k} "
                            f"hypothesis; hedge it (for example 'may have contributed', 'not tested') or drop it")
    return problems


def validate_final(ans: FinalAnswer, evidence: dict[str, Evidence], allowed_qids: set[str], extra_values=(), context: str = "",
                   dq_notes=(), version: int = CHECKS_VERSION, supported_statements=None, plan_text: str = "") -> list[str]:
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
                problems.append(f"claim {i + 1}: {cit.qid} was not used by any critic-supported claim (you may cite only: {', '.join(sorted(allowed_qids)) or 'none'}; drop or rewrite this claim)")
            if cit.sql_hash != ev.sql_hash:
                problems.append(f"claim {i + 1}: sql_hash for {cit.qid} must be {ev.sql_hash}")
            if cit.table.lower() not in [t.lower() for t in ev.tables]:
                problems.append(f"claim {i + 1}: table for {cit.qid} must be one of {ev.tables}")
    problems += check_text_numbers("\n".join([ans.headline, ans.answer, *(c.text for c in ans.claims), ans.recommendation, *ans.caveats]),
                                   evidence, allowed_qids, extra_values, strict_rates=version >= 2,
                                   day_lists=known_days(evidence, context) if version >= 4 else None,
                                   plan_pcts=plan_thresholds(plan_text) if version >= 5 else None)
    if version >= 2:
        cmp = check_comparisons_v5 if version >= 5 else check_comparisons
        for i, c in enumerate(ans.claims):
            problems += [f"claim {i + 1}: {p}" for p in cmp(c.text, evidence, {x.qid for x in c.citations})]
        problems += cmp("\n".join([ans.headline, ans.answer]), evidence, allowed_qids)
    if version >= 5:
        problems += check_units("\n".join([ans.headline, ans.answer, *(c.text for c in ans.claims), ans.recommendation, *ans.caveats]),
                                evidence, allowed_qids)
        problems += check_aggregate_labels("\n".join([ans.headline, ans.answer, *(c.text for c in ans.claims), ans.recommendation]),
                                           evidence, allowed_qids, extra_values)
    if version >= 3 and supported_statements is not None:
        problems += check_causes(ans.headline + ".\n" + ans.answer, list(supported_statements))
    problems += check_text_dates("\n".join([ans.headline, ans.answer, ans.recommendation, *ans.caveats, *(c.text for c in ans.claims)]),
                                 evidence, context)
    prose = "\n".join([ans.headline, ans.answer, ans.recommendation, *(c.text for c in ans.claims)])
    problems += check_consistency(prose + "\n" + "\n".join(ans.caveats), evidence, context, dq_notes, ans.caveats)
    return problems


_DATE_RE = re.compile(r"\b20\d\d-\d\d-\d\d\b|\b\d{1,2}(?:st|nd|rd|th)\b|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?(?:\s+20\d\d)?\b"
                      r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+20\d\d)?\b|\b20\d\d\b", re.I)
_NUM_RE = re.compile(r"(?<![\w.])[-+]?\$?(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s?(k\b|%|x\b|×)?", re.I)


_RATE_COL = re.compile(r"pct|percent|ratio|share|change|chg|diff|delta|rate", re.I)
# "(7, 14, 28)" after "December Saturdays" lists days of the month, not measurements
_DAY_LIST = re.compile(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?(?:\s+[A-Za-z]+){0,2}\s*\(\s*\d{1,2}(?:\s*(?:,|and|&)\s*\d{1,2})+\s*\)", re.I)
# "the ±5% threshold" is an analyst-chosen tolerance, not a measured value
_TOLERANCE = re.compile(r"(?:±|\+/-)\s?\d+(?:\.\d+)?\s?%")


_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
_LIST_SEP = r"\s*(?:,\s*(?:and\s+|&\s*)?|and\s+|&\s*)"
_DAY = r"\d{1,2}(?:st|nd|rd|th)?"
# "2024-11-07, 14, 21" / "Nov 7, 14 and 21": an anchor date followed by more days of the same month
_DAYS_AFTER = re.compile(rf"(?P<anchor>\b20\d\d-(?P<m1>\d\d)-\d\d\b|\b(?P<m2>{_MONTH})\s+{_DAY}\b)(?P<rest>(?:{_LIST_SEP}{_DAY}\b(?![-./]\d|\s*(?:%|x\b|k\b|{_MONTH})))+)", re.I)
# "7, 14 and 21 Dec": days of the month listed before the month that ends the list
_DAYS_BEFORE = re.compile(rf"(?P<rest>\b(?:{_DAY}{_LIST_SEP})+){_DAY}\s+(?P<m>{_MONTH})", re.I)


def blank_day_lists(text: str, known_days: set[tuple[int, int]]) -> str:
    """Blank the bare day numbers of a date list when every date they form is known (in a query result, a note or the
    question). A list with any unknown date is left alone, so its numbers are still checked as figures."""
    def days_of(rest: str) -> list[int]:
        return [int(d) for d in re.findall(r"\d{1,2}", rest)]

    def ok(month: int, days: list[int]) -> bool:
        return bool(days) and all((month, d) in known_days for d in days)

    def after(m):
        month = int(m.group("m1")) if m.group("m1") else _MON[m.group("m2").lower()[:3]]
        if ok(month, days_of(m.group("rest"))):
            return m.group("anchor") + " " * len(m.group("rest"))
        return m.group(0)

    def before(m):
        month = _MON[m.group("m").lower()[:3]]
        if ok(month, days_of(m.group("rest"))):
            return " " * len(m.group("rest")) + m.group(0)[len(m.group("rest")):]
        return m.group(0)

    return _DAYS_BEFORE.sub(before, _DAYS_AFTER.sub(after, text))


def known_days(evidence: dict[str, Evidence], context: str = "") -> set[tuple[int, int]]:
    out = month_days(context)
    for ev in evidence.values():
        if ev and ev.status == "ok":
            for r in ev.rows:
                for v in r:
                    out |= month_days(str(v))
    return out


def check_text_numbers(text: str, evidence: dict[str, Evidence], allowed_qids: set[str], extra_values=(), strict_rates: bool = True,
                       day_lists: set[tuple[int, int]] | None = None, plan_pcts: set[float] | None = None) -> list[str]:
    """Every number written in the final answer (headline, answer, claims, recommendation and caveats) must be backed
    by evidence. A % or Nx must be a derived value of a critic-supported claim (re-evaluated from its formula) or a
    cell of a percentage or ratio column of a query the answer may cite (allowed_qids), never an arbitrary cell that
    happens to be close (a zone id of 12 does not back "up ~12%"). Any other number must be a cell of a successful
    query or a verified derived value. Dates, day-of-month lists, +/- tolerances and small counts (<= 12) are exempt."""
    derived = [abs(float(x)) for x in extra_values if x is not None]
    cells, rate_cells = [], []
    for q, ev in evidence.items():
        if not ev or ev.status != "ok":
            continue
        rate_idx = {i for i, c in enumerate(ev.columns) if _RATE_COL.search(c)} if q in allowed_qids else set()
        for r in ev.rows:
            for i, v in enumerate(r):
                n = _num(v)
                if n is not None:
                    cells.append(abs(n))
                    if i in rate_idx:
                        rate_cells.append(abs(n))
    plain_pool = sorted(set(derived + cells))
    rate_pool = sorted(set(derived + rate_cells)) if strict_rates else plain_pool
    problems = []
    if plan_pcts:
        text, problems = plan_threshold_mentions(text, plan_pcts, rate_pool)
    if day_lists is not None:
        text = blank_day_lists(text, day_lists)
    text = _TOLERANCE.sub(" ", _DAY_LIST.sub(" ", _DATE_RE.sub(" ", text)))
    for m in _NUM_RE.finditer(text):
        raw, suf = m.group(1), (m.group(2) or "").lower()
        val = float(raw.replace(",", ""))
        if suf == "k":
            val *= 1000
        if not suf and "." not in raw and val <= 12:
            continue
        if suf in ("%", "x", "\u00d7"):
            cands = rate_pool
            def tol(c):
                return abs(val - c) <= max(0.51, 0.005 * val)
        else:
            cands = plain_pool
            def tol(c):
                return abs(val - c) <= max(0.0051 * abs(c), 0.51 if "." not in raw else 0.0051) or (suf == "k" and abs(val - c) <= 500)
        if not any(tol(c) for c in cands):
            problems.append(f"the number '{m.group(0).strip()}' in the answer is not backed by cited evidence")
    return problems


_CMP = re.compile(r"\b(?P<dir>higher|greater|more|above|larger|longer|lower|less|fewer|below|smaller|shorter)\s+than\s+"
                  r"(?:on\s+|for\s+|in\s+|at\s+)?(?:(?:all|each|every|any)\s+(?:of\s+)?)?(?:the\s+|those\s+)?"
                  r"(?:other|remaining|rest)\b", re.I)
_CMP_UP = {"higher", "greater", "more", "above", "larger", "longer"}
_CMP_NEG = re.compile(r"\b(?:not|n't|never|no)\b(?:\W+\w+){0,3}\W*$", re.I)


def check_comparisons(text: str, evidence: dict[str, Evidence], qids: set[str]) -> list[str]:
    """'X was 4.79 miles, higher than the other three Mondays' must hold against the cited rows: the row holding 4.79
    must be above every other row of that column (below, for 'lower than'). Comparisons with an average ('than the
    mean of the other ...') are not matched; derived percentages are checked elsewhere."""
    problems = []
    for sent in _sentences(text):
        for m in _CMP.finditer(sent):
            head = sent[:m.start()]
            if _CMP_NEG.search(head):
                continue
            nums = [float(x.replace(",", "")) for x, suf in _NUM_RE.findall(_DATE_RE.sub(" ", head)) if not suf]
            days = month_days(head)
            up = m.group("dir").lower() in _CMP_UP
            for q in sorted(qids):
                ev = evidence.get(q)
                if not ev or ev.status != "ok" or len(ev.rows) < 2:
                    continue
                for ci, col in enumerate(ev.columns):
                    vals = [_num(r[ci]) for r in ev.rows]
                    if sum(v is not None for v in vals) < 2:
                        continue
                    targets = [i for i, v in enumerate(vals) if v is not None and any(value_matches(n, v) for n in nums)]
                    if days:
                        dated = [i for i in targets if any(month_days(str(x)[:10]) & days for x in ev.rows[i])]
                        targets = dated
                    if len(targets) != 1:
                        continue
                    t = targets[0]
                    worse = [(ev.rows[i], v) for i, v in enumerate(vals) if i != t and v is not None and (v >= vals[t] if up else v <= vals[t])]
                    if worse:
                        row, v = worse[0]
                        key = next((str(x)[:10] for x in row if isinstance(x, str)), "another row")
                        problems.append(f"'{m.group(0)}' is contradicted by {q}.{col}: {vals[t]:g} vs {v:g} for {key}")
    return list(dict.fromkeys(problems))


_MON = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_DM_RE = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?|"
                    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b|"
                    r"\b20\d\d-(\d\d)-(\d\d)\b", re.I)


def month_days(text: str) -> set[tuple[int, int]]:
    out = set()
    for m in _DM_RE.finditer(text):
        if m.group(1):
            out.add((_MON[m.group(2).lower()[:3]], int(m.group(1))))
        elif m.group(3):
            out.add((_MON[m.group(3).lower()[:3]], int(m.group(4))))
        else:
            out.add((int(m.group(5)), int(m.group(6))))
    return out


def check_text_dates(text: str, evidence: dict[str, Evidence], context: str = "") -> list[str]:
    """Every calendar date written in the answer must appear in a query result, a data-quality note or the question."""
    known = month_days(context)
    for ev in evidence.values():
        if ev.status == "ok":
            for r in ev.rows:
                for v in r:
                    known |= month_days(str(v))
    return [f"the date {m:02d}-{d:02d} in the answer does not appear in any query result or note"
            for m, d in sorted(month_days(text) - known)]


# ---- Consistency checks: wording that the evidence contradicts ------------------------------------------------------
# Weather words that claim benign conditions. 'clear', 'fine', 'good' etc. only count next to a weather noun, so
# "the evidence is clear" or "a mild drop" are not weather claims.
_BENIGN = r"mild|dry|clear|calm|fair|fine|pleasant|sunny|benign|good|favou?rable|rain-free|precipitation-free|unremarkable"
_BENIGN_DAY = r"mild|dry|sunny|clear|calm|pleasant|rain-free"
_WX_PATTERNS = [
    re.compile(rf"\b(?P<adj>{_BENIGN})(?:\s+(?:and|,)\s*(?:{_BENIGN}))?\s+(?:\w+\s+)?(?:weather|skies|sky)\b", re.I),
    re.compile(rf"\b(?P<adj>{_BENIGN_DAY})(?:\s+(?:and|,)\s*(?:{_BENIGN}))?\s+(?:\w+\s+)?(?:day|conditions)\b", re.I),
    re.compile(rf"\b(?:weather(?:\s+conditions)?|skies|sky)\b(?:\s+[\w,()\-:/]+){{0,5}}?\s+(?:was|were|remained|stayed|being|is|are|looked)\s+"
               rf"(?:\w+\s+)?(?P<adj>{_BENIGN})\b", re.I),
    re.compile(r"\b(?P<adj>no|zero|without(?:\s+any)?)\s+(?:significant\s+|notable\s+|measurable\s+|meaningful\s+)?"
               r"(?P<what>rain|rainfall|precipitation|snow|snowfall)\b", re.I),
]
_NEG = re.compile(r"\b(not|n't|never|neither|nor|hardly|far from|anything but)\W+(?:\w+\W+){0,2}$", re.I)
_COMPARISON_SET = re.compile(r"\b\w+s\s+(?:with|that\s+had|having)\s*$|\b(?:other|comparison|baseline|control|reference)\b[^.;]*$", re.I)
PRECIP_MM_THRESHOLD = 5.0   # a day with >= ~5 mm of rain is not 'mild', 'dry' or 'clear'
TRACE_MM = 1.0              # 'no rain' tolerates a trace below 1 mm


def _weather_rows(evidence: dict[str, Evidence]) -> list[dict]:
    """(month, day) -> precip_mm / snow_mm for every successful query that returned weather columns."""
    out = []
    for ev in evidence.values():
        if not ev or ev.status != "ok":
            continue
        cols = [c.lower() for c in ev.columns]
        prcp = [i for i, c in enumerate(cols) if re.search(r"prcp|precip|rain", c)]
        snow = [i for i, c in enumerate(cols) if re.search(r"snow", c) and not re.search(r"snwd|depth", c)]
        if not prcp and not snow:
            continue
        for r in ev.rows:
            days = set()
            for v in r:
                if isinstance(v, str) and re.match(r"^\s*20\d\d-\d\d-\d\d", v):
                    days |= month_days(v[:10])
            if not days:
                continue
            def scale(i):  # inch columns are converted to mm
                return 25.4 if re.search(r"(_in|inch)", cols[i]) else 1.0
            p = [n * scale(i) for i in prcp if (n := _num(r[i])) is not None]
            sn = [n * scale(i) for i in snow if (n := _num(r[i])) is not None]
            out.append({"days": days, "prcp": max(p) if p else None, "snow": max(sn) if sn else None,
                        "qid": ev.qid, "table": ev.primary_table})
    return out


_BASELINE_WORDS = re.compile(r"\b(?:other|others|comparison|baseline|control|reference|previous|surrounding|remaining)\b", re.I)
_CLAUSE_BREAK = re.compile(r"[,;:()]|\b(?:whereas|while|but|although|though|unlike|compared\s+(?:with|to)|versus|vs\.?)\b", re.I)


def _days_for(sent: str, start: int, end: int):
    """The day a weather word describes: the nearest date in its own clause, else the nearest date in the sentence.
    None when the sentence has no date."""
    spans = []
    for m in _DM_RE.finditer(sent):
        spans.append((m.start(), m.end(), month_days(m.group(0))))
    if not spans:
        return None
    breaks = [0] + [b.start() for b in _CLAUSE_BREAK.finditer(sent)] + [len(sent)]
    lo = max(b for b in breaks if b <= start)
    hi = min((b for b in breaks if b >= end), default=len(sent))
    same = [sp for sp in spans if sp[0] >= lo and sp[1] <= hi] or spans
    def dist(sp):
        return start - sp[1] if sp[1] <= start else (sp[0] - end if sp[0] >= end else 0)
    return min(same, key=dist)[2]


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?;])\s+|\n+", text) if s.strip()]


def check_weather_wording(text: str, evidence: dict[str, Evidence], context: str = "") -> list[str]:
    """A benign weather word (mild, dry, clear, no rain, ...) must not describe a day whose recorded weather contradicts
    it: precipitation >= PRECIP_MM_THRESHOLD or any snowfall. The day is the date in the same sentence, else the dates
    of the question. Negated wording ('was not mild') is allowed."""
    rows = _weather_rows(evidence)
    if not rows:
        return []
    default_days = month_days(context)
    problems = []
    for sent in _sentences(text):
        for pat in _WX_PATTERNS:
            for m in pat.finditer(sent):
                if m.groupdict().get("what") and _COMPARISON_SET.search(sent[:m.start()]):
                    continue  # "other Saturdays with no snow" describes the comparison days, not the day in question
                if _NEG.search(sent[:m.start()]) or re.search(r"\b(not|never|hardly)\b|n't", m.group(0), re.I):
                    continue
                adj = m.group("adj").lower()
                what = (m.groupdict().get("what") or "").lower()
                days = _days_for(sent, m.start(), m.end())
                if days is None:  # no date in the sentence
                    if _BASELINE_WORDS.search(sent):
                        continue  # "the baseline Saturday had dry weather" is about the comparison days
                    days = default_days
                for r in rows:
                    hit = r["days"] & days
                    if not hit:
                        continue
                    day = "-".join(f"{x:02d}" for x in sorted(hit)[0])
                    wet = r["prcp"] is not None and r["prcp"] >= PRECIP_MM_THRESHOLD
                    snowy = r["snow"] is not None and r["snow"] > 0
                    if what in ("rain", "rainfall", "precipitation"):
                        bad = r["prcp"] is not None and r["prcp"] >= TRACE_MM
                    elif what in ("snow", "snowfall"):
                        bad = snowy
                    else:
                        bad = wet or snowy
                    if bad:
                        rec = ", ".join(x for x in (f"{r['prcp']:g} mm precipitation" if r["prcp"] is not None else "",
                                                    f"{r['snow']:g} mm snow" if r["snow"] is not None and r["snow"] > 0 else "") if x)
                        problems.append(f"'{m.group(0).strip()}' contradicts the recorded weather for {day} ({rec}, "
                                        f"{r['qid']} {r['table']}): describe the weather with the recorded values or drop the word")
                        break
    return list(dict.fromkeys(problems))


_NO_DQ = re.compile(
    r"\b(?:no|zero|without(?:\s+any)?|free\s+of(?:\s+any)?|absence\s+of)\s+(?:known\s+|other\s+|apparent\s+|obvious\s+)?"
    r"(?:data[- ]?quality|dq|quality|data)\s+(?:flags?|issues?|problems?|concerns?|caveats?|anomal\w+|errors?|warnings?)\b"
    r"|\b(?:data[- ]?quality|dq)\s+(?:is|was|looks|appears)\s+(?:\w+\s+)?(?:clean|fine|ok|good|unflagged)\b"
    r"|\bno\s+(?:flags?|issues?)\s+(?:on|with|in|for)\s+(?:the\s+)?data(?:[- ]quality)?\b", re.I)
_DQ_CAVEAT = re.compile(r"data[- ]?quality|d_holiday|date_dim|\bDQ\d", re.I)


def check_dq_consistency(text: str, dq_notes=(), caveats=()) -> list[str]:
    """'No data-quality flags/issues' cannot be claimed while the run raised data-quality notes (e.g. the TPCDS
    d_holiday vs Nager.Date check) or the answer itself carries a data-quality caveat."""
    flagged = list(dq_notes) + [c for c in caveats if _DQ_CAVEAT.search(c) and not _NO_DQ.search(c)]
    if not flagged:
        return []
    return [f"'{m.group(0).strip()}' contradicts the data-quality findings of this run ({len(flagged)} flagged, e.g. "
            f"{flagged[0][:140]}); say what was checked and keep the data-quality caveat instead"
            for m in _NO_DQ.finditer(text)]


def check_consistency(text: str, evidence: dict[str, Evidence], context: str = "", dq_notes=(), caveats=()) -> list[str]:
    return check_weather_wording(text, evidence, context) + check_dq_consistency(text, dq_notes, caveats)


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


# ---- v4: SQL-side checks before a query runs --------------------------------------------------------------------------
_DOW7 = re.compile(r"\bd_dow\s*(?:=|==|>=|<=|<>|!=)\s*'?7\b|\bd_dow\s+(?:not\s+)?in\s*\([^)]*\b7\b[^)]*\)", re.I)


def check_calendar_sql(sql: str) -> list[str]:
    """TPCDS_DB.date_dim.d_dow is 0=Sunday..6=Saturday (unlike Spark dayofweek(), 1=Sunday..7=Saturday), so d_dow = 7
    never matches a row: an empty result that looks like evidence of absence."""
    s = re.sub(r"'(?:[^']|'')*'", lambda m: m.group(0) if m.group(0) in ("'7'", "'6'") else "''", sql)
    if _DOW7.search(s):
        return ["TPCDS_DB.date_dim.d_dow is 0=Sunday..6=Saturday (Saturday is 6, not 7; Spark dayofweek() is the one that "
                "uses 1..7): fix the d_dow filter or use d_day_name = 'Saturday'"]
    return []


def repair_table_names(sql: str, allowed: list[str]) -> tuple[str, list[str]]:
    """Rewrite a table reference that is not allowlisted to the ONE allowlisted table it can only mean: the same table
    name under a wrong or extra prefix (whybot.date_dim, whybot.tpcds_db.date_dim -> TPCDS_DB.date_dim). Ambiguous or
    unknown names are left as they are, so the allowlist still fails closed. Returns (sql, ["old -> new", ...])."""
    from .datasource.base import DataSourceError, referenced_tables
    try:
        refs = referenced_tables(sql)
    except DataSourceError:
        return sql, []
    low = {t.lower(): t for t in allowed}
    repairs = []
    for ref in sorted(refs):
        if ref in low:
            continue
        parts = ref.split(".")
        cands = {t for lt, t in low.items() if ref.endswith("." + lt)} or {t for lt, t in low.items() if lt.split(".")[-1] == parts[-1]}
        if len(cands) != 1:
            continue
        new = cands.pop()
        pat = re.compile(r"(?<![\w.`])" + r"\.".join(rf"`?{re.escape(p)}`?" for p in parts) + r"(?![\w`])", re.I)
        sql2, n = pat.subn(new, sql)
        if n:
            sql = sql2
            repairs.append(f"{ref} -> {new}")
    return sql, repairs


_HYP_DIRECTION = re.compile(
    r"\b(?:hypothes[ie]s|H\d+)\b[^.;]{0,40}?\b(?:expects?|expected|predicts?|predicted|anticipat\w+|assum\w+|requires?|implies|implied)\b"
    r"|\bdirection of the hypothes[ie]s\b|\b(?:does not|doesn't|fails? to) (?:support|match|fit) (?:the |its )?hypothes[ie]s\b", re.I)


def check_critic_verdicts(report: CriticReport, checks: dict[str, NumericCheck]) -> list[str]:
    """A claim whose numbers pass the automated check is evidence even when it refutes its hypothesis. A verdict of
    unsupported (or weak) that rests on the hypothesis' expected direction, not on the rows, is sent back."""
    problems = []
    for v in report.verdicts:
        ch = checks.get(v.cid)
        if ch is None or not ch.ok or v.verdict == "supported":
            continue
        if _HYP_DIRECTION.search(v.reason or ""):
            problems.append(f"{v.cid}: judge the claim against the returned rows, not against what its hypothesis expected. Its numbers "
                            f"pass the automated check; if the statement says what the rows say, mark the claim supported and set the "
                            f"hypothesis to refuted (a claim that refutes a hypothesis is valid evidence). Mark it weak or unsupported "
                            f"only for a problem with the claim itself (wording, baseline, confounders)")
    return problems


# ---- v5: units -------------------------------------------------------------------------------------------------------
# The unit of a result column, from its name. TLC trip_distance is miles; NOAA prcp/snow are mm, tmax/tmin deg C.
def column_unit(col: str) -> str | None:
    c = col.lower()
    if re.search(r"(?:^|_)km(?:_|$)|kilomet", c):
        return "km"
    if "distance" in c or re.search(r"(?:^|_)(?:mi|miles)(?:_|$)", c):
        return "mi"
    if re.search(r"(?:^|_)mm(?:_|$)", c) or c in ("prcp", "snow", "snwd") or "precip" in c:
        return "mm"
    if re.search(r"(?:^|_)f$", c) and re.search(r"temp|tmax|tmin", c):
        return "F"
    if re.search(r"(?:^|_)c$", c) and re.search(r"temp|tmax|tmin", c) or c in ("tmax", "tmin"):
        return "C"
    if re.search(r"amount|fare|tip|toll|price|revenue|usd", c):
        return "USD"
    return None


UNIT_LABEL = {"mi": "miles", "km": "km", "mm": "mm", "C": "deg C", "F": "deg F", "USD": "USD"}
_UNIT_DIM = {"mi": "length", "km": "length", "mm": "depth", "cm": "depth", "in": "depth", "C": "temp", "F": "temp",
             "USD": "money", "EUR": "money", "GBP": "money"}
_UNIT_WORDS = [
    ("km", r"\bkms?\b|\bkilomet(?:er|re)s?\b"),
    ("mi", r"\bmiles?\b|\bmi\b"),
    ("mm", r"\bmm\b|\bmillimet(?:er|re)s?\b"),
    ("cm", r"\bcm\b|\bcentimet(?:er|re)s?\b"),
    ("in", r"\binch(?:es)?\b"),
    ("C", r"°\s?C\b|\bdeg(?:rees?)?\s?C\b|\bdegrees? Celsius\b|\bCelsius\b"),
    ("F", r"°\s?F\b|\bdeg(?:rees?)?\s?F\b|\bdegrees? Fahrenheit\b|\bFahrenheit\b"),
    ("USD", r"\bUSD\b|\bUS\$|\bdollars?\b"),
    ("EUR", r"€|\bEUR\b|\beuros?\b"),
    ("GBP", r"£|\bGBP\b"),
]
_UNIT_ALT = "|".join(f"(?P<u_{k}>{p})" for k, p in _UNIT_WORDS)
_NUM_UNIT = re.compile(rf"(?<![\w.])~?(?P<n>[-+]?(?:\d{{1,3}}(?:,\d{{3}})+(?:\.\d+)?|\d+(?:\.\d+)?))\s?(?:{_UNIT_ALT})", re.I)
_DOLLAR_NUM = re.compile(r"(?<![\w$])\$\s?(?P<n>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)")
_UNIT_ANY = re.compile(_UNIT_ALT, re.I)


def _unit_of(m) -> str:
    return next(k for k, _ in _UNIT_WORDS if m.group(f"u_{k}"))


def units_text(evidence) -> str:
    """The units of the returned columns, for the Analyst and Writer prompts (v5)."""
    by_unit: dict[str, list[str]] = {}
    for ev in (evidence.values() if isinstance(evidence, dict) else evidence):
        if ev and ev.status == "ok":
            for c in ev.columns:
                u = column_unit(c)
                if u and c not in by_unit.setdefault(u, []):
                    by_unit[u].append(c)
    if not by_unit:
        return ""
    parts = [f"{', '.join(cols)} = {UNIT_LABEL[u]}" + (" (TLC distances are miles: never write km)" if u == "mi" else "")
             for u, cols in by_unit.items()]
    return "UNITS: " + "; ".join(parts) + "."


def check_units(text: str, evidence: dict[str, Evidence], qids: set[str]) -> list[str]:
    """A number written with a unit must carry the unit of the column it comes from ('4.79 km' when avg_trip_distance is
    miles; '9.4 mm' when 9.4 is tmax_c). A unit word of a dimension the cited columns measure in another unit ('4.40-5.53
    km' with distances in miles) is flagged too. Numbers that are not cells of a column with a known unit are left to the
    number check."""
    cols = []  # (qid, column, unit, values)
    for q in sorted(qids):
        ev = evidence.get(q)
        if not ev or ev.status != "ok":
            continue
        for i, c in enumerate(ev.columns):
            u = column_unit(c)
            cols.append((q, c, u, [r[i] for r in ev.rows]))
    problems, flagged = [], set()
    text = _DATE_RE.sub(lambda m: " " * len(m.group(0)), text)
    found = [(m.group("n"), _unit_of(m), m.group(0).strip()) for m in _NUM_UNIT.finditer(text)]
    found += [(m.group("n"), "USD", m.group(0).strip()) for m in _DOLLAR_NUM.finditer(text)]
    for n, unit, shown in found:
        hits = [(q, c, u) for q, c, u, vals in cols if any(_num(v) is not None and value_matches(n, v) for v in vals)]
        if not hits or any(u is None for _, _, u in hits) or any(u == unit for _, _, u in hits):
            continue
        q, c, u = hits[0]
        flagged.add(unit)
        problems.append(f"'{shown}': {n} is {q}.{c}, which is in {UNIT_LABEL[u]}, not {unit}")
    known = {u for _, _, u, _ in cols if u}
    for m in _UNIT_ANY.finditer(text):
        unit = _unit_of(m)
        same_dim = sorted(u for u in known if _UNIT_DIM.get(u) == _UNIT_DIM.get(unit))
        if same_dim and unit not in same_dim and unit not in flagged:
            flagged.add(unit)
            c = next(c for _, c, u, _ in cols if u in same_dim)
            problems.append(f"the answer writes {m.group(0).strip()!r} but {c} is in {UNIT_LABEL[same_dim[0]]}")
    return list(dict.fromkeys(problems))


# ---- v5: a value labelled average / mean / baseline must be that aggregate ------------------------------------------------
_AGG_WORD = r"baseline|mean|average|averag(?:ed|es|ing)|avg|typical"
_CROSS_ROW = re.compile(r"baseline|typical|mean\s+of|average\s+of|averag(?:ed|es|ing)\b", re.I)
_AGG_STOP = (r"(?:to|vs|versus|than|compared|while|whereas|but|and|from|against|on|for|in|range|between|min\w*|max\w*|low\w*|high\w*|"
             r"each|individual|single|rows?|dates?|days?|weeks?|\w+days?|weekends?|values?|cells?|counts?|figures?|numbers?|levels?|readings?)\b")
_AGG_BEFORE = re.compile(rf"\b(?P<w>{_AGG_WORD})\b(?P<gap>(?:\W+(?!\d|{_AGG_STOP})\w+){{0,3}})[^\w]*~?\$?$", re.I)
_AGG_AFTER = re.compile(rf"^\s?(?:{_UNIT_ALT})?\s*(?P<w>{_AGG_WORD})\b", re.I)
_AGG_COL = re.compile(r"baseline|mean|avg|average|typical", re.I)
_PERROW_COL = re.compile(r"^avg_|_avg$|^mean_", re.I)


def _stated_tolerance(raw: str, suffix: str) -> float:
    """Half a unit of the last stated digit: 141,642 -> 0.5; 4.3 -> 0.05; 141,600 -> 50; 141.6k -> 50."""
    s = raw.replace(",", "")
    if "." in s:
        t = 0.5 * 10 ** -len(s.split(".")[1])
    else:
        z = len(s) - len(s.rstrip("0")) if len(s) >= 4 else 0
        t = 0.5 * 10 ** z
    return t * 1000 if suffix == "k" else t


def check_aggregate_labels(text: str, evidence: dict[str, Evidence], qids: set[str], derived=()) -> list[str]:
    """A number labelled average, mean or baseline ('baseline average of 141,642', '~4.3 miles baseline') must equal, to its
    stated precision, an aggregate of the cited rows: the mean of a numeric column over all rows or all rows but one (the
    target day), a verified derived value, or the single value of a column that is itself an aggregate (a one-value
    baseline_avg_trips; for a plain 'average' also a per-row avg_ column). Appearing somewhere in the results is not enough: 141,642 is one
    Thursday's count, not the mean of the three."""
    aggs, agg_cells, perrow_cells = [abs(float(d)) for d in derived if d is not None], [], []
    for q in sorted(qids):
        ev = evidence.get(q)
        if not ev or ev.status != "ok":
            continue
        for i, c in enumerate(ev.columns):
            vals = [_num(r[i]) for r in ev.rows]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            # an aggregate-named column is an aggregate only when it holds one value; per-day values in a column the SQL
            # agent named baseline_avg_trips are still single days
            if _AGG_COL.search(c) and not _PERROW_COL.search(c) and len(set(vals)) == 1:
                agg_cells += vals
            elif _AGG_COL.search(c):
                perrow_cells += vals
            if len(vals) >= 2:
                aggs.append(sum(vals) / len(vals))
                if len(vals) >= 3:
                    aggs += [(sum(vals) - v) / (len(vals) - 1) for v in vals]
    blanked = _DATE_RE.sub(lambda m: " " * len(m.group(0)), text)
    problems = []
    for sent_m in re.finditer(r"[^.;!?\n]+(?:[.;!?](?!\d)|$)|[^\n]+", blanked):
        sent, base = sent_m.group(0), sent_m.start()
        prev_end = 0
        for m in _NUM_RE.finditer(sent):
            raw, suf = m.group(1), (m.group(2) or "").lower()
            before, after = sent[prev_end:m.start()], sent[m.end():]
            prev_end = m.end()
            if suf in ("%", "x", "\u00d7"):
                continue
            lab = _AGG_BEFORE.search(before)
            lab_a = _AGG_AFTER.search(after)
            if not lab and not lab_a or re.match(r"\s?(?:[A-Za-z]+\s?)?,\s*[-+~$]?\d", after):
                continue  # no label, or the first value of a list ('baseline: 80127, 100492, 93060')
            label = ((lab.group(0) if lab else "") + " " + (lab_a.group("w") if lab_a else "")).strip()
            val = abs(float(raw.replace(",", ""))) * (1000 if suf == "k" else 1)
            if "." not in raw and val <= 12 and not suf:
                continue
            tol = _stated_tolerance(raw, suf) + 1e-9
            pool = aggs + agg_cells + ([] if _CROSS_ROW.search(label) else perrow_cells)
            if any(abs(val - a) <= tol for a in pool):
                continue
            near = min(aggs, key=lambda a: abs(a - val), default=None)
            hint = f"; the nearest aggregate of the cited rows is {near:,.2f}" if near is not None else ""
            shown = text[base + m.start():base + m.end()].strip()
            problems.append(f"'{shown}' is labelled {label.split()[0].lower()!r} but is not the average of the cited rows "
                            f"(a single row's value or a wrong mean){hint}: declare the average in derived with a formula, "
                            f"or drop the label")
    return list(dict.fromkeys(problems))


# ---- v5: comparison scope --------------------------------------------------------------------------------------------------
_CMP_SET = (r"(?:other|remaining|rest|comparison|baseline|control|reference|prior|previous|preceding|surrounding|normal|typical|usual)\b"
            r"|(?:\w+\s+){0,2}(?:days|mondays|tuesdays|wednesdays|thursdays|fridays|saturdays|sundays|weekdays|weekends|dates)\b")
_CMP5 = re.compile(r"\b(?P<dir>higher|greater|more|above|larger|longer|lower|less|fewer|below|smaller|shorter)\b(?P<mid>(?:[^.;]|\.(?=\d)){0,80}?)"
                   r"\bthan\s+(?:on\s+|for\s+|in\s+|at\s+|during\s+)?(?:(?:all|each|every|any)\s+(?:of\s+)?)?(?:the\s+|those\s+|its\s+)?"
                   rf"(?P<set>{_CMP_SET})", re.I)
_ON_AVERAGE = re.compile(r"\bon average\b|\bthan the (?:average|mean)\b|\baverage of the\b|\bmean of the\b", re.I)


def _row_date(row):
    import datetime as _dt
    for x in row:
        m = re.match(r"(20\d\d)-(\d\d)-(\d\d)", str(x))
        if m:
            try:
                return _dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                return None
    return None


def _compared_rows(rows, t, vals, named_days):
    """The rows a 'than the comparison days' phrase compares with: the other dates the sentence names, else the rows on
    the target's weekday (baselines here are like-for-like weekdays), else every other row."""
    others = [(i, rows[i], v) for i, v in enumerate(vals) if i != t and v is not None]
    dates = {i: _row_date(rows[i]) for i, _, _ in others}
    named = [o for o in others if dates[o[0]] and (dates[o[0]].month, dates[o[0]].day) in named_days]
    if named:
        return [(r, v) for _, r, v in named]
    td = _row_date(rows[t])
    if td:
        same = [o for o in others if dates[o[0]] and dates[o[0]].weekday() == td.weekday()]
        if same:
            return [(r, v) for _, r, v in same]
    return [(r, v) for _, r, v in others]


def check_comparisons_v5(text: str, evidence: dict[str, Evidence], qids: set[str]) -> list[str]:
    """'X was higher than (on) the comparison Saturdays' must hold for every compared row of the cited column, not just
    on average: the row holding the stated value must be above every other row (below, for 'lower than'). Wording that
    says 'on average' (or compares with the mean) is checked against the mean of the other rows instead. The
    comparison may be split ('higher on the snow day (4.79) than on comparison Saturdays')."""
    problems = []
    for sent in _sentences(text):
        for m in _CMP5.finditer(sent):
            head = sent[:m.start("mid") + len(m.group("mid"))]
            if _CMP_NEG.search(sent[:m.start()]):
                continue
            on_avg = bool(_ON_AVERAGE.search(sent))
            nums = [float(x.replace(",", "")) for x, suf in _NUM_RE.findall(_DATE_RE.sub(" ", head)) if not suf]
            days = month_days(head)
            up = m.group("dir").lower() in _CMP_UP
            for q in sorted(qids):
                ev = evidence.get(q)
                if not ev or ev.status != "ok" or len(ev.rows) < 2:
                    continue
                for ci, col in enumerate(ev.columns):
                    vals = [_num(r[ci]) for r in ev.rows]
                    if sum(v is not None for v in vals) < 2:
                        continue
                    targets = [i for i, v in enumerate(vals) if v is not None and any(value_matches(n, v) for n in nums)]
                    if days:
                        targets = [i for i in targets if any(month_days(str(x)[:10]) & days for x in ev.rows[i])]
                    if len(targets) != 1:
                        continue
                    t = targets[0]
                    others = _compared_rows(ev.rows, t, vals, month_days(sent) - days)
                    if not others:
                        continue
                    phrase = m.group(0) if len(m.group(0)) < 90 else f"{m.group('dir')} ... than {m.group('set')}"
                    if on_avg:
                        mean = sum(v for _, v in others) / len(others)
                        if (vals[t] <= mean) if up else (vals[t] >= mean):
                            problems.append(f"'{phrase}' (on average) is contradicted by {q}.{col}: {vals[t]:g} vs the mean "
                                            f"of the others {mean:g}")
                        continue
                    worse = [(row, v) for row, v in others if (v >= vals[t] if up else v <= vals[t])]
                    if worse:
                        row, v = worse[0]
                        key = next((str(x)[:10] for x in row if isinstance(x, str)), "another row")
                        problems.append(f"'{phrase}' does not hold for every compared day: {q}.{col} is {vals[t]:g} vs {v:g} for "
                                        f"{key}. Name the days it holds for, or say 'on average' if it holds against their mean")
    return list(dict.fromkeys(problems))


# ---- v5: thresholds the plan set (e.g. 'refute if within 5% of the baseline') -----------------------------------------------
_THR_OP = re.compile(r"(?:\b(?P<op>within|inside|under|below|less than|fewer than|more than|greater than|over|above|exceed\w*|beyond|"
                     r"at least|at most|outside)|(?P<sym>[<>≤≥]))\s*(?:the\s+|a\s+|an\s+|its\s+)?(?:plan(?:'s|ned)?\s+)?"
                     r"(?:materiality\s+|significance\s+)?~?(?:±|\+/-)?$", re.I)
_THR_NOUN_BEFORE = re.compile(r"\b(?:threshold|margin|band|tolerance|cut-?off)\s+(?:of\s+)?~?(?:±|\+/-)?$", re.I)
_THR_NOUN_AFTER = re.compile(r"^\s*(?:materiality\s+|significance\s+|noise\s+)?(?:threshold|margin|band|tolerance|cut-?off)\b", re.I)
_THR_NEG = re.compile(r"\b(?:not|no|never|far from|short of|without)\b[^,;]{0,40}$", re.I)
_THR_NEG_AFTER = re.compile(r"^[^,;.]{0,40}\b(?:not|never)\b", re.I)
_THR_LE = {"within", "inside", "under", "below", "less than", "fewer than", "at most", "<", "≤"}


def plan_thresholds(plan_text: str) -> set[float]:
    """The percentages the plan's hypotheses use as decision thresholds (support if / refute if)."""
    return {float(x) for x in _PCT.findall(plan_text or "")}


def plan_threshold_mentions(text: str, plan_pcts: set[float], pool: list[float]) -> tuple[str, list[str]]:
    """A plan threshold ('within 5% of the baseline', 'the 5% threshold') is a parameter, not a measurement. Named as a
    threshold ('the 5% threshold') it may be quoted as one. Used as a comparison ('within 5%', 'more than 5%') it needs
    the measured change it is compared with in the same sentence, backed (in pool), and the comparison must hold
    ('+0.15%, within the 5% threshold'). Returns the text with every threshold
    mention blanked (so the number check does not also report it) and a problem for each mention that is not backed
    that way."""
    if not plan_pcts:
        return text, []
    out, problems, pos = [], [], 0
    for sent_m in re.finditer(r"(?:[^.!?\n]|\.(?=\d))+(?:[.!?]|$)|[^\n]+", text):
        sent = sent_m.group(0)
        mentions, measured = [], []
        for m in _PCT.finditer(sent):
            val = abs(float(m.group(1)))
            before, after = sent[:m.start()], sent[m.end():]
            op = _THR_OP.search(before)
            framed = op or _THR_NOUN_BEFORE.search(before) or _THR_NOUN_AFTER.search(after)
            if framed and any(abs(val - t) < 1e-9 for t in plan_pcts):
                kind = (op.group("op") or op.group("sym")).lower() if op else ""
                # 'not a meaningful reduction (>5%)', 'far from the >5% threshold': the comparison is negated
                neg = bool(op) and bool(_THR_NEG.search(before[:op.start()]) or _THR_NEG_AFTER.search(after))
                mentions.append((m, val, kind, neg))
            elif any(abs(val - c) <= max(0.051, 0.005 * val) for c in pool):
                measured.append(val)
        for m, val, kind, neg in mentions:
            if kind:
                le = (kind in _THR_LE) != neg
                ok = any((x <= val) if le else (x >= val) for x in measured)
            else:  # named as a threshold ('the ±5% threshold for fares is analyst-defined'): a parameter, stated as one
                ok = True
            if not ok:
                problems.append(f"'{m.group(0).strip()}' is a threshold from the plan's hypotheses, not a measurement: quote it only in "
                                f"the same sentence as the measured change it is compared with (e.g. '+0.15%, within the "
                                f"{val:g}% threshold'), and only when that comparison holds; otherwise state the measured change")
        s = list(sent)
        for m, *_ in mentions:
            s[m.start():m.end()] = " " * (m.end() - m.start())
        out.append(text[pos:sent_m.start()] + "".join(s))
        pos = sent_m.end()
    out.append(text[pos:])
    return "".join(out), problems
