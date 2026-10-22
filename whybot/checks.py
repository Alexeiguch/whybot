"""Deterministic validation used by the Critic and Writer, and the branching policy. Pure functions (unit-tested)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

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
                v = v.strip().strip("'\"")
                rows = [r for r in rows if value_matches(v, r[ki]) or str(r[ki])[:10] == v[:10] and len(v) >= 8]
                if not rows:
                    problems.append(f"{cv.qid}: no row matches {cv.row_hint}")
                    continue
        if not any(value_matches(cv.value, r[idx]) for r in rows):
            problems.append(f"{cv.qid}.{cv.column}={cv.value} not found in returned rows")
    corrections: list[str] = []
    problems += check_derived_values(claim, corrections)
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


def check_derived_values(claim: Claim, corrections: list[str] | None = None) -> list[str]:
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
    for left in re.findall(r"\{[A-Za-z_]\w*\}", claim.statement):
        problems.append(f"placeholder {left} has no derived value with that name")
    for p in [float(m) for m in _PCT.findall(claim.statement)]:
        if not any(_close(p, v) for v in ok_values):
            problems.append(f"{p:g}% in the statement is not recomputable from cited cells (declare it in derived with a formula)")
    for r in [float(m) for m in _RATIO.findall(claim.statement)]:
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


def validate_final(ans: FinalAnswer, evidence: dict[str, Evidence], allowed_qids: set[str], extra_values=(), context: str = "") -> list[str]:
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
    problems += check_text_numbers(f"{ans.headline}\n{ans.answer}\n" + "\n".join(c.text for c in ans.claims),
                                   evidence, allowed_qids, extra_values)
    problems += check_text_dates("\n".join([ans.headline, ans.answer, ans.recommendation, *ans.caveats, *(c.text for c in ans.claims)]),
                                 evidence, context)
    return problems


_DATE_RE = re.compile(r"\b20\d\d-\d\d-\d\d\b|\b\d{1,2}(?:st|nd|rd|th)\b|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?(?:\s+20\d\d)?\b"
                      r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+20\d\d)?\b|\b20\d\d\b", re.I)
_NUM_RE = re.compile(r"(?<![\w.])[-+]?\$?(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s?(k\b|%|x\b|×)?", re.I)


def check_text_numbers(text: str, evidence: dict[str, Evidence], allowed_qids: set[str], extra_values=()) -> list[str]:
    """Every number written in the final answer must be a cell of allowed evidence, a verified derived value, or a
    percentage/ratio between two such cells (0.5 pp tolerance). Dates, years and small counts (<= 12) are exempt."""
    # Any cell of a successful query is a real, recorded number; claims still may only CITE critic-supported queries.
    pool = [abs(float(x)) for x in extra_values]
    for q, ev in evidence.items():
        if ev and ev.status == "ok":
            for r in ev.rows:
                pool += [abs(n) for n in (_num(v) for v in r) if n is not None]
    pool = sorted(set(pool))[:400]
    derived = set()
    for i, a in enumerate(pool):
        for b in pool:
            if b and a != b:
                derived.add(abs((a - b) / b * 100))
                derived.add(a / b)
    problems = []
    for m in _NUM_RE.finditer(_DATE_RE.sub(" ", text)):
        raw, suf = m.group(1), (m.group(2) or "").lower()
        val = float(raw.replace(",", ""))
        if suf == "k":
            val *= 1000
        if not suf and "." not in raw and val <= 12:
            continue
        cands = pool if suf in ("", "k") else list(pool) + list(derived)
        tol = (lambda c: abs(val - c) <= max(0.51, 0.005 * val)) if suf in ("%", "x", "×") else \
              (lambda c: abs(val - c) <= max(0.0051 * abs(c), 0.51 if "." not in raw else 0.0051) or (suf == "k" and abs(val - c) <= 500))
        if not any(tol(c) for c in cands):
            problems.append(f"the number '{m.group(0).strip()}' in the answer is not backed by cited evidence")
    return problems


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
