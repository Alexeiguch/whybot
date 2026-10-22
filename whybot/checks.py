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


def value_matches(claimed, actual, rel_tol: float = 0.005) -> bool:
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
    problems += check_derived_values(claim)
    return NumericCheck(claim.cid, not problems, problems)


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


def check_derived_values(claim: Claim) -> list[str]:
    """Every derived value must re-evaluate from the cited cells, and every % or Nx ratio in the statement must be one
    of the declared derived values (or a cited cell). Catches 'fell 49.5%' when the cells give 45.5%."""
    nums = [_num(cv.value) for cv in claim.cited_values]
    problems, ok_values = [], [abs(n) for n in nums if n is not None]
    for d in claim.derived:
        try:
            actual = eval_formula(d.formula, nums)
        except ValueError as e:
            problems.append(f"derived {d.name}: {e}")
            continue
        if not _close(d.value, actual):
            problems.append(f"derived {d.name}={d.value:g} but {d.formula} = {actual:.2f} from the cited cells")
        else:
            ok_values.append(abs(actual))
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


def validate_final(ans: FinalAnswer, evidence: dict[str, Evidence], allowed_qids: set[str], extra_values=()) -> list[str]:
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
    problems += check_text_numbers(f"{ans.headline}\n{ans.answer}\n" + "\n".join(c.text for c in ans.claims),
                                   evidence, allowed_qids, extra_values)
    return problems


_DATE_RE = re.compile(r"\b20\d\d-\d\d-\d\d\b|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?(?:\s+20\d\d)?\b"
                      r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+20\d\d)?\b|\b20\d\d\b", re.I)
_NUM_RE = re.compile(r"(?<![\w.])[-+]?\$?(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s?(k\b|%|x\b|×)?", re.I)


def check_text_numbers(text: str, evidence: dict[str, Evidence], allowed_qids: set[str], extra_values=()) -> list[str]:
    """Every number written in the final answer must be a cell of allowed evidence, a verified derived value, or a
    percentage/ratio between two such cells (0.5 pp tolerance). Dates, years and small counts (<= 12) are exempt."""
    pool = [abs(float(x)) for x in extra_values]
    for q in allowed_qids:
        ev = evidence.get(q)
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
