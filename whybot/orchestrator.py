"""WhyBot orchestrator: Planner -> SQL Executor -> Analyst -> Critic -> (retry | write | escalate) -> Writer.

Control flow is driven by the Critic's verdicts plus deterministic number checks (checks.decide_next), capped at
MAX_ITERATIONS (<=3) and a wall-clock budget. SQL failures trigger schema introspection and query repair; a query that
still fails marks its hypothesis untestable for that iteration (fallback) instead of crashing the run.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .agents import Analyst, Critic, Planner, SQLAgent, Writer
from .checks import check_claim_numbers, check_consistency, decide_next, drop_uncited, result_sanity, tables_in_sql, validate_final
from .config import ROOT, Settings
from .datasource import DataSource, DataSourceError, guard_sql, make_datasource, sql_hash
from .evidence import Evidence
from .llm import BaseLLM, LLMError
from .schemas import ClaimSet, CriticReport, FinalAnswer, Hypothesis, HypothesisStatus, QuerySpec
from .trace import Tracer

AGENTS = [
    {"name": "Planner", "model_role": "planner", "role": "Turns the question into falsifiable hypotheses and a SQL plan; revises it from critic feedback"},
    {"name": "Executor", "model_role": "sql", "role": "Writes Lightning SQL, runs it through the DataSource adapter, introspects schema and repairs failing queries"},
    {"name": "Analyst", "model_role": "analyst", "role": "Drafts claims that cite exact numbers from query results"},
    {"name": "Critic", "model_role": "critic", "role": "Checks every claim against the numbers, rejects weak ones, requests retries or escalates"},
    {"name": "Writer", "model_role": "writer", "role": "Writes the final answer; every claim cites query id, SQL hash and source table"},
    {"name": "Memory", "model_role": None, "role": "Meterless H-MEM recall/remember, World Model provenance, Markovian carryover"},
]


_MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def parse_dates(text: str) -> list:
    """Dates like 2024-11-28 or 28 Nov 2024 / 28 November 2024 found in free text."""
    import re
    from datetime import date
    out = []
    for y, m, d in re.findall(r"\b(20\d\d)-(\d\d)-(\d\d)\b", text):
        try:
            out.append(date(int(y), int(m), int(d)))
        except ValueError:
            pass
    for d, mon, y in re.findall(r"\b(\d{1,2})\s+([A-Za-z]{3,9})\s+(20\d\d)\b", text):
        m = _MONTHS.get(mon[:3].lower())
        if m:
            try:
                out.append(date(int(y), m, int(d)))
            except ValueError:
                pass
    return out


WRITER_RESERVE_S = 150  # stop iterating early enough for the Writer to finish inside the run budget
RUN_ID_RE = __import__("re").compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$")


def result_ok(r) -> bool:
    return bool(r) and r.get("status") != "error"


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:6]


def _load_notes() -> dict:
    p = ROOT / "whybot" / "catalog_notes.json"
    return {k: v for k, v in json.loads(p.read_text()).items() if not k.startswith("_")} if p.exists() else {}


class WhyBot:
    def __init__(self, settings: Settings, llm: BaseLLM | None = None, datasource: DataSource | None = None,
                 meterless=None, use_meterless: bool | None = None, echo: bool = True):
        self.s = settings
        self._llm = llm
        self._ds = datasource
        self._meterless = meterless
        self.use_meterless = settings.meterless_enabled if use_meterless is None else use_meterless
        self.echo = echo
        self.notes = _load_notes()

    # ------------------------------------------------------------------ setup helpers
    def _make_llm(self, tr: Tracer) -> BaseLLM:
        def on_call(ev: dict):
            role = ev.get("role", "?")
            agent = {"planner": "Planner", "sql": "Executor", "analyst": "Analyst", "critic": "Critic", "writer": "Writer"}.get(role, role)
            if ev["event"] == "llm_call":
                summ = (f"{ev['model']} recorded {ev.get('recorded_at')} (live {ev.get('original_latency_s')}s, "
                        f"tokens={ev.get('original_completion_tokens')}, prompt_match={ev.get('prompt_match')})") if ev.get("replayed") \
                    else f"{ev['model']} {ev['latency_s']}s tokens={ev.get('completion_tokens')}"
                tr.emit(agent, "llm_call", output_summary=summ,
                        status="info", data={k: v for k, v in ev.items() if k != "event"})
            else:
                tr.emit(agent, ev["event"], output_summary=str(ev.get("error", ""))[:300], status="retry",
                        retry_count=ev.get("attempt", 0), data=ev)
        if self._llm is not None:
            self._llm.on_call = on_call
            return self._llm
        import os
        from .llm import CassetteLLM, NvidiaLLM, cassette_path
        cpath = cassette_path(self.s.fixtures_dir, self._question)
        if self.s.sample_mode and os.environ.get("SAMPLE_LLM", "replay") != "live":
            llm = CassetteLLM(cpath, on_call=on_call)
            h = llm.header
            self._recorded_recall = h.get("hmem_recall")
            tr.emit("Orchestrator", "llm_mode", status="info",
                    output_summary=f"SAMPLE_MODE REPLAY of live run {h.get('live_run_id')} recorded {h.get('recorded_at')} "
                                   f"(trace {h.get('trace')}); LLM and SQL outputs come from that run, the checks and branch policy re-execute",
                    data={"mode": "replay", **h})
            return llm
        live = NvidiaLLM(self.s, on_call=on_call)
        tr.emit("Orchestrator", "llm_mode", status="info", output_summary="LIVE NVIDIA Nemotron calls", data={"mode": "live"})
        if self.s.record_fixtures:
            self._cassette = CassetteLLM(cpath, inner=live, on_call=on_call, run_id=self._run_id)
            return self._cassette
        return live

    def _make_ds(self, tr: Tracer) -> DataSource:
        if self._ds is not None:
            return self._ds
        try:
            return make_datasource(self.s)
        except DataSourceError as e:
            tr.emit("Executor", "datasource_unavailable", output_summary=str(e), status="error",
                    data={"requested": self.s.data_source})
            if self.s.data_source == "zetaris" and (Path(self.s.fixtures_dir) / "catalog.json").exists():
                from .datasource.replay import ReplayDataSource
                ds = ReplayDataSource(self.s.fixtures_dir)
                tr.emit("Executor", "fallback", output_summary=f"Zetaris unreachable; falling back to {ds.label}", status="fallback")
                return ds
            raise

    def _make_meterless(self, tr: Tracer, replay: bool = False):
        if self._meterless is not None or not self.use_meterless:
            return self._meterless
        try:
            from .memory import Meterless
            # Replayed / dev runs get their own store so they never pollute the live investigation memory.
            d = self.s.meterless_data_dir if not replay else self.s.meterless_data_dir.with_name(self.s.meterless_data_dir.name + "-replay")
            tr.emit("Memory", "meterless_store", status="info", output_summary=f"Meterless data dir: {d.name}")
            return Meterless(d)
        except Exception as e:
            tr.emit("Memory", "meterless_unavailable", output_summary=str(e)[:300], status="error")
            return None

    # ------------------------------------------------------------------ main entry
    def run(self, question: str, run_id: str | None = None, subscribe=None) -> dict:
        run_id = run_id or new_run_id()
        if not RUN_ID_RE.match(run_id):
            raise ValueError("invalid run_id: use letters, digits, '.', '_' or '-' (max 80 chars)")
        self._question, self._cassette, self._run_id = question, None, run_id
        self._recorded_recall = None
        result = None
        t0 = time.time()
        deadline = t0 + self.s.run_timeout_s
        tr = Tracer(run_id, self.s.logs_dir, echo=self.echo)
        if subscribe:
            tr.subscribe(subscribe)
        mem = ds = None
        try:
            tr.emit("Orchestrator", "start", input_summary=question,
                    output_summary=f"max_iterations={self.s.max_iterations} run_timeout_s={self.s.run_timeout_s}",
                    data={"models": {"planner": self.s.planner_model, "critic": self.s.critic_model,
                                     "sql": self.s.sql_model, "writer": self.s.writer_model}})
            llm = self._make_llm(tr)
            llm.deadline, llm.call_timeout_s = deadline, self.s.llm_timeout_s
            ds = self._make_ds(tr)
            tr.emit("Executor", "datasource", output_summary=ds.label, data={"kind": ds.kind, "dialect": ds.dialect})
            mem = self._make_meterless(tr, replay=ds.kind != "live")
            result = self._investigate(question, run_id, tr, llm, ds, mem, deadline)
        except Exception as e:  # structured error, never a silent crash
            tr.emit("Orchestrator", "error", output_summary=f"{type(e).__name__}: {e}", status="error")
            result = {"status": "error", "error": {"type": type(e).__name__, "message": str(e)[:800],
                                                   "recoverable": isinstance(e, (DataSourceError, LLMError))}}
        finally:
            if self._cassette is not None and result_ok(locals().get("result")):
                self._cassette.save(question)
            if mem is not None and self._meterless is None:
                mem.close()
            if ds is not None and self._ds is None:
                ds.close()
        result.update({"run_id": run_id, "question": question, "trace_id": run_id,
                       "log_file": str(tr.path.relative_to(ROOT)) if tr.path.is_relative_to(ROOT) else str(tr.path),
                       "execution_time_seconds": round(time.time() - t0, 1), "agents": AGENTS})
        result["wall_clock"] = {"started_at": datetime.fromtimestamp(t0, timezone.utc).isoformat(timespec="seconds"),
                                "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                "seconds": result["execution_time_seconds"]}
        tr.emit("Orchestrator", "finish", output_summary=f"status={result['status']} wall_clock={result['execution_time_seconds']}s",
                status="success" if result["status"] != "error" else "error", data={"wall_clock": result["wall_clock"]})
        (Path(self.s.logs_dir) / f"{run_id}.result.json").write_text(json.dumps(result, indent=1, default=str))
        return result

    # ------------------------------------------------------------------ the loop
    def _investigate(self, question, run_id, tr: Tracer, llm, ds: DataSource, mem, deadline) -> dict:
        planner, sqlagent, analyst, critic, writer = Planner(llm), SQLAgent(llm), Analyst(llm), Critic(llm), Writer(llm)

        catalog = ds.catalog()
        tables = [t["table"] for t in catalog]
        tr.emit("Executor", "introspect_schema", output_summary=f"{len(catalog)} tables: {', '.join(tables)}",
                data={"catalog": catalog})

        memory_text, mem_hits = self._recall(mem, question, tr)
        inv_entity = None
        if mem:
            inv_entity = self._safe(tr, lambda: mem.upsert_entity(etype="investigation", key=run_id, run_id=run_id, agent="Planner",
                                                                  attrs={"name": run_id, "question": question, "source": ds.label}))
            inv_entity = inv_entity["id"] if inv_entity else None

        hypotheses: dict[str, Hypothesis] = {}
        hyp_status: dict[str, HypothesisStatus] = {}
        prev_claims: ClaimSet | None = None
        evidence: dict[str, Evidence] = {}
        carryover, feedback = "", ""
        claims = ClaimSet(claims=[])
        report: CriticReport | None = None
        checks: dict = {}
        decision = None
        markov_steps: list[str] = []
        qcount = 0
        dq_notes = None

        for it in range(1, self.s.max_iterations + 1):
            tr.iteration = it
            if time.time() > deadline:
                tr.emit("Orchestrator", "timeout", output_summary="run budget exhausted before iteration", status="error")
                break
            # ---- Plan
            done = [f"{e.qid} [{e.status}] ({e.hypothesis_id}) {e.purpose[:140]}" for e in evidence.values()]
            required = sorted({f.hypothesis_id for f in report.follow_up_queries}) if (report and it > 1) else []
            try:
                plan, meta = planner.plan(question, catalog, self.notes, memory_text, carryover, feedback, done, it, required)
            except LLMError as e:
                if it == 1:
                    raise
                # a failed follow-up plan must not discard verified work: write (or escalate) with what the Critic accepted
                tr.emit("Planner", "failed", output_summary=f"follow-up plan failed: {str(e)[:250]}; finishing with verified claims",
                        status="error", target_agent="Writer")
                break
            for h in plan.hypotheses:
                hypotheses[h.id] = h  # revisions with the same id replace the earlier statement
            tr.emit("Planner", "plan", input_summary=feedback[:300] or question, target_agent="Executor",
                    output_summary=f"{len(plan.hypotheses)} hypotheses, {len(plan.queries)} queries; metric={plan.metric}",
                    data={"plan": plan.model_dump(), "llm": meta})
            plan_text = "\n".join(f"{h.id} [{h.kind}] {h.statement} | support if: {h.support_if} | refute if: {h.refute_if}"
                                  for h in hypotheses.values()) + f"\nMetric: {plan.metric}; target: {plan.target_window}; baseline: {plan.baseline_window}"

            # ---- Execute (with repair + fallback)
            for spec in plan.queries:
                if time.time() > deadline:
                    break
                qcount += 1
                ev = self._execute(f"Q{qcount}", spec, it, question, catalog, tables, ds, sqlagent, memory_text, tr)
                evidence[ev.qid] = ev
                if ev.status == "ok" and mem and inv_entity:
                    fact = self._safe(tr, lambda: mem.assert_evidence(
                        run_id=run_id, agent="Executor", about=inv_entity, predicate=f"evidence.{ev.qid}",
                        value={"hypothesis": ev.hypothesis_id, "purpose": ev.purpose[:200], "columns": ev.columns,
                               "rows_head": ev.rows[:5], "row_count": ev.row_count, "source": ev.source_label},
                        table=ev.primary_table, sql_hash=ev.sql_hash, confidence=0.9))
                    if fact:
                        ev.wm_fact_id = fact["id"]
                        tr.emit("Memory", "world_model.assert_fact", output_summary=f"{ev.qid} -> {fact['id']} (provenance: Executor, {ev.sql_hash}, {ev.primary_table})",
                                data={"fact_id": fact["id"], "qid": ev.qid, "sql_hash": ev.sql_hash, "table": ev.primary_table})

            if it == 1 and dq_notes is None:
                dq_notes = self._holiday_dq_check(question, plan, tables, ds, tr, evidence)

            ok_ev = [e for e in evidence.values() if e.status == "ok" and not e.qid.startswith("DQ")]
            if not ok_ev:
                feedback = "Every query failed. Use simpler queries and only columns from the catalog."
                decision_txt = "retry: no evidence"
                tr.emit("Orchestrator", "branch", output_summary=decision_txt, status="retry", target_agent="Planner")
                carryover = self._carry(mem, tr, it, "No successful queries.", "all", "simplify", carryover)
                continue

            # ---- Analyse
            claims = None
            for a_try in (1, 2):  # one retry before giving up on this iteration's analysis
                try:
                    claims, meta = analyst.draft(question, plan_text, ok_ev, feedback)
                    tr.emit("Analyst", "draft_claims", target_agent="Critic", output_summary=f"{len(claims.claims)} claims",
                            data={"claims": claims.model_dump()["claims"], "llm": meta})
                    break
                except LLMError as e:
                    tr.emit("Analyst", "failed", output_summary=f"attempt {a_try}: {str(e)[:300]}", status="error")
                    if "budget exhausted" in str(e) or "cassette" in str(e):
                        break
            if claims is None:
                if prev_claims is not None and prev_claims.claims:
                    claims = prev_claims  # keep the last drafted claims rather than discarding earlier work
                    tr.emit("Orchestrator", "reuse_claims", output_summary=f"Analyst unavailable; re-checking the {len(claims.claims)} claims of the previous iteration", status="fallback")
                else:
                    claims = ClaimSet(claims=[])
            prev_claims = claims

            # ---- Critique: deterministic number checks, then the LLM critic
            checks = {c.cid: check_claim_numbers(c, evidence, question) for c in claims.claims}
            for cid, ch in checks.items():
                tr.emit("Critic", "number_check", output_summary=f"{cid}: {'PASS' if ch.ok else 'REJECT ' + '; '.join(ch.problems)}"[:400],
                        status="success" if ch.ok else "rejected", data={"cid": cid, "ok": ch.ok, "problems": ch.problems, "corrections": ch.corrections})
                if ch.corrections:
                    tr.emit("Critic", "arithmetic_corrected", output_summary=f"{cid}: " + "; ".join(ch.corrections)[:400], status="info",
                            data={"cid": cid, "corrections": ch.corrections})
            report = None
            if claims.claims and time.time() < deadline:
                try:
                    report, meta = critic.review(question, plan_text, claims, ok_ev, checks, tables, it, self.s.max_iterations)
                except LLMError as e:
                    tr.emit("Critic", "failed", output_summary=str(e)[:300], status="error")
            if report is None:
                report = self._fallback_report(claims, checks, retry=it < self.s.max_iterations)
                tr.emit("Critic", "fallback_verdicts", output_summary="LLM critic unavailable; deterministic verdicts only", status="fallback")
            else:
                tr.emit("Critic", "review", target_agent="Orchestrator",
                        output_summary=f"decision={report.decision}; " + ", ".join(f"{v.cid}:{v.verdict}" for v in report.verdicts),
                        data={"report": report.model_dump(), "llm": meta})
            for v in report.verdicts:
                ok_num = checks.get(v.cid) and checks[v.cid].ok
                final = v.verdict if ok_num else "unsupported"
                tr.emit("Critic", "verdict", output_summary=f"{v.cid} {final} ({v.strength:.2f}): {v.reason}"[:400],
                        status="success" if final == "supported" else "rejected", confidence=v.strength,
                        data={"cid": v.cid, "verdict": final, "llm_verdict": v.verdict, "number_check": bool(ok_num)})

            for hs in report.hypothesis_status:  # latest non-'untested' status wins across iterations
                if hs.status != "untested" or hs.hypothesis_id not in hyp_status:
                    hyp_status[hs.hypothesis_id] = hs
            decision = decide_next(it, self.s.max_iterations, report, checks, evidence, out_of_time=time.time() > deadline - min(WRITER_RESERVE_S, 0.2 * self.s.run_timeout_s))
            tr.emit("Orchestrator", "branch", output_summary=f"{decision.action}: {decision.reason}",
                    target_agent={"retry": "Planner", "write": "Writer", "escalate": "Writer"}[decision.action],
                    status="retry" if decision.action == "retry" else ("escalated" if decision.action == "escalate" else "success"))
            self._record_verdicts(mem, tr, run_id, inv_entity, claims, report, checks, evidence)

            completed = "; ".join(f"{e.qid}:{e.status}" for e in evidence.values())
            remaining = "; ".join(f"{f.hypothesis_id}: {f.purpose[:100]}" for f in report.follow_up_queries) or "none"
            decisions = "; ".join(f"{h.hypothesis_id}={h.status}" for h in report.hypothesis_status)
            carryover = self._carry(mem, tr, it, report.feedback, completed, remaining, carryover, decisions)
            markov_steps.append(f"Iteration {it}: {report.feedback}\n[STATE_CHECKPOINT]\n{carryover}")
            if decision.action != "retry":
                break
            feedback = report.feedback + ("\nFOLLOW-UP TESTS REQUESTED:\n" + "\n".join(
                f"- {f.hypothesis_id}: {f.purpose} (tables: {', '.join(f.tables)})" for f in report.follow_up_queries) if report.follow_up_queries else "")
        else:
            pass

        if decision is None or decision.action == "retry":
            # loop ended by cap/timeout while still wanting to retry
            from .checks import Decision
            sup = decision.supported_cids if decision else []
            decision = Decision("write" if sup else "escalate", "iteration cap reached", sup)
            tr.emit("Orchestrator", "branch", output_summary=f"{decision.action}: {decision.reason}", status="escalated" if decision.action == "escalate" else "success", target_agent="Writer")

        status = "answered" if decision.action == "write" else "insufficient_evidence"
        if status == "insufficient_evidence":
            tr.emit("Orchestrator", "escalate", output_summary="insufficient evidence: answer will state what is and is not known", status="escalated")

        final = self._write(question, status, decision, claims, report, evidence, writer, tr, dq_notes or [])
        self._remember(mem, tr, run_id, question, status, final, evidence, report)
        if mem and markov_steps:
            rec = self._safe(tr, lambda: mem.record_run(question, markov_steps + ["[TASK_COMPLETE]"]))
            if rec:
                tr.emit("Memory", "markovian.record_run", output_summary=f"{rec['chunks']} chunks, status={rec['status']}", data=rec)

        return {
            "status": status,
            "output": {"summary": f"{final.headline}\n\n{final.answer}", "headline": final.headline, "answer": final.answer,
                       "recommendation": final.recommendation, "claims": [c.model_dump() for c in final.claims],
                       "caveats": final.caveats, "confidence": final.confidence, "markdown": render_markdown(final, evidence, ds.label),
                       "recommendations": [final.recommendation], "artifacts": []},
            "hypotheses": [{**h.model_dump(), "status": hyp_status[h.id].status if h.id in hyp_status else "untested",
                            "status_reason": hyp_status[h.id].reason if h.id in hyp_status else ""} for h in hypotheses.values()],
            "evidence": [e.to_dict() for e in evidence.values()],
            "iterations": tr.iteration,
            "data_source": {"label": ds.label, "kind": ds.kind},
            "memory": {"recalled": mem_hits, "meterless": bool(mem)},
            "data_quality_notes": dq_notes or [],
        }

    # ------------------------------------------------------------------ pieces
    def _execute(self, qid, spec: QuerySpec, it, question, catalog, tables, ds, sqlagent, memory_text, tr) -> Evidence:
        ev = Evidence(qid=qid, hypothesis_id=spec.hypothesis_id, purpose=spec.purpose, iteration=it, tables=list(spec.tables))
        failed: list[dict] = []
        cat = catalog
        for attempt in range(self.s.max_sql_repairs + 1):
            try:
                draft, meta = sqlagent.write(spec.purpose, spec.tables, cat, self.notes, ds.dialect, ds.dialect_notes,
                                             memory_text, question, failed)
            except LLMError as e:
                ev.error = str(e)
                tr.emit("Executor", "sql_generation_failed", output_summary=str(e)[:300], status="error", retry_count=attempt)
                break
            used = tables_in_sql(draft.sql, tables) or spec.tables
            try:
                draft.sql = guard_sql(draft.sql, allowed_tables=ds.allowed_tables())  # the exact SQL that runs is the SQL that is hashed and cited
            except DataSourceError:
                pass  # ds.run() raises the same error below and the repair loop handles it
            h = sql_hash(draft.sql)
            tr.emit("Executor", "sql_written", input_summary=spec.purpose, output_summary=draft.sql[:600], retry_count=attempt,
                    data={"qid": qid, "sql": draft.sql, "sql_hash": h, "tables": used, "llm": meta})
            try:
                res = ds.run(draft.sql, timeout_s=self.s.sql_timeout_s)
            except DataSourceError as e:
                failed.append({"sql": draft.sql, "error": str(e)})
                ev.attempts.append({"sql": draft.sql, "sql_hash": h, "error": str(e)[:500]})
                tr.emit("Executor", "query_failed", output_summary=f"{qid}: {str(e)[:300]}", status="error", retry_count=attempt,
                        data={"qid": qid, "sql_hash": h, "error": str(e)[:800]})
                if attempt < self.s.max_sql_repairs:
                    fresh = []
                    for t in used:
                        try:
                            fresh.append({"table": t, "columns": ds.describe(t), "note": ""})
                        except Exception as de:
                            fresh.append({"table": t, "columns": [], "note": f"describe failed: {str(de)[:100]}"})
                    cat = [c for c in catalog if c["table"] not in used] + fresh
                    tr.emit("Executor", "schema_introspect_for_repair", target_agent="Executor", retry_count=attempt + 1,
                            output_summary=f"re-described {', '.join(used)}; asking SQL agent to repair",
                            data={"tables": fresh})
                continue
            smells = result_sanity(res.columns, res.rows)
            if smells and not res.rows and attempt >= 1:
                smells = []  # an empty result confirmed twice is evidence of absence, not an error
            if smells and attempt < self.s.max_sql_repairs:
                err = "RESULT SANITY CHECK FAILED: " + "; ".join(smells)
                failed.append({"sql": draft.sql, "error": err})
                ev.attempts.append({"sql": draft.sql, "sql_hash": h, "error": err})
                tr.emit("Executor", "result_rejected", output_summary=f"{qid}: {err}", status="rejected", retry_count=attempt,
                        target_agent="Executor", data={"qid": qid, "sql_hash": h, "rows_head": res.rows[:5], "columns": res.columns})
                continue
            ev.sql, ev.sql_hash, ev.tables = draft.sql, h, used
            ev.columns, ev.rows, ev.row_count, ev.truncated = res.columns, res.rows, len(res.rows), res.truncated
            ev.elapsed_s, ev.source_label, ev.recorded_at, ev.status = res.elapsed_s, res.source_label, res.recorded_at, "ok"
            ev.attempts.append({"sql": draft.sql, "sql_hash": h, "error": ""})
            tr.emit("Executor", "query_ok", target_agent="Analyst",
                    output_summary=f"{qid}: {ev.row_count} rows in {res.elapsed_s}s from {', '.join(used)} [{res.source_label}]",
                    retry_count=attempt, data={"qid": qid, "hypothesis_id": spec.hypothesis_id, "sql": draft.sql, "sql_hash": h, "tables": used, "columns": res.columns,
                                               "rows_head": res.rows[:8], "row_count": ev.row_count, "source": res.source_label})
            return ev
        ev.status = "failed"
        ev.error = ev.error or (failed[-1]["error"] if failed else "unknown")
        tr.emit("Executor", "fallback", output_summary=f"{qid} failed after {len(failed)} attempts; hypothesis {spec.hypothesis_id} untestable this iteration",
                status="fallback", data={"qid": qid})
        return ev

    def _holiday_dq_check(self, question, plan, tables, ds, tr, evidence) -> list[str]:
        """Critic-owned deterministic data-quality check: TPCDS date_dim.d_holiday vs real Nager.Date holidays.

        Disagreements are reported as data-quality notes (never as causal evidence)."""
        low = {t.lower(): t for t in tables}
        dd, hol = low.get("tpcds_db.date_dim"), low.get("whybot.us_holidays")
        if not (dd and hol):
            return []
        dates = parse_dates(question + " " + plan.target_window + " " + plan.baseline_window)
        if not dates:
            return []
        lo, hi = min(dates) - timedelta(days=31), max(dates) + timedelta(days=31)
        sql = (f"SELECT d.d_date AS cal_date, d.d_holiday AS tpcds_d_holiday, h.holiday_name AS nager_holiday "
               f"FROM {dd} d LEFT JOIN {hol} h ON h.holiday_date = d.d_date AND h.is_national = true "
               f"WHERE d.d_date BETWEEN DATE '{lo}' AND DATE '{hi}' AND (d.d_holiday = 'Y' OR h.holiday_name IS NOT NULL) "
               f"ORDER BY d.d_date")
        ev = Evidence(qid="DQ1", hypothesis_id="data_quality", purpose="Critic check: TPCDS d_holiday vs Nager.Date national holidays",
                      iteration=1, tables=[dd, hol])
        try:
            g = guard_sql(sql)
            res = ds.run(g, timeout_s=self.s.sql_timeout_s)
        except DataSourceError as e:
            tr.emit("Critic", "data_quality_check", output_summary=f"holiday cross-check failed: {str(e)[:200]}", status="error")
            return []
        ev.sql, ev.sql_hash, ev.status = g, sql_hash(g), "ok"
        ev.columns, ev.rows, ev.row_count, ev.source_label = res.columns, res.rows, len(res.rows), res.source_label
        evidence[ev.qid] = ev
        notes = []
        for row in res.rows:
            if len(row) < 3:
                continue
            d, flag, name = row[0], row[1], row[2]
            if (str(flag).upper() == "Y") != (name is not None):
                notes.append(f"{d}: TPCDS date_dim.d_holiday={flag} but Nager.Date says "
                             f"{name + ' (national holiday)' if name else 'no national holiday'} [DQ1 {ev.sql_hash}]")
        tr.emit("Critic", "data_quality_check", target_agent="Writer",
                output_summary=(f"{len(notes)} d_holiday disagreements in {lo}..{hi} (flagged as data-quality notes, not evidence)"
                                + (f"; e.g. {notes[0]}" if notes else "")),
                status="rejected" if notes else "success",
                data={"qid": "DQ1", "sql": g, "sql_hash": ev.sql_hash, "tables": ev.tables, "rows_head": res.rows[:20], "notes": notes,
                      "columns": res.columns, "row_count": ev.row_count})
        return notes

    def _recall(self, mem, question, tr) -> tuple[str, list]:
        if self._recorded_recall is not None:
            # SAMPLE_MODE replay: H-MEM recall is an input to the agents like a SQL result, so the live run's recall is
            # replayed (labelled) and the prompts match the live ones. New memories still go to the replay store.
            hits = self._recorded_recall
            tr.emit("Memory", "hmem.recall", target_agent="Planner", status="success",
                    output_summary=f"RECORDED recall of the live run: {len(hits)} memories (threshold {self.s.hmem_threshold})"
                    + (f"; top: {hits[0]['content'][:120]}" if hits else ""), data={"hits": hits, "recorded": True})
            return self._recall_text(hits), hits
        if not mem:
            if self._cassette is not None:
                self._cassette.header["hmem_recall"] = []
            return "", []
        hits, seen = [], set()
        for q in (question, f"lesson SQL schema join filter {question}"):
            r = self._safe(tr, lambda: mem.recall(q, self.s.hmem_threshold))
            for m in (r or {}).get("memories", []):
                if m["memory"]["id"] not in seen:
                    seen.add(m["memory"]["id"])
                    hits.append({"id": m["memory"]["id"], "relevance": round(m["relevance"], 3), "content": m["memory"]["content"],
                                 "tags": m["memory"].get("tags", []), "source": m["memory"].get("source")})
        hits.sort(key=lambda h: -h["relevance"])
        hits = hits[:6]
        tr.emit("Memory", "hmem.recall", target_agent="Planner", output_summary=f"{len(hits)} memories (threshold {self.s.hmem_threshold})"
                + (f"; top: {hits[0]['content'][:120]}" if hits else ""), data={"hits": hits})
        if self._cassette is not None:
            self._cassette.header["hmem_recall"] = hits
        return self._recall_text(hits), hits

    @staticmethod
    def _recall_text(hits) -> str:
        return "\n".join(f"- ({h['relevance']}) {h['content']}" for h in hits)

    def _carry(self, mem, tr, it, summary, completed, remaining, previous, decisions="") -> str:
        chunk = (f"Iteration {it} summary: {summary}\n[STATE_CHECKPOINT]\nCOMPLETED: {completed}\nREMAINING: {remaining}\n"
                 f"DECISIONS: {decisions or 'none'}\nCONTEXT: iteration {it} of {self.s.max_iterations}")
        if not mem:
            return chunk.split("[STATE_CHECKPOINT]\n", 1)[1][:2000]
        r = self._safe(tr, lambda: mem.carry(chunk, previous))
        if not r:
            return chunk[-2000:]
        tr.emit("Memory", "markovian.carry", target_agent="Planner",
                output_summary=f"carryover {r['carryTokens']} tok (level={r['level']}) from {r['chunkTokens']} tok chunk",
                data={"carryover": r["carryover"], "level": r["level"]})
        return r["carryover"]

    def _record_verdicts(self, mem, tr, run_id, inv_entity, claims, report, checks, evidence):
        if not mem or not inv_entity:
            return
        verdicts = {v.cid: v for v in report.verdicts}
        for c in claims.claims:
            v = verdicts.get(c.cid)
            if not v or not checks.get(c.cid) or not checks[c.cid].ok or v.verdict != "supported":
                continue
            qids = sorted({cv.qid for cv in c.cited_values})
            ev = evidence[qids[0]]
            self._safe(tr, lambda: mem.assert_evidence(run_id=run_id, agent="Critic", about=inv_entity, predicate=f"claim.{c.cid}",
                                                       value={"statement": c.statement, "hypothesis": c.hypothesis_id, "qids": qids, "strength": v.strength},
                                                       table=ev.primary_table, sql_hash=",".join(evidence[q].sql_hash for q in qids), confidence=v.strength, kind="agent"))

    def _fallback_report(self, claims: ClaimSet, checks, retry: bool = False) -> CriticReport:
        from .schemas import ClaimVerdict
        return CriticReport(
            verdicts=[ClaimVerdict(cid=c.cid, verdict="weak" if checks[c.cid].ok else "unsupported", strength=0.3,
                                   reason="deterministic check only") for c in claims.claims],
            hypothesis_status=[], decision="retry" if retry else "insufficient_evidence",
            feedback="Critic unavailable this iteration; only number checks were applied." + (" Retrying." if retry else ""))

    @staticmethod
    def _dq_caveat(ans: FinalAnswer, dq_notes) -> FinalAnswer:
        """The calendar data-quality finding is rendered deterministically from the DQ1 rows, never paraphrased."""
        if not dq_notes:
            return ans
        keep = [c for c in ans.caveats if not re.search(r"d_holiday|date_dim|tpcds", c, re.I)]
        items = "; ".join(n.split(" [DQ1")[0].replace("TPCDS date_dim.d_holiday=N but Nager.Date says ", "d_holiday=N, Nager.Date: ")
                          for n in dq_notes[:6])
        tag = dq_notes[0].split("[DQ1 ")[-1].rstrip("]") if "[DQ1 " in dq_notes[0] else ""
        keep.append(f"Data quality (not evidence): the TPCDS date_dim.d_holiday flag disagrees with Nager.Date on {len(dq_notes)} "
                    f"date(s): {items}. Holidays come from Nager.Date only [DQ1 {tag}].")
        return ans.model_copy(update={"caveats": keep})

    def _write(self, question, status, decision, claims, report, evidence, writer, tr, dq_notes=()) -> FinalAnswer:
        sup_ids = set(decision.supported_cids)
        verdicts = {v.cid: v for v in (report.verdicts if report else [])}
        supported = [{**c.model_dump(), "critic": verdicts[c.cid].reason} for c in claims.claims if c.cid in sup_ids]
        rejected = [{"cid": c.cid, "statement": c.statement, "verdict": verdicts[c.cid].verdict if c.cid in verdicts else "unchecked",
                     "reason": verdicts[c.cid].reason if c.cid in verdicts else ""} for c in claims.claims if c.cid not in sup_ids]
        allowed = {cv.qid for c in claims.claims if c.cid in sup_ids for cv in c.cited_values}
        if not allowed:  # escalation with nothing supported: may cite any successful query as context
            allowed = {e.qid for e in evidence.values() if e.status == "ok" and not e.qid.startswith("DQ")}
        hstat = [h.model_dump() for h in (report.hypothesis_status if report else [])]
        if dq_notes:
            hstat.append({"data_quality_notes (mention as caveats, NOT evidence)": list(dq_notes)})

        verified_derived = []
        for c in claims.claims:
            if c.cid in sup_ids:
                for d in c.derived:
                    verified_derived.append(d.value)

        def validator(ans: FinalAnswer) -> list[str]:
            probs = validate_final(ans, evidence, allowed, verified_derived, context=question + "\n" + "\n".join(dq_notes),
                                   dq_notes=dq_notes)
            if status == "insufficient_evidence" and probs == ["no claims"]:
                return []
            return probs
        try:
            ans, meta = writer.write(question, status, supported, rejected, list(evidence.values()), hstat, validator)
            ans = self._dq_caveat(ans, dq_notes)
            tr.emit("Writer", "final_answer", output_summary=ans.headline, confidence=ans.confidence,
                    data={"answer": ans.model_dump(), "llm": meta})
            return ans
        except LLMError as e:
            tr.emit("Writer", "validation_failed", output_summary=str(e)[:300], status="error")
        # Deterministic fallback: render the verified claims directly, with their citations, minus any statement whose
        # wording contradicts the evidence or the run's data-quality findings.
        from .schemas import Citation, FinalClaim
        supported = [c for c in supported if not check_consistency(c["statement"], evidence, question, dq_notes)]
        fc = [FinalClaim(text=c["statement"], citations=[Citation(qid=q, sql_hash=evidence[q].sql_hash, table=evidence[q].primary_table)
                                                          for q in sorted({cv["qid"] for cv in c["cited_values"]})]) for c in supported]
        lead = supported[0]["statement"] if supported else ""
        head = (lead[:160] if status == "answered" and lead else
                ("Insufficient evidence" if status != "answered" else "Verified findings") + f" for: {question}")
        ans = FinalAnswer(headline=head,
                          answer=" ".join(c["statement"] for c in supported[:4]) or "No claim survived the Critic.",
                          recommendation="Review the verified claims and evidence table.", claims=fc,
                          caveats=["Writer fallback: deterministic rendering of verified claims"], confidence=0.3)
        ans = self._dq_caveat(drop_uncited(ans, evidence, allowed), dq_notes)
        tr.emit("Writer", "fallback_answer", output_summary=ans.headline, status="fallback")
        return ans

    def _remember(self, mem, tr, run_id, question, status, final: FinalAnswer, evidence, report):
        if not mem:
            return
        stored = []
        supported = [h.hypothesis_id for h in (report.hypothesis_status if report else []) if h.status == "supported"]
        content = f"Investigation {run_id}: Q='{question}' -> {status}: {final.headline} (supported: {', '.join(supported) or 'none'})"
        r = self._safe(tr, lambda: mem.remember(content, tags=["investigation"], source=f"agent:writer:{run_id}", confidence=final.confidence or 0.5, run_id=run_id))
        if r:
            stored.append(r["id"])
        for lesson in final.lessons[:3]:
            r = self._safe(tr, lambda: mem.remember(f"Lesson: {lesson}", tags=["lesson"], source=f"agent:writer:{run_id}", confidence=0.7, layer="long_term", run_id=run_id))
            if r:
                stored.append(r["id"])
        for e in evidence.values():  # repair lessons: what failed and the SQL that fixed it
            if e.status == "ok" and len(e.attempts) > 1:
                err = e.attempts[-2]["error"][:160]
                lesson = f"Lesson (SQL repair on {', '.join(e.tables)}): error '{err}' was fixed by: {e.sql[:300]}"
                r = self._safe(tr, lambda: mem.remember(lesson, tags=["lesson", "sql_repair"], source=f"agent:executor:{run_id}", confidence=0.8, layer="long_term", run_id=run_id))
                if r:
                    stored.append(r["id"])
        tr.emit("Memory", "hmem.remember", output_summary=f"stored {len(stored)} memories ({', '.join(stored)})", data={"ids": stored})

    @staticmethod
    def _safe(tr, fn):
        try:
            return fn()
        except Exception as e:
            tr.emit("Memory", "meterless_error", output_summary=str(e)[:300], status="error")
            return None


def render_markdown(ans: FinalAnswer, evidence: dict[str, Evidence], source_label: str) -> str:
    lines = [f"## {ans.headline}", "", ans.answer, "", f"**Recommendation:** {ans.recommendation}", "", "### Evidence-backed claims"]
    for i, c in enumerate(ans.claims, 1):
        cites = "; ".join(f"[{x.qid} · {x.sql_hash} · {x.table}]" for x in c.citations)
        lines.append(f"{i}. {c.text} {cites}")
    if ans.caveats:
        lines += ["", "### Caveats"] + [f"- {c}" for c in ans.caveats]
    lines += ["", f"_Confidence {ans.confidence:.2f}. Data source: {source_label}._"]
    return "\n".join(lines)
