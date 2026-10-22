"""WhyBot orchestrator: Planner -> SQL Executor -> Analyst -> Critic -> (retry | write | escalate) -> Writer.

Control flow is driven by the Critic's verdicts plus deterministic number checks (checks.decide_next), capped at
MAX_ITERATIONS (<=3) and a wall-clock budget. SQL failures trigger schema introspection and query repair; a query that
still fails marks its hypothesis untestable for that iteration (fallback) instead of crashing the run.
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .agents import Analyst, Critic, Planner, SQLAgent, Writer
from .checks import check_claim_numbers, decide_next, drop_uncited, result_sanity, tables_in_sql, validate_final
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
                tr.emit(agent, "llm_call", output_summary=f"{ev['model']} {ev['latency_s']}s tokens={ev.get('completion_tokens')}",
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
        if self.s.sample_mode and (os.environ.get("SAMPLE_LLM") == "replay" or not os.environ.get("NVIDIA_API_KEY")):
            tr.emit("Orchestrator", "llm_mode", output_summary=f"SAMPLE_MODE: replaying recorded LLM responses from {cpath.name}", status="info")
            return CassetteLLM(cpath, on_call=on_call)
        live = NvidiaLLM(self.s, on_call=on_call)
        if self.s.record_fixtures:
            self._cassette = CassetteLLM(cpath, inner=live, on_call=on_call)
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

    def _make_meterless(self, tr: Tracer):
        if self._meterless is not None or not self.use_meterless:
            return self._meterless
        try:
            from .memory import Meterless
            return Meterless(self.s.meterless_data_dir)
        except Exception as e:
            tr.emit("Memory", "meterless_unavailable", output_summary=str(e)[:300], status="error")
            return None

    # ------------------------------------------------------------------ main entry
    def run(self, question: str, run_id: str | None = None, subscribe=None) -> dict:
        run_id = run_id or new_run_id()
        self._question, self._cassette = question, None
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
            ds = self._make_ds(tr)
            tr.emit("Executor", "datasource", output_summary=ds.label, data={"kind": ds.kind, "dialect": ds.dialect})
            mem = self._make_meterless(tr)
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
        tr.emit("Orchestrator", "finish", output_summary=f"status={result['status']} in {result['execution_time_seconds']}s",
                status="success" if result["status"] != "error" else "error")
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
        evidence: dict[str, Evidence] = {}
        carryover, feedback = "", ""
        claims = ClaimSet(claims=[])
        report: CriticReport | None = None
        checks: dict = {}
        decision = None
        markov_steps: list[str] = []
        qcount = 0

        for it in range(1, self.s.max_iterations + 1):
            tr.iteration = it
            if time.time() > deadline:
                tr.emit("Orchestrator", "timeout", output_summary="run budget exhausted before iteration", status="error")
                break
            # ---- Plan
            done = [f"{e.qid} [{e.status}] ({e.hypothesis_id}) {e.purpose[:140]}" for e in evidence.values()]
            plan, meta = planner.plan(question, catalog, self.notes, memory_text, carryover, feedback, done, it)
            for h in plan.hypotheses:
                hypotheses.setdefault(h.id, h)
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

            ok_ev = [e for e in evidence.values() if e.status == "ok"]
            if not ok_ev:
                feedback = "Every query failed. Use simpler queries and only columns from the catalog."
                decision_txt = "retry: no evidence"
                tr.emit("Orchestrator", "branch", output_summary=decision_txt, status="retry", target_agent="Planner")
                carryover = self._carry(mem, tr, it, "No successful queries.", "all", "simplify", carryover)
                continue

            # ---- Analyse
            try:
                claims, meta = analyst.draft(question, plan_text, ok_ev, feedback)
                tr.emit("Analyst", "draft_claims", target_agent="Critic", output_summary=f"{len(claims.claims)} claims",
                        data={"claims": claims.model_dump()["claims"], "llm": meta})
            except LLMError as e:
                tr.emit("Analyst", "failed", output_summary=str(e)[:300], status="error")
                claims = ClaimSet(claims=[])

            # ---- Critique: deterministic number checks, then the LLM critic
            checks = {c.cid: check_claim_numbers(c, evidence) for c in claims.claims}
            for cid, ch in checks.items():
                tr.emit("Critic", "number_check", output_summary=f"{cid}: {'PASS' if ch.ok else 'REJECT ' + '; '.join(ch.problems)}"[:400],
                        status="success" if ch.ok else "rejected", data={"cid": cid, "ok": ch.ok, "problems": ch.problems})
            report = None
            if claims.claims and time.time() < deadline:
                try:
                    report, meta = critic.review(question, plan_text, claims, ok_ev, checks, tables, it, self.s.max_iterations)
                except LLMError as e:
                    tr.emit("Critic", "failed", output_summary=str(e)[:300], status="error")
            if report is None:
                report = self._fallback_report(claims, checks)
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

            decision = decide_next(it, self.s.max_iterations, report, checks, evidence, out_of_time=time.time() > deadline)
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

        final = self._write(question, status, decision, claims, report, evidence, writer, tr)
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
            "hypotheses": [{**h.model_dump(), "status": next((x.status for x in (report.hypothesis_status if report else []) if x.hypothesis_id == h.id), "untested")} for h in hypotheses.values()],
            "evidence": [e.to_dict() for e in evidence.values()],
            "iterations": tr.iteration,
            "data_source": {"label": ds.label, "kind": ds.kind},
            "memory": {"recalled": mem_hits, "meterless": bool(mem)},
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
                draft.sql = guard_sql(draft.sql)  # the exact SQL that runs is the SQL that is hashed and cited
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
            if smells and attempt < self.s.max_sql_repairs:
                err = "RESULT SANITY CHECK FAILED: " + "; ".join(smells)
                failed.append({"sql": draft.sql, "error": err})
                ev.attempts.append({"sql": draft.sql, "sql_hash": h, "error": err})
                tr.emit("Executor", "result_rejected", output_summary=f"{qid}: {err}", status="rejected", retry_count=attempt,
                        target_agent="Executor", data={"qid": qid, "sql_hash": h, "rows_head": res.rows[:5], "columns": res.columns})
                continue
            ev.sql, ev.sql_hash, ev.tables = draft.sql, h, used
            ev.columns, ev.rows, ev.row_count = res.columns, res.rows, len(res.rows)
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

    def _recall(self, mem, question, tr) -> tuple[str, list]:
        if not mem:
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
        return "\n".join(f"- ({h['relevance']}) {h['content']}" for h in hits), hits

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

    def _fallback_report(self, claims: ClaimSet, checks) -> CriticReport:
        from .schemas import ClaimVerdict
        return CriticReport(
            verdicts=[ClaimVerdict(cid=c.cid, verdict="weak" if checks[c.cid].ok else "unsupported", strength=0.3,
                                   reason="deterministic check only") for c in claims.claims],
            hypothesis_status=[], decision="insufficient_evidence", feedback="Critic unavailable; only number checks were applied.")

    def _write(self, question, status, decision, claims, report, evidence, writer, tr) -> FinalAnswer:
        sup_ids = set(decision.supported_cids)
        verdicts = {v.cid: v for v in (report.verdicts if report else [])}
        supported = [{**c.model_dump(), "critic": verdicts[c.cid].reason} for c in claims.claims if c.cid in sup_ids]
        rejected = [{"cid": c.cid, "statement": c.statement, "verdict": verdicts[c.cid].verdict if c.cid in verdicts else "unchecked",
                     "reason": verdicts[c.cid].reason if c.cid in verdicts else ""} for c in claims.claims if c.cid not in sup_ids]
        allowed = {cv.qid for c in claims.claims if c.cid in sup_ids for cv in c.cited_values}
        if not allowed:  # escalation with nothing supported: may cite any successful query as context
            allowed = {e.qid for e in evidence.values() if e.status == "ok"}
        hstat = [h.model_dump() for h in (report.hypothesis_status if report else [])]

        def validator(ans: FinalAnswer) -> list[str]:
            probs = validate_final(ans, evidence, allowed)
            if status == "insufficient_evidence" and probs == ["no claims"]:
                return []
            return probs
        try:
            ans, meta = writer.write(question, status, supported, rejected, list(evidence.values()), hstat, validator)
            tr.emit("Writer", "final_answer", output_summary=ans.headline, confidence=ans.confidence,
                    data={"answer": ans.model_dump(), "llm": meta})
            return ans
        except LLMError as e:
            tr.emit("Writer", "validation_failed", output_summary=str(e)[:300], status="error")
        # Deterministic fallback: render the verified claims directly, with their citations.
        from .schemas import Citation, FinalClaim
        fc = [FinalClaim(text=c["statement"], citations=[Citation(qid=q, sql_hash=evidence[q].sql_hash, table=evidence[q].primary_table)
                                                          for q in sorted({cv["qid"] for cv in c["cited_values"]})]) for c in supported]
        ans = FinalAnswer(headline=("Insufficient evidence" if status != "answered" else "Verified findings") + f" for: {question}",
                          answer="The Writer could not produce a valid cited answer; these are the critic-verified claims verbatim.",
                          recommendation="Review the verified claims and evidence table.", claims=fc,
                          caveats=["Writer fallback: deterministic rendering of verified claims"], confidence=0.3)
        ans = drop_uncited(ans, evidence, allowed)
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
        lines.append(f"{i}. {c.text} — {cites}")
    if ans.caveats:
        lines += ["", "### Caveats"] + [f"- {c}" for c in ans.caveats]
    lines += ["", f"_Confidence {ans.confidence:.2f}. Data source: {source_label}._"]
    return "\n".join(lines)
