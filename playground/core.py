"""Playground logic with no UI: fixture discovery, the two run paths, and rendering of the trace into panels.

Replay  -> the WhyBot HTTP API (POST /runs with options.sample_mode=true, then the SSE stream GET /runs/{id}/events),
           served by an embedded uvicorn running whybot.server:app, or by WHYBOT_API_URL if set.
Live    -> whybot.orchestrator.WhyBot in-process with the visitor's key injected as the LLM (the HTTP API reads the key
           from the server environment only, and a shared playground must never put a visitor's key there).
Both paths emit the same trace events (docs/api.md), so one renderer serves both. No agent logic lives here.
"""
from __future__ import annotations

import html
import re
import json
import os
import queue
import socket
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx

from whybot.config import ROOT, Settings
from whybot.llm import cassette_path
from whybot.runid import validate_run_id

API_HOST = os.environ.get("WHYBOT_API_HOST", "127.0.0.1")
API_PORT = int(os.environ.get("WHYBOT_API_PORT", "8000"))
TRACE_DIR = Path(os.environ.get("PLAYGROUND_TRACE_DIR", Path(tempfile.gettempdir()) / "whybot-playground-traces"))
LIVE_SLOTS = threading.BoundedSemaphore(int(os.environ.get("PLAYGROUND_MAX_LIVE_RUNS", "2")))

# ---------------------------------------------------------------- fixtures (re-read on every call: new commits show up)


def fixtures_dir() -> Path:
    return Path(Settings.from_env().fixtures_dir)


def recorded_questions(fdir: Path | None = None) -> dict[str, dict]:
    """question -> cassette header, for every recorded LLM cassette under fixtures/recorded/llm."""
    fdir = fdir or fixtures_dir()
    out = {}
    for p in sorted((fdir / "llm").glob("*.json")):
        try:
            data = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        q = (data.get("question") or "").strip()
        if q:
            out[q] = {k: v for k, v in data.items() if k != "calls"}
    return out


def sample_questions() -> list[str]:
    """The committed sample questions (input_examples/*.json), then any other recorded question."""
    qs = []
    for p in sorted((ROOT / "input_examples").glob("*.json")):
        try:
            qs.append(json.loads(p.read_text())["input"]["query"].strip())
        except (OSError, KeyError, json.JSONDecodeError):
            continue
    for q in recorded_questions():
        if q not in qs:
            qs.append(q)
    return qs


def replay_status(question: str | None = None) -> dict:
    fdir = fixtures_dir()
    cat = fdir / "catalog.json"
    n_queries = len(list((fdir / "queries").glob("*.json"))) if (fdir / "queries").is_dir() else 0
    rec = recorded_questions(fdir)
    st = {"fixtures_dir": str(fdir), "catalog": cat.is_file(), "queries": n_queries, "recorded": rec,
          "ready": cat.is_file() and n_queries > 0 and bool(rec)}
    if question is not None:
        st["question_ready"] = st["ready"] and cassette_path(fdir, question).is_file()
    return st


def friendly_missing(question: str, st: dict) -> str:
    if not st["catalog"] or not st["queries"]:
        return ("**Recorded runs are not in this checkout yet.** The recorded Zetaris results and Nemotron responses "
                "(`fixtures/recorded/`) are being re-recorded from live Zetaris and will appear on `main` shortly. "
                "Pull the latest `main` (or rebuild the Codespace) and the playground picks them up automatically, "
                "with no restart. Meanwhile, **Live** mode works with your own NVIDIA key if Zetaris is configured.")
    if not st["recorded"]:
        return ("**Recorded Zetaris results are here, but the recorded Nemotron responses (`fixtures/recorded/llm/`) "
                "are not committed yet.** They are being re-recorded; pull `main` and try again (no restart needed). "
                "You can already run **Live** with your own NVIDIA key against the recorded Zetaris results.")
    return ("**Replay only has recordings for the sample questions.** This exact question was never run live, so there "
            "is nothing to replay. Pick a sample question marked *recorded*, or switch to **Live** with your NVIDIA key.")


def zetaris_configured() -> tuple[bool, str]:
    missing = [n for n in ("ZETARIS_USERNAME", "ZETARIS_PASSWORD", "ZETARIS_JDBC_JAR") if not os.environ.get(n)]
    if missing:
        return False, f"{', '.join(missing)} not set on this playground, so data comes from RECORDED Zetaris results"
    if not Path(os.environ["ZETARIS_JDBC_JAR"]).is_file():
        return False, "ZETARIS_JDBC_JAR does not point to a file, so data comes from RECORDED Zetaris results"
    return True, "ZETARIS_* is configured: queries run on LIVE Zetaris"


# ---------------------------------------------------------------- embedded API server

_server_lock = threading.Lock()
_server_thread: threading.Thread | None = None


def api_url() -> str:
    return os.environ.get("WHYBOT_API_URL", "").rstrip("/") or f"http://127.0.0.1:{API_PORT}"


def _healthy(url: str) -> bool:
    try:
        return httpx.get(f"{url}/health", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


def ensure_api(timeout_s: float = 30) -> str:
    """Use WHYBOT_API_URL if set; otherwise start whybot.server:app in this process (once) and wait for /health."""
    global _server_thread
    url = api_url()
    if os.environ.get("WHYBOT_API_URL") or _healthy(url):
        return url
    with _server_lock:
        if _server_thread is None or not _server_thread.is_alive():
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", API_PORT)) == 0:
                    raise RuntimeError(f"port {API_PORT} is taken by something that is not WhyBot; set WHYBOT_API_PORT")
            import uvicorn
            from whybot.server import app
            server = uvicorn.Server(uvicorn.Config(app, host=API_HOST, port=API_PORT, log_level="warning"))
            _server_thread = threading.Thread(target=server.run, name="whybot-api", daemon=True)
            _server_thread.start()
    t_end = time.time() + timeout_s
    while time.time() < t_end:
        if _healthy(url):
            return url
        time.sleep(0.2)
    raise RuntimeError(f"WhyBot API did not come up on {url}")


# ---------------------------------------------------------------- run paths (both yield ("trace", ev) / ("result", res))


def new_run_id(mode: str) -> str:
    return datetime.now(timezone.utc).strftime(f"pg-{mode}-%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:6]


def stream_replay(question: str, run_id: str):
    """SAMPLE_MODE through the public API: POST /runs, then follow the SSE timeline until the result event."""
    validate_run_id(run_id)
    url = ensure_api()
    body = {"run_id": run_id, "input": {"query": question}, "options": {"sample_mode": True}}
    r = httpx.post(f"{url}/runs", json=body, timeout=30)
    if r.status_code == 429:
        raise RuntimeError("Too many investigations are running on this playground; try again in a minute.")
    if r.status_code == 422:
        raise RuntimeError("The question must be 5 to 500 characters.")
    if r.status_code in (400, 409):
        raise RuntimeError(f"Run id refused: {r.json().get('detail', '')}")
    r.raise_for_status()
    events_path = r.json()["events"]
    with httpx.stream("GET", f"{url}{events_path}", timeout=httpx.Timeout(30, read=1200)) as resp:
        resp.raise_for_status()
        kind, data = None, []
        for line in resp.iter_lines():
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                kind = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].lstrip())
            elif line == "" and kind:
                payload = json.loads("\n".join(data)) if data else None
                yield kind, payload
                if kind == "result":
                    return
                kind, data = None, []


def stream_live(question: str, run_id: str, api_key: str):
    """Live Nemotron with the visitor's key, in-process through whybot.orchestrator.WhyBot."""
    from whybot.orchestrator import WhyBot
    from .live_llm import KeyedNvidiaLLM

    validate_run_id(run_id)
    key = (api_key or "").strip()
    if not key:
        raise RuntimeError("Paste your NVIDIA API key (build.nvidia.com) to run Live, or switch to Replay.")
    if not LIVE_SLOTS.acquire(timeout=5):
        raise RuntimeError("Too many live investigations are running on this playground; try again shortly.")
    s = Settings.from_env()
    s.sample_mode, s.record_fixtures = False, False  # a playground run never records or touches recorded-* traces
    zet_ok, reason = zetaris_configured()
    s.data_source = "zetaris" if zet_ok else "replay"
    if not zet_ok and not (Path(s.fixtures_dir) / "catalog.json").is_file():
        LIVE_SLOTS.release()
        raise RuntimeError("Live needs data: Zetaris is not configured here and the recorded Zetaris results "
                           "(fixtures/recorded) are not in this checkout yet. Pull main and retry.")
    q: queue.Queue = queue.Queue()

    def scrub(obj):
        if isinstance(obj, str):
            return obj.replace(key, "<NVIDIA_API_KEY:redacted>") if key in obj else obj
        if isinstance(obj, dict):
            return {k: scrub(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [scrub(v) for v in obj]
        return obj

    def work():
        llm = None
        try:
            llm = KeyedNvidiaLLM(s, key)
            q.put(("trace", {"agent_name": "Orchestrator", "action": "playground_mode", "status": "info", "seq": 0,
                             "iteration": 0, "elapsed_s": 0, "output_summary": f"LIVE NVIDIA Nemotron (your key). {reason}",
                             "data": {"mode": "live", "data_source": s.data_source, "reason": reason}}))
            res = WhyBot(s, llm=llm).run(question, run_id=run_id, subscribe=lambda ev: q.put(("trace", scrub(ev))))
        except Exception as e:
            res = {"status": "error", "error": {"type": type(e).__name__, "message": scrub(str(e))[:300]}, "run_id": run_id}
        finally:
            del llm
            LIVE_SLOTS.release()
        q.put(("result", scrub(res)))

    threading.Thread(target=work, name=f"live-{run_id}", daemon=True).start()
    while True:
        item = q.get()
        yield item
        if item[0] == "result":
            return


# ---------------------------------------------------------------- state + rendering

STATUS_CLASS = {"rejected": "bad", "error": "bad", "retry": "retry", "escalated": "warn", "fallback": "warn",
                "success": "ok", "info": "muted"}


@dataclass
class RunView:
    mode: str                    # replay | live
    question: str
    run_id: str
    events: list[dict] = field(default_factory=list)
    result: dict | None = None
    notice: str = ""

    def add(self, ev: dict):
        self.events.append(ev)

    # ---- labelling
    def banner(self) -> str:
        mode_ev = next((e for e in self.events if e.get("action") in ("llm_mode", "playground_mode")), None)
        ds_ev = next((e for e in reversed(self.events) if e.get("action") == "datasource"), None)
        fb = [e for e in self.events if e.get("action") in ("fallback", "datasource_unavailable") and e.get("agent_name") == "Executor"
              and "qid" not in (e.get("data") or {})]
        ds_label = (ds_ev or {}).get("output_summary", "")
        if self.mode == "replay":
            d = (mode_ev or {}).get("data", {})
            head = "RECORDED · replay of a real live run"
            sub = (f"Live run <code>{_e(d.get('live_run_id', '?'))}</code> recorded {_e(d.get('recorded_at', '?'))}. "
                   "LLM and SQL outputs come from that run; the checks, branch policy and citation validator re-execute now. "
                   "No keys used.")
            cls = "rec"
        else:
            reason = ((mode_ev or {}).get("data") or {}).get("reason", "")
            head = "LIVE · NVIDIA Nemotron with your key"
            sub = _e(reason)
            cls = "live" if "LIVE Zetaris" in ds_label else "mixed"
        data_line = f"<div class='ds'>Data source: <b>{_e(ds_label or 'starting…')}</b></div>" if ds_label or self.events else ""
        fb_line = "".join(f"<div class='ds warnline'>⚠ {_e(e.get('output_summary', ''))}</div>" for e in fb)
        return f"<div class='pg-banner {cls}'><div class='head'>{head}</div><div class='sub'>{sub}</div>{data_line}{fb_line}</div>"

    # ---- panels
    def timeline(self) -> str:
        rows = []
        for e in self.events:
            st = e.get("status", "")
            cls = STATUS_CLASS.get(st, "")
            act = e.get("action", "")
            if act in ("llm_call", "world_model.assert_fact", "meterless_store"):
                cls += " dim"
            tag = {"rejected": "✕ REJECTED", "error": "✕ ERROR", "retry": "↻ RETRY", "escalated": "⚠ ESCALATED",
                   "fallback": "⚠ FALLBACK"}.get(st, "")
            pcls = STATUS_CLASS.get(st, "")
            if act == "verdict":
                tag, pcls = _verdict_tag(e)
                cls = {"ok": "", "warn": "warn", "bad": "bad"}[pcls]
            if act == "branch" and st == "retry":
                tag = "↻ RETRY → Planner"
            pill = f"<span class='pill {pcls}'>{_e(tag)}</span>" if tag else ""
            tgt = f" → {_e(e['target_agent'])}" if e.get("target_agent") else ""
            rows.append(f"<tr class='{cls}'><td class='t'><b>#{_e(e.get('seq', ''))}</b><br>{e.get('elapsed_s', '')}s<br>it{e.get('iteration', 0)}</td>"
                        f"<td><span class='ag ag-{_e(e.get('agent_name', ''))}'>{_e(e.get('agent_name', ''))}</span></td>"
                        f"<td><div class='act'>{_e(act)}{tgt} {pill}</div><div class='sum'>{_e(str(e.get('output_summary', ''))[:320])}</div></td></tr>")
        if not rows:
            return "<div class='empty'>Pick a question and press <b>Investigate</b>. The agent timeline streams here.</div>"
        return f"<div class='tl' role='log' aria-label='Agent timeline'><table>{''.join(reversed(rows))}</table></div><div class='hint'>newest first · {len(rows)} events</div>"

    def hypotheses(self) -> str:
        plans = [e for e in self.events if e.get("action") == "plan"]
        final = {h.get("id"): h for h in ((self.result or {}).get("hypotheses") or [])}
        if not plans and not final:
            return "<div class='empty'>The Planner's hypotheses appear here.</div>"
        out = []
        for p in plans:
            plan = (p.get("data") or {}).get("plan") or {}
            out.append(f"<h4>Iteration {p.get('iteration')} plan · metric <code>{_e(plan.get('metric', ''))}</code> · target "
                       f"{_e(plan.get('target_window', ''))} · baseline {_e(plan.get('baseline_window', ''))}</h4><ul>")
            for h in plan.get("hypotheses") or []:
                fh = final.get(h.get("id"), {})
                st = fh.get("status", "")
                pill = f"<span class='pill {_hyp_cls(st)}'>{_e(HYP_ICON.get(st, '') + st.upper())}</span>" if st else ""
                out.append(f"<li><b>{_e(h.get('id'))}</b> <i>{_e(h.get('kind', ''))}</i> {pill} {_e(h.get('statement', ''))}"
                           f"<div class='small'>supports if: {_e(h.get('support_if', ''))} · refutes if: {_e(h.get('refute_if', ''))}</div>"
                           f"{'<div class=small>' + _e(fh.get('status_reason', '')) + '</div>' if fh.get('status_reason') else ''}</li>")
            out.append("</ul>")
        return "<div class='pg-panel'>" + "".join(out) + "</div>"

    def sql_steps(self) -> str:
        by: dict[str, dict] = {}
        for e in self.events:
            d = e.get("data") or {}
            qid = d.get("qid")
            act = e.get("action")
            if not qid or act not in ("sql_written", "query_ok", "query_failed", "result_rejected", "data_quality_check", "fallback"):
                continue
            s = by.setdefault(qid, {"attempts": [], "status": "running", "tables": [], "hyp": d.get("hypothesis_id", ""), "it": e.get("iteration")})
            if act == "sql_written":
                s["attempts"].append({"sql": d.get("sql", ""), "hash": d.get("sql_hash", ""), "tables": d.get("tables", []), "outcome": "…"})
                s["tables"] = d.get("tables", []) or s["tables"]
            elif act == "data_quality_check":
                s["attempts"].append({"sql": d.get("sql", ""), "hash": d.get("sql_hash", ""), "tables": d.get("tables", []),
                                      "outcome": e.get("output_summary", "")})
                s["status"], s["hyp"] = "data-quality check", "Critic"
            elif s["attempts"] or act == "fallback":
                if not s["attempts"]:
                    s["attempts"].append({"sql": "", "hash": "", "tables": [], "outcome": ""})
                s["attempts"][-1]["outcome"] = e.get("output_summary", "")
                s["status"] = {"query_ok": "ok", "query_failed": "error", "result_rejected": "rejected", "fallback": "failed"}[act]
                if act == "query_ok":
                    s["hyp"] = d.get("hypothesis_id", s["hyp"])
                    s["source"] = d.get("source", "")
        if not by:
            return "<div class='empty'>Each SQL step, the table it reads and every repair attempt appear here.</div>"
        out = []
        for qid, s in by.items():
            out.append(f"<div class='step'><div class='sh'><code>{_e(qid)}</code> · {_e(s['hyp'])} · iteration {s['it']} · "
                       f"tables <code>{_e(', '.join(s['tables']))}</code> <span class='pill {_q_cls(s['status'])}'>{_e(Q_ICON.get(s['status'], '') + s['status'].upper())}</span></div>")
            for i, a in enumerate(s["attempts"]):
                lab = "repair " + str(i) if i else "attempt"
                out.append(f"<div class='att'><div class='small'>{lab} · <code>{_e(a['hash'])}</code></div><pre>{_e(a['sql'])}</pre>"
                           f"<div class='small {'badtxt' if 'error' in a['outcome'].lower() or 'FAILED' in a['outcome'] else ''}'>{_e(a['outcome'][:400])}</div></div>")
            out.append("</div>")
        return "".join(out)

    def critic(self) -> str:
        out = []
        for e in self.events:
            act, d, st = e.get("action"), e.get("data") or {}, e.get("status")
            if act == "number_check":
                cls = "ok" if st == "success" else "bad"
                lab = "✓ PASS" if st == "success" else "✕ REJECT"
                out.append(f"<div class='cv {cls}'><span class='pill {cls}'>{lab}</span> number check · {_e(e.get('output_summary', ''))}</div>")
            elif act == "verdict":
                lab, cls = _verdict_tag(e)
                out.append(f"<div class='cv {cls}'><span class='pill {cls}'>{lab}</span> it{e.get('iteration')} · {_e(e.get('output_summary', ''))}</div>")
            elif act == "data_quality_check":
                out.append(f"<div class='cv warn'><span class='pill warn'>⚠ DATA QUALITY</span> {_e(e.get('output_summary', ''))}</div>")
            elif act == "branch":
                cls = {"retry": "retry", "escalated": "warn"}.get(st, "ok")
                lab = {"retry": "↻ RETRY → Planner", "escalated": "⚠ ESCALATE"}.get(st, "✓ ACCEPT → Writer")
                out.append(f"<div class='cv {cls} branch'><span class='pill {cls}'>{lab}</span> it{e.get('iteration')} · {_e(e.get('output_summary', ''))}</div>")
            elif e.get("agent_name") == "Writer" and st in ("retry", "rejected", "error", "fallback"):
                out.append(f"<div class='cv warn'><span class='pill {"retry" if st == "retry" else "warn"}'>{"↻" if st == "retry" else "⚠"} WRITER {_e(st.upper())}</span> {_e(e.get('output_summary', ''))}</div>")
        return "".join(out) or "<div class='empty'>Critic verdicts, number checks, REJECTs and retries appear here.</div>"

    def evidence_rows(self) -> list[list]:
        evs = (self.result or {}).get("evidence")
        if evs is None:
            evs = [{"qid": d.get("qid"), "hypothesis_id": d.get("hypothesis_id"), "status": "ok", "tables": d.get("tables", []),
                    "sql_hash": d.get("sql_hash"), "row_count": d.get("row_count"), "source_label": d.get("source"),
                    "recorded_at": None, "columns": d.get("columns", []), "rows": d.get("rows_head", [])}
                   for d in (e.get("data") or {} for e in self.events if e.get("action") == "query_ok")]
        out = []
        for ev in evs:
            preview = "; ".join(", ".join(f"{c}={v}" for c, v in zip(ev.get("columns") or [], r)) for r in (ev.get("rows") or [])[:3])
            out.append([ev.get("qid"), ev.get("hypothesis_id"), ev.get("status"), ", ".join(ev.get("tables") or []),
                        ev.get("sql_hash"), ev.get("row_count"), ev.get("source_label") or "", ev.get("recorded_at") or "",
                        preview[:300]])
        return out

    def answer(self) -> str:
        if self.notice:
            return self.notice
        res = self.result
        if not res:
            return "_The cited answer appears here when the Writer finishes._" if self.events else ""
        if res.get("status") == "error":
            err = res.get("error") or {}
            msg = err.get("message", "")
            if "cassette" in msg:
                return friendly_missing(self.question, replay_status(self.question))
            return f"**Run failed** ({_e(err.get('type', ''))}): {_e(msg)}"
        o = res.get("output") or {}
        label = "RECORDED replay" if self.mode == "replay" else "LIVE run"
        lines = [f"### {o.get('headline', '')}", "",
                 f"`{res.get('status')}` · confidence {o.get('confidence')} · {res.get('iterations')} iteration(s) · "
                 f"{res.get('execution_time_seconds')} s · **{label}** · data: {(res.get('data_source') or {}).get('label', '')}", "",
                 o.get("answer", ""), "", f"**Recommendation:** {o.get('recommendation', '')}", "", "**Claims and their evidence**"]
        lines.append("<div class='claims'>" + "".join(self._claim_html(i, c, res) for i, c in enumerate(o.get("claims") or [], 1)) + "</div>")
        rej = [h for h in res.get("hypotheses") or [] if h.get("status") == "rejected"]
        if rej:
            lines += ["", "**Rejected hypotheses**"] + [f"- **{h.get('id')}** {h.get('statement', '')} _{h.get('status_reason', '')}_" for h in rej]
        if o.get("caveats"):
            lines += ["", "**Caveats**"] + [f"- {c}" for c in o["caveats"]]
        if res.get("data_quality_notes"):
            lines += ["", "**Data-quality notes** (not evidence)"] + [f"- {n}" for n in res["data_quality_notes"]]
        return _nd("\n".join(lines))

    @staticmethod
    def _claim_html(i: int, c: dict, res: dict) -> str:
        """One claim as a click-to-expand card: the summary is the claim and its citations, the body is the cited SQL."""
        by_hash = {ev.get("sql_hash"): ev for ev in res.get("evidence") or []}
        cites = " ".join(f"<code>[{_e(x.get('qid'))} · {_e(str(x.get('sql_hash', ''))[:23])} · {_e(x.get('table'))}]</code>"
                         for x in c.get("citations") or [])
        body = []
        for x in c.get("citations") or []:
            ev = by_hash.get(x.get("sql_hash")) or {}
            body.append(f"<div class='small'>{_e(x.get('qid'))} · {_e(ev.get('row_count', '?'))} rows · {_e(ev.get('source_label', ''))}</div>"
                        f"<pre>{_e(ev.get('sql') or 'SQL not in this result')}</pre>")
        return (f"<details class='claim'><summary><b>{i}.</b> {_e(c.get('text', ''))} {cites}"
                f"<span class='show-sql'>show SQL</span></summary>{''.join(body)}</details>")

    def write_trace(self) -> str | None:
        if not self.events:
            return None
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        p = TRACE_DIR / f"{self.run_id}.{'RECORDED-replay' if self.mode == 'replay' else 'LIVE'}.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for e in self.events:
                f.write(json.dumps(e, default=str) + "\n")
        return str(p)


def _nd(x: str) -> str:
    """Model text sometimes contains em dashes; the UI shows a comma instead."""
    return re.sub(r"\s*\u2014\s*", ", ", x)


def _e(x) -> str:
    return html.escape(_nd(str(x if x is not None else "")))


VERDICT_TAG = {"supported": ("✓ ACCEPT", "ok"), "weak": ("◐ WEAK", "warn")}


def _verdict_tag(e: dict) -> tuple[str, str]:
    """The orchestrator's status decides REJECT; a 'weak' claim it rejected (so the loop retries) shows as REJECT."""
    v = (e.get("data") or {}).get("verdict", "")
    if e.get("status") == "rejected":
        return ("✕ REJECT · " + v if v and v != "rejected" else "✕ REJECT"), "bad"
    return VERDICT_TAG.get(v, ("✕ REJECT", "bad"))


HYP_ICON = {"supported": "✓ ", "rejected": "✕ ", "inconclusive": "◐ ", "untested": "○ "}
Q_ICON = {"ok": "✓ ", "error": "✕ ", "rejected": "✕ ", "failed": "✕ ", "running": "… "}


def _hyp_cls(st: str) -> str:
    return {"supported": "ok", "rejected": "bad", "inconclusive": "warn", "untested": "muted"}.get(st, "muted")


def _q_cls(st: str) -> str:
    return {"ok": "ok", "error": "bad", "rejected": "bad", "failed": "bad", "running": "muted"}.get(st, "warn")
