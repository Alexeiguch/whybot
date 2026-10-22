"""Playground smoke tests (SAMPLE_MODE, offline, no keys). Run: python -m pytest playground/tests -q

The end-to-end replay test runs automatically once the recorded fixtures (fixtures/recorded, including llm/) are committed;
until then it is skipped and the missing-fixtures path is tested instead.
"""
import json
import os
import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from playground import app, core  # noqa: E402
from playground.live_llm import KeyedNvidiaLLM  # noqa: E402
from whybot.config import Settings  # noqa: E402
from whybot.llm import NvidiaLLM  # noqa: E402

EX1 = json.loads((ROOT / "input_examples" / "example_1.json").read_text())["input"]["query"]
REPLAY = "Replay (recorded real run, no keys)"
LIVE = "Live (your NVIDIA key)"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _last(gen):
    out = None
    n = 0
    for out in gen:
        n += 1
    return out, n


@pytest.fixture(autouse=True)
def _offline(monkeypatch, tmp_path):
    for k in ("NVIDIA_API_KEY", "WHYBOT_API_URL"):
        monkeypatch.delenv(k, raising=False)
    # set-but-empty wins over any .env.local on the machine, so tests never reach a live Zetaris or a real key
    for k in ("NVIDIA_API_KEY", "ZETARIS_USERNAME", "ZETARIS_PASSWORD", "ZETARIS_JDBC_JAR"):
        monkeypatch.setenv(k, "")
    monkeypatch.setenv("SAMPLE_LLM", "replay")
    monkeypatch.setenv("SAMPLE_MODE", "true")
    monkeypatch.setenv("METERLESS_ENABLED", "false")
    monkeypatch.setenv("LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(core, "TRACE_DIR", tmp_path / "traces")


def test_ui_builds():
    demo = app.build()
    assert demo is not None
    assert app.CUSTOM in app._choices()


def test_sample_questions_come_from_input_examples():
    assert EX1 in core.sample_questions()


def test_missing_fixtures_shows_friendly_message(monkeypatch, tmp_path):
    monkeypatch.setenv("FIXTURES_DIR", str(tmp_path / "nothing-here"))
    (banner, *_rest, answer, trace), _ = _last(app.investigate(EX1, "", REPLAY, "", 0))
    assert "not in this checkout yet" in answer
    assert trace is None
    assert "picked up automatically" in app.fixture_md()


def test_custom_question_without_recording_is_explained(monkeypatch, tmp_path):
    fx = tmp_path / "fx"
    (fx / "queries").mkdir(parents=True)
    (fx / "llm").mkdir()
    (fx / "catalog.json").write_text('{"source": "zetaris", "catalog": []}')
    (fx / "queries" / "a.json").write_text("{}")
    (fx / "llm" / "x.json").write_text(json.dumps({"question": "some other recorded question", "calls": []}))
    monkeypatch.setenv("FIXTURES_DIR", str(fx))
    *_rest, answer, _trace = _last(app.investigate(app.CUSTOM, "Why did trips fall on 3 Mar 2025?", REPLAY, "", 0))[0]
    assert "only has recordings for the sample questions" in answer


def test_live_without_key_is_refused_politely():
    *_rest, answer, _trace = _last(app.investigate(EX1, "", LIVE, "", 0))[0]
    assert "Paste your NVIDIA API key" in answer


def test_live_path_streams_labels_and_scrubs_the_key(monkeypatch, tmp_path):
    """Live plumbing with a stand-in LLM that echoes the key in its error: the key must never reach the UI or trace."""
    import playground.live_llm as live_llm
    from whybot.llm import BaseLLM, LLMError

    secret = "nvapi-SMOKE-not-a-real-key-123456"

    class EchoKeyLLM(BaseLLM):
        def __init__(self, settings, api_key, on_call=None):
            super().__init__(on_call)
            self.k = api_key

        def chat(self, role, messages, **kw):
            raise LLMError(f"upstream rejected key {self.k}")

    monkeypatch.setattr(live_llm, "KeyedNvidiaLLM", EchoKeyLLM)
    fx = tmp_path / "fx"
    (fx / "queries").mkdir(parents=True)
    (fx / "catalog.json").write_text(json.dumps({"source": "zetaris", "recorded_at": "2026-10-22T12:00:00Z",
                                                 "catalog": [{"table": "whybot.yellow_daily", "columns": [["trip_date", ""]], "note": ""}]}))
    monkeypatch.setenv("FIXTURES_DIR", str(fx))
    monkeypatch.setenv("RUN_TIMEOUT_S", "20")
    final, frames = _last(app.investigate(EX1, "", LIVE, secret, 0))
    banner, timeline, *_mid, answer, trace = final
    assert "LIVE" in banner and "RECORDED Zetaris results" in banner, "Live without ZETARIS_* says it uses recorded data, and why"
    blob = json.dumps(final, default=str) + Path(trace).read_text()
    assert secret not in blob
    assert Path(trace).name.endswith(".LIVE.jsonl")


def test_keyed_llm_keeps_key_out_of_env_and_matches_parent(monkeypatch):
    s = Settings()
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-parent-dummy-0000")
    parent = NvidiaLLM(s)
    monkeypatch.delenv("NVIDIA_API_KEY")
    child = KeyedNvidiaLLM(s, "nvapi-visitor-dummy-1111")
    assert set(vars(parent)) == set(vars(child)), "NvidiaLLM gained attributes: update playground/live_llm.py"
    assert "NVIDIA_API_KEY" not in os.environ
    assert "visitor" not in repr(child)


def test_render_highlights_reject_and_retry_and_writes_labelled_trace():
    v = core.RunView(mode="replay", question=EX1, run_id="pg-test")
    evs = [
        {"agent_name": "Orchestrator", "action": "llm_mode", "status": "info", "iteration": 0, "elapsed_s": 0,
         "data": {"mode": "replay", "live_run_id": "recorded-x", "recorded_at": "2026-10-22T12:00:00Z"}},
        {"agent_name": "Executor", "action": "datasource", "status": "success", "output_summary": "RECORDED zetaris results (replay)"},
        {"agent_name": "Planner", "action": "plan", "status": "success", "iteration": 1, "data": {"plan": {"metric": "trips",
         "hypotheses": [{"id": "H1", "kind": "holiday", "statement": "Thanksgiving"}]}}},
        {"agent_name": "Executor", "action": "sql_written", "iteration": 1, "data": {"qid": "Q1", "sql": "SELECT 1", "sql_hash": "sha256:ab", "tables": ["whybot.yellow_daily"]}},
        {"agent_name": "Executor", "action": "query_ok", "status": "success", "iteration": 1, "output_summary": "Q1: 7 rows",
         "data": {"qid": "Q1", "hypothesis_id": "H1", "sql_hash": "sha256:ab", "tables": ["whybot.yellow_daily"], "columns": ["trips"],
                  "rows_head": [[77244]], "row_count": 1, "source": "RECORDED zetaris results"}},
        {"agent_name": "Critic", "action": "number_check", "status": "rejected", "output_summary": "C3: FAIL"},
        {"agent_name": "Critic", "action": "verdict", "status": "rejected", "iteration": 1, "output_summary": "C3 unsupported", "data": {"verdict": "unsupported"}},
        {"agent_name": "Orchestrator", "action": "branch", "status": "retry", "iteration": 1, "output_summary": "retry: follow-ups"},
    ]
    for e in evs:
        v.add(e)
    assert "RECORDED" in v.banner() and "recorded-x" in v.banner()
    assert "REJECT" in v.critic() and "RETRY" in v.critic() and "RETRY" in v.timeline()
    assert "whybot.yellow_daily" in v.sql_steps()
    rows = v.evidence_rows()
    assert rows[0][0] == "Q1" and "RECORDED" in rows[0][6]
    p = Path(v.write_trace())
    assert p.name.endswith("RECORDED-replay.jsonl") and len(p.read_text().splitlines()) == len(evs)


def test_replay_end_to_end_through_the_api(monkeypatch):
    if not core.replay_status(EX1).get("question_ready"):
        pytest.skip("recorded fixtures (fixtures/recorded incl. llm/) are not committed yet")
    monkeypatch.setattr(core, "API_PORT", _free_port())
    final, frames = _last(app.investigate(EX1, "", REPLAY, "", 0))
    banner, timeline, hyps, sql, critic, evidence, answer, trace = final
    assert frames > 3, "the timeline should stream, not arrive in one frame"
    assert "RECORDED" in banner and "pg-banner rec" in banner
    assert "Planner" in timeline and "Critic" in timeline
    assert "<b>H" in hyps and "whybot." in sql
    assert "REJECT" in critic or "RETRY" in critic, "the recorded run shows the Critic rejecting or retrying"
    rows = evidence.get("value") if isinstance(evidence, dict) else evidence
    rows = rows.get("data", rows) if isinstance(rows, dict) else rows
    assert rows and any("RECORDED" in str(r[6]) for r in rows), "evidence rows carry RECORDED provenance"
    assert answer.startswith("### ") and "RECORDED replay" in answer and "[Q" in answer
    lines = Path(trace).read_text().splitlines()
    assert lines and json.loads(lines[-1])["action"] == "finish"
    assert "nvapi-" not in Path(trace).read_text()


def test_rejected_weak_verdict_shows_as_reject_and_timeline_shows_seq():
    v = core.RunView(mode="replay", question=EX1, run_id="pg-test")
    v.add({"agent_name": "Critic", "action": "verdict", "status": "rejected", "iteration": 1, "seq": 77,
           "output_summary": "c2 weak (0.35): average \u2014 no material reduction", "data": {"verdict": "weak"}})
    assert "REJECT · weak" in v.critic() and "WEAK" not in v.critic()
    assert "#77" in v.timeline()
    assert "\u2014" not in v.critic() + v.timeline()


def test_clicking_a_claim_reveals_its_cited_sql():
    v = core.RunView(mode="replay", question=EX1, run_id="pg-test")
    v.result = {"status": "answered", "iterations": 1, "execution_time_seconds": 1,
                "output": {"headline": "H", "answer": "A", "recommendation": "R", "confidence": 0.9,
                           "claims": [{"text": "Trips fell <45%>", "citations": [{"qid": "Q1", "sql_hash": "sha256:ab", "table": "whybot.yellow_daily"}]}]},
                "evidence": [{"qid": "Q1", "sql_hash": "sha256:ab", "sql": "SELECT trips FROM whybot.yellow_daily WHERE x < 3", "row_count": 4}]}
    md = v.answer()
    assert "<details class='claim'>" in md and "<summary>" in md
    assert "SELECT trips FROM whybot.yellow_daily WHERE x &lt; 3" in md
    assert "Trips fell &lt;45%&gt;" in md


def test_playground_refuses_reserved_or_malformed_run_ids():
    from whybot.runid import RunIdError
    for bad in ("recorded-example-1-thanksgiving", "../x"):
        with pytest.raises(RunIdError):
            next(core.stream_replay(EX1, bad))
        with pytest.raises(RunIdError):
            next(core.stream_live(EX1, bad, "dummy"))
    assert core.new_run_id("replay").startswith("pg-replay-")
