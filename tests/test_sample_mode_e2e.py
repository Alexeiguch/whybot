"""End-to-end SAMPLE_MODE: replay each committed live recording fully offline and get the same, real Writer answer."""
import json
from pathlib import Path

import pytest

from whybot.config import ROOT, Settings
from whybot.orchestrator import WhyBot

EXAMPLES = sorted((ROOT / "input_examples").glob("example_*.json"))


@pytest.mark.parametrize("path", EXAMPLES, ids=[p.stem for p in EXAMPLES])
def test_sample_mode_replay_matches_live_run_with_real_writer_answer(path, tmp_path, monkeypatch):
    live_path = ROOT / "output_examples" / path.name
    if not live_path.exists():
        pytest.skip("recorded live run not committed")
    for k in ("NVIDIA_API_KEY", "SAMPLE_LLM", "ZETARIS_PASSWORD"):
        monkeypatch.delenv(k, raising=False)
    ex = json.loads(path.read_text())
    s = Settings()
    s.sample_mode, s.data_source = True, "replay"
    s.logs_dir = tmp_path
    s.fixtures_dir = ROOT / "fixtures" / "recorded"
    s.meterless_data_dir = tmp_path / ".meterless"
    res = WhyBot(s, use_meterless=False, echo=False).run(ex["input"]["query"], run_id="e2e-" + ex["run_id"])
    live = json.loads(live_path.read_text())

    events = [json.loads(l) for l in Path(res["log_file"]).read_text().splitlines()]
    assert not [e for e in events if e["action"] in ("fallback_answer", "validation_failed")], "replay must end in a real Writer answer"
    assert not any("cassette exhausted" in e["output_summary"] for e in events)
    assert res["status"] == live["status"] and res["status"] in ("answered", "insufficient_evidence")
    assert res["output"]["headline"] == live["output"]["headline"]
    assert res["output"]["answer"] == live["output"]["answer"]
    assert res["data_source"]["kind"] == "recorded"
    assert all("RECORDED zetaris" in (e.get("source_label") or "") for e in res["evidence"] if e["status"] == "ok")
    hashes = {e["sql_hash"] for e in res["evidence"]}
    assert all(c["sql_hash"] in hashes for cl in res["output"]["claims"] for c in cl["citations"])

    mode = next(e for e in events if e["action"] == "llm_mode")
    assert mode["data"]["mode"] == "replay" and mode["data"]["live_run_id"] == "recorded-" + ex["run_id"]
    calls = [e for e in events if e["action"] == "llm_call"]
    assert calls and all(c["data"].get("replayed") and c["data"].get("recorded_at") for c in calls)
    mismatched = [(c["agent_name"], c["seq"]) for c in calls if c["data"].get("prompt_match") is not True]
    assert not mismatched, f"replayed prompts differ from the live ones: {mismatched}"


def test_at_least_one_recorded_run_has_a_critic_reject_then_retry():
    traces = sorted((ROOT / "logs").glob("recorded-*.jsonl"))
    if len(traces) < len(EXAMPLES):
        pytest.skip("the full recorded set is not committed yet")
    found = False
    for t in traces:
        ev = [json.loads(l) for l in t.read_text().splitlines()]
        rejected = any(e["agent_name"] == "Critic" and e["action"] == "verdict" and e["status"] == "rejected" for e in ev)
        retried = any(e["action"] == "branch" and e["status"] == "retry" for e in ev)
        found |= rejected and retried
    assert found


def _replay(ex_name, tmp_path, monkeypatch, fixtures=None):
    for k in ("NVIDIA_API_KEY", "SAMPLE_LLM", "ZETARIS_PASSWORD", "REPLAY_STRICT"):
        monkeypatch.delenv(k, raising=False)
    ex = json.loads((ROOT / "input_examples" / ex_name).read_text())
    s = Settings()
    s.sample_mode, s.data_source = True, "replay"
    s.logs_dir = tmp_path
    s.fixtures_dir = fixtures or ROOT / "fixtures" / "recorded"
    s.meterless_data_dir = tmp_path / ".meterless"
    res = WhyBot(s, use_meterless=False, echo=False).run(ex["input"]["query"], run_id="t-" + ex["run_id"])
    return res, [json.loads(l) for l in Path(res["log_file"]).read_text().splitlines()]


def test_replay_reports_what_the_current_checks_would_add(tmp_path, monkeypatch):
    """Run #3 was recorded with checks v1. Its replay keeps v1 (so the prompts match) and reports, without acting on it,
    that v2 rejects the '~12%' (backed only by a zone id) and claim 4's 'higher than the other three Mondays'."""
    res, events = _replay("example_3.json", tmp_path, monkeypatch)
    shadow = [e for e in events if e["action"] == "shadow_checks"]
    assert len(shadow) == 1 and shadow[0]["status"] == "warning"
    found = " ".join(shadow[0]["data"]["findings"])
    assert "'12%'" in found and "higher than the other" in found and "5.53" in found
    assert res["status"] == "insufficient_evidence"


def test_replay_drift_is_a_visible_warning_and_strict_mode_fails(tmp_path, monkeypatch):
    import shutil
    fx = tmp_path / "fx"
    shutil.copytree(ROOT / "fixtures" / "recorded", fx)
    from whybot.llm import cassette_path
    ex = json.loads((ROOT / "input_examples" / "example_1.json").read_text())
    cp = cassette_path(fx, ex["input"]["query"])
    data = json.loads(cp.read_text())
    data["calls"][0]["prompt_sha"] = "0" * 16          # as if the Planner prompt had changed since recording
    cp.write_text(json.dumps(data))
    res, events = _replay("example_1.json", tmp_path / "a", monkeypatch, fx)
    drift = [e for e in events if e["action"] == "replay_drift"]
    assert len(drift) == 1 and drift[0]["status"] == "warning" and drift[0]["agent_name"] == "Planner"
    assert res["status"] == "answered"
    monkeypatch.setenv("REPLAY_STRICT", "true")
    ex = json.loads((ROOT / "input_examples" / "example_1.json").read_text())
    s = Settings()
    s.sample_mode, s.data_source, s.logs_dir, s.fixtures_dir = True, "replay", tmp_path / "b", fx
    s.meterless_data_dir = tmp_path / "b" / ".m"
    strict = WhyBot(s, use_meterless=False, echo=False).run(ex["input"]["query"], run_id="t-strict")
    assert strict["status"] == "error" and "REPLAY_STRICT" in strict["error"]["message"]


def test_new_recordings_store_the_checks_version(tmp_path):
    from whybot.checks import CHECKS_VERSION
    from whybot.llm import BaseLLM, CassetteLLM

    class Inner(BaseLLM):
        def model_for(self, role):
            return "fake"

        def chat(self, role, messages, *, max_tokens=4096, temperature=0.2):
            return "{}", {"model": "fake", "latency_s": 0.0}

    cas = CassetteLLM(tmp_path / "c.json", inner=Inner(), run_id="recorded-x")
    cas.chat("planner", [{"role": "user", "content": "q"}])
    cas.save("q")
    data = json.loads((tmp_path / "c.json").read_text())
    assert data["checks_version"] == CHECKS_VERSION and len(data["calls"]) == 1
    assert CassetteLLM(tmp_path / "c.json").header.get("checks_version") == CHECKS_VERSION


def test_replay_uses_the_table_notes_it_was_recorded_with(tmp_path, monkeypatch):
    """Editing whybot/catalog_notes.json must not make a committed recording drift: the replay sends the notes stored
    in fixtures/recorded/catalog.json."""
    import whybot.notes as notes
    edited = tmp_path / "catalog_notes.json"
    edited.write_text(json.dumps({"whybot.yellow_daily": "EDITED NOTE: join on trip_date"}))
    monkeypatch.setattr(notes, "NOTES_PATH", edited)
    res, events = _replay("example_1.json", tmp_path, monkeypatch)
    calls = [e for e in events if e["action"] == "llm_call"]
    assert calls and all((e.get("data") or {}).get("prompt_match") is True for e in calls)
    assert not [e for e in events if e["action"] == "replay_drift"]


@pytest.mark.parametrize("path", EXAMPLES, ids=[p.stem for p in EXAMPLES])
def test_every_committed_recording_replays_its_exact_prompts(path, tmp_path, monkeypatch):
    """Whatever checks version a recording was made with, its replay sends byte-identical prompts (no drift)."""
    if not (ROOT / "output_examples" / path.name).exists():
        pytest.skip("recorded live run not committed")
    res, events = _replay(path.name, tmp_path, monkeypatch)
    calls = [e for e in events if e["action"] == "llm_call"]
    assert calls and all((e.get("data") or {}).get("prompt_match") is True for e in calls)
    assert not [e for e in events if e["action"] == "replay_drift"]
