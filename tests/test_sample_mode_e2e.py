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


def test_at_least_one_recorded_run_has_a_critic_reject_then_retry():
    traces = sorted((ROOT / "logs").glob("recorded-*.jsonl"))
    if not traces:
        pytest.skip("no recorded traces")
    found = False
    for t in traces:
        ev = [json.loads(l) for l in t.read_text().splitlines()]
        rejected = any(e["agent_name"] == "Critic" and e["action"] == "verdict" and e["status"] == "rejected" for e in ev)
        retried = any(e["action"] == "branch" and e["status"] == "retry" for e in ev)
        found |= rejected and retried
    assert found
