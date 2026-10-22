"""End-to-end SAMPLE_MODE: replay the committed live recording of the headline question, fully offline."""
import json
from pathlib import Path

import pytest

from whybot.config import ROOT, Settings
from whybot.orchestrator import WhyBot

EX1 = json.loads((ROOT / "input_examples" / "example_1.json").read_text())
LIVE = ROOT / "output_examples" / "example_1.json"


@pytest.mark.skipif(not LIVE.exists(), reason="recorded live run not committed")
def test_sample_mode_replays_headline_run_offline(tmp_path, monkeypatch):
    for k in ("NVIDIA_API_KEY", "SAMPLE_LLM", "ZETARIS_PASSWORD"):
        monkeypatch.delenv(k, raising=False)
    s = Settings()
    s.sample_mode, s.data_source = True, "replay"
    s.logs_dir = tmp_path
    s.fixtures_dir = ROOT / "fixtures" / "recorded"
    res = WhyBot(s, use_meterless=False, echo=False).run(EX1["input"]["query"], run_id="e2e-replay")
    live = json.loads(LIVE.read_text())

    assert res["status"] == live["status"] and res["status"] in ("answered", "insufficient_evidence")
    assert res["output"]["headline"] == live["output"]["headline"]
    assert res["data_source"]["kind"] == "replay"
    assert all("RECORDED zetaris" in (e.get("source_label") or "") for e in res["evidence"] if e["status"] == "ok")
    # every cited sql hash exists among the recorded queries
    hashes = {e["sql_hash"] for e in res["evidence"]}
    assert all(c["sql_hash"] in hashes for cl in res["output"]["claims"] for c in cl["citations"])

    events = [json.loads(l) for l in Path(res["log_file"]).read_text().splitlines()]
    mode = next(e for e in events if e["action"] == "llm_mode")
    assert mode["data"]["mode"] == "replay" and mode["data"]["live_run_id"] == "recorded-" + EX1["run_id"]
    calls = [e for e in events if e["action"] == "llm_call"]
    assert calls and all(c["data"].get("replayed") and c["data"].get("recorded_at") for c in calls)
    assert any(e["status"] == "rejected" for e in events), "the recorded run shows the Critic rejecting something"
