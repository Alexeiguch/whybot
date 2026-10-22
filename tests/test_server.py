from fastapi.testclient import TestClient

import whybot.server as srv


class StubBot:
    def __init__(self, s):
        self.s = s

    def run(self, q, run_id=None, subscribe=None):
        ev = {"agent_name": "Planner", "action": "plan", "status": "success", "output_summary": "x", "elapsed_s": 0, "iteration": 1, "data": {}}
        if subscribe:
            subscribe(ev)
        return {"run_id": run_id or "r1", "status": "answered", "question": q, "source": self.s.data_source}


def test_run_endpoints(monkeypatch):
    monkeypatch.setattr(srv, "WhyBot", StubBot)
    c = TestClient(srv.app)
    assert c.get("/health").json()["ok"]
    assert len(c.get("/samples").json()) == 3
    assert "WhyBot" in c.get("/").text
    r = c.post("/run", json={"input": {"query": "why?"}, "options": {"sample_mode": True}}).json()
    assert r["status"] == "answered" and r["source"] == "replay"
    rid = c.post("/runs", json={"run_id": "abc", "input": {"query": "why?"}}).json()["run_id"]
    body = c.get(f"/runs/{rid}/events").text
    assert "event: trace" in body and "event: result" in body
