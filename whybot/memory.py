"""Meterless cognition layer via the Node JSON-lines bridge (meterless_bridge/bridge.ts).

H-MEM       -> past investigations + lessons, recalled by the Planner (and SQL agent) at the start of each run.
World Model -> every evidence fact asserted with provenance (agent, run id, SQL hash, source table).
Markovian   -> bounded carryover between loop iterations; finished investigation recorded as a Markovian run.
"""
from __future__ import annotations

import itertools
import json
from .config import SECRET_ENV_NAMES
import os
import shutil
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path

from .config import ROOT

BRIDGE_DIR = ROOT / "meterless_bridge"


class MeterlessError(RuntimeError):
    pass


class Meterless:
    def __init__(self, data_dir: Path, timeout_s: float = 20.0):
        tsx = BRIDGE_DIR / "node_modules" / ".bin" / "tsx"
        if not tsx.exists():
            raise MeterlessError("Meterless bridge not installed: run `npm install` in meterless_bridge/")
        if not shutil.which("node"):
            raise MeterlessError("node is not on PATH")
        data_dir.mkdir(parents=True, exist_ok=True)
        env = {k: v for k, v in os.environ.items() if k not in SECRET_ENV_NAMES}  # the bridge needs no credentials
        env["METERLESS_DATA_DIR"] = str(data_dir)
        log_dir = BRIDGE_DIR.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        self._errlog = open(log_dir / "meterless-bridge.log", "a")
        self.p = subprocess.Popen([str(tsx), "bridge.ts"], cwd=BRIDGE_DIR, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=self._errlog, text=True, env=env)
        self.ids = itertools.count(1)
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        ready = self._read()
        if not ready.get("result", {}).get("ready"):
            raise MeterlessError(f"bridge failed to start: {ready}")

    def _read(self) -> dict:
        out: dict = {}

        def rd():
            line = self.p.stdout.readline()
            out["line"] = line
        t = threading.Thread(target=rd, daemon=True)
        t.start()
        t.join(self.timeout_s)
        if "line" not in out or not out["line"]:
            raise MeterlessError("bridge timed out or exited")
        return json.loads(out["line"])

    def call(self, method: str, **params):
        with self._lock:
            i = next(self.ids)
            self.p.stdin.write(json.dumps({"id": i, "method": method, "params": params}, default=str) + "\n")
            self.p.stdin.flush()
            resp = self._read()
            while resp.get("id") is not None and resp.get("id") < i:  # a late answer to a call that timed out earlier
                resp = self._read()
        if resp.get("id") != i:
            raise MeterlessError(f"{method}: bridge answered request {resp.get('id')} instead of {i}")
        if "error" in resp:
            raise MeterlessError(f"{method}: {resp['error']}")
        return resp["result"]

    # ---- H-MEM
    def recall(self, query: str, threshold: float, top_n: int = 5) -> dict:
        return self.call("hmem.query", query=query, opts={"topN": top_n, "threshold": threshold})

    def remember(self, content: str, *, tags: list[str], source: str, confidence: float = 0.7,
                 layer: str = "working", mtype: str = "factual", run_id: str | None = None) -> dict:
        return self.call("hmem.add", content=content, type=mtype, layer=layer, tags=tags, confidence=confidence,
                         source=source, goalRunId=run_id)

    # ---- World Model
    def assert_evidence(self, *, run_id: str, agent: str, about: str, predicate: str, value, table: str,
                        sql_hash: str, confidence: float, kind: str = "pipeline") -> dict:
        return self.call("wm.assertFact", about=about, predicate=predicate, value=value, confidence=confidence,
                         source={"kind": kind, "by": agent, "at": _now(), "runId": run_id,
                                 "url": f"zetaris://{table}", "checksum": sql_hash})

    def upsert_entity(self, *, etype: str, key: str, attrs: dict, run_id: str, agent: str) -> dict:
        return self.call("wm.upsertEntity", type=etype, externalKey={"system": "whybot", "id": key}, attrs=attrs,
                         source={"kind": "agent", "by": agent, "at": _now(), "runId": run_id})

    def relate(self, *, src: str, rel: str, dst: str, run_id: str, agent: str) -> dict:
        return self.call("wm.relate", **{"from": src, "type": rel, "to": dst,
                                         "source": {"kind": "agent", "by": agent, "at": _now(), "runId": run_id}})

    def about(self, entity_id: str) -> list:
        return self.call("wm.about", id=entity_id)

    # ---- Markovian
    def carry(self, chunk_text: str, previous: str, carryover_tokens: int = 512) -> dict:
        return self.call("markovian.carry", chunkText=chunk_text, previousCarryover=previous, carryoverTokens=carryover_tokens)

    def record_run(self, goal: str, steps: list[str]) -> dict:
        return self.call("markovian.record", goal=goal, steps=steps)

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
