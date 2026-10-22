"""FastAPI app: POST /run (sync JSON), POST /runs (async) + GET /runs/{id}/events (SSE timeline), GET / (web UI)."""
from __future__ import annotations

import asyncio
import json
import os
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from .config import ROOT, Settings
from .orchestrator import WhyBot, new_run_id

app = FastAPI(title="WhyBot", version="0.1.0", description="Explains why a metric changed, with cited evidence.")
STATIC = Path(__file__).parent / "static"
RUNS: dict[str, dict] = {}


RUN_SLOTS = threading.BoundedSemaphore(int(os.environ.get("MAX_CONCURRENT_RUNS", "2")))  # LLM + Zetaris budget guard


class RunInput(BaseModel):
    query: str = Field(min_length=5, max_length=500)


class RunOptions(BaseModel):
    sample_mode: bool | None = None
    data_source: str | None = None


class RunRequest(BaseModel):
    run_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$")
    input: RunInput
    options: RunOptions = RunOptions()


def _settings(opts: RunOptions) -> Settings:
    s = Settings.from_env()
    if opts.sample_mode:
        s.sample_mode, s.data_source = True, "replay"
    elif opts.data_source in ("zetaris", "replay", "dev") and os.environ.get("ALLOW_SOURCE_OVERRIDE", "").lower() == "true":
        s.data_source = opts.data_source  # off by default: a public endpoint cannot switch itself onto the dev source
    return s


@app.get("/health")
def health():
    s = Settings.from_env()
    return {"ok": True, "data_source": s.data_source, "sample_mode": s.sample_mode}


@app.get("/samples")
def samples():
    out = []
    for p in sorted((ROOT / "input_examples").glob("*.json")):
        out.append(json.loads(p.read_text()))
    return out


@app.post("/run")
def run_sync(req: RunRequest):
    """Blocking run: takes the hackathon JSON shape and returns the full structured result."""
    if not RUN_SLOTS.acquire(timeout=5):
        raise HTTPException(429, "too many investigations running; retry shortly")
    try:
        return WhyBot(_settings(req.options)).run(req.input.query, run_id=req.run_id)
    finally:
        RUN_SLOTS.release()


@app.post("/runs")
def run_async(req: RunRequest):
    run_id = req.run_id or new_run_id()
    if run_id in RUNS and not RUNS[run_id]["done"]:
        raise HTTPException(409, "a run with this id is in progress")
    if not RUN_SLOTS.acquire(timeout=5):
        raise HTTPException(429, "too many investigations running; retry shortly")
    state = {"events": [], "result": None, "done": False, "cond": threading.Condition()}
    RUNS[run_id] = state

    def on_event(ev):
        with state["cond"]:
            state["events"].append(ev)
            state["cond"].notify_all()

    def work():
        try:
            res = WhyBot(_settings(req.options)).run(req.input.query, run_id=run_id, subscribe=on_event)
        except Exception as e:  # never leave the SSE stream hanging
            res = {"status": "error", "error": {"type": type(e).__name__, "message": str(e)[:300]}, "run_id": run_id}
        finally:
            RUN_SLOTS.release()
        with state["cond"]:
            state["result"], state["done"] = res, True
            state["cond"].notify_all()

    threading.Thread(target=work, daemon=True).start()
    return {"run_id": run_id, "events": f"/runs/{run_id}/events", "result": f"/runs/{run_id}"}


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    st = RUNS.get(run_id)
    if st:
        return {"done": st["done"], "result": st["result"], "events": len(st["events"])}
    p = Path(Settings.from_env().logs_dir) / f"{run_id}.result.json"
    if p.exists():
        return {"done": True, "result": json.loads(p.read_text())}
    raise HTTPException(404, "unknown run")


@app.get("/runs/{run_id}/events")
async def events(run_id: str):
    st = RUNS.get(run_id)
    if not st:
        raise HTTPException(404, "unknown run")

    async def gen():
        i = 0
        while True:
            evs, done = st["events"][i:], st["done"]
            for ev in evs:
                yield f"event: trace\ndata: {json.dumps(ev, default=str)}\n\n"
            i += len(evs)
            if done and i >= len(st["events"]):
                yield f"event: result\ndata: {json.dumps(st['result'], default=str)}\n\n"
                return
            await asyncio.sleep(0.4)
            yield ": keepalive\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
