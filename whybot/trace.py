"""JSONL trace: one file per run in logs/<run_id>.jsonl, mirrored to stdout and to live subscribers (SSE)."""
from __future__ import annotations

import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .config import redact


def _scrub(obj: Any) -> Any:
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_scrub(v) for v in obj]
    return obj


class Tracer:
    def __init__(self, run_id: str, logs_dir: Path, echo: bool = True):
        self.run_id = run_id
        self.path = Path(logs_dir) / f"{run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.echo = echo
        self.seq = 0
        self.iteration = 0
        self.events: list[dict] = []
        self._subs: list[Callable[[dict], None]] = []
        self._lock = threading.Lock()
        self._t0 = time.time()

    def subscribe(self, fn: Callable[[dict], None]) -> None:
        self._subs.append(fn)

    def emit(self, agent: str, action: str, *, input_summary: str = "", output_summary: str = "",
             target_agent: str | None = None, status: str = "success", retry_count: int = 0,
             confidence: float | None = None, data: dict | None = None) -> dict:
        with self._lock:
            self.seq += 1
            ev = {
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
                "elapsed_s": round(time.time() - self._t0, 2),
                "run_id": self.run_id,
                "seq": self.seq,
                "iteration": self.iteration,
                "agent_name": agent,
                "action": action,
                "input_summary": input_summary[:600],
                "output_summary": output_summary[:1200],
                "target_agent": target_agent,
                "status": status,
                "retry_count": retry_count,
                "confidence": confidence,
                "data": data or {},
            }
            ev = _scrub(ev)
            self.events.append(ev)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(ev, default=str) + "\n")
        if self.echo:
            lvl = "WARN" if status not in ("success", "info") else "INFO"
            arrow = f" -> {target_agent}" if target_agent else ""
            print(f"[{lvl}] run_id={self.run_id} it={self.iteration} {agent}.{action}{arrow}: {ev['output_summary'][:160]}",
                  file=sys.stderr, flush=True)
        for fn in list(self._subs):
            try:
                fn(ev)
            except Exception:
                pass
        return ev
