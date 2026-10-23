"""When the recorder re-runs a live investigation from the start: an NVIDIA failure that is not the run's fault.

A 5xx or a timeout (the client's APITimeoutError, the watchdog, a gateway timeout) is transient. It warrants a re-run
when it ended the run in an error, or when the run degraded because of it (an agent fell back: deterministic critic
verdicts, a Writer fallback answer). A transient error that a per-call retry absorbed does not.
"""
from __future__ import annotations

import json
import re

NVIDIA_TRANSIENT = re.compile(
    r"\b5\d\d\b|Internal Server Error|Bad Gateway|Service Unavailable|Gateway Time-?out"
    r"|APITimeoutError|Request timed out|watchdog timeout|\btimed out\b|ReadTimeout|ConnectTimeout", re.I)
DEGRADED = {"fallback_verdicts", "fallback_answer", "validation_failed", "sql_generation_failed"}


def rerun_reason(result: dict, events: list[dict]) -> str | None:
    """Why this recorded run should be re-run, or None to keep it."""
    msg = json.dumps(result.get("error") or "", default=str)
    if result.get("status") == "error" and NVIDIA_TRANSIENT.search(msg):
        return f"run error: {msg[:160]}"
    transient = [e for e in events if e.get("agent_name") != "Memory" and e.get("status") == "error"
                 and NVIDIA_TRANSIENT.search(e.get("output_summary") or "")]
    degraded = [e for e in events if e.get("action") in DEGRADED or e.get("status") == "fallback"]
    if transient and degraded:
        t = transient[0]
        return f"{t.get('agent_name')} {t.get('action')}: {(t.get('output_summary') or '')[:120]} -> {degraded[0].get('action')}"
    return None


def read_trace(path) -> list[dict]:
    try:
        return [json.loads(l) for l in open(path) if l.strip()]
    except OSError:
        return []
