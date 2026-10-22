"""Run ids name trace files, so they are validated in one place for the CLI, the API, the playground and the Tracer.

- Format: a letter or digit, then letters, digits, '.', '_' or '-' (max 81 chars), never '..'.
- The `recorded-` prefix is reserved for the committed live recordings (scripts/record_fixtures.py).
- An existing trace is never overwritten; only a recording run may replace its own recorded-* trace.
"""
from __future__ import annotations

import re
from pathlib import Path

RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$")
RESERVED_PREFIX = "recorded-"


class RunIdError(ValueError):
    pass


def validate_run_id(run_id: str, *, allow_recorded: bool = False) -> str:
    if not isinstance(run_id, str) or not RUN_ID_RE.match(run_id) or ".." in run_id:
        raise RunIdError("invalid run_id: use letters, digits, '.', '_' or '-' (max 81 chars, starting with a letter or digit)")
    if run_id.lower().startswith(RESERVED_PREFIX) and not allow_recorded:
        raise RunIdError(f"run_id prefix '{RESERVED_PREFIX}' is reserved for the committed live recordings")
    return run_id


def trace_exists(run_id: str, logs_dir: Path | str) -> bool:
    d = Path(logs_dir)
    return (d / f"{run_id}.jsonl").exists() or (d / f"{run_id}.result.json").exists()
