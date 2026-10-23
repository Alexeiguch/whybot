#!/usr/bin/env python3
"""Run the sample questions against LIVE Zetaris and record real query results (+ the LLM responses) for SAMPLE_MODE.

  DATA_SOURCE=zetaris RECORD_FIXTURES=true python scripts/record_fixtures.py [example_2 example_3]

A run that ends in an error caused by an NVIDIA 5xx (after the client's own per-call retries) is re-run from the
start, up to RECORD_5XX_RETRIES times (default 3), so a recording is always one complete live run.

Writes fixtures/recorded/catalog.json, fixtures/recorded/queries/<sql-hash>.json and fixtures/recorded/llm/<q-hash>.json.
Only results from a live Zetaris connection can be recorded (RecordingDataSource refuses anything else).
"""
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["DATA_SOURCE"] = "zetaris"
os.environ["RECORD_FIXTURES"] = "true"
os.environ.pop("SAMPLE_MODE", None)

from whybot.config import Settings  # noqa: E402
from whybot.orchestrator import WhyBot  # noqa: E402

s = Settings.from_env()
RETRIES = int(os.environ.get("RECORD_5XX_RETRIES", "3"))
NVIDIA_5XX = re.compile(r"\b5\d\d\b|Internal Server Error|Bad Gateway|Service Unavailable|Gateway Time-?out", re.I)
only = set(sys.argv[1:])  # e.g. example_2 example_3 (default: all, in order)
for p in sorted((ROOT / "input_examples").glob("*.json")):
    if only and p.stem not in only:
        continue
    ex = json.loads(p.read_text())
    for attempt in range(1, RETRIES + 2):
        res = WhyBot(s).run(ex["input"]["query"], run_id=f"recorded-{ex['run_id']}")
        msg = json.dumps(res.get("error") or "", default=str)
        if res.get("status") != "error" or not NVIDIA_5XX.search(msg) or attempt > RETRIES:
            break
        print(p.name, f"attempt {attempt}: NVIDIA 5xx ({msg[:160]}), re-running", file=sys.stderr)
        time.sleep(30 * attempt)
    out = ROOT / "output_examples" / p.name
    out.write_text(json.dumps(res, indent=1, default=str))
    print(p.name, res["status"], res.get("execution_time_seconds"), "->", out)
