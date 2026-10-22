#!/usr/bin/env python3
"""Run the sample questions against LIVE Zetaris and record real query results (+ the LLM responses) for SAMPLE_MODE.

  DATA_SOURCE=zetaris RECORD_FIXTURES=true python scripts/record_fixtures.py [example_2 example_3]

Writes fixtures/recorded/catalog.json, fixtures/recorded/queries/<sql-hash>.json and fixtures/recorded/llm/<q-hash>.json.
Only results from a live Zetaris connection can be recorded (RecordingDataSource refuses anything else).
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["DATA_SOURCE"] = "zetaris"
os.environ["RECORD_FIXTURES"] = "true"
os.environ.pop("SAMPLE_MODE", None)

from whybot.config import Settings  # noqa: E402
from whybot.orchestrator import WhyBot  # noqa: E402

s = Settings.from_env()
only = set(sys.argv[1:])  # e.g. example_2 example_3 (default: all, in order)
for p in sorted((ROOT / "input_examples").glob("*.json")):
    if only and p.stem not in only:
        continue
    ex = json.loads(p.read_text())
    res = WhyBot(s).run(ex["input"]["query"], run_id=f"recorded-{ex['run_id']}")
    out = ROOT / "output_examples" / p.name
    out.write_text(json.dumps(res, indent=1, default=str))
    print(p.name, res["status"], res.get("execution_time_seconds"), "->", out)
