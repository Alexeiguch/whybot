#!/usr/bin/env python3
"""Run the sample questions against LIVE Zetaris and record real query results (+ the LLM responses) for SAMPLE_MODE.

  DATA_SOURCE=zetaris RECORD_FIXTURES=true python scripts/record_fixtures.py [example_2 example_3]

A run that an NVIDIA 5xx or timeout ended in an error, or degraded into a fallback (whybot/rerun.py), is re-run from
the start, up to RECORD_5XX_RETRIES times (default 3), so a recording is always one complete live run. Each re-run
starts from the Meterless store as it was before the failed attempt, and the Meterless bridge is warmed up first.

Writes fixtures/recorded/catalog.json, fixtures/recorded/queries/<sql-hash>.json and fixtures/recorded/llm/<q-hash>.json.
Only results from a live Zetaris connection can be recorded (RecordingDataSource refuses anything else).
"""
import json
import os
import shutil
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
from whybot.rerun import read_trace, rerun_reason  # noqa: E402

s = Settings.from_env()
RETRIES = int(os.environ.get("RECORD_5XX_RETRIES", "3"))
QUERIES = s.fixtures_dir / "queries"


def warm_bridge():
    """Start and stop the Meterless bridge once so the run's own start does not pay the cold tsx compile."""
    try:
        from whybot.memory import Meterless
        Meterless(s.meterless_data_dir).close()
    except Exception as e:  # the run itself reports meterless_unavailable
        print("Meterless warm-up failed:", str(e)[:200], file=sys.stderr)


only = set(sys.argv[1:])  # e.g. example_2 example_3 (default: all, in order)
for p in sorted((ROOT / "input_examples").glob("*.json")):
    if only and p.stem not in only:
        continue
    ex = json.loads(p.read_text())
    snap = s.meterless_data_dir.with_name(s.meterless_data_dir.name + ".before-" + p.stem)
    for attempt in range(1, RETRIES + 2):
        shutil.rmtree(snap, ignore_errors=True)
        if s.meterless_data_dir.exists():
            shutil.copytree(s.meterless_data_dir, snap)
        before = set(QUERIES.glob("*.json"))
        warm_bridge()
        res = WhyBot(s).run(ex["input"]["query"], run_id=f"recorded-{ex['run_id']}")
        why = rerun_reason(res, read_trace(ROOT / res.get("log_file", "")))
        if not why or attempt > RETRIES:
            break
        print(p.name, f"attempt {attempt}: transient NVIDIA failure ({why}), re-running", file=sys.stderr)
        for f in set(QUERIES.glob("*.json")) - before:  # fixtures only the discarded attempt wrote
            f.unlink()
        shutil.rmtree(s.meterless_data_dir, ignore_errors=True)
        if snap.exists():
            shutil.copytree(snap, s.meterless_data_dir)
        time.sleep(30 * attempt)
    shutil.rmtree(snap, ignore_errors=True)
    out = ROOT / "output_examples" / p.name
    out.write_text(json.dumps(res, indent=1, default=str))
    print(p.name, res["status"], res.get("execution_time_seconds"), "->", out)
