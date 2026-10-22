#!/usr/bin/env python3
"""WhyBot entry point.

  python run.py "Why did yellow taxi trips drop on 13 Feb 2024?"   # run one investigation, print JSON result
  python run.py --serve                                              # start the API + web UI on 0.0.0.0:$PORT
  python run.py --example input_examples/example_1.json
Options: --source zetaris|replay|dev  --sample (SAMPLE_MODE replay)  --markdown (print the cited answer only)
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description="WhyBot: explain why a metric changed, with cited evidence")
    ap.add_argument("question", nargs="?")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--example")
    ap.add_argument("--source", choices=["zetaris", "replay", "dev"])
    ap.add_argument("--sample", action="store_true", help="SAMPLE_MODE: replay recorded Zetaris results")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--run-id")
    a = ap.parse_args()

    if a.sample:
        os.environ["SAMPLE_MODE"] = "true"
    if a.source:
        os.environ["DATA_SOURCE"] = a.source
    if a.serve or (not a.question and not a.example):
        import uvicorn
        port = int(os.getenv("PORT", "8000"))
        uvicorn.run("whybot.server:app", host="0.0.0.0", port=port)
        return 0

    from whybot.config import Settings
    from whybot.orchestrator import WhyBot

    question, run_id = a.question, a.run_id
    if a.example:
        ex = json.load(open(a.example))
        question = ex["input"]["query"]
        if not run_id and ex.get("run_id"):  # a fresh id per run, so re-running an example never hits an existing trace
            run_id = f"{ex['run_id']}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
        if ex.get("options", {}).get("sample_mode"):
            os.environ["SAMPLE_MODE"] = "true"
    s = Settings.from_env()
    if s.data_source == "dev":
        print("[WARN] DATA_SOURCE=dev: DEV-ONLY DuckDB sample, NOT Zetaris. Do not use for the demo.", file=sys.stderr)
    from whybot.runid import RunIdError

    try:
        res = WhyBot(s).run(question, run_id=run_id)
    except RunIdError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if a.markdown and res.get("output"):
        print(res["output"]["markdown"])
    else:
        print(json.dumps(res, indent=1, default=str))
    return 0 if res["status"] != "error" else 1


if __name__ == "__main__":
    sys.exit(main())
