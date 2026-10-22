#!/usr/bin/env python3
"""Add the live Zetaris query errors of a committed recorded trace to fixtures/recorded/queries.

Recordings made before RecordingDataSource stored failures kept only the successful results, so a replay answered
the failed SQL with REPLAY MISS and the SQL agent's repair prompts no longer matched the live run (prompt_match).
The error text is copied verbatim from the trace's query_failed events (which carry the full Zetaris message);
local guard errors (allowlist, forbidden SQL) are skipped because the replay raises them itself.

  python scripts/backfill_recorded_errors.py logs/recorded-example-2-snow-saturday.jsonl [...]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QDIR = ROOT / "fixtures" / "recorded" / "queries"

for trace in sys.argv[1:]:
    sql_by_hash, added = {}, 0
    for line in Path(trace).read_text().splitlines():
        e = json.loads(line)
        d = e.get("data") or {}
        if e["action"] == "sql_written":
            sql_by_hash[d["sql_hash"]] = d["sql"]
        elif e["action"] == "query_failed" and d.get("error", "").startswith("Zetaris error"):
            h = d["sql_hash"]
            out = QDIR / f"{h.split(':')[1]}.json"
            if out.exists():
                continue
            out.write_text(json.dumps({"source": "zetaris", "sql": sql_by_hash[h], "sql_hash": h, "recorded_at": e["timestamp"],
                                       "error": d["error"], "label": "LIVE Zetaris Lightning (JDBC)",
                                       "backfilled_from": str(Path(trace).resolve().relative_to(ROOT)), "trace_seq": e["seq"]}, indent=1))
            added += 1
    print(trace, "added", added)
