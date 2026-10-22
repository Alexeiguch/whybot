# WhyBot HTTP API (stable contract)

The server is `python run.py --serve` (uvicorn on `0.0.0.0:$PORT`, default 8000). The web UI and the `playground/` client use only the endpoints below. These shapes are a stable contract: fields may be added, but none will be removed or renamed.

## Request body (`POST /run` and `POST /runs`)

```json
{
  "run_id": "optional-id",
  "input": {"query": "Why did NYC yellow taxi trips drop on Thursday 28 Nov 2024 compared with the surrounding weeks?"},
  "options": {"sample_mode": true}
}
```

- `run_id` is optional and must match `^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$`. A new id is generated when it is omitted.
- `input.query` is 5 to 500 characters.
- `options.sample_mode: true` replays a recorded live run (offline, no credentials). See "What SAMPLE_MODE is" in the README.
- `options.data_source` (`zetaris` | `replay` | `dev`) is ignored unless the server runs with `ALLOW_SOURCE_OVERRIDE=true`.
- Validation errors return **422**. When `MAX_CONCURRENT_RUNS` investigations are already running (default 2), the server returns **429**. Reusing the id of a run that is still in progress returns **409** on `/runs`.

## `POST /run` (blocking)

Returns the full result object (see below) when the investigation finishes. A live run takes minutes; a SAMPLE_MODE run takes seconds.

## `POST /runs` (async) + `GET /runs/{run_id}/events` (SSE)

`POST /runs` returns immediately:

```json
{"run_id": "abc", "events": "/runs/abc/events", "result": "/runs/abc"}
```

`GET /runs/{run_id}/events` is a `text/event-stream`:

- `event: trace`: `data` is one trace event (same shape as a line of `logs/<run_id>.jsonl`). The stream replays every event from the start of the run, so a late subscriber misses nothing.
- `event: result`: `data` is the final result object. The stream then ends.
- Lines that start with `:` are keepalive comments.

Trace event:

```json
{"timestamp": "2026-10-22T12:00:00.000Z", "elapsed_s": 12.3, "run_id": "abc", "seq": 17, "iteration": 1,
 "agent_name": "Planner|Executor|Analyst|Critic|Writer|Orchestrator|Memory", "action": "plan|sql_written|query_ok|query_failed|number_check|verdict|branch|llm_call|llm_mode|...",
 "input_summary": "", "output_summary": "", "target_agent": "Critic", "status": "success|error|retry|rejected|escalated|info|fallback",
 "retry_count": 0, "confidence": null, "data": {}}
```

Useful actions:

- `llm_mode` (`data.mode` = `live` | `replay`; on replay it also carries `live_run_id`, `recorded_at` and `trace`).
- `sql_written` and `query_ok` (`data.sql`, `data.sql_hash`).
- `verdict` (`data.cid`, `data.verdict`).
- `branch` (retry, write or escalate).
- `finish`.

## `GET /runs/{run_id}`

Returns `{"done": bool, "result": <result or null>, "events": <count>}`. It also works for finished runs found on disk as `logs/<run_id>.result.json`.

## Result object

```json
{
  "run_id": "abc", "trace_id": "...", "status": "answered|insufficient_evidence|error",
  "output": {"headline": "", "answer": "", "recommendation": "", "summary": "", "markdown": "",
             "claims": [{"text": "", "citations": [{"qid": "Q1", "sql_hash": "sha256:...", "table": "whybot.yellow_daily"}]}],
             "caveats": [], "confidence": 0.8},
  "hypotheses": [{"id": "H1", "kind": "holiday", "statement": "", "status": "supported|rejected|inconclusive|untested", "status_reason": ""}],
  "evidence": [{"qid": "Q1", "hypothesis_id": "H0", "sql": "", "sql_hash": "", "tables": [], "status": "ok|failed", "columns": [], "rows": [], "row_count": 0, "source_label": ""}],
  "iterations": 2, "data_source": {"label": "", "kind": "live|replay|dev"},
  "memory": {"recalled": 0, "meterless": true}, "data_quality_notes": [],
  "wall_clock": {"started_at": "", "finished_at": "", "seconds": 0}, "execution_time_seconds": 0,
  "log_file": "logs/abc.jsonl", "agents": [],
  "error": {"type": "", "message": ""}
}
```

`error` appears only when `status` is `error`.

## Other endpoints

- `GET /health`: `{"ok": true, "data_source": "...", "sample_mode": false}`.
- `GET /samples`: the three `input_examples/*.json` objects.
- `GET /`: the built-in web UI.
