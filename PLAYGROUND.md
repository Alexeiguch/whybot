# WhyBot Playground

Try WhyBot in a couple of minutes with no setup and no keys.

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/Alexeiguch/whybot?quickstart=1)

Pick a sample question and press **Investigate**. You can watch the agents work through it: the Planner's hypotheses, each SQL step with the table it reads, Critic verdicts with **REJECT** and **RETRY** highlighted, the evidence table with provenance, and the final answer, where every claim cites its query id, SQL hash and source table. You can download the full JSONL trace of any run.

## Two modes, always labelled

| Mode | What runs | Needs |
|---|---|---|
| **Replay** (default, yellow **RECORDED** banner) | A recorded **real** live run: the Zetaris query results and Nemotron responses come from that run, and the orchestration re-executes now (SQL guard and hashing, number checks, branch policy, citation validator). It goes through the WhyBot HTTP API (`POST /runs` + SSE, see `docs/api.md`) in `SAMPLE_MODE`. | Nothing |
| **Live** (green or blue **LIVE** banner) | Real NVIDIA Nemotron calls with **your** key, pasted into a password field. The key stays in memory for that run only. It is never written to disk, logs, the trace or the environment. Queries go to **live Zetaris** if `ZETARIS_*` is configured. Otherwise they use the recorded Zetaris results, and the banner says why. | An NVIDIA API key from [build.nvidia.com](https://build.nvidia.com) |

Replay only has recordings for the sample questions. To ask your own question, use Live.

## Launch options

**1. GitHub Codespaces (zero install).** Click the badge above. The devcontainer installs Python 3.11, Node 20 (for the Meterless bridge) and a Java 17 JRE (for the Zetaris JDBC driver), then starts the playground in `SAMPLE_MODE`. Port **7860** (the playground) opens in your browser. Port **8000** serves the WhyBot API and the built-in UI. If the tab doesn't open, go to the **Ports** panel and open 7860.

**2. Docker (one command).**

```bash
docker compose -f playground/docker-compose.yml up --build
# open http://localhost:7860   (API + built-in UI: http://localhost:8000)
```

**3. Local Python.**

```bash
bash playground/run.sh          # or: make -C playground
# open http://localhost:7860
```

`run.sh` creates `.venv` on the first run, installs `playground/requirements.txt` (WhyBot's requirements plus Gradio) and the Meterless bridge, and starts the UI with an embedded WhyBot API server. To use a WhyBot server that is already running, set `WHYBOT_API_URL=http://host:8000`.

## Recorded runs

Replay reads `fixtures/recorded/` (`catalog.json`, `queries/`, `llm/`). These files are re-read on every run, so newly committed recordings show up after a `git pull` with no restart. If they are missing, Replay shows a short notice instead of a run until they arrive.

## Tests

```bash
make -C playground test         # = python -m pytest playground/tests -q
```

The smoke tests run fully offline. They cover the UI build, the missing-fixtures message, Live labelling and key scrubbing (using a stand-in LLM), trace rendering with REJECT and RETRY highlighting, and, once recordings are present, a full Replay through the HTTP API.

## Limits

- Replay covers the recorded sample questions only.
- In Live mode without `ZETARIS_*`, the SQL agent can only reuse recorded queries. A new question will mostly end in "insufficient evidence", and the run says so.
- A live run takes about 4 to 8 minutes. Two live runs can run at once (`PLAYGROUND_MAX_LIVE_RUNS`).
- The playground adds no agent logic. Everything comes from the `whybot` package and its API.
