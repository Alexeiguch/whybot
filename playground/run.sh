#!/usr/bin/env bash
# Start the WhyBot playground (UI on :7860, WhyBot API on :8000) in SAMPLE_MODE. Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
if [[ -x .venv/bin/python ]]; then PY=.venv/bin/python; fi
if ! "$PY" -c "import gradio, fastapi" 2>/dev/null; then
  echo "[playground] installing Python deps (first run only)…"
  [[ -x .venv/bin/python ]] || { python3 -m venv .venv; PY=.venv/bin/python; }
  "$PY" -m pip install -q -r playground/requirements.txt
fi
if [[ ! -x meterless_bridge/node_modules/.bin/tsx ]] && command -v npm >/dev/null; then
  (cd meterless_bridge && npm ci --no-audit --no-fund --loglevel=error) || echo "[playground] Meterless bridge not installed; runs continue without memory"
fi
export SAMPLE_MODE="${SAMPLE_MODE:-true}" GRADIO_ANALYTICS_ENABLED=False
export GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-0.0.0.0}" GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7860}"
export WHYBOT_API_HOST="${WHYBOT_API_HOST:-0.0.0.0}" WHYBOT_API_PORT="${WHYBOT_API_PORT:-8000}"
echo "[playground] UI  -> http://localhost:${GRADIO_SERVER_PORT}   (WhyBot API + built-in UI -> http://localhost:${WHYBOT_API_PORT})"
exec "$PY" -m playground.app
