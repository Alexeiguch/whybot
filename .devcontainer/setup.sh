#!/usr/bin/env bash
# Codespaces / devcontainer setup: Python deps (WhyBot + Gradio) into .venv, and the Meterless Node bridge.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r playground/requirements.txt
(cd meterless_bridge && npm ci --no-audit --no-fund --loglevel=error)
java -version 2>&1 | head -1 || true
echo "WhyBot playground installed. It starts automatically; or run: bash playground/run.sh"
