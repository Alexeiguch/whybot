# WhyBot playground image: Python + Node (Meterless bridge) + JRE (Zetaris JDBC) + Gradio UI. Build context: repo root.
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 JAVA_HOME=/usr/lib/jvm/default-java \
    SAMPLE_MODE=true GRADIO_ANALYTICS_ENABLED=False GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=7860 \
    WHYBOT_API_HOST=0.0.0.0 WHYBOT_API_PORT=8000
RUN apt-get update && apt-get install -y --no-install-recommends nodejs npm default-jre-headless ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
COPY playground/requirements.txt playground/
RUN pip install --no-cache-dir -r playground/requirements.txt
COPY meterless_bridge/package.json meterless_bridge/package-lock.json meterless_bridge/
RUN cd meterless_bridge && npm ci --omit=dev --no-audit --no-fund
COPY . .
RUN useradd --uid 10001 --create-home whybot && mkdir -p /app/logs /app/.meterless /app/.meterless-replay \
    && chown -R 10001 /app/logs /app/.meterless /app/.meterless-replay
USER 10001
EXPOSE 7860 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -fsS http://localhost:7860/ >/dev/null || exit 1
# No secrets are baked in. Live mode takes the visitor's NVIDIA key in the UI; Zetaris live needs ZETARIS_* at runtime.
CMD ["python", "-m", "playground.app"]
