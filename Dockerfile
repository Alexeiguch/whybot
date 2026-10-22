# WhyBot: Python agents + Node (Meterless bridge) + JRE (Zetaris JDBC). No GPU; Nemotron is reached over the NVIDIA API.
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8000 JAVA_HOME=/usr/lib/jvm/default-java
RUN apt-get update && apt-get install -y --no-install-recommends nodejs npm default-jre-headless ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY meterless_bridge/package.json meterless_bridge/package-lock.json meterless_bridge/
RUN cd meterless_bridge && npm ci --omit=dev --no-audit --no-fund
COPY . .
RUN useradd --uid 10001 --create-home whybot && mkdir -p /app/logs /app/.meterless /app/.meterless-replay && chown -R 10001 /app/logs /app/.meterless /app/.meterless-replay
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -fsS http://localhost:${PORT}/health || exit 1
# Secrets are passed at runtime: docker run --env-file .env -p 8000:8000 whybot
# The Zetaris JDBC driver JAR is not redistributed: mount it and set ZETARIS_JDBC_JAR, or run with SAMPLE_MODE=true.
CMD ["python", "run.py", "--serve"]
