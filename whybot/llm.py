"""LLM access. NVIDIA build (OpenAI-compatible) with Nemotron models, JSON outputs validated by pydantic, retried with feedback."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Callable, Type, TypeVar

from pydantic import BaseModel, ValidationError

from .config import Settings

T = TypeVar("T", bound=BaseModel)

THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)


class LLMError(RuntimeError):
    pass


def strip_reasoning(text: str | None) -> str:
    """Remove reasoning traces: <think> blocks, dangling </think> prefixes, and code fences."""
    if not text:
        return ""
    text = THINK_RE.sub("", text)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def extract_json(text: str) -> Any:
    """Return the first JSON object in text (handles ```json fences and leading prose)."""
    text = strip_reasoning(text)
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found in model output")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unterminated JSON object in model output")


_WATCHDOG = concurrent.futures.ThreadPoolExecutor(max_workers=8, thread_name_prefix="llm-call")


def prompt_sha(messages: list[dict]) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()[:16]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class BaseLLM:
    """Subclasses implement chat(). complete_json() adds schema validation, retry-with-feedback, back-off and the run deadline."""

    def __init__(self, on_call: Callable[[dict], None] | None = None):
        self.on_call = on_call
        self.deadline: float | None = None   # wall-clock deadline of the current run (set by the orchestrator)
        self.call_timeout_s: float = 90.0

    def model_for(self, role: str) -> str:
        return role

    def _timeout(self) -> float:
        if self.deadline is None:
            return self.call_timeout_s
        left = self.deadline - time.time()
        if left < 5:
            raise LLMError("run budget exhausted")
        return min(self.call_timeout_s, left)

    def chat(self, role: str, messages: list[dict], *, max_tokens: int = 4096, temperature: float = 0.2) -> tuple[str, dict]:
        raise NotImplementedError

    def complete_json(self, role: str, system: str, user: str, model_cls: Type[T], *,
                      validator: Callable[[T], list[str]] | None = None, max_attempts: int = 3,
                      max_tokens: int = 4096) -> tuple[T, dict]:
        schema = json.dumps(model_cls.model_json_schema(), separators=(",", ":"))
        sys_msg = (f"{system}\n\nRespond with ONE JSON object only, no prose, matching this JSON schema:\n{schema}")
        messages = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}]
        errors: list[str] = []
        meta: dict = {}
        transport_failures = 0
        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            if self.deadline is not None and self.deadline - time.time() < 5:
                raise LLMError(f"{role}: run budget exhausted")
            try:
                raw, meta = self.chat(role, messages, max_tokens=max_tokens)
            except LLMError as e:
                errors.append(str(e))
                if self.on_call:
                    self.on_call({"role": role, "event": "llm_error_retry", "attempt": attempt, "error": str(e)[:300]})
                if "budget exhausted" in str(e) or "cassette" in str(e) or transport_failures >= 2:
                    break
                transport_failures += 1
                attempt -= 1                         # a 429/5xx/timeout does not use up a validation attempt
                time.sleep(min(2 ** transport_failures, 8))
                continue
            meta = {**meta, "attempt": attempt}
            if meta.get("finish_reason") == "length":
                max_tokens = int(max_tokens * 1.5)
            try:
                obj = model_cls.model_validate(extract_json(raw))
                problems = validator(obj) if validator else []
                if not problems:
                    meta["validation_errors"] = errors
                    return obj, meta
                err = "; ".join(problems)
            except (ValueError, ValidationError, json.JSONDecodeError) as e:
                err = str(e)[:800]
            errors.append(err)
            if self.on_call:
                self.on_call({"role": role, "event": "json_retry", "attempt": attempt, "error": err})
            messages = messages[:2] + [
                {"role": "assistant", "content": strip_reasoning(raw)[:4000] or "(empty)"},
                {"role": "user", "content": f"Your previous answer was rejected: {err}\nReturn a corrected JSON object only."},
            ]
        raise LLMError(f"{role}: no valid JSON after {max_attempts} attempts: {errors[-1] if errors else ''}")


class NvidiaLLM(BaseLLM):
    """OpenAI-compatible client against integrate.api.nvidia.com. Key from NVIDIA_API_KEY only."""

    # Roles that run with reasoning on (comma list in LLM_THINKING_ROLES, default none: Ultra thinking costs 40-140 s per call).
    # Reasoning content is always stripped and never logged verbatim; only its length is traced.

    def __init__(self, settings: Settings, on_call: Callable[[dict], None] | None = None):
        super().__init__(on_call)
        from openai import OpenAI
        key = os.environ.get("NVIDIA_API_KEY")
        if not key:
            raise LLMError("NVIDIA_API_KEY is not set")
        self.s = settings
        self.call_timeout_s = settings.llm_timeout_s
        self.client = OpenAI(base_url=settings.llm_base_url, api_key=key, timeout=settings.llm_timeout_s, max_retries=0)
        self.thinking_roles = {r.strip() for r in os.environ.get("LLM_THINKING_ROLES", "").split(",") if r.strip()}

    def model_for(self, role: str) -> str:
        return {"planner": self.s.planner_model, "critic": self.s.critic_model,
                "sql": self.s.sql_model, "writer": self.s.writer_model, "analyst": self.s.writer_model}.get(role, self.s.writer_model)

    def chat(self, role, messages, *, max_tokens=4096, temperature=0.2):
        model = self.model_for(role)
        thinking = role in self.thinking_roles
        timeout = self._timeout()
        t0 = time.time()
        def call():
            return self.client.with_options(timeout=timeout).chat.completions.create(
                model=model, messages=messages, temperature=temperature,
                max_tokens=max_tokens + (8000 if thinking else 0),
                extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
            )
        # Hard watchdog: a stalled TLS read was observed to outlive the client timeout, so the call runs in a worker
        # thread and is abandoned (the daemon thread finishes or dies with the process) once the budget is spent.
        fut = _WATCHDOG.submit(call)
        try:
            r = fut.result(timeout=timeout + 5)
        except concurrent.futures.TimeoutError as e:
            raise LLMError(f"{model}: watchdog timeout after {timeout + 5:.0f}s") from e
        except Exception as e:
            raise LLMError(f"{model}: {type(e).__name__}: {str(e)[:300]}") from e
        msg = r.choices[0].message
        content = strip_reasoning(msg.content)
        usage = getattr(r, "usage", None)
        meta = {"model": model, "latency_s": round(time.time() - t0, 2), "thinking": thinking,
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "finish_reason": r.choices[0].finish_reason, "prompt_sha": prompt_sha(messages),
                "reasoning_chars_stripped": len(getattr(msg, "reasoning_content", None) or "")}
        if self.on_call:
            self.on_call({"role": role, "event": "llm_call", **meta})
        if not content:
            raise LLMError(f"{model}: empty content (finish_reason={r.choices[0].finish_reason})")
        return content, meta


class CassetteLLM(BaseLLM):
    """Records (inner=live LLM) or replays (inner=None) the ordered LLM responses of one recorded LIVE run.

    SAMPLE_MODE replays by default so a judge sees exactly the recorded investigation. Every replayed call carries the
    original live metadata (model, latency, tokens, recorded_at, live run id) and prompt_match: whether today's prompt
    hashes to the prompt that was sent live. Drift (code or prompt changes since recording) is reported, never hidden."""

    def __init__(self, path, inner: BaseLLM | None = None, on_call=None, run_id: str | None = None):
        super().__init__(on_call)
        from pathlib import Path
        self.path = Path(path)
        self.inner = inner
        self.run_id = run_id
        self.recorded: list[dict] = []
        self.queue: dict[str, list[dict]] = {}
        self.header: dict = {}
        if inner is None:
            if not self.path.is_file():
                raise LLMError(f"no LLM cassette at {self.path}")
            data = json.loads(self.path.read_text())
            self.header = {k: v for k, v in data.items() if k != "calls"}
            for item in data["calls"]:
                self.queue.setdefault(item["role"], []).append(item)

    def model_for(self, role):
        return self.inner.model_for(role) if self.inner else f"cassette:{role}"

    def chat(self, role, messages, *, max_tokens=4096, temperature=0.2):
        if self.inner is not None:
            self.inner.on_call = self.on_call
            self.inner.deadline = self.deadline
            content, meta = self.inner.chat(role, messages, max_tokens=max_tokens, temperature=temperature)
            self.recorded.append({"role": role, "content": content, "recorded_at": _now(), "prompt_sha": prompt_sha(messages),
                                  **{k: meta.get(k) for k in ("model", "latency_s", "prompt_tokens", "completion_tokens", "finish_reason")}})
            return content, meta
        q = self.queue.get(role)
        if not q:
            raise LLMError(f"cassette exhausted for role {role}")
        item = q.pop(0)
        cur = prompt_sha(messages)
        meta = {"model": f"replay:{item.get('model')}", "latency_s": 0.0, "replayed": True,
                "recorded_at": item.get("recorded_at"), "live_run_id": self.header.get("live_run_id"),
                "original_latency_s": item.get("latency_s"), "original_completion_tokens": item.get("completion_tokens"),
                "original_prompt_tokens": item.get("prompt_tokens"), "prompt_sha": cur, "recorded_prompt_sha": item.get("prompt_sha"),
                "prompt_match": cur == item.get("prompt_sha")}
        if self.on_call:
            self.on_call({"role": role, "event": "llm_call", **meta})
        return item["content"], meta

    def save(self, question: str):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"question": question, "live_run_id": self.run_id, "trace": f"logs/{self.run_id}.jsonl",
                                         "recorded_at": _now(), "checks_version": _checks_version(),
                                         **{k: v for k, v in self.header.items() if k in ("hmem_recall", "table_notes", "dialect_notes")},
                                         "calls": self.recorded}, indent=1))


def _checks_version() -> int:
    from .checks import CHECKS_VERSION
    return CHECKS_VERSION


def cassette_path(fixtures_dir, question: str):
    from pathlib import Path
    return Path(fixtures_dir) / "llm" / (hashlib.sha256(question.strip().lower().encode()).hexdigest()[:16] + ".json")
