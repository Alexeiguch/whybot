"""Live Nemotron client that takes the visitor's NVIDIA key as an argument instead of reading NVIDIA_API_KEY.

The key lives only on this object for the length of one run. It is never written to os.environ, never traced
(whybot's tracer also masks anything shaped like nvapi-...), and the playground drops the object when the run ends.
All chat/JSON/retry behaviour is inherited unchanged from whybot.llm.NvidiaLLM.
"""
from __future__ import annotations

import os

from whybot.config import Settings
from whybot.llm import BaseLLM, LLMError, NvidiaLLM


class KeyedNvidiaLLM(NvidiaLLM):
    def __init__(self, settings: Settings, api_key: str, on_call=None):
        BaseLLM.__init__(self, on_call)
        key = (api_key or "").strip()
        if not key:
            raise LLMError("no NVIDIA API key was provided")
        from openai import OpenAI
        self.s = settings
        self.call_timeout_s = settings.llm_timeout_s
        self.client = OpenAI(base_url=settings.llm_base_url, api_key=key, timeout=settings.llm_timeout_s, max_retries=0)
        self.thinking_roles = {r.strip() for r in os.environ.get("LLM_THINKING_ROLES", "").split(",") if r.strip()}

    def __repr__(self) -> str:  # never show the key in a repr or traceback
        return "KeyedNvidiaLLM(<key hidden>)"
