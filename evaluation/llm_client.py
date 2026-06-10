"""
Unified LLM client for evaluation modules.

Provides a single interface for calling LLMs via multiple backends:
  - OpenAI-compatible APIs (DashScope, vLLM, etc.)
  - ZhipuAI native SDK (GLM-4.7 via Coding endpoint)

All evaluation-related LLM calls should go through this module to ensure
consistent error handling, retry logic, and configuration management.

Backend auto-detection:
  - Model names containing "glm" → ZhipuAI backend
  - All other models → OpenAI-compatible backend
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

logger = logging.getLogger(__name__)


# =====================================================================
# Configuration
# =====================================================================

@dataclass
class LLMConfig:
    """Configuration for a single LLM endpoint.

    Attributes:
        model_name: Model identifier (e.g., "glm-4.7", "gpt-4o").
        api_key: API key. Falls back to env vars.
        base_url: API base URL. Falls back to env vars.
        temperature: Sampling temperature.
        max_tokens: Maximum output tokens.
        backend: Force backend type ("openai" | "zhipuai" | "auto").
    """
    model_name: str = "glm-4.7"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 8192
    backend: str = "auto"  # "auto" | "openai" | "zhipuai"

    def __post_init__(self):
        if self.backend == "auto":
            self.backend = self._detect_backend()

    def _detect_backend(self) -> str:
        """Auto-detect backend from model name."""
        name_lower = self.model_name.lower()
        if "glm" in name_lower:
            return "zhipuai"
        return "openai"


# =====================================================================
# Client
# =====================================================================

class LLMClient:
    """Unified LLM client supporting multiple backends.

    Each call creates a fresh client instance for thread safety.
    All calls are wrapped with retry logic and structured error handling.
    """

    def __init__(self, config: LLMConfig):
        self.config = config

    # -- ZhipuAI backend ----------------------------------------------------

    def _call_zhipu(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call via ZhipuAI native SDK."""
        from zhipuai import ZhipuAI

        client = ZhipuAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url or None,
        )

        response = client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.95,
        )

        content = response.choices[0].message.content
        if content is None or not content.strip():
            raise ValueError("LLM returned empty content")

        # Log token usage
        if hasattr(response, 'usage') and response.usage:
            logger.info(
                f"[{self.config.model_name}] Tokens — "
                f"prompt: {response.usage.prompt_tokens}, "
                f"completion: {response.usage.completion_tokens}, "
                f"total: {response.usage.total_tokens}"
            )

        return content.strip()

    # -- OpenAI-compatible backend ------------------------------------------

    def _call_openai(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call via OpenAI-compatible API."""
        from openai import OpenAI

        client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url or None,
        )

        response = client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        content = response.choices[0].message.content
        if content is None or not content.strip():
            raise ValueError("LLM returned empty content")
        return content.strip()

    # -- Dispatch with retry ------------------------------------------------

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=lambda rs: logger.warning(
            f"[LLMClient] Retry {rs.attempt_number}/5: "
            f"{rs.outcome.exception()}"
        ),
    )
    def _call_api(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Dispatch to the correct backend with retry logic."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        temp = temperature if temperature is not None else self.config.temperature
        mtok = max_tokens if max_tokens is not None else self.config.max_tokens

        if self.config.backend == "zhipuai":
            return self._call_zhipu(messages, temp, mtok)
        else:
            return self._call_openai(messages, temp, mtok)

    # -- Public API ---------------------------------------------------------

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Optional[str]:
        """Call the LLM and return the text response.

        Returns:
            Response text, or None on failure.
        """
        try:
            return self._call_api(system_prompt, user_prompt, temperature, max_tokens)
        except Exception as e:
            logger.error(f"[LLMClient] call() failed: {e}")
            return None

    def call_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Call the LLM and parse the response as JSON.

        Falls back to regex extraction if the response is not pure JSON.

        Returns:
            Parsed dict, or None on failure.
        """
        text = self.call(system_prompt, user_prompt, temperature, max_tokens)
        if text is None:
            return None
        return _safe_parse_json(text)


# =====================================================================
# Factory
# =====================================================================

# Default API configurations per provider
_DEFAULT_CONFIGS = {
    "zhipuai": {
        "api_key": "e2bb1c9dcfea446896cdfb3735c98a10.ZwHWlBTzph3t6RIa",
        "base_url": "https://open.bigmodel.cn/api/coding/paas/v4",
    },
    "openai": {
        "api_key": None,  # from env OPENAI_API_KEY
        "base_url": None,  # from env BASE_URL
    },
}


def create_client(model_name: str = "glm-4.7", **kwargs) -> LLMClient:
    """Factory: create an LLMClient with sensible defaults.

    Auto-detects backend from model name:
      - "glm-*" → ZhipuAI backend with Coding endpoint defaults
      - others  → OpenAI-compatible backend

    Args:
        model_name: Model identifier.
        **kwargs: Override any LLMConfig field (api_key, base_url,
                  temperature, max_tokens, backend).

    Returns:
        Configured LLMClient instance.
    """
    # Detect backend
    backend = kwargs.pop("backend", "auto")
    temp_config = LLMConfig(model_name=model_name, backend=backend)
    detected = temp_config.backend

    # Fill in defaults if not explicitly provided or set to None
    defaults = _DEFAULT_CONFIGS.get(detected, {})
    if not kwargs.get("api_key"):
        kwargs["api_key"] = defaults.get("api_key")
    if not kwargs.get("base_url"):
        kwargs["base_url"] = defaults.get("base_url")

    config = LLMConfig(model_name=model_name, backend=detected, **kwargs)
    return LLMClient(config)


# =====================================================================
# Helpers
# =====================================================================

def _safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """Try to parse JSON from LLM output.

    Handles cases where the LLM wraps JSON in markdown code blocks,
    includes extra text before/after, or truncates the response.
    """
    if not text or not text.strip():
        return None

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block (case-insensitive)
    m = re.search(r'```(?:json|JSON|Json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding the outermost { ... } block
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start:i+1])
                except json.JSONDecodeError:
                    start = None

    # Last resort: try to fix truncated JSON
    if start is not None:
        snippet = text[start:]

        # Strategy: remove the last incomplete key-value pair, then close braces
        fixed = _repair_truncated_json(snippet)
        if fixed is not None:
            return fixed

    logger.warning(f"[LLMClient] Could not parse JSON from: {text[:200]}")
    return None


def _repair_truncated_json(snippet: str) -> Optional[Dict[str, Any]]:
    """Attempt multiple repair strategies on truncated JSON.

    Handles:
      - Incomplete last value: {"vote": "X", "ranking":}
      - Truncated array: {"ranking": ["a", "b
      - Missing closing braces/brackets
    """
    # Count unclosed structures
    open_braces = snippet.count('{') - snippet.count('}')
    open_brackets = snippet.count('[') - snippet.count(']')
    rstripped = snippet.rstrip()

    # --- Strategy 1: truncate to last complete key-value pair ---
    # Find the last complete value followed by a comma or end-of-object
    # Walk backwards to find the last valid JSON boundary
    attempts = []

    # Remove trailing incomplete content after last comma at depth 1
    # e.g. {"a": 1, "b":}  →  {"a": 1}
    # e.g. {"vote": "X", "ranking":}  →  {"vote": "X"}
    last_comma = rstripped.rfind(',')
    if last_comma > 0:
        # Try everything up to and including the last comma (as a complete object)
        # by removing the trailing comma and closing braces
        prefix = rstripped[:last_comma].rstrip()
        if prefix.endswith(','):
            prefix = prefix[:-1].rstrip()
        for suffix in ['}', ']}', '}']:
            try:
                # Recount braces in prefix
                p_braces = prefix.count('{') - prefix.count('}')
                p_brackets = prefix.count('[') - prefix.count(']')
                candidate = prefix + ']' * max(0, p_brackets) + '}' * max(0, p_braces)
                return json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                continue

    # --- Strategy 2: close all open strings, brackets, braces ---
    # Check for unclosed string
    fixed = rstripped
    quote_count = fixed.count('"')
    if quote_count % 2 == 1:
        # Odd quotes — close the string
        fixed += '"'
    # If inside an array value, close it
    if open_brackets > 0:
        fixed += ']' * open_brackets
    if open_braces > 0:
        fixed += '}' * open_braces
    try:
        return json.loads(fixed)
    except (json.JSONDecodeError, ValueError):
        pass

    # --- Strategy 3: aggressive — extract only completed string values ---
    # Find all "key": "value" pairs using regex
    pairs = re.findall(r'"(\w+)"\s*:\s*"([^"]*)"', snippet)
    if pairs:
        reconstructed = {k: v for k, v in pairs}
        return reconstructed

    # --- Strategy 4: extract boolean/null/number values ---
    pairs2 = re.findall(r'"(\w+)"\s*:\s*(true|false|null|\d+\.?\d*)', snippet)
    if pairs2:
        result = {}
        for k, v in pairs2:
            if v == 'true':
                result[k] = True
            elif v == 'false':
                result[k] = False
            elif v == 'null':
                result[k] = None
            else:
                result[k] = float(v) if '.' in v else int(v)
        return result

    return None
