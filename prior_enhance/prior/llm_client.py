"""DashScope OpenAI 兼容客户端。"""

from __future__ import annotations

from typing import List, Optional

from openai import OpenAI

from prior_enhance.config import LLMConfig
from prior_enhance.telemetry.run_logger import log

PHASE = "PRIOR"


class DashScopeLLMClient:
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        if not self.config.api_key:
            raise ValueError(
                "未配置 DASHSCOPE_API_KEY。"
                "请在 prior_enhance/secrets.py 中设置，或 export DASHSCOPE_API_KEY=sk-xxx"
            )
        self._client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )

    def chat(
        self,
        messages: List[dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: Optional[bool] = None,
    ) -> str:
        """发送 chat completion，返回 assistant 正文（content 部分）。"""
        model = model or self.config.model
        temperature = self.config.temperature if temperature is None else temperature
        max_tokens = max_tokens or self.config.max_tokens
        stream = self.config.stream if stream is None else stream

        log.log_input("llm_request", {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            "n_messages": len(messages),
        }, phase=PHASE)

        if stream:
            content, reasoning = self._chat_stream(messages, model, temperature, max_tokens)
        else:
            resp = self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            msg = resp.choices[0].message
            reasoning = getattr(msg, "reasoning_content", None) or ""
            content = msg.content or ""

        # deepseek-r1 系列：JSON 可能在 content 或 reasoning
        if reasoning.strip():
            log.log_output("llm_reasoning_chars", len(reasoning), phase=PHASE)
        log.log_output("llm_completion", {"response_chars": len(content)}, phase=PHASE)

        if not content.strip() and reasoning.strip():
            content = reasoning
        elif reasoning.strip() and '"priors"' in reasoning and '"priors"' not in content:
            content = reasoning
        return content

    def _chat_stream(
        self,
        messages: List[dict],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, str]:
        completion = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        for chunk in completion:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                reasoning_parts.append(delta.reasoning_content)
            if hasattr(delta, "content") and delta.content:
                content_parts.append(delta.content)
        return "".join(content_parts), "".join(reasoning_parts)
