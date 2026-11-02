from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from tenacity import(
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential
)
from openai import (
    OpenAI,
    AzureOpenAI,
    OpenAIError,
    AuthenticationError,
    RateLimitError,
    BadRequestError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError
)
from openai.types.chat import ChatCompletionMessage
from app.logger import logger
from app.tracing import trace_event

if TYPE_CHECKING:
    from app.config.config import LLMConfig


class EmptyResponseError(Exception):
    """Custom exception for empty LLM responses, used to trigger retry."""
    pass


class LLM:
    """LLM wrapper class. Each instance lazily creates its own OpenAI client."""
    
    def __init__(self, llm_config: LLMConfig):
        self._config = llm_config
        self._client = None
        self._client_lock = threading.Lock()
        logger.debug(
            f"Initialized LLM wrapper: model={llm_config.model}, "
            f"temperature={llm_config.temperature}, reasoning_effort={llm_config.reasoning_effort}"
        )
    
    @property
    def llm_config(self) -> LLMConfig:
        """Get the LLM configuration."""
        return self._config
    
    def _create_client(self):
        if self._config.api_type == "openai":
            return OpenAI(api_key=self._config.api_key, base_url=self._config.base_url)
        elif self._config.api_type == "azure":
            return AzureOpenAI(api_key=self._config.api_key, base_url=self._config.base_url, api_version=self._config.api_version)
        else:
            raise ValueError(f"Unsupported api type: {self._config.api_type}")

    def _get_client(self):
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    self._client = self._create_client()
                    logger.debug(f"Created LLM client for model={self._config.model}")
        return self._client
        
    @retry(
        wait=wait_random_exponential(multiplier=1, max=60),
        stop=stop_after_attempt(15),
        # Retry on recoverable errors (including BadRequestError for provider-specific issues like "user location not supported")
        # NOT on AuthenticationError (wrong API key won't fix itself)
        retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError, InternalServerError, BadRequestError, EmptyResponseError))
    )
    def ask(self, messages: List[Dict[str, str]],
                  system_message: Optional[Dict[str, str]] = None,
                  timeout: int = 300,
                  **kwargs) -> tuple[List[ChatCompletionMessage], Dict[str, int]]:
        # trace_label / trace_extra 仅用于执行痕迹记录，不属于请求参数。
        # 必须在 request_params.update(kwargs) 之前 pop，否则会透传给 OpenAI SDK。
        trace_label = kwargs.pop("trace_label", None)
        trace_extra = kwargs.pop("trace_extra", None)
        trace_context: Dict[str, Any] = dict(trace_extra or {})
        if trace_label:
            trace_context.setdefault("label", trace_label)

        call_started = time.perf_counter()

        if system_message:
            messages = [system_message] + messages
            
        target_n = kwargs.pop("n", 1)
        max_request_n = 1 if self._config.n_call_strategy == "split" else (self._config.max_request_n or target_n)
        
        all_choices = []
        total_token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        
        current_max_tokens = kwargs.pop("max_tokens", self._config.max_tokens)

        trace_event(
            "llm_call_start",
            model=self._config.model,
            messages=messages,
            requested_n=target_n,
            max_request_n=max_request_n,
            max_tokens=current_max_tokens,
            temperature=self._config.temperature,
            reasoning_effort=self._config.reasoning_effort,
            timeout=timeout,
            extra_request_params=kwargs,
            trace_extra=trace_context,
        )

        request_index = 0
        while len(all_choices) < target_n:
            current_n = min(target_n - len(all_choices), max_request_n)
            request_index += 1
            request_started = time.perf_counter()
            try:
                request_params = {
                    "model": self._config.model,
                    "messages": messages,
                    "max_tokens": current_max_tokens,
                    "temperature": self._config.temperature,
                    "timeout": timeout,
                    "n": current_n,
                }
                if self._config.reasoning_effort is not None:
                    request_params["reasoning_effort"] = self._config.reasoning_effort
                if self._config.extra_body:
                    request_params["extra_body"] = self._config.extra_body
                request_params.update(kwargs)

                trace_event(
                    "llm_request",
                    request_index=request_index,
                    n=current_n,
                    max_tokens=current_max_tokens,
                    temperature=self._config.temperature,
                    timeout=timeout,
                    reasoning_effort=self._config.reasoning_effort,
                    extra_body=self._config.extra_body,
                    extra_request_params=kwargs,
                    message_count=len(messages),
                    trace_extra=trace_context,
                )
                    
                response = self._get_client().chat.completions.create(**request_params)
                latency_ms = round((time.perf_counter() - request_started) * 1000, 1)
                if not response.choices:
                    trace_event(
                        "llm_empty_response",
                        reason="no_choices",
                        request_index=request_index,
                        latency_ms=latency_ms,
                        response_id=getattr(response, "id", None),
                        trace_extra=trace_context,
                    )
                    raise EmptyResponseError(f"No response from the model: {response}")
                
                # Check if any choice has None or empty content
                for choice in response.choices:
                    if choice.message.content is None or choice.message.content.strip() == "":
                        trace_event(
                            "llm_empty_response",
                            reason="empty_content",
                            request_index=request_index,
                            latency_ms=latency_ms,
                            finish_reason=getattr(choice, "finish_reason", None),
                            response_id=getattr(response, "id", None),
                            trace_extra=trace_context,
                        )
                        raise EmptyResponseError(f"Model returned empty content (possibly filtered): {response}")
                
                all_choices.extend([choice.message for choice in response.choices])

                trace_event(
                    "llm_response",
                    request_index=request_index,
                    latency_ms=latency_ms,
                    response_id=getattr(response, "id", None),
                    response_model=getattr(response, "model", None),
                    finish_reasons=[getattr(choice, "finish_reason", None) for choice in response.choices],
                    choices=[
                        {
                            "index": getattr(choice, "index", idx),
                            "finish_reason": getattr(choice, "finish_reason", None),
                            "content": choice.message.content,
                            "reasoning_content": getattr(choice.message, "reasoning_content", None),
                        }
                        for idx, choice in enumerate(response.choices)
                    ],
                    usage={
                        "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                        "completion_tokens": getattr(response.usage, "completion_tokens", None),
                        "total_tokens": getattr(response.usage, "total_tokens", None),
                    } if response.usage else None,
                    accumulated_choices=len(all_choices),
                    trace_extra=trace_context,
                )
                
                # Calculate token usage
                if response.usage:
                    total_token_usage["prompt_tokens"] += response.usage.prompt_tokens
                    total_token_usage["completion_tokens"] += response.usage.completion_tokens
                    total_token_usage["total_tokens"] += response.usage.total_tokens
                
            except OpenAIError as e:
                trace_event(
                    "llm_error",
                    request_index=request_index,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    latency_ms=round((time.perf_counter() - request_started) * 1000, 1),
                    trace_extra=trace_context,
                )
                if isinstance(e, RateLimitError):
                    logger.error(f"OpenAI error: {e}")
                    logger.error("Rate limit exceeded, please try again later.")
                elif isinstance(e, AuthenticationError):
                    logger.error(f"OpenAI error: {e}")
                    logger.error("Authentication error, please check your api key.")
                elif isinstance(e, BadRequestError):
                    error_msg = str(e).lower()
                    if "context" in error_msg or "tokens" in error_msg or "length" in error_msg:
                        new_max_tokens = int(current_max_tokens * 0.9) if current_max_tokens else 0
                        if new_max_tokens > 0:
                            logger.warning(f"Context length exceeded. Reducing max_tokens from {current_max_tokens} to {new_max_tokens} and retrying.")
                            trace_event(
                                "llm_shrink_max_tokens",
                                previous_max_tokens=current_max_tokens,
                                new_max_tokens=new_max_tokens,
                                error_message=str(e),
                                trace_extra=trace_context,
                            )
                            current_max_tokens = new_max_tokens
                            continue
                        else:
                            logger.error(f"Context length exceeded, but max_tokens cannot be reduced further (current: {current_max_tokens}).")
                    
                    logger.error(f"OpenAI error: {e}")
                    logger.error("Bad request, please check your request parameters.")
                raise e
            except Exception as e:
                trace_event(
                    "llm_error",
                    request_index=request_index,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    latency_ms=round((time.perf_counter() - request_started) * 1000, 1),
                    trace_extra=trace_context,
                )
                logger.error(f"Unexpected error: {e}")
                raise e

        trace_event(
            "llm_call_end",
            request_count=request_index,
            choice_count=len(all_choices),
            target_n=target_n,
            duration_ms=round((time.perf_counter() - call_started) * 1000, 1),
            token_usage=total_token_usage,
            trace_extra=trace_context,
        )
                
        return all_choices, total_token_usage
