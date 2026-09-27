"""Minimal OpenAI SDK compatibility for pinned third-party baselines.

The project normally uses a dependency-free OpenAI-compatible provider.  Some
vendored research code imports ``openai.OpenAI`` directly.  On experiment
hosts where the SDK is unavailable, this module installs only the small chat
completions surface required by that code.  It never replaces a real installed
SDK.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
import sys
import types
from typing import Any, Mapping
from urllib.error import HTTPError
from urllib.request import Request, urlopen


class _AttrDict(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def model_dump(self, mode: str | None = None) -> dict[str, Any]:
        del mode
        return {
            key: _plain(value)
            for key, value in self.items()
        }


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _attrs(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _AttrDict({key: _attrs(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_attrs(item) for item in value]
    return value


class ContextWindowExceededError(RuntimeError):
    """A backend rejected a request whose prompt cannot fit safely."""

    def __init__(
        self,
        message: str,
        *,
        context_window_tokens: int | None = None,
        input_tokens: int | None = None,
        requested_output_tokens: int | None = None,
    ) -> None:
        super().__init__(message)
        self.context_window_tokens = context_window_tokens
        self.input_tokens = input_tokens
        self.requested_output_tokens = requested_output_tokens


_CONTEXT_LENGTH_PATTERN = re.compile(
    r"maximum context length of\s+(?P<context>\d+)\s+tokens.*?"
    r"(?P<input>\d+)\s+tokens from the input messages and\s+"
    r"(?P<output>\d+)\s+tokens for the completion",
    flags=re.IGNORECASE | re.DOTALL,
)

_SGLANG_INPUT_CONTEXT_PATTERN = re.compile(
    r"input\s*\(\s*(?P<input>\d+)\s+tokens\s*\)\s+is longer than\s+"
    r"(?:the\s+)?model(?:'s)?\s+context length\s*"
    r"\(\s*(?P<context>\d+)\s+tokens\s*\)",
    flags=re.IGNORECASE | re.DOTALL,
)


def _context_error_details(body: str) -> tuple[int, int, int] | None:
    match = _CONTEXT_LENGTH_PATTERN.search(str(body or ""))
    if match is not None:
        return (
            int(match.group("context")),
            int(match.group("input")),
            int(match.group("output")),
        )
    match = _SGLANG_INPUT_CONTEXT_PATTERN.search(str(body or ""))
    if match is not None:
        # SGLang reports only the prompt and context-window sizes for this
        # error form. A zero sentinel is replaced by the request's actual
        # max_tokens value before the normalized exception is raised.
        return (
            int(match.group("context")),
            int(match.group("input")),
            0,
        )
    return None


def _context_retry_max_tokens(
    body: str,
    *,
    reserve_tokens: int,
    minimum_output_tokens: int,
) -> int | None:
    """Return a safe server-measured output budget for a context 400."""

    details = _context_error_details(body)
    if details is None:
        return None
    context_window, input_tokens, _ = details
    available = context_window - input_tokens - int(reserve_tokens)
    if available < int(minimum_output_tokens):
        return None
    return available


def _model_request_defaults(
    model: str,
    *,
    max_tokens: int = 8192,
) -> dict[str, Any]:
    """Return transport defaults without leaking backend-specific options.

    The local Qwen service accepts SGLang's ``chat_template_kwargs`` and a
    deterministic temperature.  Public OpenAI-compatible gateways such as
    Gemini do not necessarily expose either option, so those fields must not
    be sent merely because the vendored CUPMem client shares this transport.
    Explicit caller kwargs still override all defaults in ``create``.
    """

    defaults: dict[str, Any] = {
        "max_tokens": int(max_tokens),
        "seed": 731,
    }
    if "qwen" in model.casefold():
        defaults.update(
            {
                "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )
    return defaults


@dataclass
class _ChatCompletions:
    api_key: str
    base_url: str
    timeout: float
    default_max_tokens: int = 8192
    context_reserve_tokens: int = 512
    minimum_output_tokens: int = 512
    adaptive_context_retry: bool = False

    def create(self, *, model: str, messages: list[dict[str, str]], **kwargs: Any):
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            **_model_request_defaults(
                model,
                max_tokens=self.default_max_tokens,
            ),
            **kwargs,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        context_retry_used = False
        while True:
            request = Request(
                self.base_url.rstrip("/") + "/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    result = json.loads(response.read().decode("utf-8"))
                break
            except HTTPError as error:
                body = error.read().decode("utf-8", errors="replace")
                details = _context_error_details(body)
                retry_max_tokens = _context_retry_max_tokens(
                    body,
                    reserve_tokens=self.context_reserve_tokens,
                    minimum_output_tokens=self.minimum_output_tokens,
                )
                current_max_tokens = int(
                    payload.get("max_tokens", self.default_max_tokens)
                )
                if (
                    error.code == 400
                    and self.adaptive_context_retry
                    and not context_retry_used
                    and retry_max_tokens is not None
                    and retry_max_tokens < current_max_tokens
                ):
                    payload["max_tokens"] = retry_max_tokens
                    context_retry_used = True
                    continue
                message = f"OpenAI-compatible HTTP {error.code}: {body[:1000]}"
                if error.code == 400 and details is not None:
                    context_window, input_tokens, reported_output = details
                    requested_output = reported_output or current_max_tokens
                    raise ContextWindowExceededError(
                        message,
                        context_window_tokens=context_window,
                        input_tokens=input_tokens,
                        requested_output_tokens=requested_output,
                    ) from error
                raise RuntimeError(message) from error
        return _attrs(result)


class OpenAICompat:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout: float = 600.0,
        default_max_tokens: int = 8192,
        context_reserve_tokens: int = 512,
        minimum_output_tokens: int = 512,
        adaptive_context_retry: bool = False,
        **_: Any,
    ) -> None:
        completions = _ChatCompletions(
            api_key=api_key,
            base_url=base_url,
            timeout=float(timeout),
            default_max_tokens=int(default_max_tokens),
            context_reserve_tokens=int(context_reserve_tokens),
            minimum_output_tokens=int(minimum_output_tokens),
            adaptive_context_retry=bool(adaptive_context_retry),
        )
        self.chat = types.SimpleNamespace(completions=completions)


def install_openai_compat_if_missing() -> bool:
    """Install the compatibility module only when ``openai`` is unavailable."""

    try:
        __import__("openai")
        return False
    except ModuleNotFoundError:
        module = types.ModuleType("openai")
        module.OpenAI = OpenAICompat  # type: ignore[attr-defined]
        sys.modules["openai"] = module
        return True


__all__ = [
    "ContextWindowExceededError",
    "OpenAICompat",
    "install_openai_compat_if_missing",
]
