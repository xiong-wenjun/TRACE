from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
from http.client import RemoteDisconnected
import json
import math
import os
import threading
import time
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen


JUDGE_SYSTEM_PROMPT = "You are a careful multi-agent reasoning component."
JUDGE_SYSTEM_PROMPT_SHA256 = hashlib.sha256(
    JUDGE_SYSTEM_PROMPT.encode("utf-8")
).hexdigest()


class _RejectRedirects(HTTPRedirectHandler):
    """Prevent an authenticated request from forwarding Bearer credentials."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        del req, fp, code, msg, headers, newurl
        return None


def _open_external_no_redirect(request: Request, *, timeout: float):
    return build_opener(_RejectRedirects()).open(request, timeout=timeout)


def _resolve_generation_request_lock_path(
    request_lock_path: str | None,
) -> str | None:
    """Resolve an explicit provider lock, falling back to the legacy env var."""

    value = request_lock_path
    if value is None:
        value = os.environ.get("TRACE_GENERATION_LOCK_PATH", "")
    path = str(value).strip()
    if not path:
        return None
    if not os.path.isabs(path):
        raise ValueError("generation request lock path must be absolute")
    return path


@contextmanager
def _generation_request_slot(request_lock_path: str | None):
    """Optionally serialize calls made by one provider lock namespace."""

    path = request_lock_path
    if path is None:
        yield
        return
    descriptor = os.open(
        path,
        os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_seconds: float
    transport_mode: str | None = None


def endpoint_origin(url: str) -> str:
    """Return the non-secret origin used in public experiment metadata."""

    parsed = urlsplit(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("endpoint must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("endpoint URL must not contain credentials")
    return urlunsplit((parsed.scheme.lower(), parsed.netloc, "", "", ""))


def _chat_completions_url(base_url: str) -> str:
    parsed = urlsplit(base_url.strip().rstrip("/"))
    origin = endpoint_origin(base_url)
    if parsed.query or parsed.fragment:
        raise ValueError("endpoint URL must not contain a query or fragment")
    path = parsed.path.rstrip("/")
    if path.endswith("/chat/completions"):
        suffix = path
    elif path.endswith(("/v1", "/v1beta/openai", "/v1alpha/openai")):
        suffix = f"{path}/chat/completions"
    elif parsed.hostname == "api.deepseek.com" and not path:
        suffix = "/chat/completions"
    else:
        suffix = f"{path}/v1/chat/completions" if path else "/v1/chat/completions"
    return f"{origin}{suffix}"


def _openai_api_url(base_url: str, endpoint: str) -> str:
    """Resolve an OpenAI-compatible endpoint without duplicating ``/v1``."""

    endpoint = endpoint.strip("/")
    if not endpoint:
        raise ValueError("API endpoint must be non-empty")
    if endpoint == "chat/completions":
        return _chat_completions_url(base_url)
    parsed = urlsplit(base_url.strip().rstrip("/"))
    origin = endpoint_origin(base_url)
    if parsed.query or parsed.fragment:
        raise ValueError("endpoint URL must not contain a query or fragment")
    path = parsed.path.rstrip("/")
    if path.endswith(("/v1", "/v1beta/openai", "/v1alpha/openai")):
        api_root = path
    elif parsed.hostname == "api.deepseek.com" and not path:
        api_root = ""
    else:
        api_root = f"{path}/v1"
    return f"{origin}{api_root}/{endpoint}"


def response_format_sha256(response_format: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        response_format,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _copy_response_format(
    response_format: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if response_format is None:
        return None
    if not isinstance(response_format, Mapping):
        raise TypeError("response_format must be a mapping")
    try:
        copied = json.loads(json.dumps(dict(response_format)))
    except (TypeError, ValueError) as error:
        raise ValueError("response_format must be JSON serializable") from error
    if not isinstance(copied, dict) or not isinstance(copied.get("type"), str):
        raise ValueError("response_format requires a string type")
    return copied


_RESERVED_CHAT_REQUEST_FIELDS = frozenset(
    {
        "model",
        "messages",
        "temperature",
        "seed",
        "max_tokens",
        "tools",
        "tool_choice",
        "response_format",
        "chat_template_kwargs",
    }
)


def _copy_chat_request_overrides(
    overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if overrides is None:
        return {}
    if not isinstance(overrides, Mapping):
        raise TypeError("chat_request_overrides must be a mapping")
    collisions = sorted(_RESERVED_CHAT_REQUEST_FIELDS.intersection(overrides))
    if collisions:
        raise ValueError(
            "chat_request_overrides contains reserved fields: "
            + ", ".join(collisions)
        )
    try:
        copied = json.loads(json.dumps(dict(overrides)))
    except (TypeError, ValueError) as error:
        raise ValueError(
            "chat_request_overrides must be JSON serializable"
        ) from error
    if not isinstance(copied, dict):
        raise TypeError("chat_request_overrides must encode a JSON object")
    return copied


class OpenAICompatibleChatClient:
    """Authenticated, chat-only client for an independent external Judge.

    The API secret is intentionally accepted only as an in-memory constructor
    argument.  ``safe_metadata`` and ``repr`` expose neither the secret nor a
    fingerprint that could become a stable identifier across experiments.
    """

    _TOKEN_FIELDS = frozenset({"max_tokens", "max_completion_tokens"})

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str,
        timeout: float = 180.0,
        retries: int = 3,
        max_tokens_field: str = "max_tokens",
        temperature: float = 0.0,
        send_seed: bool = False,
        response_format: Mapping[str, Any] | None = None,
    ) -> None:
        model = model.strip()
        if not model:
            raise ValueError("judge model must be non-empty")
        if not api_key or not api_key.strip():
            raise ValueError("judge API key must be non-empty")
        if "\r" in api_key or "\n" in api_key:
            raise ValueError("judge API key contains invalid header characters")
        if max_tokens_field not in self._TOKEN_FIELDS:
            raise ValueError(
                "max_tokens_field must be max_tokens or max_completion_tokens"
            )
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if retries < 1:
            raise ValueError("retries must be at least one")
        if not math.isfinite(float(temperature)) or float(temperature) < 0:
            raise ValueError("temperature must be a finite non-negative number")

        self._url = _chat_completions_url(base_url)
        self._origin = endpoint_origin(base_url)
        self.model = model
        self._api_key = api_key.strip()
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.max_tokens_field = max_tokens_field
        self.temperature = float(temperature)
        self.send_seed = bool(send_seed)
        self._response_format = _copy_response_format(response_format)
        self._lock = threading.Lock()
        self._usage: dict[str, int | float] = {
            "chat_calls": 0,
            "request_attempts": 0,
            "request_retries": 0,
            "http_errors": 0,
            "transport_errors": 0,
            "malformed_responses": 0,
            "invalid_envelopes": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "generation_latency_seconds": 0.0,
        }

    def __repr__(self) -> str:
        return (
            "OpenAICompatibleChatClient("
            f"origin={self._origin!r}, model={self.model!r}, secret_present=True)"
        )

    def safe_metadata(self) -> dict[str, Any]:
        metadata = {
            "mode": "external",
            "endpoint_origin": self._origin,
            "model": self.model,
            "system_prompt_sha256": JUDGE_SYSTEM_PROMPT_SHA256,
            "secret_present": True,
        }
        if self._response_format is not None:
            metadata.update(
                {
                    "response_format_type": self._response_format["type"],
                    "response_schema_sha256": response_format_sha256(
                        self._response_format
                    ),
                }
            )
        return metadata

    def snapshot(self) -> dict[str, int | float]:
        with self._lock:
            return dict(self._usage)

    def chat(self, prompt: str, *, seed: int, max_tokens: int = 512) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": JUDGE_SYSTEM_PROMPT,
                },
                {"role": "user", "content": prompt},
            ],
            self.max_tokens_field: int(max_tokens),
            "temperature": self.temperature,
        }
        if self.send_seed:
            payload["seed"] = int(seed)
        if self._response_format is not None:
            payload["response_format"] = self._response_format
        encoded = json.dumps(payload).encode("utf-8")
        started = time.perf_counter()
        text: str | None = None
        last_failure = "unknown_error"
        for attempt in range(self.retries):
            with self._lock:
                self._usage["request_attempts"] += 1
            request = Request(
                self._url,
                data=encoded,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with _open_external_no_redirect(
                    request, timeout=self.timeout
                ) as handle:
                    decoded = json.loads(handle.read().decode("utf-8"))
                if not isinstance(decoded, dict):
                    raise ValueError("response is not an object")
                choice = decoded["choices"][0]
                usage = decoded.get("usage") or {}
                prompt_tokens = int(usage.get("prompt_tokens", 0))
                completion_tokens = int(usage.get("completion_tokens", 0))
                total_tokens = int(
                    usage.get("total_tokens", prompt_tokens + completion_tokens)
                )
                # Count physical API usage even when a completion is discarded
                # and retried.  Otherwise long-run cost reports are biased low.
                with self._lock:
                    self._usage["prompt_tokens"] += prompt_tokens
                    self._usage["completion_tokens"] += completion_tokens
                    self._usage["total_tokens"] += total_tokens
                if decoded.get("model") != self.model:
                    raise LookupError("served model differs from requested model")
                if choice.get("finish_reason") != "stop":
                    raise LookupError("completion did not finish normally")
                candidate = choice["message"]["content"]
                if not isinstance(candidate, str) or not candidate.strip():
                    raise LookupError("empty content")
                text = candidate
                break
            except HTTPError as error:
                last_failure = f"http_status_{error.code}"
                with self._lock:
                    self._usage["http_errors"] += 1
                retryable = error.code == 429 or 500 <= error.code < 600
                if not retryable:
                    break
            except (
                URLError,
                TimeoutError,
                RemoteDisconnected,
                ConnectionResetError,
                BrokenPipeError,
            ):
                last_failure = "transport_error"
                with self._lock:
                    self._usage["transport_errors"] += 1
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                last_failure = "malformed_response"
                with self._lock:
                    self._usage["malformed_responses"] += 1
            except (KeyError, IndexError, TypeError, LookupError):
                last_failure = "invalid_completion_envelope"
                with self._lock:
                    self._usage["invalid_envelopes"] += 1
            if attempt + 1 < self.retries:
                with self._lock:
                    self._usage["request_retries"] += 1
                time.sleep(float(attempt + 1))
        if text is None:
            raise RuntimeError(
                "independent Judge request failed closed after "
                f"{self.retries} attempt(s): {last_failure}"
            )

        with self._lock:
            self._usage["chat_calls"] += 1
            self._usage["generation_latency_seconds"] += (
                time.perf_counter() - started
            )
        return text


def _generation_seed(model: str, seed: int) -> int:
    if model.casefold().startswith("gemini-"):
        return int(seed) & 0x7FFFFFFF
    return int(seed)


def _textual_content(value: object) -> str | None:
    """Normalize OpenAI strings and Anthropic-style text block arrays."""

    if isinstance(value, str):
        return value if value.strip() else None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    pieces: list[str] = []
    for block in value:
        if isinstance(block, str):
            if block.strip():
                pieces.append(block)
            continue
        if not isinstance(block, Mapping):
            continue
        for key in ("text", "output_text", "content"):
            text = _textual_content(block.get(key))
            if text is not None:
                pieces.append(text)
                break
    joined = "\n".join(pieces)
    return joined if joined.strip() else None


def _completion_envelope_diagnostic(
    response: Mapping[str, Any],
    message: Mapping[str, Any],
) -> str:
    """Describe an empty envelope without persisting prompt or response text."""

    choices = response.get("choices")
    choice = (
        choices[0]
        if isinstance(choices, Sequence)
        and not isinstance(choices, (str, bytes))
        and choices
        and isinstance(choices[0], Mapping)
        else {}
    )
    usage = response.get("usage")
    usage = usage if isinstance(usage, Mapping) else {}
    content = message.get("content")
    reasoning = message.get("reasoning_content")
    return (
        f"finish_reason={choice.get('finish_reason')!r}, "
        f"message_keys={sorted(str(key) for key in message)}, "
        f"content_type={type(content).__name__}, "
        f"reasoning_content_type={type(reasoning).__name__}, "
        f"completion_tokens={int(usage.get('completion_tokens', 0) or 0)}"
    )


def _retryable_empty_completion(
    response: Mapping[str, Any],
    message: Mapping[str, Any],
) -> bool:
    choices = response.get("choices")
    choice = (
        choices[0]
        if isinstance(choices, Sequence)
        and not isinstance(choices, (str, bytes))
        and choices
        and isinstance(choices[0], Mapping)
        else {}
    )
    return (
        choice.get("finish_reason") in {"length", "stop"}
        and _textual_content(message.get("content")) is None
        and _textual_content(message.get("reasoning_content")) is None
    )


def _retryable_missing_tool_call(response: Mapping[str, Any]) -> bool:
    choices = response.get("choices")
    choice = (
        choices[0]
        if isinstance(choices, Sequence)
        and not isinstance(choices, (str, bytes))
        and choices
        and isinstance(choices[0], Mapping)
        else {}
    )
    return choice.get("finish_reason") in {"length", "stop", "tool_calls"}


def _visible_output_retry_payload(
    payload: Mapping[str, Any],
    *,
    require_tool: bool,
) -> dict[str, Any]:
    copied = json.loads(json.dumps(dict(payload)))
    messages = copied.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RuntimeError("chat payload has no messages to repair")
    user_message = messages[-1]
    if not isinstance(user_message, dict):
        raise RuntimeError("chat payload has an invalid user message")
    content = user_message.get("content")
    if not isinstance(content, str):
        raise RuntimeError("chat payload user content must be text")
    if require_tool:
        suffix = (
            "\n\nThe previous attempt returned no visible output. Call the "
            "required tool now and provide non-empty JSON arguments."
        )
    else:
        suffix = (
            "\n\nThe previous attempt returned no visible output. Return a "
            "non-empty final answer now."
        )
    user_message["content"] = content + suffix
    return copied


_FORCED_TOOL_FALLBACK_MODES = frozenset(
    {"disabled", "validated_content_then_prompt_json"}
)


def _strict_json_object(text: str) -> Mapping[str, Any] | None:
    """Decode one JSON object, allowing only prose outside its outer braces."""

    candidate = text.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3:
            candidate = "\n".join(lines[1:-1]).strip()
    candidates = [candidate]
    first = candidate.find("{")
    last = candidate.rfind("}")
    if 0 <= first < last:
        candidates.append(candidate[first : last + 1])
    for fragment in candidates:
        try:
            value = json.loads(fragment)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            return value
    return None


def _json_schema_validation_error(
    value: Any,
    schema: Mapping[str, Any],
    *,
    path: str = "$",
) -> str | None:
    """Validate the bounded JSON-Schema subset used by forced tools."""

    if "enum" in schema:
        enum = schema["enum"]
        if not isinstance(enum, Sequence) or isinstance(enum, (str, bytes)):
            return f"{path}: schema enum must be an array"
        if value not in enum:
            return f"{path}: value is outside enum"
    if "const" in schema and value != schema["const"]:
        return f"{path}: value does not match const"

    expected = schema.get("type")
    if expected == "object":
        if not isinstance(value, Mapping):
            return f"{path}: expected object"
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            return f"{path}: schema properties must be an object"
        required = schema.get("required", ())
        if not isinstance(required, Sequence) or isinstance(required, (str, bytes)):
            return f"{path}: schema required must be an array"
        missing = [key for key in required if key not in value]
        if missing:
            return f"{path}: missing required fields {missing}"
        additional = schema.get("additionalProperties", True)
        extras = set(value) - set(properties)
        if additional is False and extras:
            return f"{path}: unexpected fields {sorted(extras)}"
        for key, child in properties.items():
            if key not in value:
                continue
            if not isinstance(child, Mapping):
                return f"{path}.{key}: property schema must be an object"
            error = _json_schema_validation_error(
                value[key], child, path=f"{path}.{key}"
            )
            if error is not None:
                return error
        if isinstance(additional, Mapping):
            for key in extras:
                error = _json_schema_validation_error(
                    value[key], additional, path=f"{path}.{key}"
                )
                if error is not None:
                    return error
    elif expected == "array":
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            return f"{path}: expected array"
        minimum = schema.get("minItems")
        maximum = schema.get("maxItems")
        if minimum is not None and len(value) < int(minimum):
            return f"{path}: array is shorter than minItems"
        if maximum is not None and len(value) > int(maximum):
            return f"{path}: array is longer than maxItems"
        items = schema.get("items")
        if isinstance(items, Mapping):
            for index, item in enumerate(value):
                error = _json_schema_validation_error(
                    item, items, path=f"{path}[{index}]"
                )
                if error is not None:
                    return error
    elif expected == "string":
        if not isinstance(value, str):
            return f"{path}: expected string"
        if "minLength" in schema and len(value) < int(schema["minLength"]):
            return f"{path}: string is shorter than minLength"
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            return f"{path}: string is longer than maxLength"
    elif expected == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            return f"{path}: expected integer"
    elif expected == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"{path}: expected number"
        if not math.isfinite(float(value)):
            return f"{path}: number must be finite"
    elif expected == "boolean":
        if not isinstance(value, bool):
            return f"{path}: expected boolean"
    elif expected == "null":
        if value is not None:
            return f"{path}: expected null"
    elif expected is not None:
        return f"{path}: unsupported schema type {expected!r}"

    if expected in {"integer", "number"}:
        if "minimum" in schema and value < schema["minimum"]:
            return f"{path}: number is below minimum"
        if "maximum" in schema and value > schema["maximum"]:
            return f"{path}: number is above maximum"
    return None


def _validated_tool_content(
    content: Any,
    parameters: Mapping[str, Any],
) -> str | None:
    text = _textual_content(content)
    if text is None:
        return None
    value = _strict_json_object(text)
    if value is None:
        return None
    if _json_schema_validation_error(value, parameters) is not None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _prompt_json_tool_payload(
    payload: Mapping[str, Any],
    *,
    tool_name: str,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    copied = json.loads(json.dumps(dict(payload)))
    copied.pop("tools", None)
    copied.pop("tool_choice", None)
    messages = copied.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RuntimeError("chat payload has no messages for JSON fallback")
    user_message = messages[-1]
    if not isinstance(user_message, dict) or not isinstance(
        user_message.get("content"), str
    ):
        raise RuntimeError("chat payload has an invalid user message")
    schema = json.dumps(parameters, ensure_ascii=False, sort_keys=True)
    user_message["content"] += (
        "\n\nThe function-call transport was not honored. Return only the JSON "
        f"arguments for `{tool_name}` matching this schema exactly: {schema}"
    )
    return copied


class OpenAICompatibleProvider:
    """Small dependency-free client for local vLLM OpenAI-compatible servers."""

    def __init__(
        self,
        *,
        generation_base_url: str,
        generation_model: str,
        generation_api_key: str | None = None,
        embedding_base_url: str | None = None,
        embedding_model: str | None = None,
        timeout: float = 180.0,
        retries: int = 3,
        chat_response_format: Mapping[str, Any] | None = None,
        chat_template_kwargs: Mapping[str, Any] | None = None,
        chat_request_overrides: Mapping[str, Any] | None = None,
        empty_content_retries: int = 0,
        omit_temperature: bool = False,
        request_lock_path: str | None = None,
        forced_tool_fallback_mode: str = "disabled",
    ) -> None:
        self.generation_base_url = generation_base_url.rstrip("/")
        self.generation_model = generation_model
        if generation_api_key is None:
            generation_api_key = os.environ.get(
                "TRACE_ACTOR_API_KEY"
            )
        if generation_api_key is not None and any(
            value in generation_api_key for value in ("\r", "\n")
        ):
            raise ValueError("generation API key contains invalid header characters")
        self._generation_api_key = (
            generation_api_key.strip() if generation_api_key else None
        )
        self.embedding_base_url = (
            embedding_base_url.rstrip("/") if embedding_base_url else None
        )
        self.embedding_model = embedding_model
        self.timeout = timeout
        self.retries = retries
        if empty_content_retries < 0:
            raise ValueError("empty_content_retries must be non-negative")
        self.empty_content_retries = int(empty_content_retries)
        self.omit_temperature = bool(omit_temperature)
        self.request_lock_path = _resolve_generation_request_lock_path(
            request_lock_path
        )
        if forced_tool_fallback_mode not in _FORCED_TOOL_FALLBACK_MODES:
            raise ValueError(
                "forced_tool_fallback_mode must be one of: "
                + ", ".join(sorted(_FORCED_TOOL_FALLBACK_MODES))
            )
        self.forced_tool_fallback_mode = forced_tool_fallback_mode
        self.chat_response_format = _copy_response_format(chat_response_format)
        self.chat_template_kwargs = (
            dict(chat_template_kwargs) if chat_template_kwargs is not None else None
        )
        self.chat_request_overrides = _copy_chat_request_overrides(
            chat_request_overrides
        )

    def __repr__(self) -> str:
        return (
            "OpenAICompatibleProvider("
            f"origin={endpoint_origin(self.generation_base_url)!r}, "
            f"model={self.generation_model!r}, "
            f"secret_present={self._generation_api_key is not None})"
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._generation_api_key is not None:
            headers["Authorization"] = f"Bearer {self._generation_api_key}"
        return headers

    def _post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        encoded = json.dumps(payload).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self.retries):
            request = Request(
                url,
                data=encoded,
                headers=self._headers(),
                method="POST",
            )
            try:
                with _generation_request_slot(self.request_lock_path):
                    with urlopen(request, timeout=self.timeout) as response:
                        return json.loads(response.read().decode("utf-8"))
            except HTTPError as error:
                last_error = RuntimeError(self._safe_http_error_detail(error))
                if attempt + 1 < self.retries:
                    time.sleep(float(attempt + 1))
            except (
                URLError,
                TimeoutError,
                RemoteDisconnected,
                ConnectionResetError,
                BrokenPipeError,
                json.JSONDecodeError,
            ) as error:
                last_error = error
                if attempt + 1 < self.retries:
                    time.sleep(float(attempt + 1))
        raise RuntimeError(f"request failed after {self.retries} attempts: {last_error}")

    def _safe_http_error_detail(self, error: HTTPError) -> str:
        """Return bounded provider diagnostics without echoing request payloads."""

        status = getattr(error, "code", None)
        reason = str(getattr(error, "reason", "HTTP error"))[:160]
        parts = [f"HTTP {status}: {reason}"]
        try:
            raw = error.read(65536).decode("utf-8", errors="replace")
            envelope = json.loads(raw)
        except (AttributeError, OSError, UnicodeError, json.JSONDecodeError):
            envelope = None
        if isinstance(envelope, Mapping):
            detail = envelope.get("error", envelope)
            if isinstance(detail, Mapping):
                for key in ("type", "code", "message"):
                    value = detail.get(key)
                    if isinstance(value, (str, int, float, bool)):
                        text = str(value)
                        if self._generation_api_key:
                            text = text.replace(
                                self._generation_api_key,
                                "[REDACTED]",
                            )
                        parts.append(f"{key}={text[:500]!r}")
        return ", ".join(parts)

    def complete(
        self,
        prompt: str,
        *,
        system: str = JUDGE_SYSTEM_PROMPT,
        seed: int = 0,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> Completion:
        started = time.perf_counter()
        payload: dict[str, Any] = {
            "model": self.generation_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "seed": _generation_seed(self.generation_model, seed),
            "max_tokens": max_tokens,
        }
        if not self.omit_temperature:
            payload["temperature"] = temperature
        if self.chat_template_kwargs is not None:
            payload["chat_template_kwargs"] = dict(self.chat_template_kwargs)
        if self.chat_response_format is not None:
            payload["response_format"] = self.chat_response_format
        payload.update(self.chat_request_overrides)
        url = _openai_api_url(self.generation_base_url, "chat/completions")
        response: dict[str, Any] = {}
        message: Mapping[str, Any] = {}
        candidate: str | None = None
        request_payload = payload
        for response_attempt in range(self.empty_content_retries + 1):
            response = self._post(url, request_payload)
            message = response["choices"][0]["message"]
            candidate = _textual_content(message.get("content"))
            if candidate is None:
                # SGLang serves Qwen thinking-mode answers in this extension
                # field with OpenAI ``content=null``. Never turn null into the
                # literal string "None", which can be cached as a fake answer.
                candidate = _textual_content(message.get("reasoning_content"))
            if candidate is not None:
                break
            if (
                response_attempt >= self.empty_content_retries
                or not _retryable_empty_completion(response, message)
            ):
                raise RuntimeError(
                    "completion omitted textual content ("
                    + _completion_envelope_diagnostic(response, message)
                    + f", response_attempts={response_attempt + 1})"
                )
            request_payload = _visible_output_retry_payload(
                payload,
                require_tool=False,
            )
        assert candidate is not None
        usage = response.get("usage") or {}
        text = candidate
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        return Completion(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=int(
                usage.get("total_tokens", prompt_tokens + completion_tokens)
            ),
            latency_seconds=time.perf_counter() - started,
        )

    def complete_with_forced_tool(
        self,
        prompt: str,
        *,
        tool_name: str,
        tool_description: str,
        tool_parameters: Mapping[str, Any],
        system: str = JUDGE_SYSTEM_PROMPT,
        seed: int = 0,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> Completion:
        if not tool_name or not tool_name.strip():
            raise ValueError("tool_name must be non-empty")
        parameters = json.loads(json.dumps(dict(tool_parameters)))
        started = time.perf_counter()
        payload: dict[str, Any] = {
            "model": self.generation_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "seed": _generation_seed(self.generation_model, seed),
            "max_tokens": max_tokens,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": tool_description,
                        "parameters": parameters,
                    },
                }
            ],
            "tool_choice": {
                "type": "function",
                "function": {"name": tool_name},
            },
        }
        if not self.omit_temperature:
            payload["temperature"] = temperature
        if self.chat_template_kwargs is not None:
            payload["chat_template_kwargs"] = dict(self.chat_template_kwargs)
        payload.update(self.chat_request_overrides)
        url = _openai_api_url(self.generation_base_url, "chat/completions")
        response: dict[str, Any] = {}
        message: Mapping[str, Any] = {}
        function: Mapping[str, Any] = {}
        request_payload = payload
        transport_mode = "forced_tool"
        for response_attempt in range(self.empty_content_retries + 1):
            response = self._post(url, request_payload)
            message = response["choices"][0]["message"]
            tool_calls = message.get("tool_calls") or ()
            if tool_calls:
                function = tool_calls[0].get("function") or {}
            else:
                function = message.get("function_call") or {}
            if function:
                break
            if (
                self.forced_tool_fallback_mode
                == "validated_content_then_prompt_json"
            ):
                validated = _validated_tool_content(
                    message.get("content"), parameters
                )
                if validated is not None:
                    transport_mode = "validated_content_fallback"
                    function = {"name": tool_name, "arguments": validated}
                    break
                fallback_payload = _prompt_json_tool_payload(
                    payload,
                    tool_name=tool_name,
                    parameters=parameters,
                )
                fallback_response: dict[str, Any] = {}
                fallback_message: Mapping[str, Any] = {}
                for fallback_attempt in range(
                    self.empty_content_retries + 1
                ):
                    fallback_response = self._post(url, fallback_payload)
                    fallback_message = fallback_response["choices"][0][
                        "message"
                    ]
                    validated = _validated_tool_content(
                        fallback_message.get("content"), parameters
                    )
                    if validated is not None:
                        response = fallback_response
                        message = fallback_message
                        transport_mode = "validated_content_fallback"
                        function = {
                            "name": tool_name,
                            "arguments": validated,
                        }
                        break
                    if fallback_attempt < self.empty_content_retries:
                        fallback_payload = _visible_output_retry_payload(
                            fallback_payload,
                            require_tool=False,
                        )
                if function:
                    break
                raise RuntimeError(
                    "forced tool and validated content fallback failed ("
                    + _completion_envelope_diagnostic(
                        fallback_response, fallback_message
                    )
                    + f", response_attempts={fallback_attempt + 2})"
                )
            if (
                response_attempt >= self.empty_content_retries
                or not _retryable_missing_tool_call(response)
            ):
                raise RuntimeError(
                    "forced tool response omitted a tool call ("
                    + _completion_envelope_diagnostic(response, message)
                    + f", response_attempts={response_attempt + 1})"
                )
            request_payload = _visible_output_retry_payload(
                payload,
                require_tool=True,
            )
        if function.get("name") != tool_name:
            raise RuntimeError("forced tool response used an unexpected function")
        arguments = function.get("arguments")
        if isinstance(arguments, Mapping):
            text = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
        elif isinstance(arguments, str) and arguments.strip():
            text = arguments
        else:
            raise RuntimeError("forced tool response omitted JSON arguments")
        usage = response.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        return Completion(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=int(
                usage.get("total_tokens", prompt_tokens + completion_tokens)
            ),
            latency_seconds=time.perf_counter() - started,
            transport_mode=transport_mode,
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if self.embedding_base_url is None or self.embedding_model is None:
            raise RuntimeError("embedding endpoint/model is not configured")
        response = self._post(
            _openai_api_url(self.embedding_base_url, "embeddings"),
            {"model": self.embedding_model, "input": list(texts)},
        )
        rows = sorted(response["data"], key=lambda item: int(item["index"]))
        return [[float(value) for value in item["embedding"]] for item in rows]

    def health(self) -> dict[str, Any]:
        request = Request(
            _openai_api_url(self.generation_base_url, "models"),
            headers=self._headers(),
            method="GET",
        )
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=min(self.timeout, 10.0)) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (
            HTTPError,
            URLError,
            TimeoutError,
            RemoteDisconnected,
            ConnectionResetError,
            BrokenPipeError,
            json.JSONDecodeError,
        ) as error:
            return {"ok": False, "error": str(error)}
        return {
            "ok": True,
            "latency_seconds": time.perf_counter() - started,
            "models": [item.get("id") for item in payload.get("data", [])],
        }


class OpenAICompatibleEmbeddingClient:
    """Independent OpenAI-compatible embedding client.

    Keeping this client separate from the generation provider prevents actor
    credentials from being forwarded to a local embedding endpoint.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 180.0,
        retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_id = model.strip()
        if not self.model_id:
            raise ValueError("embedding model must be non-empty")
        if api_key is not None and any(
            value in api_key for value in ("\r", "\n")
        ):
            raise ValueError("embedding API key contains invalid header characters")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if retries < 1:
            raise ValueError("retries must be at least one")
        self._api_key = api_key.strip() if api_key else None
        self.timeout = float(timeout)
        self.retries = int(retries)

    def __repr__(self) -> str:
        return (
            "OpenAICompatibleEmbeddingClient("
            f"origin={endpoint_origin(self.base_url)!r}, "
            f"model={self.model_id!r}, "
            f"secret_present={self._api_key is not None})"
        )

    def safe_metadata(self) -> dict[str, Any]:
        return {
            "mode": "external",
            "endpoint_origin": endpoint_origin(self.base_url),
            "model": self.model_id,
            "secret_present": self._api_key is not None,
        }

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = {
            "model": self.model_id,
            "input": [str(text) for text in texts],
        }
        encoded = json.dumps(payload).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self.retries):
            request = Request(
                _openai_api_url(self.base_url, "embeddings"),
                data=encoded,
                headers=self._headers(),
                method="POST",
            )
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    decoded = json.loads(response.read().decode("utf-8"))
                rows = sorted(
                    decoded["data"], key=lambda item: int(item["index"])
                )
                vectors = [
                    [float(value) for value in item["embedding"]]
                    for item in rows
                ]
                if len(vectors) != len(texts):
                    raise ValueError("embedding response count mismatch")
                return vectors
            except (
                HTTPError,
                URLError,
                TimeoutError,
                RemoteDisconnected,
                ConnectionResetError,
                BrokenPipeError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
            ) as error:
                last_error = error
                if attempt + 1 < self.retries:
                    time.sleep(float(attempt + 1))
        raise RuntimeError(
            f"embedding request failed after {self.retries} attempts: "
            f"{type(last_error).__name__ if last_error else 'unknown_error'}"
        )

    def health(self) -> dict[str, Any]:
        request = Request(
            _openai_api_url(self.base_url, "models"),
            headers=self._headers(),
            method="GET",
        )
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=min(self.timeout, 10.0)) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (
            HTTPError,
            URLError,
            TimeoutError,
            RemoteDisconnected,
            ConnectionResetError,
            BrokenPipeError,
            json.JSONDecodeError,
        ) as error:
            return {"ok": False, "error": str(error)}
        return {
            "ok": True,
            "latency_seconds": time.perf_counter() - started,
            "models": [item.get("id") for item in payload.get("data", [])],
        }
