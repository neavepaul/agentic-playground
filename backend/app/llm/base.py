import json
import logging
import time
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

_metrics_log = logging.getLogger("agentic_friend.llm.metrics")

_ROLE_COUNTERS = {"coordinator": "coordinator_llm_calls", "explorer": "explorer_llm_calls",
                  "critic": "critic_llm_calls", "conversation": "conversation_llm_calls"}


class ModelError(RuntimeError):
    pass


class LLMClient(Protocol):
    async def generate(self, messages: list[dict], response_schema: dict) -> str: ...


def validation_summary(schema: type[BaseModel], error: ValidationError) -> str:
    """Only schema field names and error codes; never rejected model content."""
    issues = []
    for issue in error.errors(include_input=False, include_context=False, include_url=False)[:4]:
        location = issue["loc"]
        field = location[0] if location and location[0] in schema.model_fields else "response"
        code = issue["type"]
        if not code.replace("_", "").isalnum() or len(code) > 60:
            code = "invalid_output"
        issues.append(f"{field}: {code}")
    return "; ".join(issues)


def _record_metrics(client, metrics, role: str, schema_name: str, seconds: float) -> None:
    """Fold one model call into the task counters and emit a measurable log line.

    Token and duration fields come straight from the backend response when it
    supplies them; nothing is synthesised when it does not.
    """
    reported = getattr(client, "last_metrics", None) or {}
    if metrics is not None:
        metrics.llm_calls += 1
        metrics.llm_seconds += round(seconds, 3)
        metrics.prompt_tokens += reported.get("prompt_tokens") or 0
        metrics.completion_tokens += reported.get("completion_tokens") or 0
        counter = _ROLE_COUNTERS.get(role)
        if counter:
            setattr(metrics, counter, getattr(metrics, counter) + 1)
    rate = reported.get("completion_tokens", 0) / reported.get("eval_seconds", 0) \
        if reported.get("eval_seconds") else None
    _metrics_log.info(
        "llm_call role=%s schema=%s wall_seconds=%.2f %s",
        role or "unknown", schema_name, seconds,
        " ".join(f"{key}={value}" for key, value in
                 {**reported, "tokens_per_second": round(rate, 1) if rate else None}.items()
                 if value is not None))


async def structured(client: LLMClient, schema: type[T], prompt: str, context: dict,
                     repair_hint: str = "", metrics=None, role: str = "") -> T:
    output_schema = schema.model_json_schema()
    messages = [{"role": "system", "content": prompt + "\nOutput JSON schema:\n" + json.dumps(output_schema)},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False)}]
    for attempt in range(2):
        started = time.monotonic()
        raw = await client.generate(messages, output_schema)
        _record_metrics(client, metrics, role, schema.__name__, time.monotonic() - started)
        try:
            return schema.model_validate_json(raw)
        except (ValidationError, ValueError, TypeError) as exc:
            # Never retain/log invalid raw content: it might contain private reasoning.
            details = validation_summary(schema, exc) if isinstance(exc, ValidationError) else "response: invalid_output"
            logging.getLogger("agentic_friend.llm").warning(
                "%s validation attempt %d failed: %s", schema.__name__, attempt + 1, details)
            if attempt:
                raise ModelError(f"Model returned invalid structured output twice. {schema.__name__}: {details}.") from None
            messages.append({"role": "user", "content":
                             f"Your response did not match the schema ({details}). Retry with one valid JSON object. "
                             "Use only the specified fields and brief public operational summaries. "
                             "Do not include thinking, analysis, markdown or a scratchpad. " + repair_hint})
    raise ModelError("No valid response.")
