import json
import logging
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


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


async def structured(client: LLMClient, schema: type[T], prompt: str, context: dict,
                     repair_hint: str = "") -> T:
    output_schema = schema.model_json_schema()
    messages = [{"role": "system", "content": prompt + "\nOutput JSON schema:\n" + json.dumps(output_schema)},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False)}]
    for attempt in range(2):
        raw = await client.generate(messages, output_schema)
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
