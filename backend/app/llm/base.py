import json
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class ModelError(RuntimeError):
    pass


class LLMClient(Protocol):
    async def generate(self, messages: list[dict], response_schema: dict) -> str: ...


async def structured(client: LLMClient, schema: type[T], prompt: str, context: dict,
                     repair_hint: str = "") -> T:
    output_schema = schema.model_json_schema()
    messages = [{"role": "system", "content": prompt + "\nOutput JSON schema:\n" + json.dumps(output_schema)},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False)}]
    for attempt in range(2):
        raw = await client.generate(messages, output_schema)
        try:
            return schema.model_validate_json(raw)
        except (ValidationError, ValueError, TypeError):
            # Never retain/log invalid raw content: it might contain private reasoning.
            if attempt:
                raise ModelError("Model returned invalid structured output twice.") from None
            messages.append({"role": "user", "content":
                             "Your response did not match the schema. Retry with one valid JSON object. "
                             "Use only the specified fields and brief public operational summaries. "
                             "Do not include thinking, analysis, markdown or a scratchpad. " + repair_hint})
    raise ModelError("No valid response.")
