"""Small structured-output smoke test: python -m app.check_model."""
import asyncio

from app.agents.schemas import CriticReview
from app.config import Settings
from app.llm.base import structured
from app.llm.ollama import OllamaClient


async def main():
    client = OllamaClient(Settings())
    try:
        response = await structured(client, CriticReview,
                                    "Return approved=true and summary=Ready as JSON. No reasoning.", {})
        print(response.model_dump_json())
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
