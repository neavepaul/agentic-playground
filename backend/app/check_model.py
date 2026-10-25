"""Small structured-output smoke test: python -m app.check_model."""
import asyncio

from app.agents.schemas import CriticReview
from app.config import Settings
from app.llm.base import structured
from app.llm.factory import create_client


async def main():
    client = create_client(Settings())
    try:
        response = await structured(client, CriticReview,
                                    "Return approved=true and summary=Ready as JSON. No reasoning.", {})
        print(response.model_dump_json())
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
