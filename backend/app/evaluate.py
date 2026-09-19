"""Run with python -m app.evaluate [--mission 1]. Requires local Ollama."""
import argparse
import asyncio
import json
import logging
import time

from app.config import Settings
from app.events.bus import EventBus
from app.llm.ollama import OllamaClient
from app.tasks.manager import TaskManager
from app.world.engine import WorldEngine
from app.world.tools import WorldTools

MISSIONS = {
    1: "Find the charger.",
    2: "Find Neave and tell him dinner is ready.",
    3: "Find out who needs the charger and deliver it.",
    4: "Find the missing keys.",
}


def mission_satisfied(number: int, context, world: dict) -> bool:
    successful = [a for a in context.action_history if a["success"]]
    if number in (1, 4):
        item = "charger" if number == 1 else "keys"
        return any(a["tool"] == "look" and any(o["id"] == item for o in a["observation"]["objects"])
                   for a in successful)
    if number == 2:
        return any(a["tool"] == "talk_to" and a["observation"]["person"] == "neave"
                   and "dinner" in a["observation"]["message"].lower()
                   and "ready" in a["observation"]["message"].lower() for a in successful)
    if number == 3:
        need = any(a["tool"] == "talk_to" and a["observation"]["person"] in {"dad", "neave"}
                   and "charger" in a["observation"]["message"].lower() for a in successful)
        return need and world["objects"]["charger"]["location"] == {"kind": "person", "id": "neave"}
    return False


async def evaluate(numbers: list[int]) -> bool:
    settings = Settings()
    client = OllamaClient(settings)
    results = []
    try:
        for number in numbers:
            started = time.monotonic()
            engine, bus = WorldEngine(), EventBus()
            manager = TaskManager(client, WorldTools(engine, bus), bus, settings)
            task = manager.start(MISSIONS[number])
            await manager.runner
            passed = task.status == "completed" and mission_satisfied(number, task, engine.snapshot())
            results.append(passed)
            print(json.dumps({"mission": number, "passed": passed, "status": task.status,
                              "summary": task.summary, "tools": task.tool_count,
                              "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
    finally:
        await client.close()
    return all(results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument("--mission", type=int, choices=list(MISSIONS))
    args = parser.parse_args()
    raise SystemExit(0 if asyncio.run(evaluate([args.mission] if args.mission else list(MISSIONS))) else 1)
