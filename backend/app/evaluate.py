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
from app.world.clock import WorldClock
from app.world.engine import WorldEngine
from app.world.tools import WorldTools

from pathlib import Path
from app.agents.schemas import GoalCondition
from app.tasks.goals import check_conditions

SCENARIOS = {int(key): value for key, value in json.loads(
    (Path(__file__).resolve().parents[1] / "evaluations" / "missions.json").read_text(encoding="utf-8")
).items()}
MISSIONS = {key: scenario["goal"] for key, scenario in SCENARIOS.items()}


def mission_satisfied(number: int, context, world: dict) -> bool:
    scenario = SCENARIOS.get(number)
    if scenario is None:
        return False
    expected = context.model_copy(update={
        "conditions": [GoalCondition.model_validate(c) for c in scenario["conditions"]]
    })
    return all(c["satisfied"] for c in check_conditions(expected)) and all(
        world["objects"].get(key, {}).get("location") == location
        for key, location in scenario.get("object_locations", {}).items()
    )

async def evaluate(numbers: list[int]) -> bool:
    settings = Settings()
    client = OllamaClient(settings)
    results = []
    try:
        for number in numbers:
            started = time.monotonic()
            clock = WorldClock(speed=settings.clock_speed, start_hour=settings.clock_start_hour,
                               mode=settings.simulation_mode, action_seconds=settings.action_seconds)
            engine, bus = WorldEngine(settings.world_file, clock=clock), EventBus()
            manager = TaskManager(client, WorldTools(engine, bus), bus, settings)
            task = manager.start(MISSIONS[number])
            await manager.runner
            passed = task.status == "completed" and mission_satisfied(number, task, engine.snapshot())
            results.append(passed)
            metrics = task.metrics
            print(json.dumps({"mission": number, "passed": passed, "status": task.status,
                              "summary": task.summary, "tools": task.tool_count,
                              "elapsed_seconds": round(time.monotonic() - started, 1),
                              "simulation_mode": settings.simulation_mode,
                              # The headline number: inference calls per completed task.
                              "llm_calls": metrics.llm_calls,
                              "llm_seconds": round(metrics.llm_seconds, 1),
                              "coordinator_llm_calls": metrics.coordinator_llm_calls,
                              "explorer_llm_calls": metrics.explorer_llm_calls,
                              "critic_llm_calls": metrics.critic_llm_calls,
                              "conversation_llm_calls": metrics.conversation_llm_calls,
                              "reflex_actions": metrics.reflex_actions,
                              "route_actions": metrics.route_actions,
                              "useful_observations": metrics.useful_observations,
                              "repeat_room_visits": metrics.repeat_room_visits,
                              "stall_events": metrics.stall_events,
                              "coordinator_handoffs": metrics.coordinator_handoffs,
                              "simulated_start": metrics.simulated_start,
                              "simulated_end": metrics.simulated_end,
                              "prompt_tokens": metrics.prompt_tokens,
                              "completion_tokens": metrics.completion_tokens}), flush=True)
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
