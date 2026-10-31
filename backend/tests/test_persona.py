"""The persona layer: deterministic state, bounded prompt views, and safe injection."""
import json
from pathlib import Path

import pytest

from app.agents import prompts
from app.config import Settings
from app.events.bus import EventBus
from app.memory.store import PersistentMemory
from app.mind.agent import AgentMind, _INTENTION_GENERATOR
from app.persona import Persona
from app.persona.engine import FOLLOWUP_DELAY, VIEW_LIMIT, VOICE_VIEW_LIMIT
from app.persona.profile import BASELINE, DRIFT
from app.tasks.manager import TaskManager
from app.world.engine import WorldEngine
from app.world.tools import WorldTools
from tests.fakes import ScriptedLLM, complete, tool

LEGACY_WORLD = Path(__file__).parent / "fixtures" / "legacy_house.json"


class Clock:
    def __init__(self):
        self.seconds = 8 * 3600.0

    def __call__(self):
        return self.seconds


def event(kind, **data):
    return {"type": kind, "data": data}


def given(person="neave", thing="medicine"):
    return event("object_given", arguments={"object": thing, "person": person})


def test_delivery_builds_a_relationship_lifts_mood_and_plans_a_check_in():
    clock = Clock()
    persona = Persona(now=clock)
    before = persona.state.mood.valence
    persona.observe(given())
    bond = persona.state.relationships["neave"]
    assert bond.times_helped == 1 and bond.warmth > 0.5 and bond.familiarity > 0
    assert bond.moments == ["brought them the medicine"]
    assert persona.state.mood.valence > before
    assert persona.state.episodes[0].summary == "Met neave for the first time."
    assert persona.due_followups() == []
    clock.seconds += FOLLOWUP_DELAY
    assert [f.person for f in persona.due_followups()] == ["neave"]
    # Talking to them once the check-in is due settles it.
    persona.observe(event("person_spoken_to", arguments={"person": "neave", "message": "How are you?"}))
    assert persona.due_followups() == []


def test_failures_make_it_cautious_and_methodical_but_traits_stay_bounded():
    persona = Persona(now=Clock())
    for _ in range(200):
        persona.observe(event("task_failed", task={"goal": "Find the keys"}))
    assert persona.mood_label() in {"determined", "concerned", "tired"}
    assert persona.trait("caution") == pytest.approx(BASELINE["caution"] + DRIFT["caution"])
    assert persona.trait("honesty") == BASELINE["honesty"]
    assert persona.state.episodes[-1].summary == "Could not finish: Find the keys"
    assert len(persona.state.episodes) == 12
    assert not hasattr(persona, "reaction")  # no stock reaction lines


def test_tool_failure_dips_are_capped_per_task():
    persona = Persona(now=Clock())
    for _ in range(50):
        persona.observe(event("tool_failed", task_id="t1", tool="look", error="x"))
    assert persona.state.mood.valence == pytest.approx(0.2 - 5 * 0.02)


def test_mood_decays_on_the_simulated_clock_and_survives_clock_resets():
    clock = Clock()
    persona = Persona(now=clock)
    persona.observe(event("task_failed", task={"goal": "x"}))
    low = persona.state.mood.valence
    clock.seconds -= 5000  # world reset: time must not run backwards
    persona.due_followups()
    assert persona.state.mood.valence == pytest.approx(low)
    clock.seconds += 4 * 1800
    persona.due_followups()
    # Drifts back toward its resting point (0.1) rather than staying where the failure left it.
    assert abs(persona.state.mood.valence - 0.1) < abs(low - 0.1) / 10


def test_state_persists_and_old_memory_files_still_load(tmp_path):
    path = tmp_path / "mem.json"
    path.write_text(json.dumps({"version": 1, "people": {}}), encoding="utf-8")
    memory = PersistentMemory.load(path)
    assert memory.persona.traits == BASELINE
    persona = Persona(memory.persona, now=Clock())
    persona.observe(given())
    memory.save(path)
    assert PersistentMemory.load(path).persona.relationships["neave"].times_helped == 1


def test_prompt_views_stay_small():
    persona = Persona(now=Clock())
    for index in range(30):
        persona.observe(given(person=f"person_{index}", thing="a very long object name " * 3))
    assert len(json.dumps(persona.mind_view())) <= VIEW_LIMIT
    assert len(json.dumps(persona.voice_view([f"person_{i}" for i in range(30)]))) <= VOICE_VIEW_LIMIT
    assert persona.voice_view()["voice"]


def test_empathy_lowers_and_low_mood_raises_the_initiative_threshold():
    persona = Persona(now=Clock())
    relaxed = persona.intention_threshold(0.6)
    assert relaxed < 0.6
    for _ in range(4):
        persona.observe(event("task_failed", task={"goal": "x"}))
    assert persona.intention_threshold(0.6) > relaxed
    assert 0 <= persona.helpfulness() <= 1


def test_bus_listener_errors_never_break_emitting():
    bus = EventBus()
    bus.add_listener(lambda e: 1 / 0)
    assert bus.emit("task_started", summary="ok")["type"] == "task_started"


def test_persona_text_stays_out_of_static_planning_prompts():
    for name in ("COORDINATOR", "CRITIC", "GOAL_PLANNER", "EXPLORER", "CAPABILITY_EXPLORER"):
        text = getattr(prompts, name, "")
        assert "persona" not in text.casefold(), name
    assert "persona" not in _INTENTION_GENERATOR.casefold()


def _manager(client, persona):
    engine, bus = WorldEngine(LEGACY_WORLD), EventBus()
    if persona:
        persona.attach(bus)
    return TaskManager(client, WorldTools(engine, bus), bus, Settings(), persona=persona), bus


@pytest.mark.asyncio
async def test_persona_voices_speech_keeps_dictated_text_and_stays_out_of_planning():
    persona = Persona(now=Clock())
    styled = "Hello Neave! Dinner is ready, and it smells wonderful. I hope you're hungry."
    fake = ScriptedLLM([
        {"action": "delegate", "summary": "Notify.", "task": "Tell Neave dinner is ready."},
        tool("look"), tool("move_to", room="bedroom"), tool("look"),
        tool("talk_to", person="neave", message=styled),
        {"action": "report", "summary": "Message delivered."}, complete,
        {"approved": True, "summary": "Notification evidenced."},
    ])
    mgr, bus = _manager(fake, persona)
    messages = []
    bus.add_listener(lambda e: messages.append(e) if e["type"] == "agent_message" else None)
    task = mgr.start("Tell Neave dinner is ready.")
    await mgr.runner
    assert task.status == "completed", task.summary
    with_persona = [(m, s) for m, s in fake.calls if "persona" in json.loads(m[1]["content"])]
    assert with_persona, "speech decision should carry the persona"
    for sent, schema in with_persona:
        assert "talk_to" in json.dumps(schema)
        assert "word for word" in sent[0]["content"]
    for sent, schema in fake.calls:
        if "action" in schema.get("properties", {}):  # Coordinator decisions
            assert "persona" not in json.loads(sent[1]["content"])
    assert "neave" in persona.state.relationships
    # The outcome is appraised, but no canned reaction line is broadcast after it.
    assert not any(m["agent"] == "Kip" for m in messages)
    assert persona.state.skills["passing on messages"].successes == 1


@pytest.mark.asyncio
async def test_without_persona_prompts_and_payloads_are_unchanged():
    fake = ScriptedLLM([
        {"action": "delegate", "summary": "Notify.", "task": "Tell Neave dinner is ready."},
        tool("look"), tool("move_to", room="bedroom"), tool("look"),
        tool("talk_to", person="neave", message="Dinner is ready."),
        {"action": "report", "summary": "Message delivered."}, complete,
        {"approved": True, "summary": "Notification evidenced."},
    ])
    mgr, _ = _manager(fake, None)
    task = mgr.start("Tell Neave dinner is ready.")
    await mgr.runner
    assert task.status == "completed", task.summary
    for sent, _ in fake.calls:
        assert "persona" not in json.loads(sent[1]["content"])
        assert "persona" not in sent[0]["content"]


@pytest.mark.asyncio
async def test_idle_mind_uses_persona_for_initiative_and_memory_formation():
    seen = []

    class Model:
        async def generate(self, messages, schema):
            seen.append((schema["title"], messages))
            if schema["title"] == "IntentionOrIdle":
                return json.dumps({"intention": None})
            return json.dumps({"upsert_edges": [], "remove_edges": []})

    clock = Clock()
    persona = Persona(now=clock)
    engine, bus = WorldEngine(LEGACY_WORLD), EventBus()
    persona.attach(bus)
    settings = Settings(idle_reflection_interval=0.001)
    mgr = TaskManager(Model(), WorldTools(engine, bus), bus, settings, persona=persona)
    mind = AgentMind(Model(), mgr, engine, bus, settings, persona=persona)
    bus.emit("object_given", "explorer", arguments={"object": "medicine", "person": "neave"},
             summary="Gave the medicine to neave.")
    clock.seconds += FOLLOWUP_DELAY
    await mind._tick()
    titles = {title: messages for title, messages in seen}
    intention = json.loads(titles["IntentionOrIdle"][1]["content"])
    assert intention["persona"]["due_followups"][0]["person"] == "neave"
    assert "due_followups" in titles["IntentionOrIdle"][0]["content"]
    assert intention["drive_helpfulness"] == persona.helpfulness()
    assert set(titles) == {"IntentionOrIdle"}  # no model call maintains memory any more
