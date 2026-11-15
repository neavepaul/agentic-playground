"""The persona layer: deterministic state, bounded prompt views, and safe injection."""
import json

import pytest

from app.agents import prompts
from app.config import Settings
from app.events.bus import EventBus
from app.memory.store import PersistentMemory
from app.persona import Persona
from app.persona.engine import FOLLOWUP_DELAY, VOICE_VIEW_LIMIT
from app.persona.profile import BASELINE, DRIFT
from app.persona.state import Followup
from app.tasks.manager import TaskManager
from tests.fakes import ScriptedLLM, tool


class Clock:
    def __init__(self):
        self.seconds = 8 * 3600.0

    def __call__(self):
        return self.seconds


def event(kind, **data):
    return {"type": kind, "data": data}


def chat(text, person="neave"):
    return event("chat_message", turn={"role": "user", "text": text, "speaker": person})


def check_in_later(persona, person="neave", reason="they had a cold"):
    """A check-in carried in Kip's saved state, due a couple of hours from now."""
    now = persona.state.age_seconds
    persona.state.followups.append(Followup(person=person, reason=reason, due_at=now + FOLLOWUP_DELAY,
                                            expires_at=now + 24 * 3600))


def test_chat_builds_a_relationship_lifts_mood_and_a_due_check_in_settles_when_asked():
    clock = Clock()
    persona = Persona(now=clock)
    before = persona.state.mood.valence
    persona.observe(chat("haha, good one"))
    bond = persona.state.relationships["neave"]
    assert bond.chats == 1 and bond.warmth > 0.5 and bond.familiarity > 0 and bond.humour_reception > 0.5
    assert persona.state.mood.valence > before
    assert persona.state.episodes[0].summary == "Met neave for the first time."
    check_in_later(persona)
    assert persona.due_followups() == []
    clock.seconds += FOLLOWUP_DELAY
    assert [f.person for f in persona.due_followups()] == ["neave"]
    # A reply that does not ask after them leaves it due; one that does settles it.
    persona.observe(event("chat_message", turn={"role": "kip", "text": "Nice weather.", "to": "neave"}))
    assert [f.person for f in persona.due_followups()] == ["neave"]
    persona.observe(event("chat_message", turn={"role": "kip", "text": "How are you?", "to": "neave"}))
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
        persona.observe(event("tool_failed", task_id="t1", tool="web_search", error="x"))
    assert persona.state.mood.valence == pytest.approx(0.2 - 5 * 0.02)


def test_mood_decays_on_its_clock_and_survives_clock_resets():
    clock = Clock()
    persona = Persona(now=clock)
    persona.observe(event("task_failed", task={"goal": "x"}))
    low = persona.state.mood.valence
    clock.seconds -= 5000  # the clock set back: time must not run backwards
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
    persona.observe(chat("Morning, Kip."))
    memory.save(path)
    assert PersistentMemory.load(path).persona.relationships["neave"].chats == 1


def test_prompt_views_stay_small():
    persona = Persona(now=Clock())
    for index in range(30):
        persona.observe(chat("I'm stressed about a very long list of things " * 3, person=f"person_{index}"))
    assert len(json.dumps(persona.voice_view([f"person_{i}" for i in range(30)]))) <= VOICE_VIEW_LIMIT
    assert persona.voice_view()["voice"]


def test_bus_listener_errors_never_break_emitting():
    bus = EventBus()
    bus.add_listener(lambda e: 1 / 0)
    assert bus.emit("task_started", summary="ok")["type"] == "task_started"


def test_persona_text_stays_out_of_static_planning_prompts():
    for name in ("COORDINATOR", "CRITIC", "GOAL_PLANNER", "CAPABILITY_EXPLORER"):
        text = getattr(prompts, name, "")
        assert "persona" not in text.casefold(), name


@pytest.fixture
def web(monkeypatch):
    from app.capabilities import tools as capabilities
    monkeypatch.setattr(capabilities, "web_search", lambda query: {
        "query": query, "results": [{"url": "https://weather.example", "title": "Weather",
                                     "snippet": "Sunny, 31 degrees."}]})


def _manager(client, persona):
    bus = EventBus()
    if persona:
        persona.attach(bus)
    return TaskManager(client, bus, Settings(), persona=persona), bus


def _answer(text):
    def answer(payload):
        ids = [o["evidence_id"] for o in payload["tool_observations"] if o["success"]]
        return {"summary": "Ready.", "action": {"tool": "finish_answer", "answer": text, "evidence_ids": ids}}
    return answer


@pytest.mark.asyncio
async def test_persona_voices_the_answer_and_stays_out_of_planning_and_tool_choice(web):
    persona = Persona(now=Clock())
    styled = "Sunny. 31 degrees. Hat weather, I think."
    fake = ScriptedLLM([tool("web_search", query="weather today"), _answer(styled)])
    mgr, bus = _manager(fake, persona)
    messages = []
    bus.add_listener(lambda e: messages.append(e) if e["type"] == "agent_message" else None)
    task = mgr.start("Search the web for today's weather")
    await mgr.runner
    assert task.status == "completed", task.summary
    assert task.answer["answer"] == styled
    with_persona = [(m, s) for m, s in fake.calls if "persona" in json.loads(m[1]["content"])]
    assert len(with_persona) == 1, "only the decision that can write the answer carries the persona"
    for sent, schema in with_persona:
        assert "finish_answer" in json.dumps(schema)
        assert "character's voice" in sent[0]["content"]
    choosing = [(m, s) for m, s in fake.calls if "finish_answer" not in json.dumps(s)]
    assert choosing and all("persona" not in json.loads(m[1]["content"]) for m, _ in choosing)
    # The outcome is appraised, but no canned reaction line is broadcast after it.
    assert not any(m["agent"] == "Kip" for m in messages)
    assert persona.state.skills["looking things up"].successes == 1


@pytest.mark.asyncio
async def test_without_persona_prompts_and_payloads_are_unchanged(web):
    fake = ScriptedLLM([tool("web_search", query="weather today"), _answer("Sunny, 31 degrees.")])
    mgr, _ = _manager(fake, None)
    task = mgr.start("Search the web for today's weather")
    await mgr.runner
    assert task.status == "completed", task.summary
    for sent, _ in fake.calls:
        assert "persona" not in json.loads(sent[1]["content"])
        assert "persona" not in sent[0]["content"]
        assert "character" not in sent[0]["content"]
