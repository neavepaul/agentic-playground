"""Conversation that is truthful about tools, permissions and observations, in Kip's voice.

Built from the 26 October live Colibri transcript. Every test drives the real ChatSession
and TaskManager with deterministic fake models and records what actually ran; nothing
touches stores outside the test sandbox.
"""
import asyncio
import json
from pathlib import Path

import pytest

from app.chat.evidence import capabilities, contradicted_presence, last_seen
from app.chat.intent import read_intent
from app.chat.prompt import CHAT_SYSTEM
from app.chat.session import ChatSession, describe_findings
from app.config import Settings
from app.events.bus import EventBus
from app.llm.priority import ConversationFirst
from app.memory.recall import MemoryStore
from app.memory.store import PersistentMemory, PersonMemory
from app.persona import Persona
from app.persona.profile import IDENTITY
from app.tasks.manager import TaskManager
from app.tasks.models import TaskContext
from app.tasks.permissions import Grant
from app.world.engine import WorldEngine
from app.world.tools import WorldTools
from tests.test_permissions import LEGACY_WORLD, Model, physical, said, settle

HOUSE_WORLD = Path(__file__).resolve().parents[1] / "worlds" / "house.json"


class PlanningModel(Model):
    """Model that can also plan initial tool calls, as the real goal planner does."""

    def __init__(self, chat=(), task=(), conditions=None, initial=()):
        super().__init__(chat, task, conditions)
        self.initial, self.titles = list(initial), []

    async def generate(self, messages, response_schema):
        title = response_schema.get("title")
        self.titles.append(title)
        if title == "GoalPlan" and self.initial:
            return json.dumps({"summary": "Planned.", "conditions": self.conditions or [{"kind": "answer"}],
                               "initial_actions": self.initial})
        return await super().generate(messages, response_schema)


@pytest.fixture
def ran(monkeypatch):
    """Every tool that really ran. Python really evaluates simple arithmetic."""
    calls = []
    from app.capabilities import tools as capabilities_module

    def search(query):
        calls.append("web_search")
        return {"query": query, "results": [{"url": "https://example.com", "title": "Example Domain"}]}

    async def read(url, start_char=0):
        calls.append("read_webpage")
        return {"title": "Example Domain", "url": url, "text": "Example Domain. For use in documentation."}

    async def python(code, data=None):
        calls.append("run_python")
        expression = code.removeprefix("print(").removesuffix(")")
        return {"stdout": f"{eval(expression, {'__builtins__': {}})}\n", "result": "None", "result_truncated": False}

    monkeypatch.setattr(capabilities_module, "web_search", search)
    monkeypatch.setattr(capabilities_module, "read_webpage", read)
    monkeypatch.setattr(capabilities_module, "run_python", python)
    original = WorldTools.execute

    def execute(self, name, arguments, task_id=None):
        calls.append(name)
        return original(self, name, arguments, task_id)

    monkeypatch.setattr(WorldTools, "execute", execute)
    return calls


def session_with(model, world=LEGACY_WORLD, persistent=None, persona=None, settings=None):
    engine, bus = WorldEngine(world), EventBus()
    settings = settings or Settings(direct_initial_delegation=True)
    manager = TaskManager(model, WorldTools(engine, bus), bus, settings, persistent, persona=persona)
    session = ChatSession(model, bus, manager, engine, settings, store=MemoryStore(":memory:"), persona=persona)
    session.attach()
    return session, manager


async def done(session, manager):
    await session.close()
    await manager.close()


def kip_lines(session, source=None):
    return [t["text"] for t in session.history() if t["role"] == "kip" and t.get("source") == source]


def latest_prompt(model):
    return model.prompts[-1][-1]["content"]


# ── 1. execution truthfulness ──────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_asking_to_run_python_runs_it_and_fabricated_output_is_corrected(ran):
    fabricated = "Here it is:\n```python\nprint(3 * 60 * 60)\n```\nOutput: 10800"
    model = PlanningModel([said(fabricated)], conditions=[{"kind": "answer", "required_tools": ["run_python"]}],
                          initial=[{"tool": "run_python", "arguments": {"code": "print(3 * 60 * 60)"}}])
    session, manager = session_with(model)
    session.submit("Please actually run Python for 3 * 60 * 60 and show me the code and output.")
    await settle(session, manager)
    assert "Never write its result" in latest_prompt(model)  # told before replying
    correction = kip_lines(session, "correction")
    assert correction and "no Python has run for that" in correction[0] and "running now" in correction[0]
    await manager.runner
    task = next(iter(manager.tasks.values()))
    assert task.requested_tools == ["run_python"] and task.status == "completed"
    assert ran.count("run_python") == 1
    assert task.answer["answer"] == "Ran the numbers: 10800."  # said aloud: the result, not the code
    assert kip_lines(session, "narration")[-1] == task.answer["answer"]
    assert model.titles.count("GoalPlan") == 1 and len(model.titles) == 1  # no extra model call to restate it
    await done(session, manager)


@pytest.mark.asyncio
async def test_a_calculation_request_is_not_answered_by_a_guess(ran):
    model = PlanningModel([said("20.")], conditions=[{"kind": "answer", "required_tools": ["run_python"]}],
                          initial=[{"tool": "run_python", "arguments": {"code": "print(3 * 60 * 60)"}}])
    session, manager = session_with(model)
    manager.restrictions.update("Don't run Python")
    session.submit("You can use Python again. Use Python to calculate how many seconds are in three hours, "
                   "and tell me the result.")
    await settle(session, manager)
    assert not manager.restrictions  # the grant was applied first
    await manager.runner
    assert ran.count("run_python") == 1
    assert kip_lines(session, "narration")[-1] == "Ran the numbers: 10800."
    await done(session, manager)


@pytest.mark.asyncio
async def test_a_prohibited_tool_the_request_needs_gets_a_refusal_not_a_fake(ran):
    model = PlanningModel([said("I'd need Python for that, and you've confiscated it.")])
    session, manager = session_with(model)
    session.submit("Don't use Python, but run this in Python and show the output: 2 + 2")
    await settle(session, manager)
    prompt = latest_prompt(model)
    assert "Turn it down with one blunt, good-natured line" in prompt and "Don't fake it" in prompt
    assert not manager.tasks and "run_python" not in ran
    await done(session, manager)


@pytest.mark.asyncio
async def test_a_refused_tool_is_reported_as_not_done_and_an_invented_run_is_flagged(ran):
    plan = [{"kind": "answer", "required_tools": ["read_webpage"]}]

    def answer(payload):
        ids = [o["evidence_id"] for o in payload["tool_observations"] if o["success"]]
        return {"summary": "Ready.", "action": {"tool": "finish_answer", "evidence_ids": ids, "answer":
                "It's a documentation placeholder. 17 * 23 = 391: I ran that calculation locally."}}
    model = PlanningModel(task=[answer], conditions=plan, initial=[
        {"tool": "run_python", "arguments": {"code": "print(17 * 23)"}},
        {"tool": "read_webpage", "arguments": {"url": "https://example.com"}}])
    session, manager = session_with(model)
    task = manager.start("Read https://example.com and use Python to calculate 17 * 23. Tell me both results.")
    manager.restrictions.update("Don't run Python for now.")  # arrives after the errand started
    await manager.runner
    assert "run_python" not in ran and "read_webpage" in ran
    assert task.status == "completed" and task.summary.startswith("Partly done.")
    assert task.blocked == ["run Python (you asked me not to)"]
    narration = kip_lines(session, "narration")[-1]
    assert "No program actually ran for this" in narration and "Not done: run Python" in narration
    await done(session, manager)


@pytest.mark.asyncio
async def test_a_failed_run_is_never_claimed_as_a_success(ran, monkeypatch):
    from app.capabilities import tools as capabilities_module

    async def broken(code, data=None):
        raise ValueError("SyntaxError: invalid syntax")
    monkeypatch.setattr(capabilities_module, "run_python", broken)
    model = PlanningModel([said("Done. I ran it and the output was 4.")],
                          conditions=[{"kind": "answer", "required_tools": ["run_python"]}],
                          initial=[{"tool": "run_python", "arguments": {"code": "print(2 +"}}])
    session, manager = session_with(model)
    session.submit("Run Python for 2 + 2 please.")
    await settle(session, manager)
    assert any("no Python has run for that" in line for line in kip_lines(session, "correction"))
    task = next(iter(manager.tasks.values()))
    assert not any(e["tool"] == "run_python" and e["success"] for e in task.action_history)
    assert task.answer is None
    await done(session, manager)


# ── 2. capability self-awareness ───────────────────────────────────────────────
@pytest.mark.asyncio
async def test_kip_knows_his_working_tools_and_corrects_a_false_denial(ran):
    model = PlanningModel([said("I don't have access to external websites, so I can't check.")])
    session, manager = session_with(model)
    session.submit("What is on https://example.com, with a short summary passed along to Paul?")
    await settle(session, manager)
    assert "available: search the web, read webpages, run Python" in latest_prompt(model)
    assert "I can search the web and read webpages" in kip_lines(session, "correction")[0]
    task = next(iter(manager.tasks.values()))  # the page read was started from their own words
    assert task.grant.source == "question" and task.requested_tools == ["read_webpage"]
    await done(session, manager)


def test_the_tool_line_reflects_restrictions_switches_and_failures():
    engine, bus = WorldEngine(LEGACY_WORLD), EventBus()
    manager = TaskManager(Model(), WorldTools(engine, bus), bus, Settings())
    assert capabilities(manager, Settings()).startswith("available: search the web, read webpages, run Python")
    manager.restrictions.update("Don't read any webpages, and don't go into the kitchen", {"kitchen": "Kitchen"})
    line = capabilities(manager, Settings())
    assert "you asked me not to: read webpages" in line and "(not the kitchen)" in line
    task = TaskContext(goal="x")
    task.action_history.append({"tool": "web_search", "success": False, "error": "Search provider unavailable"})
    manager.tasks[task.id] = task
    assert "last attempt failed: search the web (Search provider unavailable)" in capabilities(manager, Settings())
    assert "switched off: search the web, read webpages, run Python" in capabilities(
        manager, Settings(digital_tools_enabled=False))


# ── 3. no stale replay ─────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_lifting_a_restriction_and_declining_replays_nothing(ran):
    replay = "Understood. I'll calculate it.\n```python\n3 * 60 * 60\n```\nThe result is 10800 seconds."
    model = PlanningModel([said("I could check the office.", offer="Check the office"),
                           said(replay, task="Use Python to calculate the seconds in three hours")])
    session, manager = session_with(model)
    manager.restrictions.update("Don't run Python")
    session.submit("Could you offer to check the office?")
    await settle(session, manager)
    assert session.pending_offer == "Check the office"
    session.submit("No thanks. You can use Python again.")
    await settle(session, manager)
    assert "Acknowledge in a word or two" in latest_prompt(model)
    assert not manager.tasks and not ran                       # no replayed calculation, no trip
    assert session.pending_offer is None and not manager.restrictions
    assert any("no Python has run for that" in line for line in kip_lines(session, "correction"))
    await done(session, manager)


@pytest.mark.parametrize("message, state_only", [
    ("No thanks. You can use Python again.", True), ("Don't run Python for now.", True),
    ("Stop that.", True), ("You can use Python again. Use Python to calculate 2 + 2.", False),
    ("What did you do today?", False), ("Find my keys", False)])
def test_state_changes_are_told_apart_from_new_requests(message, state_only):
    granted = ["run Python"] if "can use Python" in message else []
    revoked = ["run Python"] if message.startswith("Don't run") else []
    assert read_intent(message, granted, revoked).changes_state_only is state_only


@pytest.mark.asyncio
async def test_stop_cancels_the_current_errand_and_the_queue(ran):
    model = PlanningModel([said("Stopping."), said("Stopped.")])
    session, manager = session_with(model)
    manager.start("Find the keys")
    session.pending_goals.append("Check the study")
    session.submit("Stop that.")
    await settle(session, manager)
    await asyncio.sleep(0.05)
    assert manager.active_id is None and not session.pending_goals
    assert next(iter(manager.tasks.values())).status == "cancelled"
    assert any(line.startswith("Stopped:") for line in kip_lines(session, "narration"))
    await done(session, manager)


@pytest.mark.asyncio
async def test_asking_for_an_offer_with_wait_for_me_registers_the_offer(ran):
    model = PlanningModel([said("I can check the office. Say the word.", task="Check the office")])
    session, manager = session_with(model)
    session.submit("Could you offer to check the office, and wait for my confirmation before going?")
    await settle(session, manager)
    assert session.pending_offer == "Check the office" and not manager.tasks and not ran
    await done(session, manager)


# ── 4. claims grounded in evidence ─────────────────────────────────────────────
def _office_checked(manager):
    """Kip really looked in the office (an empty room), as in the live run."""
    task = TaskContext(goal="Check the office for current items", status="completed")
    task.action_history.append({"tool": "look", "success": True, "observed_at": "06:04", "evidence_id": "e1",
                                "observation": {"room": "office", "connections": ["entrance"], "people": [],
                                                "objects": [], "held_objects": []}})
    manager.tasks[task.id] = task


@pytest.mark.asyncio
async def test_an_empty_office_cannot_support_a_claim_that_paul_was_there(ran):
    memory = PersistentMemory(people={"paul": PersonMemory(name="Paul", last_seen_room="office")})
    model = PlanningModel([said("I didn't speak to Paul. He was in the office, but I kept my distance.")])
    session, manager = session_with(model, world=HOUSE_WORLD, persistent=memory)
    _office_checked(manager)
    world = manager.tools._engine.snapshot()
    assert last_seen(manager, world)["Paul"] == ("older record only: last seen in the office; "
                                                 "not there when I looked at 06:04")
    session.submit("What did you actually do today, and did you speak to Paul?")
    await settle(session, manager)
    prompt = latest_prompt(model)
    assert "not there when I looked at 06:04" in prompt and "the office: nobody there" in prompt
    correction = kip_lines(session, "correction")[0]
    assert "when I looked in the office at 06:04, Paul wasn't there" in correction
    await done(session, manager)


def test_hedged_or_supported_presence_is_not_flagged():
    engine, bus = WorldEngine(HOUSE_WORLD), EventBus()
    manager = TaskManager(Model(), WorldTools(engine, bus), bus, Settings())
    _office_checked(manager)
    world = engine.snapshot()
    for reply in ("Paul is usually in the office.", "My older record has Paul in the office.",
                  "Paul might be in the kitchen.", "I checked the office."):
        assert contradicted_presence(reply, manager, world) == [], reply
    assert contradicted_presence("Paul was in the office.", manager, world) == [("paul", "office")]


# ── 5. goal completeness ────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_a_room_check_reports_what_was_actually_in_the_room(ran):
    engine, bus = WorldEngine(LEGACY_WORLD), EventBus()
    manager = TaskManager(Model(), WorldTools(engine, bus), bus, Settings())
    task = manager.start("Look around the hall and tell me what is there", Grant.full("api", "x"))
    observation = manager.call_tool(task, "look", {})["observation"]
    text = describe_findings(task.public())
    names = [p["name"] for p in observation["people"]] + [o["name"].lower() for o in observation["objects"]]
    assert text.startswith("I looked in the hall (")
    for name in names:
        assert name in text
    if not names:
        assert "(empty)" in text
    await manager.close()


# ── 6. persona through conversation and tool handoffs ──────────────────────────
@pytest.mark.asyncio
async def test_persona_reaches_the_conversation_and_an_introduction_names_the_speaker(ran):
    model = PlanningModel([said("Neave. The house is still standing.")])
    session, manager = session_with(model, persona=Persona())
    session.submit("Hi Kip, it's Neave. How are you doing today?")
    await settle(session, manager)
    system, latest = model.prompts[-1][0]["content"], latest_prompt(model)
    assert IDENTITY in system and "ready to assist" in system  # named as something never to say
    assert "Stance:" in latest and "don't know who is speaking" not in latest
    assert session.history()[0]["speaker"] == "neave"
    await done(session, manager)


@pytest.mark.asyncio
async def test_the_same_character_writes_the_answer_after_a_tool_handoff(ran):
    def answer(payload):
        assert payload["persona"]["mood"] and "identity" not in payload["persona"]  # it's in the system prompt
        ids = [o["evidence_id"] for o in payload["tool_observations"] if o["success"]]
        return {"summary": "Ready.", "action": {"tool": "finish_answer", "answer": "It's a placeholder page.",
                                                "evidence_ids": ids}}
    model = PlanningModel(task=[answer], conditions=[{"kind": "answer", "required_tools": ["read_webpage"]}],
                          initial=[{"tool": "read_webpage", "arguments": {"url": "https://example.com"}}])
    session, manager = session_with(model, persona=Persona())
    task = manager.start("Read https://example.com and summarise it")
    await manager.runner
    assert task.status == "completed" and task.answer["answer"] == "It's a placeholder page."
    answering = [messages for messages, _ in model.calls if "persona" in json.loads(messages[1]["content"])]
    assert answering and all(IDENTITY in messages[0]["content"] for messages in answering)
    assert IDENTITY in CHAT_SYSTEM.format(name="Kip", identity=IDENTITY + "\n", voice="")
    await done(session, manager)


# ── 7. no errand-pushing in small talk ─────────────────────────────────────────
@pytest.mark.asyncio
async def test_small_talk_does_not_keep_proposing_errands(ran):
    model = PlanningModel([said("Nice.", offer="Check the kitchen"), said("Mm.", offer="Tidy the study"),
                           said("Indeed.", offer="Check the kitchen"), said("Quite.", offer="Look for the keys")])
    session, manager = session_with(model)
    offers = []
    for line in ("Lovely evening.", "The cat's asleep again.", "I like this song.", "Long week."):
        session.submit(line)
        await settle(session, manager)
        offers.append(session.pending_offer)
    assert offers[0] == "Check the kitchen"
    assert offers[1:] == ["Check the kitchen"] * 3  # nothing piled on top; the same one never re-offered
    assert "Small talk: no suggestions to do things around the house." in latest_prompt(model)
    assert not manager.tasks and not physical(ran)
    await done(session, manager)


# ── 8. performance: a waiting person goes first ────────────────────────────────
@pytest.mark.asyncio
async def test_errand_model_calls_wait_while_someone_waits_for_a_reply():
    order = []

    class Recorder:
        last_metrics = {}

        async def generate(self, messages, schema):
            order.append("errand")
            return "{}"
    gate = ConversationFirst(Recorder())
    gate.conversation_waiting(+1)
    pending = asyncio.create_task(gate.generate([], {}))
    await asyncio.sleep(0.01)
    assert order == [] and not pending.done()
    order.append("reply")
    gate.conversation_waiting(-1)
    await pending
    assert order == ["reply", "errand"] and gate.last_metrics == {}


# ── 9. free to use the web and Python on his own ───────────────────────────────
@pytest.mark.asyncio
async def test_kip_runs_python_on_his_own_when_he_decides_it_helps(ran):
    model = PlanningModel([said("Let me count properly.", task="Use Python to calculate 366 * 24")],
                          conditions=[{"kind": "answer", "required_tools": ["run_python"]}],
                          initial=[{"tool": "run_python", "arguments": {"code": "print(366 * 24)"}}])
    session, manager = session_with(model)
    session.submit("I always forget how many hours there are in a leap year.")  # no request, no question
    await settle(session, manager)
    await manager.runner
    task = next(iter(manager.tasks.values()))
    assert task.grant.source == "question" and ran.count("run_python") == 1  # read-only, no yes needed
    assert kip_lines(session, "narration")[-1] == "Ran the numbers: 8784."
    assert len(model.titles) == 1  # planned once; the run itself is the answer
    await done(session, manager)


@pytest.mark.asyncio
async def test_kip_looks_things_up_unasked_but_house_errands_still_wait_for_a_yes(ran):
    model = PlanningModel([said("Let me check.", offer="Search the web for today's weather in Mumbai"),
                           said("I could take a look.", offer="Check the kitchen")],
                          conditions=[{"kind": "answer", "required_tools": ["web_search"]}])
    session, manager = session_with(model)
    session.submit("I wonder what the weather's like in Mumbai today.")
    await settle(session, manager)
    lookup = manager.tasks[manager.active_id]
    assert lookup.goal == "Search the web for today's weather in Mumbai" and lookup.grant.source == "question"
    assert session.pending_offer is None
    await manager.cancel(lookup.id)
    session.submit("Something smells odd downstairs.")
    await settle(session, manager)
    assert session.pending_offer == "Check the kitchen" and len(manager.tasks) == 1
    assert not physical(ran)
    await done(session, manager)


# ── 10. greetings: fresh each time, in his own voice ───────────────────────────
@pytest.mark.asyncio
async def test_a_repeated_greeting_never_shows_kip_his_old_reply(ran):
    old = "Better than yesterday, Neave. I'm just here in the hall. How is your morning going?"
    model = PlanningModel([said(old), said("Still here. The hall hasn't moved either.")])
    session, manager = session_with(model, world=HOUSE_WORLD, persona=Persona())
    session.submit("Hi Kip, it's Neave. How are you doing today?")
    await settle(session, manager)
    again = ChatSession(model, session.bus, manager, session.engine, session.settings, store=session.store,
                        persona=session.persona)  # the next test run: same memory, new conversation
    again.attach()
    again.submit("Hi Kip, it's Neave. How are you doing today?")
    await settle(again, manager)
    prompt = latest_prompt(model)
    assert "They said: Hi Kip, it's Neave." in prompt   # remembered what she said
    assert old not in prompt and "I replied" not in prompt  # but not his own line to parrot
    await again.close()
    await done(session, manager)


@pytest.mark.asyncio
async def test_a_housemates_greeting_invites_his_own_voice(ran):
    model = PlanningModel([said("Neave. Still upright.")])
    session, manager = session_with(model, world=HOUSE_WORLD, persona=Persona())
    session.submit("Hi Kip, it's Neave. How are you doing today?")
    await settle(session, manager)
    prompt = latest_prompt(model)
    assert "You live with them" in prompt and "Humour suits this" in prompt
    assert "A greeting:" in prompt and "Don't ask them back unless you want to know" in prompt
    assert "barely know" not in prompt and "reserved" not in prompt
    await done(session, manager)


# ── 11. a voice companion: live facts looked up, everything said aloud ─────────
@pytest.mark.asyncio
async def test_live_facts_go_straight_to_the_lookup_and_its_answer_is_the_reply(ran):
    def answer(payload):
        ids = [o["evidence_id"] for o in payload["tool_observations"] if o["success"]]
        return {"summary": "Ready.", "action": {"tool": "finish_answer", "evidence_ids": ids, "answer":
                "**31°C** and muggy, says [the forecast](https://example.com/mumbai). Umbrella optional."}}
    model = PlanningModel([said("Probably 31 degrees and muggy.")], task=[answer])
    session, manager = session_with(model)
    session.submit("What's the weather like in Mumbai today?")
    await settle(session, manager)
    await manager.runner
    # No chat reply comes first, so nothing can guess a figure or promise a lookup that never starts.
    assert model.prompts == []
    task = next(iter(manager.tasks.values()))
    assert task.grant.source == "question" and task.status == "completed"
    assert ran[:1] == ["web_search"] and "get_status" not in ran  # a web lookup never reads the house
    assert "GoalPlan" not in model.titles  # the plan was known: no planning call
    lines = [t for t in session.history() if t["role"] == "kip"]
    assert [t["text"] for t in lines] == ["31°C and muggy, says the forecast. Umbrella optional."]
    await done(session, manager)


@pytest.mark.asyncio
async def test_a_guessed_live_figure_in_ordinary_chat_is_retracted(ran):
    # Small talk that mentions the weather still gets a normal reply, and a guessed figure is corrected.
    model = PlanningModel([said("Probably 31 degrees and muggy.")])
    session, manager = session_with(model)
    session.submit("I love this weather.")
    await settle(session, manager)
    assert model.prompts and "that figure was a guess" in kip_lines(session, "correction")[0]
    await done(session, manager)


@pytest.mark.asyncio
async def test_a_follow_up_on_a_lookup_is_looked_up_too_with_its_topic(ran):
    answers = iter(["New Zealand beat India by 87 runs in the fourth T20I and lead the series 3-1.",
                    "Yes: the fifth T20I is on Sunday."])

    def answer(payload):
        ids = [o["evidence_id"] for o in payload["tool_observations"] if o["success"]]
        return {"summary": "Ready.", "action": {"tool": "finish_answer", "evidence_ids": ids,
                                                "answer": next(answers)}}
    model = PlanningModel([said("Fifth match? No idea, honestly.")], task=[answer, answer])
    session, manager = session_with(model)
    session.submit("What happened in the last India vs New Zealand cricket game?")
    await settle(session, manager)
    await manager.runner
    session.submit("Is there a fifth match still to play in that series?")  # no live-fact word of its own
    await settle(session, manager)
    await manager.runner
    assert model.prompts == []  # neither question waited for a chat reply
    follow_up = list(manager.tasks.values())[-1]
    assert follow_up.status == "completed"
    assert "following up on: What happened in the last India vs New Zealand cricket game?" in follow_up.goal
    assert kip_lines(session, "narration")[-1] == "Yes: the fifth T20I is on Sunday."
    await done(session, manager)


@pytest.mark.asyncio
async def test_an_unrelated_question_after_a_lookup_gets_an_ordinary_reply(ran):
    def answer(payload):
        ids = [o["evidence_id"] for o in payload["tool_observations"] if o["success"]]
        return {"summary": "Ready.", "action": {"tool": "finish_answer", "evidence_ids": ids,
                                                "answer": "Sunny, 24 degrees."}}
    model = PlanningModel([said("Blue. Very blue.")], task=[answer])
    session, manager = session_with(model)
    session.submit("What's the weather in Pune today?")
    await settle(session, manager)
    await manager.runner
    session.submit("What's your favourite colour?")
    await settle(session, manager)
    assert len(model.prompts) == 1 and len(manager.tasks) == 1
    await done(session, manager)


@pytest.mark.asyncio
async def test_a_sports_question_after_small_talk_about_someone_is_still_looked_up_at_once(ran):
    def answer(payload):
        ids = [o["evidence_id"] for o in payload["tool_observations"] if o["success"]]
        return {"summary": "Ready.", "action": {"tool": "finish_answer", "evidence_ids": ids,
                                                "answer": "Next race is Sunday, says the calendar."}}
    model = PlanningModel([said("I check the web.")], task=[answer])
    session, manager = session_with(model, world=HOUSE_WORLD)
    session.submit("Ha, good to hear. Let Moira enjoy her book. Hey, when's the next F1 race? "
                   "And who's leading the championship right now?")
    await settle(session, manager)
    assert session.offer is None  # not held for a yes: a lookup changes nothing in the house
    await manager.runner
    task = next(iter(manager.tasks.values()))
    assert task.grant.source == "question" and task.status == "completed"
    assert ran[:1] == ["web_search"] and "get_status" not in ran  # a web lookup never reads the house
    await done(session, manager)


def test_house_mentions_that_ask_for_something_still_make_a_lookup_house_work():
    session, _ = session_with(PlanningModel([]), world=HOUSE_WORLD)
    assert not session._involves_the_house("Let Moira enjoy her book. When's the next F1 race?")
    assert session._involves_the_house("Tell Paul. When's the next F1 race?")
    assert session._involves_the_house("When's the next F1 race? Then let Paul know.")
    assert session._involves_the_house("What's the weather like for Moira's walk?")


@pytest.mark.asyncio
async def test_nothing_unspeakable_reaches_the_conversation(ran):
    written = "Sure thing.\n\n```python\nprint(2 + 2)\n```\n\n- first point\n- **second** point, see https://example.com/x."
    model = PlanningModel([said(written)])
    session, manager = session_with(model)
    session.submit("Tell me something.")
    await settle(session, manager)
    for line in [t["text"] for t in session.history() if t["role"] == "kip"]:
        assert "```" not in line and "**" not in line and "http" not in line and "\n" not in line, line
    assert kip_lines(session)[0] == "Sure thing. first point. second point, see example.com."
    await done(session, manager)
