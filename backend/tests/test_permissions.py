"""The capability boundary: what a task may do is decided by a grant and checked where tools run.

Every test here records what was actually dispatched (world tools, web search, page reads,
Python), not what a classifier returned. Models are deterministic fakes; no live services,
and no store outside the test sandbox is touched.
"""
import asyncio
import json
from pathlib import Path

import pytest

from app.capabilities.schemas import ACTION_SCHEMAS
from app.chat import ChatSession
from app.config import Settings
from app.events.bus import EventBus
from app.memory.recall import MemoryStore
from app.tasks.manager import TaskManager
from app.tasks.permissions import (COMMUNICATION, EVERYTHING, MANIPULATION, NAVIGATION, READ_ONLY, TOOL_EFFECTS,
                                   Grant, Restrictions)
from app.world.engine import WorldEngine
from app.world.tools import WorldTools
from tests.fakes import ScriptedLLM, complete, tool

LEGACY_WORLD = Path(__file__).parent / "fixtures" / "legacy_house.json"
PHYSICAL_TOOLS = {"look", "move_to", "pick_up", "drop", "give", "talk_to"}
_SPECIAL = {"GoalPlan", "ConversationMove", "ConversationMeaning"}


class Model(ScriptedLLM):
    """Chat replies are streamed from `chat`; task decisions come from `task` and then block,
    so a task never does anything that was not scripted."""

    def __init__(self, chat=(), task=(), conditions=None):
        super().__init__(list(task), conditions)
        self.chat, self.left, self.prompts, self.last_metrics = iter(chat), len(task), [], {}

    async def stream_text(self, messages, max_tokens=300, temperature=None):
        self.prompts.append(messages)
        yield next(self.chat)

    async def generate(self, messages, response_schema):
        if response_schema.get("title") not in _SPECIAL:
            if self.left <= 0:
                await asyncio.Event().wait()
            self.left -= 1
        return await super().generate(messages, response_schema)


def said(text, task="", offer=""):
    return f"{text}\n###\n" + json.dumps({"task": task, "offer": offer, "remember": [], "forget": []})


@pytest.fixture
def dispatched(monkeypatch):
    """Every tool that actually ran, in order."""
    calls = []
    from app.capabilities import tools as capabilities

    def search(query):
        calls.append("web_search")
        return {"query": query, "results": [{"url": "https://example.com", "title": "Source"}]}

    async def read(url, start_char=0):
        calls.append("read_webpage")
        return {"title": "Source", "url": url, "text": "AQI 150"}

    async def python(code, data=None):
        calls.append("run_python")
        return {"stdout": "", "result": "42", "result_truncated": False}

    monkeypatch.setattr(capabilities, "web_search", search)
    monkeypatch.setattr(capabilities, "read_webpage", read)
    monkeypatch.setattr(capabilities, "run_python", python)
    original = WorldTools.execute

    def execute(self, name, arguments, task_id=None):
        calls.append(name)
        return original(self, name, arguments, task_id)

    monkeypatch.setattr(WorldTools, "execute", execute)
    return calls


def physical(calls):
    return [name for name in calls if name in PHYSICAL_TOOLS]


def session_with(model, world=LEGACY_WORLD, settings=None):
    engine, bus = WorldEngine(world), EventBus()
    settings = settings or Settings(direct_initial_delegation=True)
    manager = TaskManager(model, WorldTools(engine, bus), bus, settings)
    session = ChatSession(model, bus, manager, engine, settings, store=MemoryStore(":memory:"))
    session.attach()
    return session, manager, bus


async def settle(session, manager):
    await session.drain()
    for _ in range(50):  # let a started task reach its first blocking decision
        await asyncio.sleep(0)


async def finish(session, manager):
    await session.close()
    await manager.close()


def answer_from(tool_name):
    def answer(payload):
        ids = [o["evidence_id"] for o in payload["tool_observations"] if o["success"] and o["tool"] == tool_name]
        return {"summary": "Ready.", "action": {"tool": "finish_answer", "answer": "The AQI is 150.",
                                                "evidence_ids": ids}}
    return answer


# ── the registry ────────────────────────────────────────────────────────────────
def test_every_executable_tool_has_a_registered_effect():
    assert set(WorldTools.schemas()) | set(ACTION_SCHEMAS) == set(TOOL_EFFECTS)
    assert READ_ONLY.isdisjoint({NAVIGATION, MANIPULATION, COMMUNICATION})
    assert Grant(effects=sorted(READ_ONLY)).deny("teleport", {}) == "teleport is not a registered tool"


# ── the four review reproductions, judged by what ran ──────────────────────────
@pytest.mark.asyncio
async def test_a_location_question_never_starts_a_physical_search(dispatched):
    session, manager, _ = session_with(Model([said("I don't know offhand.", task="Search for the medicine")]))
    session.submit("Where is the medicine?")
    await settle(session, manager)
    assert manager.active() is None and not physical(dispatched)
    assert session.pending_offer == "Search for the medicine"  # offered once, not done
    await finish(session, manager)


@pytest.mark.asyncio
async def test_a_python_prohibition_stops_python_and_the_reply_is_told_about_it(dispatched):
    model = Model([said("42.", task="Calculate 6 times 7 using Python")])
    session, manager, _ = session_with(model)
    session.submit("Don't run Python; what is 6 times 7?")
    await settle(session, manager)
    assert "run_python" not in dispatched and manager.active() is None
    assert '"asked_not_to":["run Python"]' in model.prompts[-1][-1]["content"]
    await finish(session, manager)


@pytest.mark.asyncio
async def test_a_webpage_prohibition_holds_inside_an_allowed_search(dispatched):
    model = Model([said("Let me check.", task="Search the web for the current AQI in Mumbai")],
                  task=[{"summary": "Searching.", "action": {"tool": "web_search", "query": "AQI Mumbai"}},
                        answer_from("web_search")],
                  conditions=[{"kind": "answer", "required_tools": ["web_search", "read_webpage"]}])
    session, manager, _ = session_with(model)
    session.submit("Don't read any webpages; what is the current AQI in Mumbai?")
    await settle(session, manager)
    await manager.runner
    task = next(iter(manager.tasks.values()))
    assert task.status == "completed", task.summary
    assert dispatched == ["get_status", "get_map", "web_search"]
    assert task.conditions[0].required_tools == ["web_search"]  # the forbidden tool left the plan
    # The Explorer was never even offered a page read as a choice (dispatch would refuse it anyway).
    choices = [schema["properties"]["action"]["discriminator"]["mapping"] for _, schema in model.calls]
    assert choices and all("read_webpage" not in mapping and "web_search" in mapping for mapping in choices)
    await finish(session, manager)


@pytest.mark.asyncio
async def test_tell_me_in_a_lookup_is_a_reply_not_a_message_to_someone(dispatched):
    model = Model([said("Looking.", task="Search the web and tell me the current AQI in Mumbai")],
                  task=[{"summary": "Searching.", "action": {"tool": "web_search", "query": "AQI Mumbai"}},
                        answer_from("web_search")],
                  conditions=[{"kind": "answer", "required_tools": ["web_search"]}])
    session, manager, _ = session_with(model)
    session.submit("What is the current AQI in Mumbai?")
    await settle(session, manager)
    await manager.runner
    task = next(iter(manager.tasks.values()))
    assert task.status == "completed" and task.grant.source == "question"
    assert dispatched == ["get_status", "get_map", "web_search"] and session.pending_offer is None
    await finish(session, manager)


# ── requests, questions and third parties ───────────────────────────────────────
@pytest.mark.asyncio
async def test_search_the_web_and_tell_me_the_weather_is_a_plain_lookup(dispatched):
    model = Model([said("On it.", task="Search the web and tell me the weather")],
                  task=[{"summary": "Searching.", "action": {"tool": "web_search", "query": "weather"}},
                        answer_from("web_search")],
                  conditions=[{"kind": "answer", "required_tools": ["web_search"]}])
    session, manager, _ = session_with(model)
    session.submit("Search the web and tell me the weather")
    await settle(session, manager)
    await manager.runner
    assert next(iter(manager.tasks.values())).status == "completed"
    assert not physical(dispatched)
    await finish(session, manager)


@pytest.mark.asyncio
async def test_a_question_whose_plan_contacts_someone_stops_and_offers_once(dispatched):
    model = Model([said("Let me see.", task="Search the web for the weather and tell Paul"),
                   said("Fair enough."), said("Sure.", task="Search the web for the weather and tell Paul")],
                  conditions=[{"kind": "answer", "required_tools": ["web_search"]},
                              {"kind": "notify", "person": "paul", "message": "the weather"}])
    session, manager, _ = session_with(model, world=None)
    session.submit("What's the weather like for Paul?")
    await settle(session, manager)
    await manager.runner
    assert dispatched == ["get_status", "get_map"]  # stopped after planning: nothing looked up, nobody told
    assert next(iter(manager.tasks.values())).status == "cancelled"
    assert "talking to someone else" in session.history()[-1]["text"]
    assert session.pending_offer == "Search the web for the weather and tell Paul"
    session.submit("No thanks.")
    await settle(session, manager)
    session.submit("What's the weather like for Paul?")  # asked again: not offered again
    await settle(session, manager)
    assert session.pending_offer is None and len(manager.tasks) == 1
    await finish(session, manager)


@pytest.mark.asyncio
async def test_an_explicit_message_to_a_third_party_is_delivered(dispatched):
    model = Model([said("Will do.", task="Tell Neave dinner is ready.")], task=[
        tool("look"), tool("move_to", room="bedroom"), tool("look"),
        tool("talk_to", person="neave", message="Dinner is ready."),
        {"action": "report", "summary": "Message delivered."}, complete,
        {"approved": True, "summary": "Notification evidenced."}])
    session, manager, _ = session_with(model, settings=Settings())
    model.left += 1  # the Coordinator's first decision
    model.replies = iter([{"action": "delegate", "summary": "Notify.", "task": "Tell Neave dinner is ready."},
                          *list(model.replies)])
    session.submit("Please tell Neave dinner is ready.")
    await settle(session, manager)
    await manager.runner
    assert next(iter(manager.tasks.values())).status == "completed"
    assert "talk_to" in dispatched
    await finish(session, manager)


@pytest.mark.asyncio
async def test_where_are_my_keys_does_not_move_the_robot(dispatched):
    session, manager, _ = session_with(Model([said("Not sure.", task="Find the keys")]))
    session.submit("Where are my keys?")
    await settle(session, manager)
    assert manager.active() is None and not physical(dispatched)
    await finish(session, manager)


@pytest.mark.asyncio
async def test_find_my_keys_is_an_authorized_physical_request(dispatched):
    session, manager, _ = session_with(Model([said("On it.", task="Find the keys")]))
    session.submit("Find my keys")
    await settle(session, manager)
    assert manager.active()["grant"]["source"] == "request"
    assert "look" in dispatched  # it really goes and looks
    await finish(session, manager)


# ── prohibitions bind the executable calls ───────────────────────────────────────
@pytest.mark.asyncio
async def test_dont_use_python_calculate_runs_no_python(dispatched):
    model = Model([said("6 × 7 is 42.", task="Calculate 6 × 7 using Python")])
    session, manager, _ = session_with(model)
    session.submit("Don't use Python; calculate 6 × 7")
    await settle(session, manager)
    assert "run_python" not in dispatched and manager.active() is None
    await finish(session, manager)


@pytest.mark.asyncio
async def test_a_prohibited_room_is_never_entered_or_routed_through(dispatched):
    session, manager, _ = session_with(Model([said("Looking now.", task="Find the keys")]))
    session.submit("Don't go to the kitchen; find my keys")
    await settle(session, manager)
    task = manager.tasks[manager.active_id]
    result = manager.call_tool(task, "move_to", {"room": "kitchen"})  # as if the Explorer chose it
    assert result["refused"] and "kitchen" in result["error"]
    assert "move_to" not in dispatched
    assert task._find_path("hall", "kitchen") == [] and task._find_path("hall", "study") == ["study"]
    assert "kitchen" not in task.compact()["known_but_unobserved_rooms"]
    await finish(session, manager)


@pytest.mark.asyncio
async def test_a_web_lookup_cannot_turn_into_physical_exploration(dispatched):
    model = Model([said("Checking.", task="Search the web for the AQI")],
                  conditions=[{"kind": "answer", "required_tools": ["web_search"]}])
    session, manager, _ = session_with(model)
    session.submit("What's the AQI today?")
    await settle(session, manager)
    task = manager.tasks[manager.active_id]
    for name, arguments in [("look", {}), ("move_to", {"room": "kitchen"}), ("pick_up", {"object": "keys"}),
                            ("talk_to", {"person": "neave", "message": "hi"})]:
        assert manager.call_tool(task, name, arguments)["refused"]
    assert not physical(dispatched)
    await finish(session, manager)


@pytest.mark.asyncio
async def test_a_task_cannot_run_tools_beyond_its_grant_even_in_a_batch(dispatched):
    engine, bus = WorldEngine(LEGACY_WORLD), EventBus()
    manager = TaskManager(Model(), WorldTools(engine, bus), bus, Settings())
    manager.restrictions.update("Don't run Python")
    task = manager.start("Look something up", Grant.read_only("What is it?"))
    results = await manager.call_capability_batch(task, [
        {"tool": "web_search", "arguments": {"query": "x"}},
        {"tool": "run_python", "arguments": {"code": "6 * 7"}}])
    assert [r["success"] for r in results] == [True, False]
    assert dispatched.count("web_search") == 1 and "run_python" not in dispatched
    assert manager.call_tool(task, "give", {"object": "keys", "person": "neave"})["refused"]
    await manager.close()


@pytest.mark.asyncio
async def test_a_queued_errand_is_revalidated_when_it_runs(dispatched):
    model = Model([said("After this one.", task="Find the keys"), said("Noted.")])
    session, manager, _ = session_with(model)
    manager.start("Busy with something else")
    session.submit("Then find my keys")
    await settle(session, manager)
    assert list(session.pending_goals) == ["Find the keys"]
    session.submit("Actually, don't go into the kitchen.")  # said after it was queued
    await settle(session, manager)
    await manager.cancel(manager.active_id)
    await asyncio.sleep(0.1)  # the queue starts the next errand
    task = manager.tasks[manager.active_id]
    assert task.goal == "Find the keys" and task.grant.source == "request"
    assert manager.call_tool(task, "move_to", {"room": "kitchen"})["refused"]
    assert "move_to" not in dispatched
    await finish(session, manager)


@pytest.mark.asyncio
async def test_a_prohibition_said_while_an_offer_is_pending_declines_it(dispatched):
    model = Model([said("I could look that up.", offer="Search the web for the AQI"), said("Understood."),
                   said("Mm.")])
    session, manager, _ = session_with(model)
    session.submit("I wonder what the air is like today.")
    await settle(session, manager)
    assert session.pending_offer == "Search the web for the AQI"
    session.submit("Don't use the internet.")
    await settle(session, manager)
    session.submit("yes")
    await settle(session, manager)
    assert not manager.tasks and not dispatched


@pytest.mark.asyncio
async def test_an_accepted_offer_still_obeys_a_prohibition_made_after_it(dispatched):
    model = Model([said("I could look that up.", offer="Search the web for the AQI"), said("Starting.")],
                  conditions=[{"kind": "answer", "required_tools": ["web_search"]}])
    session, manager, _ = session_with(model)
    session.submit("I wonder what the air is like today.")
    await settle(session, manager)
    manager.restrictions.update("Don't use the internet.")  # e.g. said by someone else, elsewhere
    session.submit("yes")
    await settle(session, manager)
    task = manager.tasks[manager.active_id]
    assert task.grant.source == "offer" and task.goal == "Search the web for the AQI"
    assert task.conditions[0].required_tools == []
    for name, arguments in (("web_search", {"query": "AQI"}), ("read_webpage", {"url": "https://example.com"})):
        assert (await manager.call_capability(task, name, arguments))["refused"]
    assert not {"web_search", "read_webpage"} & set(dispatched)
    await finish(session, manager)


def test_prohibitions_can_be_lifted_and_ordinary_phrases_impose_none():
    restrictions = Restrictions()
    for text in ("No, use Python.", "No, go into the kitchen.", "I don't know if the web is down.", "Never mind."):
        restrictions.update(text, {"kitchen": "Kitchen"})
    assert not restrictions
    restrictions.update("Don't use Python", {})
    assert restrictions.forbids_tool("run_python")
    restrictions.update("OK, you can use Python again", {})
    assert not restrictions


@pytest.mark.asyncio
async def test_consequential_handling_needs_the_item_named_by_whoever_authorized_it(dispatched):
    engine, bus = WorldEngine(), EventBus()  # the house: medicine is marked sensitive
    manager = TaskManager(Model(), WorldTools(engine, bus), bus, Settings())
    task = manager.start("Check on everyone", Grant.full("autonomy"))
    assert "needs someone to ask me for it by name" in manager.call_tool(task, "pick_up", {"object": "medicine"})["error"]
    await manager.close()
    task = manager.start("Bring the medicine to Paul", Grant.full("request", "Please bring the medicine to Paul"))
    result = manager.call_tool(task, "pick_up", {"object": "medicine"})
    assert not result.get("refused")  # permitted; the world may still say it isn't here
    assert dispatched.count("pick_up") == 1
    await manager.close()


def test_a_read_only_grant_reports_what_a_physical_plan_would_need():
    from app.agents.schemas import GoalCondition
    plan = [GoalCondition(kind="notify", person="paul", message="the weather")]
    assert COMMUNICATION in Grant.read_only("?").uncovered(plan)
    assert not Grant.full("request").uncovered(plan)
    assert set(Grant.full("request").effects) == EVERYTHING
