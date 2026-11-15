"""The capability boundary: what a task may do is decided by a grant and checked where tools run.

Every test here records what was actually dispatched (web search, page reads, Python), not
what a classifier returned. Models are deterministic fakes; no live services,
and no store outside the test sandbox is touched.
"""
import asyncio
import json

import pytest

from app.capabilities.schemas import ACTION_SCHEMAS
from app.chat import ChatSession
from app.config import Settings
from app.events.bus import EventBus
from app.memory.recall import MemoryStore
from app.tasks.manager import TaskManager
from app.tasks.permissions import (CALCULATION, DIGITAL_LOOKUP, EVERYTHING, READ_ONLY, REPLY, TOOL_EFFECTS,
                                   Grant, Restrictions)
from tests.fakes import ScriptedLLM

_SPECIAL = {"GoalPlan"}


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
    return calls


def session_with(model, settings=None):
    bus = EventBus()
    settings = settings or Settings()
    manager = TaskManager(model, bus, settings)
    session = ChatSession(model, bus, manager, settings, store=MemoryStore(":memory:"))
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
    assert set(ACTION_SCHEMAS) == set(TOOL_EFFECTS)
    assert set(TOOL_EFFECTS.values()) == EVERYTHING == {DIGITAL_LOOKUP, CALCULATION, REPLY}
    assert Grant(effects=sorted(READ_ONLY)).deny("teleport", {}) == "teleport is not a registered tool"


# ── review reproductions, judged by what ran ──────────────────────────
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
                  task=[answer_from("web_search")],
                  conditions=[{"kind": "answer", "required_tools": ["web_search", "read_webpage"]}])
    session, manager, _ = session_with(model)
    session.submit("Don't read any webpages; what is the current AQI in Mumbai?")
    await settle(session, manager)
    await manager.runner
    task = next(iter(manager.tasks.values()))
    assert task.status == "completed", task.summary
    assert dispatched == ["web_search"]
    assert task.conditions[0].required_tools == ["web_search"]  # the forbidden tool left the plan
    # The Explorer was never even offered a page read as a choice (dispatch would refuse it anyway).
    choices = [schema["properties"]["action"]["discriminator"]["mapping"] for _, schema in model.calls]
    assert choices and all("read_webpage" not in mapping and "web_search" in mapping for mapping in choices)
    await finish(session, manager)


@pytest.mark.asyncio
async def test_tell_me_in_a_lookup_is_a_reply_not_a_message_to_someone(dispatched):
    model = Model([said("Looking.", task="Search the web and tell me the current AQI in Mumbai")],
                  task=[answer_from("web_search")],
                  conditions=[{"kind": "answer", "required_tools": ["web_search"]}])
    session, manager, _ = session_with(model)
    session.submit("What is the current AQI in Mumbai?")
    await settle(session, manager)
    await manager.runner
    task = next(iter(manager.tasks.values()))
    assert task.status == "completed" and task.grant.source == "question"
    assert dispatched == ["web_search"] and session.pending_offer is None
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
    assert set(dispatched) == {"web_search"}
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
async def test_a_task_cannot_run_tools_beyond_its_grant_even_in_a_batch(dispatched):
    from app.tasks.models import TaskContext
    manager = TaskManager(Model(), EventBus(), Settings())
    manager.restrictions.update("Don't run Python")
    # Exercise dispatch directly, without a competing planner terminating the run.
    task = TaskContext(goal="Look something up", grant=Grant.read_only("What is it?"))
    results = await manager.call_capability_batch(task, [
        {"tool": "web_search", "arguments": {"query": "x"}},
        {"tool": "run_python", "arguments": {"code": "6 * 7"}}])
    assert [r["success"] for r in results] == [True, False]
    assert dispatched.count("web_search") == 1 and "run_python" not in dispatched
    await manager.close()


@pytest.mark.asyncio
async def test_a_queued_errand_is_revalidated_when_it_runs(dispatched):
    model = Model([said("After this one.", task="Search the web for the AQI"), said("Noted.")])
    session, manager, _ = session_with(model)
    manager.start("Busy with something else")
    session.submit("Then search the web for the AQI")  # an explicit tool request: queued as said
    await settle(session, manager)
    assert list(session.pending_goals) == ["Then search the web for the AQI"]
    session.submit("Actually, don't use the internet.")  # said after it was queued
    await settle(session, manager)
    await manager.cancel(manager.active_id)
    await asyncio.sleep(0.1)  # the queue starts the next errand
    task = manager.tasks[manager.active_id]
    assert task.goal == "Then search the web for the AQI" and task.grant.source == "request"
    assert (await manager.call_capability(task, "web_search", {"query": "AQI"}))["refused"]
    assert "web_search" not in dispatched
    await finish(session, manager)


@pytest.mark.asyncio
async def test_a_lookup_kip_starts_on_his_own_is_read_only_and_obeys_a_later_prohibition(dispatched):
    model = Model([said("Let me look.", offer="Search the web for the AQI"), said("Understood.")],
                  conditions=[{"kind": "answer", "required_tools": ["web_search"]}])
    session, manager, _ = session_with(model)
    session.submit("I wonder what the air is like today.")  # not a request, not a question
    await settle(session, manager)
    task = manager.tasks[manager.active_id]
    assert task.goal == "Search the web for the AQI" and task.grant.source == "question"  # no yes needed
    assert session.pending_offer is None
    session.submit("Don't use the internet.")
    await settle(session, manager)
    for name, arguments in (("web_search", {"query": "AQI"}), ("read_webpage", {"url": "https://example.com"})):
        assert (await manager.call_capability(task, name, arguments))["refused"]
    assert not {"web_search", "read_webpage"} & set(dispatched)
    await finish(session, manager)


def test_prohibitions_can_be_lifted_and_ordinary_phrases_impose_none():
    restrictions = Restrictions()
    for text in ("No, use Python.", "No, go into the kitchen.", "I don't know if the web is down.", "Never mind."):
        restrictions.update(text)
    assert not restrictions
    restrictions.update("Don't use Python")
    assert restrictions.forbids_tool("run_python")
    restrictions.update("OK, you can use Python again")
    assert not restrictions


def test_a_narrow_grant_reports_what_a_plan_would_need():
    from app.agents.schemas import GoalCondition
    plan = [GoalCondition(kind="answer", required_tools=["web_search", "run_python"])]
    assert Grant(effects=[DIGITAL_LOOKUP, REPLY]).uncovered(plan) == {CALCULATION}
    assert not Grant.read_only("?").uncovered(plan)
    assert not Grant.full("request").uncovered(plan)
    assert set(Grant.full("request").effects) == EVERYTHING


@pytest.mark.asyncio
async def test_clarifying_a_permission_offer_preserves_its_effects(dispatched):
    model = Model([said("I mean I can look it up."), said("On my way.")])
    session, manager, _ = session_with(model)
    session._needs_permission({"goal": "Search the web for the AQI", "needs": [DIGITAL_LOOKUP]})
    effects = session.offer.effects
    session.submit("Huh?")
    await settle(session, manager)
    assert session.offer.effects == effects
    session.submit("Yes")
    await settle(session, manager)
    task = manager.tasks[manager.active_id]
    assert set(task.grant.effects) == effects
    await finish(session, manager)


@pytest.mark.asyncio
async def test_accepting_an_offer_with_no_effects_does_not_grant_everything(dispatched):
    from app.chat.consent import Offer
    session, manager, _ = session_with(Model([said("Understood.")]))
    session.offer = Offer(goal="Wait here", reply_turn_id="", generation=session.generation,
                          effects=frozenset())
    session.submit("Yes")
    await settle(session, manager)
    task = next(iter(manager.tasks.values()))
    assert task.grant.effects == []
    assert not dispatched
    await finish(session, manager)


def test_task_api_applies_restrictions_from_the_authorizing_goal(dispatched):
    from fastapi.testclient import TestClient
    from app.main import create_app
    from tests.fakes import WaitingLLM
    app = create_app(client=WaitingLLM(), settings=Settings())
    with TestClient(app) as client:
        response = client.post("/api/tasks", json={"goal": "Work out 6 times 7; don't use Python"})
        assert response.status_code == 202
        manager = app.state.manager
        task = manager.tasks[response.json()["id"]]
        assert manager.restrictions.forbids_tool("run_python")
        assert task.restricted == {"tools": ["run_python"]}
        assert not task.permits("run_python", {"code": "6 * 7"})
        assert "run_python" not in dispatched


@pytest.mark.asyncio
async def test_model_written_goals_do_not_change_standing_restrictions():
    session, manager, _ = session_with(Model())
    manager.start("Don't use Python", Grant.full("request", "Look up the AQI"))
    assert not manager.restrictions
    await finish(session, manager)


@pytest.mark.asyncio
async def test_batch_rechecks_restrictions_when_each_worker_starts(dispatched):
    session, manager, _ = session_with(Model())
    task = manager.start("Look something up", Grant.read_only("What is it?"))
    # This runs when the batch yields, before its workers have dispatched anything.
    asyncio.get_running_loop().call_soon(manager.restrictions.update, "Don't use Python")
    results = await manager.call_capability_batch(task, [
        {"tool": "web_search", "arguments": {"query": "test"}},
        {"tool": "run_python", "arguments": {"code": "6 * 7"}},
    ])
    assert [result["success"] for result in results] == [True, False]
    assert results[1]["refused"]
    assert dispatched.count("web_search") == 1 and "run_python" not in dispatched
    assert task.tool_count == len(dispatched)
    await finish(session, manager)


@pytest.mark.parametrize("said,ruled_out", [
    ("stop using python", {"run_python"}),
    ("stop using the internet", {"web_search", "read_webpage"}),
    ("avoid the internet", {"web_search", "read_webpage"}),
    ("never touch python", {"run_python"}),
    ("Don't you dare use Python", {"run_python"}),
    ("don't do any web searches", {"web_search"}),
    ("please stay offline", {"web_search", "read_webpage"}),
    ("stop it", set()), ("never mind", set()), ("avoid spoilers please", set()),
    ("I don't know if the web is down", set()), ("no, use python", set()),
])
def test_the_ways_people_say_dont_are_recognised(said, ruled_out):
    restrictions = Restrictions()
    restrictions.update(said)
    assert restrictions.tools == ruled_out


@pytest.mark.parametrize("said", ["I know you can run code, but please don't", "You can use python later, not now",
                                  "ok to use python? no, don't"])
def test_a_message_that_also_forbids_or_defers_lifts_nothing(said):
    restrictions = Restrictions()
    restrictions.update("don't use python")
    restrictions.update(said)
    assert restrictions.tools == {"run_python"}
    restrictions.update("Fine. You can use Python again.")
    assert restrictions.tools == set()
