import json

import httpx
import pytest

from app.agents.coordinator import Coordinator
from app.agents.explorer import Explorer
from app.agents.schemas import CriticReview, GoalCondition
from app.config import Settings
from app.llm.base import ModelError, structured
from app.llm.factory import create_hosted_client
from app.llm.groq import GroqClient, HostedFirst, reset_seconds
from app.tasks.models import TaskContext
from app.world.tools import WorldTools
from tests.fakes import ScriptedLLM

KEY = "gsk_private_test_key"
REVIEW = '{"approved":true,"summary":"Ready"}'


class Clock:
    def __init__(self):
        self.value = 1000.0

    def __call__(self):
        return self.value


async def groq(respond, models="openai/gpt-oss-120b,qwen/qwen3.8-27b", clock=None):
    client = GroqClient(Settings(groq_api_key=KEY, groq_models=models), now=clock or Clock())
    await client.http.aclose()
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(respond), base_url="https://groq.test/v1/",
                                    headers={"Authorization": f"Bearer {KEY}"})
    return client


def ok(content=REVIEW, model="openai/gpt-oss-120b", headers=None, finish="stop"):
    return httpx.Response(200, headers=headers or {}, json={
        "model": model, "usage": {"prompt_tokens": 50, "completion_tokens": 9},
        "choices": [{"finish_reason": finish, "message": {"content": content}}]})


@pytest.mark.parametrize("value,seconds", [("7.66s", 7.66), ("2m59.56s", 179.56), ("1h2m", 3720),
                                           ("240ms", 0.24), ("", None), ("soon", None)])
def test_reset_headers_are_read_as_seconds(value, seconds):
    assert reset_seconds(value) == (pytest.approx(seconds) if seconds is not None else None)


def test_no_key_means_no_hosted_client():
    assert create_hosted_client(Settings(groq_api_key=None)) is None
    assert create_hosted_client(Settings(groq_api_key="  ")) is None


async def test_request_is_a_plain_chat_completion():
    bodies = []

    def respond(request):
        assert request.headers["authorization"] == f"Bearer {KEY}"
        bodies.append(json.loads(request.content))
        return ok()

    client = await groq(respond)
    try:
        result = await structured(client, CriticReview, "Review", {})
        assert result.approved
        body = bodies[0]
        # System and user messages as given: no JSON mode, tools or reasoning parameters.
        assert set(body) == {"model", "messages", "temperature", "max_completion_tokens"}
        assert body["model"] == "openai/gpt-oss-120b"
        assert [m["role"] for m in body["messages"]] == ["system", "user"]
        assert body["messages"][0]["content"].startswith("Review\nOutput JSON schema:")
        assert client.last_metrics["provider"] == "groq" and client.last_metrics["completion_tokens"] == 9
    finally:
        await client.close()


async def test_rate_limited_model_is_skipped_until_its_reset():
    clock, seen = Clock(), []

    def respond(request):
        model = json.loads(request.content)["model"]
        seen.append(model)
        if model == "openai/gpt-oss-120b":
            return httpx.Response(429, headers={"retry-after": "30"}, json={"error": {"code": "rate_limit_exceeded"}})
        return ok(model=model)

    client = await groq(respond, clock=clock)
    try:
        assert await client.generate([{"role": "user", "content": "JSON"}], {}) == REVIEW
        assert seen == ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
        await client.generate([{"role": "user", "content": "JSON"}], {})
        assert seen[-1] == "qwen/qwen3.8-27b" and seen.count("openai/gpt-oss-120b") == 1
        clock.value += 31
        await client.generate([{"role": "user", "content": "JSON"}], {})
        # After its reset the first model is tried again (and here is still limited).
        assert seen[-2:] == ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
    finally:
        await client.close()


async def test_model_without_enough_token_quota_is_not_called():
    seen = []

    def respond(request):
        model = json.loads(request.content)["model"]
        seen.append(model)
        # Only 900 tokens left this minute on the first model: too few for another call.
        return ok(model=model, headers={"x-ratelimit-remaining-tokens": "900", "x-ratelimit-reset-tokens": "40s",
                                        "x-ratelimit-remaining-requests": "999"})

    client = await groq(respond)
    try:
        await client.generate([{"role": "user", "content": "JSON"}], {})
        await client.generate([{"role": "user", "content": "JSON"}], {})
        assert seen == ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
    finally:
        await client.close()


async def test_refused_request_benches_the_model_but_a_bad_generation_does_not():
    seen = []

    def respond(request):
        model = json.loads(request.content)["model"]
        seen.append(model)
        code = "invalid_request_error" if model == "openai/gpt-oss-120b" else "tool_use_failed"
        return httpx.Response(400, json={"error": {"code": code}})

    client = await groq(respond)
    try:
        with pytest.raises(ModelError, match="every model failed"):
            await client.generate([{"role": "user", "content": "JSON"}], {})
        with pytest.raises(ModelError):
            await client.generate([{"role": "user", "content": "JSON"}], {})
        # A made-up tool call is one bad generation: that model gets one more try, now and next time.
        assert seen == ["openai/gpt-oss-120b"] + ["qwen/qwen3.8-27b"] * 4
    finally:
        await client.close()


@pytest.mark.parametrize("mode", ["auth", "timeout", "offline"])
async def test_errors_never_reveal_the_key_or_try_every_model(mode):
    seen = []

    def respond(request):
        seen.append(request)
        if mode == "auth":
            return httpx.Response(401, text=f"bad key {KEY}")
        if mode == "timeout":
            raise httpx.ReadTimeout(f"private {KEY}", request=request)
        raise httpx.ConnectError(f"private {KEY}", request=request)

    client = await groq(respond)
    try:
        with pytest.raises(ModelError) as caught:
            await client.generate([{"role": "user", "content": "JSON"}], {})
        assert KEY not in str(caught.value) and len(seen) == 1
    finally:
        await client.close()


async def test_hosted_first_caps_hosted_calls_and_falls_back_locally():
    hosted_calls = []

    def respond(request):
        hosted_calls.append(request)
        return ok() if len(hosted_calls) == 1 else httpx.Response(503)

    client = await groq(respond, models="openai/gpt-oss-120b")
    local = ScriptedLLM([{"approved": True, "summary": "Local"}, {"approved": True, "summary": "Local again"}])
    try:
        bounded = HostedFirst(client, local, limit=2, purpose="test")
        assert (await structured(bounded, CriticReview, "Review", {})).summary == "Ready"
        assert bounded.last_metrics["provider"] == "groq"
        # Groq fails: the local model answers at once, and the failed try still counts.
        assert (await structured(bounded, CriticReview, "Review", {})).summary == "Local"
        assert "provider" not in bounded.last_metrics
        # Cap reached: no further hosted request is made.
        assert (await structured(bounded, CriticReview, "Review", {})).summary == "Local again"
        assert len(hosted_calls) == 2 and bounded.used == 2
    finally:
        await client.close()


async def test_goal_planning_uses_groq_and_decisions_stay_local():
    plans, bodies = [], []

    def respond(request):
        bodies.append(json.loads(request.content))
        payload = json.loads(bodies[-1]["messages"][1]["content"])
        plans.append(payload)
        return ok(json.dumps({"summary": "Find the keys.",
                              "conditions": [{"kind": "find_object", "object": "keys"}]}))

    client = await groq(respond)
    local = ScriptedLLM([])
    try:
        coordinator = Coordinator(local, hosted=client, hosted_calls=2)
        plan = await coordinator.define_goal("Find the keys.", {"rooms": {"kitchen": {"name": "Kitchen",
                                                                                     "outline": [[0, 0]]}}})
        assert plan.conditions[0].object == "keys" and local.calls == []
        assert "tools" not in bodies[0] and "response_format" not in bodies[0]
        # Only the goal and room names leave the machine: no geometry, memory or history.
        assert plans == [{"goal": "Find the keys.", "floor_plan": {"rooms": {"kitchen": "Kitchen"}}}]
    finally:
        await client.close()


async def test_web_research_uses_groq_but_robot_tasks_stay_local(monkeypatch):
    import app.agents.explorer as explorer_module
    # Skip robot reflexes (e.g. an opening get_status) so the model choice itself is tested.
    monkeypatch.setattr(explorer_module, "reflex_action", lambda *args: None)
    hosted_requests = []

    def respond(request):
        hosted_requests.append(request)
        return ok(json.dumps({"summary": "Searching.", "action": {"tool": "web_search", "query": "a fact"}}))

    client = await groq(respond)
    try:
        explorer = Explorer(ScriptedLLM([]), WorldTools.schemas(), hosted=client, hosted_calls=4)
        research = TaskContext(goal="Look up a fact.",
                               conditions=[GoalCondition(kind="answer", required_tools=["web_search"])])
        action = await explorer.decide(research, research.goal)
        assert action.tool == "web_search" and len(hosted_requests) == 1
        body = json.loads(hosted_requests[0].content)
        assert "tools" not in body and "response_format" not in body

        mixed = TaskContext(goal="Look up a fact and find the keys.",
                            conditions=[GoalCondition(kind="answer", required_tools=["web_search"]),
                                        GoalCondition(kind="find_object", object="keys")])
        local = ScriptedLLM([{"summary": "Searching.", "action": {"tool": "web_search", "query": "a fact"}}])
        explorer.client = local
        await explorer.decide(mixed, mixed.goal)
        assert len(hosted_requests) == 1 and len(local.calls) == 1
    finally:
        await client.close()


async def test_a_server_error_moves_on_to_the_next_model_without_a_retry():
    seen = []

    def respond(request):
        seen.append(json.loads(request.content)["model"])
        return httpx.Response(500, json={"error": {"type": "internal_server_error"}}) if len(seen) == 1 \
            else ok(model=seen[-1])

    client = await groq(respond)
    try:
        assert await client.generate([{"role": "user", "content": "JSON"}], {}) == REVIEW
        assert seen == ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
    finally:
        await client.close()


@pytest.mark.parametrize("content", [f"<think>private</think>\n{REVIEW}", f"```json\n{REVIEW}\n```"])
async def test_inline_reasoning_and_a_fence_are_stripped_from_the_reply(content):
    client = await groq(lambda request: ok(content))
    try:
        assert await client.generate([{"role": "user", "content": "JSON"}], {}) == REVIEW
    finally:
        await client.close()


# ── Groq's built-in browser search: a lookup and its answer in one call ─────────
SEARCHED = [
    {"type": "browser_search", "arguments": '{"query": "Lando Norris qualifying Mexico"}',
     "search_results": {"results": [
         {"title": "2026 Mexico City GP qualifying report", "url": "https://example.com/report", "content": ""},
         {"title": "Starting grid", "url": "https://example.com/grid", "content": ""}]}},
    {"type": "browser.open", "arguments": '{"cursor": 1, "id": 0}',
     "output": "L3: McLaren report\nL4: Lando Norris qualified sixth for the 2026 Mexico City Grand Prix.",
     "search_results": {"results": [
         {"title": "example.com - viewing lines [0 - 64] of 64", "url": "https://example.com/report"}]}},
]


QUOTED = ("Lando qualify sixth. Bad day. Very bad.【2†L26-L28】\n"
          "SOURCE: Lando Norris qualified sixth for the 2026 Mexico City Grand Prix")


def searched(content=QUOTED, tools=SEARCHED,
             model="openai/gpt-oss-120b"):
    return httpx.Response(200, json={"model": model, "usage": {"prompt_tokens": 5000, "completion_tokens": 40},
                                     "choices": [{"finish_reason": "stop", "message": {
                                         "role": "assistant", "content": content, "executed_tools": tools}}]})


async def test_search_runs_browser_search_on_a_gpt_oss_model_and_cleans_the_answer():
    bodies = []

    def respond(request):
        bodies.append(json.loads(request.content))
        return searched()

    client = await groq(respond, models="qwen/qwen3.8-27b,openai/gpt-oss-120b")
    try:
        assert client.can_search()
        found = await client.search([{"role": "user", "content": "where did lando qualify?"}])
        # Only models that support browser_search are asked; the inline citation is dropped.
        assert [b["model"] for b in bodies] == ["openai/gpt-oss-120b"]
        assert bodies[0]["tools"] == [{"type": "browser_search"}] and bodies[0]["include_reasoning"] is False
        # Low effort and a browsing budget: one opened page can be ~50k tokens, re-read every step.
        assert bodies[0]["reasoning_effort"] == "low"
        assert found.answer == "Lando qualify sixth. Bad day. Very bad."
        # The SOURCE line is split off, and what was read is kept to check the answer against.
        assert found.quote == "Lando Norris qualified sixth for the 2026 Mexico City Grand Prix"
        assert "qualified sixth" in found.evidence and (found.searches, found.pages) == (1, 1)
        # The page it opened is the source, titled from the search result rather than the viewer.
        assert found.sources == [{"title": "2026 Mexico City GP qualifying report",
                                  "url": "https://example.com/report", "snippet": ""}]
    finally:
        await client.close()


async def test_a_refused_search_never_benches_the_model_for_plain_calls():
    def respond(request):
        if "tools" in json.loads(request.content):
            return httpx.Response(400, json={"error": {"code": "invalid_request"}})
        return ok()

    client = await groq(respond, models="openai/gpt-oss-120b")
    try:
        with pytest.raises(ModelError):
            await client.search([{"role": "user", "content": "q"}])
        assert await client.generate([{"role": "user", "content": "JSON"}], {}) == REVIEW
    finally:
        await client.close()


async def test_without_a_gpt_oss_model_there_is_no_search():
    client = await groq(lambda request: ok(), models="qwen/qwen3.8-27b")
    try:
        assert not client.can_search()
    finally:
        await client.close()


async def _lookup(respond, monkeypatch, local=None):
    from datetime import date
    from app.chat.session import _lookup_plan
    from app.events.bus import EventBus
    from app.persona import Persona
    from app.tasks.manager import TaskManager
    from app.tasks.permissions import Grant
    from app.world.engine import WorldEngine
    from tests.test_persona import LEGACY_WORLD
    import app.capabilities.tools as tools
    monkeypatch.setattr(tools, "web_search", lambda query: {"query": query, "results": [
        {"url": "https://example.org", "title": "Local search", "snippet": "P6"}]})
    client = await groq(respond, models="openai/gpt-oss-120b")
    engine, bus = WorldEngine(LEGACY_WORLD), EventBus()
    persona = Persona()
    persona.attach(bus)
    local = local or ScriptedLLM([])
    manager = TaskManager(local, WorldTools(engine, bus), bus, Settings(), persona=persona, hosted=client)
    said = "where did lando qualify in mexico?"
    task = manager.start("web lookup: " + said, Grant.read_only(said), plan=_lookup_plan(said))
    await manager.runner
    await client.close()
    return task, local, date.today().strftime("%d %B %Y")


async def test_a_lookup_is_answered_by_one_groq_search_in_kips_voice(monkeypatch):
    from app.persona.profile import VOICE
    bodies = []

    def respond(request):
        bodies.append(json.loads(request.content))
        return searched()

    task, local, today = await _lookup(respond, monkeypatch)
    assert task.status == "completed", task.summary
    assert task.answer["answer"] == "Lando qualify sixth. Bad day. Very bad."
    assert [s["url"] for s in task.answer["sources"]] == ["https://example.com/report"]
    assert [a["tool"] for a in task.action_history] == ["web_search", "finish_answer"]  # no house bootstrap
    assert local.calls == [] and len(bodies) == 1  # no local model call, no second search
    system = bodies[0]["messages"][0]["content"]
    assert "Browsing budget: one" in system and "never open a page to double-check" in system
    assert today in system and "browser_search" in system and VOICE in system
    assert "Say every clock time in that timezone" in system and "UTC" in system
    # The date and local time travel with the question too, or the model searches last year's race.
    asked = bodies[0]["messages"][1]["content"]
    assert asked.startswith(f"web lookup: where did lando qualify in mexico?\n(Asked on {today} at ")


async def test_an_answer_with_nothing_searched_falls_back_to_the_usual_lookup(monkeypatch):
    finish = json.dumps({"summary": "Done.", "action": {"tool": "finish_answer", "answer": "Sixth.",
                                                        "evidence_ids": ["e1"]}})

    def respond(request):
        return searched(tools=[]) if "tools" in json.loads(request.content) else ok(finish)

    task, local, _ = await _lookup(respond, monkeypatch)
    # Groq's unsearched answer is a failed search; the usual search and research step take over,
    # still without reading the house.
    tools_run = [(a["tool"], a["success"]) for a in task.action_history]
    assert tools_run[:2] == [("web_search", False), ("web_search", True)]
    assert not {"get_status", "get_map"} & {tool for tool, _ in tools_run}
    assert task.status == "completed" and task.answer["answer"] == "Sixth."


async def test_an_answer_stating_a_time_no_page_gave_is_never_spoken(monkeypatch):
    """The live failure: "starts 2:00 am GMT = 7:30 am IST" when the page said 7:00 am UTC."""
    page = [{"type": "browser.open", "arguments": "{}",
             "output": "L14: T20I Sun, 1 Nov 2026, 7:00 am\nL30: 7:00 am UTC starts\nL32: Seddon Park, Hamilton",
             "search_results": {"results": [{"title": "NZ vs India, 5th T20I", "url": "https://bcci.example/m"}]}}]
    wrong = "5th T20I today starts 2:00 am GMT = 7:30 am IST, already past.\nSOURCE: 7:00 am UTC starts"
    finish = json.dumps({"summary": "Done.", "action": {"tool": "finish_answer", "answer": "Twelve thirty.",
                                                        "evidence_ids": ["e1"]}})

    def respond(request):
        return searched(wrong, page) if "tools" in json.loads(request.content) else ok(finish)

    task, _, _ = await _lookup(respond, monkeypatch)
    failed = task.action_history[0]
    assert failed["tool"] == "web_search" and not failed["success"]
    assert "02:00 GMT is not in the quoted source" in failed["error"]
    assert task.answer["answer"] == "Twelve thirty."  # the usual lookup answered instead


@pytest.mark.parametrize("answer,quote,ok_", [
    ("Starts 12:30 pm IST, already under way.", "Time 7:00 am UTC starts", True),   # converted in code
    ("12:30 today.", "Time 7:00 am UTC starts", True),                                # the listener's zone
    ("Starts 7:30 am IST.", "Time 7:00 am UTC starts", False),
    ("Starts 12:30 pm IST.", "", False),                                               # no source quoted
    ("Starts 12:30 pm IST.", "Match starts 12:30 pm IST", False),                      # quote not in the page
    ("India 185/4 after 20 overs.", "Time 7:00 am UTC starts", False),                 # figures from nowhere
    ("12:30 pm today, Saturday.", "Time 7:00 am UTC starts", False),                   # 1 Nov 2026 is a Sunday
    ("12:30 pm today, Sunday.", "Time 7:00 am UTC starts", True),
])
def test_grounding_checks_times_and_figures_against_the_quoted_page(answer, quote, ok_):
    from app.capabilities.grounding import unsupported
    evidence = "L28: Time\nL29:\nL30: 7:00 am UTC starts\nL32: Seddon Park, Hamilton, 1 Nov 2026"
    from datetime import date
    assert (unsupported(answer, quote, evidence, local_offset=330, today=date(2026, 11, 1)) == []) is ok_
