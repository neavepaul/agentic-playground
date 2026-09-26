"""Deterministic execution, search coherence, stall detection and simulation modes.

These cover the layer that decides whether a step needs the model at all.
"""
import json

import pytest

from app.agents.explorer import Explorer, observable_commands, reflex_action
from app.agents.schemas import GoalCondition
from app.config import Settings
from app.events.bus import EventBus
from app.tasks.manager import TaskManager
from app.tasks.models import Intention, TaskContext
from app.world.clock import WorldClock
from app.world.engine import WorldEngine
from app.world.tools import WorldTools
from tests.fakes import ScriptedLLM, complete, tool

# house.json topology:
#   hall <-> kitchen
#   hall <-> bedroom_corridor <-> master_bedroom / second_bedroom
#   hall <-> entrance <-> office
# medicine and minu are in second_bedroom; paul's schedule puts him in the office
# during working hours; moira starts in the hall.


def build(goal="Deliver medicine to paul.", conditions=None, world=None, clock=None):
    tools = WorldTools(WorldEngine(world, clock=clock), EventBus())
    task = TaskContext(goal=goal, conditions=conditions if conditions is not None
                       else [GoalCondition(kind="deliver", object="medicine", person="paul")])
    for name in ("get_status", "get_map"):
        task.record(name, {}, tools.execute(name, {}))
    return task, tools


def act(task, tools, name, **args):
    result = tools.execute(name, args)
    assert result["success"], f"{name}{args} failed: {result.get('error')}"
    task.record(name, args, result)
    return result


def decision(task):
    commands, view = observable_commands(task)
    return reflex_action(task, commands, view)


class TestDeterministicExecution:
    def test_unobserved_room_is_scanned_without_the_model(self):
        task, _ = build()
        choice = decision(task)
        assert choice is not None
        assert (choice.tool, choice.source) == ("look", "reflex")

    async def test_scan_of_new_room_consumes_no_inference(self):
        task, tools = build()
        model = ScriptedLLM([tool("report")])
        choice = await Explorer(model, tools.schemas()).decide(task, "Look around.")
        assert choice.tool == "look" and choice.source == "reflex"
        assert model.calls == []
        assert task.metrics.reflex_actions == 1

    def test_visible_required_object_is_acquired_without_the_model(self):
        task, tools = build()
        for name, args in [("look", {}), ("move_to", {"room": "bedroom_corridor"}), ("look", {}),
                           ("move_to", {"room": "second_bedroom"}), ("look", {})]:
            act(task, tools, name, **args)
        choice = decision(task)
        assert (choice.tool, choice.arguments, choice.source) == (
            "pick_up", {"object": "medicine"}, "reflex")

    def test_irrelevant_visible_object_is_never_auto_acquired(self):
        """The reflex acts on goal prerequisites, not on whatever is lying around."""
        task, tools = build(goal="Go to the second bedroom.",
                            conditions=[GoalCondition(kind="visit_room", room="second_bedroom")])
        for name, args in [("look", {}), ("move_to", {"room": "bedroom_corridor"}), ("look", {}),
                           ("move_to", {"room": "second_bedroom"}), ("look", {})]:
            act(task, tools, name, **args)
        assert decision(task) is None

    def test_present_recipient_receives_held_object_without_the_model(self):
        task, tools = build()
        for name, args in [("look", {}), ("move_to", {"room": "bedroom_corridor"}), ("look", {}),
                           ("move_to", {"room": "second_bedroom"}), ("look", {}),
                           ("pick_up", {"object": "medicine"}), ("move_to", {"room": "bedroom_corridor"}),
                           ("move_to", {"room": "hall"}), ("move_to", {"room": "entrance"}),
                           ("move_to", {"room": "office"}), ("look", {})]:
            act(task, tools, name, **args)
        choice = decision(task)
        assert choice.tool == "give" and choice.source == "reflex"
        assert choice.arguments == {"object": "medicine", "person": "paul"}

    def test_delivered_goal_does_not_trigger_another_handoff(self):
        task, tools = build()
        for name, args in [("look", {}), ("move_to", {"room": "bedroom_corridor"}), ("look", {}),
                           ("move_to", {"room": "second_bedroom"}), ("look", {}),
                           ("pick_up", {"object": "medicine"}), ("move_to", {"room": "bedroom_corridor"}),
                           ("move_to", {"room": "hall"}), ("move_to", {"room": "entrance"}),
                           ("move_to", {"room": "office"}), ("look", {}),
                           ("give", {"object": "medicine", "person": "paul"})]:
            act(task, tools, name, **args)
        assert decision(task) is None

    def test_speech_is_never_reflexive(self):
        """Talking is a judgement call even when it is the only remaining option."""
        task, tools = build(goal="Ask about the medicine.",
                            conditions=[GoalCondition(kind="find_person", person="moira")])
        act(task, tools, "look", **{})
        commands, view = observable_commands(task)
        assert "talk_to:moira" in commands
        assert decision(task) is None

    async def test_committed_route_crosses_rooms_without_inference_per_hop(self):
        """One model decision buys a whole multi-hop journey plus its arrival scans."""
        task, tools = build()
        task.long_term_memory = {"people": {"paul": {"last_seen_room": "office"}}}
        act(task, tools, "look")
        explorer = Explorer(ScriptedLLM([tool("move_to", room="entrance")]), tools.schemas())
        first = await explorer.decide(task, "Find paul.")
        assert first.source == "llm" and first.arguments == {"room": "entrance"}
        assert task.intention.destination == "office"
        act(task, tools, "move_to", room="entrance")

        silent = ScriptedLLM([])
        scan = await Explorer(silent, tools.schemas()).decide(task, "Find paul.")
        assert (scan.tool, scan.source) == ("look", "reflex")
        act(task, tools, "look")

        hop = await Explorer(silent, tools.schemas()).decide(task, "Find paul.")
        assert (hop.tool, hop.arguments, hop.source) == ("move_to", {"room": "office"}, "route_executor")
        act(task, tools, "move_to", room="office")
        assert silent.calls == []

    def test_route_is_abandoned_when_the_model_leaves_it(self):
        task, tools = build()
        act(task, tools, "look")
        task.intention = Intention(kind="search_person", target="paul", destination="office",
                                   route=["entrance", "office"])
        act(task, tools, "move_to", room="kitchen")
        assert task.intention is None


class TestSearchCoherence:
    def test_empty_scan_overrides_stale_remembered_location(self):
        task, tools = build()
        task.long_term_memory = {"people": {"paul": {"last_seen_room": "kitchen"}}}
        act(task, tools, "look")
        act(task, tools, "move_to", room="kitchen")
        act(task, tools, "look")
        act(task, tools, "move_to", room="hall")
        evidence = task.locate_evidence("person", "paul")
        assert evidence["room"] != "kitchen"
        assert evidence["source"] in {"search_frontier", "unknown"}
        assert task._navigation_hints()["recipient:paul"]["target"] != "kitchen"

    def test_direct_sighting_outranks_remembered_location(self):
        task, tools = build()
        task.long_term_memory = {"people": {"paul": {"last_seen_room": "kitchen"}}}
        for name, args in [("look", {}), ("move_to", {"room": "entrance"}), ("look", {}),
                           ("move_to", {"room": "office"}), ("look", {})]:
            act(task, tools, name, **args)
        evidence = task.locate_evidence("person", "paul")
        assert evidence == {"room": "office", "source": "direct_observation",
                            "age_actions": 0, "confidence": "observed"}

    def test_unsearched_rooms_are_preferred_over_freshly_cleared_ones(self):
        task, tools = build()
        act(task, tools, "look")
        act(task, tools, "move_to", room="kitchen")
        act(task, tools, "look")
        act(task, tools, "move_to", room="hall")
        coverage = next(s for s in task.search_state() if s["target"] == "paul")
        assert set(coverage["searched_rooms"]) == {"hall", "kitchen"}
        assert "kitchen" not in coverage["unsearched_or_stale_rooms"]
        assert task._navigation_hints()["recipient:paul"]["target"] in coverage["unsearched_or_stale_rooms"]

    def test_old_negative_observations_go_stale_and_free_the_room_again(self):
        """A cleared room must not be excluded forever: occupants move."""
        task, tools = build()
        task.search_stale_after = 2
        act(task, tools, "look")
        act(task, tools, "move_to", room="kitchen")
        act(task, tools, "look")
        coverage = next(s for s in task.search_state() if s["target"] == "paul")
        assert "kitchen" not in coverage["unsearched_or_stale_rooms"]
        for room in ["hall", "bedroom_corridor", "hall", "entrance"]:
            act(task, tools, "move_to", room=room)
        coverage = next(s for s in task.search_state() if s["target"] == "paul")
        assert coverage["searched_rooms"]["kitchen"]["stale"] is True
        assert "kitchen" in coverage["unsearched_or_stale_rooms"]

    def test_schedule_prediction_ranks_above_long_term_memory(self):
        task, tools = build(clock=WorldClock(speed=0, start_hour=10.0))
        task.long_term_memory = {
            "people": {"paul": {"last_seen_room": "kitchen"}},
            "_schedules": {"paul": [{"from_hour": 9, "to_hour": 17, "room": "office"}]}}
        act(task, tools, "get_status")
        act(task, tools, "look")
        evidence = task.locate_evidence("person", "paul")
        assert evidence["room"] == "office"
        assert evidence["source"] == "schedule_prediction"

    def test_search_destination_survives_travel(self):
        task, tools = build()
        act(task, tools, "look")
        task.intention = Intention(kind="search_person", target="paul", destination="office",
                                   route=["entrance", "office"])
        act(task, tools, "move_to", room="entrance")
        assert task.intention is not None and task.intention.destination == "office"
        assert task.intention.route == ["office"]


DELIVER_MEDICINE = [{"kind": "deliver", "object": "medicine", "person": "paul"}]


def two_room_world(tmp_path):
    """A world small enough that search coverage can genuinely be exhausted."""
    data = {"name": "Pair", "robot": {"room": "entry"},
            "rooms": {"entry": {"id": "entry", "name": "Entry", "connections": ["annex"]},
                      "annex": {"id": "annex", "name": "Annex", "connections": ["entry"]}},
            "people": {}, "objects": {}}
    path = tmp_path / "pair.json"
    path.write_text(json.dumps(data))
    return path


class TestStallDetection:
    def _manager(self, replies, conditions=DELIVER_MEDICINE, **limits):
        engine, bus = WorldEngine(), EventBus()
        return TaskManager(ScriptedLLM(replies, conditions), WorldTools(engine, bus), bus,
                           Settings(**limits))

    def test_movement_alone_is_not_semantic_progress(self):
        task, tools = build()
        act(task, tools, "look")
        before = task.progress_signature()
        act(task, tools, "move_to", room="kitchen")
        assert task.progress_signature() == before

    def test_scanning_a_new_room_is_semantic_progress(self):
        task, tools = build()
        act(task, tools, "look")
        before = task.progress_signature()
        act(task, tools, "move_to", room="kitchen")
        act(task, tools, "look")
        assert task.progress_signature() != before

    async def test_room_ping_pong_without_discoveries_ends_the_delegation(self, tmp_path):
        """Bouncing between two cleared rooms while seeking someone absent must replan."""
        engine, bus = WorldEngine(two_room_world(tmp_path)), EventBus()
        model = ScriptedLLM([
            {"action": "delegate", "summary": "Search.", "task": "Find the visitor."},
            tool("move_to", room="annex"), tool("move_to", room="entry"),
            tool("move_to", room="annex"), tool("move_to", room="entry"),
            tool("move_to", room="annex"), tool("move_to", room="entry"),
            {"action": "fail", "summary": "Stop."},
        ], conditions=[{"kind": "find_person", "person": "visitor"}])
        manager = TaskManager(model, WorldTools(engine, bus), bus,
                              Settings(max_semantic_stall=3, max_coordinator_cycles=2))
        task = manager.start("Find the visitor.")
        await manager.runner
        assert task.metrics.stall_events >= 1
        assert task.metrics.repeat_room_visits >= 1
        # Caught after three fruitless moves, far short of the 40-action ceiling.
        assert task.tool_count <= 8
        assert any("No semantic progress" in e["data"].get("summary", "")
                   for e in bus.history if e["type"] == "agent_message")

    async def test_movement_does_not_reset_the_stall_counter(self, tmp_path):
        engine, bus = WorldEngine(two_room_world(tmp_path)), EventBus()
        model = ScriptedLLM([
            {"action": "delegate", "summary": "Search.", "task": "Find the visitor."},
            tool("move_to", room="annex"), tool("move_to", room="entry"),
            tool("move_to", room="annex"), tool("move_to", room="entry"),
            {"action": "fail", "summary": "Stop."},
        ], conditions=[{"kind": "find_person", "person": "visitor"}])
        manager = TaskManager(model, WorldTools(engine, bus), bus,
                              Settings(max_semantic_stall=2, max_coordinator_cycles=2))
        task = manager.start("Find the visitor.")
        await manager.runner
        # Two consecutive barren moves are enough: neither one reset the counter.
        moves = [a for a in task.action_history if a["tool"] == "move_to"]
        assert len(moves) == 3  # First move reaches unscanned ground and is exempt.
        assert task.metrics.stall_events == 1

    async def test_useful_observation_resets_the_stall_counter(self):
        manager = self._manager([
            {"action": "delegate", "summary": "Search.", "task": "Find medicine."},
            tool("move_to", room="bedroom_corridor"),
            tool("move_to", room="second_bedroom"),
            {"action": "fail", "summary": "Stop."},
        ], max_semantic_stall=3, max_coordinator_cycles=2)
        task = manager.start("Deliver medicine to paul.")
        await manager.runner
        # Every leg ended in a scan that revealed something, so nothing stalled.
        assert task.metrics.stall_events == 0
        assert any(a["tool"] == "pick_up" and a["success"] for a in task.action_history)


class TestMemoryPrecedence:
    def test_named_recipient_survives_a_conflicting_recurring_need(self):
        task, _ = build()
        task.long_term_memory = {"people": {
            "neave": {"name": "Neave", "known_needs": ["medicine"], "last_seen_room": "master_bedroom"},
            "paul": {"name": "Paul", "known_needs": [], "last_seen_room": "office"}}}
        targets = task.authoritative_targets()
        assert targets["recipients"] == ["paul"] and targets["recipient_is_fixed_by_goal"]
        scoped = task.compact()["long_term_memory"]
        assert scoped["people"]["neave"]["known_needs"] == []
        assert task.delivery_state()[0]["recipient"] == "paul"

    def test_unnamed_recipient_keeps_remembered_needs_available(self):
        """With no recipient in the goal, remembered needs are legitimate evidence."""
        task, _ = build(goal="Find who needs the medicine and deliver it.",
                        conditions=[GoalCondition(kind="deliver", object="medicine")])
        task.long_term_memory = {"people": {"neave": {"name": "Neave", "known_needs": ["medicine"]}}}
        assert not task.authoritative_targets()["recipient_is_fixed_by_goal"]
        assert task.compact()["long_term_memory"]["people"]["neave"]["known_needs"] == ["medicine"]

    def test_schedules_are_machine_read_and_kept_out_of_the_prompt(self):
        task, _ = build()
        task.long_term_memory = {"_schedules": {"paul": [{"from_hour": 9, "to_hour": 17, "room": "office"}]}}
        assert "_schedules" not in task.compact()["long_term_memory"]


class TestDelegationLifetime:
    def _manager(self, replies, conditions=DELIVER_MEDICINE, **limits):
        engine, bus = WorldEngine(), EventBus()
        return TaskManager(ScriptedLLM(replies, conditions), WorldTools(engine, bus), bus,
                           Settings(**limits)), engine

    async def test_healthy_delegation_runs_well_past_the_old_budget(self):
        manager, engine = self._manager([
            {"action": "delegate", "summary": "Deliver.", "task": "Take the medicine to paul."},
            tool("move_to", room="bedroom_corridor"),
            tool("move_to", room="second_bedroom"),
        ])
        task = manager.start("Deliver medicine to paul.")
        await manager.runner
        assert task.status == "completed", task.summary
        assert engine.snapshot()["objects"]["medicine"]["location"] == {"kind": "person", "id": "paul"}
        # Many embodied actions inside ONE delegation: the old budget was eight,
        # and every doorway used to cost an inference. Recipient search is now reflex.
        assert task.metrics.coordinator_handoffs == 1
        assert task.tool_count > 8
        assert task.metrics.explorer_llm_calls == 2
        assert task.metrics.reflex_actions + task.metrics.route_actions == 14

    async def test_completed_objective_returns_control_before_the_ceiling(self):
        manager, _ = self._manager([
            {"action": "delegate", "summary": "Scan.", "task": "Visit the kitchen."},
            tool("move_to", room="kitchen"),
            complete, {"approved": True, "summary": "Room visited."},
        ], conditions=[{"kind": "visit_room", "room": "kitchen"}], max_explorer_actions=40)
        task = manager.start("Go to the kitchen.")
        await manager.runner
        assert task.status == "completed", task.summary
        assert task.tool_count < 10  # Stopped on success, not on the ceiling.

    async def test_hard_ceiling_still_stops_a_runaway_delegation(self, tmp_path):
        engine, bus = WorldEngine(two_room_world(tmp_path)), EventBus()
        model = ScriptedLLM([
            {"action": "delegate", "summary": "Wander.", "task": "Keep moving."},
            *([tool("move_to", room="annex"), tool("move_to", room="entry")] * 6),
            {"action": "fail", "summary": "Stop."},
        ], conditions=[{"kind": "find_person", "person": "visitor"}])
        manager = TaskManager(model, WorldTools(engine, bus), bus,
                              Settings(max_explorer_actions=3, max_semantic_stall=20,
                                       max_coordinator_cycles=2))
        task = manager.start("Find the visitor.")
        await manager.runner
        # Bootstrap plus exactly the ceiling's worth of delegated actions.
        assert task.tool_count == 2 + 3
        assert any("safety ceiling" in e["data"].get("summary", "")
                   for e in bus.history if e["type"] == "agent_message")


class TestSimulationModes:
    def test_action_driven_clock_ignores_wall_time(self, monkeypatch):
        now = [1000.0]
        monkeypatch.setattr("app.world.clock.time.monotonic", lambda: now[0])
        clock = WorldClock(speed=60, start_hour=8.0, mode="action_driven", action_seconds=60)
        assert clock.time_str() == "08:00"
        now[0] += 3600  # An hour of inference latency.
        assert clock.time_str() == "08:00"
        clock.advance()
        assert clock.time_str() == "08:01"

    def test_realtime_clock_still_follows_wall_time(self, monkeypatch):
        now = [1000.0]
        monkeypatch.setattr("app.world.clock.time.monotonic", lambda: now[0])
        clock = WorldClock(speed=60, start_hour=8.0)
        now[0] += 600
        assert clock.time_str() == "18:00"
        clock.advance()  # No effect in realtime mode.
        assert clock.time_str() == "18:00"

    def test_world_advances_per_action_not_per_second(self, monkeypatch):
        now = [500.0]
        monkeypatch.setattr("app.world.clock.time.monotonic", lambda: now[0])
        clock = WorldClock(speed=60, start_hour=8.0, mode="action_driven", action_seconds=600)
        tools = WorldTools(WorldEngine(clock=clock), EventBus())
        assert tools.execute("get_status", {})["observation"]["time"] == "08:00"
        now[0] += 10_000  # Slow model; the world must not move.
        assert tools.execute("get_status", {})["observation"]["time"] == "08:00"
        tools.execute("look", {})
        assert tools.execute("get_status", {})["observation"]["time"] == "08:10"

    def test_npc_movement_is_driven_by_actions_in_action_driven_mode(self, monkeypatch):
        """Paul reaches his office shift through robot turns, not through latency."""
        now = [0.0]
        monkeypatch.setattr("app.world.clock.time.monotonic", lambda: now[0])
        clock = WorldClock(speed=60, start_hour=8.0, mode="action_driven", action_seconds=1800)
        engine = WorldEngine(clock=clock)
        tools = WorldTools(engine, EventBus())
        assert engine.snapshot()["people"]["paul"]["room"] != "office"
        now[0] += 100_000
        assert engine.snapshot()["people"]["paul"]["room"] != "office"
        for _ in range(4):  # Four half-hour turns reach 10:00.
            tools.execute("look", {})
        assert clock.time_str() == "10:00"
        assert engine.snapshot()["people"]["paul"]["room"] == "office"

    def test_action_driven_runs_are_reproducible(self, monkeypatch):
        def sequence():
            clock = WorldClock(speed=60, start_hour=8.0, mode="action_driven", action_seconds=900)
            tools = WorldTools(WorldEngine(clock=clock), EventBus())
            times = []
            for room in ["kitchen", "hall", "entrance", "office"]:
                tools.execute("move_to", {"room": room})
                tools.execute("look", {})
                times.append(tools.execute("get_status", {})["observation"]["time"])
            return times

        now = [0.0]
        monkeypatch.setattr("app.world.clock.time.monotonic", lambda: now[0])
        first = sequence()
        now[0] += 54_321  # Arbitrary latency between runs changes nothing.
        assert sequence() == first


class TestConversationValue:
    async def test_verbatim_repeat_is_rejected_without_a_classifier_call(self):
        class NoClassifier(ScriptedLLM):
            async def generate(self, messages, response_schema):
                assert response_schema.get("title") != "ConversationMove", \
                    "a verbatim repeat must be caught locally, not by the model"
                return await super().generate(messages, response_schema)

        engine, bus = WorldEngine(), EventBus()
        model = NoClassifier([
            {"action": "delegate", "summary": "Ask.", "task": "Ask moira about paul."},
            tool("talk_to", person="moira", message="Where is Paul?"),
            tool("talk_to", person="moira", message="Where is Paul?"),
            {"action": "fail", "summary": "Stop."},
        ], conditions=[{"kind": "find_person", "person": "paul"}])
        manager = TaskManager(model, WorldTools(engine, bus), bus,
                              Settings(max_coordinator_cycles=2))
        task = manager.start("Find paul.")
        await manager.runner
        assert sum(a["tool"] == "talk_to" for a in task.action_history) == 1
        assert any("Conversation move rejected" in m for m in task.action_feedback)
        assert task.metrics.conversation_llm_calls == 1  # Only the first interpretation.

    def test_exhausted_source_is_flagged_but_still_reachable(self):
        task, tools = build(goal="Ask about paul.",
                            conditions=[GoalCondition(kind="find_person", person="paul")])
        act(task, tools, "look")
        result = act(task, tools, "talk_to", person="moira", message="Where is Paul?")
        task.memory.remember_conversation(
            "moira", "thread_1", "Locate paul", True,
            result["observation"]["message"], result["observation"]["response"],
            result["evidence_id"])
        commands, _ = observable_commands(task)
        assert "talk_to:moira" in commands  # Never removed: new purposes remain valid.
        assert "expect no new information" in commands["talk_to:moira"]["description"]


class TestInstrumentation:
    async def test_task_metrics_separate_inference_from_reflex(self):
        engine, bus = WorldEngine(), EventBus()
        model = ScriptedLLM([
            {"action": "delegate", "summary": "Deliver.", "task": "Take medicine to paul."},
            tool("move_to", room="bedroom_corridor"),
            tool("move_to", room="second_bedroom"),
            tool("move_to", room="bedroom_corridor"),
            tool("move_to", room="bedroom_corridor"),
            tool("move_to", room="office"),
            {"approved": True, "summary": "Held and present."},
            complete, {"approved": True, "summary": "Evidenced."},
        ], DELIVER_MEDICINE)
        manager = TaskManager(model, WorldTools(engine, bus), bus, Settings())
        task = manager.start("Deliver medicine to paul.")
        await manager.runner
        metrics = task.metrics
        assert task.status == "completed", task.summary
        assert metrics.llm_calls == (metrics.coordinator_llm_calls + metrics.explorer_llm_calls
                                     + metrics.critic_llm_calls + metrics.conversation_llm_calls)
        assert metrics.reflex_actions + metrics.route_actions > metrics.explorer_llm_calls
        assert metrics.tool_actions == task.tool_count
        finished = next(e for e in bus.history if e["type"] == "task_completed")
        assert finished["data"]["metrics"]["explorer_llm_calls"] == metrics.explorer_llm_calls

    def test_ollama_metrics_are_read_from_the_response_not_invented(self):
        import httpx
        from app.llm.ollama import OllamaClient

        def respond(request):
            return httpx.Response(200, json={
                "model": "test-model",
                "message": {"content": '{"approved":true,"summary":"OK"}'},
                "total_duration": 2_000_000_000, "prompt_eval_count": 120,
                "prompt_eval_duration": 500_000_000, "eval_count": 30,
                "eval_duration": 1_500_000_000})

        async def go():
            client = OllamaClient(Settings())
            await client.http.aclose()
            client.http = httpx.AsyncClient(transport=httpx.MockTransport(respond),
                                            base_url="http://test")
            await client.generate([], {})
            await client.close()
            return client.last_metrics

        import asyncio
        metrics = asyncio.run(go())
        assert metrics == {"model": "test-model", "total_seconds": 2.0, "prompt_eval_seconds": 0.5,
                           "eval_seconds": 1.5, "prompt_tokens": 120, "completion_tokens": 30}
        assert "load_seconds" not in metrics  # Absent from the response, so never reported.
