import asyncio
import logging

from app.agents.coordinator import Coordinator
from app.agents.explorer import Explorer
from app.agents.conversation import classify_conversation_move, interpret_conversation
from app.agents.prompts import CRITIC
from app.agents.schemas import CriticReview
from app.config import Settings
from app.events.bus import EventBus
from app.llm.base import LLMClient, ModelError, structured
from app.memory.store import PersistentMemory
from app.world.tools import WorldTools
from .models import TaskContext
from .goals import check_conditions, normalize


class TaskBusy(ValueError):
    pass


_log = logging.getLogger("agentic_friend.tasks")


async def critic_review(client: LLMClient, context: TaskContext, proposal: str, kind: str,
                        evidence: list[dict] | None = None,
                        proposed_action: dict | None = None) -> CriticReview:
    """Run one independent critic review. Strips prior verdicts so each review is unbiased."""
    state = context.compact()
    state.pop("critic_feedback", None)
    return await structured(client, CriticReview, CRITIC,
                            {**state, "proposal": proposal,
                             "proposed_action": proposed_action,
                             "review_type": kind, "cited_evidence": evidence or []},
                            metrics=context.metrics, role="critic")


def _repeats_previous_message(context: TaskContext, person: str, message: str) -> bool:
    """Cheap local check for an already-asked question.

    Catches the common repeat without spending an inference call on the
    conversation classifier; genuinely new wording still goes to the model.
    """
    known = context.memory.people.get(person)
    if not known:
        return False
    current = normalize(message)
    return any(normalize(turn.message) == current
               for thread in known.conversation_threads.values() for turn in thread.turns)


class LimitReached(RuntimeError):
    pass


class TaskManager:
    def __init__(self, client: LLMClient, tools: WorldTools, bus: EventBus, settings: Settings,
                 persistent: PersistentMemory | None = None) -> None:
        self.tools, self.bus, self.settings = tools, bus, settings
        self.client = client
        self.persistent = persistent
        self.coordinator = Coordinator(client)
        self.explorer = Explorer(client, tools.schemas())
        self.tasks: dict[str, TaskContext] = {}
        self.active_id: str | None = None
        self.runner: asyncio.Task | None = None

    def start(self, goal: str) -> TaskContext:
        if self.active_id:
            raise TaskBusy("One task is already running. Cancel it or wait for completion.")
        context = TaskContext(goal=goal, search_stale_after=self.settings.search_stale_after)
        if self.persistent:
            context.long_term_memory = self.persistent.prompt()
            context.self_model = self.persistent.self_model_prompt()
        # Inject NPC schedules so location evidence can be predicted from clock time.
        context.long_term_memory["_schedules"] = self.tools._engine.people_schedules()
        context.metrics.simulated_start = self.tools._engine.simulated_time() or ""
        # Keep recent tasks in memory only. Active task is never evicted.
        if len(self.tasks) >= 50:
            del self.tasks[next(iter(self.tasks))]
        self.tasks[context.id] = context
        self.active_id = context.id
        self.bus.emit("task_started", task=context.public(), summary=goal)
        self.runner = asyncio.create_task(self._run(context), name=f"task-{context.id}")
        return context

    def active(self) -> dict | None:
        return self.tasks[self.active_id].public() if self.active_id else None

    def message(self, context: TaskContext, agent: str, summary: str) -> None:
        context.summary = summary
        self.bus.emit("agent_message", agent, task_id=context.id, summary=summary)

    def finish(self, context: TaskContext, status: str, summary: str) -> None:
        context.status = status
        context.summary = summary
        context.metrics.tool_actions = context.tool_count
        context.metrics.simulated_end = self.tools._engine.simulated_time() or ""
        report = context.metrics.model_dump()
        _log.info("task_metrics task=%s status=%s %s", context.id, status,
                  " ".join(f"{key}={value}" for key, value in report.items() if value not in ("", 0, 0.0)))
        self.bus.emit(f"task_{status}", task=context.public(), summary=summary, metrics=report)

    async def cancel(self, task_id: str) -> TaskContext:
        context = self.tasks[task_id]
        if context.status == "running" and self.runner:
            self.runner.cancel()
            try:
                await self.runner
            except asyncio.CancelledError:
                pass
            # Covers cancellation before the coroutine starts executing.
            if context.status == "running":
                self.finish(context, "cancelled", "Task cancelled. Completed world actions are retained.")
            self.active_id = None
        return context

    async def close(self) -> None:
        if self.active_id:
            await self.cancel(self.active_id)

    def call_tool(self, context: TaskContext, tool: str, arguments: dict) -> dict:
        if context.tool_count >= self.settings.max_tool_calls:
            raise LimitReached("Maximum total tool calls reached.")
        context.tool_count += 1
        result = self.tools.execute(tool, arguments, context.id)
        context.record(tool, arguments, result)
        self.bus.emit("task_updated", task=context.public())
        return result

    async def review(self, context: TaskContext, proposal: str, kind: str,
                     evidence: list[dict] | None = None,
                     proposed_action: dict | None = None) -> bool:
        if context.critic_count >= self.settings.max_critic_reviews:
            raise LimitReached("Maximum Critic reviews reached; stopping repeated debate.")
        context.critic_count += 1
        self.bus.emit("agent_active", "critic", task_id=context.id)
        review = await critic_review(self.client, context, proposal, kind, evidence, proposed_action)
        context.critic_feedback.append({"proposal": proposal, **review.model_dump(),
                                       "kind": kind,
                                       "observed_action_count": len(context.action_history)})
        self.bus.emit("critic_review", "critic", task_id=context.id, **review.model_dump())
        return review.approved

    def _progress_changed(self, context: TaskContext, previous: tuple | None = None) -> bool:
        signature = context.progress_signature()
        current = previous if previous is not None else context.last_progress_signature
        context.last_progress_signature = signature
        return current != signature

    async def _loop(self, context: TaskContext) -> None:
        # Bootstrap through a permitted tool; no agent gets an omniscient snapshot.
        self.call_tool(context, "get_status", {})
        self.call_tool(context, "get_map", {})
        self.bus.emit("agent_active", "coordinator", task_id=context.id)
        goal_plan = await self.coordinator.define_goal(context.goal, context.floor_plan)
        context.conditions = goal_plan.conditions
        self.message(context, "coordinator", "Required outcomes defined. Locating targets requires tool observations.")
        for cycle in range(self.settings.max_coordinator_cycles):
            context.cycle_count = cycle + 1
            self.bus.emit("task_updated", task=context.public())
            self.bus.emit("agent_active", "coordinator", task_id=context.id)
            decision = await self.coordinator.decide(context)
            public_summary = {
                "complete": "Checking goal completion against tool evidence.",
                "delegate": "Delegating: " + decision.task,
                "consult_critic": "Reviewing plan: " + decision.plan,
                "fail": decision.summary,
            }[decision.action]
            self.message(context, "coordinator", public_summary)
            if decision.action == "fail":
                self.finish(context, "failed", decision.summary)
                return
            if decision.action == "consult_critic":
                await self.review(context, decision.plan, "plan")
            elif decision.action == "complete":
                unmet = [condition for condition in check_conditions(context) if not condition["satisfied"]]
                if unmet:
                    context.critic_feedback.append({"source": "system", "approved": False,
                        "observed_action_count": len(context.action_history),
                        "summary": "Completion rejected: required outcomes lack tool evidence.",
                        "unmet_conditions": unmet,
                        "suggestion": "Complete the missing outcomes using tools."})
                    self.message(context, "system", "Completion rejected: required outcomes have not been observed.")
                    continue
                by_id = {entry["evidence_id"]: entry for entry in context.action_history
                         if entry["success"] and entry["tool"] != "get_status"}
                # The deterministic condition check above is authoritative; evidence IDs
                # are advisory hints that help the Critic focus its review. If the model
                # cited valid IDs use those; otherwise fall back to all successful
                # observations so a hallucinated UUID cannot block a verified completion.
                cited = [by_id[id] for id in decision.evidence_ids if id in by_id]
                evidence = cited if cited else list(by_id.values())
                if await self.review(context, decision.summary, "completion", evidence):
                    self.finish(context, "completed", decision.summary)
                    return
            elif decision.action == "delegate":
                context.current_plan = decision.task
                starting_tool_count = context.tool_count
                context.stall_count = 0
                for step in range(self.settings.max_explorer_actions):
                    # A healthy delegation ends here, not at the action ceiling.
                    if step and not [c for c in check_conditions(context) if not c["satisfied"]]:
                        self.message(context, "system",
                                     "Delegated objective satisfied; returning to Coordinator.")
                        break
                    self.bus.emit("agent_active", "explorer", task_id=context.id)
                    action = await self.explorer.decide(context, decision.task)
                    if action.action == "report":
                        if (any(condition.kind == "deliver" for condition in context.conditions)
                            and context.robot_status.get("room") not in context.memory.rooms):
                            correction = "Report rejected: the current room has not been observed. Scan it before reporting or replanning."
                            context.feedback(correction)
                            self.message(context, "system", correction)
                            continue
                        if context.tool_count == starting_tool_count:
                            previous_rejection = next((review for review in reversed(context.critic_feedback)
                                if review.get("kind") == "delegation_report"
                                and not review["approved"]
                                and review.get("observed_action_count") == len(context.action_history)), None)
                            approved = False if previous_rejection else await self.review(
                                context, action.summary, "delegation_report")
                            if not approved:
                                context.consecutive_report_rejections += 1
                                if context.consecutive_report_rejections >= 3:
                                    raise LimitReached("Explorer repeatedly reported without acting after feedback. "
                                                       "Stopping the unchanged decision loop; task remains incomplete.")
                                unexplored = context.compact()["known_but_unobserved_rooms"]
                                correction = ("Report rejected: no new tool evidence supports progress. "
                                              "Choose a tool action to continue the unfinished work; "
                                              "an intention to search is not an executed search. "
                                              f"Rooms without a look observation: {unexplored}.")
                                context.feedback(correction)
                                self.message(context, "system", correction)
                                continue
                        context.explorer_reports.append(action.summary)
                        self.message(context, "explorer", "Returning observations to Coordinator.")
                        break
                    self.message(context, "explorer", f"Next action: {action.tool}.")
                    previous = context.action_history[-1] if context.action_history else None
                    if (previous and previous["success"] and action.tool != "talk_to"
                            and previous["tool"] == action.tool and previous["arguments"] == action.arguments):
                        correction = "Repeated identical command rejected. Use its existing observation and choose a different action."
                        context.feedback(correction)
                        self.message(context, "system", correction)
                        continue
                    conversation_move = None
                    conversation_thread_id = None
                    if action.tool == "talk_to":
                        person = action.arguments["person"]
                        verbatim_repeat = _repeats_previous_message(
                            context, person, action.arguments["message"])
                        conversation_move = None if verbatim_repeat else await classify_conversation_move(
                            self.client, context, person, action.arguments["message"],
                        )
                        if verbatim_repeat or not conversation_move.advances_thread:
                            context.conversation_rejections[person] = context.conversation_rejections.get(person, 0) + 1
                            target = ("a question already put to this person" if verbatim_repeat
                                      else conversation_move.existing_thread_id or "an existing discussion")
                            correction = (
                                f"Conversation move rejected: it repeats or does not advance {target}. "
                                "Choose a movement or object action next; speech resumes after successful physical action."
                            )
                            context.feedback(correction)
                            self.message(context, "system", correction)
                            continue
                        known = context.memory.people.get(action.arguments["person"])
                        if (conversation_move.existing_thread_id and known
                                and conversation_move.existing_thread_id in known.conversation_threads):
                            conversation_thread_id = conversation_move.existing_thread_id
                        else:
                            conversation_thread_id = context.memory.next_thread_id(action.arguments["person"])
                    if action.tool == "give" and action.source != "reflex":
                        # Reflex give: state machine already verified object-in-inventory
                        # AND recipient-present, so the Critic cannot add information and
                        # only risks blocking a correct action or introducing latency that
                        # lets the recipient move away before the tool fires.
                        proposal = f"Execute {action.tool} with arguments {action.arguments}."
                        if not await self.review(context, proposal, "object_transfer",
                                                 proposed_action={"tool": action.tool, "arguments": action.arguments}):
                            self.message(context, "system", "Object action rejected by Critic; revising the plan.")
                            break
                    previous_signature = context.progress_signature()
                    # Travelling to unscanned ground is purposeful even before the
                    # scan proves it; only re-entering cleared rooms can be a loop.
                    on_route = (action.source == "route_executor"
                                or (action.tool == "move_to"
                                    and action.arguments.get("room") not in context.memory.rooms))
                    revisit = action.arguments.get("room") in context.memory.visited_rooms
                    result = self.call_tool(context, action.tool, action.arguments)
                    if result.get("success") and action.tool == "give":
                        # Delivery conditions are machine-checkable from world tool
                        # evidence: no LLM can add information the state machine
                        # doesn't already have. Skip the Coordinator + Critic
                        # round-trip (~2 min at qwen3:4b latency).
                        if not [c for c in check_conditions(context) if not c["satisfied"]]:
                            obj = action.arguments.get("object", "object")
                            person = action.arguments.get("person", "recipient")
                            context.metrics.coordinator_handoffs += 1
                            self.finish(context, "completed",
                                        f"{obj.capitalize()} delivered to {person}. All goal conditions met.")
                            return
                    if result.get("success") and action.tool == "talk_to":
                        await interpret_conversation(
                            self.client, context, result, conversation_thread_id,
                            conversation_move.thread_summary,
                        )
                    if self._stalled(context, action, result, previous_signature, on_route, revisit):
                        self.message(context, "system",
                                     "No semantic progress after repeated actions; replanning with the Coordinator.")
                        break
                    # Let cancellation, sockets and other API requests run between tools.
                    await asyncio.sleep(0)
                else:
                    self.message(context, "system",
                                 "Delegation safety ceiling reached; returning to Coordinator.")
                context.metrics.coordinator_handoffs += 1
            self.bus.emit("task_updated", task=context.public())
        raise LimitReached("Maximum Coordinator cycles reached.")

    def _stalled(self, context: TaskContext, action, result: dict,
                 previous: tuple, on_route: bool, revisit: bool = False) -> bool:
        """Track semantic progress and report when the current strategy is spent.

        Travelling along a committed route is purposeful even though it changes
        no task state, so hops are exempt. A leg that ends with the destination
        scanned and the target still missing counts once, which is what turns
        repeated fruitless room visits into a replan instead of a loop.
        """
        if not result.get("success"):
            return False
        if action.tool == "move_to" and revisit:
            context.metrics.repeat_room_visits += 1
        advanced = self._progress_changed(context, previous)
        if advanced:
            context.metrics.useful_observations += 1
            context.stall_count = 0
            return False
        if on_route:
            return False
        context.stall_count += 1
        if context.stall_count < self.settings.max_semantic_stall:
            return False
        context.metrics.stall_events += 1
        context.stall_count = 0
        # The committed destination produced nothing; let the planner pick another.
        context.intention = None
        return True

    async def _run(self, context: TaskContext) -> None:
        try:
            async with asyncio.timeout(self.settings.task_timeout_seconds):
                await self._loop(context)
        except asyncio.CancelledError:
            self.finish(context, "cancelled", "Task cancelled. Completed world actions are retained.")
        except TimeoutError:
            self.finish(context, "failed", "Task time limit reached.")
        except (ModelError, LimitReached) as exc:
            self.finish(context, "failed", str(exc))
        except Exception:
            logging.getLogger("agentic_friend.tasks").exception("Unexpected task failure")
            self.finish(context, "failed", "Unexpected task error. Check the backend log.")
        finally:
            if self.active_id == context.id:
                self.active_id = None
            if self.persistent:
                try:
                    self.persistent.update_from_task(context)
                    self.persistent.save(self.settings.memory_file)
                except Exception:
                    logging.getLogger("agentic_friend.tasks").exception("Failed to save persistent memory")
