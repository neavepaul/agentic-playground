import asyncio
import logging
import re

from app.agents.coordinator import Coordinator
from app.agents.explorer import Explorer
from app.agents.conversation import classify_conversation_move, interpret_conversation
from app.agents.prompts import CRITIC
from app.agents.schemas import CoordinatorDecision, CriticReview
from app.config import Settings
from app.capabilities.tools import CapabilityTools, DIGITAL_TOOLS
from app.events.bus import EventBus
from app.llm.base import LLMClient, ModelError, structured
from app.memory.store import PersistentMemory
from app.world.tools import WorldTools
from .models import TaskContext
from .goals import check_conditions, normalize
from .permissions import (DIGITAL_LOOKUP, CALCULATION, TOOL_PLAIN, Grant, Restrictions,
                          explicitly_requested_tools, plain)
from app.agents.schemas import GoalCondition


_CLAIMS_A_RUN = re.compile(r"\b(ran|run|executed|computed|calculated)\b[^.]{0,40}\b(python|code|program|script|"
                           r"locally|calculation)\b|```python")


def answer_notes(context: TaskContext) -> list[str]:
    """Footnotes the final answer must carry to stay true to the evidence."""
    notes = []
    answer = (context.answer or {}).get("answer", "")
    ran_python = any(e["tool"] == "run_python" and e["success"] for e in context.action_history)
    if answer and not ran_python and _CLAIMS_A_RUN.search(answer.casefold()):
        notes.append("No program actually ran for this: any figure above was worked out, not executed.")
    return notes


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


TERMINAL = {"completed", "failed", "cancelled"}
# Outcomes that rest on interpretation (who needs something) still get a Critic review;
# everything else is decided by deterministic checks against tool evidence.
JUDGMENT_KINDS = {"identify_recipient"}


def verified_complete(context: TaskContext) -> bool:
    """Every required outcome is satisfied by tool evidence and none needs judgment."""
    outcomes = check_conditions(context)
    return (bool(outcomes) and all(c["satisfied"] for c in outcomes)
            and not any(c.kind in JUDGMENT_KINDS for c in context.conditions))


def relevant_evidence(context: TaskContext, limit: int = 12) -> list[dict]:
    """Recent, task-relevant observations for a review; never the whole map or history."""
    useful = [e for e in context.action_history
              if e["success"] and e["tool"] not in {"get_status", "get_map"}]
    return useful[-limit:]


class TaskManager:
    def __init__(self, client: LLMClient, tools: WorldTools, bus: EventBus, settings: Settings,
                 persistent: PersistentMemory | None = None, persona=None, memory_store=None) -> None:
        self.tools, self.bus, self.settings = tools, bus, settings
        self.client = client
        self.persistent = persistent
        self.persona = persona
        self.memory_store = memory_store
        self.coordinator = Coordinator(client, digital_enabled=settings.digital_tools_enabled)
        self.capabilities = CapabilityTools(bus, enabled=settings.digital_tools_enabled)
        self.explorer = Explorer(client, tools.schemas(), persona=persona)
        self.tasks: dict[str, TaskContext] = {}
        self.active_id: str | None = None
        self.runner: asyncio.Task | None = None
        # What people asked Kip not to do. Shared and live: it binds every task,
        # including ones already queued or running, at the moment a tool would run.
        self.restrictions = Restrictions()

    def start(self, goal: str, grant: Grant | None = None, plan=None) -> TaskContext:
        """Start a task. Without a grant it is a direct instruction (API): the goal
        text is the person's own words and authorizes every kind of action."""
        if self.active_id:
            raise TaskBusy("One task is already running. Cancel it or wait for completion.")
        if grant is None:
            # API goals are the person's own words. Internal/model-written goals carry
            # an explicit grant and must never change the household's restrictions.
            rooms = {key: room.get("name", key)
                     for key, room in self.tools._engine.snapshot().get("rooms", {}).items()}
            self.restrictions.update(goal, rooms)
        grant = grant or Grant.full("api", goal)
        context = TaskContext(goal=goal, search_stale_after=self.settings.search_stale_after, grant=grant,
                              requested_tools=explicitly_requested_tools(grant.authorized_text))
        self._sync_permissions(context)
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
        self.runner = asyncio.create_task(self._run(context, plan), name=f"task-{context.id}")
        return context

    def active(self) -> dict | None:
        return self.tasks[self.active_id].public() if self.active_id else None

    def message(self, context: TaskContext, agent: str, summary: str) -> None:
        context.summary = summary
        self.bus.emit("agent_message", agent, task_id=context.id, summary=summary)

    def finish(self, context: TaskContext, status: str, summary: str) -> None:
        if context.status in TERMINAL:
            # Terminal transitions happen once; a late cancel cannot rewrite an outcome.
            _log.info("Ignoring %s for task %s, already %s", status, context.id, context.status)
            return
        if status == "completed":
            context.answer_notes = answer_notes(context)
            if context.blocked:
                # Never present a partial result as the whole thing.
                summary = f"Partly done. {summary} Not done: {'; '.join(context.blocked)}."
        context.status = status
        context.summary = summary
        context.metrics.tool_actions = context.tool_count
        context.metrics.simulated_end = self.tools._engine.simulated_time() or ""
        report = context.metrics.model_dump()
        _log.info("task_metrics task=%s status=%s %s", context.id, status,
                  " ".join(f"{key}={value}" for key, value in report.items() if value not in ("", 0, 0.0)))
        # The persona appraises the outcome from this event; it adds no stock reaction line.
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
            if self.active_id == task_id:
                self.active_id = None
        return context

    async def close(self) -> None:
        if self.active_id:
            await self.cancel(self.active_id)

    # ── permission at the point of execution ──────────────────────────────────
    def _sync_permissions(self, context: TaskContext) -> None:
        objects = self.tools._engine.snapshot().get("objects", {})
        context.restricted = {"tools": sorted(self.restrictions.tools), "effects": sorted(self.restrictions.effects),
                              "rooms": sorted(self.restrictions.rooms),
                              "objects": {key: {"name": item.get("name", key), "sensitive": True}
                                          for key, item in objects.items() if item.get("sensitive")}}

    def _refused(self, context: TaskContext, tool: str, arguments: dict) -> dict | None:
        """Checked immediately before every tool call, against the live restrictions."""
        self._sync_permissions(context)
        reason = context.grant.deny(tool, arguments, self.restrictions, context.restricted["objects"])
        if not reason:
            return None
        error = f"Not permitted: {reason}. Choose a permitted action or report the blockage."
        event = self.bus.emit("tool_refused", "system", task_id=context.id, tool=tool, arguments=arguments,
                              success=False, error=error, summary=f"Refused {tool}: {reason}.")
        result = {"success": False, "error": error, "refused": True, "evidence_id": event["id"]}
        context.record(tool, arguments, result)
        context.feedback(error)
        if self.restrictions.forbids_tool(tool):
            # Ruled out mid-task: the rest can still finish, and this is reported as not done.
            self._drop_requirement(context, tool)
        _log.info("Refused %s for task %s: %s", tool, context.id, reason)
        return result

    def _drop_requirement(self, context: TaskContext, tool: str) -> None:
        for index, condition in enumerate(context.conditions):
            if tool in condition.required_tools:
                kept = [t for t in condition.required_tools if t != tool]
                context.conditions[index] = condition.model_copy(update={"required_tools": kept})
        note = f"{TOOL_PLAIN.get(tool, tool)} (you asked me not to)"
        if tool in context.requested_tools and note not in context.blocked:
            context.blocked.append(note)

    def _require_requested_tools(self, context: TaskContext) -> None:
        """What the person explicitly asked to have run is a required outcome, tracked on
        its own: a plan that forgets it cannot be reported as complete."""
        wanted = [t for t in context.requested_tools if t in {"web_search", "read_webpage", "run_python"}]
        if not wanted:
            return
        for index, condition in enumerate(context.conditions):
            if condition.kind == "answer":
                tools = sorted(set(condition.required_tools) | set(wanted))[:3]
                context.conditions[index] = condition.model_copy(update={"required_tools": tools})
                return
        context.conditions.append(GoalCondition(kind="answer", required_tools=wanted[:3]))

    def _outside_grant(self, context: TaskContext) -> bool:
        """After planning, before acting: stop if the plan needs more than was authorized."""
        self._require_requested_tools(context)
        for condition in list(context.conditions):
            for tool in condition.required_tools:
                if self.restrictions.forbids_tool(tool):
                    # A tool the person ruled out is dropped from the plan rather than run.
                    self._drop_requirement(context, tool)
        uncovered = context.grant.uncovered(context.conditions, self.restrictions)
        ruled_out = {effect for effect in uncovered if effect in self.restrictions.effects}
        if any(c.kind == "answer" for c in context.conditions) and not any(
                context.permits(tool, {}) for tool in ("web_search", "read_webpage", "run_python")):
            ruled_out.add(DIGITAL_LOOKUP if context.grant.effects and DIGITAL_LOOKUP in context.grant.effects
                          else CALCULATION)
            uncovered |= ruled_out
        if not uncovered:
            return False
        if ruled_out:
            summary = ("Not started: you asked me not to do what this needs: "
                       + ", ".join(self.restrictions.describe()) + ".")
        else:
            summary = f"Needs a go-ahead first: this would mean {plain(uncovered)}."
        self.bus.emit("task_needs_permission", "system", task_id=context.id, goal=context.goal,
                      needs=sorted(uncovered), ruled_out=sorted(ruled_out), summary=summary)
        self.finish(context, "cancelled", summary)
        return True

    async def _report_execution(self, context: TaskContext) -> bool:
        """A request to *run* Python, answered by the run itself: once the program has run,
        its code and output are the answer. No model call is needed to restate them."""
        wanted = {t for c in context.conditions if c.kind == "answer" for t in c.required_tools}
        # Asked for by the person, or chosen by Kip himself ("Use Python to ..." as his own goal).
        chosen = "run_python" in context.requested_tools or "python" in context.goal.casefold()
        if wanted != {"run_python"} or len(context.conditions) != 1 or not chosen:
            return False
        runs = [e for e in context.action_history if e["tool"] == "run_python" and e["success"]]
        if not runs:
            return False
        run = runs[-1]
        obs = run["observation"]
        stdout, value = (obs.get("stdout") or "").strip(), obs.get("result", "")
        output = stdout or ("" if value in ("", "None") else value)
        if not output:
            return False
        code = run["arguments"].get("code", "").strip()
        # Said aloud: the result, not the program. The code stays in the task record.
        text = f"Ran the numbers: {output}."
        result = await self.call_capability(context, "finish_answer", {"answer": text,
                                                                      "evidence_ids": [run["evidence_id"]]})
        if result.get("success") and not [c for c in check_conditions(context) if not c["satisfied"]]:
            self.finish(context, "completed", "Answer ready from the program's own output.")
            return True
        return False

    def call_tool(self, context: TaskContext, tool: str, arguments: dict) -> dict:
        refused = self._refused(context, tool, arguments)
        if refused:
            return refused
        if context.tool_count >= self.settings.max_tool_calls:
            raise LimitReached("Maximum total tool calls reached.")
        context.tool_count += 1
        result = self.tools.execute(tool, arguments, context.id)
        # When it was seen, on the house clock: recaps must tell fresh from old observations.
        result["observed_at"] = self.tools._engine.simulated_time() or ""
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

    async def call_capability(self, context: TaskContext, tool: str, arguments: dict) -> dict:
        refused = self._refused(context, tool, arguments)
        if refused:
            return refused
        if context.tool_count >= self.settings.max_tool_calls:
            raise LimitReached("Maximum total tool calls reached.")
        context.tool_count += 1
        result = await self.capabilities.execute(tool, arguments, context)
        self._record_capability(context, tool, arguments, result)
        return result

    def _record_capability(self, context: TaskContext, tool: str, arguments: dict, result: dict) -> None:
        context.record(tool, arguments, result)
        if result["success"]:
            context.metrics.useful_observations += 1
        if not result["success"]:
            context.feedback(result["error"])
            self.message(context, "system", "Tool attempt rejected; continuing with the error feedback.")
        self.bus.emit("task_updated", task=context.public())

    async def call_capability_batch(self, context: TaskContext, calls: list[dict]) -> list[dict]:
        from app.capabilities.schemas import BatchAction
        from app.capabilities.tools import validation_reasons
        from pydantic import ValidationError
        try:
            validated = BatchAction.model_validate({"tool": "batch_capabilities", "actions": [
                {**call.get("arguments", {}), "tool": call["tool"]} for call in calls]})
        except ValidationError as exc:
            # Like a single invalid call, a malformed batch is feedback for the
            # next decision, not a task failure. Nothing in it was started.
            error = ("Tool batch rejected; none of its calls ran. Use distinct web_search, "
                     "read_webpage or run_python calls with valid arguments. " + validation_reasons(exc))
            context.feedback(error)
            self.bus.emit("tool_failed", "explorer", task_id=context.id, tool="batch_capabilities",
                          arguments={"actions": calls}, success=False, error=error)
            self.message(context, "system", "Tool batch rejected; continuing with the error feedback.")
            return []
        requested = [(action.tool, action.model_dump(exclude={"tool"})) for action in validated.actions]
        refusals = [self._refused(context, tool, arguments) for tool, arguments in requested]
        actions = [action for action, refused in zip(requested, refusals) if not refused]
        if not actions:
            return refusals
        if context.tool_count + len(actions) > self.settings.max_tool_calls:
            raise LimitReached("Maximum total tool calls reached; batch was not started.")
        context.tool_count += len(actions)
        # Public reads and restricted Python can run independently. Physical
        # mutations and final answers are excluded by the batch schema.
        async def execute(tool, arguments):
            # Restrictions may change after preflight but before this worker runs.
            refused = self._refused(context, tool, arguments)
            if refused:
                context.tool_count -= 1  # release the slot reserved for this unexecuted call
                return refused
            result = await self.capabilities.execute(tool, arguments, context)
            self._record_capability(context, tool, arguments, result)
            return result
        async with asyncio.TaskGroup() as group:
            running = iter([group.create_task(execute(tool, arguments)) for tool, arguments in actions])
        # Results in the order requested, refusals included.
        return [refused or next(running).result() for refused in refusals]

    def _check_model_budget(self, context: TaskContext) -> None:
        if context.metrics.llm_calls >= self.settings.max_task_llm_calls:
            raise LimitReached(f"Model-call budget ({self.settings.max_task_llm_calls}) used up before the task "
                               "finished; stopping. The task is incomplete.")

    def _rejected(self, context: TaskContext, count: int, correction: str) -> int:
        """Record one rejected suggestion; too many in a row means the plan is stuck."""
        context.feedback(correction)
        self.message(context, "system", correction)
        count += 1
        if count >= self.settings.max_consecutive_rejections:
            raise LimitReached(f"{count} suggestions in a row were rejected without progress; stopping. "
                               f"Last reason: {correction}")
        return count

    def _progress_changed(self, context: TaskContext, previous: tuple | None = None) -> bool:
        signature = context.progress_signature()
        current = previous if previous is not None else context.last_progress_signature
        context.last_progress_signature = signature
        return current != signature

    async def _loop(self, context: TaskContext, plan=None) -> None:
        # Bootstrap through a permitted tool; no agent gets an omniscient snapshot.
        self.call_tool(context, "get_status", {})
        self.call_tool(context, "get_map", {})
        self.bus.emit("agent_active", "coordinator", task_id=context.id)
        # A plan known in advance (a plain lookup of a live fact) needs no planning call.
        goal_plan = plan or await self.coordinator.define_goal(context.goal, context.floor_plan,
                                                               metrics=context.metrics)
        context.conditions = goal_plan.conditions
        if self._outside_grant(context):
            return
        digital_only = all(c.kind == "answer" for c in context.conditions)
        if self.settings.digital_tools_enabled and digital_only:
            # These initial calls are explicitly independent, fully specified
            # by the model, and validated by the same capability executor.
            self.bus.emit("agent_active", "explorer", task_id=context.id)
            if goal_plan.initial_actions:
                self.message(context, "explorer", "Running the model's independent tool calls together.")
                await self.call_capability_batch(context, [action.model_dump() for action in goal_plan.initial_actions])
                if await self._report_execution(context):
                    return
        self.message(context, "coordinator", "Required outcomes defined. Locating targets requires tool observations.")
        disagreements = 0
        rejections = 0                     # consecutive rejected suggestions, reset by real progress
        stagnant_cycles, last_evidence = 0, -1
        for cycle in range(self.settings.max_coordinator_cycles):
            context.cycle_count = cycle + 1
            self.bus.emit("task_updated", task=context.public())
            if verified_complete(context):
                # Every outcome is proven by tool evidence: done. No further review can add
                # information, and extra actions could only undo verified progress.
                self.finish(context, "completed", "All requested outcomes are verified by tool evidence.")
                return
            evidence_count = sum(1 for e in context.action_history if e["success"])
            stagnant_cycles = stagnant_cycles + 1 if evidence_count == last_evidence else 0
            last_evidence = evidence_count
            if stagnant_cycles >= self.settings.max_cycles_without_progress:
                raise LimitReached(f"No new evidence after {stagnant_cycles} planning cycles; stopping instead of "
                                   "reasoning in circles. The task is incomplete.")
            self._check_model_budget(context)
            self.bus.emit("agent_active", "coordinator", task_id=context.id)
            # GoalPlan already interpreted the complete goal. Hand it directly
            # to Explorer instead of paying for a second initial planning call.
            # Later cycles still use the Coordinator for recovery/replanning.
            if cycle == 0 and (self.settings.direct_initial_delegation or digital_only):
                decision = CoordinatorDecision(action="delegate", summary="Execute the required outcomes.",
                                               task=context.goal)
            else:
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
                if verified_complete(context):
                    self.finish(context, "completed", decision.summary)
                    return
                cited = [by_id[id] for id in decision.evidence_ids if id in by_id]
                evidence = cited[-12:] if cited else relevant_evidence(context)
                if await self.review(context, decision.summary, "completion", evidence):
                    self.finish(context, "completed", decision.summary)
                    return
                disagreements += 1
                if disagreements >= self.settings.max_critic_disagreements:
                    raise LimitReached("The reviewer kept rejecting completion although the checks pass; "
                                       "stopping instead of debating. The outcome needs a person's judgment.")
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
                    self._check_model_budget(context)
                    self.bus.emit("agent_active", "explorer", task_id=context.id)
                    action = await self.explorer.decide(context, decision.task)
                    if action.action == "report":
                        if (any(condition.kind == "deliver" for condition in context.conditions)
                            and context.robot_status.get("room") not in context.memory.rooms):
                            rejections = self._rejected(context, rejections, "Report rejected: the current room has "
                                                        "not been observed. Scan it before reporting or replanning.")
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
                                rejections = self._rejected(context, rejections, correction)
                                continue
                        context.explorer_reports.append(action.summary)
                        self.message(context, "explorer", "Returning observations to Coordinator.")
                        break
                    self.message(context, "explorer", f"Next action: {action.tool}.")
                    if action.tool == "batch_capabilities":
                        await self.call_capability_batch(context, [
                            {"tool": call["tool"], "arguments": {k: v for k, v in call.items() if k != "tool"}}
                            for call in action.arguments["actions"]])
                        continue
                    previous = context.action_history[-1] if context.action_history else None
                    if (previous and previous["success"] and action.tool != "talk_to"
                            and previous["tool"] == action.tool and previous["arguments"] == action.arguments):
                        rejections = self._rejected(context, rejections, "Repeated identical command rejected. Use "
                                                    "its existing observation and choose a different action.")
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
                            rejections = self._rejected(context, rejections, correction)
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
                    if action.tool in DIGITAL_TOOLS:
                        result = await self.call_capability(context, action.tool, action.arguments)
                        if result.get("success"):
                            rejections = 0
                        if result.get("success") and not [c for c in check_conditions(context) if not c["satisfied"]]:
                            self.finish(context, "completed", "Answer ready. All requested outcomes completed.")
                            return
                        continue
                    result = self.call_tool(context, action.tool, action.arguments)
                    if result.get("success"):
                        rejections = 0
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
        # Deterministic search sweeps are purposeful even when rooms are empty:
        # don't penalise them or escalate to the Coordinator mid-frontier.
        if on_route or action.source == "reflex_search":
            return False
        context.stall_count += 1
        if context.stall_count < self.settings.max_semantic_stall:
            return False
        context.metrics.stall_events += 1
        context.stall_count = 0
        # The committed destination produced nothing; let the planner pick another.
        context.intention = None
        return True

    async def _recall_for(self, context: TaskContext) -> None:
        # Searchable memory is advisory like the rest of long_term_memory: skills and
        # past episodes suggest where to look, observations still decide.
        if not self.memory_store:
            return
        from app.memory.recall import prompt_memories
        try:
            found = await self.memory_store.recall(context.goal, kinds=["skill", "episode", "fact"], k=3)
        except Exception:
            logging.getLogger("agentic_friend.tasks").exception("Memory recall failed")
            return
        if found:
            context.long_term_memory["relevant_memories"] = {
                "note": "Recalled from past experience; may be stale. Verify by observation.",
                "items": prompt_memories(found)}

    async def _remember(self, context: TaskContext) -> None:
        if not self.memory_store or context.status not in {"completed", "failed"}:
            return
        from app.memory.recall import task_memories
        feeling = self.persona.mood_label() if self.persona else None
        try:
            for memory in task_memories(context, feeling):
                await self.memory_store.add(source="task", **memory)
        except Exception:
            logging.getLogger("agentic_friend.tasks").exception("Failed to store task memories")

    async def _run(self, context: TaskContext, plan=None) -> None:
        try:
            async with asyncio.timeout(self.settings.task_timeout_seconds):
                await self._recall_for(context)
                await self._loop(context, plan)
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
            # Save what this task learned BEFORE releasing the robot, so a queued or
            # autonomous errand that starts next sees today's locations and failures.
            try:
                await self._remember(context)
                if self.persistent:
                    try:
                        self.persistent.update_from_task(context)
                        self.persistent.save()
                        # Beliefs are computed from these tallies, so they just changed.
                        self.bus.emit("beliefs_updated", summary="Beliefs updated from the finished task.")
                    except Exception:
                        logging.getLogger("agentic_friend.tasks").exception("Failed to save persistent memory")
            finally:
                if self.active_id == context.id:
                    self.active_id = None
