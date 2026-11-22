import asyncio
import json
import logging
import re

from pydantic import ValidationError

from app import clock
from app.agents.coordinator import Coordinator
from app.agents.explorer import Explorer
from app.agents.prompts import CRITIC
from app.agents.schemas import CoordinatorDecision, CriticReview, GoalCondition
from app.config import Settings
from app.capabilities.tools import CapabilityTools
from app.events.bus import EventBus
from app.llm.base import LLMClient, ModelError, structured
from app.memory.store import PersistentMemory
from .models import TaskContext
from .goals import check_conditions
from .permissions import (DIGITAL_LOOKUP, CALCULATION, TOOL_PLAIN, Grant, Restrictions,
                          explicitly_requested_tools, plain)


_CLAIMS_A_RUN = re.compile(r"\b(ran|run|executed|computed|calculated)\b[^.]{0,40}\b(python|code|program|script|"
                           r"locally|calculation)\b|```python")


def answer_notes(context: TaskContext) -> list[str]:
    """Footnotes the final answer must carry to stay true to the evidence."""
    notes = []
    if (context.answer or {}).get('verification') == 'evidence_linked':
        notes.append('This answer cites tool evidence; its interpretation has not been independently verified.')
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


class LimitReached(RuntimeError):
    pass


TERMINAL = {"completed", "failed", "cancelled"}
JOURNAL_TEXT = 2000  # characters of any one tool-output string the task journal keeps


def _clipped(value):
    """A copy with every string longer than JOURNAL_TEXT cut to it (for the task journal)."""
    if isinstance(value, str):
        return value if len(value) <= JOURNAL_TEXT else value[:JOURNAL_TEXT] + " …"
    if isinstance(value, list):
        return [_clipped(item) for item in value]
    if isinstance(value, dict):
        return {key: _clipped(item) for key, item in value.items()}
    return value


def verified_complete(context: TaskContext) -> bool:
    """An answer was accepted and the required tool executions are present."""
    outcomes = check_conditions(context)
    return bool(outcomes) and all(c["satisfied"] for c in outcomes)


def unmet(context: TaskContext) -> list[dict]:
    return [c for c in check_conditions(context) if not c["satisfied"]]


def relevant_evidence(context: TaskContext, limit: int = 12) -> list[dict]:
    """Recent, task-relevant observations for a review; never the whole history."""
    return [e for e in context.action_history if e["success"]][-limit:]


class TaskManager:
    def __init__(self, client: LLMClient, bus: EventBus, settings: Settings,
                 persistent: PersistentMemory | None = None, persona=None, memory_store=None,
                 hosted=None) -> None:
        self.bus, self.settings = bus, settings
        self.client = client
        self.persistent = persistent
        self.persona = persona
        self.memory_store = memory_store
        self.coordinator = Coordinator(client, hosted=hosted, hosted_calls=settings.groq_plan_calls)
        self.capabilities = CapabilityTools(bus, enabled=settings.digital_tools_enabled, hosted=hosted,
                                            cricket_key=settings.cricketdata_api_key,
                                            football_key=settings.footballdata_api_key,
                                            groww=(settings.groww_api_key, settings.groww_api_secret))
        self.explorer = Explorer(client, persona=persona, hosted=hosted, hosted_calls=settings.groq_research_calls)
        self.tasks: dict[str, TaskContext] = {}
        self.active_id: str | None = None
        self.runner: asyncio.Task | None = None
        # What the person asked Kip not to do. Shared and live: it binds every task,
        # including ones already queued or running, at the moment a tool would run.
        self.restrictions = Restrictions()
        if self.memory_store:
            saved = json.loads(self.memory_store.get_state("task-restrictions") or "{}")
            self.restrictions.tools = set(saved.get("tools", []))
            records = json.loads(self.memory_store.get_state("task-journal") or "[]")
            for record in records[-50:]:
                try:
                    context = TaskContext.model_validate(record)
                except ValidationError:
                    continue  # an errand from the retired house simulation: nothing to show for it now
                if context.status == "running":
                    context.status = "failed"
                    context.summary = ("Interrupted by restart. An in-flight action may have an unknown outcome; "
                                       "nothing was replayed. Ask again after checking the current state.")
                self.tasks[context.id] = context
            if records:
                self.checkpoint()

    def save_restrictions(self) -> None:
        if self.memory_store:
            self.memory_store.set_state({"task-restrictions": json.dumps({"tools": sorted(self.restrictions.tools)})})

    def checkpoint(self) -> None:
        """Bounded durable task audit. Recovery never replays an interrupted action.

        Saved several times per tool call, so long tool output (a page's text, a program's output)
        is kept to its first JOURNAL_TEXT characters: tens of kilobytes, not tens of megabytes, to
        rewrite on a Pi's microSD card. A recovered task is finished; its record is for reading."""
        if self.memory_store:
            records = []
            for task in self.tasks.values():
                record = task.model_dump(mode="json")
                record["action_history"] = _clipped(record["action_history"])
                records.append(record)
            self.memory_store.set_state({"task-journal": json.dumps(records)})

    def _before_dispatch(self, context: TaskContext, tool: str, arguments: dict) -> None:
        self._ensure_running(context)
        context.pending_action = {"tool": tool, "arguments": arguments}
        self.checkpoint()

    @staticmethod
    def _ensure_running(context: TaskContext) -> None:
        if context.status != "running":
            raise LimitReached("A terminal task cannot execute more tools.")

    def start(self, goal: str, grant: Grant | None = None, plan=None, session_id: str | None = None,
              speaker: str | None = None, portfolio: bool = True, owner: bool = False,
              queue_id: str | None = None) -> TaskContext:
        """Start a task. Without a grant it is a direct instruction (API): the goal
        text is the person's own words and authorizes every kind of action. session_id is the
        conversation that asked; speaker and portfolio say who that is and what they may see."""
        if self.active_id:
            raise TaskBusy("One task is already running. Cancel it or wait for completion.")
        if grant is None:
            # API goals are the person's own words. Internal/model-written goals carry
            # an explicit grant and must never change the person's restrictions.
            if self.restrictions.update(goal):
                self.save_restrictions()
        grant = grant or Grant.full("api", goal)
        context = TaskContext(goal=goal, grant=grant, requested_tools=explicitly_requested_tools(grant.authorized_text),
                              session_id=session_id, speaker=speaker, portfolio=portfolio, owner=owner, queue_id=queue_id)
        context.metrics.llm_call_limit = self.settings.max_task_llm_calls
        self._sync_permissions(context)
        if self.persistent:
            context.long_term_memory = self.persistent.prompt()
            context.self_model = self.persistent.self_model_prompt()
        # Keep recent tasks in memory only. Active task is never evicted.
        if len(self.tasks) >= 50:
            del self.tasks[next(iter(self.tasks))]
        self.tasks[context.id] = context
        self.checkpoint()
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
        self.checkpoint()
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
                self.finish(context, "cancelled", "Task cancelled.")
            if self.active_id == task_id:
                self.active_id = None
        return context

    async def close(self) -> None:
        if self.active_id:
            await self.cancel(self.active_id)

    # ── permission at the point of execution ──────────────────────────────────
    def _sync_permissions(self, context: TaskContext) -> None:
        context.restricted = {"tools": sorted(self.restrictions.tools)}

    def _refused(self, context: TaskContext, tool: str, arguments: dict) -> dict | None:
        """Checked immediately before every tool call, against the live restrictions."""
        self._sync_permissions(context)
        reason = context.grant.deny(tool, arguments, self.restrictions)
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
        uncovered = context.grant.uncovered(context.conditions)
        ruled_out: set[str] = set()
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

    async def _feed_answer(self, context: TaskContext, goal_plan) -> bool:
        """A question a data feed covers (sport, weather, markets, news, their portfolio), answered
        from it: one request, exact facts, no model call. Anything a feed does not cover goes on to
        the web lookup."""
        question = getattr(goal_plan, "feed", None)
        if not question or not self.settings.digital_tools_enabled:
            return False
        arguments = {"query": goal_plan.initial_actions[0].arguments.get("query") or context.goal}
        self._ensure_running(context)
        if self._refused(context, "web_search", arguments) or context.tool_count >= self.settings.max_tool_calls:
            return False
        source = self.capabilities.feed_source(question, portfolio=context.portfolio)
        if source:  # "Let me check Groww for that."
            self.bus.emit("feed_checking", "explorer", task_id=context.id, source=source)
        self._before_dispatch(context, "web_search", arguments)
        result, answer = await self.capabilities.feed_answer(arguments["query"], question, context)
        context.pending_action = None
        if result is None:
            return False
        # context.private is already set (feed_answer, before any event) for their own accounts.
        context.tool_count += 1
        self._record_capability(context, "web_search", arguments, result)
        finished = await self.call_capability(context, "finish_answer", {"answer": answer,
                                                                        "evidence_ids": [result["evidence_id"]]})
        if finished.get("success") and not [c for c in check_conditions(context) if not c["satisfied"]]:
            self.finish(context, "completed", f"Answered from the {result['observation']['provider']} results feed.")
            return True
        if context.private:
            # Their own accounts never go on to a model: its prompt would carry the figures.
            self.finish(context, "failed", "Could not finish the answer from their own accounts.")
            return True
        return False

    async def _web_answer(self, context: TaskContext, goal_plan) -> bool:
        """A plain lookup answered in one Groq call: its browser search finds the pages and
        writes the answer in Kip's voice. Anything it cannot settle goes the usual way."""
        actions = goal_plan.initial_actions
        if (len(context.conditions) != 1 or context.conditions[0].required_tools != ["web_search"]
                or len(actions) != 1 or actions[0].tool != "web_search"
                or not self.capabilities.can_answer_from_web()):
            return False
        if self.restrictions.forbids_tool("read_webpage") or "read_webpage" in context.restricted.get("tools", []):
            # Groq's browser search opens the pages it finds: not after "don't read webpages".
            # The usual lookup searches without reading.
            return False
        arguments = {"query": actions[0].arguments.get("query") or context.goal}
        self._ensure_running(context)
        if self._refused(context, "web_search", arguments) or context.tool_count >= self.settings.max_tool_calls:
            return False
        from app.agents.prompts import WEB_ANSWER, persona_system
        now = clock.now()
        today = now.date()
        stamp = f"{now.strftime('%A %d %B %Y, %H:%M')} {now.tzname()} (UTC{now.strftime('%z')})"
        system = WEB_ANSWER.format(now=stamp, year=today.year)
        if self.persona:
            system += persona_system(self.persona.voice_view())
        # The date travels with the question too: these models otherwise search last year.
        asked = f"{context.goal}\n(Asked on {today.strftime('%d %B %Y')} at {now.strftime('%H:%M')} {now.tzname()}.)"
        context.tool_count += 1
        self._before_dispatch(context, "web_search", arguments)
        # A question about today must be answered from a source about today.
        about_today = bool(re.search(r"\btoday\b|\bright now\b|\blive\b|\btonight\b", context.goal.casefold()))
        # ...and a question about the latest one from a dated source no newer one beats.
        about_latest = bool(re.search(r"\blatest\b|\bmost recent\b|\blast\b", context.goal.casefold()))
        result, answer = await self.capabilities.web_answer(
            arguments["query"], [{"role": "system", "content": system}, {"role": "user", "content": asked}], context,
            about_today=about_today, about_latest=about_latest)
        reported = self.capabilities.hosted.last_metrics
        context.metrics.llm_calls += 1
        context.metrics.explorer_llm_calls += 1
        context.metrics.llm_seconds += reported.get("groq_seconds", 0)
        context.metrics.prompt_tokens += reported.get("prompt_tokens", 0)
        context.metrics.completion_tokens += reported.get("completion_tokens", 0)
        self._record_capability(context, "web_search", arguments, result)
        context.pending_action = None
        if not result["success"]:
            return False
        finished = await self.call_capability(context, "finish_answer", {"answer": answer,
                                                                        "evidence_ids": [result["evidence_id"]]})
        if finished.get("success") and not [c for c in check_conditions(context) if not c["satisfied"]]:
            self.finish(context, "completed", "Answered from Groq's web search.")
            return True
        return False

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
        self._ensure_running(context)
        refused = self._refused(context, tool, arguments)
        if refused:
            return refused
        if context.tool_count >= self.settings.max_tool_calls:
            raise LimitReached("Maximum total tool calls reached.")
        context.tool_count += 1
        self._before_dispatch(context, tool, arguments)
        result = await self.capabilities.execute(tool, arguments, context)
        self._record_capability(context, tool, arguments, result)
        context.pending_action = None
        self.checkpoint()
        return result

    def _record_capability(self, context: TaskContext, tool: str, arguments: dict, result: dict) -> None:
        context.record(tool, arguments, result)
        self.checkpoint()
        if result["success"]:
            context.metrics.useful_observations += 1
        if not result["success"]:
            context.feedback(result["error"])
            self.message(context, "system", "Tool attempt rejected; continuing with the error feedback.")
        self.bus.emit("task_updated", task=context.public())

    async def call_capability_batch(self, context: TaskContext, calls: list[dict]) -> list[dict]:
        self._ensure_running(context)
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
        self._before_dispatch(context, "batch_capabilities", {"actions": actions})
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
        context.pending_action = None
        self.checkpoint()
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

    async def _loop(self, context: TaskContext, plan=None) -> None:
        self.bus.emit("agent_active", "coordinator", task_id=context.id)
        # A plan known in advance (a plain lookup of a live fact) needs no planning call.
        # A request that only asks for Python needs just the code: one narrow call.
        if plan is None and context.requested_tools == ["run_python"]:
            plan = await self.coordinator.python_plan(context.goal, metrics=context.metrics)
        goal_plan = plan or await self.coordinator.define_goal(context.goal, metrics=context.metrics)
        context.conditions = goal_plan.conditions
        if self._outside_grant(context):
            return
        if not self.settings.digital_tools_enabled:
            self.finish(context, "failed", "My web and Python tools are switched off.")
            return
        # These initial calls are explicitly independent, fully specified
        # by the model, and validated by the same capability executor.
        self.bus.emit("agent_active", "explorer", task_id=context.id)
        if await self._feed_answer(context, goal_plan) or await self._web_answer(context, goal_plan):
            return
        if goal_plan.initial_actions:
            self.message(context, "explorer", "Running the model's independent tool calls together.")
            await self.call_capability_batch(context, [action.model_dump() for action in goal_plan.initial_actions])
            if await self._report_execution(context):
                return
        disagreements = 0
        rejections = 0                     # consecutive rejected suggestions, reset by real progress
        stagnant_cycles, last_evidence = 0, -1
        for cycle in range(self.settings.max_coordinator_cycles):
            context.cycle_count = cycle + 1
            self.bus.emit("task_updated", task=context.public())
            if verified_complete(context):
                # Every outcome is proven by tool evidence: done.
                self.finish(context, "completed", "Answer prepared from tool evidence.")
                return
            evidence_count = sum(1 for e in context.action_history if e["success"])
            stagnant_cycles = stagnant_cycles + 1 if evidence_count == last_evidence else 0
            last_evidence = evidence_count
            if stagnant_cycles >= self.settings.max_cycles_without_progress:
                raise LimitReached(f"No new evidence after {stagnant_cycles} planning cycles; stopping instead of "
                                   "reasoning in circles. The task is incomplete.")
            self._check_model_budget(context)
            self.bus.emit("agent_active", "coordinator", task_id=context.id)
            # GoalPlan already interpreted the complete goal. Hand it directly to Explorer
            # instead of paying for a second planning call; later cycles replan.
            if cycle == 0:
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
                missing = unmet(context)
                if missing:
                    context.critic_feedback.append({"source": "system", "approved": False,
                        "observed_action_count": len(context.action_history),
                        "summary": "Completion rejected: required outcomes lack tool evidence.",
                        "unmet_conditions": missing,
                        "suggestion": "Complete the missing outcomes using tools."})
                    self.message(context, "system", "Completion rejected: required outcomes have not been observed.")
                    continue
                by_id = {entry["evidence_id"]: entry for entry in context.action_history if entry["success"]}
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
                for step in range(self.settings.max_explorer_actions):
                    # A healthy delegation ends here, not at the action ceiling.
                    if step and not unmet(context):
                        self.message(context, "system", "Delegated objective satisfied; returning to Coordinator.")
                        break
                    self._check_model_budget(context)
                    self.bus.emit("agent_active", "explorer", task_id=context.id)
                    action = await self.explorer.decide(context, decision.task)
                    if action.action == "report":
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
                    if (previous and previous["success"]
                            and previous["tool"] == action.tool and previous["arguments"] == action.arguments):
                        rejections = self._rejected(context, rejections, "Repeated identical command rejected. Use "
                                                    "its existing observation and choose a different action.")
                        continue
                    result = await self.call_capability(context, action.tool, action.arguments)
                    if result.get("success"):
                        rejections = 0
                        if not unmet(context):
                            self.finish(context, "completed", "Answer prepared from tool evidence.")
                            return
                    # Let cancellation, sockets and other API requests run between tools.
                    await asyncio.sleep(0)
                else:
                    self.message(context, "system",
                                 "Delegation safety ceiling reached; returning to Coordinator.")
                context.metrics.coordinator_handoffs += 1
            self.bus.emit("task_updated", task=context.public())
        raise LimitReached("Maximum Coordinator cycles reached.")

    async def _recall_for(self, context: TaskContext) -> None:
        # Searchable memory is advisory like the rest of long_term_memory: past
        # episodes and answers suggest where to look, observations still decide.
        if not self.memory_store:
            return
        from app.memory.recall import prompt_memories
        try:
            # Someone's errand recalls only what they told Kip (and, for the owner, what Kip knew
            # before there were profiles), never another person's facts or lookups.
            who = {"viewer": context.speaker, "legacy": context.owner} if context.speaker else {}
            found = await self.memory_store.recall(context.goal, kinds=["skill", "episode", "fact"], k=3, **who)
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
                if context.speaker:
                    # Filed under whoever asked: other people never see what they looked up.
                    memory = {**memory, "meta": {**(memory.get("meta") or {}), "speaker": context.speaker}}
                await self.memory_store.add(source="task", **memory)
        except Exception:
            logging.getLogger("agentic_friend.tasks").exception("Failed to store task memories")

    async def _run(self, context: TaskContext, plan=None) -> None:
        try:
            async with asyncio.timeout(self.settings.task_timeout_seconds):
                await self._recall_for(context)
                await self._loop(context, plan)
        except asyncio.CancelledError:
            self.finish(context, "cancelled", "Task cancelled.")
        except TimeoutError:
            self.finish(context, "failed", "Task time limit reached.")
        except (ModelError, LimitReached) as exc:
            self.finish(context, "failed", str(exc))
        except Exception:
            logging.getLogger("agentic_friend.tasks").exception("Unexpected task failure")
            self.finish(context, "failed", "Unexpected task error. Check the backend log.")
        finally:
            # Save what this task learned BEFORE releasing the task slot, so a queued
            # errand that starts next sees it.
            try:
                await self._remember(context)
                if self.persistent:
                    try:
                        self.persistent.update_from_task(context)
                        self.persistent.save()
                    except Exception:
                        logging.getLogger("agentic_friend.tasks").exception("Failed to save persistent memory")
            finally:
                if self.active_id == context.id:
                    self.active_id = None
