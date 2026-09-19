import asyncio
import logging

from app.agents.coordinator import Coordinator
from app.agents.critic import Critic
from app.agents.explorer import Explorer
from app.config import Settings
from app.events.bus import EventBus
from app.llm.base import LLMClient, ModelError
from app.world.tools import WorldTools
from .models import TaskContext


class TaskBusy(ValueError):
    pass


class LimitReached(RuntimeError):
    pass


class TaskManager:
    def __init__(self, client: LLMClient, tools: WorldTools, bus: EventBus, settings: Settings) -> None:
        self.tools, self.bus, self.settings = tools, bus, settings
        self.coordinator = Coordinator(client)
        self.explorer = Explorer(client, tools.schemas())
        self.critic = Critic(client)
        self.tasks: dict[str, TaskContext] = {}
        self.active_id: str | None = None
        self.runner: asyncio.Task | None = None

    def start(self, goal: str) -> TaskContext:
        if self.active_id:
            raise TaskBusy("One task is already running. Cancel it or wait for completion.")
        context = TaskContext(goal=goal)
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
        self.bus.emit(f"task_{status}", task=context.public(), summary=summary)

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
                     evidence: list[dict] | None = None) -> bool:
        if context.critic_count >= self.settings.max_critic_reviews:
            raise LimitReached("Maximum Critic reviews reached; stopping repeated debate.")
        context.critic_count += 1
        self.bus.emit("agent_active", "critic", task_id=context.id)
        review = await self.critic.review(context, proposal, kind, evidence)
        context.critic_feedback.append({"proposal": proposal, **review.model_dump()})
        self.bus.emit("critic_review", "critic", task_id=context.id, **review.model_dump())
        return review.approved

    async def _loop(self, context: TaskContext) -> None:
        # Bootstrap through a permitted tool; no agent gets an omniscient snapshot.
        self.call_tool(context, "get_status", {})
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
                by_id = {entry["evidence_id"]: entry for entry in context.action_history
                         if entry["success"] and entry["tool"] != "get_status"}
                if not all(id in by_id for id in decision.evidence_ids):
                    context.critic_feedback.append({"approved": False,
                        "summary": "Completion referenced unknown or failed tool evidence.",
                        "suggestion": "Gather evidence and cite successful tool observation IDs."})
                    self.message(context, "system", "Completion rejected: evidence IDs are not valid.")
                    continue
                evidence = [by_id[id] for id in decision.evidence_ids]
                if await self.review(context, decision.summary, "completion", evidence):
                    self.finish(context, "completed", decision.summary)
                    return
            elif decision.action == "delegate":
                context.current_plan = decision.task
                starting_tool_count = context.tool_count
                for _ in range(self.settings.max_explorer_actions):
                    self.bus.emit("agent_active", "explorer", task_id=context.id)
                    action = await self.explorer.decide(context, decision.task)
                    if action.action == "report":
                        context.explorer_reports.append(action.summary)
                        self.message(context, "explorer", "Returning observations to Coordinator.")
                        if context.tool_count == starting_tool_count:
                            approved = await self.review(context, action.summary, "delegation_report")
                            if not approved:
                                continue
                        break
                    self.message(context, "explorer", f"Next action: {action.tool}.")
                    self.call_tool(context, action.tool, action.arguments)
                    # Let cancellation, sockets and other API requests run between tools.
                    await asyncio.sleep(0)
                else:
                    self.message(context, "system", "Delegation action limit reached; returning to Coordinator.")
            self.bus.emit("task_updated", task=context.public())
        raise LimitReached("Maximum Coordinator cycles reached.")

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
