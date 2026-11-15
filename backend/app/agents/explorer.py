import logging

from pydantic import Field, create_model

from app.llm.base import LLMClient, structured
from app.tasks.models import TaskContext
from .schemas import ExplorerDecision

_log = logging.getLogger("agentic_friend.decisions")


class Explorer:
    def __init__(self, client: LLMClient, persona=None, hosted=None, hosted_calls: int = 0) -> None:
        self.client = client
        self.persona = persona
        # Web-research tasks may use a hosted model, under one call budget per task.
        self.hosted, self.hosted_calls = hosted, hosted_calls
        self._research: dict[str, object] = {}

    def _research_client(self, task_id: str):
        if self.hosted is None or not self.hosted_calls:
            return self.client
        if task_id not in self._research:
            from app.llm.groq import HostedFirst
            if len(self._research) >= 32:  # budgets of long-finished tasks
                self._research.pop(next(iter(self._research)))
            self._research[task_id] = HostedFirst(self.hosted, self.client, self.hosted_calls, "web research")
        return self._research[task_id]

    async def decide(self, context: TaskContext, task: str) -> ExplorerDecision:
        from app.capabilities.schemas import CapabilityChoice, SearchAction, ReadAction, PythonAction, AnswerAction, BatchAction
        from app.capabilities.tools import required_tool_progress
        from app.capabilities.context import evidence_view
        from .prompts import CAPABILITY_EXPLORER

        progress = required_tool_progress(context)
        # Only tools this task may use are offered; dispatch refuses the rest anyway.
        tools = [action for tool, action in (("web_search", SearchAction), ("read_webpage", ReadAction),
                                             ("run_python", PythonAction)) if context.permits(tool, {})]
        available = BatchAction
        for action in tools:
            available |= action
        if progress["can_finish"]:
            available |= AnswerAction
        schema = create_model("CapabilityChoice", __base__=CapabilityChoice,
                              action=(available, Field(discriminator="tool")))
        observations, aliases = evidence_view(context)
        originals = {original: alias for alias, original in aliases.items()}
        progress["successful_evidence_ids"] = {tool: [originals[key] for key in ids]
                                               for tool, ids in progress["successful_evidence_ids"].items()}
        payload = {"goal": context.goal,
                   "required_tool_progress": progress,
                   "required_outcomes": [c.model_dump(exclude_defaults=True) for c in context.conditions],
                   "tool_observations": observations, "feedback": context.action_feedback[-3:]}
        if task != context.goal:
            payload["delegated_task"] = task
        prompt = CAPABILITY_EXPLORER
        if self.persona and progress["can_finish"]:
            from .prompts import PERSONA_ANSWER, persona_payload, persona_system
            view = self.persona.voice_view()
            prompt += PERSONA_ANSWER + persona_system(view)
            payload["persona"] = persona_payload(view)
        choice = await structured(self._research_client(context.id), schema, prompt, payload,
                                  metrics=context.metrics, role="explorer", compact_schema=True,
                                  repair_hint="Choose an available tool. Run missing tools before finish_answer.")
        selected = choice.action
        if selected.tool == "finish_answer":
            selected.evidence_ids = [aliases.get(key, key) for key in selected.evidence_ids]
        _log.info("Explorer choice: task=%s tool=%s summary=%s", context.id, selected.tool, choice.summary)
        return ExplorerDecision(action="tool", tool=selected.tool,
                                arguments=selected.model_dump(exclude={"tool"}), summary=choice.summary)
