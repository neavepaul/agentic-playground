from copy import deepcopy
from uuid import uuid4
from typing import Literal

from pydantic import BaseModel, Field

from app.agents.schemas import GoalCondition
from .goals import check_conditions
from .permissions import Grant, Restrictions


class TaskMetrics(BaseModel):
    """Per-task execution counters used to compare inference cost across runs."""
    llm_calls: int = 0
    llm_call_limit: int = 0  # zero only for standalone/unbudgeted callers
    coordinator_llm_calls: int = 0
    explorer_llm_calls: int = 0
    critic_llm_calls: int = 0
    llm_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_actions: int = 0
    useful_observations: int = 0
    coordinator_handoffs: int = 0


class TaskContext(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    goal: str
    status: Literal["running", "completed", "failed", "cancelled"] = "running"
    summary: str = "Starting task."
    cycle_count: int = 0
    tool_count: int = 0
    critic_count: int = 0
    current_plan: str = ""
    # What this task was authorized to do. The manager re-checks it, with the live
    # restrictions, right before every tool call; this copy lets planning avoid
    # choices that would only be refused.
    grant: Grant = Field(default_factory=Grant)
    restricted: dict = Field(default_factory=dict)  # snapshot: tools the person ruled out
    # Tools the person's own words asked to have run: required outcomes, whatever the plan said.
    requested_tools: list[str] = Field(default_factory=list)
    # Requested outcomes that were not done, and why ("run Python (you asked me not to)").
    blocked: list[str] = Field(default_factory=list)
    # Honest footnotes on the final answer (e.g. it describes a run that never happened).
    answer_notes: list[str] = Field(default_factory=list)
    # Audit evidence is task-local and bounded by the execution tool budget.
    action_history: list[dict] = Field(default_factory=list)
    critic_feedback: list[dict] = Field(default_factory=list)
    explorer_reports: list[str] = Field(default_factory=list)
    conditions: list[GoalCondition] = Field(default_factory=list)
    answer: dict | None = None
    # Answered from their own accounts (Groww holdings): shown to them and kept on the device, but
    # never put in a later model prompt, a memory or the status line, so it never reaches Groq.
    private: bool = False
    # The conversation that asked for this errand (None: the task API). Its events, narration and
    # answer reach only that conversation's person.
    session_id: str | None = None
    speaker: str | None = None
    # Whether the person who asked may see the owner's Groww portfolio.
    portfolio: bool = True
    # Written before dispatch and cleared only after result evidence is journaled.
    # A restart with this set has an uncertain outcome, never an automatic retry.
    pending_action: dict | None = None
    action_feedback: list[str] = Field(default_factory=list)
    consecutive_report_rejections: int = 0
    long_term_memory: dict = Field(default_factory=dict)
    self_model: dict = Field(default_factory=dict)
    metrics: TaskMetrics = Field(default_factory=TaskMetrics)

    def feedback(self, message: str) -> None:
        self.action_feedback.append(message)
        self.action_feedback[:] = self.action_feedback[-6:]

    def restrictions(self) -> Restrictions:
        snapshot = Restrictions()
        snapshot.tools = set(self.restricted.get("tools", []))
        return snapshot

    def permits(self, tool: str, arguments: dict) -> bool:
        return not self.grant.deny(tool, arguments, self.restrictions())

    def record(self, tool: str, arguments: dict, result: dict) -> None:
        entry = deepcopy({"tool": tool, "arguments": arguments, **result})
        self.action_history.append(entry)
        if not result["success"]:
            return
        self.consecutive_report_rejections = 0
        if tool == "finish_answer":
            self.answer = deepcopy(entry["observation"])

    def compact(self) -> dict:
        return deepcopy({"goal": self.goal, "current_plan": self.current_plan,
                "long_term_memory": self.long_term_memory,
                "self_model": self.self_model,
                "action_feedback": self.action_feedback[-6:],
                "required_outcomes": check_conditions(self),
                "permissions": {"cleared_for": self.grant.effects,
                                "asked_not_to": self.restrictions().describe()},
                "recent_actions": [
                    {key: action[key] for key in ("tool", "arguments", "success", "error", "evidence_id")
                     if key in action}
                    for action in self.action_history[-8:]],
                "critic_feedback": [f for f in self.critic_feedback
                                    if f.get("observed_action_count") == len(self.action_history)
                                    and f.get("kind") != "delegation_report"][-2:],
                "explorer_reports_unverified": self.explorer_reports[-2:],
                "cycle_count": self.cycle_count, "tool_count": self.tool_count})

    def public(self) -> dict:
        """The task as pages, events and the API see it. A private task keeps its sources but not its
        answer's words: those reach the person only as Kip's reply."""
        shown = self.model_dump(mode="json", exclude={"action_history", "critic_feedback", "explorer_reports"})
        if self.private:
            from app.events.bus import redact
            shown = {**shown, **redact({key: shown[key] for key in ("answer", "pending_action")})}
        return shown
