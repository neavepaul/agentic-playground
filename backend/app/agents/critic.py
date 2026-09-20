from app.llm.base import LLMClient, structured
from app.tasks.models import TaskContext
from .prompts import CRITIC
from .schemas import CriticReview


class Critic:
    def __init__(self, client: LLMClient) -> None:
        self.client = client

    async def review(self, context: TaskContext, proposal: str, kind: str,
                     evidence: list[dict] | None = None,
                     proposed_action: dict | None = None) -> CriticReview:
        state = context.compact()
        # Prior verdicts are opinions, not evidence for the next independent review.
        # In particular, a previously missing prerequisite may now be satisfied.
        state.pop("critic_feedback", None)
        return await structured(self.client, CriticReview, CRITIC,
                                {**state, "proposal": proposal,
                                 "proposed_action": proposed_action,
                                 "review_type": kind, "cited_evidence": evidence or []})
