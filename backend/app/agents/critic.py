from app.llm.base import LLMClient, structured
from app.tasks.models import TaskContext
from .prompts import CRITIC
from .schemas import CriticReview


class Critic:
    def __init__(self, client: LLMClient) -> None:
        self.client = client

    async def review(self, context: TaskContext, proposal: str, kind: str,
                     evidence: list[dict] | None = None) -> CriticReview:
        return await structured(self.client, CriticReview, CRITIC,
                                {**context.compact(), "proposal": proposal,
                                 "review_type": kind, "cited_evidence": evidence or []})
