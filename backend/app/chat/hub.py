"""Many conversations, one Kip: each person's chats are separate ChatSessions over one task manager.

A conversation sees only the errands it asked for (OwnTasks), so its replies, status line and
"what did you do" answers never carry another person's lookups. The single errand slot is still
shared: a conversation whose errand must wait queues it, as one conversation always has.
"""
import logging
from collections import OrderedDict

_log = logging.getLogger("agentic_friend.chat")


class OwnTasks:
    """The task manager as one conversation sees it. active_id stays the shared slot (what decides
    whether a new errand can start); tasks and active() are only this conversation's own."""

    def __init__(self, manager, session_id: str, speaker: str | None, portfolio: bool) -> None:
        self._manager, self.session_id = manager, session_id
        self.speaker, self.portfolio = speaker, portfolio

    def __getattr__(self, name):
        return getattr(self._manager, name)

    @property
    def manager(self):
        return self._manager

    @property
    def tasks(self) -> dict:
        return {key: task for key, task in self._manager.tasks.items() if task.session_id == self.session_id}

    @property
    def active_id(self) -> str | None:
        return self._manager.active_id

    def owns(self, task_id: str | None) -> bool:
        task = self._manager.tasks.get(task_id) if task_id else None
        return task is not None and task.session_id == self.session_id

    def active(self) -> dict | None:
        return self._manager.active() if self.owns(self._manager.active_id) else None

    def start(self, goal: str, grant=None, plan=None):
        return self._manager.start(goal, grant, plan=plan, session_id=self.session_id,
                                   speaker=self.speaker, portfolio=self.portfolio)


class ConversationHub:
    """Open ChatSessions by conversation id, made on first use. Idle ones beyond `limit` are closed
    (their turns stay in kip.db and reload when the conversation is opened again)."""

    def __init__(self, make, bus, limit: int = 6) -> None:
        self._make, self._bus, self._limit = make, bus, limit
        self._open: OrderedDict[str, object] = OrderedDict()

    def get(self, conversation_id: str, profile):
        session = self._open.get(conversation_id)
        if session is None:
            session = self._make(conversation_id, profile)
            self._open[conversation_id] = session
        self._open.move_to_end(conversation_id)
        return session

    def peek(self, conversation_id: str):
        return self._open.get(conversation_id)

    async def trim(self) -> None:
        idle = [key for key, session in self._open.items() if not session.busy and not session.pending_goals]
        while len(self._open) > self._limit and idle:
            await self.drop(idle.pop(0))

    async def drop(self, conversation_id: str) -> None:
        session = self._open.pop(conversation_id, None)
        if session is not None:
            self._bus.remove_listener(session._narrate)
            await session.close()

    async def close(self) -> None:
        for key in list(self._open):
            await self.drop(key)
