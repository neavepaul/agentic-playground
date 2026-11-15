import asyncio
import copy
import logging
from collections import deque
from datetime import datetime, timezone
from uuid import uuid4

logger = logging.getLogger("agentic_friend.events")


def _private(data: dict) -> bool:
    """An event about their own accounts (a Groww answer, its task or its chat turn): its words stay
    out of the log, which is kept in plain text and may be shared when asking for help."""
    return bool(data.get("private") or (data.get("turn") or {}).get("private")
                or (data.get("task") or {}).get("private"))


PRIVATE_TEXT = "(private: shown only in the answer)"


# Every field that can carry an answer's words, wherever it sits (a turn, a tool's arguments or
# observation, a task's answer or the action it is about to dispatch).
_WORDS = frozenset({"answer", "quote", "snippet", "summary", "text", "content", "screen_text", "spoken_text"})


def redact(value):
    """A copy with the words and figures of every _WORDS field replaced; ids, kinds, statuses,
    tools, goals and sources stay, so a page can still list what happened."""
    if isinstance(value, dict):
        return {key: PRIVATE_TEXT if key in _WORDS and isinstance(item, str) else redact(item)
                for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    return copy.deepcopy(value)


def _redacted(data: dict) -> dict:
    """A private event as replay history keeps it. The chat log comes from /api/chat instead."""
    return redact(data)


class EventBus:
    def __init__(self) -> None:
        self.sequence = 0
        self.history: deque[dict] = deque(maxlen=300)
        self.subscribers: set[asyncio.Queue] = set()
        self.listeners: list = []

    def add_listener(self, listener) -> None:
        """Synchronous in-process observers (e.g. the persona). Must be cheap."""
        self.listeners.append(listener)

    def remove_listener(self, listener) -> None:
        if listener in self.listeners:
            self.listeners.remove(listener)

    def emit(self, type: str, agent: str = "system", skip_history: bool = False, **data) -> dict:
        self.sequence += 1
        event = {"id": str(uuid4()), "sequence": self.sequence,
                 "timestamp": datetime.now(timezone.utc).isoformat(),
                 "type": type, "agent": agent, "data": data}
        private = _private(data)
        if not skip_history:
            # Live pages get the whole event (the reply must show); a page that connects later
            # replays history, which never holds their own accounts' figures.
            self.history.append({**event, "data": _redacted(data)} if private else event)
        logger.info("%s [%s] %s", type, agent, "(private: not logged)" if private else
                    data.get("summary", data.get("observation", data.get("error", ""))))
        for queue in tuple(self.subscribers):
            if queue.full():
                # Force a snapshot resync rather than silently lose state changes.
                while not queue.empty():
                    queue.get_nowait()
                queue.put_nowait({"type": "resync_required"})
            else:
                queue.put_nowait(event)
        # Listeners run after delivery, so events they emit always follow their cause.
        for listener in tuple(self.listeners):
            try:
                listener(event)
            except Exception:
                logger.exception("Event listener failed for %s", type)
        return event

    def subscribe(self) -> asyncio.Queue:
        queue = asyncio.Queue(maxsize=128)
        self.subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self.subscribers.discard(queue)
