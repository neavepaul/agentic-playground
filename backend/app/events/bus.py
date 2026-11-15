import asyncio
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


class EventBus:
    def __init__(self) -> None:
        self.sequence = 0
        self.history: deque[dict] = deque(maxlen=300)
        self.subscribers: set[asyncio.Queue] = set()
        self.listeners: list = []

    def add_listener(self, listener) -> None:
        """Synchronous in-process observers (e.g. the persona). Must be cheap."""
        self.listeners.append(listener)

    def emit(self, type: str, agent: str = "system", skip_history: bool = False, **data) -> dict:
        self.sequence += 1
        event = {"id": str(uuid4()), "sequence": self.sequence,
                 "timestamp": datetime.now(timezone.utc).isoformat(),
                 "type": type, "agent": agent, "data": data}
        if not skip_history:
            self.history.append(event)
        logger.info("%s [%s] %s", type, agent, "(private: not logged)" if _private(data) else
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
