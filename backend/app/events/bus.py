import asyncio
import logging
from collections import deque
from datetime import datetime, timezone
from uuid import uuid4

logger = logging.getLogger("agentic_friend.events")


class EventBus:
    def __init__(self) -> None:
        self.sequence = 0
        self.history: deque[dict] = deque(maxlen=300)
        self.subscribers: set[asyncio.Queue] = set()

    def emit(self, type: str, agent: str = "system", **data) -> dict:
        self.sequence += 1
        event = {"id": str(uuid4()), "sequence": self.sequence,
                 "timestamp": datetime.now(timezone.utc).isoformat(),
                 "type": type, "agent": agent, "data": data}
        self.history.append(event)
        logger.info("%s [%s] %s", type, agent,
                    data.get("summary", data.get("observation", data.get("error", ""))))
        for queue in tuple(self.subscribers):
            if queue.full():
                # Force a snapshot resync rather than silently lose state changes.
                while not queue.empty():
                    queue.get_nowait()
                queue.put_nowait({"type": "resync_required"})
            else:
                queue.put_nowait(event)
        return event

    def subscribe(self) -> asyncio.Queue:
        queue = asyncio.Queue(maxsize=128)
        self.subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self.subscribers.discard(queue)
