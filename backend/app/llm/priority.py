"""Conversation first on a shared model.

When chat and errands use the same local model, an errand's next model call waits while a
person is waiting for a reply. A call already in flight is never interrupted, and the
errand continues as soon as the conversation is quiet. No calls are added or removed.
"""
import asyncio


class ConversationFirst:
    def __init__(self, client) -> None:
        self.client = client
        self._waiting = 0
        self._quiet = asyncio.Event()
        self._quiet.set()

    def conversation_waiting(self, delta: int) -> None:
        self._waiting = max(0, self._waiting + delta)
        (self._quiet.set if self._waiting == 0 else self._quiet.clear)()

    async def generate(self, messages, response_schema):
        await self._quiet.wait()
        return await self.client.generate(messages, response_schema)

    def __getattr__(self, name):  # last_metrics, stream_text, health, close, ...
        return getattr(self.client, name)
