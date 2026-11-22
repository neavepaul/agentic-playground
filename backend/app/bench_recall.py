"""Compare KIP recall on small and large synthetic histories without touching kip.db."""
import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from app.chat.session import ChatSession
from app.config import Settings
from app.events.bus import EventBus
from app.llm.factory import create_client, create_hosted_client
from app.llm.groq import HostedChat
from app.memory.recall import Embedder, MemoryStore
from app.tasks.manager import TaskManager


CASES = (
    ("What is my favourite tea?", "silver needle", "My favourite tea is silver needle."),
    ("Where do I keep the telescope?", "cedar cupboard", "I keep the telescope in the cedar cupboard."),
    ("What name did I give my sourdough starter?", "bubbles", "My sourdough starter is named Bubbles."),
    ("Which instrument am I learning?", "mandolin", "I am learning to play the mandolin."),
)


class BenchmarkSession(ChatSession):
    async def _remember(self, action, user_turn, reply, record=True):
        # Keep the same corpus for each question and repeat.
        pass


class Capture:
    def __init__(self, client):
        self.client = client
        self.prompt = ()

    def __getattr__(self, name):
        return getattr(self.client, name)

    async def stream_text(self, messages, **kwargs):
        self.prompt = messages
        async for chunk in self.client.stream_text(messages, **kwargs):
            yield chunk
        if self.last_metrics.get("provider") != "groq":
            self.prompt = kwargs.get("local_messages") or messages


async def seed(path, count):
    store = MemoryStore(path)
    try:
        for _, _, text in CASES:
            await store.add("preference" if "favourite" in text else "fact", text, meta={"speaker": "benchmark"})
        topics = ("jazz", "cycling", "ceramics", "hiking", "gardening", "astronomy", "cooking", "painting")
        for index in range(count - len(CASES)):
            await store.add("fact", f"Note {index}: my {topics[index % len(topics)]} journal uses page {index * 7}.",
                            meta={"speaker": "benchmark"})
    finally:
        store.close()


async def run_case(client, settings, store, question, expected):
    bus = EventBus()
    manager = TaskManager(client, bus, settings)
    session = BenchmarkSession(client, bus, manager, settings, store=store, speaker="Benchmark", owner=False)
    first_word = None
    started = time.perf_counter()

    def observe(event):
        nonlocal first_word
        if event["type"] == "chat_delta" and event["data"].get("text", "").strip() and first_word is None:
            first_word = time.perf_counter() - started

    bus.add_listener(observe)
    try:
        session.submit(question)
        await asyncio.wait_for(session.drain(), timeout=settings.llm_timeout_seconds + 30)
        reply = next((turn for turn in reversed(session.history()) if turn["role"] == "kip"), {})
        metrics = reply.get("metrics") or {}
        if first_word is None or metrics.get("estimated_prompt_tokens") is None:
            raise RuntimeError(f"No measurable model reply for {question!r}: {reply.get('error') or reply.get('text')}")
        prompt = "\n".join(message["content"] for message in client.prompt or ())
        return {
            "first_word_s": round(first_word, 3) if first_word is not None else None,
            "recall_s": metrics.get("memory_recall_seconds"),
            "prompt_tokens_est": metrics.get("estimated_prompt_tokens"),
            "fact_in_prompt": expected in prompt.casefold(),
            "fact_in_answer": expected in reply.get("text", "").casefold(),
            "provider": metrics.get("provider", "ollama"),
            "model": metrics.get("model"),
        }
    finally:
        await session.close()
        await manager.close()


async def run(args):
    settings = Settings(intent_router="off", chat_compaction=False, chat_extraction=False,
                        digital_tools_enabled=False, persona_enabled=False)
    local = create_client(settings)
    hosted = create_hosted_client(settings) if args.hosted else None
    if args.hosted and hosted is None:
        await local.close()
        raise SystemExit("--hosted requires a Groq key in the KIP backend configuration.")
    if hosted:
        models = [name.strip() for name in settings.groq_chat_models.split(",") if name.strip()]
        model = HostedChat(hosted, local, models)
    else:
        model = local
    embedder = (Embedder(settings.embedding_model, settings.embedding_cache_dir)
                if settings.embeddings_enabled and not args.keyword_only else None)
    try:
        with TemporaryDirectory(prefix="kip-recall-bench-") as directory:
            for size in args.sizes:
                path = Path(directory) / f"memory-{size}.db"
                await seed(path, size)
                store = MemoryStore(path, embedder)
                rows = []
                try:
                    indexed = await store.backfill() if embedder else 0
                    print(json.dumps({"memories": size, "indexed": indexed}), flush=True)
                    if embedder and indexed != size:
                        raise SystemExit("Embeddings did not index every seed record. Retry with --keyword-only for a keyword baseline.")
                    await run_case(Capture(model), settings, store, *CASES[0][:2])  # warm both model and recall
                    for _ in range(args.repeats):
                        for question, expected, _ in CASES:
                            result = await run_case(Capture(model), settings, store, question, expected)
                            rows.append(result)
                            print(json.dumps({"question": question, **result}), flush=True)
                finally:
                    store.close()
                summary = {"memories": size, "questions": len(rows),
                           "facts_in_prompt": sum(row["fact_in_prompt"] for row in rows),
                           "facts_in_answer": sum(row["fact_in_answer"] for row in rows)}
                for field in ("first_word_s", "recall_s", "prompt_tokens_est"):
                    values = [row[field] for row in rows if row[field] is not None]
                    summary[f"median_{field}"] = statistics.median(values) if values else None
                print(json.dumps({"summary": summary}), flush=True)
    finally:
        if hosted:
            await hosted.close()
        await local.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs=2, default=[50, 2000], metavar=("SMALL", "LARGE"))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--keyword-only", action="store_true")
    parser.add_argument("--hosted", action="store_true", help="Use configured Groq chat with Ollama fallback.")
    args = parser.parse_args()
    if not len(CASES) <= args.sizes[0] < args.sizes[1] or args.repeats < 1:
        parser.error("provide increasing sizes of at least four memories and a positive repeat count")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
