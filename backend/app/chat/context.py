"""Bounded conversational context without additional inference.

The fallback counts UTF-8 bytes conservatively (one byte per token plus framing),
not characters divided by four. Deployments can inject their local tokenizer.
This is an upper estimate, not reported provider usage or a Qwen benchmark.
"""
import json
import re
from collections.abc import Callable

from app.llm.base import ModelError

TokenCounter = Callable[[str], int]


class ContextBudgetError(ModelError):
    """Required instructions and current request cannot fit without losing meaning."""


def estimate_tokens(text: str) -> int:
    return len(text.encode("utf-8"))


def message_tokens(messages: list[dict], counter: TokenCounter = estimate_tokens) -> int:
    return sum(8 + counter(m["content"]) for m in messages)


_IMPORTANT = re.compile(r"\b(actually|correction|instead|remember|promise|promised|commitment|"
                        r"allergic|allergy|don't|do not|never|must|prefer)\b", re.I)
_WORD = re.compile(r"\w{3,}")


def select_history(turns: list[dict], query: str, budget: int,
                   counter: TokenCounter = estimate_tokens) -> list[dict]:
    """Recent complete turns first; then exact older excerpts by relevance/recency.

    No fabricated summary facts, sliced code, truncated corrections or replayed
    control metadata. Older selections are quoted with their roles, not promoted
    to system instructions. The full transcript remains in SQLite.
    """
    if budget <= 0:
        return []
    def content(turn):
        prefix = "[update] " if turn.get("source") == "narration" else ""
        return prefix + turn.get("content", turn["text"])

    messages = [{"role": "user" if t["role"] == "user" else "assistant",
                 "content": content(t)} for t in turns]
    if message_tokens(messages, counter) <= budget:
        return messages
    # Reserve a quarter for older relevant context, especially corrections and
    # commitments. Never skip a large recent turn to stitch unrelated old dialogue.
    recent, used = [], 0
    for index in range(len(messages) - 1, -1, -1):
        cost = message_tokens([messages[index]], counter)
        if used + cost > budget * 0.75:
            break
        recent.insert(0, index)
        used += cost
    query_words = set(_WORD.findall(query.casefold()))
    cutoff = recent[0] if recent else len(messages)
    candidates = []
    for index in range(cutoff):
        text = messages[index]["content"]
        overlap = len(query_words & set(_WORD.findall(text.casefold())))
        important = messages[index]["role"] == "user" and bool(_IMPORTANT.search(text))
        if overlap or important:
            candidates.append((int(important), overlap, index))
    heading = "Earlier conversation excerpts (incomplete quoted history; later corrections take precedence):\n"
    excerpts = []
    available = budget - used
    for _, _, index in sorted(candidates, reverse=True):
        item = {"role": messages[index]["role"], "text": messages[index]["content"]}
        trial = sorted([*excerpts, (index, item)])
        summary = heading + json.dumps([value for _, value in trial], ensure_ascii=False)
        if message_tokens([{"content": summary}], counter) <= available:
            excerpts = trial
    result = []
    if excerpts:
        result.append({"role": "user", "content": heading + json.dumps(
            [item for _, item in excerpts], ensure_ascii=False)})
    result.extend(messages[index] for index in recent)
    return result


def assemble(system: str, latest: str, turns: list[dict], query: str, budget: int,
             counter: TokenCounter = estimate_tokens) -> list[dict]:
    mandatory = [{"role": "system", "content": system}, {"role": "user", "content": latest}]
    remaining = budget - message_tokens(mandatory, counter)
    if remaining < 0:
        raise ContextBudgetError("This message and its required context exceed the configured conversation budget. "
                                 "Please split the request or increase CHAT_CONTEXT_TOKENS.")
    return [mandatory[0], *select_history(turns, query, remaining, counter), mandatory[1]]
