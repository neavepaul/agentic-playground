import asyncio
import logging

import httpx
from pydantic import ValidationError

from app import clock
from app.events.bus import EventBus
from .schemas import ACTION_SCHEMAS
from .python import run_python
from .web import read_webpage, web_search

DIGITAL_TOOLS = frozenset(ACTION_SCHEMAS)


def validation_reasons(exc: ValidationError, skip: int = 0) -> str:
    """Field-level reasons the model can act on, without echoing input values."""
    reasons = []
    for error in exc.errors():
        field = ".".join(str(part) for part in error["loc"][skip:]) or "input"
        reasons.append(f"{field}: {error['msg']}")
    return "; ".join(reasons)[:400]


def required_tool_progress(context) -> dict:
    required = {tool for c in context.conditions if c.kind == "answer" for tool in c.required_tools}
    evidence = {}
    for entry in context.action_history:
        if entry["success"] and entry["tool"] in DIGITAL_TOOLS - {"finish_answer"}:
            evidence.setdefault(entry["tool"], []).append(entry["evidence_id"])
    missing = sorted(required - evidence.keys())
    return {"missing_tools": missing, "can_finish": bool(evidence) and not missing,
            "successful_evidence_ids": {tool: ids[-6:] for tool, ids in evidence.items()}}


NOT_SHARED = ("That's the owner's Groww portfolio, and they haven't shared it with you. "
              "They can turn that on for your profile.")


class CapabilityTools:
    def __init__(self, bus: EventBus, enabled: bool = True, hosted=None, cricket_key=None, football_key=None, groww=(None, None)):
        self.bus, self.enabled = bus, enabled
        # Groq, whose built-in browser search can look a question up and answer it in one call.
        self.hosted = hosted
        self.cricket_key = cricket_key.get_secret_value().strip() if cricket_key else ""
        self.football_key = football_key.get_secret_value().strip() if football_key else ""
        self.groww = tuple(part.get_secret_value().strip() if part else "" for part in groww)

    def can_answer_from_web(self) -> bool:
        return self.enabled and self.hosted is not None and self.hosted.can_search()

    async def web_answer(self, query: str, messages: list[dict], context,
                         about_today: bool = False, about_latest: bool = False) -> tuple[dict, str]:
        """(web_search result, answer): the search run by Groq's browser search, which also
        writes the answer. Its pages are the evidence, recorded like any web_search.

        The answer is used only if what it states is in what Groq read: its quoted source must
        be there, and every time and figure must come from it (grounding.py). Otherwise the
        search counts as failed and the usual lookup, which reads the pages itself, takes over."""
        from app.llm.base import ModelError
        from .grounding import stale_latest, unbacked_results, unsupported, wrong_day, wrong_winner
        answer = ""
        try:
            found = await self.hosted.search(messages)
            answer = found.answer
            if not found.sources or not answer:
                raise ValueError("Groq answered without searching the web.")
            now = clock.now()
            problems = unsupported(answer, found.quote, found.evidence,
                                   int(now.utcoffset().total_seconds() // 60),
                                   today=now.date())
            # "7 runs" when the page says 7 wickets; "series 4-1" when it says 3-1.
            problems += unbacked_results(answer, f"{found.quote}\n{found.evidence}")
            winner = wrong_winner(answer, found.quote)  # "Antonelli won" from "...as Verstappen wins"
            if winner:
                problems.append(winner)
            stale = wrong_day([s["title"] for s in found.sources] + [found.quote], now.date()) if about_today else None
            if stale:
                problems.append(stale)  # a March final is not today's match
            if about_latest and not about_today:
                old = stale_latest([s["title"] for s in found.sources] + [found.quote], found.evidence, now.date())
                if old:
                    problems.append(old)  # an October race is not the most recent one in November
            if problems:
                raise ValueError("Groq's answer is not backed by what it read: " + "; ".join(problems)
                                 + f". It said: {answer[:200]!r}")
            result = {"success": True, "observation": {"query": query, "results": found.sources, "provider": "groq",
                                                       "quote": found.quote, "untrusted_content": True}}
            event_type = "capability_observed"
        except (ModelError, ValueError) as exc:
            logging.getLogger("agentic_friend.capabilities").warning("Groq web search not used: %s", exc)
            result, event_type, answer = {"success": False, "error": str(exc)[:400]}, "tool_failed", ""
        event = self.bus.emit(event_type, "explorer", task_id=context.id,
                              tool="web_search", arguments={"query": query}, **result)
        return {**result, "evidence_id": event["id"]}, answer

    def feed_source(self, question: dict, portfolio: bool = True) -> str | None:
        """The feed about to be asked, by name ("Groww"), or None when no feed will try. Without
        portfolio (the person may not see the owner's holdings), never Groww."""
        from . import feeds
        return feeds.source_for(question, cricket_key=self.cricket_key, football_key=self.football_key,
                                groww=self.groww if portfolio else ("", ""))

    async def feed_answer(self, query: str, question: dict, context) -> tuple[dict | None, str]:
        """(web_search result, answer) from a results feed (feeds.py), recorded like any search;
        (None, "") when the feed does not cover the question or cannot be reached."""
        from . import feeds
        if not getattr(context, "portfolio", True) and feeds.is_portfolio(question):
            # The holdings are the owner's: only people they allow may see them, and the question
            # never goes on to a web search or a model either.
            found = feeds.FeedAnswer(NOT_SHARED, "", "Groww holdings", "kip")
            # Nothing here is worth remembering, and it is about the owner's accounts: no memory.
            context.private = True
            result = {"success": True, "observation": {
                "query": query, "results": [{"title": found.title, "url": "", "snippet": found.answer}],
                "provider": found.provider, "quote": found.answer, "untrusted_content": False}}
            event = self.bus.emit("capability_observed", "explorer", task_id=context.id,
                                  tool="web_search", arguments={"query": query}, **result)
            return {**result, "evidence_id": event["id"]}, found.answer
        try:
            found = await feeds.answer(question, clock.now(), cricket_key=self.cricket_key,
                                       football_key=self.football_key, groww=self.groww)
        except feeds.FEED_ERRORS as exc:
            logging.getLogger("agentic_friend.capabilities").info("results feed not used: %s", type(exc).__name__)
            return None, ""
        if found is None:
            return None, ""
        result = {"success": True, "observation": {
            # The feed's facts are the result's text: answer checks read evidence as titles and snippets.
            "query": query, "results": [{"title": found.title, "url": found.url, "snippet": found.answer}],
            "provider": found.provider, "quote": found.answer, "untrusted_content": True}}
        shown = result
        if found.provider in feeds.PRIVATE_PROVIDERS:
            # Their own accounts: private from here on (prompts, memories, logs). The task record keeps
            # the figures for the answer checks; the event, its history and the log carry none.
            context.private = True
            hidden = "(private: shown only in the answer)"
            shown = {"success": True, "private": True, "observation": {
                **result["observation"], "quote": hidden,
                "results": [{"title": found.title, "url": found.url, "snippet": hidden}]}}
        event = self.bus.emit("capability_observed", "explorer", task_id=context.id,
                              tool="web_search", arguments={"query": query}, **shown)
        return {**result, "evidence_id": event["id"]}, found.answer

    async def execute(self, tool: str, arguments: dict, context) -> dict:
        try:
            if not self.enabled or tool not in ACTION_SCHEMAS:
                raise ValueError("This capability is unavailable.")
            args = ACTION_SCHEMAS[tool].model_validate({**arguments, "tool": tool})
            async with asyncio.timeout(30):
                if tool == "web_search":
                    observation = await asyncio.to_thread(web_search, args.query)
                elif tool == "read_webpage":
                    observation = await read_webpage(args.url, args.start_char)
                elif tool == "run_python":
                    observation = await run_python(args.code, args.data)
                else:
                    observation = self.finish_answer(args, context)
            result = {"success": True, "observation": observation}
            event_type = "capability_observed"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logging.getLogger("agentic_friend.capabilities").info("%s failed: %s", tool, type(exc).__name__)
            if isinstance(exc, ValidationError):
                error = "Invalid tool arguments: " + validation_reasons(exc)
            elif isinstance(exc, (ValueError, TimeoutError)):
                error = str(exc)[:400] or "Tool time limit reached."
            elif type(exc).__module__.startswith("pydantic_monty"):
                error = (type(exc).__name__ + ": " + str(exc))[:500]
            else:
                # Provider/runtime exceptions can contain environment details.
                error = f"{tool} failed ({type(exc).__name__}). Check the input, simplify it, or try another source."
            result, event_type = {"success": False, "error": error}, "tool_failed"
        shown = {"tool": tool, "arguments": arguments, **result}
        if getattr(context, "private", False):
            from app.events.bus import _redacted
            shown = {**_redacted(shown), "private": True}
        event = self.bus.emit(event_type, "explorer", task_id=context.id, **shown)
        return {**result, "evidence_id": event["id"]}

    @staticmethod
    def _wrong_event(answer: str, evidence: list[dict], goal: str) -> list[str]:
        """Times in a sports answer the cited pages give only for another match."""
        from app.chat.router import SPORTS, read_slots
        from .grounding import wrong_event
        slots = read_slots(goal)
        teams = list(dict.fromkeys(c for c, d in slots.entities + slots.ambiguous if d in SPORTS))
        if not teams:
            return []
        pages = []
        for item in evidence:
            obs = item["observation"]
            pages.append(str(obs.get("text") or ""))
            pages += [f"{r.get('title', '')} {r.get('snippet', '')}" for r in obs.get("results", [])]
        offset = int(clock.now().utcoffset().total_seconds() // 60)
        from .grounding import unbacked_results
        pages = [p for p in pages if p.strip()]
        return wrong_event(answer, pages, teams, offset) + unbacked_results(answer, "\n".join(pages))

    @staticmethod
    def finish_answer(args, context) -> dict:
        by_id = {a["evidence_id"]: a for a in context.action_history
                 if a["success"] and a["tool"] in DIGITAL_TOOLS - {"finish_answer"}}
        if any(key not in by_id for key in args.evidence_ids):
            raise ValueError("Answer must cite successful search, page or Python evidence from this task.")
        evidence = [by_id[key] for key in args.evidence_ids]
        required = {tool for c in context.conditions if c.kind == "answer" for tool in c.required_tools}
        missing = required - {a["tool"] for a in evidence}
        # A required tool that genuinely succeeded in this task but was not cited is
        # attached automatically: the guard exists to stop claims about tools that never
        # ran, not to reject a correct answer over citation bookkeeping.
        ran = {a["tool"]: a for a in by_id.values()}
        attached = [ran[tool] for tool in sorted(missing) if tool in ran]
        missing -= {a["tool"] for a in attached}
        if missing:
            raise ValueError("Answer is missing cited tool evidence for: " + ", ".join(sorted(missing))
                             + ". Run any unfinished tools and cite their successful evidence IDs.")
        wrong = CapabilityTools._wrong_event(args.answer, evidence + attached, context.goal)
        if wrong:
            # "8:05 a.m." from a fixture list where 8:05 is another match's start, or "won by 7
            # runs" from a page that says 7 wickets: try again.
            raise ValueError("Answer does not match the cited pages: " + "; ".join(wrong)
                             + ". Use the time and result the pages give for the match that was asked about.")
        sources = []
        for item in evidence + [a for a in attached if a["tool"] == "read_webpage"]:
            obs = item["observation"]
            candidates = obs.get("results", []) if item["tool"] == "web_search" else (
                [obs] if item["tool"] == "read_webpage" else [])
            for source in candidates:
                if not any(s["url"] == source["url"] for s in sources):
                    sources.append({"title": source.get("title", source["url"]), "url": source["url"]})
        return {"answer": args.answer, "sources": sources,
                "evidence_ids": args.evidence_ids + [a["evidence_id"] for a in attached]}
