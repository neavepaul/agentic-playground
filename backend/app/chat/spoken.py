"""Kip speaks; he doesn't print. Turn text meant for a screen into something said aloud.

The models are asked for plain speech already; this is the deterministic last step so a
stray code block, bullet list or link never reaches the voice. Evidence (sources, code,
tool output) stays in the turn's metadata and task records, just not in what Kip says.
"""
import re

_FENCE = re.compile(r"```([\w+-]*)\n?(.*?)```", re.S)
_CODE_CHARS = re.compile(r"[=()\[\]{};]|\bdef\b|\bimport\b|\bprint\b")
_INLINE_CODE = re.compile(r"`([^`]*)`")
_EMPHASIS = re.compile(r"(\*\*|__|\*|_)(?=\S)(.+?)(?<=\S)\1")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.M)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.M)
_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
_URL = re.compile(r"https?://(?:www\.)?([^/\s]+?)(?:/[^\s]*?)?(?=[.,;:!?)]*(?:\s|$))")


def spoken(text: str) -> str:
    def fence(match: re.Match) -> str:
        language, body = match.group(1), match.group(2).strip()
        # A bare one-line result ("10800") can be said; a program can't, and never needs to be.
        if language or not body or "\n" in body or len(body) > 40 or _CODE_CHARS.search(body):
            return ""
        return body
    text = _FENCE.sub(fence, text)
    text = _LINK.sub(lambda m: m.group(1), text)
    text = _URL.sub(lambda m: m.group(1), text)
    text = _INLINE_CODE.sub(lambda m: m.group(1), text)
    text = _EMPHASIS.sub(lambda m: m.group(2), text)
    text = _HEADING.sub("", text)
    lines = [line for line in (_BULLET.sub("", line).strip() for line in text.splitlines()) if line]
    # Lines become sentences of one utterance; a single line is left exactly as written.
    said = " ".join(line if index == len(lines) - 1 or line[-1] in ".!?:;,…" else line + "."
                    for index, line in enumerate(lines))
    said = re.sub(r"\s+([.,!?;:])", r"\1", said)
    said = re.sub(r"(?<=[.!?])\.+", "", said)
    return re.sub(r"\s{2,}", " ", said).strip()
