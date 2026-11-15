"""Live reply-time benchmark against a running Kip: python -m app.bench [BASE_URL] [--timeout S].

Sends a fixed set of messages one at a time, waits until Kip is idle (replies, running and queued
errands), and prints the time to Kip's first and last reply for each. It talks to Kip like the
browser does, so it writes to whatever kip.db that backend uses: point it at a backend started
with MEMORY_DB_FILE set to a copy if the conversation should not be kept.
"""
import argparse
import json
import statistics
import sys
import time
import urllib.request

MESSAGES = [
    ("chat", "Hi Kip, how are you doing today?"),
    ("chat", "Tell me a short joke."),
    ("memory", "Remember that my favourite fruit is mango."),
    ("memory", "What's my favourite fruit?"),
    ("live", "What's the weather in London today?"),
    ("python", "Use Python to calculate the average of 12, 18, 21 and 29."),
    ("read", "Read https://example.com and tell me what it says."),
]


def _call(base: str, method: str, path: str, body=None):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def run(base: str, text: str, timeout: float) -> dict:
    start = time.perf_counter()
    sent = _call(base, "POST", "/api/chat", {"message": text})
    first = last = None
    replies: list[dict] = []
    while time.perf_counter() - start < timeout:
        time.sleep(0.25)
        chat = _call(base, "GET", "/api/chat")
        turns = chat["turns"]
        index = next((i for i, t in enumerate(turns) if t["id"] == sent["id"]), None)
        after = [t for t in turns[index + 1:] if t["role"] != "user"] if index is not None else []
        now = time.perf_counter() - start
        if after and first is None:
            first = now
        if len(after) != len(replies):
            last, replies = now, after
        # "busy" covers replies only: also wait for the running errand and the queue.
        task = _call(base, "GET", "/api/state").get("task") or {}
        if after and not chat["busy"] and task.get("status") != "running" and not chat.get("queued_errands"):
            break
    return {"first_s": first, "last_s": last, "timed_out": time.perf_counter() - start >= timeout,
            "replies": [t["text"] for t in replies]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base", nargs="?", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=600, help="seconds per message")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(_call(args.base, "GET", "/api/health"))
    done = []
    for kind, text in MESSAGES:
        result = run(args.base, text, args.timeout)
        seconds = lambda value: f"{value:.1f}s" if value is not None else "-"  # noqa: E731
        print(f"[{kind:6}] first {seconds(result['first_s']):>7}  last {seconds(result['last_s']):>7}  "
              f"{'TIMEOUT ' if result['timed_out'] else ''}{text}")
        for reply in result["replies"]:
            print("         ->", reply[:200].replace("\n", " "))
        if result["last_s"] is not None:
            done.append(result["last_s"])
    if done:
        print(f"median {statistics.median(done):.1f}s, max {max(done):.1f}s, {len(done)}/{len(MESSAGES)} answered")


if __name__ == "__main__":
    main()
