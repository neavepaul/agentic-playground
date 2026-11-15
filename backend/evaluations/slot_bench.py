"""How well, and how fast, a feed question is read into that feed's form.

    cd backend && .venv/bin/python -m evaluations.slot_bench --reader [--only f1|finance]
    cd backend && .venv/bin/python -m evaluations.slot_bench [--runs 2] [--only f1|finance] [--out results.json]

For each question in slot_cases.jsonl the intent router picks the domain (as Kip does), then:
  --reader   the example-trained readers Kip uses (app/capabilities/readers.py) fill the form. Local
             and instant; a reading too unsure to use is counted apart (the web lookup answers it).
  otherwise  the local Ollama model (the one in .env, through Kip's own client, same context size
             so the loaded model is reused) fills a JSON form. On the Pi 4 with qwen3.5 2B this took
             11-73 s a question (15 November 2026), which is why Kip uses the readers.
Each form is scored field by field against what was expected. Questions tagged "fresh" are
phrasings no example or rule was written for: the fair test. The regex rules the readers replaced
scored 33/34 on the questions they were written for and 8/14 on the fresh ones.
Nothing here runs inside a conversation, and nothing is fetched.
"""
import argparse
import asyncio
import json
import re
import statistics
import time
from datetime import date
from pathlib import Path

from app.capabilities import feeds
from app.chat.router import default_router
from app.config import Settings
from app.llm.ollama import OllamaClient

CASES = Path(__file__).with_name("slot_cases.jsonl")
TEXT = {"anyOf": [{"type": "string"}, {"type": "null"}]}

FORMS = {
    "f1": {
        "schema": {"title": "F1Question", "type": "object", "properties": {
            "ask": {"enum": ["standings", "race result", "qualifying", "sprint", "starting grid", "next race", "other"]},
            "table": {"enum": ["drivers", "constructors"]},
            "name": TEXT,
            "season": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
            "race": TEXT,
        }, "required": ["ask", "table", "name", "season", "race"]},
        "guide": """Read a Formula 1 question into this form.
ask: standings (championship, points, title, champion, leader), race result (who won, how someone finished,
podium), qualifying, sprint, starting grid, next race (when, what time, schedule), other.
table: constructors when a team (McLaren, Ferrari, Red Bull, Mercedes...) or the constructors' or teams'
championship is asked about; drivers otherwise. A driver named with their team is the driver.
name: the driver (full name if you know it) or team asked about, else null.
season: a year they name, else null. race: the Grand Prix or place they name, else null.
Examples:
"is norris leading the championship" -> {"ask":"standings","table":"drivers","name":"Lando Norris","season":null,"race":null}
"how many points have mercedes got" -> {"ask":"standings","table":"constructors","name":"Mercedes","season":null,"race":null}
"who won in monaco" -> {"ask":"race result","table":"drivers","name":null,"season":null,"race":"Monaco"}
"when is qualifying in las vegas" -> {"ask":"next race","table":"drivers","name":null,"season":null,"race":"Las Vegas"}""",
    },
    "finance": {
        "schema": {"title": "FinanceQuestion", "type": "object", "properties": {
            "kind": {"enum": ["my portfolio", "my holding", "share price", "index", "crypto", "exchange rate", "other"]},
            "name": TEXT,
            "when": {"enum": ["now", "past"]},
        }, "required": ["kind", "name", "when"]},
        "guide": """Read a markets question into this form.
kind: my portfolio (their own holdings as a whole: worth, gains, best or worst, top holding),
my holding (one company they own: how much, how many shares, am I up on it), share price (a company's
price, not about what they own), index (Nifty, Sensex...), crypto, exchange rate, other.
name: the company, index or coin named, else null.
when: past if they ask about an earlier date or period, else now.
Examples:
"what are my stocks worth" -> {"kind":"my portfolio","name":null,"when":"now"}
"how many infosys shares have i got" -> {"kind":"my holding","name":"Infosys","when":"now"}
"tata motors share price" -> {"kind":"share price","name":"Tata Motors","when":"now"}
"where was the sensex last month" -> {"kind":"index","name":"Sensex","when":"past"}""",
    },
}

_TEAMS = {"mclaren", "ferrari", "red bull", "mercedes", "williams", "aston martin", "alpine", "haas", "sauber",
          "racing bulls", "audi", "cadillac"}
_F1_ASK = {"standings": "standings", "finishing position": "race result", "result": "race result",
           "qualifying position": "qualifying", "sprint result": "sprint", "starting position": "starting grid",
           "schedule": "next race", "start time": "next race"}
_MARKET = {"index": "index", "coin": "crypto", "currency": "exchange rate", "share": "share price"}


def cases(only: str | None, path: Path = CASES) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if only in (None, r["domain"])]


def _bare(text) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text or "").casefold())


def score(expect: dict, got: dict) -> list[str]:
    """The fields that came out wrong. A name matches loosely: "Norris" is "Lando Norris"."""
    wrong = []
    for key, want in expect.items():
        have = got.get(key)
        if key in ("name", "race") and want is not None:
            ok = bool(_bare(have)) and (_bare(want) in _bare(have) or _bare(have) in _bare(want))
        elif want is None:
            ok = have in (None, "")
        else:
            ok = have == want
        if not ok:
            wrong.append(f"{key}={have!r} (want {want!r})")
    return wrong


def reader_reading(domain: str, route, said: str) -> tuple[dict, bool]:
    """The example-trained reader's form (app/capabilities/readers.py), as the feed would use it,
    and whether it is sure enough to be used (unsure: the web lookup answers instead)."""
    reading = feeds.reader(domain).read(said)
    if reading.untaught:
        print(f"        untaught words: {' '.join(reading.untaught)}")
    if domain == "finance":
        name = reading.known[0] if reading.known else feeds._holding_name(reading, {"entities": route.entities})
        # The feed uses a name only where the kind asks about one.
        named = reading.form["kind"] in ("my holding", "share price", "index", "crypto")
        return {**reading.form, "name": name if named else None}, reading.confident
    drivers = [e for e in route.entities if e.casefold() not in _TEAMS]
    year = re.search(r"\b(19[5-9]\d|20\d\d)\b", said)
    # The F1 feed takes names from the router (and checks them against the standings), not the reader.
    return ({**reading.form, "name": (drivers or route.entities or [None])[0],
             "season": int(year.group(1)) if year else None, "race": route.place},
            reading.confident)


def run_readers(rows: list[dict]) -> None:
    router = default_router()
    totals: dict[str, list[int]] = {}
    for case in rows:
        route = router.route(case["say"], None, 0)
        started = time.perf_counter()
        got, sure = reader_reading(case["domain"], route, case["say"])
        ms = (time.perf_counter() - started) * 1000
        wrong = score(case["expect"], got)
        # Not a feed question: unsure and "other" both leave it to the web lookup, which is right.
        off_topic = "other" in (case["expect"].get("ask"), case["expect"].get("kind"))
        verdict = "ok" if off_topic and not sure else "unsure" if not sure else "ok" if not wrong else "BAD"
        routed = "" if route.domain == case["domain"] else f"   [router domain: {route.domain}]"
        print(f"{verdict:6} {ms:5.1f}ms  {case['say']}{routed}")
        if wrong:
            print(f"        reader: {'; '.join(wrong)}")
        tags = case.get("tags", [])
        for group in (case["domain"], "off-topic" if "off-topic" in tags else "fresh" if "fresh" in tags
                      else "independent" if "independent" in tags else "written-for"):
            t = totals.setdefault(group, [0, 0, 0, 0, 0])  # right, wrong, unsure, router right, count
            t[0 if verdict == "ok" else 1 if verdict == "BAD" else 2] += 1
            t[3] += route.domain == case["domain"]
            t[4] += 1
    print("\nsummary (unsure goes to the web lookup, so it is not a wrong answer)")
    for group, (right, bad, unsure, routed, n) in totals.items():
        print(f"  {group:12} reader right {right}/{n}, wrong {bad}, unsure {unsure}   router domain right {routed}/{n}")


async def ask_model(client: OllamaClient, domain: str, said: str, today: str) -> tuple[dict, dict]:
    form = FORMS[domain]
    # The guide is the same every call (the date changes once a day), so Ollama can reuse it.
    messages = [{"role": "system", "content": f"{form['guide']}\nToday is {today}. Reply with the JSON form only."},
                {"role": "user", "content": said}]
    started = time.perf_counter()
    raw = await client.generate(messages, form["schema"])
    metrics = {**client.last_metrics, "wall_seconds": round(time.perf_counter() - started, 2)}
    try:
        return json.loads(raw), metrics
    except json.JSONDecodeError:
        return {"_unparsed": raw}, metrics


def _pct(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(p * (len(ordered) - 1)))]


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=1, help="ask each question this many times (timing spread)")
    parser.add_argument("--only", choices=sorted(FORMS), help="one domain")
    parser.add_argument("--out", type=Path, help="write every result here as JSON")
    parser.add_argument("--cases", type=Path, default=CASES,
                        help="questions to score (slot_cases_independent.jsonl: written blind, never trained on)")
    parser.add_argument("--reader", action="store_true",
                        help="score the example-trained readers instead of the model (no Ollama needed)")
    args = parser.parse_args()
    if args.reader:
        run_readers(cases(args.only, args.cases))
        return

    settings = Settings()
    client = OllamaClient(settings)
    router = default_router()
    today = f"{date.today().day} {date.today():%B %Y}"
    rows = cases(args.only, args.cases)
    print(f"model {settings.ollama_model} at {settings.ollama_base_url}, context {settings.context_tokens}, "
          f"{len(rows)} questions x {args.runs}\n")

    results = []
    # One call first so a model load is not counted against the first question.
    _, warm = await ask_model(client, rows[0]["domain"], rows[0]["say"], today)
    print(f"warm-up call: {warm.get('wall_seconds')} s (load {warm.get('load_seconds', 0)} s, "
          f"prompt {warm.get('prompt_tokens')} tokens)\n")
    for case in rows:
        started = time.perf_counter()
        route = router.route(case["say"], None, 0)
        router_ms = (time.perf_counter() - started) * 1000
        for run in range(args.runs):
            got, metrics = await ask_model(client, case["domain"], case["say"], today)
            model_wrong = score(case["expect"], got)
            results.append({"say": case["say"], "domain": case["domain"], "tags": case.get("tags", []),
                            "router_domain": route.domain,
                            "router_ms": round(router_ms, 1), "run": run, "model": got, "model_wrong": model_wrong, **metrics})
            mark = "ok " if not model_wrong else "BAD"
            print(f"{mark} {metrics['wall_seconds']:5.1f}s  p{metrics.get('prompt_tokens', '?'):>4} "
                  f"c{metrics.get('completion_tokens', '?'):>3}  {case['say']}")
            if model_wrong:
                print(f"        model: {'; '.join(model_wrong)}")

    print("\nsummary")
    for domain in sorted({r["domain"] for r in results}):
        mine = [r for r in results if r["domain"] == domain]
        secs = [r["wall_seconds"] for r in mine]
        model_ok = sum(not r["model_wrong"] for r in mine)
        firsts = sum(r["run"] == 0 for r in mine)
        routed = sum(r["router_domain"] == domain for r in mine if r["run"] == 0)
        print(f"  {domain:8} model right {model_ok}/{len(mine)}  "
              f"router domain right {routed}/{firsts}  "
              f"seconds median {statistics.median(secs):.1f} p90 {_pct(secs, 0.9):.1f} max {max(secs):.1f}")
    fresh = [r for r in results if "fresh" in r["tags"]]
    if fresh:
        # Phrasings no regex rule was written for: the fair comparison.
        print(f"  fresh phrasings only: model right {sum(not r['model_wrong'] for r in fresh)}/{len(fresh)}")
    prompts = [r.get("prompt_eval_seconds", 0) for r in results]
    evals = [r.get("eval_seconds", 0) for r in results]
    print(f"  time split (median): reading the prompt {statistics.median(prompts):.1f} s, "
          f"writing the form {statistics.median(evals):.1f} s")
    if args.out:
        args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nevery result: {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
