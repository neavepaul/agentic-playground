"""Offline tools for the intent router. Nothing here runs inside a conversation.

    python -m app.chat.router_train eval [--embeddings] [--compare]
                                                       measure the router against evaluations/intent_eval.jsonl
    python -m app.chat.router_train review             list corrections people made, awaiting review
    python -m app.chat.router_train accept ID [LABEL]  use a correction as a training example
    python -m app.chat.router_train reject ID          drop it
    python -m app.chat.router_train shadow             how the router compared with the live path

Accepted corrections join the training data the next time Kip starts; the router is never
retrained during a conversation.
"""
import json
import re
import statistics
import sys
import time
from datetime import date
from pathlib import Path

from app.chat.intent import is_live_fact, read_intent
from app.chat.router import (LIVE, THRESHOLD, IntentRouter, TfidfLogistic, fast_path, frame_of, normalize,
                             training_rows)

EVAL = Path(__file__).resolve().parents[2] / "evaluations" / "intent_eval.jsonl"
HOUSE = re.compile(r"\b(paul|moira|neave|kitchen|hall|bedroom|study)\b")
TODAY = date(2026, 11, 1)


def cases(path: Path = EVAL) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _legacy(case: dict, intent, earlier_fast: str | None) -> bool:
    """What the regex path decided before the router: a live-fact word in a question, or a
    follow-up sharing a word with the lookup just answered."""
    said = case["say"]
    if intent.cancel or intent.changes_state_only or intent.about_past or intent.wants_offer or intent.needs_tools:
        return False
    if not (intent.question or intent.request) or "web_search" in case.get("forbidden", []):
        return False
    if HOUSE.search(normalize(said)):
        return False
    if intent.live_fact:
        return True
    words = lambda text: {w for w in re.findall(r"[a-z]{4,}", normalize(text))}
    return bool(earlier_fast and words(said) & words(earlier_fast))


def evaluate(router: IntentRouter, rows: list[dict]) -> dict:
    tally: dict[str, list[bool]] = {}
    mark = lambda key, ok: tally.setdefault(key, []).append(bool(ok))
    latencies, failures = [], []
    legacy_fast = router_fast = avoided = wrong_fast = legacy_wrong = 0
    for case in rows:
        frame, earlier_fast = None, None
        for turn, said in enumerate(case.get("context", [])):
            earlier = router.route(said, frame, turn, TODAY)
            frame = frame_of(earlier, said, turn) or frame
            earlier_fast = said if _legacy({"say": said}, read_intent(said), None) else earlier_fast
        started = time.perf_counter()
        route = router.route(case["say"], frame, len(case.get("context", [])), TODAY)
        latencies.append((time.perf_counter() - started) * 1000)
        intent = read_intent(case["say"], revoked=case.get("revoked", []))
        fast = fast_path(route, intent, web_allowed="web_search" not in case.get("forbidden", []),
                         involves_house=bool(HOUSE.search(normalize(case["say"]))))
        old = _legacy(case, intent, earlier_fast)
        problems = []
        if case.get("intent"):
            mark("intent", route.intent == case["intent"])
            if route.intent != case["intent"]:
                problems.append(f"intent {route.intent} ({route.confidence:.2f})")
        want_tool = case.get("tool") == "web_search"
        mark("tool routing", (route.capability == "web_search") == want_tool)
        if (route.capability == "web_search") != want_tool:
            problems.append(f"tool {route.capability}")
        mark("fast path", fast == case["fast"])
        if fast != case["fast"]:
            problems.append(f"fast={fast}")
        mark("legacy fast path", old == case["fast"])
        for key in ("domain", "focus", "time"):
            if key in case:
                got = getattr(route, key)
                mark(key, got == case[key])
                if got != case[key]:
                    problems.append(f"{key} {got!r}")
        if case.get("entities"):
            ok = set(case["entities"]) <= set(route.entities)
            mark("entities", ok)
            if not ok:
                problems.append(f"entities {route.entities}")
        if "clarify" in case:
            mark("clarification", bool(route.clarify) == case["clarify"])
        if "correction" in case:
            mark("correction detection", route.correction == case["correction"])
            if route.correction != case["correction"]:
                problems.append(f"correction={route.correction}")
        if "context" in case.get("tags", []) or (case.get("context") and "correction" not in case):
            mark("context resolution", not problems)
        if "regression" in case.get("tags", []):
            mark("regression cases", not problems)
        legacy_fast += old
        router_fast += fast
        wrong_fast += fast and not case["fast"]
        legacy_wrong += old and not case["fast"]
        avoided += fast and case["fast"] and not old
        if problems:
            failures.append(f"{case['say']!r}: " + ", ".join(problems))
    expected_fast = sum(case["fast"] for case in rows)
    return {"cases": len(rows), "accuracy": {k: (sum(v), len(v)) for k, v in tally.items()},
            "false_positive_fast": wrong_fast, "legacy_false_positive_fast": legacy_wrong,
            "fast_routes": router_fast, "legacy_fast_routes": legacy_fast, "expected_fast": expected_fast,
            "chat_calls_avoided": avoided,
            "fallback_share": round(1 - router_fast / len(rows), 3),
            "latency_ms": {"p50": round(statistics.median(latencies), 2),
                           "p95": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 2)},
            "failures": failures}


def _softmax_fit(X, labels, epochs=400, rate=8.0, l2=1e-4):
    import numpy as np
    classes = sorted(set(labels))
    Y = np.zeros((len(labels), len(classes)), np.float32)
    Y[np.arange(len(labels)), [classes.index(l) for l in labels]] = 1
    W, b = np.zeros((X.shape[1], len(classes)), np.float32), np.zeros(len(classes), np.float32)
    for _ in range(epochs):
        Z = X @ W + b
        P = np.exp(Z - Z.max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        G = (P - Y) / len(labels)
        W -= rate * (X.T @ G + l2 * W)
        b -= rate * G.sum(0)
    return lambda x: classes[int(np.argmax(x @ W + b))]


def _approaches():
    """name -> (fit(texts, labels) -> predict(text), setup seconds). Embedding approaches only
    when the local model is downloaded."""
    import numpy as np
    found = {"TF-IDF + logistic regression (chosen)":
             lambda texts, labels: (lambda m: lambda t: max((p := m.predict(t)), key=p.get))(
                 TfidfLogistic(texts, labels))}
    try:
        from app.config import Settings
        from app.memory.recall import Embedder
        settings = Settings()
        embedder = Embedder(settings.embedding_model, settings.embedding_cache_dir)
        cache: dict[str, np.ndarray] = {}

        def embed(texts):
            missing = [t for t in texts if t not in cache]
            if missing:
                for text, vector in zip(missing, embedder.embed_sync(missing)):
                    cache[text] = np.array(vector, np.float32)
            return np.stack([cache[t] for t in texts])
        embed(["warm up"])

        def logistic(texts, labels):
            predict = _softmax_fit(embed(texts), labels)
            return lambda t: predict(embed([t])[0])

        def knn(texts, labels, k=5):
            vectors = embed(texts)

            def predict(text):
                sims = vectors @ embed([text])[0]
                votes: dict[str, float] = {}
                for i in np.argsort(-sims)[:k]:
                    votes[labels[i]] = votes.get(labels[i], 0) + float(sims[i])
                return max(votes, key=votes.get)
            return predict
        found["bge-small embeddings + logistic regression"] = logistic
        found["bge-small embeddings + 5-nearest neighbours"] = knn
    except Exception as exc:
        print(f"(embedding approaches skipped: {type(exc).__name__})")
    return found


def compare(rows: list[dict], folds: int = 5) -> list[dict]:
    """Candidate classifiers on the evaluation utterances, and in 5-fold cross-validation on
    the training examples (which nothing was tuned against). "live" is the decision that
    matters for routing: does this need current information at all."""
    train = training_rows()
    texts, labels = [t for _, t in train], [l for l, _ in train]
    scored = [c for c in rows if c.get("intent") and not c.get("context")]
    results = []

    def regex_fit(texts, labels):
        def predict(text):
            intent = read_intent(text)
            return "information.sports" if intent.live_fact and (intent.question or intent.request) else "other"
        return predict
    for name, fit in {"regex live-fact rule (current)": regex_fit, **_approaches()}.items():
        started = time.perf_counter()
        predict = fit(texts, labels)
        setup = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        guesses = [predict(c["say"]) for c in scored]
        per = (time.perf_counter() - started) * 1000 / len(scored)
        live_ok = sum((g in LIVE) == (c["intent"] in LIVE) for g, c in zip(guesses, scored))
        false_live = sum(g in LIVE and c["intent"] not in LIVE for g, c in zip(guesses, scored))
        cv_right = cv_live = 0
        for fold in range(folds):
            held = [(l, t) for i, (l, t) in enumerate(train) if i % folds == fold]
            kept = [(l, t) for i, (l, t) in enumerate(train) if i % folds != fold]
            model = fit([t for _, t in kept], [l for l, _ in kept])
            for label, text in held:
                guess = model(text)
                cv_right += guess == label
                cv_live += (guess in LIVE) == (label in LIVE)
        regex = name.startswith("regex")
        results.append({"approach": name,
                        "eval_intent": "n/a" if regex else f"{sum(g == c['intent'] for g, c in zip(guesses, scored))}/{len(scored)}",
                        "eval_live_vs_not": f"{live_ok}/{len(scored)}", "eval_false_live": false_live,
                        "cv_intent": "n/a" if regex else f"{cv_right}/{len(train)}",
                        "cv_live_vs_not": f"{cv_live}/{len(train)}",
                        "ms_per_message": round(per, 2), "setup_ms": round(setup)})
    return results


def _embedder():
    from app.config import Settings
    from app.memory.recall import Embedder
    settings = Settings()
    return Embedder(settings.embedding_model, settings.embedding_cache_dir)


def _store():
    from app.config import Settings
    from app.memory.recall import MemoryStore
    return MemoryStore(Settings().memory_db_file)


def main(argv: list[str]) -> int:
    command = argv[0] if argv else "eval"
    if command == "eval":
        rows = cases()
        started = time.perf_counter()
        router = IntentRouter(_reviewed(), embedder=_embedder() if "--embeddings" in argv else None)
        router.add_embeddings()
        print(f"{router.backend} router trained on {len(router.rows)} examples in "
              f"{(time.perf_counter() - started) * 1000:.0f} ms; threshold {THRESHOLD}")
        report = evaluate(router, rows)
        for key, (right, total) in report.pop("accuracy").items():
            print(f"  {key:<22} {right}/{total} ({right / total:.0%})")
        failures = report.pop("failures")
        print(json.dumps(report, indent=2))
        if failures:
            print("misses:\n  " + "\n  ".join(failures))
        if "--compare" in argv:
            for row in compare(rows):
                print(json.dumps(row))
        return 0
    store = _store()
    if command == "review":
        for row in store.intent_corrections(status="unreviewed"):
            print(json.dumps(row, ensure_ascii=False))
        return 0
    if command in ("accept", "reject") and len(argv) >= 2:
        label = argv[2] if len(argv) > 2 else None
        ok = store.review_intent_correction(int(argv[1]), "accepted" if command == "accept" else "rejected", label)
        print("updated" if ok else "no such correction")
        return 0 if ok else 1
    if command == "shadow":
        print(json.dumps(store.intent_shadow_summary(), indent=2))
        return 0
    print(__doc__)
    return 1


def _reviewed() -> list[tuple[str, str]]:
    try:
        return _store().accepted_intent_examples()
    except Exception:
        return []


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
