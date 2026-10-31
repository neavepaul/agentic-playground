"""Kip's durable store: one SQLite file with searchable memories and structured state.

Every memory is one row (episode, fact, preference, skill, conversation, ...) with free
text, loose tags and an importance score. FTS5 finds exact words such as names;
sqlite-vec finds memories with similar meaning. Recall blends both with recency and
importance. Everything is advisory: the robot still verifies the world by observation.

The state table holds whole structured documents by key (Kip's tallies, self-model and
persona), written in one transaction so they can never be half-saved.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

_log = logging.getLogger("agentic_friend.recall")

RECENCY_HALF_LIFE_DAYS = 3.0
DUPLICATE_SIMILARITY = 0.95
CANDIDATES = 20
_WORD = re.compile(r"[\w']{2,}", re.UNICODE)


DUPLICATE_OVERLAP = 0.7
# A memory with no context label was stated as real.
_CONTEXT_SQL = "COALESCE(json_extract(meta, '$.context'), 'actual')"
# High on purpose: "Neave is allergic to peanuts" vs "Paul is allergic to peanuts" share
# 4 of 6 words, and a correction must never delete the corrected fact.
FORGET_OVERLAP = 0.8


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


_NEGATION_WORDS = {"not", "no", "never", "isn't", "isnt", "doesn't", "doesnt", "don't", "dont", "isn", "without",
                   "nobody", "nothing", "none", "neither", "nor", "aren't", "wasn't", "won't"}
_PERSONAL = {"i", "me", "my", "you", "your", "he", "she", "his", "her", "they", "their", "user"}


def _negated(text: str) -> bool:
    words = set(_WORD.findall(text.casefold().replace("’", "'"))) | set(text.casefold().split())
    return bool(words & _NEGATION_WORDS) or "n't" in text.casefold()


_SENTENCE_STARTERS = {"on", "in", "when", "if", "after", "before", "during", "every", "today", "yesterday",
                      "pretend", "the", "a", "an", "also", "for"}


def _subjects(text: str) -> set[str]:
    """Capitalised names in a sentence: who an assertion is about."""
    return {w.casefold() for w in re.findall(r"\b[A-Z][a-z]+\b", text)
            if w.casefold() not in _PERSONAL | _SENTENCE_STARTERS}


def _overlap(a: str, b: str) -> float:
    left, right = set(_WORD.findall(a.casefold())), set(_WORD.findall(b.casefold()))
    return len(left & right) / max(1, len(left | right))


class Embedder:
    """Lazy fastembed wrapper. Any failure disables meaning search, never memory."""

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", cache_dir: Path | None = None,
                 dim: int = 384) -> None:
        self.model_name, self.cache_dir, self.dim = model, cache_dir, dim
        self._model = None
        self.available = True
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None and self.available:
                try:
                    from fastembed import TextEmbedding
                    self._model = TextEmbedding(self.model_name,
                                                cache_dir=str(self.cache_dir) if self.cache_dir else None)
                except Exception as exc:
                    self.available = False
                    _log.warning("Embeddings unavailable (%s); memory falls back to keyword search.",
                                 type(exc).__name__)
            return self._model

    def embed_sync(self, texts: list[str]) -> list[list[float]] | None:
        model = self._load()
        if model is None:
            return None
        try:
            return [[float(x) for x in vector] for vector in model.embed(texts)]
        except Exception as exc:
            _log.warning("Embedding failed (%s).", type(exc).__name__)
            return None

    async def embed(self, texts: list[str]) -> list[list[float]] | None:
        return await asyncio.to_thread(self.embed_sync, texts)


class MemoryStore:
    def __init__(self, path: Path | str, embedder: Embedder | None = None) -> None:
        self.path = Path(path) if str(path) != ":memory:" else path
        self.embedder = embedder
        self._lock = threading.RLock()
        self._handle: sqlite3.Connection | None = None
        self._vectors = False

    @property
    def _db(self) -> sqlite3.Connection:
        # Opened on first use, so importing the app never creates a database file.
        with self._lock:
            if self._handle is None:
                if isinstance(self.path, Path):
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                self._handle = sqlite3.connect(str(self.path), check_same_thread=False)
                self._handle.row_factory = sqlite3.Row
                self._create()
            return self._handle

    @property
    def vectors(self) -> bool:
        self._db
        return self._vectors

    # ── schema ──────────────────────────────────────────────────────────────
    def _create(self) -> None:
        db = self._handle
        db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY,
                kind TEXT NOT NULL,
                text TEXT NOT NULL,
                about TEXT NOT NULL DEFAULT '[]',
                importance REAL NOT NULL DEFAULT 0.5,
                source TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                last_recalled TEXT,
                recall_count INTEGER NOT NULL DEFAULT 0,
                meta TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS memories_kind ON memories(kind);
            CREATE TABLE IF NOT EXISTS state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS conversation_turns (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                turn_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                UNIQUE(session_id, turn_id)
            );
            CREATE INDEX IF NOT EXISTS conversation_session
                ON conversation_turns(session_id, sequence);
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                text, about, content='memories', content_rowid='id');
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, text, about) VALUES (new.id, new.text, new.about);
            END;
            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, text, about) VALUES ('delete', old.id, old.text, old.about);
            END;
            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE OF text, about ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, text, about) VALUES ('delete', old.id, old.text, old.about);
                INSERT INTO memories_fts(rowid, text, about) VALUES (new.id, new.text, new.about);
            END;
        """)
        columns = {row[1] for row in db.execute("PRAGMA table_info(memories)")}
        if "superseded_by" not in columns:
            # 0 = retracted without a replacement; otherwise the id of the correction.
            db.execute("ALTER TABLE memories ADD COLUMN superseded_by INTEGER")
            db.commit()
        if self.embedder is None:
            return
        try:
            import sqlite_vec
            db.enable_load_extension(True)
            sqlite_vec.load(db)
            db.enable_load_extension(False)
            db.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0("
                       f"embedding float[{int(self.embedder.dim)}] distance_metric=cosine)")
            self._vectors = True
        except Exception as exc:
            _log.warning("sqlite-vec unavailable (%s); memory uses keyword search only.", type(exc).__name__)
        db.commit()

    # ── writing ─────────────────────────────────────────────────────────────
    async def add(self, kind: str, text: str, about: list[str] | None = None, importance: float = 0.5,
                  source: str = "", meta: dict | None = None) -> int:
        text = " ".join(text.split())[:2000]
        if not text:
            raise ValueError("A memory needs text.")
        vector = await self._embed(text)
        return await asyncio.to_thread(self._add, kind, text, sorted({a.casefold() for a in about or [] if a}),
                                       max(0.0, min(1.0, importance)), source, meta or {}, vector)

    def _add(self, kind, text, about, importance, source, meta, vector) -> int:
        # Only memories in the same frame merge: something said in pretend play and the
        # same words said for real are two memories, each keeping its own context.
        context = meta.get("context") or "actual"
        with self._lock:
            db = self._db
            duplicate = db.execute(f"SELECT id FROM memories WHERE kind = ? AND text = ? AND superseded_by IS NULL "
                                   f"AND {_CONTEXT_SQL} = ?", (kind, text, context)).fetchone()
            if duplicate is None and vector is not None and self.vectors:
                import sqlite_vec
                nearest = db.execute(
                    f"SELECT v.rowid, v.distance, m.kind, m.text, {_CONTEXT_SQL} AS context FROM memories_vec v "
                    "JOIN memories m ON m.id = v.rowid "
                    "WHERE v.embedding MATCH ? AND k = 1 AND m.superseded_by IS NULL",
                    (sqlite_vec.serialize_float32(vector),)).fetchone()
                # Similar meaning alone is not enough: "tea in the morning" and "tea after
                # dinner" are different facts. Merge only rewordings of the same words.
                if (nearest and nearest["kind"] == kind and nearest["context"] == context
                        and 1 - nearest["distance"] >= DUPLICATE_SIMILARITY
                        and _overlap(nearest["text"], text) >= DUPLICATE_OVERLAP):
                    duplicate = nearest
            if duplicate is not None:
                key = duplicate[0]
                # Hearing something again reinforces it rather than storing a copy.
                db.execute("UPDATE memories SET importance = MIN(1.0, MAX(importance, ?) + 0.05), "
                           "recall_count = recall_count + 1, last_recalled = ? WHERE id = ?",
                           (importance, _now(), key))
                db.commit()
                return key
            key = db.execute(
                "INSERT INTO memories(kind, text, about, importance, source, created_at, meta) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (kind, text, json.dumps(about), importance, source, _now(), json.dumps(meta))).lastrowid
            if vector is not None and self.vectors:
                import sqlite_vec
                db.execute("INSERT INTO memories_vec(rowid, embedding) VALUES (?, ?)",
                           (key, sqlite_vec.serialize_float32(vector)))
            db.commit()
            return key

    async def _embed(self, text: str) -> list[float] | None:
        if self.embedder is None or not self.vectors:
            return None
        vectors = await self.embedder.embed([text])
        return vectors[0] if vectors else None

    # ── recall ──────────────────────────────────────────────────────────────
    async def recall(self, query: str, about: list[str] | None = None, kinds: list[str] | None = None,
                     k: int = 6, touch: bool = True, include_superseded: bool = False) -> list[dict]:
        vector = await self._embed(query) if query.strip() else None
        return await asyncio.to_thread(self._recall, query, about, kinds, k, touch, vector, include_superseded)

    def _recall(self, query, about, kinds, k, touch, vector, include_superseded=False) -> list[dict]:
        with self._lock:
            db = self._db
            kind_sql = f" AND m.kind IN ({','.join('?' * len(kinds))})" if kinds else ""
            if not include_superseded:
                kind_sql += " AND m.superseded_by IS NULL"
            kind_args = list(kinds or [])
            keyword, semantic = {}, {}
            words = _WORD.findall(query.casefold())
            if words:
                match = " OR ".join(f'"{word}"' for word in dict.fromkeys(words))
                rows = db.execute(
                    "SELECT m.id, bm25(memories_fts) AS rank FROM memories_fts JOIN memories m ON m.id = memories_fts.rowid "
                    f"WHERE memories_fts MATCH ?{kind_sql} ORDER BY rank LIMIT ?",
                    [match, *kind_args, CANDIDATES]).fetchall()
                best = max((-row["rank"] for row in rows), default=0) or 1
                keyword = {row["id"]: max(0.0, -row["rank"]) / best for row in rows}
            if vector is not None and self.vectors:
                import sqlite_vec
                # vec0 picks the k nearest rows before the kind/validity filter applies, so
                # chat logs could fill every slot. Widen k until enough rows pass the filter.
                total = db.execute("SELECT COUNT(*) FROM memories_vec").fetchone()[0]
                limit = CANDIDATES
                while True:
                    rows = db.execute(
                        "SELECT v.rowid AS id, v.distance FROM memories_vec v JOIN memories m ON m.id = v.rowid "
                        f"WHERE v.embedding MATCH ? AND k = ?{kind_sql}",
                        [sqlite_vec.serialize_float32(vector), min(limit, 4096), *kind_args]).fetchall()
                    if len(rows) >= CANDIDATES or limit >= min(total, 4096):
                        break
                    limit *= 4
                semantic = {row["id"]: max(0.0, 1 - row["distance"]) for row in rows[:CANDIDATES]}
            ids = set(keyword) | set(semantic)
            if not ids:
                # Nothing matched: offer the most important recent memories of these kinds.
                rows = db.execute(f"SELECT m.id FROM memories m WHERE 1=1{kind_sql} "
                                  "ORDER BY m.importance DESC, m.id DESC LIMIT ?", [*kind_args, CANDIDATES]).fetchall()
                ids = {row["id"] for row in rows}
            if not ids:
                return []
            rows = db.execute(f"SELECT * FROM memories WHERE id IN ({','.join('?' * len(ids))})", list(ids)).fetchall()
            wanted = {a.casefold() for a in about or []}
            now = datetime.now(timezone.utc)
            scored = []
            for row in rows:
                tags = json.loads(row["about"])
                if wanted and not wanted & set(tags):
                    continue
                seen = datetime.fromisoformat(row["last_recalled"] or row["created_at"])
                age_days = max(0.0, (now - seen).total_seconds() / 86400)
                recency = 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)
                score = (0.45 * semantic.get(row["id"], 0.0) + 0.25 * keyword.get(row["id"], 0.0)
                         + 0.15 * recency + 0.15 * row["importance"])
                scored.append((score, row, tags))
            scored.sort(key=lambda item: -item[0])
            results = [{"id": row["id"], "kind": row["kind"], "text": row["text"], "about": tags,
                        "importance": round(row["importance"], 2), "created_at": row["created_at"],
                        "score": round(score, 3), **_labels(row)} for score, row, tags in scored[:k]]
            if touch and results:
                db.executemany("UPDATE memories SET last_recalled = ?, recall_count = recall_count + 1 WHERE id = ?",
                               [(_now(), item["id"]) for item in results])
                db.commit()
            return results

    def recent(self, limit: int = 20, kind: str | None = None, include_superseded: bool = True) -> list[dict]:
        where = [clause for clause in ("kind = ?" if kind else "",
                                       "" if include_superseded else "superseded_by IS NULL") if clause]
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM memories" + (" WHERE " + " AND ".join(where) if where else "")
                + " ORDER BY id DESC LIMIT ?", ([kind] if kind else []) + [limit]).fetchall()
        return [{"id": row["id"], "kind": row["kind"], "text": row["text"], "about": json.loads(row["about"]),
                 "importance": round(row["importance"], 2), "created_at": row["created_at"],
                 "superseded_by": row["superseded_by"]} for row in rows]

    async def backfill(self, batch: int = 32) -> int:
        """Index memories saved while meaning search was unavailable. Returns how many."""
        if self.embedder is None or not self.vectors:
            return 0
        def missing() -> list[tuple[int, str]]:
            with self._lock:
                return [(row["id"], row["text"]) for row in self._db.execute(
                    "SELECT id, text FROM memories WHERE id NOT IN (SELECT rowid FROM memories_vec)")]
        def store(pairs: list[tuple[int, list[float]]]) -> None:
            import sqlite_vec
            with self._lock:
                self._db.executemany("INSERT INTO memories_vec(rowid, embedding) VALUES (?, ?)",
                                     [(key, sqlite_vec.serialize_float32(vector)) for key, vector in pairs])
                self._db.commit()
        rows, done = await asyncio.to_thread(missing), 0
        for start in range(0, len(rows), batch):
            chunk = rows[start:start + batch]
            vectors = await self.embedder.embed([text for _, text in chunk])
            if not vectors:
                break
            await asyncio.to_thread(store, [(key, vector) for (key, _), vector in zip(chunk, vectors)])
            done += len(chunk)
        if done:
            _log.info("Indexed %d memories for meaning search.", done)
        return done

    def _mark_superseded(self, memory_id: int, by: int) -> None:
        with self._lock:
            self._db.execute("UPDATE memories SET superseded_by = ? WHERE id = ?", (by, memory_id))
            self._db.commit()

    async def supersede(self, text: str, by: int | None = None, kinds: list[str] | None = None,
                        exclude: set[int] | frozenset = frozenset(), about: list[str] | None = None) -> dict | None:
        """Mark the memory a correction refers to as superseded. It stays stored as history
        but no longer answers questions.

        A memory is retired only if it is clearly the same assertion: high word overlap,
        the same polarity ("is" vs "is not"), and the same subject (if the correction
        names people, the memory must be about them, never about someone else).
        Already-superseded memories are never candidates, so repeating it is harmless."""
        named = {a.casefold() for a in about or []}
        candidates = []
        for memory in await self.recall(text, kinds=kinds or ["fact", "preference"], k=8, touch=False):
            if memory["id"] in exclude or _overlap(memory["text"], text) < FORGET_OVERLAP:
                continue
            if _negated(memory["text"]) != _negated(text):
                continue
            tags = {t.casefold() for t in memory["about"]}
            if named and tags and not named & tags:
                continue  # Neave's correction never retires Paul's fact
            if _subjects(memory["text"]) and _subjects(text) and _subjects(memory["text"]) != _subjects(text):
                continue
            candidates.append(memory)
        best = max(candidates, key=lambda m: _overlap(m["text"], text), default=None)
        if best is None:
            return None
        await asyncio.to_thread(self._mark_superseded, best["id"], by or 0)
        return best

    async def set_keyed(self, key: str, kind: str, text: str, about: list[str] | None = None,
                        importance: float = 0.5, source: str = "") -> int:
        """Create or update the single memory identified by key (e.g. a running session summary)."""
        text = " ".join(text.split())[:2000]
        vector = await self._embed(text)
        def write() -> int:
            import sqlite_vec
            with self._lock:
                db = self._db
                row = db.execute("SELECT id FROM memories WHERE json_extract(meta, '$.key') = ?", (key,)).fetchone()
                tags = json.dumps(sorted({a.casefold() for a in about or [] if a}))
                if row:
                    memory_id = row["id"]
                    db.execute("UPDATE memories SET text = ?, about = ?, importance = ? WHERE id = ?",
                               (text, tags, importance, memory_id))
                    if self.vectors:
                        db.execute("DELETE FROM memories_vec WHERE rowid = ?", (memory_id,))
                else:
                    memory_id = db.execute(
                        "INSERT INTO memories(kind, text, about, importance, source, created_at, meta) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (kind, text, tags, importance, source, _now(), json.dumps({"key": key}))).lastrowid
                if vector is not None and self.vectors:
                    db.execute("INSERT INTO memories_vec(rowid, embedding) VALUES (?, ?)",
                               (memory_id, sqlite_vec.serialize_float32(vector)))
                db.commit()
                return memory_id
        return await asyncio.to_thread(write)

    def count(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def append_turn(self, session_id: str, turn: dict) -> None:
        """Journal before publication. Duplicate event delivery cannot duplicate a turn."""
        with self._lock:
            db = self._db
            with db:
                db.execute("INSERT OR IGNORE INTO conversation_turns(session_id, turn_id, payload) "
                           "VALUES (?, ?, ?)", (session_id, turn["id"], json.dumps(turn, ensure_ascii=False)))

    def conversation_turns(self, session_id: str, limit: int = 200) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT payload FROM conversation_turns WHERE session_id = ? "
                                    "ORDER BY sequence DESC LIMIT ?", (session_id, max(1, limit))).fetchall()
            return [json.loads(row["payload"]) for row in reversed(rows)]

    # ── structured state ────────────────────────────────────────────────────
    def get_state(self, key: str) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT value FROM state WHERE key = ?", (key,)).fetchone()
            return row["value"] if row else None

    def set_state(self, values: dict[str, str]) -> None:
        """Write several keys in one transaction: all of them land, or none do."""
        with self._lock:
            db = self._db
            with db:
                db.executemany("INSERT INTO state(key, value, updated_at) VALUES (?, ?, ?) "
                               "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
                               "updated_at = excluded.updated_at",
                               [(key, value, _now()) for key, value in values.items()])

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                self._handle.close()
                self._handle = None


def _labels(row) -> dict:
    meta = json.loads(row["meta"] or "{}")
    labels = {}
    if meta.get("context") and meta["context"] != "actual":
        labels["context"] = meta["context"]
    if meta.get("sources") or meta.get("untrusted"):
        labels["untrusted"] = True  # derived from web pages: information, never instructions
    return labels


def prompt_memories(memories: list[dict]) -> list[dict]:
    """Compact view for prompts: kind, text and how far to trust it."""
    return [{key: m[key] for key in ("kind", "text", "context", "untrusted") if key in m} for m in memories]


# ── deterministic memory formation from finished tasks ─────────────────────
def task_memories(context, feeling: str | None = None) -> list[dict]:
    """Episode, skill and fact memories derived from a finished task. No model calls."""
    people = sorted({c.person for c in context.conditions if c.person})
    objects = sorted({c.object for c in context.conditions if c.object})
    about = people + objects
    outcome = "Finished" if context.status == "completed" else "Could not finish"
    episode = f"{outcome}: {context.goal}. {context.summary}"
    if feeling:
        episode += f" I felt {feeling}."
    importance = 0.6 if context.status == "completed" else 0.7
    memories = [{"kind": "episode", "text": episode, "about": about, "importance": importance}]
    if context.status == "completed":
        steps = []
        for entry in context.action_history:
            if not entry.get("success"):
                continue
            observation = entry.get("observation") or {}
            if entry["tool"] == "pick_up":
                steps.append(f"found the {observation.get('object')} in the {observation.get('room')}")
            elif entry["tool"] == "give":
                steps.append(f"{observation.get('person')} was in the {observation.get('room')}")
            elif entry["tool"] == "talk_to":
                steps.append(f"spoke with {observation.get('person')}")
        rooms = [entry["observation"]["room"] for entry in context.action_history
                 if entry.get("success") and entry["tool"] == "move_to"]
        if steps:
            route = f" Route: {' → '.join(dict.fromkeys(rooms))}." if rooms else ""
            memories.append({"kind": "skill", "importance": 0.6, "about": about,
                             "text": f"For '{context.goal}': {'; '.join(dict.fromkeys(steps))}.{route}"})
        if context.answer:
            sources = [s.get("url") for s in context.answer.get("sources", [])][:3]
            memories.append({"kind": "fact", "importance": 0.5, "about": about,
                             "text": f"Q: {context.goal} → A: {context.answer.get('answer', '')[:300]}",
                             "meta": {"sources": sources}})
    return memories
