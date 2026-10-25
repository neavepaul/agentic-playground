"""Searchable long-term memory: one flexible SQLite log with keyword and meaning search.

Every memory is one row (episode, fact, preference, skill, conversation, ...) with free
text, loose tags and an importance score. FTS5 finds exact words such as names;
sqlite-vec finds memories with similar meaning. Recall blends both with recency and
importance. Everything is advisory: the robot still verifies the world by observation.
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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        with self._lock:
            db = self._db
            duplicate = db.execute("SELECT id FROM memories WHERE kind = ? AND text = ?", (kind, text)).fetchone()
            if duplicate is None and vector is not None and self.vectors:
                import sqlite_vec
                nearest = db.execute(
                    "SELECT v.rowid, v.distance, m.kind, m.text FROM memories_vec v JOIN memories m ON m.id = v.rowid "
                    "WHERE v.embedding MATCH ? AND k = 1", (sqlite_vec.serialize_float32(vector),)).fetchone()
                # Similar meaning alone is not enough: "tea in the morning" and "tea after
                # dinner" are different facts. Merge only rewordings of the same words.
                if (nearest and nearest["kind"] == kind and 1 - nearest["distance"] >= DUPLICATE_SIMILARITY
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
                     k: int = 6, touch: bool = True) -> list[dict]:
        vector = await self._embed(query) if query.strip() else None
        return await asyncio.to_thread(self._recall, query, about, kinds, k, touch, vector)

    def _recall(self, query, about, kinds, k, touch, vector) -> list[dict]:
        with self._lock:
            db = self._db
            kind_sql = f" AND m.kind IN ({','.join('?' * len(kinds))})" if kinds else ""
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
                rows = db.execute(
                    "SELECT v.rowid AS id, v.distance FROM memories_vec v JOIN memories m ON m.id = v.rowid "
                    f"WHERE v.embedding MATCH ? AND k = ?{kind_sql}",
                    [sqlite_vec.serialize_float32(vector), CANDIDATES, *kind_args]).fetchall()
                semantic = {row["id"]: max(0.0, 1 - row["distance"]) for row in rows}
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
                        "score": round(score, 3)} for score, row, tags in scored[:k]]
            if touch and results:
                db.executemany("UPDATE memories SET last_recalled = ?, recall_count = recall_count + 1 WHERE id = ?",
                               [(_now(), item["id"]) for item in results])
                db.commit()
            return results

    def recent(self, limit: int = 20, kind: str | None = None) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM memories" + (" WHERE kind = ?" if kind else "") + " ORDER BY id DESC LIMIT ?",
                ([kind] if kind else []) + [limit]).fetchall()
        return [{"id": row["id"], "kind": row["kind"], "text": row["text"], "about": json.loads(row["about"]),
                 "importance": round(row["importance"], 2), "created_at": row["created_at"]} for row in rows]

    def count(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                self._handle.close()
                self._handle = None


def prompt_memories(memories: list[dict]) -> list[dict]:
    """Compact view for prompts: kind and text only."""
    return [{"kind": m["kind"], "text": m["text"]} for m in memories]


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
