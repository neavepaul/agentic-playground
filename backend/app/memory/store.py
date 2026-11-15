"""Persistent memory that survives across tasks and restarts.

Evidence gathered during a task is task-scoped. This store holds what lasts: Kip's
personality, his recent errands and his own track record. It is advisory only.

It lives as one document in the state table of kip.db. persistent.json is
only read once, to import it, and is then left untouched as a backup.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import logging

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from app.memory.durable import UnreadableStoreError, atomic_write_text

from app.persona.state import PersonaState


STATE_KEY = "persistent"
IMPORT_KEY = "imported_json"


class SelfModel(BaseModel):
    """Kip's record of his own performance across tasks."""
    tool_failure_counts: dict[str, int] = Field(default_factory=dict)


class TaskRecord(BaseModel):
    task_id: str
    goal: str
    outcome: str
    timestamp: str
    summary: str


class PersistentMemory(BaseModel):
    version: int = 1
    recent_tasks: list[TaskRecord] = Field(default_factory=list)
    self_model: SelfModel = Field(default_factory=SelfModel)
    persona: PersonaState = Field(default_factory=PersonaState)
    # Monotonic count of finished tasks; recent_tasks is capped, this is not.
    tasks_recorded: int = 0
    _load_error: str = PrivateAttr(default="")
    _target: object = PrivateAttr(default=None)  # where this memory saves to

    @property
    def task_total(self) -> int:
        """Finished tasks ever recorded (never capped like recent_tasks)."""
        return max(self.tasks_recorded, len(self.recent_tasks))

    @model_validator(mode="after")
    def _count_existing_history(self):
        # Older files lack the counter: start from the history they do contain.
        self.tasks_recorded = max(self.tasks_recorded, len(self.recent_tasks))
        return self

    @property
    def load_error(self) -> str:
        return self._load_error

    @classmethod
    def load(cls, source) -> PersistentMemory:
        """Load from a JSON file (a Path) or from a store's state table.

        Missing data starts a fresh memory. Unreadable data is NOT treated as empty: the
        app runs with a blank in-memory store, the error is reported, and saving is
        refused so the original (and Kip's real memories) stay untouched until fixed.
        """
        memory = cls()
        where = Path(source).name if isinstance(source, (str, Path)) else f"kip.db state '{STATE_KEY}'"
        try:
            if isinstance(source, (str, Path)):
                path = Path(source)
                text = path.read_text(encoding="utf-8") if path.exists() else None
            else:
                text = source.get_state(STATE_KEY)
        except (OSError, UnicodeDecodeError) as exc:
            text, read_error = None, exc
        else:
            read_error = None
        if read_error is not None or text is not None:
            try:
                if read_error is not None:
                    raise read_error
                memory = cls.model_validate_json(text)
            except Exception as exc:
                logging.getLogger("agentic_friend.memory").error(
                    "Persistent memory in %s could not be read (%s). It will not be overwritten; "
                    "restore it from a backup in memory/backups/.", where, type(exc).__name__)
                memory._load_error = f"{where} could not be read ({type(exc).__name__}); saving is disabled."
        memory._target = source
        return memory

    def save(self, target=None) -> bool:
        """Save to the given file, or to wherever this memory was loaded from.

        A memory that was never loaded (a bare PersistentMemory() in tests) has nowhere to
        go and is not saved. The app always loads it, so it always has a target.
        """
        if self._load_error:
            raise UnreadableStoreError(self._load_error)
        target = target if target is not None else self._target
        if target is None:
            return False
        text = self.model_dump_json(indent=2)
        if isinstance(target, (str, Path)):
            atomic_write_text(Path(target), text)
        else:
            target.set_state({STATE_KEY: text})
        return True

    def update_from_task(self, context) -> None:
        """Record a finished task and any tool failures in it, whatever the outcome."""
        for entry in context.action_history:
            if not entry["success"] and not entry.get("refused"):
                self.self_model.tool_failure_counts[entry["tool"]] = (
                    self.self_model.tool_failure_counts.get(entry["tool"], 0) + 1)
        self.recent_tasks.append(TaskRecord(
            task_id=context.id,
            goal=context.goal,
            outcome=context.status,
            timestamp=datetime.now(timezone.utc).isoformat(),
            summary=context.summary,
        ))
        self.recent_tasks[:] = self.recent_tasks[-20:]
        self.tasks_recorded += 1

    def self_model_prompt(self) -> dict:
        """Compact self-model payload for the Coordinator. Empty when no task history exists."""
        if not self.recent_tasks:
            return {}
        by_outcome: dict[str, int] = {}
        for t in self.recent_tasks:
            by_outcome[t.outcome] = by_outcome.get(t.outcome, 0) + 1
        return {
            "note": "Kip's own performance history — use to plan more honestly.",
            "task_summary": {
                "total": len(self.recent_tasks),
                "completed": by_outcome.get("completed", 0),
                "failed": by_outcome.get("failed", 0),
                "cancelled": by_outcome.get("cancelled", 0),
            },
            "tool_failures": dict(
                sorted(self.self_model.tool_failure_counts.items(), key=lambda x: -x[1])[:5]
            ),
        }

    def prompt(self) -> dict:
        """Compact payload for the LLM. Advisory only."""
        return {
            "note": "Recent errands and how they went.",
            "recent_tasks": [{"goal": t.goal, "outcome": t.outcome, "summary": t.summary}
                             for t in self.recent_tasks[-5:]],
        }


def open_persistent(store, memory_file: Path, graph_file: Path | None = None,
                    backups_dir: Path | None = None) -> PersistentMemory:
    """Kip's persistent memory from kip.db, importing the old JSON files the first time.

    The import happens once, only when kip.db has no persistent state yet and the JSON
    file is readable. The originals are copied to backups_dir and never modified; the old
    belief graph is kept verbatim in the state table as history (beliefs are now computed).
    """
    if store.get_state(STATE_KEY) is not None or not Path(memory_file).exists():
        return PersistentMemory.load(store)
    imported = PersistentMemory.load(memory_file)
    if imported.load_error:
        # Never import, or start fresh over, unreadable memories: run read-only and say so.
        return imported
    sources = [Path(memory_file)] + ([Path(graph_file)] if graph_file and Path(graph_file).exists() else [])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record: dict = {"at": stamp, "files": {}}
    if backups_dir is not None:
        folder = Path(backups_dir) / f"{stamp}-import"
        folder.mkdir(parents=True, exist_ok=True)
        for source in sources:
            shutil.copy2(source, folder / source.name)
            if (folder / source.name).read_bytes() != source.read_bytes():
                raise OSError(f"Backup of {source} did not verify; import aborted.")
        record["backup"] = str(folder)
    for source in sources:
        record["files"][source.name] = {"path": str(source),
                                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    values = {STATE_KEY: imported.model_dump_json(indent=2), IMPORT_KEY: json.dumps(record)}
    for source in sources[1:]:
        # Kept only as history: an odd byte in the old graph must not block the import.
        values["imported:" + source.name] = source.read_bytes().decode("utf-8", errors="replace")
    store.set_state(values)
    logging.getLogger("agentic_friend.memory").info(
        "Imported %s into kip.db (backup: %s). The JSON files are no longer read or written.",
        ", ".join(source.name for source in sources), record.get("backup", "none"))
    return PersistentMemory.load(store)
