"""Crash-safe writes for Kip's durable JSON stores.

A save writes a temporary file in the same directory, flushes and fsyncs it, keeps the
previous version as <name>.bak, then atomically replaces the original. An interrupted
write therefore leaves either the old file or the new one, never a truncated mix.
"""
import os
import tempfile
from pathlib import Path


class UnreadableStoreError(RuntimeError):
    """A durable store exists but cannot be read. It must never be overwritten."""


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            backup = path.with_name(path.name + ".bak")
            backup.write_bytes(path.read_bytes())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
