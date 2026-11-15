"""Errors shared by Kip's durable stores (kip.db)."""


class UnreadableStoreError(RuntimeError):
    """A durable store exists but cannot be read. It must never be overwritten."""
