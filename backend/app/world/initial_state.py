from pathlib import Path

from .models import World

DEFAULT_WORLD = Path(__file__).resolve().parents[2] / "worlds" / "house.json"


def initial_world(path: str | Path | None = None) -> World:
    """Read and validate a fresh scenario; never share mutable initial state."""
    return World.model_validate_json(Path(path or DEFAULT_WORLD).read_text(encoding="utf-8"))
