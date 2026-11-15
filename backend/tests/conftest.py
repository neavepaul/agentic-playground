"""Keep the test suite away from Kip's real memory and local configuration.

Settings read these environment variables, so every Settings() built by a test (and the
module-level app in app.main) writes to a throwaway directory instead of backend/memory/.
A developer's backend/.env (model, timeouts, delegation, API keys) is never read.
"""
import os
import tempfile
from pathlib import Path

_SANDBOX = Path(tempfile.mkdtemp(prefix="kip-tests-"))
os.environ["MEMORY_FILE"] = str(_SANDBOX / "persistent.json")
os.environ["GRAPH_FILE"] = str(_SANDBOX / "beliefs.json")
os.environ["MEMORY_DB_FILE"] = str(_SANDBOX / "kip.db")
os.environ["EMBEDDINGS_ENABLED"] = "false"

from app.config import Settings  # noqa: E402  (after the environment is set)

Settings.model_config["env_file"] = None


import httpx  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _no_results_feeds(monkeypatch):
    """No test reaches the live F1 or cricket feeds: a lookup falls back as when they are down.
    A test that wants a feed patches app.capabilities.feeds.fetch itself."""
    from app.capabilities import feeds

    async def offline(url):
        raise httpx.ConnectError("results feeds are offline in tests")
    monkeypatch.setattr(feeds, "fetch", offline)
    feeds._CACHE.clear()  # no test sees another's cached scoreboard
