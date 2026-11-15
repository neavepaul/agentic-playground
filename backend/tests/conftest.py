"""Keep the test suite away from Kip's real memory and local configuration.

Settings read these environment variables, so every Settings() built by a test (and the
module-level app in app.main) writes to a throwaway directory instead of backend/memory/.
A developer's backend/.env (model, timeouts, delegation, API keys) is never read.
"""
import os
import tempfile
from pathlib import Path

_SANDBOX = Path(tempfile.mkdtemp(prefix="kip-tests-"))
os.environ["MEMORY_DB_FILE"] = str(_SANDBOX / "kip.db")
os.environ["EMBEDDINGS_ENABLED"] = "false"
# No built page unless a test makes one: the apps under test serve only /api and /ws.
os.environ["FRONTEND_DIST"] = str(_SANDBOX / "no-dist")
# Scripted test models stand in for any model, Groq included, whose action lines are acted on: the
# offer, consent and errand tests need that. Tests of LOCAL_CHAT_ERRANDS=false set it themselves.
os.environ["LOCAL_CHAT_ERRANDS"] = "true"
# FastAPI's TestClient calls itself "testserver"; the real server answers only to loopback names.
os.environ["ALLOWED_HOSTS"] = "127.0.0.1,localhost,testserver"

from app.config import Settings  # noqa: E402  (after the environment is set)

Settings.model_config["env_file"] = None


import httpx  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _no_results_feeds(monkeypatch):
    """No test reaches the live F1 or cricket feeds: a lookup falls back as when they are down.
    A test that wants a feed patches app.capabilities.feeds.fetch itself."""
    from app.capabilities import feeds

    async def offline(*args, **kwargs):
        raise httpx.ConnectError("results feeds are offline in tests")
    monkeypatch.setattr(feeds, "fetch", offline)
    monkeypatch.setattr(feeds, "post", offline)
    feeds._CACHE.clear()  # no test sees another's cached scoreboard
    feeds._GROWW_TOKEN.clear()
