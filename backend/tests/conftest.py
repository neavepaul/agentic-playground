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
