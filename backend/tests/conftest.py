"""Keep the test suite away from Kip's real memory.

Settings read these environment variables, so every Settings() built by a test (and the
module-level app in app.main) writes to a throwaway directory instead of backend/memory/.
"""
import os
import tempfile
from pathlib import Path

_SANDBOX = Path(tempfile.mkdtemp(prefix="kip-tests-"))
os.environ["MEMORY_FILE"] = str(_SANDBOX / "persistent.json")
os.environ["GRAPH_FILE"] = str(_SANDBOX / "beliefs.json")
os.environ["MEMORY_DB_FILE"] = str(_SANDBOX / "kip.db")
os.environ["EMBEDDINGS_ENABLED"] = "false"
