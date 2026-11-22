"""Profile-scoped Web Push for messages delivered while a chat is unattended."""
import asyncio
import base64
import hashlib
import logging
import os
import time
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import BaseModel, Field
from app.people import SESSION_SECONDS


class Subscription(BaseModel):
    endpoint: str = Field(max_length=2048)
    keys: dict[str, str]

    def valid(self):
        try:
            url = urlsplit(self.endpoint)
            host = url.hostname or ""
            allowed = host in {"fcm.googleapis.com", "updates.push.services.mozilla.com",
                               "web.push.apple.com"} or host.endswith(".push.apple.com")
            return (url.scheme == "https" and allowed and url.netloc == host
                    and not url.fragment and bool(url.path.strip("/"))
                    and all(len(self.keys.get(k, "")) in range(1, 150) for k in ("auth", "p256dh")))
        except ValueError:
            return False


class PhonePush:
    def __init__(self, store):
        self.store = store
        self.key = os.environ.get("KIP_PUSH_KEY", "")
        self.subject = os.environ.get("KIP_PUSH_SUBJECT", "")
        self.public_key = ""
        if self.key and self.subject:
            from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat, load_pem_private_key
            key = load_pem_private_key(Path(self.key).read_bytes(), password=None)
            point = key.public_key().public_bytes(Encoding.X962, PublicFormat.UncompressedPoint)
            self.public_key = base64.urlsafe_b64encode(point).decode().rstrip("=")
        self.views = {}
        self.jobs = set()
        with store._lock, store._db:
            store._db.execute("""CREATE TABLE IF NOT EXISTS phone_push (
                endpoint TEXT PRIMARY KEY, profile TEXT NOT NULL, session TEXT NOT NULL, body TEXT NOT NULL
            )""")

    def subscribe(self, profile, token, subscription):
        session = hashlib.sha256(token.encode()).hexdigest()
        with self.store._lock, self.store._db:
            self.store._db.execute("INSERT OR REPLACE INTO phone_push VALUES (?, ?, ?, ?)",
                                   (subscription.endpoint, profile, session, subscription.model_dump_json()))

    def revoke_session(self, token):
        if token:
            session = hashlib.sha256(token.encode()).hexdigest()
            with self.store._lock, self.store._db:
                self.store._db.execute("DELETE FROM phone_push WHERE session = ?", (session,))
            for key in list(self.views):
                if key[0] == session:
                    del self.views[key]

    def view(self, token, tab, profile, conversation):
        now = time.monotonic()
        self.views = {k: v for k, v in self.views.items() if v[2] > now}
        key = (hashlib.sha256(token.encode()).hexdigest(), tab)
        if conversation:
            self.views[key] = (profile, conversation, now + 25)
        else:
            self.views.pop(key, None)

    def notify(self, profile, conversation):
        if not self.public_key or not profile or len(self.jobs) >= 64:
            return
        task = asyncio.create_task(self.send(profile, conversation))
        self.jobs.add(task)
        task.add_done_callback(self.jobs.discard)

    async def close(self):
        for task in self.jobs:
            task.cancel()
        await asyncio.gather(*self.jobs, return_exceptions=True)

    async def send(self, profile, conversation):
        from pywebpush import webpush, WebPushException
        import json

        if any(p == profile and c == conversation and until > time.monotonic()
               for p, c, until in self.views.values()):
            return
        with self.store._lock:
            rows = self.store._db.execute("""
                SELECT phone_push.* FROM phone_push JOIN signins
                ON token_hash = session AND profile_id = profile
                WHERE profile = ? AND seen_at >= ?
                """, (profile, time.time() - SESSION_SECONDS)).fetchall()
        for row in rows:
            try:
                await asyncio.to_thread(webpush, json.loads(row["body"]),
                                        "{}",
                                        vapid_private_key=self.key,
                                        vapid_claims={"sub": self.subject},
                                        ttl=300, timeout=8)
            except WebPushException as exc:
                if exc.response is not None and exc.response.status_code in (404, 410):
                    with self.store._lock, self.store._db:
                        self.store._db.execute("DELETE FROM phone_push WHERE endpoint = ? AND session = ?",
                                               (row["endpoint"], row["session"]))
                else:
                    logging.getLogger(__name__).warning("Phone push was rejected")
            except Exception:
                logging.getLogger(__name__).warning("Phone push could not be sent")
