"""The people who talk to Kip: profiles with a 4-digit PIN, the devices they signed in on, and
their conversations. Everything lives in kip.db beside Kip's memories.

Anyone who reaches Kip can make a profile; the first one made is the owner. Only the owner (and
whoever the owner allows) may ask about the owner's Groww portfolio, and only the owner (and whoever
the owner allows) may change their own PIN. A PIN is short, so wrong
guesses lock a profile for longer and longer; signed-in devices hold a random token whose hash
alone is stored.

A profile can also be made without a PIN (python -m app.people add NAME): its person chooses one
the first time they sign in. Until then anyone who picks that name can claim it.

    python -m app.people add NAME [PIN]   make a profile (the first one made is the owner)
    python -m app.people list             who has a profile, and who still has to choose a PIN
"""
from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

COOKIE = "kip_session"
SESSION_SECONDS = 180 * 86400
MAX_PROFILES = 20
FREE_TRIES = 5           # wrong PINs before a profile locks
LOCK_SECONDS = 30        # first lock; doubles with each further wrong PIN
MAX_LOCK_SECONDS = 3600
_PIN = re.compile(r"\d{4}")
_NAME = re.compile(r"[^\W_](?:[\w' .-]*[^\W_])?")
_SCRYPT = {"n": 2 ** 14, "r": 8, "p": 1, "dklen": 32}


class SignInError(ValueError):
    """Shown to the person as written."""


class Locked(SignInError):
    def __init__(self, seconds: int) -> None:
        minutes = (seconds + 59) // 60
        wait = f"{seconds} seconds" if seconds < 90 else f"{minutes} minutes"
        super().__init__(f"Too many wrong PINs. Try again in {wait}.")
        self.seconds = seconds


@dataclass(frozen=True)
class Profile:
    id: str
    name: str
    owner: bool
    portfolio: bool
    needs_pin: bool = False  # made without one: they choose it the first time they sign in
    pin_change: bool = False  # may change their own PIN (the owner always may)

    @property
    def speaker(self) -> str:
        """The name memories are filed under (as ChatSession files them)."""
        return "_".join(self.name.casefold().split())

    def public(self) -> dict:
        return {"id": self.id, "name": self.name, "owner": self.owner, "portfolio": self.portfolio,
                "needs_pin": self.needs_pin, "pin_change": self.pin_change}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_pin(pin: str, salt: bytes) -> bytes:
    return hashlib.scrypt(pin.encode(), salt=salt, **_SCRYPT)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def clean_name(name: str) -> str:
    name = " ".join(str(name or "").split())
    if not 1 <= len(name) <= 30 or not _NAME.fullmatch(name):
        raise SignInError("Use a name of up to 30 letters, numbers or spaces.")
    return name


def check_pin(pin: str) -> str:
    pin = str(pin or "")
    if not _PIN.fullmatch(pin):
        raise SignInError("The PIN is 4 digits.")
    return pin


class People:
    def __init__(self, store, legacy_session: str | None = None) -> None:
        # The memory store owns the kip.db connection and its lock; these tables sit beside its own.
        self._store = store
        self._legacy = legacy_session
        with store._lock:
            store._db.executescript("""
                CREATE TABLE IF NOT EXISTS profiles (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    pin_hash BLOB NOT NULL,
                    salt BLOB NOT NULL,
                    owner INTEGER NOT NULL DEFAULT 0,
                    portfolio INTEGER NOT NULL DEFAULT 0,
                    failures INTEGER NOT NULL DEFAULT 0,
                    locked_until REAL NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS signins (
                    token_hash TEXT PRIMARY KEY,
                    profile_id TEXT NOT NULL,
                    seen_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    profile_id TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS conversations_profile ON conversations(profile_id, updated_at);
            """)
            # Added after the first profiles were made: older kip.db files gain it here.
            columns = {row[1] for row in store._db.execute("PRAGMA table_info(profiles)")}
            if "pin_change" not in columns:
                with store._db:
                    store._db.execute("ALTER TABLE profiles ADD COLUMN pin_change INTEGER NOT NULL DEFAULT 0")

    def _db(self):
        return self._store._db

    @staticmethod
    def _profile(row) -> Profile:
        return Profile(row["id"], row["name"], bool(row["owner"]), bool(row["owner"] or row["portfolio"]),
                       not row["pin_hash"], bool(row["owner"] or row["pin_change"]))

    # ── profiles ────────────────────────────────────────────────────────────
    def profiles(self) -> list[Profile]:
        with self._store._lock:
            rows = self._db().execute("SELECT * FROM profiles ORDER BY owner DESC, name COLLATE NOCASE").fetchall()
        return [self._profile(row) for row in rows]

    def get(self, profile_id: str | None) -> Profile | None:
        if not profile_id:
            return None
        with self._store._lock:
            row = self._db().execute("SELECT * FROM profiles WHERE id = ?", (profile_id,)).fetchone()
        return self._profile(row) if row else None

    def owner(self) -> Profile | None:
        with self._store._lock:
            row = self._db().execute("SELECT * FROM profiles WHERE owner = 1").fetchone()
        return self._profile(row) if row else None

    def create(self, name: str, pin: str | None) -> Profile:
        """A new profile. Without a PIN (None), its person chooses one on first sign-in."""
        name = clean_name(name)
        salt = secrets.token_bytes(16)
        digest = _hash_pin(check_pin(pin), salt) if pin is not None else b""
        speaker = "_".join(name.casefold().split())
        with self._store._lock:
            db = self._db()
            with db:
                existing = db.execute("SELECT name FROM profiles").fetchall()
                if len(existing) >= MAX_PROFILES:
                    raise SignInError(f"Kip already has {MAX_PROFILES} profiles.")
                if any("_".join(row["name"].casefold().split()) == speaker for row in existing):
                    raise SignInError("That name is taken. Pick another, or sign in to it.")
                first = not existing
                profile_id = str(uuid4())
                db.execute("INSERT INTO profiles(id, name, pin_hash, salt, owner, portfolio, created_at) "
                           "VALUES (?, ?, ?, ?, ?, ?, ?)", (profile_id, name, digest, salt, int(first), int(first), _now()))
                if first and self._legacy:
                    # The chat from before there were profiles is the owner's.
                    had = db.execute("SELECT 1 FROM conversation_turns WHERE session_id = ? LIMIT 1",
                                     (self._legacy,)).fetchone()
                    if had:
                        db.execute("INSERT OR IGNORE INTO conversations(id, profile_id, title, created_at, updated_at) "
                                   "VALUES (?, ?, ?, ?, ?)", (self._legacy, profile_id, "Earlier chats", _now(), _now()))
        return self.get(profile_id)

    def verify(self, profile_id: str, pin: str) -> Profile:
        """The profile, when the PIN is right. Wrong PINs count towards a lock."""
        with self._store._lock:
            row = self._db().execute("SELECT * FROM profiles WHERE id = ?", (profile_id,)).fetchone()
        if row is None:
            raise SignInError("That profile no longer exists.")
        if not row["pin_hash"]:
            raise SignInError("This profile has no PIN yet. Choose one first.")
        wait = int(row["locked_until"] - time.time())
        if wait > 0:
            raise Locked(wait)
        right = _PIN.fullmatch(str(pin or "")) and hmac.compare_digest(_hash_pin(pin, row["salt"]), row["pin_hash"])
        with self._store._lock:
            db = self._db()
            with db:
                if right:
                    db.execute("UPDATE profiles SET failures = 0, locked_until = 0 WHERE id = ?", (profile_id,))
                    return self._profile(row)
                failures = row["failures"] + 1
                lock = 0.0
                if failures >= FREE_TRIES:
                    lock = time.time() + min(MAX_LOCK_SECONDS, LOCK_SECONDS * 2 ** (failures - FREE_TRIES))
                db.execute("UPDATE profiles SET failures = ?, locked_until = ? WHERE id = ?",
                           (failures, lock, profile_id))
        if lock:
            raise Locked(int(lock - time.time()))
        left = FREE_TRIES - failures
        raise SignInError(f"Wrong PIN. {left} {'try' if left == 1 else 'tries'} left before it locks.")

    def choose_pin(self, profile_id: str, pin: str) -> Profile:
        """The first PIN of a profile made without one. Once set, only the PIN signs in."""
        salt = secrets.token_bytes(16)
        digest = _hash_pin(check_pin(pin), salt)
        with self._store._lock:
            db = self._db()
            with db:
                # Only while it is still empty: two people racing for one profile cannot both win.
                changed = db.execute("UPDATE profiles SET pin_hash = ?, salt = ? WHERE id = ? AND length(pin_hash) = 0",
                                     (digest, salt, profile_id)).rowcount
        profile = self.get(profile_id)
        if profile is None:
            raise SignInError("That profile no longer exists.")
        if not changed:
            raise SignInError("This profile already has a PIN. Sign in with it.")
        return profile

    def change_pin(self, profile_id: str, current: str, new: str) -> Profile:
        """Their own PIN, changed: the current one first (wrong guesses count towards the lock).
        Every other device signed in to this profile is signed out."""
        check_pin(new)
        profile = self.verify(profile_id, current)
        if not profile.pin_change:
            raise SignInError("The owner hasn't allowed this profile to change its PIN.")
        salt = secrets.token_bytes(16)
        digest = _hash_pin(new, salt)
        with self._store._lock:
            db = self._db()
            with db:
                db.execute("UPDATE profiles SET pin_hash = ?, salt = ? WHERE id = ?", (digest, salt, profile_id))
                db.execute("DELETE FROM signins WHERE profile_id = ?", (profile_id,))
        return profile

    def set_access(self, profile_id: str, portfolio: bool | None = None, pin_change: bool | None = None) -> Profile:
        """What the owner allows someone: the Groww portfolio, changing their own PIN. The owner's
        own profile always has both."""
        with self._store._lock:
            db = self._db()
            with db:
                for column, allowed in (("portfolio", portfolio), ("pin_change", pin_change)):
                    if allowed is not None:
                        db.execute(f"UPDATE profiles SET {column} = ? WHERE id = ? AND owner = 0",  # a fixed column name  # nosec B608
                                   (int(allowed), profile_id))
        profile = self.get(profile_id)
        if profile is None:
            raise SignInError("That profile no longer exists.")
        return profile

    # ── signed-in devices ───────────────────────────────────────────────────
    def sign_in(self, profile: Profile) -> str:
        token = secrets.token_urlsafe(32)
        with self._store._lock:
            db = self._db()
            with db:
                db.execute("DELETE FROM signins WHERE seen_at < ?", (time.time() - SESSION_SECONDS,))
                db.execute("INSERT INTO signins(token_hash, profile_id, seen_at) VALUES (?, ?, ?)",
                           (_token_hash(token), profile.id, time.time()))
        return token

    def signed_in(self, token: str | None) -> Profile | None:
        if not token:
            return None
        key = _token_hash(token)
        with self._store._lock:
            db = self._db()
            row = db.execute("SELECT profile_id, seen_at FROM signins WHERE token_hash = ?", (key,)).fetchone()
            if row is None or row["seen_at"] < time.time() - SESSION_SECONDS:
                return None
            if row["seen_at"] < time.time() - 3600:
                with db:
                    db.execute("UPDATE signins SET seen_at = ? WHERE token_hash = ?", (time.time(), key))
        return self.get(row["profile_id"])

    def sign_out(self, token: str | None) -> None:
        if token:
            with self._store._lock:
                db = self._db()
                with db:
                    db.execute("DELETE FROM signins WHERE token_hash = ?", (_token_hash(token),))

    # ── conversations ───────────────────────────────────────────────────────
    def new_conversation(self, profile: Profile, first_message: str, conversation_id: str | None = None) -> str:
        conversation_id = conversation_id or str(uuid4())
        title = " ".join(first_message.split())
        title = title if len(title) <= 60 else title[:59].rstrip() + "…"
        with self._store._lock:
            db = self._db()
            with db:
                db.execute("INSERT INTO conversations(id, profile_id, title, created_at, updated_at) "
                           "VALUES (?, ?, ?, ?, ?)", (conversation_id, profile.id, title, _now(), _now()))
        return conversation_id

    def conversations(self, profile: Profile, limit: int = 100) -> list[dict]:
        with self._store._lock:
            rows = self._db().execute("SELECT id, title, created_at, updated_at FROM conversations "
                                      "WHERE profile_id = ? ORDER BY updated_at DESC LIMIT ?",
                                      (profile.id, limit)).fetchall()
        return [dict(row) for row in rows]

    def conversation_owner(self, conversation_id: str | None) -> str | None:
        if not conversation_id:
            return None
        with self._store._lock:
            row = self._db().execute("SELECT profile_id FROM conversations WHERE id = ?",
                                     (conversation_id,)).fetchone()
        return row["profile_id"] if row else None

    def touch(self, conversation_id: str) -> None:
        with self._store._lock:
            db = self._db()
            with db:
                db.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (_now(), conversation_id))

    def delete_conversation(self, conversation_id: str) -> None:
        with self._store._lock:
            db = self._db()
            with db:
                db.execute("DELETE FROM conversation_turns WHERE session_id = ?", (conversation_id,))
                db.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))


def main(argv: list[str] | None = None) -> None:
    import sys
    from app.config import get_settings
    from app.memory.recall import MemoryStore
    args = list(sys.argv[1:] if argv is None else argv)
    settings = get_settings()
    store = MemoryStore(settings.memory_db_file)
    people = People(store, legacy_session=settings.chat_session_id)
    try:
        if args[:1] == ["add"] and len(args) in (2, 3):
            profile = people.create(args[1], args[2] if len(args) == 3 else None)
            print(f"Made {profile.name}" + (" (the owner)" if profile.owner else "")
                  + (": they choose a PIN the first time they sign in." if profile.needs_pin else "."))
        elif args == ["list"]:
            for profile in people.profiles():
                notes = [n for n, on in (("owner", profile.owner), ("sees the portfolio", profile.portfolio),
                                         ("may change their PIN", profile.pin_change and not profile.owner),
                                         ("no PIN yet", profile.needs_pin)) if on]
                print(profile.name + (f" ({', '.join(notes)})" if notes else ""))
        else:
            raise SystemExit(__doc__.split("\n\n")[-1])
    except SignInError as exc:
        raise SystemExit(str(exc)) from None
    finally:
        store.close()


if __name__ == "__main__":
    main()
