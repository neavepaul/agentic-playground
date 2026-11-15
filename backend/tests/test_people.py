"""Profiles, signing in, and each person's own conversations, errands and memories."""
import asyncio

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.people import FREE_TRIES
from tests.conftest import sign_in
from tests.fakes import WaitingLLM


class Device:
    """One person's phone: the shared test client wearing their sign-in cookie for each request."""

    def __init__(self, client, jar):
        self.client, self.jar = client, jar

    def __getattr__(self, name):
        self.client.cookies.clear()
        self.client.cookies.update(self.jar)
        return getattr(self.client, name)


def two_people(client):
    """Neave (the owner, made first) and Asha, each on their own device."""
    me = sign_in(client, "Neave", "1234")
    neave = Device(client, dict(client.cookies))
    client.cookies.clear()
    other = sign_in(client, "Asha", "5678")
    asha = Device(client, dict(client.cookies))
    return neave, asha, me, other


def test_the_first_profile_is_the_owner_and_names_are_unique():
    with TestClient(create_app(client=WaitingLLM())) as client:
        assert client.get("/api/profiles").json() == {"profiles": [], "me": None, "name": "Kip"}
        owner = sign_in(client, "Neave")
        assert owner["owner"] and owner["portfolio"]
        other = client.post("/api/profiles", json={"name": "Asha", "pin": "5678"}).json()
        assert not other["owner"] and not other["portfolio"]
        assert client.post("/api/profiles", json={"name": "neave", "pin": "1111"}).status_code == 409
        assert client.post("/api/profiles", json={"name": "Bad", "pin": "12a4"}).status_code == 409
        assert client.post("/api/profiles", json={"name": "<b>", "pin": "1234"}).status_code == 409
        names = [p["name"] for p in client.get("/api/profiles").json()["profiles"]]
        assert names == ["Neave", "Asha"]  # the owner first, and no PINs or flags for the picker


def test_wrong_pins_lock_the_profile_and_the_right_one_signs_in():
    with TestClient(create_app(client=WaitingLLM())) as client:
        profile = sign_in(client, "Neave", "1234")
        client.post("/api/signout")
        assert client.get("/api/me").status_code == 401
        for attempt in range(FREE_TRIES - 1):
            wrong = client.post("/api/signin", json={"profile_id": profile["id"], "pin": "0000"})
            assert wrong.status_code == 403 and "Wrong PIN" in wrong.json()["detail"]
        locked = client.post("/api/signin", json={"profile_id": profile["id"], "pin": "0000"})
        assert locked.status_code == 403 and "Too many wrong PINs" in locked.json()["detail"]
        # Locked: even the right PIN waits, so 10,000 PINs cannot be tried quickly.
        assert client.post("/api/signin", json={"profile_id": profile["id"], "pin": "1234"}).status_code == 403


def test_a_right_pin_signs_in_and_signing_out_ends_it():
    with TestClient(create_app(client=WaitingLLM())) as client:
        profile = sign_in(client, "Neave", "1234")
        client.post("/api/signout")
        assert client.post("/api/signin", json={"profile_id": profile["id"], "pin": "1234"}).status_code == 200
        assert client.get("/api/me").json()["name"] == "Neave"
        client.post("/api/signout")
        assert client.get("/api/me").status_code == 401


def test_every_message_without_a_conversation_starts_a_new_one():
    with TestClient(create_app(client=WaitingLLM())) as client:
        sign_in(client)
        first = client.post("/api/chat", json={"message": "What's the weather in Kochi?"}).json()["session_id"]
        again = client.post("/api/chat", json={"message": "and tomorrow?", "conversation_id": first}).json()
        assert again["session_id"] == first
        second = client.post("/api/chat", json={"message": "Hi Kip"}).json()["session_id"]
        listed = client.get("/api/conversations").json()["conversations"]
        assert [c["id"] for c in listed] == [second, first]  # newest first
        assert listed[1]["title"] == "What's the weather in Kochi?"
        turns = client.get("/api/chat", params={"conversation_id": first}).json()["turns"]
        assert [t["text"] for t in turns if t["role"] == "user"] == ["What's the weather in Kochi?", "and tomorrow?"]
        assert client.get("/api/chat").json()["turns"] == []  # a new chat starts empty
        assert client.delete(f"/api/conversations/{first}").status_code == 204
        assert [c["id"] for c in client.get("/api/conversations").json()["conversations"]] == [second]
        # The page names a new chat itself, so its live events can be matched before the reply.
        named = client.post("/api/chat", json={"message": "hey", "conversation_id": "a1b2c3d4e5f60789"}).json()
        assert named["session_id"] == "a1b2c3d4e5f60789"
        assert client.post("/api/chat", json={"message": "x", "conversation_id": "../etc"}).status_code == 422


def test_people_never_see_each_others_conversations():
    app = create_app(client=WaitingLLM())
    with TestClient(app) as client:
        neave, asha, _, _ = two_people(client)
        mine = neave.post("/api/chat", json={"message": "my secret plan"}).json()["session_id"]
        assert asha.get("/api/conversations").json()["conversations"] == []
        assert asha.get("/api/chat", params={"conversation_id": mine}).status_code == 404
        assert asha.post("/api/chat", json={"message": "hi", "conversation_id": mine}).status_code == 404
        assert neave.get("/api/chat", params={"conversation_id": mine}).json()["turns"][0]["text"] == "my secret plan"
        assert asha.delete(f"/api/conversations/{mine}").status_code == 404
        with asha.websocket_connect("/ws") as socket:
            snapshot = socket.receive_json()
            assert "my secret plan" not in str(snapshot)
            neave.post("/api/chat", json={"message": "another secret", "conversation_id": mine})
            theirs = asha.post("/api/chat", json={"message": "hello"}).json()["session_id"]
            while True:  # everything before Asha's own message must be hers too
                event = socket.receive_json()
                assert "secret" not in str(event)
                if event["type"] == "chat_message":
                    assert event["data"]["turn"]["session_id"] == theirs
                    break


def test_an_errand_is_narrated_only_in_the_conversation_that_asked():
    app = create_app(client=WaitingLLM())
    with TestClient(app) as client:
        neave, asha, _, _ = two_people(client)
        mine = neave.post("/api/chat", json={"message": "hi"}).json()["session_id"]
        theirs = asha.post("/api/chat", json={"message": "hi"}).json()["session_id"]
        hub = app.state.hub
        task = neave.portal.call(hub.peek(mine).manager.start, "Look up online: the score")
        assert task.session_id == mine and task.speaker == "neave"
        assert hub.peek(theirs).manager.active() is None   # Asha's chat never sees Neave's errand
        assert hub.peek(theirs).manager.tasks == {}
        assert asha.get(f"/api/tasks/{task.id}").status_code == 404
        assert neave.get(f"/api/tasks/{task.id}").json()["id"] == task.id
        texts = [t["text"] for t in hub.peek(theirs).history()]
        assert not any("score" in text for text in texts)
        assert any("score" in t["text"] for t in hub.peek(mine).history())  # "Start now: …"


def test_each_person_sees_only_their_own_memories_and_the_owner_keeps_the_old_ones():
    app = create_app(client=WaitingLLM())
    with TestClient(app) as client:
        neave, asha, _, _ = two_people(client)
        store = app.state.manager.memory_store

        async def remember():
            await store.add("fact", "I live in Kochi", about=["neave"], meta={"speaker": "neave"})
            await store.add("fact", "I am a nurse", about=["asha"], meta={"speaker": "asha"})
            await store.add("fact", "The garden gate sticks", meta={})  # from before profiles
        asyncio.run(remember())
        mine = {m["text"] for m in neave.get("/api/memories").json()}
        theirs = {m["text"] for m in asha.get("/api/memories").json()}
        assert mine == {"I live in Kochi", "The garden gate sticks"}
        assert theirs == {"I am a nurse"}
        assert {m["text"] for m in asha.get("/api/memories", params={"q": "Kochi"}).json()} <= {"I am a nurse"}


def test_the_chat_from_before_profiles_becomes_the_owners():
    from app.memory.recall import MemoryStore
    store = MemoryStore(":memory:")
    store.append_turn("home", {"id": "t1", "role": "user", "text": "old hello", "session_id": "home"})
    with TestClient(create_app(client=WaitingLLM(), memory_store=store)) as client:
        sign_in(client)
        listed = client.get("/api/conversations").json()["conversations"]
        assert [(c["id"], c["title"]) for c in listed] == [("home", "Earlier chats")]
        assert client.get("/api/chat", params={"conversation_id": "home"}).json()["turns"][0]["text"] == "old hello"


def test_only_the_owner_starts_api_errands_and_shares_the_portfolio():
    app = create_app(client=WaitingLLM())
    with TestClient(app) as client:
        neave, asha, _, other = two_people(client)
        assert asha.post("/api/tasks", json={"goal": "Find charger."}).status_code == 403
        assert asha.post(f"/api/profiles/{other['id']}/access", json={"portfolio": True}).status_code == 403
        assert [p["name"] for p in neave.get("/api/me").json()["people"]] == ["Asha"]
        shared = neave.post(f"/api/profiles/{other['id']}/access", json={"portfolio": True}).json()
        assert shared["portfolio"] and not shared["pin_change"]  # only what was asked changes
        assert asha.get("/api/me").json()["portfolio"]


def test_a_signed_out_page_gets_nothing():
    from starlette.websockets import WebSocketDisconnect
    with TestClient(create_app(client=WaitingLLM())) as client:
        sign_in(client)
        client.cookies.clear()
        for path in ("/api/state", "/api/conversations", "/api/memories", "/api/persona", "/api/me"):
            assert client.get(path).status_code == 401
        assert client.post("/api/chat", json={"message": "hi"}).status_code == 401
        # The local-tool header counts only from the Pi itself (TestClient is not loopback).
        assert client.get("/api/state", headers={"x-kip-local": "1"}).status_code == 401
        with pytest.raises(WebSocketDisconnect) as refused:
            with client.websocket_connect("/ws") as socket:
                socket.receive_json()
        assert refused.value.code == 1008


@pytest.mark.asyncio
async def test_someone_without_access_is_told_the_portfolio_is_not_theirs(monkeypatch):
    from app.capabilities import feeds
    from app.capabilities.tools import NOT_SHARED, CapabilityTools
    from app.events.bus import EventBus
    from app.tasks.models import TaskContext
    asked = []

    async def holdings(*args, **kwargs):
        asked.append(args)
        return feeds.FeedAnswer("Worth 46,905.", "https://groww.in/", "Groww holdings", "groww")
    monkeypatch.setattr(feeds, "answer", holdings)
    tools = CapabilityTools(EventBus(), groww=(None, None))
    question = {"domain": "finance", "said": "how is my portfolio doing", "time": "now"}
    assert feeds.is_portfolio(question)
    context = TaskContext(goal="Look up online: how is my portfolio doing", portfolio=False)
    result, answer = await tools.feed_answer("how is my portfolio doing", question, context)
    assert answer == NOT_SHARED.format(holder="the owner's") and result["success"] and not asked  # Groww is never asked
    assert tools.feed_source(question, portfolio=False) is None     # and no "Let me check Groww"


def test_phones_on_the_home_network_reach_kip_but_other_sites_do_not():
    """Kip answers at the Pi's own name or a private address; a page may act only when Kip served it."""
    import socket
    machine = socket.gethostname().lower()
    with TestClient(create_app(client=WaitingLLM())) as client:
        for host in ("192.168.29.96:8000", "10.0.0.5:8000", f"{machine}.local:8000", f"{machine}:8000", "[::1]:8000"):
            assert client.get("/api/health", headers={"host": host}).status_code == 200, host
        for host in ("kip.attacker.example", "8.8.8.8", "evil.local"):
            assert client.get("/api/health", headers={"host": host}).status_code == 400, host
        lan = {"host": "192.168.29.96:8000"}
        made = client.post("/api/profiles", json={"name": "Neave", "pin": "1234"},
                           headers={**lan, "origin": "http://192.168.29.96:8000"})
        assert made.status_code == 201  # the page Kip served at that address
        foreign = client.post("/api/chat", json={"message": "hi"},
                              headers={**lan, "origin": "http://192.168.29.50:3000"})
        assert foreign.status_code == 403  # another device's page, even on the same network


def test_a_profile_made_without_a_pin_is_claimed_once_by_choosing_one():
    app = create_app(client=WaitingLLM())
    with TestClient(app) as client:
        people = app.state.people
        owner = people.create("Neave", "2468")
        paul = people.create("Paul", None)
        assert owner.owner and not owner.needs_pin and paul.needs_pin and not paul.owner
        listed = {p["name"]: p["needs_pin"] for p in client.get("/api/profiles").json()["profiles"]}
        assert listed == {"Neave": False, "Paul": True}
        # No PIN yet: signing in is refused, choosing one signs in.
        assert client.post("/api/signin", json={"profile_id": paul.id, "pin": "0000"}).status_code == 403
        assert client.post(f"/api/profiles/{paul.id}/pin", json={"pin": "12a4"}).status_code == 409
        chosen = client.post(f"/api/profiles/{paul.id}/pin", json={"pin": "4321"})
        assert chosen.status_code == 200 and client.get("/api/me").json()["name"] == "Paul"
        # Once chosen, nobody can choose again: only the PIN signs in.
        client.cookies.clear()
        assert client.post(f"/api/profiles/{paul.id}/pin", json={"pin": "1111"}).status_code == 409
        assert client.post("/api/signin", json={"profile_id": paul.id, "pin": "4321"}).status_code == 200
        assert client.post(f"/api/profiles/{owner.id}/pin", json={"pin": "1111"}).status_code == 409


def test_profiles_are_made_from_the_command_line(tmp_path, monkeypatch, capsys):
    from app.config import get_settings
    from app.people import main
    monkeypatch.setenv("MEMORY_DB_FILE", str(tmp_path / "kip.db"))
    get_settings.cache_clear()  # settings are read once; this test points them at its own kip.db
    try:
        main(["add", "Neave", "2468"])
        main(["add", "Minu"])
        main(["list"])
        out = capsys.readouterr().out
        assert "Made Neave (the owner)." in out and "Made Minu: they choose a PIN" in out
        assert "Neave (owner, sees the portfolio)" in out and "Minu (no PIN yet)" in out
        with pytest.raises(SystemExit, match="taken"):
            main(["add", "minu"])
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()


def test_the_kip_panel_shows_each_person_only_their_own_bond_moments_and_check_ins():
    app = create_app(client=WaitingLLM())
    with TestClient(app) as client:
        neave, asha, _, _ = two_people(client)
        persona = app.state.persona

        def said(text, speaker):
            persona.observe({"type": "chat_message", "data": {"turn": {"role": "user", "text": text, "speaker": speaker}}})
        said("Morning Kip, lovely day", "neave")
        said("I'm so stressed and exhausted about work", "asha")
        persona.observe({"type": "task_failed", "data": {"task": {"goal": "Look up online: asha's secret", "speaker": "asha"}}})
        from app.persona.state import Followup
        persona.state.followups.append(Followup(person="asha", reason="asha was stressed about work",
                                                due_at=0, expires_at=10 ** 9))
        assert {"neave", "asha"} <= set(persona.state.relationships)

        mine = neave.get("/api/persona").json()
        assert set(mine["relationships"]) == {"neave"}
        assert "asha" not in str(mine["moments"]) and "secret" not in str(mine) and mine["followups"] == []
        assert mine["mood"] and mine["traits"]  # Kip himself is everyone's
        theirs = asha.get("/api/persona").json()
        assert set(theirs["relationships"]) == {"asha"} and theirs["followups"]
        assert any("secret" in m["summary"] for m in theirs["moments"])

        # The live mood update is personal too.
        with neave.websocket_connect("/ws") as socket:
            socket.receive_json()  # snapshot
            said("Still stressed, honestly", "asha")
            while True:
                event = socket.receive_json()
                if event["type"] == "persona_updated":
                    assert set(event["data"]["persona"]["relationships"]) <= {"neave"}
                    assert "asha" not in str(event["data"]["persona"])
                    break


def test_people_change_their_own_pin_only_when_the_owner_allows():
    app = create_app(client=WaitingLLM())
    with TestClient(app) as client:
        neave, asha, me, other = two_people(client)
        assert me["pin_change"] and not other["pin_change"]  # the owner always may
        assert asha.post("/api/me/pin", json={"current": "5678", "pin": "1111"}).status_code == 403
        assert neave.post(f"/api/profiles/{other['id']}/access", json={"pin_change": True}).json()["pin_change"]
        # The current PIN first; a wrong one counts towards the lock like any wrong PIN.
        wrong = asha.post("/api/me/pin", json={"current": "0000", "pin": "1111"})
        assert wrong.status_code == 403 and "Wrong PIN" in wrong.json()["detail"]
        changed = asha.post("/api/me/pin", json={"current": "5678", "pin": "1111"})
        assert changed.status_code == 200
        old_jar, asha.jar = asha.jar, {"kip_session": changed.cookies["kip_session"]}  # a fresh sign-in here
        assert asha.get("/api/me").json()["name"] == "Asha"
        assert Device(client, old_jar).get("/api/me").status_code == 401  # every other device is signed out
        client.cookies.clear()
        assert client.post("/api/signin", json={"profile_id": other["id"], "pin": "5678"}).status_code == 403
        assert client.post("/api/signin", json={"profile_id": other["id"], "pin": "1111"}).status_code == 200
        # The owner changes their own (and the old PIN no longer works).
        assert neave.post("/api/me/pin", json={"current": "1234", "pin": "8642"}).status_code == 200
        client.cookies.clear()
        assert client.post("/api/signin", json={"profile_id": me["id"], "pin": "1234"}).status_code == 403


def test_an_older_kip_db_gains_the_pin_change_setting():
    import sqlite3
    from app.memory.recall import MemoryStore
    from app.people import People
    store = MemoryStore(":memory:")
    with store._lock:
        store._db.executescript("""CREATE TABLE profiles (id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE COLLATE NOCASE,
            pin_hash BLOB NOT NULL, salt BLOB NOT NULL, owner INTEGER NOT NULL DEFAULT 0,
            portfolio INTEGER NOT NULL DEFAULT 0, failures INTEGER NOT NULL DEFAULT 0,
            locked_until REAL NOT NULL DEFAULT 0, created_at TEXT NOT NULL);
            INSERT INTO profiles(id, name, pin_hash, salt, owner, created_at) VALUES ('p1', 'Paul', x'', x'', 0, 'then');""")
    people = People(store)
    assert people.get("p1").pin_change is False and people.get("p1").needs_pin
    assert people.set_access("p1", pin_change=True).pin_change
    assert isinstance(store._db, sqlite3.Connection)


@pytest.mark.asyncio
async def test_someone_allowed_asks_about_the_owners_holdings_by_name(monkeypatch):
    """Minu, allowed the portfolio, asks "how much HDFC Bank does Neave own": Groww answers, about Neave."""
    from app.capabilities import feeds
    from app.capabilities.tools import CapabilityTools
    from app.chat.router import Route
    from app.chat.session import _feed_question
    from app.events.bus import EventBus
    from app.tasks.models import TaskContext
    route = Route(intent="information.finance", confidence=0.9, domain="finance", entities=["HDFC Bank"], focus="price",
                  time="now")
    question = _feed_question(route, "How much HDFC bank stocks does Neave own", holder="Neave")
    assert question["said"] == "How much HDFC bank stocks do i own" and question["holder"] == "Neave"
    assert feeds.is_portfolio(question)
    assert not feeds.is_portfolio(_feed_question(route, "How much HDFC bank stocks does Neave own"))

    async def holdings(*args, **kwargs):
        return feeds.FeedAnswer("You hold 10 HDFCBANK shares, worth ₹8,040: up ₹40 on the ₹8,000 you paid.",
                                "https://groww.in/", "Groww holdings", "groww")
    monkeypatch.setattr(feeds, "answer", holdings)
    tools = CapabilityTools(EventBus(), groww=(None, None))
    _, answer = await tools.feed_answer("q", question, TaskContext(goal="Look up online: q", portfolio=True))
    assert answer == "Neave holds 10 HDFCBANK shares, worth ₹8,040: up ₹40 on the ₹8,000 Neave paid."
    _, refused = await tools.feed_answer("q", question, TaskContext(goal="Look up online: q", portfolio=False))
    assert refused.startswith("That's Neave's Groww portfolio")


def test_kip_knows_who_has_a_profile_and_whose_the_portfolio_is():
    app = create_app(client=WaitingLLM())
    with TestClient(app) as client:
        neave, asha, _, _ = two_people(client)
        theirs = asha.post("/api/chat", json={"message": "hi"}).json()["session_id"]
        session = app.state.hub.peek(theirs)
        status = session._status("asha")
        assert status["talking_with"] == "Asha"
        assert "Neave (owner: the Groww portfolio is theirs)" in status["people_with_profiles"]
        assert "Asha" in status["people_with_profiles"]
        assert session._holder() == "Neave"
        mine = neave.post("/api/chat", json={"message": "hi"}).json()["session_id"]
        assert app.state.hub.peek(mine)._holder() is None  # the owner asks in their own words
