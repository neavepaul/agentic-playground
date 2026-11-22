import sys
from types import SimpleNamespace

from app.memory.recall import MemoryStore
from app.people import People
from app.phone_push import PhonePush, Subscription


async def test_alerts_follow_conversation_owner_and_stop_on_signout(monkeypatch):
    store = MemoryStore(":memory:")
    people = People(store)
    push = PhonePush(store)
    neave = people.create("Neave", "1234")
    moira = people.create("Moira", "5678")
    neave_token = people.sign_in(neave)
    moira_token = people.sign_in(moira)
    for person, token in [(neave, neave_token), (moira, moira_token)]:
        push.subscribe(person.id, token, Subscription(
            endpoint=f"https://fcm.googleapis.com/{person.name}",
            keys={"auth": "test", "p256dh": "test"},
        ))
    sent = []
    monkeypatch.setitem(sys.modules, "pywebpush", SimpleNamespace(
        webpush=lambda sub, data, **kwargs: sent.append((sub["endpoint"], data)),
        WebPushException=RuntimeError,
    ))
    try:
        push.view(neave_token, "tab", neave.id, "neave-chat")
        await push.send(moira.id, "moira-chat")
        await push.send(neave.id, "neave-chat")
        assert sent == [("https://fcm.googleapis.com/Moira", "{}")]
        push.view(neave_token, "tab", neave.id, "")
        await push.send(neave.id, "neave-chat")
        assert sent[-1] == ("https://fcm.googleapis.com/Neave", "{}")
        push.revoke_session(moira_token)
        await push.send(moira.id, "moira-chat")
        assert len(sent) == 2
    finally:
        store.close()
