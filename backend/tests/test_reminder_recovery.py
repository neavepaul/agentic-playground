import asyncio
import json
from datetime import datetime
from zoneinfo import ZoneInfo

from app.chat.when import read_request
from app.chat.hub import ConversationHub
from app.chat.session import ChatSession
from app.tasks.permissions import Grant
from tests.test_chat import make_session
from tests.test_reminders import build, said


def test_startup_delivers_missed_journaled_reply_once(tmp_path):
    from fastapi.testclient import TestClient
    from app.main import create_app
    from app.memory.recall import MemoryStore
    from app.people import People
    from app.tasks.models import TaskContext
    from app.config import Settings
    from tests.fakes import WaitingLLM
    path = tmp_path / 'recovery.db'
    store = MemoryStore(path)
    people = People(store)
    profile = people.create('Review', '1234')
    cid = people.new_conversation(profile, 'test')
    task = TaskContext(goal='test', session_id=cid, speaker='review', status='completed',
                       answer={'answer': 'Recovered reply', 'sources': []})
    store.set_state({'task-journal': json.dumps([task.model_dump(mode='json')])})
    store.close()
    for _ in range(2):
        store = MemoryStore(path)
        app = create_app(client=WaitingLLM(), settings=Settings(intent_router='off'), memory_store=store)
        with TestClient(app):
            turns = store.conversation_turns(cid)
            assert sum(t.get('task_id') == task.id for t in turns) == 1
            assert turns[-1]['text'] == 'Recovered reply'


def test_reminder_language_regressions():
    now = datetime(2026, 11, 21, 10, tzinfo=ZoneInfo('Asia/Kolkata'))
    request = read_request('Remind me in 1 hour and 30 minutes to stretch', now)
    assert (request.schedule.first - now).total_seconds() == 5400
    assert request.what == 'stretch'
    for text in ["Don't remind me in 10 minutes to stretch", 'Do not cancel all reminders']:
        assert read_request(text, now).action == 'noop'
    assert read_request('Remind me every day at 25:00 to stretch', now).action == 'ask_when'


async def test_followup_cannot_bypass_interval_limit():
    session, scheduler, clock, model, manager = build()
    said(session, 'Remind me to stretch')
    said(session, 'every minute')
    assert not scheduler.active(session.session_id)
    await session.close()


async def test_queue_survives_close_and_preserves_grant():
    session, model, manager, events = make_session([])
    session._start_or_queue('first')
    session._start_or_queue('second', grant=Grant.read_only('second'))
    cid, store = session.session_id, session.store
    await session.close()
    await manager.close()
    restored = ChatSession(model, session.bus, manager, session.settings, store=store, session_id=cid)
    restored.attach()
    await asyncio.sleep(0.1)
    task = manager.tasks[manager.active_id]
    assert task.goal == 'second' and task.grant == Grant.read_only('second')
    assert json.loads(store.get_state(f'queue:{cid}')) == []
    await restored.close()
    await manager.close()


async def test_active_task_prevents_conversation_eviction():
    session, model, manager, events = make_session([])
    hub = ConversationHub(lambda cid, profile: session, session.bus, limit=0)
    hub.get(session.session_id, None)
    session._start_or_queue('running task')
    await hub.trim()
    assert hub.peek(session.session_id) is session
    context = manager.tasks[manager.active_id]
    context.answer = {'answer': 'Finished result', 'sources': []}
    manager.finish(context, 'completed', 'Done')
    assert any(t['text'] == 'Finished result' for t in session.store.conversation_turns(session.session_id))
    manager.runner.cancel()
    try:
        await manager.runner
    except asyncio.CancelledError:
        pass
    await hub.close()
