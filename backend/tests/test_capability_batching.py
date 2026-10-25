import asyncio
from copy import deepcopy
import json

import pytest
from pydantic import ValidationError

from app.agents.explorer import Explorer
from app.agents.schemas import GoalCondition, StrictModel
from app.capabilities.context import evidence_view
from app.capabilities.schemas import BatchAction
from app.config import Settings
from app.events.bus import EventBus
from app.llm.base import structured
from app.tasks.manager import TaskManager, LimitReached
from app.tasks.models import TaskContext
from app.world.engine import WorldEngine
from app.world.tools import WorldTools
from tests.fakes import ScriptedLLM


def manager(client, **settings):
    bus, engine = EventBus(), WorldEngine()
    return TaskManager(client, WorldTools(engine, bus), bus, Settings(**settings))


@pytest.mark.asyncio
async def test_literal_request_still_requires_model_and_two_calls_can_complete(monkeypatch):
    from app.capabilities import tools
    async def read(url, start_char):
        return {'url': url, 'title': 'Example Domain', 'text': 'Actual page', 'next_start_char': None}
    monkeypatch.setattr(tools, 'read_webpage', read)
    calls = []
    class Model:
        async def generate(self, messages, schema):
            calls.append(schema['title'])
            if schema['title'] == 'GoalPlan':
                return json.dumps({'summary': 'Read and calculate.',
                    'conditions': [{'kind': 'answer', 'required_tools': ['read_webpage', 'run_python']}],
                    'initial_actions': [
                        {'tool': 'read_webpage', 'arguments': {'url': 'https://example.com'}},
                        {'tool': 'run_python', 'arguments': {'code': 'print(6 * 7)'}}]})
            payload = json.loads(messages[1]['content'])
            assert payload['required_tool_progress']['can_finish']
            assert any(o.get('observation', {}).get('stdout') == '42\n' for o in payload['tool_observations'])
            return json.dumps({'summary': 'Verified.', 'action': {'tool': 'finish_answer',
                'answer': 'Example Domain; 42.', 'evidence_ids': [o['evidence_id'] for o in payload['tool_observations']]}})
    runner = manager(Model(), direct_initial_delegation=False)
    task = runner.start('Read https://example.com and use Python to calculate 6 × 7. Tell me both results.')
    await runner.runner
    assert task.status == 'completed', task.summary
    assert calls == ['GoalPlan', 'CapabilityChoice']
    assert task.metrics.llm_calls == 2 and task.metrics.coordinator_llm_calls == 1
    assert len(task.answer['evidence_ids']) == 2
    assert all(key in {a['evidence_id'] for a in task.action_history} for key in task.answer['evidence_ids'])


@pytest.mark.asyncio
async def test_model_can_batch_during_later_decisions(monkeypatch):
    from app.capabilities import tools
    async def read(url, start_char):
        return {'url': url, 'title': 'Source', 'text': 'Data'}
    monkeypatch.setattr(tools, 'read_webpage', read)
    def answer(payload):
        return {'summary': 'Ready.', 'action': {'tool': 'finish_answer', 'answer': 'Source; 42.',
            'evidence_ids': [o['evidence_id'] for o in payload['tool_observations'] if o['success']]}}
    client = ScriptedLLM([{'summary': 'Run independent work.', 'action': {'tool': 'batch_capabilities', 'actions': [
        {'tool': 'read_webpage', 'url': 'https://example.org'}, {'tool': 'run_python', 'code': 'print(6 * 7)'}]}}, answer],
        conditions=[{'kind': 'answer', 'required_tools': ['read_webpage', 'run_python']}])
    runner = manager(client)
    task = runner.start('Read a source and calculate.')
    await runner.runner
    assert task.status == 'completed', task.summary
    assert task.metrics.llm_calls == 3
    assert {a['tool'] for a in task.action_history} >= {'read_webpage', 'run_python', 'finish_answer'}


@pytest.mark.asyncio
async def test_batch_is_concurrent_and_reserves_budget():
    runner = manager(ScriptedLLM([]), max_tool_calls=2)
    context = TaskContext(goal='Independent reads')
    started = set()
    ready = asyncio.Event()
    async def execute(tool, args, task):
        started.add(args['url'])
        if len(started) == 2:
            ready.set()
        await asyncio.wait_for(ready.wait(), timeout=1)
        return {'success': True, 'observation': {'url': args['url']}, 'evidence_id': args['url']}
    runner.capabilities.execute = execute
    await runner.call_capability_batch(context, [
        {'tool': 'read_webpage', 'arguments': {'url': 'https://one.example'}},
        {'tool': 'read_webpage', 'arguments': {'url': 'https://two.example'}}])
    assert len(context.action_history) == 2 and context.tool_count == 2
    with pytest.raises(LimitReached):
        await runner.call_capability_batch(context, [{'tool': 'run_python', 'arguments': {'code': '42'}}])
    assert context.tool_count == 2


@pytest.mark.asyncio
async def test_batch_cancellation_keeps_completed_evidence_and_cancels_pending():
    runner = manager(ScriptedLLM([]))
    context = TaskContext(goal='Read two sources')
    completed, cancelled = asyncio.Event(), asyncio.Event()
    async def execute(tool, args, task):
        if args['url'] == 'https://one.example':
            completed.set()
            return {'success': True, 'observation': {}, 'evidence_id': 'done'}
        try:
            await asyncio.sleep(100)
        except asyncio.CancelledError:
            cancelled.set()
            raise
    runner.capabilities.execute = execute
    task = asyncio.create_task(runner.call_capability_batch(context, [
        {'tool': 'read_webpage', 'arguments': {'url': 'https://one.example'}},
        {'tool': 'read_webpage', 'arguments': {'url': 'https://two.example'}}]))
    await completed.wait()
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set() and context.action_history[0]['evidence_id'] == 'done'


@pytest.mark.parametrize('actions', [
    [{'tool': 'finish_answer', 'answer': 'Claim', 'evidence_ids': ['e1']}],
    [{'tool': 'give', 'object': 'medicine', 'person': 'neave'}],
    [{'tool': 'run_python', 'code': '42'}] * 2,
])
def test_batch_rejects_mutations_answers_and_duplicates(actions):
    with pytest.raises(ValidationError):
        BatchAction(tool='batch_capabilities', actions=actions)


def test_prompt_views_are_short_without_changing_evidence():
    context = TaskContext(goal='Research')
    original = {'url': 'https://example.com', 'title': 'Source', 'text': 'x' * 6000,
                'start_char': 100, 'next_start_char': 6100, 'links': [{'url': 'https://example.com'}] * 15}
    context.record('read_webpage', {}, {'success': True, 'observation': original, 'evidence_id': 'long-real-uuid'})
    before = deepcopy(context.action_history)
    view, aliases = evidence_view(context)
    obs = view[0]['observation']
    assert len(obs['text']) == 2500 and obs['next_start_char'] == 2600
    assert view[0]['evidence_id'] == 'e1' and aliases['e1'] == 'long-real-uuid'
    assert context.action_history == before


@pytest.mark.asyncio
async def test_compact_schema_preserves_named_properties_and_runtime_validation():
    class Output(StrictModel):
        title: str
        description: str
    class Client:
        async def generate(self, messages, schema):
            prompt_schema = json.loads(messages[0]['content'].split('Output JSON schema:\n')[1])
            assert set(prompt_schema['properties']) == {'title', 'description'}
            assert 'title' not in prompt_schema
            assert schema['title'] == 'Output'
            return '{"title":"A","description":"B"}'
    result = await structured(Client(), Output, 'Test', {}, compact_schema=True)
    assert result.title == 'A'
