import asyncio
import json
import socket

import httpx
import pytest

from app.agents.explorer import Explorer
from app.agents.schemas import GoalCondition
from app.capabilities import web
from app.capabilities.python import run_python
from app.capabilities.schemas import AnswerAction
from app.capabilities.tools import CapabilityTools
from app.config import Settings
from app.events.bus import EventBus
from app.tasks.goals import check_conditions
from app.tasks.manager import TaskManager
from app.tasks.models import TaskContext
from app.world.engine import WorldEngine
from app.world.tools import WorldTools
from tests.fakes import ScriptedLLM


@pytest.mark.parametrize('url', ['file:///etc/passwd', 'ftp://example.com', 'https://user:secret@example.com',
                                 'http://example.com:8000', 'https://example.com/\nheader',
                                 'https://example.com\\@localhost'])
def test_reader_rejects_invalid_urls(url):
    with pytest.raises((ValueError, httpx.InvalidURL)):
        web.public_url(url)


@pytest.mark.asyncio
@pytest.mark.parametrize('ip', ['127.0.0.1', '::1', '192.168.1.1', '169.254.169.254',
                                '0.0.0.0', '::ffff:127.0.0.1'])
async def test_reader_rejects_private_addresses(monkeypatch, ip):
    async def addresses(*args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, '', (ip, 443))]
    monkeypatch.setattr(asyncio.get_running_loop(), 'getaddrinfo', addresses)
    with pytest.raises(ValueError, match='network addresses'):
        await web.public_address('example.com', 443)


@pytest.mark.asyncio
async def test_reader_checks_redirect_and_pins_connection(monkeypatch):
    checked, requests = [], []
    async def address(host, port):
        checked.append(host)
        if host == 'internal.test':
            raise ValueError('Private address blocked')
        return '93.184.216.34'
    def respond(request):
        requests.append(request)
        return httpx.Response(302, headers={'location': 'http://internal.test/'})
    original = httpx.AsyncClient
    monkeypatch.setattr(web, 'public_address', address)
    monkeypatch.setattr(web.httpx, 'AsyncClient', lambda **kwargs: original(transport=httpx.MockTransport(respond), **kwargs))
    with pytest.raises(ValueError, match='blocked'):
        await web.read_webpage('https://example.com/')
    assert checked == ['example.com', 'internal.test']
    assert requests[0].url.host == '93.184.216.34'
    assert requests[0].headers['host'] == 'example.com'
    assert requests[0].extensions['sni_hostname'] == 'example.com'


@pytest.mark.asyncio
@pytest.mark.parametrize('body,kind,error', [(b'x' * 1_000_001, 'text/plain', '1 MB'),
                                           (b'%PDF', 'application/pdf', 'HTML and plain text')], ids=['oversized', 'pdf'])
async def test_reader_bounds_downloads(monkeypatch, body, kind, error):
    async def address(*args):
        return '93.184.216.34'
    original = httpx.AsyncClient
    monkeypatch.setattr(web, 'public_address', address)
    transport = httpx.MockTransport(lambda req: httpx.Response(200, content=body, headers={'content-type': kind}))
    monkeypatch.setattr(web.httpx, 'AsyncClient', lambda **kwargs: original(transport=transport, **kwargs))
    with pytest.raises(ValueError, match=error):
        await web.read_webpage('https://example.com/')


def test_reader_extracts_text_links_and_continuations():
    body = '<title>Manual</title><script>secret()</script><nav>Menu</nav><main>Hello <a href="/more">More</a>' + 'x' * 7000 + '</main>'
    first = web.extract_page(body, 'https://example.com', 0)
    assert first['title'] == 'Manual'
    assert 'secret' not in first['text'] and 'Menu' not in first['text']
    assert first['links'][0]['url'] == 'https://example.com/more'
    assert first['next_start_char'] == 6000
    assert web.extract_page(body, 'https://example.com', 6000)['next_start_char'] is None
    assert first['untrusted_content']


def test_search_preserves_sources_and_filters_non_web_links(monkeypatch):
    import ddgs
    class Search:
        def __init__(self, **kwargs):
            pass
        def text(self, query, **kwargs):
            return [{'title': 'Source', 'href': 'https://example.com', 'body': 'Snippet'},
                    {'title': 'Unsafe', 'href': 'file:///private', 'body': 'Ignore'}]
    monkeypatch.setattr(ddgs, 'DDGS', Search)
    result = web.web_search('test query')
    assert result['query'] == 'test query'
    assert result['results'] == [{'title': 'Source', 'url': 'https://example.com', 'snippet': 'Snippet'}]


@pytest.mark.asyncio
async def test_python_calculates_with_input_and_has_no_host_access():
    result = await run_python('values = data["values"]\nprint(sum(values))\nsum(values) / len(values)', {'values': [4, 6]})
    assert result['stdout'] == '10\n'
    assert result['result'] == '5.0'
    for code in ['open("backend/.env").read()', 'import subprocess\nsubprocess.run(["whoami"])',
                 'import socket\nsocket.socket()', 'import os\nos.environ["PATH"]']:
        with pytest.raises(Exception):
            await run_python(code, {})


@pytest.mark.asyncio
async def test_python_infinite_loop_output_and_input_limits():
    with pytest.raises(Exception):
        await asyncio.wait_for(run_python('while True:\n    pass', {}, seconds=0.05), timeout=5)
    with pytest.raises(Exception):
        await run_python('print("x" * 9000)', {})
    with pytest.raises(ValueError, match='32,000'):
        await run_python('data', {'x': 'x' * 33000})
    # A failed/terminated program does not poison the following execution.
    assert (await run_python('6 * 7', {}))['result'] == '42'


def evidence(context, tool, obs, key='e1'):
    context.record(tool, {}, {'success': True, 'observation': obs, 'evidence_id': key})


def test_answer_requires_real_task_evidence_and_requested_tools():
    context = TaskContext(goal='Research', conditions=[GoalCondition(kind='answer', required_tools=['read_webpage'])])
    evidence(context, 'web_search', {'results': [{'url': 'https://example.com', 'title': 'Example'}]})
    for ids in [['invented'], ['e1']]:
        with pytest.raises(ValueError):
            CapabilityTools.finish_answer(AnswerAction(tool='finish_answer', answer='Claim', evidence_ids=ids), context)
    evidence(context, 'read_webpage', {'url': 'https://example.com', 'title': 'Example', 'text': 'Facts'}, 'e2')
    result = CapabilityTools.finish_answer(AnswerAction(tool='finish_answer', answer='Facts', evidence_ids=['e1', 'e2']), context)
    assert len(result['sources']) == 1
    assert not check_conditions(context)[0]['satisfied']
    evidence(context, 'finish_answer', result, 'e3')
    assert check_conditions(context)[0]['satisfied']
    assert context.public()['answer']['answer'] == 'Facts'


@pytest.mark.asyncio
async def test_digital_goal_executes_program_and_finishes_without_moving():
    def answer(payload):
        entry = next(o for o in payload['tool_observations'] if o['tool'] == 'run_python')
        assert entry['observation']['result'] == '42'
        return {'summary': 'Calculation checked.', 'action': {'tool': 'finish_answer', 'answer': 'The answer is 42.',
                                                            'evidence_ids': [entry['evidence_id']]}}
    llm = ScriptedLLM([{'summary': 'Calculating.', 'action': {'tool': 'run_python', 'code': '6 * 7'}}, answer],
                      conditions=[{'kind': 'answer', 'required_tools': ['run_python']}])
    engine, bus = WorldEngine(), EventBus()
    before = engine.snapshot()
    manager = TaskManager(llm, WorldTools(engine, bus), bus,
                          Settings(direct_initial_delegation=True, idle_reflection_enabled=False))
    task = manager.start('Use Python to calculate 6 * 7.')
    await manager.runner
    assert task.status == 'completed', task.summary
    assert task.answer['answer'] == 'The answer is 42.'
    assert engine.snapshot() == before
    assert [a['tool'] for a in task.action_history] == ['get_status', 'get_map', 'run_python', 'finish_answer']


@pytest.mark.asyncio
async def test_search_read_answer_workflow(monkeypatch):
    from app.capabilities import tools
    monkeypatch.setattr(tools, 'web_search', lambda query: {'query': query, 'results': [{'url': 'https://example.com', 'title': 'Source'}]})
    async def read(*args):
        return {'title': 'Source', 'url': 'https://example.com', 'text': 'Verified fact'}
    monkeypatch.setattr(tools, 'read_webpage', read)
    def answer(payload):
        return {'summary': 'Ready.', 'action': {'tool': 'finish_answer', 'answer': 'Verified fact.',
                    'evidence_ids': [o['evidence_id'] for o in payload['tool_observations'] if o['success']]}}
    llm = ScriptedLLM([{'summary': 'Searching.', 'action': {'tool': 'web_search', 'query': 'a fact'}},
                      {'summary': 'Reading.', 'action': {'tool': 'read_webpage', 'url': 'https://example.com'}}, answer],
                     conditions=[{'kind': 'answer', 'required_tools': ['web_search', 'read_webpage']}])
    bus, engine = EventBus(), WorldEngine()
    manager = TaskManager(llm, WorldTools(engine, bus), bus, Settings(direct_initial_delegation=True))
    task = manager.start('Research a fact online.')
    await manager.runner
    assert task.status == 'completed', task.summary
    assert task.answer['sources'] == [{'title': 'Source', 'url': 'https://example.com'}]


@pytest.mark.asyncio
async def test_failed_capability_does_not_satisfy_answer_and_disabled_tools_fail():
    task = TaskContext(goal='Calculate', conditions=[GoalCondition(kind='answer')])
    result = await CapabilityTools(EventBus(), enabled=False).execute('run_python', {'code': '1'}, task)
    task.record('run_python', {}, result)
    assert not result['success']
    assert not check_conditions(task)[0]['satisfied']


@pytest.mark.asyncio
async def test_capability_cancellation_is_propagated(monkeypatch):
    from app.capabilities import tools
    async def block(*args):
        await asyncio.sleep(100)
    monkeypatch.setattr(tools, 'read_webpage', block)
    task = asyncio.create_task(CapabilityTools(EventBus()).execute('read_webpage', {'url': 'https://example.com'}, TaskContext(goal='Read')))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_robot_only_goals_keep_original_explorer_schema():
    task = TaskContext(goal='Find keys', conditions=[GoalCondition(kind='find_object', object='keys')])
    evidence(task, 'get_status', {'room': 'hall', 'inventory': []})
    action = await Explorer(ScriptedLLM([]), WorldTools.schemas()).decide(task, task.goal)
    assert action.tool == 'look' and action.source == 'reflex'


@pytest.mark.asyncio
async def test_failed_program_can_be_corrected_from_observed_error():
    def repair(payload):
        failure = payload['tool_observations'][-1]
        assert not failure['success'] and failure['arguments']['code'] == '1 / 0'
        assert 'error' in failure
        return {'summary': 'Correcting calculation.', 'action': {'tool': 'run_python', 'code': '6 * 7'}}
    def answer(payload):
        good = next(o for o in payload['tool_observations'] if o['tool'] == 'run_python' and o['success'])
        return {'summary': 'Checked.', 'action': {'tool': 'finish_answer', 'answer': '42',
                                               'evidence_ids': [good['evidence_id']]}}
    llm = ScriptedLLM([{'summary': 'Calculating.', 'action': {'tool': 'run_python', 'code': '1 / 0'}}, repair, answer],
                      conditions=[{'kind': 'answer', 'required_tools': ['run_python']}])
    bus, engine = EventBus(), WorldEngine()
    manager = TaskManager(llm, WorldTools(engine, bus), bus, Settings(direct_initial_delegation=True))
    task = manager.start('Calculate 6 * 7 with Python.')
    await manager.runner
    assert task.status == 'completed', task.summary
    assert task.answer['answer'] == '42'
    assert any(not a['success'] for a in task.action_history)


@pytest.mark.asyncio
async def test_completion_choice_is_unavailable_until_required_tools_succeed():
    context = TaskContext(goal='Read a page and execute Python.', conditions=[
        GoalCondition(kind='answer', required_tools=['read_webpage', 'run_python'])])
    evidence(context, 'read_webpage', {'url': 'https://example.com', 'title': 'Example'}, 'page-id')
    context.record('finish_answer', {'answer': 'UNSUPPORTED_CLAIM', 'evidence_ids': ['page-id']},
                   {'success': False, 'error': 'Python has not run.', 'evidence_id': 'rejected-id'})
    def calculate(payload):
        assert payload['required_tool_progress']['missing_tools'] == ['run_python']
        assert not payload['required_tool_progress']['can_finish']
        assert 'UNSUPPORTED_CLAIM' not in json.dumps(payload)
        return {'summary': 'Executing the missing calculation.', 'action': {'tool': 'run_python', 'code': '6 * 7'}}
    llm = ScriptedLLM([calculate, {'summary': 'Evidence complete.', 'action': {
        'tool': 'finish_answer', 'answer': 'Example; 42.', 'evidence_ids': ['page-id', 'python-id']}}])
    explorer = Explorer(llm, WorldTools.schemas())
    first = await explorer.decide(context, context.goal)
    assert first.tool == 'run_python'
    assert 'finish_answer' not in llm.calls[0][1]['properties']['action']['discriminator']['mapping']
    evidence(context, 'run_python', {'result': '42', 'stdout': ''}, 'python-id')
    second = await explorer.decide(context, context.goal)
    assert second.tool == 'finish_answer'
    assert 'finish_answer' in llm.calls[1][1]['properties']['action']['discriminator']['mapping']


def test_missing_evidence_error_names_only_the_missing_tool():
    context = TaskContext(goal='Read and calculate', conditions=[
        GoalCondition(kind='answer', required_tools=['read_webpage', 'run_python'])])
    evidence(context, 'read_webpage', {'url': 'https://example.com', 'title': 'Example'}, 'page-id')
    with pytest.raises(ValueError) as exc:
        CapabilityTools.finish_answer(AnswerAction(tool='finish_answer', answer='42', evidence_ids=['page-id']), context)
    assert 'run_python' in str(exc.value)
    assert 'read_webpage' not in str(exc.value)


@pytest.mark.asyncio
async def test_mixed_goal_can_read_page_then_complete_physical_delivery(monkeypatch):
    from pathlib import Path
    from app.capabilities import tools
    async def read(*args):
        return {'title': 'Example', 'url': 'https://example.com', 'text': 'Example page'}
    monkeypatch.setattr(tools, 'read_webpage', read)
    def answer(payload):
        page = next(o for o in payload['tool_observations'] if o['tool'] == 'read_webpage')
        return {'summary': 'Page read.', 'action': {'tool': 'finish_answer', 'answer': 'Example page.',
                                                  'evidence_ids': [page['evidence_id']]}}
    def move(room):
        return {'summary': 'Moving.', 'action': {'tool': 'robot_command', 'command_id': 'move_to:' + room}}
    llm = ScriptedLLM([{'summary': 'Reading.', 'action': {'tool': 'read_webpage', 'url': 'https://example.com'}},
                      answer, move('study'), move('hall'), move('bedroom')], conditions=[
                          {'kind': 'answer', 'required_tools': ['read_webpage']},
                          {'kind': 'deliver', 'object': 'charger', 'person': 'neave'}])
    engine = WorldEngine(Path(__file__).parent / 'fixtures' / 'legacy_house.json')
    bus = EventBus()
    manager = TaskManager(llm, WorldTools(engine, bus), bus, Settings(direct_initial_delegation=True))
    task = manager.start('Read example.com and bring the charger to Neave.')
    await manager.runner
    assert task.status == 'completed', task.summary
    assert task.answer['answer'] == 'Example page.'
    assert all(c['satisfied'] for c in check_conditions(task))
    assert any(a['tool'] == 'give' and a['success'] for a in task.action_history)
