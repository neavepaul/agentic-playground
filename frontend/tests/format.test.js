import test from 'node:test';
import assert from 'node:assert/strict';
import { describeEvent, modelStatus, parseRich, providerLabel, relativeTime, safeSources, sourceKey, taskTitle }
  from '../src/format.js';

test('sources keep only http(s) links, once each', () => {
  const sources = safeSources([
    { title: 'A', url: 'https://www.example.com/a' },
    { title: 'A again', url: 'https://www.example.com/a' },
    { title: '', url: 'http://news.test/b' },
    { title: 'Evil', url: 'javascript:alert(1)' },
    { title: 'Broken', url: 'not a url' },
    null,
  ]);
  assert.deepEqual(sources.map(s => [s.host, s.title]), [['example.com', 'A'], ['news.test', 'news.test']]);
  assert.deepEqual(safeSources(undefined), []);
  assert.equal(sourceKey([{ url: 'https://b.test/' }, { url: 'https://a.test/' }]),
    sourceKey([{ url: 'https://a.test/' }, { url: 'https://b.test/' }]));
});

test('lookup goals read like the question that was asked', () => {
  assert.equal(taskTitle({ goal: 'Look up online: latest cricket score' }), 'latest cricket score');
  assert.equal(taskTitle({ goal: 'Look up online: who won? (following up on: cricket)' }), 'who won?');
  assert.equal(taskTitle({ goal: '' }), 'Looking something up');
});

test('model status names the configured provider instead of assuming a local one', () => {
  assert.equal(providerLabel('groq'), 'Groq');
  assert.equal(providerLabel('somethingnew'), 'Somethingnew');
  assert.deepEqual(modelStatus({ provider: 'groq', reachable: true, model: 'llama-x', model_available: true }),
    { state: 'ready', label: 'Groq · llama-x', detail: '' });
  assert.equal(modelStatus({ provider: 'ollama', reachable: false, model: 'm' }).state, 'offline');
  assert.equal(modelStatus({ provider: 'ollama', reachable: true, model: 'm', model_available: false }).state, 'warn');
  assert.equal(modelStatus(null).label, 'Backend offline');
});

test('relative times are short and human', () => {
  const now = Date.parse('2026-01-02T12:00:00Z');
  assert.equal(relativeTime('2026-01-02T11:59:50Z', now), 'just now');
  assert.equal(relativeTime('2026-01-02T11:30:00Z', now), '30 min ago');
  assert.equal(relativeTime('2026-01-02T09:00:00Z', now), '3 h ago');
  assert.equal(relativeTime('2026-01-01T09:00:00Z', now), 'yesterday');
  assert.equal(relativeTime('garbage', now), '');
});

test('rich text splits code fences and inline marks without producing HTML', () => {
  const blocks = parseRich('Use `x` and **this**.\nSee https://example.com/page.\n```py\nprint(1)\n```\nDone');
  assert.equal(blocks.length, 3);
  assert.deepEqual(blocks[1], { type: 'code', lang: 'py', text: 'print(1)' });
  const spans = blocks[0].spans;
  assert.deepEqual(spans.filter(s => s.type !== 'text'), [
    { type: 'code', text: 'x' }, { type: 'bold', text: 'this' },
    { type: 'link', text: 'https://example.com/page', href: 'https://example.com/page' },
  ]);
  assert.deepEqual(parseRich('[docs](https://a.test/x)')[0].spans, [{ type: 'link', text: 'docs', href: 'https://a.test/x' }]);
  assert.deepEqual(parseRich('[bad](javascript:alert(1))')[0].spans.map(s => s.type), ['text']);
});

test('activity shows lookups plainly and marks internal steps as detail', () => {
  const done = describeEvent({ type: 'task_completed', data: { task: { goal: 'Look up online: rain today',
    answer: { sources: [{ url: 'https://w.test/' }] } } } });
  assert.equal(done.text, 'Finished: rain today');
  assert.equal(done.note, '1 source');
  assert.ok(!done.detail);
  assert.ok(describeEvent({ type: 'capability_observed', data: { tool: 'web_search', observation: { query: 'q', results: [] } } }).detail);
  assert.ok(describeEvent({ type: 'critic_review', data: { approved: true } }).detail);
  assert.equal(describeEvent({ type: 'chat_delta', data: {} }), null);
  assert.equal(describeEvent({ type: 'unknown_event', data: {} }), null);
});
