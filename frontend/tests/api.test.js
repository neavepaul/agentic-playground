import test from 'node:test';
import assert from 'node:assert/strict';
import { api } from '../src/api.js';

function respond(status, body) {
  globalThis.fetch = async () => new Response(body, { status });
}

test('a plain-text server error becomes a readable message, not a JSON parse error', async () => {
  respond(500, 'Internal Server Error');
  await assert.rejects(api('/chat', { message: 'hi' }), /server had a problem \(HTTP 500\)/);
});

test('a JSON error detail is shown as written', async () => {
  respond(409, JSON.stringify({ detail: 'Conversation is shutting down.' }));
  await assert.rejects(api('/chat', { message: 'hi' }), /Conversation is shutting down\./);
});

test('validation errors get a neutral message (not "household goal" for chat)', async () => {
  respond(422, JSON.stringify({ detail: [{ msg: 'String should have at least 1 character' }] }));
  await assert.rejects(api('/chat', { message: ' ' }), /check what you entered/);
});

test('a successful response that is not JSON is reported, not returned as garbage', async () => {
  respond(200, '<html>proxy page</html>');
  await assert.rejects(api('/world'), /could not be read/);
});

test('a normal JSON response is returned', async () => {
  respond(200, JSON.stringify({ ok: true }));
  assert.deepEqual(await api('/world'), { ok: true });
});
