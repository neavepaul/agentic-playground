import test from 'node:test';
import assert from 'node:assert/strict';
import { handleChatEvent, renderTurn } from '../src/chat.js';

/** Just enough DOM for chat.js; setting innerHTML is forbidden for model output. */
class FakeElement {
  constructor(tag) {
    this.tagName = tag.toUpperCase();
    this.children = [];
    this.parent = null;
    this.dataset = {};
    this.style = {};
    this.attributes = {};
    this.className = '';
    this.ownText = '';
    const self = this;
    this.classList = {
      add(...names) { self.className = [...new Set([...self.className.split(' ').filter(Boolean), ...names])].join(' '); },
      remove(...names) { self.className = self.className.split(' ').filter(n => n && !names.includes(n)).join(' '); },
      contains(name) { return self.className.split(' ').includes(name); },
      toggle(name, force) { (force ?? !this.contains(name)) ? this.add(name) : this.remove(name); },
    };
  }
  get textContent() { return this.ownText + this.children.map(child => child.textContent).join(''); }
  set textContent(value) { this.children = []; this.ownText = String(value); }
  set innerHTML(_) { throw new Error('Untrusted model output must not become HTML'); }
  get firstElementChild() { return this.children[0]; }
  get isConnected() { return true; }
  append(...nodes) { for (const node of nodes) { node.parent = this; this.children.push(node); } }
  prepend(...nodes) { for (const node of nodes) { node.parent = this; this.children.unshift(node); } }
  replaceChildren(...nodes) { this.children = []; this.ownText = ''; this.append(...nodes); }
  remove() { if (this.parent) this.parent.children = this.parent.children.filter(child => child !== this); }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  removeAttribute(name) { delete this.attributes[name]; }
  matches(selector) { return selector.startsWith('.') && this.classList.contains(selector.slice(1)); }
  querySelectorAll(selector) {
    const found = [];
    const walk = (node) => node.children.forEach(child => { if (child.matches(selector)) found.push(child); walk(child); });
    walk(this);
    return found;
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
}

class FakeText extends FakeElement {
  constructor(text) { super('#text'); this.ownText = String(text); }
}

function withDom(run) {
  const log = new FakeElement('div');
  log.scrollHeight = 0; log.scrollTop = 0; log.clientHeight = 0;
  const previous = globalThis.document;
  globalThis.document = {
    getElementById(id) { return id === 'chat-log' ? log : null; },
    createElement(tag) { return new FakeElement(tag); },
    createTextNode(text) { return new FakeText(text); },
    querySelectorAll() { return []; },
  };
  try { run(log); } finally { globalThis.document = previous; }
}

test('Kip replies render as text nodes and code blocks, never HTML', () => withDom((log) => {
  const rich = '### Example\n```html\n<script>alert(1)</script>\n```';
  renderTurn({ id: 'rich-test', role: 'kip', text: 'The code is on screen.', screen_text: rich });
  const row = log.children[0];
  assert.ok(row.classList.contains('kip'));
  assert.equal(row.querySelector('.code-block').textContent, '<script>alert(1)</script>');
  assert.match(row.querySelector('.msg-text').textContent, /Example/);
}));

test('legacy turns without screen_text still render, and duplicates are ignored', () => withDom((log) => {
  renderTurn({ id: 'legacy-test', role: 'kip', text: 'Hello.' });
  renderTurn({ id: 'legacy-test', role: 'kip', text: 'Hello.' });
  assert.equal(log.children.length, 1);
  assert.equal(log.children[0].querySelector('.msg-text').textContent, 'Hello.');
}));

test('user text stays verbatim', () => withDom((log) => {
  renderTurn({ id: 'user-test', role: 'user', text: '**not bold** <b>' });
  assert.ok(log.children[0].classList.contains('user'));
  assert.equal(log.children[0].querySelector('.msg-text').textContent, '**not bold** <b>');
}));

test('a streamed reply fills one bubble and is finalised by chat_message', () => withDom((log) => {
  handleChatEvent({ type: 'chat_thinking', data: { reply_id: 'r1' } });
  assert.ok(log.children[0].querySelector('.typing'));
  handleChatEvent({ type: 'chat_delta', data: { reply_id: 'r1', text: 'Hi ' } });
  handleChatEvent({ type: 'chat_delta', data: { reply_id: 'r1', text: 'there.' } });
  assert.equal(log.children[0].querySelector('.msg-text').textContent, 'Hi there.');
  handleChatEvent({ type: 'chat_message', data: { turn: { id: 'final-1', reply_id: 'r1', role: 'kip', text: 'Hi there!',
    sources: [{ title: 'Example', url: 'https://example.com/a' }, { title: 'Bad', url: 'javascript:alert(1)' }] } } });
  assert.equal(log.children.length, 1);
  const row = log.children[0];
  assert.equal(row.querySelector('.msg-text').textContent, 'Hi there!');
  const chips = row.querySelectorAll('.source-chip');
  assert.equal(chips.length, 1);
  assert.equal(chips[0].href, 'https://example.com/a');
}));

test('an interrupted reply that never started typing is removed', () => withDom((log) => {
  handleChatEvent({ type: 'chat_thinking', data: { reply_id: 'r2' } });
  handleChatEvent({ type: 'chat_reply_ended', data: { reply_id: 'r2' } });
  assert.equal(log.children.length, 0);
}));
