import test from 'node:test';
import assert from 'node:assert/strict';
import { renderTurn } from '../src/chat.js';

test('chat preserves screen content safely and supports legacy turns', () => {
  const nodes = [];
  const log = { children: nodes, scrollHeight: 0, scrollTop: 0, clientHeight: 0,
    append(node) { nodes.push(node); } };
  const previous = globalThis.document;
  globalThis.document = {
    getElementById(id) { return id === 'chat-log' ? log : null; },
    createElement() {
      return { children: [], style: {}, classList: { add() {}, remove() {} },
        append(node) { this.children.push(node); },
        querySelector() { return this.children[0]; },
        set innerHTML(_) { throw new Error('Untrusted model output must not become HTML'); } };
    },
  };
  try {
    const rich = '### Example\n```html\n<script>alert(1)</script>\n```';
    renderTurn({ id: 'rich-test', role: 'kip', text: 'The code is on screen.', screen_text: rich });
    assert.equal(nodes[0].children[0].textContent, rich);
    assert.equal(nodes[0].children[0].style.whiteSpace, 'pre-wrap');
    renderTurn({ id: 'legacy-test', role: 'kip', text: 'Hello.' });
    assert.equal(nodes[1].children[0].textContent, 'Hello.');
  } finally {
    globalThis.document = previous;
  }
});
