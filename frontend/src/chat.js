/**
 * Talk to Kip — renders the conversation, streams replies from chat_delta events,
 * and sends messages to POST /api/chat. Replies arrive over the shared WebSocket.
 */
import { api } from './api.js';

const $ = (id) => document.getElementById(id);
const seen = new Set();
const streaming = new Map(); // reply_id -> bubble element

function scrollToEnd(log, force = false) {
  const nearBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 80;
  if (force || nearBottom) log.scrollTop = log.scrollHeight;
}

function bubble(role, extra = '') {
  const node = document.createElement('div');
  node.className = `chat-bubble ${role} ${extra}`.trim();
  const text = document.createElement('p');
  node.append(text);
  $('chat-empty')?.remove();
  $('chat-log').append(node);
  return node;
}

function addSources(node, sources) {
  const valid = (sources || []).filter(source => {
    try { return ['http:', 'https:'].includes(new URL(source.url).protocol); } catch { return false; }
  });
  if (!valid.length) return;
  const list = document.createElement('div');
  list.className = 'chat-sources';
  for (const source of valid) {
    const link = document.createElement('a');
    link.href = source.url; link.textContent = source.title || source.url;
    link.target = '_blank'; link.rel = 'noopener noreferrer';
    list.append(link);
  }
  node.append(list);
}

export function renderTurn(turn) {
  if (!turn || seen.has(turn.id)) return;
  seen.add(turn.id);
  const log = $('chat-log');
  let node = turn.reply_id ? streaming.get(turn.reply_id) : null;
  if (node) streaming.delete(turn.reply_id);
  else node = bubble(turn.role === 'user' ? 'user' : 'kip');
  node.classList.remove('thinking');
  if (turn.source === 'narration') node.classList.add('narration');
  if (turn.error) node.classList.add('error');
  node.querySelector('p').textContent = turn.text;
  if (turn.action?.task) {
    const tag = document.createElement('span');
    tag.className = 'chat-task';
    tag.textContent = `Errand: ${turn.action.task}`;
    node.append(tag);
  }
  addSources(node, turn.sources);
  if (turn.metrics?.first_token_seconds !== undefined) {
    node.title = `First words after ${turn.metrics.first_token_seconds}s · total ${turn.metrics.total_seconds ?? '?'}s`;
  }
  scrollToEnd(log, turn.role === 'user');
}

export function handleChatEvent(event) {
  const d = event.data || {};
  if (event.type === 'chat_message') { renderTurn(d.turn); return true; }
  if (event.type === 'chat_thinking') {
    if (!streaming.has(d.reply_id)) {
      const node = bubble('kip', 'thinking');
      node.querySelector('p').textContent = d.waiting_on_errand
        ? 'Thinking… (the model is also busy with an errand, so this may take a while)'
        : 'Thinking…';
      streaming.set(d.reply_id, node);
      scrollToEnd($('chat-log'));
    }
    return true;
  }
  if (event.type === 'chat_delta') {
    const node = streaming.get(d.reply_id);
    if (node) {
      const text = node.querySelector('p');
      if (node.classList.contains('thinking')) { node.classList.remove('thinking'); text.textContent = ''; }
      text.textContent += d.text;
      scrollToEnd($('chat-log'));
    }
    return true;
  }
  return false;
}

export async function initChat() {
  try {
    const { turns, name } = await api('/chat');
    $('chat-name').textContent = name;
    turns.forEach(renderTurn);
  } catch { /* The WebSocket snapshot still replays recent chat events. */ }

  const form = $('chat-form');
  const input = $('chat-input');
  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const message = input.value.trim();
    if (!message) return;
    input.value = '';
    $('chat-send').disabled = true;
    try { renderTurn(await api('/chat', { message })); }
    catch (error) {
      const node = bubble('kip', 'error');
      node.querySelector('p').textContent = error.message;
      input.value = message;
    } finally { $('chat-send').disabled = false; input.focus(); }
  });
  input.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); form.requestSubmit(); }
  });
  document.querySelectorAll('[data-say]').forEach(button => button.addEventListener('click', () => {
    input.value = button.dataset.say;
    input.focus();
  }));
}
