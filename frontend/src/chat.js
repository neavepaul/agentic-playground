/**
 * Talk to Kip — renders the conversation, streams replies from chat_delta events,
 * and sends messages to POST /api/chat. Replies arrive over the shared WebSocket.
 *
 * The server's /api/chat history is authoritative: on load, reconnect and reset the
 * log is rebuilt from it in server order, so nothing is lost or duplicated.
 */
import { api } from './api.js';

const $ = (id) => document.getElementById(id);
const MAX_BUBBLES = 200;
const seen = new Set();
const streaming = new Map(); // reply_id -> bubble element
let reloading = null;        // pending reload promise
let queued = [];             // live chat events that arrived during a reload

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
  const log = $('chat-log');
  log.append(node);
  while (log.children.length > MAX_BUBBLES) log.firstElementChild.remove();
  if (seen.size > MAX_BUBBLES * 3) seen.delete(seen.values().next().value);
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
  if (turn.source === 'correction' || turn.source === 'clarification') node.classList.add('narration');
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

/** Remove "Thinking…"/partial bubbles that will never be finished by this page. */
function releaseStreaming(note = '') {
  for (const node of streaming.values()) {
    if (note && node.querySelector('p').textContent && !node.classList.contains('thinking')) {
      node.classList.add('error');
      node.querySelector('p').textContent += ` ${note}`;
    } else {
      node.remove();
    }
  }
  streaming.clear();
}

export function isChatEvent(event) {
  return event.type.startsWith('chat_');
}

export function handleChatEvent(event) {
  if (!isChatEvent(event)) return false;
  if (reloading) { queued.push(event); return true; }
  const d = event.data || {};
  if (event.type === 'chat_message') renderTurn(d.turn);
  else if (event.type === 'chat_thinking') {
    if (!streaming.has(d.reply_id)) {
      const node = bubble('kip', 'thinking');
      node.querySelector('p').textContent = d.waiting_on_errand
        ? 'Thinking… (the model is also busy with an errand, so this may take a while)'
        : 'Thinking…';
      streaming.set(d.reply_id, node);
      scrollToEnd($('chat-log'));
    }
  } else if (event.type === 'chat_delta') {
    const node = streaming.get(d.reply_id);
    if (node) {
      const text = node.querySelector('p');
      if (node.classList.contains('thinking')) { node.classList.remove('thinking'); text.textContent = ''; }
      text.textContent += d.text;
      scrollToEnd($('chat-log'));
    }
  } else if (event.type === 'chat_reply_ended') {
    const node = streaming.get(d.reply_id);
    if (node) {
      streaming.delete(d.reply_id);
      if (node.classList.contains('thinking')) node.remove();
      else { node.classList.add('error'); node.querySelector('p').textContent += ' (reply interrupted)'; }
    }
  }
  return true;
}

/** Rebuild the log from the server's history (load, reconnect, reset). */
export function reloadChat() {
  if (reloading) return reloading;
  queued = [];
  reloading = (async () => {
    try {
      const { turns, name } = await api('/chat');
      $('chat-name').textContent = name;
      releaseStreaming();
      seen.clear();
      $('chat-log').replaceChildren();
      turns.forEach(renderTurn);
      if (!turns.length) {
        const empty = document.createElement('p');
        empty.id = 'chat-empty'; empty.className = 'subtle';
        empty.textContent = 'Say hello, ask a question, or ask for help around the house.';
        $('chat-log').append(empty);
      }
    } catch { /* keep what is shown; the next reconnect retries */ }
    finally {
      const pending = queued;
      reloading = null;
      queued = [];
      pending.forEach(handleChatEvent); // events that arrived meanwhile, deduplicated by id
    }
  })();
  return reloading;
}

export function resetChat() {
  releaseStreaming();
  return reloadChat();
}

/** Offer the people in the world as possible speakers ("Who's talking?"). */
export function setSpeakers(people) {
  const select = $('chat-speaker');
  if (!select) return;
  const current = select.value;
  const options = [['', 'Someone (unknown)'],
    ...Object.entries(people || {}).map(([id, person]) => [id, person.name || id])];
  const signature = options.map(([id]) => id).join('|');
  if (select.dataset.signature === signature) return;
  select.dataset.signature = signature;
  select.replaceChildren(...options.map(([id, label]) => {
    const option = document.createElement('option');
    option.value = id; option.textContent = label;
    return option;
  }));
  select.value = options.some(([id]) => id === current) ? current : '';
}

export function initChat() {
  reloadChat();
  const form = $('chat-form');
  const input = $('chat-input');
  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const message = input.value.trim();
    if (!message) return;
    input.value = '';
    $('chat-send').disabled = true;
    const speaker = $('chat-speaker')?.value || undefined;
    try { renderTurn(await api('/chat', speaker ? { message, speaker } : { message })); }
    catch (error) {
      const node = bubble('kip', 'error');
      node.querySelector('p').textContent = error.message;
      input.value = message;
    } finally { $('chat-send').disabled = false; input.focus(); }
  });
  input.addEventListener('keydown', (event) => {
    // Never send while an input method (e.g. Chinese/Japanese) is still composing.
    if (event.isComposing || event.keyCode === 229) return;
    if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); form.requestSubmit(); }
  });
  document.querySelectorAll('[data-say]').forEach(button => button.addEventListener('click', () => {
    input.value = button.dataset.say;
    input.focus();
  }));
}
