/**
 * Talk to Kip — renders the conversation, streams replies from chat_delta events,
 * and sends messages to POST /api/chat. Replies arrive over the shared WebSocket.
 *
 * The server's /api/chat history is authoritative: on load and on every reconnect the
 * log is rebuilt from it in server order, so nothing is lost or duplicated.
 *
 * One conversation is open at a time. A new chat gets its id here, before its first message is
 * sent, so the reply streaming in over the socket is recognised as this chat's.
 */
import { api } from './api.js';
import { clockTime, parseRich, safeSources, sourceKey, taskTitle } from './format.js';

const $ = (id) => document.getElementById(id);
const MAX_ITEMS = 200;
const seen = new Set();
const streaming = new Map(); // reply_id -> message row
const reloadHooks = [];
let reloading = null;        // pending reload promise
let queued = [];             // live chat events that arrived during a reload
let sending = false;
let noteTimer;
let conversation = null;     // the open chat's id; null for a new chat not yet sent
let startedHook = () => {};
let missingHook = () => {};

/** A fresh chat id, made where crypto.randomUUID may be missing (plain http on the home network). */
export function newChatId() {
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  return [...bytes].map(byte => byte.toString(16).padStart(2, '0')).join('');
}

export function activeConversation() { return conversation; }
export function onStarted(callback) { startedHook = callback; }
export function onMissing(callback) { missingHook = callback; }

/** Show a chat (an id) or a new, empty one (null). */
export function setConversation(id) {
  if (id === conversation && id !== null) return reloadChat();
  conversation = id;
  globalThis.window?.dispatchEvent(new Event('kip:conversation'));
  return reloadChat();
}

/** The chat an event belongs to, when it names one. */
export function eventConversation(event) {
  const d = event?.data || {};
  return d.session_id || d.turn?.session_id || d.task?.session_id || null;
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function scroller() { return $('chat-scroll') || $('chat-log'); }

function nearBottom() {
  const box = scroller();
  return !box || box.scrollHeight - box.scrollTop - box.clientHeight < 120;
}

/** Keep the newest message in view unless the reader has scrolled up to read history. */
export function scrollToEnd(force = false) {
  const box = scroller();
  if (!box) return;
  if (force || nearBottom()) {
    box.scrollTop = box.scrollHeight;
    const jump = $('jump-latest');
    if (jump) jump.hidden = true;
  } else if ($('jump-latest')) {
    $('jump-latest').hidden = false;
  }
}

function updateEmpty() {
  const empty = $('chat-empty');
  if (empty) empty.hidden = $('chat-log').children.length > 0;
}

/** Add any element (message, lookup card) to the end of the conversation. */
export function appendToLog(node) {
  const log = $('chat-log');
  const stick = nearBottom();
  log.append(node);
  while (log.children.length > MAX_ITEMS) log.firstElementChild.remove();
  if (seen.size > MAX_ITEMS * 3) seen.delete(seen.values().next().value);
  updateEmpty();
  if (stick) scrollToEnd(true);
  return node;
}

function message(role, extra = '') {
  const row = el('div', `msg ${role} ${extra}`.trim());
  const bubble = el('div', 'bubble');
  const text = el('div', 'msg-text');
  bubble.append(text);
  row.append(bubble);
  row.textEl = text;
  row.bubbleEl = bubble;
  return appendToLog(row);
}

/** Build safe rich text: every piece of model output goes through textContent. */
export function renderRich(target, text) {
  target.replaceChildren();
  for (const block of parseRich(text)) {
    if (block.type === 'code') {
      const pre = el('pre', 'code-block');
      if (block.lang) pre.dataset.lang = block.lang;
      pre.append(el('code', '', block.text));
      target.append(pre);
      continue;
    }
    const paragraph = el('p');
    for (const span of block.spans) {
      if (span.type === 'text') paragraph.append(document.createTextNode(span.text));
      else if (span.type === 'code') paragraph.append(el('code', 'inline-code', span.text));
      else if (span.type === 'bold') paragraph.append(el('strong', '', span.text));
      else if (span.type === 'link') {
        const link = el('a', '', span.text);
        link.href = span.href; link.target = '_blank'; link.rel = 'noopener noreferrer';
        paragraph.append(link);
      }
    }
    target.append(paragraph);
  }
}

const VISIBLE_SOURCES = 4;

/** Small link chips for the pages an answer came from. */
export function sourceChips(sources) {
  const valid = safeSources(sources);
  if (!valid.length) return null;
  const list = el('div', 'sources');
  list.dataset.sourceKey = sourceKey(valid);
  list.setAttribute('aria-label', 'Sources');
  valid.forEach((source, index) => {
    const link = el('a', 'source-chip');
    link.href = source.url; link.target = '_blank'; link.rel = 'noopener noreferrer';
    link.title = source.title === source.host ? source.url : `${source.title} — ${source.url}`;
    link.append(el('span', 'source-host', source.host));
    if (source.title !== source.host) link.append(el('span', 'source-title', source.title));
    // A long list buries the answer: show the first few, the rest on request.
    if (index >= VISIBLE_SOURCES) link.hidden = true;
    list.append(link);
  });
  if (valid.length > VISIBLE_SOURCES) {
    const more = el('button', 'source-chip source-more', `+${valid.length - VISIBLE_SOURCES} more`);
    more.type = 'button';
    more.addEventListener('click', () => {
      list.querySelectorAll('.source-chip[hidden]').forEach(link => { link.hidden = false; });
      more.remove();
    });
    list.append(more);
  }
  return list;
}

/** True when chips for exactly these sources are already in the conversation. */
export function sourcesShown(key) {
  if (!key) return false;
  return [...($('chat-log')?.querySelectorAll('.sources') || [])].some(node => node.dataset.sourceKey === key);
}

function setMeta(row, turn) {
  const time = clockTime(turn.at);
  if (!time) return;
  row.querySelector('.msg-meta')?.remove();
  const meta = el('div', 'msg-meta');
  const stamp = el('time', '', time);
  stamp.dateTime = turn.at;
  meta.append(stamp);
  if (turn.metrics?.first_token_seconds !== undefined) {
    meta.title = `First words after ${turn.metrics.first_token_seconds}s · total ${turn.metrics.total_seconds ?? '?'}s`;
  }
  row.append(meta);
}

export function renderTurn(turn) {
  if (!turn || seen.has(turn.id)) return;
  seen.add(turn.id);
  const user = turn.role === 'user';
  const narration = ['narration', 'correction', 'clarification'].includes(turn.source);
  // Narration that is not a lookup result ("Start now: …") is a quiet status line.
  const note = !user && narration && !turn.outcome && !safeSources(turn.sources).length;
  let row = turn.reply_id ? streaming.get(turn.reply_id) : null;
  if (row) {
    streaming.delete(turn.reply_id);
    row.className = 'msg kip';
  } else {
    row = message(user ? 'user' : note ? 'note' : 'kip');
  }
  row.querySelector('.typing')?.remove();
  if (turn.outcome === 'failed') row.classList.add('failed');
  if (turn.error) row.classList.add('error');
  const text = turn.screen_text ?? turn.content ?? turn.text ?? '';
  if (user || note) row.textEl.textContent = text;
  else renderRich(row.textEl, text);
  if (turn.action?.task) {
    row.bubbleEl.append(el('span', 'msg-tag', `Looking into: ${taskTitle({ goal: turn.action.task })}`));
  }
  const key = sourceKey(turn.sources);
  if (key && !sourcesShown(key)) {
    const chips = sourceChips(turn.sources);
    if (chips) row.bubbleEl.append(chips);
  }
  setMeta(row, turn);
  scrollToEnd(user);
}

/** Remove typing/partial bubbles that will never be finished by this page. */
function releaseStreaming() {
  for (const row of streaming.values()) row.remove();
  streaming.clear();
}

function typing(row, waitingOnErrand) {
  const dots = el('div', 'typing');
  dots.append(el('i'), el('i'), el('i'), el('span', 'sr-only', 'Kip is typing'));
  row.textEl.replaceChildren(dots);
  if (waitingOnErrand) row.textEl.append(el('p', 'typing-note', 'Finishing a lookup first — this may take a moment.'));
}

export function isChatEvent(event) {
  return typeof event?.type === 'string' && event.type.startsWith('chat_');
}

export function handleChatEvent(event) {
  if (!isChatEvent(event)) return false;
  if (reloading) { queued.push(event); return true; }
  const owner = eventConversation(event);
  if (owner && owner !== conversation) return true; // another of your chats: shown when you open it
  const d = event.data || {};
  if (event.type === 'chat_message') renderTurn(d.turn);
  else if (event.type === 'chat_thinking') {
    if (!streaming.has(d.reply_id)) {
      const row = message('kip', 'thinking');
      typing(row, d.waiting_on_errand);
      streaming.set(d.reply_id, row);
      scrollToEnd();
    }
  } else if (event.type === 'chat_delta') {
    const row = streaming.get(d.reply_id);
    if (row) {
      if (row.classList.contains('thinking')) {
        row.classList.remove('thinking');
        row.classList.add('streaming');
        row.textEl.replaceChildren();
      }
      row.textEl.append(document.createTextNode(d.text ?? ''));
      scrollToEnd();
    }
  } else if (event.type === 'chat_reply_ended') {
    const row = streaming.get(d.reply_id);
    if (row) {
      streaming.delete(d.reply_id);
      if (row.classList.contains('thinking')) { row.remove(); updateEmpty(); }
      else {
        row.classList.remove('streaming');
        row.classList.add('error');
        row.bubbleEl.append(el('span', 'msg-tag', 'Reply interrupted'));
      }
    }
  }
  return true;
}

/** Run after every rebuild of the log (e.g. to restore a running lookup card). */
export function onChatReload(callback) { reloadHooks.push(callback); }

export function setName(name) {
  const value = String(name || 'Kip');
  document.querySelectorAll('.kip-name').forEach(node => { node.textContent = value; });
  if ($('kip-name')) $('kip-name').textContent = value;
  if ($('kip-monogram')) $('kip-monogram').textContent = value.charAt(0).toUpperCase();
  document.querySelector('.empty-monogram span')?.replaceChildren(value.charAt(0).toUpperCase());
  if ($('chat-input')) $('chat-input').placeholder = `Message ${value}…`;
  document.title = value;
}

/** Rebuild the log from the server's history (load and every reconnect). */
export function reloadChat() {
  if (reloading) return reloading;
  queued = [];
  const showing = conversation;
  reloading = (async () => {
    // Always yield first: a new chat needs no fetch, and finishing synchronously would clear
    // `reloading` before it is even assigned, leaving every later event queued forever.
    await null;
    try {
      const { turns, name } = showing
        ? await api(`/chat?conversation_id=${encodeURIComponent(showing)}`)
        : { turns: [], name: null };
      if (showing !== conversation) return; // another chat was opened meanwhile; it reloads itself
      if (name) setName(name);
      releaseStreaming();
      seen.clear();
      $('chat-log').replaceChildren();
      (turns || []).forEach(renderTurn);
      updateEmpty();
      reloadHooks.forEach(hook => hook());
      scrollToEnd(true);
    } catch (error) {
      // A deleted chat, or one that is not yours: start a new one. Otherwise keep what is shown;
      // the next reconnect retries.
      if (error?.status === 404 && showing === conversation) missingHook();
    }
    finally {
      const pending = queued;
      reloading = null;
      queued = [];
      pending.forEach(handleChatEvent); // events that arrived meanwhile, deduplicated by id
    }
  })();
  return reloading;
}

export function showNote(text = '', tone = 'error') {
  const note = $('error');
  if (!note) return;
  clearTimeout(noteTimer);
  note.textContent = text;
  note.dataset.tone = tone;
  note.hidden = !text;
  if (text && tone !== 'sticky') noteTimer = setTimeout(() => { note.hidden = true; }, 8000);
}

function autosize(input) {
  input.style.height = 'auto';
  input.style.height = `${Math.min(input.scrollHeight, 200)}px`;
}

function updateSend() {
  const input = $('chat-input');
  $('chat-send').disabled = sending || !input.value.trim();
}

async function send(text) {
  const input = $('chat-input');
  const message = text.trim();
  if (!message || sending) return;
  sending = true;
  input.value = '';
  autosize(input);
  updateSend();
  $('chat-form').setAttribute('aria-busy', 'true');
  const fresh = conversation === null;
  if (fresh) {
    conversation = newChatId();
    seen.clear();
  }
  const sentTo = conversation;
  try {
    renderTurn(await api('/chat', { message, conversation_id: sentTo }));
    globalThis.window?.dispatchEvent(new Event('kip:conversation'));
    if (fresh) startedHook(sentTo);
    showNote('');
    scrollToEnd(true);
  } catch (error) {
    if (fresh && conversation === sentTo) conversation = null;
    // 409 and friends: the backend explains why it cannot take the message right now.
    showNote(error.message || 'Kip could not take that message. Please try again.');
    if (!input.value) input.value = message;
    autosize(input);
  } finally {
    sending = false;
    $('chat-form').removeAttribute('aria-busy');
    updateSend();
    input.focus();
  }
}

export function initChat() {
  const form = $('chat-form');
  const input = $('chat-input');
  form.addEventListener('submit', (event) => { event.preventDefault(); send(input.value); });
  input.addEventListener('keydown', (event) => {
    // Never send while an input method (e.g. Chinese/Japanese) is still composing.
    if (event.isComposing || event.keyCode === 229) return;
    if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); form.requestSubmit(); }
  });
  input.addEventListener('input', () => { autosize(input); updateSend(); });
  document.querySelectorAll('[data-say]').forEach(button => button.addEventListener('click', () => send(button.dataset.say)));
  scroller().addEventListener('scroll', () => { if (nearBottom()) $('jump-latest').hidden = true; }, { passive: true });
  $('jump-latest').addEventListener('click', () => scrollToEnd(true));
  updateEmpty();
  updateSend();
}
