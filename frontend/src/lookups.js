/**
 * Lookups in the conversation — a compact card while Kip searches the web or runs
 * Python for a task (task_started → task_completed / failed / cancelled), with a
 * cancel button while it runs and source chips once it has an answer.
 */
import { api } from './api.js';
import { appendToLog, onChatReload, scrollToEnd, showNote, sourceChips, sourcesShown } from './chat.js';
import { sourceKey, taskTitle } from './format.js';

const cards = new Map(); // task id -> card element
let current = null;      // latest task the server told us about

const STATUS = {
  running: 'Looking it up…',
  completed: 'Looked it up',
  failed: 'Couldn’t finish the lookup',
  cancelled: 'Lookup stopped',
};

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function icon(status) {
  const node = el('span', 'lookup-icon');
  node.setAttribute('aria-hidden', 'true');
  if (status === 'running') node.append(el('span', 'spinner'));
  else node.innerHTML = status === 'completed'
    ? '<svg viewBox="0 0 16 16" width="14" height="14"><path d="m3.5 8.5 3 3 6-7" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>'
    : '<svg viewBox="0 0 16 16" width="14" height="14"><path d="m4.5 4.5 7 7m0-7-7 7" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>';
  return node;
}

async function cancel(task, button) {
  button.disabled = true;
  button.textContent = 'Stopping…';
  try { update(await api(`/tasks/${encodeURIComponent(task.id)}/cancel`, {})); }
  catch (error) {
    showNote(error.message);
    button.disabled = false;
    button.textContent = 'Stop';
  }
}

function fill(card, task) {
  const status = STATUS[task.status] ? task.status : 'running';
  card.dataset.status = status;
  const head = el('div', 'lookup-head');
  head.append(icon(status));
  const text = el('div', 'lookup-text');
  text.append(el('span', 'lookup-status', STATUS[status]), el('span', 'lookup-goal', taskTitle(task)));
  head.append(text);
  if (status === 'running') {
    const stop = el('button', 'lookup-cancel', 'Stop');
    stop.type = 'button';
    stop.setAttribute('aria-label', `Stop looking up: ${taskTitle(task)}`);
    stop.addEventListener('click', () => cancel(task, stop));
    head.append(stop);
  }
  const parts = [head];
  if (status === 'failed' && task.summary) parts.push(el('p', 'lookup-summary', task.summary));
  if (status === 'completed') {
    const key = sourceKey(task.answer?.sources);
    if (key && !sourcesShown(key)) {
      const chips = sourceChips(task.answer.sources);
      if (chips) parts.push(chips);
    }
  }
  card.replaceChildren(...parts);
}

/** Apply a TaskPublic from the snapshot or a task_* event. */
export function update(task) {
  if (!task?.id) return;
  current = task;
  let card = cards.get(task.id);
  if (!card) {
    // Only a lookup this page sees running gets a card; finished ones speak through Kip's reply.
    if (task.status !== 'running') return;
    card = el('div', 'lookup');
    card.setAttribute('role', 'status');
    cards.set(task.id, card);
    fill(card, task);
    appendToLog(card);
    return;
  }
  if (card.dataset.status === task.status) return;
  fill(card, task);
  if (!card.isConnected) cards.delete(task.id);
  scrollToEnd();
}

export function initLookups() {
  // The chat log is rebuilt from history on reconnect; keep a still-running lookup visible.
  onChatReload(() => {
    cards.clear();
    if (current?.status === 'running') update(current);
  });
}
