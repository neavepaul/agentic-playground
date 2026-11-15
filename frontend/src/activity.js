/**
 * Activity tab — recent lookups and notable events in plain language, newest first.
 * Internal steps (searches, tool errors, reviews) stay hidden unless asked for.
 */
import { clockTime, describeEvent } from './format.js';

const MAX_ENTRIES = 150;
const DETAILS_KEY = 'kip.activity.details';
const seen = new Set();
let panel;
let list;
let onNotable = () => {};

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function updateEmpty() {
  const visible = [...list.children].some(item => !item.classList.contains('detail')
    || panel.classList.contains('show-details'));
  panel.querySelector('.activity-empty').hidden = visible;
}

export function initActivity(target, notify) {
  panel = target;
  onNotable = notify || onNotable;
  let details = false;
  try { details = localStorage.getItem(DETAILS_KEY) === '1'; } catch { /* private mode */ }
  panel.classList.toggle('show-details', details);
  const head = el('div', 'panel-intro');
  head.append(el('p', 'panel-lede', 'What Kip has been up to — lookups, results and anything that needs you.'));
  const toggle = el('label', 'switch');
  const box = el('input');
  box.type = 'checkbox'; box.checked = details;
  box.addEventListener('change', () => {
    panel.classList.toggle('show-details', box.checked);
    try { localStorage.setItem(DETAILS_KEY, box.checked ? '1' : '0'); } catch { /* private mode */ }
    updateEmpty();
  });
  toggle.append(box, el('span', 'switch-track'), el('span', 'switch-label', 'Show internal steps'));
  head.append(toggle);
  list = el('ol', 'activity-list');
  const empty = el('div', 'panel-empty activity-empty');
  empty.append(el('strong', '', 'Nothing yet'), el('span', '', 'When Kip looks something up, it shows here.'));
  panel.replaceChildren(head, empty, list);
}

export function clearActivity() {
  seen.clear();
  list?.replaceChildren();
  if (list) updateEmpty();
}

export function addEvent(event) {
  if (!list || !event?.id || seen.has(event.id)) return;
  const entry = describeEvent(event);
  if (!entry) return;
  seen.add(event.id);
  if (seen.size > MAX_ENTRIES * 4) seen.delete(seen.values().next().value);
  const item = el('li', `activity-item tone-${entry.tone}${entry.detail ? ' detail' : ''}`);
  const marker = el('span', 'activity-marker');
  marker.setAttribute('aria-hidden', 'true');
  const body = el('div', 'activity-body');
  const top = el('div', 'activity-top');
  top.append(el('span', 'activity-label', entry.label));
  const time = clockTime(event.timestamp);
  if (time) {
    const stamp = el('time', 'activity-time', time);
    stamp.dateTime = event.timestamp;
    top.append(stamp);
  }
  body.append(top, el('p', 'activity-text', entry.text));
  if (entry.note) body.append(el('p', 'activity-note', entry.note));
  item.append(marker, body);
  list.prepend(item);
  while (list.children.length > MAX_ENTRIES) list.lastElementChild.remove();
  updateEmpty();
  if (!entry.detail) onNotable(entry);
}
