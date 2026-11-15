/**
 * "About you" tab — what Kip remembers (GET /api/memories?q=&kind=&limit=).
 * Without a query it lists the most recent memories; a query searches by meaning.
 */
import { api } from './api.js';
import { relativeTime } from './format.js';

const KINDS = [
  ['', 'All'], ['fact', 'Facts'], ['preference', 'Likes'], ['episode', 'Moments'],
  ['conversation', 'Chats'], ['skill', 'Skills'],
];
const KIND_LABELS = { fact: 'Fact', preference: 'Preference', episode: 'Moment', conversation: 'Chat', skill: 'Skill' };

let timer;
let latestRequest = 0;
let kind = '';

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function row(memory) {
  const item = el('article', 'memory-card');
  const top = el('div', 'memory-top');
  top.append(el('span', `memory-kind kind-${memory.kind}`, KIND_LABELS[memory.kind] || memory.kind));
  const when = relativeTime(memory.created_at);
  if (when) {
    const stamp = el('time', 'memory-time', when);
    stamp.dateTime = memory.created_at;
    top.append(stamp);
  }
  item.append(top, el('p', 'memory-text', memory.text));
  const importance = Number(memory.importance);
  if (memory.about?.length || Number.isFinite(importance)) {
    const foot = el('div', 'memory-foot');
    if (memory.about?.length) {
      const tags = el('div', 'memory-tags');
      for (const tag of memory.about.slice(0, 5)) tags.append(el('span', 'memory-tag', tag));
      foot.append(tags);
    }
    if (Number.isFinite(importance)) {
      const meter = el('span', 'importance');
      const level = Math.max(1, Math.min(3, Math.ceil(importance * 3)));
      meter.title = `Importance ${importance.toFixed(2)}`;
      meter.setAttribute('aria-label', meter.title);
      for (let i = 1; i <= 3; i += 1) meter.append(el('i', i <= level ? 'on' : ''));
      foot.append(meter);
    }
    item.append(foot);
  }
  return item;
}

function emptyState(panel, title, text) {
  const box = el('div', 'panel-empty');
  box.append(el('strong', '', title), el('span', '', text));
  panel.querySelector('#memory-results').replaceChildren(box);
}

export function initMemories(panel) {
  panel.replaceChildren();
  const intro = el('div', 'panel-intro');
  intro.append(el('p', 'panel-lede', 'Things Kip has learned from your conversations.'));
  const search = el('div', 'search-field');
  search.innerHTML = '<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><circle cx="11" cy="11" r="6.5" fill="none" stroke="currentColor" stroke-width="1.7"/><path d="m16 16 4 4" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/></svg>';
  const input = el('input');
  input.id = 'memory-query'; input.type = 'search';
  input.placeholder = 'Search memories';
  input.setAttribute('aria-label', 'Search memories');
  search.append(input);
  const filters = el('div', 'filter-chips');
  filters.setAttribute('role', 'group');
  filters.setAttribute('aria-label', 'Filter by kind');
  for (const [value, label] of KINDS) {
    const chip = el('button', 'filter-chip', label);
    chip.type = 'button';
    chip.setAttribute('aria-pressed', String(value === kind));
    chip.addEventListener('click', () => {
      kind = value;
      filters.querySelectorAll('.filter-chip').forEach(other => other.setAttribute('aria-pressed', String(other === chip)));
      loadMemories(panel);
    });
    filters.append(chip);
  }
  intro.append(search, filters);
  const results = el('div', 'memory-results');
  results.id = 'memory-results';
  results.setAttribute('aria-live', 'polite');
  panel.append(intro, results);
  input.addEventListener('input', () => {
    clearTimeout(timer);
    timer = setTimeout(() => loadMemories(panel), 250);
  });
}

export async function loadMemories(panel) {
  const query = panel.querySelector('#memory-query')?.value.trim() || '';
  const results = panel.querySelector('#memory-results');
  if (!results) return;
  const request = ++latestRequest;
  const params = new URLSearchParams({ limit: '40' });
  if (query) params.set('q', query);
  if (kind) params.set('kind', kind);
  results.setAttribute('aria-busy', 'true');
  try {
    const memories = await api(`/memories?${params}`);
    if (request !== latestRequest) return; // a newer search has started; never show stale results
    if (!memories.length) {
      emptyState(panel, query || kind ? 'No matches' : 'Nothing remembered yet',
        query || kind ? 'Try a different search or filter.' : 'Tell Kip about yourself and it will remember.');
    } else {
      results.replaceChildren(...memories.map(row));
    }
  } catch {
    if (request === latestRequest) emptyState(panel, 'Could not load memories', 'Check that the backend is running.');
  } finally {
    if (request === latestRequest) results.removeAttribute('aria-busy');
  }
}
