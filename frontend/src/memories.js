/**
 * Memories tab — search Kip's long-term memory (GET /api/memories?q=).
 * Without a query it lists the most recent memories.
 */
import { api } from './api.js';

let timer;

function row(memory) {
  const item = document.createElement('article');
  item.className = 'memory-item';
  const top = document.createElement('div');
  top.className = 'memory-top';
  const kind = document.createElement('span');
  kind.className = `memory-kind ${memory.kind}`;
  kind.textContent = memory.kind;
  const meta = document.createElement('span');
  meta.className = 'subtle';
  meta.textContent = `importance ${memory.importance.toFixed(2)}${memory.score !== undefined ? ` · match ${memory.score.toFixed(2)}` : ''}`;
  top.append(kind, meta);
  const text = document.createElement('p');
  text.textContent = memory.text;
  item.append(top, text);
  if (memory.about?.length) {
    const tags = document.createElement('span');
    tags.className = 'memory-tags';
    tags.textContent = memory.about.join(' · ');
    item.append(tags);
  }
  return item;
}

export function initMemories(panel) {
  panel.innerHTML = `
    <div class="memory-search">
      <input id="memory-query" type="search" placeholder="Search memories — e.g. “what does Neave like?”" aria-label="Search memories">
    </div>
    <div id="memory-results" class="memory-results"></div>`;
  panel.querySelector('#memory-query').addEventListener('input', () => {
    clearTimeout(timer);
    timer = setTimeout(() => loadMemories(panel), 250);
  });
}

export async function loadMemories(panel) {
  const query = panel.querySelector('#memory-query')?.value.trim() || '';
  const results = panel.querySelector('#memory-results');
  try {
    const memories = await api(`/memories?limit=30${query ? `&q=${encodeURIComponent(query)}` : ''}`);
    results.replaceChildren(...memories.map(row));
    if (!memories.length) {
      results.innerHTML = '<p class="beliefs-empty">No memories yet. Talk to Kip or give it an errand.</p>';
    }
  } catch {
    results.innerHTML = '<p class="beliefs-empty">Could not load memories.</p>';
  }
}
