/**
 * Pure helpers shared by the UI: no DOM access, so they are unit-tested in Node.
 */

const LOOKUP_PREFIX = 'Look up online: ';
const PROVIDERS = { groq: 'Groq', ollama: 'Ollama' };

/** Keep only http(s) sources, de-duplicated by URL, with a readable host name. */
export function safeSources(sources) {
  const seen = new Set();
  const result = [];
  for (const source of Array.isArray(sources) ? sources : []) {
    let url;
    try { url = new URL(source?.url); } catch { continue; }
    if (!['http:', 'https:'].includes(url.protocol) || seen.has(url.href)) continue;
    seen.add(url.href);
    const host = url.hostname.replace(/^www\./, '');
    result.push({ url: url.href, host, title: (source.title || '').trim() || host });
  }
  return result;
}

/** A stable identity for a set of sources, so the same chips are never shown twice. */
export function sourceKey(sources) {
  return safeSources(sources).map(source => source.url).sort().join('\n');
}

/** "Look up online: who won? (following up on: cricket)" -> "who won?" */
export function taskTitle(task) {
  let goal = String(task?.goal || '').trim();
  if (goal.startsWith(LOOKUP_PREFIX)) goal = goal.slice(LOOKUP_PREFIX.length);
  goal = goal.replace(/\s+\((following up on|they corrected):[\s\S]*\)$/, '');
  return goal || 'Looking something up';
}

export function clockTime(iso) {
  const date = new Date(iso);
  if (!iso || Number.isNaN(date.getTime())) return '';
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

export function relativeTime(iso, now = Date.now()) {
  const date = new Date(iso);
  if (!iso || Number.isNaN(date.getTime())) return '';
  const seconds = Math.max(0, (now - date.getTime()) / 1000);
  if (seconds < 45) return 'just now';
  if (seconds < 3600) return `${Math.round(seconds / 60)} min ago`;
  if (seconds < 86400) return `${Math.round(seconds / 3600)} h ago`;
  if (seconds < 172800) return 'yesterday';
  if (seconds < 7 * 86400) return `${Math.floor(seconds / 86400)} days ago`;
  return date.toLocaleDateString([], { day: 'numeric', month: 'short' });
}

export function providerLabel(provider) {
  const key = String(provider || '').toLowerCase();
  if (!key) return 'Model';
  return PROVIDERS[key] || key.charAt(0).toUpperCase() + key.slice(1);
}

/**
 * Turn the `llm` block of GET /api/health into a short status line.
 * state: "ready" | "warn" | "offline"
 */
export function modelStatus(llm) {
  if (!llm) return { state: 'offline', label: 'Backend offline', detail: '' };
  const provider = providerLabel(llm.provider);
  const model = llm.model ? String(llm.model) : '';
  const named = model ? `${provider} · ${model}` : provider;
  if (llm.mode === 'test') return { state: 'warn', label: `${provider} · test mode`, detail: '' };
  if (llm.reachable === false) return { state: 'offline', label: `${provider} unreachable`, detail: model };
  if (llm.model_available === false) return { state: 'warn', label: `${named}`, detail: 'Model not found' };
  return { state: 'ready', label: named, detail: '' };
}

export function moodPhrase(mood) {
  return mood ? `Feeling ${String(mood).toLowerCase()}` : '';
}

const INLINE = /\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)|`([^`\n]+)`|\*\*([^*\n]+)\*\*|(https?:\/\/[^\s<>()]*[^\s<>().,;:!?'"])/g;

function inlineSpans(text) {
  const spans = [];
  const push = (span) => {
    const last = spans[spans.length - 1];
    if (span.type === 'text' && last?.type === 'text') last.text += span.text;
    else if (span.text) spans.push(span);
  };
  for (const line of text.split(/(\n)/)) {
    const heading = /^#{1,6}\s+(.+)$/.exec(line);
    if (heading) { push({ type: 'bold', text: heading[1] }); continue; }
    let last = 0;
    for (const match of line.matchAll(INLINE)) {
      push({ type: 'text', text: line.slice(last, match.index) });
      if (match[1]) push({ type: 'link', text: match[1], href: match[2] });
      else if (match[3]) push({ type: 'code', text: match[3] });
      else if (match[4]) push({ type: 'bold', text: match[4] });
      else push({ type: 'link', text: match[5], href: match[5] });
      last = match.index + match[0].length;
    }
    push({ type: 'text', text: line.slice(last) });
  }
  return spans;
}

/**
 * Split a reply into blocks a renderer can build with textContent only (never HTML):
 * fenced code blocks, and text with inline code, bold, headings-as-bold and http(s) links.
 */
export function parseRich(text) {
  const source = String(text ?? '');
  const blocks = [];
  const fence = /```([\w+#.-]*)[^\S\n]*\n?([\s\S]*?)```/g;
  let last = 0;
  const addText = (chunk) => {
    const trimmed = chunk.replace(/^\n+|\n+$/g, '');
    if (trimmed) blocks.push({ type: 'text', spans: inlineSpans(trimmed) });
  };
  for (const match of source.matchAll(fence)) {
    addText(source.slice(last, match.index));
    blocks.push({ type: 'code', lang: match[1] || '', text: match[2].replace(/\n$/, '') });
    last = match.index + match[0].length;
  }
  addText(source.slice(last));
  return blocks;
}

const TOOL_NAMES ={ web_search: 'Searched the web', read_webpage: 'Read a page', run_python: 'Ran Python' };

/**
 * Plain-language Activity entry for a bus event, or null if it should not be shown.
 * `detail: true` marks internal steps that are hidden unless the user asks for them.
 */
export function describeEvent(event) {
  const d = event?.data || {};
  const o = d.observation || {};
  const task = d.task;
  switch (event?.type) {
    case 'task_started':
      return { tone: 'info', label: 'Lookup', text: `Started: ${taskTitle(task || { goal: d.summary })}` };
    case 'task_completed': {
      const count = safeSources(task?.answer?.sources).length;
      return { tone: 'success', label: 'Lookup', text: `Finished: ${taskTitle(task || { goal: d.summary })}`,
        note: count ? `${count} source${count === 1 ? '' : 's'}` : '' };
    }
    case 'task_failed':
      return { tone: 'error', label: 'Lookup', text: `Could not finish: ${taskTitle(task || { goal: '' })}`,
        note: d.summary || task?.summary || '' };
    case 'task_cancelled':
      return { tone: 'muted', label: 'Lookup', text: `Stopped: ${taskTitle(task || { goal: '' })}`,
        note: d.summary || '' };
    case 'task_needs_permission':
      return { tone: 'warn', label: 'Permission', text: d.summary || 'Kip needs a go-ahead first.' };
    case 'tool_refused':
      return { tone: 'warn', label: 'Blocked', text: d.summary || `Not permitted: ${d.tool}`, detail: true };
    case 'tool_failed':
      return { tone: 'error', label: 'Tool', text: `${d.tool || 'A tool'} failed`, note: d.error || '', detail: true };
    case 'capability_observed':
      if (d.tool === 'web_search') {
        const count = Array.isArray(o.results) ? o.results.length : 0;
        return { tone: 'muted', label: 'Search', text: `Searched for “${o.query || '…'}”`,
          note: `${count} result${count === 1 ? '' : 's'}`, detail: true };
      }
      if (d.tool === 'read_webpage') return { tone: 'muted', label: 'Read', text: o.title || o.url || 'Read a page', detail: true };
      if (d.tool === 'run_python') return { tone: 'muted', label: 'Python', text: 'Ran a calculation', detail: true };
      if (d.tool === 'finish_answer') return null;
      return d.tool ? { tone: 'muted', label: 'Tool', text: TOOL_NAMES[d.tool] || `Used ${d.tool}`, detail: true } : null;
    case 'critic_review':
      return { tone: d.approved ? 'muted' : 'warn', label: 'Check',
        text: d.approved ? 'Double-checked the answer' : 'Answer needed more work', note: d.suggestion || '', detail: true };
    case 'agent_message':
      return d.summary ? { tone: 'muted', label: 'Step', text: d.summary, detail: true } : null;
    case 'persona_updated':
      return null; // shown in the header and the Kip tab instead
    default:
      return null;
  }
}
