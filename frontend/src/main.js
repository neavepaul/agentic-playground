import './style.css';
import { api } from './api.js';
import { createScene } from './scene.js';
import { connectEvents } from './websocket.js';
import { addEvent, clearFeed, getTask, setAgent, setConnection, setMemoryOpen, setTask, setWorld, showError } from './ui.js';
import { renderBeliefs } from './beliefs.js';
import { renderPersona } from './persona.js';
import { handleChatEvent, initChat, isChatEvent, reloadChat, resetChat, setSpeakers } from './chat.js';
import { initMemories, loadMemories } from './memories.js';

let scene;
try { scene = createScene(document.getElementById('scene')); }
catch { showError('The 3D view requires WebGL. Enable hardware acceleration or try another browser.'); }
let sequence = -1;
let activeTab = 'activity';
const beliefsPanel = document.getElementById('beliefs-panel');
const personaPanel = document.getElementById('persona-panel');
const memoriesPanel = document.getElementById('memories-panel');
initMemories(memoriesPanel);
initChat();

async function fetchAndRenderBeliefs() {
  try {
    const [data, worldData] = await Promise.all([api('/beliefs'), api('/world')]);
    const people = new Set(Object.keys(worldData?.world?.people ?? {}));
    const rooms = new Set(Object.keys(worldData?.world?.rooms ?? {}));
    renderBeliefs(beliefsPanel, data, people, rooms);
  } catch {
    beliefsPanel.innerHTML = '<p class="beliefs-empty">Could not load beliefs.</p>';
  }
}

async function fetchAndRenderPersona() {
  try { renderPersona(personaPanel, await api('/persona')); }
  catch { personaPanel.innerHTML = '<p class="beliefs-empty">Could not load the personality.</p>'; }
}

const tabs = {
  activity: { panel: document.getElementById('feed') },
  beliefs: { panel: beliefsPanel, load: fetchAndRenderBeliefs },
  persona: { panel: personaPanel, load: fetchAndRenderPersona },
  memories: { panel: memoriesPanel, load: () => loadMemories(memoriesPanel) },
};
for (const name of Object.keys(tabs)) {
  document.getElementById(`tab-${name}`).addEventListener('click', () => {
    if (activeTab === name) return;
    activeTab = name;
    for (const [other, { panel }] of Object.entries(tabs)) {
      const button = document.getElementById(`tab-${other}`);
      button.classList.toggle('active', other === name);
      button.setAttribute('aria-selected', String(other === name));
      panel.hidden = other !== name;
    }
    document.getElementById(`tab-${name}`).classList.remove('has-update');
    if (tabs[name].load) requestAnimationFrame(tabs[name].load);
  });
}

let firstSnapshot = true;
function snapshot(data, events) {
  sequence = data.sequence;
  scene?.update(data.world, true);
  setWorld(data.world);
  setSpeakers(data.world.people);
  setTask(data.task);
  if (events) {
    clearFeed();
    // Chat is rebuilt from /api/chat (complete, ordered); replayed chat events are skipped.
    events.forEach(e => isChatEvent(e) || addEvent(e));
  }
  if (!firstSnapshot) {
    // Reconnected: recover anything missed while away.
    reloadChat();
    if (activeTab === 'persona') fetchAndRenderPersona();
    else document.getElementById('tab-persona').classList.add('has-update');
    if (activeTab === 'beliefs') fetchAndRenderBeliefs();
    else document.getElementById('tab-beliefs').classList.add('has-update');
  }
  firstSnapshot = false;
}
connectEvents({ snapshot, connection: setConnection, error: showError, event(event) {
  if (event.sequence <= sequence) return;
  sequence = event.sequence;
  const d = event.data;
  if (event.type === 'world_reset') { clearFeed(); setTask(null); resetChat(); }
  if (d.world) { scene?.update(d.world, event.type === 'world_reset'); setWorld(d.world); }
  if (d.task) setTask(d.task);
  if (event.type === 'agent_active' || event.type === 'agent_message' || event.type === 'critic_review') setAgent(event.agent);
  if (event.type === 'agent_message') document.getElementById('task-summary').textContent = d.summary;
  if (event.type === 'world_tick') return;
  if (handleChatEvent(event)) return;
  if (event.type === 'persona_updated') {
    if (activeTab === 'persona') renderPersona(personaPanel, d.persona);
    else document.getElementById('tab-persona').classList.add('has-update');
    return;
  }
  if (event.type === 'beliefs_updated') {
    if (activeTab === 'beliefs') requestAnimationFrame(fetchAndRenderBeliefs);
    else {
      document.getElementById('tab-beliefs').classList.add('has-update');
    }
    return;
  }
  addEvent(event);
} });

document.getElementById('goal-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const goal = document.getElementById('goal').value.trim();
  if (!goal) return showError('Enter a goal first.');
  showError();
  document.getElementById('run').disabled = true;
  try {
    const created = await api('/tasks', { goal });
    // A WebSocket update may arrive before this HTTP response. Preserve newer state.
    if (getTask()?.id !== created.id) setTask(created);
  }
  catch (error) { showError(error.message); setTask(getTask()); }
});
document.getElementById('reset').addEventListener('click', async () => {
  try {
    snapshot(await api('/world/reset', {}));
    // The reset's own event may be dropped as already seen (same sequence), so clear here too.
    clearFeed(); setTask(null); resetChat();
    showError();
  }
  catch (error) { showError(error.message); }
});
document.getElementById('cancel').addEventListener('click', async () => {
  if (!getTask()) return;
  try { setTask(await api(`/tasks/${getTask().id}/cancel`, {})); }
  catch (error) { showError(error.message); }
});
document.getElementById('view-reset').addEventListener('click', () => scene?.resetView());
document.getElementById('memory-open').addEventListener('click', () => setMemoryOpen(true));
document.getElementById('memory-close').addEventListener('click', () => setMemoryOpen(false));
document.getElementById('memory-overlay').addEventListener('click', (event) => {
  if (event.target.id === 'memory-overlay') setMemoryOpen(false);
});
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') setMemoryOpen(false);
});
document.querySelectorAll('[data-goal]').forEach(button => button.addEventListener('click', () => {
  document.getElementById('goal').value = button.dataset.goal;
  document.getElementById('goal').focus();
}));
async function checkModel() {
  try {
    const { llm } = await api('/health');
    const provider = llm.provider === 'colibri' ? 'Colibri' : 'Ollama';
    document.getElementById('model-status').textContent = llm.model_available
      ? `${llm.model} · ready locally`
      : llm.reachable ? `Model missing · check ${provider}: ${llm.model}` : `${provider} offline · start the model server`;
  } catch { document.getElementById('model-status').textContent = 'Backend offline'; }
}
checkModel();
setInterval(checkModel, 30000);
