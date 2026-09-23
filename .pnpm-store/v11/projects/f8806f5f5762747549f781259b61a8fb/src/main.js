import './style.css';
import { api } from './api.js';
import { createScene } from './scene.js';
import { connectEvents } from './websocket.js';
import { addEvent, clearFeed, getTask, setAgent, setConnection, setMemoryOpen, setTask, setWorld, showError } from './ui.js';
import { renderBeliefs } from './beliefs.js';

let scene;
try { scene = createScene(document.getElementById('scene')); }
catch { showError('The 3D view requires WebGL. Enable hardware acceleration or try another browser.'); }
let sequence = -1;
let activeTab = 'activity';
let beliefsDirty = false;
const beliefsPanel = document.getElementById('beliefs-panel');

async function fetchAndRenderBeliefs() {
  try {
    const [data, worldData] = await Promise.all([api('/beliefs'), api('/world')]);
    const people = new Set();
    const rooms = new Set();
    for (const [id, room] of Object.entries(worldData?.world?.rooms ?? {})) {
      rooms.add(id);
      for (const pid of room?.people ?? []) people.add(pid);
    }
    renderBeliefs(beliefsPanel, data, people, rooms);
  } catch {
    beliefsPanel.innerHTML = '<p class="beliefs-empty">Could not load beliefs.</p>';
  }
}

document.getElementById('tab-activity').addEventListener('click', () => {
  if (activeTab === 'activity') return;
  activeTab = 'activity';
  document.getElementById('tab-activity').classList.add('active');
  document.getElementById('tab-beliefs').classList.remove('active');
  document.getElementById('feed').hidden = false;
  beliefsPanel.hidden = true;
});

document.getElementById('tab-beliefs').addEventListener('click', () => {
  if (activeTab === 'beliefs') return;
  activeTab = 'beliefs';
  document.getElementById('tab-beliefs').classList.add('active');
  document.getElementById('tab-activity').classList.remove('active');
  document.getElementById('feed').hidden = true;
  beliefsPanel.hidden = false;
  requestAnimationFrame(fetchAndRenderBeliefs);
  beliefsDirty = false;
});

function snapshot(data, events) {
  sequence = data.sequence;
  scene?.update(data.world, true);
  setWorld(data.world);
  setTask(data.task);
  if (events) { clearFeed(); events.forEach(addEvent); }
}
connectEvents({ snapshot, connection: setConnection, error: showError, event(event) {
  if (event.sequence <= sequence) return;
  sequence = event.sequence;
  const d = event.data;
  if (event.type === 'world_reset') { clearFeed(); setTask(null); }
  if (d.world) { scene?.update(d.world, event.type === 'world_reset'); setWorld(d.world); }
  if (d.task) setTask(d.task);
  if (event.type === 'agent_active' || event.type === 'agent_message' || event.type === 'critic_review') setAgent(event.agent);
  if (event.type === 'agent_message') document.getElementById('task-summary').textContent = d.summary;
  if (event.type === 'beliefs_updated') {
    if (activeTab === 'beliefs') requestAnimationFrame(fetchAndRenderBeliefs);
    else beliefsDirty = true;
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
  try { snapshot(await api('/world/reset', {})); showError(); }
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
    const { ollama } = await api('/health');
    document.getElementById('model-status').textContent = ollama.model_available
      ? `${ollama.model} · ready locally`
      : ollama.reachable ? `Model missing · run ollama pull ${ollama.model}` : 'Ollama offline · run ollama serve';
  } catch { document.getElementById('model-status').textContent = 'Backend offline'; }
}
checkModel();
setInterval(checkModel, 30000);
