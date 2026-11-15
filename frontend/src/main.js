import './style.css';
import { api } from './api.js';
import { connectEvents } from './websocket.js';
import { handleChatEvent, initChat, isChatEvent, reloadChat, showNote } from './chat.js';
import { initLookups, update as updateLookup } from './lookups.js';
import { addEvent, clearActivity, initActivity } from './activity.js';
import { initMemories, loadMemories } from './memories.js';
import { renderPersona } from './persona.js';
import { initPanel, initTabs, setConnection, setModel, setMood } from './ui.js';

const memoriesPanel = document.getElementById('memories-panel');
const personaPanel = document.getElementById('persona-panel');
const activityPanel = document.getElementById('activity-panel');

let sequence = -1;
let replaying = false;
let persona = null;
let memoriesTimer;

initChat();
initLookups();
initPanel();
initMemories(memoriesPanel);
const tabs = initTabs(['memories', 'persona', 'activity'], (name) => {
  if (name === 'memories') loadMemories(memoriesPanel);
  if (name === 'persona') loadPersona();
});
initActivity(activityPanel, () => { if (!replaying) tabs.flag('activity'); });
loadMemories(memoriesPanel);

async function loadPersona() {
  try { showPersona(await api('/persona')); }
  catch {
    if (tabs.active() === 'persona') {
      personaPanel.innerHTML = '<div class="panel-empty"><strong>Could not load Kip’s personality</strong><span>Check that the backend is running.</span></div>';
    }
  }
}

function showPersona(data) {
  persona = data;
  setMood(data);
  if (tabs.active() === 'persona') renderPersona(personaPanel, data);
}

/** New memories are written just after Kip replies; refresh the list if it is open. */
function refreshMemoriesSoon() {
  if (tabs.active() !== 'memories') return;
  clearTimeout(memoriesTimer);
  memoriesTimer = setTimeout(() => loadMemories(memoriesPanel), 1500);
}

function snapshot(data, events) {
  sequence = Math.max(sequence, Number(data?.sequence ?? -1));
  if (data?.task) updateLookup(data.task);
  if (!events) return; // the HTTP pre-fetch; the socket's own snapshot follows
  replaying = true;
  clearActivity();
  for (const event of events) if (!isChatEvent(event)) addEvent(event);
  replaying = false;
  reloadChat(); // authoritative history, rebuilt on every (re)connect
}

function onEvent(event) {
  if (typeof event?.sequence === 'number') {
    if (event.sequence <= sequence) return;
    sequence = event.sequence;
  }
  const d = event.data || {};
  if (handleChatEvent(event)) {
    if (event.type === 'chat_message') refreshMemoriesSoon();
    return;
  }
  if (event.type === 'persona_updated') {
    const moodChanged = d.persona?.mood !== persona?.mood;
    if (d.persona) showPersona(d.persona);
    if (moodChanged) tabs.flag('persona');
    return;
  }
  if (d.task) updateLookup(d.task);
  addEvent(event);
}

connectEvents({
  snapshot,
  event: onEvent,
  connection: setConnection,
  error(message) {
    if (message) showNote(message, 'sticky');
    else if (document.getElementById('error').dataset.tone === 'sticky') showNote('');
  },
});

async function checkModel() {
  // The conversation model is the one people feel: Groq when chat is hosted.
  try { const health = await api('/health'); setModel(health.chat || health.llm); }
  catch { setModel(null); }
}
checkModel();
setInterval(checkModel, 30000);
loadPersona();
