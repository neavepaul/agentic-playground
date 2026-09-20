const $ = (id) => document.getElementById(id);
let task = null;
let connected = false;
let eventCount = 0;
const seen = new Set();

export function showError(message = '') { $('error').hidden = !message; $('error').textContent = message; }
export function setConnection(value) {
  connected = value;
  $('connection').textContent = value ? 'Live connection' : 'Reconnecting…';
  $('connection-dot').classList.toggle('online', value);
  $('run').disabled = !value || task?.status === 'running';
}
export function setWorld(world) {
  $('robot-room').textContent = world.robot.room;
  $('inventory').textContent = world.robot.inventory.join(', ') || 'Empty';
}
export function setTask(next) {
  task = next;
  const running = task?.status === 'running';
  $('task-status').textContent = task?.status || 'Idle';
  $('task-status').className = `status ${task?.status || ''}`;
  $('task-summary').textContent = task?.summary || 'Ready when you are.';
  $('budget').textContent = `${task?.cycle_count || 0} cycles · ${task?.tool_count || 0} tool calls`;
  $('run').disabled = running || !connected;
  $('cancel').hidden = !running;
  renderMemory(task?.task_memory);
  if (!running) setAgent(null);
}
export function getTask() { return task; }
function renderMemory(memory) {
  $('memory-content').textContent = memory
    ? JSON.stringify(memory, null, 2)
    : 'No task memory yet.';
}
export function setMemoryOpen(open) {
  const overlay = $('memory-overlay');
  overlay.hidden = !open;
  if (open) $('memory-close').focus();
}
export function setAgent(name) {
  document.querySelectorAll('[data-agent]').forEach(el => el.classList.toggle('active', el.dataset.agent === name));
}
export function clearFeed() {
  $('feed').replaceChildren();
  seen.clear();
  eventCount = 0;
  $('event-count').textContent = '0 events';
}
function eventText(event) {
  const d = event.data;
  const o = d?.observation;
  if (event.type === 'room_observed') return `In ${o.room}: ${[...o.people.map(p => p.name), ...o.objects.map(x => x.name)].join(', ') || 'no people or objects'}. Exits: ${o.connections.join(', ')}.`;
  if (event.type === 'robot_moved') return `Moved from ${o.from_room} to ${o.room}.`;
  if (event.type === 'person_spoken_to') return `${o.person}: “${o.response}”`;
  if (event.type === 'object_picked_up') return `Picked up ${o.object}.`;
  if (event.type === 'object_dropped') return `Dropped ${o.object} in ${o.room}.`;
  if (event.type === 'object_given') return `Gave ${o.object} to ${o.person}.`;
  if (event.type === 'tool_failed') return `${d.tool}: ${d.error}`;
  if (event.type === 'critic_review') return `${d.approved ? 'Approved' : 'Needs work'}: ${d.summary}${d.suggestion ? ' ' + d.suggestion : ''}`;
  if (['agent_message', 'task_started', 'task_completed', 'task_failed', 'task_cancelled', 'world_reset'].includes(event.type)) return d.summary;
  return null;
}
export function addEvent(event) {
  if (seen.has(event.id)) return;
  const text = eventText(event);
  if (!text) return;
  seen.add(event.id);
  if (seen.size > 600) seen.delete(seen.values().next().value);
  $('feed-empty')?.remove();
  const entry = document.createElement('article');
  entry.className = `feed-entry ${event.agent} ${event.data?.observation ? 'evidence' : ''} ${event.type.includes('failed') ? 'error-event' : ''}`;
  const top = document.createElement('div'); top.className = 'feed-top';
  const role = document.createElement('span'); role.className = 'feed-role';
  role.textContent = event.type === 'person_spoken_to' ? event.data.observation.person : event.agent;
  const time = document.createElement('span'); time.className = 'feed-time';
  time.textContent = new Date(event.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  const paragraph = document.createElement('p'); paragraph.textContent = text;
  top.append(role, time); entry.append(top, paragraph);
  const feed = $('feed');
  const atBottom = feed.scrollHeight - feed.scrollTop - feed.clientHeight < 70;
  feed.append(entry);
  while (feed.children.length > 200) feed.firstElementChild.remove();
  if (atBottom) feed.scrollTop = feed.scrollHeight;
  $('event-count').textContent = `${++eventCount} events`;
}
