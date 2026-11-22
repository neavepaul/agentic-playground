/**
 * The chats sidebar, like a chat app's: "New chat", your recent chats newest first, who is signed
 * in, and Settings. The page opens on a new chat; #chat=<id> in the address reopens one (and survives a
 * reload). On narrow screens the sidebar is a drawer.
 */
import { api } from './api.js';
import { relativeTime } from './format.js';
import { initials } from './signin.js';

const $ = (id) => document.getElementById(id);
const NARROW = '(max-width: 1023px)';
let open = () => {};
let current = null;
let loadVersion = 0;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

export function fromAddress() {
  const match = /^#chat=([A-Za-z0-9_-]{1,64})$/.exec(location.hash);
  return match ? match[1] : null;
}

function setAddress(id) {
  const url = id ? `#chat=${id}` : location.pathname + location.search;
  if ((id ? `#chat=${id}` : '') !== location.hash) history.pushState(null, '', url);
}

function drawer(show) {
  document.body.classList.toggle('chats-open', show);
  $('chats-toggle').setAttribute('aria-expanded', String(show));
  $('chats-scrim').hidden = !(show && matchMedia(NARROW).matches);
  $('chats').inert = matchMedia(NARROW).matches && !show;
}

function markActive() {
  document.querySelectorAll('#chat-list .chat-item').forEach((item) => {
    const active = item.dataset.id === current;
    item.classList.toggle('active', active);
    item.querySelector('.chat-open')?.setAttribute('aria-current', active ? 'page' : 'false');
  });
}

/** Open a chat by id, or a new one (null). */
export function select(id, { address = true } = {}) {
  current = id;
  if (address) setAddress(id);
  markActive();
  if (matchMedia(NARROW).matches) drawer(false);
  open(id);
}

/** A new chat's first message made it real: it joins the list and the address. */
export function started(id) {
  current = id;
  setAddress(id);
  load();
}

async function remove(conversation) {
  if (!confirm(`Delete “${conversation.title || 'this chat'}”? Kip still remembers what you told him.`)) return;
  try {
    await api(`/conversations/${encodeURIComponent(conversation.id)}`, undefined, 'DELETE');
    if (conversation.id === current) select(null);
    load();
  } catch (error) { alert(error.message); }
}

export async function load() {
  const version = ++loadVersion;
  const query = $('chat-search').value.trim();
  let conversations = [];
  try {
    ({ conversations } = await api(query ? '/conversations?q=' + encodeURIComponent(query) : '/conversations'));
  } catch { return; }
  if (version !== loadVersion || query !== $('chat-search').value.trim()) return;
  $('chat-list-empty').textContent = query ? 'No chats match your search.' : 'Your chats will show here.';
  $('chat-list-empty').hidden = conversations.length > 0;
  $('chat-list').replaceChildren(...conversations.map((conversation) => {
    const item = el('li', 'chat-item');
    item.dataset.id = conversation.id;
    const button = el('button', 'chat-open');
    button.type = 'button';
    button.append(el('span', 'chat-title', conversation.title || 'New chat'),
      el('span', 'chat-time', relativeTime(conversation.updated_at)));
    button.addEventListener('click', () => select(conversation.id));
    const del = el('button', 'chat-delete');
    del.type = 'button';
    del.title = 'Delete chat';
    del.setAttribute('aria-label', `Delete chat: ${conversation.title}`);
    del.innerHTML = '<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path d="M5 7h14M10 7V5h4v2M7 7l1 12h8l1-12" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>';
    del.addEventListener('click', () => remove(conversation));
    item.append(button, del);
    return item;
  }));
  markActive();
}

/** One switch in the owner's People list: what this person may do. */
function permission(person, key, text) {
  const label = el('label', 'switch');
  const box = el('input');
  box.type = 'checkbox';
  box.checked = Boolean(person[key]);
  box.addEventListener('change', async () => {
    box.disabled = true;
    try { box.checked = Boolean((await api(`/profiles/${encodeURIComponent(person.id)}/access`, { [key]: box.checked }))[key]); }
    catch (error) { box.checked = !box.checked; alert(error.message); }
    finally { box.disabled = false; }
  });
  label.append(box, el('span', 'switch-track'), el('span', 'switch-label', text));
  return label;
}

/** Settings: the owner decides who may ask about the Groww portfolio and who may change their PIN;
 * anyone allowed changes their own PIN here. */
async function settings() {
  let profile;
  try { profile = await api('/me'); } catch (error) { alert(error.message); return; }
  const people = profile.people || [];
  $('settings-people').hidden = !profile.owner;
  $('sharing-empty').hidden = people.length > 0;
  $('sharing-list').replaceChildren(...people.map((person) => {
    const item = el('li', 'sharing-item');
    const who = el('div', 'sharing-who');
    who.append(el('span', 'me-avatar', initials(person.name)), el('span', '', person.name));
    if (person.needs_pin) who.append(el('span', 'sharing-hint', 'no PIN yet'));
    const switches = el('div', 'sharing-switches');
    switches.append(permission(person, 'portfolio', 'Groww portfolio'), permission(person, 'pin_change', 'Change their PIN'));
    item.append(who, switches);
    return item;
  }));
  $('settings-pin').hidden = !profile.pin_change;
  $('change-pin-form').reset();
  $('change-pin-note').hidden = true;
  $('settings').showModal();
}

function initPinChange() {
  const note = (text, ok = false) => {
    $('change-pin-note').textContent = text;
    $('change-pin-note').className = ok ? 'signin-note' : 'signin-error';
    $('change-pin-note').hidden = false;
  };
  for (const id of ['pin-current', 'pin-new', 'pin-new-again']) {
    $(id).addEventListener('input', () => { $(id).value = $(id).value.replace(/\D/g, '').slice(0, 4); });
  }
  $('change-pin-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    const pin = $('pin-new').value;
    if (!/^\d{4}$/.test(pin)) return note('The new PIN is 4 digits.');
    if (pin !== $('pin-new-again').value) return note('The two new PINs are different.');
    const button = $('change-pin-form').querySelector('[type="submit"]');
    button.disabled = true;
    try {
      await api('/me/pin', { current: $('pin-current').value, pin });
      $('change-pin-form').reset();
      note('PIN changed. Any other device signed in as you has been signed out.', true);
    } catch (error) { note(error.message); }
    finally { button.disabled = false; }
  });
}

export function initConversations(me, onOpen) {
  open = onOpen;
  $('me-name').textContent = me.name;
  $('me-avatar').textContent = initials(me.name);
  $('settings-open').hidden = !(me.owner || me.pin_change);
  $('settings-open').addEventListener('click', settings);
  initPinChange();
  let searchTimer;
  $('chat-search').addEventListener('input', () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(load, 200);
  });
  $('new-chat').addEventListener('click', () => { select(null); $('chat-input').focus(); });
  $('sign-out').addEventListener('click', async () => {
    try { await api('/signout', {}); } finally { location.hash = ''; location.reload(); }
  });
  $('chats-toggle').addEventListener('click', () => drawer(!document.body.classList.contains('chats-open')));
  $('chats-scrim').addEventListener('click', () => drawer(false));
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && document.body.classList.contains('chats-open')) drawer(false);
  });
  matchMedia(NARROW).addEventListener('change', () => drawer(false));
  window.addEventListener('popstate', () => { if (fromAddress() !== current) select(fromAddress(), { address: false }); });
  drawer(false);
  load();
  select(fromAddress(), { address: false });
}
