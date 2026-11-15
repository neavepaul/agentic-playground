/**
 * App chrome: header status pills, Kip's mood line, the side panel and its tabs.
 */
import { modelStatus, moodPhrase } from './format.js';

const $ = (id) => document.getElementById(id);
const NARROW = '(max-width: 1023px)';

export function setConnection(online) {
  $('connection-pill').dataset.state = online ? 'ready' : 'offline';
  $('connection').textContent = online ? 'Connected' : 'Reconnecting…';
}

export function setModel(llm) {
  const status = modelStatus(llm);
  $('model-pill').dataset.state = status.state;
  $('model-status').textContent = status.label;
  $('model-pill').title = status.detail ? `${status.label} — ${status.detail}` : `Language model: ${status.label}`;
}

export function setMood(persona) {
  const text = persona && persona.enabled !== false ? moodPhrase(persona.mood) : '';
  $('kip-mood').dataset.mood = persona?.mood || '';
  $('kip-mood-text').textContent = text || 'Your companion';
}

/** Tabs with arrow-key navigation; `onShow(name)` runs when a tab becomes visible. */
export function initTabs(names, onShow) {
  let active = names[0];
  const select = (name, focus = false) => {
    active = name;
    for (const other of names) {
      const tab = $(`tab-${other}`);
      const selected = other === name;
      tab.setAttribute('aria-selected', String(selected));
      tab.tabIndex = selected ? 0 : -1;
      $(`${other}-panel`).hidden = !selected;
    }
    $(`tab-${name}`).classList.remove('has-update');
    if (focus) $(`tab-${name}`).focus();
    onShow(name);
  };
  names.forEach((name, index) => {
    const tab = $(`tab-${name}`);
    tab.addEventListener('click', () => { if (active !== name) select(name); });
    tab.addEventListener('keydown', (event) => {
      const step = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
      if (step) { event.preventDefault(); select(names[(index + step + names.length) % names.length], true); }
      if (event.key === 'Home') { event.preventDefault(); select(names[0], true); }
      if (event.key === 'End') { event.preventDefault(); select(names[names.length - 1], true); }
    });
  });
  return {
    active: () => active,
    flag(name) { if (name !== active || !panelVisible()) $(`tab-${name}`).classList.add('has-update'); },
  };
}

function panelVisible() {
  return matchMedia(NARROW).matches
    ? document.body.classList.contains('panel-open')
    : !document.body.classList.contains('panel-collapsed');
}

/** Side panel: a column on wide screens (collapsible), a drawer on narrow ones. */
export function initPanel() {
  const toggle = $('panel-toggle');
  const scrim = $('scrim');
  const sync = () => {
    const narrow = matchMedia(NARROW).matches;
    const open = panelVisible();
    toggle.setAttribute('aria-expanded', String(open));
    scrim.hidden = !(narrow && open);
    $('side-panel').inert = !open;
  };
  const setOpen = (open) => {
    if (matchMedia(NARROW).matches) document.body.classList.toggle('panel-open', open);
    else {
      document.body.classList.toggle('panel-collapsed', !open);
      try { localStorage.setItem('kip.panel.collapsed', open ? '0' : '1'); } catch { /* private mode */ }
    }
    sync();
    if (open && matchMedia(NARROW).matches) document.querySelector('#side-panel [aria-selected="true"]')?.focus();
  };
  try { document.body.classList.toggle('panel-collapsed', localStorage.getItem('kip.panel.collapsed') === '1'); }
  catch { /* private mode */ }
  toggle.addEventListener('click', () => setOpen(!panelVisible()));
  $('panel-close').addEventListener('click', () => { setOpen(false); toggle.focus(); });
  scrim.addEventListener('click', () => setOpen(false));
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && matchMedia(NARROW).matches && panelVisible()) { setOpen(false); toggle.focus(); }
  });
  matchMedia(NARROW).addEventListener('change', () => { document.body.classList.remove('panel-open'); sync(); });
  sync();
}
