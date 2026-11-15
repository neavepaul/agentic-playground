/**
 * "Kip" tab — Kip's evolving mood, traits, track record, the people it knows and
 * memorable moments, as reported by GET /api/persona and persona_updated events.
 */

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

const number = (value, fallback = 0) => (Number.isFinite(Number(value)) ? Number(value) : fallback);
const clamp = (value) => Math.max(0, Math.min(1, number(value)));
const label = (name) => String(name).replace(/_/g, ' ');

function meter(name, value, baseline) {
  const row = el('div', 'meter');
  row.append(el('span', 'meter-label', label(name)));
  const track = el('span', 'meter-track');
  track.setAttribute('role', 'meter');
  track.setAttribute('aria-label', label(name));
  track.setAttribute('aria-valuemin', '0');
  track.setAttribute('aria-valuemax', '1');
  track.setAttribute('aria-valuenow', clamp(value).toFixed(2));
  const fill = el('span', 'meter-fill');
  fill.style.width = `${Math.round(clamp(value) * 100)}%`;
  track.append(fill);
  if (baseline !== undefined) {
    const mark = el('span', 'meter-baseline');
    mark.style.left = `${Math.round(clamp(baseline) * 100)}%`;
    mark.title = `Where Kip started: ${number(baseline).toFixed(2)}`;
    track.append(mark);
  }
  row.append(track, el('span', 'meter-value', number(value).toFixed(2)));
  return row;
}

function section(title, hint) {
  const wrap = el('section', 'card-section');
  const head = el('div', 'section-head');
  head.append(el('h3', '', title));
  if (hint) head.append(el('span', 'section-hint', hint));
  wrap.append(head);
  return wrap;
}

function empty(text) { return el('p', 'section-empty', text); }

export function renderPersona(panel, data) {
  panel.replaceChildren();
  if (!data || data.enabled === false) {
    const box = el('div', 'panel-empty');
    box.append(el('strong', '', 'Personality is off'), el('span', '', 'Kip is running without its personality layer.'));
    panel.append(box);
    return;
  }

  const hero = el('div', 'persona-hero');
  hero.dataset.mood = data.mood || '';
  const heading = el('div', 'persona-hero-text');
  heading.append(el('span', 'persona-mood', data.mood ? `Feeling ${data.mood}` : 'Mood unknown'));
  if (data.tone) heading.append(el('p', 'persona-tone', data.tone));
  hero.append(heading);
  panel.append(hero);

  const now = section('Right now');
  now.append(meter('mood', (number(data.valence) + 1) / 2), meter('energy', data.energy));
  for (const [name, value] of Object.entries(data.affect || {})) now.append(meter(name, value));
  panel.append(now);

  const traits = Object.entries(data.traits || {});
  if (traits.length) {
    const block = section('Personality', 'drifts with experience');
    for (const [name, value] of traits) block.append(meter(name, value, data.baseline?.[name]));
    panel.append(block);
  }

  const record = Object.entries(data.track_record || {});
  if (record.length) {
    const block = section('Track record');
    const list = el('ul', 'plain-list');
    for (const [skill, result] of record) {
      const item = el('li');
      item.append(el('strong', '', label(skill)), el('span', '', String(result)));
      list.append(item);
    }
    block.append(list);
    panel.append(block);
  }

  const people = Object.entries(data.relationships || {});
  const bonds = section('People Kip knows');
  if (!people.length) bonds.append(empty('No one yet — say hello.'));
  for (const [person, bond] of people) {
    const card = el('div', 'person-card');
    const top = el('div', 'person-top');
    top.append(el('span', 'person-initial', String(person).charAt(0).toUpperCase()), el('strong', '', person));
    const facts = [];
    if (bond.chats) facts.push(`${bond.chats} chat${bond.chats === 1 ? '' : 's'}`);
    if (bond.times_helped) facts.push(`helped ${bond.times_helped}×`);
    if (bond.do_not_disturb) facts.push('asked for space');
    if (facts.length) top.append(el('span', 'person-facts', facts.join(' · ')));
    card.append(top, meter('warmth', bond.warmth), meter('familiarity', bond.familiarity));
    if (number(bond.concern) > 0.05) card.append(meter('concern', bond.concern));
    for (const moment of bond.moments || []) card.append(el('p', 'person-moment', moment));
    bonds.append(card);
  }
  panel.append(bonds);

  const moments = section('Memorable moments');
  const list = data.moments || [];
  if (!list.length) moments.append(empty('Nothing memorable yet.'));
  else {
    const items = el('ul', 'moment-list');
    for (const moment of [...list].reverse()) {
      const item = el('li');
      item.append(el('span', '', moment.summary));
      if (moment.feeling) item.append(el('span', 'moment-feeling', moment.feeling));
      items.append(item);
    }
    moments.append(items);
  }
  panel.append(moments);

  if (data.followups?.length) {
    const block = section('Planned check-ins');
    const items = el('ul', 'moment-list');
    for (const followup of data.followups) items.append(el('li', '', followup.reason));
    block.append(items);
    panel.append(block);
  }
}
