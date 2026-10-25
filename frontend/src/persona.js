/**
 * Personality panel — the robot's evolving traits, mood, relationships and
 * memorable moments, as reported by GET /api/persona.
 */

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function bar(label, value, baseline) {
  const row = el('div', 'persona-bar');
  row.append(el('span', 'persona-bar-label', label));
  const track = el('span', 'persona-bar-track');
  const fill = el('span', 'persona-bar-fill');
  fill.style.width = `${Math.round(value * 100)}%`;
  track.append(fill);
  if (baseline !== undefined) {
    const mark = el('span', 'persona-bar-baseline');
    mark.style.left = `${Math.round(baseline * 100)}%`;
    mark.title = `Baseline ${baseline.toFixed(2)}`;
    track.append(mark);
  }
  row.append(track, el('span', 'persona-bar-value', value.toFixed(2)));
  return row;
}

function section(title) {
  const wrap = el('section', 'persona-section');
  wrap.append(el('span', 'eyebrow', title));
  return wrap;
}

export function renderPersona(panel, data) {
  panel.replaceChildren();
  if (!data || data.enabled === false) {
    panel.append(el('p', 'beliefs-empty', 'The personality layer is turned off.'));
    return;
  }
  const head = el('div', 'persona-head');
  head.append(el('strong', 'persona-name', data.name),
              el('span', `persona-mood ${data.mood}`, data.mood),
              el('span', 'subtle', data.tone));
  panel.append(head);

  const traits = section('TRAITS · drift from experience');
  for (const [name, value] of Object.entries(data.traits)) traits.append(bar(name, value, data.baseline[name]));
  traits.append(bar('mood', (data.valence + 1) / 2), bar('energy', data.energy));
  panel.append(traits);

  const people = section('RELATIONSHIPS');
  const entries = Object.entries(data.relationships);
  if (!entries.length) people.append(el('p', 'persona-empty', 'No one met yet.'));
  for (const [person, bond] of entries) {
    const card = el('div', 'persona-person');
    card.append(el('strong', '', person),
                el('span', 'subtle', `warmth ${bond.warmth.toFixed(2)} · familiarity ${bond.familiarity.toFixed(2)} · helped ${bond.times_helped}×`));
    if (bond.moments.length) card.append(el('span', 'persona-moment', bond.moments.join(' · ')));
    people.append(card);
  }
  panel.append(people);

  const moments = section('MEMORABLE MOMENTS');
  if (!data.moments.length) moments.append(el('p', 'persona-empty', 'Nothing memorable yet.'));
  for (const moment of [...data.moments].reverse()) {
    moments.append(el('p', 'persona-moment', `${moment.summary} (${moment.feeling})`));
  }
  if (data.followups.length) {
    moments.append(el('span', 'eyebrow', 'PLANNED CHECK-INS'));
    for (const followup of data.followups) moments.append(el('p', 'persona-moment', followup.reason));
  }
  panel.append(moments);
}
