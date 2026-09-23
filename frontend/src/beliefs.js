/**
 * Belief graph renderer — Fruchterman-Reingold force layout → SVG.
 * Nodes are entities (people, objects, rooms); edges are relations
 * coloured by effective_confidence (green ≥ 0.7, amber ≥ 0.4, red < 0.4).
 */

// ── Force layout ──────────────────────────────────────────────────────────────

function layoutGraph(nodes, links, W, H) {
  const n = nodes.length;
  if (n === 0) return;
  if (n === 1) { nodes[0].x = W / 2; nodes[0].y = H / 2; return; }

  const k = Math.sqrt((W * H) / n) * 0.75;
  const R = Math.min(W, H) * 0.32;

  // Circular seed positions for stable convergence
  nodes.forEach((node, i) => {
    const a = (2 * Math.PI * i) / n;
    node.x = W / 2 + R * Math.cos(a);
    node.y = H / 2 + R * Math.sin(a);
  });

  const idx = Object.fromEntries(nodes.map((node, i) => [node.id, i]));
  const fx = new Float32Array(n);
  const fy = new Float32Array(n);
  let temp = 0.12 * Math.sqrt(W * H);
  const cooling = temp / 150;

  for (let iter = 0; iter < 150; iter++) {
    fx.fill(0); fy.fill(0);

    // Repulsion between all pairs
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const dx = (nodes[i].x - nodes[j].x) || 0.01;
        const dy = (nodes[i].y - nodes[j].y) || 0.01;
        const d = Math.sqrt(dx * dx + dy * dy) || 0.01;
        const f = k * k / d;
        fx[i] += dx / d * f; fy[i] += dy / d * f;
        fx[j] -= dx / d * f; fy[j] -= dy / d * f;
      }
    }

    // Spring attraction along edges
    for (const e of links) {
      const si = idx[e.subject], ti = idx[e.target];
      if (si == null || ti == null || si === ti) continue;
      const dx = nodes[ti].x - nodes[si].x;
      const dy = nodes[ti].y - nodes[si].y;
      const d = Math.sqrt(dx * dx + dy * dy) || 0.01;
      const f = d * d / k;
      fx[si] += dx / d * f; fy[si] += dy / d * f;
      fx[ti] -= dx / d * f; fy[ti] -= dy / d * f;
    }

    // Light gravity toward centre
    for (let i = 0; i < n; i++) {
      fx[i] += (W / 2 - nodes[i].x) * 0.01;
      fy[i] += (H / 2 - nodes[i].y) * 0.01;
    }

    // Apply with temperature clamping + boundary
    for (let i = 0; i < n; i++) {
      const d = Math.sqrt(fx[i] * fx[i] + fy[i] * fy[i]) || 0.01;
      const scale = Math.min(d, temp) / d;
      nodes[i].x = Math.max(52, Math.min(W - 52, nodes[i].x + fx[i] * scale));
      nodes[i].y = Math.max(36, Math.min(H - 52, nodes[i].y + fy[i] * scale));
    }
    temp = Math.max(0.5, temp - cooling);
  }
}

// ── SVG helpers ───────────────────────────────────────────────────────────────

const NS = 'http://www.w3.org/2000/svg';
function svgEl(tag, attrs = {}) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  return e;
}
function svgText(attrs, content) {
  const e = svgEl('text', attrs);
  e.textContent = content;
  return e;
}

// ── Colour helpers ────────────────────────────────────────────────────────────

const EDGE_COLORS = { hi: '#52795e', mid: '#c49a3c', lo: '#c97963' };

function edgeTier(eff) { return eff >= 0.7 ? 'hi' : eff >= 0.4 ? 'mid' : 'lo'; }
function edgeColor(eff) { return EDGE_COLORS[edgeTier(eff)]; }

function nodeColor(id, people, rooms) {
  if (people.has(id)) return '#7c9fd8';
  if (rooms.has(id))  return '#7dba87';
  return '#e8a85a';
}

const REL_SHORT = { located_in: 'in', needs: 'needs', has: 'has', recurring_need: '★ need' };
function shortRel(rel) { return REL_SHORT[rel] ?? rel; }

// ── Public render ─────────────────────────────────────────────────────────────

/**
 * Render the belief graph into `container`.
 * @param {Element}  container  DOM element to paint into
 * @param {Object}   data       Response from GET /api/beliefs
 * @param {Set}      people     Set of person IDs from world snapshot
 * @param {Set}      rooms      Set of room IDs from world snapshot
 */
export function renderBeliefs(container, data, people, rooms) {
  const edges = data?.beliefs ?? [];

  if (edges.length === 0) {
    container.innerHTML = '<p class="beliefs-empty">No beliefs recorded yet.<br>Run a task and the agent will consolidate its observations here.</p>';
    return;
  }

  const W = container.clientWidth  || 360;
  const H = container.clientHeight || 360;

  // Collect unique node IDs
  const nodeIds = new Set();
  edges.forEach(e => { nodeIds.add(e.subject); nodeIds.add(e.target); });
  const nodes = [...nodeIds].map(id => ({ id, x: 0, y: 0 }));
  layoutGraph(nodes, edges, W, H);
  const nodeMap = Object.fromEntries(nodes.map(n => [n.id, n]));

  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, width: '100%', height: H, class: 'belief-svg' });

  // Arrow-head markers
  const defs = svgEl('defs');
  for (const [id, fill] of Object.entries(EDGE_COLORS)) {
    const m = svgEl('marker', { id: `arr-${id}`, markerWidth: '8', markerHeight: '6', refX: '7', refY: '3', orient: 'auto' });
    m.appendChild(svgEl('path', { d: 'M0,0 L0,6 L8,3 z', fill }));
    defs.appendChild(m);
  }
  svg.appendChild(defs);

  // ── Edges ────────────────────────────────────────────────────────────────
  const NODE_R = 20;
  const edgeGroup = svgEl('g');

  for (const edge of edges) {
    const s = nodeMap[edge.subject], t = nodeMap[edge.target];
    if (!s || !t || s === t) continue;
    const eff = edge.effective_confidence ?? 0;
    const color = edgeColor(eff);
    const tier = edgeTier(eff);

    const dx = t.x - s.x, dy = t.y - s.y;
    const d  = Math.sqrt(dx * dx + dy * dy) || 1;
    const ux = dx / d,    uy = dy / d;

    // Offset endpoints to the node circle's edge
    const x1 = s.x + ux * NODE_R,        y1 = s.y + uy * NODE_R;
    const x2 = t.x - ux * (NODE_R + 7),  y2 = t.y - uy * (NODE_R + 7);

    edgeGroup.appendChild(svgEl('line', {
      x1, y1, x2, y2,
      stroke: color, 'stroke-width': '1.5', 'stroke-opacity': '0.8',
      'marker-end': `url(#arr-${tier})`,
    }));

    // Label at midpoint, offset perpendicularly so it clears the line
    const mx = (x1 + x2) / 2 + (-uy * 10);
    const my = (y1 + y2) / 2 + ( ux * 10);
    edgeGroup.appendChild(svgText({
      x: mx, y: my, 'text-anchor': 'middle', 'dominant-baseline': 'middle',
      fill: color, 'font-size': '9', 'font-weight': '600',
      stroke: '#f4f5f1', 'stroke-width': '2.5', 'paint-order': 'stroke',
    }, `${shortRel(edge.relation)} ${eff.toFixed(2)}`));
  }
  svg.appendChild(edgeGroup);

  // ── Nodes ────────────────────────────────────────────────────────────────
  const nodeGroup = svgEl('g');
  for (const node of nodes) {
    const fill = nodeColor(node.id, people, rooms);
    const g = svgEl('g', { transform: `translate(${node.x.toFixed(1)},${node.y.toFixed(1)})` });
    g.appendChild(svgEl('circle', { r: NODE_R, fill, stroke: '#fff', 'stroke-width': '2', style: 'cursor:default' }));
    const label = node.id.length > 9 ? node.id.slice(0, 9) + '…' : node.id;
    g.appendChild(svgText({ y: '4', 'text-anchor': 'middle', fill: '#fff', 'font-size': '9', 'font-weight': '700' }, label));
    nodeGroup.appendChild(g);
  }
  svg.appendChild(nodeGroup);

  // ── Legend ───────────────────────────────────────────────────────────────
  const legY = H - 44;
  const leg = svgEl('g', { transform: `translate(10,${legY})` });

  const entityLegend = [['#7c9fd8', 'Person'], ['#7dba87', 'Room'], ['#e8a85a', 'Object']];
  entityLegend.forEach(([fill, label], i) => {
    const g = svgEl('g', { transform: `translate(${i * 76},0)` });
    g.appendChild(svgEl('circle', { cx: '5', cy: '5', r: '5', fill }));
    g.appendChild(svgText({ x: '14', y: '9', 'font-size': '9', fill: '#84937b' }, label));
    leg.appendChild(g);
  });

  const confLegend = [['#52795e', '≥70%'], ['#c49a3c', '40–70%'], ['#c97963', '<40%']];
  confLegend.forEach(([stroke, label], i) => {
    const g = svgEl('g', { transform: `translate(${i * 76},18)` });
    g.appendChild(svgEl('line', { x1: '0', y1: '5', x2: '10', y2: '5', stroke, 'stroke-width': '2' }));
    g.appendChild(svgText({ x: '14', y: '9', 'font-size': '9', fill: '#84937b' }, label));
    leg.appendChild(g);
  });
  svg.appendChild(leg);

  container.innerHTML = '';
  container.appendChild(svg);
}
