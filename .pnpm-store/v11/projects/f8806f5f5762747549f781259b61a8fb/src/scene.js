import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DObject, CSS2DRenderer } from 'three/addons/renderers/CSS2DRenderer.js';
import { MovementQueue } from './animations.js';

const ROOMS = { hall: [0, 0], kitchen: [-5, 0], bedroom: [5, 0], study: [0, -5] };
const roomPosition = (room) => new THREE.Vector3(ROOMS[room][0], 0, ROOMS[room][1]);

export function createScene(container) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#edf0e8');
  const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.85;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  container.append(renderer.domElement);
  const labels = new CSS2DRenderer();
  Object.assign(labels.domElement.style, { position: 'absolute', inset: '0', pointerEvents: 'none' });
  container.append(labels.domElement);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.maxPolarAngle = Math.PI / 2.35;
  controls.minDistance = 10;
  controls.maxDistance = 35;
  controls.enablePan = false;
  const resetView = () => { camera.position.set(13, 17, 19); controls.target.set(0, 0, -1); controls.update(); };
  resetView();
  scene.add(new THREE.HemisphereLight(0xffffff, 0xaab397, 2.5));
  const sun = new THREE.DirectionalLight(0xfff6df, 3);
  sun.position.set(-7, 16, 9);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, { left: -14, right: 14, top: 14, bottom: -14 });
  sun.shadow.bias = -0.001;
  scene.add(sun);

  function box(parent, size, position, color) {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(...size), new THREE.MeshStandardMaterial({ color, roughness: 0.8 }));
    mesh.position.set(...position);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    parent.add(mesh);
    return mesh;
  }
  function label(parent, text, position, type = '') {
    const div = document.createElement('div');
    div.className = `scene-label ${type}`;
    div.textContent = text;
    const object = new CSS2DObject(div);
    object.position.set(...position);
    parent.add(object);
    return object;
  }
  box(scene, [200, .1, 200], [0, -.31, 0], '#edf0e8').castShadow = false;
  const colors = { hall: '#e5dfce', kitchen: '#dae4d3', bedroom: '#dedfe7', study: '#d3e1de' };
  for (const [id, [x, z]] of Object.entries(ROOMS)) {
    const room = new THREE.Group();
    room.position.set(x, 0, z);
    scene.add(room);
    box(room, [4.6, .25, 4.6], [0, -.15, 0], colors[id]);
    label(room, id, [0, .02, 1.88], 'room');
    // Low walls leave room contents visible; openings face the hall.
    const openings = { hall: ['west', 'east', 'north'], kitchen: ['east'], bedroom: ['west'], study: ['south'] }[id];
    for (const side of ['north', 'south', 'east', 'west']) {
      const horizontal = side === 'north' || side === 'south';
      const sign = side === 'north' || side === 'west' ? -1 : 1;
      const parts = openings.includes(side) ? [[-1.65, 1.3], [1.65, 1.3]] : [[0, 4.6]];
      for (const [offset, length] of parts) {
        box(room, horizontal ? [length, .48, .12] : [.12, .48, length],
          horizontal ? [offset, .13, sign * 2.25] : [sign * 2.25, .13, offset], '#f7f8ef');
      }
    }
  }
  box(scene, [.5, .08, 1.3], [-2.5, -.09, 0], '#d3d6c4');
  box(scene, [.5, .08, 1.3], [2.5, -.09, 0], '#d3d6c4');
  box(scene, [1.3, .08, .5], [0, -.09, -2.5], '#d3d6c4');
  // A few fixed furnishings for orientation, with no simulation semantics.
  box(scene, [2.8, .65, .65], [-5, .25, -1.6], '#9cad92');
  box(scene, [1.45, .45, 2.0], [5.8, .18, -.6], '#b6b8c5');
  box(scene, [1.35, .12, .5], [5.8, .47, -1.15], '#f1f0ee');
  box(scene, [2, .65, .7], [.25, .26, -6.55], '#8ea79c');

  function person(color, robot = false) {
    const group = new THREE.Group();
    const material = new THREE.MeshStandardMaterial({ color, roughness: .6 });
    const body = new THREE.Mesh(robot ? new THREE.BoxGeometry(.58, .64, .48) : new THREE.CapsuleGeometry(.22, .4, 4, 12), material);
    body.position.y = .6;
    body.castShadow = true;
    group.add(body);
    const head = new THREE.Mesh(new THREE.SphereGeometry(.21, 16, 12), material);
    head.position.y = 1.1;
    head.castShadow = true;
    group.add(head);
    if (robot) {
      box(group, [.38, .12, .03], [0, .72, .255], '#414d41');
      for (const x of [-.25, .25]) box(group, [.14, .18, .38], [x, .17, 0], '#596052');
      label(group, 'Avatar', [0, 1.55, 0], 'avatar');
    }
    scene.add(group);
    return group;
  }
  const avatar = person('#e89a58', true);
  const movement = new MovementQueue(avatar.position, matchMedia('(prefers-reduced-motion: reduce)').matches ? 1 : 850);
  const people = new Map();
  const objects = new Map();
  let lastRoom = null;
  let world = null;

  function update(nextWorld, snap = false) {
    world = nextWorld;
    if (snap || !lastRoom) movement.snap(roomPosition(world.robot.room));
    else if (world.robot.room !== lastRoom) movement.move(roomPosition(world.robot.room));
    lastRoom = world.robot.room;
    for (const npc of Object.values(world.people)) {
      if (!people.has(npc.id)) {
        const mesh = person({ mom: '#7f9b72', dad: '#689b91', neave: '#8e93b0' }[npc.id]);
        label(mesh, npc.name, [0, 1.48, 0]);
        people.set(npc.id, mesh);
      }
      people.get(npc.id).position.copy(roomPosition(npc.room)).add(new THREE.Vector3(-.8, 0, -.5));
    }
    for (const item of Object.values(world.objects)) {
      if (!objects.has(item.id)) {
        const group = new THREE.Group();
        const sizes = { charger: [.32, .18, .38], keys: [.3, .07, .16], laptop: [.6, .08, .43] };
        box(group, sizes[item.id] || [.3, .2, .3], [0, .13, 0], item.id === 'keys' ? '#bc9b55' : '#637280');
        label(group, item.name, [0, .5, 0], 'object');
        scene.add(group);
        objects.set(item.id, group);
      }
    }
    placeObjects();
  }
  function placeObjects() {
    if (!world) return;
    let carried = 0;
    for (const item of Object.values(world.objects)) {
      const mesh = objects.get(item.id);
      if (item.location.kind === 'robot') {
        mesh.position.copy(avatar.position).add(new THREE.Vector3(.55 + carried++ * .3, .65, .15));
      } else if (item.location.kind === 'person') {
        mesh.position.copy(people.get(item.location.id).position).add(new THREE.Vector3(.5, .6, 0));
      } else {
        mesh.position.copy(roomPosition(item.location.id)).add(new THREE.Vector3(.6, 0, .5));
      }
    }
  }
  const resize = new ResizeObserver(() => {
    const { width, height } = container.getBoundingClientRect();
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
    labels.setSize(width, height);
  });
  resize.observe(container);
  renderer.setAnimationLoop((time) => {
    movement.update(time);
    placeObjects();
    controls.update();
    renderer.render(scene, camera);
    labels.render(scene, camera);
  });
  return { update, resetView };
}
