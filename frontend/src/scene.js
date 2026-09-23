import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DObject, CSS2DRenderer } from 'three/addons/renderers/CSS2DRenderer.js';
import { MovementQueue } from './animations.js';


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
  controls.maxDistance = 60;
  let viewDistance = 18;
  controls.enablePan = false;
  const resetView = () => { camera.position.set(0, viewDistance, viewDistance * .45); controls.target.set(0, 0, 0); controls.update(); };
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
  let layoutKey = '';
  let layout = new THREE.Group();
  scene.add(layout);
  let roomDefinitions = {};
  let origin = [0, 0];
  const point = ([x, z]) => new THREE.Vector3(x - origin[0], 0, z - origin[1]);
  const roomPosition = (id) => point(roomDefinitions[id].anchor);

  function disposeTree(group) {
    group.traverse(object => {
      object.element?.remove();
      object.geometry?.dispose();
      if (Array.isArray(object.material)) object.material.forEach(m => m.dispose());
      else object.material?.dispose();
    });
    group.removeFromParent();
  }

  function buildLayout(world) {
    const key = JSON.stringify([world.rooms, world.doors]);
    if (key === layoutKey) return false;
    layoutKey = key;
    disposeTree(layout);
    layout = new THREE.Group();
    scene.add(layout);
    roomDefinitions = Object.fromEntries(Object.entries(world.rooms).map(([id, room], index) => {
      if (room.outline?.length) return [id, room];
      // Geometry is optional for headless fixtures; give unknown rooms a generic tile.
      const x = (index % 3) * 5, z = Math.floor(index / 3) * 5;
      return [id, { ...room, anchor: [x, z], outline: [[x-2,z-2],[x+2,z-2],[x+2,z+2],[x-2,z+2]] }];
    }));
    const points = Object.values(roomDefinitions).flatMap(room => room.outline);
    const xs = points.map(p => p[0]), zs = points.map(p => p[1]);
    const width = Math.max(...xs) - Math.min(...xs), depth = Math.max(...zs) - Math.min(...zs);
    origin = [(Math.max(...xs) + Math.min(...xs)) / 2, (Math.max(...zs) + Math.min(...zs)) / 2];
    viewDistance = Math.max(14, width * 1.55, depth * 1.8);
    resetView();
    for (const room of Object.values(roomDefinitions)) {
      const vertices = room.outline.map(point);
      const shape = new THREE.Shape(vertices.map(p => new THREE.Vector2(p.x, -p.z)));
      const floor = new THREE.Mesh(new THREE.ShapeGeometry(shape), new THREE.MeshStandardMaterial({ color: room.color, side: THREE.DoubleSide }));
      floor.rotation.x = -Math.PI / 2;
      floor.position.y = -.025;
      floor.receiveShadow = true;
      layout.add(floor);
      const center = roomPosition(room.id);
      label(layout, room.name, [center.x, .04, center.z + .65], 'room');
      for (let i = 0; i < vertices.length; i++) {
        const a = vertices[i], b = vertices[(i + 1) % vertices.length];
        const length = a.distanceTo(b), direction = b.clone().sub(a).normalize();
        const gaps = (world.doors || []).filter(door => door.rooms.includes(room.id)).flatMap(door => {
          const delta = point(door.position).sub(a), along = delta.dot(direction);
          const perpendicular = delta.clone().sub(direction.clone().multiplyScalar(along)).length();
          return perpendicular < .02 && along >= 0 && along <= length
            ? [[Math.max(0, along-door.width/2), Math.min(length, along+door.width/2)]] : [];
        }).sort((a,b) => a[0]-b[0]);
        let cursor = 0;
        for (const [start, end] of [...gaps, [length, length]]) {
          if (start > cursor) {
            const middle = a.clone().addScaledVector(direction, (cursor+start)/2);
            const wall = box(layout, [start-cursor, .42, .075], [middle.x, .21, middle.z], '#536359');
            wall.rotation.y = -Math.atan2(direction.z, direction.x);
          }
          cursor = Math.max(cursor, end);
        }
      }
    }
    for (const door of world.doors || []) {
      const position = point(door.position);
      const marker = new THREE.Mesh(new THREE.CircleGeometry(.14, 20), new THREE.MeshBasicMaterial({ color: '#fff8df', side: THREE.DoubleSide }));
      marker.rotation.x = -Math.PI / 2;
      marker.position.set(position.x, .005, position.z);
      layout.add(marker);
    }
    return true;
  }

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
    const changed = buildLayout(world);
    if (snap || changed || !lastRoom) movement.snap(roomPosition(world.robot.room));
    else if (world.robot.room !== lastRoom) {
      const door = (world.doors || []).find(d => d.rooms.includes(lastRoom) && d.rooms.includes(world.robot.room));
      if (door) movement.move(point(door.position));
      movement.move(roomPosition(world.robot.room));
    }
    lastRoom = world.robot.room;
    for (const [id, mesh] of people) if (!world.people[id]) { disposeTree(mesh); people.delete(id); }
    for (const [id, mesh] of objects) if (!world.objects[id]) { disposeTree(mesh); objects.delete(id); }
    // Group by room so we can spread occupants instead of stacking them.
    const roomOccupants = {};
    for (const npc of Object.values(world.people)) {
      (roomOccupants[npc.room] ??= []).push(npc.id);
    }
    for (const npc of Object.values(world.people)) {
      if (!people.has(npc.id)) {
        const mesh = person('#748caa');
        label(mesh, npc.name, [0, 1.48, 0]);
        people.set(npc.id, mesh);
      }
      const occupants = roomOccupants[npc.room];
      const idx = occupants.indexOf(npc.id);
      const n = occupants.length;
      const spreadX = (idx - (n - 1) / 2) * 0.6;
      people.get(npc.id).position.copy(roomPosition(npc.room)).add(new THREE.Vector3(spreadX, 0, -.3));
    }
    for (const item of Object.values(world.objects)) {
      if (!objects.has(item.id)) {
        const group = new THREE.Group();
        box(group, [.25, .18, .3], [0, .13, 0], '#637280');
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
        mesh.position.copy(roomPosition(item.location.id)).add(new THREE.Vector3(.35, 0, .25));
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
