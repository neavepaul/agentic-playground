import { Vector3 } from 'three';

// Spread along the screen's horizontal axis, so orbiting cannot hide roommates
// behind each other. Keep the offsets on the floor plane.
export function occupantOffset(index, count, camera) {
  const right = new Vector3(1, 0, 0).applyQuaternion(camera.quaternion);
  right.y = 0;
  return right.normalize().multiplyScalar((index - (count - 1) / 2) * 1.4);
}

// Include the capsule's radius and a small gap from every wall, including
// interior corners of concave rooms.
export function fitsInRoom(position, outline, clearance = .3) {
  const { x, z } = position;
  let inside = false;
  for (let i = 0, j = outline.length - 1; i < outline.length; j = i++) {
    const [ax, az] = outline[j], [bx, bz] = outline[i];
    if ((az > z) !== (bz > z) && x < (bx - ax) * (z - az) / (bz - az) + ax) inside = !inside;
    const dx = bx - ax, dz = bz - az;
    const lengthSquared = dx * dx + dz * dz;
    const t = lengthSquared ? Math.max(0, Math.min(1, ((x - ax) * dx + (z - az) * dz) / lengthSquared)) : 0;
    if (Math.hypot(x - ax - t * dx, z - az - t * dz) < clearance) return false;
  }
  return inside;
}

export function occupantPositions(count, camera, room) {
  const center = new Vector3(room.anchor[0], 0, room.anchor[1] - .3);
  if (!fitsInRoom(center, room.outline)) center.z = room.anchor[1];
  const offsets = Array.from({ length: count }, (_, index) => occupantOffset(index, count, camera));
  // Compress the whole row uniformly; clamping each person separately would
  // stack multiple people against the same wall.
  for (let step = 40; step >= 0; step--) {
    const positions = offsets.map(offset => center.clone().addScaledVector(offset, step / 40));
    if (positions.every(position => fitsInRoom(position, room.outline, .38))) {
      if (count < 2 || positions[0].distanceTo(positions[1]) >= .75) return positions;
      break;
    }
  }
  // Crowded rooms need more than one row, rather than ever-tighter spacing.
  const xs = room.outline.map(p => p[0]), zs = room.outline.map(p => p[1]);
  const candidates = [];
  for (let x = Math.min(...xs) + .38; x <= Math.max(...xs) - .38; x += .25) {
    for (let z = Math.min(...zs) + .38; z <= Math.max(...zs) - .38; z += .25) {
      const position = new Vector3(x, 0, z);
      if (fitsInRoom(position, room.outline, .38)) candidates.push(position);
    }
  }
  const positions = [];
  const right = new Vector3(1, 0, 0).applyQuaternion(camera.quaternion);
  right.y = 0;
  right.normalize();
  while (positions.length < count && candidates.length) {
    let best = 0, bestScore = -Infinity;
    candidates.forEach((candidate, index) => {
      let score = positions.length ? Math.min(...positions.map(position => {
        const delta = candidate.clone().sub(position);
        return delta.lengthSq() * .5 + delta.dot(right) ** 2;
      })) : -candidate.distanceToSquared(center);
      if (positions.some(position => position.distanceTo(candidate) < .7)) score -= 1000;
      if (score > bestScore) { best = index; bestScore = score; }
    });
    positions.push(candidates.splice(best, 1)[0]);
  }
  return Array.from({ length: count }, (_, index) => positions[index] ?? center.clone());
}

export function roomEntities(world) {
  const rooms = {};
  const add = (room, kind, id) => (rooms[room] ??= []).push({ kind, id });
  add(world.robot.room, 'robot', 'robot');
  for (const person of Object.values(world.people)) add(person.room, 'person', person.id);
  for (const item of Object.values(world.objects)) {
    if (item.location.kind === 'room') add(item.location.id, 'object', item.id);
  }
  return rooms;
}
