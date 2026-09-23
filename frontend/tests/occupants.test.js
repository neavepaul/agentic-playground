import test from 'node:test';
import assert from 'node:assert/strict';
import { PerspectiveCamera, Vector3 } from 'three';
import { occupantOffset } from '../src/occupants.js';
import { occupantPositions, fitsInRoom, roomEntities } from '../src/occupants.js';
import { readFileSync } from 'node:fs';

test('three master-bedroom occupants stay inside the walls throughout an orbit', () => {
  const world = JSON.parse(readFileSync(new URL('../../backend/worlds/house.json', import.meta.url)));
  const room = world.rooms.master_bedroom;
  const camera = new PerspectiveCamera(38, 1, .1, 100);
  const center = new Vector3(...[room.anchor[0], 0, room.anchor[1]]);
  for (let angle = 0; angle < Math.PI * 2; angle += Math.PI / 36) {
    camera.position.copy(center).add(new Vector3(Math.cos(angle) * 10, 18, Math.sin(angle) * 10));
    camera.lookAt(center);
    camera.updateMatrixWorld();
    const positions = occupantPositions(3, camera, room);
    for (const position of positions) {
      assert.ok(fitsInRoom(position, room.outline));
      assert.ok(position.x >= .6 && position.x <= 2.25);
      assert.ok(position.z >= 5.35 && position.z <= 8.3);
    }
    for (let i = 1; i < positions.length; i++) {
      assert.ok(positions[i].distanceTo(positions[i - 1]) > .44, 'Capsules must not overlap');
      assert.ok(positions[i].clone().project(camera).distanceTo(positions[i - 1].clone().project(camera)) > .02);
    }
  }
});

test('avatar, people and floor objects share slots; carried objects follow owners', () => {
  const world = JSON.parse(readFileSync(new URL('../../backend/worlds/house.json', import.meta.url)));
  world.robot.room = 'master_bedroom';
  world.people.moira.room = 'master_bedroom';
  world.people.paul.room = 'master_bedroom';
  world.objects.extra = { id: 'extra', location: { kind: 'room', id: 'master_bedroom' } };
  world.objects.held = { id: 'held', location: { kind: 'person', id: 'moira' } };
  const entities = roomEntities(world).master_bedroom;
  assert.ok(entities.some(entity => entity.kind === 'robot'));
  assert.equal(entities.filter(entity => entity.kind === 'person').length, 3);
  assert.ok(entities.some(entity => entity.id === 'extra'));
  assert.ok(!entities.some(entity => entity.id === 'held'));
  const camera = new PerspectiveCamera();
  const room = world.rooms.master_bedroom;
  for (let angle = 0; angle < Math.PI * 2; angle += Math.PI / 12) {
    camera.position.set(Math.cos(angle) * 10, 18, Math.sin(angle) * 10);
    camera.lookAt(0, 0, 0);
    const positions = occupantPositions(entities.length, camera, room);
    for (let i = 0; i < positions.length; i++) {
      assert.ok(fitsInRoom(positions[i], room.outline, .37));
      for (let j = 0; j < i; j++) assert.ok(positions[i].distanceTo(positions[j]) > .65);
    }
  }
});

test('room clearance excludes concave cutouts and walls', () => {
  const outline = [[0, 0], [3, 0], [3, 1], [1, 1], [1, 3], [0, 3]];
  assert.ok(fitsInRoom(new Vector3(.5, 0, 2), outline));
  assert.ok(!fitsInRoom(new Vector3(2, 0, 2), outline));
  assert.ok(!fitsInRoom(new Vector3(.9, 0, 2), outline));
});

test('roommates remain separated on screen from every orbit angle', () => {
  const camera = new PerspectiveCamera(38, 1, .1, 100);
  const center = new Vector3(0, 0, 0);
  for (let angle = 0; angle < Math.PI * 2; angle += Math.PI / 12) {
    camera.position.set(Math.cos(angle) * 10, 18, Math.sin(angle) * 10);
    camera.lookAt(center);
    camera.updateMatrixWorld();
    const moira = occupantOffset(0, 2, camera);
    const neave = occupantOffset(1, 2, camera);
    assert.ok(moira.y === 0);
    assert.ok(neave.y === 0);
    assert.ok(Math.abs(moira.distanceTo(neave) - 1.4) < 1e-10);
    // At a 470px viewport, allow room for both name labels as well as bodies.
    const gap = (neave.clone().project(camera).x - moira.clone().project(camera).x) * 235;
    assert.ok(gap > 45, `Labels overlap at orbit angle ${angle}: ${gap}px`);
  }
});

test('a person stays at the room anchor when their roommate leaves', () => {
  const camera = new PerspectiveCamera();
  assert.equal(occupantOffset(0, 1, camera).length(), 0);
});
