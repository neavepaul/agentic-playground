import test from 'node:test';
import assert from 'node:assert/strict';
import { Vector3 } from 'three';
import { MovementQueue } from '../src/animations.js';

test('spacing retargets the destination without replacing the doorway waypoint', () => {
  const position = new Vector3();
  const motion = new MovementQueue(position, 100);
  motion.move(new Vector3(1, 0, 0));
  motion.move(new Vector3(2, 0, 0));
  motion.retargetDestination(new Vector3(2, 0, 1));
  motion.update(0);
  motion.update(100);
  assert.deepEqual(position.toArray(), [1, 0, 0]);
  motion.update(101);
  motion.retargetDestination(new Vector3(2, 0, 2));
  motion.update(201);
  assert.deepEqual(position.toArray(), [2, 0, 2]);
  motion.retargetDestination(new Vector3(3, 0, 2));
  assert.deepEqual(position.toArray(), [3, 0, 2]);
});

test('movement preserves room-to-room order and resync clears stale animation', () => {
  const position = new Vector3();
  const motion = new MovementQueue(position, 100);
  motion.move(new Vector3(1, 0, 0));
  motion.move(new Vector3(1, 0, 1));
  motion.update(0);
  motion.update(50);
  assert.equal(position.x, .5);
  assert.equal(position.z, 0);
  motion.update(100);
  motion.update(101);
  motion.update(201);
  assert.deepEqual(position.toArray(), [1, 0, 1]);
  motion.move(new Vector3(4, 0, 0));
  motion.snap(new Vector3());
  motion.update(400);
  assert.deepEqual(position.toArray(), [0, 0, 0]);
});
