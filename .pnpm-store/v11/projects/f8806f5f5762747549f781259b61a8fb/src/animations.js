// Visualization only. Never sends movement back to the backend.
export class MovementQueue {
  constructor(position, duration = 800) {
    this.position = position;
    this.duration = duration;
    this.queue = [];
    this.active = null;
  }
  snap(target) {
    this.queue = [];
    this.active = null;
    this.position.copy(target);
  }
  move(target) {
    this.queue.push(target.clone());
  }
  update(now) {
    if (!this.active && this.queue.length) {
      this.active = { from: this.position.clone(), to: this.queue.shift(), start: now };
    }
    if (!this.active) return;
    const t = Math.min(1, (now - this.active.start) / this.duration);
    this.position.lerpVectors(this.active.from, this.active.to, t * t * (3 - 2 * t));
    if (t === 1) this.active = null;
  }
}
