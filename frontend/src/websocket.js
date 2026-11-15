import { api } from './api.js';

/**
 * One WebSocket to /ws with exponential-backoff reconnect. Every (re)connection first
 * re-fetches GET /api/state, then the socket's own snapshot replays recent history, so
 * events missed while offline are recovered and live events are filtered by sequence.
 */
export function connectEvents({ snapshot, event, connection, error }) {
  let retries = 0;
  let socket;
  let stopped = false;
  let timer;
  async function connect() {
    if (stopped) return;
    try {
      snapshot(await api('/state'));
      const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
      socket = new WebSocket(`${protocol}//${location.host}/ws`);
      socket.onopen = () => { retries = 0; connection(true); error(''); };
      socket.onmessage = ({ data }) => {
        let message;
        try { message = JSON.parse(data); } catch { return; }
        if (message.type === 'snapshot') snapshot(message.data, message.events || []);
        // The server dropped events for this client: reconnect to get a fresh snapshot.
        else if (message.type === 'resync_required') socket.close();
        else event(message);
      };
      socket.onclose = reconnect;
      socket.onerror = () => socket.close();
    } catch {
      error('Can’t reach Kip’s server. Is the backend running on port 8000?');
      reconnect();
    }
  }
  function reconnect() {
    connection(false);
    clearTimeout(timer);
    if (!stopped) timer = setTimeout(connect, Math.min(1000 * 2 ** retries++, 10000));
  }
  connect();
  return () => { stopped = true; clearTimeout(timer); socket?.close(); };
}
