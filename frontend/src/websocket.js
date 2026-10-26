import { api } from './api.js';

export function connectEvents({ snapshot, event, connection, error }) {
  let retries = 0;
  let socket;
  let stopped = false;
  let timer;
  async function connect() {
    if (stopped) return;
    try {
      // Re-fetch on every connection, including reconnect after missed events.
      snapshot(await api('/world'));
      const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
      socket = new WebSocket(`${protocol}//${location.host}/ws`);
      socket.onopen = () => { retries = 0; connection(true); error(''); };
      socket.onmessage = ({ data }) => {
        const message = JSON.parse(data);
        if (message.type === 'snapshot') snapshot(message.data, message.events);
        else event(message);
      };
      socket.onclose = reconnect;
      socket.onerror = () => socket.close();
    } catch {
      error('Cannot reach the backend. Start FastAPI on port 8000.');
      reconnect();
    }
  }
  function reconnect() {
    connection(false);
    if (!stopped) timer = setTimeout(connect, Math.min(1000 * 2 ** retries++, 10000));
  }
  connect();
  return () => { stopped = true; clearTimeout(timer); socket?.close(); };
}
