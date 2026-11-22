import { api } from './api.js';
import { activeConversation } from './chat.js';

export async function initPhonePush() {
  const enable = document.getElementById('phone-push-enable');
  const disable = document.getElementById('phone-push-disable');
  const status = document.getElementById('phone-push-status');
  const settings = await api('/phone-push');
  if (!settings.public_key) return;
  if (!window.isSecureContext || !navigator.serviceWorker || !window.PushManager) {
    status.textContent = 'Open Kip over HTTPS. On iPhone, add Kip to your Home Screen first.';
    return;
  }
  const registration = await navigator.serviceWorker.register('/kip-push.js');
  const tab = crypto.randomUUID();
  function reportView(event) {
    const conversation = event?.type !== 'pagehide' && document.visibilityState === 'visible' && document.hasFocus()
      ? activeConversation() : null;
    fetch('/api/chat-view/' + tab + '?conversation=' + encodeURIComponent(conversation || ''), {
      method: 'POST', keepalive: true,
    }).catch(() => {});
  }
  for (const name of ['focus', 'blur', 'hashchange', 'pagehide', 'pageshow']) {
    window.addEventListener(name, reportView);
  }
  document.addEventListener('visibilitychange', reportView);
  window.addEventListener('kip:conversation', reportView);
  setInterval(reportView, 8000);
  reportView();
  enable.hidden = disable.hidden = false;
  enable.onclick = async () => {
    enable.disabled = true;
    try {
      const sub = await registration.pushManager.subscribe({
        userVisibleOnly: true, applicationServerKey: settings.public_key,
      });
      await api('/phone-push', sub.toJSON());
      status.textContent = 'Notifications enabled for this profile on this device.';
    } catch (error) {
      status.textContent = error.message || 'Could not enable notifications.';
    } finally {
      enable.disabled = false;
    }
  };
  disable.onclick = async () => {
    try {
      await api('/phone-push', undefined, 'DELETE');
      status.textContent = 'Notifications disabled on this sign-in.';
    } catch (error) {
      status.textContent = error.message;
    }
  };
}
