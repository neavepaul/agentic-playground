self.addEventListener('push', (event) => {
  event.waitUntil(self.registration.showNotification('Kip', {
    body: 'You have a new message from Kip.',
    tag: 'kip-message',
  }));
});

self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  event.waitUntil((async () => {
    const windows = await self.clients.matchAll({ type: 'window', includeUncontrolled: true });
    if (windows.length) await windows[0].focus();
    else await self.clients.openWindow('/');
  })());
});
