/**
 * JSON calls to Kip's /api. A body makes it a POST unless a method is given. A 401 means this
 * device is no longer signed in: the page hears 'kip:signed-out' and shows the sign-in screen.
 */
export async function api(path, body, method) {
  const response = await fetch(`/api${path}`, {
    method: method || (body === undefined ? 'GET' : 'POST'),
    credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json' },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
  if (response.status === 401 && typeof window !== 'undefined') {
    window.dispatchEvent(new Event('kip:signed-out'));
  }
  if (response.status === 204) return null;
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error(response.ok
      ? 'The server response could not be read. Please try again.'
      : `The server had a problem (HTTP ${response.status}). Please try again.`);
  }
  if (!response.ok) {
    const error = new Error(typeof data?.detail === 'string'
      ? data.detail
      : 'Please check what you entered and try again.');
    error.status = response.status;
    throw error;
  }
  return data;
}
