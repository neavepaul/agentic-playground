/** JSON API helper. Errors always become readable messages, even for non-JSON replies. */
export async function api(path, body) {
  const response = await fetch(`/api${path}`, {
    method: body === undefined ? 'GET' : 'POST',
    headers: { 'Content-Type': 'application/json' },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
  const text = await response.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = null; }
  if (!response.ok) {
    const detail = data?.detail;
    if (typeof detail === 'string') throw new Error(detail);
    if (response.status === 422) throw new Error('Please check what you entered and try again.');
    throw new Error(`The server had a problem (HTTP ${response.status}). Please try again.`);
  }
  if (data === null) throw new Error('The server sent a response that could not be read.');
  return data;
}
