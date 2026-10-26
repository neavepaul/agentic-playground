export async function api(path, body) {
  const response = await fetch(`/api${path}`, {
    method: body === undefined ? 'GET' : 'POST',
    headers: { 'Content-Type': 'application/json' },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error(response.ok
      ? 'The server response could not be read. Please try again.'
      : `The server had a problem (HTTP ${response.status}). Please try again.`);
  }
  if (!response.ok) throw new Error(typeof data?.detail === 'string'
    ? data.detail
    : 'Please check what you entered and try again.');
  return data;
}
