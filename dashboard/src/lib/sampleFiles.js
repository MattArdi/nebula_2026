// Fetches a bundled sample file (served from public/sample-data/) and wraps
// it as a File object, so it can flow through exactly the same parse/predict
// path as a real drag-and-drop upload — the dashboard's default view and a
// manual upload are the same code path, not two separate implementations.
export async function fetchAsFile(url, name) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Could not load sample file ${name} (${res.status})`);
  const blob = await res.blob();
  return new File([blob], name, { type: blob.type });
}

export async function fetchAllAsFiles(entries) {
  return Promise.all(entries.map(({ url, name }) => fetchAsFile(url, name)));
}
