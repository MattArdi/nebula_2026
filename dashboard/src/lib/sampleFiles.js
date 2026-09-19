// Fetches a bundled sample file (served from public/sample-data/) and wraps
// it as a File object, so it can flow through exactly the same parse/predict
// path as a real drag-and-drop upload — the dashboard's default view and a
// manual upload are the same code path, not two separate implementations.
export async function fetchAsFile(url, name) {
  const res = await fetch(url);
  // Dev servers with SPA-style fallback routing (Vite included) return a
  // 200 text/html response — the app shell — for ANY unmatched path,
  // including a missing non-HTML asset like this one. res.ok alone can't
  // tell a real sample file from that fallback page, so a bundled sample
  // is treated as missing whenever content-type says html but the file
  // being requested isn't.
  const contentType = res.headers.get("content-type") ?? "";
  const looksLikeSpaFallback = contentType.includes("text/html") && !url.toLowerCase().endsWith(".html");
  if (!res.ok || looksLikeSpaFallback) {
    throw new Error(`Could not load sample file ${name} (${looksLikeSpaFallback ? "not found" : res.status})`);
  }
  const blob = await res.blob();
  return new File([blob], name, { type: blob.type });
}

export async function fetchAllAsFiles(entries) {
  return Promise.all(entries.map(({ url, name }) => fetchAsFile(url, name)));
}
