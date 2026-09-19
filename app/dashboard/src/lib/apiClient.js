// Talks to the FastAPI backend (backend/api/main.py) that wraps our real,
// validated Python pipelines via subprocess. Run it with:
//   uvicorn backend.api.main:app --reload --port 8000
// Override the URL with VITE_API_BASE_URL if it isn't on localhost:8000.
const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

async function postFiles(path, fieldName, files) {
  const form = new FormData();
  for (const file of files) form.append(fieldName, file, file.name);

  let res;
  try {
    res = await fetch(`${API_BASE}${path}`, { method: "POST", body: form });
  } catch {
    throw new Error(
      `Could not reach the prediction backend at ${API_BASE} — is it running? ` +
        `(uvicorn backend.api.main:app --reload --port 8000)`
    );
  }

  if (!res.ok) {
    let detail = null;
    try {
      detail = (await res.json())?.detail;
    } catch {
      // response body wasn't JSON — fall through to the generic message below
    }
    const message = typeof detail === "string" ? detail : detail?.message;
    throw new Error(message ?? `Backend request failed (${res.status})`);
  }

  return res.json();
}

// Each call resolves { columns, rows, csv, filename, log_tail } — `rows` is
// the parsed predictions table, `csv` the exact submission-format text
// (byte-identical to what predict.py would write for the same input).
export function predictDoor(file) {
  return postFiles("/api/door/predict", "file", [file]);
}

export function predictAcv(file) {
  return postFiles("/api/acv/predict", "file", [file]);
}

export function predictRail(files) {
  return postFiles("/api/rail/predict", "files", files);
}

export function predictShm(files) {
  return postFiles("/api/shm/predict", "files", files);
}
