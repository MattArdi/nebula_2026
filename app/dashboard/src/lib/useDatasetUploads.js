import { useRef, useState } from "react";

// Uploads dropped into a subsystem popup. Each starts as "processing" and
// becomes "ready" (with whatever computeRun resolved) or "error". Held in
// memory only, so a refresh clears them. `appliedKey` is the upload that
// was last applied as the page's current dataset, if any.
//
// By default every dropped file is its own upload. A subsystem whose dataset
// is really a set of files (Rail: one file per second) passes `groupFiles`,
// which turns the dropped files into [{ name, baseName, input }] — one entry
// per upload — and `input` is what computeRun receives.
export function useDatasetUploads(computeRun, groupFiles) {
  const [uploads, setUploads] = useState([]); // [{ key, name, baseName, status, progress, run, error }]
  const [appliedKey, setAppliedKey] = useState(null);
  const nextKeyRef = useRef(0);

  async function handleFiles(files) {
    let groups;
    try {
      groups = groupFiles
        ? await groupFiles(files)
        : files.map((file) => ({ name: file.name, baseName: file.name.replace(/\.[^.]+$/, ""), input: file }));
    } catch (err) {
      const key = nextKeyRef.current++;
      setUploads((prev) => [...prev, { key, name: files[0]?.name ?? "Upload", status: "error", error: err.message }]);
      return;
    }

    const entries = groups.map((group) => ({ key: nextKeyRef.current++, ...group }));
    setUploads((prev) => [
      ...prev,
      ...entries.map(({ key, name, baseName }) => ({ key, name, baseName, status: "processing" })),
    ]);

    // One at a time, so several uploads don't hit the backend at once.
    for (const { key, name, input } of entries) {
      const patch = (changes) => setUploads((prev) => prev.map((u) => (u.key === key ? { ...u, ...changes } : u)));
      try {
        const run = await computeRun(input, (done, total) => patch({ progress: `${done}/${total}` }));
        patch({ status: "ready", run });
      } catch (err) {
        patch({ status: "error", error: `${name}: ${err.message}` });
      }
    }
  }

  return { uploads, appliedKey, setAppliedKey, handleFiles };
}
