import { useCallback, useState } from "react";

// Keeps every loaded file's computed result around (keyed by file name) so
// dropping File 2 adds a run instead of replacing File 1 — with a dropdown
// to pick which one is currently shown. Re-dropping a same-named file
// updates its existing run in place rather than duplicating it.
export function useFileRuns() {
  const [runs, setRuns] = useState([]); // [{ id, ...anything }]
  const [selectedId, setSelectedId] = useState(null);

  const addRun = useCallback((run) => {
    setRuns((prev) => {
      const idx = prev.findIndex((r) => r.id === run.id);
      if (idx >= 0) {
        const copy = [...prev];
        copy[idx] = run;
        return copy;
      }
      return [...prev, run];
    });
    setSelectedId(run.id);
  }, []);

  const removeRun = useCallback((id) => {
    setRuns((prev) => {
      const idx = prev.findIndex((r) => r.id === id);
      const next = prev.filter((r) => r.id !== id);
      // Removing the currently-selected file falls back to its neighbor
      // (or whatever's left) instead of leaving nothing selected.
      setSelectedId((cur) => {
        if (cur !== id) return cur;
        if (!next.length) return null;
        return next[Math.min(idx, next.length - 1)].id;
      });
      return next;
    });
  }, []);

  return { runs, selectedId, setSelectedId, addRun, removeRun };
}
