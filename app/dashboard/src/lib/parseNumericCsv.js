import Papa from "papaparse";

// Parses a headerless CSV of numbers into a 2D array (rows x cols) — used
// by Rail Corrugation (129 columns) and SHM (1 column) files, neither of
// which has a header row.
export function parseNumericMatrix(file) {
  return new Promise((resolve, reject) => {
    Papa.parse(file, {
      header: false,
      dynamicTyping: true,
      skipEmptyLines: true,
      worker: true,
      complete: (results) => {
        // PapaParse reports plenty of non-fatal notices as "errors" (e.g.
        // "Unable to auto-detect delimiting character" on a genuinely
        // single-column file, where it correctly defaults to comma anyway)
        // — the only thing that actually matters is whether usable rows
        // came out the other end.
        if (!results.data?.length) {
          const msg = results.errors?.[0]?.message ?? "No data rows were parsed.";
          return reject(new Error(msg));
        }
        resolve(results.data);
      },
      error: reject,
    });
  });
}
