// Builds and triggers a browser download for a prediction CSV. Values are
// escaped minimally (quote + comma + newline) since every column here is a
// filename, timestamp, label, or number — none of which legitimately
// contain quotes, but we escape defensively anyway.
function escapeCell(v) {
  const s = String(v ?? "");
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

export function toCsv(headers, rows) {
  const lines = [headers.join(",")];
  for (const row of rows) {
    lines.push(headers.map((h) => escapeCell(row[h])).join(","));
  }
  return lines.join("\n");
}

export function downloadCsv(filename, headers, rows) {
  downloadCsvText(filename, toCsv(headers, rows));
}

// Same download mechanics as downloadCsv, but for CSV text the caller
// already has (e.g. the backend's own submission-format output) rather
// than headers/rows to build it from.
export function downloadCsvText(filename, csvText) {
  const blob = new Blob([csvText], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
