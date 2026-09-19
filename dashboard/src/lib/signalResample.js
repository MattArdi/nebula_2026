// Chart-support utilities for the Door signal-detail view. None of this
// classifies anything — the real prediction comes from the backend
// (backend/Door/v2_rule-based via apiClient.js). This just slices and
// resamples the raw stream client-side so it can be charted next to that
// prediction.

// Mirrors backend/Door/v2_rule-based/segmentation.py's detect_cycles +
// flag_implausible exactly (same constants, same gap rule): Train.csv /
// Test.csv are individual door cycles concatenated back-to-back with the
// idle time between them stripped out, so a cycle boundary is just a
// >100ms gap in the timestamp column. Used here purely to slice the raw
// signal for charting — the predictions themselves are matched in by
// order from the backend's response, not recomputed.
export const GAP_THRESHOLD_S = 0.1;
export const MIN_CYCLE_ROWS = 50;
export const MAX_CYCLE_ROWS = 250;

// Parses "YYYY-M-D-H-M-S-ms" (not zero-padded) into epoch ms.
export function parseDoorTimestamp(dateStr) {
  const [y, mo, d, h, mi, se, ms] = dateStr.split("-").map(Number);
  return Date.UTC(y, mo - 1, d, h, mi, se, ms);
}

export function groupDoorCycles(rows) {
  const chunks = [];
  let start = 0;
  for (let i = 1; i <= rows.length; i++) {
    const atEnd = i === rows.length;
    const gapMs = atEnd ? Infinity : parseDoorTimestamp(rows[i].Datetime) - parseDoorTimestamp(rows[i - 1].Datetime);
    if (gapMs > GAP_THRESHOLD_S * 1000) {
      const chunk = rows.slice(start, i);
      if (chunk.length >= MIN_CYCLE_ROWS && chunk.length <= MAX_CYCLE_ROWS) {
        chunks.push(chunk);
      }
      start = i;
    }
  }
  return chunks;
}

// "Close command"/"Open command" flags are read directly off the stream,
// not predicted — see backend/Door/v2_rule-based/rules.py's get_operation.
export function getOperation(chunk) {
  const closeMean = chunk.reduce((sum, r) => sum + Number(r["Close command"]), 0) / chunk.length;
  return closeMean > 0.5 ? "Close" : "Open";
}

const RESAMPLE_POINTS = 40;

// Resamples one cycle's raw current/voltage/back-EMF onto a shared 0-100%
// grid (linear interpolation) so cycles of different lengths can be
// overlaid and averaged.
export function resampleCycleSignal(chunk) {
  const n = chunk.length;
  const current = chunk.map((r) => Number(r["Motor current(mA)"]));
  const voltage = chunk.map((r) => Number(r["Motor Voltage(10mV)"]));
  const bemf = chunk.map((r) => Number(r["Motor electrodynamic force"]));

  const points = [];
  for (let i = 0; i <= RESAMPLE_POINTS; i++) {
    const pct = (i / RESAMPLE_POINTS) * 100;
    const pos = (pct / 100) * (n - 1);
    const lo = Math.floor(pos);
    const hi = Math.min(n - 1, lo + 1);
    const frac = pos - lo;
    const lerp = (arr) => arr[lo] + (arr[hi] - arr[lo]) * frac;
    points.push({ pct, current: lerp(current), voltage: lerp(voltage), bemf: lerp(bemf) });
  }
  return points;
}

// Reference "what a normal cycle typically looks like" curve, averaged
// pointwise from every currently-loaded Normal cycle's resampled signal,
// split by operation (Open/Close have very different shapes).
export function averageNormalCycleSignal(segments) {
  const byOp = { Open: [], Close: [] };
  for (const seg of segments) {
    if (seg.prediction === "Normal" && seg.operation && seg.rawSeries?.length) {
      byOp[seg.operation].push(resampleCycleSignal(seg.rawSeries));
    }
  }
  const avg = {};
  for (const op of ["Open", "Close"]) {
    const samples = byOp[op];
    if (!samples.length) {
      avg[op] = [];
      continue;
    }
    avg[op] = samples[0].map((_, i) => ({
      pct: samples[0][i].pct,
      current: samples.reduce((s, c) => s + c[i].current, 0) / samples.length,
      voltage: samples.reduce((s, c) => s + c[i].voltage, 0) / samples.length,
      bemf: samples.reduce((s, c) => s + c[i].bemf, 0) / samples.length,
    }));
  }
  return avg;
}
