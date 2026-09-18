// Door subsystem: temporal segmentation of a continuous door-controller
// stream into cycles, then Normal / Abnormal-resistance classification.
//
// Segmentation is NOT based on the opening/closing command flags — the info
// kit warns not to assume those are the most robust signal, and they
// aren't: on the real Train.csv, the stream simply has NO ROWS at all
// during the idle gap between cycles (confirmed: 110 labelled segments'
// n_rows sum to exactly the file's total row count). The reliable signal
// is a jump in consecutive-row timestamps far above the ~20ms sample
// interval — every one of the 109 gaps between Train's 110 true segments
// is >100ms, and there are exactly 109 such gaps in the file.
//
// Classification is a logistic regression on 4 per-segment features,
// fitted and 5-fold cross-validated against Train.csv + its 110 labelled
// segments (96.6% CV F1) — see PS3 Door_Subsystem_Info_Kit.md for the
// labelled data this was fit against.
//
// Features are standardized (z-scored using the Train fit's own mean/std)
// before the linear layer — fitting on the raw, unscaled feature values
// (current in mA, up to ~2000+; back-EMF up to ~1800) drove the fitted
// coefficients to produce |z| in the range of 8-60 for ordinary points, not
// just extreme ones, which made EVERY prediction's confidence round to
// 100% even under honest cross-validation — a real statistical artifact
// (the classes are genuinely linearly separable on these 4 features, so an
// unregularized-in-effect fit pushes toward complete separation), not a
// display bug. Standardizing first, at the same regularization strength,
// keeps accuracy close to the original (98.2% train fit here vs 100%
// before) while spreading confidence out to a believable 41%-99.8% range
// (mean 92%) instead of saturating at ~100% for every single segment.

const GAP_THRESHOLD_MS = 100; // normal sample interval is 20ms

// Parses the dataset's native "Y-M-D-H-Mi-S-MS" timestamp (not zero-padded)
// into epoch milliseconds.
export function parseDoorTimestamp(raw) {
  const parts = raw.trim().split("-").map(Number);
  const [y, mo, d, h, mi, s, ms] = parts;
  return new Date(y, mo - 1, d, h, mi, s, ms).getTime();
}

export function segmentStream(rows) {
  const segments = [];
  let current = [];
  for (let i = 0; i < rows.length; i++) {
    if (i > 0 && rows[i].ts - rows[i - 1].ts > GAP_THRESHOLD_MS) {
      if (current.length) segments.push(current);
      current = [];
    }
    current.push(rows[i]);
  }
  if (current.length) segments.push(current);
  return segments;
}

function mean(arr) {
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}
function std(arr) {
  const m = mean(arr);
  const variance = arr.reduce((a, b) => a + (b - m) ** 2, 0) / Math.max(1, arr.length - 1);
  return Math.sqrt(variance);
}

// Fitted on all 110 Train.csv segments; feature order matters.
const FEATURE_ORDER = ["current_mean", "current_std", "current_max", "bemf_mean"];
const SCALER_MEAN = [587.2236973402883, 625.3215084657759, 2305.5636363636363, 1034.931493389984];
const SCALER_SCALE = [139.1214161931272, 127.27578797993216, 203.56283493947276, 146.11182797741625];
const COEF = [3.658764616527377, -0.40597332012442944, -1.5175230825675867, -0.8646184232321892];
const INTERCEPT = -1.4841891924833082;

export function segmentFeatures(segmentRows) {
  const current = segmentRows.map((r) => r.current);
  const bemf = segmentRows.map((r) => r.bemf);
  return {
    current_mean: mean(current),
    current_std: std(current),
    current_max: Math.max(...current),
    bemf_mean: mean(bemf),
  };
}

export function classifySegment(segmentRows) {
  const feats = segmentFeatures(segmentRows);
  const z =
    INTERCEPT +
    FEATURE_ORDER.reduce((sum, k, i) => sum + COEF[i] * ((feats[k] - SCALER_MEAN[i]) / SCALER_SCALE[i]), 0);
  const probAbnormal = 1 / (1 + Math.exp(-z));
  return {
    prediction: probAbnormal >= 0.5 ? "Abnormal resistance" : "Normal",
    probAbnormal,
  };
}

// Open vs Close is informational only (not scored — see Door_Subsystem_Info_Kit.md
// Section 2.1), but the raw stream already carries it directly: "Door is
// opening"/"Door is closing" are 0/1 flags, and within one real cycle only
// one of them is ever set (confirmed against Train_Segments_Answer.csv's
// `operation` column) — so majority vote across the segment's rows recovers
// it with no fitting involved.
function classifyOperation(segmentRows) {
  let opening = 0;
  let closing = 0;
  for (const r of segmentRows) {
    if (r.opening === 1) opening++;
    if (r.closing === 1) closing++;
  }
  return opening >= closing ? "Open" : "Close";
}

/**
 * Full pipeline: raw parsed CSV rows (from PapaParse, header:true) -> predicted segments.
 * @param {Array<Record<string,string>>} rawRows
 */
export function runDoorPipeline(rawRows) {
  const rows = rawRows
    .filter((r) => r["Datetime"])
    .map((r) => ({
      ts: parseDoorTimestamp(r["Datetime"]),
      rawTimestamp: r["Datetime"].trim(),
      current: Number(r["Motor current(mA)"]),
      voltage: Number(r["Motor Voltage(10mV)"]),
      bemf: Number(r["Motor electrodynamic force"]),
      opening: Number(r["Door is opening"]),
      closing: Number(r["Door is closing"]),
    }))
    .sort((a, b) => a.ts - b.ts);

  const segments = segmentStream(rows);

  return segments.map((seg) => {
    const { prediction, probAbnormal } = classifySegment(seg);
    const start_ts = seg[0].ts;
    const end_ts = seg[seg.length - 1].ts;
    const span = Math.max(1, end_ts - start_ts);
    return {
      start_time: seg[0].rawTimestamp,
      end_time: seg[seg.length - 1].rawTimestamp,
      start_ts,
      end_ts,
      operation: classifyOperation(seg),
      prediction,
      n_rows: seg.length,
      confidence: prediction === "Abnormal resistance" ? probAbnormal : 1 - probAbnormal,
      // Consistent 0-1 "how abnormal" signal across every cycle, unlike
      // `confidence` above (which is always about whichever class was
      // predicted) — needed for a trend chart where the same axis has to
      // mean the same thing for every point.
      probAbnormal,
      // Raw per-row signal, x-axis normalized to "% of this cycle elapsed"
      // so cycles of different real durations overlay on the same 0-100
      // scale — used by the click-to-inspect detail charts. Real
      // observed data, not derived/smoothed.
      rawSeries: seg.map((r) => ({
        pct: ((r.ts - start_ts) / span) * 100,
        current: r.current,
        voltage: r.voltage,
        bemf: r.bemf,
      })),
    };
  });
}

const SIGNAL_KEYS = ["current", "voltage", "bemf"];

// Linear-interpolates one cycle's rawSeries (real, unevenly-sampled points)
// onto an evenly-spaced 0-100% grid, so cycles of different real durations
// and row counts can be compared/averaged point-for-point.
function resampleToGrid(rawSeries, grid) {
  return grid.map((g) => {
    let lo = rawSeries[0];
    let hi = rawSeries[rawSeries.length - 1];
    for (let i = 0; i < rawSeries.length - 1; i++) {
      if (rawSeries[i].pct <= g && rawSeries[i + 1].pct >= g) {
        lo = rawSeries[i];
        hi = rawSeries[i + 1];
        break;
      }
    }
    const t = hi.pct === lo.pct ? 0 : (g - lo.pct) / (hi.pct - lo.pct);
    const point = { pct: g };
    for (const key of SIGNAL_KEYS) point[key] = lo[key] + t * (hi[key] - lo[key]);
    return point;
  });
}

/**
 * Resamples one cycle's real raw signal onto a fixed 0-100% grid (101
 * points, i.e. every 1%) — used both to build the average-normal-cycle
 * baseline and to align a selected cycle onto the same grid as that
 * baseline for the comparison charts.
 * @param {Array<{pct:number,current:number,voltage:number,bemf:number}>} rawSeries
 * @param {{gridPoints?: number}} [opts]
 */
export function resampleCycleSignal(rawSeries, opts = {}) {
  const { gridPoints = 101 } = opts;
  const grid = Array.from({ length: gridPoints }, (_, i) => (i / (gridPoints - 1)) * 100);
  return resampleToGrid(rawSeries, grid);
}

function averageOneOperation(normals, grid) {
  if (!normals.length) return [];
  const resampled = normals.map((s) => resampleToGrid(s.rawSeries, grid));
  return grid.map((pct, i) => {
    const point = { pct };
    for (const key of SIGNAL_KEYS) point[key] = mean(resampled.map((r) => r[i][key]));
    return point;
  });
}

/**
 * Averages every currently-loaded Normal cycle's signal (current, voltage,
 * back-EMF), each resampled onto the same 0-100%-of-cycle grid first, into
 * a reference "what a normal cycle typically looks like" curve — the
 * baseline a selected cycle's own signal is compared against in the detail
 * charts. Computed SEPARATELY for Open and Close cycles (an opening motion
 * and a closing motion have genuinely different current/voltage/back-EMF
 * shapes, so pooling them would blur the baseline into something neither
 * one actually looks like). Real observed data averaged together, not a
 * fitted/synthetic curve; recomputed whenever the loaded file changes.
 * @param {Array<{prediction:string, operation:string, rawSeries: any[]|null}>} segments
 * @param {{gridPoints?: number}} [opts]
 * @returns {{Open: any[], Close: any[]}}
 */
export function averageNormalCycleSignal(segments, opts = {}) {
  const { gridPoints = 101 } = opts;
  const grid = Array.from({ length: gridPoints }, (_, i) => (i / (gridPoints - 1)) * 100);
  const normals = segments.filter((s) => s.prediction === "Normal" && s.rawSeries?.length);

  return {
    Open: averageOneOperation(normals.filter((s) => s.operation === "Open"), grid),
    Close: averageOneOperation(normals.filter((s) => s.operation === "Close"), grid),
  };
}
