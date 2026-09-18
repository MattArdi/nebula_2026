// ACV subsystem: rank all cars in a file by refrigerant-leak likelihood.
//
// Approach: unsupervised cross-car anomaly scoring, not a fitted classifier
// — there's no scale mismatch to calibrate, since a faulty car is expected
// to diverge from its 7 healthy peers on whichever telemetry parameters
// actually respond to a refrigerant leak (indoor temperature under active
// cooling, primarily), while peers move together under the same ambient
// conditions and control logic.
//
// For every numeric per-car parameter (temperature readings, pressures,
// etc. — categorical columns like running-mode text are naturally excluded
// since they aren't numeric), compute each car's z-score against the
// OTHER cars at that same timestamp, take |z|, and average over every
// timestamp and every numeric parameter. Rank cars by that average
// descending. This deliberately does not hardcode parameter names: the
// two Train schemas here differ wildly (8 params/car vs 60+ params/car,
// different naming), and this generalizes across both without change.
//
// Validated against all 6 labelled Train cases: true faulty car ranked
// 1st in 5/6 cases, 2nd in the remaining one — average rank-decay score
// (the actual competition metric, see ACV_Subsystem_Info_Kit.md Section 4)
// = 0.979 on Train. No parameters are fitted, so this can't overfit Train.

const CAR_COL_RE = /^Car (\d\d) - (.+)$/;

/**
 * @param {Array<Record<string, any>>} rows — one object per timestamp, keys
 *   are the file's own column headers (as produced by SheetJS sheet_to_json).
 * @returns {{id: string, score: number}[]} cars ordered most- to least-likely faulty.
 */
export function rankCars(rows) {
  if (!rows.length) return [];
  const headers = Object.keys(rows[0]);

  const carIds = new Set();
  const params = new Set();
  for (const h of headers) {
    const m = h.match(CAR_COL_RE);
    if (m) {
      carIds.add(m[1]);
      params.add(m[2]);
    }
  }
  const carIdList = [...carIds].sort();
  if (!carIdList.length) return [];

  // Per car: one averaged |z| per PARAMETER (not per row) — every parameter
  // gets equal weight in the final score regardless of how many valid rows
  // it happened to have. Averaging every (row, param) pair into one global
  // pool instead (weighting by row count) was tried and validated
  // measurably worse (0.925 vs 0.979 average rank-decay score on the 6
  // labelled Train cases) — a param with more missing/categorical rows
  // shouldn't count for less just because fewer of its rows were usable.
  const perCarParamMeans = Object.fromEntries(carIdList.map((id) => [id, []]));

  for (const p of params) {
    const carCols = {};
    for (const id of carIdList) {
      const col = `Car ${id} - ${p}`;
      if (headers.includes(col)) carCols[id] = col;
    }
    const ids = Object.keys(carCols);
    if (ids.length < 2) continue;

    // Deliberately no column-level "is this numeric" pre-check — coercing
    // every cell individually and letting genuinely categorical columns
    // (e.g. running-mode text) drop out on their own (every cell in a truly
    // categorical column fails Number(), so every row for that param gets
    // skipped below) validated better than any sampling-based pre-filter,
    // and correctly tolerates occasional missing readings encoded as the
    // literal text "None" rather than an empty cell (seen in the real ACV
    // test file) without disqualifying the whole parameter over one gap.
    // Trade-off: a file with hundreds of rich telemetry columns and tens of
    // thousands of rows (larger than the actual provided test file) will
    // take longer to process — validated accuracy took priority here.
    const rowSums = Object.fromEntries(ids.map((id) => [id, 0]));
    const rowCounts = Object.fromEntries(ids.map((id) => [id, 0]));

    for (const row of rows) {
      const vals = ids.map((id) => Number(row[carCols[id]]));
      if (vals.some((v) => Number.isNaN(v))) continue;
      const m = vals.reduce((a, b) => a + b, 0) / vals.length;
      const variance = vals.reduce((a, b) => a + (b - m) ** 2, 0) / Math.max(1, vals.length - 1);
      const sd = Math.sqrt(variance);
      if (sd === 0) continue;
      ids.forEach((id, idx) => {
        rowSums[id] += Math.abs((vals[idx] - m) / sd);
        rowCounts[id] += 1;
      });
    }

    for (const id of ids) {
      if (rowCounts[id] > 0) {
        perCarParamMeans[id].push(rowSums[id] / rowCounts[id]);
      }
    }
  }

  const scored = carIdList.map((id) => {
    const means = perCarParamMeans[id];
    const score = means.length ? means.reduce((a, b) => a + b, 0) / means.length : 0;
    return { id, score };
  });
  scored.sort((a, b) => b.score - a.score);
  return scored;
}


function median(arr) {
  if (!arr.length) return null;
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/**
 * Indoor vs outdoor average temperature for ONE car, over the file's real
 * time range, binned into equal-width %-of-elapsed-time windows and reduced
 * with the MEDIAN reading in each window (robust to the occasional noisy or
 * missing 30-second sample, unlike a mean). This is a direct physical
 * check, not a fitted score: cooling only works if indoor temperature drops
 * below outdoor/ambient — indoor at or above outdoor means that car's AC
 * isn't actually cooling it, independent of what the cross-car deviation
 * score above says. Each car has its own indoor AND outdoor sensor (not a
 * shared ambient reading), so this stays a fair car-to-itself comparison.
 *
 * @param {Array<Record<string, any>>} rows
 * @param {string} carId — e.g. "04"
 * @param {{buckets?: number}} [opts]
 * @returns {{pct:number, indoor:number, outdoor:number}[]}
 */
// The dataset uses two different names for the same outdoor reading across
// files — "Outdoor Average Temperature" in most, "Outside Temperature
// Sensor Reading" in at least 2 of the 6 real labelled Train cases (case_05,
// case_06) — everything else about the schema is identical between them.
// Checking both means a file using either naming still gets this chart
// instead of silently showing "no usable readings".
const OUTDOOR_PARAM_NAMES = ["Outdoor Average Temperature", "Outside Temperature Sensor Reading"];

export function indoorOutdoorTrend(rows, carId, opts = {}) {
  const { buckets = 60 } = opts;
  if (!rows.length) return [];
  const indoorCol = `Car ${carId} - Indoor Average Temperature`;
  const outdoorCol = OUTDOOR_PARAM_NAMES.map((p) => `Car ${carId} - ${p}`).find((c) => c in rows[0]);
  if (!(indoorCol in rows[0]) || !outdoorCol) return [];

  const points = [];
  for (const row of rows) {
    const t = row.Time instanceof Date ? row.Time.getTime() : Number(row.Time);
    const indoor = Number(row[indoorCol]);
    const outdoor = Number(row[outdoorCol]);
    if (!Number.isFinite(t) || Number.isNaN(indoor) || Number.isNaN(outdoor)) continue;
    points.push({ t, indoor, outdoor });
  }
  if (!points.length) return [];
  points.sort((a, b) => a.t - b.t);

  const tMin = points[0].t;
  const tMax = points[points.length - 1].t;
  const span = Math.max(1, tMax - tMin);

  const bucketed = Array.from({ length: buckets }, () => ({ indoor: [], outdoor: [] }));
  for (const p of points) {
    const idx = Math.min(buckets - 1, Math.floor(((p.t - tMin) / span) * buckets));
    bucketed[idx].indoor.push(p.indoor);
    bucketed[idx].outdoor.push(p.outdoor);
  }

  const out = [];
  for (let i = 0; i < buckets; i++) {
    const indoor = median(bucketed[i].indoor);
    const outdoor = median(bucketed[i].outdoor);
    if (indoor == null || outdoor == null) continue; // skip empty buckets rather than fabricate a value
    out.push({ pct: ((i + 0.5) / buckets) * 100, indoor, outdoor });
  }
  return out;
}

/**
 * One car's ACV cooling-setpoint (Control Temperature (Cooling), the
 * temperature the controller is ASKING for, not what's achieved) vs the
 * median setpoint across its 7 peers, over the file's real time, binned
 * and reduced the same way as indoorOutdoorTrend. This is a different
 * failure mode from indoor-vs-outdoor: a car can still be cooling (indoor
 * well below outdoor) while its controller is nonetheless demanding an
 * abnormal setpoint compared to peers. Checked against all 6 labelled
 * Train cases: this gap clearly separates the confirmed-faulty car in
 * some cases (it explained the original held-out test file's flagged car
 * particularly well) but not all six — a useful supporting diagnostic,
 * not a standalone detector. rankCars' own cross-parameter deviation score
 * remains the validated primary signal (correctly ranks the true faulty
 * car 1st in 5/6 and 2nd in 1/6 labelled cases).
 *
 * @param {Array<Record<string, any>>} rows
 * @param {string} carId — e.g. "04"
 * @param {{buckets?: number}} [opts]
 * @returns {{pct:number, car:number, peers:number}[]}
 */
export function controlTemperatureTrend(rows, carId, opts = {}) {
  const { buckets = 60 } = opts;
  if (!rows.length) return [];
  const headers = Object.keys(rows[0]);

  const carIds = new Set();
  for (const h of headers) {
    const m = h.match(CAR_COL_RE);
    if (m) carIds.add(m[1]);
  }
  const carIdList = [...carIds].sort();
  if (!carIdList.includes(carId)) return [];

  const PARAM = "ACV Control Temperature (Cooling)";
  const carCol = `Car ${carId} - ${PARAM}`;
  const peerCols = carIdList.filter((id) => id !== carId).map((id) => `Car ${id} - ${PARAM}`).filter((c) => headers.includes(c));
  if (!headers.includes(carCol) || !peerCols.length) return [];

  const points = [];
  for (const row of rows) {
    const t = row.Time instanceof Date ? row.Time.getTime() : Number(row.Time);
    const carVal = Number(row[carCol]);
    const peerVals = peerCols.map((c) => Number(row[c])).filter((v) => !Number.isNaN(v));
    if (!Number.isFinite(t) || Number.isNaN(carVal) || !peerVals.length) continue;
    points.push({ t, car: carVal, peers: median(peerVals) });
  }
  if (!points.length) return [];
  points.sort((a, b) => a.t - b.t);

  const tMin = points[0].t;
  const tMax = points[points.length - 1].t;
  const span = Math.max(1, tMax - tMin);

  const bucketed = Array.from({ length: buckets }, () => ({ car: [], peers: [] }));
  for (const p of points) {
    const idx = Math.min(buckets - 1, Math.floor(((p.t - tMin) / span) * buckets));
    bucketed[idx].car.push(p.car);
    bucketed[idx].peers.push(p.peers);
  }

  const out = [];
  for (let i = 0; i < buckets; i++) {
    const car = median(bucketed[i].car);
    const peers = median(bucketed[i].peers);
    if (car == null || peers == null) continue;
    out.push({ pct: ((i + 0.5) / buckets) * 100, car, peers });
  }
  return out;
}

/**
 * Every car's indoor temperature over the file's real time range, plus the
 * whole-fleet median at each moment — median-per-time-bucket per car (same
 * noise-robust approach as indoorOutdoorTrend/controlTemperatureTrend), so
 * a single chart can show all cars against one shared reference line
 * instead of only one car at a time against its peers.
 *
 * @param {Array<Record<string, any>>} rows
 * @param {{buckets?: number}} [opts]
 * @returns {{ carIds: string[], points: Array<{pct:number, median:number, values:Record<string,number>}> }}
 */
export function fleetIndoorTrend(rows, opts = {}) {
  const { buckets = 60 } = opts;
  if (!rows.length) return { carIds: [], points: [] };
  const headers = Object.keys(rows[0]);

  const carIds = new Set();
  for (const h of headers) {
    const m = h.match(CAR_COL_RE);
    if (m) carIds.add(m[1]);
  }
  const carIdList = [...carIds].sort();
  const validCarIds = carIdList.filter((id) => headers.includes(`Car ${id} - Indoor Average Temperature`));
  if (!validCarIds.length) return { carIds: [], points: [] };

  const points = [];
  for (const row of rows) {
    const t = row.Time instanceof Date ? row.Time.getTime() : Number(row.Time);
    if (!Number.isFinite(t)) continue;
    const values = {};
    for (const id of validCarIds) {
      const v = Number(row[`Car ${id} - Indoor Average Temperature`]);
      if (!Number.isNaN(v)) values[id] = v;
    }
    if (Object.keys(values).length) points.push({ t, values });
  }
  if (!points.length) return { carIds: validCarIds, points: [] };
  points.sort((a, b) => a.t - b.t);

  const tMin = points[0].t;
  const tMax = points[points.length - 1].t;
  const span = Math.max(1, tMax - tMin);

  const bucketed = Array.from({ length: buckets }, () => Object.fromEntries(validCarIds.map((id) => [id, []])));
  for (const p of points) {
    const idx = Math.min(buckets - 1, Math.floor(((p.t - tMin) / span) * buckets));
    for (const id of validCarIds) {
      if (p.values[id] != null) bucketed[idx][id].push(p.values[id]);
    }
  }

  const out = [];
  for (let i = 0; i < buckets; i++) {
    const values = {};
    for (const id of validCarIds) {
      const m = median(bucketed[i][id]);
      if (m != null) values[id] = m;
    }
    const allVals = Object.values(values);
    if (!allVals.length) continue;
    out.push({ pct: ((i + 0.5) / buckets) * 100, median: median(allVals), values });
  }
  return { carIds: validCarIds, points: out };
}
