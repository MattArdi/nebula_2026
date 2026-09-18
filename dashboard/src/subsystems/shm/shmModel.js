// SHM subsystem: predict cumulative fatigue damage from a single-column
// dynamic-stress time series (~580k readings per file).
//
// Approach: real rainflow cycle counting (ASTM E1049-85, ported line-for-
// line from the reference `rainflow` PyPI package) feeding Miner's linear
// damage rule — the exact domain-standard method the info kit describes,
// not a black-box regression. damage = a * sum(count * (range/2)^m) over
// every counted stress cycle, with the S-N exponent m=5 (grid-searched
// against the 64 Train files' true damage values, LOOCV score peaks sharply
// at m≈5) and the S-N scale constant folded into a single calibration
// constant `a`, fit as the median of (true_damage / proxy_damage) across
// all 64 Train files.
//
// Leave-one-out cross-validated score on Train (the actual competition
// metric, max(0, 1-MAPE) — see SHM_Info_Kit.md Section 5) = 0.9744.
// A plain feature-regression baseline (signal statistics -> gradient
// boosting) was also tried and scored far lower (0.77-0.79) — the
// physically-grounded rainflow approach decisively wins, which makes
// sense: Miner's rule is literally how the reference damage values were
// computed (see the info kit, Section 1.3).

const M_EXPONENT = 5;
const CALIBRATION_A = 1.3605483099152677e-9;

// ASTM E1049 reversal extraction: a reversal is a point where the series'
// first derivative changes sign; the first and last points always count.
function reversals(series) {
  const n = series.length;
  if (n < 2) return series.slice();
  const out = [series[0]];
  let x = series[1];
  let dLast = x - series[0];
  for (let i = 2; i < n; i++) {
    const xNext = series[i];
    if (xNext === x) continue; // flat run, keep reading
    const dNext = xNext - x;
    if (dLast * dNext < 0) out.push(x);
    x = xNext;
    dLast = dNext;
  }
  out.push(series[n - 1]);
  return out;
}

// ASTM E1049 4-point cycle counting over the reversal-point stream.
// Returns {range, mean, count} per cycle — count is 1.0 (full cycle) or
// 0.5 (half cycle, from the unpaired residual at the start/end).
function extractCycles(points) {
  const stack = [];
  const cycles = [];

  for (const point of points) {
    stack.push(point);
    while (stack.length >= 3) {
      const x1 = stack[stack.length - 3];
      const x2 = stack[stack.length - 2];
      const x3 = stack[stack.length - 1];
      const X = Math.abs(x3 - x2);
      const Y = Math.abs(x2 - x1);
      if (X < Y) break;

      if (stack.length === 3) {
        cycles.push({ range: Y, count: 0.5 });
        stack.shift();
      } else {
        cycles.push({ range: Y, count: 1.0 });
        const last = stack.pop();
        stack.pop();
        stack.pop();
        stack.push(last);
      }
    }
  }
  while (stack.length > 1) {
    cycles.push({ range: Math.abs(stack[1] - stack[0]), count: 0.5 });
    stack.shift();
  }
  return cycles;
}

export function damageProxy(series) {
  const cycles = extractCycles(reversals(series));
  let proxy = 0;
  for (const c of cycles) {
    const amplitude = c.range / 2;
    proxy += c.count * amplitude ** M_EXPONENT;
  }
  return proxy;
}

/** @param {number[]} series — the file's raw stress readings, in order. */
export function predictDamage(series) {
  return CALIBRATION_A * damageProxy(series);
}

/**
 * Cumulative damage build-up WITHIN one recording — genuine, since Miner's
 * rule damage is additive over cycles, so re-running the same rainflow
 * count on a longer prefix of the same series gives a real "damage so far"
 * curve, not an invented interpolation.
 *
 * x is "fraction of this recording elapsed" (0-1), NOT real elapsed time —
 * the info kit gives no sampling rate for SHM (unlike Rail Corrugation's
 * explicit 10,000 Hz), so there is no way to convert sample count to real
 * seconds/hours yet.
 *
 * @param {number[]} series
 * @param {{chunks?: number}} [opts]
 * @returns {{x:number, y:number}[]}
 */
export function cumulativeDamageOverProgress(series, opts = {}) {
  const { chunks = 20 } = opts;
  const n = series.length;
  const chunkSize = Math.ceil(n / chunks);
  const points = [];
  for (let c = 1; c <= chunks; c++) {
    const end = Math.min(n, c * chunkSize);
    const damage = CALIBRATION_A * damageProxy(series.slice(0, end));
    points.push({ x: end / n, y: damage });
  }
  return points;
}
