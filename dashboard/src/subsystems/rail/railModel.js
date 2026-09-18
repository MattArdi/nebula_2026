// Rail Corrugation: 3-class classification (Normal / Side I / Side II) from
// a 1-second, 10,000 Hz, 129-column axle-box vibration+shock recording.
//
// Model: multinomial logistic regression on 25 hand-engineered time-domain
// features (no FFT — kept fully portable to plain JS), standardized then
// softmax. Fitted on all 272 Train files; 5-fold stratified cross-validated
// macro F1 = 0.770 (vs. 0.33 for a trivial "always predict Normal"
// baseline) — see PS3 Rail_Corrugation_Info_Kit.md Section 4 for why macro
// F1 (not accuracy) is the scored metric, given the dataset is only ~14%
// fault examples.
//
// Column layout (confirmed from the real Train files): column 0 is a
// rotating-speed toggle signal (0/1, from a 90-tooth wheel); columns 1-128
// are 8 cars x 16 columns each, and within each 16-column car block the
// offsets are [pos1_vib, pos1_shock, pos2_vib, pos2_shock, ..., pos8_vib,
// pos8_shock] — Side I = positions {1,3,5,7} (vib offsets 0,4,8,12; shock
// offsets 1,5,9,13), Side II = positions {2,4,6,8} (vib offsets 2,6,10,14;
// shock offsets 3,7,11,15).

const WHEEL_DIAMETER_M = 0.85;
const TEETH_PER_REV = 90;

const FEATURE_ORDER = [
  "speed_kmh", "s1_vib_rms_mean", "s1_vib_rms_max", "s1_vib_p2p_mean",
  "s1_vib_kurt_mean", "s1_vib_crest_mean", "s1_vib_zcr_mean", "s1_shock_rms_mean",
  "s1_shock_rms_max", "s1_shock_kurt_mean", "s1_shock_p95_mean", "s2_vib_rms_mean",
  "s2_vib_rms_max", "s2_vib_p2p_mean", "s2_vib_kurt_mean", "s2_vib_crest_mean",
  "s2_vib_zcr_mean", "s2_shock_rms_mean", "s2_shock_rms_max", "s2_shock_kurt_mean",
  "s2_shock_p95_mean", "ratio_vib_rms_s1_s2", "ratio_shock_rms_s1_s2",
  "ratio_vib_kurt_s1_s2", "ratio_shock_kurt_s1_s2",
];

const SCALER_MEAN = [
  32.02578820977605, 0.30904630635858366, 0.7430420523104563, 3.266714600955739,
  6.053119013641211, 5.147007037911996, 0.06031698068703926, 1.3246545000294623,
  2.7894743369780572, 22.065418389582266, 2.8280009942896203, 0.3179630809972286,
  0.7831846978564689, 3.4014112809125114, 5.886981917619984, 5.39424273745059,
  0.07097526158865883, 1.4637846339864704, 2.6751183055763352, 22.223988918498176,
  3.1063216574051773, 1.043792379390879, 0.9056860614367591, 1.0151938688566087,
  1.0072428237061009,
];
const SCALER_SCALE = [
  21.378207395631655, 0.1853654383656689, 0.496465357126578, 2.348698050725062,
  3.943920435780631, 1.1589769821094242, 0.025241839666229518, 0.09923178304739783,
  0.4383932950280848, 9.073869755066434, 0.29822985195266977, 0.20967097285738684,
  0.5570510623732005, 2.4847687336320754, 3.8758327914262063, 0.972308609064952,
  0.015690610598551705, 0.11096383475849289, 0.4578501885223898, 9.025101190971867,
  0.3257412799809616, 0.16154697088025735, 0.03514094168939338, 0.10155401533379703,
  0.16014716964679163,
];
const CLASSES = ["Normal", "Side I", "Side II"];
const COEF = [
  [-0.06916676050083516, -0.19222087487176162, -1.3369341426691934, 0.04427436671792956,
    0.3247804540959482, 0.14212970582506804, -0.14781625461605868, -0.056403941356403815,
    -0.5276506201240948, -0.03338378410102802, 0.060191923643197635, -0.28362233926351477,
    -0.9106242240677939, 0.08512909903636151, 0.36363046287722706, 0.39846799318917464,
    -0.036966611629983666, -0.07308911925711231, 0.16278286029431846, 0.12313270212560277,
    -0.0224838726944873, -0.034202516608033924, -0.03928173720377996, -0.026676207781704635,
    -0.18670605927249886],
  [0.02519339936171959, 0.09615637280891375, 1.1483704592742598, 0.14024394135639537,
    -0.0799286237404726, 0.0072606980302295406, -0.1370229172599813, 0.3730644572463439,
    0.5760164799888267, -0.5286883289974688, -0.47826882899201, -0.28276227148025457,
    -0.47487894144128895, -0.10316938486530794, 0.030567441569561333, -0.06102195858420238,
    0.15420122124908062, 0.23997799777261908, -0.1235367331920987, -0.622634229765903,
    -0.16897101540635037, 0.7852868829630972, 0.23989680219650802, -0.3270242518019872,
    0.04828920965151962],
  [0.043973361139116374, 0.09606450206284706, 0.1885636833949344, -0.18451830807432462,
    -0.24485183035547656, -0.14939040385529703, 0.28483917187604024, -0.3166605158899399,
    -0.04836585986473192, 0.5620721130984971, 0.41807690534881325, 0.5663846107437687,
    1.3855031655090835, 0.018040285828945977, -0.39419790444678854, -0.33744603460497213,
    -0.11723460961909729, -0.16688887851550588, -0.039246127102219586, 0.4995015276403005,
    0.19145488810083805, -0.751084366355064, -0.20061506499272855, 0.35370045958369134,
    0.13841684962097953],
];
const INTERCEPT = [3.136633262951482, -0.9156580119794953, -2.22097525097199];

function channelStats(x) {
  const n = x.length;
  let sum = 0;
  for (let i = 0; i < n; i++) sum += x[i];
  const mean = sum / n;

  let sqSum = 0, m4Sum = 0, sumSq = 0, maxAbs = 0, max = -Infinity, min = Infinity;
  for (let i = 0; i < n; i++) {
    const v = x[i];
    const d = v - mean;
    const d2 = d * d;
    sqSum += d2;
    m4Sum += d2 * d2;
    sumSq += v * v;
    const a = Math.abs(v);
    if (a > maxAbs) maxAbs = a;
    if (v > max) max = v;
    if (v < min) min = v;
  }
  const variance = sqSum / n;
  const std = Math.max(Math.sqrt(variance), 1e-12);
  const rms = Math.sqrt(sumSq / n);
  const kurt = m4Sum / n / std ** 4 - 3;
  const p2p = max - min;
  const crest = maxAbs / Math.max(rms, 1e-12);

  let flips = 0;
  let prevSign = x[0] >= 0 ? 1 : -1;
  for (let i = 1; i < n; i++) {
    const s = x[i] >= 0 ? 1 : -1;
    if (s !== prevSign) flips++;
    prevSign = s;
  }
  const zcr = flips / (n - 1);

  const absSorted = Array.from(x, Math.abs).sort((a, b) => a - b);
  const h = 0.95 * (n - 1);
  const lo = Math.floor(h);
  const hi = Math.ceil(h);
  const p95 = absSorted[lo] + (absSorted[hi] - absSorted[lo]) * (h - lo);

  return { rms, kurt, p2p, crest, zcr, p95 };
}

function aggregate(channels, key) {
  const vals = channels.map((s) => s[key]);
  return vals.reduce((a, b) => a + b, 0) / vals.length;
}
function aggregateMax(channels, key) {
  return Math.max(...channels.map((s) => s[key]));
}

const VIB_OFFSETS = { side1: [0, 4, 8, 12], side2: [2, 6, 10, 14] };
const SHOCK_OFFSETS = { side1: [1, 5, 9, 13], side2: [3, 7, 11, 15] };

/**
 * @param {number[][]} rows — rows x 129 numeric columns, in original file order
 *   (column 0 = speed toggle, columns 1-128 = the 8x16 vibration/shock block).
 */
export function extractFeatures(rows) {
  const n = rows.length;
  const speedCol = new Array(n);
  for (let i = 0; i < n; i++) speedCol[i] = rows[i][0];

  let rising = 0;
  for (let i = 1; i < n; i++) {
    if (speedCol[i - 1] === 0 && speedCol[i] === 1) rising++;
  }
  const revolutions = rising / TEETH_PER_REV;
  const distanceM = revolutions * Math.PI * WHEEL_DIAMETER_M;
  const speedKmh = distanceM * 3.6; // 1 file = 1 second, so distance/s -> already m/s, x3.6 -> km/h

  function extractChannel(car, offset) {
    const col = 1 + car * 16 + offset;
    const arr = new Array(n);
    for (let i = 0; i < n; i++) arr[i] = rows[i][col];
    return channelStats(arr);
  }

  const sideFeatures = {};
  for (const side of ["side1", "side2"]) {
    const vibChannels = [];
    const shockChannels = [];
    for (let car = 0; car < 8; car++) {
      for (const off of VIB_OFFSETS[side]) vibChannels.push(extractChannel(car, off));
      for (const off of SHOCK_OFFSETS[side]) shockChannels.push(extractChannel(car, off));
    }
    const prefix = side === "side1" ? "s1" : "s2";
    sideFeatures[`${prefix}_vib_rms_mean`] = aggregate(vibChannels, "rms");
    sideFeatures[`${prefix}_vib_rms_max`] = aggregateMax(vibChannels, "rms");
    sideFeatures[`${prefix}_vib_p2p_mean`] = aggregate(vibChannels, "p2p");
    sideFeatures[`${prefix}_vib_kurt_mean`] = aggregate(vibChannels, "kurt");
    sideFeatures[`${prefix}_vib_crest_mean`] = aggregate(vibChannels, "crest");
    sideFeatures[`${prefix}_vib_zcr_mean`] = aggregate(vibChannels, "zcr");
    sideFeatures[`${prefix}_shock_rms_mean`] = aggregate(shockChannels, "rms");
    sideFeatures[`${prefix}_shock_rms_max`] = aggregateMax(shockChannels, "rms");
    sideFeatures[`${prefix}_shock_kurt_mean`] = aggregate(shockChannels, "kurt");
    sideFeatures[`${prefix}_shock_p95_mean`] = aggregate(shockChannels, "p95");
  }

  const eps = 1e-9;
  const features = {
    speed_kmh: speedKmh,
    ...sideFeatures,
    ratio_vib_rms_s1_s2: sideFeatures.s1_vib_rms_mean / (sideFeatures.s2_vib_rms_mean + eps),
    ratio_shock_rms_s1_s2: sideFeatures.s1_shock_rms_mean / (sideFeatures.s2_shock_rms_mean + eps),
    ratio_vib_kurt_s1_s2:
      (sideFeatures.s1_vib_kurt_mean + 10) / (sideFeatures.s2_vib_kurt_mean + 10 + eps),
    ratio_shock_kurt_s1_s2:
      (sideFeatures.s1_shock_kurt_mean + 10) / (sideFeatures.s2_shock_kurt_mean + 10 + eps),
  };

  return FEATURE_ORDER.map((k) => features[k]);
}

function softmax(zs) {
  const max = Math.max(...zs);
  const exps = zs.map((z) => Math.exp(z - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

export function classify(featureVector) {
  const scaled = featureVector.map((v, i) => (v - SCALER_MEAN[i]) / SCALER_SCALE[i]);
  const zs = COEF.map(
    (coefRow, k) => INTERCEPT[k] + coefRow.reduce((sum, c, i) => sum + c * scaled[i], 0)
  );
  const probs = softmax(zs);
  let best = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
  return { prediction: CLASSES[best], probabilities: Object.fromEntries(CLASSES.map((c, i) => [c, probs[i]])) };
}

/** @param {number[][]} rows */
export function predictRailFile(rows) {
  const features = extractFeatures(rows);
  const { prediction, probabilities } = classify(features);
  // Signed severity score = P(Side I) − P(Side II), straight from the
  // classifier's own softmax output (so it reflects all 25 features
  // together, the same way the prediction itself is decided) — validated
  // against the real bundled Train files: Normal sits within ±0.07 of 0,
  // every Side I file lands well above +0.6, every Side II file well below
  // -0.5. A single raw feature (e.g. the Side I/Side II vibration RMS
  // ratio alone) was tried first and was too noisy/overlapping to use —
  // this margin is what the model actually decides with, not a proxy.
  const severityScore = probabilities["Side I"] - probabilities["Side II"];
  return { prediction, probabilities, severityScore };
}

