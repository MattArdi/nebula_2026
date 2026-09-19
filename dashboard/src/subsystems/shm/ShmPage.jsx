import { useEffect, useState } from "react";
import Papa from "papaparse";
import { parseNumericMatrix } from "../../lib/parseNumericCsv.js";
import BatchSubsystemPage from "../../components/BatchSubsystemPage.jsx";
import DamageProgressChart from "../../components/DamageProgressChart.jsx";
import ShmCombinedChart from "../../components/ShmCombinedChart.jsx";
import { predictDamage, cumulativeDamageOverProgress } from "./shmModel.js";
import { SHM_SAMPLES, SHM_LABELS_URL } from "../../lib/sampleManifest.js";

async function parseShmFile(file) {
  const rows = await parseNumericMatrix(file);
  const series = rows.map((r) => r[0]).filter((v) => typeof v === "number" && !Number.isNaN(v));
  if (!series.length) throw new Error("No numeric readings found in file.");
  return series;
}

async function predictFile(series) {
  const damage = predictDamage(series);
  return { prediction: damage.toFixed(4) };
}

function tierFor(damage) {
  if (damage >= 0.6) return { label: "High", tone: "text-status-critical" };
  if (damage >= 0.3) return { label: "Medium", tone: "text-status-serious" };
  return { label: "Low", tone: "text-status-good" };
}

// Within 15% of the true damage value counts as a match — damage is a
// continuous prediction, so exact equality would never happen.
function matchesTrueLabel(prediction, trueLabel) {
  const pred = Number(prediction);
  const truth = Number(trueLabel);
  if (truth === 0) return Math.abs(pred) < 0.01;
  return Math.abs(pred - truth) / Math.abs(truth) <= 0.15;
}

function computeStats(results) {
  const values = results.map((r) => Number(r.prediction));
  const avg = values.reduce((a, b) => a + b, 0) / values.length;
  const high = values.filter((v) => v >= 0.6).length;
  const medium = values.filter((v) => v >= 0.3 && v < 0.6).length;
  const withTruth = results.filter((r) => r.trueLabel != null);
  const stats = [
    { label: "Files", value: results.length },
    { label: "Avg. predicted damage", value: avg.toFixed(3) },
    { label: "Medium damage", value: medium, tone: medium > 0 ? "serious" : "good" },
    { label: "High damage", value: high, tone: high > 0 ? "critical" : "good" },
  ];
  if (withTruth.length) {
    const avgErrorPct =
      (withTruth.reduce((sum, r) => sum + Math.abs(Number(r.prediction) - Number(r.trueLabel)) / Math.max(1e-9, Math.abs(Number(r.trueLabel))), 0) /
        withTruth.length) *
      100;
    stats.push({ label: "Avg. error vs Train_Labels.csv", value: `${avgErrorPct.toFixed(1)}%`, tone: avgErrorPct < 15 ? "good" : "serious" });
  }
  return stats;
}

// Health = (1 - average predicted damage) * 100 — damage is already on a
// physically meaningful 0-1 scale where 1.0 means fatigue failure per
// Miner's rule, so this falls straight out of the model with no invented
// scaling. Clamped since a (rare, off-scale) prediction above 1.0 shouldn't
// go negative.
function computeHealth(results) {
  const values = results.map((r) => Number(r.prediction));
  const avg = values.reduce((a, b) => a + b, 0) / values.length;
  return Math.max(0, (1 - avg) * 100);
}

// Per-file health, same (1-damage)*100 conversion as the aggregate.
function computeEntities(results) {
  return results.map((r) => ({
    id: r.file_id,
    label: tierFor(Number(r.prediction)).label,
    value: Math.max(0, (1 - Number(r.prediction)) * 100),
    valueType: "health",
  }));
}

function renderCell(row) {
  const damage = Number(row.prediction);
  const tier = tierFor(damage);
  return (
    <span className="inline-flex items-center gap-2">
      <span className={`font-semibold tabular-nums ${tier.tone}`}>{row.prediction}</span>
      <span className="text-xs text-ink-muted">{tier.label}</span>
    </span>
  );
}

function defaultSelect(rows) {
  return [...rows].sort((a, b) => Number(b.prediction) - Number(a.prediction))[0]?.file_id;
}

function renderDetail(row) {
  const points = cumulativeDamageOverProgress(row.parsed, { chunks: 20 });
  const truthNote =
    row.trueLabel != null
      ? ` True damage (Train_Labels.csv): ${row.trueLabel} — predicted ${row.prediction} (${row.matchesTruth ? "within 15%" : "off by more than 15%"}).`
      : "";

  return (
    <DamageProgressChart
      title={`${row.file_id} — cumulative damage build-up`}
      subtitle={`Cumulative Miner's-rule damage recomputed on successively longer prefixes of this recording.${truthNote}`}
      data={points}
      caveat="X-axis is fraction of this recording elapsed, not real time — SHM's sampling rate isn't published."
    />
  );
}

function renderCombined(results) {
  return (
    <ShmCombinedChart
      title="All files — predicted vs true damage"
      subtitle="Every loaded file plotted as one point, ordered by true damage ascending, against Train_Labels.csv."
      results={results}
      caveat="Points are ordered by damage value, not file name."
    />
  );
}

export default function ShmPage({ onSummary }) {
  const [trueLabels, setTrueLabels] = useState(null);

  useEffect(() => {
    fetch(SHM_LABELS_URL)
      .then((res) => (res.ok ? res.text() : null))
      .then((text) => {
        if (!text) return;
        Papa.parse(text, {
          header: true,
          skipEmptyLines: true,
          complete: (results) => {
            const map = {};
            for (const row of results.data) map[row.filename] = row.damage;
            setTrueLabels(map);
          },
        });
      })
      .catch(() => {});
  }, []);

  return (
    <BatchSubsystemPage
      title="Cumulative fatigue damage predictions"
      description={
        <>
          Drop one or many dynamic-stress time-series files (e.g. <code className="text-ink-secondary">train01.csv</code>
          ...<code className="text-ink-secondary">train64.csv</code>, or a .zip of several). Each file runs through
          rainflow cycle counting and Miner's linear damage rule to predict a cumulative fatigue-damage number.
          Opens pre-loaded with 16 labelled Train files, checked against{" "}
          <code className="text-ink-secondary">Train_Labels.csv</code>. Click a row for that file's damage build-up
          chart. Drop your own files (Train or Test) to replace them.
        </>
      }
      csvFilename="shm_predictions.csv"
      accept=".csv"
      parseFile={parseShmFile}
      predictFile={predictFile}
      predictionHeader="Predicted damage"
      sampleFiles={SHM_SAMPLES}
      trueLabels={trueLabels}
      matchesTrueLabel={matchesTrueLabel}
      computeStats={computeStats}
      computeHealth={computeHealth}
      computeEntities={computeEntities}
      renderCell={renderCell}
      renderCombined={renderCombined}
      renderDetail={renderDetail}
      defaultSelect={defaultSelect}
      onSummary={onSummary}
    />
  );
}
