import { useEffect, useState } from "react";
import Papa from "papaparse";
import BatchSubsystemPage from "../../components/BatchSubsystemPage.jsx";
import ShmCombinedChart from "../../components/ShmCombinedChart.jsx";
import DamageProgressChart from "../../components/DamageProgressChart.jsx";
import { predictShm } from "../../lib/apiClient.js";
import { SHM_SAMPLES, SHM_LABELS_URL } from "../../lib/sampleManifest.js";

// The backend runs rainflow counting + Miner's-rule damage directly on the
// raw file — no client-side parsing needed.
async function parseShmFile(file) {
  return file;
}

async function predictFile(file) {
  const result = await predictShm([file]);
  const row = result.rows[0]; // { file_id, prediction }
  const diag = result.diagnostics?.[0]; // { remaining_life_pct, error_band_pct, extrapolation_*, damage_progress }
  return {
    prediction: row.prediction,
    remainingLifePct: diag?.remaining_life_pct ?? null,
    errorBandPct: diag?.error_band_pct ?? null,
    extrapolationFlagged: diag?.extrapolation_flagged ?? false,
    extrapolationReason: diag?.extrapolation_reason ?? null,
    damageProgress: diag?.damage_progress ?? null,
  };
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
  const avgRemaining = results.reduce((sum, r) => sum + (r.remainingLifePct ?? (1 - Number(r.prediction)) * 100), 0) / results.length;
  const high = values.filter((v) => v >= 0.6).length;
  const medium = values.filter((v) => v >= 0.3 && v < 0.6).length;
  const extrapolated = results.filter((r) => r.extrapolationFlagged).length;
  const withTruth = results.filter((r) => r.trueLabel != null);
  const stats = [
    { label: "Files", value: results.length },
    { label: "Avg. fatigue life remaining", value: `${avgRemaining.toFixed(0)}%`, tone: avgRemaining < 40 ? "critical" : avgRemaining < 70 ? "serious" : "good" },
    { label: "Medium damage", value: medium, tone: medium > 0 ? "serious" : "good" },
    { label: "High damage", value: high, tone: high > 0 ? "critical" : "good" },
  ];
  if (extrapolated > 0) {
    stats.push({ label: "Extrapolated (outside Train's range)", value: extrapolated, tone: "serious" });
  }
  if (withTruth.length) {
    const avgErrorPct =
      (withTruth.reduce((sum, r) => sum + Math.abs(Number(r.prediction) - Number(r.trueLabel)) / Math.max(1e-9, Math.abs(Number(r.trueLabel))), 0) /
        withTruth.length) *
      100;
    stats.push({ label: "Avg. error vs Train_Labels.csv", value: `${avgErrorPct.toFixed(1)}%`, tone: avgErrorPct < 15 ? "good" : "serious" });
  }
  return stats;
}

// Health = fatigue life remaining, the same (1-damage)*100 the backend now
// computes directly (damage is already on a physically meaningful 0-1
// scale where 1.0 means fatigue failure per Miner's rule).
function computeHealth(results) {
  const values = results.map((r) => r.remainingLifePct ?? (1 - Number(r.prediction)) * 100);
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function computeEntities(results) {
  return results.map((r) => ({
    id: r.file_id,
    label: tierFor(Number(r.prediction)).label,
    value: r.remainingLifePct ?? Math.max(0, (1 - Number(r.prediction)) * 100),
    valueType: "health",
  }));
}

function renderCell(row) {
  const damage = Number(row.prediction);
  const tier = tierFor(damage);
  const remaining = row.remainingLifePct;
  return (
    <span className="inline-flex items-center gap-2">
      {remaining != null ? (
        <>
          <span className={`font-semibold tabular-nums ${tier.tone}`}>{remaining.toFixed(0)}% life remaining</span>
          {row.errorBandPct != null && <span className="text-xs text-ink-muted">(± {row.errorBandPct.toFixed(1)}%)</span>}
        </>
      ) : (
        <span className={`font-semibold tabular-nums ${tier.tone}`}>{row.prediction}</span>
      )}
      <span className="text-xs text-ink-muted">{tier.label} damage</span>
      {row.extrapolationFlagged && <span className="text-xs text-status-warning">extrapolated</span>}
    </span>
  );
}

function renderDetail(row) {
  if (!row.damageProgress?.length) return null;
  const points = row.damageProgress.map((p) => ({ x: p.pct / 100, y: p.damage }));
  const truthNote =
    row.trueLabel != null
      ? ` True damage (Train_Labels.csv): ${row.trueLabel} — predicted ${row.prediction} (${row.matchesTruth ? "within 15%" : "off by more than 15%"}).`
      : "";
  const remaining = row.remainingLifePct;
  const errorBand = row.errorBandPct;

  return (
    <DamageProgressChart
      title={`${row.file_id} — cumulative damage build-up`}
      subtitle={`Fatigue life remaining: ${remaining != null ? remaining.toFixed(0) : "—"}%${
        errorBand != null ? ` (± ${errorBand.toFixed(1)}%, from the shipped calibration's leave-one-out error)` : ""
      }. Real rainflow counting recomputed on successively longer prefixes of this recording.${truthNote}`}
      data={points}
      caveat={
        row.extrapolationFlagged
          ? `Extrapolation: ${row.extrapolationReason}`
          : "X-axis is fraction of this recording elapsed, not real time — SHM's sampling rate isn't published."
      }
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
      description="Observes the stress readings from each recording over time, then estimates how much fatigue damage has built up."
      csvFilename="shm_predictions.csv"
      accept=".csv"
      parseFile={parseShmFile}
      predictFile={predictFile}
      predictionHeader="Fatigue life remaining"
      sampleFiles={SHM_SAMPLES}
      trueLabels={trueLabels}
      matchesTrueLabel={matchesTrueLabel}
      computeStats={computeStats}
      computeHealth={computeHealth}
      computeEntities={computeEntities}
      renderCell={renderCell}
      renderCombined={renderCombined}
      renderDetail={renderDetail}
      onSummary={onSummary}
    />
  );
}
