import { useEffect, useState } from "react";
import Papa from "papaparse";
import BatchSubsystemPage from "../../components/BatchSubsystemPage.jsx";
import { LabelBadge } from "../../components/ui.jsx";
import RailProbabilityChart from "../../components/RailProbabilityChart.jsx";
import { predictRail } from "../../lib/apiClient.js";
import { RAIL_SAMPLES, RAIL_LABELS_URL } from "../../lib/sampleManifest.js";

// The backend needs the raw File (feature extraction happens server-side,
// in the real validated pipeline) — no client-side parsing needed.
async function parseRailFile(file) {
  return file;
}

async function predictFile(file) {
  const result = await predictRail([file]);
  const row = result.rows[0]; // { file_id, prediction }
  const diag = result.diagnostics?.[0]; // { probabilities, speed_kmh, worst_offender }
  return {
    prediction: row.prediction,
    probabilities: diag?.probabilities ?? null,
    speedKmh: diag?.speed_kmh ?? null,
    worstOffender: diag?.worst_offender ?? null,
  };
}

function computeStats(results) {
  const counts = { Normal: 0, "Side I": 0, "Side II": 0 };
  for (const r of results) counts[r.prediction] = (counts[r.prediction] ?? 0) + 1;
  const stats = [
    { label: "Files", value: results.length },
    { label: "Normal", value: counts.Normal, tone: "good" },
    { label: "Side I", value: counts["Side I"], tone: counts["Side I"] > 0 ? "critical" : "good" },
    { label: "Side II", value: counts["Side II"], tone: counts["Side II"] > 0 ? "critical" : "good" },
  ];
  const withTruth = results.filter((r) => r.trueLabel != null);
  if (withTruth.length) {
    const correct = withTruth.filter((r) => r.matchesTruth).length;
    stats.push({
      label: "Match Train_Labels.csv",
      value: `${correct}/${withTruth.length}`,
      tone: correct === withTruth.length ? "good" : "serious",
    });
  }
  return stats;
}

// Health = % of the currently loaded files classified Normal.
function computeHealth(results) {
  const normal = results.filter((r) => r.prediction === "Normal").length;
  return (normal / results.length) * 100;
}

// Per-file health is binary (Normal/faulty) — the model doesn't output a
// graded severity for this subsystem, only a 3-class label.
function computeEntities(results) {
  return results.map((r) => ({
    id: r.file_id,
    label: r.prediction,
    value: r.prediction === "Normal" ? 100 : 0,
    valueType: "health",
  }));
}

function renderCell(row) {
  const topProb = row.probabilities?.[row.prediction];
  return (
    <span className="inline-flex items-center gap-2">
      <LabelBadge label={row.prediction} />
      {topProb != null && <span className="text-xs text-ink-muted tabular-nums">{(topProb * 100).toFixed(0)}%</span>}
    </span>
  );
}

function renderCombined(results) {
  return (
    <RailProbabilityChart
      title="All files — class probability"
      subtitle="Every loaded file's real ensemble output: P(Normal) / P(Side I) / P(Side II), stacked to 100%, not a hard label."
      results={results}
      caveat="Each file is an isolated 1-second snapshot — bars are ordered by load order only, not a real timeline."
    />
  );
}

export default function RailPage({ onSummary }) {
  const [trueLabels, setTrueLabels] = useState(null);

  useEffect(() => {
    fetch(RAIL_LABELS_URL)
      .then((res) => (res.ok ? res.text() : null))
      .then((text) => {
        if (!text) return;
        Papa.parse(text, {
          header: true,
          skipEmptyLines: true,
          complete: (results) => {
            const map = {};
            for (const row of results.data) map[row.filename] = row.label;
            setTrueLabels(map);
          },
        });
      })
      .catch(() => {});
  }, []);

  return (
    <BatchSubsystemPage
      title="Rail corrugation predictions"
      description="Observes the vibration and shock readings from each rail recording, then classifies it as Normal, Side I, or Side II corrugation."
      csvFilename="rail_predictions.csv"
      accept=".csv"
      parseFile={parseRailFile}
      predictFile={predictFile}
      predictionHeader="Prediction"
      sampleFiles={RAIL_SAMPLES}
      trueLabels={trueLabels}
      computeStats={computeStats}
      computeHealth={computeHealth}
      computeEntities={computeEntities}
      renderCell={renderCell}
      renderCombined={renderCombined}
      onSummary={onSummary}
    />
  );
}
