import { useEffect, useState } from "react";
import Papa from "papaparse";
import BatchSubsystemPage from "../../components/BatchSubsystemPage.jsx";
import RailSeverityChart from "../../components/RailSeverityChart.jsx";
import { predictRail } from "../../lib/apiClient.js";
import { RAIL_SAMPLES, RAIL_LABELS_URL } from "../../lib/sampleManifest.js";

// The backend needs the raw File (feature extraction happens server-side,
// in the real validated pipeline) — no client-side parsing needed.
async function parseRailFile(file) {
  return file;
}

// The ensemble's own class-probability margin isn't exposed by the
// submission CSV (file_id, prediction only), so severity here is the
// predicted class encoded on the chart's existing +1/0/-1 scale, not a
// continuous confidence score.
const SEVERITY = { Normal: 0, "Side I": 1, "Side II": -1 };

async function predictFile(file) {
  const result = await predictRail([file]);
  const row = result.rows[0]; // { file_id, prediction }
  return { prediction: row.prediction, severityScore: SEVERITY[row.prediction] ?? 0 };
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

function renderCombined(results) {
  return (
    <RailSeverityChart
      title="All files — Side I / Side II severity"
      subtitle="Every loaded file as one bar: height is the predicted class (+1 Side I, 0 Normal, −1 Side II)."
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
      description={
        <>
          Drop one or many axle-box vibration/shock recordings (e.g. <code className="text-ink-secondary">Train1.csv</code>
          ...<code className="text-ink-secondary">Train272.csv</code>, or a .zip of several). Each file runs through
          the real validated ensemble (feature extraction + CatBoost/XGBoost/LogReg soft voting) and is classified
          as Normal, Side I, or Side II corrugation. Check against{" "}
          <code className="text-ink-secondary">Train_Labels.csv</code> by dropping Train files, or drop Test files
          for a real submission-ready run.
        </>
      }
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
      renderCombined={renderCombined}
      onSummary={onSummary}
    />
  );
}
