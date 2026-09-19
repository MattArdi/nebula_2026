import { useEffect, useState } from "react";
import Papa from "papaparse";
import BatchSubsystemPage from "../../components/BatchSubsystemPage.jsx";
import RailSeverityChart from "../../components/RailSeverityChart.jsx";
import { predictRailFile } from "./railModel.js";
import { RAIL_SAMPLES, RAIL_LABELS_URL } from "../../lib/sampleManifest.js";

function parseRailFile(file) {
  return new Promise((resolve, reject) => {
    Papa.parse(file, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      worker: true,
      complete: (results) => {
        // PapaParse's "errors" include harmless notices — only an empty
        // result actually means failure.
        if (!results.data?.length) {
          const msg = results.errors?.[0]?.message ?? "No data rows found.";
          return reject(new Error(msg));
        }
        resolve(results.data.map((r) => Object.values(r)));
      },
      error: reject,
    });
  });
}

async function predictFile(rows) {
  const { prediction, severityScore } = predictRailFile(rows);
  return { prediction, severityScore };
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
      subtitle="Every loaded file as one bar: height is the classifier's own P(Side I) − P(Side II) margin — near 0 for Normal, up for Side I, down for Side II."
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
          ...<code className="text-ink-secondary">Train272.csv</code>, or a .zip of several). Each file's vibration
          features are extracted separately for the Side I and Side II rails, then classified as Normal, Side I, or
          Side II corrugation. Opens pre-loaded with 12 labelled Train files, checked against{" "}
          <code className="text-ink-secondary">Train_Labels.csv</code>. Drop your own files (Train or Test) to
          replace them.
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
