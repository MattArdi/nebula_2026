import { useEffect, useRef, useState } from "react";
import { FileDrop, StatCard } from "../../components/ui.jsx";
import { CurrentDatasetHeader, UploadHeading } from "../../components/DatasetSections.jsx";
import UploadCard from "../../components/UploadCard.jsx";
import AcvTemperatureChart from "../../components/AcvTemperatureChart.jsx";
import { predictAcv } from "../../lib/apiClient.js";
import { fetchAsFile } from "../../lib/sampleFiles.js";
import { ACV_SAMPLE } from "../../lib/sampleManifest.js";
import { TRAIN_IDS } from "../../lib/trainIds.js";
import { useDatasetUploads } from "../../lib/useDatasetUploads.js";

async function computeRun(file) {
  const result = await predictAcv(file);
  const row = result.rows[0]; // { file_id, ranked_cars }
  return {
    fileName: file.name,
    rankedCars: row.ranked_cars.split("|"),
    indoorTemperature: result.diagnostics?.indoor_temperature ?? null,
    csvText: result.csv,
  };
}

// Stat tiles + indoor temperature chart for one file's result — used for both
// the current dataset and each uploaded file's "View Prediction" panel.
function AcvResultView({ run }) {
  const { rankedCars, indoorTemperature } = run;
  const faultyCarId = rankedCars[0];

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-3">
        <StatCard label="Cars Analysed" value={rankedCars.length} />
        <StatCard label="Most likely faulty car" value={`Car ${faultyCarId}`} tone="critical" />
      </div>

      <AcvTemperatureChart indoor={indoorTemperature} faultyCarId={faultyCarId} />
    </div>
  );
}

export default function AcvPage({ onSummary }) {
  // The dataset the stats, chart and Overview card are showing — starts as
  // the bundled sample, and is swapped by "Upload to current dataset".
  // Everything here is in-memory only, so a refresh resets it.
  const [currentRun, setCurrentRun] = useState(null);
  const [loadError, setLoadError] = useState(null);
  const { uploads, appliedKey, setAppliedKey, handleFiles } = useDatasetUploads(computeRun);
  const autoLoadedRef = useRef(false);

  useEffect(() => {
    if (autoLoadedRef.current) return;
    autoLoadedRef.current = true;
    (async () => {
      try {
        const file = await fetchAsFile(ACV_SAMPLE.url, ACV_SAMPLE.name);
        setCurrentRun(await computeRun(file));
      } catch (err) {
        setLoadError(`Could not load bundled sample data: ${err.message}`);
      }
    })();
  }, []);

  function applyUpload(upload) {
    setCurrentRun(upload.run);
    setAppliedKey(upload.key);
  }

  const rankedCars = currentRun?.rankedCars ?? null;

  useEffect(() => {
    if (!rankedCars || !onSummary) return;
    onSummary({
      fileCount: 1 + uploads.length,
      stats: [
        { label: "Cars analysed", value: rankedCars.length },
        { label: "Most likely faulty", value: `Car ${rankedCars[0]}`, tone: "critical" },
      ],
      // Exact acv_predictions.csv schema (file_id, ranked_cars) — one row
      // per case file, for the Overview's per-subsystem box.
      predictions: [{ file_id: currentRun.fileName, ranked_cars: rankedCars.join("|") }],
      predictionCount: 1,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rankedCars]);

  return (
    <div className="space-y-5">
      <CurrentDatasetHeader trainNumber={TRAIN_IDS.acv} />

      {loadError && (
        <div className="text-xs rounded-md px-3 py-2 border text-status-critical border-status-critical/40 bg-status-critical/10">
          {loadError}
        </div>
      )}
      {!currentRun && !loadError && <div className="text-xs text-ink-muted">Processing… large files can take a few seconds.</div>}

      {currentRun && <AcvResultView run={currentRun} />}

      <div className="space-y-3">
        <UploadHeading />
        <FileDrop onFiles={handleFiles} accept=".xlsx" multiple />
      </div>

      {uploads.length > 0 && (
        <div className="space-y-5">
          {uploads.map((u) => (
            <UploadCard key={u.key} upload={u} isCurrent={appliedKey === u.key} onApply={() => applyUpload(u)}>
              {(run) => <AcvResultView run={run} />}
            </UploadCard>
          ))}
        </div>
      )}
    </div>
  );
}
