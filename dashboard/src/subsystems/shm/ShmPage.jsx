import { useEffect, useRef, useState } from "react";
import { FileDrop, StatCard } from "../../components/ui.jsx";
import { CurrentDatasetHeader, UploadHeading } from "../../components/DatasetSections.jsx";
import UploadCard from "../../components/UploadCard.jsx";
import DamageProgressChart from "../../components/DamageProgressChart.jsx";
import { predictShm } from "../../lib/apiClient.js";
import { fetchAsFile } from "../../lib/sampleFiles.js";
import { SHM_SAMPLES } from "../../lib/sampleManifest.js";
import { TRAIN_IDS } from "../../lib/trainIds.js";
import { useDatasetUploads } from "../../lib/useDatasetUploads.js";

// The backend runs rainflow counting + Miner's-rule damage directly on the
// raw file — no client-side parsing needed.
async function computeRun(file) {
  const result = await predictShm([file]);
  const row = result.rows[0]; // { file_id, prediction }
  const diag = result.diagnostics?.[0]; // { remaining_life_pct, damage_progress, ... }
  const damage = Number(row.prediction);
  return {
    fileName: file.name,
    damage,
    // Damage is on a 0-1 scale where 1.0 means fatigue failure, so the life
    // left is what remains of it.
    remainingLifePct: diag?.remaining_life_pct ?? (1 - damage) * 100,
    damageProgress: diag?.damage_progress ?? null,
    csvText: result.csv,
  };
}

// Stat tiles + cumulative damage chart for one file's result — used for both
// the current dataset and each uploaded file's "View Prediction" panel.
function ShmResultView({ run }) {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-3">
        <StatCard label="Current Train Damage (%)" value={`${(run.damage * 100).toFixed(2)}%`} />
        <StatCard label="Fatigue Life Remaining (%)" value={`${run.remainingLifePct.toFixed(2)}%`} tone="good" />
      </div>

      <DamageProgressChart
        title="Cumulative Damage Build-up"
        subtitle="Damage overtime based on recording"
        data={run.damageProgress}
      />
    </div>
  );
}

export default function ShmPage({ onSummary }) {
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
        const file = await fetchAsFile(SHM_SAMPLES[0].url, SHM_SAMPLES[0].name);
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

  useEffect(() => {
    if (!currentRun || !onSummary) return;
    onSummary({
      fileCount: 1 + uploads.length,
      damage: currentRun.damage,
      stats: [
        { label: "Current Train Damage (%)", value: `${(currentRun.damage * 100).toFixed(2)}%` },
        { label: "Fatigue Life Remaining (%)", value: `${currentRun.remainingLifePct.toFixed(2)}%`, tone: "good" },
      ],
      health: currentRun.remainingLifePct,
      // Exact shm_predictions.csv schema (file_id, prediction), for the
      // Overview's per-subsystem box.
      predictions: [{ file_id: currentRun.fileName, prediction: currentRun.damage }],
      predictionCount: 1,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentRun]);

  return (
    <div className="space-y-5">
      <CurrentDatasetHeader trainNumber={TRAIN_IDS.shm} />

      {loadError && (
        <div className="text-xs rounded-md px-3 py-2 border text-status-critical border-status-critical/40 bg-status-critical/10">
          {loadError}
        </div>
      )}
      {!currentRun && !loadError && <div className="text-xs text-ink-muted">Processing…</div>}

      {currentRun && <ShmResultView run={currentRun} />}

      <div className="space-y-3">
        <UploadHeading />
        <FileDrop onFiles={handleFiles} accept=".csv" multiple />
      </div>

      {uploads.length > 0 && (
        <div className="space-y-5">
          {uploads.map((u) => (
            <UploadCard key={u.key} upload={u} isCurrent={appliedKey === u.key} onApply={() => applyUpload(u)}>
              {(run) => <ShmResultView run={run} />}
            </UploadCard>
          ))}
        </div>
      )}
    </div>
  );
}
