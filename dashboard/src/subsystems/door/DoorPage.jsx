import { useEffect, useMemo, useRef, useState } from "react";
import Papa from "papaparse";
import { FileDrop, StatCard } from "../../components/ui.jsx";
import { CurrentDatasetHeader, UploadHeading } from "../../components/DatasetSections.jsx";
import UploadCard from "../../components/UploadCard.jsx";
import SegmentTimeline from "../../components/SegmentTimeline.jsx";
import CycleSignalDetail from "../../components/CycleSignalDetail.jsx";
import { predictDoor } from "../../lib/apiClient.js";
import { groupDoorCycles, getOperation, parseDoorTimestamp, averageNormalCycleSignal } from "../../lib/signalResample.js";
import { fetchAsFile } from "../../lib/sampleFiles.js";
import { DOOR_SAMPLE } from "../../lib/sampleManifest.js";
import { TRAIN_IDS } from "../../lib/trainIds.js";
import { useDatasetUploads } from "../../lib/useDatasetUploads.js";

// The stream's timestamps are parsed as UTC (see parseDoorTimestamp), so
// format them as UTC too — otherwise the browser's own timezone shifts them.
function formatTimeOfDay(ms) {
  return new Date(ms).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit", timeZone: "UTC" });
}

// Runs the backend pipeline + client-side charting prep for one file and
// resolves its segments — doesn't touch React state itself, so the caller
// can await several of these in a row (one per dropped file) without them
// racing each other's setState calls.
function computeRun(file) {
  return new Promise((resolve, reject) => {
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      worker: true,
      complete: async (results) => {
        try {
          if (!results.data?.length) {
            throw new Error(results.errors?.[0]?.message ?? "No data rows found.");
          }

          // The real prediction comes from the backend (the validated
          // Python pipeline); the client also chunks the same raw rows
          // into cycles (mirroring the backend's own gap rule) purely to
          // have a raw signal to chart and to read the Open/Close flag off.
          const backendResult = await predictDoor(file);
          const predicted = backendResult.rows; // [{ start_time, end_time, prediction }]
          const diagnostics = backendResult.diagnostics ?? []; // per-segment, same order as `rows`
          const cycles = groupDoorCycles(results.data);

          const alignedByCount = cycles.length === predicted.length;
          const segments = predicted.map((p, i) => {
            const chunk = alignedByCount ? cycles[i] : null;
            const diag = diagnostics[i] ?? null;
            return {
              ...p,
              start_ts: parseDoorTimestamp(p.start_time),
              end_ts: parseDoorTimestamp(p.end_time),
              // The backend's own operation flag is authoritative — prefer
              // it over the client-side chunk guess, which only exists for
              // charting and can miss on an edge case even when the count
              // lines up.
              operation: diag?.operation ?? (chunk ? getOperation(chunk) : null),
              rawSeries: chunk,
            };
          });

          resolve({ fileName: file.name, segments, csvText: backendResult.csv });
        } catch (err) {
          reject(err);
        }
      },
      error: reject,
    });
  });
}

// Stat tiles + cycle timeline + click-a-bar signal detail for one set of
// segments — used for both the current dataset and each uploaded file's
// "View Prediction" panel.
function DoorResultView({ segments }) {
  const [selectedCycle, setSelectedCycle] = useState(null);

  // A different dataset invalidates whichever cycle was selected — it
  // belongs to the previous segment array.
  useEffect(() => {
    setSelectedCycle(null);
  }, [segments]);

  const abnormalCount = segments.filter((s) => s.prediction === "Abnormal resistance").length;

  // Reference "what a normal cycle typically looks like" curve, averaged
  // from every Normal cycle in this dataset — the baseline the signal-detail
  // charts compare a selected cycle against.
  const normalAverage = useMemo(() => averageNormalCycleSignal(segments), [segments]);

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-3 gap-3">
        <StatCard label="Cycles found" value={segments.length} />
        <StatCard label="Normal Resistance" value={segments.length - abnormalCount} tone="good" />
        <StatCard label="Abnormal Resistance" value={abnormalCount} tone={abnormalCount > 0 ? "critical" : "good"} />
      </div>

      {segments.length >= 4 && (
        <>
          <SegmentTimeline
            title="Cycle timeline"
            subtitle="Hover or click each bar for more details."
            historical={segments}
            xFormat={formatTimeOfDay}
            selected={selectedCycle}
            onSelect={setSelectedCycle}
          />

          {selectedCycle && <CycleSignalDetail segment={selectedCycle} normalAverage={normalAverage} />}
        </>
      )}
    </div>
  );
}

export default function DoorPage({ onSummary }) {
  // The dataset the timeline and the Overview card are showing — starts as
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
        const file = await fetchAsFile(DOOR_SAMPLE.url, DOOR_SAMPLE.name);
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

  const segments = currentRun?.segments ?? null;

  useEffect(() => {
    if (!segments || !onSummary) return;
    const abnormalCount = segments.filter((s) => s.prediction === "Abnormal resistance").length;
    onSummary({
      fileCount: 1 + uploads.length,
      stats: [
        { label: "Cycles found", value: segments.length },
        { label: "Normal", value: segments.length - abnormalCount, tone: "good" },
        { label: "Abnormal resistance", value: abnormalCount, tone: abnormalCount > 0 ? "critical" : "good" },
      ],
      // Health = % of cycles found NOT abnormal — real, from the loaded file.
      health: ((segments.length - abnormalCount) / segments.length) * 100,
      // The rule-based classifier has no calibrated confidence, so
      // per-cycle health is binary (100 if Normal, 0 if Abnormal), same
      // convention as the other subsystems that lack a graded score.
      entities: segments.map((s, i) => ({
        id: `Cycle ${i + 1} (${s.start_time})`,
        label: s.prediction,
        value: s.prediction === "Normal" ? 100 : 0,
        valueType: "health",
      })),
      // Preview rows in the exact door_predictions.csv schema (start_time,
      // end_time, prediction) for the Overview's per-subsystem box.
      predictions: segments.slice(0, 4).map((s) => ({
        start_time: s.start_time,
        end_time: s.end_time,
        prediction: s.prediction,
      })),
      predictionCount: segments.length,
      // Abnormal cycles split by door operation, for the Overview's ranking row.
      abnormalByOperation: {
        Open: segments.filter((s) => s.prediction === "Abnormal resistance" && s.operation === "Open").length,
        Close: segments.filter((s) => s.prediction === "Abnormal resistance" && s.operation === "Close").length,
      },
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [segments]);

  return (
    <div className="space-y-5">
      <CurrentDatasetHeader trainNumber={TRAIN_IDS.door} />

      <div className="text-sm font-semibold text-ink-primary">Cycle-by-cycle Health Timeline</div>

      {loadError && (
        <div className="text-xs rounded-md px-3 py-2 border text-status-critical border-status-critical/40 bg-status-critical/10">
          {loadError}
        </div>
      )}
      {!segments && !loadError && <div className="text-xs text-ink-muted">Processing…</div>}

      {segments && <DoorResultView segments={segments} />}

      <div className="space-y-3">
        <UploadHeading />
        <FileDrop onFiles={handleFiles} accept=".csv" multiple />
      </div>

      {uploads.length > 0 && (
        <div className="space-y-5">
          {uploads.map((u) => (
            <UploadCard key={u.key} upload={u} isCurrent={appliedKey === u.key} onApply={() => applyUpload(u)}>
              {(run) => <DoorResultView segments={run.segments} />}
            </UploadCard>
          ))}
        </div>
      )}
    </div>
  );
}
