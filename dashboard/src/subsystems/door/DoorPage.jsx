import { useEffect, useMemo, useRef, useState } from "react";
import Papa from "papaparse";
import { FileDrop, PrimaryButton, Card, StatCard, LabelBadge } from "../../components/ui.jsx";
import SegmentTimeline from "../../components/SegmentTimeline.jsx";
import CycleSignalDetail from "../../components/CycleSignalDetail.jsx";
import { downloadCsv } from "../../lib/csvExport.js";
import { runDoorPipeline, averageNormalCycleSignal } from "./doorModel.js";
import { fetchAsFile } from "../../lib/sampleFiles.js";
import { DOOR_SAMPLE, DOOR_ANSWERS_URL } from "../../lib/sampleManifest.js";

function formatTimeOfDay(ms) {
  return new Date(ms).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

// Ground truth (Train_Segments_Answer.csv) only exists for Train.csv, the
// bundled default — matched to predicted segments by order.
async function fetchGroundTruth() {
  const res = await fetch(DOOR_ANSWERS_URL);
  if (!res.ok) return null;
  const text = await res.text();
  return new Promise((resolve) => {
    Papa.parse(text, {
      header: true,
      skipEmptyLines: true,
      complete: (results) => resolve(results.data),
      error: () => resolve(null),
    });
  });
}

export default function DoorPage({ onSummary }) {
  const [fileName, setFileName] = useState(null);
  const [status, setStatus] = useState(null); // { type, message }
  const [segments, setSegments] = useState(null);
  const [groundTruth, setGroundTruth] = useState(null); // array aligned by index, or null
  const [busy, setBusy] = useState(false);
  const [selectedCycle, setSelectedCycle] = useState(null);
  const autoLoadedRef = useRef(false);

  function runFile(file, { withGroundTruth = false } = {}) {
    setFileName(file.name);
    setStatus(null);
    setSegments(null);
    setGroundTruth(null);
    setSelectedCycle(null);
    setBusy(true);

    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      worker: true,
      complete: async (results) => {
        try {
          if (!results.data?.length) {
            throw new Error(results.errors?.[0]?.message ?? "No data rows found.");
          }
          const predicted = runDoorPipeline(results.data);
          setSegments(predicted);
          const abnormal = predicted.filter((s) => s.prediction === "Abnormal resistance").length;

          let truth = null;
          let accuracyMsg = "";
          if (withGroundTruth) {
            truth = await fetchGroundTruth();
            if (truth && truth.length === predicted.length) {
              setGroundTruth(truth);
              const correct = predicted.filter((s, i) => s.prediction === truth[i].status).length;
              accuracyMsg = ` — ${correct}/${predicted.length} match Train_Segments_Answer.csv`;
            }
          }

          setStatus({
            type: "ok",
            message: `Found ${predicted.length} cycles (${abnormal} abnormal-resistance, ${
              predicted.length - abnormal
            } normal) from ${results.data.length} rows${accuracyMsg}.`,
          });
        } catch (err) {
          setStatus({ type: "error", message: err.message });
        } finally {
          setBusy(false);
        }
      },
      error: (err) => {
        setStatus({ type: "error", message: err.message });
        setBusy(false);
      },
    });
  }

  useEffect(() => {
    if (autoLoadedRef.current) return;
    autoLoadedRef.current = true;
    (async () => {
      try {
        const file = await fetchAsFile(DOOR_SAMPLE.url, DOOR_SAMPLE.name);
        runFile(file, { withGroundTruth: true });
      } catch (err) {
        setStatus({ type: "error", message: `Could not load bundled sample data: ${err.message}` });
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function handleDownload() {
    downloadCsv(
      "door_predictions.csv",
      ["start_time", "end_time", "prediction", "confidence"],
      segments.map((s) => ({
        start_time: s.start_time,
        end_time: s.end_time,
        prediction: s.prediction,
        confidence: s.confidence.toFixed(3),
      }))
    );
  }

  const abnormalCount = segments ? segments.filter((s) => s.prediction === "Abnormal resistance").length : 0;
  const correctCount = groundTruth ? segments.filter((s, i) => s.prediction === groundTruth[i].status).length : null;

  // Reference "what a normal cycle typically looks like" curve, averaged
  // from every currently-loaded Normal cycle's real signal — the baseline
  // the signal-detail charts compare a selected cycle against.
  const normalAverage = useMemo(() => (segments ? averageNormalCycleSignal(segments) : []), [segments]);

  useEffect(() => {
    if (!segments || !onSummary) return;
    onSummary({
      fileCount: 1,
      stats: [
        { label: "Cycles found", value: segments.length },
        { label: "Normal", value: segments.length - abnormalCount, tone: "good" },
        { label: "Abnormal resistance", value: abnormalCount, tone: abnormalCount > 0 ? "critical" : "good" },
      ],
      // Health = % of cycles found NOT abnormal — real, from the loaded file.
      health: ((segments.length - abnormalCount) / segments.length) * 100,
      // Per-cycle breakdown for the Overview's entity view — health here is
      // always "probability this cycle is Normal", regardless of which
      // class was actually predicted, so it's on a consistent 0-100 scale
      // across entities (the underlying `confidence` field is instead
      // "probability of whichever class was predicted").
      entities: segments.map((s, i) => ({
        id: `Cycle ${i + 1} (${s.start_time})`,
        label: s.prediction,
        value: s.prediction === "Normal" ? s.confidence * 100 : (1 - s.confidence) * 100,
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
      accuracy: correctCount != null ? { correct: correctCount, total: segments.length } : null,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [segments, groundTruth]);

  return (
    <div className="space-y-5">
      <Card>
        <div className="text-sm font-medium text-ink-primary mb-1">What this does</div>
        <p className="text-xs text-ink-muted">
          Finds every door-open/close cycle in a continuous door-controller stream, then classifies each cycle as{" "}
          <span className="text-ink-secondary">Normal</span> or{" "}
          <span className="text-ink-secondary">Abnormal resistance</span> from its motor current and back-EMF
          profile. Opens pre-loaded with the labelled <code className="text-ink-secondary">Train.csv</code> stream,
          checked against <code className="text-ink-secondary">Train_Segments_Answer.csv</code>. Drop your own{" "}
          <code className="text-ink-secondary">.csv</code> stream (Train or the real Test.csv) to replace it.
        </p>
      </Card>

      {segments && (
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
          <StatCard label="Cycles found" value={segments.length} />
          <StatCard label="Normal" value={segments.length - abnormalCount} tone="good" />
          <StatCard label="Abnormal resistance" value={abnormalCount} tone={abnormalCount > 0 ? "critical" : "good"} />
        </div>
      )}

      {correctCount != null && (
        <div
          className={`text-xs rounded-md px-3 py-2 border ${
            correctCount === segments.length
              ? "text-status-good border-status-good/40 bg-status-good/10"
              : "text-status-serious border-status-serious/40 bg-status-serious/10"
          }`}
        >
          {correctCount}/{segments.length} cycles match Train_Segments_Answer.csv's true label
        </div>
      )}

      {segments && segments.length >= 4 && (
        <>
          <SegmentTimeline
            title="Cycle timeline"
            subtitle="Each bar is one door-open/close cycle at its real start/end time, split into an Open lane and a Close lane. Click a bar for its raw signal."
            historical={segments}
            xFormat={formatTimeOfDay}
            selected={selectedCycle}
            onSelect={(seg) => setSelectedCycle(seg)}
            caveat="Open/Close comes directly from the stream's own opening/closing flags (informational, not scored)."
          />

          {selectedCycle && <CycleSignalDetail segment={selectedCycle} normalAverage={normalAverage} />}
        </>
      )}

      <FileDrop onFiles={(files) => runFile(files[0])} accept=".csv" hint="A single continuous-stream CSV, e.g. Train.csv or Test.csv" />

      {fileName && <div className="text-xs text-ink-muted">{fileName}</div>}
      {busy && <div className="text-xs text-ink-muted">Processing…</div>}

      {status && (
        <div
          className={`text-xs rounded-md px-3 py-2 border ${
            status.type === "ok"
              ? "text-status-good border-status-good/40 bg-status-good/10"
              : "text-status-critical border-status-critical/40 bg-status-critical/10"
          }`}
        >
          {status.message}
        </div>
      )}

      {segments && segments.length > 0 && (
        <Card className="overflow-hidden !px-0 !py-0">
          <div className="flex items-center justify-between px-4 py-3 border-b border-line-hairline">
            <div className="text-sm font-medium text-ink-primary">Predicted cycles ({segments.length})</div>
            <PrimaryButton onClick={handleDownload}>Download door_predictions.csv</PrimaryButton>
          </div>
          <div className="overflow-x-auto max-h-[480px] overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-surface-card">
                <tr className="text-left text-xs text-ink-muted border-b border-line-hairline">
                  <th className="px-4 py-2 font-normal">#</th>
                  <th className="px-4 py-2 font-normal">Start time</th>
                  <th className="px-4 py-2 font-normal">End time</th>
                  <th className="px-4 py-2 font-normal">Prediction</th>
                  <th className="px-4 py-2 font-normal">Confidence</th>
                  {groundTruth && <th className="px-4 py-2 font-normal">True label</th>}
                </tr>
              </thead>
              <tbody>
                {segments.map((s, i) => {
                  const truth = groundTruth?.[i];
                  const match = truth ? s.prediction === truth.status : null;
                  return (
                    <tr key={i} className="border-b border-line-hairline last:border-0">
                      <td className="px-4 py-2 text-ink-muted tabular-nums">{i + 1}</td>
                      <td className="px-4 py-2 text-ink-secondary tabular-nums">{s.start_time}</td>
                      <td className="px-4 py-2 text-ink-secondary tabular-nums">{s.end_time}</td>
                      <td className="px-4 py-2">
                        <LabelBadge label={s.prediction} />
                      </td>
                      <td className="px-4 py-2 text-ink-muted tabular-nums">{(s.confidence * 100).toFixed(0)}%</td>
                      {groundTruth && (
                        <td className="px-4 py-2">
                          <span className={match ? "text-status-good" : "text-status-critical"}>
                            {truth.status}
                          </span>
                        </td>
                      )}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}
