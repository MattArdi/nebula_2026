import { useEffect, useMemo, useRef, useState } from "react";
import Papa from "papaparse";
import { FileDrop, PrimaryButton, Card, StatCard, LabelBadge } from "../../components/ui.jsx";
import SegmentTimeline from "../../components/SegmentTimeline.jsx";
import CycleSignalDetail from "../../components/CycleSignalDetail.jsx";
import DoorThresholdChart from "../../components/DoorThresholdChart.jsx";
import { downloadCsvText } from "../../lib/csvExport.js";
import { predictDoor } from "../../lib/apiClient.js";
import { groupDoorCycles, getOperation, parseDoorTimestamp, averageNormalCycleSignal } from "../../lib/signalResample.js";
import { fetchAsFile } from "../../lib/sampleFiles.js";
import { DOOR_SAMPLE, DOOR_ANSWERS_URL } from "../../lib/sampleManifest.js";

function formatTimeOfDay(ms) {
  return new Date(ms).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

// Ground truth (Train_Segments_Answer.csv) only exists for Train.csv, so
// this quietly resolves to null for any other file — the caller only uses
// it once the row count also lines up with what was predicted.
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
  const [csvText, setCsvText] = useState(null);
  const [busy, setBusy] = useState(false);
  const [selectedCycle, setSelectedCycle] = useState(null);
  const autoLoadedRef = useRef(false);

  function runFile(file) {
    setFileName(file.name);
    setStatus(null);
    setSegments(null);
    setGroundTruth(null);
    setCsvText(null);
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

          // The real prediction comes from the backend (the validated
          // Python pipeline); the client also chunks the same raw rows
          // into cycles (mirroring the backend's own gap rule) purely to
          // have a raw signal to chart and to read the Open/Close flag off.
          const [backendResult, truth] = await Promise.all([predictDoor(file), fetchGroundTruth()]);
          const predicted = backendResult.rows; // [{ start_time, end_time, prediction }]
          const diagnostics = backendResult.diagnostics ?? []; // per-segment, same order as `rows`
          const cycles = groupDoorCycles(results.data);

          const alignedByCount = cycles.length === predicted.length;
          const segs = predicted.map((p, i) => {
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
              decisionFeature: diag?.decision_feature ?? null,
              decisionValue: diag?.decision_value ?? null,
              thresholdUsed: diag?.threshold_used ?? null,
              baselineUsed: diag?.baseline_used ?? null,
              lowConfidence: diag?.low_confidence ?? false,
              outOfRange: diag?.out_of_range ?? false,
            };
          });

          setSegments(segs);
          setCsvText(backendResult.csv);
          const abnormal = segs.filter((s) => s.prediction === "Abnormal resistance").length;

          let accuracyMsg = "";
          if (truth && truth.length === segs.length) {
            setGroundTruth(truth);
            const correct = segs.filter((s, i) => s.prediction === truth[i].status).length;
            accuracyMsg = ` — ${correct}/${segs.length} match Train_Segments_Answer.csv`;
          }

          setStatus({
            type: "ok",
            message: `Found ${segs.length} cycles (${abnormal} abnormal-resistance, ${
              segs.length - abnormal
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
        runFile(file);
      } catch (err) {
        setStatus({ type: "error", message: `Could not load bundled sample data: ${err.message}` });
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function handleDownload() {
    downloadCsvText("door_predictions.csv", csvText);
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
          profile, using the real validated Door pipeline. Opens pre-loaded with the labelled{" "}
          <code className="text-ink-secondary">Train.csv</code> stream, checked against{" "}
          <code className="text-ink-secondary">Train_Segments_Answer.csv</code>. Drop your own{" "}
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

          <DoorThresholdChart
            title="Decision value vs. threshold, in cycle order"
            subtitle="Every cycle's real current reading against the threshold it was judged against — a value creeping toward the line across successive cycles is visible here before it actually crosses."
            segments={segments}
            caveat="From the deployed v3_adaptive pipeline's own diagnostics — the threshold and confidence checks it already computes, not re-derived client-side."
          />
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
                      <td className="px-4 py-2">
                        <div className="flex items-center gap-1.5">
                          {s.lowConfidence && (
                            <span className="text-[11px] text-status-warning" title="Within the bootstrap confidence half-width of the threshold — a close call.">
                              Low confidence
                            </span>
                          )}
                          {s.outOfRange && (
                            <span className="text-[11px] text-status-critical" title="Outside anything Train ever demonstrated for this operation.">
                              Out of range
                            </span>
                          )}
                          {!s.lowConfidence && !s.outOfRange && <span className="text-[11px] text-ink-muted">—</span>}
                        </div>
                      </td>
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
