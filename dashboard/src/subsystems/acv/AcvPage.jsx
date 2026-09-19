import { useEffect, useMemo, useRef, useState } from "react";
import Papa from "papaparse";
import { FileDrop, PrimaryButton, Card, StatCard, FileRunPicker } from "../../components/ui.jsx";
import AcvCusumChart from "../../components/AcvCusumChart.jsx";
import { downloadCsvText } from "../../lib/csvExport.js";
import { predictAcv } from "../../lib/apiClient.js";
import { fetchAsFile } from "../../lib/sampleFiles.js";
import { ACV_SAMPLE, ACV_LABELS_URL } from "../../lib/sampleManifest.js";
import { useFileRuns } from "../../lib/useFileRuns.js";

// Ground truth from Train_Labels.csv — only exists for the 6 labelled
// Train case files. acv_test_case.xlsx's answer is deliberately
// unpublished (used by the organisers to grade submissions).
async function fetchTrainLabels() {
  const res = await fetch(ACV_LABELS_URL);
  if (!res.ok) return {};
  const text = await res.text();
  return new Promise((resolve) => {
    Papa.parse(text, {
      header: true,
      skipEmptyLines: true,
      complete: (results) => {
        const map = {};
        for (const row of results.data) map[row.filename] = row.faulty_car;
        resolve(map);
      },
      error: () => resolve({}),
    });
  });
}

async function computeRun(file) {
  const result = await predictAcv(file);
  const row = result.rows[0]; // { file_id, ranked_cars }
  const rankedCars = row.ranked_cars.split("|");
  return {
    id: file.name,
    fileName: file.name,
    rankedCars,
    diagnostics: result.diagnostics ?? null,
    csvText: result.csv,
    statusMessage: `Ranked ${rankedCars.length} cars from the real ACV pipeline. Most likely faulty: Car ${rankedCars[0]}.`,
  };
}

export default function AcvPage({ onSummary }) {
  const { runs, selectedId, setSelectedId, addRun, removeRun } = useFileRuns();
  const [status, setStatus] = useState(null);
  const [trainLabels, setTrainLabels] = useState({});
  const [busy, setBusy] = useState(false);
  const [visibleCarIds, setVisibleCarIds] = useState(null); // Set, null until first loaded (defaults to "all")
  const autoLoadedRef = useRef(false);

  useEffect(() => {
    fetchTrainLabels().then(setTrainLabels);
  }, []);

  async function runFiles(files) {
    setStatus(null);
    setBusy(true);
    const errors = [];
    let lastOk = null;
    for (const file of files) {
      try {
        const run = await computeRun(file);
        addRun(run);
        lastOk = run;
      } catch (err) {
        errors.push(`${file.name}: ${err.message}`);
      }
    }
    setBusy(false);
    setStatus(
      errors.length
        ? { type: "error", message: errors.join("; ") }
        : lastOk
          ? { type: "ok", message: lastOk.statusMessage }
          : null
    );
  }

  useEffect(() => {
    if (autoLoadedRef.current) return;
    autoLoadedRef.current = true;
    (async () => {
      try {
        const file = await fetchAsFile(ACV_SAMPLE.url, ACV_SAMPLE.name);
        await runFiles([file]);
      } catch (err) {
        setStatus({ type: "error", message: `Could not load bundled sample data: ${err.message}` });
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // A file switch invalidates the chart's ticked-car selection — a
  // different case file can have a different faulty car / trajectory set.
  useEffect(() => {
    setVisibleCarIds(null);
  }, [selectedId]);

  const selected = runs.find((r) => r.id === selectedId) ?? null;
  const fileName = selected?.fileName ?? null;
  const rankedCars = selected?.rankedCars ?? null;
  const diagnostics = selected?.diagnostics ?? null;

  function handleDownload() {
    downloadCsvText("acv_predictions.csv", selected.csvText);
  }

  // Ground truth from Train_Labels.csv — null for any file it doesn't cover.
  const groundTruthCarId = trainLabels[fileName] ?? null;
  const modelMatchesGroundTruth = groundTruthCarId != null && rankedCars?.[0] === groundTruthCarId;

  const scoreById = useMemo(() => {
    const map = {};
    for (const c of diagnostics?.cars ?? []) map[c.id] = c;
    return map;
  }, [diagnostics]);
  const maxScore = Math.max(1e-9, ...Object.values(scoreById).map((c) => c.primary_score ?? 0));

  const trajectoryCarIds = Object.keys(diagnostics?.trajectories ?? {});
  const effectiveVisibleCarIds = visibleCarIds ?? new Set(trajectoryCarIds);
  function toggleCarVisible(id) {
    const next = new Set(effectiveVisibleCarIds);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    setVisibleCarIds(next);
  }

  useEffect(() => {
    if (!rankedCars || !onSummary) return;
    onSummary({
      fileCount: runs.length,
      groundTruthCarId,
      stats: [
        { label: "Cars analysed", value: rankedCars.length },
        { label: "Most likely faulty", value: `Car ${rankedCars[0]}`, tone: "critical" },
      ],
      // Exact acv_predictions.csv schema (file_id, ranked_cars) — one row
      // per uploaded case file, for the Overview's per-subsystem box.
      predictions: [{ file_id: fileName, ranked_cars: rankedCars.join("|") }],
      predictionCount: 1,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rankedCars]);

  return (
    <div className="space-y-5">
      <Card>
        <div className="text-sm font-medium text-ink-primary mb-1">What this does</div>
        <p className="text-xs text-ink-muted">
          Ranks every car in a file by how much its readings deviate from its peers at the same moments, using the
          real validated ACV pipeline. Opens pre-loaded with <code className="text-ink-secondary">acv_case_01.xlsx</code>,
          a labelled Train case checked against <code className="text-ink-secondary">Train_Labels.csv</code>. Drop any
          other <code className="text-ink-secondary">.xlsx</code> case file(s) (Train or Test) to add more — switch
          between them with the file picker below.
        </p>
      </Card>

      <FileRunPicker runs={runs} selectedId={selectedId} onSelect={setSelectedId} onRemove={removeRun} />

      {rankedCars && (
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
          <StatCard label="Cars analysed" value={rankedCars.length} />
          <StatCard
            label="Model's most likely faulty"
            value={`Car ${rankedCars[0]}`}
            tone={groundTruthCarId ? (modelMatchesGroundTruth ? "good" : "critical") : "critical"}
          />
          {groundTruthCarId ? (
            <StatCard label="Confirmed faulty (Train_Labels.csv)" value={`Car ${groundTruthCarId}`} tone="good" />
          ) : diagnostics ? (
            <StatCard
              label="Margin vs. runner-up"
              value={diagnostics.margin != null ? diagnostics.margin.toFixed(2) : "—"}
              tone={diagnostics.margin_flagged ? "serious" : "good"}
              hint={diagnostics.margin_flagged ? "Thinner than usual — a closer call" : undefined}
            />
          ) : null}
        </div>
      )}

      {groundTruthCarId && rankedCars && (
        <div
          className={`text-xs rounded-md px-3 py-2 border ${
            modelMatchesGroundTruth
              ? "text-status-good border-status-good/40 bg-status-good/10"
              : "text-status-critical border-status-critical/40 bg-status-critical/10"
          }`}
        >
          {modelMatchesGroundTruth
            ? `Model's top pick (Car ${rankedCars[0]}) matches the confirmed faulty car from Train_Labels.csv.`
            : `Model's top pick (Car ${rankedCars[0]}) does NOT match the confirmed faulty car (Car ${groundTruthCarId}) from Train_Labels.csv.`}
        </div>
      )}

      {diagnostics && !groundTruthCarId && (
        <div
          className={`text-xs rounded-md px-3 py-2 border ${
            diagnostics.margin_flagged
              ? "text-status-serious border-status-serious/40 bg-status-serious/10"
              : "text-status-good border-status-good/40 bg-status-good/10"
          }`}
        >
          {diagnostics.margin_flagged ? "Thin margin — " : ""}
          {diagnostics.margin_reason}
        </div>
      )}

      {diagnostics?.trajectories && (
        <AcvCusumChart
          title="Per-car CUSUM score over the trip"
          subtitle="Every data-bearing car's running deviation score across the file's real time axis — shows when a car's score started climbing, not just which one ended up on top."
          trajectories={diagnostics.trajectories}
          topCarId={rankedCars?.[0]}
          visibleCarIds={effectiveVisibleCarIds}
          onToggleCar={toggleCarVisible}
          caveat="A rising, non-resetting climb is the ranking's real signal — ordinary jitter keeps decaying back toward 0."
        />
      )}

      <FileDrop
        onFiles={runFiles}
        accept=".xlsx"
        multiple
        hint="One or many .xlsx case files, e.g. acv_case_01.xlsx — each adds a file to compare"
      />

      {fileName && <div className="text-xs text-ink-muted">{fileName}</div>}
      {busy && <div className="text-xs text-ink-muted">Processing… large files can take a few seconds.</div>}

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

      {rankedCars && (
        <Card className="!px-0 !py-0 overflow-hidden">
          <div className="flex items-center justify-between px-4 py-3 border-b border-line-hairline">
            <div className="text-sm font-medium text-ink-primary">Ranked cars, most likely faulty first</div>
            <PrimaryButton onClick={handleDownload}>Download acv_predictions.csv</PrimaryButton>
          </div>
          <div className="px-4 py-3 space-y-2">
            {rankedCars.map((id, i) => {
              const c = scoreById[id];
              return (
                <div key={id} className="flex items-center gap-3">
                  <span className="w-6 text-xs text-ink-muted tabular-nums">{i + 1}</span>
                  <span className="w-16 text-sm font-medium text-ink-primary">Car {id}</span>
                  {c?.primary_score != null && (
                    <>
                      <div className="flex-1 h-2 rounded-full bg-surface-raised overflow-hidden">
                        <div
                          className={`h-full ${i === 0 ? "bg-status-critical" : "bg-series-blue"}`}
                          style={{ width: `${Math.max(4, (c.primary_score / maxScore) * 100)}%` }}
                        />
                      </div>
                      <span className="w-16 text-xs text-ink-muted tabular-nums text-right">{c.primary_score.toFixed(2)}</span>
                    </>
                  )}
                  {id === groundTruthCarId && (
                    <span className="text-[11px] text-status-good font-medium shrink-0">confirmed faulty</span>
                  )}
                </div>
              );
            })}
          </div>
        </Card>
      )}
    </div>
  );
}
