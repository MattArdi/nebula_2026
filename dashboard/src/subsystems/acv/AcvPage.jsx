import { useEffect, useMemo, useRef, useState } from "react";
import * as XLSX from "xlsx";
import { FileDrop, PrimaryButton, Card, StatCard } from "../../components/ui.jsx";
import IndoorOutdoorChart from "../../components/IndoorOutdoorChart.jsx";
import PeerComparisonChart from "../../components/PeerComparisonChart.jsx";
import FleetComparisonChart from "../../components/FleetComparisonChart.jsx";
import { downloadCsv } from "../../lib/csvExport.js";
import { rankCars, indoorOutdoorTrend, controlTemperatureTrend, fleetIndoorTrend } from "./acvModel.js";
import { ACV_TRAIN_LABELS } from "./acvTrainLabels.js";
import { fetchAsFile } from "../../lib/sampleFiles.js";
import { ACV_SAMPLE } from "../../lib/sampleManifest.js";

export default function AcvPage({ onSummary }) {
  const [fileName, setFileName] = useState(null);
  const [status, setStatus] = useState(null);
  const [ranking, setRanking] = useState(null);
  const [rawRows, setRawRows] = useState(null);
  const [busy, setBusy] = useState(false);
  const [selectedCarId, setSelectedCarId] = useState(null);
  const [visibleCarIds, setVisibleCarIds] = useState(null); // Set, null until first loaded (defaults to "all")
  const autoLoadedRef = useRef(false);

  function runFile(file) {
    setFileName(file.name);
    setStatus(null);
    setRanking(null);
    setRawRows(null);
    setSelectedCarId(null); // re-default to the new file's top-ranked car
    setVisibleCarIds(null); // re-default to "all cars visible" for the new file
    setBusy(true);

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const data = new Uint8Array(e.target.result);
        // cellDates: true is required — without it SheetJS returns the
        // "Time" column as raw Excel serial numbers, not real dates, which
        // would silently break the trend chart's x-axis.
        const workbook = XLSX.read(data, { type: "array", cellDates: true });
        const sheet = workbook.Sheets[workbook.SheetNames[0]];
        const rows = XLSX.utils.sheet_to_json(sheet, { defval: null });
        if (!rows.length) throw new Error("No rows found in the first sheet.");

        const scored = rankCars(rows);
        if (!scored.length) {
          throw new Error('No "Car NN - <parameter>" columns found — check this is an ACV telemetry file.');
        }
        setRanking(scored);
        setRawRows(rows);
        setStatus({
          type: "ok",
          message: `Ranked ${scored.length} cars from ${rows.length} timestamped readings. Most likely faulty: Car ${scored[0].id}.`,
        });
      } catch (err) {
        setStatus({ type: "error", message: err.message });
      } finally {
        setBusy(false);
      }
    };
    reader.onerror = () => {
      setStatus({ type: "error", message: "Could not read the file." });
      setBusy(false);
    };
    reader.readAsArrayBuffer(file);
  }

  useEffect(() => {
    if (autoLoadedRef.current) return;
    autoLoadedRef.current = true;
    (async () => {
      try {
        const file = await fetchAsFile(ACV_SAMPLE.url, ACV_SAMPLE.name);
        runFile(file);
      } catch (err) {
        setStatus({ type: "error", message: `Could not load bundled sample data: ${err.message}` });
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function handleDownload() {
    downloadCsv("acv_predictions.csv", ["file_id", "ranked_cars"], [
      { file_id: fileName, ranked_cars: ranking.map((c) => c.id).join("|") },
    ]);
  }

  const maxScore = ranking ? Math.max(...ranking.map((c) => c.score), 1e-9) : 1;
  const gap = ranking && ranking.length > 1 ? ranking[0].score - ranking[1].score : null;

  // Ground truth from Train_Labels.csv — only exists for the 6 labelled
  // Train case files. acv_test_case.xlsx's answer is deliberately
  // unpublished (used by the organisers to grade submissions), so this is
  // null for it, same as for any file the labels don't cover.
  const groundTruthCarId = ACV_TRAIN_LABELS[fileName] ?? null;
  const modelMatchesGroundTruth = groundTruthCarId != null && ranking?.[0]?.id === groundTruthCarId;

  // Default the comparison charts to the CONFIRMED faulty car (ground
  // truth) when it's known, not the model's own guess — falls back to the
  // model's top pick only for files with no published answer.
  const activeCarId = selectedCarId ?? groundTruthCarId ?? ranking?.[0]?.id ?? null;

  const indoorOutdoor = useMemo(() => {
    if (!activeCarId || !rawRows) return [];
    return indoorOutdoorTrend(rawRows, activeCarId, { buckets: 60 });
  }, [activeCarId, rawRows]);

  const controlTempVsPeers = useMemo(() => {
    if (!activeCarId || !rawRows) return [];
    return controlTemperatureTrend(rawRows, activeCarId, { buckets: 60 });
  }, [activeCarId, rawRows]);

  const fleetData = useMemo(() => {
    if (!rawRows) return { carIds: [], points: [] };
    return fleetIndoorTrend(rawRows, { buckets: 60 });
  }, [rawRows]);

  // null means "not touched yet" -> default to every car ticked.
  const effectiveVisibleCarIds = visibleCarIds ?? new Set(fleetData.carIds);
  function toggleCarVisible(id) {
    const next = new Set(effectiveVisibleCarIds);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    setVisibleCarIds(next);
  }

  useEffect(() => {
    if (!ranking || !onSummary) return;
    onSummary({
      fileCount: 1,
      trainId: rawRows?.[0]?.["Train number"] ?? null,
      groundTruthCarId,
      stats: [
        { label: "Cars analysed", value: ranking.length },
        { label: "Most likely faulty", value: `Car ${ranking[0].id}`, tone: "critical" },
        { label: "Top deviation score", value: ranking[0].score.toFixed(2) },
      ],
      // Not a "health" percentage — cross-car z-score deviation is
      // unbounded, so it's exposed as a raw score for ranking, not a
      // 0-100 scale (see valueType).
      entities: ranking.map((c, i) => ({
        id: `Car ${c.id}`,
        label: i === 0 ? "Most likely faulty" : `Rank ${i + 1}`,
        value: c.score,
        valueType: "score",
      })),
      // Exact acv_predictions.csv schema (file_id, ranked_cars) — one row
      // per uploaded case file, for the Overview's per-subsystem box.
      predictions: [{ file_id: fileName, ranked_cars: ranking.map((c) => c.id).join("|") }],
      predictionCount: 1,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ranking]);

  return (
    <div className="space-y-5">
      <Card>
        <div className="text-sm font-medium text-ink-primary mb-1">What this does</div>
        <p className="text-xs text-ink-muted">
          Ranks every car in a file by how much its readings deviate from its peers at the same moments. Opens
          pre-loaded with <code className="text-ink-secondary">acv_case_01.xlsx</code>, a labelled Train case checked
          against <code className="text-ink-secondary">Train_Labels.csv</code>. Drop any other{" "}
          <code className="text-ink-secondary">.xlsx</code> case file (Train or Test) to replace it.
        </p>
      </Card>

      {ranking && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <StatCard label="Cars analysed" value={ranking.length} />
          <StatCard
            label="Model's most likely faulty"
            value={`Car ${ranking[0].id}`}
            tone={groundTruthCarId ? (modelMatchesGroundTruth ? "good" : "critical") : "critical"}
          />
          {groundTruthCarId ? (
            <StatCard label="Confirmed faulty (Train_Labels.csv)" value={`Car ${groundTruthCarId}`} tone="good" />
          ) : (
            <StatCard label="Top deviation score" value={ranking[0].score.toFixed(2)} />
          )}
          <StatCard label="Gap to runner-up" value={gap != null ? gap.toFixed(2) : "—"} />
        </div>
      )}

      {groundTruthCarId && ranking && (
        <div
          className={`text-xs rounded-md px-3 py-2 border ${
            modelMatchesGroundTruth
              ? "text-status-good border-status-good/40 bg-status-good/10"
              : "text-status-critical border-status-critical/40 bg-status-critical/10"
          }`}
        >
          {modelMatchesGroundTruth
            ? `Model's top pick (Car ${ranking[0].id}) matches the confirmed faulty car from Train_Labels.csv.`
            : `Model's top pick (Car ${ranking[0].id}) does NOT match the confirmed faulty car (Car ${groundTruthCarId}) from Train_Labels.csv.`}
        </div>
      )}

      {fleetData.points.length > 0 && (
        <FleetComparisonChart
          title="All cars vs fleet median — indoor temperature"
          subtitle="Every car's indoor temperature over the file's real time range, against the whole-fleet median (dashed). Tick/untick cars below to compare any subset — the confirmed-faulty car (from Train_Labels.csv) is drawn with a bolder line."
          carIds={fleetData.carIds}
          points={fleetData.points}
          faultyCarId={groundTruthCarId}
          visibleCarIds={effectiveVisibleCarIds}
          onToggleCar={toggleCarVisible}
          caveat="Median-per-time-bucket per car. Fleet median = median across all cars' own medians at each moment."
        />
      )}

      {ranking && (
        <IndoorOutdoorChart
          title={`Car ${activeCarId} — indoor vs outdoor temperature`}
          subtitle="A working AC keeps indoor temperature below outdoor/ambient. Each point is the median reading in that %-of-file-elapsed window. Red band = indoor at or above outdoor for that stretch."
          data={indoorOutdoor}
          carOptions={ranking.map((c) => c.id)}
          selectedCarId={activeCarId}
          onSelectCar={setSelectedCarId}
          caveat="Uses this car's own indoor and outdoor sensors. Car list is ordered most- to least-likely faulty."
        />
      )}

      {ranking && (
        <PeerComparisonChart
          title={`Car ${activeCarId} — cooling setpoint vs peers`}
          subtitle="What the AC controller is asking for (its cooling-temperature setpoint), not what's achieved. A useful supporting view alongside the deviation-score ranking, not a standalone fault detector."
          data={controlTempVsPeers}
          unit="°C"
          carOptions={ranking.map((c) => c.id)}
          selectedCarId={activeCarId}
          onSelectCar={setSelectedCarId}
          caveat="Peer median = median cooling setpoint across the other 7 cars at the same moment. No 'good' direction here — any large gap from peers is the anomaly signal."
        />
      )}

      <FileDrop onFiles={(files) => runFile(files[0])} accept=".xlsx" hint="A single .xlsx case file, e.g. acv_case_01.xlsx" />

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

      {ranking && (
        <Card className="!px-0 !py-0 overflow-hidden">
          <div className="flex items-center justify-between px-4 py-3 border-b border-line-hairline">
            <div className="text-sm font-medium text-ink-primary">Ranked cars, most likely faulty first</div>
            <PrimaryButton onClick={handleDownload}>Download acv_predictions.csv</PrimaryButton>
          </div>
          <div className="px-4 py-3 space-y-2">
            {ranking.map((c, i) => (
              <div key={c.id} className="flex items-center gap-3">
                <span className="w-6 text-xs text-ink-muted tabular-nums">{i + 1}</span>
                <span className="w-16 text-sm font-medium text-ink-primary">Car {c.id}</span>
                <div className="flex-1 h-2 rounded-full bg-surface-raised overflow-hidden">
                  <div
                    className={`h-full ${i === 0 ? "bg-status-critical" : "bg-series-blue"}`}
                    style={{ width: `${Math.max(4, (c.score / maxScore) * 100)}%` }}
                  />
                </div>
                <span className="w-16 text-xs text-ink-muted tabular-nums text-right">
                  {c.score.toFixed(2)}
                </span>
                {c.id === groundTruthCarId && (
                  <span className="text-[11px] text-status-good font-medium shrink-0">confirmed faulty</span>
                )}
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
