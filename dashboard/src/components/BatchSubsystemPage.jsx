import { useEffect, useMemo, useRef, useState } from "react";
import { FileDrop, PrimaryButton, Card, StatCard, ProgressBar, LabelBadge } from "./ui.jsx";
import { downloadCsv } from "../lib/csvExport.js";
import { fetchAllAsFiles } from "../lib/sampleFiles.js";
import { expandZipFiles } from "../lib/zip.js";

// Shared "drop many files -> one prediction row per file" dashboard, used by
// Rail Corrugation and SHM. Door and ACV have different-shaped output.
export default function BatchSubsystemPage({
  title,
  description,
  csvFilename,
  accept,
  parseFile,
  predictFile,
  predictionHeader = "prediction",
  formatPrediction = (v) => String(v),
  sampleFiles,
  computeStats,
  computeHealth,
  computeEntities,
  renderCell,
  renderDetail,
  // Renders a chart/summary using ALL currently loaded files together, e.g.
  // every file plotted as one point/bar in a single combined chart.
  renderCombined,
  defaultSelect, // (results) => file_id to select once loaded
  // Ground truth (e.g. from a Train_Labels.csv), keyed by file_id.
  trueLabels,
  matchesTrueLabel = (prediction, trueLabel) => String(prediction) === String(trueLabel),
  onSummary,
}) {
  const [results, setResults] = useState(null);
  const [selectedId, setSelectedId] = useState(null);
  const [progress, setProgress] = useState({ current: 0, total: 0 });
  const [status, setStatus] = useState(null);
  const [busy, setBusy] = useState(false);
  const [zipStatus, setZipStatus] = useState(null);
  const autoLoadedRef = useRef(false);

  async function runFiles(files, sourceLabel) {
    setStatus(null);
    setResults(null);
    setSelectedId(null);
    setBusy(true);
    setProgress({ current: 0, total: files.length });

    const rows = [];
    const errors = [];

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      try {
        const parsed = await parseFile(file);
        // Spread `extra` through — only prediction/label are special-cased,
        // anything else a subsystem's predictFile returns rides along.
        const { prediction, label, ...extra } = await predictFile(parsed, file);
        rows.push({
          file_id: file.name,
          prediction,
          label: label ?? formatPrediction(prediction),
          parsed: renderDetail ? parsed : undefined,
          ...extra,
        });
      } catch (err) {
        errors.push(`${file.name}: ${err.message}`);
      }
      setProgress({ current: i + 1, total: files.length });
    }

    setBusy(false);
    setResults(rows);
    if (rows.length) setSelectedId(defaultSelect ? defaultSelect(rows) : rows[0].file_id);
    setStatus(
      errors.length
        ? { type: "error", message: `${rows.length} file(s) predicted; ${errors.length} failed: ${errors.slice(0, 3).join("; ")}${errors.length > 3 ? "…" : ""}` }
        : { type: "ok", message: `${sourceLabel ?? "Predicted"} ${rows.length} file${rows.length === 1 ? "" : "s"}.` }
    );
  }

  useEffect(() => {
    if (!sampleFiles?.length || autoLoadedRef.current) return;
    autoLoadedRef.current = true;
    (async () => {
      try {
        const files = await fetchAllAsFiles(sampleFiles);
        await runFiles(files, "Loaded PS3 sample data —");
      } catch (err) {
        setStatus({ type: "error", message: `Could not load bundled sample data: ${err.message}` });
        setBusy(false);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function handleDownload() {
    downloadCsv(
      csvFilename,
      ["file_id", "prediction"],
      results.map((r) => ({ file_id: r.file_id, prediction: r.prediction }))
    );
  }

  // Derived separately so ground truth arriving after the sample files
  // finish loading still attaches correctly.
  const resultsWithTruth = useMemo(() => {
    if (!results) return results;
    if (!trueLabels) return results;
    return results.map((r) => {
      const trueLabel = trueLabels[r.file_id] ?? null;
      return { ...r, trueLabel, matchesTruth: trueLabel != null ? matchesTrueLabel(r.prediction, trueLabel) : null };
    });
  }, [results, trueLabels]);

  const stats = resultsWithTruth && computeStats ? computeStats(resultsWithTruth) : null;

  useEffect(() => {
    if (resultsWithTruth && onSummary) {
      onSummary({
        stats,
        fileCount: resultsWithTruth.length,
        health: computeHealth ? computeHealth(resultsWithTruth) : undefined,
        entities: computeEntities ? computeEntities(resultsWithTruth) : undefined,
        // Preview rows in the exact <subsystem>_predictions.csv schema
        // (file_id, prediction) for the Overview's per-subsystem box.
        predictions: resultsWithTruth.slice(0, 4).map((r) => ({ file_id: r.file_id, prediction: r.prediction })),
        predictionCount: resultsWithTruth.length,
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [resultsWithTruth]);

  const selected = resultsWithTruth?.find((r) => r.file_id === selectedId);

  async function handleDroppedFiles(files) {
    const hasZip = files.some((f) => f.name.toLowerCase().endsWith(".zip"));
    if (!hasZip) return runFiles(files, "Predicted");
    setZipStatus("Unzipping…");
    try {
      const expanded = await expandZipFiles(files, { extensions: [".csv"] });
      setZipStatus(null);
      if (!expanded.length) {
        setStatus({ type: "error", message: "No .csv files found inside the uploaded zip." });
        return;
      }
      await runFiles(expanded, `Unzipped and predicted`);
    } catch (err) {
      setZipStatus(null);
      setStatus({ type: "error", message: `Could not read zip file: ${err.message}` });
    }
  }

  return (
    <div className="space-y-5">
      <Card>
        <div className="text-sm font-medium text-ink-primary mb-1">What this does</div>
        <p className="text-xs text-ink-muted">{description}</p>
      </Card>

      {stats && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {stats.map((s) => (
            <StatCard key={s.label} label={s.label} value={s.value} tone={s.tone} />
          ))}
        </div>
      )}

      {renderCombined && resultsWithTruth && renderCombined(resultsWithTruth)}

      {renderDetail && resultsWithTruth && resultsWithTruth.length > 0 && (
        <div className="flex items-center gap-2">
          <span className="text-xs text-ink-muted">Inspect file:</span>
          <select
            value={selectedId ?? ""}
            onChange={(e) => setSelectedId(e.target.value)}
            className="text-xs bg-surface-raised border border-line-border rounded-md px-1.5 py-1 text-ink-primary"
          >
            {resultsWithTruth.map((r) => (
              <option key={r.file_id} value={r.file_id}>
                {r.file_id}
              </option>
            ))}
          </select>
        </div>
      )}

      {renderDetail && selected && renderDetail(selected)}

      <FileDrop
        onFiles={handleDroppedFiles}
        accept={accept ? `${accept},.zip` : ".zip"}
        multiple
        hint="Drop one or many files (or a single .zip of files) to replace the sample data above — one prediction row per file"
      />
      {zipStatus && <div className="text-xs text-ink-muted">{zipStatus}</div>}

      {busy && <ProgressBar current={progress.current} total={progress.total} />}

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

      {resultsWithTruth && resultsWithTruth.length > 0 && (
        <Card className="!px-0 !py-0 overflow-hidden">
          <div className="flex items-center justify-between px-4 py-3 border-b border-line-hairline">
            <div className="text-sm font-medium text-ink-primary">{title} — {resultsWithTruth.length} files</div>
            <PrimaryButton onClick={handleDownload}>Download {csvFilename}</PrimaryButton>
          </div>
          <div className="overflow-x-auto max-h-[480px] overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-surface-card">
                <tr className="text-left text-xs text-ink-muted border-b border-line-hairline">
                  <th className="px-4 py-2 font-normal">File</th>
                  <th className="px-4 py-2 font-normal">{predictionHeader}</th>
                  {trueLabels && <th className="px-4 py-2 font-normal">True label</th>}
                </tr>
              </thead>
              <tbody>
                {resultsWithTruth.map((r) => (
                  <tr
                    key={r.file_id}
                    onClick={renderDetail ? () => setSelectedId(r.file_id) : undefined}
                    className={`border-b border-line-hairline last:border-0 ${
                      renderDetail ? "cursor-pointer hover:bg-surface-raised transition-colors" : ""
                    } ${renderDetail && r.file_id === selectedId ? "bg-series-blue/10" : ""}`}
                  >
                    <td className="px-4 py-2 text-ink-secondary">{r.file_id}</td>
                    <td className="px-4 py-2">
                      {renderCell
                        ? renderCell(r)
                        : typeof r.prediction === "string"
                          ? <LabelBadge label={r.prediction} />
                          : r.label}
                    </td>
                    {trueLabels && (
                      <td className="px-4 py-2">
                        {r.trueLabel != null && (
                          <span className={r.matchesTruth ? "text-status-good" : "text-status-critical"}>
                            {r.trueLabel}
                          </span>
                        )}
                      </td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}
