import { useEffect, useRef, useState } from "react";
import Papa from "papaparse";
import { FileDrop, StatCard } from "../../components/ui.jsx";
import { CurrentDatasetHeader, UploadHeading } from "../../components/DatasetSections.jsx";
import UploadCard from "../../components/UploadCard.jsx";
import RailAbnormalitiesChart from "../../components/RailAbnormalitiesChart.jsx";
import { predictRail } from "../../lib/apiClient.js";
import { fetchAsFile } from "../../lib/sampleFiles.js";
import { RAIL_SAMPLE_PREDICTIONS } from "../../lib/sampleManifest.js";
import { useDatasetUploads } from "../../lib/useDatasetUploads.js";
import { expandZipFiles } from "../../lib/zip.js";

// Files per backend request — each is ~17 MB, so this keeps a single
// request modest while still paying the pipeline's startup cost only once
// per chunk instead of once per file.
const CHUNK_SIZE = 5;

const naturalCompare = (a, b) => a.localeCompare(b, undefined, { numeric: true });

// Puts predictions in recording order (Train2 before Train10) and numbers each
// second — by the number in its file name (Train37.csv is second 37) when
// every file has one, otherwise by position.
function buildRun(predictions) {
  const sorted = [...predictions].sort((a, b) => naturalCompare(a.file_id, b.file_id));
  const numbers = sorted.map((r) => /(\d+)\.[^.]*$/.exec(r.file_id)?.[1]);
  const useNumbers = numbers.every((n) => n != null);
  const rows = sorted.map((r, i) => ({ ...r, second: useNumbers ? Number(numbers[i]) : i + 1 }));

  const csvText =
    Papa.unparse({ fields: ["file_id", "prediction"], data: rows.map((r) => [r.file_id, r.prediction]) }, { newline: "\n" }) +
    "\n";
  return { rows, csvText };
}

// A rail dataset is a set of files, each one second of the recording. Runs
// the backend on all of them.
async function computeRun(files, onProgress) {
  const predictions = [];
  for (let i = 0; i < files.length; i += CHUNK_SIZE) {
    const result = await predictRail(files.slice(i, i + CHUNK_SIZE));
    predictions.push(...result.rows);
    onProgress?.(Math.min(i + CHUNK_SIZE, files.length), files.length);
  }
  return buildRun(predictions);
}

async function loadSampleRun() {
  const file = await fetchAsFile(RAIL_SAMPLE_PREDICTIONS.url, RAIL_SAMPLE_PREDICTIONS.name);
  const parsed = Papa.parse(await file.text(), { header: true, skipEmptyLines: true });
  return buildRun(parsed.data);
}

// Everything dropped in one go — loose CSVs and/or zips of CSVs — is one
// dataset. A single file or zip keeps its own name; several files are named
// by count.
async function groupRailFiles(files) {
  const csvs = await expandZipFiles(files, { extensions: [".csv"] });
  if (!csvs.length) throw new Error("No .csv files found.");
  if (files.length === 1) {
    return [{ name: files[0].name, baseName: files[0].name.replace(/\.[^.]+$/, ""), input: csvs }];
  }
  return [{ name: `${csvs.length} files`, baseName: "rail", input: csvs }];
}

function countByClass(rows) {
  const counts = { Normal: 0, "Side I": 0, "Side II": 0 };
  for (const r of rows) counts[r.prediction] = (counts[r.prediction] ?? 0) + 1;
  return counts;
}

const TILE_TONES = {
  "Side I": { text: "text-status-serious", border: "border-status-serious/60" },
  "Side II": { text: "text-status-critical", border: "border-status-critical/60" },
};

// A Side I / Side II count that opens a list of exactly which seconds are
// affected. With none, it's a plain tile.
function AbnormalityTile({ label, count, open, onToggle }) {
  const tone = TILE_TONES[label];
  if (count === 0) return <StatCard label={label} value={count} tone="good" />;

  return (
    <button
      onClick={onToggle}
      aria-expanded={open}
      className={`w-full text-left rounded-lg border bg-surface-card px-4 py-3.5 transition-colors hover:bg-surface-raised ${
        open ? tone.border : "border-line-border"
      }`}
    >
      <div className="flex items-center justify-between text-xs text-ink-muted">
        <span>{label}</span>
        <svg
          width="12"
          height="12"
          viewBox="0 0 24 24"
          fill="none"
          aria-hidden="true"
          className={`transition-transform ${open ? "rotate-180" : ""}`}
        >
          <path d="m6 9 6 6 6-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </div>
      <div className={`text-2xl font-semibold tabular-nums mt-1 ${tone.text}`}>{count}</div>
    </button>
  );
}

// Stat tiles + abnormalities dot graph for one dataset's result — used for
// both the current dataset and each upload's "View Prediction" panel.
function RailResultView({ run }) {
  const [openSide, setOpenSide] = useState(null); // "Side I" | "Side II" | null
  const [activeIndex, setActiveIndex] = useState(null); // index into run.rows pinned on the graph

  // A different dataset has different seconds.
  useEffect(() => {
    setOpenSide(null);
    setActiveIndex(null);
  }, [run]);

  const counts = countByClass(run.rows);

  function toggleSide(side) {
    setOpenSide((cur) => (cur === side ? null : side));
    setActiveIndex(null);
  }

  const openRows = openSide
    ? run.rows.map((r, index) => ({ ...r, index })).filter((r) => r.prediction === openSide)
    : [];

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-3 gap-3">
        <StatCard label="Normal" value={counts.Normal} tone="good" />
        <AbnormalityTile label="Side I" count={counts["Side I"]} open={openSide === "Side I"} onToggle={() => toggleSide("Side I")} />
        <AbnormalityTile label="Side II" count={counts["Side II"]} open={openSide === "Side II"} onToggle={() => toggleSide("Side II")} />
      </div>

      {openSide && (
        <div className="rounded-lg border border-line-border bg-surface-card px-4 py-3">
          <div className="text-xs text-ink-muted mb-2">
            {openSide} at these seconds — pick one to see it on the graph
          </div>
          <div className="flex flex-wrap gap-2 max-h-32 overflow-y-auto">
            {openRows.map((r) => (
              <button
                key={r.index}
                onClick={() => setActiveIndex((cur) => (cur === r.index ? null : r.index))}
                className={`rounded-md border px-2.5 py-1 text-xs tabular-nums transition-colors ${
                  activeIndex === r.index
                    ? "border-series-blue bg-series-blue/10 text-ink-primary font-medium"
                    : "border-line-border text-ink-secondary hover:bg-surface-raised"
                }`}
              >
                Second {r.second}
              </button>
            ))}
          </div>
        </div>
      )}

      <RailAbnormalitiesChart rows={run.rows} activeIndex={activeIndex} />
    </div>
  );
}

export default function RailPage({ onSummary }) {
  // The dataset the stats, chart and Overview card are showing — starts as
  // the bundled sample, and is swapped by "Upload to current dataset".
  // Everything here is in-memory only, so a refresh resets it.
  const [currentRun, setCurrentRun] = useState(null);
  const [loadError, setLoadError] = useState(null);
  const { uploads, appliedKey, setAppliedKey, handleFiles } = useDatasetUploads(computeRun, groupRailFiles);
  const autoLoadedRef = useRef(false);

  useEffect(() => {
    if (autoLoadedRef.current) return;
    autoLoadedRef.current = true;
    (async () => {
      try {
        setCurrentRun(await loadSampleRun());
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
    const counts = countByClass(currentRun.rows);
    onSummary({
      fileCount: 1 + uploads.length,
      abnormalBySide: { "Side I": counts["Side I"], "Side II": counts["Side II"] },
      stats: [
        { label: "Normal", value: counts.Normal, tone: "good" },
        { label: "Side I", value: counts["Side I"] },
        { label: "Side II", value: counts["Side II"] },
      ],
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentRun]);

  return (
    <div className="space-y-5">
      <CurrentDatasetHeader id="rail" />

      {loadError && (
        <div className="text-xs rounded-md px-3 py-2 border text-status-critical border-status-critical/40 bg-status-critical/10">
          {loadError}
        </div>
      )}
      {!currentRun && !loadError && <div className="text-xs text-ink-muted">Processing…</div>}

      {currentRun && <RailResultView run={currentRun} />}

      <div className="space-y-3">
        <UploadHeading />
        <FileDrop onFiles={handleFiles} accept=".csv,.zip" multiple />
      </div>

      {uploads.length > 0 && (
        <div className="space-y-5">
          {uploads.map((u) => (
            <UploadCard key={u.key} upload={u} isCurrent={appliedKey === u.key} onApply={() => applyUpload(u)}>
              {(run) => <RailResultView run={run} />}
            </UploadCard>
          ))}
        </div>
      )}
    </div>
  );
}
