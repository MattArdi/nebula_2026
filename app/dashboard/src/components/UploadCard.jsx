import { useState } from "react";
import { downloadCsvText } from "../lib/csvExport.js";

const BUTTON_BASE = "inline-flex items-center rounded-md px-3 py-1.5 text-xs font-medium transition-colors";

function Chevron({ open }) {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
      className={`shrink-0 text-ink-muted transition-transform ${open ? "rotate-90" : ""}`}
    >
      <path d="m9 6 6 6-6 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

/**
 * One uploaded file as a card: a tinted header (file name + actions) over a
 * cream body. Processing → [View Prediction] → once viewed, the header gets
 * a dropdown chevron that opens/closes the body and shows
 * [Download …] / [Upload to current dataset] / [Uploaded].
 *
 * `children` is a function of the computed run that renders the body.
 */
export default function UploadCard({ upload, isCurrent, onApply, children }) {
  const [viewed, setViewed] = useState(false);
  const [expanded, setExpanded] = useState(true);
  const { name, baseName, status, progress, run, error } = upload;

  return (
    <div className="rounded-lg border border-line-border bg-surface-card overflow-hidden">
      <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2 bg-surface-raised px-4 py-2.5">
        {viewed ? (
          <button
            onClick={() => setExpanded((e) => !e)}
            aria-expanded={expanded}
            className="flex items-center gap-2 min-w-0 text-left"
          >
            <Chevron open={expanded} />
            <span className="text-sm font-medium text-ink-primary truncate">{name}</span>
          </button>
        ) : (
          <span className="text-sm font-medium text-ink-primary truncate">{name}</span>
        )}

        <div className="flex flex-wrap items-center gap-2">
          {status === "processing" && <span className="text-xs text-ink-muted">Processing…{progress ? ` ${progress}` : ""}</span>}
          {status === "error" && <span className="text-xs text-status-critical">{error}</span>}

          {status === "ready" && !viewed && (
            <button
              onClick={() => setViewed(true)}
              className={`${BUTTON_BASE} bg-series-blue text-white hover:bg-series-blue/85`}
            >
              View Prediction
            </button>
          )}

          {status === "ready" && viewed && (
            <>
              <button
                onClick={() => downloadCsvText(`${baseName}_predictions.csv`, run.csvText)}
                className={`${BUTTON_BASE} bg-series-blue text-white hover:bg-series-blue/85`}
              >
                Download {baseName}_predictions.csv
              </button>
              {isCurrent ? (
                <span className={`${BUTTON_BASE} border border-status-good/40 bg-status-good/10 text-status-good`}>
                  Uploaded
                </span>
              ) : (
                <button
                  onClick={onApply}
                  className={`${BUTTON_BASE} border border-line-border bg-surface-card text-ink-primary hover:border-series-blue/60`}
                >
                  Upload to current dataset
                </button>
              )}
            </>
          )}
        </div>
      </div>

      {status === "ready" && viewed && expanded && (
        <div className="border-t border-line-border bg-surface-page px-4 py-4">{children(run)}</div>
      )}
    </div>
  );
}
