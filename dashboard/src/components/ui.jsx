import { useRef, useState } from "react";

export function FileDrop({ onFiles, accept, multiple = false, hint }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  function handleFiles(fileList) {
    const files = Array.from(fileList ?? []);
    if (files.length) onFiles(files);
  }

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        handleFiles(e.dataTransfer.files);
      }}
      onClick={() => inputRef.current?.click()}
      className={`rounded-lg border-2 border-dashed px-6 py-10 text-center cursor-pointer transition-colors ${
        dragging
          ? "border-series-blue bg-series-blue/10"
          : "border-line-border bg-surface-card hover:border-series-blue/50"
      }`}
    >
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none" className="mx-auto mb-2 text-ink-muted" aria-hidden="true">
        <path
          d="M12 4v11m0-11 4 4m-4-4-4 4M5 17v2a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-2"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
      <div className="text-sm text-ink-primary font-medium">
        Drag &amp; drop {multiple ? "file(s)" : "a file"} here, or click to browse
      </div>
      {hint && <div className="text-xs text-ink-muted mt-1.5">{hint}</div>}
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        multiple={multiple}
        className="hidden"
        onChange={(e) => handleFiles(e.target.files)}
      />
    </div>
  );
}

export function ProgressBar({ current, total, label }) {
  if (total === 0) return null;
  const pct = Math.round((current / total) * 100);
  return (
    <div>
      <div className="flex items-center justify-between text-xs text-ink-muted mb-1">
        <span>{label ?? `Processing ${current} of ${total}`}</span>
        <span className="tabular-nums">{pct}%</span>
      </div>
      <div className="h-1.5 rounded-full bg-surface-raised overflow-hidden">
        <div
          className="h-full bg-series-blue transition-all duration-150"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

const LABEL_TONE = {
  Normal: "good",
  Operational: "good",
  "Side I": "critical",
  "Side II": "critical",
  "Abnormal resistance": "critical",
};

const TONE_CLASSES = {
  critical: "bg-status-critical/15 text-status-critical border-status-critical/40",
  serious: "bg-status-serious/15 text-status-serious border-status-serious/40",
  good: "bg-status-good/15 text-status-good border-status-good/40",
  neutral: "bg-surface-raised text-ink-secondary border-line-border",
};

export function LabelBadge({ label }) {
  const tone = LABEL_TONE[label] ?? "neutral";
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2 py-0.5 text-xs font-medium ${TONE_CLASSES[tone]}`}
    >
      {label}
    </span>
  );
}

export function PrimaryButton({ children, onClick, disabled }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="inline-flex items-center gap-2 rounded-md bg-series-blue px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-series-blue/85 disabled:opacity-40 disabled:cursor-not-allowed"
    >
      {children}
    </button>
  );
}

export function StatCard({ label, value, tone = "default", hint }) {
  const toneClass =
    {
      critical: "text-status-critical",
      serious: "text-status-serious",
      good: "text-status-good",
      default: "text-ink-primary",
    }[tone] ?? "text-ink-primary";

  return (
    <div className="rounded-lg border border-line-border bg-surface-card px-4 py-3.5">
      <div className="text-xs text-ink-muted">{label}</div>
      <div className={`text-2xl font-semibold tabular-nums mt-1 ${toneClass}`}>{value}</div>
      {hint && <div className="text-xs text-ink-muted mt-1">{hint}</div>}
    </div>
  );
}

export function Card({ children, className = "" }) {
  return (
    <div className={`rounded-lg border border-line-border bg-surface-card px-4 py-3.5 ${className}`}>
      {children}
    </div>
  );
}
