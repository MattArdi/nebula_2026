import { useState } from "react";

const TONE_BAR = { good: "bg-status-good", serious: "bg-status-serious", critical: "bg-status-critical", neutral: "bg-series-blue" };
const TONE_TEXT = { good: "text-status-good", serious: "text-status-serious", critical: "text-status-critical", neutral: "text-ink-primary" };

function toneFor(entity) {
  if (entity.valueType !== "health") return "neutral";
  if (entity.value >= 80) return "good";
  if (entity.value >= 50) return "serious";
  return "critical";
}

function EntityRow({ entity }) {
  const tone = toneFor(entity);
  const isHealth = entity.valueType === "health";
  // ACV's deviation score is unbounded (not 0-100) — the bar width below is
  // a rough visual aid only (capped at 100%), the displayed number is the
  // real, unscaled value.
  const barWidth = isHealth ? Math.max(2, entity.value) : Math.min(100, entity.value * 40);

  return (
    <div className="flex items-center gap-2 text-[11px] py-1 border-b border-line-hairline last:border-0">
      <span className="w-24 truncate text-ink-secondary" title={entity.id}>
        {entity.id}
      </span>
      <span className="w-16 truncate text-ink-muted">{entity.label}</span>
      <div className="flex-1 h-1.5 rounded-full bg-surface-raised overflow-hidden">
        <div className={`h-full ${TONE_BAR[tone]}`} style={{ width: `${barWidth}%` }} />
      </div>
      <span className={`w-10 text-right tabular-nums ${TONE_TEXT[tone]}`}>
        {isHealth ? entity.value.toFixed(0) : entity.value.toFixed(2)}
      </span>
    </div>
  );
}

/**
 * Collapsible per-entity breakdown (worst-first), meant to be embedded
 * directly inside a subsystem's own summary box on the Overview page —
 * a sibling of the box's navigate-button, not nested inside it, since the
 * toggle below is itself a button.
 */
export function EntityList({ entities }) {
  const [open, setOpen] = useState(false);
  if (!entities?.length) return null;

  const sorted = [...entities].sort((a, b) => (a.valueType === "health" ? a.value - b.value : b.value - a.value));

  return (
    <div className="border-t border-line-hairline">
      <button
        onClick={(e) => {
          e.stopPropagation();
          setOpen((o) => !o);
        }}
        className="w-full flex items-center justify-between px-4 py-2 text-left hover:bg-surface-raised transition-colors"
      >
        <span className="text-[11px] font-medium text-ink-muted uppercase tracking-wide">Per-entity breakdown</span>
        <span className="text-[11px] text-ink-muted">
          {open ? "Hide" : "Show"} {sorted.length} {sorted.length === 1 ? "entry" : "entries"} {open ? "▲" : "▼"}
        </span>
      </button>
      {open && (
        <div className="px-4 pb-3 max-h-64 overflow-y-auto">
          {sorted.map((e, i) => (
            <EntityRow key={`${e.id}-${i}`} entity={e} />
          ))}
        </div>
      )}
    </div>
  );
}
