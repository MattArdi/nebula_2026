import { useState } from "react";
import { Card } from "./ui.jsx";

// Matches tailwind.config.js's status colors exactly, since SVG fill can't
// read Tailwind classes directly.
const COLORS = { Normal: "#0ca30c", "Abnormal resistance": "#e66767" };
const SELECT_COLOR = "#3987e5";

const WIDTH = 1000;
const HEIGHT = 175;
const PAD_LEFT = 40; // room for "Open"/"Close" row labels
const PAD_RIGHT = 10;
const ROW_OPEN_Y = 40;
const ROW_CLOSE_Y = 88;
const BAR_H = 32;
const AXIS_Y = HEIGHT - 30;
const MIN_BAR_W = 4; // real cycle durations are often sub-pixel at this scale — floor so short cycles stay visible

function LegendDot({ color, label }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="w-2 h-2 rounded-full shrink-0" style={{ background: color }} />
      {label}
    </span>
  );
}

/**
 * A Gantt-style timeline of observed door cycles, split into an Open lane
 * and a Close lane (the dataset's own "Door is opening"/"Door is closing"
 * flags, not predicted — see doorModel.js's classifyOperation), so an Open
 * cycle and the Close cycle that follows it don't visually collide on one
 * row.
 *
 * Click a bar to select it (for a detail panel elsewhere on the page);
 * hover shows a quick tooltip either way.
 */
export default function SegmentTimeline({ title, subtitle, historical, xFormat, caveat, selected, onSelect }) {
  const [hover, setHover] = useState(null);

  if (!historical?.length) return null;

  const xMin = historical[0].start_ts;
  const xMax = historical[historical.length - 1].end_ts;
  const span = Math.max(1, xMax - xMin);
  const usableW = WIDTH - PAD_LEFT - PAD_RIGHT;
  const scaleX = (t) => PAD_LEFT + ((t - xMin) / span) * usableW;

  const tickCount = 5;
  const ticks = Array.from({ length: tickCount + 1 }, (_, i) => xMin + (span * i) / tickCount);

  function renderBar(seg, i) {
    const x1 = scaleX(seg.start_ts);
    const x2 = scaleX(seg.end_ts);
    const w = Math.max(MIN_BAR_W, x2 - x1);
    const rowY = seg.operation === "Close" ? ROW_CLOSE_Y : ROW_OPEN_Y;
    const color = COLORS[seg.prediction] ?? "#898781";
    const isSelected = selected === seg;
    return (
      <rect
        key={`${seg.operation}-${i}`}
        x={x1}
        y={rowY}
        width={w}
        height={BAR_H}
        rx={4}
        fill={color}
        fillOpacity={0.85}
        stroke={isSelected ? SELECT_COLOR : "none"}
        strokeWidth={isSelected ? 2.5 : 0}
        style={{ cursor: "pointer" }}
        onMouseEnter={() => setHover({ seg, xPct: ((x1 + x2) / 2 / WIDTH) * 100 })}
        onMouseLeave={() => setHover((h) => (h?.seg === seg ? null : h))}
        onClick={() => onSelect?.(seg)}
      />
    );
  }

  return (
    <Card>
      <div className="flex flex-wrap items-center justify-between gap-2 mb-1">
        <div className="text-sm font-medium text-ink-primary">{title}</div>
        <div className="flex items-center gap-3 text-[11px] text-ink-muted">
          <LegendDot color={COLORS.Normal} label="Normal" />
          <LegendDot color={COLORS["Abnormal resistance"]} label="Abnormal resistance" />
        </div>
      </div>
      {subtitle && <p className="text-xs text-ink-muted mb-3">{subtitle}</p>}

      <div className="relative">
        <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="w-full" style={{ height: 175, display: "block" }}>
          <text x={4} y={ROW_OPEN_Y + BAR_H / 2 + 4} fontSize={11} fill="#c3c2b7">
            Open
          </text>
          <text x={4} y={ROW_CLOSE_Y + BAR_H / 2 + 4} fontSize={11} fill="#c3c2b7">
            Close
          </text>

          <line x1={PAD_LEFT} y1={AXIS_Y} x2={WIDTH - PAD_RIGHT} y2={AXIS_Y} stroke="#2c2c2a" strokeWidth={1} />

          {ticks.map((t, i) => (
            <g key={i}>
              <line x1={scaleX(t)} y1={AXIS_Y} x2={scaleX(t)} y2={AXIS_Y + 4} stroke="#2c2c2a" />
              <text x={scaleX(t)} y={AXIS_Y + 17} fontSize={10} fill="#898781" textAnchor="middle">
                {xFormat ? xFormat(t) : t}
              </text>
            </g>
          ))}

          {historical.map((s, i) => renderBar(s, i))}
        </svg>

        {hover && (
          <div
            className="absolute pointer-events-none bg-surface-raised border border-line-border rounded-md px-2.5 py-1.5 text-[11px] shadow-xl whitespace-nowrap z-10"
            style={{ left: `${hover.xPct}%`, top: 0, transform: "translate(-50%, -100%)" }}
          >
            <div className="font-medium" style={{ color: COLORS[hover.seg.prediction] }}>
              {hover.seg.operation} — {hover.seg.prediction}
            </div>
            <div className="text-ink-muted">
              {xFormat ? xFormat(hover.seg.start_ts) : hover.seg.start_ts} –{" "}
              {xFormat ? xFormat(hover.seg.end_ts) : hover.seg.end_ts}
            </div>
            <div className="text-ink-muted">probability abnormal: {(hover.seg.probAbnormal * 100).toFixed(0)}%</div>
            <div className="text-ink-muted mt-0.5">Click for signal detail</div>
          </div>
        )}
      </div>

      {caveat && <p className="text-[11px] text-ink-muted mt-2 pt-2 border-t border-line-hairline">{caveat}</p>}
    </Card>
  );
}
