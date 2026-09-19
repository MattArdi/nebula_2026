import { useEffect, useRef, useState } from "react";
import { Card } from "./ui.jsx";
import { usePx } from "../lib/useRem.js";

// Matches tailwind.config.js's status colors exactly, since SVG fill can't
// read Tailwind classes directly.
const COLORS = { Normal: "#0ca30c", "Abnormal resistance": "#e66767" };
const SELECT_COLOR = "#3987e5";

// Below this the bars get too thin to hover, so the timeline stops shrinking
// and scrolls sideways instead.
const MIN_WIDTH = 720;
const MIN_BAR_W = 4; // real cycle durations are often sub-pixel at this scale — floor so short cycles stay visible

function LegendDot({ color, label }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: color }} />
      {label}
    </span>
  );
}

/**
 * A Gantt-style timeline of observed door cycles, split into an Open lane
 * and a Close lane (the dataset's own "Close command" flag, not predicted),
 * so an Open cycle and the Close cycle that follows it don't visually
 * collide on one row.
 *
 * Drawn at real pixel size — its height and text don't grow with the popup —
 * and it fills the popup's width, scrolling sideways once that drops below
 * MIN_WIDTH. Click a bar to select it (for a detail panel elsewhere on the
 * page); hover shows a quick tooltip either way.
 */
export default function SegmentTimeline({ title, subtitle, historical, xFormat, caveat, selected, onSelect }) {
  const px = usePx();
  const scrollRef = useRef(null);
  const [containerW, setContainerW] = useState(0);
  const [hover, setHover] = useState(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setContainerW(el.clientWidth));
    ro.observe(el);
    setContainerW(el.clientWidth);
    return () => ro.disconnect();
  }, []);

  if (!historical?.length) return null;

  const width = Math.max(containerW, px(MIN_WIDTH));
  const padLeft = px(64); // room for "Open"/"Close" row labels
  const padRight = px(14);
  const barH = px(34);
  const rowOpenY = px(12);
  const rowCloseY = rowOpenY + barH + px(14);
  const axisY = rowCloseY + barH + px(16);
  const height = axisY + px(34);

  const xMin = historical[0].start_ts;
  const xMax = historical[historical.length - 1].end_ts;
  const span = Math.max(1, xMax - xMin);
  const usableW = width - padLeft - padRight;
  const scaleX = (t) => padLeft + ((t - xMin) / span) * usableW;

  const tickCount = 5;
  const ticks = Array.from({ length: tickCount + 1 }, (_, i) => xMin + (span * i) / tickCount);

  function renderBar(seg, i) {
    const x1 = scaleX(seg.start_ts);
    const x2 = scaleX(seg.end_ts);
    const w = Math.max(MIN_BAR_W, x2 - x1);
    const rowY = seg.operation === "Close" ? rowCloseY : rowOpenY;
    const color = COLORS[seg.prediction] ?? "#908e87";
    const isSelected = selected === seg;
    return (
      <rect
        key={`${seg.operation}-${i}`}
        x={x1}
        y={rowY}
        width={w}
        height={barH}
        rx={4}
        fill={color}
        fillOpacity={0.85}
        stroke={isSelected ? SELECT_COLOR : "none"}
        strokeWidth={isSelected ? 2.5 : 0}
        style={{ cursor: "pointer" }}
        onMouseEnter={(e) => {
          const r = e.currentTarget.getBoundingClientRect();
          setHover({ seg, x: r.left + r.width / 2, y: r.top });
        }}
        onMouseLeave={() => setHover((h) => (h?.seg === seg ? null : h))}
        onClick={() => onSelect?.(seg)}
      />
    );
  }

  return (
    <Card>
      <div className="flex flex-wrap items-center justify-between gap-2 mb-1">
        <div className="text-base font-medium text-ink-primary">{title}</div>
        <div className="flex items-center gap-3 text-sm text-ink-muted">
          <LegendDot color={COLORS.Normal} label="Normal" />
          <LegendDot color={COLORS["Abnormal resistance"]} label="Abnormal" />
        </div>
      </div>
      {subtitle && <p className="text-sm text-ink-muted mb-3">{subtitle}</p>}

      <div ref={scrollRef} className="overflow-x-auto">
        <svg width={width} height={height} style={{ display: "block" }}>
          <text x={0} y={rowOpenY + barH / 2 + px(5)} fontSize={px(14)} fill="#605f5a">
            Open
          </text>
          <text x={0} y={rowCloseY + barH / 2 + px(5)} fontSize={px(14)} fill="#605f5a">
            Close
          </text>

          <line x1={padLeft} y1={axisY} x2={width - padRight} y2={axisY} stroke="#e3dfd3" strokeWidth={1} />

          {ticks.map((t, i) => (
            <g key={i}>
              <line x1={scaleX(t)} y1={axisY} x2={scaleX(t)} y2={axisY + 5} stroke="#e3dfd3" />
              <text x={scaleX(t)} y={axisY + px(22)} fontSize={px(13)} fill="#908e87" textAnchor="middle">
                {xFormat ? xFormat(t) : t}
              </text>
            </g>
          ))}

          {historical.map((s, i) => renderBar(s, i))}
        </svg>
      </div>

      {/* Fixed-position so the scroll container above can't clip it. */}
      {hover && (
        <div
          className="fixed pointer-events-none bg-surface-raised border border-line-border rounded-md px-3 py-2 text-sm shadow-xl whitespace-nowrap z-50"
          style={{
            left: Math.min(Math.max(hover.x, 110), window.innerWidth - 110),
            top: hover.y - 8,
            transform: "translate(-50%, -100%)",
          }}
        >
          <div className="font-medium" style={{ color: COLORS[hover.seg.prediction] }}>
            {hover.seg.operation} — {hover.seg.prediction}
          </div>
          <div className="text-ink-muted">
            {xFormat ? xFormat(hover.seg.start_ts) : hover.seg.start_ts} –{" "}
            {xFormat ? xFormat(hover.seg.end_ts) : hover.seg.end_ts}
          </div>
          <div className="text-ink-muted mt-0.5">Click for signal detail</div>
        </div>
      )}

      {caveat && <p className="text-xs text-ink-muted mt-2 pt-2 border-t border-line-hairline">{caveat}</p>}
    </Card>
  );
}
