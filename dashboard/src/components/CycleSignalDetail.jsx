import { ResponsiveContainer, ComposedChart, Area, Line, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";
import { Card } from "./ui.jsx";
import { resampleCycleSignal } from "../subsystems/door/doorModel.js";

const STATUS_COLOR = { Normal: "#0ca30c", "Abnormal resistance": "#e66767" };
const REFERENCE_COLOR = "#898781";
const GRID = "#2c2c2a";
const AXIS = "#898781";

function ChartTooltip({ active, payload, label, unit }) {
  if (!active || !payload?.length) return null;
  const actual = payload.find((p) => p.dataKey === "actual");
  const reference = payload.find((p) => p.dataKey === "reference");
  return (
    <div className="rounded-md border border-line-border bg-surface-raised px-3 py-2 text-xs shadow-lg">
      <div className="text-ink-muted mb-1">{label.toFixed(0)}% of cycle</div>
      {actual && (
        <div style={{ color: actual.color }}>
          This cycle: {actual.value.toFixed(1)} {unit}
        </div>
      )}
      {reference && (
        <div style={{ color: REFERENCE_COLOR }}>
          Normal average: {reference.value.toFixed(1)} {unit}
        </div>
      )}
    </div>
  );
}

function formatStart(segment) {
  if (segment.start_time) return segment.start_time;
  return new Date(segment.start_ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function SignalPanel({ segment, reference, dataKey, label, unit }) {
  const color = STATUS_COLOR[segment.prediction] ?? REFERENCE_COLOR;

  // Both series share the same 0-100% grid so the shaded band between them
  // (the "gap" from the reference) lines up point-for-point.
  const actualGrid = resampleCycleSignal(segment.rawSeries);
  const data = actualGrid.map((p, i) => {
    const actual = p[dataKey];
    const ref = reference[i]?.[dataKey] ?? actual;
    return {
      pct: p.pct,
      actual,
      reference: ref,
      lower: Math.min(actual, ref),
      gap: Math.abs(actual - ref),
    };
  });

  return (
    <div>
      <div className="text-xs text-ink-secondary mb-1">
        {label}
        {unit && <span className="text-ink-muted"> ({unit})</span>}
      </div>
      <ResponsiveContainer width="100%" height={140}>
        <ComposedChart data={data} margin={{ top: 4, right: 12, left: -12, bottom: 0 }}>
          <CartesianGrid stroke={GRID} vertical={false} />
          <XAxis
            dataKey="pct"
            type="number"
            domain={[0, 100]}
            tickFormatter={(v) => `${v.toFixed(0)}%`}
            tick={{ fill: AXIS, fontSize: 10 }}
            axisLine={{ stroke: GRID }}
            tickLine={false}
          />
          <YAxis tick={{ fill: AXIS, fontSize: 10 }} axisLine={{ stroke: GRID }} tickLine={false} width={44} domain={["auto", "auto"]} />
          <Tooltip content={<ChartTooltip unit={unit} />} cursor={{ stroke: AXIS, strokeWidth: 1 }} />

          {/* Stacked-area trick to shade the band between "actual" and
              "reference": an invisible base up to the lower of the two,
              then a colored fill for just the gap on top of it. */}
          <Area dataKey="lower" stackId="band" stroke="none" fill="transparent" isAnimationActive={false} />
          <Area dataKey="gap" stackId="band" stroke="none" fill={color} fillOpacity={0.18} isAnimationActive={false} />

          <Line type="monotone" dataKey="reference" stroke={REFERENCE_COLOR} strokeWidth={1.5} strokeDasharray="4 3" dot={false} isAnimationActive={false} />
          <Line type="monotone" dataKey="actual" stroke={color} strokeWidth={1.5} dot={false} isAnimationActive={false} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

/**
 * Click-to-inspect detail for one selected cycle from SegmentTimeline: 3
 * stacked charts (motor current, voltage, back-EMF) against % of the cycle
 * elapsed, each showing the selected cycle's real signal against a
 * reference "what a normal cycle typically looks like" curve averaged from
 * every currently-loaded Normal cycle. The gap between them is shaded red
 * when this cycle is Abnormal resistance, green when Normal — the same
 * color the line itself uses — so a wide colored band is a visual
 * explanation of *why* the model called it abnormal, not just a label.
 */
export default function CycleSignalDetail({ segment, normalAverage }) {
  if (!segment) return null;
  const color = STATUS_COLOR[segment.prediction] ?? REFERENCE_COLOR;
  const reference = normalAverage?.[segment.operation] ?? [];

  return (
    <Card>
      <div className="flex flex-wrap items-center justify-between gap-2 mb-1">
        <div className="text-sm font-medium text-ink-primary">{segment.operation} cycle — {formatStart(segment)}</div>
        <div className="text-xs" style={{ color }}>
          {segment.prediction} ({(segment.probAbnormal * 100).toFixed(0)}% abnormal)
        </div>
      </div>

      {!reference.length ? (
        <p className="text-xs text-ink-muted py-8 text-center">
          Not enough Normal {segment.operation} cycles loaded yet to build a comparison baseline.
        </p>
      ) : (
        <>
          <div className="flex items-center gap-3 text-[11px] text-ink-muted mb-3">
            <span className="inline-flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full" style={{ background: color }} />
              This cycle
            </span>
            <span className="inline-flex items-center gap-1.5">
              <span className="w-3 h-0 border-t border-dashed" style={{ borderColor: REFERENCE_COLOR }} />
              Normal average (all loaded Normal {segment.operation} cycles)
            </span>
            <span>Shaded gap = deviation from normal, colored by this cycle's prediction</span>
          </div>
          <div className="space-y-4">
            <SignalPanel segment={segment} reference={reference} dataKey="current" label="Motor current" unit="mA" />
            <SignalPanel segment={segment} reference={reference} dataKey="voltage" label="Motor voltage" unit="×10mV" />
            <SignalPanel segment={segment} reference={reference} dataKey="bemf" label="Motor back electromotive force" unit="" />
          </div>
        </>
      )}
    </Card>
  );
}
