import { ResponsiveContainer, ComposedChart, Area, Line, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";
import { Card } from "./ui.jsx";
import ScrollableChart from "./ScrollableChart.jsx";
import { resampleCycleSignal } from "../lib/signalResample.js";
import { usePx } from "../lib/useRem.js";

const STATUS_COLOR = { Normal: "#0ca30c", "Abnormal resistance": "#e66767" };
const REFERENCE_COLOR = "#908e87";
const GRID = "#e3dfd3";
const AXIS = "#908e87";

function ChartTooltip({ active, payload, label, unit }) {
  if (!active || !payload?.length) return null;
  const actual = payload.find((p) => p.dataKey === "actual");
  const reference = payload.find((p) => p.dataKey === "reference");
  return (
    <div className="rounded-md border border-line-border bg-surface-raised px-3 py-2 text-sm shadow-lg">
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

// The stream's timestamps are parsed as UTC (see parseDoorTimestamp), so format
// them as UTC too — otherwise the browser's own timezone shifts them.
function formatClock(ms) {
  return new Date(ms).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    fractionalSecondDigits: 3,
    hour12: false,
    timeZone: "UTC",
  });
}

function formatStart(segment) {
  if (segment.start_time) return segment.start_time;
  return new Date(segment.start_ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function SignalPanel({ segment, reference, dataKey, label, unit }) {
  const px = usePx();
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
      <div className="text-sm text-ink-secondary mb-1">
        {label}
        {unit && <span className="text-ink-muted"> ({unit})</span>}
      </div>
      <ResponsiveContainer width="100%" height={px(190)}>
        <ComposedChart data={data} margin={{ top: 4, right: 12, left: -12, bottom: 0 }}>
          <CartesianGrid stroke={GRID} vertical={false} />
          <XAxis
            dataKey="pct"
            type="number"
            domain={[0, 100]}
            tickFormatter={(v) => `${v.toFixed(0)}%`}
            tick={{ fill: AXIS, fontSize: px(13) }}
            axisLine={{ stroke: GRID }}
            tickLine={false}
            height={px(50)}
            label={{ value: "Percentage of Cycle (%)", position: "insideBottom", offset: 2, fill: AXIS, fontSize: px(13) }}
          />
          <YAxis tick={{ fill: AXIS, fontSize: px(13) }} axisLine={{ stroke: GRID }} tickLine={false} width={px(52)} domain={["auto", "auto"]} />
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
  const reference = segment.operation ? normalAverage?.[segment.operation] ?? [] : [];

  if (!segment.rawSeries?.length) {
    return (
      <Card>
        <div className="text-base font-medium text-ink-primary mb-1">
          {segment.prediction} cycle — {formatStart(segment)}
        </div>
        <p className="text-xs text-ink-muted py-8 text-center">
          Raw signal isn't available for this cycle (the client-side cycle slicing didn't line up with the
          backend's segmentation for this file).
        </p>
      </Card>
    );
  }

  return (
    <Card>
      <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
        <div className="text-base font-medium text-ink-primary">{segment.operation} Cycle</div>
        {reference.length > 0 && (
          <div className="flex items-center gap-3 text-sm text-ink-muted">
            <span className="inline-flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full" style={{ background: color }} />
              This cycle
            </span>
            <span className="inline-flex items-center gap-1.5">
              <span className="w-3 h-0 border-t border-dashed" style={{ borderColor: REFERENCE_COLOR }} />
              Normal average
            </span>
          </div>
        )}
      </div>

      <div className="text-sm text-ink-secondary space-y-0.5 mb-3">
        <div>Start Time: {formatClock(segment.start_ts)}</div>
        <div>End Time: {formatClock(segment.end_ts)}</div>
        <div>Total Cycle Time: {((segment.end_ts - segment.start_ts) / 1000).toFixed(2)} s</div>
      </div>

      {!reference.length ? (
        <p className="text-xs text-ink-muted py-8 text-center">
          Not enough Normal {segment.operation} cycles loaded yet to build a comparison baseline.
        </p>
      ) : (
        <ScrollableChart minWidth={560}>
        <div className="space-y-4">
          <SignalPanel segment={segment} reference={reference} dataKey="current" label="Motor current" unit="mA" />
          <SignalPanel segment={segment} reference={reference} dataKey="voltage" label="Motor voltage" unit="×10mV" />
          <SignalPanel segment={segment} reference={reference} dataKey="bemf" label="Motor Electrodynamic Force" unit="" />
        </div>
        </ScrollableChart>
      )}
    </Card>
  );
}
