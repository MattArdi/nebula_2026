import { ResponsiveContainer, ComposedChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ReferenceLine } from "recharts";
import { Card } from "./ui.jsx";

const STATUS_COLOR = { Normal: "#0ca30c", "Abnormal resistance": "#e66767" };
const THRESHOLD_COLOR = "#908e87";
const LOW_CONFIDENCE_COLOR = "#fab219";
const GRID = "#e3dfd3";
const AXIS = "#908e87";

function Dot({ cx, cy, payload }) {
  if (cx == null || cy == null) return null;
  const color = STATUS_COLOR[payload.prediction] ?? THRESHOLD_COLOR;
  return (
    <g>
      {payload.lowConfidence && <circle cx={cx} cy={cy} r={6} fill="none" stroke={LOW_CONFIDENCE_COLOR} strokeWidth={1.5} />}
      <circle cx={cx} cy={cy} r={3} fill={color} stroke={payload.outOfRange ? "#232322" : "none"} strokeWidth={1} />
    </g>
  );
}

function ChartTooltip({ active, payload, unit }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div className="rounded-md border border-line-border bg-surface-raised px-3 py-2 text-xs shadow-lg">
      <div className="text-ink-muted mb-1">{p.start_time}</div>
      <div style={{ color: STATUS_COLOR[p.prediction] }}>
        {p.prediction}: {p.decisionValue.toFixed(1)} {unit} (threshold {p.thresholdUsed.toFixed(1)})
      </div>
      {p.lowConfidence && <div style={{ color: LOW_CONFIDENCE_COLOR }}>Low confidence — a close call near the threshold</div>}
      {p.outOfRange && <div className="text-status-critical">Out of range — outside anything Train demonstrated</div>}
    </div>
  );
}

function Panel({ title, feature, unit, data }) {
  if (!data.length) return null;
  const chartData = data.map((s, i) => ({ i, ...s }));
  return (
    <div>
      <div className="text-xs text-ink-secondary mb-1">
        {title} <span className="text-ink-muted">({feature}, {unit})</span>
      </div>
      <ResponsiveContainer width="100%" height={140}>
        <ComposedChart data={chartData} margin={{ top: 4, right: 12, left: -12, bottom: 0 }}>
          <CartesianGrid stroke={GRID} vertical={false} />
          <XAxis dataKey="i" tick={{ fill: AXIS, fontSize: 10 }} axisLine={{ stroke: GRID }} tickLine={false} />
          <YAxis tick={{ fill: AXIS, fontSize: 10 }} axisLine={{ stroke: GRID }} tickLine={false} width={44} domain={["auto", "auto"]} />
          <Tooltip content={<ChartTooltip unit={unit} />} cursor={{ stroke: AXIS, strokeWidth: 1 }} />
          <ReferenceLine
            y={chartData[0].thresholdUsed}
            stroke={THRESHOLD_COLOR}
            strokeDasharray="4 3"
            label={{ value: "threshold", fill: THRESHOLD_COLOR, fontSize: 10, position: "insideTopRight" }}
          />
          <Line
            type="monotone"
            dataKey="decisionValue"
            stroke={THRESHOLD_COLOR}
            strokeWidth={1}
            strokeOpacity={0.5}
            dot={<Dot />}
            isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

/**
 * Early-warning view the per-cycle table can't give: every Close/Open
 * cycle's real decision value (cur_max / cur_mean) plotted in order against
 * the threshold it was judged against, so a value creeping toward the line
 * over successive cycles is visible before it actually crosses — not just
 * a binary pass/fail per cycle. Dot color is the backend's own prediction;
 * an amber ring marks a low-confidence call (within the bootstrap
 * threshold uncertainty); a white dot outline marks an out-of-range value
 * (outside anything Train ever demonstrated). Both are v3's own diagnostic
 * output, not derived here.
 */
export default function DoorThresholdChart({ title, subtitle, segments, caveat }) {
  const withDiagnostics = (segments ?? []).filter((s) => s.thresholdUsed != null);
  if (!withDiagnostics.length) return null;

  const closeData = withDiagnostics.filter((s) => s.operation === "Close");
  const openData = withDiagnostics.filter((s) => s.operation === "Open");

  return (
    <Card>
      <div className="flex flex-wrap items-center justify-between gap-2 mb-1">
        <div className="text-sm font-medium text-ink-primary">{title}</div>
        <div className="flex items-center gap-3 text-[11px] text-ink-muted">
          <span className="inline-flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full" style={{ background: STATUS_COLOR.Normal }} />
            Normal
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full" style={{ background: STATUS_COLOR["Abnormal resistance"] }} />
            Abnormal
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full border" style={{ borderColor: LOW_CONFIDENCE_COLOR }} />
            Low confidence
          </span>
        </div>
      </div>
      {subtitle && <p className="text-xs text-ink-muted mb-3">{subtitle}</p>}

      <div className="space-y-4">
        <Panel title="Close cycles" feature="cur_max" unit="mA" data={closeData} />
        <Panel title="Open cycles" feature="cur_mean" unit="mA" data={openData} />
      </div>

      {caveat && <p className="text-[11px] text-ink-muted mt-2 pt-2 border-t border-line-hairline">{caveat}</p>}
    </Card>
  );
}
