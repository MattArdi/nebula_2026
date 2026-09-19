import { ResponsiveContainer, ComposedChart, Bar, Cell, XAxis, YAxis, Tooltip, CartesianGrid, ReferenceLine } from "recharts";
import { Card } from "./ui.jsx";

const COLORS = { Normal: "#0ca30c", "Side I": "#e66767", "Side II": "#3987e5" };
const GRID = "#e3dfd3";
const AXIS = "#908e87";

function ChartTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="rounded-md border border-line-border bg-surface-raised px-3 py-2 text-xs shadow-lg">
      <div className="text-ink-primary font-medium mb-1">{d.file_id}</div>
      <div style={{ color: COLORS[d.prediction] }}>Predicted: {d.prediction}</div>
      {d.trueLabel != null && (
        <div className={d.matchesTruth ? "text-status-good" : "text-status-critical"}>
          True: {d.trueLabel}
        </div>
      )}
      <div className="text-ink-muted mt-0.5">Severity score: {d.severityScore.toFixed(3)}</div>
    </div>
  );
}

/**
 * Every currently-loaded file plotted together as one bar each: bar height
 * encodes the predicted class (+1 Side I, 0 Normal, −1 Side II) — the
 * ensemble's own class-probability margin isn't exposed by the submission
 * CSV, so this is categorical rather than a continuous confidence score.
 */
export default function RailSeverityChart({ title, subtitle, results, caveat }) {
  if (!results?.length) return null;

  const data = results.map((r) => ({
    file_id: r.file_id,
    prediction: r.prediction,
    severityScore: r.severityScore,
    trueLabel: r.trueLabel,
    matchesTruth: r.matchesTruth,
  }));

  return (
    <Card>
      <div className="flex flex-wrap items-center justify-between gap-2 mb-1">
        <div className="text-sm font-medium text-ink-primary">{title}</div>
        <div className="flex items-center gap-3 text-[11px] text-ink-muted">
          <span className="inline-flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full" style={{ background: COLORS.Normal }} />
            Normal
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full" style={{ background: COLORS["Side I"] }} />
            Side I (up)
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full" style={{ background: COLORS["Side II"] }} />
            Side II (down)
          </span>
        </div>
      </div>
      {subtitle && <p className="text-xs text-ink-muted mb-3">{subtitle}</p>}

      <ResponsiveContainer width="100%" height={240}>
        <ComposedChart data={data} margin={{ top: 8, right: 16, left: -8, bottom: 0 }}>
          <CartesianGrid stroke={GRID} vertical={false} />
          <XAxis
            dataKey="file_id"
            tick={{ fill: AXIS, fontSize: 10 }}
            axisLine={{ stroke: GRID }}
            tickLine={false}
            interval={0}
            angle={-35}
            textAnchor="end"
            height={50}
          />
          <YAxis
            tick={{ fill: AXIS, fontSize: 11 }}
            axisLine={{ stroke: GRID }}
            tickLine={false}
            width={44}
            domain={[-1, 1]}
            label={{ value: "← Side II   Side I →", angle: -90, position: "insideLeft", fill: AXIS, fontSize: 10 }}
          />
          <Tooltip content={<ChartTooltip />} cursor={{ fill: "rgba(35,35,34,0.04)" }} />
          <ReferenceLine y={0} stroke="#605f5a" strokeWidth={1} />
          <Bar dataKey="severityScore" radius={[3, 3, 3, 3]} isAnimationActive={false}>
            {data.map((d, i) => (
              <Cell key={i} fill={COLORS[d.prediction] ?? "#908e87"} />
            ))}
          </Bar>
        </ComposedChart>
      </ResponsiveContainer>

      {caveat && <p className="text-[11px] text-ink-muted mt-2 pt-2 border-t border-line-hairline">{caveat}</p>}
    </Card>
  );
}
