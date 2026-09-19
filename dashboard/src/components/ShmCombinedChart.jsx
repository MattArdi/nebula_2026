import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";
import { Card } from "./ui.jsx";

const PRED_COLOR = "#3987e5";
const TRUE_COLOR = "#898781";
const GRID = "#2c2c2a";
const AXIS = "#898781";

function ChartTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="rounded-md border border-line-border bg-surface-raised px-3 py-2 text-xs shadow-lg">
      <div className="text-ink-primary font-medium mb-1">{d.file_id}</div>
      <div style={{ color: PRED_COLOR }}>Predicted: {d.predicted.toFixed(4)}</div>
      {d.trueLabel != null && <div style={{ color: TRUE_COLOR }}>True (Train_Labels.csv): {d.trueLabel}</div>}
    </div>
  );
}

/**
 * Every currently-loaded file as one point on a single shared graph —
 * "train01, train02, ... each counted as a point" — predicted damage
 * (solid) plotted alongside the true label from Train_Labels.csv (dashed),
 * so you can see across the whole loaded set at once how closely
 * predictions track ground truth, not just file-by-file in a table. Points
 * are ordered by true damage ascending (when known) so both lines read as
 * a clean trend rather than the arbitrary file-name order the real dataset
 * uses (file numbers are randomly assigned, not sequential by damage).
 */
export default function ShmCombinedChart({ title, subtitle, results, caveat }) {
  if (!results?.length) return null;

  const hasTruth = results.some((r) => r.trueLabel != null);
  const sorted = [...results].sort((a, b) => {
    if (hasTruth) return Number(a.trueLabel ?? Number(a.prediction)) - Number(b.trueLabel ?? Number(b.prediction));
    return Number(a.prediction) - Number(b.prediction);
  });

  const data = sorted.map((r) => ({
    file_id: r.file_id,
    predicted: Number(r.prediction),
    trueLabel: r.trueLabel != null ? Number(r.trueLabel) : null,
  }));

  return (
    <Card>
      <div className="flex flex-wrap items-center justify-between gap-2 mb-1">
        <div className="text-sm font-medium text-ink-primary">{title}</div>
        <div className="flex items-center gap-3 text-[11px] text-ink-muted">
          <span className="inline-flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full" style={{ background: PRED_COLOR }} />
            Predicted
          </span>
          {hasTruth && (
            <span className="inline-flex items-center gap-1.5">
              <span className="w-3 h-0 border-t border-dashed" style={{ borderColor: TRUE_COLOR }} />
              True (Train_Labels.csv)
            </span>
          )}
        </div>
      </div>
      {subtitle && <p className="text-xs text-ink-muted mb-3">{subtitle}</p>}

      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={data} margin={{ top: 8, right: 16, left: -8, bottom: 0 }}>
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
          <YAxis tick={{ fill: AXIS, fontSize: 11 }} axisLine={{ stroke: GRID }} tickLine={false} width={44} domain={["auto", "auto"]} />
          <Tooltip content={<ChartTooltip />} cursor={{ stroke: AXIS, strokeWidth: 1 }} />
          {hasTruth && (
            <Line type="monotone" dataKey="trueLabel" stroke={TRUE_COLOR} strokeWidth={1.5} strokeDasharray="4 3" dot={{ r: 3 }} isAnimationActive={false} connectNulls />
          )}
          <Line type="monotone" dataKey="predicted" stroke={PRED_COLOR} strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>

      {caveat && <p className="text-[11px] text-ink-muted mt-2 pt-2 border-t border-line-hairline">{caveat}</p>}
    </Card>
  );
}
