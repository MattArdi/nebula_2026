import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ReferenceLine } from "recharts";
import { Card } from "./ui.jsx";

const COLOR = "#3987e5";
const THRESHOLD_COLOR = "#e66767";
const GRID = "#e3dfd3";
const AXIS = "#908e87";

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-md border border-line-border bg-surface-raised px-3 py-2 text-xs shadow-lg">
      <div className="text-ink-muted mb-1">{(label * 100).toFixed(0)}% of recording</div>
      <div style={{ color: COLOR }}>Cumulative damage: {payload[0].value.toFixed(3)}</div>
    </div>
  );
}

/**
 * Real cumulative Miner's-rule damage, recomputed on successively longer
 * prefixes of THIS file's own recording — a genuine within-file
 * accumulation (rainflow counting re-run on more data as it "arrives"),
 * not an extrapolation or forecast beyond what was actually recorded.
 */
export default function DamageProgressChart({ title, subtitle, data, caveat }) {
  if (!data?.length) return null;

  return (
    <Card>
      <div className="text-sm font-medium text-ink-primary mb-1">{title}</div>
      {subtitle && <p className="text-xs text-ink-muted mb-3">{subtitle}</p>}

      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={data} margin={{ top: 8, right: 16, left: -8, bottom: 0 }}>
          <CartesianGrid stroke={GRID} vertical={false} />
          <XAxis
            dataKey="x"
            type="number"
            domain={[0, 1]}
            tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
            tick={{ fill: AXIS, fontSize: 11 }}
            axisLine={{ stroke: GRID }}
            tickLine={false}
          />
          <YAxis tick={{ fill: AXIS, fontSize: 11 }} axisLine={{ stroke: GRID }} tickLine={false} width={44} domain={["auto", "auto"]} />
          <Tooltip content={<ChartTooltip />} cursor={{ stroke: AXIS, strokeWidth: 1 }} />
          <ReferenceLine y={1.0} stroke={THRESHOLD_COLOR} strokeDasharray="4 3" label={{ value: "D=1.0 (failure)", fill: THRESHOLD_COLOR, fontSize: 10, position: "insideTopRight" }} />
          <Line type="monotone" dataKey="y" stroke={COLOR} strokeWidth={2} dot={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>

      {caveat && <p className="text-[11px] text-ink-muted mt-2 pt-2 border-t border-line-hairline">{caveat}</p>}
    </Card>
  );
}
