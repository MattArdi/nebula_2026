import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ReferenceLine } from "recharts";
import { Card } from "./ui.jsx";
import ScrollableChart from "./ScrollableChart.jsx";
import { usePx } from "../lib/useRem.js";

const COLOR = "#3987e5";
const THRESHOLD_COLOR = "#e66767";
const GRID = "#e3dfd3";
const AXIS = "#908e87";

// Miner's rule: cumulative damage of 1.0 is fatigue failure.
const FAILURE_DAMAGE = 1;

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-md border border-line-border bg-surface-raised px-3 py-2 text-sm shadow-lg">
      <div className="text-ink-muted mb-1">{Number(label).toFixed(0)}% of cycle</div>
      <div style={{ color: COLOR }}>Damage: {payload[0].value.toFixed(3)}</div>
    </div>
  );
}

/**
 * Real cumulative Miner's-rule damage, recomputed on successively longer
 * prefixes of THIS file's own recording — a genuine within-file
 * accumulation (rainflow counting re-run on more data as it "arrives"),
 * not an extrapolation or forecast beyond what was actually recorded.
 *
 * `data` is [{ pct, damage }], pct being 0-100.
 */
export default function DamageProgressChart({ title, subtitle, data }) {
  const px = usePx();
  if (!data?.length) return null;

  return (
    <Card>
      <div className="flex flex-wrap items-start justify-between gap-x-4 gap-y-1">
        <div>
          <div className="text-base font-medium text-ink-primary">{title}</div>
          {subtitle && <p className="text-sm text-ink-muted mt-0.5">{subtitle}</p>}
        </div>
        <span className="inline-flex items-center gap-1.5 text-sm text-ink-muted">
          <span className="w-4 h-0 border-t-2 border-dashed" style={{ borderColor: THRESHOLD_COLOR }} />
          Maximum Damage Possible (Fatigue Failure)
        </span>
      </div>

      <div className="mt-3">
        <ScrollableChart minWidth={px(620)}>
<ResponsiveContainer width="100%" height={px(290)}>
          <LineChart data={data} margin={{ top: 8, right: 16, left: 4, bottom: 0 }}>
            <CartesianGrid stroke={GRID} vertical={false} />
            <XAxis
              dataKey="pct"
              type="number"
              domain={[0, 100]}
              tickFormatter={(v) => `${v.toFixed(0)}%`}
              tick={{ fill: AXIS, fontSize: px(13) }}
              axisLine={{ stroke: GRID }}
              tickLine={false}
              height={px(52)}
              label={{ value: "Percentage of Cycle (%)", position: "insideBottom", offset: 2, fill: AXIS, fontSize: px(13) }}
            />
            <YAxis
              tick={{ fill: AXIS, fontSize: px(13) }}
              axisLine={{ stroke: GRID }}
              tickLine={false}
              width={px(58)}
              domain={[0, FAILURE_DAMAGE]}
              label={{ value: "Damage", angle: -90, position: "insideLeft", offset: 8, style: { textAnchor: "middle", fill: AXIS, fontSize: px(13) } }}
            />
            <Tooltip content={<ChartTooltip />} cursor={{ stroke: AXIS, strokeWidth: 1 }} />
            <ReferenceLine y={FAILURE_DAMAGE} stroke={THRESHOLD_COLOR} strokeDasharray="4 3" />
            <Line type="monotone" dataKey="damage" stroke={COLOR} strokeWidth={2} dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
</ScrollableChart>
      </div>
    </Card>
  );
}
