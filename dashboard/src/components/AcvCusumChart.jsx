import { ResponsiveContainer, ComposedChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";
import { Card } from "./ui.jsx";

const CAR_COLORS = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"];
const GRID = "#2c2c2a";
const AXIS = "#898781";
const MAX_POINTS = 150;

// Downsamples a trajectory to ~MAX_POINTS by stride -- CUSUM is already a
// decayed running sum (smooth by construction), so a stride sample reads
// the same shape as the full ~9000-point series without rendering all of
// it.
function downsample(arr, maxPoints) {
  if (arr.length <= maxPoints) return arr.map((v, i) => [i, v]);
  const stride = Math.ceil(arr.length / maxPoints);
  const out = [];
  for (let i = 0; i < arr.length; i += stride) out.push([i, arr[i]]);
  const last = arr.length - 1;
  if (out[out.length - 1][0] !== last) out.push([last, arr[last]]);
  return out;
}

function ChartTooltip({ active, payload, label, topCarId }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-md border border-line-border bg-surface-raised px-3 py-2 text-xs shadow-lg max-w-[220px]">
      <div className="text-ink-muted mb-1">{label.toFixed(0)}% elapsed</div>
      {payload.map((p) => (
        <div key={p.dataKey} style={{ color: p.color }}>
          Car {p.dataKey}
          {p.dataKey === topCarId ? " (top suspect)" : ""}: {p.value?.toFixed(2)}
        </div>
      ))}
    </div>
  );
}

/**
 * Every data-bearing car's CUSUM score plotted across the file's real time
 * axis, not just the final ranking -- shows WHEN a car's score started
 * climbing, not only which car ended up on top. Rising from ~0 and staying
 * there is healthy (the decayed CUSUM keeps resetting toward 0 on ordinary
 * jitter, per Section 3 Step 3); a sustained one-directional climb is what
 * the ranking is actually keying on.
 */
export default function AcvCusumChart({ title, subtitle, trajectories, topCarId, visibleCarIds, onToggleCar, caveat }) {
  const carIds = Object.keys(trajectories ?? {}).sort();
  if (!carIds.length) return null;

  const length = trajectories[carIds[0]].length;
  const sampledIdx = downsample(trajectories[carIds[0]], MAX_POINTS).map(([i]) => i);
  const points = sampledIdx.map((i) => {
    const row = { pct: (i / Math.max(1, length - 1)) * 100 };
    for (const id of carIds) row[id] = trajectories[id][i];
    return row;
  });

  return (
    <Card>
      <div className="flex flex-wrap items-center justify-between gap-2 mb-1">
        <div className="text-sm font-medium text-ink-primary">{title}</div>
      </div>
      {subtitle && <p className="text-xs text-ink-muted mb-3">{subtitle}</p>}

      <div className="flex flex-wrap gap-x-4 gap-y-1.5 mb-3 text-[11px]">
        {carIds.map((id, i) => {
          const color = CAR_COLORS[i % CAR_COLORS.length];
          const checked = visibleCarIds.has(id);
          return (
            <label key={id} className="inline-flex items-center gap-1.5 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={checked}
                onChange={() => onToggleCar(id)}
                className="w-3 h-3"
                style={{ accentColor: color }}
              />
              <span className="w-2 h-2 rounded-full shrink-0" style={{ background: color, opacity: checked ? 1 : 0.3 }} />
              <span className={checked ? "text-ink-secondary" : "text-ink-muted"}>
                Car {id}
                {id === topCarId && <span className="text-status-critical"> (top suspect)</span>}
              </span>
            </label>
          );
        })}
      </div>

      <ResponsiveContainer width="100%" height={240}>
        <ComposedChart data={points} margin={{ top: 8, right: 16, left: -8, bottom: 0 }}>
          <CartesianGrid stroke={GRID} vertical={false} />
          <XAxis
            dataKey="pct"
            type="number"
            domain={[0, 100]}
            tickFormatter={(v) => `${v.toFixed(0)}%`}
            tick={{ fill: AXIS, fontSize: 11 }}
            axisLine={{ stroke: GRID }}
            tickLine={false}
          />
          <YAxis tick={{ fill: AXIS, fontSize: 11 }} axisLine={{ stroke: GRID }} tickLine={false} width={44} domain={["auto", "auto"]} />
          <Tooltip content={<ChartTooltip topCarId={topCarId} />} cursor={{ stroke: AXIS, strokeWidth: 1 }} />

          {carIds.map((id, i) => {
            if (!visibleCarIds.has(id)) return null;
            const color = CAR_COLORS[i % CAR_COLORS.length];
            const isTop = id === topCarId;
            return (
              <Line
                key={id}
                type="monotone"
                dataKey={id}
                stroke={color}
                strokeWidth={isTop ? 3 : 1.5}
                dot={false}
                isAnimationActive={false}
                connectNulls
              />
            );
          })}
        </ComposedChart>
      </ResponsiveContainer>

      {caveat && <p className="text-[11px] text-ink-muted mt-2 pt-2 border-t border-line-hairline">{caveat}</p>}
    </Card>
  );
}
