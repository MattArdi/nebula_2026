import { ResponsiveContainer, ComposedChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend } from "recharts";
import { Card } from "./ui.jsx";

// Fixed categorical order (never cycled/reassigned), matching tailwind
// config's `series` palette exactly — happens to be 8 colors for 8 cars.
const CAR_COLORS = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"];
const MEDIAN_COLOR = "#c3c2b7";
const GRID = "#2c2c2a";
const AXIS = "#898781";

function ChartTooltip({ active, payload, label, faultyCarId }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-md border border-line-border bg-surface-raised px-3 py-2 text-xs shadow-lg max-w-[220px]">
      <div className="text-ink-muted mb-1">{label.toFixed(0)}% elapsed</div>
      {payload.map((p) => (
        <div key={p.dataKey} style={{ color: p.color }}>
          {p.dataKey === "median" ? "Fleet median" : `Car ${p.dataKey}`}
          {p.dataKey === faultyCarId ? " (confirmed faulty)" : ""}: {p.value?.toFixed(1)}°C
        </div>
      ))}
    </div>
  );
}

/**
 * Every car's indoor temperature plotted together against the whole-fleet
 * median (thick dashed reference), with a checkbox per car to toggle which
 * lines are shown — lets you visually compare any subset of cars to the
 * fleet at once instead of one car at a time. The confirmed-faulty car
 * (from Train_Labels.csv, when known) gets a bold stroke so it stands out
 * regardless of which other cars are ticked.
 */
export default function FleetComparisonChart({ title, subtitle, carIds, points, faultyCarId, visibleCarIds, onToggleCar, caveat }) {
  if (!points?.length) return null;

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
                className="w-3 h-3 accent-series-blue"
                style={{ accentColor: color }}
              />
              <span className="w-2 h-2 rounded-full shrink-0" style={{ background: color, opacity: checked ? 1 : 0.3 }} />
              <span className={checked ? "text-ink-secondary" : "text-ink-muted"}>
                Car {id}
                {id === faultyCarId && <span className="text-status-critical"> (confirmed)</span>}
              </span>
            </label>
          );
        })}
      </div>

      <ResponsiveContainer width="100%" height={260}>
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
          <Tooltip content={<ChartTooltip faultyCarId={faultyCarId} />} cursor={{ stroke: AXIS, strokeWidth: 1 }} />

          <Line type="monotone" dataKey="median" name="Fleet median" stroke={MEDIAN_COLOR} strokeWidth={2.5} strokeDasharray="5 3" dot={false} isAnimationActive={false} />

          {carIds.map((id, i) => {
            if (!visibleCarIds.has(id)) return null;
            const color = CAR_COLORS[i % CAR_COLORS.length];
            const isFaulty = id === faultyCarId;
            return (
              <Line
                key={id}
                type="monotone"
                dataKey={(p) => p.values[id]}
                name={`Car ${id}`}
                stroke={color}
                strokeWidth={isFaulty ? 3 : 1.5}
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
