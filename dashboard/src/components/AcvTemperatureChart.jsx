import { useEffect, useState } from "react";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";
import { Card } from "./ui.jsx";

const FAULTY_COLOR = "#e66767";
const FLEET_COLOR = "#908e87";
// Every non-faulty car gets one of these when ticked. Red is reserved for the
// faulty car, so there is no red, orange or pink here, and no grey (the fleet
// median's colour); the seven are spread across blue, green, gold, purple,
// cyan, brown and near-black so no two look alike.
const OTHER_COLORS = ["#3987e5", "#0ca30c", "#c98500", "#8e44ad", "#17b3c7", "#8b5a2b", "#232322"];
const GRID = "#e3dfd3";
const AXIS = "#908e87";

function ChartTooltip({ active, payload, label, faultyCarId }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-md border border-line-border bg-surface-raised px-3 py-2 text-xs shadow-lg">
      <div className="text-ink-muted mb-1">{Number(label).toFixed(0)}% of cycle</div>
      {payload.map((p) => (
        <div key={p.dataKey} style={{ color: p.color }}>
          {p.dataKey === "fleet" ? "Fleet median" : `Car ${p.dataKey}`}
          {p.dataKey === faultyCarId ? " (faulty)" : ""}: {p.value?.toFixed(1)} °C
        </div>
      ))}
    </div>
  );
}

/**
 * Indoor temperature over the file for the fleet median (grey, dotted) and
 * whichever cars are ticked — by default just the model's faulty car, drawn
 * in red. `indoor` is the pipeline's diagnostics.indoor_temperature.
 */
export default function AcvTemperatureChart({ indoor, faultyCarId }) {
  const [selected, setSelected] = useState(() => new Set([faultyCarId]));

  // A different dataset has its own faulty car and its own cars.
  useEffect(() => {
    setSelected(new Set([faultyCarId]));
  }, [indoor, faultyCarId]);

  if (!indoor?.pct?.length) return null;

  const carIds = Object.keys(indoor.cars).sort();
  const otherIds = carIds.filter((id) => id !== faultyCarId);
  const colorFor = (id) => (id === faultyCarId ? FAULTY_COLOR : OTHER_COLORS[otherIds.indexOf(id) % OTHER_COLORS.length]);

  const points = indoor.pct.map((pct, i) => {
    const row = { pct, fleet: indoor.fleet_median[i] };
    for (const id of carIds) row[id] = indoor.cars[id][i];
    return row;
  });

  function toggle(id) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  // Faulty car first so it draws (and lists in the tooltip) right after the
  // fleet median.
  const drawOrder = [...carIds.filter((id) => id === faultyCarId), ...otherIds].filter((id) => selected.has(id));

  return (
    <Card>
      <div className="text-sm font-medium text-ink-primary">Indoor Temperature of Car</div>
      <p className="text-xs text-ink-muted mt-0.5 mb-3">Car indicated as red is the faulty car.</p>

      <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 mb-3 text-[11px]">
        {carIds.map((id) => {
          const checked = selected.has(id);
          const faulty = id === faultyCarId;
          return (
            <label key={id} className="inline-flex items-center gap-1.5 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={checked}
                onChange={() => toggle(id)}
                className="w-3 h-3"
                style={{ accentColor: colorFor(id) }}
              />
              <span className={faulty ? "text-status-critical font-medium" : checked ? "text-ink-secondary" : "text-ink-muted"}>
                Car {id}
              </span>
            </label>
          );
        })}
        <span className="inline-flex items-center gap-1.5 text-ink-muted">
          <span className="w-4 h-0 border-t-2 border-dotted" style={{ borderColor: FLEET_COLOR }} />
          Fleet Median
        </span>
      </div>

      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={points} margin={{ top: 8, right: 16, left: 4, bottom: 0 }}>
          <CartesianGrid stroke={GRID} vertical={false} />
          <XAxis
            dataKey="pct"
            type="number"
            domain={[0, 100]}
            tickFormatter={(v) => `${v.toFixed(0)}%`}
            tick={{ fill: AXIS, fontSize: 11 }}
            axisLine={{ stroke: GRID }}
            tickLine={false}
            height={44}
            label={{ value: "Percentage of Cycle (%)", position: "insideBottom", offset: 2, fill: AXIS, fontSize: 11 }}
          />
          <YAxis
            tick={{ fill: AXIS, fontSize: 11 }}
            axisLine={{ stroke: GRID }}
            tickLine={false}
            width={56}
            domain={[(min) => Math.floor(min) - 1, (max) => Math.ceil(max) + 1]}
            label={{
              value: "Indoor Car Temperature (°C)",
              angle: -90,
              position: "insideLeft",
              offset: 4,
              style: { textAnchor: "middle", fill: AXIS, fontSize: 11 },
            }}
          />
          <Tooltip content={<ChartTooltip faultyCarId={faultyCarId} />} cursor={{ stroke: AXIS, strokeWidth: 1 }} />

          <Line
            type="monotone"
            dataKey="fleet"
            stroke={FLEET_COLOR}
            strokeWidth={1.5}
            strokeDasharray="2 4"
            strokeLinecap="round"
            dot={false}
            isAnimationActive={false}
            connectNulls
          />
          {drawOrder.map((id) => (
            <Line
              key={id}
              type="monotone"
              dataKey={id}
              stroke={colorFor(id)}
              strokeWidth={id === faultyCarId ? 2.5 : 1.5}
              dot={false}
              isAnimationActive={false}
              connectNulls
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </Card>
  );
}
