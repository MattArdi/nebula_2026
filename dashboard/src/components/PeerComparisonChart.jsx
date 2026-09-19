import { ResponsiveContainer, ComposedChart, Area, Line, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";
import { Card } from "./ui.jsx";

const CAR_COLOR = "#3987e5";
const PEER_COLOR = "#908e87";
const DEVIATION_COLOR = "#e66767";
const GRID = "#e3dfd3";
const AXIS = "#908e87";

function ChartTooltip({ active, payload, label, unit }) {
  if (!active || !payload?.length) return null;
  const car = payload.find((p) => p.dataKey === "car");
  const peers = payload.find((p) => p.dataKey === "peers");
  if (!car || !peers) return null;
  return (
    <div className="rounded-md border border-line-border bg-surface-raised px-3 py-2 text-xs shadow-lg">
      <div className="text-ink-muted mb-1">{label.toFixed(0)}% elapsed</div>
      <div style={{ color: CAR_COLOR }}>
        This car: {car.value.toFixed(1)} {unit}
      </div>
      <div style={{ color: PEER_COLOR }}>
        Peer median: {peers.value.toFixed(1)} {unit}
      </div>
      <div className="mt-1 text-ink-muted">Gap: {Math.abs(car.value - peers.value).toFixed(1)} {unit}</div>
    </div>
  );
}

/**
 * A car's own value vs the median of its peers at the same moments, with
 * the gap between them shaded — unlike IndoorOutdoorChart, there's no
 * physically "good" direction here (a controller asking for either a
 * higher or lower setpoint than peers is equally anomalous), so the band
 * is one consistent color regardless of sign, sized by how far this car
 * has drifted from the pack.
 */
export default function PeerComparisonChart({ title, subtitle, data, unit, caveat, carOptions, selectedCarId, onSelectCar }) {
  if (!data?.length && !carOptions?.length) return null;

  const chartData = (data ?? []).map((d) => ({
    pct: d.pct,
    car: d.car,
    peers: d.peers,
    lower: Math.min(d.car, d.peers),
    gap: Math.abs(d.car - d.peers),
  }));

  return (
    <Card>
      <div className="flex flex-wrap items-center justify-between gap-2 mb-1">
        <div className="flex items-center gap-2">
          <div className="text-sm font-medium text-ink-primary">{title}</div>
          {carOptions?.length > 1 && (
            <select
              value={selectedCarId}
              onChange={(e) => onSelectCar?.(e.target.value)}
              className="text-xs bg-surface-raised border border-line-border rounded-md px-1.5 py-0.5 text-ink-primary"
            >
              {carOptions.map((id) => (
                <option key={id} value={id}>
                  Car {id}
                </option>
              ))}
            </select>
          )}
        </div>
        <div className="flex items-center gap-3 text-[11px] text-ink-muted">
          <span className="inline-flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full" style={{ background: CAR_COLOR }} />
            This car
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="w-3 h-0 border-t border-dashed" style={{ borderColor: PEER_COLOR }} />
            Peer median (other 7 cars)
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-sm" style={{ background: DEVIATION_COLOR, opacity: 0.5 }} />
            Gap from peers
          </span>
        </div>
      </div>
      {subtitle && <p className="text-xs text-ink-muted mb-3">{subtitle}</p>}

      {!chartData.length ? (
        <p className="text-xs text-ink-muted py-8 text-center">No usable readings for this car/parameter.</p>
      ) : (
        <ResponsiveContainer width="100%" height={220}>
          <ComposedChart data={chartData} margin={{ top: 8, right: 16, left: -8, bottom: 0 }}>
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
            <Tooltip content={<ChartTooltip unit={unit} />} cursor={{ stroke: AXIS, strokeWidth: 1 }} />

            <Area dataKey="lower" stackId="dev" stroke="none" fill="transparent" isAnimationActive={false} />
            <Area dataKey="gap" stackId="dev" stroke="none" fill={DEVIATION_COLOR} fillOpacity={0.2} isAnimationActive={false} />

            <Line type="monotone" dataKey="peers" stroke={PEER_COLOR} strokeWidth={1.5} strokeDasharray="4 3" dot={false} isAnimationActive={false} />
            <Line type="monotone" dataKey="car" stroke={CAR_COLOR} strokeWidth={2} dot={false} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      )}

      {caveat && <p className="text-[11px] text-ink-muted mt-2 pt-2 border-t border-line-hairline">{caveat}</p>}
    </Card>
  );
}
