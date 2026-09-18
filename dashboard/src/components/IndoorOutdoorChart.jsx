import { ResponsiveContainer, ComposedChart, Area, Line, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";
import { Card } from "./ui.jsx";

const INDOOR_COLOR = "#3987e5";
const OUTDOOR_COLOR = "#898781";
const FAULT_COLOR = "#e66767";
const OK_COLOR = "#0ca30c";
const GRID = "#2c2c2a";
const AXIS = "#898781";

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  const indoor = payload.find((p) => p.dataKey === "indoor");
  const outdoor = payload.find((p) => p.dataKey === "outdoor");
  if (!indoor || !outdoor) return null;
  const faulty = indoor.value >= outdoor.value;
  return (
    <div className="rounded-md border border-line-border bg-surface-raised px-3 py-2 text-xs shadow-lg">
      <div className="text-ink-muted mb-1">{label.toFixed(0)}% elapsed</div>
      <div style={{ color: INDOOR_COLOR }}>Indoor: {indoor.value.toFixed(1)}°C</div>
      <div style={{ color: OUTDOOR_COLOR }}>Outdoor: {outdoor.value.toFixed(1)}°C</div>
      <div className="mt-1 font-medium" style={{ color: faulty ? FAULT_COLOR : OK_COLOR }}>
        {faulty ? "Indoor ≥ outdoor — not cooling" : "Indoor < outdoor — cooling normally"}
      </div>
    </div>
  );
}

/**
 * Physical AC-fault indicator: a working AC keeps a car's indoor
 * temperature below outdoor/ambient — if indoor is at or above outdoor,
 * that car isn't actually being cooled, independent of any fitted score.
 * Median-per-time-bucket indoor/outdoor lines (x = % of the file's real
 * time elapsed), with the gap between them shaded per-point: red wherever
 * indoor >= outdoor (fault), green wherever indoor < outdoor (working) —
 * the sign can flip within one file, so the band is colored pointwise via
 * two independently-stacked Areas, not one fixed color for the whole chart.
 */
export default function IndoorOutdoorChart({ title, subtitle, data, caveat, carOptions, selectedCarId, onSelectCar }) {
  // Don't early-return on empty data when a car selector is present — that
  // would make the whole card (selector included) vanish for a car with no
  // usable readings, trapping the user with no way to switch back.
  if (!data?.length && !carOptions?.length) return null;

  const chartData = (data ?? []).map((d) => ({
    pct: d.pct,
    indoor: d.indoor,
    outdoor: d.outdoor,
    lower: Math.min(d.indoor, d.outdoor),
    faultGap: Math.max(0, d.indoor - d.outdoor),
    okGap: Math.max(0, d.outdoor - d.indoor),
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
            <span className="w-2 h-2 rounded-full" style={{ background: INDOOR_COLOR }} />
            Indoor
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="w-3 h-0 border-t border-dashed" style={{ borderColor: OUTDOOR_COLOR }} />
            Outdoor
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-sm" style={{ background: FAULT_COLOR, opacity: 0.5 }} />
            Indoor ≥ outdoor
          </span>
        </div>
      </div>
      {subtitle && <p className="text-xs text-ink-muted mb-3">{subtitle}</p>}

      {!chartData.length ? (
        <p className="text-xs text-ink-muted py-8 text-center">No usable indoor/outdoor readings for this car.</p>
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
          <Tooltip content={<ChartTooltip />} cursor={{ stroke: AXIS, strokeWidth: 1 }} />

          {/* Red band, pointwise: only where indoor > outdoor. */}
          <Area dataKey="outdoor" stackId="fault" stroke="none" fill="transparent" isAnimationActive={false} />
          <Area dataKey="faultGap" stackId="fault" stroke="none" fill={FAULT_COLOR} fillOpacity={0.22} isAnimationActive={false} />

          {/* Green band, pointwise: only where indoor < outdoor. */}
          <Area dataKey="lower" stackId="ok" stroke="none" fill="transparent" isAnimationActive={false} />
          <Area dataKey="okGap" stackId="ok" stroke="none" fill={OK_COLOR} fillOpacity={0.15} isAnimationActive={false} />

          <Line type="monotone" dataKey="outdoor" stroke={OUTDOOR_COLOR} strokeWidth={1.5} strokeDasharray="4 3" dot={false} isAnimationActive={false} />
          <Line type="monotone" dataKey="indoor" stroke={INDOOR_COLOR} strokeWidth={2} dot={false} isAnimationActive={false} />
        </ComposedChart>
      </ResponsiveContainer>
      )}

      {caveat && <p className="text-[11px] text-ink-muted mt-2 pt-2 border-t border-line-hairline">{caveat}</p>}
    </Card>
  );
}
