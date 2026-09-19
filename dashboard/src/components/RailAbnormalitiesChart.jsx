import { ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, ZAxis, Tooltip, CartesianGrid, ReferenceDot } from "recharts";
import { Card } from "./ui.jsx";

// Bottom to top: more severe sits higher.
const LEVELS = { Normal: 0, "Side I": 1, "Side II": 2 };
const COLORS = { Normal: "#0ca30c", "Side I": "#ec835a", "Side II": "#e66767" };
const GRID = "#e3dfd3";
const AXIS = "#908e87";

const CALLOUT_W = 148;
const CALLOUT_H = 26;

function ChartTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const point = payload[0].payload;
  return (
    <div className="rounded-md border border-line-border bg-surface-raised px-3 py-2 text-xs shadow-lg">
      <div className="text-ink-muted mb-1">Second {point.second}</div>
      <div style={{ color: COLORS[point.label] }}>{point.label}</div>
    </div>
  );
}

// The pinned marker for a point picked from the list above: a ring round the
// dot and a callout saying which second and which issue. The callout sits
// below the top row (there's no room above it) and hangs off whichever side
// of the dot has room.
function PinnedPoint({ cx, cy, point }) {
  if (cx == null || cy == null) return null;
  const color = COLORS[point.label];
  const below = point.level === LEVELS["Side II"];
  const y = below ? cy + 12 : cy - 12 - CALLOUT_H;
  const x = point.pct > 50 ? cx - CALLOUT_W + 12 : cx - 12;
  return (
    <g pointerEvents="none">
      <circle cx={cx} cy={cy} r={9} fill="none" stroke={color} strokeWidth={2} />
      <rect x={x} y={y} width={CALLOUT_W} height={CALLOUT_H} rx={5} fill="#ffffff" stroke={color} strokeWidth={1.5} />
      <text x={x + CALLOUT_W / 2} y={y + CALLOUT_H / 2 + 4} textAnchor="middle" fontSize={11} fontWeight={600} fill={color}>
        Second {point.second} — {point.label}
      </text>
    </g>
  );
}

/**
 * One dot per input file — each file is one second of the recording — placed
 * at its position in the recording (x) and on its predicted class's row (y).
 * `rows` is [{ file_id, prediction, second }] in recording order. When
 * `activeIndex` is set, that dot is ringed and labelled with its second and
 * issue.
 */
export default function RailAbnormalitiesChart({ rows, activeIndex = null }) {
  if (!rows?.length) return null;

  // Each file sits in the middle of its own slice of the recording, so a
  // single file lands mid-chart instead of on the axis.
  const points = rows.map((r, i) => ({
    pct: ((i + 0.5) / rows.length) * 100,
    level: LEVELS[r.prediction] ?? 0,
    label: r.prediction,
    second: r.second,
  }));
  const active = activeIndex != null ? points[activeIndex] : null;

  return (
    <Card>
      <div className="text-sm font-medium text-ink-primary mb-3">Abnormalities Detected</div>

      <ResponsiveContainer width="100%" height={240}>
        <ScatterChart margin={{ top: 12, right: 16, left: 4, bottom: 0 }}>
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
            dataKey="level"
            type="number"
            domain={[-0.5, 2.5]}
            ticks={[0, 1, 2]}
            tickFormatter={(v) => Object.keys(LEVELS)[v]}
            tick={{ fill: AXIS, fontSize: 11 }}
            axisLine={{ stroke: GRID }}
            tickLine={false}
            width={56}
          />
          <ZAxis range={[50, 50]} />
          <Tooltip content={<ChartTooltip />} cursor={{ strokeDasharray: "3 3", stroke: AXIS }} />
          {Object.keys(LEVELS).map((label) => (
            <Scatter
              key={label}
              data={points.filter((p) => p.label === label)}
              fill={COLORS[label]}
              fillOpacity={0.85}
              isAnimationActive={false}
            />
          ))}
          {active && (
            <ReferenceDot
              x={active.pct}
              y={active.level}
              ifOverflow="visible"
              shape={({ cx, cy }) => <PinnedPoint cx={cx} cy={cy} point={active} />}
            />
          )}
        </ScatterChart>
      </ResponsiveContainer>
    </Card>
  );
}
