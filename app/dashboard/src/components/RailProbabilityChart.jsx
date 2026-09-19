import { ResponsiveContainer, ComposedChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";
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
      <div style={{ color: COLORS.Normal }}>Normal: {(d.Normal * 100).toFixed(1)}%</div>
      <div style={{ color: COLORS["Side I"] }}>Side I: {(d["Side I"] * 100).toFixed(1)}%</div>
      <div style={{ color: COLORS["Side II"] }}>Side II: {(d["Side II"] * 100).toFixed(1)}%</div>
      {d.trueLabel != null && (
        <div className={d.matchesTruth ? "text-status-good mt-1" : "text-status-critical mt-1"}>True: {d.trueLabel}</div>
      )}
      {d.worstOffender && (
        <div className="text-ink-muted mt-1 pt-1 border-t border-line-hairline">
          Worst on predicted side: car {d.worstOffender.car}, position {d.worstOffender.position}
        </div>
      )}
    </div>
  );
}

/**
 * The ensemble's real class-probability distribution per file (from its
 * own predict_proba, already fitted -- not a new model), stacked to 100%
 * per bar. Replaces a categorical +1/0/-1 severity encoding with the
 * actual confidence the ensemble has in each class, which the submission
 * CSV alone can't carry (it only has the hard label).
 */
export default function RailProbabilityChart({ title, subtitle, results, caveat }) {
  if (!results?.length) return null;

  const data = results.map((r) => ({
    file_id: r.file_id,
    prediction: r.prediction,
    Normal: r.probabilities?.Normal ?? (r.prediction === "Normal" ? 1 : 0),
    "Side I": r.probabilities?.["Side I"] ?? (r.prediction === "Side I" ? 1 : 0),
    "Side II": r.probabilities?.["Side II"] ?? (r.prediction === "Side II" ? 1 : 0),
    trueLabel: r.trueLabel,
    matchesTruth: r.matchesTruth,
    worstOffender: r.worstOffender?.[r.prediction === "Side I" ? "side1" : "side2"] ?? null,
  }));

  return (
    <Card>
      <div className="flex flex-wrap items-center justify-between gap-2 mb-1">
        <div className="text-sm font-medium text-ink-primary">{title}</div>
        <div className="flex items-center gap-3 text-[11px] text-ink-muted">
          <span className="inline-flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full" style={{ background: COLORS.Normal }} />
            P(Normal)
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full" style={{ background: COLORS["Side I"] }} />
            P(Side I)
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full" style={{ background: COLORS["Side II"] }} />
            P(Side II)
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
            domain={[0, 1]}
            tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
          />
          <Tooltip content={<ChartTooltip />} cursor={{ fill: "rgba(35,35,34,0.04)" }} />
          <Bar dataKey="Normal" stackId="p" fill={COLORS.Normal} isAnimationActive={false} />
          <Bar dataKey="Side I" stackId="p" fill={COLORS["Side I"]} isAnimationActive={false} />
          <Bar dataKey="Side II" stackId="p" fill={COLORS["Side II"]} radius={[3, 3, 0, 0]} isAnimationActive={false} />
        </ComposedChart>
      </ResponsiveContainer>

      {caveat && <p className="text-[11px] text-ink-muted mt-2 pt-2 border-t border-line-hairline">{caveat}</p>}
    </Card>
  );
}
