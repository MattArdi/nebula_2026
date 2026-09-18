import { Card, LabelBadge } from "../components/ui.jsx";
import { EntityList } from "../components/EntityBreakdown.jsx";

const SUBSYSTEMS = [
  {
    id: "door",
    title: "Door",
    task: "Temporal segmentation + binary classification",
    blurb:
      "Finds every door-open/close cycle in a continuous motor-current stream, then flags each as Normal or Abnormal resistance.",
  },
  {
    id: "acv",
    title: "ACV",
    task: "Fault localisation / ranking",
    blurb:
      "Ranks all 8 cars in a train by how likely each is to have a refrigerant leak, from cabin/ambient telemetry.",
  },
  {
    id: "rail",
    title: "Rail Corrugation",
    task: "3-class classification",
    blurb:
      "Classifies a 1-second axle-box vibration/shock recording as Normal, Side I, or Side II corrugation.",
  },
  {
    id: "shm",
    title: "SHM",
    task: "Regression",
    blurb: "Estimates cumulative fatigue damage from a dynamic-stress time series.",
  },
];

// A train that isn't the one bundled dataset has no data behind it — these
// rows exist purely to show what the ranking would look like with more
// trains, and are deliberately inert (stopPropagation so clicking one
// doesn't navigate into a page with nothing to show).
const PLACEHOLDER_TRAIN_IDS = ["6868", "6969"];

function DoorTrainRanking({ stats, accuracy }) {
  const abnormalCount = stats?.find((s) => s.label === "Abnormal resistance")?.value ?? 0;

  return (
    <div className="mt-3 pt-3 border-t border-line-hairline">
      <table className="w-full text-[11px]">
        <thead>
          <tr className="text-ink-muted">
            <th className="text-left font-normal pb-1">Rank</th>
            <th className="text-left font-normal pb-1">Train ID</th>
            <th className="text-right font-normal pb-1">Predicted abnormalities</th>
          </tr>
        </thead>
        <tbody>
          <tr className="text-ink-primary">
            <td className="py-1">1</td>
            <td className="py-1 tabular-nums">6767</td>
            <td
              className={`py-1 text-right font-semibold tabular-nums ${
                abnormalCount > 0 ? "text-status-critical" : "text-status-good"
              }`}
            >
              {abnormalCount}
              {accuracy && (
                <span className="ml-1.5 font-normal text-ink-muted">
                  ({accuracy.correct}/{accuracy.total} match ground truth)
                </span>
              )}
            </td>
          </tr>
          {PLACEHOLDER_TRAIN_IDS.map((id, i) => (
            <tr
              key={id}
              className="text-ink-muted/50 cursor-not-allowed"
              onClick={(e) => e.stopPropagation()}
              title="No dataset provided for this train yet"
            >
              <td className="py-1">{i + 2}</td>
              <td className="py-1 tabular-nums">{id}</td>
              <td className="py-1 text-right">— no dataset</td>
            </tr>
          ))}
          <tr className="text-ink-muted/50" onClick={(e) => e.stopPropagation()}>
            <td className="py-1" colSpan={3}>
              ⋯
            </td>
          </tr>
        </tbody>
      </table>
      <div className="text-[10px] text-ink-muted mt-1.5">
        Only train 6767 has a real dataset loaded. Other train IDs are placeholders.
      </div>
    </div>
  );
}

// ACV's xlsx carries a real "Train number" column (unlike Door's plain CSV,
// which has no train identifier at all) — so this one uses the file's own
// ID rather than an invented one, same placeholder-row treatment as Door
// otherwise.
const ACV_PLACEHOLDER_TRAIN_IDS = ["0621", "0622"];

function AcvTrainRanking({ stats, trainId, groundTruthCarId }) {
  const faultyCar = stats?.find((s) => s.label === "Most likely faulty")?.value ?? "—";
  const modelCarId = faultyCar.replace("Car ", "");
  const matches = groundTruthCarId != null && modelCarId === groundTruthCarId;

  return (
    <div className="mt-3 pt-3 border-t border-line-hairline">
      <table className="w-full text-[11px]">
        <thead>
          <tr className="text-ink-muted">
            <th className="text-left font-normal pb-1">Rank</th>
            <th className="text-left font-normal pb-1">Train ID</th>
            <th className="text-right font-normal pb-1">
              {groundTruthCarId ? "Faulty car (confirmed)" : "Most likely faulty car"}
            </th>
          </tr>
        </thead>
        <tbody>
          <tr className="text-ink-primary">
            <td className="py-1">1</td>
            <td className="py-1 tabular-nums">{trainId ?? "—"}</td>
            <td className="py-1 text-right font-semibold text-status-critical">
              {groundTruthCarId ? `Car ${groundTruthCarId}` : faultyCar}
              {groundTruthCarId && (
                <span className={`ml-1 font-normal ${matches ? "text-status-good" : "text-ink-muted"}`}>
                  ({matches ? "model agrees" : `model said Car ${modelCarId}`})
                </span>
              )}
            </td>
          </tr>
          {ACV_PLACEHOLDER_TRAIN_IDS.map((id, i) => (
            <tr
              key={id}
              className="text-ink-muted/50 cursor-not-allowed"
              onClick={(e) => e.stopPropagation()}
              title="No dataset provided for this train yet"
            >
              <td className="py-1">{i + 2}</td>
              <td className="py-1 tabular-nums">{id}</td>
              <td className="py-1 text-right">— no dataset</td>
            </tr>
          ))}
          <tr className="text-ink-muted/50" onClick={(e) => e.stopPropagation()}>
            <td className="py-1" colSpan={3}>
              ⋯
            </td>
          </tr>
        </tbody>
      </table>
      <div className="text-[10px] text-ink-muted mt-1.5">
        {groundTruthCarId
          ? "Faulty car shown is the published ground truth from Train_Labels.csv."
          : `Only train ${trainId ?? "—"} has a real dataset loaded.`}{" "}
        Other train IDs are placeholders.
      </div>
    </div>
  );
}

// One compact preview row per subsystem's actual prediction schema — same
// columns as its downloadable CSV in 04_Example_Submission, just truncated
// to a handful of rows so the Overview box stays scannable.
function PredictionPreview({ id, summary }) {
  const { predictions, predictionCount, stats, trainId, groundTruthCarId, accuracy } = summary;

  if (id === "door") return <DoorTrainRanking stats={stats} accuracy={accuracy} />;
  if (id === "acv") return <AcvTrainRanking stats={stats} trainId={trainId} groundTruthCarId={groundTruthCarId} />;

  if (!predictions?.length) return null;
  const hiddenCount = (predictionCount ?? predictions.length) - predictions.length;

  // rail / shm — both file_id + prediction
  return (
    <div className="mt-3 pt-3 border-t border-line-hairline space-y-1">
      {predictions.map((p, i) => (
        <div key={i} className="flex items-center justify-between gap-2 text-[11px]">
          <span className="text-ink-muted truncate">{p.file_id}</span>
          {typeof p.prediction === "string" && Number.isNaN(Number(p.prediction)) ? (
            <LabelBadge label={p.prediction} />
          ) : (
            <span className="text-ink-secondary tabular-nums">{p.prediction}</span>
          )}
        </div>
      ))}
      {hiddenCount > 0 && <div className="text-[11px] text-ink-muted pt-0.5">+{hiddenCount} more files</div>}
    </div>
  );
}

export default function Home({ onNavigate, summaries }) {
  return (
    <div className="space-y-5">
      <div className="grid sm:grid-cols-2 gap-3">
        {SUBSYSTEMS.map((s) => {
          const summary = summaries?.[s.id];
          return (
            <div
              key={s.id}
              className="rounded-lg border border-line-border bg-surface-card overflow-hidden hover:border-series-blue/60 transition-colors"
            >
              <button onClick={() => onNavigate(s.id)} className="w-full text-left px-4 py-3.5">
                <div className="text-sm font-semibold text-ink-primary">{s.title}</div>
                <div className="text-xs text-series-blue mt-0.5">{s.task}</div>
                <p className="text-xs text-ink-muted mt-1.5">{s.blurb}</p>

                {summary ? (
                  <>
                    <div className="grid grid-cols-2 gap-2 mt-3 pt-3 border-t border-line-hairline">
                      {summary.stats.map((stat) => (
                        <div key={stat.label}>
                          <div className="text-[11px] text-ink-muted">{stat.label}</div>
                          <div
                            className={`text-base font-semibold tabular-nums ${
                              stat.tone === "critical"
                                ? "text-status-critical"
                                : stat.tone === "good"
                                  ? "text-status-good"
                                  : "text-ink-primary"
                            }`}
                          >
                            {stat.value}
                          </div>
                        </div>
                      ))}
                    </div>
                    <PredictionPreview id={s.id} summary={summary} />
                  </>
                ) : (
                  <div className="text-xs text-ink-muted mt-3 pt-3 border-t border-line-hairline">
                    Loading sample data…
                  </div>
                )}
              </button>

              <EntityList entities={summary?.entities} />
            </div>
          );
        })}
      </div>

      <Card>
        <div className="text-xs font-medium text-ink-muted uppercase tracking-wide mb-1.5">
          Why there's no single "train status"
        </div>
        <p className="text-xs text-ink-muted">
          The four datasets don't share a train or car identifier, so each box above reports its own subsystem's
          predictions independently rather than a single combined status.
        </p>
      </Card>
    </div>
  );
}
