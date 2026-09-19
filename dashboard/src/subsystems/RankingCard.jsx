import { FLEET_RANKINGS } from "../lib/fleetRankings.js";
import { TRAIN_IDS } from "../lib/trainIds.js";

export const SUBSYSTEMS = [
  {
    id: "door",
    title: "Door",
    blurb: "Observes open and close door cycles and flags each cycle as Normal or Abnormal resistance",
  },
  {
    id: "acv",
    title: "ACV",
    blurb: "Observes the faulty car amongst the 8 cars in each train.",
  },
  {
    id: "shm",
    title: "SHM",
    blurb: "Observes stress readings over time and estimates amount of fatigue damage.",
  },
  {
    id: "rail",
    title: "Rail Corrugation",
    blurb:
      "Observes the vibration and shock readings and classifies each rail segment as Normal, having Side I, or Side II corrugation.",
  },
];

function damageTone(damage) {
  if (damage >= 0.6) return "text-status-critical";
  if (damage >= 0.3) return "text-status-serious";
  return "text-status-good";
}

// The live row's value and colour for each subsystem, from its summary.
function liveValue(id, summary) {
  switch (id) {
    case "door": {
      const close = summary.abnormalByOperation?.Close ?? 0;
      const open = summary.abnormalByOperation?.Open ?? 0;
      return {
        text: `${close + open} (${close} Close | ${open} Open)`,
        tone: close + open > 0 ? "text-status-critical" : "text-status-good",
      };
    }
    case "acv":
      return {
        text: summary.stats?.find((s) => s.label === "Most likely faulty")?.value ?? "—",
        tone: "text-status-critical",
      };
    case "shm": {
      const damage = summary.damage;
      return damage != null ? { text: damage.toFixed(12), tone: damageTone(damage) } : { text: "—", tone: "" };
    }
    case "rail": {
      const sideI = summary.abnormalBySide?.["Side I"] ?? 0;
      const sideII = summary.abnormalBySide?.["Side II"] ?? 0;
      return {
        text: `${sideI + sideII} (${sideI} Side I | ${sideII} Side II)`,
        tone: sideI + sideII > 0 ? "text-status-critical" : "text-status-good",
      };
    }
    default:
      return { text: "—", tone: "" };
  }
}

// A ranking of trains. Only the row for the loaded train (TRAIN_IDS[id]) is
// live and opens its dataset — and only when that row itself is clicked; every
// other row is hardcoded and does nothing.
function RankingTable({ id, summary, rows, onOpen }) {
  const live = liveValue(id, summary);

  return (
    <div className="mt-4 pt-4 border-t border-line-hairline">
      <table className="w-full text-base">
        <thead>
          <tr className="text-ink-muted text-sm">
            <th className="text-left font-normal pb-2 px-3">Rank</th>
            <th className="text-left font-normal pb-2 px-3">Train ID</th>
            <th className="text-right font-normal pb-2 px-3">{FLEET_RANKINGS[id].valueHeader}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) =>
            row.id === TRAIN_IDS[id] ? (
              <tr
                key={row.id}
                role="button"
                tabIndex={0}
                onClick={onOpen}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    onOpen();
                  }
                }}
                title="Open this dataset"
                className="text-ink-primary cursor-pointer transition-colors hover:bg-black/[0.07] focus-visible:bg-black/[0.07] outline-none"
              >
                <td className="py-2.5 px-3">{i + 1}</td>
                <td className="py-2.5 px-3 tabular-nums">{row.id}</td>
                <td className={`py-2.5 px-3 text-right font-semibold tabular-nums ${live.tone}`}>{live.text}</td>
              </tr>
            ) : (
              <tr key={row.id} className="text-ink-muted/50 cursor-not-allowed">
                <td className="py-2.5 px-3">{i + 1}</td>
                <td className="py-2.5 px-3 tabular-nums">{row.id}</td>
                <td className="py-2.5 px-3 text-right tabular-nums">{row.value}</td>
              </tr>
            ),
          )}
        </tbody>
      </table>
    </div>
  );
}

// One subsystem's card: title, description, and its train ranking. Used for
// the Overview grid and for each top-nav popup, so they look identical.
export default function RankingCard({ id, summary, rows, onOpen }) {
  const info = SUBSYSTEMS.find((s) => s.id === id);

  return (
    <div className="rounded-lg border border-line-border bg-surface-card overflow-hidden px-5 py-4">
      <div className="text-xl font-bold text-ink-primary">{info.title}</div>
      <p className="text-base text-ink-muted mt-2">{info.blurb}</p>

      {summary ? (
        <RankingTable id={id} summary={summary} rows={rows} onOpen={onOpen} />
      ) : (
        <div className="flex items-center gap-2 text-sm text-ink-muted mt-4 pt-4 border-t border-line-hairline">
          <svg className="animate-spin shrink-0" width="12" height="12" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="3" opacity="0.25" />
            <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
          </svg>
          Running the model — larger files take longer
        </div>
      )}
    </div>
  );
}
