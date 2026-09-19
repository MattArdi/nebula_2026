import { TRAIN_IDS } from "../lib/trainIds.js";

const SUBSYSTEMS = [
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

// Only the first row of each ranking has a real dataset behind it. The rows
// below it are hardcoded and deliberately inert.
const PLACEHOLDER_ROWS = {
  door: [
    { id: "6868", value: "20 (10 Close | 10 Open)" },
    { id: "6969", value: "24 (14 Close | 10 Open)" },
  ],
  acv: [
    { id: "6767", value: "Car 06" },
    { id: "6868", value: "Car 07" },
  ],
  // Kept below the real damage so the table stays sorted highest-damage-first
  // for the bundled sample.
  shm: [
    { id: "7676", value: "0.089234798" },
    { id: "9696", value: "0.061735412" },
  ],
  rail: [
    { id: "6767", value: "9 (2 Side I | 7 Side II)" },
    { id: "9696", value: "7 (1 Side I | 6 Side II)" },
  ],
};

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

const VALUE_HEADERS = {
  door: "Predicted no. of abnormalities",
  acv: "Faulty Car",
  shm: "Current Train Damage",
  rail: "No. of Abnormalities",
};

// The train ranking under a subsystem's description. Only the first row —
// the loaded dataset — opens the popup, and only when that row itself is
// clicked; the hardcoded rows and everything else on the card do nothing.
function RankingTable({ id, summary, onOpen }) {
  const live = liveValue(id, summary);

  return (
    <div className="mt-3 pt-3 border-t border-line-hairline">
      <table className="w-full text-[11px]">
        <thead>
          <tr className="text-ink-muted">
            <th className="text-left font-normal pb-1 px-2">Rank</th>
            <th className="text-left font-normal pb-1 px-2">Train ID</th>
            <th className="text-right font-normal pb-1 px-2">{VALUE_HEADERS[id]}</th>
          </tr>
        </thead>
        <tbody>
          <tr
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
            <td className="py-1.5 px-2">1</td>
            <td className="py-1.5 px-2 tabular-nums">{TRAIN_IDS[id]}</td>
            <td className={`py-1.5 px-2 text-right font-semibold tabular-nums ${live.tone}`}>{live.text}</td>
          </tr>
          {PLACEHOLDER_ROWS[id].map((row, i) => (
            <tr key={row.id} className="text-ink-muted/50 cursor-not-allowed">
              <td className="py-1.5 px-2">{i + 2}</td>
              <td className="py-1.5 px-2 tabular-nums">{row.id}</td>
              <td className="py-1.5 px-2 text-right tabular-nums">{row.value}</td>
            </tr>
          ))}
        </tbody>
      </table>
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
            <div key={s.id} className="rounded-lg border border-line-border bg-surface-card overflow-hidden px-4 py-3.5">
              <div className="text-sm font-semibold text-ink-primary">{s.title}</div>
              <p className="text-xs text-ink-muted mt-1.5">{s.blurb}</p>

              {summary ? (
                <RankingTable id={s.id} summary={summary} onOpen={() => onNavigate(s.id)} />
              ) : (
                <div className="flex items-center gap-2 text-xs text-ink-muted mt-3 pt-3 border-t border-line-hairline">
                  <svg className="animate-spin shrink-0" width="12" height="12" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                    <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="3" opacity="0.25" />
                    <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
                  </svg>
                  Running the model — larger files take longer
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
