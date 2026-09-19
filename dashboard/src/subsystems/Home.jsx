import RankingCard, { SUBSYSTEMS } from "./RankingCard.jsx";
import { TRAIN_IDS } from "../lib/trainIds.js";

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

export default function Home({ onNavigate, summaries }) {
  return (
    <div className="space-y-5">
      <div className="grid sm:grid-cols-2 gap-3">
        {SUBSYSTEMS.map((s) => (
          <RankingCard
            key={s.id}
            id={s.id}
            summary={summaries?.[s.id]}
            rows={[{ id: TRAIN_IDS[s.id] }, ...PLACEHOLDER_ROWS[s.id]]}
            onOpen={() => onNavigate(s.id)}
          />
        ))}
      </div>
    </div>
  );
}
