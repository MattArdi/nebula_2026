import { FLEET_RANKINGS } from "../lib/fleetRankings.js";
import { TRAIN_IDS } from "../lib/trainIds.js";

// Headings shared by every subsystem popup: the dataset currently shown at
// the top, and the upload area below it.
export function CurrentDatasetHeader({ id }) {
  const trainNumber = TRAIN_IDS[id];
  // Where this train sits in the subsystem's fleet ranking.
  const rank = FLEET_RANKINGS[id].rows.findIndex((row) => row.id === trainNumber) + 1;
  return (
    <div>
      <div className="text-xl font-bold text-ink-primary">Current Dataset</div>
      <div className="text-sm text-ink-secondary mt-0.5">Train Number: {trainNumber}</div>
      <div className="text-sm text-ink-secondary">Rank: {rank}</div>
    </div>
  );
}

// The rule above the heading separates the upload area from the dataset shown
// above it.
export function UploadHeading() {
  return <div className="border-t border-line-border pt-5 text-xl font-bold text-ink-primary">Upload New Datasets</div>;
}
