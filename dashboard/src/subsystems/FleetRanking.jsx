import RankingCard from "./RankingCard.jsx";
import { FLEET_RANKINGS } from "../lib/fleetRankings.js";

// The popup behind each top navigation button: the same card as on the
// Overview, but listing the whole fleet (up to 10 trains). The loaded train's
// row opens its dataset; the rest are hardcoded and inert.
export default function FleetRanking({ id, summary, onOpen }) {
  return <RankingCard id={id} summary={summary} rows={FLEET_RANKINGS[id].rows} onOpen={onOpen} />;
}
