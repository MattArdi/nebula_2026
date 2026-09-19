// Bundled real PS3 Train (labelled) files, auto-loaded so every dashboard
// opens already populated AND already checkable against a published
// correct answer — no upload required, and no unverifiable guesswork.
// None of these default to the unpublished Test set anymore (its answers
// are held back by the organisers to grade submissions after the
// hackathon) — upload a Test file manually via each page's file drop
// once you're ready to run the final submission.

export const DOOR_SAMPLE = { url: "/sample-data/door/Train.csv", name: "Train.csv" };
export const DOOR_ANSWERS_URL = "/sample-data/door/Train_Segments_Answer.csv";

export const ACV_SAMPLE = { url: "/sample-data/acv/acv_case_01.xlsx", name: "acv_case_01.xlsx" };
export const ACV_LABELS_URL = "/sample-data/acv/Train_Labels.csv";

// 4 Normal / 4 Side I / 4 Side II — a curated, class-balanced subset of the
// 272 labelled Train files (the real class split is heavily imbalanced,
// ~86% Normal, so an unfiltered sample would show almost no faults).
export const RAIL_SAMPLES = [
  "Train1.csv", "Train3.csv", "Train4.csv", "Train5.csv", // Normal
  "Train62.csv", "Train83.csv", "Train100.csv", "Train106.csv", // Side I
  "Train2.csv", "Train12.csv", "Train26.csv", "Train30.csv", // Side II
].map((name) => ({ url: `/sample-data/rail/${name}`, name }));
export const RAIL_LABELS_URL = "/sample-data/rail/Train_Labels.csv";

// 16 of the 64 labelled Train files, evenly spaced across the real damage
// range (0.03 to 0.79) so the sample spans low/medium/high, not clustered.
export const SHM_SAMPLES = [
  "train13.csv", "train33.csv", "train09.csv", "train50.csv",
  "train62.csv", "train21.csv", "train24.csv", "train59.csv",
  "train01.csv", "train51.csv", "train44.csv", "train45.csv",
  "train36.csv", "train06.csv", "train16.csv", "train23.csv",
].map((name) => ({ url: `/sample-data/shm/${name}`, name }));
export const SHM_LABELS_URL = "/sample-data/shm/Train_Labels.csv";
