// Bundled real PS3 Train (labelled) files, auto-loaded so every dashboard
// opens already populated AND already checkable against a published
// correct answer — no upload required, and no unverifiable guesswork.
// Standardized to exactly ONE bundled file per subsystem (the first Train
// file, i.e. "Train 1") to keep the repo's storage and initial-load size
// small — drop any other Train/Test file(s) in via each page's file drop to
// load more, which now adds to what's loaded rather than replacing it.
// None of these default to the unpublished Test set (its answers are held
// back by the organisers to grade submissions after the hackathon).

export const DOOR_SAMPLE = { url: "/sample-data/door/Train.csv", name: "Train.csv" };
export const DOOR_ANSWERS_URL = "/sample-data/door/Train_Segments_Answer.csv";

export const ACV_SAMPLE = { url: "/sample-data/acv/acv_case_01.xlsx", name: "acv_case_01.xlsx" };

// Rail's sample is a whole recording (one file per second, each ~17 MB), too
// big to bundle — so this is the model's own predictions over the training
// files that were available, one row per second, in the same file_id,prediction
// format as a run's download.
export const RAIL_SAMPLE_PREDICTIONS = { url: "/sample-data/rail/Train_Predictions.csv", name: "Train_Predictions.csv" };

export const SHM_SAMPLES = [{ url: "/sample-data/shm/train01.csv", name: "train01.csv" }];
