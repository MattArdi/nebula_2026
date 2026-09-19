// Headings shared by every subsystem popup: the dataset currently shown at
// the top, and the upload area below it.
export function CurrentDatasetHeader({ trainNumber }) {
  return (
    <div>
      <div className="text-xl font-bold text-ink-primary">Current Dataset</div>
      <div className="text-sm text-ink-secondary mt-0.5">Train Number: {trainNumber}</div>
    </div>
  );
}

// The rule above the heading separates the upload area from the dataset shown
// above it.
export function UploadHeading() {
  return <div className="border-t border-line-border pt-5 text-xl font-bold text-ink-primary">Upload New Datasets</div>;
}
