// Keeps a chart at least `minWidth` wide and lets it scroll sideways when its
// popup is narrower than that, so it never gets squashed to fit.
export default function ScrollableChart({ minWidth = 620, children }) {
  return (
    <div className="overflow-x-auto">
      <div style={{ minWidth }}>{children}</div>
    </div>
  );
}
