const NAV_ITEMS = [
  { id: "home", label: "Overview" },
  { id: "door", label: "Door" },
  { id: "acv", label: "ACV" },
  { id: "rail", label: "Rail Corrugation" },
  { id: "shm", label: "SHM" },
];

// "home" is the permanent backdrop, not a closable window, so it's always
// shown active; every other item highlights only while its window is open.
export function Sidebar({ openIds, onNavigate }) {
  return (
    <aside className="hidden md:flex w-60 shrink-0 flex-col border-r border-line-hairline bg-surface-card">
      <div className="px-5 py-5 border-b border-line-hairline">
        <div className="text-sm font-semibold tracking-wide text-ink-primary">
          Train Condition Monitoring
        </div>
        <div className="text-xs text-ink-muted mt-0.5">PS3 — 4 subsystems</div>
      </div>

      <nav className="flex-1 px-2 py-3 space-y-1">
        {NAV_ITEMS.map((item) => {
          const isOn = item.id === "home" || openIds.has(item.id);
          return (
            <button
              key={item.id}
              onClick={() => onNavigate(item.id)}
              className={`w-full rounded-md px-3 py-2 text-sm text-left transition-colors ${
                isOn
                  ? "bg-series-blue/15 text-ink-primary"
                  : "text-ink-secondary hover:bg-surface-raised hover:text-ink-primary"
              }`}
            >
              {item.label}
            </button>
          );
        })}
      </nav>

      <div className="px-4 py-3 border-t border-line-hairline text-[11px] text-ink-muted">
        Click to open a window. Drag the header to move it, the bottom-right corner to resize, or double-click the
        header to maximize.
      </div>
    </aside>
  );
}

export function MobileNav({ openIds, onNavigate }) {
  return (
    <nav className="md:hidden flex overflow-x-auto border-t border-line-hairline bg-surface-card shrink-0">
      {NAV_ITEMS.map((item) => {
        const isOn = item.id === "home" || openIds.has(item.id);
        return (
          <button
            key={item.id}
            onClick={() => onNavigate(item.id)}
            className={`flex-1 min-w-[72px] py-2.5 text-[11px] transition-colors ${
              isOn ? "text-series-blue" : "text-ink-muted"
            }`}
          >
            {item.label}
          </button>
        );
      })}
    </nav>
  );
}

export function TopBar({ title, subtitle }) {
  return (
    <header className="flex items-center justify-between border-b border-line-hairline bg-surface-page px-6 py-4">
      <div>
        <h1 className="text-lg font-semibold text-ink-primary">{title}</h1>
        {subtitle && <p className="text-sm text-ink-muted mt-0.5">{subtitle}</p>}
      </div>
    </header>
  );
}
