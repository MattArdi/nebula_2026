import { useEffect, useRef, useState } from "react";

const NAV_ITEMS = [
  { id: "home", label: "Overview" },
  { id: "door", label: "Door" },
  { id: "acv", label: "ACV" },
  { id: "rail", label: "Rail Corrugation" },
  { id: "shm", label: "SHM" },
];

function BrandMark() {
  return (
    <div className="w-8 h-8 rounded-lg bg-white/15 border border-white/30 flex items-center justify-center shrink-0">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <path d="M4 7h16M4 12h16M4 17h16" stroke="#ffffff" strokeWidth="1.8" strokeLinecap="round" />
        <path d="M7 4v3M7 17v3M12 4v3M12 17v3M17 4v3M17 17v3" stroke="#ffffff" strokeWidth="1.8" strokeLinecap="round" />
      </svg>
    </div>
  );
}

export function TopBar({ title, subtitle, user, onLogout }) {
  return (
    <header className="flex items-center justify-between gap-3 border-b border-white/10 bg-brand-navy px-4 sm:px-6 py-3.5">
      <div className="flex items-center gap-3 min-w-0">
        <BrandMark />
        <div className="min-w-0">
          <h1 className="text-[15px] font-semibold text-white truncate leading-tight">{title}</h1>
          {subtitle && <p className="text-xs text-white/60 truncate leading-tight mt-0.5">{subtitle}</p>}
        </div>
      </div>
      {onLogout && (
        <div className="flex items-center gap-3 shrink-0">
          {user && <span className="hidden sm:inline text-xs text-white/60">{user}</span>}
          <button
            onClick={onLogout}
            className="text-xs text-white/80 hover:text-white border border-white/20 rounded-md px-3 py-1.5 transition-colors hover:bg-white/10"
          >
            Log out
          </button>
        </div>
      )}
    </header>
  );
}

// Replaces the old left sidebar (desktop) + bottom tab bar (mobile) pair with
// a single horizontal nav that works at every width — a narrow viewport
// scrolls it sideways instead of switching to a different component, so
// there's exactly one navigation affordance to learn regardless of device.
// "home" is the permanent backdrop, not a closable window, so it's always
// shown active; every other item highlights only while its window is open.
export function TopNav({ openIds, onNavigate }) {
  const scrollerRef = useRef(null);
  const [canScrollLeft, setCanScrollLeft] = useState(false);
  const [canScrollRight, setCanScrollRight] = useState(false);

  // Edge fades only appear when there's actually more to scroll to — on a
  // wide viewport where all 5 tabs fit, neither ever shows.
  useEffect(() => {
    const el = scrollerRef.current;
    if (!el) return;
    function update() {
      setCanScrollLeft(el.scrollLeft > 4);
      setCanScrollRight(el.scrollLeft + el.clientWidth < el.scrollWidth - 4);
    }
    update();
    el.addEventListener("scroll", update, { passive: true });
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => {
      el.removeEventListener("scroll", update);
      ro.disconnect();
    };
  }, []);

  return (
    <div className="relative border-b border-white/10 bg-brand-navy">
      <nav ref={scrollerRef} className="flex items-center gap-1 overflow-x-auto px-3 sm:px-5">
        {NAV_ITEMS.map((item) => {
          const isOn = item.id === "home" || openIds.has(item.id);
          return (
            <button
              key={item.id}
              onClick={() => onNavigate(item.id)}
              className={`relative shrink-0 px-3.5 py-2.5 text-sm font-medium whitespace-nowrap transition-colors ${
                isOn ? "text-white" : "text-white/55 hover:text-white/80"
              }`}
            >
              {item.label}
              <span
                className={`absolute left-3 right-3 -bottom-px h-0.5 rounded-full transition-colors ${
                  isOn ? "bg-white" : "bg-transparent"
                }`}
              />
            </button>
          );
        })}
      </nav>
      {canScrollLeft && (
        <div className="pointer-events-none absolute left-0 top-0 bottom-0 w-8 bg-gradient-to-r from-brand-navy to-transparent" />
      )}
      {canScrollRight && (
        <div className="pointer-events-none absolute right-0 top-0 bottom-0 w-8 bg-gradient-to-l from-brand-navy to-transparent" />
      )}
    </div>
  );
}
