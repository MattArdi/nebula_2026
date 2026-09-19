import { useRef } from "react";

const MIN_WIDTH = 380;
const MIN_HEIGHT = 280;

/**
 * A draggable, resizable, maximizable floating panel. `open` only toggles
 * visibility (display:none) — it never unmounts `children`, so a
 * subsystem page's already-loaded data survives closing and reopening its
 * window.
 *
 * Drag/resize deliberately do NOT call onMove/onResize on every mousemove.
 * With four subsystem pages (tables + charts) always mounted, pushing a
 * React state update on every tick forced a full-tree re-render per pixel
 * of movement, which is what made dragging feel laggy. Instead the drag
 * mutates this window's own DOM node directly (cheap, no reconciliation)
 * and only commits the final position/size to React state once, on
 * mouseup — the source of truth stays in App's `windows` state either way.
 */
export default function Window({
  title,
  open,
  x,
  y,
  width,
  height,
  z,
  maximized,
  onClose,
  onFocus,
  onMove,
  onResize,
  onToggleMaximize,
  children,
}) {
  const startRef = useRef(null);
  const rootRef = useRef(null);

  function clamp(px, py, w) {
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    // Keep at least ~80px of the header reachable so a dragged-off window
    // can always be pulled back.
    return {
      x: Math.min(Math.max(px, -(w - 80)), vw - 80),
      y: Math.min(Math.max(py, 0), vh - 40),
    };
  }

  function startDrag(e) {
    if (maximized || e.button !== 0) return;
    e.preventDefault(); // stop native text-selection drag from firing alongside the move
    document.body.style.userSelect = "none"; // belt-and-braces: also blocks selection starting on any other element mid-drag
    onFocus();
    startRef.current = { mouseX: e.clientX, mouseY: e.clientY, baseX: x, baseY: y, liveX: x, liveY: y };
    window.addEventListener("mousemove", handleDrag);
    window.addEventListener("mouseup", stopDrag);
  }
  function handleDrag(e) {
    const s = startRef.current;
    if (!s) return;
    const dx = e.clientX - s.mouseX;
    const dy = e.clientY - s.mouseY;
    const resolved = clamp(s.baseX + dx, s.baseY + dy, width);
    s.liveX = resolved.x;
    s.liveY = resolved.y;
    if (rootRef.current) {
      rootRef.current.style.left = `${resolved.x}px`;
      rootRef.current.style.top = `${resolved.y}px`;
    }
  }
  function stopDrag() {
    const s = startRef.current;
    startRef.current = null;
    document.body.style.userSelect = "";
    window.removeEventListener("mousemove", handleDrag);
    window.removeEventListener("mouseup", stopDrag);
    if (s) onMove(s.liveX, s.liveY); // commit once, on release
  }

  function startResize(e) {
    if (maximized) return;
    e.stopPropagation();
    if (e.button !== 0) return;
    e.preventDefault();
    document.body.style.userSelect = "none";
    onFocus();
    startRef.current = { mouseX: e.clientX, mouseY: e.clientY, baseW: width, baseH: height, liveW: width, liveH: height };
    window.addEventListener("mousemove", handleResize);
    window.addEventListener("mouseup", stopResize);
  }
  function handleResize(e) {
    const s = startRef.current;
    if (!s) return;
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const w = Math.min(Math.max(MIN_WIDTH, s.baseW + (e.clientX - s.mouseX)), vw - x);
    const h = Math.min(Math.max(MIN_HEIGHT, s.baseH + (e.clientY - s.mouseY)), vh - y);
    s.liveW = w;
    s.liveH = h;
    if (rootRef.current) {
      rootRef.current.style.width = `${w}px`;
      rootRef.current.style.height = `${h}px`;
    }
  }
  function stopResize() {
    const s = startRef.current;
    startRef.current = null;
    document.body.style.userSelect = "";
    window.removeEventListener("mousemove", handleResize);
    window.removeEventListener("mouseup", stopResize);
    if (s) onResize(s.liveW, s.liveH);
  }

  const style = maximized
    ? { left: 8, top: 8, right: 8, bottom: 8, zIndex: z, display: open ? "flex" : "none" }
    : { left: x, top: y, width, height, zIndex: z, display: open ? "flex" : "none" };

  return (
    <div
      ref={rootRef}
      onMouseDown={onFocus}
      className="absolute rounded-lg border border-line-border bg-surface-page shadow-2xl flex flex-col overflow-hidden"
      style={style}
    >
      <div
        onMouseDown={startDrag}
        onDoubleClick={onToggleMaximize}
        className={`flex items-center justify-between px-3 py-2 bg-surface-card border-b border-line-hairline select-none shrink-0 ${
          maximized ? "cursor-default" : "cursor-move"
        }`}
      >
        <span className="text-2xl font-bold text-ink-primary truncate">{title}</span>
        <div className="flex items-center gap-1 shrink-0">
          <button
            onMouseDown={(e) => e.stopPropagation()}
            onClick={onToggleMaximize}
            title={maximized ? "Restore" : "Maximize"}
            className="text-ink-muted hover:text-ink-primary hover:bg-surface-raised w-5 h-5 flex items-center justify-center rounded"
          >
            {maximized ? (
              <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
                <path d="M3 1H9V7" stroke="currentColor" strokeWidth="1.1" />
                <path d="M1 3H7V9H1V3Z" stroke="currentColor" strokeWidth="1.1" />
              </svg>
            ) : (
              <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
                <rect x="0.5" y="0.5" width="9" height="9" stroke="currentColor" strokeWidth="1.1" />
              </svg>
            )}
          </button>
          <button
            onMouseDown={(e) => e.stopPropagation()}
            onClick={onClose}
            title="Close"
            className="text-ink-muted hover:text-status-critical hover:bg-surface-raised w-5 h-5 flex items-center justify-center rounded"
          >
            <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
              <path d="M1 1L9 9M9 1L1 9" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
            </svg>
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4 min-h-0">{children}</div>

      {!maximized && (
        <div
          onMouseDown={startResize}
          title="Resize"
          className="absolute bottom-0 right-0 w-4 h-4 cursor-nwse-resize"
          style={{ background: "linear-gradient(135deg, transparent 50%, rgba(35,35,34,0.25) 50%)" }}
        />
      )}
    </div>
  );
}
