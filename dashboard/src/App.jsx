import { useState } from "react";
import { TopBar, TopNav } from "./components/Layout.jsx";
import Window from "./components/Window.jsx";
import LoginPage from "./components/LoginPage.jsx";
import Home from "./subsystems/Home.jsx";
import DoorPage from "./subsystems/door/DoorPage.jsx";
import AcvPage from "./subsystems/acv/AcvPage.jsx";
import RailPage from "./subsystems/rail/RailPage.jsx";
import ShmPage from "./subsystems/shm/ShmPage.jsx";
import { getAuthedUser, login, logout } from "./lib/auth.js";

const TITLES = {
  door: "Door",
  acv: "ACV",
  rail: "Rail Corrugation",
  shm: "SHM",
};

// Cascading defaults so windows opened one after another don't land exactly
// on top of each other — the user drags them wherever they actually want.
const DEFAULT_WINDOWS = {
  door: { open: false, x: 40, y: 24, width: 640, height: 560, z: 1, maximized: false },
  acv: { open: false, x: 90, y: 64, width: 640, height: 580, z: 1, maximized: false },
  rail: { open: false, x: 140, y: 104, width: 680, height: 600, z: 1, maximized: false },
  shm: { open: false, x: 190, y: 144, width: 680, height: 600, z: 1, maximized: false },
};

export default function App() {
  const [user, setUser] = useState(() => getAuthedUser());
  const [windows, setWindows] = useState(DEFAULT_WINDOWS);
  const [zCounter, setZCounter] = useState(1);
  // Each subsystem page reports its own latest summary up here once it has
  // one, so Overview can show real aggregated numbers without recomputing
  // anything itself or duplicating any model logic.
  const [summaries, setSummaries] = useState({});

  function reportSummary(id, summary) {
    setSummaries((prev) => ({ ...prev, [id]: summary }));
  }

  function bringToFront(id) {
    const nextZ = zCounter + 1;
    setZCounter(nextZ);
    setWindows((prev) => ({ ...prev, [id]: { ...prev[id], z: nextZ } }));
    return nextZ;
  }

  function openWindow(id) {
    if (id === "home") return; // the backdrop, not a window
    const nextZ = zCounter + 1;
    setZCounter(nextZ);
    setWindows((prev) => ({ ...prev, [id]: { ...prev[id], open: true, z: nextZ } }));
  }

  function closeWindow(id) {
    setWindows((prev) => ({ ...prev, [id]: { ...prev[id], open: false } }));
  }

  function moveWindow(id, x, y) {
    setWindows((prev) => ({ ...prev, [id]: { ...prev[id], x, y } }));
  }

  function resizeWindow(id, width, height) {
    setWindows((prev) => ({ ...prev, [id]: { ...prev[id], width, height } }));
  }

  function toggleMaximize(id) {
    bringToFront(id);
    setWindows((prev) => ({ ...prev, [id]: { ...prev[id], maximized: !prev[id].maximized } }));
  }

  const openIds = new Set(Object.keys(windows).filter((id) => windows[id].open));

  if (!user) {
    return (
      <LoginPage
        onLogin={(u) => {
          login(u);
          setUser(u);
        }}
      />
    );
  }

  return (
    <div className="flex h-screen flex-col">
      <TopBar
        title="Train Condition Monitoring"
        subtitle="Select a tab below to open its window"
        user={user}
        onLogout={() => {
          logout();
          setUser(null);
        }}
      />
      <TopNav openIds={openIds} onNavigate={openWindow} />

      {/* The desktop: Overview is the permanent backdrop; every subsystem
          page mounts once, immediately, and stays mounted for the whole
          session (Window only toggles display:none) — so opening,
          closing, and reopening a window never re-fetches or re-runs a
          pipeline that already ran, same as the earlier tab-based design. */}
      <main className="flex-1 relative overflow-hidden bg-surface-page">
        <div className="absolute inset-0 overflow-y-auto px-6 py-5">
          <Home onNavigate={openWindow} summaries={summaries} />
        </div>

        <Window
          title={TITLES.door}
          {...windows.door}
          onClose={() => closeWindow("door")}
          onFocus={() => bringToFront("door")}
          onMove={(x, y) => moveWindow("door", x, y)}
          onResize={(w, h) => resizeWindow("door", w, h)}
          onToggleMaximize={() => toggleMaximize("door")}
        >
          <DoorPage onSummary={(s) => reportSummary("door", s)} />
        </Window>

        <Window
          title={TITLES.acv}
          {...windows.acv}
          onClose={() => closeWindow("acv")}
          onFocus={() => bringToFront("acv")}
          onMove={(x, y) => moveWindow("acv", x, y)}
          onResize={(w, h) => resizeWindow("acv", w, h)}
          onToggleMaximize={() => toggleMaximize("acv")}
        >
          <AcvPage onSummary={(s) => reportSummary("acv", s)} />
        </Window>

        <Window
          title={TITLES.rail}
          {...windows.rail}
          onClose={() => closeWindow("rail")}
          onFocus={() => bringToFront("rail")}
          onMove={(x, y) => moveWindow("rail", x, y)}
          onResize={(w, h) => resizeWindow("rail", w, h)}
          onToggleMaximize={() => toggleMaximize("rail")}
        >
          <RailPage onSummary={(s) => reportSummary("rail", s)} />
        </Window>

        <Window
          title={TITLES.shm}
          {...windows.shm}
          onClose={() => closeWindow("shm")}
          onFocus={() => bringToFront("shm")}
          onMove={(x, y) => moveWindow("shm", x, y)}
          onResize={(w, h) => resizeWindow("shm", w, h)}
          onToggleMaximize={() => toggleMaximize("shm")}
        >
          <ShmPage onSummary={(s) => reportSummary("shm", s)} />
        </Window>
      </main>
    </div>
  );
}
