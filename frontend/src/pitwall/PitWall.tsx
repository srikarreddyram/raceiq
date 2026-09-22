/**
 * RaceIQ Pit Wall — PRD Section 13.2's data-dense engineering dashboard.
 * Different design language from the splash page by design: no cinematic
 * motion, no scroll choreography, everything above the fold and legible
 * at a glance.
 *
 * Organised around the race weekend, the project's centrepiece: the
 * planner comes first, then the in-race strategy call and simulator on one
 * page, then the circuit, the car (speed, characteristics and tyres on one
 * page), the driver and model performance. PRD 13.2's seven views are all
 * here, grouped where they answer the same question.
 */

import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import { EdgeBezel } from "../design/chrome";
import { Logo } from "../design/Logo";
import { DEFAULT_ACCENT } from "../design/theme";
import { C, DISPLAY, F } from "../design/tokens";
import { CarView } from "./views/CarView";
import { CircuitView } from "./views/CircuitView";
import { DriverView } from "./views/DriverView";
import { ModelPerformanceView } from "./views/ModelPerformanceView";
import { RaceView } from "./views/RaceView";
import { PlannerView } from "./views/PlannerView";
import { WeekendProvider } from "./WeekendContext";

type NavItem = { to: string; label: string };

const NAV: NavItem[] = [
  { to: "/pitwall/weekend", label: "Race Weekend" },
  { to: "/pitwall/strategy", label: "Race Strategy" },
  { to: "/pitwall/circuit", label: "Circuit" },
  { to: "/pitwall/car", label: "Car" },
  { to: "/pitwall/driver", label: "Driver" },
  { to: "/pitwall/models", label: "Model Perf" },
];

// Text on the carbon bar is white by construction — these literals belong
// to the bar, not the theme.
const ON_BAR = "#FFFFFF";
const ON_BAR_DIM = "rgba(255,255,255,0.66)";

/**
 * The header: carbon black with white type, F1 red only where F1 itself
 * uses it — the mark, the active tab, a thin line under the bar. A fully
 * red bar read as a Ferrari site rather than an F1 one.
 */
function TopBar() {
  return (
    <header
      style={{
        position: "sticky",
        top: 0,
        zIndex: 30,
        background: C.carbon,
        boxShadow: "0 2px 10px rgba(21,21,30,0.18)",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 28,
          padding: "0 20px",
          height: 60,
          maxWidth: 1680,
          margin: "0 auto",
        }}
      >
        <NavLink
          to="/"
          style={{
            ...DISPLAY,
            fontStyle: "italic",
            fontSize: 24,
            color: ON_BAR,
            textDecoration: "none",
            whiteSpace: "nowrap",
          }}
        >
          <Logo height={26} onDark />
        </NavLink>

        <div
          style={{
            fontFamily: F.mono,
            fontWeight: 700,
            fontSize: 11,
            letterSpacing: "0.14em",
            color: ON_BAR_DIM,
            whiteSpace: "nowrap",
            borderLeft: `1px solid ${ON_BAR_DIM}`,
            paddingLeft: 14,
          }}
        >
          PIT WALL
        </div>

        <nav style={{ display: "flex", gap: 4, overflowX: "auto", flex: 1, alignSelf: "stretch" }}>
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              style={({ isActive }) => ({
                display: "flex",
                alignItems: "center",
                fontFamily: F.mono,
                fontWeight: isActive ? 700 : 600,
                fontSize: 13,
                letterSpacing: "0.04em",
                textTransform: "uppercase",
                color: isActive ? ON_BAR : ON_BAR_DIM,
                textDecoration: "none",
                padding: "0 13px",
                borderBottom: `3px solid ${isActive ? DEFAULT_ACCENT : "transparent"}`,
                whiteSpace: "nowrap",
                transition: "color 200ms",
              })}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      </div>
      <div aria-hidden style={{ position: "absolute", left: 0, right: 0, bottom: -3, height: 3, background: DEFAULT_ACCENT }} />
    </header>
  );
}

export function PitWall() {
  return (
    <WeekendProvider>
    <div style={{ minHeight: "100vh", background: C.bg }}>
      <EdgeBezel />
      <TopBar />
      <main style={{ maxWidth: 1680, margin: "0 auto", padding: "26px 20px 80px" }}>
        <Routes>
          <Route index element={<Navigate to="/pitwall/weekend" replace />} />
          <Route path="weekend" element={<PlannerView />} />
          <Route path="strategy" element={<RaceView />} />
          <Route path="circuit" element={<CircuitView />} />
          <Route path="car" element={<CarView />} />
          <Route path="driver" element={<DriverView />} />
          <Route path="models" element={<ModelPerformanceView />} />
          {/* Old addresses from before the merges. */}
          <Route path="race" element={<Navigate to="/pitwall/strategy" replace />} />
          <Route path="simulation" element={<Navigate to="/pitwall/strategy" replace />} />
          <Route path="car-profile" element={<Navigate to="/pitwall/car" replace />} />
          <Route path="tyre" element={<Navigate to="/pitwall/car#tyres" replace />} />
          <Route path="*" element={<Navigate to="/pitwall/weekend" replace />} />
        </Routes>
      </main>
    </div>
    </WeekendProvider>
  );
}
