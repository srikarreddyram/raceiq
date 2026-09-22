/**
 * RaceIQ Pit Wall — PRD Section 13.2's data-dense engineering dashboard.
 * Different design language from the splash page by design: no cinematic
 * motion, no scroll choreography, everything above the fold and legible
 * at a glance.
 *
 * PRD 13.2 names seven views, and all seven are built here: Race,
 * Strategy Simulation, Circuit, Car Profile, Tyre, Driver and Model
 * Performance.
 */

import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import { EdgeBezel, HeaderStripe } from "../design/chrome";
import { DEFAULT_ACCENT } from "../design/theme";
import { C, DISPLAY, F } from "../design/tokens";
import { CarProfileView } from "./views/CarProfileView";
import { CircuitView } from "./views/CircuitView";
import { DriverView } from "./views/DriverView";
import { ModelPerformanceView } from "./views/ModelPerformanceView";
import { RaceView } from "./views/RaceView";
import { SimulationView } from "./views/SimulationView";
import { TyreView } from "./views/TyreView";

type NavItem = { to: string; label: string };

const NAV: NavItem[] = [
  { to: "/pitwall/race", label: "Race" },
  { to: "/pitwall/simulation", label: "Strategy Sim" },
  { to: "/pitwall/circuit", label: "Circuit" },
  { to: "/pitwall/car-profile", label: "Car Profile" },
  { to: "/pitwall/tyre", label: "Tyre" },
  { to: "/pitwall/driver", label: "Driver" },
  { to: "/pitwall/models", label: "Model Perf" },
];

// Text on the red bar is white by construction — these are the only
// literals in the Pit Wall, and they belong to the bar, not the theme.
const ON_BAR = "#FFFFFF";
const ON_BAR_DIM = "rgba(255,255,255,0.78)";

/** The F1 app's red header: logo, section name, and the seven views. */
function TopBar() {
  return (
    <header
      style={{
        position: "sticky",
        top: 0,
        zIndex: 30,
        background: DEFAULT_ACCENT,
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
          RACE<span style={{ fontWeight: 400 }}>IQ</span>
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
                borderBottom: `3px solid ${isActive ? ON_BAR : "transparent"}`,
                whiteSpace: "nowrap",
                transition: "color 200ms",
              })}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      </div>
      <HeaderStripe bottom={-4} />
    </header>
  );
}

export function PitWall() {
  return (
    <div style={{ minHeight: "100vh", background: C.bg }}>
      <EdgeBezel />
      <TopBar />
      <main style={{ maxWidth: 1680, margin: "0 auto", padding: "26px 20px 80px" }}>
        <Routes>
          <Route index element={<Navigate to="/pitwall/race" replace />} />
          <Route path="race" element={<RaceView />} />
          <Route path="simulation" element={<SimulationView />} />
          <Route path="circuit" element={<CircuitView />} />
          <Route path="car-profile" element={<CarProfileView />} />
          <Route path="tyre" element={<TyreView />} />
          <Route path="driver" element={<DriverView />} />
          <Route path="models" element={<ModelPerformanceView />} />
          <Route path="*" element={<Navigate to="/pitwall/race" replace />} />
        </Routes>
      </main>
    </div>
  );
}
