/**
 * RaceIQ Pit Wall — PRD Section 13.2's data-dense engineering dashboard.
 * Different design language from the splash page by design: no cinematic
 * motion, no scroll choreography, everything above the fold and legible
 * at a glance.
 *
 * PRD 13.2 names seven views. Five are built here (Race, Strategy
 * Simulation, Circuit, Car Profile, Tyre); the rest are listed in the nav
 * as unavailable, with the specific missing backend named, rather than
 * silently omitted or stubbed with fake panels:
 *   - Driver needs a per-driver history endpoint
 *   - Model Performance needs monitoring/ exposed over the API (it exists
 *     as a CLI report today, not an endpoint)
 */

import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import { EdgeBezel, HeaderStripe } from "../design/chrome";
import { C, F } from "../design/tokens";
import { CarProfileView } from "./views/CarProfileView";
import { CircuitView } from "./views/CircuitView";
import { RaceView } from "./views/RaceView";
import { SimulationView } from "./views/SimulationView";
import { TyreView } from "./views/TyreView";

type NavItem = { to: string; label: string; disabled?: string };

const NAV: NavItem[] = [
  { to: "/pitwall/race", label: "Race" },
  { to: "/pitwall/simulation", label: "Strategy Sim" },
  { to: "/pitwall/circuit", label: "Circuit" },
  { to: "/pitwall/car-profile", label: "Car Profile" },
  { to: "/pitwall/tyre", label: "Tyre" },
  { to: "/pitwall/driver", label: "Driver", disabled: "needs a driver-history endpoint" },
  { to: "/pitwall/models", label: "Model Perf", disabled: "monitoring/ is CLI-only today" },
];

function TopBar() {
  return (
    <header
      style={{
        position: "sticky",
        top: 0,
        zIndex: 30,
        background: "rgba(10,10,15,0.92)",
        backdropFilter: "blur(10px)",
        borderBottom: `1px solid ${C.edge}`,
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 30,
          padding: "0 20px",
          height: 58,
          maxWidth: 1680,
          margin: "0 auto",
        }}
      >
        <NavLink
          to="/"
          style={{
            fontFamily: F.display,
            fontSize: 22,
            letterSpacing: "0.1em",
            color: C.text,
            textDecoration: "none",
            whiteSpace: "nowrap",
          }}
        >
          RACE<span style={{ color: C.gold }}>IQ</span>
        </NavLink>

        <div
          style={{
            fontFamily: F.mono,
            fontSize: 8.5,
            letterSpacing: "0.3em",
            color: C.faint,
            whiteSpace: "nowrap",
          }}
        >
          PIT WALL
        </div>

        <nav style={{ display: "flex", gap: 2, overflowX: "auto", flex: 1 }}>
          {NAV.map((item) =>
            item.disabled ? (
              <span
                key={item.to}
                title={`Not built — ${item.disabled}`}
                style={{
                  fontFamily: F.mono,
                  fontSize: 10,
                  letterSpacing: "0.16em",
                  textTransform: "uppercase",
                  color: "rgba(240,240,240,0.22)",
                  padding: "9px 13px",
                  whiteSpace: "nowrap",
                  cursor: "not-allowed",
                }}
              >
                {item.label}
              </span>
            ) : (
              <NavLink
                key={item.to}
                to={item.to}
                style={({ isActive }) => ({
                  fontFamily: F.mono,
                  fontSize: 10,
                  letterSpacing: "0.16em",
                  textTransform: "uppercase",
                  color: isActive ? C.gold : C.dim,
                  textDecoration: "none",
                  padding: "9px 13px",
                  borderBottom: `2px solid ${isActive ? C.gold : "transparent"}`,
                  whiteSpace: "nowrap",
                  transition: "color 200ms",
                })}
              >
                {item.label}
              </NavLink>
            ),
          )}
        </nav>
      </div>
      <HeaderStripe bottom={-5} />
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
          <Route path="*" element={<Navigate to="/pitwall/race" replace />} />
        </Routes>
      </main>
    </div>
  );
}
