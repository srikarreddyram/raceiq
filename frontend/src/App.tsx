/**
 * Two surfaces, one app — PRD Section 13: "RaceIQ Public" (cinematic
 * splash) and "RaceIQ Pit Wall" (data-dense dashboard), described there
 * as "entirely separate ... different route, different design language,
 * same API backend".
 *
 * The PRD's repository layout sketches these as two directories under
 * frontend/ (splash/ and pitwall/). They're two route trees inside one
 * Vite app here instead: they share the token/primitive layer and the API
 * client, so two separate builds would mean duplicating both plus the
 * toolchain to get one product. The separation the PRD actually asks for
 * — different route, different design language — holds either way.
 */

import { lazy, Suspense } from "react";
import { Route, Routes } from "react-router-dom";
import { Animations } from "./design/Animations";
import { EmptyState } from "./design/primitives";
import { accentVars, DEFAULT_ACCENT } from "./design/theme";
import { SplashPage } from "./splash/SplashPage";

// The Pit Wall is lazy-loaded: it pulls in recharts, which is most of the
// bundle and which the splash page — the public-facing surface, and the
// one where load time actually costs something — never renders.
const PitWall = lazy(() => import("./pitwall/PitWall").then((m) => ({ default: m.PitWall })));

export function App() {
  return (
    <div style={accentVars(DEFAULT_ACCENT)}>
      <Animations />
      <Routes>
        <Route path="/" element={<SplashPage />} />
        <Route
          path="/pitwall/*"
          element={
            <Suspense fallback={<EmptyState>LOADING THE PIT WALL…</EmptyState>}>
              <PitWall />
            </Suspense>
          }
        />
      </Routes>
    </div>
  );
}
