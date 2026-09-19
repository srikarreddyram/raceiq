/**
 * Circuit View — PRD Section 13.2 asks for an interactive SVG track map
 * with sector colouring, DRS overlays and corner nodes. That needs
 * track_maps/ (GPS → smoothed path → SVG) and GET /circuits/{id}/map,
 * neither of which exists — so this view shows what the warehouse
 * genuinely knows about each circuit and names the missing piece, rather
 * than drawing a decorative squiggle that isn't the real track.
 *
 * What IS real here: every circuit in the Gold layer, its geography, and
 * its leakage-safe historical safety-car rate — the same figure the
 * strategy engine samples from when deciding whether a lap is worth
 * gambling on a caution.
 */

import { useState } from "react";
import { api } from "../../api/client";
import type { Circuit } from "../../api/types";
import { useAsync } from "../../api/useAsync";
import { Bar, Card, EmptyState, ErrorState, SectionLabel, Stat, TableSkeleton, tierLabel } from "../../design/primitives";
import { C, F, NUM } from "../../design/tokens";

function CircuitRow({
  circuit,
  active,
  onSelect,
}: {
  circuit: Circuit;
  active: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      onClick={onSelect}
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        width: "100%",
        gap: 12,
        padding: "11px 14px",
        background: active ? "var(--rq-hover, rgba(201,168,76,0.08))" : "transparent",
        border: "none",
        borderLeft: `2px solid ${active ? C.gold : "transparent"}`,
        borderBottom: `1px solid ${C.rule}`,
        cursor: "pointer",
        textAlign: "left",
        transition: "background 200ms",
      }}
    >
      <span style={{ fontFamily: F.body, fontSize: 13, color: active ? C.text : C.dim }}>{circuit.name}</span>
      <span style={{ fontFamily: F.mono, fontSize: 9, color: C.faint, letterSpacing: "0.12em", whiteSpace: "nowrap" }}>
        {circuit.country.toUpperCase()}
      </span>
    </button>
  );
}

export function CircuitView() {
  const circuits = useAsync(() => api.circuits(), []);
  const [selected, setSelected] = useState<string>("bahrain");

  const profile = useAsync(() => api.circuitProfile(selected), [selected], Boolean(selected));

  return (
    <div>
      <SectionLabel>Circuit view</SectionLabel>
      <h1 style={{ fontFamily: F.display, fontSize: 34, margin: "8px 0 20px", letterSpacing: "0.02em" }}>
        Circuit profiles
      </h1>

      {circuits.error && <ErrorState message={circuits.error} />}
      {circuits.loading && <TableSkeleton rows={8} columns={2} />}

      {circuits.data && (
        <div style={{ display: "grid", gridTemplateColumns: "minmax(240px, 320px) 1fr", gap: 18, alignItems: "start" }}>
          <Card style={{ overflow: "hidden", maxHeight: 620, overflowY: "auto" }}>
            <div
              style={{
                padding: "12px 14px",
                borderBottom: `1px solid ${C.edge}`,
                fontFamily: F.mono,
                fontSize: 9,
                letterSpacing: "0.22em",
                color: C.faint,
                position: "sticky",
                top: 0,
                background: C.surface,
              }}
            >
              {circuits.data.length} CIRCUITS
            </div>
            {[...circuits.data]
              .sort((a, b) => a.name.localeCompare(b.name))
              .map((circuit) => (
                <CircuitRow
                  key={circuit.circuit_id}
                  circuit={circuit}
                  active={circuit.circuit_id === selected}
                  onSelect={() => setSelected(circuit.circuit_id)}
                />
              ))}
          </Card>

          <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
            {profile.error && <ErrorState message={profile.error} />}
            {profile.loading && <TableSkeleton rows={4} columns={3} />}

            {profile.data && !profile.loading && (
              <>
                <Card accent={C.gold} style={{ padding: "22px 26px" }}>
                  <SectionLabel>{profile.data.locality}, {profile.data.country}</SectionLabel>
                  <h2 style={{ fontFamily: F.display, fontSize: 40, margin: "10px 0 20px", letterSpacing: "0.02em" }}>
                    {profile.data.name}
                  </h2>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))", gap: 24 }}>
                    <Stat
                      label="Prior races"
                      value={profile.data.prior_races_at_circuit ?? "—"}
                      size={26}
                    />
                    <Stat
                      label="Safety car rate"
                      value={
                        profile.data.historical_sc_rate == null
                          ? "—"
                          : `${(profile.data.historical_sc_rate * 100).toFixed(0)}%`
                      }
                      color={C.gold}
                      size={26}
                    />
                    <Stat
                      label="Baseline track temp"
                      value={
                        profile.data.circuit_baseline_track_temp == null
                          ? "—"
                          : profile.data.circuit_baseline_track_temp.toFixed(1)
                      }
                      unit="°C"
                      size={26}
                    />
                    <Stat
                      label="Coordinates"
                      value={`${profile.data.latitude.toFixed(2)}, ${profile.data.longitude.toFixed(2)}`}
                      size={18}
                    />
                  </div>
                </Card>

                <Card style={{ padding: "20px 22px" }}>
                  <SectionLabel style={{ marginBottom: 14 }}>Safety car likelihood</SectionLabel>
                  {profile.data.historical_sc_rate == null ? (
                    <EmptyState>NO PRIOR RACES AT THIS CIRCUIT YET — NOTHING TO AVERAGE</EmptyState>
                  ) : (
                    <>
                      <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 10 }}>
                        <Bar pct={profile.data.historical_sc_rate} height={6} />
                        <span style={{ ...NUM, fontSize: 15, color: C.text, minWidth: 52, textAlign: "right" }}>
                          {(profile.data.historical_sc_rate * 100).toFixed(0)}%
                        </span>
                      </div>
                      <div style={{ fontFamily: F.body, fontSize: 12, color: C.muted, lineHeight: 1.6 }}>
                        {tierLabel(profile.data.historical_sc_rate)} likelihood relative to the grid — an
                        expanding average over races at this circuit strictly before each race, never
                        including the race being predicted. The strategy engine converts this into a
                        per-lap hazard when deciding whether a stop is worth gambling on a caution.
                      </div>
                    </>
                  )}
                </Card>

                <Card style={{ padding: "20px 22px", borderStyle: "dashed" }}>
                  <SectionLabel style={{ marginBottom: 10 }}>Track map — not built</SectionLabel>
                  <div style={{ fontFamily: F.body, fontSize: 12.5, color: C.muted, lineHeight: 1.7 }}>
                    PRD §7 specifies an interactive SVG track map reconstructed from FastF1 GPS telemetry —
                    sector colouring by pace delta, DRS overlays, corner classification nodes. That needs the{" "}
                    <span style={{ fontFamily: F.mono, fontSize: 11.5, color: C.gold }}>track_maps/</span>{" "}
                    pipeline and a{" "}
                    <span style={{ fontFamily: F.mono, fontSize: 11.5, color: C.gold }}>
                      GET /circuits/{"{id}"}/map
                    </span>{" "}
                    endpoint, neither of which exists yet. Drawing a decorative shape that isn't the real
                    circuit would look finished and be worse than nothing.
                  </div>
                </Card>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
