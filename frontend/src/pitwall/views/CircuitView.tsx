/**
 * Circuit View — PRD Section 13.2. The interactive track map is
 * reconstructed by track_maps/ from FastF1 position telemetry and served by
 * GET /circuits/{id}/map, alongside the Section 10.3 geometry features.
 *
 * Every map is labelled with the season of the lap it was drawn from:
 * corner speeds, and whether DRS exists at all, belong to a regulation
 * era, not to the circuit forever.
 *
 * Also shown: each circuit's geography and its leakage-safe historical
 * safety-car rate — the same figure the strategy engine samples from when
 * deciding whether a lap is worth gambling on a caution.
 */

import { useState } from "react";
import { api } from "../../api/client";
import type { Circuit, CircuitMap } from "../../api/types";
import { useAsync } from "../../api/useAsync";
import { Bar, Card, EmptyState, ErrorState, SectionLabel, Stat, TableSkeleton, tierLabel } from "../../design/primitives";
import { C, DISPLAY, F, NUM } from "../../design/tokens";
import { TrackMap } from "../components/TrackMap";

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
        background: active ? C.hover : "transparent",
        border: "none",
        borderLeft: `2px solid ${active ? C.accent : "transparent"}`,
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
  const map = useAsync(() => api.circuitMap(selected), [selected], Boolean(selected));

  return (
    <div>
      <SectionLabel>Circuit view</SectionLabel>
      <h1 style={{ ...DISPLAY, fontSize: 34, margin: "8px 0 20px", letterSpacing: "0.02em" }}>
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
                <Card accent={C.accent} style={{ padding: "22px 26px" }}>
                  <SectionLabel>{profile.data.locality}, {profile.data.country}</SectionLabel>
                  <h2 style={{ ...DISPLAY, fontSize: 40, margin: "10px 0 20px", letterSpacing: "0.02em" }}>
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
                      color={C.accent}
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

<Card style={{ padding: "20px 22px" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 14, gap: 12 }}>
                    <SectionLabel>Track map</SectionLabel>
                    {map.data && (
                      <span style={{ fontFamily: F.mono, fontSize: 9, letterSpacing: "0.16em", color: C.faint }}>
                        REFERENCE LAP {map.data.reference_race_id.replace("_", " R")} ·{" "}
                        {map.data.reference_season >= 2026 ? "2026 REGS" : map.data.reference_season >= 2022 ? "2022–25 REGS" : "2018–21 REGS"}
                      </span>
                    )}
                  </div>
                  {map.loading && <TableSkeleton rows={6} columns={1} />}
                  {map.error && <EmptyState>NO TRACK MAP FOR THIS CIRCUIT — {map.error.toUpperCase()}</EmptyState>}
                  {map.data && !map.loading && <TrackMap map={map.data} />}
                </Card>

                {map.data && !map.loading && <GeometryCard map={map.data} />}
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function GeometryCard({ map }: { map: CircuitMap }) {
  const km = (m: number) => (m / 1000).toFixed(3);
  return (
    <Card style={{ padding: "20px 22px" }}>
      <SectionLabel style={{ marginBottom: 16 }}>Circuit geometry</SectionLabel>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))", gap: 22 }}>
        <Stat label="Lap length" value={km(map.lap_length_m)} unit="km" size={24} />
        <Stat
          label="Corners"
          value={`${map.corner_count}`}
          unit={`${map.slow_corner_count}S · ${map.medium_corner_count}M · ${map.fast_corner_count}F`}
          size={24}
        />
        <Stat label="Downforce demand" value={map.downforce_demand_index.toFixed(2)} color={C.accent} size={24} />
        <Stat label="Elevation range" value={map.elevation_range_m.toFixed(1)} unit="m" size={24} />
        <Stat label="Braking distance" value={map.total_braking_distance_m.toFixed(0)} unit="m / lap" size={24} />
        <Stat label="Pit lane loss" value={map.pit_lane_delta.toFixed(1)} unit="s" size={24} />
        <Stat
          label="Tyre stress"
          value={map.tyre_stress_index == null ? "—" : `${map.tyre_stress_index > 0 ? "+" : ""}${map.tyre_stress_index.toFixed(3)}`}
          unit="s/lap vs grid"
          color={map.tyre_stress_index != null && map.tyre_stress_index > 0 ? C.red : undefined}
          size={24}
        />
        <Stat
          label="DRS length"
          value={map.drs_zone_total_length_m == null ? "N/A" : map.drs_zone_total_length_m.toFixed(0)}
          unit={map.drs_zone_total_length_m == null ? undefined : "m"}
          size={24}
        />
      </div>
      <div style={{ display: "flex", gap: 22, marginTop: 20, fontFamily: F.mono, fontSize: 10, color: C.faint, letterSpacing: "0.12em" }}>
        {[map.sector1_avg_speed, map.sector2_avg_speed, map.sector3_avg_speed].map((v, i) => (
          <span key={i}>
            S{i + 1} AVG <span style={{ ...NUM, color: C.text }}>{v == null ? "—" : v.toFixed(0)}</span> KM/H
          </span>
        ))}
      </div>
      <div style={{ fontFamily: F.body, fontSize: 12, color: C.muted, lineHeight: 1.6, marginTop: 14 }}>
        Downforce demand is the share of the lap not spent flat out. Tyre stress is this circuit's median
        in-stint lap-time slope minus the grid-wide median, over all history — context, not a model input.
        {map.detection_recall != null &&
          ` Curvature detection found ${(map.detection_recall * 100).toFixed(0)}% of the official corners.`}
      </div>
    </Card>
  );
}
