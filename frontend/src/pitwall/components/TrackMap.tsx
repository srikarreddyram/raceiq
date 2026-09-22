/**
 * The circuit as reconstructed by track_maps/ from one reference lap of
 * FastF1 position telemetry — PRD Section 7.2. Hand-rolled SVG, per the
 * design template: one path, a circle per corner, no charting library.
 *
 * Corner nodes are coloured by PRD 7.1's apex-speed classes. DRS zones are
 * drawn only when the API returns them: `drs_zones` is null for a
 * reference lap under the 2026 regulations, which have no DRS, and the
 * legend says so instead of implying a circuit with zero zones.
 */

import { useState } from "react";
import type { CircuitMap, CornerNode } from "../../api/types";
import { C, F, NUM } from "../../design/tokens";

const CLASS_COLOURS: Record<CornerNode["speed_class"], string> = {
  slow: C.red,
  medium: C.amber,
  fast: C.green,
};

const CLASS_LABELS: Record<CornerNode["speed_class"], string> = {
  slow: "SLOW < 120",
  medium: "MEDIUM",
  fast: "FAST > 200",
};

function LegendItem({ colour, label, line }: { colour: string; label: string; line?: boolean }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
      <span
        style={{
          width: line ? 16 : 8,
          height: line ? 3 : 8,
          borderRadius: line ? 2 : "50%",
          background: colour,
        }}
      />
      {label}
    </span>
  );
}

export function TrackMap({ map }: { map: CircuitMap }) {
  const [hovered, setHovered] = useState<CornerNode | null>(null);
  const size = map.viewbox;
  const [sx, sy] = map.start_finish;

  return (
    <div>
      <div style={{ position: "relative" }}>
        <svg
          viewBox={`0 0 ${size} ${size}`}
          style={{ width: "100%", maxHeight: 520, display: "block" }}
          role="img"
          aria-label={`Track map of ${map.circuit_id}`}
        >
          {/* The F1 app's circuit drawing: a bold carbon line over a soft
              grey run-off band. */}
          <path d={map.svg_path} fill="none" stroke={C.fill} strokeWidth={26} strokeLinejoin="round" />
          <path d={map.svg_path} fill="none" stroke={C.carbon} strokeWidth={8} strokeLinejoin="round" />

          {map.drs_zones?.map((zone, i) => (
            <path key={i} d={zone} fill="none" stroke={C.green} strokeWidth={9} strokeLinecap="round" opacity={0.9} />
          ))}

          {map.sector_boundaries.map(([x, y], i) => (
            <g key={i}>
              <circle cx={x} cy={y} r={7} fill={C.surface} stroke={C.accent} strokeWidth={3} />
              <text x={x + 12} y={y - 10} fill={C.accent} fontFamily={F.mono} fontWeight={700} fontSize={22}>
                S{i + 2}
              </text>
            </g>
          ))}

          <rect x={sx - 5} y={sy - 18} width={10} height={36} fill="#E10600" />

          {map.corners.map((corner) => {
            const active = hovered?.number === corner.number;
            return (
              <g
                key={corner.number}
                onMouseEnter={() => setHovered(corner)}
                onMouseLeave={() => setHovered(null)}
                style={{ cursor: "default" }}
              >
                <circle
                  cx={corner.x}
                  cy={corner.y}
                  r={active ? 19 : 14}
                  fill={CLASS_COLOURS[corner.speed_class]}
                  stroke={C.surface}
                  strokeWidth={3}
                  style={{ transition: "r 150ms" }}
                />
                <text
                  x={corner.x}
                  y={corner.y + 4}
                  textAnchor="middle"
                  fill={C.surface}
                  fontFamily={F.mono}
                  fontSize={14}
                  fontWeight={700}
                  pointerEvents="none"
                >
                  {corner.number}
                </text>
              </g>
            );
          })}
        </svg>

        {hovered && (
          <div
            style={{
              position: "absolute",
              top: 8,
              right: 8,
              background: C.raised,
              border: `1px solid ${C.edge}`,
              borderLeft: `3px solid ${CLASS_COLOURS[hovered.speed_class]}`,
              borderRadius: 4,
              padding: "10px 14px",
              fontFamily: F.mono,
              fontSize: 11,
              color: C.dim,
              lineHeight: 1.7,
            }}
          >
            <div style={{ color: C.text, letterSpacing: "0.18em" }}>TURN {hovered.number}</div>
            <div>
              APEX <span style={{ ...NUM, color: C.text }}>{hovered.min_speed_kph.toFixed(0)}</span> KM/H
            </div>
            <div>
              RADIUS{" "}
              <span style={{ ...NUM, color: C.text }}>
                {hovered.radius_m == null ? "—" : `${hovered.radius_m.toFixed(0)} M`}
              </span>
            </div>
          </div>
        )}
      </div>

      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: 18,
          marginTop: 14,
          fontFamily: F.mono,
          fontSize: 9,
          letterSpacing: "0.16em",
          color: C.faint,
        }}
      >
        {(["slow", "medium", "fast"] as const).map((cls) => (
          <LegendItem key={cls} colour={CLASS_COLOURS[cls]} label={CLASS_LABELS[cls]} />
        ))}
        {map.drs_zones ? (
          <LegendItem colour={C.green} label="DRS ZONE" line />
        ) : (
          <span>
            {map.reference_season >= 2026 ? "NO DRS — 2026 REGULATIONS" : "DRS NOT OBSERVED ON REFERENCE LAP"}
          </span>
        )}
        <LegendItem colour="#E10600" label="START / FINISH" line />
      </div>
    </div>
  );
}
