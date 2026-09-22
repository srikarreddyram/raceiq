/**
 * Season + team control shared by the Car Profile and Tyre views. The team
 * list comes from the season itself (GET /teams?season=), so a 2026 pick
 * offers Cadillac and Audi and a 2019 pick offers Toro Rosso — the grid as
 * it actually was, not today's names projected backwards.
 */

import { useEffect, useState } from "react";
import { api } from "../../api/client";
import { useAsync } from "../../api/useAsync";
import { C, F } from "../../design/tokens";

const SEASONS = [2026, 2025, 2024, 2023, 2022, 2021, 2020, 2019, 2018];

export function useTeamSelection(initialTeam = "ferrari", initialSeason = 2026) {
  const [season, setSeason] = useState(initialSeason);
  const [teamId, setTeamId] = useState(initialTeam);
  const teams = useAsync(() => api.teams(season), [season]);

  // A team that didn't race this season (Sauber -> Audi) falls back to the
  // first team on that grid rather than requesting a profile that 404s.
  useEffect(() => {
    if (teams.data && !teams.data.some((t) => t.team_id === teamId) && teams.data.length) {
      setTeamId(teams.data[0].team_id);
    }
  }, [teams.data, teamId]);

  const ready = Boolean(teams.data?.some((t) => t.team_id === teamId));
  const teamName = teams.data?.find((t) => t.team_id === teamId)?.name ?? teamId;
  return { season, setSeason, teamId, setTeamId, teams, ready, teamName };
}

export function TeamPicker({ state }: { state: ReturnType<typeof useTeamSelection> }) {
  const { season, setSeason, teamId, setTeamId, teams } = state;
  return (
    <div
      style={{
        display: "flex",
        flexWrap: "wrap",
        alignItems: "center",
        gap: 14,
        background: C.surface,
        border: `1px solid ${C.edge}`,
        borderRadius: 6,
        padding: "12px 16px",
        marginBottom: 20,
      }}
    >
      <select
        value={season}
        onChange={(e) => setSeason(Number(e.target.value))}
        aria-label="Season"
        style={{
          background: C.raised,
          color: C.text,
          border: `1px solid ${C.edge}`,
          borderRadius: 4,
          padding: "8px 10px",
          fontFamily: F.mono,
          fontSize: 12,
          outline: "none",
        }}
      >
        {SEASONS.map((s) => (
          <option key={s} value={s}>
            {s}
          </option>
        ))}
      </select>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
        {(teams.data ?? []).map((t) => {
          const active = t.team_id === teamId;
          return (
            <button
              key={t.team_id}
              onClick={() => setTeamId(t.team_id)}
              style={{
                fontFamily: F.mono,
                fontSize: 10,
                letterSpacing: "0.12em",
                textTransform: "uppercase",
                padding: "7px 11px",
                borderRadius: 4,
                cursor: "pointer",
                border: `1px solid ${active ? C.accent : C.edge}`,
                background: active ? C.hover : "transparent",
                color: active ? C.text : C.dim,
                transition: "all 200ms",
              }}
            >
              {t.name.replace(/ F1 Team$/, "")}
            </button>
          );
        })}
      </div>
    </div>
  );
}
