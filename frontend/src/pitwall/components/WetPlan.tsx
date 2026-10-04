/**
 * The wet-race plan (race_plan/wet.py) on the Race Weekend page.
 *
 * When rain is likely it leads the page and the dry plan below it becomes
 * "if the track stays dry". It shows the chance of rain hour by hour
 * through the race, the tyre to start on, and the calls teams have
 * actually made once the track was wet — measured, with the number of
 * races behind each, rather than a simulated wet race the data can't
 * support (the module says why).
 */

import type { WetPlan } from "../../api/types";
import { Card, SectionLabel } from "../../design/primitives";
import { C, DISPLAY, F, NUM, compoundColor } from "../../design/tokens";

const pct = (v: number | null) => (v == null ? "—" : `${Math.round(v * 100)}%`);

const SOURCE_LABEL: Record<WetPlan["source"], string> = {
  forecast: "Forecast · Open-Meteo",
  typical: "Typical for this circuit",
  measured: "It was wet in this race",
  override: "Rain set by hand",
};

function RainHours({ plan }: { plan: WetPlan }) {
  if (!plan.hourly.length) return null;
  return (
    <div style={{ display: "flex", gap: 10, alignItems: "flex-end" }}>
      {plan.hourly.map((h, i) => {
        const p = h.rain_probability ?? 0;
        return (
          <div key={h.hour_utc} style={{ width: 64, textAlign: "center" }}>
            <div style={{ ...NUM, fontSize: 13, color: C.text, fontWeight: 700, marginBottom: 4 }}>{pct(h.rain_probability)}</div>
            <div style={{ height: 52, background: C.fill, borderRadius: 4, display: "flex", alignItems: "flex-end", overflow: "hidden" }}>
              <div style={{ width: "100%", height: `${Math.max(4, p * 100)}%`, background: compoundColor("WET"), opacity: 0.35 + 0.65 * p }} />
            </div>
            <div style={{ fontFamily: F.mono, fontSize: 10.5, fontWeight: 700, color: C.faint, marginTop: 5 }}>
              {i === 0 ? "START" : `+${i}H`}
            </div>
            <div style={{ fontFamily: F.mono, fontSize: 10.5, color: C.muted }}>
              {h.rain_mm == null ? "" : `${h.rain_mm.toFixed(1)} mm`}
            </div>
          </div>
        );
      })}
    </div>
  );
}

export function WetPlanCard({ plan }: { plan: WetPlan }) {
  const tyre = plan.start_tyre === "WET" ? "full wets" : "intermediates";
  return (
    <Card accent={compoundColor("WET")} style={{ padding: "22px 26px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 24, flexWrap: "wrap", alignItems: "flex-start" }}>
        <div>
          <SectionLabel style={{ marginBottom: 8 }}>
            {plan.leads ? "Wet race plan" : "If it rains"} · {SOURCE_LABEL[plan.source]}
          </SectionLabel>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <span
              style={{
                display: "inline-flex",
                alignItems: "center",
                justifyContent: "center",
                width: 38,
                height: 38,
                borderRadius: 19,
                border: `4px solid ${compoundColor(plan.start_tyre)}`,
                fontFamily: F.mono,
                fontWeight: 800,
                fontSize: 15,
                color: C.text,
              }}
            >
              {plan.start_tyre === "WET" ? "W" : "I"}
            </span>
            <div style={{ ...DISPLAY, fontSize: 24 }}>Start on {tyre}</div>
          </div>
          <div style={{ fontFamily: F.body, fontSize: 13.5, color: C.dim, marginTop: 10, lineHeight: 1.55, maxWidth: 560 }}>
            {plan.source === "measured"
              ? "This race was wet. These are the calls teams make once the track is wet, measured across every wet race since 2018."
              : plan.source === "override"
                ? "Planning for a wet race, as set above. These are the calls teams make once the track is wet, measured across every wet race since 2018."
                : `Chance of rain at the start ${pct(plan.chance_at_start)}, at some point in the race ${pct(plan.chance_during)}. ${
                    plan.leads ? "The plan below is for if the track stays dry." : "The dry plan above is the plan unless the track gets wet."
                  }`}
          </div>
        </div>
        <RainHours plan={plan} />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 14, marginTop: 18 }}>
        {plan.calls.map((c) => (
          <div key={c.title} style={{ background: C.raised, border: `1px solid ${C.edge}`, borderRadius: 8, padding: "14px 16px" }}>
            <div style={{ fontFamily: F.body, fontWeight: 800, fontSize: 14.5, color: C.text, marginBottom: 6 }}>{c.title}</div>
            <div style={{ fontFamily: F.body, fontSize: 13, color: C.dim, lineHeight: 1.6 }}>{c.text}</div>
          </div>
        ))}
      </div>

      <div style={{ fontFamily: F.body, fontSize: 12, color: C.muted, lineHeight: 1.6, marginTop: 14 }}>
        Why calls and not a simulated wet race: only about twenty races since 2018 were run mostly on wet-weather tyres, too few to
        fit wet pace and tyre wear the way the dry plan is fitted, and an hourly forecast can't say which laps will be wet — rain at
        one circuit is too local. Whether it's this plan or the dry one is decided on the day, looking at the track.
      </div>
    </Card>
  );
}
