/**
 * RaceIQ Public — PRD Section 13.1's cinematic splash page, built on the
 * design system template §7 scrollytelling pattern.
 *
 * A 600vh container wraps a `position: sticky` 100vh viewport: scrolling
 * moves through six scenes that cross-fade in place, with no
 * scroll-jacking and no JS-driven smooth-scroll interception (real
 * browser scroll the whole way).
 *
 * PRD 13.1 specifies video backdrops per scene. There are no video assets
 * in this repo and inventing stock footage would be a worse lie than
 * going without, so each scene instead gets a CSS-composed backdrop —
 * corner-anchored radial washes plus one scene-specific animated element.
 * Same crossfade mechanism, same scrim system, so dropping real clips in
 * later is a swap of the backdrop layer, not a rewrite.
 *
 * Every number quoted in the copy is a real, measured figure from this
 * project's own warehouse and model runs, not marketing filler.
 */

import type { CSSProperties, ReactNode } from "react";
import { Link } from "react-router-dom";
import { C, F, NUM } from "../design/tokens";
import { SPRING } from "../design/Animations";
import { EdgeBezel } from "../design/chrome";
import { DEFAULT_ACCENT } from "../design/theme";
import { useSectionScroll } from "./useSectionScroll";

const SECTIONS = 6;

/** Word-by-word reveal; a whole-line fade never reads as intentional. */
function WordLine({
  text,
  active,
  size = 76,
  color = C.text,
  delayBase = 0,
}: {
  text: string;
  active: boolean;
  size?: number;
  color?: string;
  delayBase?: number;
}) {
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: "0 0.32em" }}>
      {text.split(" ").map((word, i) => (
        <span
          key={`${word}-${i}`}
          style={{
            fontFamily: F.display,
            fontSize: size,
            lineHeight: 1.02,
            letterSpacing: "0.01em",
            color,
            display: "inline-block",
            opacity: active ? 1 : 0,
            animation: active ? `wordDrop 0.62s ${SPRING} both` : "none",
            animationDelay: `${delayBase + i * 0.09}s`,
          }}
        >
          {word}
        </span>
      ))}
    </div>
  );
}

function Eyebrow({ children, active }: { children: ReactNode; active: boolean }) {
  return (
    <div
      style={{
        fontFamily: F.mono,
        fontSize: 11,
        letterSpacing: "0.45em",
        textTransform: "uppercase",
        color: C.gold,
        marginBottom: 22,
        opacity: active ? 1 : 0,
        animation: active ? "fadeUp 0.5s ease both" : "none",
      }}
    >
      {children}
    </div>
  );
}

function Prose({ children, active, delay = 0.5 }: { children: ReactNode; active: boolean; delay?: number }) {
  return (
    <p
      style={{
        fontFamily: F.body,
        fontSize: 17,
        lineHeight: 1.75,
        color: C.muted,
        maxWidth: 560,
        marginTop: 26,
        opacity: active ? 1 : 0,
        animation: active ? "fadeUp 0.6s ease both" : "none",
        animationDelay: `${delay}s`,
      }}
    >
      {children}
    </p>
  );
}

function SceneStat({
  value,
  label,
  active,
  delay,
}: {
  value: string;
  label: string;
  active: boolean;
  delay: number;
}) {
  return (
    <div
      style={{
        opacity: active ? 1 : 0,
        animation: active ? `statPop 0.55s ${SPRING} both` : "none",
        animationDelay: `${delay}s`,
      }}
    >
      <div style={{ ...NUM, fontFamily: F.display, fontSize: 52, color: C.gold, lineHeight: 1 }}>{value}</div>
      <div
        style={{
          fontFamily: F.mono,
          fontSize: 9.5,
          letterSpacing: "0.24em",
          textTransform: "uppercase",
          color: C.faint,
          marginTop: 9,
        }}
      >
        {label}
      </div>
    </div>
  );
}

function SectionOverlay({ children, active }: { children: ReactNode; active: boolean }) {
  return (
    <div
      style={{
        position: "absolute",
        inset: 0,
        display: "flex",
        alignItems: "center",
        padding: "0 clamp(20px, 9vw, 150px)",
        opacity: active ? 1 : 0,
        pointerEvents: active ? "auto" : "none",
        transition: "opacity 0.55s ease",
      }}
    >
      <div style={{ width: "100%", maxWidth: 1180 }}>{children}</div>
    </div>
  );
}

/** Scene-specific backdrop: washes plus one animated element per scene. */
function Backdrop({ section }: { section: number }) {
  const washes = [
    `radial-gradient(70% 60% at 14% 22%, ${DEFAULT_ACCENT}24 0%, transparent 60%)`,
    `radial-gradient(70% 70% at 82% 30%, #E1060030 0%, transparent 62%)`,
    `radial-gradient(75% 65% at 20% 78%, #2563EB26 0%, transparent 62%)`,
    `radial-gradient(70% 60% at 76% 24%, ${DEFAULT_ACCENT}2E 0%, transparent 60%)`,
    `radial-gradient(70% 65% at 18% 30%, #16A34A22 0%, transparent 60%)`,
    `radial-gradient(90% 80% at 50% 50%, ${DEFAULT_ACCENT}1F 0%, transparent 66%)`,
  ];

  return (
    <div aria-hidden style={{ position: "absolute", inset: 0, overflow: "hidden" }}>
      {washes.map((wash, i) => (
        <div
          key={i}
          style={{
            position: "absolute",
            inset: 0,
            background: wash,
            opacity: section === i ? 1 : 0,
            transition: "opacity 700ms ease",
          }}
        />
      ))}

      {/* Timing-screen grid: the one persistent texture, subtle enough to
          sit behind text at every scene. */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          backgroundImage: `linear-gradient(${C.rule} 1px, transparent 1px), linear-gradient(90deg, ${C.rule} 1px, transparent 1px)`,
          backgroundSize: "88px 88px",
          maskImage: "radial-gradient(75% 65% at 50% 45%, #000 0%, transparent 78%)",
          WebkitMaskImage: "radial-gradient(75% 65% at 50% 45%, #000 0%, transparent 78%)",
        }}
      />

      {/* Permanent radial vignette layer (§7's scrim system). */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          background: "radial-gradient(120% 90% at 50% 50%, transparent 30%, rgba(0,0,0,0.72) 100%)",
        }}
      />
      {/* Per-section scrim, biased toward the side the copy sits on. */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          background:
            section === 5
              ? "linear-gradient(180deg, rgba(10,10,15,0.55) 0%, rgba(10,10,15,0.35) 100%)"
              : "linear-gradient(90deg, rgba(10,10,15,0.88) 0%, rgba(10,10,15,0.42) 55%, transparent 100%)",
          transition: "background 700ms ease",
        }}
      />
    </div>
  );
}

function NavDots({
  section,
  onJump,
}: {
  section: number;
  onJump: (i: number) => void;
}) {
  const names = ["Formation lap", "Lights out", "First corner", "Pit window", "Final laps", "Chequered flag"];
  return (
    <div
      style={{
        position: "absolute",
        right: "clamp(18px, 4vw, 54px)",
        top: "50%",
        transform: "translateY(-50%)",
        display: "flex",
        flexDirection: "column",
        gap: 10,
        alignItems: "flex-end",
        zIndex: 20,
      }}
    >
      {Array.from({ length: SECTIONS }).map((_, i) => (
        <button
          key={i}
          onClick={() => onJump(i)}
          aria-label={`Go to scene ${i + 1}: ${names[i]}`}
          aria-current={section === i}
          title={names[i]}
          style={{
            // The active dot GROWS into a dash — that's what makes the nav
            // readable as a progress indicator, not just a row of bullets.
            width: section === i ? 3 : 2,
            height: section === i ? 28 : 8,
            background: section === i ? C.gold : `${DEFAULT_ACCENT}33`,
            borderRadius: 2,
            border: "none",
            padding: 0,
            cursor: "pointer",
            transition: `all 400ms ${SPRING}`,
          }}
        />
      ))}
    </div>
  );
}

/** Static chrome: rendered ONCE, explicitly not wired to `section`. */
function BrandMark() {
  return (
    <div
      style={{
        position: "absolute",
        top: "clamp(20px, 4vh, 40px)",
        left: "clamp(20px, 9vw, 150px)",
        zIndex: 20,
        display: "flex",
        alignItems: "center",
        gap: 14,
      }}
    >
      <div style={{ fontFamily: F.display, fontSize: 25, letterSpacing: "0.1em", color: C.text }}>
        RACE<span style={{ color: C.gold }}>IQ</span>
      </div>
      <div style={{ width: 1, height: 17, background: C.edge }} />
      <div style={{ fontFamily: F.mono, fontSize: 9, letterSpacing: "0.3em", color: C.faint }}>
        STRATEGY INTELLIGENCE
      </div>
    </div>
  );
}

const linkButton = (primary: boolean): CSSProperties => ({
  fontFamily: F.mono,
  fontSize: 11,
  letterSpacing: "0.22em",
  textTransform: "uppercase",
  padding: "15px 30px",
  borderRadius: 3,
  textDecoration: "none",
  display: "inline-block",
  transition: "all 220ms",
  background: primary ? C.gold : "transparent",
  color: primary ? "var(--rq-on-accent, #000)" : C.text,
  border: `1px solid ${primary ? "transparent" : C.edge}`,
});

export function SplashPage() {
  const { wrapRef, section, progress, jumpTo } = useSectionScroll(SECTIONS);

  return (
    <div ref={wrapRef} style={{ height: `${SECTIONS * 100}vh`, position: "relative" }}>
      <div style={{ position: "sticky", top: 0, height: "100vh", overflow: "hidden", background: C.bg }}>
        <Backdrop section={section} />
        {/* Static chrome: rendered once, outside the per-section animation
            logic, explicitly NOT wired to `section` — a persistent frame
            around a moving scene, not part of the scene (§7). */}
        <EdgeBezel />
        <BrandMark />
        <NavDots section={section} onJump={jumpTo} />

        {/* Scene 1 — Formation lap */}
        <SectionOverlay active={section === 0}>
          <Eyebrow active={section === 0}>Formation lap</Eyebrow>
          <WordLine text="Race intelligence." active={section === 0} />
          <WordLine text="Built for the pit wall." active={section === 0} color={C.gold} delayBase={0.18} />
          <Prose active={section === 0}>
            A circuit-adaptive, team-aware strategy engine trained on nine seasons of real Formula 1
            telemetry — not a lap-time calculator with a pit-stop constant bolted on.
          </Prose>
          <div
            style={{
              display: "flex",
              gap: 16,
              marginTop: 40,
              opacity: section === 0 ? 1 : 0,
              animation: section === 0 ? "fadeUp 0.6s ease both" : "none",
              animationDelay: "0.72s",
            }}
          >
            <Link to="/pitwall" style={linkButton(true)}>
              Open the pit wall
            </Link>
            <a href="#scroll" onClick={(e) => { e.preventDefault(); jumpTo(1); }} style={linkButton(false)}>
              How it works
            </a>
          </div>
        </SectionOverlay>

        {/* Scene 2 — Lights out */}
        <SectionOverlay active={section === 1}>
          <Eyebrow active={section === 1}>Lights out</Eyebrow>
          <WordLine text="0.4 seconds" active={section === 1} size={92} color={C.gold} />
          <WordLine text="to decide." active={section === 1} delayBase={0.2} />
          <Prose active={section === 1}>
            Safety car deploys on lap 32. Pit now and you rejoin in traffic; stay out and you burn the
            tyre advantage you spent twenty laps building. Every rival is solving the same problem with
            the same public data — and every public model treats your car like the average car.
          </Prose>
        </SectionOverlay>

        {/* Scene 3 — First corner */}
        <SectionOverlay active={section === 2}>
          <Eyebrow active={section === 2}>First corner</Eyebrow>
          <WordLine text="Nine seasons." active={section === 2} />
          <WordLine text="Every lap." active={section === 2} color={C.gold} delayBase={0.18} />
          <Prose active={section === 2}>
            FastF1 telemetry, Ergast results, and circuit weather reconciled into a leakage-safe feature
            store — every feature computed strictly from information available at or before the lap it
            describes.
          </Prose>
          <div style={{ display: "flex", gap: 58, marginTop: 44, flexWrap: "wrap" }}>
            <SceneStat value="202,577" label="Laps ingested" active={section === 2} delay={0.55} />
            <SceneStat value="184" label="Races" active={section === 2} delay={0.66} />
            <SceneStat value="31" label="Circuits" active={section === 2} delay={0.77} />
            <SceneStat value="2018-26" label="Seasons" active={section === 2} delay={0.88} />
          </div>
        </SectionOverlay>

        {/* Scene 4 — Pit window */}
        <SectionOverlay active={section === 3}>
          <Eyebrow active={section === 3}>Pit window</Eyebrow>
          <WordLine text="Five thousand races," active={section === 3} size={64} />
          <WordLine text="before you call one." active={section === 3} size={64} color={C.gold} delayBase={0.2} />
          <Prose active={section === 3}>
            The engine enumerates every feasible remaining strategy, then runs each one through a Monte
            Carlo simulation sharing one set of safety-car and retirement draws — so candidates are
            compared under identical luck, not against each other's noise.
          </Prose>
          <div style={{ display: "flex", gap: 58, marginTop: 44, flexWrap: "wrap" }}>
            <SceneStat value="~89" label="Candidates / call" active={section === 3} delay={0.55} />
            <SceneStat value="5,000" label="Simulations each" active={section === 3} delay={0.66} />
            <SceneStat value="2.3s" label="End to end" active={section === 3} delay={0.77} />
          </div>
        </SectionOverlay>

        {/* Scene 5 — Final laps */}
        <SectionOverlay active={section === 4}>
          <Eyebrow active={section === 4}>Final laps</Eyebrow>
          <WordLine text="Your car." active={section === 4} />
          <WordLine text="Your strategy." active={section === 4} color={C.gold} delayBase={0.18} />
          <Prose active={section === 4}>
            Seven models run underneath: lap time, tyre degradation, pit timing, safety car probability,
            finishing position and win probability — plus a sequence model that scores a whole remaining
            race in one pass, because chaining a single-lap model across thirty laps compounds its own
            error into fiction.
          </Prose>
        </SectionOverlay>

        {/* Scene 6 — Chequered flag */}
        <SectionOverlay active={section === 5}>
          <div style={{ textAlign: "center", display: "flex", flexDirection: "column", alignItems: "center" }}>
            <Eyebrow active={section === 5}>Chequered flag</Eyebrow>
            <div style={{ display: "flex", justifyContent: "center" }}>
              <WordLine text="Call it with evidence." active={section === 5} size={68} />
            </div>
            <Prose active={section === 5} delay={0.4}>
              The pit wall dashboard runs against the live API — real races, real laps, real
              recommendations with the reasoning shown.
            </Prose>
            <div
              style={{
                display: "flex",
                gap: 16,
                marginTop: 40,
                justifyContent: "center",
                opacity: section === 5 ? 1 : 0,
                animation: section === 5 ? "fadeUp 0.6s ease both" : "none",
                animationDelay: "0.62s",
              }}
            >
              <Link to="/pitwall" style={{ ...linkButton(true), animation: "goldPulse 3.2s ease infinite" }}>
                Enter the pit wall
              </Link>
              <a
                href="https://github.com/srikarreddyram/raceiq"
                target="_blank"
                rel="noreferrer"
                style={linkButton(false)}
              >
                View the source
              </a>
            </div>
          </div>
        </SectionOverlay>

        {/* Scroll affordance, only while there's more to scroll. */}
        <div
          style={{
            position: "absolute",
            bottom: 30,
            left: "50%",
            transform: "translateX(-50%)",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: 9,
            opacity: section === SECTIONS - 1 ? 0 : 1,
            transition: "opacity 400ms ease",
            pointerEvents: "none",
          }}
        >
          <div style={{ fontFamily: F.mono, fontSize: 8.5, letterSpacing: "0.3em", color: C.faint }}>SCROLL</div>
          <div style={{ width: 1, height: 26, background: C.gold, animation: "scrollNudge 1.9s ease infinite" }} />
        </div>

        {/* Scrub bar — the only element that tracks continuous progress. */}
        <div style={{ position: "absolute", left: 0, right: 0, bottom: 0, height: 2, background: C.rule }}>
          <div
            style={{
              height: "100%",
              width: `${progress * 100}%`,
              background: C.gold,
              transition: "width 120ms linear",
            }}
          />
        </div>
      </div>
    </div>
  );
}
