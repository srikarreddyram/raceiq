# RaceIQ — Product Requirements Document

**Document Version:** 2.0
**Status:** Draft
**Classification:** Internal
**Owner:** Engineering & Data Science

---

## 1. Executive Summary

RaceIQ is an end-to-end Formula 1 race strategy intelligence platform. It ingests historical and live telemetry data, engineers race-level and circuit-geometry features, builds team-specific car characteristic profiles, trains and serves multiple machine learning models, and powers real-time pit strategy recommendations through a REST API and two-surface frontend.

The platform ships with two distinct frontend surfaces:

- **RaceIQ Public** — a cinematic marketing splash page built with sticky-scroll, video crossfade, and animated stat reveals. No animation libraries; raw CSS keyframes and React state.
- **RaceIQ Pit Wall** — a data-dense strategy dashboard built for race engineers. Real-time lap deltas, tyre degradation curves, strategy comparison panels, and the circuit-adaptive optimal strategy engine.

The strategy engine's core differentiator is threefold: it is **circuit-adaptive** (strategy logic changes based on learned circuit geometry and historical patterns), **team-specific** (recommendations account for each car's tyre temperature handling, downforce philosophy, and pace characteristics in the current season), and **track-map-aware** (circuit geometry reconstructed from GPS telemetry is used both as model input and as a visual canvas in the dashboard).

---

## 2. Problem Statement

Race strategy tools that treat all teams, all cars, and all circuits as equivalent are systematically wrong. A Mercedes that runs its tyres in a narrow cold operating window will degrade differently at a cool Interlagos versus a hot Bahrain — not just because of the track, but because of the car. A Red Bull with exceptional aero efficiency handles tyre temperature differently under safety car periods than a car with more mechanical grip dependency. A strategy that works for Ferrari's 2024 high-downforce setup at Monza is wrong for the same team at Silverstone.

Existing public analytics models ignore this. They predict strategy for "a driver" on "a compound" at "a circuit." RaceIQ predicts strategy for a specific car-driver-circuit-weather combination, using team characteristic profiles learned from the current season's data and seeded with known engineering priors.

Additionally, circuit geometry — the shape, corner profiles, elevation changes, and sector speed characteristics of each track — is a first-class input to the model, not just a circuit ID lookup. The track map is both a model feature and a visual element of the dashboard.

---

## 3. Goals

- Collect and store 12–15 seasons of F1 data from all available sources, producing clean analytics-ready datasets
- Reconstruct per-circuit track maps from FastF1 GPS telemetry and use them as both model input and dashboard visuals
- Build and maintain team car characteristic profiles per season, combining data-inferred behaviour and manually seeded engineering priors
- Engineer a comprehensive feature set covering drivers, tyres, circuits, track geometry, weather, team characteristics, and live race state
- Train, evaluate, and track six ML models: lap time prediction, tyre degradation, pit stop recommendation, safety car probability, final position prediction, and win probability
- Build a circuit-adaptive, team-aware optimal strategy engine powered by Monte Carlo simulation
- Serve predictions and recommendations through a REST API
- Retrain models nightly on new race data
- Ship two frontend surfaces: a public splash page and a data-dense pit wall dashboard

---

## 4. Non-Goals

- Real-time optical tracking or live car telemetry ingestion (requires FIA/team access)
- Actual tyre temperature sensor data (not publicly available)
- Wind tunnel, CFD, or factory-level car setup data (not public)
- Autonomous race strategy decisions — this is a decision support tool only
- Mobile application in v1
- Multi-team competitive telemetry sharing

---

## 5. Target Users

**Primary — Race Strategy Engineer:** Needs real-time strategy recommendations, team-specific pit window analysis, and tyre life projections during a race. Primary user of the Pit Wall dashboard.

**Secondary — Performance Engineer:** Uses the platform between race weekends to analyse historical telemetry, build circuit profiles, and evaluate team characteristic model drift between seasons.

**Secondary — Team Principal:** Consumes high-level race outcome probabilities and strategy summaries. Does not interact with raw model outputs.

**Tertiary — Fans and Fantasy F1 Players:** Consume the public splash page and a simplified read-only strategy view exposed via public API.

---

## 6. Data Sources

### 6.1 FastF1

Primary source for lap-level telemetry, session data, tyre compounds, and GPS-based track geometry reconstruction.

**Data retrieved:**
- Lap times, sector times, tyre compound per stint
- Pit stop laps and durations
- Driver position per lap
- Speed traces, throttle, brake, DRS, gear, RPM channels
- GPS coordinates per telemetry sample (used for track map reconstruction)
- Track status events: safety car, VSC, red flag

### 6.2 Ergast API

Supplementary historical source. Full modern F1 era coverage (1950 onward).

> ⚠️ **Deprecation Risk:** Ergast has signalled potential deprecation. Identify a fallback before building ingestion around it — likely the official F1 API or Jolpica mirror.

**Data retrieved:** Race results, grid positions, fastest laps, constructor standings, circuit metadata.

### 6.3 OpenF1

Near-real-time data for live race weekends. Supplements FastF1 to reduce recommendation latency during active sessions.

### 6.4 Weather API

External weather keyed to circuit GPS coordinates and session timestamps.

**Data retrieved:** Ambient temp, track temp, humidity, wind speed and direction, rain probability, rainfall events.

### 6.5 Technical Journalism (Scraped — Team Characteristics Seeding)

Engineering priors for team car profiles are partially seeded from structured scraping of F1 technical journalism. Sources include Racefans.net, The Race, and Autosport technical analysis articles.

**Data extracted (semi-structured, manually curated per season):**
- Downforce philosophy (low / medium / high)
- Tyre operating window descriptor (narrow-cold / wide / narrow-hot)
- Setup sensitivity flags (understeery, oversteery, aerodynamically sensitive)
- Known weaknesses (e.g. "struggles in slow corners", "poor tyre warmup in cool conditions")

This data is encoded as categorical and ordinal features in the Team Characteristics table and updated at the start of each season plus after major car upgrade packages.

---

## 7. Track Map System

The track map system serves two purposes simultaneously: it provides circuit geometry features to the ML models, and it renders interactive SVG track maps in the Pit Wall dashboard.

### 7.1 Track Map Reconstruction Pipeline

GPS coordinates are extracted from FastF1 telemetry at ~10Hz resolution. The reconstruction pipeline:

1. **Coordinate normalisation** — convert raw GPS (lat/lng) to a local Cartesian frame centred on the circuit
2. **Path smoothing** — apply a Savitzky-Golay filter to remove GPS noise while preserving corner shape
3. **Corner detection** — identify corners by local curvature maxima (second derivative of path direction). Classify by minimum speed through corner: slow (<120 km/h), medium (120–200 km/h), fast (>200 km/h)
4. **Sector boundary mapping** — overlay official sector boundaries from FastF1 session metadata
5. **Feature extraction** — derive per-circuit geometry features (see Section 8.3)
6. **SVG serialisation** — normalise the smoothed path to a viewport coordinate system and store as an SVG path string for dashboard rendering

The pipeline runs once per new circuit added to the calendar. Existing circuits are re-run at the start of each season to account for layout changes (e.g. Abu Dhabi 2021 redesign).

### 7.2 Track Map as Dashboard Visual

The SVG track map is rendered in the Pit Wall dashboard as an interactive canvas:

- **Sector colouring** — sectors highlighted by current pace delta vs. season average (green = gaining, red = losing)
- **Pit entry/exit overlay** — pit lane entry and exit points marked, with estimated pit lane delta annotated
- **DRS zone overlay** — DRS activation and detection zones rendered as coloured path segments
- **Corner classification overlay** — corner speed class (slow/medium/fast) shown as colour-coded nodes
- **Live position markers** — during active sessions, driver position markers update per lap

---

## 8. Team Car Characteristic Profiles

### 8.1 Architecture

Each team gets a `CarProfile` per season. Profiles are built from two layers:

**Layer 1 — Data-inferred characteristics** (computed from FastF1 telemetry and race results, updated after every race):

| Characteristic | How Inferred |
|---|---|
| `tyre_temp_sensitivity` | Correlation between track temperature and team's tyre degradation rate vs. field average |
| `tyre_warmup_rate` | Pace delta on laps 1–3 of a stint vs. laps 4–10 — cars that warm tyres slowly show more variance |
| `cold_tyre_pace_loss` | Lap time penalty on out-lap vs. next lap, per compound |
| `downforce_proxy` | Sector 2 (high-downforce) pace delta vs. Sector 1+3 (low-downforce) pace delta, relative to field |
| `aero_sensitivity` | Lap time variance in windy conditions vs. still conditions, relative to field |
| `degradation_rate` | Fitted tyre degradation curve steepness per compound per team per circuit type |
| `safety_car_restart_pace` | Pace on lap immediately after safety car restart vs. steady-state pace |
| `undercut_vulnerability` | Historical track position loss rate when rivals pit first at this circuit type |

**Layer 2 — Manually seeded engineering priors** (from technical journalism, updated per season and per major upgrade):

| Field | Type | Description |
|---|---|---|
| `downforce_philosophy` | categorical | `low` / `medium` / `high` |
| `tyre_operating_window` | categorical | `narrow_cold` / `wide` / `narrow_hot` |
| `setup_sensitivity` | ordinal 0–10 | How much performance changes with setup changes |
| `known_weaknesses` | text array | e.g. `["slow_corner_pace", "poor_tyre_warmup_cool_conditions"]` |
| `upgrade_package_laps` | int | Lap number of most recent major upgrade package this season |

### 8.2 How Car Profiles Feed the Strategy Engine

At inference time, the strategy engine selects the `CarProfile` for the requesting team and current season. The profile modifies strategy recommendations in the following ways:

- **Tyre compound selection** — teams with `narrow_cold` tyre operating windows are penalised for compound choices that don't reach operating temperature at the current track temperature
- **Stint length recommendation** — team-specific degradation curves replace the field-average degradation curve in the Tyre Degradation model
- **Undercut timing** — teams with high `undercut_vulnerability` get earlier recommended pit windows
- **Safety car restart strategy** — teams with high `safety_car_restart_pace` scores are recommended to extend stints further when SC probability is high, banking on the restart lap recovering positions
- **Compound operating window filter** — at circuits with track temperatures outside a team's known tyre operating window, the engine flags this as a risk and adjusts expected lap time projections accordingly

### 8.3 CarProfile Table Schema

```
team_id
season
constructor_name
tyre_temp_sensitivity          FLOAT
tyre_warmup_rate               FLOAT
cold_tyre_pace_loss            FLOAT
downforce_proxy                FLOAT
aero_sensitivity               FLOAT
degradation_rate_soft          FLOAT
degradation_rate_medium        FLOAT
degradation_rate_hard          FLOAT
safety_car_restart_pace        FLOAT
undercut_vulnerability         FLOAT
downforce_philosophy           VARCHAR   -- seeded
tyre_operating_window          VARCHAR   -- seeded
setup_sensitivity              INT       -- seeded
known_weaknesses               TEXT[]    -- seeded
last_inferred_update           TIMESTAMP
last_manual_update             TIMESTAMP
upgrade_package_lap            INT
```

---

## 9. Data Architecture

Data flows through a four-layer storage model:

**Raw Layer** — ingested data exactly as received. No transformations. Source of truth for pipeline replay.

**Bronze Layer** — schema enforcement, type casting, deduplication, null handling, ingestion timestamps.

**Silver Layer** — joins across entities, ID normalisation across sources, business logic derivations (stint number, gap-to-leader, track status flags).

**Gold Layer** — analytics-ready feature store. One row per lap per driver with all engineered features pre-computed and validated. Consumed by model training and serving.

### 9.1 Core Tables

**Drivers:** `driver_id`, `name`, `nationality`, `date_of_birth`, `constructor_id`, `season`

**Constructors:** `constructor_id`, `name`, `nationality`, `season`

**Races:** `race_id`, `season`, `round`, `circuit_id`, `date`, `name`

**Circuits:** `circuit_id`, `name`, `country`, `location`, `lap_length_km`, `corner_count`, `drs_zones`, `pit_lane_delta_seconds`, `overtaking_difficulty_score`, `tyre_stress_index`, `track_evolution_rate`, `svg_path` (serialised track map)

**CircuitGeometry:** `circuit_id`, `corner_id`, `corner_type` (slow/medium/fast), `corner_x`, `corner_y`, `min_speed_kmh`, `braking_zone_start_m`, `traction_zone_end_m`, `sector_id`, `drs_zone_flag`, `elevation_m`

**CircuitProfiles:** `circuit_id`, `avg_winning_stops`, `dominant_compound_soft_pct`, `dominant_compound_medium_pct`, `dominant_compound_hard_pct`, `sc_probability_per_race`, `typical_undercut_window_laps`, `track_position_importance_score`

**CarProfiles:** (see Section 8.3)

**Laps:** `lap_id`, `race_id`, `driver_id`, `team_id`, `lap_number`, `lap_time_seconds`, `position`, `compound`, `tyre_age`, `stint_number`, `is_pit_lap`, `pit_stop_duration`, `track_status`, `gap_to_leader`

**Telemetry:** `telemetry_id`, `lap_id`, `timestamp`, `speed`, `throttle`, `brake`, `drs`, `gear`, `rpm`, `distance_on_lap`, `gps_lat`, `gps_lng`, `gps_x_local`, `gps_y_local`

**Weather:** `weather_id`, `race_id`, `lap_number`, `air_temp`, `track_temp`, `humidity`, `wind_speed`, `wind_direction`, `rain_probability`, `rainfall`

**PitStops:** `pitstop_id`, `race_id`, `driver_id`, `lap_number`, `compound_in`, `compound_out`, `stop_duration_seconds`

**TrackStatus:** `status_id`, `race_id`, `lap_number`, `status_type`, `duration_laps`

---

## 10. Feature Engineering

### 10.1 Driver Features

| Feature | Description |
|---|---|
| `avg_pace_delta` | Lap time delta vs. field average at this circuit |
| `qualifying_delta` | Gap to pole normalised by circuit length |
| `overtaking_score` | Historical positions gained per race at this circuit type |
| `tyre_conservation_index` | Degradation rate relative to teammate on same compound |
| `wet_weather_rating` | Performance delta in wet vs. dry (historical) |
| `aggression_score` | Incidents per race, weighted by fault attribution |
| `consistency_score` | Std deviation of lap times in clean air, normalised |

### 10.2 Tyre Features

| Feature | Description |
|---|---|
| `compound` | Soft / Medium / Hard / Inter / Wet |
| `tyre_age_laps` | Laps completed on this set |
| `degradation_rate` | Lap time delta per lap of tyre age, team-specific curve |
| `grip_estimate` | Pace delta vs. fresh-tyre baseline |
| `predicted_remaining_life` | Output of Tyre Degradation model |
| `in_operating_window` | Whether current track temp is inside team's tyre operating window |
| `operating_window_delta` | Degrees outside operating window (signed) |

### 10.3 Circuit Geometry Features

Derived from the track map reconstruction pipeline. These are the features that make circuit geometry a model input, not just a label.

| Feature | Description |
|---|---|
| `slow_corner_count` | Corners with min speed < 120 km/h |
| `fast_corner_count` | Corners with min speed > 200 km/h |
| `slow_corner_pct` | Slow corners as % of total — proxy for mechanical grip demand |
| `avg_corner_radius` | Mean radius across all corners |
| `total_braking_distance_m` | Sum of braking zone lengths per lap |
| `elevation_range_m` | Max elevation minus min elevation per lap |
| `elevation_variance` | Variance of elevation across telemetry samples — affects tyre load |
| `sector1_avg_speed` | Average speed through Sector 1 (typically low-downforce) |
| `sector2_avg_speed` | Average speed through Sector 2 (typically high-downforce) |
| `sector3_avg_speed` | Average speed through Sector 3 |
| `downforce_demand_index` | Weighted index from corner speed distribution — high = high downforce needed |
| `tyre_stress_index` | Circuit tyre degradation rate vs. global average |
| `track_evolution_rate` | Lap time improvement from lap 1 to lap 10 as rubber lays down |
| `pit_lane_delta` | Time lost in pit lane vs. lap time |
| `drs_zone_total_length_m` | Combined length of all DRS zones |

### 10.4 Team Car Features

| Feature | Description |
|---|---|
| `team_tyre_temp_sensitivity` | From CarProfile — correlation of track temp to degradation rate |
| `team_downforce_proxy` | Inferred from sector pace splits |
| `team_in_operating_window` | Whether current track temp suits this team's tyre window |
| `team_operating_window_delta` | Degrees outside team's optimal window (signed) |
| `team_cold_tyre_pace_loss` | Expected out-lap pace penalty for this team |
| `team_degradation_rate` | Team-specific fitted degradation curve for current compound |
| `team_undercut_vulnerability` | Historical probability of losing position when pitting second |
| `team_sc_restart_pace` | Pace recovery rate after safety car restarts |
| `downforce_circuit_match` | Dot product of team downforce philosophy and circuit downforce demand |

### 10.5 Weather Features

| Feature | Description |
|---|---|
| `air_temp`, `track_temp` | Direct |
| `humidity`, `wind_speed`, `wind_direction` | Direct |
| `rain_probability_next_10_laps` | Rolling forward-looking window |
| `rainfall_flag` | Binary |
| `condition_delta` | Deviation from historical average conditions at this circuit |
| `temp_vs_team_window` | Track temp minus team's optimal tyre operating temp (signed) |

### 10.6 Race State Features

| Feature | Description |
|---|---|
| `laps_remaining` | Derived from race length and current lap |
| `current_position` | Live or historical |
| `gap_to_car_ahead`, `gap_to_car_behind` | Seconds |
| `traffic_flag` | Gap < 1.0s |
| `safety_car_active` | Binary |
| `yellow_flag_sectors` | Bitmask |
| `rival_tyre_age` | Tyre age of car ahead |
| `rival_compound` | Compound of car ahead |
| `rival_team_id` | Enables rival-specific strategy inference |

---

## 11. Machine Learning Models

All models use strict temporal train/test splits (train on seasons 1–N, test on season N+1). All experiments tracked in MLflow.

### 11.1 Lap Time Prediction

**Task:** Regression — next lap time given current state.  
**Target:** `next_lap_time_seconds`  
**Algorithms:** LightGBM (primary), XGBoost, CatBoost  
**Evaluation:** RMSE, MAE  
**Key added inputs vs. v1:** `team_degradation_rate`, `team_in_operating_window`, `downforce_circuit_match`, `slow_corner_pct`, `elevation_variance`

---

### 11.2 Tyre Degradation

**Task:** Regression — predicted remaining tyre life in laps.  
**Target:** `predicted_remaining_life_laps`  
**Key added inputs:** `team_tyre_temp_sensitivity`, `team_operating_window_delta`, `tyre_stress_index`, `track_temp`

---

### 11.3 Pit Stop Recommendation

**Task:** Binary classification — pit this lap or stay out.  
**Target:** `should_pit` (0/1)  
**Evaluation:** F1 Score, Precision (prioritised)  
**Key added inputs:** `team_undercut_vulnerability`, `team_cold_tyre_pace_loss`, `downforce_circuit_match`, `typical_undercut_window`

---

### 11.4 Safety Car Probability

**Task:** Binary classification — SC within next N laps?  
**Target:** `safety_car_within_N_laps`  
**Evaluation:** AUC-ROC, F1 Score

---

### 11.5 Final Race Position

**Task:** Multi-class classification — predicted finishing position.  
**Target:** `final_position` (P1–P20)  
**Evaluation:** Top-3 accuracy, mean absolute position error

---

### 11.6 Win Probability

**Task:** Binary classification.  
**Target:** `wins_race` (0/1)  
**Evaluation:** Log-loss, AUC-ROC

---

## 12. Circuit-Adaptive, Team-Aware Optimal Strategy Engine

### 12.1 Overview

The strategy engine combines circuit profile, track geometry features, and team car characteristics to produce a strategy recommendation that is specific to this car on this track under these conditions — not a generic recommendation for "a driver."

### 12.2 Inputs

```
circuit_id
team_id
driver_id
current_lap
tyre_compound
tyre_age_laps
current_position
gap_to_car_ahead
gap_to_car_behind
laps_remaining
weather_state
available_compounds
rival_states: [{ driver_id, team_id, compound, tyre_age, position }]
```

### 12.3 Strategy Search

Candidate strategies are enumerated as sequences of `(pit_lap, compound)` pairs from current lap to race end.

Candidates are filtered by:
- Tyre life feasibility using **team-specific** degradation curves
- Compound usage rules (two-compound minimum in dry)
- Tyre operating window feasibility — compounds that won't reach this team's operating temperature at the current track temperature are flagged and deprioritised
- Circuit-specific constraints (track position penalty applied at high `track_position_importance` circuits like Monaco and Singapore)
- `downforce_circuit_match` — strategies requiring strong pace in circuit sectors that don't match the team's strength profile are penalised

### 12.4 Monte Carlo Simulation

For each candidate strategy, N=5,000 simulations:

1. Sample stochastic variables — SC deployment (circuit-specific probability), weather changes, rival pit windows — from model distributions
2. Simulate lap-by-lap using Lap Time model (with team-specific inputs) as oracle and Tyre Degradation model (team-specific curve) as constraint
3. Apply team-specific out-lap pace penalty (`cold_tyre_pace_loss`) on laps immediately after pit stops
4. Apply circuit pit lane delta, traffic interactions, track position changes
5. Record final position

### 12.5 Scoring Function

```
Strategy Score = (Expected Points × 0.6) + (Podium Probability × 0.3) + (Win Probability × 0.1)
```

Weights configurable per team objective (championship leader vs. midfield points hunter).

### 12.6 Recommendation Output Example

```json
{
  "circuit": "Bahrain International Circuit",
  "team": "Oracle Red Bull Racing",
  "season": "2024",
  "car_profile_snapshot": {
    "tyre_operating_window": "wide",
    "downforce_philosophy": "high",
    "current_track_temp": 42,
    "team_in_operating_window": true,
    "operating_window_delta": 0,
    "downforce_circuit_match": 0.81,
    "team_degradation_rate_medium": 0.043
  },
  "circuit_profile": {
    "overtaking_difficulty": 28,
    "tyre_stress_index": 74,
    "track_position_importance": 41,
    "dominant_strategy": "2-stop",
    "historical_sc_rate": 0.44,
    "slow_corner_pct": 0.31,
    "downforce_demand_index": 0.74
  },
  "recommended_strategy": {
    "action": "Pit on Lap 18",
    "compound": "Hard",
    "remaining_stints": [
      { "stint": 2, "compound": "Hard", "laps": 18, "to_lap": 36 },
      { "stint": 3, "compound": "Medium", "laps": 21, "to_lap": 57 }
    ],
    "expected_finish": "P2",
    "expected_points": 18,
    "win_probability": 0.24,
    "podium_probability": 0.67,
    "risk_score": 11.2,
    "strategy_score": 0.812
  },
  "reasoning": [
    "Undercut window opens vs P1 in 2 laps — Bahrain historically rewards early undercuts",
    "RB20 degradation curve on Medium projects cliff in 3 laps at 42°C track temp",
    "RB20 in operating window — no compound warming penalty expected on Hard",
    "Downforce-circuit match 0.81 — car well-suited to Sector 2 pace at this circuit",
    "SC probability 44% — Hard compound viable across all plausible SC scenarios",
    "Rain probability low for remainder of race"
  ],
  "alternatives": [
    {
      "action": "Extend to Lap 22, pit on Hard",
      "expected_finish": "P3",
      "expected_points": 15,
      "win_probability": 0.11,
      "podium_probability": 0.54,
      "risk_score": 8.1,
      "reasoning": "Lower variance. Loses undercut window but reduces traffic exposure."
    }
  ]
}
```

---

## 13. Frontend — Two Surfaces

### 13.1 RaceIQ Public (Splash Page)

A cinematic marketing frontend. No animation libraries — raw CSS keyframes and React state.

**Technical architecture (from reference implementation):**
- Sticky-scroll wrapper: `600vh` tall container with `position: sticky; top: 0` inner viewport — creates 6 scroll-driven "scenes" from one screen
- Scroll listener normalises `rect.top` to 0–1 and maps to section index; uses a ref mirror to avoid re-rendering on every scroll pixel — only 6 React renders per full scroll
- Video backdrop: all clips mounted simultaneously, switched by `opacity` not mount/unmount — no flash, crossfade free. Lazy-arming loads only current ± 1 clip's `src`; all others show poster
- CSS color grade applied per clip via `filter` string; gold `mix-blend-mode: overlay` unifies all clips into one visual identity
- WordLine component: word-by-word reveal with `translateY` + `blur` keyframe, staggered by `i * 0.09s`
- Two keyframe primitives cover all text motion: `fadeUp` (prose, subtle) and `statPop` (numbers, springy overshoot via `cubic-bezier(0.34,1.56,0.64,1)`)
- Scrim system: per-section gradient scrims angle-matched to text alignment; radial vignette layer permanent
- Nav dots: pill/dot toggle driven by CSS transition on `width`/`height`, click scrolls to computed section offset

**Six scenes map to RaceIQ narrative:**
1. Formation lap — platform intro, "Race intelligence. Built for the pit wall."
2. Lights out — the problem statement, "0.4 seconds to decide."
3. First corner — the data system, circuit map visual
4. Pit window — the strategy engine, live recommendation demo
5. Final laps — team characteristic profiles, "Your car. Your strategy."
6. Chequered flag — CTA, "Request access."

**Typography and palette:**
- Display: Bebas Neue (headlines, stats, numbers)
- Body: Inter
- Data/labels: JetBrains Mono (reads as instrumentation, not marketing)
- Background: `#0a0a0f`
- Accent: `#C9A84C` (gold) — all highlights, CTAs, active states
- Secondary text: `#9aa7bd`
- Danger/negative: `#DC2626`

### 13.2 RaceIQ Pit Wall (Strategy Dashboard)

A data-dense engineering dashboard. Entirely separate from the splash page — different route, different design language, same API backend.

**Views:**

**Race View** — live or historical race state. Positions, gaps, compounds, lap-by-lap pace delta chart, current strategy recommendation with simulation outcome distribution.

**Circuit View** — interactive SVG track map for the selected circuit. Sector colouring by pace delta, DRS zone overlay, corner classification nodes, pit entry/exit markers. Team-specific pace heatmap by sector overlaid on map.

**Car Profile View** — team car characteristic profile for the current season. Inferred metrics with confidence intervals, manually seeded priors, tyre operating window visualisation vs. current and historical track temps.

**Tyre View** — team-specific degradation curves per compound. Predicted remaining life. Compound operating window range vs. current track temp.

**Strategy Simulation View** — candidate strategies side-by-side. Outcome distributions as box plots. Risk score vs. expected points scatter.

**Driver View** — per-driver performance profile, circuit history, wet/dry splits.

**Model Performance View** — live metrics, prediction vs. actual overlays, drift indicators per model, car profile inference confidence over the season.

---

## 14. API Layer

| Method | Endpoint | Description |
|---|---|---|
| GET | `/circuits` | All circuits with geometry and profile data |
| GET | `/circuits/{id}/map` | SVG track map with overlay data |
| GET | `/circuits/{id}/profile` | Full circuit profile including strategy priors |
| GET | `/teams/{id}/car-profile` | Team car profile for current or specified season |
| GET | `/drivers` | Driver list with metadata |
| GET | `/races` | Race list with circuit and date |
| GET | `/telemetry/{lap_id}` | Telemetry channels for a lap |
| POST | `/predict/laptime` | Lap time prediction (team-aware) |
| POST | `/predict/strategy` | Full strategy recommendation |
| POST | `/predict/strategy/optimal` | Circuit-adaptive, team-specific optimal strategy |
| POST | `/simulate` | Monte Carlo simulation for given strategies |
| POST | `/retrain` | Trigger manual retraining (authenticated) |
| PUT | `/teams/{id}/car-profile/seed` | Update manually seeded car profile priors (authenticated) |

---

## 15. MLOps & Nightly Retraining

1. Check for new completed sessions
2. Download and validate raw data from all sources
3. Apply Bronze → Silver → Gold transformations
4. Run Great Expectations data quality checks — abort on critical failures
5. Recompute features for affected races
6. **Update CarProfiles** — re-infer team characteristics from new race data
7. Update circuit profiles with new race data
8. Retrain all six models
9. Evaluate against held-out test season
10. Promote if metrics improve or hold within tolerance; log alert otherwise
11. Update MLflow model registry

Orchestrated via Airflow. dbt for Silver/Gold transformations. Docker for reproducibility.

---

## 16. Monitoring

| Signal | Threshold |
|---|---|
| Prediction drift (KL divergence) | > 0.1 |
| Data drift (PSI on key features) | > 0.2 |
| Missing features | > 2% null on critical features |
| CarProfile staleness | > 2 races since last inferred update |
| Circuit profile staleness | > 14 days before a race at that circuit |
| API latency P95 | > 500ms |
| API error rate | > 1% over 5 minutes |
| Simulation throughput | > 10s for 5,000 simulations |
| Model accuracy degradation | > 10% vs. baseline |

---

## 17. Tech Stack

| Layer | Tools |
|---|---|
| Data collection | `fastf1`, `requests`, Ergast, OpenF1, Weather API |
| Track map reconstruction | `numpy`, `scipy` (Savitzky-Golay), `shapely`, `svgwrite` |
| Data processing | `pandas`, `polars`, `pyarrow`, `duckdb` |
| Data transformation | `dbt` |
| Data validation | `great_expectations` |
| Storage | PostgreSQL |
| ML training | `lightgbm`, `xgboost`, `catboost`, `scikit-learn` |
| Hyperparameter tuning | `optuna` |
| Experiment tracking | `mlflow` |
| Explainability | `shap` |
| Pipeline orchestration | `airflow` |
| Serving | `fastapi`, `redis` |
| Frontend — Splash | React, raw CSS keyframes, no animation libraries |
| Frontend — Pit Wall | React, `plotly`, `recharts` |
| Containerisation | `docker`, `docker-compose` |
| CI/CD | GitHub Actions |
| Infrastructure (v2) | AWS, Terraform |
| Language | Python 3.10+ (backend), TypeScript (frontend) |

---

## 18. Repository Structure

```
raceiq/
├── ingestion/
│   ├── fastf1/
│   ├── ergast/
│   ├── openf1/
│   └── weather/
├── pipelines/
│   ├── bronze/
│   ├── silver/
│   └── gold/
├── track_maps/
│   ├── reconstruction/       # GPS → smoothed path → SVG
│   ├── geometry_features/    # Corner detection, sector profiles
│   └── assets/               # Stored SVG track maps per circuit
├── car_profiles/
│   ├── inference/            # Data-inferred characteristic computation
│   ├── seeding/              # Manual prior ingestion and validation
│   └── profiles/             # Stored CarProfile per team per season
├── feature_store/
│   ├── definitions/
│   └── validation/
├── models/
│   ├── lap_time/
│   ├── tyre_degradation/
│   ├── pit_stop/
│   ├── safety_car/
│   ├── race_position/
│   └── win_probability/
├── strategy_engine/
│   ├── search/
│   ├── simulation/
│   ├── scoring/
│   └── recommendation/
├── serving/
│   └── api/
├── frontend/
│   ├── splash/               # Public marketing page
│   └── pitwall/              # Strategy dashboard
├── monitoring/
├── tests/
├── docker/
├── dbt/
├── airflow/
└── docs/
    ├── PRD.md
    ├── ARCHITECTURE.md
    └── API_SPEC.md
```

---

## 19. Success Criteria

- Lap Time Prediction RMSE below 0.4s on held-out test season, measured per-team (not just global average)
- Tyre Degradation model predicts remaining life within 2 laps on average, using team-specific curves
- Team-specific features provide measurable lift vs. team-agnostic baseline on held-out season
- Pit Stop Recommendation precision above 0.75
- Safety Car Probability AUC-ROC above 0.70
- Track map reconstruction pipeline produces clean SVG for all 24 current calendar circuits
- Circuit geometry features provide measurable lift vs. circuit-ID-only baseline
- Strategy recommendation API responds within 500ms at P95
- Simulation engine completes 5,000 scenarios in under 10 seconds
- CarProfile inference updates within 2 hours of race completion
- Nightly retraining pipeline runs without manual intervention
- Zero data leakage on all temporal splits
- Splash page renders at 60fps on mid-range hardware with six video clips

---

## 20. Open Questions

- **Tyre operating window cold start:** For new teams (e.g. a new constructor entering the calendar), there's no historical data to infer the tyre operating window. Define fallback — use constructor's engine supplier's historical profile, or fall back to field average?
- **CarProfile update cadence after major upgrades:** A significant upgrade package can change a car's characteristics mid-season. How many races of new data are needed before the inferred profile is trusted over the pre-upgrade profile?
- **Rival team strategy inference:** The engine receives `rival_states` including `rival_team_id`. Should rival strategies also be modelled using the rival team's CarProfile (more accurate) or remain team-agnostic (simpler, avoids modelling error compounding)?
- **Track map circuit coverage:** FastF1 GPS telemetry quality varies by circuit and session. Define the minimum telemetry sample threshold required for a clean reconstruction, and the fallback for circuits below it.
- **Splash page video sourcing:** Six video clips are needed for the marketing page. Source, licensing, and color grading pipeline to be confirmed.
- **Simulation compute under live race conditions:** 5,000 simulations × multiple candidate strategies × P95 <500ms may require pre-computation or simulation depth reduction during live sessions. Confirm acceptable latency vs. accuracy tradeoff.
- **Dashboard hosting:** VPN-gated internal only, or externally accessible for fan/fantasy use case?