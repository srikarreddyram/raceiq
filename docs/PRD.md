# RaceIQ — Product Requirements Document

**Document Version:** 1.0  
**Status:** Draft  
**Classification:** Internal  
**Owner:** Engineering & Data Science

---

## 1. Executive Summary

RaceIQ is an end-to-end Formula 1 race strategy intelligence platform. It ingests historical and live telemetry data, engineers race-level features, trains and serves multiple machine learning models, and powers real-time pit strategy recommendations through a REST API and interactive dashboard.

The platform is designed to assist race strategy engineers in making faster, data-backed decisions during a race weekend — covering pit timing, tyre selection, undercut and overcut windows, safety car responses, and final position projections.

The platform's centrepiece is a **circuit-adaptive optimal strategy engine**: given any track, any weather conditions, and any race state, it computes the highest expected-points strategy for a driver from that moment to the chequered flag.

---

## 2. Problem Statement

A race strategy engineer makes hundreds of high-stakes decisions across a race weekend, often under extreme time pressure with incomplete information. Decisions such as when to pit, which compound to switch to, whether an undercut on a rival is viable, or whether a set of tyres can survive another 12 laps are currently made by combining experience, gut feel, and whatever data the team's existing tooling surfaces in the moment.

These decisions draw on a large set of interdependent variables: historical race patterns at the specific circuit, live telemetry, evolving weather conditions, tyre degradation curves, driver behaviour under pressure, and competitor strategies. No single engineer can hold all of this in their head simultaneously.

A platform that ingests this data, models the relevant relationships, and surfaces ranked strategy options with explicit reasoning reduces the cognitive load on the engineer and improves the quality of in-race decisions. Critically, this reasoning must be **circuit-aware** — a strategy that works at Monaco (low degradation, impossible overtaking, track position everything) is catastrophically wrong at Bahrain (high degradation, easy overtaking, undercuts decisive).

---

## 3. Goals

The platform must:

- Collect and store historical F1 data across multiple seasons from all available sources, producing a clean analytics-ready dataset
- Engineer a comprehensive feature set covering drivers, tyres, circuits, weather, and live race state
- Train, evaluate, and track six purpose-built ML models covering lap time prediction, tyre degradation, pit stop recommendation, safety car probability, final position prediction, and win probability
- Build a **circuit-adaptive optimal strategy engine** that recommends the highest-EP strategy for any track under any conditions
- Run Monte Carlo simulations across thousands of race scenarios to quantify outcome distributions for alternative strategy choices
- Surface ranked pit strategy recommendations with explicit reasoning through a REST API
- Retrain models automatically on a nightly schedule when new race data becomes available
- Monitor prediction drift, data quality, and API health continuously
- Expose all outputs through an interactive dashboard accessible to engineers and analysts

---

## 4. Non-Goals

- RaceIQ does not replace the race strategy engineer. It is a decision support tool, not an autonomous decision-making system.
- Live video or broadcast feed ingestion is out of scope.
- Driver contract, commercial, or regulatory compliance data is out of scope.
- Multi-team competitive intelligence (accessing other teams' telemetry) is out of scope.
- Mobile application development is out of scope for v1.

---

## 5. Target Users

**Primary — Race Strategy Engineer:** The core user. Needs real-time strategy recommendations, pit window analysis, and tyre life projections during a race. Interacts primarily through the dashboard and API during race weekends.

**Secondary — Performance Engineer:** Uses the platform between race weekends to analyse historical telemetry, evaluate model performance, and prepare circuit-specific feature profiles.

**Secondary — Team Principal:** Consumes high-level race outcome probabilities and strategy summaries. Does not interact with raw model outputs.

**Tertiary — Fans and Fantasy F1 Players:** External-facing, lower-priority use case. Would consume a simplified public API surface or read-only dashboard view if exposed in a future version.

---

## 6. Data Sources

### 6.1 FastF1

Primary source for lap-level telemetry, session data, tyre compounds, and timing. Provides Python API access to official FIA timing data and onboard telemetry channels. Covers 2018 onward with full telemetry; partial coverage for earlier seasons.

**Data retrieved:** Lap times, sector times, tyre compound per stint, pit stop laps, driver position per lap, speed traces, throttle/brake/DRS channels, track status events (safety car, VSC, red flag).

### 6.2 Ergast API

Supplementary historical source covering the full modern F1 era (1950 onward). Used for race results, qualifying results, constructor standings, and circuit metadata where FastF1 coverage is insufficient.

> ⚠️ **Deprecation Risk:** Ergast has signalled potential deprecation. Identify a fallback (official F1 API or community mirror) before building ingestion around it.

**Data retrieved:** Race results, grid positions, fastest laps, constructor points, circuit coordinates and characteristics.

### 6.3 OpenF1

Supplementary real-time and near-real-time data source for live race weekends. Used alongside FastF1 during active sessions to reduce latency on live telemetry ingestion.

### 6.4 Weather API

External weather data keyed to circuit coordinates and session timestamps. Used to build weather features at lap-level granularity.

**Data retrieved:** Ambient temperature, track temperature, humidity, wind speed and direction, rain probability, actual rainfall events.

---

## 7. Data Architecture

Data flows through a four-layer storage model:

**Raw Layer** stores ingested data exactly as received from the source — no transformations, no schema enforcement. Serves as the source of truth for replaying any pipeline step.

**Bronze Layer** applies schema enforcement, type casting, deduplication, and basic null handling. Records are traceable back to their source with ingestion timestamps.

**Silver Layer** applies joins across entities, normalises identifiers across data sources (e.g., driver IDs between FastF1 and Ergast differ), and applies business logic for derived fields such as stint number, gap-to-leader, and track status flags.

**Gold Layer** is the analytics-ready feature store consumed by model training and the serving layer. Contains one row per lap per driver, with all engineered features pre-computed and validated.

### 7.1 Core Tables

**Drivers:** `driver_id`, `name`, `nationality`, `date_of_birth`, `constructor_id`, `season`

**Constructors:** `constructor_id`, `name`, `nationality`, `season`

**Races:** `race_id`, `season`, `round`, `circuit_id`, `date`, `name`

**Circuits:** `circuit_id`, `name`, `country`, `location`, `lap_length_km`, `corner_count`, `drs_zones`, `pit_lane_delta_seconds`, `overtaking_difficulty_score`, `tyre_stress_index`, `track_evolution_rate`

**Laps:** `lap_id`, `race_id`, `driver_id`, `lap_number`, `lap_time_seconds`, `position`, `compound`, `tyre_age`, `stint_number`, `is_pit_lap`, `pit_stop_duration`, `track_status`, `gap_to_leader`

**Telemetry:** `telemetry_id`, `lap_id`, `timestamp`, `speed`, `throttle`, `brake`, `drs`, `gear`, `rpm`, `distance_on_lap`

**Weather:** `weather_id`, `race_id`, `lap_number`, `air_temp`, `track_temp`, `humidity`, `wind_speed`, `wind_direction`, `rain_probability`, `rainfall`

**PitStops:** `pitstop_id`, `race_id`, `driver_id`, `lap_number`, `compound_in`, `compound_out`, `stop_duration_seconds`

**TrackStatus:** `status_id`, `race_id`, `lap_number`, `status_type` (SC, VSC, Yellow, Red, Clear), `duration_laps`

**CircuitProfiles:** `circuit_id`, `avg_winning_stops`, `dominant_compound_soft_pct`, `dominant_compound_medium_pct`, `dominant_compound_hard_pct`, `sc_probability_per_race`, `typical_undercut_window_laps`, `track_position_importance_score`

---

## 8. Feature Engineering

All features are pre-computed in the Gold Layer and versioned in the feature store. Features are grouped by entity.

### 8.1 Driver Features

| Feature | Description |
|---|---|
| `avg_pace_delta` | Average lap time delta vs. field average at this circuit |
| `qualifying_delta` | Gap to pole in qualifying, normalised by circuit length |
| `overtaking_score` | Historical positions gained per race at this circuit type |
| `tyre_conservation_index` | Degradation rate relative to teammate on same compound |
| `wet_weather_rating` | Performance delta in wet vs. dry conditions (historical) |
| `aggression_score` | Incidents per race, weighted by fault attribution |
| `consistency_score` | Standard deviation of lap times in clean air, normalised |

### 8.2 Tyre Features

| Feature | Description |
|---|---|
| `compound` | Soft / Medium / Hard / Inter / Wet |
| `tyre_age_laps` | Number of laps completed on this set |
| `stint_length` | Total planned or observed stint length |
| `degradation_rate` | Lap time delta per lap of tyre age, fitted per stint |
| `grip_estimate` | Derived from pace delta vs. fresh-tyre baseline |
| `predicted_remaining_life` | Output of Tyre Degradation model (Section 9.2) |

### 8.3 Circuit Features

These are the key inputs to the circuit-adaptive strategy engine. Each circuit gets its own profile, learned from historical races at that track.

| Feature | Description |
|---|---|
| `overtaking_difficulty` | Index derived from historical position change data (0–100) |
| `pit_lane_delta` | Time lost in pit lane vs. lap time at this circuit |
| `drs_zones` | Number of DRS activation zones |
| `corner_count` | Total corners per lap |
| `tyre_stress_index` | How aggressively this circuit degrades tyres (0–100) |
| `track_evolution_rate` | How much grip increases lap-by-lap as rubber is laid down |
| `track_position_importance` | How much finishing position correlates with track position vs. pace |
| `historical_sc_rate` | Safety car appearances per race at this circuit (historical) |
| `typical_undercut_window` | Historical laps-ahead gap at which undercuts succeeded here |
| `dominant_strategy_stops` | Modal number of pit stops in winning strategies at this circuit |

### 8.4 Weather Features

| Feature | Description |
|---|---|
| `air_temp`, `track_temp` | Direct from weather source |
| `humidity` | Affects tyre performance and grip |
| `wind_speed`, `wind_direction` | Affects aero balance |
| `rain_probability_next_10_laps` | Rolling forward-looking probability window |
| `rainfall_flag` | Binary; derived from actual rainfall data |
| `condition_delta` | Deviation from historical average conditions at this circuit |

### 8.5 Race State Features

| Feature | Description |
|---|---|
| `laps_remaining` | Derived from race length and current lap |
| `current_position` | Live or historical position |
| `gap_to_car_ahead` | Seconds; determines undercut window viability |
| `gap_to_car_behind` | Seconds; determines overcut vulnerability |
| `traffic_flag` | Whether driver is stuck in traffic (gap < 1.0s) |
| `safety_car_active` | Binary; from track status table |
| `yellow_flag_sectors` | Bitmask of active yellow flag sectors |
| `rival_tyre_age` | Tyre age of the car ahead — key for undercut timing |
| `rival_compound` | Compound of car ahead — determines pace delta window |

---

## 9. Machine Learning Models

All models are trained using strict temporal splits — training on seasons 1 through N, validation on season N, test on season N+1. No random splits are used anywhere in the pipeline. All experiments are tracked in MLflow including hyperparameters, feature sets, evaluation metrics, and model artifacts.

### 9.1 Lap Time Prediction

**Task:** Regression — predict the lap time for the next lap given current race state and features.

**Target:** `next_lap_time_seconds`

**Algorithms:** LightGBM (primary), XGBoost, CatBoost (comparison)

**Evaluation:** RMSE, MAE

**Key features:** `tyre_age_laps`, `compound`, `fuel_load_estimate`, `gap_to_car_ahead`, `track_temp`, `driver_avg_pace_delta`, `safety_car_active`, `tyre_stress_index`, `track_evolution_rate`

**Use cases:** Feeds into strategy simulation as the core lap time oracle. Used to project stint pace under alternative tyre strategies.

---

### 9.2 Tyre Degradation

**Task:** Regression — predict remaining tyre life in laps given current stint state.

**Target:** `predicted_remaining_life_laps`

**Key features:** `compound`, `tyre_age_laps`, `track_temp`, `driver_tyre_conservation_index`, `rain_probability`, `tyre_stress_index`

**Evaluation:** RMSE on remaining life prediction

**Use cases:** Feeds into pit stop recommendation to determine whether a driver can extend their stint. Feeds into the optimal strategy engine as a hard constraint on strategy viability.

---

### 9.3 Pit Stop Recommendation

**Task:** Binary classification — pit this lap or stay out.

**Target:** `should_pit` (0/1)

**Key features:** `tyre_age_laps`, `predicted_remaining_life_laps`, `gap_to_car_ahead`, `gap_to_car_behind`, `pit_lane_delta`, `laps_remaining`, `safety_car_active`, `rain_probability_next_10_laps`, `track_position_importance`, `typical_undercut_window`

**Evaluation:** F1 Score, Precision, Recall. Precision is prioritised — a false positive (unnecessary pit call) is more costly than a false negative in most race scenarios.

---

### 9.4 Safety Car Probability

**Task:** Binary classification — will a safety car be deployed within the next N laps?

**Target:** `safety_car_within_N_laps` (binary, N configurable — default 5)

**Key features:** `historical_sc_rate_this_circuit`, `laps_remaining`, `current_lap`, `incident_rate_this_race`, `rain_probability`, `gap_between_cars`

**Evaluation:** AUC-ROC, F1 Score

**Use cases:** Feeds into simulation engine as a stochastic event. A high SC probability significantly shifts the optimal pit window — pitting under SC is free track position.

---

### 9.5 Final Race Position

**Task:** Multi-class classification — predict the final finishing position for a driver.

**Target:** `final_position` (P1 through P20)

**Algorithms:** LightGBM with softmax objective

**Evaluation:** Top-3 accuracy, mean absolute position error

**Key features:** `current_position`, `gap_to_leader`, `laps_remaining`, `tyre_compound`, `tyre_age_laps`, `safety_car_active`, `driver_overtaking_score`, `overtaking_difficulty`

---

### 9.6 Win Probability

**Task:** Binary classification — does this driver win the race?

**Target:** `wins_race` (0/1)

**Evaluation:** Log-loss, AUC-ROC

**Use cases:** Headline output on the dashboard. Used in the optimal strategy engine to compare win probability across alternative strategies.

---

## 10. Circuit-Adaptive Optimal Strategy Engine

This is the platform's core differentiator. Given any circuit, any weather conditions, and any current race state, it computes the optimal sequence of pit stops and tyre compounds that maximises expected championship points from that moment to the chequered flag.

### 10.1 Why Circuit-Adaptive Matters

A one-size-fits-all strategy model fails because circuits are fundamentally different strategic environments:

| Circuit Type | Strategic Logic |
|---|---|
| **Monaco** (street, low overtaking) | Track position is everything. Pit as late as possible. Avoid pitting under green flag at all costs. One-stop almost always dominant. |
| **Bahrain** (high degradation, easy overtaking) | Undercuts are decisive. Pit early to jump rivals. Two-stop often optimal. Fresh tyres = positions gained. |
| **Spa** (weather-variable, long lap) | Safety car probability high. Weather window management critical. Intermediate/wet compound timing is race-defining. |
| **Singapore** (street, SC-heavy) | Safety car probability extremely high. Strategy built around SC windows. Track position matters almost as much as Monaco. |

The engine learns these profiles from historical race data at each circuit and adjusts its strategy search accordingly.

### 10.2 Inputs

```
circuit_id
driver_id
current_lap
tyre_compound
tyre_age_laps
current_position
gap_to_car_ahead
gap_to_car_behind
laps_remaining
weather_state
available_compounds        # what sets remain in the allocation
rival_states               # compound + tyre age for cars around driver
```

### 10.3 Strategy Search

The engine enumerates all viable strategy candidates for the given circuit. A strategy is a sequence of `(pit_lap, compound)` pairs from current lap to race end.

Candidate strategies are filtered by:
- Tyre life feasibility (each stint must be within compound's predicted life)
- Compound usage rules (at least two compounds must be used in dry conditions)
- Circuit-specific constraints (e.g., at Monaco, strategies requiring a track position overtake to gain places are penalised heavily)

Remaining candidates are scored by Monte Carlo simulation.

### 10.4 Monte Carlo Simulation

For each candidate strategy, the engine runs N simulations (default 5,000):

1. Sample stochastic variables — safety car deployment probability (circuit-specific), weather changes, rival pit windows — from their model distributions
2. Simulate lap-by-lap race progression using the Lap Time Prediction model as the time oracle and the Tyre Degradation model as the tyre constraint
3. Apply circuit-specific pit lane delta, traffic interactions, and track position changes at each simulated lap
4. Record final position for this run

Repeat N times. Aggregate into outcome distribution.

### 10.5 Scoring Function

Each strategy is scored as:

```
Strategy Score = (Expected Points × 0.6) + (Podium Probability × 0.3) + (Win Probability × 0.1)
```

Weights are configurable per team objective (e.g., a team fighting for the championship weights win probability higher; a midfield team weights points finishes higher).

### 10.6 Outputs

For each candidate strategy, the engine returns:

| Output | Description |
|---|---|
| `win_probability` | % of simulations ending P1 |
| `podium_probability` | % of simulations ending P1–P3 |
| `points_probability` | % of simulations ending P1–P10 |
| `expected_finish` | Mean finishing position across simulations |
| `expected_points` | Mean championship points across simulations |
| `finish_distribution` | Full position distribution P1–P20 |
| `risk_score` | Variance of finish position — higher = more volatile strategy |
| `strategy_score` | Composite ranking score |

### 10.7 Recommendation Output

```json
{
  "circuit": "Bahrain International Circuit",
  "circuit_profile": {
    "overtaking_difficulty": 28,
    "tyre_stress_index": 74,
    "track_position_importance": 41,
    "dominant_strategy": "2-stop",
    "historical_sc_rate": 0.44
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
    "Undercut window opens against P1 within 2 laps — Bahrain historically rewards early undercuts",
    "Current tyres predicted to fall off cliff in 3 laps (tyre stress index: 74)",
    "Hard compound projected to last comfortably to secondary stop",
    "Rain probability low for remainder of race",
    "SC probability 44% — Hard tyres will still be viable if SC deployed mid-stint"
  ],
  "alternatives": [
    {
      "action": "Extend to Lap 22, then pit on Hard",
      "expected_finish": "P3",
      "expected_points": 15,
      "win_probability": 0.11,
      "podium_probability": 0.54,
      "risk_score": 8.1,
      "reasoning": "Lower risk, lower ceiling. Loses undercut window but reduces exposure to traffic."
    },
    {
      "action": "Pit now on Medium — aggressive undercut",
      "expected_finish": "P2",
      "expected_points": 17,
      "win_probability": 0.31,
      "podium_probability": 0.59,
      "risk_score": 19.4,
      "reasoning": "Highest win probability but high variance. Medium may not survive to race end without third stop."
    }
  ]
}
```

---

## 11. API Layer

All model outputs, simulations, and recommendations are exposed via a FastAPI REST service.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/circuits` | List all circuits with profile data |
| GET | `/circuits/{circuit_id}/profile` | Full circuit profile including historical strategy data |
| GET | `/drivers` | List all drivers with metadata |
| GET | `/races` | List all races with circuit and date |
| GET | `/telemetry/{lap_id}` | Telemetry channels for a specific lap |
| GET | `/prediction/laptime` | Lap time prediction for given state |
| POST | `/predict/strategy` | Full strategy recommendation with reasoning |
| POST | `/predict/strategy/optimal` | Circuit-adaptive optimal strategy for any track + conditions |
| POST | `/simulate` | Run Monte Carlo simulation for given state and strategies |
| POST | `/retrain` | Trigger manual model retraining (authenticated) |

All endpoints return JSON. Authentication on write endpoints via API key. Rate limiting applied on `/simulate` and `/predict/strategy/optimal` given compute cost.

---

## 12. Dashboard

The dashboard is built in Plotly Dash. It exposes the following views:

**Race View:** Live or historical race state. Shows current positions, gaps, tyre compounds, lap-by-lap pace, and the current top strategy recommendation with simulation outcomes.

**Circuit Strategy View:** Circuit profile for the current or selected track. Shows historical winning strategy distributions, tyre compound usage, SC frequency, undercut success rates, and the circuit-adaptive strategy recommendation for a given scenario.

**Telemetry View:** Speed traces, throttle/brake/DRS channels, sector time breakdowns for any lap and driver.

**Tyre View:** Degradation curves per stint, compound comparison, predicted remaining life.

**Strategy Simulation View:** Side-by-side comparison of candidate strategies with full outcome distributions rendered as box plots or violin plots.

**Driver View:** Per-driver performance profiles, historical circuit stats, wet/dry splits.

**Model Performance View:** Live model metrics, prediction vs. actual overlays, drift indicators per model.

---

## 13. MLOps & Nightly Retraining

The nightly pipeline runs on a fixed schedule after all race-day data has been confirmed available:

1. Check for new completed sessions since the last pipeline run
2. Download and validate raw data from all sources
3. Apply Bronze → Silver → Gold transformations
4. Run data quality checks via Great Expectations — abort pipeline if critical checks fail
5. Recompute features for affected races in the feature store
6. Update circuit profiles with new race data
7. Retrain all six models with updated data
8. Evaluate each model against the held-out test season
9. Compare new model metrics against currently deployed model metrics
10. Promote new model to production if evaluation metric improves or holds within tolerance; otherwise retain current model and log an alert
11. Update the MLflow model registry with the new run metadata

All pipeline steps are orchestrated via Airflow. dbt handles Silver and Gold layer SQL transformations. Docker ensures environment reproducibility across pipeline stages.

---

## 14. Monitoring

| Signal | Description | Alert Threshold |
|---|---|---|
| Prediction drift | Distribution shift in model outputs vs. historical baseline | KL divergence > 0.1 |
| Data drift | Feature distribution shift in incoming data | PSI > 0.2 on key features |
| Missing features | Null rate on Gold layer features | > 2% null on any critical feature |
| Circuit profile staleness | Days since circuit profile last updated | > 14 days before a race at that circuit |
| API latency | P95 response time per endpoint | > 500ms on prediction endpoints |
| API error rate | 4xx and 5xx rates | > 1% over 5-minute window |
| Simulation throughput | Time to complete 5,000 simulations | > 10 seconds |
| Model accuracy | Rolling MAE / log-loss on recent races | > 10% degradation vs. baseline |

---

## 15. Tech Stack

| Layer | Tools |
|---|---|
| Data collection | `fastf1`, `requests`, Ergast API, OpenF1, Weather API |
| Data processing | `pandas`, `polars`, `pyarrow`, `duckdb` |
| Data transformation | `dbt` |
| Data validation | `great_expectations` |
| Storage | PostgreSQL (Gold layer), raw file storage for Bronze |
| ML training | `lightgbm`, `xgboost`, `catboost`, `scikit-learn` |
| Hyperparameter tuning | `optuna` |
| Experiment tracking | `mlflow` |
| Explainability | `shap` |
| Pipeline orchestration | `airflow` |
| Serving | `fastapi`, `redis` (caching for circuit profiles and simulation results) |
| Dashboard | `plotly dash` |
| Containerisation | `docker`, `docker-compose` |
| CI/CD | GitHub Actions |
| Infrastructure (optional v2) | AWS, Terraform |
| Language | Python 3.10+ |

---

## 16. Repository Structure

```
raceiq/
├── ingestion/
│   ├── fastf1/             # FastF1 session and telemetry ingestion
│   ├── ergast/             # Historical results ingestion
│   ├── openf1/             # Live session ingestion
│   └── weather/            # Weather API ingestion
├── pipelines/
│   ├── bronze/             # Schema enforcement and deduplication
│   ├── silver/             # Joins, ID normalisation, business logic
│   └── gold/               # Feature store construction
├── feature_store/
│   ├── definitions/        # Feature definitions and versioning
│   └── validation/         # Great Expectations suites
├── circuit_profiles/       # Per-circuit learned profiles and strategy priors
├── models/
│   ├── lap_time/
│   ├── tyre_degradation/
│   ├── pit_stop/
│   ├── safety_car/
│   ├── race_position/
│   └── win_probability/
├── strategy_engine/
│   ├── search/             # Strategy candidate enumeration
│   ├── simulation/         # Monte Carlo engine
│   ├── scoring/            # Strategy scoring and ranking
│   └── recommendation/     # Reasoning generation
├── serving/
│   └── api/                # FastAPI application
├── dashboard/              # Plotly Dash application
├── monitoring/             # Drift detection and alerting
├── tests/
├── docker/
├── dbt/                    # Silver and Gold dbt models
├── airflow/                # DAG definitions
└── docs/
    ├── PRD.md
    ├── ARCHITECTURE.md
    └── API_SPEC.md
```

---

## 17. Success Criteria

- Lap Time Prediction RMSE below 0.4 seconds on held-out test season
- Tyre Degradation model predicts remaining life within 2 laps on average across all compounds
- Pit Stop Recommendation precision above 0.75 on held-out test season
- Safety Car Probability AUC-ROC above 0.70
- Circuit-adaptive strategy engine produces a ranked recommendation for any circuit + conditions input within 500ms at P95
- Simulation engine completes 5,000 scenarios in under 10 seconds
- Nightly retraining pipeline completes without manual intervention
- Zero data leakage verified on all temporal train/test splits
- All six models tracked and versioned in MLflow with full reproducibility
- Circuit profiles demonstrably improve strategy recommendations vs. circuit-agnostic baseline (measured by historical backtest on 2023 season)

---

## 18. Open Questions

- **Live race data latency:** OpenF1 provides near-real-time data but with variable delay. What is the acceptable latency for strategy recommendations during a live race — 5 seconds, 30 seconds, 1 lap?
- **Simulation compute budget:** 5,000 simulations per strategy call is the default. Under live race conditions with multiple strategy candidates, this could take several seconds. Should simulation depth be reduced during live races in favour of lower latency?
- **Ergast deprecation:** A fallback data source for pre-2018 historical results needs to be identified before ingestion is built around Ergast.
- **Circuit profile cold start:** For circuits with fewer than 5 historical races in the dataset (e.g., newer additions to the calendar like Las Vegas, Qatar), the circuit profile will be sparse. Define fallback behaviour — nearest-neighbour circuit type, or global average?
- **Strategy scoring weights:** The default scoring function weights expected points (0.6), podium probability (0.3), and win probability (0.1). These should be configurable per team objective. Define the configuration interface.
- **Dashboard hosting:** Internal only (VPN-gated), or externally accessible for the fan/fantasy use case? This affects authentication design and infrastructure cost significantly.
