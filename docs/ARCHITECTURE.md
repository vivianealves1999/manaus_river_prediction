# Architecture

## 1. What this system does

Given the historical stage (water level) record of the Rio Negro at the Porto de Manaus
gauge, plus climate and upstream-hydrology covariates, it produces a **monthly-mean stage
forecast three or more months ahead**, with a prediction interval, a backtested estimate of
how much that forecast can be trusted, and an explicit ENSO what-if layer.

The unit throughout is **metres above the Porto de Manaus gauge zero** — a local datum, not
sea level. Reference points on that datum: record high 30.02 m, record low 12.11 m
(October 2024), typical annual swing roughly 10 m.

## 2. Layered design

```
                         ┌──────────────────────────────────┐
  main.py  (thin CLI) ──▶│        pipeline.run()            │
                         └───────────────┬──────────────────┘
                                         │  orchestrates, owns no logic
   ┌─────────────────────────────────────┼──────────────────────────────────┐
   │                                     │                                  │
   ▼                                     ▼                                  ▼
┌────────────────┐            ┌────────────────────┐            ┌────────────────────┐
│  data layer    │            │  feature layer     │            │  model layer       │
│                │            │                    │            │                    │
│ ana_client     │            │ features.py        │            │ baselines          │
│ porto_manaus   │──panel────▶│  calendar          │──X, y────▶ │ sarimax            │
│ climate        │            │  target lags       │            │ ets                │
│ build_dataset  │            │  climate lags      │            │ gbm                │
└────────────────┘            │  upstream lags     │            │ ensemble           │
        │                     └────────────────────┘            └──────────┬─────────┘
        │                                                                  │
        │                     ┌────────────────────┐                       │
        └────────────────────▶│  backtest.py       │◀──────────────────────┘
                              │  rolling origin    │
                              └─────────┬──────────┘
                                        │ skill table → ensemble weights
                    ┌───────────────────┼────────────────────┐
                    ▼                   ▼                    ▼
              ┌───────────┐      ┌─────────────┐      ┌─────────────┐
              │ plots.py  │      │ scenarios.py│      │  report .md │
              └───────────┘      └─────────────┘      └─────────────┘
```

Each layer depends only on the layer above it in that list. `config.py` is the only module
everything imports, and it holds every constant so nothing else contains magic numbers.

## 3. Data flow, stage by stage

### Stage 1 — acquisition (`src/manaus_river/data/`)

| Module | Source | Coverage | Role |
|---|---|---|---|
| `ana_client.py` | ANA / HidroWeb legacy SOAP-over-GET | Manaus **1902–2014**, upstream gauges to ~2020s | Authoritative, quality-checked ("consisted") backbone |
| `porto_manaus.py` | portodemanaus.com.br (WordPress/JetEngine) | **2000–present**, daily | Recent tail; same physical gauge |
| `climate.py` | NOAA PSL | ONI/TNA/TSA/SOI **1948–present**, AMO to 2023 | Exogenous climate drivers |

Three engineering details in this layer are load-bearing:

**Chunking.** The ANA service silently returns an empty result set — not an error — for
date ranges beyond a few years. Requests are split into 5-year windows
(`ANA_CHUNK_YEARS`). Getting this wrong looks like "the station has no data".

**Consistency-level resolution.** Each month can appear twice, `NivelConsistencia` 1 (raw)
and 2 (quality-checked). We sort by consistency and keep the last, so the checked value
always wins where it exists and raw fills the rest.

**Splicing.** ANA's consisted Manaus record stops at 2014-12, and every ANA route to newer
data is now behind an API token. The Porto de Manaus web archive covers 2000-present. We
validated the two against each other over their 5,459-day overlap:

> correlation **0.9995**, mean difference **−0.002 m**, standard deviation **0.127 m**

That is the same gauge, so `build_dataset.build_target_daily()` takes ANA wherever it
exists and appends Porto de Manaus only for dates ANA lacks. ANA always wins ties.

The Porto de Manaus archive needed two quality fixes, both documented in the module:
it is **not unit-consistent** (recent years publish metres, `24,09`; older years publish
whole centimetres, `1641`), and it contains a handful of **transcription typos** (one
reading 7 m off its neighbours). `_normalise_level()` and `_drop_impossible_jumps()`
handle these respectively.

Requests are made **one month at a time** rather than six-per-page. The paged listing
interleaves several tables whose month must be inferred from surrounding HTML headings,
and those headings are not reliably rendered — which silently mis-assigned whole months
during development. Pinning the month in the request means the date comes from what we
asked for, not from parsing.

### Stage 2 — the panel (`build_dataset.py`)

Daily readings are aggregated to monthly means, discarding months with fewer than 10
daily observations (averaging a month from three readings would bias the seasonal cycle).
The result is one tidy frame indexed by month:

| Column | Meaning |
|---|---|
| `level_m` | **target** — monthly mean stage at Manaus |
| `level_min_m`, `level_max_m`, `n_days` | context and QA, never predictors |
| `manacapuru_level_m`, `caracarai_level_m`, `tabatinga_level_m`, `porto_velho_level_m` | upstream gauges |
| `oni`, `tna`, `tsa`, `soi`, `amo` | climate indices |
| `enso_phase` | categorical CPC phase label |

### Stage 3 — features (`features.py`)

**The single rule that governs this module: every feature must be knowable at forecast
time.** When forecasting month `t+h` from origin `t`, any covariate may enter only at a lag
of at least `h`. `build_design_matrix()` enforces this itself rather than trusting callers,
because leaking the future is the easiest way to produce a backtest that looks superb and a
forecast that is worthless.

Feature families:

- **Calendar** — three Fourier harmonic pairs of month-of-year (the flood pulse is a smooth
  annual wave, so harmonics beat 11 dummies), plus a centred linear time index.
- **Autoregressive** — target lags at `h, h+1, h+2, h+3, h+6, h+12`; a 3-month rate of
  change (distinguishes a filling basin from a draining one at the same stage); the anomaly
  against the origin month's climatology; a 12-month rolling mean as a basin-storage proxy.
- **Climate** — ONI/TNA/TSA/SOI/AMO at lags `max(lag, h)`, plus their 3-month running means.
- **Upstream** — each upstream gauge at lags `max(lag, h)`.

`exog_for_sarimax()` returns a deliberately smaller, dense block (ONI and TNA running means
only). SARIMAX already models seasonality and autocorrelation internally, so its exogenous
matrix should carry only *external* drivers, and it must be dense and extendable across the
forecast window — which rules out the upstream gauges, whose own future values are unknown.

### Stage 4 — models (`models/`)

| Model | Type | Why it is in the stack |
|---|---|---|
| `Climatology` | Monthly mean, recency-weighted | The honest bar. The Rio Negro flood pulse is so regular that climatology is genuinely hard to beat at 3 months. |
| `Seasonal Naive + drift` | Last year same month + annual anomaly | Cheap, and the drift term carries "the basin is 2 m below normal right now". |
| `Holt-Winters (ETS)` | Additive, damped trend | Models level/trend/season as evolving *states*, not autocorrelation — structurally different errors from SARIMAX. |
| `SARIMAX` | Seasonal ARIMA | Captures the annual pulse plus the near-unit-root persistence of stage. |
| `SARIMAX+ENSO` | SARIMAX + exogenous ONI/TNA | The requested ENSO-aware statistical model. |
| `Gradient Boosting` | Direct multi-horizon GBM | The only member that sees the full feature space and can express non-linear interactions (El Niño bites harder when storage is already low). |
| `Ensemble` | Inverse-MSE weighted | The headline. |

Every model implements the same three-method `BaseForecaster` protocol, so the backtester,
plots and report never branch on model type.

Two design choices worth calling out:

**Direct, not recursive, multi-horizon.** The GBM fits a separate model per lead time,
each mapping origin-time features straight onto the target month. This avoids the error
compounding of feeding one-step predictions back in, and makes the no-leakage rule trivial
to enforce.

**Additive, not multiplicative, seasonality.** Stage is measured against an arbitrary gauge
datum, so a "20% seasonal effect" has no physical meaning.

Prediction intervals: SARIMAX uses its analytic predictive distribution; the GBM uses two
extra quantile-regression fits (stage errors are visibly skewed in drought years); ETS and
the baselines widen a residual scale with √lead. The ensemble combines the weighted member
spread with between-member disagreement in quadrature — disagreement is itself a real and
often dominant uncertainty signal.

### Stage 5 — backtesting (`backtest.py`)

Rolling origin, **refitting every model from scratch at every origin**. For each origin `t`:
truncate the panel to `≤ t`, refit, forecast `t+1…t+h`, score against truth, bucket by lead
time.

Refitting is expensive — the SARIMAX order search dominates runtime — but fitting once on
the full history and "forecasting" the interior leaks the answer through the estimated
parameters, which is exactly the self-deception this project is trying to avoid.

Metrics (MAE, RMSE, MAPE, bias, interval coverage) are reported **per lead time**, never
pooled: a 1-month forecast of this river is nearly trivial and a 3-month one is not, so
pooling flatters every model. `skill_vs_baseline()` expresses everything as fractional
improvement over climatology — a model that cannot beat climatology is not adding value and
the report says so.

Backtest RMSE feeds `WeightedEnsemble.weights_from_scores()` as inverse-MSE weights, the
closed-form optimum for unbiased uncorrelated errors and far more robust out-of-sample than
solving for unconstrained optimal weights on a short backtest.

### Stage 6 — outputs

- `outputs/figures/` — the comparison chart and diagnostics (see §5)
- `outputs/forecasts/forecast_<date>.csv` — every model's path, with intervals
- `outputs/reports/` — skill tables as CSV, plus a Markdown report
- `data/processed/monthly_panel.csv` — the analysis-ready panel

## 4. How El Niño enters the system

There are two distinct pathways, deliberately kept separate.

**Pathway A — observed indices as lagged exogenous features (default, in the headline
number).** ONI and TNA enter `SARIMAX+ENSO` and the GBM at lags of at least the forecast
horizon. Because the lag is ≥ h, forecasting `t+1…t+h` only ever uses climate values from
`t+1-h…t` — already observed. **No climate forecast is required and nothing leaks.**

**Pathway B — assumed future ENSO state (`scenarios.py`, reported separately).** Re-runs the
best model with the ONI path overridden, producing the forecast under "strong El Niño",
"neutral", "La Niña" and so on. This is presented as a labelled what-if rather than folded
into the headline, because it rests on an assumption the data cannot verify.

The physical chain, which is why any of this works:

> Pacific SST anomaly → Walker circulation shifts east → anomalous subsidence over the
> Amazon → suppressed convection and rainfall deficit in the catchment → (1–2 season lag for
> soil storage and channel routing) → stage anomaly at Manaus.

The reason NOAA/CPC can flag this months ahead is **thermal inertia**: a Niño 3.4 SST
anomaly is a slow, high-heat-capacity ocean signal that persists and evolves predictably,
unlike the fast atmosphere it drives. Slow ocean forcing a fast atmosphere is what gives
seasonal forecasting any skill at all.

**The caveat this project takes seriously:** the Pacific is not the only driver. A warm
Tropical North Atlantic pulls the ITCZ northward and caused some of the Amazon's worst
droughts (2005, 2010) with ENSO near neutral. TNA is therefore carried alongside ONI
everywhere, and `enso_phase_summary()` reports the observed spread within each phase so the
reader can see how much ENSO alone actually explains.

## 5. Figures

Charting rules applied throughout `plots.py`:

- **One y-axis per panel.** Stage in metres and error in metres are different measures and
  get different panels, never a twin axis.
- **Colour follows the entity, never its rank** — each model keeps its palette slot whatever
  its skill turns out to be, so figures are comparable across runs.
- **Identity never rests on colour alone.** Three of the categorical hues fall below 3:1
  contrast against the chart surface, so every forecast line is directly labelled at its
  endpoint and every figure is backed by a CSV/Markdown table.
- Thin marks (2 px lines), recessive grid, a single translucent interval band belonging to
  the headline model rather than one band per model.

The categorical palette was validated for colour-vision deficiency rather than eyeballed:
worst adjacent-pair CVD ΔE 9.1 (target ≥ 8), worst normal-vision ΔE 19.6 (floor ≥ 15).

| File | Content |
|---|---|
| `forecast_comparison.png` | **The headline deliverable.** All models' paths to the target month, ensemble interval, backtest error by lead, and a close-up of the headline forecast. |
| `model_skill.png` | RMSE by lead time; predicted-vs-observed for the best model. |
| `enso_influence.png` | Observed stage anomaly by ENSO phase and month; lagged ONI vs Oct–Dec anomaly. |
| `history_overview.png` | The full instrumental record and the seasonal envelope. |
| `enso_scenarios.png` | Forecast paths under each assumed ENSO state. |

## 6. Failure behaviour

The pipeline is built to degrade rather than abort, because it depends on four external
services that are individually unreliable:

- Any single **climate index** that fails to download is skipped with an error log; the rest
  proceed.
- Any single **upstream gauge** that fails is dropped from the panel; its features simply do
  not appear.
- The **Porto de Manaus** scraper never raises; on failure the pipeline runs on ANA alone.
- A **model** that fails to fit at a backtest origin is logged and skipped for that origin
  rather than killing the run; a model that fails entirely is dropped from the ensemble
  (with its weight redistributed, not set to zero).
- **Figure generation** is wrapped so a plotting error cannot lose a completed forecast.

Network calls retry with linear backoff, and all raw downloads are cached to `data/raw/`, so
re-runs are offline and fast. `--refresh-data` forces re-download.

## 7. Performance

A full run (48 rolling origins × 6 models, each refitted from scratch) takes roughly 25–30
minutes. Three things dominate and each has a documented mitigation:

| Cost | Mitigation | Where |
|---|---|---|
| SARIMAX order search at every origin | Search once per variant, then reuse the specification | `models/sarimax.py::_ORDER_CACHE` |
| State-space filtering over 120+ years | Fit on the most recent 50 years | `models/sarimax.py::max_train_years` |
| Gradient boosting × 3 leads × 3 quantile fits | 250 trees, depth 3 | `config.py::gbm_*` |

**BLAS threading is the non-obvious one.** This workload is thousands of *small*
linear-algebra calls, not a few large ones, and multi-threaded BLAS spends more time
synchronising threads than doing arithmetic at that size. Measured on this project, the
default settings left the process at about **20% CPU utilisation** and made a full backtest
several times slower than single-threaded. `main.py` therefore pins `OMP_NUM_THREADS` and
friends to 1 *before* numpy is imported — which must happen before the import, not after, or
the setting has no effect.

For a quick answer, `--origins 12` gives a usable skill estimate in a few minutes, and
`--figures-only` redraws every figure from the previous run's artefacts without refitting
anything.

## 8. Extending it

- **A new data source** — add a module under `data/` exposing a function that returns a
  date-indexed frame, and join it in `build_dataset.build_panel()`.
- **A new model** — subclass `BaseForecaster`, implement `fit`/`predict`, register it in
  `pipeline.model_factories()`. Backtesting, weighting, plotting and reporting pick it up
  with no further changes.
- **A new feature family** — add an `add_*_features()` function in `features.py` and call it
  from `build_design_matrix()`. Respect the lag rule: shift by at least `horizon`.
