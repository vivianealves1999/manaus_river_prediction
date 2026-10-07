# Modelling notes

This document explains *why* the modelling stack looks the way it does — the reasoning a
reviewer would want to interrogate. For the code layout see `ARCHITECTURE.md`; for where the
numbers come from see `DATA_SOURCES.md`.

## 1. What is actually being predicted

**Target:** monthly mean stage of the Rio Negro at the Porto de Manaus gauge, in metres
above the local gauge zero.

Three consequences of that definition, all of which matter when reading the output:

1. **It is a stage, not a discharge.** Stage responds to the Solimões backwater as well as
   to the Negro's own flow (see §4), and it is referenced to a local datum, so "20 m" is only
   meaningful relative to this gauge's own history.
2. **It is a monthly mean.** The daily minimum inside the target month will sit *below* the
   forecast — typically 0.5–1.5 m below during the falling limb. Headlines about "the river
   hit 12.11 m" refer to daily extremes, not monthly means.
3. **Percentage error is meaningless.** Because the datum is arbitrary, MAPE is reported only
   for comparability with other work; MAE and RMSE in metres are the metrics to read.

## 2. Why this series is both easy and hard

**Easy:** the Amazon flood pulse is one of the most regular large-scale hydrological signals
on Earth. Manaus rises from a November low to a June–July peak and falls back, every year,
with a ~10 m amplitude. Month-of-year alone explains the great majority of the variance.

**Hard:** everything anyone actually wants to know is in the *residual*. Whether December
comes in at 21 m or 17 m — the difference between normal navigation and a shipping
emergency — is a 3–4 m anomaly on top of a perfectly predictable 10 m cycle.

This is why **climatology is the reference model and not a straw man.** A model that
"explains 95% of variance" here has done nothing; the only question that matters is whether
it beats the monthly mean on the anomaly. `skill_vs_baseline()` exists to force that
comparison into the report, and if a model comes out negative the report says so.

## 3. Persistence and memory

Two properties drive the feature design:

**Extreme persistence.** Stage is an integral of catchment storage, so month-to-month
autocorrelation is very high, and a 1-month-ahead forecast is close to trivial. That is
exactly why every metric is bucketed **per lead time** — pooling a 1-month and a 3-month
forecast flatters everything.

**Long memory.** A drought year leaves the basin depleted going into the next wet season, so
a 12-month rolling mean carries genuine information beyond the recent lags. Hence
`level_roll12_*` in the feature set.

The autoregressive features also include a **rate of change** (`level_delta_*`): two months
can sit at the same absolute stage with one on the rising limb and one on the falling limb,
and those have completely different futures. Absolute level alone cannot distinguish them.

## 4. The Solimões backwater — the most important non-obvious fact

The project brief asked whether other Amazon rivers influence the Rio Negro. They do, and
one of them dominates.

The Rio Negro meets the Rio Solimões just downstream of Manaus. The Solimões carries several
times the Negro's discharge, and its flood wave is **out of phase** with the Negro's. When
the Solimões runs high it acts as a hydraulic dam, **backing water up into the Negro and
raising the stage at Manaus** with no change whatsoever in the Negro's own flow.

So the Manaus gauge is partly measuring a different river. Practical implications:

- The Solimões gauge at **Manacapuru** (ANA 14100000) is the strongest external predictor
  available, and it is in the feature set.
- Far-upstream **Tabatinga** (13150000) has a 2–3 month travel time to Manaus — a genuine
  physical lead at exactly this project's horizon.
- **Caracaraí** on the Rio Branco (14710000) drains the Negro's largest tributary from the
  *northern* hemisphere, so it peaks out of phase with the southern basin.

### The catch

These gauges lag badly in the public feed (Manacapuru and Tabatinga stop around 2025-09).
For a forecast issued in September 2026 at 3-month lead, the legal lag is ≥ 3 months, so a
stale gauge yields nothing usable. Rather than impute it — which would mean feeding the model
a fabricated predictor and calling the output a forecast — `build_design_matrix()` **drops
any predictor with no observation in the last 12 months**. Losing a feature is honest;
inventing one is not.

The cost of including them at all is that requiring every feature to be present restricts
GBM training to 1972 onward (~643 months) instead of 1950 onward (~900). That is a real
trade-off, made in favour of the physically-motivated features, and it is visible in the
backtest.

## 5. Model-by-model rationale

| Model | What it is good at | Where it fails |
|---|---|---|
| **Climatology** | The seasonal cycle, exactly. Unbeatable at long lead when no anomaly information exists. | Cannot represent an anomaly at all — it forecasts a normal year, every year. Catastrophic in 2024. |
| **Seasonal Naive + drift** | Carries the current anomaly forward cheaply. | Assumes this year resembles last year plus a constant; wrong whenever the anomaly is itself evolving. |
| **Holt-Winters (ETS)** | Adapts level/trend/season as evolving states; handles a changing seasonal amplitude. | Purely endogenous — blind to ENSO. Damped trend can under-react to a fast recession. |
| **SARIMAX** | The seasonal + near-unit-root structure, with a principled predictive distribution. | Linear and Gaussian; cannot express "El Niño bites harder when storage is already low". |
| **SARIMAX+ENSO** | Adds the climate signal in a form whose coefficients are interpretable and testable. | Same linearity limit; a small dense exogenous block by necessity. |
| **Gradient Boosting** | The full feature space, non-linear interactions, skewed intervals via quantile loss. | Cannot extrapolate beyond the training range — a genuine worry for record-breaking events. Needs the most data. |
| **Ensemble** | Averages decorrelated errors, which is most valuable exactly at the seasonal turning point the December forecast sits on. | Inherits every member's blind spots in proportion to its weight. |

### Direct rather than recursive multi-horizon

The GBM fits a **separate model per lead time**, each mapping origin-time features straight
onto the target month. The alternative — predict `t+1`, feed it back, predict `t+2` — compounds
its own errors and makes the no-leakage rule much harder to verify. Direct forecasting costs
`h` fits instead of one and is worth it.

### Why additive, not multiplicative, seasonality

Stage is measured against an arbitrary gauge zero. A "20% seasonal effect" would change
meaning if the datum were redefined, which is a sign the model form is wrong. Additive
seasonality is the only choice with physical meaning here.

### Prediction intervals

| Model | Method | Why |
|---|---|---|
| SARIMAX | Analytic state-space predictive distribution | It has one; use it |
| GBM | Two extra quantile-regression fits | Stage errors are visibly right-skewed in drought years; a symmetric normal interval would understate downside risk |
| ETS, baselines | Residual scale widened by √lead | No analytic interval available; uncertainty must at least grow with lead time |
| Ensemble | Weighted member spread ⊕ between-member disagreement (in quadrature) | Disagreement is itself a real uncertainty signal, and often the dominant one |

Interval quality is not assumed — **empirical coverage is a reported backtest metric**. An
80% interval that covers 55% of outcomes is a broken interval, and the table will show it.

## 6. How El Niño is used

### The physical chain

> Pacific SST anomaly → Walker circulation shifts east → anomalous subsidence over the
> Amazon → suppressed convection, rainfall deficit in the catchment → (1–2 season lag for
> soil storage and channel routing) → stage anomaly at Manaus.

**Why NOAA/CPC can call it months ahead.** The ocean has orders of magnitude more thermal
inertia than the atmosphere. A Niño 3.4 SST anomaly is a slow, high-heat-capacity signal
that persists and evolves predictably over seasons, and the atmospheric teleconnection it
forces is statistically reliable. Slow ocean forcing a fast atmosphere is the entire basis of
seasonal climate prediction — without that asymmetry there would be no skill at any lead.

The specific things forecasters watch: sub-surface warm-water volume in the equatorial
Pacific (which leads surface expression by months), westerly wind bursts, thermocline depth
along the equator, and the Southern Oscillation pressure gradient. The ONI itself is the
3-month running mean of the Niño 3.4 SST anomaly, with an event declared at ±0.5 °C sustained
over five overlapping seasons — that is what `classify_enso()` implements.

### Two pathways, deliberately separated

**Pathway A — observed indices as lagged features (in the headline number).** ONI and TNA
enter `SARIMAX+ENSO` and the GBM at lags ≥ horizon. Forecasting `t+1…t+h` therefore uses only
climate values from `t+1-h…t`, all observed. **No climate forecast is required and nothing
leaks.** The limitation is that it cannot exploit CPC's forward outlook.

**Pathway B — assumed future ENSO state (`scenarios.py`, reported separately).** Re-runs the
best model with the ONI path overridden: strong El Niño, moderate El Niño, neutral, La Niña,
or persistence of the current value. This is reported as a labelled what-if, never folded
into the headline, because it rests on an assumption the historical data cannot verify.

### What the data actually says about lag and about which ocean matters

`climate_lag_diagnostics()` correlates each climate index at lag *L* against the Oct–Dec
stage anomaly at Manaus. Two results are worth stating up front because neither is the
obvious one, and both shaped the feature design:

**1. ENSO acts with a 6–8 month delay, not contemporaneously.**

| ONI lag (months) | 0 | 3 | 6 | **7** | 8 | 12 |
|---|---|---|---|---|---|---|
| r vs Oct–Dec anomaly | −0.10 | −0.19 | −0.28 | **−0.29** | −0.28 | −0.22 |

The sign is negative throughout — El Niño means lower water, as expected — but the
association at lag 0 is less than half the strength at lag 7. This is precisely the
catchment-storage and channel-routing delay the physical chain predicts, and it is why a
"current ENSO phase" composite badly understates ENSO's real influence. It also validates
carrying ONI at lags 3–12 rather than only at the shortest legal lag.

**2. For December specifically, the Atlantic outranks the Pacific.**

| Predictor (6-month mean ending September) | r vs December anomaly |
|---|---|
| ONI (Pacific) | −0.13 |
| **TNA (tropical North Atlantic)** | **−0.46** |

This is a large difference, and it is consistent with the published attribution of the 2005
and 2010 Amazon droughts to a warm tropical North Atlantic with ENSO near neutral. A
Pacific-only model would be leaving the stronger signal on the table.

**3. The stage response to ENSO is threshold-like, not linear.** Banding December anomalies
by ENSO intensity rather than phase label shows the effect concentrated in genuine events
rather than scaling smoothly with ONI, which is why `enso_intensity_summary()` reports bands
and why a single regression coefficient on ONI understates strong-event risk.

None of this argues against including ENSO — it argues for including it *lagged*, alongside
the Atlantic, and with honest error bars.

### The caveat that keeps this honest

**The Pacific is not the only driver.** The Amazon's two worst droughts before 2023 — 2005
and 2010 — occurred with ENSO near neutral. Both were driven by a warm Tropical North
Atlantic pulling the ITCZ northward and starving the central basin of rainfall. A
Pacific-only model would have missed both.

That is why TNA, TSA and SOI ride alongside ONI everywhere, and why
`enso_phase_summary()` reports the **spread within each phase**, not just the mean. If the
within-phase standard deviation is comparable to the between-phase difference — which for
this river it largely is — then ENSO shifts the odds without determining the outcome, and any
presentation that implies otherwise is overselling. The report is written to make that
visible rather than to hide it.

## 7. Validation protocol

Rolling origin, **refitting every model from scratch at every origin.**

```
origin t:   [────────── train ──────────]│ forecast t+1..t+h │
origin t+1:  [────────── train ─────────────]│ forecast ...   │
```

Non-negotiables:

- **No feature lagged by less than the horizon**, enforced in `build_design_matrix()` rather
  than left to the caller.
- **Climatology, rolling means and medians are recomputed on the truncated slice** at each
  origin. Computing them once on the full history is a subtle and very common leak.
- **Refit, never reuse.** Fitting once on everything and scoring the interior leaks the
  answer through the estimated parameters.
- **Per-lead metrics**, never pooled.

Two concessions are made for runtime, both documented in `models/sarimax.py`: SARIMAX fits on
the most recent 50 years, and its order is selected once per variant then held fixed across
origins. The second is arguably the more correct experiment anyway — it measures the chosen
specification rather than the search procedure.

## 8. Known limitations

1. **Extrapolation.** The GBM cannot predict outside its training range, and the record low
   of October 2024 (12.11 m) was outside anything seen before. Tree models will systematically
   under-predict novel extremes. The SARIMAX and ETS members can extrapolate, which is part of
   why the ensemble includes them.
2. **Stale upstream gauges.** The physically strongest external predictors are the ones the
   public feed serves least promptly, so they contribute less to a live forecast than their
   physical importance warrants.
3. **No rainfall or soil-moisture input.** Satellite precipitation (CHIRPS, GPM) and GRACE
   terrestrial water storage would plausibly add real skill at 1–3 months. They are the
   highest-value extension.
4. **Monthly resolution.** Navigation and water-supply decisions turn on daily extremes; a
   monthly mean is the right quantity for a seasonal outlook but not for operations.
5. **Non-stationarity.** Amazon hydrological extremes have intensified since ~2000. Every
   model here assumes the relationships it learned still hold, and all of them will be
   slow to notice if that stops being true.
6. **Gauge datum.** Stage is not discharge and not sea level. Comparing these numbers to any
   other station or dataset requires care.
