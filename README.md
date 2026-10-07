# Manaus River Level Prediction — Rio Negro at Porto de Manaus

Seasonal (3+ month) forecasting of the Rio Negro water level at Manaus, Amazonas, Brazil,
with El Niño / ENSO as an explicit driver.

Built on **1,489 months of gauge data (September 1902 – present)** spliced from two public
sources, plus NOAA Pacific and Atlantic climate indices and four upstream Amazon-basin
gauges. Six models are backtested against each other on a leak-free rolling-origin protocol
and combined into a skill-weighted ensemble.

---

## Quick start

```bash
# from the project root, with the venv active
pip install -r requirements.txt

# forecast to December of the current hydrological year
python main.py

# a specific target, with a faster backtest
python main.py --target-month 12 --target-year 2026 --origins 24

# re-download every source (~6 min; otherwise cached CSVs are used)
python main.py --refresh-data
```

Outputs land in:

| Path | Content |
|---|---|
| `outputs/figures/forecast_comparison.png` | **The headline chart** — every model's path to the target month |
| `outputs/figures/model_skill.png` | Error by lead time; predicted vs observed |
| `outputs/figures/enso_influence.png` | Observed ENSO effect on Manaus stage |
| `outputs/figures/history_overview.png` | The full 1902– record and the flood pulse |
| `outputs/figures/enso_scenarios.png` | Forecast under assumed ENSO states |
| `outputs/reports/forecast_report_<date>.md` | Narrative report with all skill tables |
| `outputs/reports/backtest_*.csv` | Skill overall, by lead, and every backtest prediction |
| `outputs/reports/climate_lag_correlations.csv` | Which climate index leads the low-water season, by lag |
| `outputs/reports/enso_*.csv` | Phase composite, intensity bands, what-if scenarios |
| `outputs/forecasts/forecast_<date>.csv` | Machine-readable forecasts with intervals |
| `data/processed/monthly_panel.csv` | The analysis-ready monthly panel |

## What the numbers mean

Stage is **metres above the Porto de Manaus gauge zero** — a local datum, not sea level.
Anchors on that datum:

| | Value |
|---|---|
| Record high | 30.02 m (June 2021) |
| Record low | 12.11 m (October 2024) |
| Typical annual swing | ~10 m |
| Low water | October–November |
| Peak | June–July |

The target is the **monthly mean**, so the daily minimum inside a month sits below the
forecast — typically 0.5–1.5 m below on the falling limb.

## Project layout

```
manaus_river_prediction/
├── main.py                       CLI entry point
├── requirements.txt
├── docs/
│   ├── ARCHITECTURE.md           layers, data flow, failure behaviour
│   ├── DATA_SOURCES.md           every source, how it is reached, what broke
│   └── MODELING.md               why each model is here; validation; limitations
├── src/manaus_river/
│   ├── config.py                 all paths, station codes, URLs, hyper-parameters
│   ├── features.py               lag-legal feature engineering
│   ├── backtest.py               rolling-origin evaluation
│   ├── scenarios.py              ENSO what-if layer
│   ├── plots.py                  figures
│   ├── pipeline.py               orchestration
│   ├── data/
│   │   ├── ana_client.py         ANA / HidroWeb (1902–2014)
│   │   ├── porto_manaus.py       portodemanaus.com.br (2000–present)
│   │   ├── climate.py            NOAA PSL climate indices
│   │   └── build_dataset.py      splice + monthly panel
│   └── models/
│       ├── base.py               the shared forecaster protocol
│       ├── baselines.py          climatology, seasonal naive + drift
│       ├── sarimax.py            SARIMAX and SARIMAX+ENSO
│       ├── ets.py                Holt-Winters
│       ├── gbm.py                gradient boosting, direct multi-horizon
│       └── ensemble.py           inverse-MSE weighted combination
├── data/{raw,processed,external}/
└── outputs/{figures,forecasts,reports}/
```

## Data

| Source | Coverage | Role |
|---|---|---|
| ANA / HidroWeb station 14990000 | **1902-09 → 2014-12**, 41,009 days | Authoritative quality-checked backbone |
| portodemanaus.com.br | **2000-01 → present**, 9,676 days | Recent tail (same physical gauge) |
| NOAA PSL — ONI, TNA, TSA, SOI, AMO | 1948/1950 → present | Climate drivers |
| ANA — Manacapuru, Caracaraí, Tabatinga, Porto Velho | 1967/1972 → 2025-26 | Upstream hydrology |

The two stage sources were validated against each other on their 5,459-day overlap:
**correlation 0.9995, mean difference −0.002 m**. ANA wins wherever both exist.

Full details, including the several ANA endpoints that are now token-gated and the two data
quality bugs in the Porto de Manaus archive, are in [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md).

## Models

| Model | Kind |
|---|---|
| Climatology | Recency-weighted monthly mean — **the reference bar** |
| Seasonal Naive + drift | Last year, shifted by the current anomaly |
| Holt-Winters (ETS) | Additive, damped trend |
| SARIMAX | Seasonal ARIMA |
| SARIMAX+ENSO | + exogenous lagged ONI / TNA |
| Gradient Boosting | Direct multi-horizon, full feature space, quantile intervals |
| **Ensemble** | Inverse-MSE weighted — the headline |

Climatology is the reference model for a reason: the Amazon flood pulse is so regular that
month-of-year alone explains most of the variance. The useful question is not "what is the
December level" but "**how far from normal** will December be", and that is what the skill
tables measure. See [`docs/MODELING.md`](docs/MODELING.md).

## How El Niño enters

Two pathways, kept deliberately separate:

**A — observed indices, lagged (in the headline forecast).** ONI and TNA enter the models at
lags of at least the forecast horizon, so a 3-month forecast only ever uses climate values
already observed. **No climate forecast is required, and nothing leaks.**

**B — assumed future ENSO state (`scenarios.py`, reported separately).** The same model
re-run under "strong El Niño", "neutral", "La Niña" etc., as a labelled what-if.

The physical chain: *Pacific SST anomaly → Walker circulation shifts east → anomalous
subsidence over the Amazon → rainfall deficit → (1–2 season lag through soil and channel
storage) → stage anomaly at Manaus.* What makes it forecastable months ahead is the ocean's
thermal inertia — a slow, persistent SST signal driving a fast atmosphere.

### Two findings from the data that shaped the design

**ENSO acts with a 6–8 month delay, not contemporaneously.** Correlation between ONI and the
Oct–Dec stage anomaly at Manaus:

| ONI lag (months) | 0 | 3 | 6 | **7** | 8 | 12 |
|---|---|---|---|---|---|---|
| r | −0.10 | −0.19 | −0.28 | **−0.29** | −0.28 | −0.22 |

Negative throughout, as the physics predicts — but less than half as strong at lag 0 as at
lag 7. That delay is catchment storage plus channel routing, and it means a "current ENSO
phase" composite substantially *understates* ENSO's real influence.

**For December, the Atlantic outranks the Pacific.** Against the December anomaly, using the
6-month mean ending in September: ONI gives r = −0.13, while **TNA gives r = −0.46**.

This is consistent with the 2005 and 2010 Amazon droughts, both attributed to a warm tropical
North Atlantic with ENSO near neutral. A Pacific-only model leaves the stronger signal on the
table. TNA and TSA therefore ride alongside ONI throughout, and the report shows the spread
*within* each ENSO phase so it stays clear that ENSO **shifts the odds rather than
determining the outcome** — the within-phase standard deviation is comparable to the
between-phase difference in means.

## Do other Amazon rivers affect the Rio Negro?

Yes — and one of them dominates, which is worth knowing before trusting any Negro-only model.

The Negro meets the **Rio Solimões** just downstream of Manaus. The Solimões carries several
times the Negro's discharge and its flood wave is out of phase. When it runs high it acts as a
**hydraulic dam, backing water up into the Negro and raising the level at Manaus** with no
change in the Negro's own flow. The Manaus gauge is therefore partly measuring a different
river.

Hence the upstream gauges in the panel, all from the same ANA endpoint:

| Station | Code | Why |
|---|---|---|
| Manacapuru (Solimões) | 14100000 | The backwater effect — strongest external predictor |
| Caracaraí (Branco) | 14710000 | Largest Negro tributary; northern-hemisphere rainfall regime |
| Tabatinga (Solimões) | 13150000 | Far upstream; 2–3 month travel time = real lead |
| Porto Velho (Madeira) | 15400000 | Southern-basin monsoon proxy |

Browse other station codes at [snirh.gov.br/hidroweb](https://www.snirh.gov.br/hidroweb/).

## Methodological guarantees

- **No feature is lagged by less than the forecast horizon.** Enforced in
  `build_design_matrix()`, not left to the caller.
- **Models are refit from scratch at every backtest origin**, and climatology/rolling
  statistics are recomputed on the truncated slice only.
- **Metrics are reported per lead time**, never pooled — a 1-month forecast of this river is
  nearly trivial, a 3-month one is not.
- **Stale predictors are dropped, not imputed.** Any feature with no observation in the last
  12 months is discarded rather than filled with a median, so the forecast never rests on a
  fabricated input.
- **Interval coverage is measured**, not assumed.

## Limitations

1. Tree models cannot extrapolate — the October 2024 record low was outside all prior
   experience, and the GBM would have under-predicted it. SARIMAX and ETS in the ensemble
   partially offset this.
2. The physically strongest upstream predictors are the least promptly published, so they
   contribute less to a live forecast than they should.
3. No rainfall or soil-moisture input yet. Satellite precipitation (CHIRPS, GPM) and GRACE
   water storage are the highest-value extension.
4. Monthly resolution — right for a seasonal outlook, not for daily navigation decisions.
5. Amazon hydrological extremes have intensified since ~2000; every model here assumes the
   relationships it learned still hold.

## CLI reference

```
python main.py [options]

  --target-month 1-12     month to forecast to           (default 12)
  --target-year YEAR      year of the target month       (default: next occurrence)
  --origins N             rolling-origin evaluations     (default 48)
  --refresh-data          re-download all sources
  --data-only             build the dataset and exit
  --figures-only          redraw figures from the last run, without refitting
  --no-figures            skip figure generation
  --no-scenarios          skip ENSO what-if scenarios
  -v, --verbose           debug logging
```

A full run takes roughly 25–30 minutes, dominated by refitting every model from scratch at
each of the 48 backtest origins. `--origins 12` gives a usable answer in a few minutes, and
`--figures-only` redraws the charts in seconds.

Note that `main.py` pins BLAS to a single thread before importing numpy. This workload is
thousands of small linear-algebra calls rather than a few large ones, and multi-threaded BLAS
spends more time synchronising than computing at that size — left alone it ran at about 20%
CPU utilisation and took several times longer.
