"""End-to-end orchestration: data -> features -> backtest -> forecast -> figures -> report.

This module is the single place that knows the order of operations, so ``main.py`` stays a
thin CLI and every stage below stays independently testable.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from . import plots
from .backtest import BacktestResult, rolling_origin_backtest, skill_vs_baseline
from .config import FORECAST_DIR, MODEL_CONFIG, REPORT_DIR, ModelConfig
from .data.build_dataset import build_panel, load_panel
from .models.baselines import ClimatologyForecaster, SeasonalNaiveForecaster
from .models.ensemble import WeightedEnsemble
from .models.ets import ETSForecaster
from .models.gbm import GBMForecaster
from .models.sarimax import SarimaxForecaster
from .scenarios import (
    climate_lag_diagnostics,
    enso_composite,
    enso_intensity_summary,
    enso_phase_summary,
    scenario_forecast,
)

log = logging.getLogger(__name__)


def model_factories(config: ModelConfig = MODEL_CONFIG) -> dict[str, callable]:
    """Name -> zero-argument factory returning a fresh unfitted model.

    Factories rather than instances: the backtester must refit from scratch at every
    origin, so it needs to be able to create new models, not reuse one.
    """
    return {
        "Climatology": lambda: ClimatologyForecaster(),
        "Seasonal Naive + drift": lambda: SeasonalNaiveForecaster(),
        "Holt-Winters (ETS)": lambda: ETSForecaster(),
        "SARIMAX": lambda: SarimaxForecaster(use_exog=False, config=config),
        "SARIMAX+ENSO": lambda: SarimaxForecaster(use_exog=True, config=config),
        "Gradient Boosting": lambda: GBMForecaster(config=config),
    }


@dataclass
class PipelineOutput:
    """Everything a caller (CLI, notebook, test) needs after a run."""

    panel: pd.DataFrame
    backtest: BacktestResult
    forecasts: dict[str, pd.DataFrame]
    ensemble_weights: dict[str, float]
    figures: list[str] = field(default_factory=list)
    report_path: str = ""
    horizon: int = MODEL_CONFIG.horizon

    @property
    def headline(self) -> pd.DataFrame:
        return self.forecasts.get("Ensemble", pd.DataFrame())


def months_until(target_month: int, target_year: int | None, last_observed: pd.Timestamp) -> int:
    """Horizon needed to reach ``target_month`` of ``target_year`` from the last observation.

    If the target month of the current year has already passed, the next occurrence is used,
    so "forecast to December" keeps meaning something sensible in January.
    """
    year = target_year or last_observed.year
    target = pd.Timestamp(year=year, month=target_month, day=1)
    if target <= last_observed:
        target = pd.Timestamp(year=year + 1, month=target_month, day=1)
    return (target.year - last_observed.year) * 12 + (target.month - last_observed.month)


def run(
    target_month: int = 12,
    target_year: int | None = None,
    refresh_data: bool = False,
    config: ModelConfig = MODEL_CONFIG,
    n_origins: int | None = None,
    make_figures: bool = True,
    run_scenarios: bool = True,
) -> PipelineOutput:
    """Execute the full pipeline and return its artefacts."""
    # ---- 1. data ---------------------------------------------------------------------
    log.info("=" * 78)
    log.info("STAGE 1/6  Data assembly")
    panel = build_panel(cache=not refresh_data) if refresh_data else load_panel()
    observed = panel["level_m"].dropna()
    last_observed = observed.index.max()
    log.info(
        "Panel: %d months, target observed %s..%s (%d monthly observations)",
        len(panel), observed.index.min().date(), last_observed.date(), len(observed),
    )

    # ---- 2. horizon ------------------------------------------------------------------
    horizon = max(months_until(target_month, target_year, last_observed), config.horizon)
    target_date = last_observed + pd.DateOffset(months=horizon)
    log.info(
        "STAGE 2/6  Horizon: %d months (origin %s -> target %s)",
        horizon, last_observed.date(), target_date.date(),
    )
    run_config = ModelConfig(**{**config.__dict__, "horizon": horizon})
    if n_origins is not None:
        run_config.backtest_origins = n_origins

    # ---- 3. backtest -----------------------------------------------------------------
    log.info("STAGE 3/6  Rolling-origin backtest")
    factories = model_factories(run_config)
    backtest = rolling_origin_backtest(panel, factories, run_config)
    log.info("Overall skill:\n%s", backtest.overall.round(3).to_string())
    improvement = skill_vs_baseline(backtest)
    if not improvement.empty:
        log.info("RMSE improvement vs Climatology:\n%s",
                 (improvement * 100).round(1).to_string())

    # ---- 4. final fit and forecast ---------------------------------------------------
    log.info("STAGE 4/6  Fitting final models on the full history")
    weights = WeightedEnsemble.weights_from_scores(backtest.rmse_by_model())
    log.info("Ensemble weights (inverse-MSE):\n%s",
             pd.Series(weights).sort_values(ascending=False).round(4).to_string())

    forecasts: dict[str, pd.DataFrame] = {}
    members = []
    for name, factory in factories.items():
        try:
            model = factory()
            model.fit(panel, horizon)
            result = model.predict(panel, horizon, run_config.alpha)
            forecasts[name] = result.to_frame()
            members.append(model)
            log.info("  %-24s %s -> %.2f m",
                     name, target_date.date(), result.mean.iloc[-1])
        except Exception as exc:  # noqa: BLE001
            log.error("  %-24s FAILED: %s", name, exc)

    if members:
        ensemble = WeightedEnsemble(members, weights)
        # Members are already fitted; predict directly rather than refitting.
        ensemble_result = ensemble.predict(panel, horizon, run_config.alpha)
        forecasts["Ensemble"] = ensemble_result.to_frame()
        log.info("  %-24s %s -> %.2f m",
                 "Ensemble", target_date.date(), ensemble_result.mean.iloc[-1])

    # ---- 5. persist ------------------------------------------------------------------
    log.info("STAGE 5/6  Writing forecasts and tables")
    stamp = date.today().isoformat()
    combined = pd.concat(forecasts.values()).rename_axis("date")
    forecast_path = FORECAST_DIR / f"forecast_{stamp}.csv"
    combined.to_csv(forecast_path, index_label="date")

    backtest.overall.round(4).to_csv(REPORT_DIR / "backtest_overall.csv")
    backtest.metrics.round(4).to_csv(REPORT_DIR / "backtest_by_lead.csv")
    backtest.predictions.to_csv(REPORT_DIR / "backtest_predictions.csv", index=False)
    (REPORT_DIR / "ensemble_weights.json").write_text(
        json.dumps(weights, indent=2), encoding="utf-8"
    )

    composite = enso_composite(panel)
    if not composite.empty:
        composite.round(3).to_csv(REPORT_DIR / "enso_monthly_composite.csv")
    phase_summary = enso_phase_summary(panel)
    if not phase_summary.empty:
        phase_summary.round(3).to_csv(REPORT_DIR / "enso_phase_summary.csv")
        log.info("Low-water (Sep-Dec) anomaly by ENSO phase:\n%s",
                 phase_summary.round(3).to_string())

    lag_diagnostics = climate_lag_diagnostics(panel)
    if not lag_diagnostics.empty:
        lag_diagnostics.round(3).to_csv(REPORT_DIR / "climate_lag_correlations.csv")
        log.info("Climate index vs Oct-Dec stage anomaly, by lag:\n%s",
                 lag_diagnostics.round(3).to_string())

    intensity_summary = enso_intensity_summary(panel, target_month=target_date.month)
    if not intensity_summary.empty:
        intensity_summary.round(3).to_csv(REPORT_DIR / "enso_intensity_summary.csv")
        log.info("%s anomaly by ENSO intensity:\n%s",
                 target_date.strftime("%B"), intensity_summary.round(3).to_string())

    # ---- 6. figures and report -------------------------------------------------------
    figures: list[str] = []
    scenario_frame = pd.DataFrame()
    if run_scenarios:
        log.info("STAGE 6/6  ENSO scenarios and figures")
        best = backtest.best_model()
        scenario_frame = scenario_forecast(
            panel, factories.get(best, factories["SARIMAX+ENSO"]), horizon, run_config
        )
        if not scenario_frame.empty:
            scenario_frame.to_csv(REPORT_DIR / "enso_scenarios.csv", index_label="date")

    if make_figures:
        try:
            figures.append(
                plots.plot_forecast_comparison(
                    panel["level_m"], forecasts,
                    headline_model="Ensemble" if "Ensemble" in forecasts else backtest.best_model(),
                    target_label=f"{target_date:%B %Y}",
                    metrics=backtest.metrics,
                )
            )
            figures.append(plots.plot_model_skill(backtest.metrics, backtest.predictions))
            if not composite.empty:
                figures.append(plots.plot_enso_influence(composite, panel))
            figures.append(plots.plot_history_overview(load_daily_target(), panel))
            if not scenario_frame.empty:
                figures.append(plots.plot_scenarios(panel["level_m"], scenario_frame))
        except Exception as exc:  # noqa: BLE001 - a figure failure must not lose the forecast
            log.exception("Figure generation failed: %s", exc)

    report_path = write_report(
        panel, backtest, forecasts, weights, phase_summary, scenario_frame,
        horizon, target_date, figures, lag_diagnostics, intensity_summary,
    )

    return PipelineOutput(
        panel=panel,
        backtest=backtest,
        forecasts=forecasts,
        ensemble_weights=weights,
        figures=[f for f in figures if f],
        report_path=report_path,
        horizon=horizon,
    )


def load_daily_target() -> pd.Series:
    """Load the spliced daily target series written during data assembly."""
    from .data.build_dataset import DAILY_TARGET_PATH

    daily = pd.read_csv(DAILY_TARGET_PATH, parse_dates=["date"]).set_index("date")
    return daily["level_m"].sort_index()


def write_report(
    panel: pd.DataFrame,
    backtest: BacktestResult,
    forecasts: dict[str, pd.DataFrame],
    weights: dict[str, float],
    phase_summary: pd.DataFrame,
    scenario_frame: pd.DataFrame,
    horizon: int,
    target_date: pd.Timestamp,
    figures: list[str],
    lag_diagnostics: pd.DataFrame | None = None,
    intensity_summary: pd.DataFrame | None = None,
) -> str:
    """Write the human-readable Markdown forecast report."""
    observed = panel["level_m"].dropna()
    origin = observed.index.max()
    lines: list[str] = []

    lines.append(f"# Rio Negro at Manaus - forecast to {target_date:%B %Y}\n")
    lines.append(f"*Generated {date.today().isoformat()}*\n")

    lines.append("## Headline\n")
    headline_name = "Ensemble" if "Ensemble" in forecasts else backtest.best_model()
    headline = forecasts.get(headline_name)
    if headline is not None and not headline.empty:
        value = headline["forecast_m"].iloc[-1]
        lines.append(
            f"**{target_date:%B %Y} monthly mean stage: {value:.2f} m** "
            f"above the Porto de Manaus gauge zero, from the *{headline_name}* model.\n"
        )
        if {"lower_m", "upper_m"} <= set(headline.columns):
            lines.append(
                f"80% prediction interval: **{headline['lower_m'].iloc[-1]:.2f} m "
                f"to {headline['upper_m'].iloc[-1]:.2f} m**.\n"
            )
        lines.append(
            f"Forecast origin: last observed month {origin:%B %Y} "
            f"({observed.iloc[-1]:.2f} m), horizon {horizon} months.\n"
        )

        climatology = observed[observed.index.month == target_date.month]
        if len(climatology):
            anomaly = value - climatology.mean()
            lines.append(
                f"That is **{anomaly:+.2f} m** against the {target_date:%B} mean of "
                f"{climatology.mean():.2f} m "
                f"(observed {target_date:%B} range {climatology.min():.2f}-"
                f"{climatology.max():.2f} m over {len(climatology)} years).\n"
            )

    lines.append("\n## Forecast by model\n")
    rows = []
    for name, frame in forecasts.items():
        if frame.empty:
            continue
        row = {"Model": name}
        for stamp, value in frame["forecast_m"].items():
            row[f"{stamp:%b %Y}"] = round(float(value), 2)
        rows.append(row)
    if rows:
        lines.append(pd.DataFrame(rows).set_index("Model").to_markdown())
        lines.append("")

    lines.append("\n## Backtest skill (rolling origin, refit at every origin)\n")
    lines.append(backtest.overall.round(3).to_markdown())
    lines.append("\n\n### By lead time\n")
    lines.append(backtest.metrics.round(3).to_markdown())

    improvement = skill_vs_baseline(backtest)
    if not improvement.empty:
        lines.append("\n\n### RMSE improvement over Climatology\n")
        lines.append((improvement * 100).round(1).rename("improvement_%").to_markdown())

    lines.append("\n\n## Ensemble weights (inverse-MSE from the backtest)\n")
    lines.append(
        pd.Series(weights).sort_values(ascending=False).round(4).rename("weight").to_markdown()
    )

    if not phase_summary.empty:
        lines.append("\n\n## ENSO influence, observed (Sep-Dec stage anomaly)\n")
        lines.append(phase_summary.round(3).to_markdown())
        lines.append(
            "\n\nCompare the within-phase standard deviation against the between-phase "
            "difference in means. For this river they are of comparable size, which means the "
            "CPC phase label **shifts the odds without determining the outcome**. Any "
            "presentation of ENSO as the deciding factor for a single season overstates it.\n"
        )

    if intensity_summary is not None and not intensity_summary.empty:
        lines.append(f"\n\n### {target_date:%B} anomaly banded by ENSO intensity\n")
        lines.append(
            "Banded by the 6-month ONI mean ending three months before the target - the "
            "freshest ENSO state a 3-month-lead forecast could legitimately know. The phase "
            "label alone lumps a marginal event together with a record one, and the stage "
            "response is closer to threshold-like than linear.\n"
        )
        lines.append(intensity_summary.round(3).to_markdown())

    if lag_diagnostics is not None and not lag_diagnostics.empty:
        lines.append("\n\n### Which climate index actually leads the low-water season\n")
        lines.append(
            "Correlation between each index at lag L and the Oct-Dec stage anomaly. This is "
            "the empirical basis for the lag structure in the feature set - and the reason "
            "this project does not treat the Pacific as the only driver.\n"
        )
        lines.append(lag_diagnostics.round(3).to_markdown())

        strongest: list[str] = []
        for column in lag_diagnostics.columns:
            series = lag_diagnostics[column].dropna()
            if series.empty:
                continue
            peak_lag = int(series.abs().idxmax())
            strongest.append(
                f"- **{str(column).upper()}**: strongest at lag {peak_lag} months, "
                f"r = {series.loc[peak_lag]:+.3f}"
            )
        if strongest:
            lines.append("\n\nStrongest lag per index:\n")
            lines.extend(strongest)
            lines.append("")

    if not scenario_frame.empty:
        lines.append("\n\n## ENSO what-if scenarios\n")
        flat = scenario_frame.reset_index()
        date_column = flat.columns[0]
        pivot = flat.pivot_table(
            index="scenario", columns=date_column, values="forecast_m", sort=False
        )
        pivot.columns = [f"{pd.Timestamp(c):%b %Y}" for c in pivot.columns]
        lines.append(pivot.round(2).to_markdown())

    if figures:
        lines.append("\n\n## Figures\n")
        for figure in figures:
            if figure:
                name = figure.replace("\\", "/").split("/")[-1]
                lines.append(f"- `outputs/figures/{name}`")

    lines.append("\n\n## How to read this\n")
    lines.append(
        "Stage is metres above the local Porto de Manaus gauge zero, not metres above sea "
        "level, and the forecast target is the **monthly mean**, so the daily minimum "
        "inside the month will sit below it.\n"
    )
    lines.append(
        "Every number in the backtest table comes from models refitted on data available "
        "at the forecast origin only; no feature in the design matrix is lagged by less "
        "than the forecast horizon.\n"
    )

    path = REPORT_DIR / f"forecast_report_{date.today().isoformat()}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Report written to %s", path)
    return str(path)


def figures_from_artifacts(target_label: str | None = None) -> list[str]:
    """Regenerate every figure from the artefacts of the last run, without refitting.

    Backtesting and fitting cost far more than plotting, so iterating on a figure should
    not mean re-running the models. Everything the plots need was persisted by :func:`run`.
    """
    panel = load_panel()
    figures: list[str] = []

    forecast_files = sorted(FORECAST_DIR.glob("forecast_*.csv"))
    if not forecast_files:
        raise FileNotFoundError(
            f"No forecasts in {FORECAST_DIR}; run the pipeline at least once first"
        )
    combined = pd.read_csv(forecast_files[-1], parse_dates=["date"]).set_index("date")
    forecasts = {
        name: group.drop(columns="model") for name, group in combined.groupby("model")
    }
    headline = "Ensemble" if "Ensemble" in forecasts else next(iter(forecasts))
    label = target_label or f"{combined.index.max():%B %Y}"

    metrics_path = REPORT_DIR / "backtest_by_lead.csv"
    metrics = (
        pd.read_csv(metrics_path).set_index(["model", "lead"])
        if metrics_path.exists() else None
    )

    figures.append(
        plots.plot_forecast_comparison(
            panel["level_m"], forecasts, headline_model=headline,
            target_label=label, metrics=metrics,
        )
    )

    predictions_path = REPORT_DIR / "backtest_predictions.csv"
    if metrics is not None and predictions_path.exists():
        predictions = pd.read_csv(predictions_path, parse_dates=["origin", "target_date"])
        figures.append(plots.plot_model_skill(metrics, predictions))

    composite = enso_composite(panel)
    if not composite.empty:
        figures.append(plots.plot_enso_influence(composite, panel))

    figures.append(plots.plot_history_overview(load_daily_target(), panel))

    scenario_path = REPORT_DIR / "enso_scenarios.csv"
    if scenario_path.exists():
        scenario_frame = (
            pd.read_csv(scenario_path, parse_dates=["date"]).set_index("date")
        )
        figures.append(plots.plot_scenarios(panel["level_m"], scenario_frame))

    return [f for f in figures if f]
