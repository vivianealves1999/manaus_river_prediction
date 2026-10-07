"""Rolling-origin backtesting.

This is the part of the project that decides which number the user is told in December, so
it is written to be pessimistic rather than flattering.

Protocol
--------
For each origin month ``t`` in the evaluation window:

1. Truncate the panel to ``<= t``. *Everything* downstream sees only that slice - the
   models are re-fitted from scratch, and the feature pipeline recomputes climatology,
   rolling means and scaling on the truncated data only.
2. Forecast ``t+1 ... t+h``.
3. Score against the observed values, bucketed by lead time.

Re-fitting at every origin is expensive (SARIMAX order search dominates) but it is the only
way to get an honest estimate: fitting once on the full history and then "forecasting" the
interior would leak the answer through the estimated parameters.

Metrics are reported per lead time rather than pooled, because a 1-month forecast of this
river is nearly trivial and a 3-month forecast is not; pooling the two flatters the model.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import MODEL_CONFIG, ModelConfig
from .models.base import BaseForecaster

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Per-origin, per-lead predictions plus the derived skill table."""

    predictions: pd.DataFrame       # origin, target_date, lead, model, y_true, y_pred, lower, upper
    metrics: pd.DataFrame          # model x lead -> MAE, RMSE, MAPE, bias, coverage
    overall: pd.DataFrame          # model -> metrics averaged across leads
    config: ModelConfig = field(default_factory=lambda: MODEL_CONFIG)

    def best_model(self, metric: str = "RMSE") -> str:
        return str(self.overall[metric].idxmin())

    def rmse_by_model(self) -> dict[str, float]:
        return self.overall["RMSE"].to_dict()


def _origins(panel: pd.DataFrame, config: ModelConfig) -> list[pd.Timestamp]:
    """Choose evaluation origins: the most recent ``backtest_origins`` months that leave
    both a long enough training history behind them and a full horizon ahead."""
    observed = panel["level_m"].dropna()
    if observed.empty:
        return []

    first_allowed = observed.index.min() + pd.DateOffset(years=config.min_train_years)
    last_allowed = observed.index.max() - pd.DateOffset(months=config.horizon)
    candidates = [
        stamp for stamp in observed.index if first_allowed <= stamp <= last_allowed
    ]
    return candidates[-config.backtest_origins:]


def rolling_origin_backtest(
    panel: pd.DataFrame,
    model_factory: dict[str, callable],
    config: ModelConfig = MODEL_CONFIG,
    origins: list[pd.Timestamp] | None = None,
) -> BacktestResult:
    """Evaluate every model in ``model_factory`` over rolling origins.

    ``model_factory`` maps a display name to a zero-argument callable returning a *fresh*
    unfitted forecaster. A factory (rather than an instance) is required precisely because
    every origin must start from an unfitted model.
    """
    origins = origins if origins is not None else _origins(panel, config)
    if not origins:
        raise ValueError("No valid backtest origins: history too short for the configuration")

    log.info(
        "Backtesting %d models over %d origins (%s..%s), horizon %d",
        len(model_factory), len(origins), origins[0].date(), origins[-1].date(), config.horizon,
    )

    truth = panel["level_m"]
    records: list[dict] = []

    for position, origin in enumerate(origins, start=1):
        train = panel.loc[:origin]
        if position == 1 or position % 10 == 0 or position == len(origins):
            log.info("  origin %d/%d: %s (train n=%d)",
                     position, len(origins), origin.date(), len(train))

        for name, factory in model_factory.items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model: BaseForecaster = factory()
                    model.fit(train, config.horizon)
                    result = model.predict(train, config.horizon, config.alpha)
            except Exception as exc:  # noqa: BLE001 - record the gap, keep the run alive
                log.warning("    %s failed at origin %s: %s", name, origin.date(), exc)
                continue

            for lead, target_date in enumerate(result.mean.index, start=1):
                if target_date not in truth.index:
                    continue
                y_true = truth.loc[target_date]
                if pd.isna(y_true):
                    continue
                records.append(
                    {
                        "origin": origin,
                        "target_date": target_date,
                        "lead": lead,
                        "model": name,
                        "y_true": float(y_true),
                        "y_pred": float(result.mean.loc[target_date]),
                        "lower": (
                            float(result.lower.loc[target_date])
                            if result.lower is not None else np.nan
                        ),
                        "upper": (
                            float(result.upper.loc[target_date])
                            if result.upper is not None else np.nan
                        ),
                    }
                )

    if not records:
        raise RuntimeError("Backtest produced no scoreable predictions")

    predictions = pd.DataFrame(records)
    metrics = _score(predictions, group_by=["model", "lead"])
    overall = _score(predictions, group_by=["model"])
    return BacktestResult(predictions, metrics, overall, config)


def _score(predictions: pd.DataFrame, group_by: list[str]) -> pd.DataFrame:
    """Compute the skill table for a grouping of the prediction records."""

    def _metrics(group: pd.DataFrame) -> pd.Series:
        error = group["y_pred"] - group["y_true"]
        covered = (
            ((group["y_true"] >= group["lower"]) & (group["y_true"] <= group["upper"])).mean()
            if group["lower"].notna().any() else np.nan
        )
        return pd.Series(
            {
                "n": len(group),
                "MAE": error.abs().mean(),
                "RMSE": np.sqrt((error ** 2).mean()),
                # Stage is metres above an arbitrary datum, so percentage error is not
                # physically meaningful; it is reported only for comparability.
                "MAPE": (error.abs() / group["y_true"].abs()).mean() * 100.0,
                "bias": error.mean(),
                "coverage": covered,
            }
        )

    scored = predictions.groupby(group_by, observed=True).apply(_metrics, include_groups=False)
    # A single-key table (model) reads best ranked by skill; a multi-key table (model x lead)
    # reads best in key order, so the lead progression within each model is legible.
    if len(group_by) > 1:
        return scored.sort_index()
    return scored.sort_values("RMSE") if "RMSE" in scored.columns else scored


def skill_vs_baseline(
    result: BacktestResult, baseline: str = "Climatology", metric: str = "RMSE"
) -> pd.Series:
    """Fractional improvement over a reference model, per model (higher is better).

    A value of 0.20 means the model cuts the baseline's error by 20%. Negative means the
    model is *worse* than the baseline and should not be trusted.
    """
    overall = result.overall
    if baseline not in overall.index:
        return pd.Series(dtype=float)
    reference = overall.loc[baseline, metric]
    return ((reference - overall[metric]) / reference).sort_values(ascending=False)
