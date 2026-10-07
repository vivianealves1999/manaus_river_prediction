"""Weighted ensemble of the individual forecasters.

Ensembling is not a flourish here, it is the main defence against a specific failure mode:
each model in this stack is wrong in a different way at the seasonal turning points, and
the Rio Negro forecast that matters (the October-December low-water trough) sits exactly on
one. Averaging decorrelated errors is the cheapest reliable accuracy gain available.

Weights are derived from backtest skill rather than set by hand. We use inverse-MSE
weighting over the rolling-origin results, which is the closed-form optimum when errors
are unbiased and uncorrelated, and is far more robust out-of-sample than solving for
unconstrained optimal weights on a short backtest.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .base import BaseForecaster, ForecastResult

log = logging.getLogger(__name__)


class WeightedEnsemble(BaseForecaster):
    """Combine member forecasts with fixed or skill-derived weights."""

    name = "Ensemble"

    def __init__(
        self,
        members: list[BaseForecaster],
        weights: dict[str, float] | None = None,
    ) -> None:
        if not members:
            raise ValueError("An ensemble needs at least one member")
        self.members = members
        self.weights = weights

    @staticmethod
    def weights_from_scores(
        rmse_by_model: dict[str, float], floor: float = 1e-6
    ) -> dict[str, float]:
        """Inverse-MSE weights, normalised to sum to one.

        Models with a non-finite score are dropped rather than given zero weight, so a
        model that failed to backtest cannot silently dilute the ensemble.
        """
        inverse = {
            name: 1.0 / max(rmse ** 2, floor)
            for name, rmse in rmse_by_model.items()
            if np.isfinite(rmse) and rmse > 0
        }
        total = sum(inverse.values())
        if total <= 0:
            equal = 1.0 / len(rmse_by_model)
            return {name: equal for name in rmse_by_model}
        return {name: value / total for name, value in inverse.items()}

    def fit(self, panel: pd.DataFrame, horizon: int) -> "WeightedEnsemble":
        for member in self.members:
            try:
                member.fit(panel, horizon)
            except Exception as exc:  # noqa: BLE001 - one bad member must not kill the run
                log.error("Ensemble member %s failed to fit: %s", member.name, exc)
        return self

    def predict(self, panel: pd.DataFrame, horizon: int, alpha: float = 0.2) -> ForecastResult:
        results: dict[str, ForecastResult] = {}
        for member in self.members:
            try:
                results[member.name] = member.predict(panel, horizon, alpha)
            except Exception as exc:  # noqa: BLE001
                log.error("Ensemble member %s failed to predict: %s", member.name, exc)

        if not results:
            raise RuntimeError("No ensemble member produced a forecast")

        if self.weights:
            active = {n: self.weights.get(n, 0.0) for n in results}
            total = sum(active.values())
            active = (
                {n: w / total for n, w in active.items()}
                if total > 0
                else {n: 1.0 / len(results) for n in results}
            )
        else:
            active = {n: 1.0 / len(results) for n in results}

        index = self.future_index(panel, horizon)
        mean = pd.Series(0.0, index=index)
        for name, result in results.items():
            mean = mean.add(result.mean.reindex(index) * active[name], fill_value=0.0)

        # Ensemble interval: weighted member spread combined with the between-member
        # disagreement, which is itself a useful (and often the dominant) uncertainty signal.
        half_widths = pd.Series(0.0, index=index)
        for name, result in results.items():
            if result.lower is None or result.upper is None:
                continue
            width = (result.upper.reindex(index) - result.lower.reindex(index)) / 2.0
            half_widths = half_widths.add(width * active[name], fill_value=0.0)

        member_matrix = pd.DataFrame(
            {n: r.mean.reindex(index) for n, r in results.items()}
        )
        disagreement = member_matrix.std(axis=1).fillna(0.0)
        total_half_width = np.sqrt(half_widths ** 2 + disagreement ** 2)

        self.active_weights = active
        return ForecastResult(
            mean, mean - total_half_width, mean + total_half_width, self.name
        )
