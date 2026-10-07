"""Gradient-boosted regression as a direct h-step forecaster.

This is the only model in the stack that can use the full feature space - upstream gauge
levels, several climate indices at several lags, calendar harmonics and autoregressive
terms - and that can express non-linear interactions such as "El Nino matters much more
when the basin is already in a low-storage state".

It is fitted in *direct* multi-horizon form: a separate model per lead time, each mapping
origin-time features straight onto the target month. That avoids the error compounding of
recursive one-step models and keeps the leakage rule trivially enforceable (see
:mod:`manaus_river.features`).

Prediction intervals come from quantile regression - two extra fits, at the alpha/2 and
1-alpha/2 quantiles - rather than from a normal approximation, because stage errors are
visibly skewed in drought years.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import MODEL_CONFIG, ModelConfig
from ..features import build_design_matrix
from .base import BaseForecaster, ForecastResult

log = logging.getLogger(__name__)

#: Minimum training rows before a GBM fit at a given lead time is attempted.
MIN_TRAIN_ROWS = 60


class GBMForecaster(BaseForecaster):
    """Direct multi-horizon gradient boosting over the full lagged feature set."""

    name = "Gradient Boosting"

    def __init__(self, config: ModelConfig = MODEL_CONFIG, with_intervals: bool = True) -> None:
        self.config = config
        self.with_intervals = with_intervals
        self._models: dict[int, object] = {}
        self._lower_models: dict[int, object] = {}
        self._upper_models: dict[int, object] = {}
        self._columns: dict[int, list[str]] = {}
        self._medians: dict[int, pd.Series] = {}

    def _make_model(self, quantile: float | None = None):
        from sklearn.ensemble import GradientBoostingRegressor

        kwargs = dict(
            n_estimators=self.config.gbm_n_estimators,
            learning_rate=self.config.gbm_learning_rate,
            max_depth=self.config.gbm_max_depth,
            subsample=self.config.gbm_subsample,
            random_state=self.config.random_state,
        )
        if quantile is not None:
            return GradientBoostingRegressor(loss="quantile", alpha=quantile, **kwargs)
        return GradientBoostingRegressor(loss="squared_error", **kwargs)

    def fit(self, panel: pd.DataFrame, horizon: int) -> "GBMForecaster":
        self._models.clear()
        self._lower_models.clear()
        self._upper_models.clear()
        self._columns.clear()
        self._medians.clear()

        alpha = self.config.alpha
        for step in range(1, horizon + 1):
            # Build on the panel *extended* over the forecast window, exactly as predict()
            # will, so the staleness filter in build_design_matrix() sees the same column
            # availability in both places. Rows without a target drop out below.
            extended = panel.reindex(panel.index.union(self.future_index(panel, step)))
            design, target = build_design_matrix(extended, step, self.config)
            if len(design) < MIN_TRAIN_ROWS:
                log.warning("Too few rows (%d) to fit GBM at step %d", len(design), step)
                continue
            self._columns[step] = list(design.columns)
            self._medians[step] = design.median()
            self._models[step] = self._make_model().fit(design.values, target.values)
            if self.with_intervals:
                self._lower_models[step] = self._make_model(quantile=alpha / 2.0).fit(
                    design.values, target.values
                )
                self._upper_models[step] = self._make_model(quantile=1.0 - alpha / 2.0).fit(
                    design.values, target.values
                )

        if not self._models:
            raise RuntimeError("GBM could not be fitted at any horizon")
        return self

    def _features_at_origin(self, panel: pd.DataFrame, step: int) -> np.ndarray | None:
        """Feature row for the target month ``origin + step``.

        The design matrix is rebuilt on a panel extended by ``step`` empty months so the
        lag machinery lines up exactly as it did in training; we then take the last row.
        """
        future_index = self.future_index(panel, step)
        extended = panel.reindex(panel.index.union(future_index))
        design, _ = build_design_matrix(extended, step, self.config, dropna=False)
        design = design.reindex(columns=self._columns[step])
        row = design.iloc[[-1]]
        if row.isna().to_numpy().any():
            row = row.fillna(self._medians[step])
        if row.isna().to_numpy().any():
            return None
        return row.to_numpy(dtype=float)

    def predict(self, panel: pd.DataFrame, horizon: int, alpha: float = 0.2) -> ForecastResult:
        index = self.future_index(panel, horizon)
        means: list[float] = []
        lowers: list[float] = []
        uppers: list[float] = []

        for step in range(1, horizon + 1):
            features = (
                self._features_at_origin(panel, step) if step in self._models else None
            )
            if features is None:
                means.append(np.nan)
                lowers.append(np.nan)
                uppers.append(np.nan)
                continue
            means.append(float(self._models[step].predict(features)[0]))
            if self.with_intervals and step in self._lower_models:
                lowers.append(float(self._lower_models[step].predict(features)[0]))
                uppers.append(float(self._upper_models[step].predict(features)[0]))
            else:
                lowers.append(np.nan)
                uppers.append(np.nan)

        mean = pd.Series(means, index=index).interpolate(limit_direction="both")
        lower = pd.Series(lowers, index=index)
        upper = pd.Series(uppers, index=index)
        if lower.isna().all() or upper.isna().all():
            return ForecastResult(mean, None, None, self.name)
        # Quantile fits are independent, so pin the interval around the mean forecast.
        lower = pd.concat([lower, mean], axis=1).min(axis=1)
        upper = pd.concat([upper, mean], axis=1).max(axis=1)
        return ForecastResult(mean, lower, upper, self.name)

    def feature_importance(self, step: int = 1, top_n: int = 20) -> pd.Series:
        """Impurity-based importances for one lead time, most important first."""
        if step not in self._models:
            return pd.Series(dtype=float)
        importances = pd.Series(
            self._models[step].feature_importances_, index=self._columns[step]
        )
        return importances.sort_values(ascending=False).head(top_n)
