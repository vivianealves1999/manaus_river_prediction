"""Exponential smoothing (Holt-Winters) for monthly stage.

Included as a structurally different competitor to SARIMAX: it models level, trend and a
seasonal profile as explicitly evolving states rather than as an autocorrelation
structure. On a series whose seasonal amplitude itself varies between wet and dry decades
that difference is real, and it gives the ensemble genuinely decorrelated errors.

Additive seasonality is used rather than multiplicative because stage is measured against
an arbitrary gauge datum, so proportional seasonal effects have no physical meaning here.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from .base import BaseForecaster, ForecastResult

log = logging.getLogger(__name__)


class ETSForecaster(BaseForecaster):
    """Holt-Winters additive with damped trend."""

    name = "Holt-Winters (ETS)"

    def __init__(self, damped_trend: bool = True, seasonal_periods: int = 12) -> None:
        self.damped_trend = damped_trend
        self.seasonal_periods = seasonal_periods
        self._result = None

    def fit(self, panel: pd.DataFrame, horizon: int) -> "ETSForecaster":
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        series = panel["level_m"].astype(float)
        first, last = series.first_valid_index(), series.last_valid_index()
        series = series.loc[first:last]
        series = series.asfreq("MS").interpolate(method="time", limit_direction="both")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                series,
                trend="add",
                damped_trend=self.damped_trend,
                seasonal="add",
                seasonal_periods=self.seasonal_periods,
                initialization_method="estimated",
            )
            self._result = model.fit(optimized=True)
        return self

    def predict(self, panel: pd.DataFrame, horizon: int, alpha: float = 0.2) -> ForecastResult:
        if self._result is None:
            raise RuntimeError("fit() must be called before predict()")
        index = self.future_index(panel, horizon)
        mean = pd.Series(np.asarray(self._result.forecast(horizon)), index=index)

        # Holt-Winters has no analytic predictive interval here, so we widen the in-sample
        # residual scale with the square root of lead time.
        resid_std = float(np.nanstd(self._result.resid))
        scale = resid_std * np.sqrt(np.arange(1, horizon + 1))
        lower, upper = self._empirical_interval(scale, mean, alpha)
        return ForecastResult(mean, lower, upper, self.name)
