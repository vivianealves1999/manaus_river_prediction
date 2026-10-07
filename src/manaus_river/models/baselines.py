"""Reference models.

Every forecasting claim in this project is measured against these. A sophisticated model
that cannot beat monthly climatology on the Rio Negro is not adding value: the flood
pulse is so regular that climatology alone is a genuinely strong 3-month forecast.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseForecaster, ForecastResult


class ClimatologyForecaster(BaseForecaster):
    """Predict the historical mean stage for the target calendar month.

    Optionally weights recent decades more heavily, which matters because the Amazon has
    shown increased hydrological extremes since roughly 2000.
    """

    name = "Climatology"

    def __init__(self, recent_years: int | None = 40) -> None:
        self.recent_years = recent_years
        self._monthly_mean: pd.Series | None = None
        self._monthly_std: pd.Series | None = None

    def fit(self, panel: pd.DataFrame, horizon: int) -> "ClimatologyForecaster":
        series = panel["level_m"].dropna()
        if self.recent_years is not None:
            cutoff = series.index.max() - pd.DateOffset(years=self.recent_years)
            recent = series[series.index >= cutoff]
            if len(recent) >= 12 * 10:
                series = recent
        self._monthly_mean = series.groupby(series.index.month).mean()
        self._monthly_std = series.groupby(series.index.month).std()
        return self

    def predict(self, panel: pd.DataFrame, horizon: int, alpha: float = 0.2) -> ForecastResult:
        if self._monthly_mean is None:
            raise RuntimeError("fit() must be called before predict()")
        index = self.future_index(panel, horizon)
        mean = pd.Series([self._monthly_mean[m] for m in index.month], index=index)
        std = np.array([self._monthly_std.get(m, mean.std()) for m in index.month])
        lower, upper = self._empirical_interval(std, mean, alpha)
        return ForecastResult(mean, lower, upper, self.name)


class SeasonalNaiveForecaster(BaseForecaster):
    """Last year's value for the same month, shifted by the current annual anomaly.

    The drift term is what makes this competitive: it carries forward the fact that the
    basin is currently, say, 2 m below normal, instead of blindly repeating last year.
    """

    name = "Seasonal Naive + drift"

    def __init__(self, drift_window: int = 6) -> None:
        self.drift_window = drift_window
        self._series: pd.Series | None = None
        self._drift: float = 0.0
        self._resid_std: float = 1.0

    def fit(self, panel: pd.DataFrame, horizon: int) -> "SeasonalNaiveForecaster":
        series = panel["level_m"].dropna()
        self._series = series
        recent = series.iloc[-self.drift_window:]
        year_ago = series.shift(12).iloc[-self.drift_window:]
        self._drift = float((recent - year_ago).mean()) if year_ago.notna().any() else 0.0
        residuals = (series - series.shift(12)).dropna()
        self._resid_std = float(residuals.std()) if len(residuals) > 12 else 1.0
        return self

    def predict(self, panel: pd.DataFrame, horizon: int, alpha: float = 0.2) -> ForecastResult:
        if self._series is None:
            raise RuntimeError("fit() must be called before predict()")
        index = self.future_index(panel, horizon)
        values = []
        for stamp in index:
            reference = stamp - pd.DateOffset(years=1)
            if reference in self._series.index and pd.notna(self._series[reference]):
                values.append(self._series[reference] + self._drift)
            else:
                same_month = self._series[self._series.index.month == stamp.month]
                values.append(float(same_month.mean()) if len(same_month) else float(self._series.mean()))
        mean = pd.Series(values, index=index)
        # Uncertainty widens with lead time.
        scale = self._resid_std * np.sqrt(np.arange(1, horizon + 1) / horizon)
        lower, upper = self._empirical_interval(scale, mean, alpha)
        return ForecastResult(mean, lower, upper, self.name)
