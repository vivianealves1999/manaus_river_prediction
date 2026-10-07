"""Common interface for every forecaster in the project.

Keeping a single tiny protocol means the backtester, the evaluation report and the
plotting layer never need to know which model they are holding, and adding a new model is
a matter of implementing three methods.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ForecastResult:
    """A forecast with optional prediction interval, indexed by target month."""

    mean: pd.Series
    lower: pd.Series | None = None
    upper: pd.Series | None = None
    model_name: str = ""

    def to_frame(self) -> pd.DataFrame:
        data = {"forecast_m": self.mean}
        if self.lower is not None:
            data["lower_m"] = self.lower
        if self.upper is not None:
            data["upper_m"] = self.upper
        frame = pd.DataFrame(data)
        frame.insert(0, "model", self.model_name)
        return frame


class BaseForecaster(ABC):
    """All models are fit on a panel up to an origin and asked for h steps ahead."""

    name: str = "base"
    #: Set False for models that cannot express a prediction interval.
    supports_intervals: bool = True

    @abstractmethod
    def fit(self, panel: pd.DataFrame, horizon: int) -> "BaseForecaster":
        """Fit on the full history contained in ``panel`` (target column ``level_m``)."""

    @abstractmethod
    def predict(self, panel: pd.DataFrame, horizon: int, alpha: float = 0.2) -> ForecastResult:
        """Forecast months ``1..horizon`` after the last observed month in ``panel``."""

    # -- helpers shared by concrete models -------------------------------------------
    @staticmethod
    def future_index(panel: pd.DataFrame, horizon: int) -> pd.DatetimeIndex:
        last = panel.index.max()
        return pd.date_range(last + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")

    @staticmethod
    def _empirical_interval(
        residual_std: float | np.ndarray, mean: pd.Series, alpha: float
    ) -> tuple[pd.Series, pd.Series]:
        """Normal-approximation interval from a residual scale.

        Used by models that have no analytic predictive distribution. The scale may be a
        scalar or a per-horizon array, which is the honest choice: uncertainty grows with
        lead time.
        """
        from scipy.stats import norm

        z = norm.ppf(1.0 - alpha / 2.0)
        spread = z * np.asarray(residual_std, dtype=float)
        return mean - spread, mean + spread

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} name={self.name!r}>"
