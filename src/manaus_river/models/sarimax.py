"""SARIMAX models for monthly Rio Negro stage.

Two variants are registered:

``SARIMAX``      - pure endogenous seasonal ARIMA. Captures the annual flood pulse and
                   the very strong month-to-month persistence of river stage.
``SARIMAX+ENSO`` - the same specification plus an exogenous block of lagged ONI/TNA
                   running means.

Why the exogenous block is legitimate at forecast time
------------------------------------------------------
Every exogenous column is the climate index lagged by at least ``horizon`` months
(see :func:`manaus_river.features.exog_for_sarimax`). Forecasting month ``t+3`` therefore
uses the ONI of month ``t`` or earlier - a value that is already observed. No climate
forecast is required, and no future information leaks in. The cost is that we cannot
exploit CPC's own ENSO outlook; :mod:`manaus_river.scenarios` covers that separately.

Order selection is a small guided search rather than a full auto-ARIMA: the candidate set
in the config encodes what is already known about this series (strong seasonality,
near-unit-root persistence), and AIC picks among them on the training window only.

Two runtime concessions, both of which also happen to be defensible modelling choices:

``max_train_years``
    The state-space filter is linear in sample length, and the panel holds 120+ years. We
    fit on the most recent ``max_train_years`` only. Beyond being ~2x faster this is
    arguably *better*: the Amazon's hydrological regime has shifted measurably since the
    mid-20th century, and 1900s data pulls the seasonal profile toward a climate that no
    longer exists. (The long record is still used in full by the climatology baseline and
    the ENSO composites, where more history is unambiguously better.)

``_ORDER_CACHE``
    The order search is performed once per model variant and the winner reused across
    backtest origins. Re-searching at every origin roughly quintuples runtime, and holding
    the specification fixed across the evaluation is the more honest test of *that
    specification* - otherwise the backtest silently measures the search procedure too.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from ..config import MODEL_CONFIG, ModelConfig
from ..features import exog_for_sarimax
from .base import BaseForecaster, ForecastResult

log = logging.getLogger(__name__)

#: Order chosen per model variant on first fit, then reused. See the module docstring.
_ORDER_CACHE: dict[str, tuple] = {}


def clear_order_cache() -> None:
    """Forget cached SARIMAX specifications (call between unrelated experiments)."""
    _ORDER_CACHE.clear()


class SarimaxForecaster(BaseForecaster):
    """Seasonal ARIMA with an optional exogenous climate block."""

    def __init__(
        self,
        use_exog: bool = False,
        config: ModelConfig = MODEL_CONFIG,
        order: tuple[int, int, int] | None = None,
        seasonal_order: tuple[int, int, int, int] | None = None,
        max_train_years: int = 50,
        cache_order: bool = True,
    ) -> None:
        self.use_exog = use_exog
        self.config = config
        self.order = order
        self.seasonal_order = seasonal_order
        self.max_train_years = max_train_years
        self.cache_order = cache_order
        self.name = "SARIMAX+ENSO" if use_exog else "SARIMAX"
        self._result = None
        self._endog: pd.Series | None = None
        self._selected: tuple | None = None

    # -- internals --------------------------------------------------------------------
    def _prepare_endog(self, panel: pd.DataFrame) -> pd.Series:
        """Monthly target on a gap-free MS index, trimmed to the recent regime.

        statsmodels needs a regular frequency. Interior gaps (a handful of months with too
        few daily readings) are interpolated; the series is trimmed to the first and last
        genuine observation so we never extrapolate at the edges, then to the most recent
        ``max_train_years``.
        """
        series = panel["level_m"].astype(float)
        first, last = series.first_valid_index(), series.last_valid_index()
        series = series.loc[first:last]
        series = series.asfreq("MS").interpolate(method="time", limit_direction="both")
        if self.max_train_years:
            cutoff = series.index.max() - pd.DateOffset(years=self.max_train_years)
            trimmed = series.loc[series.index >= cutoff]
            # Never trim below what a seasonal model needs to see.
            if len(trimmed) >= 12 * 15:
                series = trimmed
        return series

    def _build_exog(self, panel: pd.DataFrame, horizon: int, n_future: int = 0) -> pd.DataFrame | None:
        """Exogenous matrix covering the history and, if asked, the forecast window."""
        if not self.use_exog:
            return None
        frame = panel
        if n_future:
            future_index = self.future_index(panel, n_future)
            frame = panel.reindex(panel.index.union(future_index))
        exog = exog_for_sarimax(frame, horizon, self.config)
        if exog.empty or exog.shape[1] == 0:
            return None
        return exog

    # -- API --------------------------------------------------------------------------
    def fit(self, panel: pd.DataFrame, horizon: int) -> "SarimaxForecaster":
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        endog = self._prepare_endog(panel)
        self._endog = endog

        exog = self._build_exog(panel, horizon)
        exog_train = exog.reindex(endog.index) if exog is not None else None
        if exog_train is not None:
            exog_train = exog_train.interpolate(limit_direction="both")
            if exog_train.isna().any().any():
                log.warning("Dropping exogenous block for %s: unfillable gaps", self.name)
                exog_train = None

        if self.order is not None:
            candidates = [(self.order, self.seasonal_order)]
        elif self.cache_order and self.name in _ORDER_CACHE:
            candidates = [_ORDER_CACHE[self.name]]
        else:
            candidates = list(self.config.sarimax_orders)

        best_aic, best_result, best_spec = np.inf, None, None
        for order, seasonal_order in candidates:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = SARIMAX(
                        endog,
                        exog=exog_train,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    result = model.fit(disp=False, maxiter=200)
                if np.isfinite(result.aic) and result.aic < best_aic:
                    best_aic, best_result, best_spec = result.aic, result, (order, seasonal_order)
            except Exception as exc:  # noqa: BLE001 - a failed candidate is not fatal
                log.debug("SARIMAX%s%s failed: %s", order, seasonal_order, exc)

        if best_result is None:
            raise RuntimeError(f"No SARIMAX specification converged for {self.name}")

        self._result = best_result
        self._selected = best_spec
        if self.cache_order and self.order is None and self.name not in _ORDER_CACHE:
            _ORDER_CACHE[self.name] = best_spec
            log.info("%s selected SARIMAX%s%s (AIC=%.1f), caching specification",
                     self.name, *best_spec, best_aic)
        else:
            log.debug("%s fitted SARIMAX%s%s (AIC=%.1f)", self.name, *best_spec, best_aic)
        return self

    def predict(self, panel: pd.DataFrame, horizon: int, alpha: float = 0.2) -> ForecastResult:
        if self._result is None:
            raise RuntimeError("fit() must be called before predict()")

        exog_future = None
        if self.use_exog and self._result.model.exog is not None:
            exog_all = self._build_exog(panel, horizon, n_future=horizon)
            if exog_all is not None:
                index = self.future_index(panel, horizon)
                exog_future = exog_all.reindex(index).ffill().bfill()

        forecast = self._result.get_forecast(steps=horizon, exog=exog_future)
        index = self.future_index(panel, horizon)
        mean = pd.Series(np.asarray(forecast.predicted_mean), index=index)
        conf = forecast.conf_int(alpha=alpha)
        lower = pd.Series(np.asarray(conf.iloc[:, 0]), index=index)
        upper = pd.Series(np.asarray(conf.iloc[:, 1]), index=index)
        return ForecastResult(mean, lower, upper, self.name)

    @property
    def selected_order(self) -> tuple | None:
        return self._selected

    def summary(self) -> str:
        return "" if self._result is None else str(self._result.summary())
