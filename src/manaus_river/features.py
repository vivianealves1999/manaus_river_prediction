"""Feature engineering for the monthly stage panel.

Design rule that governs this whole module
------------------------------------------
Every feature must be *knowable at forecast time*. When we stand at origin month ``t``
and predict month ``t + h``, a feature attached to ``t + h`` may only use information
available at ``t``. Concretely this means an upstream gauge reading or climate index can
only enter the design matrix at a lag of at least ``h`` months. :func:`build_design_matrix`
enforces this rather than leaving it to the caller, because it is the single easiest way
to leak the future into a hydrological model and silently produce a backtest that looks
excellent and forecasts badly.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import MODEL_CONFIG, ModelConfig

log = logging.getLogger(__name__)

TARGET = "level_m"
CLIMATE_COLUMNS = ("oni", "tna", "tsa", "soi", "amo")


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Month-of-year encoded as Fourier harmonics plus a slow linear trend.

    The Rio Negro flood pulse is a smooth annual wave (low water Oct-Nov, peak Jun-Jul),
    so two or three harmonics capture it with far fewer parameters than 11 monthly dummies.
    """
    out = frame.copy()
    month = out.index.month.to_numpy()
    for k in (1, 2, 3):
        out[f"sin_{k}"] = np.sin(2 * np.pi * k * month / 12.0)
        out[f"cos_{k}"] = np.cos(2 * np.pi * k * month / 12.0)
    # Centred decimal year: captures any slow secular drift without exploding in scale.
    out["time_idx"] = (out.index.year + (month - 1) / 12.0) - 1980.0
    return out


def add_target_lags(frame: pd.DataFrame, horizon: int, max_lag: int = 24) -> pd.DataFrame:
    """Autoregressive features on the target, all shifted by at least ``horizon``.

    ``level_lag_h`` is the most recent observation available at the forecast origin; the
    remaining lags describe the shape of the recession/rise the basin is currently on.
    """
    out = frame.copy()
    base = out[TARGET]
    for lag in (horizon, horizon + 1, horizon + 2, horizon + 3, horizon + 6, horizon + 12):
        if lag <= max_lag + horizon:
            out[f"level_lag_{lag}"] = base.shift(lag)

    # Rate of change over the three months leading up to the origin: distinguishes a
    # basin that is filling from one that is draining at the same absolute stage.
    out[f"level_delta_{horizon}"] = base.shift(horizon) - base.shift(horizon + 3)
    # Anomaly relative to the long-run climatology of the origin month.
    climatology = base.groupby(base.index.month).transform("mean")
    out[f"level_anom_{horizon}"] = (base - climatology).shift(horizon)
    # 12-month rolling mean at the origin: basin storage / multi-year wet-dry state.
    out[f"level_roll12_{horizon}"] = base.rolling(12, min_periods=6).mean().shift(horizon)
    return out


def add_climate_features(
    frame: pd.DataFrame, horizon: int, config: ModelConfig = MODEL_CONFIG
) -> pd.DataFrame:
    """Lagged ENSO and Atlantic SST features.

    Lags below ``horizon`` are skipped: a 3-month-ahead forecast cannot use the ONI value
    of the month it is predicting. Where a lag would be illegal we shift to ``horizon``,
    which is the freshest legitimate value.
    """
    out = frame.copy()
    for column in CLIMATE_COLUMNS:
        if column not in out.columns:
            continue
        series = out[column]
        for lag in config.enso_lags:
            effective_lag = max(lag, horizon)
            out[f"{column}_lag_{effective_lag}"] = series.shift(effective_lag)
        # ONI 3-month running mean is the operational ENSO definition; its lagged value
        # is a smoother, less noisy signal than a single month.
        out[f"{column}_run3_lag_{horizon}"] = (
            series.rolling(3, min_periods=2).mean().shift(horizon)
        )
    return out


def add_upstream_features(
    frame: pd.DataFrame, horizon: int, config: ModelConfig = MODEL_CONFIG
) -> pd.DataFrame:
    """Lagged stage at the upstream gauges (water already in the system).

    This is the physically strongest class of predictor at short lead: water measured at
    Manacapuru or Caracarai today is, to a first approximation, water that will be past
    Manaus within weeks. Same lag legality rule as the climate features.
    """
    out = frame.copy()
    upstream_columns = [
        c for c in out.columns if c.endswith("_level_m") and c != TARGET
    ]
    for column in upstream_columns:
        series = out[column]
        for lag in config.upstream_lags:
            effective_lag = max(lag, horizon)
            out[f"{column}_lag_{effective_lag}"] = series.shift(effective_lag)
    return out


def build_design_matrix(
    panel: pd.DataFrame,
    horizon: int,
    config: ModelConfig = MODEL_CONFIG,
    dropna: bool = True,
    require_recent: int = 12,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, y)`` for direct h-step-ahead regression.

    Row ``t`` of ``X`` holds only information observable at month ``t - horizon`` or
    earlier, and ``y[t]`` is the stage at month ``t``. Fitting ``X -> y`` therefore learns
    a direct h-step map with no recursion and no leakage.

    ``require_recent`` drops *stale* predictors: a column with no observation anywhere in
    the final ``require_recent`` rows is discarded even if its historical coverage is
    excellent. This matters because several sources in this project lag badly - the ANA
    upstream gauges stop around 2025-09 and the AMO index at 2023-01. Keeping such a column
    would mean imputing it at forecast time, i.e. feeding the model a fabricated value for
    one of its predictors and presenting the result as a forecast. Better to lose the
    feature than to invent it.
    """
    frame = panel.copy()
    frame = add_calendar_features(frame)
    frame = add_target_lags(frame, horizon)
    frame = add_climate_features(frame, horizon, config)
    frame = add_upstream_features(frame, horizon, config)

    target = frame[TARGET]
    drop_columns = [
        TARGET, "level_min_m", "level_max_m", "n_days", "enso_phase", "consistency",
    ]
    # Raw contemporaneous predictors are dropped: only their lagged versions are legal.
    drop_columns += [c for c in frame.columns if c.endswith("_level_m")]
    drop_columns += [c for c in CLIMATE_COLUMNS if c in frame.columns]

    design = frame.drop(columns=[c for c in drop_columns if c in frame.columns])
    design = design.loc[:, design.notna().mean() > 0.30]  # discard near-empty predictors

    if require_recent and len(design) > require_recent:
        tail = design.iloc[-require_recent:]
        fresh = tail.notna().any(axis=0)
        stale = [str(c) for c in design.columns[~fresh]]
        if stale:
            log.debug("Dropping %d stale predictor(s): %s", len(stale), ", ".join(stale))
        design = design.loc[:, fresh]

    if dropna:
        mask = target.notna() & design.notna().all(axis=1)
        return design.loc[mask], target.loc[mask]
    return design, target


def exog_for_sarimax(
    panel: pd.DataFrame, horizon: int, config: ModelConfig = MODEL_CONFIG
) -> pd.DataFrame:
    """A compact, dense exogenous block for SARIMAX.

    SARIMAX already models seasonality and autocorrelation internally, so the exogenous
    matrix should contain only the *external* drivers. It must also be dense over the
    whole estimation window and extendable into the forecast window, which rules out the
    upstream gauges (their records are shorter and themselves need forecasting) and
    favours a handful of well-behaved climate lags.
    """
    columns: dict[str, pd.Series] = {}
    for column in ("oni", "tna"):
        if column not in panel.columns:
            continue
        running = panel[column].rolling(3, min_periods=1).mean()
        for lag in (horizon, horizon + 3, horizon + 6):
            columns[f"{column}_run3_lag_{lag}"] = running.shift(lag)

    if not columns:
        return pd.DataFrame(index=panel.index)

    exog = pd.DataFrame(columns, index=panel.index)
    # Short interior gaps in the index record are safe to interpolate; the climate
    # indices are smooth by construction.
    return exog.interpolate(limit_direction="both")
