"""ENSO scenario analysis.

The rest of the modelling stack only uses *observed* climate indices, lagged far enough
back to be legal at forecast time (see :mod:`manaus_river.features`). That is the right
default: it needs no external forecast and cannot leak.

But it leaves something on the table. Agencies such as NOAA CPC and the IRI publish an
ENSO *outlook* months ahead, and the user explicitly wants to know how an El Nino changes
the Manaus answer. This module supplies that as an explicit, clearly-labelled
what-if layer rather than folding an unverifiable assumption into the headline number.

Two things are provided:

``enso_composite``      What the historical record says stage anomalies at Manaus look like
                        during each ENSO phase, by calendar month. Pure observation, no model.
``scenario_forecast``   Re-runs a model with the future ONI path overridden, so the
                        forecast can be read under "ENSO persists", "strong El Nino",
                        "La Nina" and so on.

Why the teleconnection is physically real, and predictable
----------------------------------------------------------
El Nino warms the central/eastern equatorial Pacific. That shifts the ascending branch of
the Walker circulation east, off the Amazon, and places anomalous *subsidence* over
northern South America. Subsiding air suppresses convection, so catchment rainfall drops -
most strongly in the northern basin, which is exactly the Rio Negro's drainage. Reduced
rainfall depletes soil and channel storage, and the stage at Manaus falls with a lag of
roughly one to two seasons because the catchment integrates rainfall before routing it past
the gauge.

The reason this is forecastable at 3-6 month lead is the ocean's thermal inertia: a Nino
3.4 SST anomaly is a slow, high-heat-capacity signal that persists and evolves
predictably, unlike the atmosphere it then drives. That asymmetry - slow ocean forcing a
fast atmosphere - is what gives seasonal forecasting any skill at all.

One important caveat this project takes seriously: the Pacific is not the only driver. A
warm Tropical North Atlantic displaces the ITCZ northward and has caused some of the
Amazon's worst droughts (2005, 2010) with ENSO near neutral. TNA is therefore carried
alongside ONI throughout.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import MODEL_CONFIG, ModelConfig
from .data.climate import classify_enso
from .models.base import BaseForecaster, ForecastResult

log = logging.getLogger(__name__)

#: Canonical ONI paths for the what-if scenarios, in degrees C.
SCENARIO_ONI_OFFSETS: dict[str, float | None] = {
    "Observed persistence": None,   # hold the latest observed ONI flat
    "Strong El Nino (+1.5)": 1.5,
    "Moderate El Nino (+1.0)": 1.0,
    "Neutral (0.0)": 0.0,
    "Moderate La Nina (-1.0)": -1.0,
}


def enso_composite(panel: pd.DataFrame, min_count: int = 8) -> pd.DataFrame:
    """Observed stage anomaly at Manaus by calendar month and ENSO phase.

    The anomaly is taken against each month's own long-run mean, so the seasonal cycle is
    removed and what remains is attributable to the phase. Cells backed by fewer than
    ``min_count`` months are blanked rather than reported, because a composite built on
    three events is noise.
    """
    if "oni" not in panel.columns:
        return pd.DataFrame()

    # Restrict to the ENSO era before computing anything. The climatology the anomaly is
    # measured against must come from the *same* months that are being classified,
    # otherwise the comparison is partly against a different (pre-1950) climate.
    frame = panel[["level_m", "oni"]].dropna().copy()
    frame["phase"] = classify_enso(frame["oni"])
    frame = frame.dropna(subset=["phase"])
    frame["month"] = frame.index.month
    climatology = frame.groupby("month")["level_m"].transform("mean")
    frame["anomaly_m"] = frame["level_m"] - climatology

    composite = frame.pivot_table(
        index="month", columns="phase", values="anomaly_m", aggfunc="mean"
    )
    counts = frame.pivot_table(
        index="month", columns="phase", values="anomaly_m", aggfunc="count"
    )
    return composite.where(counts >= min_count)


def enso_phase_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Low-water-season (Sep-Dec) stage anomaly by ENSO phase, with sample sizes.

    This is the single table that most directly answers "how much does El Nino matter for
    the December level at Manaus".
    """
    if "oni" not in panel.columns:
        return pd.DataFrame()

    frame = panel[["level_m", "oni"]].dropna().copy()
    frame["phase"] = classify_enso(frame["oni"])
    frame = frame.dropna(subset=["phase"])
    climatology = frame.groupby(frame.index.month)["level_m"].transform("mean")
    frame["anomaly_m"] = frame["level_m"] - climatology

    low_water = frame[frame.index.month.isin((9, 10, 11, 12))]
    summary = low_water.groupby("phase")["anomaly_m"].agg(
        n="count", mean_anomaly_m="mean", std_m="std", min_m="min", max_m="max"
    )
    return summary.sort_values("mean_anomaly_m")


def _override_future_oni(
    panel: pd.DataFrame, horizon: int, target_oni: float | None
) -> pd.DataFrame:
    """Extend the panel with ``horizon`` future months carrying an assumed ONI path.

    The exogenous features used by the models are ONI lagged by at least ``horizon``
    months, so the values that actually drive a forecast of ``t+1..t+h`` are the ONI of
    ``t+1-h..t`` - already observed. Overriding the *future* ONI therefore only bites for
    the lags shorter than the full horizon, plus any recursive use. To make the scenario
    bite as intended we also overwrite the most recent ``horizon`` observed months, which
    is the honest interpretation of "assume ENSO is in this state going into the season".
    """
    extended = panel.reindex(
        panel.index.union(
            pd.date_range(
                panel.index.max() + pd.offsets.MonthBegin(1), periods=horizon, freq="MS"
            )
        )
    )
    if "oni" not in extended.columns:
        return extended

    if target_oni is None:
        latest = extended["oni"].dropna()
        fill_value = float(latest.iloc[-1]) if len(latest) else 0.0
    else:
        fill_value = float(target_oni)

    window = extended.index[-(2 * horizon):]
    extended.loc[window, "oni"] = fill_value
    return extended


def scenario_forecast(
    panel: pd.DataFrame,
    model_factory: callable,
    horizon: int,
    config: ModelConfig = MODEL_CONFIG,
    scenarios: dict[str, float | None] | None = None,
) -> pd.DataFrame:
    """Forecast the same horizon under several assumed ENSO states.

    ``model_factory`` must return a fresh unfitted forecaster; each scenario is fitted on
    its own modified history so the assumed ONI genuinely propagates into the fit.
    """
    scenarios = scenarios or SCENARIO_ONI_OFFSETS
    rows: list[pd.DataFrame] = []

    for label, target_oni in scenarios.items():
        modified = _override_future_oni(panel, horizon, target_oni)
        # The models expect the target column to end at the last observation.
        trimmed = modified.loc[: panel["level_m"].last_valid_index()].copy()
        trimmed["oni"] = modified["oni"].reindex(trimmed.index)
        try:
            model: BaseForecaster = model_factory()
            model.fit(trimmed, horizon)
            result: ForecastResult = model.predict(trimmed, horizon, config.alpha)
        except Exception as exc:  # noqa: BLE001
            log.error("Scenario %r failed: %s", label, exc)
            continue
        frame = result.to_frame()
        frame.insert(0, "scenario", label)
        frame.insert(1, "assumed_oni", np.nan if target_oni is None else target_oni)
        rows.append(frame)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows).rename_axis("date")


def climate_lag_diagnostics(
    panel: pd.DataFrame,
    target_months: tuple[int, ...] = (10, 11, 12),
    indices: tuple[str, ...] = ("oni", "tna", "tsa", "soi"),
    max_lag: int = 12,
) -> pd.DataFrame:
    """Correlation between each climate index at lag L and the low-water stage anomaly.

    This is the empirical justification for the lag structure in :mod:`manaus_river.features`,
    and it is reported rather than assumed because the answer is not the obvious one.

    Two results this consistently surfaces for Manaus:

    * ONI's association with the Oct-Dec stage anomaly is **much stronger at 6-8 months lag
      than contemporaneously** - which is exactly the catchment-storage and routing delay the
      physical chain predicts, and why a "current ENSO phase" composite understates ENSO.
    * For **December specifically the tropical North Atlantic (TNA) outranks the Pacific.**
      That is consistent with the 2005 and 2010 Amazon droughts, both Atlantic-driven with
      ENSO near neutral, and it is the reason this project does not treat ENSO as the
      single driver.
    """
    frame = panel[["level_m"]].dropna().copy()
    climatology = frame.groupby(frame.index.month)["level_m"].transform("mean")
    frame["anomaly_m"] = frame["level_m"] - climatology
    subset = frame[frame.index.month.isin(target_months)]

    rows: list[dict] = []
    for name in indices:
        if name not in panel.columns:
            continue
        for lag in range(max_lag + 1):
            lagged = panel[name].shift(lag).reindex(subset.index)
            paired = pd.concat([subset["anomaly_m"], lagged], axis=1).dropna()
            if len(paired) < 20:
                continue
            rows.append(
                {
                    "index": name,
                    "lag_months": lag,
                    "n": len(paired),
                    "correlation": paired.iloc[:, 0].corr(paired.iloc[:, 1]),
                }
            )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).pivot(
        index="lag_months", columns="index", values="correlation"
    )


def enso_intensity_summary(
    panel: pd.DataFrame, target_month: int = 12, oni_window: int = 6
) -> pd.DataFrame:
    """Target-month stage anomaly banded by ENSO *intensity* going into the season.

    The CPC +/-0.5 degC phase label lumps a marginal event together with a record one. Since
    the stage response is closer to threshold-like than linear, banding by the strength of
    the ONI running mean ending three months before the target is more informative than the
    phase label alone - and it is the form that actually answers "how much does a strong El
    Nino cost us in December".
    """
    if "oni" not in panel.columns:
        return pd.DataFrame()

    frame = panel[["level_m"]].dropna().copy()
    climatology = frame.groupby(frame.index.month)["level_m"].transform("mean")
    frame["anomaly_m"] = frame["level_m"] - climatology

    running = panel["oni"].rolling(oni_window, min_periods=oni_window - 2).mean()
    targets = frame[frame.index.month == target_month].copy()
    # ONI state three months before the target month: the freshest value a 3-month-lead
    # forecast could legitimately know.
    lead_stamps = targets.index - pd.DateOffset(months=3)
    targets["oni_lead"] = running.reindex(lead_stamps).to_numpy()
    targets = targets.dropna(subset=["oni_lead"])
    if targets.empty:
        return pd.DataFrame()

    def _band(value: float) -> str:
        if value >= 1.0:
            return "Strong El Nino (ONI >= 1.0)"
        if value >= 0.5:
            return "Weak/moderate El Nino"
        if value <= -1.0:
            return "Strong La Nina (ONI <= -1.0)"
        if value <= -0.5:
            return "Weak/moderate La Nina"
        return "Neutral"

    targets["band"] = targets["oni_lead"].map(_band)
    summary = targets.groupby("band")["anomaly_m"].agg(
        n="count", mean_anomaly_m="mean", std_m="std", min_m="min", max_m="max"
    )
    return summary.sort_values("mean_anomaly_m")
