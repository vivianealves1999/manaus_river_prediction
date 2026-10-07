"""Figure generation.

Charting decisions worth stating once
------------------------------------
* **One y-axis per panel, always.** Stage in metres and model error in metres are
  different measures; they get separate panels rather than a twin axis.
* **Colour carries model identity, never rank.** Each model keeps its slot whatever its
  skill turns out to be, so the reader can compare figures across runs.
* **Direct labels on the December values.** Three of the categorical hues fall below 3:1
  against the light surface, so identity is never left to colour alone - every forecast
  line is labelled at its endpoint and every figure is backed by a CSV/Markdown table.
* Marks are thin (2px lines), the grid is recessive, and the prediction interval is a
  single translucent band belonging to the headline model rather than one band per model,
  which would be unreadable.
"""
from __future__ import annotations

import logging

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # headless: we only ever write files
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

from .config import FIGURE_DIR

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Design tokens (validated categorical palette; see docs/ARCHITECTURE.md)
# --------------------------------------------------------------------------------------
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#87867f"
GRID = "#e6e5e1"

#: Fixed slot order. Assigned by entity, never cycled, never re-assigned by rank.
SERIES_COLORS: dict[str, str] = {
    "Observed": TEXT_PRIMARY,
    "Ensemble": "#2a78d6",
    "SARIMAX+ENSO": "#eb6834",
    "SARIMAX": "#1baf7a",
    "Gradient Boosting": "#eda100",
    "Holt-Winters (ETS)": "#e87ba4",
    "Climatology": "#008300",
    "Seasonal Naive + drift": "#4a3aa7",
}
FALLBACK_COLOR = "#87867f"

ENSO_COLORS = {"El Nino": "#e34948", "Neutral": "#87867f", "La Nina": "#2a78d6"}

_LINE_WIDTH = 2.0
_HEADLINE_WIDTH = 2.8


def _style_axes(ax, ylabel: str | None = None, xlabel: str | None = None) -> None:
    """Recessive grid and axes; no chartjunk."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=10)


def _color_for(model: str) -> str:
    return SERIES_COLORS.get(model, FALLBACK_COLOR)


def plot_forecast_comparison(
    history: pd.Series,
    forecasts: dict[str, pd.DataFrame],
    headline_model: str = "Ensemble",
    history_months: int = 42,
    target_label: str = "December",
    filename: str = "forecast_comparison.png",
    metrics: pd.DataFrame | None = None,
) -> str:
    """The headline figure: every model's path to the target month, plus backtest skill.

    ``forecasts`` maps model name -> frame indexed by date with ``forecast_m`` and
    optionally ``lower_m`` / ``upper_m``.
    """
    fig = plt.figure(figsize=(14, 9), facecolor=SURFACE)
    grid = fig.add_gridspec(2, 2, height_ratios=[2.1, 1.0], hspace=0.46, wspace=0.22)
    ax_main = fig.add_subplot(grid[0, :])
    ax_skill = fig.add_subplot(grid[1, 0])
    ax_zoom = fig.add_subplot(grid[1, 1])

    recent = history.dropna().iloc[-history_months:]

    # ---- main panel -----------------------------------------------------------------
    ax_main.plot(
        recent.index, recent.values, color=SERIES_COLORS["Observed"],
        linewidth=_HEADLINE_WIDTH, label="Observed", zorder=6,
    )

    headline = forecasts.get(headline_model)
    if headline is not None and {"lower_m", "upper_m"} <= set(headline.columns):
        ax_main.fill_between(
            headline.index, headline["lower_m"], headline["upper_m"],
            color=SERIES_COLORS.get(headline_model, FALLBACK_COLOR), alpha=0.16,
            linewidth=0, zorder=2, label=f"{headline_model} 80% interval",
        )

    # Join the last observation to each forecast so the lines do not float.
    last_date, last_value = recent.index[-1], float(recent.iloc[-1])

    for model, frame in forecasts.items():
        if frame.empty:
            continue
        color = _color_for(model)
        is_headline = model == headline_model
        dates = [last_date] + list(frame.index)
        values = [last_value] + list(frame["forecast_m"])
        ax_main.plot(
            dates, values, color=color,
            linewidth=_HEADLINE_WIDTH if is_headline else _LINE_WIDTH,
            linestyle="-" if is_headline else "--",
            marker="o", markersize=5.5 if is_headline else 4.5,
            markeredgecolor=SURFACE, markeredgewidth=1.2,
            label=model, zorder=5 if is_headline else 4,
        )

    # Direct endpoint labels: identity must never rest on colour alone.
    endpoints = sorted(
        (
            (float(frame["forecast_m"].iloc[-1]), model, frame.index[-1])
            for model, frame in forecasts.items() if not frame.empty
        ),
        reverse=True,
    )
    if endpoints:
        span = max(e[0] for e in endpoints) - min(e[0] for e in endpoints)
        offset_step = max(span * 0.22, 0.30)
        for rank, (value, model, date) in enumerate(endpoints):
            ax_main.annotate(
                f"{model}  {value:.2f} m",
                xy=(date, value),
                xytext=(14, (len(endpoints) / 2 - rank) * offset_step * 14),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold" if model == headline_model else "normal",
                color=TEXT_PRIMARY if model == headline_model else TEXT_SECONDARY,
                va="center",
                arrowprops=dict(
                    arrowstyle="-", color=_color_for(model), linewidth=1.0, alpha=0.7
                ),
            )

    if forecasts:
        first_forecast = min(f.index[0] for f in forecasts.values() if not f.empty)
        ax_main.axvline(
            last_date, color=TEXT_MUTED, linewidth=1.0, linestyle=":", zorder=1
        )
        ax_main.annotate(
            "forecast origin",
            xy=(last_date, ax_main.get_ylim()[0]),
            xytext=(4, 6), textcoords="offset points",
            fontsize=8, color=TEXT_MUTED,
        )
        del first_forecast

    ax_main.set_title(
        f"Rio Negro at Manaus - monthly mean stage, forecast to {target_label}",
        color=TEXT_PRIMARY, fontsize=14, fontweight="bold", loc="left", pad=12,
    )
    _style_axes(ax_main, ylabel="Stage (m above gauge zero)")
    ax_main.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax_main.yaxis.set_major_locator(MultipleLocator(2))
    ax_main.margins(x=0.13)
    # Below the axes rather than inside them: the observed series fills the lower-left of
    # this panel every year at low water, and an inset legend sat on top of it.
    legend = ax_main.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.09), frameon=False,
        fontsize=9, ncols=4, labelcolor=TEXT_SECONDARY,
    )
    legend.set_zorder(10)

    # ---- skill panel -----------------------------------------------------------------
    if metrics is not None and not metrics.empty:
        table = metrics.reset_index()
        models = [m for m in SERIES_COLORS if m in set(table["model"])]
        leads = sorted(table["lead"].unique())
        width = 0.8 / max(len(models), 1)
        for slot, model in enumerate(models):
            subset = table[table["model"] == model].set_index("lead")["RMSE"]
            positions = [lead + slot * width - 0.4 + width / 2 for lead in leads]
            ax_skill.bar(
                positions, [subset.get(lead, np.nan) for lead in leads],
                width=width * 0.88, color=_color_for(model), label=model,
                edgecolor=SURFACE, linewidth=1.4, zorder=3,
            )
        ax_skill.set_xticks(leads)
        ax_skill.set_xticklabels([f"{lead} mo" for lead in leads])
        ax_skill.set_title(
            "Backtest error by lead time (lower is better)",
            color=TEXT_PRIMARY, fontsize=11, fontweight="bold", loc="left", pad=8,
        )
        _style_axes(ax_skill, ylabel="RMSE (m)", xlabel="Forecast lead")
    else:
        ax_skill.axis("off")

    # ---- zoom panel -----------------------------------------------------------------
    zoom_start = last_date - pd.DateOffset(months=9)
    zoom_hist = history.dropna().loc[zoom_start:]
    ax_zoom.plot(
        zoom_hist.index, zoom_hist.values, color=SERIES_COLORS["Observed"],
        linewidth=_HEADLINE_WIDTH, zorder=5,
    )
    if headline is not None:
        if {"lower_m", "upper_m"} <= set(headline.columns):
            ax_zoom.fill_between(
                headline.index, headline["lower_m"], headline["upper_m"],
                color=SERIES_COLORS.get(headline_model, FALLBACK_COLOR),
                alpha=0.18, linewidth=0, zorder=2,
            )
        ax_zoom.plot(
            [last_date] + list(headline.index),
            [last_value] + list(headline["forecast_m"]),
            color=SERIES_COLORS.get(headline_model, FALLBACK_COLOR),
            linewidth=_HEADLINE_WIDTH, marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=6,
        )
        # Stagger the labels: consecutive forecast points sit close enough to collide.
        for position, (date, value) in enumerate(headline["forecast_m"].items()):
            ax_zoom.annotate(
                f"{value:.2f}", xy=(date, value),
                xytext=(0, 13 if position % 2 == 0 else -18),
                textcoords="offset points", ha="center", fontsize=9,
                fontweight="bold", color=TEXT_PRIMARY,
            )
    ax_zoom.set_title(
        f"{headline_model} forecast, close up",
        color=TEXT_PRIMARY, fontsize=11, fontweight="bold", loc="left", pad=8,
    )
    _style_axes(ax_zoom, ylabel="Stage (m)")
    ax_zoom.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax_zoom.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_zoom.margins(y=0.22)

    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    log.info("Wrote %s", path)
    return str(path)


def plot_model_skill(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    filename: str = "model_skill.png",
) -> str:
    """Diagnostics: error by lead, and predicted-vs-observed for the best model."""
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(13, 5.2), facecolor=SURFACE, gridspec_kw={"wspace": 0.24}
    )

    table = metrics.reset_index()
    models = [m for m in SERIES_COLORS if m in set(table["model"])]
    for model in models:
        subset = table[table["model"] == model].sort_values("lead")
        ax_left.plot(
            subset["lead"], subset["RMSE"], color=_color_for(model),
            linewidth=_LINE_WIDTH, marker="o", markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=1.3, label=model, zorder=4,
        )
        if not subset.empty:
            last = subset.iloc[-1]
            ax_left.annotate(
                f" {model}", xy=(last["lead"], last["RMSE"]),
                xytext=(6, 0), textcoords="offset points",
                fontsize=8.5, color=TEXT_SECONDARY, va="center",
            )
    ax_left.set_title(
        "Rolling-origin RMSE by forecast lead",
        color=TEXT_PRIMARY, fontsize=12, fontweight="bold", loc="left", pad=10,
    )
    _style_axes(ax_left, ylabel="RMSE (m)", xlabel="Lead time (months)")
    ax_left.set_xticks(sorted(table["lead"].unique()))
    ax_left.margins(x=0.30)

    # Predicted vs observed for the best model, all leads pooled.
    best = table.groupby("model")["RMSE"].mean().idxmin()
    subset = predictions[predictions["model"] == best]
    limits = [
        min(subset["y_true"].min(), subset["y_pred"].min()) - 0.5,
        max(subset["y_true"].max(), subset["y_pred"].max()) + 0.5,
    ]
    ax_right.plot(limits, limits, color=TEXT_MUTED, linewidth=1.2, linestyle="--", zorder=2)
    for lead, marker in zip(sorted(subset["lead"].unique()), ("o", "s", "^", "D", "v")):
        group = subset[subset["lead"] == lead]
        ax_right.scatter(
            group["y_true"], group["y_pred"], s=42, marker=marker,
            color=_color_for(best), alpha=0.55,
            edgecolor=SURFACE, linewidth=1.0, label=f"lead {lead} mo", zorder=4,
        )
    ax_right.set_title(
        f"Predicted vs observed - {best}",
        color=TEXT_PRIMARY, fontsize=12, fontweight="bold", loc="left", pad=10,
    )
    _style_axes(ax_right, ylabel="Predicted stage (m)", xlabel="Observed stage (m)")
    ax_right.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, loc="upper left")
    ax_right.set_xlim(limits)
    ax_right.set_ylim(limits)

    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    log.info("Wrote %s", path)
    return str(path)


def plot_enso_influence(
    composite: pd.DataFrame,
    panel: pd.DataFrame,
    filename: str = "enso_influence.png",
) -> str:
    """Observed evidence that ENSO phase shifts the Manaus stage, by calendar month."""
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(13, 5.2), facecolor=SURFACE, gridspec_kw={"wspace": 0.22}
    )

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    phases = [p for p in ("El Nino", "Neutral", "La Nina") if p in composite.columns]
    width = 0.8 / max(len(phases), 1)
    for slot, phase in enumerate(phases):
        values = composite[phase].reindex(range(1, 13))
        positions = np.arange(1, 13) + slot * width - 0.4 + width / 2
        ax_left.bar(
            positions, values.values, width=width * 0.88,
            color=ENSO_COLORS.get(phase, FALLBACK_COLOR), label=phase,
            edgecolor=SURFACE, linewidth=1.4, zorder=3,
        )
    ax_left.axhline(0, color=TEXT_SECONDARY, linewidth=1.1, zorder=4)
    ax_left.set_xticks(np.arange(1, 13))
    ax_left.set_xticklabels(month_labels)
    ax_left.set_title(
        "Stage anomaly at Manaus by ENSO phase",
        color=TEXT_PRIMARY, fontsize=12, fontweight="bold", loc="left", pad=10,
    )
    _style_axes(ax_left, ylabel="Anomaly vs monthly mean (m)", xlabel="Calendar month")
    ax_left.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY)

    # Right: ONI against the following low-water-season stage anomaly.
    if "oni" in panel.columns:
        frame = panel[["level_m", "oni"]].dropna().copy()
        # Same restriction as scenarios.enso_composite: anomalies are measured against the
        # ENSO-era climatology, not the full 1902- record.
        climatology = frame.groupby(frame.index.month)["level_m"].transform("mean")
        frame["anomaly_m"] = frame["level_m"] - climatology
        # ONI averaged over the preceding boreal winter vs the Sep-Dec trough anomaly.
        low_water = frame[frame.index.month.isin((10, 11, 12))].copy()
        low_water["oni_lead"] = frame["oni"].rolling(6, min_periods=3).mean().shift(6).reindex(
            low_water.index
        )
        low_water = low_water.dropna(subset=["oni_lead"])
        ax_right.scatter(
            low_water["oni_lead"], low_water["anomaly_m"], s=46,
            color=SERIES_COLORS["Ensemble"], alpha=0.6,
            edgecolor=SURFACE, linewidth=1.0, zorder=4,
        )
        if len(low_water) > 5:
            slope, intercept = np.polyfit(low_water["oni_lead"], low_water["anomaly_m"], 1)
            xs = np.linspace(low_water["oni_lead"].min(), low_water["oni_lead"].max(), 50)
            correlation = low_water["oni_lead"].corr(low_water["anomaly_m"])
            ax_right.plot(
                xs, slope * xs + intercept, color=TEXT_PRIMARY,
                linewidth=_LINE_WIDTH, zorder=5,
                label=f"slope {slope:.2f} m per degC  (r = {correlation:.2f})",
            )
            ax_right.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY)
        ax_right.axhline(0, color=TEXT_MUTED, linewidth=1.0, linestyle=":", zorder=2)
        ax_right.axvline(0, color=TEXT_MUTED, linewidth=1.0, linestyle=":", zorder=2)
        ax_right.set_title(
            "Preceding 6-month ONI vs Oct-Dec stage anomaly",
            color=TEXT_PRIMARY, fontsize=12, fontweight="bold", loc="left", pad=10,
        )
        _style_axes(
            ax_right, ylabel="Oct-Dec anomaly (m)", xlabel="Mean ONI, 6 months earlier (degC)"
        )
    else:
        ax_right.axis("off")

    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    log.info("Wrote %s", path)
    return str(path)


def plot_history_overview(
    daily: pd.Series,
    panel: pd.DataFrame,
    filename: str = "history_overview.png",
) -> str:
    """The full instrumental record, with the seasonal envelope that motivates the models."""
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(13, 7.5), facecolor=SURFACE,
        gridspec_kw={"hspace": 0.34, "height_ratios": [1.3, 1.0]},
    )

    annual_max = daily.resample("YS").max()
    annual_min = daily.resample("YS").min()
    ax_top.fill_between(
        annual_max.index, annual_min.values, annual_max.values,
        color=SERIES_COLORS["Ensemble"], alpha=0.18, linewidth=0, zorder=2,
        label="Annual range (min-max)",
    )
    ax_top.plot(
        annual_max.index, annual_max.values, color=SERIES_COLORS["Ensemble"],
        linewidth=1.6, zorder=4, label="Annual maximum",
    )
    ax_top.plot(
        annual_min.index, annual_min.values, color=SERIES_COLORS["SARIMAX+ENSO"],
        linewidth=1.6, zorder=4, label="Annual minimum",
    )
    record_low_date, record_low = daily.idxmin(), daily.min()
    ax_top.annotate(
        f"record low {record_low:.2f} m\n{record_low_date:%b %Y}",
        xy=(record_low_date, record_low), xytext=(-90, 26),
        textcoords="offset points", fontsize=9, color=TEXT_PRIMARY,
        arrowprops=dict(arrowstyle="-", color=TEXT_MUTED, linewidth=1.0),
    )
    ax_top.set_title(
        f"Porto de Manaus gauge, {daily.index.min():%Y}-{daily.index.max():%Y}"
        " - annual high and low water",
        color=TEXT_PRIMARY, fontsize=13, fontweight="bold", loc="left", pad=10,
    )
    _style_axes(ax_top, ylabel="Stage (m)")
    ax_top.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, ncols=3)

    monthly = panel["level_m"].dropna()
    by_month = monthly.groupby(monthly.index.month)
    months = np.arange(1, 13)
    median = by_month.median().reindex(months)
    ax_bottom.fill_between(
        months, by_month.min().reindex(months), by_month.max().reindex(months),
        color=TEXT_MUTED, alpha=0.14, linewidth=0, zorder=2, label="Full observed range",
    )
    ax_bottom.fill_between(
        months, by_month.quantile(0.25).reindex(months),
        by_month.quantile(0.75).reindex(months),
        color=SERIES_COLORS["Ensemble"], alpha=0.25, linewidth=0, zorder=3,
        label="Interquartile range",
    )
    ax_bottom.plot(
        months, median.values, color=SERIES_COLORS["Ensemble"],
        linewidth=_HEADLINE_WIDTH, marker="o", markersize=6,
        markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=5, label="Median",
    )
    ax_bottom.set_xticks(months)
    ax_bottom.set_xticklabels(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    )
    ax_bottom.set_title(
        "The flood pulse: monthly mean stage distribution",
        color=TEXT_PRIMARY, fontsize=12, fontweight="bold", loc="left", pad=10,
    )
    _style_axes(ax_bottom, ylabel="Stage (m)", xlabel="Calendar month")
    ax_bottom.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, ncols=3)

    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    log.info("Wrote %s", path)
    return str(path)


def plot_scenarios(
    history: pd.Series,
    scenario_frame: pd.DataFrame,
    filename: str = "enso_scenarios.png",
    history_months: int = 24,
) -> str:
    """Forecast paths under each assumed ENSO state."""
    if scenario_frame.empty:
        return ""

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=SURFACE)
    recent = history.dropna().iloc[-history_months:]
    ax.plot(
        recent.index, recent.values, color=SERIES_COLORS["Observed"],
        linewidth=_HEADLINE_WIDTH, label="Observed", zorder=6,
    )

    last_date, last_value = recent.index[-1], float(recent.iloc[-1])
    palette = ["#2a78d6", "#e34948", "#eb6834", "#87867f", "#1baf7a", "#4a3aa7"]
    for slot, (label, group) in enumerate(scenario_frame.groupby("scenario", sort=False)):
        color = palette[slot % len(palette)]
        ax.plot(
            [last_date] + list(group.index), [last_value] + list(group["forecast_m"]),
            color=color, linewidth=_LINE_WIDTH, marker="o", markersize=5.5,
            markeredgecolor=SURFACE, markeredgewidth=1.2, label=label, zorder=5,
        )
        end_value = float(group["forecast_m"].iloc[-1])
        ax.annotate(
            f" {end_value:.2f} m", xy=(group.index[-1], end_value),
            xytext=(8, 0), textcoords="offset points", fontsize=9,
            color=TEXT_SECONDARY, va="center",
        )

    ax.set_title(
        "Manaus stage forecast under assumed ENSO states",
        color=TEXT_PRIMARY, fontsize=13, fontweight="bold", loc="left", pad=12,
    )
    _style_axes(ax, ylabel="Stage (m above gauge zero)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, ncols=2)
    ax.margins(x=0.16)

    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    log.info("Wrote %s", path)
    return str(path)
