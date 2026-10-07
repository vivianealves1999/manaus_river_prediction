"""Assemble the analysis-ready monthly panel from all raw sources."""
from __future__ import annotations

import logging

import pandas as pd

from ..config import PROCESSED_DIR, TARGET_STATION, UPSTREAM_STATIONS
from .ana_client import download_station, to_monthly
from .climate import classify_enso, load_climate_indices
from .porto_manaus import fetch_porto_manaus

log = logging.getLogger(__name__)

PANEL_PATH = PROCESSED_DIR / "monthly_panel.csv"
DAILY_TARGET_PATH = PROCESSED_DIR / "manaus_daily.csv"


def build_target_daily(cache: bool = True) -> pd.DataFrame:
    """Daily Manaus stage: ANA history spliced with the Porto de Manaus tail."""
    ana = download_station(TARGET_STATION, cache=cache)[["level_m"]]
    log.info(
        "ANA Manaus: %d days %s..%s",
        len(ana), ana.index.min().date(), ana.index.max().date(),
    )

    porto = fetch_porto_manaus(cache=cache)
    if not porto.empty:
        # ANA always wins on overlapping dates; the scraper only extends the tail.
        new_dates = porto.index.difference(ana.index)
        if len(new_dates):
            log.info("Appending %d days from Porto de Manaus", len(new_dates))
            ana = pd.concat([ana, porto.loc[new_dates, ["level_m"]]]).sort_index()

    ana = ana[~ana.index.duplicated(keep="first")].sort_index()
    ana.to_csv(DAILY_TARGET_PATH, index_label="date")
    return ana


def build_panel(cache: bool = True) -> pd.DataFrame:
    """Build and persist the monthly modelling panel.

    Columns
    -------
    level_m                 target: monthly mean stage at Manaus
    level_min_m/level_max_m monthly extremes (context, not predictors)
    <station>_level_m       monthly mean stage at each upstream gauge
    oni/tna/tsa/soi/amo     monthly climate indices
    enso_phase              categorical CPC phase label
    """
    daily = build_target_daily(cache=cache)
    panel = to_monthly(daily)
    log.info("Target panel: %d months %s..%s", len(panel),
             panel.index.min().date(), panel.index.max().date())

    for station in UPSTREAM_STATIONS:
        try:
            station_daily = download_station(station, cache=cache)
            station_monthly = to_monthly(station_daily)
            column = f"{station.name.lower().replace(' ', '_')}_level_m"
            panel[column] = station_monthly["level_m"]
            coverage = panel[column].notna().mean()
            log.info("Upstream %s -> %s (%.0f%% coverage)", station.name, column, 100 * coverage)
        except Exception as exc:  # noqa: BLE001 - upstream gauges are optional predictors
            log.error("Could not add upstream station %s: %s", station.name, exc)

    try:
        climate = load_climate_indices(cache=cache)
        panel = panel.join(climate, how="left")
        if "oni" in panel.columns:
            panel["enso_phase"] = classify_enso(panel["oni"])
    except Exception as exc:  # noqa: BLE001
        log.error("Climate indices unavailable: %s", exc)

    panel = panel.sort_index()
    panel.index.name = "date"
    panel.to_csv(PANEL_PATH, index_label="date")
    log.info("Panel written to %s (%d rows, %d columns)", PANEL_PATH, *panel.shape)
    return panel


def load_panel() -> pd.DataFrame:
    """Load the persisted panel, building it first if necessary."""
    if not PANEL_PATH.exists():
        return build_panel()
    return pd.read_csv(PANEL_PATH, parse_dates=["date"]).set_index("date").sort_index()
