"""Climate-index loaders (ENSO and tropical-Atlantic SST modes).

Why these indices matter for the Rio Negro
------------------------------------------
The physical chain that lets NOAA ENSO monitoring say something about Manaus is:

  Pacific SST anomaly  ->  shifted Walker circulation  ->  anomalous subsidence over
  the Amazon  ->  rainfall deficit/surplus in the catchment  ->  (1-3 month lag for
  soil storage + channel routing)  ->  stage anomaly at the Porto de Manaus gauge.

That is why NOAA/CPC can flag an El Nino months in advance: the ocean has far more
thermal inertia than the atmosphere, so the Nino 3.4 SST anomaly is both persistent and
predictable at 3-6 month lead, and the atmospheric teleconnection that follows it is
statistically reliable.

Crucially for the Amazon, the Pacific is NOT the only driver. The 2005 and 2010
droughts were driven mainly by a warm Tropical North Atlantic (TNA), which pulls the
ITCZ northward and starves the central basin of rainfall. We therefore load
TNA/TSA/AMO alongside ONI and SOI and let the models decide.

All series come from NOAA PSL as fixed-width ASCII: a header line with the first and last
year, then one row per year with 12 monthly values, then a trailing missing-value
sentinel and a provenance footer.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
import requests

from ..config import CLIMATE_INDEX_URLS, EXTERNAL_DIR

log = logging.getLogger(__name__)

_REQUEST_TIMEOUT = 60
_MAX_RETRIES = 3


def _fetch_text(url: str) -> str:
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.text
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            log.warning("Fetch failed (%s, attempt %d): %s", url, attempt, exc)
            if attempt < _MAX_RETRIES:
                time.sleep(3.0 * attempt)
    raise RuntimeError(f"Could not download {url}") from last_exc


def parse_psl_ascii(text: str, name: str) -> pd.Series:
    """Parse the NOAA PSL fixed-width monthly-index format into a monthly Series."""
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"Empty payload for index {name}")

    header = lines[0].split()
    try:
        first_year, last_year = int(header[0]), int(header[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Unexpected PSL header for {name}: {lines[0]!r}") from exc

    # The footer line right after the data block holds the missing-value sentinel.
    missing_values = {-99.99, -99.9, -999.0, 999.9, -9999.0}

    records: dict[pd.Timestamp, float] = {}
    for line in lines[1:]:
        parts = line.split()
        if len(parts) != 13:
            # Either the sentinel line or the provenance footer -> data block is over.
            if len(parts) == 1:
                try:
                    missing_values.add(float(parts[0]))
                except ValueError:
                    pass
            continue
        try:
            year = int(parts[0])
            values = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        if not (first_year <= year <= last_year):
            continue
        for month, value in enumerate(values, start=1):
            records[pd.Timestamp(year=year, month=month, day=1)] = value

    series = pd.Series(records, name=name).sort_index()
    series = series.replace(sorted(missing_values), np.nan)
    # Guard against any sentinel we failed to detect.
    series[series.abs() > 50] = np.nan
    return series.dropna()


def load_climate_indices(cache: bool = True) -> pd.DataFrame:
    """Return a monthly DataFrame of all configured climate indices."""
    cache_path = EXTERNAL_DIR / "climate_indices_monthly.csv"
    if cache and cache_path.exists():
        log.info("Using cached climate indices (%s)", cache_path.name)
        return pd.read_csv(cache_path, parse_dates=["date"]).set_index("date").sort_index()

    series_list: list[pd.Series] = []
    for name, url in CLIMATE_INDEX_URLS.items():
        log.info("Downloading climate index %s", name.upper())
        try:
            series = parse_psl_ascii(_fetch_text(url), name)
            log.info("  -> %d months, %s..%s", len(series), series.index.min().date(),
                     series.index.max().date())
            series_list.append(series)
        except Exception as exc:  # noqa: BLE001 - a single missing index is not fatal
            log.error("Skipping index %s: %s", name, exc)

    if not series_list:
        raise RuntimeError("No climate indices could be downloaded")

    frame = pd.concat(series_list, axis=1).sort_index()
    frame.index.name = "date"
    frame = frame.asfreq("MS")

    if cache:
        frame.to_csv(cache_path, index_label="date")
    return frame


def classify_enso(oni: pd.Series) -> pd.Series:
    """Label each month with the operational CPC ENSO phase.

    CPC declares an event when the 3-month running mean of the Nino 3.4 anomaly exceeds
    +/-0.5 degC for five consecutive overlapping seasons. We apply the threshold on the
    3-month running mean, which is the ONI definition itself.

    Months with no ONI observation are left as NaN, **not** labelled "Neutral". The
    distinction matters: the ONI record starts in 1950 while the Manaus gauge starts in
    1902, so defaulting to "Neutral" would silently file 48 years of unclassified history
    into the neutral composite and bias every phase comparison against it.
    """
    running = oni.rolling(3, center=True, min_periods=2).mean()
    phase = pd.Series(pd.NA, index=oni.index, dtype=object)
    phase[running.notna()] = "Neutral"
    phase[running >= 0.5] = "El Nino"
    phase[running <= -0.5] = "La Nina"
    return phase
