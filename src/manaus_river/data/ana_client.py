"""Client for ANA / HidroWeb historical stage ("cota") series.

The legacy SOAP-over-GET endpoint returns one XML record per (station, month,
consistency-level).  Each record carries up to 31 daily readings in ``Cota01``..``Cota31``,
expressed in **centimetres** above the local gauge zero.

Two practical quirks drive the design of this module:

1. The service silently returns an *empty* result set when the requested date range is
   too long, so requests are chunked into :data:`~manaus_river.config.ANA_CHUNK_YEARS`
   windows.
2. Every month can appear twice: ``NivelConsistencia`` 1 = raw ("bruto") and 2 =
   quality-checked ("consistido").  We always prefer 2 and fall back to 1, which is what
   gives us a long *consistent* history plus up-to-date recent months.
"""
from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from ..config import ANA_CHUNK_YEARS, ANA_SERVICE_URL, ANA_START_YEAR, RAW_DIR, Station

log = logging.getLogger(__name__)

_NS_RE = re.compile(r"\{.*?\}")
_REQUEST_TIMEOUT = 120
_MAX_RETRIES = 3
_RETRY_SLEEP = 5.0


def _strip_ns(tag: str) -> str:
    return _NS_RE.sub("", tag)


def _fetch_chunk(station_code: str, start: date, end: date) -> str:
    """Fetch one date window of raw XML, with retries."""
    params = {
        "codEstacao": station_code,
        "dataInicio": start.strftime("%d/%m/%Y"),
        "dataFim": end.strftime("%d/%m/%Y"),
        "tipoDados": "1",          # 1 = Cotas (stage)
        "nivelConsistencia": "",   # empty = return both raw and consisted
    }
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = requests.get(ANA_SERVICE_URL, params=params, timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.text
        except Exception as exc:  # noqa: BLE001 - network layer, retry on anything
            last_exc = exc
            log.warning(
                "ANA request failed (station=%s %s..%s attempt %d/%d): %s",
                station_code, start, end, attempt, _MAX_RETRIES, exc,
            )
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_SLEEP * attempt)
    raise RuntimeError(f"ANA request permanently failed for {station_code}") from last_exc


def _parse_xml(xml_text: str) -> list[dict]:
    """Explode monthly XML records into one row per calendar day."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        log.error("Unparseable XML from ANA: %s", exc)
        return []

    rows: list[dict] = []
    for record in root.iter():
        if _strip_ns(record.tag) != "SerieHistorica":
            continue
        fields = {_strip_ns(child.tag): (child.text or "").strip() for child in record}

        raw_month = fields.get("DataHora")
        if not raw_month:
            continue
        try:
            month_start = pd.Timestamp(raw_month).normalize().replace(day=1)
        except (ValueError, TypeError):
            continue

        try:
            consistency = int(fields.get("NivelConsistencia") or 1)
        except ValueError:
            consistency = 1

        for day in range(1, 32):
            value = fields.get(f"Cota{day:02d}")
            if value in (None, ""):
                continue
            try:
                cm = float(value)
            except ValueError:
                continue
            # ANA uses sentinel/implausible values for missing data; the Manaus gauge
            # has never read below ~1100 cm nor above ~3100 cm.
            if not (0.0 < cm < 5000.0):
                continue
            try:
                stamp = month_start.replace(day=day)
            except ValueError:
                continue  # e.g. Cota31 in a 30-day month
            rows.append(
                {
                    "date": stamp,
                    "level_cm": cm,
                    "consistency": consistency,
                }
            )
    return rows


def download_station(
    station: Station,
    start_year: int = ANA_START_YEAR,
    end_year: int | None = None,
    cache: bool = True,
) -> pd.DataFrame:
    """Download the full daily stage series for one station.

    Returns a tidy frame indexed by ``date`` with a single ``level_m`` column, having
    resolved the raw/consisted duplication in favour of quality-checked data.
    """
    end_year = end_year or date.today().year
    cache_path: Path = RAW_DIR / f"ana_{station.code}_{station.name.lower()}_daily.csv"

    if cache and cache_path.exists():
        log.info("Using cached series for %s (%s)", station.name, cache_path.name)
        cached = pd.read_csv(cache_path, parse_dates=["date"])
        return cached.set_index("date").sort_index()

    all_rows: list[dict] = []
    for chunk_start_year in range(start_year, end_year + 1, ANA_CHUNK_YEARS):
        chunk_start = date(chunk_start_year, 1, 1)
        chunk_end = date(min(chunk_start_year + ANA_CHUNK_YEARS - 1, end_year), 12, 31)
        log.info("Fetching %s (%s) %s..%s", station.name, station.code, chunk_start, chunk_end)
        xml_text = _fetch_chunk(station.code, chunk_start, chunk_end)
        rows = _parse_xml(xml_text)
        log.info("  -> %d daily observations", len(rows))
        all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError(f"No data returned by ANA for station {station.code}")

    frame = pd.DataFrame(all_rows)

    # Prefer consisted (2) over raw (1) for any duplicated day.
    frame = (
        frame.sort_values(["date", "consistency"])
        .drop_duplicates(subset="date", keep="last")
        .set_index("date")
        .sort_index()
    )
    frame["level_m"] = frame["level_cm"] / 100.0
    frame = frame[["level_m", "consistency"]]

    if cache:
        frame.to_csv(cache_path, index_label="date")
        log.info("Cached %d rows -> %s", len(frame), cache_path.name)
    return frame


def to_monthly(daily: pd.DataFrame, min_days: int = 10) -> pd.DataFrame:
    """Aggregate a daily stage series to monthly statistics.

    Months with fewer than ``min_days`` observations are treated as missing rather than
    being averaged from a handful of days, which would bias the seasonal cycle.
    """
    grouped = daily["level_m"].resample("MS")
    monthly = pd.DataFrame(
        {
            "level_m": grouped.mean(),
            "level_min_m": grouped.min(),
            "level_max_m": grouped.max(),
            "n_days": grouped.count(),
        }
    )
    monthly.loc[monthly["n_days"] < min_days, ["level_m", "level_min_m", "level_max_m"]] = pd.NA
    return monthly
