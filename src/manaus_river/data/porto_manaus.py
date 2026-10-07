"""Porto de Manaus daily stage archive (2000 - present).

Why this source exists in the pipeline
--------------------------------------
The ANA *consisted* (quality-checked) series for the Manaus gauge is the gold standard
and reaches back to 1902, but the public legacy web service stops delivering it at
2014-12. Every other ANA route to recent data (the new HidroWebService REST API, the
SNIRH document endpoint) is behind an API token, and the telemetry endpoint only retains
the last few weeks.

portodemanaus.com.br publishes the official Porto de Manaus ruler reading for every day
since 2000. That is the same physical gauge as ANA station 14990000, so splicing the two
gives an unbroken 1902 - present daily record.

How the archive is reached
--------------------------
The site is WordPress + JetEngine. Each month is a ``nivel-do-rio`` post carrying a daily
table, surfaced through a JetSmartFilters listing. The listing is server-rendered and
accepts filters directly in the URL::

    /nivel-do-rio-negro/?jsf=jet-engine:result&tax=ano:<year_term>;meses:<month_term>

We deliberately request **one month per call** rather than paging six at a time. Paged
responses interleave several tables whose month has to be recovered from the surrounding
HTML headings, and those headings are not reliably rendered (some are split across
elements or carry entity-encoded accents), which silently mis-assigns whole months. With
a single-month filter the date comes from the request we made, not from parsing.

Both taxonomy term id maps are discovered at run time from the public WP REST API rather
than hard-coded, since the ids are not sequential.

Two data-quality quirks are handled explicitly: the archive is not unit-consistent (recent
years publish metres, older years whole centimetres) and it contains occasional
transcription typos. See :func:`_normalise_level` and :func:`_drop_impossible_jumps`.
"""
from __future__ import annotations

import io
import json
import logging
import time
import unicodedata

import pandas as pd
import requests

from ..config import RAW_DIR

log = logging.getLogger(__name__)

BASE_URL = "https://portodemanaus.com.br"
LEVEL_PAGE = f"{BASE_URL}/nivel-do-rio-negro/"
YEAR_TERMS_API = f"{BASE_URL}/wp-json/wp/v2/ano"
MONTH_TERMS_API = f"{BASE_URL}/wp-json/wp/v2/meses"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

_MONTH_NUMBERS = {
    "janeiro": 1, "fevereiro": 2, "marco": 3, "abril": 4, "maio": 5, "junho": 6,
    "julho": 7, "agosto": 8, "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12,
}
_REQUEST_TIMEOUT = 90
_MAX_RETRIES = 3
_POLITE_DELAY = 0.7  # seconds between requests


# Physically plausible envelope for this gauge (record low 12.11 m in Oct 2024, record
# high 30.02 m in Jun 2021). Used only to reject parsing artefacts.
_MIN_PLAUSIBLE_M = 5.0
_MAX_PLAUSIBLE_M = 35.0


def _normalise_level(value: float) -> float | None:
    """Coerce a published reading to metres, or return None if implausible.

    The site is not unit-consistent across its archive: recent years publish metres
    ("24,09") while older years publish whole centimetres ("1641"). Since the gauge can
    never read above 35 m nor below 5 m, the magnitude alone disambiguates the two
    unambiguously.
    """
    if _MIN_PLAUSIBLE_M < value < _MAX_PLAUSIBLE_M:
        return float(value)
    if 100.0 * _MIN_PLAUSIBLE_M < value < 100.0 * _MAX_PLAUSIBLE_M:
        return float(value) / 100.0
    return None


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def _get(url: str, params: dict | None = None) -> str | None:
    """GET with retries. Returns None rather than raising: this source is best-effort."""
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, headers=_HEADERS, timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.text
        except Exception as exc:  # noqa: BLE001
            log.warning("GET failed (%s attempt %d/%d): %s", url, attempt, _MAX_RETRIES, exc)
            if attempt < _MAX_RETRIES:
                time.sleep(2.0 * attempt)
    return None


def _fetch_terms(url: str) -> list[dict]:
    text = _get(url, {"per_page": 100})
    if text is None:
        return []
    try:
        terms = json.loads(text)
    except json.JSONDecodeError:
        return []
    return terms if isinstance(terms, list) else []


def discover_year_terms() -> dict[int, int]:
    """Map calendar year -> JetEngine ``ano`` taxonomy term id."""
    mapping: dict[int, int] = {}
    for term in _fetch_terms(YEAR_TERMS_API):
        try:
            mapping[int(str(term["name"]).strip())] = int(term["id"])
        except (ValueError, KeyError, TypeError):
            continue
    return mapping


def discover_month_terms() -> dict[int, int]:
    """Map month number (1-12) -> JetEngine ``meses`` taxonomy term id.

    Matching is done on the term *slug* rather than its name: names arrive with
    inconsistently encoded accents ("Mar\\xe7o" vs "Mar&ccedil;o"), whereas slugs are
    already ASCII ("marco").
    """
    mapping: dict[int, int] = {}
    for term in _fetch_terms(MONTH_TERMS_API):
        try:
            slug = _strip_accents(str(term["slug"])).lower()
            month = _MONTH_NUMBERS.get(slug)
            if month is None:
                month = _MONTH_NUMBERS.get(_strip_accents(str(term["name"])).lower())
            if month is not None:
                mapping[month] = int(term["id"])
        except (KeyError, TypeError, ValueError):
            continue
    return mapping


def _parse_month_table(html_text: str, year: int, month: int) -> pd.DataFrame:
    """Extract the daily readings for one known (year, month) from a filtered page.

    Because the request already pinned the month, we do not parse any heading: we take the
    first table that has the expected ``Dia`` / ``Cota`` columns.
    """
    # A handful of months render malformed markup that lxml gives up on; pandas then
    # falls back to html5lib. Catch broadly so one bad month cannot abort a 320-month run.
    try:
        tables = pd.read_html(io.StringIO(html_text))
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not parse tables for %04d-%02d: %s", year, month, exc)
        return pd.DataFrame(columns=["date", "level_m"])

    for table in tables:
        table = table.copy()
        table.columns = [_strip_accents(str(c)).strip().lower() for c in table.columns]
        day_col = next((c for c in table.columns if c.startswith("dia")), None)
        level_col = next((c for c in table.columns if "cota" in c), None)
        if day_col is None or level_col is None:
            continue

        days = pd.to_numeric(table[day_col], errors="coerce")
        levels = pd.to_numeric(
            table[level_col].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )
        rows: list[dict] = []
        for day, level in zip(days, levels):
            if pd.isna(day) or pd.isna(level):
                continue
            level_m = _normalise_level(float(level))
            if level_m is None:
                continue
            try:
                stamp = pd.Timestamp(year=year, month=month, day=int(day))
            except ValueError:
                continue  # e.g. day 31 listed in a 30-day month
            rows.append({"date": stamp, "level_m": level_m})
        if rows:
            return pd.DataFrame(rows)

    return pd.DataFrame(columns=["date", "level_m"])


def _drop_impossible_jumps(series: pd.Series, max_daily_change_m: float = 0.6) -> pd.Series:
    """Remove transcription typos from a daily stage series.

    The Rio Negro at Manaus is a very large, heavily damped system: day-to-day changes are
    of the order of a few centimetres and essentially never exceed ~30 cm. A reading that
    sits more than ``max_daily_change_m`` away from *both* its neighbours is therefore a
    digit error, not hydrology (the archive contains a handful, e.g. a value 7 m off on
    2012-12-14). Isolated spikes are dropped; genuine sustained moves keep at least one
    consistent neighbour and survive.
    """
    if len(series) < 3:
        return series
    before = series.diff()
    after = series.diff(-1)
    spike = (before.abs() > max_daily_change_m) & (after.abs() > max_daily_change_m)
    # Only treat it as a typo if the two jumps point in opposite directions (a spike),
    # rather than a real step change in the same direction.
    spike &= (before * after) > 0
    if spike.any():
        log.info("Dropping %d implausible daily reading(s) as transcription typos",
                 int(spike.sum()))
    return series[~spike.fillna(False)]


def fetch_porto_manaus(cache: bool = True) -> pd.DataFrame:
    """Download the full Porto de Manaus daily archive, one month per request.

    Never raises: on any failure the pipeline falls back to ANA data alone.
    """
    cache_path = RAW_DIR / "porto_manaus_daily.csv"
    if cache and cache_path.exists():
        log.info("Using cached Porto de Manaus archive (%s)", cache_path.name)
        return pd.read_csv(cache_path, parse_dates=["date"]).set_index("date").sort_index()

    empty = pd.DataFrame(columns=["level_m"], index=pd.DatetimeIndex([], name="date"))

    year_terms = discover_year_terms()
    month_terms = discover_month_terms()
    if not year_terms or len(month_terms) != 12:
        log.warning(
            "Porto de Manaus taxonomy discovery incomplete (%d years, %d months); "
            "skipping source", len(year_terms), len(month_terms),
        )
        return empty
    log.info("Porto de Manaus archive: %d years (%d-%d)",
             len(year_terms), min(year_terms), max(year_terms))

    frames: list[pd.DataFrame] = []
    for year in sorted(year_terms):
        year_rows = 0
        for month in range(1, 13):
            params = {
                "jsf": "jet-engine:result",
                "tax": f"ano:{year_terms[year]};meses:{month_terms[month]}",
            }
            html_text = _get(LEVEL_PAGE, params)
            time.sleep(_POLITE_DELAY)
            if html_text is None:
                continue
            frame = _parse_month_table(html_text, year, month)
            if not frame.empty:
                frames.append(frame)
                year_rows += len(frame)
        log.info("  %d -> %d daily readings", year, year_rows)

    if not frames:
        log.warning("Porto de Manaus returned no parseable readings")
        return empty

    combined = (
        pd.concat(frames, ignore_index=True)
        .dropna()
        .drop_duplicates(subset="date", keep="last")
        .set_index("date")
        .sort_index()
    )
    cleaned = _drop_impossible_jumps(combined["level_m"]).to_frame("level_m")
    log.info("Porto de Manaus: %d readings %s..%s", len(cleaned),
             cleaned.index.min().date(), cleaned.index.max().date())
    if cache:
        cleaned.to_csv(cache_path, index_label="date")
    return cleaned
