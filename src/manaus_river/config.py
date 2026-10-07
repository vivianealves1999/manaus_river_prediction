from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
FORECAST_DIR = OUTPUT_DIR / "forecasts"
REPORT_DIR = OUTPUT_DIR / "reports"

for _p in (RAW_DIR, PROCESSED_DIR, EXTERNAL_DIR, FIGURE_DIR, FORECAST_DIR, REPORT_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------------------
# Hydrological stations (ANA / HidroWeb codes)
@dataclass(frozen=True)
class Station:
    code: str
    name: str
    river: str
    role: str  # "target" or "upstream"
    note: str = ""


TARGET_STATION = Station(
    code="14990000",
    name="Manaus",
    river="Rio Negro",
    role="target",
    note="Porto de Manaus gauge. Datum (gauge zero) = 0 cm on the local ruler.",
)

# Upstream / neighbouring gauges used as exogenous hydrological predictors.
# Rationale for each one is documented in docs/DATA_SOURCES.md.
UPSTREAM_STATIONS: tuple[Station, ...] = (
    Station(
        "14100000", "Manacapuru", "Rio Solimoes", "upstream",
        "~80 km upstream of the confluence. The Solimoes backwater physically dams the "
        "Rio Negro at Manaus, so this is the single strongest non-local predictor.",
    ),
    Station(
        "14710000", "Caracarai", "Rio Branco", "upstream",
        "Largest tributary of the Rio Negro; Northern-Hemisphere rainfall regime.",
    ),
    Station(
        "13150000", "Tabatinga", "Rio Solimoes", "upstream",
        "Far-upstream Solimoes (Peru/Colombia border); ~2-3 month travel time to Manaus.",
    ),
    Station(
        "15400000", "Porto Velho", "Rio Madeira", "upstream",
        "Southern Amazon regime; proxy for the southern-basin monsoon.",
    ),
)

ALL_STATIONS: tuple[Station, ...] = (TARGET_STATION,) + UPSTREAM_STATIONS

# --------------------------------------------------------------------------------------
# Data sources
ANA_SERVICE_URL = "http://telemetriaws1.ana.gov.br/ServiceANA.asmx/HidroSerieHistorica"
ANA_CHUNK_YEARS = 5          # the service silently returns an empty set on long ranges
ANA_START_YEAR = 1900        # the loader clips to whatever the service actually returns
PORTO_MANAUS_URL = "https://portodemanaus.com.br/nivel-do-rio-negro/"

# NOAA PSL climate indices. Fixed-width ASCII, one row per year, 12 monthly columns.
CLIMATE_INDEX_URLS: dict[str, str] = {
    "oni": "https://psl.noaa.gov/data/correlation/oni.data",   # Nino 3.4 SST anomaly (ENSO)
    "tna": "https://psl.noaa.gov/data/correlation/tna.data",   # Tropical North Atlantic SST anomaly
    "tsa": "https://psl.noaa.gov/data/correlation/tsa.data",   # Tropical South Atlantic SST anomaly
    "soi": "https://psl.noaa.gov/data/correlation/soi.data",   # Southern Oscillation Index
    "amo": "https://psl.noaa.gov/data/correlation/amon.us.data",  # Atlantic Multidecadal Oscillation
}

# --------------------------------------------------------------------------------------
# Modelling
@dataclass
class ModelConfig:
    """Hyper-parameters shared by the whole modelling stack."""

    horizon: int = 3                      # months ahead to forecast (>= 3 per requirement)
    seasonal_period: int = 12             # monthly data
    min_train_years: int = 25             # first rolling-origin fit uses at least this much
    backtest_origins: int = 48            # number of rolling-origin evaluations
    alpha: float = 0.20                   # 80% prediction interval

    # SARIMAX search space (small, ordered by prior plausibility for a strongly
    # seasonal, highly persistent stage series).
    sarimax_orders: tuple[tuple[tuple[int, int, int], tuple[int, int, int, int]], ...] = field(
        default_factory=lambda: (
            ((1, 0, 0), (1, 1, 1, 12)),
            ((2, 0, 1), (1, 1, 1, 12)),
            ((1, 1, 1), (0, 1, 1, 12)),
            ((2, 1, 1), (1, 1, 1, 12)),
            ((3, 0, 0), (2, 1, 0, 12)),
        )
    )

    # Climate-index lags (months). ENSO leads Amazon hydrology by roughly 1-2 seasons:
    # the atmospheric teleconnection sets rainfall, then catchment storage and routing
    # add further delay before the stage at Manaus responds.
    enso_lags: tuple[int, ...] = (0, 3, 6, 9, 12)
    upstream_lags: tuple[int, ...] = (0, 1, 2, 3, 6)

    # Gradient boosting (direct multi-horizon)
    gbm_n_estimators: int = 250
    gbm_learning_rate: float = 0.05
    gbm_max_depth: int = 3
    gbm_subsample: float = 0.9

    random_state: int = 42


MODEL_CONFIG = ModelConfig()
