"""Tests for the parts of the pipeline where a silent bug would be most expensive.

Run with::

    python -m pytest tests -q          # if pytest is installed
    python tests/test_pipeline.py      # otherwise: plain-assert self-check

The emphasis is deliberate. Most of this project is glue around external services that
cannot be unit-tested meaningfully. What *can* be tested, and badly needs to be, is the
no-leakage contract in the feature layer: a leak there produces an excellent-looking backtest
and a worthless forecast, with no error message anywhere.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from manaus_river.config import ModelConfig  # noqa: E402
from manaus_river.data.climate import classify_enso, parse_psl_ascii  # noqa: E402
from manaus_river.data.porto_manaus import (  # noqa: E402
    _drop_impossible_jumps,
    _normalise_level,
)
from manaus_river.features import build_design_matrix  # noqa: E402
from manaus_river.models.baselines import (  # noqa: E402
    ClimatologyForecaster,
    SeasonalNaiveForecaster,
)
from manaus_river.models.ensemble import WeightedEnsemble  # noqa: E402
from manaus_river.pipeline import months_until  # noqa: E402

CONFIG = ModelConfig()


def _synthetic_panel(n_years: int = 40) -> pd.DataFrame:
    """A Rio-Negro-shaped series: strong annual pulse, persistence, an ENSO response."""
    index = pd.date_range("1980-01-01", periods=n_years * 12, freq="MS")
    month = index.month.to_numpy()
    seasonal = 5.0 * np.sin(2 * np.pi * (month - 4) / 12.0)
    rng = np.random.default_rng(0)
    oni = pd.Series(rng.normal(0, 0.8, len(index)), index=index).rolling(
        6, min_periods=1
    ).mean()
    level = 22.0 + seasonal - 0.8 * oni.shift(4).fillna(0.0).to_numpy() + rng.normal(0, 0.4, len(index))
    return pd.DataFrame(
        {
            "level_m": level,
            "oni": oni.to_numpy(),
            "tna": rng.normal(0, 0.3, len(index)),
            "manacapuru_level_m": level + rng.normal(0, 1.0, len(index)),
        },
        index=index,
    )


# ---------------------------------------------------------------------------------------
# The contract that matters most: no feature may know the future
# ---------------------------------------------------------------------------------------
def test_no_feature_leaks_the_future() -> None:
    """Every design-matrix value for target month t must be computable at t - horizon.

    Verified behaviourally rather than by inspection: perturb the panel *after* the forecast
    origin and assert that not one feature value for that origin changes. If any lag were
    shorter than the horizon, the perturbation would show up.
    """
    panel = _synthetic_panel()
    horizon = 3

    design_a, target_a = build_design_matrix(panel, horizon, CONFIG)
    origin_row = design_a.index[-1]

    # Corrupt everything strictly after (target month - horizon).
    cutoff = origin_row - pd.DateOffset(months=horizon)
    perturbed = panel.copy()
    mask = perturbed.index > cutoff
    perturbed.loc[mask, "level_m"] += 7.5
    perturbed.loc[mask, "oni"] += 4.0
    perturbed.loc[mask, "tna"] += 4.0
    perturbed.loc[mask, "manacapuru_level_m"] += 7.5

    design_b, _ = build_design_matrix(perturbed, horizon, CONFIG)

    shared = [c for c in design_a.columns if c in design_b.columns]
    row_a = design_a.loc[origin_row, shared]
    row_b = design_b.loc[origin_row, shared]
    changed = [c for c in shared if not np.isclose(row_a[c], row_b[c], atol=1e-9)]
    assert not changed, f"Leaking features (changed when the future changed): {changed}"
    assert len(shared) > 10, "Design matrix suspiciously small; test would be vacuous"


def test_target_lags_respect_horizon() -> None:
    """No level lag shorter than the horizon should ever be generated."""
    panel = _synthetic_panel()
    for horizon in (1, 3, 6):
        design, _ = build_design_matrix(panel, horizon, CONFIG)
        for column in design.columns:
            if column.startswith("level_lag_"):
                assert int(column.rsplit("_", 1)[1]) >= horizon, column
            for prefix in ("oni_lag_", "tna_lag_", "tsa_lag_", "soi_lag_"):
                if column.startswith(prefix):
                    assert int(column.rsplit("_", 1)[1]) >= horizon, column


def test_stale_predictors_are_dropped_not_imputed() -> None:
    """A predictor that stops updating must leave the design matrix entirely."""
    panel = _synthetic_panel()
    panel.loc[panel.index[-40:], "manacapuru_level_m"] = np.nan
    design, _ = build_design_matrix(panel, 3, CONFIG, require_recent=12)
    stale = [c for c in design.columns if c.startswith("manacapuru")]
    assert not stale, f"Stale predictor survived: {stale}"


# ---------------------------------------------------------------------------------------
# Forecast plumbing
# ---------------------------------------------------------------------------------------
def test_forecast_index_is_the_months_after_the_origin() -> None:
    panel = _synthetic_panel()
    model = ClimatologyForecaster().fit(panel, 3)
    result = model.predict(panel, 3)
    expected = pd.date_range(
        panel.index.max() + pd.offsets.MonthBegin(1), periods=3, freq="MS"
    )
    assert list(result.mean.index) == list(expected)
    assert result.lower is not None and (result.lower < result.mean).all()
    assert result.upper is not None and (result.upper > result.mean).all()


def test_baselines_produce_physically_plausible_values() -> None:
    panel = _synthetic_panel()
    for factory in (ClimatologyForecaster, SeasonalNaiveForecaster):
        result = factory().fit(panel, 3).predict(panel, 3)
        assert result.mean.notna().all()
        assert result.mean.between(10.0, 35.0).all(), f"{factory.__name__}: {result.mean.tolist()}"


def test_months_until_handles_the_year_boundary() -> None:
    # Target still ahead in the same year.
    assert months_until(12, None, pd.Timestamp("2026-09-01")) == 3
    # Target already passed -> next year's occurrence.
    assert months_until(3, None, pd.Timestamp("2026-09-01")) == 6
    # Explicit year.
    assert months_until(12, 2027, pd.Timestamp("2026-09-01")) == 15
    # Target month equal to the origin month -> next year, never zero.
    assert months_until(9, None, pd.Timestamp("2026-09-01")) == 12


def test_ensemble_weights_are_normalised_and_favour_skill() -> None:
    weights = WeightedEnsemble.weights_from_scores(
        {"good": 0.5, "bad": 2.0, "broken": float("nan")}
    )
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert "broken" not in weights, "Non-finite score should be dropped, not zero-weighted"
    assert weights["good"] > weights["bad"]
    # Inverse-MSE: a 4x worse RMSE should get 16x less weight.
    assert abs(weights["good"] / weights["bad"] - 16.0) < 1e-6


# ---------------------------------------------------------------------------------------
# Data-quality helpers
# ---------------------------------------------------------------------------------------
def test_level_unit_normalisation() -> None:
    assert _normalise_level(24.09) == 24.09        # already metres
    assert _normalise_level(1641.0) == 16.41       # centimetres
    assert _normalise_level(2649.0) == 26.49
    assert _normalise_level(0.0) is None           # implausible
    assert _normalise_level(99999.0) is None


def test_typo_spikes_are_removed_but_real_steps_survive() -> None:
    index = pd.date_range("2012-12-01", periods=7, freq="D")
    series = pd.Series([24.0, 24.05, 31.5, 24.1, 24.15, 24.2, 24.25], index=index)
    cleaned = _drop_impossible_jumps(series)
    assert 31.5 not in cleaned.values, "Isolated spike should be dropped"
    assert len(cleaned) == 6

    # A genuine monotone move must not be mistaken for a typo.
    rising = pd.Series(
        [20.0, 20.8, 21.6, 22.4, 23.2, 24.0, 24.8],
        index=pd.date_range("2013-01-01", periods=7, freq="D"),
    )
    assert len(_drop_impossible_jumps(rising)) == len(rising)


def test_psl_parser_honours_the_sentinel() -> None:
    text = "\n".join(
        [
            "  1950         1952",
            " 1950  -1.53  -1.34  -1.16  -1.18  -1.07  -0.85  -0.54  -0.42  -0.39  -0.44  -0.60  -0.80",
            " 1951  -0.82 -99.99  -0.17   0.18   0.36   0.58   0.70   0.89   0.99   1.15   1.04   0.81",
            " 1952   0.53   0.37   0.34   0.29   0.20   0.00  -0.08   0.00   0.15   0.10   0.04   0.15",
            "  -99.99",
            "  NOAA PSL provenance footer",
        ]
    )
    series = parse_psl_ascii(text, "oni")
    assert len(series) == 35, "One sentinel value should have been dropped"
    assert pd.Timestamp("1951-02-01") not in series.index
    assert abs(series.loc[pd.Timestamp("1950-01-01")] - (-1.53)) < 1e-9
    assert series.abs().max() < 50


def test_enso_classification_matches_the_oni_definition() -> None:
    index = pd.date_range("2000-01-01", periods=12, freq="MS")
    oni = pd.Series([2.0] * 4 + [0.0] * 4 + [-2.0] * 4, index=index)
    phases = classify_enso(oni)
    assert phases.iloc[1] == "El Nino"
    assert phases.iloc[5] == "Neutral"
    assert phases.iloc[10] == "La Nina"


# ---------------------------------------------------------------------------------------
def _main() -> int:
    """Run every test in this module without requiring pytest."""
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  ERROR {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
