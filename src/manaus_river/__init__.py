"""Seasonal forecasting of the Rio Negro water level at Porto de Manaus.

Typical use is through the CLI (``python main.py``), but the pipeline is importable::

    from manaus_river.pipeline import run
    output = run(target_month=12, target_year=2026)
    print(output.headline)          # the ensemble forecast with intervals
    print(output.backtest.overall)  # how much to trust it

See ``docs/ARCHITECTURE.md`` for the layering and ``docs/MODELING.md`` for the reasoning
behind the model stack and the validation protocol.
"""
from __future__ import annotations

__version__ = "1.0.0"

__all__ = [
    "config",
    "features",
    "backtest",
    "scenarios",
    "plots",
    "pipeline",
]
