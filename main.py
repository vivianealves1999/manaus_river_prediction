"""Manaus (Rio Negro) river-level forecasting - command line entry point.

Examples
--------
Full run, forecasting to December of the current hydrological year::

    python main.py

Re-download every source before running (slow; ~6 min for the Porto de Manaus archive)::

    python main.py --refresh-data

Forecast to a specific month, with a quicker backtest::

    python main.py --target-month 12 --target-year 2026 --origins 24

Only rebuild the dataset, without modelling::

    python main.py --data-only
"""
from __future__ import annotations

import os

# Pin BLAS to a single thread *before* numpy is imported anywhere.
# This workload is thousands of *small* linear-algebra calls (SARIMAX state-space filtering,
# gradient-boosting splits), not a few large ones. Multi-threaded BLAS spends more time on
# thread synchronisation than on arithmetic at this size: measured here, the default settings
# left the process at roughly 20% CPU utilisation and made a full backtest several times
# slower than single-threaded. Set these to taste if you change the model mix.
for _thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_var, "1")

import argparse  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from manaus_river.config import MODEL_CONFIG  # noqa: E402
from manaus_river.pipeline import run  # noqa: E402


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    # These are chatty and say nothing we act on.
    for noisy in ("urllib3", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="manaus-river",
        description="Forecast the Rio Negro stage at Manaus three or more months ahead.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--target-month", type=int, default=12, choices=range(1, 13), metavar="1-12",
        help="Calendar month to forecast to",
    )
    parser.add_argument(
        "--target-year", type=int, default=None,
        help="Calendar year of the target month (default: the next occurrence)",
    )
    parser.add_argument(
        "--refresh-data", action="store_true",
        help="Re-download every source instead of using the cached CSVs",
    )
    parser.add_argument(
        "--origins", type=int, default=MODEL_CONFIG.backtest_origins,
        help="Number of rolling-origin backtest evaluations",
    )
    parser.add_argument(
        "--no-figures", action="store_true", help="Skip figure generation",
    )
    parser.add_argument(
        "--no-scenarios", action="store_true", help="Skip the ENSO what-if scenarios",
    )
    parser.add_argument(
        "--data-only", action="store_true",
        help="Build the dataset and exit without modelling",
    )
    parser.add_argument(
        "--figures-only", action="store_true",
        help="Redraw the figures from the last run without refitting anything",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    log = logging.getLogger("manaus_river")

    if args.figures_only:
        from manaus_river.pipeline import figures_from_artifacts

        for figure in figures_from_artifacts():
            print(f"  Figure:  {figure}")
        return 0

    if args.data_only:
        from manaus_river.data.build_dataset import build_panel

        panel = build_panel(cache=not args.refresh_data)
        log.info("Dataset ready: %d rows x %d columns", *panel.shape)
        return 0

    try:
        output = run(
            target_month=args.target_month,
            target_year=args.target_year,
            refresh_data=args.refresh_data,
            n_origins=args.origins,
            make_figures=not args.no_figures,
            run_scenarios=not args.no_scenarios,
        )
    except Exception:  # noqa: BLE001 - surface the traceback, exit non-zero
        log.exception("Pipeline failed")
        return 1

    headline = output.headline
    print()
    print("=" * 78)
    if not headline.empty:
        target = headline.index[-1]
        print(f"  {target:%B %Y} forecast: {headline['forecast_m'].iloc[-1]:.2f} m")
        if {"lower_m", "upper_m"} <= set(headline.columns):
            print(
                f"  80% interval:     {headline['lower_m'].iloc[-1]:.2f} m "
                f"to {headline['upper_m'].iloc[-1]:.2f} m"
            )
    print(f"  Best backtest model: {output.backtest.best_model()}")
    print(f"  Report:  {output.report_path}")
    for figure in output.figures:
        print(f"  Figure:  {figure}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
