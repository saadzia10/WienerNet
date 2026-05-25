"""Command-line entrypoint for the data pipeline.

Usage:
    # Convert raw FLUXNET data to canonical schema
    python -m data_pipeline.cli preprocess --site woodwalton

    # Fit (E0, rb) on the canonical data
    python -m data_pipeline.cli partition --site woodwalton --strategy astral

    # Both in one go
    python -m data_pipeline.cli all --site woodwalton --strategy astral
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from . import config
from .partitioning import estimate_for_site
from .preprocessing import SitePreprocessor

# Logger config kept simple here — the wienernet.utils.logging setup is for
# training runs; this script just prints to stderr.
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("data_pipeline.cli")


SITES_ROOT = Path("data_manipulation/other_sites")
MAIN_DATA = Path("data_manipulation/complete_data.parquet")
MAIN_NIGHT_OUT = Path("data_manipulation/final_night_data.parquet")
MAIN_DAY_OUT = Path("data_manipulation/final_day_data.parquet")


def _site_paths(site: str) -> dict[str, Path]:
    base = SITES_ROOT / site
    return {
        "raw": base / "data.parquet",
        "processed": base / "processed_data.parquet",
        "night": base / "final_night_data.parquet",
        "day": base / "final_day_data.parquet",
    }


def cmd_preprocess(args: argparse.Namespace) -> None:
    paths = _site_paths(args.site)
    if not paths["raw"].exists():
        raise FileNotFoundError(f"Raw data not found: {paths['raw']}")
    pp = SitePreprocessor()
    pp.process_file(
        paths["raw"],
        paths["processed"],
        radiation_source=args.radiation_source,
        swin_column=args.swin_column,
    )


def cmd_partition(args: argparse.Namespace) -> None:
    paths = _site_paths(args.site) if args.site != "main" else None
    if paths is None:
        # Main Rosedene site
        in_path = MAIN_DATA
        night_out = MAIN_NIGHT_OUT
        day_out = MAIN_DAY_OUT if not args.skip_day else None
        site_arg = None
        strategy = "column"
    else:
        in_path = paths["processed"]
        night_out = paths["night"]
        day_out = paths["day"] if not args.skip_day else None
        site_arg = args.site
        strategy = args.strategy

    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}. Run 'preprocess' first.")

    estimate_for_site(
        in_path,
        night_output_path=night_out,
        day_output_path=day_out,
        site=site_arg,
        night_mask_strategy=strategy,
        reco_column=args.reco_column,
    )


def cmd_all(args: argparse.Namespace) -> None:
    cmd_preprocess(args)
    cmd_partition(args)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="WienerNet data pipeline CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # --- preprocess ---
    p_pre = sub.add_parser("preprocess", help="Raw FLUXNET -> canonical schema")
    p_pre.add_argument("--site", required=True, choices=sorted(config.SITES.keys()))
    p_pre.add_argument("--radiation-source", default="components", choices=("components", "swin"))
    p_pre.add_argument("--swin-column", default="SWin")
    p_pre.set_defaults(func=cmd_preprocess)

    # --- partition ---
    p_part = sub.add_parser("partition", help="Fit E0, rb (and optionally alpha, beta)")
    p_part.add_argument("--site", required=True, help=f"Site name (or 'main'). Known: {sorted(config.SITES.keys())}")
    p_part.add_argument("--strategy", default="astral", choices=("column", "astral"),
                        help="How to identify night (ignored for 'main')")
    p_part.add_argument("--reco-column", default="TER", help="Falls back to NEE if missing")
    p_part.add_argument("--skip-day", action="store_true", help="Skip day-time alpha/beta fit")
    p_part.set_defaults(func=cmd_partition)

    # --- all ---
    p_all = sub.add_parser("all", help="preprocess + partition in one go")
    p_all.add_argument("--site", required=True, choices=sorted(config.SITES.keys()))
    p_all.add_argument("--radiation-source", default="components", choices=("components", "swin"))
    p_all.add_argument("--swin-column", default="SWin")
    p_all.add_argument("--strategy", default="astral", choices=("column", "astral"))
    p_all.add_argument("--reco-column", default="TER")
    p_all.add_argument("--skip-day", action="store_true")
    p_all.set_defaults(func=cmd_all)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
