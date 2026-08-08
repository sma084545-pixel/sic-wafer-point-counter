#!/usr/bin/env python3
"""Create a provenance-bound DIC/KOH/expert reference annotation template."""

from __future__ import annotations

import argparse
from pathlib import Path

from sic_wafer_counter.paper_alignment import write_reference_template


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create an independent-reference CSV template. The template starts "
            "as uncertain/not_registered and never fabricates confirmed points."
        )
    )
    parser.add_argument("source_image", type=Path, help="XRT image used by the analysis")
    parser.add_argument("reference_image", type=Path, help="Independent DIC/KOH image")
    parser.add_argument("--output", type=Path, required=True, help="Destination CSV")
    parser.add_argument(
        "--method",
        choices=("DIC", "KOH", "expert_annotation"),
        default="DIC",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    path = write_reference_template(
        args.output,
        source_image_path=args.source_image,
        reference_image_path=args.reference_image,
        reference_method=args.method,
    )
    print(f"Reference template: {path}")
    print("Status: uncertain / not_registered (edit only after independent registration)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
