#!/usr/bin/env python3
"""Compose one or two comparable run outputs into a publication-style plate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np


def _load_summary(folder: Path) -> Mapping[str, Any]:
    path = folder / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _display_scale(summary: Mapping[str, Any]) -> tuple[float | None, float | None]:
    density = summary.get("spatial_density", {}).get("density_heatmap", {})
    return density.get("display_vmin_cm2"), density.get("display_vmax_cm2")


def _read(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Could not read {path}")
    return image


def build_comparison(run_dirs: list[Path], output_path: Path) -> Path:
    if not 1 <= len(run_dirs) <= 2:
        raise ValueError("Provide one or two result directories")
    summaries = [_load_summary(folder) for folder in run_dirs]
    scales = [_display_scale(summary) for summary in summaries]
    if len(scales) == 2 and scales[0] != scales[1]:
        raise ValueError(
            "Both heatmaps must use the same fixed heatmap_vmin_cm2 and "
            "heatmap_vmax_cm2 for a quantitative side-by-side comparison"
        )
    fields = [_read(folder / "paper_detection_field.png") for folder in run_dirs]
    heatmaps = [_read(folder / "density_heatmap.png") for folder in run_dirs]
    cell_width = 720

    def resize(image: np.ndarray) -> np.ndarray:
        scale = cell_width / float(image.shape[1])
        return cv2.resize(
            image,
            (cell_width, max(1, int(round(image.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
        )

    fields = [resize(image) for image in fields]
    heatmaps = [resize(image) for image in heatmaps]
    field_height = max(image.shape[0] for image in fields)
    heatmap_height = max(image.shape[0] for image in heatmaps)
    gap = 24
    header = 78
    footer = 64
    columns = len(run_dirs)
    canvas = np.full(
        (header + field_height + gap + heatmap_height + footer, columns * cell_width, 3),
        255,
        dtype=np.uint8,
    )
    cv2.putText(
        canvas,
        "Paper-aligned XRT point-target comparison",
        (24, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (25, 34, 38),
        2,
        cv2.LINE_AA,
    )
    for index, (folder, field, heatmap, summary) in enumerate(
        zip(run_dirs, fields, heatmaps, summaries)
    ):
        x0 = index * cell_width
        canvas[header:header + field.shape[0], x0:x0 + field.shape[1]] = field
        y_heatmap = header + field_height + gap
        canvas[y_heatmap:y_heatmap + heatmap.shape[0], x0:x0 + heatmap.shape[1]] = heatmap
        density = summary.get("point_density_cm2")
        label = f"{folder.name} | mean point-target density: {float(density):.6g} cm^-2"
        cv2.putText(
            canvas,
            label,
            (x0 + 18, y_heatmap + heatmap_height + 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (50, 60, 64),
            1,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise OSError(f"Could not write {output_path}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dirs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(build_comparison(args.result_dirs, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
