#!/usr/bin/env python3
"""Generate deterministic wafer images with independent ground-truth labels.

The detector never imports or reads these labels.  They exist solely for tests,
threshold calibration, and precision/recall evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd

KINDS = ("clean", "noisy", "difficult")
DEFAULT_SEED = 20230112
IMAGE_SIZE = 1024
WAFER_DIAMETER_MM = 100.0

SCENARIO_SETTINGS: dict[str, dict[str, Any]] = {
    "clean": {
        "point_count": 96,
        "noise_sigma": 1.2,
        "gradient": 3.0,
        "stripes": False,
        "touching_pairs": 0,
    },
    "noisy": {
        "point_count": 132,
        "noise_sigma": 7.0,
        "gradient": 20.0,
        "stripes": True,
        "touching_pairs": 0,
    },
    "difficult": {
        # 144 isolated + 8 pairs (16 individual points) = 160 true targets.
        "point_count": 160,
        "noise_sigma": 8.0,
        "gradient": 25.0,
        "stripes": True,
        "touching_pairs": 8,
    },
}


def _inside_circle(x: float, y: float, center: tuple[int, int], radius: float) -> bool:
    return (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2


def _sample_centers(
    rng: np.random.Generator,
    count: int,
    center: tuple[int, int],
    max_radius: float,
    *,
    minimum_distance: float,
    existing: list[tuple[float, float]] | None = None,
    allowed: Callable[[float, float], bool] | None = None,
) -> list[tuple[int, int]]:
    """Sample separated positions uniformly by area inside a circle."""

    positions: list[tuple[float, float]] = list(existing or [])
    new_positions: list[tuple[int, int]] = []
    attempts = 0
    while len(new_positions) < count and attempts < count * 2000:
        attempts += 1
        radial = max_radius * math.sqrt(float(rng.random()))
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        candidate = (
            int(round(center[0] + radial * math.cos(angle))),
            int(round(center[1] + radial * math.sin(angle))),
        )
        if (allowed is None or allowed(*candidate)) and all(
            (candidate[0] - prior[0]) ** 2 + (candidate[1] - prior[1]) ** 2
            >= minimum_distance**2
            for prior in positions
        ):
            positions.append(candidate)
            new_positions.append(candidate)
    if len(new_positions) != count:
        raise RuntimeError(f"Could only place {len(new_positions)} of {count} synthetic points")
    return new_positions


def _object_record(
    object_id: int,
    kind: str,
    *,
    is_true_defect: bool,
    should_count: bool,
    artifact_type: str,
    center_x_px: float,
    center_y_px: float,
    radius_px: float | None = None,
    contrast: float | None = None,
    touching_group: int | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Create one row of the independent truth/audit table."""

    return {
        "object_id": object_id,
        "defect_id": object_id if is_true_defect else "",
        "scenario": kind,
        "is_true_defect": bool(is_true_defect),
        "should_count": bool(should_count),
        "artifact_type": artifact_type,
        "center_x_px": float(center_x_px),
        "center_y_px": float(center_y_px),
        "radius_px": float(radius_px) if radius_px is not None else np.nan,
        "diameter_px": float(radius_px * 2.0) if radius_px is not None else np.nan,
        "contrast": float(contrast) if contrast is not None else np.nan,
        "touching_group": touching_group if touching_group is not None else "",
        "notes": notes,
    }


def generate_synthetic_wafer(
    kind: str,
    output_dir: Path,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Generate one synthetic wafer and its independent label table.

    Parameters
    ----------
    kind:
        ``clean``, ``noisy``, or ``difficult``.
    output_dir:
        Destination directory.  Existing unrelated files are not removed.
    seed:
        Random seed.  The same kind and seed produce byte-identical PNG pixels
        and the same label rows.

    Returns
    -------
    dict
        Paths, true count, and geometry used by integration tests and demos.
    """

    if kind not in KINDS:
        raise ValueError(f"Unknown synthetic kind {kind!r}; choose one of {KINDS}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setting = SCENARIO_SETTINGS[kind]
    # A stable per-scenario offset means --all does not produce correlated noise.
    scenario_seed = int(seed) + KINDS.index(kind) * 100_003
    rng = np.random.default_rng(scenario_seed)

    size = IMAGE_SIZE
    center = (size // 2 + 3, size // 2 - 5)
    radius = 448
    yy, xx = np.ogrid[:size, :size]
    radial_distance = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    wafer_mask = radial_distance <= radius

    # Dark exterior, bright wafer, smoothly changing illumination.
    image = np.full((size, size), 34.0, dtype=np.float32)
    gradient = float(setting["gradient"])
    wafer_background = (
        208.0
        + gradient * (xx - center[0]) / (2.0 * radius)
        - 0.55 * gradient * (yy - center[1]) / (2.0 * radius)
        + 3.0 * (radial_distance / radius) ** 2
    )
    image[wafer_mask] = wafer_background[wafer_mask]
    if bool(setting["stripes"]):
        horizontal = 3.0 * np.sin(np.arange(size, dtype=np.float32)[:, None] * 2.0 * np.pi / 37.0)
        vertical = 1.8 * np.sin(np.arange(size, dtype=np.float32)[None, :] * 2.0 * np.pi / 83.0)
        stripe_field = horizontal + vertical
        image[wafer_mask] += np.broadcast_to(stripe_field, image.shape)[wafer_mask]
    noise = rng.normal(0.0, float(setting["noise_sigma"]), size=image.shape).astype(np.float32)
    image[wafer_mask] += noise[wafer_mask]
    # Outside has lighter, lower-amplitude sensor noise so outside dots remain visible.
    image[~wafer_mask] += rng.normal(0.0, 1.0, int((~wafer_mask).sum())).astype(np.float32)
    image = np.clip(image, 0, 255).astype(np.uint8)

    records: list[dict[str, Any]] = []
    occupied: list[tuple[float, float]] = []
    next_object_id = 1
    total_points = int(setting["point_count"])
    touching_pairs = int(setting["touching_pairs"])
    isolated_count = total_points - touching_pairs * 2

    difficult_lines = [
        ((250, 320), (465, 335), 3),
        ((585, 710), (835, 650), 4),
        ((315, 780), (335, 575), 2),
    ]

    def difficult_point_allowed(x: float, y: float) -> bool:
        """Keep true points visibly separate from the known false artifacts."""

        if kind != "difficult":
            return True
        if any((x - cx) ** 2 + (y - cy) ** 2 < clearance**2 for cx, cy, clearance in ((660, 290, 48), (390, 620, 54))):
            return False
        if 140 <= x <= 270 and 420 <= y <= 525:  # text plus point-radius margin
            return False
        if 680 <= x <= 855 and 785 <= y <= 845:  # scale bar plus margin
            return False
        point = np.array([x, y], dtype=float)
        for start, end, _ in difficult_lines:
            a = np.asarray(start, dtype=float)
            b = np.asarray(end, dtype=float)
            segment = b - a
            fraction = float(np.clip(np.dot(point - a, segment) / np.dot(segment, segment), 0.0, 1.0))
            if float(np.linalg.norm(point - (a + fraction * segment))) < 18.0:
                return False
        return True

    isolated = _sample_centers(
        rng,
        isolated_count,
        center,
        radius - 35,
        minimum_distance=15.0 if kind == "clean" else 12.0,
        allowed=difficult_point_allowed,
    )
    occupied.extend(isolated)
    for x, y in isolated:
        point_radius = int(rng.integers(3, 7))
        contrast = float(rng.uniform(80, 145) if kind == "clean" else rng.uniform(45, 150))
        local_level = float(image[y, x])
        intensity = int(np.clip(local_level - contrast, 5, 155))
        cv2.circle(image, (x, y), point_radius, intensity, -1, lineType=cv2.LINE_AA)
        records.append(
            _object_record(
                next_object_id,
                kind,
                is_true_defect=True,
                should_count=True,
                artifact_type="point",
                center_x_px=x,
                center_y_px=y,
                radius_px=point_radius,
                contrast=local_level - intensity,
            )
        )
        next_object_id += 1

    # Each member of a touching pair remains an independent physical truth row.
    pair_bases = _sample_centers(
        rng,
        touching_pairs,
        center,
        radius - 55,
        minimum_distance=35.0,
        existing=occupied,
        allowed=difficult_point_allowed,
    )
    for group, (base_x, base_y) in enumerate(pair_bases, start=1):
        point_radius = int(rng.integers(5, 8))
        angle = float(rng.uniform(0, 2 * math.pi))
        separation = max(5.0, point_radius * 1.35)
        dx, dy = separation * math.cos(angle), separation * math.sin(angle)
        pair = [
            (int(round(base_x - dx / 2)), int(round(base_y - dy / 2))),
            (int(round(base_x + dx / 2)), int(round(base_y + dy / 2))),
        ]
        for x, y in pair:
            local_level = float(image[y, x])
            contrast = float(rng.uniform(75, 135))
            intensity = int(np.clip(local_level - contrast, 8, 130))
            cv2.circle(image, (x, y), point_radius, intensity, -1, lineType=cv2.LINE_AA)
            records.append(
                _object_record(
                    next_object_id,
                    kind,
                    is_true_defect=True,
                    should_count=True,
                    artifact_type="touching_point",
                    center_x_px=x,
                    center_y_px=y,
                    radius_px=point_radius,
                    contrast=local_level - intensity,
                    touching_group=group,
                )
            )
            next_object_id += 1

    # Known false objects are recorded too, allowing explicit false-positive QA.
    if kind in {"noisy", "difficult"}:
        outside_positions = [(35, 155), (950, 180), (87, 880), (945, 860)]
        for x, y in outside_positions:
            cv2.circle(image, (x, y), 5, 3, -1, lineType=cv2.LINE_AA)
            records.append(
                _object_record(
                    next_object_id,
                    kind,
                    is_true_defect=False,
                    should_count=False,
                    artifact_type="outside_point",
                    center_x_px=x,
                    center_y_px=y,
                    radius_px=5,
                    notes="Outside the wafer mask; must not be counted.",
                )
            )
            next_object_id += 1

    if kind == "difficult":
        for start, end, thickness in difficult_lines:
            cv2.line(image, start, end, 28, thickness, lineType=cv2.LINE_AA)
            records.append(
                _object_record(
                    next_object_id,
                    kind,
                    is_true_defect=False,
                    should_count=False,
                    artifact_type="line",
                    center_x_px=(start[0] + end[0]) / 2,
                    center_y_px=(start[1] + end[1]) / 2,
                    notes=f"line_start={start}; line_end={end}; thickness={thickness}",
                )
            )
            next_object_id += 1
        for x, y, blob_radius in [(660, 290, 24), (390, 620, 30)]:
            cv2.circle(image, (x, y), blob_radius, 22, -1, lineType=cv2.LINE_AA)
            records.append(
                _object_record(
                    next_object_id,
                    kind,
                    is_true_defect=False,
                    should_count=False,
                    artifact_type="large_blob",
                    center_x_px=x,
                    center_y_px=y,
                    radius_px=blob_radius,
                    notes="Oversized dark artifact; must be rejected.",
                )
            )
            next_object_id += 1
        # A dark label plate with light lettering forms one oversized artifact;
        # it exercises text/label rejection without watershed splitting letters
        # into many superficially point-like candidates.
        cv2.rectangle(image, (150, 430), (265, 505), 42, thickness=-1)
        cv2.putText(image, "SiC", (169, 485), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 195, 3, cv2.LINE_AA)
        records.append(
            _object_record(
                next_object_id,
                kind,
                is_true_defect=False,
                should_count=False,
                artifact_type="text",
                center_x_px=207.5,
                center_y_px=467.5,
                notes="Synthetic dark label plate with light text; oversized invalid artifact.",
            )
        )
        next_object_id += 1
        cv2.line(image, (710, 815), (830, 815), 20, 6, lineType=cv2.LINE_AA)
        records.append(
            _object_record(
                next_object_id,
                kind,
                is_true_defect=False,
                should_count=False,
                artifact_type="scale_bar",
                center_x_px=770,
                center_y_px=815,
                notes="Synthetic scale-bar artifact.",
            )
        )

    image_path = output_dir / f"synthetic_{kind}.png"
    truth_path = output_dir / f"synthetic_{kind}_ground_truth.csv"
    metadata_path = output_dir / f"synthetic_{kind}_metadata.json"
    if not cv2.imwrite(str(image_path), image):
        raise OSError(f"Could not save synthetic image: {image_path}")
    truth = pd.DataFrame(records)
    truth.to_csv(truth_path, index=False)

    diameter_px = radius * 2.0
    mm_per_pixel = WAFER_DIAMETER_MM / diameter_px
    valid_pixel_count = int(wafer_mask.sum())
    valid_area_cm2 = valid_pixel_count * (mm_per_pixel / 10.0) ** 2
    true_count = int(truth["should_count"].sum())
    metadata = {
        "scenario": kind,
        "seed": scenario_seed,
        "base_seed": int(seed),
        "image_path": str(image_path),
        "ground_truth_path": str(truth_path),
        "true_valid_count": true_count,
        "image_shape": [size, size],
        "geometry": {
            "center_x_px": center[0],
            "center_y_px": center[1],
            "radius_px": radius,
            "diameter_px": diameter_px,
            "wafer_diameter_mm": WAFER_DIAMETER_MM,
            "mm_per_pixel": mm_per_pixel,
            "valid_pixel_count": valid_pixel_count,
            "valid_area_cm2": valid_area_cm2,
            "theoretical_area_cm2": math.pi * 5.0**2,
        },
        "artifact_counts": truth["artifact_type"].value_counts().sort_index().to_dict(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "image_path": image_path,
        "ground_truth_path": truth_path,
        "metadata_path": metadata_path,
        "true_valid_count": true_count,
        "geometry": metadata["geometry"],
        "scenario": kind,
        "seed": scenario_seed,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成带独立真值表的可重复 SiC 晶圆模拟图。"
    )
    parser.add_argument("--kind", choices=KINDS, default="clean", help="单个场景（默认 clean）")
    parser.add_argument("--all", action="store_true", help="一次生成 clean/noisy/difficult 三组数据")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "sample_data" / "generated",
        help="输出目录",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="固定随机种子")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    kinds = KINDS if args.all else (args.kind,)
    for kind in kinds:
        result = generate_synthetic_wafer(kind, args.output_dir, args.seed)
        geometry = result["geometry"]
        print(
            f"{kind}: image={result['image_path']} truth={result['ground_truth_path']} "
            f"n_true={result['true_valid_count']} valid_area={geometry['valid_area_cm2']:.6f} cm^2"
        )
    print("真值 CSV 仅用于评估/校准，分析程序不会读取它。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
