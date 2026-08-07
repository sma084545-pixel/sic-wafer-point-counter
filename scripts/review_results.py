#!/usr/bin/env python3
"""Keyboard-driven review of every candidate produced by an analysis.

The script never overwrites ``defects_all.csv``.  It writes a new
``reviewed_defects.csv`` and a ``reviewed_summary.json`` containing a density
recomputed with the *same valid analysis area* recorded by the original run.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import pandas as pd
import tifffile
from scipy.stats import chi2


def _load_pyplot() -> Any:
    """Import the interactive plotting backend only when a window is opened."""

    import matplotlib.pyplot as plt

    return plt


def _truthy(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "accepted", "accept"}


def _nested_value(data: Mapping[str, Any], *paths: str) -> Any:
    for path in paths:
        current: Any = data
        found = True
        for part in path.split("."):
            if not isinstance(current, Mapping) or part not in current:
                found = False
                break
            current = current[part]
        if found and current is not None:
            return current
    return None


def valid_area_from_summary(summary: Mapping[str, Any]) -> float:
    """Extract the original actual valid area and reject unsafe fallbacks."""

    value = _nested_value(
        summary,
        "valid_analysis_area_cm2",
        "valid_area_cm2",
        "areas.valid_analysis_area_cm2",
        "areas.valid_analysis_cm2",
        "areas.valid_cm2",
    )
    if value is None:
        raise KeyError(
            "summary.json 中没有 valid_analysis_area_cm2；不能用理论 78.5398 cm² 代替。"
        )
    area = float(value)
    if not math.isfinite(area) or area <= 0:
        raise ValueError(f"Invalid valid analysis area: {area!r}")
    return area


def recalculate_reviewed_density(frame: pd.DataFrame, valid_area_cm2: float) -> dict[str, float | int]:
    """Recalculate n/S, Poisson sigma, and exact Garwood 95% interval."""

    if valid_area_cm2 <= 0:
        raise ValueError("valid_area_cm2 must be positive")
    if "accepted" not in frame:
        raise KeyError("reviewed table must contain an accepted column")
    accepted = frame["accepted"].map(_truthy)
    count = int(accepted.sum())
    density = count / valid_area_cm2
    sigma = math.sqrt(count) / valid_area_cm2
    lower_count = 0.0 if count == 0 else float(chi2.ppf(0.025, 2 * count) / 2.0)
    upper_count = float(chi2.ppf(0.975, 2 * (count + 1)) / 2.0)
    return {
        "accepted_count": count,
        "valid_analysis_area_cm2": float(valid_area_cm2),
        "point_density_cm2": float(density),
        "counting_uncertainty_cm2": float(sigma),
        "poisson_95_ci_lower_cm2": lower_count / valid_area_cm2,
        "poisson_95_ci_upper_cm2": upper_count / valid_area_cm2,
    }


class ImageAccessor:
    """Read local crops while avoiding a full TIFF copy whenever possible."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._array: np.ndarray
        if path.suffix.lower() in {".tif", ".tiff"}:
            try:
                self._array = tifffile.memmap(path)
            except (ValueError, OSError):
                # Some compressed TIFFs cannot be memory-mapped; tifffile remains
                # the least surprising accurate fallback for 16-bit data.
                self._array = tifffile.imread(path)
        else:
            loaded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if loaded is None:
                raise OSError(f"无法读取原图：{path}")
            self._array = loaded
        array = np.asarray(self._array)
        while array.ndim > 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim not in {2, 3}:
            raise ValueError(f"只支持单张二维灰度/RGB 图，当前 shape={array.shape}")
        self._array = array

    @property
    def shape(self) -> tuple[int, int]:
        return int(self._array.shape[0]), int(self._array.shape[1])

    def read_crop(self, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        crop = np.asarray(self._array[y0:y1, x0:x1])
        if crop.ndim == 3:
            if crop.shape[2] == 4:
                crop = cv2.cvtColor(crop, cv2.COLOR_BGRA2GRAY)
            else:
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        finite = crop[np.isfinite(crop)]
        if finite.size == 0:
            return np.zeros(crop.shape[:2], dtype=np.uint8)
        low, high = np.percentile(finite, [0.5, 99.5])
        if high <= low:
            high = low + 1.0
        return np.clip((crop.astype(np.float32) - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)


def _coordinate_columns(frame: pd.DataFrame) -> tuple[str, str]:
    for x_name, y_name in (
        ("centroid_x_px", "centroid_y_px"),
        ("center_x_px", "center_y_px"),
        ("x_px", "y_px"),
    ):
        if x_name in frame and y_name in frame:
            return x_name, y_name
    raise KeyError("候选 CSV 缺少 centroid_x_px/centroid_y_px 坐标列")


def save_reviewed_results(
    frame: pd.DataFrame,
    output_csv: Path,
    summary: Mapping[str, Any],
) -> dict[str, float | int]:
    """Persist reviewed states and a recomputed density summary."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    saved = frame.copy()
    saved["accepted"] = saved["accepted"].map(_truthy).astype(bool)
    saved["reviewed_at_utc"] = datetime.now(timezone.utc).isoformat()
    saved.to_csv(output_csv, index=False)
    metrics = recalculate_reviewed_density(saved, valid_area_from_summary(summary))
    reviewed_summary = dict(summary)
    reviewed_summary.update(metrics)
    reviewed_summary["source_summary"] = str(summary.get("input_file_name", ""))
    reviewed_summary["reviewed_defects_csv"] = output_csv.name
    reviewed_summary["reviewed_at_utc"] = datetime.now(timezone.utc).isoformat()
    reviewed_summary["counting_uncertainty_scope"] = (
        "该统计不确定度只反映有限计数造成的随机误差，不包含图像分割、漏检、误检以及物理判定错误造成的系统误差。"
    )
    summary_path = output_csv.with_name("reviewed_summary.json")
    summary_path.write_text(
        json.dumps(reviewed_summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return metrics


class CandidateReviewer:
    """Matplotlib state machine for accept/reject/restore review."""

    def __init__(
        self,
        image_path: Path,
        frame: pd.DataFrame,
        summary: Mapping[str, Any],
        output_csv: Path,
        crop_half_size: int = 64,
    ) -> None:
        if frame.empty:
            raise ValueError("候选 CSV 为空，无目标可复核。")
        self.image = ImageAccessor(image_path)
        self.frame = frame.reset_index(drop=True).copy()
        if "accepted" not in self.frame:
            self.frame["accepted"] = False
        self.frame["accepted"] = self.frame["accepted"].map(_truthy).astype(bool)
        self.frame["auto_accepted"] = self.frame["accepted"].copy()
        if "review_action" not in self.frame:
            self.frame["review_action"] = "unreviewed"
        self.x_column, self.y_column = _coordinate_columns(self.frame)
        self.summary = summary
        self.output_csv = output_csv
        self.crop_half_size = max(12, int(crop_half_size))
        self.index = 0
        self._saved = False
        self.plt = _load_pyplot()
        self.figure, self.axis = self.plt.subplots(figsize=(7.4, 7.0))
        self.figure.canvas.mpl_connect("key_press_event", self._on_key)
        self.figure.canvas.mpl_connect("close_event", self._on_close)

    def _status(self) -> str:
        record = self.frame.iloc[self.index]
        action = str(record.get("review_action", "unreviewed"))
        decision = "接受" if bool(record["accepted"]) else "拒绝"
        return f"{decision} / {action}"

    def _draw(self) -> None:
        record = self.frame.iloc[self.index]
        x = int(round(float(record[self.x_column])))
        y = int(round(float(record[self.y_column])))
        height, width = self.image.shape
        x0, x1 = max(0, x - self.crop_half_size), min(width, x + self.crop_half_size + 1)
        y0, y1 = max(0, y - self.crop_half_size), min(height, y + self.crop_half_size + 1)
        crop = self.image.read_crop(x0, y0, x1, y1)
        local_x, local_y = x - x0, y - y0
        self.axis.clear()
        self.axis.imshow(crop, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        colour = "#35e06f" if bool(record["accepted"]) else "#ff4b4b"
        self.axis.axvline(local_x, color=colour, lw=0.8, alpha=0.8)
        self.axis.axhline(local_y, color=colour, lw=0.8, alpha=0.8)
        circle_radius = float(record.get("equivalent_diameter_px", 12.0) or 12.0) / 2.0
        if not math.isfinite(circle_radius):
            circle_radius = 6.0
        self.axis.add_patch(
            self.plt.Circle(
                (local_x, local_y), max(4.0, circle_radius), fill=False, color=colour, lw=2
            )
        )
        defect_id = record.get("defect_id", self.index + 1)
        rejection = record.get("rejection_reason", "")
        self.axis.set_title(
            f"候选 {self.index + 1}/{len(self.frame)} · ID={defect_id} · {self._status()}\n"
            f"自动拒绝原因: {rejection or '无'}"
        )
        self.axis.set_axis_off()
        self.figure.suptitle(
            "A=接受  R=拒绝  U=恢复自动结果  ←/→或空格=浏览  S=保存  Q=保存并退出",
            fontsize=10,
        )
        self.figure.canvas.draw_idle()

    def _set_decision(self, accepted: bool, action: str) -> None:
        self.frame.at[self.index, "accepted"] = accepted
        self.frame.at[self.index, "review_action"] = action
        self._saved = False

    def _save(self) -> None:
        metrics = save_reviewed_results(self.frame, self.output_csv, self.summary)
        self._saved = True
        print(
            f"已保存 {self.output_csv}\n"
            f"复核后 n={metrics['accepted_count']}, S={metrics['valid_analysis_area_cm2']:.8g} cm^2, "
            f"rho={metrics['point_density_cm2']:.8g} ± {metrics['counting_uncertainty_cm2']:.8g} cm^-2"
        )

    def _on_key(self, event: Any) -> None:
        key = (event.key or "").lower()
        if key in {"a", "y", "1"}:
            self._set_decision(True, "manual_accept")
            self.index = min(len(self.frame) - 1, self.index + 1)
        elif key in {"r", "n", "0"}:
            self._set_decision(False, "manual_reject")
            self.index = min(len(self.frame) - 1, self.index + 1)
        elif key in {"u", "backspace"}:
            original = bool(self.frame.at[self.index, "auto_accepted"])
            self._set_decision(original, "restored_auto")
        elif key in {"right", " ", "space", "enter"}:
            self.index = min(len(self.frame) - 1, self.index + 1)
        elif key == "left":
            self.index = max(0, self.index - 1)
        elif key == "home":
            self.index = 0
        elif key == "end":
            self.index = len(self.frame) - 1
        elif key == "s":
            self._save()
        elif key in {"q", "escape"}:
            self._save()
            self.plt.close(self.figure)
            return
        else:
            return
        self._draw()

    def _on_close(self, _event: Any) -> None:
        if not self._saved:
            self._save()

    def run(self) -> None:
        self._draw()
        self.plt.show()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="逐个复核点状目标候选并重算 n/S。")
    parser.add_argument("original_image", type=Path, help="本次分析使用的原始图像")
    parser.add_argument("candidates_csv", type=Path, help="通常为 results/.../defects_all.csv")
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="summary.json；默认与候选 CSV 同目录",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 reviewed_defects.csv；默认与候选 CSV 同目录",
    )
    parser.add_argument("--crop-half-size", type=int, default=64, help="局部窗口半边长（像素）")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary_path = args.summary or args.candidates_csv.with_name("summary.json")
    output_path = args.output or args.candidates_csv.with_name("reviewed_defects.csv")
    if not args.original_image.exists():
        raise FileNotFoundError(args.original_image)
    if not args.candidates_csv.exists():
        raise FileNotFoundError(args.candidates_csv)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    frame = pd.read_csv(args.candidates_csv)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    # Validate the area before opening a window; no silent theoretical-area fallback.
    valid_area_from_summary(summary)
    reviewer = CandidateReviewer(
        args.original_image,
        frame,
        summary,
        output_path,
        crop_half_size=args.crop_half_size,
    )
    reviewer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
