#!/usr/bin/env python3
"""Validate saved automatic SiC point detections against expert annotation CSVs.

This script does not tune the detector. It reads already-produced result
folders, preserves raw reviewer decisions, and refuses to turn unclear labels
into negatives. With no annotation input it writes only a template and an
honest ``not validated on real SiC data`` status.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from sic_wafer_counter.validation import (  # noqa: E402
    annotation_provenance_status,
    arbitrate_annotations,
    assign_wafer_splits,
    bootstrap_classification_bias,
    evaluate_automatic_detections,
    load_annotations,
    regional_validation_metrics,
    reviewer_agreement,
    source_image_sha256,
    write_unvalidated_bundle,
    write_validation_outputs,
)


def _comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_result_folders(paths: list[str]) -> tuple[pd.DataFrame, dict[str, float], dict[str, float], dict[str, str | None]]:
    frames: list[pd.DataFrame] = []
    area_by_wafer: dict[str, float] = {}
    scale_by_image: dict[str, float] = {}
    source_hash_by_image: dict[str, str | None] = {}
    for value in paths:
        folder = Path(value).expanduser().resolve()
        summary_path, defects_path = folder / "summary.json", folder / "defects_accepted.csv"
        if not summary_path.is_file() or not defects_path.is_file():
            raise ValueError(f"Result folder needs summary.json and defects_accepted.csv: {folder}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        image_id = str(summary.get("input_file_name", folder.name))
        wafer_id = str(summary.get("wafer_id", folder.name))
        frame = pd.read_csv(defects_path)
        frame["image_id"] = image_id
        frame["wafer_id"] = wafer_id
        frame["accepted"] = True
        frames.append(frame)
        area_by_wafer[wafer_id] = float(summary["valid_analysis_area_cm2"])
        scale_by_image[image_id] = float(summary["mm_per_pixel"])
        source_path = Path(str(summary.get("input_path", ""))).expanduser()
        source_hash_by_image[image_id] = source_image_sha256(source_path) if source_path.is_file() else None
    return (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(),
        area_by_wafer,
        scale_by_image,
        source_hash_by_image,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", nargs="*", default=[], help="Versioned expert-annotation CSV file(s)")
    parser.add_argument("--result-dir", action="append", default=[], help="Completed result directory; repeat for multiple wafers")
    parser.add_argument("--output", required=True, help="Validation output directory")
    parser.add_argument("--matching-tolerance-um", type=float, default=25.0)
    parser.add_argument("--annotation-coverage-complete", action="store_true", help="Declare unmatched automatic points auditable false positives")
    parser.add_argument("--calibration-wafers", default="", help="Comma-separated wafer_id values")
    parser.add_argument("--validation-wafers", default="", help="Comma-separated wafer_id values")
    parser.add_argument("--locked-test-wafers", default="", help="Comma-separated wafer_id values")
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    if not args.annotations:
        paths = write_unvalidated_bundle(output)
        print(f"No real annotations supplied. Wrote template/status: {paths['validation_summary']}")
        return 0
    if not args.result_dir:
        parser.error("--result-dir is required when --annotations are supplied")

    raw = load_annotations(args.annotations)
    agreement, disagreements = reviewer_agreement(raw)
    adjudicated = arbitrate_annotations(raw)
    automatic, areas, scales, source_hashes = _load_result_folders(args.result_dir)
    provenance = annotation_provenance_status(raw, source_hashes)
    if provenance["status"].isin(["hash_mismatch", "inconsistent_annotation_hashes"]).any():
        raise ValueError("Annotation provenance failed; source-image hash mismatch or inconsistency")
    known_wafers = sorted(set(raw["wafer_id"].astype(str)) | set(automatic["wafer_id"].astype(str)))
    calibration = _comma_list(args.calibration_wafers)
    validation = _comma_list(args.validation_wafers)
    locked = _comma_list(args.locked_test_wafers)
    if calibration or validation or locked:
        splits = assign_wafer_splits(
            known_wafers,
            calibration_wafers=calibration,
            validation_wafers=validation,
            locked_test_wafers=locked,
        )
        split_status = "configured_by_wafer_id"
    else:
        splits = pd.DataFrame(columns=["wafer_id", "split"])
        split_status = "not configured; no parameter tuning was performed"

    evaluation = evaluate_automatic_detections(
        automatic,
        adjudicated,
        matching_tolerance_um=args.matching_tolerance_um,
        annotation_coverage_complete=args.annotation_coverage_complete,
        area_cm2_by_wafer=areas,
        mm_per_pixel_by_image=scales,
    )
    regional = regional_validation_metrics(evaluation.matches)
    bootstrap = bootstrap_classification_bias(evaluation.per_wafer, seed=args.bootstrap_seed)
    uncertainty = pd.DataFrame(
        [
            {"component": "counting_uncertainty", "status": "reported in each original analysis summary", "value": None},
            {"component": "classification_uncertainty", "status": bootstrap["status"], "value": json.dumps(bootstrap)},
            {"component": "parameter_sensitivity", "status": "not run: this validator never tunes parameters", "value": None},
            {"component": "area_calibration_uncertainty", "status": "not quantified: no diameter/pixel/mask uncertainty supplied", "value": None},
            {"component": "spatial_heterogeneity", "status": "not quantified: no valid-area block table supplied", "value": None},
        ]
    )
    sensitivity = pd.DataFrame(
        [{"status": "not run", "reason": "Use calibration wafers only; locked test wafers are never used to choose parameters."}]
    )
    summary = {
        **evaluation.metrics,
        "annotation_schema_version": "1.0",
        "annotation_count": int(len(raw)),
        "adjudicated_annotation_count": int(len(adjudicated)),
        "wafer_count": int(len(known_wafers)),
        "annotation_agreement": agreement,
        "split_status": split_status,
        "calibration_wafers": calibration,
        "validation_wafers": validation,
        "locked_test_wafers": locked,
        "classification_bootstrap": bootstrap,
        "annotation_provenance_statuses": provenance["status"].tolist(),
        "scientific_limit": "Agreement with image labels does not prove that each accepted black point is a physical dislocation.",
    }
    paths = write_validation_outputs(
        output,
        summary=summary,
        evaluation=evaluation,
        regional_metrics=regional,
        disagreements=disagreements,
        parameter_sensitivity=sensitivity,
        uncertainty_budget=uncertainty,
    )
    raw.to_csv(output / "annotations_raw.csv", index=False)
    adjudicated.to_csv(output / "annotations_arbitrated.csv", index=False)
    provenance.to_csv(output / "annotation_provenance.csv", index=False)
    splits.to_csv(output / "wafer_splits.csv", index=False)
    print(f"Validation status: {evaluation.metrics['validation_status']}")
    print(f"Output directory: {output}")
    print(f"Report: {paths['validation_report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
