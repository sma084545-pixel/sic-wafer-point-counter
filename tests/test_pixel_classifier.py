from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import cv2

from sic_wafer_counter.pixel_classifier import (
    BACKGROUND_LABEL,
    IGNORE_LABEL,
    PixelClassifierError,
    PixelTrainingSample,
    TARGET_LABEL,
    build_training_project,
    decode_label_mask_rle,
    encode_label_mask_rle,
    extract_pixel_features,
    feature_names,
    object_metrics,
    pixel_metrics,
    predict_pixel_probability,
    predict_pixel_probability_tiled,
    probability_to_mask,
    train_pixel_classifier,
    training_project_sha256,
    validate_pixel_model,
    validate_training_project,
)
from sic_wafer_counter.point_detection import detection_from_probability
from sic_wafer_counter.image_io import load_image
from sic_wafer_counter.pipeline import analyze_image


def _training_scene(seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    image = np.clip(0.74 + rng.normal(0.0, 0.018, (128, 144)), 0.0, 1.0).astype(np.float32)
    labels = np.zeros(image.shape, dtype=np.uint8)
    yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
    for y, x, radius in ((28, 34, 4), (65, 78, 5), (92, 116, 4), (35, 108, 3)):
        target = (yy - y) ** 2 + (xx - x) ** 2 <= radius**2
        image[target] -= np.float32(0.34)
        labels[target] = TARGET_LABEL
    # Explicit background/artefact strokes, including a dark line that is not target.
    labels[6:14, 8:132] = BACKGROUND_LABEL
    labels[110:121, 10:136] = BACKGROUND_LABEL
    image[112:116, 15:130] -= np.float32(0.18)
    labels[48:58, 8:18] = IGNORE_LABEL
    return np.clip(image, 0.0, 1.0), labels


def _sample(split: str = "calibration", wafer: str = "wafer-A") -> PixelTrainingSample:
    image, labels = _training_scene()
    digest = hashlib.sha256(image.tobytes()).hexdigest()
    return PixelTrainingSample(
        image=image,
        labels=labels,
        image_sha256=digest,
        image_name="training.tif",
        wafer_id=wafer,
        split=split,
        roi_xywh=(120, 240, image.shape[1], image.shape[0]),
    )


def _model() -> dict:
    return train_pixel_classifier(
        [_sample()],
        n_trees=10,
        max_depth=6,
        min_samples_leaf=5,
        maximum_pixels_per_class=800,
        bootstrap_pixels_per_class=200,
        random_seed=77,
    )


def test_multiscale_features_are_position_free_and_do_not_modify_source() -> None:
    image, _ = _training_scene()
    before = image.copy()
    features = extract_pixel_features(image)
    assert features.shape == (*image.shape, len(feature_names()))
    assert np.array_equal(image, before)
    assert np.isfinite(features).all()
    assert "intensity" in feature_names()
    assert not any("coordinate" in name for name in feature_names())


def test_balanced_fixed_seed_training_serialization_and_tamper_detection() -> None:
    first = _model()
    second = _model()
    assert first["trees"] == second["trees"]
    assert first["label_counts"] == second["label_counts"]
    assert first["training_parameters"]["absolute_coordinates_used"] is False
    assert first["label_counts"]["ignored_pixels"] > 0
    assert validate_pixel_model(json.loads(json.dumps(first)))["model_sha256"] == first["model_sha256"]
    tampered = json.loads(json.dumps(first))
    tampered["trees"][0][0]["probability"] = 0.123
    with pytest.raises(PixelClassifierError, match="SHA-256"):
        validate_pixel_model(tampered)
    missing_sources = json.loads(json.dumps(first))
    missing_sources.pop("model_sha256")
    missing_sources["training_sources"] = []
    with pytest.raises(PixelClassifierError, match="训练图像来源"):
        validate_pixel_model(missing_sources)


def test_model_and_project_hash_survive_browser_json_number_semantics() -> None:
    def browser_numbers(value):
        if isinstance(value, dict):
            return {key: browser_numbers(item) for key, item in value.items()}
        if isinstance(value, list):
            return [browser_numbers(item) for item in value]
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value

    model = _model()
    browser_model = browser_numbers(json.loads(json.dumps(model)))
    assert validate_pixel_model(browser_model)["model_sha256"] == model["model_sha256"]
    image, labels = _training_scene()
    project = build_training_project(
        image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
        image_name="source.tif",
        source_shape=image.shape,
        wafer_id="wafer-A",
        split="calibration",
        annotations=[
            {
                "annotation_id": "roi-1",
                "roi_xywh": [0, 0, image.shape[1], image.shape[0]],
                "labels": encode_label_mask_rle(labels),
            }
        ],
        model=model,
    )
    assert training_project_sha256(browser_numbers(project)) == project["project_sha256"]
    assert validate_training_project(browser_numbers(project))["project_sha256"] == project[
        "project_sha256"
    ]


def test_uncertain_pixels_are_not_used_as_negative_training_labels() -> None:
    model = _model()
    image, labels = _training_scene()
    probability = predict_pixel_probability(image, model)
    metrics = pixel_metrics(labels, probability, threshold=0.5)
    assert metrics["ignored_pixel_count"] == int(np.sum(labels == IGNORE_LABEL))
    assert metrics["evaluated_pixel_count"] == int(
        np.sum((labels == TARGET_LABEL) | (labels == BACKGROUND_LABEL))
    )


def test_calibration_classes_may_be_painted_in_separate_rois() -> None:
    image, labels = _training_scene()
    digest = hashlib.sha256(image.tobytes()).hexdigest()
    target_only = labels.copy()
    target_only[target_only != TARGET_LABEL] = 0
    background_only = labels.copy()
    background_only[background_only != BACKGROUND_LABEL] = 0
    common = {
        "image": image,
        "image_sha256": digest,
        "image_name": "training.tif",
        "wafer_id": "wafer-A",
        "split": "calibration",
        "roi_xywh": (0, 0, image.shape[1], image.shape[0]),
    }
    model = train_pixel_classifier(
        [
            PixelTrainingSample(labels=target_only, **common),
            PixelTrainingSample(labels=background_only, **common),
        ],
        n_trees=4,
        max_depth=5,
        bootstrap_pixels_per_class=60,
        maximum_pixels_per_class=300,
        random_seed=19,
    )
    assert model["label_counts"]["sampled_per_class"] == min(
        model["label_counts"]["labelled_target_pixels"],
        model["label_counts"]["labelled_background_pixels"],
        300,
    )
    assert model["training_parameters"]["random_seed"] == 19


def test_tiled_probability_matches_full_frame_without_seams() -> None:
    model = _model()
    image, _ = _training_scene()
    full = predict_pixel_probability(image, model)
    tiled = predict_pixel_probability_tiled(image, model, tile_size=48)
    assert np.allclose(tiled, full, atol=1e-7)


def test_probability_to_detection_respects_valid_mask_and_area() -> None:
    probability = np.zeros((64, 64), dtype=np.float32)
    probability[10:14, 10:14] = 0.95
    probability[30, 30] = 0.99
    probability[50:55, 50:55] = 0.98
    valid = np.ones_like(probability, dtype=bool)
    valid[48:, 48:] = False
    detection = detection_from_probability(
        probability,
        valid,
        {"detection": {"min_area_px": 4, "max_area_px": 500, "use_watershed": False}},
        probability_threshold=0.7,
        minimum_object_area_px=4,
    )
    assert detection.post_watershed_count == 1
    assert detection.labels[11, 11] > 0
    assert detection.labels[30, 30] == 0
    assert detection.labels[52, 52] == 0


def test_label_rle_and_project_round_trip_are_lossless() -> None:
    image, labels = _training_scene()
    encoded = encode_label_mask_rle(labels)
    assert np.array_equal(decode_label_mask_rle(encoded), labels)
    digest = hashlib.sha256(image.tobytes()).hexdigest()
    project = build_training_project(
        image_sha256=digest,
        image_name="source.tif",
        source_shape=image.shape,
        wafer_id="wafer-A",
        split="calibration",
        annotations=[{"annotation_id": "roi-1", "roi_xywh": [0, 0, image.shape[1], image.shape[0]], "labels": encoded}],
        model=_model(),
        operation_history=[{"action": "paint", "class": "target"}],
    )
    restored = validate_training_project(json.loads(json.dumps(project)))
    assert restored["project_sha256"] == project["project_sha256"]
    restored_labels = decode_label_mask_rle(restored["annotations"][0]["labels"])
    assert np.array_equal(restored_labels, labels)


def test_wafer_split_leakage_is_rejected() -> None:
    with pytest.raises(PixelClassifierError, match="不能跨"):
        train_pixel_classifier(
            [_sample("calibration", "same-wafer"), _sample("validation", "same-wafer")],
            n_trees=2,
            bootstrap_pixels_per_class=40,
        )
    same_image_different_wafer = _sample("validation", "wafer-B")
    with pytest.raises(PixelClassifierError, match="image:"):
        train_pixel_classifier(
            [_sample("calibration", "wafer-A"), same_image_different_wafer],
            n_trees=2,
            bootstrap_pixels_per_class=40,
        )


def test_synthetic_end_to_end_pixel_training_probability_objects_and_metrics() -> None:
    image, labels = _training_scene()
    model = _model()
    probability = predict_pixel_probability(image, model)
    prediction = probability_to_mask(
        probability,
        threshold=0.5,
        minimum_object_area_px=5,
    )
    pixels = pixel_metrics(labels, probability, threshold=0.5)
    objects = object_metrics(labels, prediction, matching_tolerance_px=5)
    assert pixels["precision"] is not None and pixels["precision"] >= 0.95
    assert pixels["recall"] is not None and pixels["recall"] >= 0.95
    assert objects["true_positive"] == 4
    assert objects["false_negative"] == 0
    assert objects["mean_localization_error_px"] is not None


def test_pixel_model_is_used_by_main_pipeline_and_preserves_audit_outputs(
    tmp_path,
    default_config: dict,
) -> None:
    image = np.full((321, 321), 28, dtype=np.uint8)
    cv2.circle(image, (160, 160), 145, 205, thickness=-1)
    labels = np.zeros(image.shape, dtype=np.uint8)
    for x, y in ((105, 105), (210, 110), (112, 215), (215, 212)):
        cv2.circle(image, (x, y), 5, 42, thickness=-1)
        cv2.circle(labels, (x, y), 5, TARGET_LABEL, thickness=-1)
    cv2.line(image, (75, 160), (245, 160), 52, thickness=3)
    cv2.line(labels, (75, 160), (245, 160), BACKGROUND_LABEL, thickness=5)
    labels[40:52, 72:248] = BACKGROUND_LABEL
    input_path = tmp_path / "pixel_pipeline.png"
    assert cv2.imwrite(str(input_path), image)
    with load_image(input_path, default_config) as loaded:
        normalized = loaded.require_full().copy()
    sample = PixelTrainingSample(
        normalized,
        labels,
        hashlib.sha256(input_path.read_bytes()).hexdigest(),
        input_path.name,
        "synthetic-pixel-wafer",
        "calibration",
        (0, 0, 321, 321),
    )
    model = train_pixel_classifier(
        [sample], n_trees=12, max_depth=7, bootstrap_pixels_per_class=300,
        maximum_pixels_per_class=1000, random_seed=31,
    )
    config = json.loads(json.dumps(default_config))
    config["pixel_classifier"] = {"enabled": True, "model": model}
    config["output"]["generate_local_field_package"] = False
    result = analyze_image(
        input_path,
        tmp_path / "pixel_result",
        config,
        center_x=160,
        center_y=160,
        radius_px=145,
    )
    assert result.summary["pixel_classifier"]["status"] == "applied"
    assert result.summary["pixel_classifier"]["model_sha256"] == model["model_sha256"]
    assert result.summary["decision_basis"] == "pixel_segmentation_then_configured_image_rules"
    assert result.summary["method_comparison"]["traditional_candidate_detector"]["status"] == "not_run_counterfactually"
    assert result.summary["method_comparison"]["final_accepted_count"] == result.summary["accepted_count"]
    assert (tmp_path / "pixel_result" / "pixel_classifier.json").is_file()
    assert (tmp_path / "pixel_result" / "pixel_target_probability.png").is_file()
    assert result.defects["pixel_model_applied"].fillna(False).all()
    assert result.defects["pixel_model_sha256"].eq(model["model_sha256"]).all()
    assert result.defects["rule_accepted"].dtype == bool
    assert result.defects["classifier_applied"].eq(False).all()
    assert result.defects["decision_basis"].eq(
        "pixel_segmentation_then_configured_image_rules"
    ).all()
