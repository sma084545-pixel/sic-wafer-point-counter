"""Command-line interface for the SiC wafer point counter."""

from __future__ import annotations

import argparse
import logging
import sys
from importlib import resources
from pathlib import Path
from typing import Sequence

from . import __version__
from .image_io import ImageReadError
from .pipeline import analyze_image
from .utils import ConfigurationError, deep_merge, load_config, setup_logging
from .wafer_detection import WaferDetectionError


LOGGER = logging.getLogger(__name__)
# Kept as a package resource so an installed wheel works from any directory;
# the editable repository copy remains at config/default.yaml for users to edit.
DEFAULT_CONFIG_PATH = Path(resources.files("sic_wafer_counter").joinpath("resources/default.yaml"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sic-wafer-counter",
        description=(
            "使用可复核的传统计算机视觉规则统计 SiC 晶圆图像中的黑色点状目标，"
            "并按最终有效像素面积计算 cm^-2 密度。"
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser(
        "analyze", help="分析一张晶圆图像并创建完整结果目录"
    )
    analyze.add_argument("input_image", type=Path, help="PNG/JPG/BMP/TIFF/BigTIFF 输入图像")
    analyze.add_argument("--output", type=Path, required=True, help="本次独立输出目录")
    analyze.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML 参数文件（默认：{DEFAULT_CONFIG_PATH}）",
    )
    analyze.add_argument("--wafer-diameter-mm", type=float, default=None, help="晶圆实际直径，默认读取配置（100 mm）")
    analyze.add_argument("--center-x", type=float, default=None, help="手工圆心 x（原图像素）")
    analyze.add_argument("--center-y", type=float, default=None, help="手工圆心 y（原图像素）")
    analyze.add_argument("--radius-px", type=float, default=None, help="手工半径（原图像素）")
    analyze.add_argument("--exclude-edge-mm", type=float, default=None, help="从真实晶圆轮廓向内排除的宽度，默认0 mm")
    analyze.add_argument(
        "--threshold-method",
        choices=("otsu", "adaptive", "quantile"),
        default=None,
        help="候选响应阈值方法",
    )
    analyze.add_argument(
        "--save-intermediates",
        action="store_true",
        help="强制保存预处理响应和候选掩膜",
    )
    analyze.add_argument("--no-watershed", action="store_true", help="关闭粘连点分水岭分离")
    analyze.add_argument("--verbose", action="store_true", help="输出调试级日志")

    web = subparsers.add_parser("web", help="启动本机浏览器分析工作台")
    web.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="上传文件与 results/ 输出的本机工作目录（默认当前目录）",
    )
    web.add_argument("--host", default="127.0.0.1", help="仅允许 loopback 地址，默认 127.0.0.1")
    web.add_argument("--port", type=int, default=8765, help="本机浏览器页面端口，默认 8765")
    web.add_argument("--max-workers", type=int, default=1, help="并行分析任务数，默认 1 以保护大图内存")
    web.add_argument("--max-upload-mb", type=int, default=4096, help="单个图像最大上传大小（MB）")
    return parser


def _effective_config(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    overrides: dict = {}
    if args.wafer_diameter_mm is not None:
        overrides.setdefault("wafer", {})["diameter_mm"] = args.wafer_diameter_mm
    if args.exclude_edge_mm is not None:
        overrides.setdefault("wafer", {})["exclude_edge_mm"] = args.exclude_edge_mm
    if args.threshold_method is not None:
        overrides.setdefault("detection", {})["threshold_method"] = args.threshold_method
    if args.save_intermediates:
        overrides.setdefault("output", {})["save_intermediates"] = True
    if args.no_watershed:
        overrides.setdefault("detection", {})["use_watershed"] = False
    if args.verbose:
        overrides.setdefault("logging", {})["level"] = "DEBUG"
    return deep_merge(config, overrides)


def _run_analyze(args: argparse.Namespace) -> int:
    manual = (args.center_x, args.center_y, args.radius_px)
    if any(value is not None for value in manual) and not all(
        value is not None for value in manual
    ):
        raise ConfigurationError(
            "手工标定必须同时提供 --center-x、--center-y 和 --radius-px。"
        )
    config = _effective_config(args)
    log_config = config.get("logging", {})
    setup_logging(
        args.output,
        level=log_config.get("level", "INFO"),
        verbose=args.verbose,
        logger_name="sic_wafer_counter",
    )
    result = analyze_image(
        args.input_image,
        args.output,
        config,
        center_x=args.center_x,
        center_y=args.center_y,
        radius_px=args.radius_px,
    )
    summary = result.summary
    print(f"Input image: {Path(summary['input_path'])}")
    print(
        "Detected wafer diameter: "
        f"{summary['detected_wafer_diameter_px']:.3f} px "
        f"(calibrated as {summary['wafer_diameter_mm']:.6g} mm)"
    )
    print(f"Pixel scale: {summary['mm_per_pixel']:.9g} mm/px")
    print(f"Valid area: {summary['valid_analysis_area_cm2']:.8g} cm^2")
    print(f"Accepted point count: {summary['accepted_count']}")
    print(f"Point-like target density: {summary['point_density_cm2']:.8g} cm^-2")
    print(f"Counting uncertainty: ± {summary['counting_uncertainty_cm2']:.8g} cm^-2")
    print(f"Output directory: {args.output.expanduser().resolve()}")
    if summary.get("warnings"):
        print("Warnings:")
        for warning in summary["warnings"]:
            print(f"  - {warning}")
    return 0


def _run_web(args: argparse.Namespace) -> int:
    """Start the local-only browser workbench without changing CLI analysis."""

    from .web import run_local_server

    if args.port < 1 or args.port > 65535:
        raise ConfigurationError("--port 必须在 1 到 65535 之间")
    print(f"Local SiC browser workbench: http://{args.host}:{args.port}")
    run_local_server(
        args.workspace,
        host=args.host,
        port=args.port,
        max_workers=args.max_workers,
        max_upload_mb=args.max_upload_mb,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, run the selected command, and return a process code."""

    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "analyze":
            return _run_analyze(args)
        if args.command == "web":
            return _run_web(args)
        parser.error(f"Unknown command: {args.command}")
    except WaferDetectionError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        print(
            "未输出密度。请查看输出目录中的 wafer_detection_preview.png，"
            "然后同时提供 --center-x、--center-y、--radius-px 重试。",
            file=sys.stderr,
        )
        return 2
    except (ConfigurationError, ImageReadError, ValueError, OSError) as error:
        LOGGER.exception("Analysis failed")
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
