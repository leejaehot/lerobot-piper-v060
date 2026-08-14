from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path
from threading import Event
from typing import Any

from lerobot_piper.console_ui import phase
from lerobot_piper.record_cli import DEFAULT_CONFIG, _load, _section
from lerobot_piper.reset_grid import ResetGridGuide


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="piper_grid_preview",
        description="Preview the egoview reset grid in Rerun without connecting either arm.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="recording YAML")
    parser.add_argument("--camera", help="camera key from the YAML cameras mapping")
    parser.add_argument("--fps", type=float, help="Rerun preview refresh rate")
    parser.add_argument("--debug", action="store_true", help="show a traceback on failure")
    return parser.parse_args()


def _settings(data: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cameras = _section(data, "cameras")
    capture = _section(data, "capture")
    reset_grid = _section(data, "reset_grid")
    camera_name = args.camera or reset_grid.get("camera", "egoview")
    if camera_name not in cameras:
        raise ValueError(f"Camera {camera_name!r} is not present in the YAML cameras mapping")

    fps = float(args.fps if args.fps is not None else capture.get("rerun_fps", 10))
    if fps <= 0:
        raise ValueError("--fps must be positive")

    return {
        "camera_name": str(camera_name),
        "serial": str(cameras[camera_name]),
        "camera_fps": int(capture.get("dataset_fps", 30)),
        "preview_fps": fps,
        "width": int(capture.get("width", 640)),
        "height": int(capture.get("height", 480)),
        "columns": int(reset_grid.get("columns", 16)),
        "rows": int(reset_grid.get("rows", 12)),
        "corners": reset_grid.get(
            "corners",
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        ),
        "initial_poses": reset_grid.get("initial_poses", {}),
    }


def main() -> None:
    args = _arguments()
    camera = None
    listener = None
    visualization_started = False
    try:
        cfg = _settings(_load(args.config), args)

        from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
        from lerobot.utils.keyboard_input import create_key_listener
        from lerobot.utils.visualization_utils import (
            init_visualization,
            log_visualization_data,
            shutdown_visualization,
        )

        guide = ResetGridGuide(
            camera_name=cfg["camera_name"],
            image_width=cfg["width"],
            image_height=cfg["height"],
            columns=cfg["columns"],
            rows=cfg["rows"],
            corners=cfg["corners"],
            initial_poses=cfg["initial_poses"],
            rerun_enabled=True,
        )

        camera = RealSenseCamera(
            RealSenseCameraConfig(
                serial_number_or_name=cfg["serial"],
                fps=cfg["camera_fps"],
                width=cfg["width"],
                height=cfg["height"],
                use_rgb=True,
                use_depth=False,
            )
        )

        phase(
            "GRID",
            (
                f"{cfg['camera_name']} {cfg['width']}×{cfg['height']} · "
                f"{cfg['columns']}×{cfg['rows']} grid · {guide.num_positions} fixed object pose(s) · "
                "arms/CAN remain disconnected"
            ),
        )
        phase("CAMERA", f"Connecting RealSense {cfg['serial']}")
        camera.connect()
        init_visualization("rerun", session_name="piper_grid_preview")
        visualization_started = True

        stop = Event()

        def on_key(name: str) -> None:
            if name.lower() in {"esc", "q"}:
                stop.set()

        listener = create_key_listener(
            on_key,
            controls_help="q/Esc=quit",
        )
        guide.on_phase(0, "reset")
        initial_pose_text = " · ".join(pose.label for pose in guide.initial_poses)
        phase("INITIAL", initial_pose_text or "No object initial poses configured")
        phase(
            "READY",
            f"Rerun camera-only view at {cfg['preview_fps']:g} Hz · q quits",
            "green",
        )

        interval_s = 1.0 / cfg["preview_fps"]
        while not stop.is_set():
            started = time.perf_counter()

            frame = camera.read_latest(max_age_ms=200)
            log_visualization_data(
                "rerun",
                observation={cfg["camera_name"]: frame},
                compress_images=False,
            )
            guide.log_visualization()
            time.sleep(max(interval_s - (time.perf_counter() - started), 0.0))
    except KeyboardInterrupt:
        phase("STOPPED", "Grid preview interrupted", "yellow")
    except Exception as exc:
        phase("ERROR", str(exc), "red", stream=sys.stderr)
        if args.debug:
            traceback.print_exc()
        raise SystemExit(1) from None
    finally:
        if listener is not None:
            listener.stop()
        if camera is not None and camera.is_connected:
            camera.disconnect()
        if visualization_started:
            shutdown_visualization("rerun")


if __name__ == "__main__":
    main()
