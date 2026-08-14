from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

POSITION_ID = "annotation.reset.initial_pose.position_id"
GRID_COL = "annotation.reset.initial_pose.grid_col"
GRID_ROW = "annotation.reset.initial_pose.grid_row"
GRID_X_NORM = "annotation.reset.initial_pose.x_norm"
GRID_Y_NORM = "annotation.reset.initial_pose.y_norm"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResetGridSettings:
    camera_name: str
    camera_serial: str
    camera_fps: int
    image_width: int
    image_height: int
    columns: int
    rows: int
    corners: list[list[float]]
    initial_poses: dict[str, list[int] | list[list[int]]]


def load_reset_grid_settings(path: str | Path) -> ResetGridSettings:
    """Load the shared egoview/reset-grid settings used by record and teleop."""
    config_path = Path(path).expanduser()
    try:
        data = yaml.safe_load(config_path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"Reset-grid config does not exist: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid reset-grid YAML in {config_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Reset-grid config must contain a YAML mapping: {config_path}")

    cameras = data.get("cameras", {})
    capture = data.get("capture", {})
    reset_grid = data.get("reset_grid", {})
    if not all(isinstance(section, dict) for section in (cameras, capture, reset_grid)):
        raise ValueError("cameras, capture, and reset_grid must be YAML mappings")
    if not reset_grid.get("enabled", False):
        raise ValueError(f"reset_grid.enabled is false in {config_path}")

    camera_name = str(reset_grid.get("camera", "egoview"))
    if camera_name not in cameras:
        raise ValueError(f"Reset-grid camera {camera_name!r} is missing from {config_path}")

    return ResetGridSettings(
        camera_name=camera_name,
        camera_serial=str(cameras[camera_name]),
        camera_fps=int(capture.get("dataset_fps", 30)),
        image_width=int(capture.get("width", 640)),
        image_height=int(capture.get("height", 480)),
        columns=int(reset_grid.get("columns", 16)),
        rows=int(reset_grid.get("rows", 12)),
        corners=reset_grid.get(
            "corners",
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        ),
        initial_poses=reset_grid.get("initial_poses", {}),
    )


@dataclass(frozen=True)
class GridPoint:
    col: int
    row: int
    position_id: int
    x_norm: float
    y_norm: float


@dataclass(frozen=True)
class InitialPose:
    object_name: str
    points: tuple[GridPoint, ...]

    @property
    def is_box(self) -> bool:
        return len(self.points) == 4

    @property
    def label(self) -> str:
        if not self.is_box:
            point = self.points[0]
            return f"{self.object_name} · C{point.col:02d}-R{point.row:02d}"
        top_left, _, bottom_right, _ = self.points
        return (
            f"{self.object_name} · BOX "
            f"C{top_left.col:02d}-R{top_left.row:02d}.."
            f"C{bottom_right.col:02d}-R{bottom_right.row:02d}"
        )


class ResetGridGuide:
    """Fixed per-object reset poses, dense annotations, and a Rerun overlay."""

    def __init__(
        self,
        *,
        camera_name: str,
        image_width: int,
        image_height: int,
        columns: int = 16,
        rows: int = 12,
        corners: list[list[float]] | tuple[tuple[float, float], ...],
        initial_poses: dict[str, list[int] | list[list[int]]] | None = None,
        rerun_enabled: bool = True,
    ) -> None:
        if columns < 2 or rows < 2:
            raise ValueError("reset_grid columns and rows must both be at least 2")
        if image_width < 1 or image_height < 1:
            raise ValueError("reset_grid image dimensions must be positive")
        if len(corners) != 4 or any(len(point) != 2 for point in corners):
            raise ValueError(
                "reset_grid corners must contain four [x, y] points in TL, TR, BR, BL order"
            )

        normalized_corners = np.asarray(corners, dtype=np.float64)
        if not np.isfinite(normalized_corners).all() or not (
            (normalized_corners >= 0).all() and (normalized_corners <= 1).all()
        ):
            raise ValueError("reset_grid corner coordinates must be finite values from 0 to 1")

        self.camera_name = camera_name
        self.image_width = image_width
        self.image_height = image_height
        self.columns = columns
        self.rows = rows
        self.rerun_enabled = rerun_enabled
        self._entity_path = f"observation.{camera_name}/reset_grid"

        pixel_corners = normalized_corners * np.asarray(
            [image_width - 1, image_height - 1], dtype=np.float64
        )
        self._homography = _homography_from_unit_square(pixel_corners)

        if initial_poses is None:
            initial_poses = {}
        if not isinstance(initial_poses, dict):
            raise ValueError("reset_grid initial_poses must be a YAML mapping")

        self._initial_poses: list[InitialPose] = []
        for object_name, coordinates in initial_poses.items():
            if not isinstance(object_name, str) or not object_name.strip():
                raise ValueError("reset_grid initial_poses object names must be non-empty strings")

            raw_points: list[list[int] | tuple[int, int]]
            if _is_grid_coordinate(coordinates):
                raw_points = [coordinates]
            elif (
                isinstance(coordinates, (list, tuple))
                and len(coordinates) == 4
                and all(_is_grid_coordinate(point) for point in coordinates)
            ):
                raw_points = list(coordinates)
            else:
                raise ValueError(
                    f"reset_grid initial pose for {object_name!r} must be either "
                    "[column, row] or four [[column, row], ...] box corners"
                )

            for col, row in raw_points:
                if not (0 <= col < columns and 0 <= row < rows):
                    raise ValueError(
                        f"reset_grid initial pose for {object_name!r} is outside "
                        f"the {columns}x{rows} grid: [{col}, {row}]"
                    )
            if len(raw_points) == 4:
                raw_points = _order_box_corners(object_name, raw_points)

            self._initial_poses.append(
                InitialPose(
                    object_name=object_name,
                    points=tuple(
                        GridPoint(
                            col=col,
                            row=row,
                            position_id=row * columns + col,
                            x_norm=col / (columns - 1),
                            y_norm=row / (rows - 1),
                        )
                        for col, row in raw_points
                    ),
                )
            )

        self._current_phase: str | None = None
        self._logged_overlay: str | None = None

    @property
    def dataset_features(self) -> dict[str, dict[str, Any]]:
        if not self._initial_poses:
            return {}
        point_names = self._annotation_names()
        shape = (len(point_names),)
        return {
            POSITION_ID: {"dtype": "int64", "shape": shape, "names": point_names},
            GRID_COL: {"dtype": "int64", "shape": shape, "names": point_names},
            GRID_ROW: {"dtype": "int64", "shape": shape, "names": point_names},
            GRID_X_NORM: {"dtype": "float32", "shape": shape, "names": point_names},
            GRID_Y_NORM: {"dtype": "float32", "shape": shape, "names": point_names},
        }

    def _annotation_names(self) -> list[str]:
        names: list[str] = []
        for pose in self._initial_poses:
            if pose.is_box:
                names.extend(
                    f"{pose.object_name}.corner_{index}"
                    for index in range(1, len(pose.points) + 1)
                )
            else:
                names.append(pose.object_name)
        return names

    def _annotation_points(self) -> list[GridPoint]:
        return [point for pose in self._initial_poses for point in pose.points]

    @property
    def num_positions(self) -> int:
        return len(self._initial_poses)

    @property
    def initial_poses(self) -> tuple[InitialPose, ...]:
        return tuple(self._initial_poses)

    def annotations_for_episode(self, episode_index: int) -> dict[str, np.ndarray]:
        if episode_index < 0:
            raise ValueError("episode_index must be non-negative")
        if not self._initial_poses:
            return {}
        points = self._annotation_points()
        return {
            POSITION_ID: np.asarray(
                [point.position_id for point in points], dtype=np.int64
            ),
            GRID_COL: np.asarray([point.col for point in points], dtype=np.int64),
            GRID_ROW: np.asarray([point.row for point in points], dtype=np.int64),
            GRID_X_NORM: np.asarray(
                [point.x_norm for point in points], dtype=np.float32
            ),
            GRID_Y_NORM: np.asarray(
                [point.y_norm for point in points], dtype=np.float32
            ),
        }

    def on_phase(self, episode_index: int, phase: str) -> None:
        if phase not in {"reset", "recording"}:
            raise ValueError(f"Unknown reset-grid phase: {phase}")
        if episode_index < 0:
            raise ValueError("episode_index must be non-negative")
        self._current_phase = phase
        self._logged_overlay = None
        if phase == "reset" and self._initial_poses:
            logger.info(
                "INITIAL POSES  Episode %d · %s",
                episode_index + 1,
                " · ".join(pose.label for pose in self._initial_poses),
            )

    def log_visualization(self) -> None:
        """Log the overlay after the camera image and blueprint exist."""
        if not self.rerun_enabled or self._current_phase is None:
            return
        if self._current_phase == self._logged_overlay:
            return
        self._log_rerun(phase=self._current_phase)

        # This runs once per RESET/RECORDING transition. A bounded flush makes
        # startup deterministic without adding work to subsequent preview frames.
        import rerun as rr

        recording = rr.get_global_data_recording()
        if recording is not None:
            recording.flush(timeout_sec=2.0)
        self._logged_overlay = self._current_phase

    def _project(self, x_norm: float, y_norm: float) -> list[float]:
        point = self._homography @ np.asarray([x_norm, y_norm, 1.0])
        return [float(point[0] / point[2]), float(point[1] / point[2])]

    def _log_rerun(self, *, phase: str) -> None:
        import rerun as rr

        vertical = [
            [self._project(col / (self.columns - 1), 0), self._project(col / (self.columns - 1), 1)]
            for col in range(self.columns)
        ]
        horizontal = [
            [self._project(0, row / (self.rows - 1)), self._project(1, row / (self.rows - 1))]
            for row in range(self.rows)
        ]
        rr.log(
            f"{self._entity_path}/lines",
            rr.LineStrips2D(
                vertical + horizontal,
                radii=1.0,
                colors=[70, 180, 255, 150],
                draw_order=10,
            ),
            static=True,
        )

        if self._initial_poses:
            target_color = (
                [50, 255, 100, 255] if phase == "reset" else [255, 190, 40, 255]
            )
            point_poses = [pose for pose in self._initial_poses if not pose.is_box]
            box_poses = [pose for pose in self._initial_poses if pose.is_box]
            if point_poses:
                rr.log(
                    f"{self._entity_path}/initial_pose_points",
                    rr.Points2D(
                        [
                            self._project(pose.points[0].x_norm, pose.points[0].y_norm)
                            for pose in point_poses
                        ],
                        radii=11.0,
                        colors=[target_color] * len(point_poses),
                        labels=[pose.label for pose in point_poses],
                        show_labels=True,
                        draw_order=30,
                    ),
                    static=True,
                )
            if box_poses:
                strips = []
                for pose in box_poses:
                    projected = [
                        self._project(point.x_norm, point.y_norm) for point in pose.points
                    ]
                    strips.append(projected + [projected[0]])
                rr.log(
                    f"{self._entity_path}/initial_pose_boxes",
                    rr.LineStrips2D(
                        strips,
                        radii=4.0,
                        colors=[target_color] * len(box_poses),
                        labels=[pose.label for pose in box_poses],
                        show_labels=True,
                        draw_order=31,
                    ),
                    static=True,
                )


def _is_grid_coordinate(value: object) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    )


def _order_box_corners(
    object_name: str,
    points: list[list[int] | tuple[int, int]],
) -> list[tuple[int, int]]:
    columns = {point[0] for point in points}
    rows = {point[1] for point in points}
    if len(columns) != 2 or len(rows) != 2:
        raise ValueError(
            f"reset_grid box for {object_name!r} must contain four distinct "
            "axis-aligned corners"
        )
    left, right = sorted(columns)
    top, bottom = sorted(rows)
    ordered = [(left, top), (right, top), (right, bottom), (left, bottom)]
    if set(map(tuple, points)) != set(ordered):
        raise ValueError(
            f"reset_grid box for {object_name!r} must contain each of the four "
            "axis-aligned corners exactly once"
        )
    return ordered


def _homography_from_unit_square(destination: np.ndarray) -> np.ndarray:
    source = np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.float64)
    matrix: list[list[float]] = []
    values: list[float] = []
    for (x, y), (u, v) in zip(source, destination, strict=True):
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        values.append(u)
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        values.append(v)
    try:
        coefficients = np.linalg.solve(np.asarray(matrix), np.asarray(values))
    except np.linalg.LinAlgError as exc:
        raise ValueError("reset_grid corners form a degenerate quadrilateral") from exc
    return np.append(coefficients, 1.0).reshape(3, 3)
