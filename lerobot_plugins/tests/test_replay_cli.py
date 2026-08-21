from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from lerobot_piper import replay_cli


def _base_config(root: Path) -> dict:
    return {
        "replay": {
            "default_profile": "hyu_lemon",
            "episode": 0,
            "start_frame": 0,
            "rate": 1,
            "hold": True,
            "compress_images": True,
        },
        "arm": {
            "follower_can": "can_follower",
            "speed_percent": 20,
            "max_relative_target": 3,
            "gripper_speed_mm_s": 40,
            "align_start": True,
            "align_speed_percent": 20,
            "align_timeout_seconds": 20,
            "joint_tolerance_degrees": 2,
            "gripper_tolerance_mm": 2,
            "return_to_start": True,
            "wait_for_support": True,
        },
        "audio": {"enabled": False},
        "dataset": {
            "repo_id": "test/piper",
            "root": str(root),
        },
    }


def _replay_data(num_frames: int = 3) -> replay_cli.ReplayData:
    states = np.asarray(
        [[1, -2, 3, -4, 5, -6, 50], [2, -1, 4, -3, 6, -5, 60], [3, 0, 5, -2, 7, -4, 70]],
        dtype=np.float64,
    )[:num_frames]
    return replay_cli.ReplayData(
        dataset=SimpleNamespace(fps=30, num_frames=num_frames),
        actions=states + 0.5,
        states=states,
        action_names=replay_cli.MOTOR_NAMES,
        state_names=replay_cli.MOTOR_NAMES,
        image_keys=("observation.images.front", "observation.images.right"),
        first_frame=0,
        last_frame=num_frames,
    )


def test_default_profile_merges_and_cli_overrides(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "meta").mkdir()
    (dataset_root / "meta/info.json").write_text("{}")
    config_path = tmp_path / "replay.yaml"
    config_path.write_text(yaml.safe_dump(_base_config(dataset_root)))
    profiles = tmp_path / "replays"
    profiles.mkdir()
    (profiles / "hyu_lemon.yaml").write_text(
        yaml.safe_dump(
            {
                "profile": {"description": "HYU"},
                "dataset": {"repo_id": "oms524/hyu", "root": str(dataset_root)},
            }
        )
    )
    args = replay_cli._arguments(
        ["--config", str(config_path), "--episode", "4", "--seconds", "2", "--rate", "0.5"]
    )

    cfg = replay_cli._effective(replay_cli._load_configuration(args), args)

    assert cfg["profile_name"] == "hyu_lemon"
    assert cfg["repo_id"] == "oms524/hyu"
    assert cfg["episode"] == 4
    assert cfg["seconds"] == 2
    assert cfg["rate"] == 0.5
    assert cfg["hardware"] is False


def test_preview_is_default_and_hardware_flags_are_explicit(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "meta").mkdir()
    (dataset_root / "meta/info.json").write_text("{}")
    data = _base_config(dataset_root)

    preview_args = replay_cli._arguments([])
    preview_cfg = replay_cli._effective(data, preview_args)
    replay_cli._validate_settings(preview_cfg, preview_args)
    assert preview_cfg["hardware"] is False

    init_args = replay_cli._arguments(["--init-can"])
    init_cfg = replay_cli._effective(data, init_args)
    with pytest.raises(ValueError, match="only valid together with --hardware"):
        replay_cli._validate_settings(init_cfg, init_args)


def test_load_replay_data_validates_names_and_selects_frame_range(monkeypatch, tmp_path: Path):
    class FakeSelection:
        def __init__(self, key: str, values: np.ndarray):
            self.key = key
            self.values = values

        def __getitem__(self, item):
            assert isinstance(item, slice)
            return {self.key: self.values.tolist()}

    class FakeDataset:
        fps = 30
        num_frames = 10
        num_episodes = 1
        features = {
            "observation.images.front": {"dtype": "video"},
            "observation.state": {"dtype": "float32", "names": list(replay_cli.MOTOR_NAMES)},
            "action": {"dtype": "float32", "names": list(replay_cli.MOTOR_NAMES)},
        }

        def __init__(self, repo_id, root, episodes):
            assert repo_id == "test/piper"
            assert root == tmp_path
            assert episodes == [2]
            self.values = np.zeros((self.num_frames, 7), dtype=np.float32)

        def select_columns(self, key):
            return FakeSelection(key, self.values)

    monkeypatch.setattr("lerobot.datasets.LeRobotDataset", FakeDataset)
    cfg = {
        "repo_id": "test/piper",
        "root": tmp_path,
        "episode": 2,
        "start_frame": 3,
        "frames": None,
        "seconds": 0.1,
    }

    data = replay_cli._load_replay_data(cfg)

    assert data.first_frame == 3
    assert data.last_frame == 6
    assert data.num_frames == 3
    assert data.image_keys == ("observation.images.front",)


def test_to_image_converts_chw_float_to_hwc_uint8():
    source = np.ones((3, 4, 5), dtype=np.float32) * 0.5

    image = replay_cli._to_image(source)

    assert image.shape == (4, 5, 3)
    assert image.dtype == np.uint8
    assert np.all(image == 127)


def test_hardware_replay_aligns_limits_and_safely_disconnects(monkeypatch):
    events: list[str] = []

    class FakeBus:
        def move_follower_to_normalized_pose(self, pose, **kwargs):
            assert set(pose) == {f"joint{index}" for index in range(1, 7)} | {"gripper"}
            events.append("return")

    class FakeRobot:
        def __init__(self, config):
            assert config.startup_pose is not None
            self.bus = FakeBus()

        def connect(self):
            events.append("connect")

        def send_action(self, action):
            events.append("send")
            return action

        def prepare_for_disconnect(self):
            events.append("prepare")

        def disconnect(self):
            events.append("disconnect")

    monkeypatch.setattr("lerobot_robot_piper.PiperFollower", FakeRobot)
    monkeypatch.setattr("lerobot.utils.robot_utils.precise_sleep", lambda seconds: None)
    cfg = {
        "follower_can": "can_follower",
        "speed_percent": 20,
        "max_relative_target": 3,
        "gripper_speed_mm_s": 40,
        "play_sounds": False,
        "wait_for_support": True,
        "align_start": True,
        "align_speed": 20,
        "align_timeout": 20,
        "joint_tolerance": 2,
        "gripper_tolerance": 2,
        "return_to_start": True,
        "rate": 1,
    }

    replay_cli._hardware_replay(cfg, _replay_data(num_frames=2))

    assert events == ["connect", "send", "send", "return", "prepare", "disconnect"]


def test_hardware_replay_disconnects_when_prepare_fails(monkeypatch):
    events: list[str] = []

    class FakeRobot:
        def __init__(self, _config):
            self.bus = SimpleNamespace()

        def connect(self):
            events.append("connect")

        def send_action(self, action):
            return action

        def prepare_for_disconnect(self):
            events.append("prepare")
            raise RuntimeError("prompt failed")

        def disconnect(self):
            events.append("disconnect")

    monkeypatch.setattr("lerobot_robot_piper.PiperFollower", FakeRobot)
    monkeypatch.setattr("lerobot.utils.robot_utils.precise_sleep", lambda seconds: None)
    cfg = {
        "follower_can": "can_follower",
        "speed_percent": 20,
        "max_relative_target": 3,
        "gripper_speed_mm_s": 40,
        "play_sounds": False,
        "wait_for_support": True,
        "align_start": False,
        "align_speed": 20,
        "align_timeout": 20,
        "joint_tolerance": 2,
        "gripper_tolerance": 2,
        "return_to_start": False,
        "rate": 1,
    }

    with pytest.raises(RuntimeError, match="prompt failed"):
        replay_cli._hardware_replay(cfg, _replay_data(num_frames=1))

    assert events == ["connect", "prepare", "disconnect"]
