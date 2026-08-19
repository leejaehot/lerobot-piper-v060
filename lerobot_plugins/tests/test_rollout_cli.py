from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from lerobot_piper import rollout_cli


def _write_checkpoint(
    root: Path,
    *,
    policy_type: str = "act",
    cameras: tuple[str, ...] = ("egoview", "wristcam"),
) -> Path:
    checkpoint = root / "outputs" / "test_policy"
    checkpoint.mkdir(parents=True)
    config: dict = {
        "type": policy_type,
        "device": "cuda",
        "use_amp": False,
        "input_features": {
            "observation.state": {"type": "STATE", "shape": [7]},
            **{
                f"observation.images.{name}": {"type": "VISUAL", "shape": [3, 480, 640]}
                for name in cameras
            },
        },
        "output_features": {"action": {"type": "ACTION", "shape": [7]}},
        "n_action_steps": 10,
    }
    if policy_type == "act":
        config["chunk_size"] = 10
    else:
        config["horizon"] = 16
    (checkpoint / "config.json").write_text(json.dumps(config))
    (checkpoint / "model.safetensors").write_bytes(b"checkpoint")
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        (checkpoint / name).write_text(json.dumps({"name": name, "steps": []}))
    return checkpoint


def _config(checkpoint: Path) -> dict:
    return {
        "policy": {
            "default": "act",
            "checkpoints": {"act": str(checkpoint)},
            "device": "cuda",
        },
        "rollout": {
            "task": "pick the can",
            "seconds": 10,
            "fps": 30,
        },
        "teleop_initial_pose": {
            "enabled": True,
            "speed_percent": 20,
            "timeout_seconds": 15,
            "joint_tolerance_degrees": 2,
            "gripper_tolerance_mm": 2,
            "normalized": dict(rollout_cli.DEFAULT_TELEOP_INITIAL_POSE),
        },
        "cameras": {
            "egoview": "111111111111",
            "wristcam": "222222222222",
        },
        "capture": {"camera_fps": 30, "width": 640, "height": 480},
        "arm": {
            "follower_can": "can_follower",
            "speed_percent": 30,
            "max_relative_target": 5,
            "gripper_speed_mm_s": 40,
        },
    }


def test_effective_resolves_policy_alias_and_cli_overrides(tmp_path: Path):
    checkpoint = _write_checkpoint(tmp_path)
    args = rollout_cli._arguments(["act", "--seconds=5", "--speed=20", "--amp"])

    cfg = rollout_cli._effective(_config(checkpoint), args)

    assert cfg["checkpoint"] == checkpoint.resolve()
    assert cfg["duration"] == 5
    assert cfg["speed_percent"] == 20
    assert cfg["use_amp"] is True
    assert cfg["align_start"] is True


def test_validate_accepts_matching_act_checkpoint(tmp_path: Path):
    checkpoint = _write_checkpoint(tmp_path)
    cfg = rollout_cli._effective(_config(checkpoint), rollout_cli._arguments(["act"]))

    info = rollout_cli._validate(cfg)

    assert info["type"] == "act"
    assert info["model_bytes"] > 0
    assert info["n_action_steps"] == 10


def test_validate_rejects_checkpoint_camera_mismatch(tmp_path: Path):
    checkpoint = _write_checkpoint(tmp_path, cameras=("front",))
    cfg = rollout_cli._effective(_config(checkpoint), rollout_cli._arguments(["act"]))

    with pytest.raises(ValueError, match="Checkpoint/camera mismatch"):
        rollout_cli._validate(cfg)


def test_validate_rejects_missing_processor_state(tmp_path: Path):
    checkpoint = _write_checkpoint(tmp_path)
    (checkpoint / "policy_preprocessor.json").write_text(
        json.dumps(
            {
                "name": "policy_preprocessor",
                "steps": [{"registry_name": "normalizer_processor", "state_file": "missing.safetensors"}],
            }
        )
    )
    cfg = rollout_cli._effective(_config(checkpoint), rollout_cli._arguments(["act"]))

    with pytest.raises(ValueError, match="references missing state file"):
        rollout_cli._validate(cfg)


def test_rollout_config_uses_piper_safety_settings(tmp_path: Path):
    checkpoint = _write_checkpoint(tmp_path)
    cfg = rollout_cli._effective(_config(checkpoint), rollout_cli._arguments(["act"]))
    rollout_cfg = rollout_cli._rollout_config(cfg)

    assert rollout_cfg.robot.type == "piper_follower"
    assert rollout_cfg.robot.port == "can_follower"
    assert rollout_cfg.robot.speed_percent == 30
    assert rollout_cfg.robot.max_relative_target == 5
    assert rollout_cfg.robot.wait_for_enter_on_disconnect is True
    assert rollout_cfg.robot.startup_pose == pytest.approx(
        rollout_cli.DEFAULT_TELEOP_INITIAL_POSE
    )
    assert rollout_cfg.robot.startup_pose_speed_percent == 20
    assert rollout_cfg.robot.startup_pose_timeout_s == 15
    assert rollout_cfg.duration == 10
    assert rollout_cfg.policy.pretrained_backbone_weights is None


def test_no_align_start_disables_startup_motion(tmp_path: Path):
    checkpoint = _write_checkpoint(tmp_path)
    cfg = rollout_cli._effective(
        _config(checkpoint),
        rollout_cli._arguments(["act", "--no-align-start"]),
    )

    assert cfg["align_start"] is False
    assert rollout_cli._rollout_config(cfg).robot.startup_pose is None


def test_realsense_preflight_uses_lerobot_isolated_context(monkeypatch):
    from lerobot.cameras.realsense import camera_realsense

    class FakeDevice:
        def supports(self, _info):
            return True

        def get_info(self, _info):
            return "123456789012"

    isolated_context = SimpleNamespace(query_devices=lambda: [FakeDevice()])
    calls = []

    def fake_isolated_context():
        calls.append(True)
        return isolated_context

    monkeypatch.setattr(camera_realsense, "_make_isolated_rs_context", fake_isolated_context)

    assert rollout_cli._detected_realsense_serials() == {"123456789012"}
    assert calls == [True]


def test_main_dry_run_never_checks_or_connects_hardware(tmp_path: Path, monkeypatch, capsys):
    checkpoint = _write_checkpoint(tmp_path)
    config_path = tmp_path / "rollout.yaml"
    config_path.write_text(yaml.safe_dump(_config(checkpoint)))

    monkeypatch.setattr(
        rollout_cli,
        "_preflight_hardware",
        lambda _cfg: pytest.fail("dry-run touched hardware"),
    )
    monkeypatch.setattr(
        rollout_cli,
        "_rollout_config",
        lambda _cfg: pytest.fail("dry-run built a live rollout"),
    )

    rollout_cli.main(["act", "--config", str(config_path), "--dry-run"])

    output = capsys.readouterr().out
    assert "PIPER ROLLOUT" in output
    assert "teleop initial pose ON" in output


def test_main_requires_yes_for_noninteractive_hardware_rollout(tmp_path: Path, monkeypatch):
    checkpoint = _write_checkpoint(tmp_path)
    config_path = tmp_path / "rollout.yaml"
    config_path.write_text(yaml.safe_dump(_config(checkpoint)))

    monkeypatch.setattr(rollout_cli, "_preflight_hardware", lambda _cfg: "cuda:0 (test)")
    monkeypatch.setattr(sys, "stdin", io.StringIO())
    monkeypatch.setattr(
        rollout_cli,
        "_rollout_config",
        lambda _cfg: pytest.fail("non-interactive rollout started without --yes"),
    )

    with pytest.raises(SystemExit) as exc_info:
        rollout_cli.main(["act", "--config", str(config_path)])

    assert exc_info.value.code == 1
