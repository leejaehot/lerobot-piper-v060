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
    args = rollout_cli._arguments(
        ["act", "--seconds=5", "--speed=20", "--n-action-steps=3", "--amp"]
    )

    cfg = rollout_cli._effective(_config(checkpoint), args)

    assert cfg["checkpoint"] == checkpoint.resolve()
    assert cfg["duration"] == 5
    assert cfg["speed_percent"] == 20
    assert cfg["n_action_steps"] == 3
    assert cfg["use_amp"] is True
    assert cfg["align_start"] is True


def test_named_profile_merges_policy_and_remaps_camera_names(tmp_path: Path):
    checkpoint = _write_checkpoint(tmp_path, cameras=("front", "right"))
    config_path = tmp_path / "rollout.yaml"
    config_path.write_text(yaml.safe_dump(_config(checkpoint)))
    profile_dir = tmp_path / "rollouts"
    profile_dir.mkdir()
    (profile_dir / "hyu_act.yaml").write_text(
        yaml.safe_dump(
            {
                "profile": {
                    "description": "Hanyang test profile",
                    "camera_aliases": {"front": "egoview", "right": "wristcam"},
                },
                "policy": {
                    "default": "hyu_act",
                    "checkpoints": {"hyu_act": str(checkpoint)},
                    "n_action_steps": 4,
                },
                "rollout": {"task": "place the lemon on the red plate"},
                "teleop_initial_pose": {
                    "normalized": {
                        "joint1": 12.0,
                        "joint2": -47.0,
                        "joint3": 40.0,
                        "joint4": 0.0,
                        "joint5": 98.0,
                        "joint6": -4.0,
                        "gripper": 63.6,
                    }
                },
            }
        )
    )
    args = rollout_cli._arguments(["hyu_act", "--config", str(config_path)])

    cfg = rollout_cli._effective(rollout_cli._load_configuration(args), args)
    info = rollout_cli._validate(cfg)

    assert cfg["profile_name"] == "hyu_act"
    assert cfg["policy_label"] == "hyu_act"
    assert cfg["checkpoint"] == checkpoint.resolve()
    assert cfg["task"] == "place the lemon on the red plate"
    assert cfg["n_action_steps"] == 4
    assert cfg["startup_pose"]["joint5"] == 98.0
    assert cfg["cameras"] == {
        "front": "111111111111",
        "right": "222222222222",
    }
    assert info["type"] == "act"
    assert info["n_action_steps"] == 4
    assert info["checkpoint_n_action_steps"] == 10


def test_explicit_unknown_profile_lists_available_names(tmp_path: Path):
    checkpoint = _write_checkpoint(tmp_path)
    config_path = tmp_path / "rollout.yaml"
    config_path.write_text(yaml.safe_dump(_config(checkpoint)))
    profile_dir = tmp_path / "rollouts"
    profile_dir.mkdir()
    (profile_dir / "sju_act.yaml").write_text("profile: {}\n")
    args = rollout_cli._arguments(
        ["--profile", "missing", "--config", str(config_path)]
    )

    with pytest.raises(ValueError, match="available: sju_act"):
        rollout_cli._load_configuration(args)


def test_list_profiles_does_not_validate_or_connect_hardware(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    config_path = tmp_path / "rollout.yaml"
    profile_dir = tmp_path / "rollouts"
    profile_dir.mkdir()
    (profile_dir / "hyu_act.yaml").write_text(
        yaml.safe_dump(
            {
                "profile": {"description": "Hanyang ACT"},
                "policy": {
                    "default": "hyu_act",
                    "checkpoints": {"hyu_act": "outputs/hyu/act_latest"},
                },
            }
        )
    )
    monkeypatch.setattr(
        rollout_cli,
        "_preflight_hardware",
        lambda _cfg: pytest.fail("profile listing touched hardware"),
    )

    rollout_cli.main(["--config", str(config_path), "--list-profiles"])

    output = capsys.readouterr().out
    assert "hyu_act" in output
    assert "Hanyang ACT" in output


def test_validate_accepts_matching_act_checkpoint(tmp_path: Path):
    checkpoint = _write_checkpoint(tmp_path)
    cfg = rollout_cli._effective(_config(checkpoint), rollout_cli._arguments(["act"]))

    info = rollout_cli._validate(cfg)

    assert info["type"] == "act"
    assert info["model_bytes"] > 0
    assert info["n_action_steps"] == 10


def test_validate_rejects_unsupported_policy(tmp_path: Path):
    checkpoint = _write_checkpoint(tmp_path, policy_type="unsupported")
    cfg = rollout_cli._effective(_config(checkpoint), rollout_cli._arguments(["act"]))

    with pytest.raises(ValueError, match="supports ACT and Diffusion"):
        rollout_cli._validate(cfg)


def test_validate_rejects_checkpoint_camera_mismatch(tmp_path: Path):
    checkpoint = _write_checkpoint(tmp_path, cameras=("front",))
    cfg = rollout_cli._effective(_config(checkpoint), rollout_cli._arguments(["act"]))

    with pytest.raises(ValueError, match="Checkpoint/camera mismatch"):
        rollout_cli._validate(cfg)


def test_validate_rejects_action_steps_larger_than_chunk(tmp_path: Path):
    checkpoint = _write_checkpoint(tmp_path)
    data = _config(checkpoint)
    data["policy"]["n_action_steps"] = 11
    cfg = rollout_cli._effective(data, rollout_cli._arguments(["act"]))

    with pytest.raises(ValueError, match="cannot exceed checkpoint chunk_size"):
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
    assert rollout_cfg.policy.n_action_steps == 10
    assert rollout_cfg.policy.pretrained_backbone_weights is None


def test_rollout_config_applies_action_step_override(tmp_path: Path):
    checkpoint = _write_checkpoint(tmp_path)
    data = _config(checkpoint)
    data["policy"]["n_action_steps"] = 4
    cfg = rollout_cli._effective(data, rollout_cli._arguments(["act"]))

    rollout_cfg = rollout_cli._rollout_config(cfg)

    assert rollout_cfg.policy.chunk_size == 10
    assert rollout_cfg.policy.n_action_steps == 4


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
    preview_config = tmp_path / "record.yaml"
    preview_config.write_text(
        yaml.safe_dump({"cameras": {"egoview": "111111111111"}})
    )
    config_path = tmp_path / "rollout.yaml"
    data = _config(checkpoint)
    data["rollout"].update(
        {
            "setup_preview": True,
            "setup_preview_config": str(preview_config),
            "setup_preview_camera": "egoview",
        }
    )
    config_path.write_text(yaml.safe_dump(data))

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
    monkeypatch.setattr(
        rollout_cli,
        "_prepare_environment",
        lambda _cfg: pytest.fail("dry-run opened the setup preview"),
    )

    rollout_cli.main(["act", "--config", str(config_path), "--dry-run"])

    output = capsys.readouterr().out
    assert "PIPER ROLLOUT" in output
    assert "SETUP GRID" in output
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


def test_setup_preview_confirms_before_hardware_preflight_for_any_policy(
    tmp_path: Path,
    monkeypatch,
):
    checkpoint = _write_checkpoint(tmp_path)
    preview_config = tmp_path / "record.yaml"
    preview_config.write_text(
        yaml.safe_dump({"cameras": {"egoview": "111111111111"}})
    )
    config_path = tmp_path / "rollout.yaml"
    data = _config(checkpoint)
    data["rollout"].update(
        {
            "setup_preview": True,
            "setup_preview_config": str(preview_config),
            "setup_preview_camera": "egoview",
        }
    )
    config_path.write_text(yaml.safe_dump(data))
    calls: list[str] = []

    adapter = SimpleNamespace(
        checkpoint_info=lambda _cfg, _model_config: {
            "type": "test-runtime",
            "model_bytes": 10,
            "n_action_steps": 1,
            "plan_detail": "test runtime",
        },
        check_inference=lambda _cfg: ([0.0] * 7, 0.1, "cpu"),
        rollout=lambda _cfg: calls.append("rollout"),
    )
    monkeypatch.setattr(rollout_cli, "_runtime_adapter", lambda _cfg: adapter)
    monkeypatch.setattr(
        rollout_cli,
        "_prepare_environment",
        lambda _cfg: calls.append("prepare") or True,
    )
    monkeypatch.setattr(
        rollout_cli,
        "_preflight_hardware",
        lambda _cfg: calls.append("preflight") or "cuda:0 (test)",
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO())

    rollout_cli.main(["act", "--config", str(config_path)])

    assert calls == ["prepare", "preflight", "rollout"]
