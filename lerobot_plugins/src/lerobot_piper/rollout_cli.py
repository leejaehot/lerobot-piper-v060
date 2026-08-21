from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import subprocess
import sys
import textwrap
import time
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
from lerobot_piper.cli_utils import (
    available_profiles,
    check_can_interfaces,
    deep_merge as _deep_merge,
    load_yaml as _load,
    override as _override,
    profile_directory,
    resolve_profile_path,
    section as _section,
    wrapped_lines as _line,
)
from lerobot_piper.console_ui import paint, phase, supports_color
from lerobot_piper.project_paths import PIPER_ROOT

DEFAULT_CONFIG = PIPER_ROOT / "configs/rollout.yaml"
PROFILE_DIRECTORY = "rollouts"
CAN_INIT = PIPER_ROOT / "scripts/can_init.sh"

_REQUIRED_LEROBOT_CHECKPOINT_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
)

DEFAULT_TELEOP_INITIAL_POSE = {
    "joint1": 0.0,
    "joint2": -100.0,
    "joint3": 100.0,
    "joint4": 0.0,
    "joint5": 0.0,
    "joint6": -13.0434782609,
    "gripper": 0.0,
}


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="piper_rollout",
        description="Run a local policy checkpoint on the Piper follower.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "policy_name",
        nargs="?",
        help="rollout profile, checkpoint alias from the config, or checkpoint directory",
    )
    parser.add_argument(
        "--policy",
        dest="policy_option",
        help="same as the positional policy argument",
    )
    parser.add_argument(
        "--profile",
        dest="profile_option",
        help="profile name from configs/rollouts or a profile YAML path",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="list available rollout profiles and exit",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML defaults")
    parser.add_argument("--task", help="task instruction passed to the policy")
    parser.add_argument("--seconds", type=float, help="bounded rollout duration")
    parser.add_argument("--fps", type=float, help="policy control frequency")
    parser.add_argument(
        "--n-action-steps",
        type=int,
        help="predicted actions to execute before observing again and replanning",
    )
    parser.add_argument("--device", help="PyTorch device, for example cuda or cuda:0")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable CUDA automatic mixed precision",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable torch.compile (leave off for the first hardware rollout)",
    )
    parser.add_argument("--speed", type=int, help="follower controller speed percent")
    parser.add_argument(
        "--max-relative-target",
        type=float,
        help="maximum normalized joint change accepted per control tick",
    )
    parser.add_argument(
        "--gripper-speed-mm-s",
        type=float,
        help="maximum commanded gripper travel speed",
    )
    parser.add_argument(
        "--interpolation-multiplier",
        type=int,
        help="number of smooth control ticks per policy action",
    )
    parser.add_argument(
        "--setup-preview",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="require camera/grid confirmation before any live rollout",
    )
    parser.add_argument(
        "--align-start",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="align follower to the teleop recording pose before policy control",
    )
    parser.add_argument(
        "--align-speed",
        type=int,
        help="follower speed percent used only for startup alignment",
    )
    parser.add_argument(
        "--align-timeout",
        type=float,
        help="maximum seconds to wait for startup alignment",
    )
    parser.add_argument("--rerun", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--return-to-start",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="smoothly return to the pose captured at startup when stopping",
    )
    parser.add_argument(
        "--wait-for-support",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="wait for Enter before releasing follower torque",
    )
    parser.add_argument("--sounds", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--init-can", action="store_true", help="initialize and rename CAN adapters first")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="validate files and print the plan only")
    mode.add_argument(
        "--check",
        action="store_true",
        help="load the checkpoint and run one synthetic inference without connecting hardware",
    )
    parser.add_argument("--yes", "-y", action="store_true", help="start without the confirmation prompt")
    parser.add_argument("--debug", action="store_true", help="show a traceback when startup fails")
    return parser.parse_args(argv)


def _print_profiles(config_path: Path) -> None:
    profiles = available_profiles(config_path, PROFILE_DIRECTORY)
    if not profiles:
        print(f"No rollout profiles found in {profile_directory(config_path, PROFILE_DIRECTORY)}")
        return
    print("Available Piper rollout profiles:")
    for path in profiles:
        data = _load(path)
        metadata = _section(data, "profile")
        description = str(metadata.get("description", "")).strip()
        policy = _section(data, "policy")
        selected = str(policy.get("default", ""))
        checkpoints = policy.get("checkpoints", {})
        checkpoint = checkpoints.get(selected, "") if isinstance(checkpoints, dict) else ""
        detail = description or str(checkpoint)
        print(f"  {path.stem:<16} {detail}")


def _load_configuration(args: argparse.Namespace) -> dict[str, Any]:
    base = _load(args.config)
    if args.profile_option and args.policy_name:
        raise ValueError("Specify a profile once: positional profile or --profile, not both")

    selector = args.profile_option or args.policy_name
    if selector is None:
        return base
    profile_path = resolve_profile_path(str(selector), args.config, PROFILE_DIRECTORY)
    if profile_path is None:
        if args.profile_option:
            names = ", ".join(
                path.stem for path in available_profiles(args.config, PROFILE_DIRECTORY)
            ) or "none"
            raise ValueError(f"Unknown rollout profile {selector!r}; available: {names}")
        return base

    profile_data = _load(profile_path)
    metadata = dict(_section(profile_data, "profile"))
    metadata.setdefault("name", profile_path.stem)
    camera_aliases = metadata.get("camera_aliases", {})
    if not isinstance(camera_aliases, dict):
        raise ValueError("profile.camera_aliases must be a YAML mapping")
    if camera_aliases and "cameras" in profile_data:
        raise ValueError("A profile cannot define both cameras and profile.camera_aliases")

    merged = _deep_merge(base, profile_data)
    merged["profile"] = metadata
    if "cameras" in profile_data:
        merged["cameras"] = copy.deepcopy(profile_data["cameras"])
    elif camera_aliases:
        base_cameras = _section(base, "cameras")
        missing = sorted({str(source) for source in camera_aliases.values()} - set(base_cameras))
        if missing:
            raise ValueError(
                "Profile camera alias references missing base camera(s): "
                + ", ".join(missing)
            )
        merged["cameras"] = {
            str(policy_name): base_cameras[str(hardware_name)]
            for policy_name, hardware_name in camera_aliases.items()
        }

    if args.profile_option is None:
        args.policy_name = None
    return merged


def _path_from_lerobot_root(value: str | Path, lerobot_root: Path) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not expanded.is_absolute():
        expanded = lerobot_root / expanded
    return expanded.resolve()


def _effective(data: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    policy = _section(data, "policy")
    rollout = _section(data, "rollout")
    arm = _section(data, "arm")
    cameras = _section(data, "cameras")
    capture = _section(data, "capture")
    audio = _section(data, "audio")
    initial_pose = _section(data, "teleop_initial_pose")
    profile = _section(data, "profile")
    runtime = _section(data, "runtime")

    if args.policy_name and args.policy_option:
        raise ValueError("Specify the policy once: positional act/dp or --policy, not both")
    selected = args.policy_option or args.policy_name or policy.get("default", "act")
    aliases = policy.get("checkpoints", {})
    if not isinstance(aliases, dict):
        raise ValueError("policy.checkpoints must be a YAML mapping")

    lerobot_root = Path(
        os.path.expandvars(
            os.path.expanduser(
                os.getenv("PIPER_LEROBOT_ROOT", str(policy.get("lerobot_root", "~/lerobot_v060")))
            )
        )
    ).resolve()
    checkpoint_value = aliases.get(selected, selected)
    checkpoint = _path_from_lerobot_root(str(checkpoint_value), lerobot_root)
    n_action_steps = _override(args.n_action_steps, policy.get("n_action_steps"))

    return {
        "profile_name": str(profile.get("name", "")).strip() or None,
        "policy_label": str(selected),
        "checkpoint": checkpoint,
        "device": str(_override(args.device, policy.get("device", "cuda"))),
        "use_amp": bool(_override(args.amp, policy.get("use_amp", False))),
        "use_torch_compile": bool(
            _override(args.compile, policy.get("use_torch_compile", False))
        ),
        "n_action_steps": (
            int(n_action_steps) if n_action_steps is not None else None
        ),
        "runtime_module": str(runtime.get("module", "")).strip() or None,
        "runtime_options": {
            str(key): copy.deepcopy(value)
            for key, value in runtime.items()
            if key != "module"
        },
        "disable_pretrained_backbone_download": bool(
            policy.get("disable_pretrained_backbone_download", True)
        ),
        "task": str(_override(args.task, rollout.get("task", ""))),
        "duration": float(_override(args.seconds, rollout.get("seconds", 30))),
        "fps": float(_override(args.fps, rollout.get("fps", 30))),
        "interpolation_multiplier": int(
            _override(
                args.interpolation_multiplier,
                rollout.get("interpolation_multiplier", 1),
            )
        ),
        "setup_preview": bool(
            _override(args.setup_preview, rollout.get("setup_preview", False))
        ),
        "setup_preview_config": Path(
            os.path.expandvars(
                os.path.expanduser(
                    str(
                        rollout.get(
                            "setup_preview_config",
                            PIPER_ROOT / "configs" / "record.yaml",
                        )
                    )
                )
            )
        ).resolve(),
        "setup_preview_camera": str(
            rollout.get("setup_preview_camera", "egoview")
        ),
        "rerun": bool(_override(args.rerun, rollout.get("rerun", False))),
        "rerun_compress_images": bool(rollout.get("rerun_compress_images", False)),
        "return_to_start": bool(
            _override(args.return_to_start, rollout.get("return_to_start", True))
        ),
        "follower_can": os.getenv(
            "PIPER_FOLLOWER_CAN", str(arm.get("follower_can", "can_follower"))
        ),
        "speed_percent": int(_override(args.speed, arm.get("speed_percent", 30))),
        "max_relative_target": float(
            _override(args.max_relative_target, arm.get("max_relative_target", 5))
        ),
        "gripper_speed_mm_s": float(
            _override(args.gripper_speed_mm_s, arm.get("gripper_speed_mm_s", 40))
        ),
        "wait_for_support": bool(
            _override(args.wait_for_support, arm.get("wait_for_support", True))
        ),
        "align_start": bool(
            _override(args.align_start, initial_pose.get("enabled", False))
        ),
        "startup_pose": initial_pose.get(
            "normalized",
            dict(DEFAULT_TELEOP_INITIAL_POSE),
        ),
        "startup_pose_speed_percent": int(
            _override(args.align_speed, initial_pose.get("speed_percent", 20))
        ),
        "startup_pose_timeout_s": float(
            _override(args.align_timeout, initial_pose.get("timeout_seconds", 20))
        ),
        "startup_pose_joint_tolerance_degrees": float(
            initial_pose.get("joint_tolerance_degrees", 2)
        ),
        "startup_pose_gripper_tolerance_mm": float(
            initial_pose.get("gripper_tolerance_mm", 2)
        ),
        "cameras": cameras,
        "camera_fps": int(capture.get("camera_fps", rollout.get("fps", 30))),
        "width": int(capture.get("width", 640)),
        "height": int(capture.get("height", 480)),
        "play_sounds": bool(_override(args.sounds, audio.get("enabled", True))),
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return value


def _runtime_adapter(cfg: dict[str, Any]):
    module_name = cfg.get("runtime_module")
    if module_name is None:
        return None
    try:
        module = importlib.import_module(str(module_name))
    except ImportError as exc:
        raise ValueError(f"Cannot load local policy runtime {module_name!r}: {exc}") from exc
    missing = [
        name
        for name in ("checkpoint_info", "check_inference", "rollout")
        if not callable(getattr(module, name, None))
    ]
    if missing:
        raise ValueError(
            f"Local policy runtime {module_name!r} is missing callable(s): "
            + ", ".join(missing)
        )
    return module


def _checkpoint_info(cfg: dict[str, Any]) -> dict[str, Any]:
    checkpoint = Path(cfg["checkpoint"])
    if not checkpoint.is_dir():
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint}")

    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise ValueError("Checkpoint is missing required file(s): config.json")
    model_config = _load_json(config_path)
    adapter = _runtime_adapter(cfg)
    if adapter is not None:
        info = adapter.checkpoint_info(cfg, model_config)
        if not isinstance(info, dict):
            raise ValueError("Local policy runtime checkpoint_info() must return a mapping")
        return info

    policy_type = model_config.get("type")
    if policy_type not in {"act", "diffusion"}:
        raise ValueError(
            "piper_rollout supports ACT and Diffusion checkpoints directly; "
            f"configure runtime.module for {policy_type!r}"
        )
    missing = [
        name for name in _REQUIRED_LEROBOT_CHECKPOINT_FILES if not (checkpoint / name).is_file()
    ]
    if missing:
        raise ValueError(f"Checkpoint is missing required file(s): {', '.join(missing)}")
    if (checkpoint / "model.safetensors").stat().st_size == 0:
        raise ValueError("model.safetensors is empty")
    inputs = model_config.get("input_features", {})
    outputs = model_config.get("output_features", {})
    if not isinstance(inputs, dict) or not isinstance(outputs, dict):
        raise ValueError("Checkpoint input_features/output_features must be mappings")
    state = inputs.get("observation.state")
    action = outputs.get("action")
    if not isinstance(state, dict) or state.get("shape") != [7]:
        raise ValueError("Checkpoint must expect a 7-dimensional observation.state")
    if not isinstance(action, dict) or action.get("shape") != [7]:
        raise ValueError("Checkpoint must produce a 7-dimensional action")

    expected_visuals = {f"observation.images.{name}" for name in cfg["cameras"]}
    actual_visuals = {
        key
        for key, feature in inputs.items()
        if isinstance(feature, dict) and feature.get("type") == "VISUAL"
    }
    if actual_visuals != expected_visuals:
        raise ValueError(
            "Checkpoint/camera mismatch: "
            f"checkpoint expects {sorted(actual_visuals)}, config provides {sorted(expected_visuals)}"
        )
    expected_shape = [3, int(cfg["height"]), int(cfg["width"])]
    wrong_shapes = {
        key: inputs[key].get("shape")
        for key in actual_visuals
        if inputs[key].get("shape") != expected_shape
    }
    if wrong_shapes:
        raise ValueError(
            f"Checkpoint camera shape mismatch: expected {expected_shape}, got {wrong_shapes}"
        )

    for filename in ("policy_preprocessor.json", "policy_postprocessor.json"):
        processor_config = _load_json(checkpoint / filename)
        for step in processor_config.get("steps", []):
            if not isinstance(step, dict):
                continue
            state_file = step.get("state_file")
            if state_file and not (checkpoint / str(state_file)).is_file():
                raise ValueError(f"{filename} references missing state file: {state_file}")

    checkpoint_n_action_steps = model_config.get("n_action_steps")
    effective_n_action_steps = (
        cfg["n_action_steps"]
        if cfg["n_action_steps"] is not None
        else checkpoint_n_action_steps
    )
    if not isinstance(effective_n_action_steps, int) or effective_n_action_steps < 1:
        raise ValueError("policy.n_action_steps must be a positive integer")
    chunk_size = model_config.get("chunk_size")
    if isinstance(chunk_size, int) and effective_n_action_steps > chunk_size:
        raise ValueError(
            "policy.n_action_steps cannot exceed checkpoint chunk_size "
            f"({effective_n_action_steps} > {chunk_size})"
        )
    if (
        policy_type == "act"
        and model_config.get("temporal_ensemble_coeff") is not None
        and effective_n_action_steps != 1
    ):
        raise ValueError(
            "ACT temporal ensembling requires policy.n_action_steps=1"
        )

    return {
        "type": str(policy_type),
        "model_bytes": (checkpoint / "model.safetensors").stat().st_size,
        "n_action_steps": effective_n_action_steps,
        "checkpoint_n_action_steps": checkpoint_n_action_steps,
        "chunk_size": chunk_size,
        "horizon": model_config.get("horizon"),
    }


def _setup_preview_info(cfg: dict[str, Any]) -> dict[str, Any] | None:
    if not cfg["setup_preview"]:
        return None
    config_path = Path(cfg["setup_preview_config"])
    preview_data = _load(config_path)
    preview_cameras = _section(preview_data, "cameras")
    camera_name = str(cfg["setup_preview_camera"])
    if camera_name not in preview_cameras:
        raise ValueError(
            f"Setup preview camera {camera_name!r} is not present in {config_path}"
        )
    preview_serial = str(preview_cameras[camera_name])
    rollout_serials = set(map(str, cfg["cameras"].values()))
    if preview_serial not in rollout_serials:
        raise ValueError(
            "Setup preview camera is not used by this rollout: "
            f"preview={camera_name} ({preview_serial}), "
            f"rollout serials={sorted(rollout_serials)}"
        )
    return {
        "config": config_path,
        "camera": camera_name,
        "serial": preview_serial,
    }


def _prepare_environment(cfg: dict[str, Any]) -> bool:
    preview = _setup_preview_info(cfg)
    if preview is None:
        return False

    from lerobot_piper import grid_preview_cli

    return grid_preview_cli.main(
        [
            "--config",
            str(preview["config"]),
            "--camera",
            str(preview["camera"]),
            "--confirm",
        ]
    )


def _validate(cfg: dict[str, Any]) -> dict[str, Any]:
    if not cfg["task"].strip():
        raise ValueError("rollout.task must not be empty")
    if cfg["duration"] <= 0:
        raise ValueError("rollout seconds must be positive and bounded")
    if cfg["fps"] <= 0:
        raise ValueError("rollout fps must be positive")
    if cfg["camera_fps"] <= 0 or cfg["width"] <= 0 or cfg["height"] <= 0:
        raise ValueError("camera_fps, width, and height must be positive")
    if cfg["interpolation_multiplier"] < 1:
        raise ValueError("interpolation_multiplier must be at least 1")
    if not 1 <= cfg["speed_percent"] <= 100:
        raise ValueError("speed_percent must be between 1 and 100")
    if cfg["max_relative_target"] <= 0:
        raise ValueError("max_relative_target must be positive")
    if cfg["gripper_speed_mm_s"] <= 0:
        raise ValueError("gripper_speed_mm_s must be positive")
    if not cfg["cameras"] or any(not str(serial).isdigit() for serial in cfg["cameras"].values()):
        raise ValueError("cameras must map policy camera names to numeric RealSense SDK serials")
    if len(set(map(str, cfg["cameras"].values()))) != len(cfg["cameras"]):
        raise ValueError("camera serials must be unique")
    if not isinstance(cfg["wait_for_support"], bool):
        raise ValueError("wait_for_support must be true or false")
    if not isinstance(cfg["setup_preview"], bool):
        raise ValueError("rollout.setup_preview must be true or false")
    _setup_preview_info(cfg)
    if cfg["align_start"]:
        if not 1 <= cfg["startup_pose_speed_percent"] <= 100:
            raise ValueError("teleop_initial_pose.speed_percent must be between 1 and 100")
        if cfg["startup_pose_timeout_s"] <= 0:
            raise ValueError("teleop_initial_pose.timeout_seconds must be positive")
        if cfg["startup_pose_joint_tolerance_degrees"] <= 0:
            raise ValueError("teleop_initial_pose.joint_tolerance_degrees must be positive")
        if cfg["startup_pose_gripper_tolerance_mm"] <= 0:
            raise ValueError("teleop_initial_pose.gripper_tolerance_mm must be positive")
        pose = cfg["startup_pose"]
        expected = set(DEFAULT_TELEOP_INITIAL_POSE)
        if not isinstance(pose, dict) or set(pose) != expected:
            raise ValueError(
                "teleop_initial_pose.normalized must contain exactly joint1..joint6 and gripper"
            )
        for name, value in pose.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"teleop_initial_pose.normalized.{name} must be numeric"
                ) from exc
            lower = 0.0 if name == "gripper" else -100.0
            if not np.isfinite(numeric) or not lower <= numeric <= 100.0:
                raise ValueError(
                    f"teleop_initial_pose.normalized.{name} must be finite and in "
                    f"[{lower:g}, 100]"
                )
    return _checkpoint_info(cfg)


def _device_description(device: str) -> str:
    import torch

    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        return str(torch_device)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available to PyTorch. Run piper_rollout --check outside a restricted container, "
            "or use --device=cpu only for debugging."
        )
    index = torch_device.index if torch_device.index is not None else torch.cuda.current_device()
    return f"cuda:{index} ({torch.cuda.get_device_name(index)})"


def _detected_realsense_serials() -> set[str]:
    try:
        import pyrealsense2 as rs
        from lerobot.cameras.realsense.camera_realsense import _make_isolated_rs_context
    except ImportError as exc:
        raise RuntimeError("pyrealsense2 is not installed in the active environment") from exc
    try:
        # Match LeRobot's RealSense driver.  A RealSense Viewer installation may
        # leave `"context": ""` in ~/.realsense-config.json; bare rs.context()
        # then fails before USB discovery with "expecting an object; got \"\"".
        devices = _make_isolated_rs_context().query_devices()
        return {
            str(device.get_info(rs.camera_info.serial_number))
            for device in devices
            if device.supports(rs.camera_info.serial_number)
        }
    except Exception as exc:
        raise RuntimeError(f"Could not enumerate RealSense cameras: {exc}") from exc


def _preflight_hardware(cfg: dict[str, Any]) -> str:
    check_can_interfaces((str(cfg["follower_can"]),))

    configured_serials = set(map(str, cfg["cameras"].values()))
    detected_serials = _detected_realsense_serials()
    missing_serials = configured_serials - detected_serials
    if missing_serials:
        raise RuntimeError(
            f"Missing RealSense camera serial(s): {', '.join(sorted(missing_serials))}; "
            f"detected: {', '.join(sorted(detected_serials)) or 'none'}"
        )
    return _device_description(str(cfg["device"]))


def _plan(cfg: dict[str, Any], checkpoint_info: dict[str, Any], *, mode: str) -> None:
    camera_text = "  ·  ".join(f"{name} ({serial})" for name, serial in cfg["cameras"].items())
    chunk = checkpoint_info.get("n_action_steps")
    if checkpoint_info.get("plan_detail"):
        policy_detail = str(checkpoint_info["plan_detail"])
    elif checkpoint_info["type"] == "act":
        policy_detail = (
            f"ACT · predict {checkpoint_info.get('chunk_size')} · "
            f"execute {chunk} then replan"
        )
    else:
        policy_detail = (
            f"Diffusion · horizon {checkpoint_info.get('horizon')} · execute {chunk} actions/chunk"
        )
    rows: list[str] = []
    rows += _line("MODE", mode)
    if cfg["profile_name"]:
        rows += _line("PROFILE", str(cfg["profile_name"]))
    rows += _line("POLICY", f"{cfg['policy_label']} · {policy_detail}")
    rows += _line("CHECKPOINT", str(cfg["checkpoint"]))
    rows += _line("MODEL", f"{checkpoint_info['model_bytes'] / (1024**2):.1f} MiB")
    rows += _line("TASK", str(cfg["task"]))
    rows += _line("CAMERAS", camera_text)
    rows += _line(
        "SETUP GRID",
        (
            f"ON · {cfg['setup_preview_camera']} · Enter confirms, q/Esc cancels"
            if cfg["setup_preview"]
            else "OFF"
        ),
    )
    control_rate = f"{cfg['fps']:g} Hz"
    if cfg["interpolation_multiplier"] > 1:
        control_rate = (
            f"policy {cfg['fps']:g} Hz · send "
            f"{cfg['fps'] * cfg['interpolation_multiplier']:g} Hz "
            f"({cfg['interpolation_multiplier']}× linear)"
        )
    rows += _line(
        "CONTROL",
        (
            f"{control_rate} · {cfg['duration']:g}s · speed {cfg['speed_percent']}% · "
            f"max Δ {cfg['max_relative_target']:g} · gripper {cfg['gripper_speed_mm_s']:g} mm/s"
        ),
    )
    rows += _line(
        "START ALIGN",
        (
            f"teleop initial pose {'ON' if cfg['align_start'] else 'OFF'}"
            + (
                f" · speed {cfg['startup_pose_speed_percent']}% · "
                f"timeout {cfg['startup_pose_timeout_s']:g}s"
                if cfg["align_start"]
                else ""
            )
        ),
    )
    if cfg["align_start"]:
        pose = cfg["startup_pose"]
        rows += _line(
            "START POSE",
            "["
            + ", ".join(
                f"{float(pose[name]):.2f}" for name in DEFAULT_TELEOP_INITIAL_POSE
            )
            + "]",
        )
    rows += _line(
        "COMPUTE",
        (
            f"{cfg['device']} · AMP {'ON' if cfg['use_amp'] else 'OFF'} · "
            f"torch.compile {'ON' if cfg['use_torch_compile'] else 'OFF'}"
        ),
    )
    rows += _line(
        "SHUTDOWN",
        (
            f"return-to-start {'ON' if cfg['return_to_start'] else 'OFF'} · "
            f"support confirmation {'ON' if cfg['wait_for_support'] else 'OFF'}"
        ),
    )
    rows += _line(
        "OPTIONS",
        f"Rerun {'ON' if cfg['rerun'] else 'OFF'} · voice {'ON' if cfg['play_sounds'] else 'OFF'}",
    )

    width = 76
    color = supports_color(sys.stdout)
    print(
        paint(
            "╭─ PIPER ROLLOUT " + "─" * (width - 17) + "╮",
            "cyan",
            bold=True,
            enabled=color,
        )
    )
    for row in rows:
        print(f"│ {row:<{width - 2}} │")
    print("├" + "─" * width + "┤")
    caution = (
        "CAUTION: startup alignment and autonomous policy control enable follower torque; "
        "keep E-stop ready."
    )
    for row in textwrap.wrap(caution, width=width - 2):
        print(paint(f"│ {row:<{width - 2}} │", "yellow", bold=True, enabled=color))
    print("╰" + "─" * width + "╯")


def _policy_config(cfg: dict[str, Any]):
    from lerobot.configs import PreTrainedConfig

    policy_config = PreTrainedConfig.from_pretrained(str(cfg["checkpoint"]))
    policy_config.pretrained_path = str(cfg["checkpoint"])
    policy_config.device = str(cfg["device"])
    policy_config.use_amp = bool(cfg["use_amp"])
    if cfg["n_action_steps"] is not None:
        policy_config.n_action_steps = int(cfg["n_action_steps"])
    if cfg["disable_pretrained_backbone_download"] and hasattr(
        policy_config, "pretrained_backbone_weights"
    ):
        # The complete backbone is already present in model.safetensors. Avoid a
        # redundant ImageNet download when moving a checkpoint to an offline Jetson.
        policy_config.pretrained_backbone_weights = None
    return policy_config


def _rollout_config(cfg: dict[str, Any]):
    from lerobot.cameras.realsense import RealSenseCameraConfig
    from lerobot.rollout import BaseStrategyConfig, RolloutConfig, SyncInferenceConfig
    from lerobot_robot_piper import PiperFollowerConfig

    cameras = {
        name: RealSenseCameraConfig(
            serial_number_or_name=str(serial),
            fps=int(cfg["camera_fps"]),
            width=int(cfg["width"]),
            height=int(cfg["height"]),
            use_rgb=True,
            use_depth=False,
        )
        for name, serial in cfg["cameras"].items()
    }
    return RolloutConfig(
        robot=PiperFollowerConfig(
            id="piper_follower",
            port=str(cfg["follower_can"]),
            cameras=cameras,
            speed_percent=int(cfg["speed_percent"]),
            max_relative_target=float(cfg["max_relative_target"]),
            gripper_speed_mm_s=float(cfg["gripper_speed_mm_s"]),
            terminal_update_hz=0,
            play_sounds=bool(cfg["play_sounds"]),
            disable_torque_on_disconnect=True,
            wait_for_enter_on_disconnect=bool(cfg["wait_for_support"]),
            startup_pose=(
                {name: float(value) for name, value in cfg["startup_pose"].items()}
                if cfg["align_start"]
                else None
            ),
            startup_pose_speed_percent=int(cfg["startup_pose_speed_percent"]),
            startup_pose_timeout_s=float(cfg["startup_pose_timeout_s"]),
            startup_pose_joint_tolerance_degrees=float(
                cfg["startup_pose_joint_tolerance_degrees"]
            ),
            startup_pose_gripper_tolerance_mm=float(
                cfg["startup_pose_gripper_tolerance_mm"]
            ),
        ),
        policy=_policy_config(cfg),
        strategy=BaseStrategyConfig(),
        inference=SyncInferenceConfig(),
        fps=float(cfg["fps"]),
        duration=float(cfg["duration"]),
        interpolation_multiplier=int(cfg["interpolation_multiplier"]),
        device=str(cfg["device"]),
        task=str(cfg["task"]),
        display_data=bool(cfg["rerun"]),
        display_mode="rerun",
        display_compressed_images=bool(cfg["rerun_compress_images"]),
        play_sounds=bool(cfg["play_sounds"]),
        return_to_initial_position=bool(cfg["return_to_start"]),
        use_torch_compile=bool(cfg["use_torch_compile"]),
    )


def _check_inference(cfg: dict[str, Any]) -> tuple[list[float], float, str]:
    adapter = _runtime_adapter(cfg)
    if adapter is not None:
        return adapter.check_inference(cfg)

    import torch

    from lerobot.configs import FeatureType
    from lerobot.policies import get_policy_class, make_pre_post_processors
    from lerobot.policies.utils import prepare_observation_for_inference

    device_description = _device_description(str(cfg["device"]))
    policy_config = _policy_config(cfg)
    policy_class = get_policy_class(policy_config.type)
    start = time.perf_counter()
    policy = policy_class.from_pretrained(str(cfg["checkpoint"]), config=policy_config)
    policy = policy.to(str(cfg["device"]))
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=str(cfg["checkpoint"]),
        preprocessor_overrides={"device_processor": {"device": str(cfg["device"])}},
    )

    observation: dict[str, np.ndarray] = {}
    for key, feature in policy_config.input_features.items():
        if feature.type is FeatureType.VISUAL:
            channels, height, width = feature.shape
            observation[key] = np.zeros((height, width, channels), dtype=np.uint8)
        elif feature.type is FeatureType.STATE:
            if key == "observation.state" and cfg["align_start"]:
                observation[key] = np.asarray(
                    [cfg["startup_pose"][name] for name in DEFAULT_TELEOP_INITIAL_POSE],
                    dtype=np.float32,
                )
            else:
                observation[key] = np.zeros(feature.shape, dtype=np.float32)

    prepared = prepare_observation_for_inference(
        observation,
        torch.device(str(cfg["device"])),
        task=str(cfg["task"]),
        robot_type="piper_follower",
    )
    autocast = (
        torch.autocast(device_type="cuda")
        if torch.device(str(cfg["device"])).type == "cuda" and policy_config.use_amp
        else nullcontext()
    )
    policy.reset()
    with torch.inference_mode(), autocast:
        action = postprocessor(policy.select_action(preprocessor(prepared)))
    action = action.squeeze(0).detach().cpu()
    if tuple(action.shape) != (7,):
        raise RuntimeError(f"Synthetic inference returned shape {tuple(action.shape)}, expected (7,)")
    if not torch.isfinite(action).all():
        raise RuntimeError("Synthetic inference returned non-finite actions")
    elapsed = time.perf_counter() - start
    values = [float(value) for value in action]
    del policy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return values, elapsed, device_description


def _failure_hint(message: str) -> str | None:
    lower = message.lower()
    if "cuda is not available" in lower:
        return "Run outside the restricted container and verify the Jetson PyTorch CUDA build."
    if "missing can" in lower or "unhealthy can" in lower:
        return "Run piper_rollout with --init-can after stopping other Piper processes."
    if "realsense" in lower:
        return "Close RealSense Viewer/other camera processes and reconnect the camera hub."
    if "failed to enable follower" in lower:
        return "Release the E-stop, power-cycle the follower controller, then retry."
    if "out of memory" in lower:
        return "Stop other GPU processes; test ACT first, then retry DP with --no-rerun."
    return None


def main(argv: list[str] | None = None) -> None:
    args = _arguments(argv)
    try:
        if args.list_profiles:
            _print_profiles(args.config)
            return
        cfg = _effective(_load_configuration(args), args)
        checkpoint_info = _validate(cfg)
        adapter = _runtime_adapter(cfg)
        preparation_confirmed = False

        if not args.dry_run and not args.check and cfg["setup_preview"]:
            preparation_confirmed = _prepare_environment(cfg)
            if not preparation_confirmed:
                phase("CANCELLED", "Pre-rollout environment check was cancelled", "yellow")
                return

        if args.init_can:
            phase("CAN", "Initializing USB-CAN adapters")
            command = [str(CAN_INIT)]
            if args.dry_run or args.check:
                command.append("--dry-run")
            subprocess.run(command, check=True)
            phase("CAN", "CAN initialization completed", "green")

        mode = "dry run" if args.dry_run else "checkpoint check" if args.check else "hardware rollout"
        _plan(cfg, checkpoint_info, mode=mode)
        if args.dry_run:
            phase("DRY RUN", "Checkpoint layout is valid; no hardware or model was activated", "green")
            return

        if args.check:
            phase("CHECK", "Loading checkpoint and running one synthetic inference")
            action, elapsed, device_description = _check_inference(cfg)
            phase("CHECK", f"Synthetic inference passed on {device_description}", "green")
            print(f"  Load + first inference: {elapsed:.2f}s")
            print("  Action: " + " ".join(f"{value:+.3f}" for value in action))
            return

        phase("CHECK", "Checking CUDA, follower CAN, and exact RealSense serials")
        device_description = _preflight_hardware(cfg)
        phase("READY", f"Preflight passed on {device_description}", "green")
        if not args.yes and not preparation_confirmed and not sys.stdin.isatty():
            raise RuntimeError("Non-interactive hardware rollout requires explicit --yes")
        if not args.yes and not preparation_confirmed:
            prompt = (
                "Clear the workspace and press Enter to align the follower and start, "
                "or type q to cancel: "
                if cfg["align_start"]
                else "Clear the workspace and press Enter to start, or type q to cancel: "
            )
            answer = input(prompt).strip().lower()
            if answer in {"q", "quit", "n", "no"}:
                phase("CANCELLED", "No hardware was activated", "yellow")
                return

        connect_message = "Loading policy, connecting cameras, and enabling follower torque"
        if cfg["align_start"]:
            connect_message = (
                "Loading policy, enabling follower torque, aligning the startup pose, "
                "and connecting cameras"
            )
        phase("CONNECT", connect_message)
        if adapter is None:
            from lerobot.scripts import lerobot_rollout

            lerobot_rollout.rollout(_rollout_config(cfg))
        else:
            adapter.rollout(cfg)
        phase("COMPLETE", "Policy rollout finished", "green")
    except KeyboardInterrupt:
        phase("STOPPED", "Interrupted by user; safe teardown was requested", "yellow", stream=sys.stderr)
        raise SystemExit(130) from None
    except Exception as exc:
        phase("ERROR", str(exc), "red", stream=sys.stderr)
        if hint := _failure_hint(str(exc)):
            phase("HINT", hint, "yellow", stream=sys.stderr)
        if args.debug:
            traceback.print_exc()
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
