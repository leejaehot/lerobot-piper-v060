from __future__ import annotations

import copy
import os
import subprocess
import textwrap
from pathlib import Path
from typing import Any, Iterable

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"Config file does not exist: {path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")
    return data


def section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"'{name}' must be a YAML mapping")
    return value


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def profile_directory(config_path: Path, directory_name: str) -> Path:
    return config_path.expanduser().resolve().parent / directory_name


def available_profiles(config_path: Path, directory_name: str) -> list[Path]:
    directory = profile_directory(config_path, directory_name)
    return sorted({*directory.glob("*.yaml"), *directory.glob("*.yml")})


def resolve_profile_path(
    selector: str,
    config_path: Path,
    directory_name: str,
) -> Path | None:
    expanded = Path(os.path.expandvars(os.path.expanduser(selector)))
    candidates: list[Path] = []
    if expanded.suffix.lower() in {".yaml", ".yml"}:
        candidates.append(expanded if expanded.is_absolute() else Path.cwd() / expanded)
        candidates.append(config_path.expanduser().resolve().parent / expanded)
    else:
        directory = profile_directory(config_path, directory_name)
        candidates.extend((directory / f"{selector}.yaml", directory / f"{selector}.yml"))
    return next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)


def override(value: Any, fallback: Any) -> Any:
    return fallback if value is None else value


def wrapped_lines(
    label: str,
    value: str,
    *,
    width: int = 72,
    label_width: int = 11,
) -> list[str]:
    prefix = f"{label:<{label_width}} "
    parts = textwrap.wrap(value, width=width - len(prefix)) or [""]
    return [
        prefix + part if index == 0 else " " * len(prefix) + part
        for index, part in enumerate(parts)
    ]


def check_can_interfaces(names: Iterable[str]) -> None:
    interfaces = tuple(dict.fromkeys(map(str, names)))
    missing = [name for name in interfaces if not Path(f"/sys/class/net/{name}").exists()]
    if missing:
        raise RuntimeError(f"Missing CAN interface(s): {', '.join(missing)}; retry with --init-can")

    unhealthy: list[str] = []
    for name in interfaces:
        status = subprocess.run(
            ["ip", "-details", "link", "show", name],
            check=False,
            capture_output=True,
            text=True,
        )
        if status.returncode != 0 or "can state ERROR-ACTIVE" not in status.stdout:
            unhealthy.append(name)
    if unhealthy:
        raise RuntimeError(
            f"Unhealthy CAN interface(s): {', '.join(unhealthy)}; "
            "stop other Piper processes and retry with --init-can"
        )
