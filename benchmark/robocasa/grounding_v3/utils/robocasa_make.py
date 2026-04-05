from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def add_robocasa_to_pythonpath() -> None:
    """
    Makes `import robocasa` work from this repo checkout without installing it.

    Expects the upstream RoboCasa code to live at:
      `<repo_root>/benchmark/robocasa/robocasa`
    """

    repo_root = Path(__file__).resolve().parents[2]
    robocasa_root = repo_root / "benchmark" / "robocasa"
    if not robocasa_root.exists():
        raise FileNotFoundError(
            f"Expected RoboCasa repo at {robocasa_root}, but it does not exist."
        )

    robocasa_root_str = str(robocasa_root)
    if robocasa_root_str not in sys.path:
        sys.path.insert(0, robocasa_root_str)


def make_robocasa_env(
    *,
    env_name: str,
    robots: str | list[str] = "PandaOmron",
    controller: str | None = None,
    camera: str | None = "robot0_frontview",
    renderer: str = "mjviewer",
    has_renderer: bool = False,
    has_offscreen_renderer: bool = False,
    use_camera_obs: bool = False,
    ignore_done: bool = True,
    control_freq: int = 20,
    seed: int | None = None,
    env_kwargs: dict[str, Any] | None = None,
) -> tuple[Any, str]:
    """
    Creates a RoboCasa env and returns (env, env_info_json).

    Notes:
      - `env` is an instance of robosuite's MujocoEnv subclass.
      - `env_info_json` is a JSON string storing the make-config (useful for datasets).
    """

    add_robocasa_to_pythonpath()

    # Import after path fix
    import robosuite  # type: ignore
    from robosuite.controllers import (  # type: ignore
        load_composite_controller_config,
    )

    import robocasa  # noqa: F401  # type: ignore

    env_kwargs = dict(env_kwargs or {})

    # Controller config
    robot_for_controller = robots if isinstance(robots, str) else robots[0]
    controller_config = load_composite_controller_config(
        controller=controller,
        robot=robot_for_controller,
    )

    config: dict[str, Any] = {
        "env_name": env_name,
        "robots": robots,
        "controller_configs": controller_config,
    }
    if seed is not None:
        config["seed"] = seed
    config.update(env_kwargs)

    env = robosuite.make(
        **config,
        has_renderer=has_renderer,
        has_offscreen_renderer=has_offscreen_renderer,
        render_camera=camera,
        ignore_done=ignore_done,
        use_camera_obs=use_camera_obs,
        control_freq=control_freq,
        renderer=renderer,
    )

    env_info_json = json.dumps(
        {
            "make_config": config,
            "renderer": renderer,
            "camera": camera,
            "has_renderer": has_renderer,
            "has_offscreen_renderer": has_offscreen_renderer,
            "use_camera_obs": use_camera_obs,
            "ignore_done": ignore_done,
            "control_freq": control_freq,
            "cwd": os.getcwd(),
        },
        ensure_ascii=False,
        indent=2,
    )
    return env, env_info_json

