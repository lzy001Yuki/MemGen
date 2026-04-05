from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
from typing import Any

from grounding_v3.tasks import get_task_spec
from grounding_v3.utils import make_robocasa_env


def _timestamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="microwave_relocation_v1")
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "human"),
        help="Output directory root",
    )
    parser.add_argument("--controller", type=str, default=None)
    parser.add_argument("--renderer", type=str, default="mjviewer", choices=["mjviewer", "mujoco"])
    parser.add_argument("--camera", type=str, default="robot0_frontview")
    parser.add_argument("--arm", type=str, default="right")
    parser.add_argument("--device", type=str, default=None, choices=["keyboard", "spacemouse"])
    parser.add_argument("--pos-sensitivity", type=float, default=4.0)
    parser.add_argument("--rot-sensitivity", type=float, default=4.0)
    parser.add_argument("--max-fr", type=int, default=30)
    parser.add_argument("--num-scenarios", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    task = get_task_spec(args.task)

    # Late imports (robosuite / robocasa are heavy)
    from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper  # type: ignore
    import hid  # type: ignore

    import robocasa.macros as macros  # type: ignore
    from robocasa.scripts.collect_demos import (  # type: ignore
        collect_human_trajectory,
        gather_demonstrations_as_hdf5,
    )
    from robocasa.wrappers.enclosing_wall_render_wrapper import (  # type: ignore
        EnclosingWallHotkeyHandler,
        EnclosingWallRenderWrapper,
        install_enclosing_wall_hotkeys,
    )

    out_root = Path(args.out).resolve()
    _ensure_dir(out_root)

    scenario_root = out_root / f"{_timestamp()}_{task.name}"
    _ensure_dir(scenario_root)

    scenario_meta: dict[str, Any] = {
        "task": task.name,
        "note": task.note,
        "stages": [],
        "created_at": datetime.datetime.now().isoformat(),
    }

    # Pick device
    selected_device = args.device
    if selected_device is None:
        spacemouse_found = False
        for dev in hid.enumerate():
            vendor_id, product_id = dev["vendor_id"], dev["product_id"]
            if vendor_id == macros.SPACEMOUSE_VENDOR_ID and product_id == macros.SPACEMOUSE_PRODUCT_ID:
                spacemouse_found = True
                break
        selected_device = "spacemouse" if spacemouse_found else "keyboard"

    for scenario_idx in range(args.num_scenarios):
        print(f"\n=== Scenario {scenario_idx + 1}/{args.num_scenarios}: {task.name} ===")

        for stage_idx, stage in enumerate(task.stages):
            stage_dir = scenario_root / f"{stage_idx:02d}_{stage.name}_{stage.env_name}"
            _ensure_dir(stage_dir)
            episodes_dir = stage_dir / "episodes"
            _ensure_dir(episodes_dir)

            env, env_info_json = make_robocasa_env(
                env_name=stage.env_name,
                controller=args.controller,
                camera=args.camera,
                renderer=args.renderer,
                has_renderer=True,
                has_offscreen_renderer=False,
                use_camera_obs=False,
                ignore_done=True,
                control_freq=20,
                seed=args.seed,
                env_kwargs=stage.env_kwargs,
            )

            env = VisualizationWrapper(env)
            env = EnclosingWallRenderWrapper(env, alpha=0.15, enabled=False)
            env = DataCollectionWrapper(env, str(episodes_dir), use_env_xml_for_reset=True)
            install_enclosing_wall_hotkeys(env)

            # Initialize device instance (Keyboard / SpaceMouse)
            if selected_device == "keyboard":
                from robosuite.devices import Keyboard  # type: ignore

                device = Keyboard(
                    env=env,
                    pos_sensitivity=args.pos_sensitivity,
                    rot_sensitivity=args.rot_sensitivity,
                )
            elif selected_device == "spacemouse":
                from robosuite.devices import SpaceMouse  # type: ignore

                device = SpaceMouse(
                    env=env,
                    pos_sensitivity=args.pos_sensitivity,
                    rot_sensitivity=args.rot_sensitivity,
                    vendor_id=macros.SPACEMOUSE_VENDOR_ID,
                    product_id=macros.SPACEMOUSE_PRODUCT_ID,
                )
            else:
                raise ValueError(selected_device)

            print(f"\n--- Stage {stage_idx + 1}/{len(task.stages)}: {stage.env_name} ---")
            if stage.note:
                print(f"Note: {stage.note}")

            ep_directory, discard_traj = collect_human_trajectory(
                env,
                device,
                args.arm,
                env_configuration="single-arm-opposed",
                mirror_actions=True,
                render=(args.renderer != "mjviewer"),
                max_fr=args.max_fr,
                print_info=True,
            )

            # Save per-stage meta
            stage_meta = {
                "stage_idx": stage_idx,
                "stage_name": stage.name,
                "env_name": stage.env_name,
                "env_kwargs": stage.env_kwargs,
                "note": stage.note,
                "ep_directory": ep_directory,
                "success": not discard_traj,
            }
            scenario_meta["stages"].append(stage_meta)
            with open(stage_dir / "stage_meta.json", "w", encoding="utf-8") as f:
                json.dump(stage_meta, f, ensure_ascii=False, indent=2)

            if ep_directory is not None:
                # hdf5 export (per-stage) - keep only successful trajectories
                if not discard_traj:
                    gather_demonstrations_as_hdf5(
                        directory=str(episodes_dir),
                        out_dir=str(stage_dir),
                        env_info=env_info_json,
                        out_name="demo.hdf5",
                        successful_episodes=[Path(ep_directory).name],
                        verbose=True,
                    )

            env.close()

    with open(scenario_root / "scenario_meta.json", "w", encoding="utf-8") as f:
        json.dump(scenario_meta, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Data saved under: {scenario_root}")


if __name__ == "__main__":
    main()
