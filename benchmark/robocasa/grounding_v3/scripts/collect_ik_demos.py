from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from grounding_v3.policies import IKPickPlacePolicy
from grounding_v3.tasks import get_task_spec
from grounding_v3.utils import make_robocasa_env


def _timestamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_rmtree(path: str | Path) -> None:
    p = Path(path)
    if p.exists() and p.is_dir():
        shutil.rmtree(p)


def _get_microwave_pos(env: Any) -> list[float] | None:
    try:
        from robocasa.models.fixtures import FixtureType  # type: ignore

        micro = env.get_fixture(FixtureType.MICROWAVE)
        return [float(x) for x in micro.pos]
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="microwave_relocation_v1")
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "ik"),
        help="Output directory root",
    )
    parser.add_argument("--controller", type=str, default="OSC_POSE")
    parser.add_argument("--renderer", type=str, default="mujoco", choices=["mjviewer", "mujoco"])
    parser.add_argument("--camera", type=str, default="robot0_frontview")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--num-scenarios", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=50)
    args = parser.parse_args()

    task = get_task_spec(args.task)

    # Late imports
    from robosuite.wrappers import DataCollectionWrapper  # type: ignore
    from robocasa.scripts.collect_demos import gather_demonstrations_as_hdf5  # type: ignore

    out_root = Path(args.out).resolve()
    _ensure_dir(out_root)

    run_root = out_root / f"{_timestamp()}_{task.name}"
    _ensure_dir(run_root)

    scenario_meta: dict[str, Any] = {
        "task": task.name,
        "note": task.note,
        "created_at": datetime.datetime.now().isoformat(),
        "policy": "IKPickPlacePolicy",
        "controller": args.controller,
        "stages": [],
    }

    policy = IKPickPlacePolicy()

    for scenario_idx in range(args.num_scenarios):
        print(f"\n=== Auto scenario {scenario_idx + 1}/{args.num_scenarios}: {task.name} ===")

        microwave_positions_this_scenario: list[list[float] | None] = []

        for stage_idx, stage in enumerate(task.stages):
            stage_dir = run_root / f"{scenario_idx:04d}" / f"{stage_idx:02d}_{stage.name}_{stage.env_name}"
            _ensure_dir(stage_dir)
            episodes_dir = stage_dir / "episodes"
            _ensure_dir(episodes_dir)

            retries = 0
            success = False
            last_ep_dir = None
            last_micro_pos = None

            while retries < args.max_retries and not success:
                retries += 1

                env, env_info_json = make_robocasa_env(
                    env_name=stage.env_name,
                    controller=args.controller,
                    camera=args.camera,
                    renderer=args.renderer,
                    has_renderer=args.render,
                    has_offscreen_renderer=False,
                    use_camera_obs=False,
                    ignore_done=True,
                    control_freq=20,
                    seed=(None if args.seed is None else args.seed + scenario_idx * 100 + stage_idx * 10 + retries),
                    env_kwargs=stage.env_kwargs,
                )

                env = DataCollectionWrapper(env, str(episodes_dir), use_env_xml_for_reset=True)

                env.reset()
                last_ep_dir = getattr(env, "ep_directory", None)
                last_micro_pos = _get_microwave_pos(env)

                # Run policy
                success = policy.run(env, max_steps=600, render=args.render)

                # Hold a few steps so _check_success latches in datasets
                if success:
                    for _ in range(15):
                        env.step(np.zeros(env.action_dim))
                        if args.render:
                            env.render()

                # If failed, delete the recorded episode directory to keep dataset 100% success
                if not success and last_ep_dir is not None:
                    _safe_rmtree(last_ep_dir)

                env.close()

            stage_record = {
                "scenario_idx": scenario_idx,
                "stage_idx": stage_idx,
                "stage_name": stage.name,
                "env_name": stage.env_name,
                "env_kwargs": stage.env_kwargs,
                "note": stage.note,
                "success": success,
                "retries": retries,
                "microwave_pos": last_micro_pos,
                "last_ep_directory": last_ep_dir,
            }
            scenario_meta["stages"].append(stage_record)
            microwave_positions_this_scenario.append(last_micro_pos)

            with open(stage_dir / "stage_meta.json", "w", encoding="utf-8") as f:
                json.dump(stage_record, f, ensure_ascii=False, indent=2)

            if not success:
                raise RuntimeError(
                    f"Failed to collect a successful demo after {args.max_retries} retries for "
                    f"stage {stage.env_name} (stage_dir={stage_dir})"
                )

            # Per-stage hdf5 export
            gather_demonstrations_as_hdf5(
                directory=str(episodes_dir),
                out_dir=str(stage_dir),
                env_info=env_info_json,
                out_name="demo.hdf5",
                successful_episodes=None,
                verbose=False,
            )

        # Sanity check: time-a vs time-b microwave pose should differ
        if len(microwave_positions_this_scenario) >= 2:
            micro_a = microwave_positions_this_scenario[0]
            micro_b = microwave_positions_this_scenario[-1]
            if micro_a is not None and micro_b is not None:
                if float(sum((np.array(micro_a) - np.array(micro_b)) ** 2)) < 1e-8:
                    raise RuntimeError(
                        "Expected microwave position to change between time-a and time-b, "
                        f"but got identical poses: a={micro_a}, b={micro_b}"
                    )

    with open(run_root / "scenario_meta.json", "w", encoding="utf-8") as f:
        json.dump(scenario_meta, f, ensure_ascii=False, indent=2)

    print(f"\nDone. IK demos saved under: {run_root}")


if __name__ == "__main__":
    main()
