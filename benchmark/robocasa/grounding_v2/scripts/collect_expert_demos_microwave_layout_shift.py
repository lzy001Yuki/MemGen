"""
Headless, scripted data collection for `GroundingV2MicrowavePickPlaceLayoutShift`.

This mirrors the general structure of:
  `grounding/scripts/collect_expert_demos_layout_shift.py`

but executes the actual long-horizon task:
  - Stage A: pick from counter -> place into microwave onto a plate
  - Distractors: navigate base to a few fixtures (env-provided targets)
  - Stage B: move microwave (layout shift)
  - Stage C: pick from shifted microwave -> place onto container on counter
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path
from typing import Any

import imageio
import mujoco
import numpy as np
import robosuite
from robosuite.controllers import load_composite_controller_config

import robocasa
import robocasa.utils.env_utils as EnvUtils
import robocasa.utils.object_utils as OU

# Import registers env via robosuite EnvMeta / RoboCasa KitchenEnvMeta.
from grounding_v2.envs.microwave_pick_place_layout_shift import (  # noqa: F401
    GroundingV2MicrowavePickPlaceLayoutShift,
)

# Reuse the battle-tested expert + helpers from v1 grounding.
from grounding.scripts.collect_expert_demos_layout_shift import (  # noqa: E402
    DeltaPoseExpert,
    _render_rgb,
    _save_episode_npz,
    _try_get_model_xml,
)


def _now_compact() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _obj_body_pos(env, obj_name: str) -> np.ndarray:
    body_id = env.obj_body_id[obj_name]
    return np.array(env.sim.data.body_xpos[body_id], dtype=np.float64).copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", type=str, default="pretrain", choices=["pretrain", "target", "all"])
    parser.add_argument("--layout", type=int, nargs="+", default=None)
    parser.add_argument("--style", type=int, nargs="+", default=None)
    parser.add_argument("--obj_groups", type=str, nargs="+", default=None)
    parser.add_argument("--num_distractors", type=int, default=3, choices=[2, 3])
    parser.add_argument("--robots", nargs="+", type=str, default="PandaOmron")
    parser.add_argument("--config", type=str, default="single-arm-opposed")
    parser.add_argument("--controller", type=str, default="WHOLE_BODY_IK")
    parser.add_argument("--control_freq", type=int, default=20)
    parser.add_argument("--arm", type=str, default="right")
    parser.add_argument("--max_pos_step", type=float, default=0.02)
    parser.add_argument("--pos_thresh", type=float, default=0.01)
    parser.add_argument("--settle_steps", type=int, default=10)
    parser.add_argument(
        "--disable_base_nav",
        action="store_true",
        help="Disable mobile-base navigation (not recommended for this task).",
    )
    parser.add_argument("--camera", type=str, default="robot0_agentview_center")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--video_format", type=str, default="mp4", choices=["none", "mp4", "gif", "frames"])
    parser.add_argument("--video_fps", type=int, default=20)
    parser.add_argument("--out_dir", type=str, default="/tmp/robocasa_grounding_v2_expert")
    args = parser.parse_args()
    enable_base_nav = not bool(args.disable_base_nav)

    run_dir = Path(args.out_dir) / f"{_now_compact()}_GroundingV2MicrowavePickPlaceLayoutShift_expert"
    episodes_root = run_dir / "episodes"
    episodes_root.mkdir(parents=True, exist_ok=True)

    controller_config = load_composite_controller_config(
        controller=args.controller,
        robot=args.robots if isinstance(args.robots, str) else args.robots[0],
    )

    env_kwargs: dict[str, Any] = dict(
        env_name="GroundingV2MicrowavePickPlaceLayoutShift",
        robots=args.robots,
        controller_configs=controller_config,
        env_configuration=args.config,
        split=args.split,
        seed=int(args.seed),
        num_distractors=int(args.num_distractors),
        render_onscreen=False,
        randomize_cameras=False,
        camera_names=[args.camera],
        camera_heights=args.height,
        camera_widths=args.width,
        layout_ids=args.layout,
        style_ids=args.style,
        use_camera_obs=False,  # render manually
        control_freq=int(args.control_freq),
    )
    if args.obj_groups is not None:
        env_kwargs["obj_groups"] = args.obj_groups

    env = EnvUtils.create_env(**env_kwargs)

    env_info = json.dumps(
        dict(
            env_name="GroundingV2MicrowavePickPlaceLayoutShift",
            robots=args.robots,
            controller=args.controller,
            env_configuration=args.config,
            split=args.split,
            layout_ids=args.layout,
            style_ids=args.style,
            obj_groups=env_kwargs.get("obj_groups"),
            num_distractors=int(args.num_distractors),
            camera=args.camera,
            height=int(args.height),
            width=int(args.width),
            control_freq=int(args.control_freq),
        )
    )

    manifest: dict[str, Any] = dict(
        env_name="GroundingV2MicrowavePickPlaceLayoutShift",
        created_at=_now_compact(),
        robocasa_version=getattr(robocasa, "__version__", None),
        robosuite_version=getattr(robosuite, "__version__", None),
        mujoco_version=getattr(mujoco, "__version__", None),
        num_episodes=int(args.num_episodes),
        out_dir=str(run_dir),
        config=json.loads(env_info),
        episodes=[],
    )

    try:
        for ep_i in range(int(args.num_episodes)):
            ep_name = f"ep_{ep_i:06d}"
            ep_dir = episodes_root / ep_name
            ep_dir.mkdir(parents=True, exist_ok=True)

            env.reset()

            # Save model + meta
            ep_meta = {}
            try:
                ep_meta = env.get_ep_meta()
            except Exception:
                ep_meta = {}
            _json_dump(ep_dir / "ep_meta.json", ep_meta)

            model_xml = _try_get_model_xml(env)
            if model_xml is not None:
                (ep_dir / "model.xml").write_text(model_xml, encoding="utf-8")

            expert = DeltaPoseExpert(
                env=env,
                arm=str(args.arm),
                max_pos_step=float(args.max_pos_step),
                pos_thresh=float(args.pos_thresh),
                settle_steps=int(args.settle_steps),
            )

            states: list[np.ndarray] = []
            actions: list[np.ndarray] = []

            frames_dir = None
            video_path = None
            video_writer = None
            if args.video_format == "frames":
                frames_dir = ep_dir / "frames"
                frames_dir.mkdir(parents=True, exist_ok=True)
            elif args.video_format == "mp4":
                video_path = ep_dir / "episode.mp4"
                video_writer = imageio.get_writer(video_path, fps=int(args.video_fps))
            elif args.video_format == "gif":
                video_path = ep_dir / "episode.gif"
                video_writer = imageio.get_writer(
                    video_path, mode="I", duration=float(1.0 / max(1, int(args.video_fps)))
                )

            def record_frame(frame_index: int) -> None:
                rgb = _render_rgb(env, args.camera, args.height, args.width)
                if frames_dir is not None:
                    imageio.imwrite(frames_dir / f"frame_{frame_index:06d}.png", rgb)
                if video_writer is not None:
                    video_writer.append_data(rgb)

            # Initial
            states.append(np.array(env.sim.get_state().flatten(), dtype=np.float64).copy())
            frame_i = 0
            if args.video_format != "none":
                record_frame(frame_i)
                frame_i += 1

            # --------------------------
            # Stage A: store into microwave (onto plate)
            # --------------------------
            plate_pos = _obj_body_pos(env, "microwave_plate")
            place_pos = plate_pos.copy()
            try:
                obj_size = np.array(getattr(env.objects["obj"], "size", [0.04, 0.04, 0.04]), dtype=np.float64)
                plate_size = np.array(
                    getattr(env.objects["microwave_plate"], "size", [0.12, 0.12, 0.02]),
                    dtype=np.float64,
                )
                # Aim for the object's center to be just above the plate top surface.
                place_pos[2] = float(plate_pos[2] + plate_size[2] / 2 + obj_size[2] / 2 + 0.005)
            except Exception:
                pass
            _json_dump(ep_dir / "stage_a_targets.json", {"plate_pos": plate_pos.tolist(), "place_pos": place_pos.tolist()})

            for act in expert.pick_object(obj_name="obj"):
                env.step(act)
                actions.append(np.array(act, dtype=np.float32).copy())
                states.append(np.array(env.sim.get_state().flatten(), dtype=np.float64).copy())
                if args.video_format != "none":
                    record_frame(frame_i)
                    frame_i += 1

            for act in expert.place_object_into_fixture(fixture=env.microwave, place_pos=place_pos):
                env.step(act)
                actions.append(np.array(act, dtype=np.float32).copy())
                states.append(np.array(env.sim.get_state().flatten(), dtype=np.float64).copy())
                if args.video_format != "none":
                    record_frame(frame_i)
                    frame_i += 1

            store_ok = False
            try:
                store_ok = bool(env._check_store_success())  # noqa: SLF001 (expert script)
            except Exception:
                store_ok = False

            # --------------------------
            # Distractors: navigate base to env-provided targets (optional)
            # --------------------------
            nav_ok = True
            nav_targets = []
            try:
                nav_targets = [
                    dict(
                        fixture_name=t.fixture_name,
                        target_pos=t.target_pos.tolist(),
                        target_yaw=float(t.target_yaw),
                    )
                    for t in getattr(env, "_nav_targets", [])  # noqa: SLF001
                ]
            except Exception:
                nav_targets = []
            _json_dump(ep_dir / "distractor_nav_targets.json", {"targets": nav_targets})

            if enable_base_nav and nav_targets:
                for t in getattr(env, "_nav_targets", []):  # noqa: SLF001
                    for act in expert.navigate_base_to(t.target_pos, t.target_yaw):
                        env.step(act)
                        actions.append(np.array(act, dtype=np.float32).copy())
                        states.append(np.array(env.sim.get_state().flatten(), dtype=np.float64).copy())
                        if args.video_format != "none":
                            record_frame(frame_i)
                            frame_i += 1

            # --------------------------
            # Stage B: layout shift (microwave moves)
            # --------------------------
            shift_info = {}
            shift_err = None
            try:
                shift_info = env.apply_layout_shift(obj_names=("obj", "microwave_plate")) or {}
            except Exception as e:
                shift_err = repr(e)
            _json_dump(ep_dir / "layout_shift_info.json", shift_info)
            if shift_err is not None:
                _json_dump(ep_dir / "layout_shift_error.json", {"error": shift_err})

            # Capture 1 frame right after shift with an idle action
            idle_arm = np.zeros(expert.arm_cmd_dim, dtype=np.float32)
            idle_base = np.zeros(expert.base_cmd_dim, dtype=np.float32) if expert.has_base else None
            act = expert._make_action(arm_cmd=idle_arm, gripper_cmd=None, base_cmd=idle_base, base_mode=-1.0)
            env.step(act)
            actions.append(np.array(act, dtype=np.float32).copy())
            states.append(np.array(env.sim.get_state().flatten(), dtype=np.float64).copy())
            if args.video_format != "none":
                record_frame(frame_i)
                frame_i += 1

            # Optionally navigate to the shifted microwave before retrieval
            if enable_base_nav:
                try:
                    nav_pos, nav_ori = EnvUtils.compute_robot_base_placement_pose(env, env.microwave)
                    nav_pos = np.array(nav_pos, dtype=np.float64).reshape(3)
                    nav_yaw = float(np.array(nav_ori, dtype=np.float64).reshape(-1)[2])
                    _json_dump(ep_dir / "post_shift_nav_target.json", {"pos": nav_pos.tolist(), "yaw": nav_yaw})
                    for act in expert.navigate_base_to(nav_pos, nav_yaw):
                        env.step(act)
                        actions.append(np.array(act, dtype=np.float32).copy())
                        states.append(np.array(env.sim.get_state().flatten(), dtype=np.float64).copy())
                        if args.video_format != "none":
                            record_frame(frame_i)
                            frame_i += 1
                except Exception:
                    pass

            # --------------------------
            # Stage C: retrieve from microwave, place onto container
            # --------------------------
            for act in expert.pick_object_from_fixture(fixture=env.microwave, obj_name="obj"):
                env.step(act)
                actions.append(np.array(act, dtype=np.float32).copy())
                states.append(np.array(env.sim.get_state().flatten(), dtype=np.float64).copy())
                if args.video_format != "none":
                    record_frame(frame_i)
                    frame_i += 1

            cont_pos = _obj_body_pos(env, "container")
            place_on_container = cont_pos.copy()
            _json_dump(ep_dir / "stage_c_targets.json", {"container_pos": cont_pos.tolist(), "place_pos": place_on_container.tolist()})

            for act in expert.place_object(place_pos=place_on_container):
                env.step(act)
                actions.append(np.array(act, dtype=np.float32).copy())
                states.append(np.array(env.sim.get_state().flatten(), dtype=np.float64).copy())
                if args.video_format != "none":
                    record_frame(frame_i)
                    frame_i += 1

            retrieve_ok = False
            try:
                retrieve_ok = bool(OU.check_obj_in_receptacle(env, "obj", "container"))
            except Exception:
                retrieve_ok = False

            if video_writer is not None:
                try:
                    video_writer.close()
                except Exception:
                    pass

            _save_episode_npz(ep_dir, env_name="GroundingV2MicrowavePickPlaceLayoutShift", states=states, actions=actions)
            _json_dump(ep_dir / "ep_stats.json", dict(store_success=store_ok, nav_success=nav_ok, retrieve_success=retrieve_ok))

            manifest["episodes"].append(
                dict(
                    name=ep_name,
                    path=str(ep_dir),
                    steps=len(actions),
                    store_success=store_ok,
                    retrieve_success=retrieve_ok,
                    video=str(video_path) if video_path is not None else None,
                    frames_dir=str(frames_dir) if frames_dir is not None else None,
                )
            )

            print(f"[{ep_name}] steps={len(actions)} store={store_ok} retrieve={retrieve_ok} saved={ep_dir}")

    finally:
        try:
            env.close()
        except Exception:
            pass

    _json_dump(run_dir / "manifest.json", manifest)
    print(f"Done. Dataset root: {run_dir}")


if __name__ == "__main__":
    main()
