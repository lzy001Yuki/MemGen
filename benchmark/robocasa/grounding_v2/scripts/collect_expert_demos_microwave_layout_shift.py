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
from grounding.scripts.collect_data import (  # noqa: E402
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
    parser.add_argument(
        "--mw_place_approach_dist",
        type=float,
        default=0.25,
        help="Stage A: approach distance (outside) when placing into microwave.",
    )
    parser.add_argument(
        "--mw_place_insert_dist",
        type=float,
        default=-0.02,
        help="Stage A: insert distance (near plate) when placing into microwave. Can be negative to push deeper.",
    )
    parser.add_argument(
        "--mw_place_insert_dist_candidates",
        type=float,
        nargs="+",
        default=None,
        help="Optional override: a list of insert distances to try (in order) for Stage A placement retries.",
    )
    parser.add_argument(
        "--mw_place_pre_z",
        type=float,
        default=0.04,
        help="Stage A: z offset for the pre-place pose when placing into microwave.",
    )
    parser.add_argument(
        "--mw_pick_approach_dist",
        type=float,
        default=0.25,
        help="Stage C: approach distance (outside) when picking from microwave.",
    )
    parser.add_argument(
        "--mw_pick_insert_dist",
        type=float,
        default=0.06,
        help="Stage C: insert distance (near object) when picking from microwave.",
    )
    parser.add_argument(
        "--mw_pick_pre_z",
        type=float,
        default=0.04,
        help="Stage C: z offset for the pre-grasp pose when picking from microwave.",
    )
    parser.add_argument(
        "--mw_pick_lift_z",
        type=float,
        default=0.18,
        help="Stage C: lift distance after grasping the object from microwave.",
    )
    parser.add_argument(
        "--debug_controller",
        action="store_true",
        help="Print controller interface details (input_type / cmd_dim / scales).",
    )
    parser.add_argument(
        "--pick_max_attempts",
        type=int,
        default=6,
        help="Stage A: max grasp retries (with XY jitter) when picking the object.",
    )
    parser.add_argument(
        "--pick_jitter_xy",
        type=float,
        default=0.015,
        help="Stage A: XY jitter magnitude used across grasp retries.",
    )
    parser.add_argument(
        "--stage_a_max_attempts",
        type=int,
        default=4,
        help="Stage A: max store attempts (pick + place) before giving up.",
    )
    parser.add_argument(
        "--shift_nearest_if_static_base",
        action="store_true",
        help="If base cannot be actuated, move microwave to the nearest valid counter instead of the farthest.",
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
            if bool(args.debug_controller):
                print(
                    "[controller] arm_input_type=%s arm_cmd_dim=%s arm_action_scale=%s has_base=%s base_cmd_dim=%s base_input_type=%s"
                    % (
                        getattr(expert, "arm_input_type", None),
                        getattr(expert, "arm_cmd_dim", None),
                        getattr(expert, "arm_action_scale", None),
                        getattr(expert, "has_base", None),
                        getattr(expert, "base_cmd_dim", None),
                        getattr(expert, "base_input_type", None),
                    )
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

            # Retry schedule for Stage A "place into microwave".
            if args.mw_place_insert_dist_candidates is not None:
                insert_candidates = [float(x) for x in args.mw_place_insert_dist_candidates]
            else:
                base_ins = float(args.mw_place_insert_dist)
                insert_candidates = [
                    base_ins,
                    base_ins - 0.04,
                    base_ins - 0.08,
                    base_ins + 0.04,
                ]
            # De-dup while keeping order
            _seen = set()
            insert_candidates = [x for x in insert_candidates if not (x in _seen or _seen.add(x))]

            store_ok = False
            stage_a_attempts: list[dict[str, Any]] = []

            # Used for short idle / settle steps
            idle_arm = np.zeros(expert.arm_cmd_dim, dtype=np.float32)
            idle_base = np.zeros(expert.base_cmd_dim, dtype=np.float32) if expert.has_base else None

            for attempt_i in range(int(args.stage_a_max_attempts)):
                ins = insert_candidates[attempt_i % len(insert_candidates)]

                # If the object accidentally ended up inside the microwave (e.g., failed place),
                # retrieve it using the fixture-aware primitive; otherwise, use a robust top-down grasp.
                obj_inside_mw = False
                try:
                    obj_inside_mw = bool(OU.obj_inside_of(env, "obj", env.microwave, partial_check=True))
                except Exception:
                    obj_inside_mw = False

                if obj_inside_mw:
                    pick_gen = expert.pick_object_from_fixture(
                        fixture=env.microwave,
                        obj_name="obj",
                        approach_dist=float(args.mw_pick_approach_dist),
                        insert_dist=float(args.mw_pick_insert_dist),
                        lift_z=float(args.mw_pick_lift_z),
                        pre_z=float(args.mw_pick_pre_z),
                    )
                else:
                    pick_gen = expert.pick_object_robust(
                        obj_name="obj",
                        max_attempts=int(args.pick_max_attempts),
                        jitter_xy=float(args.pick_jitter_xy),
                    )

                for act in pick_gen:
                    env.step(act)
                    actions.append(np.array(act, dtype=np.float32).copy())
                    states.append(np.array(env.sim.get_state().flatten(), dtype=np.float64).copy())
                    if args.video_format != "none":
                        record_frame(frame_i)
                        frame_i += 1

                for act in expert.place_object_into_fixture(
                    fixture=env.microwave,
                    place_pos=place_pos,
                    approach_dist=float(args.mw_place_approach_dist),
                    insert_dist=float(ins),
                    pre_z=float(args.mw_place_pre_z),
                ):
                    env.step(act)
                    actions.append(np.array(act, dtype=np.float32).copy())
                    states.append(np.array(env.sim.get_state().flatten(), dtype=np.float64).copy())
                    if args.video_format != "none":
                        record_frame(frame_i)
                        frame_i += 1

                # Idle a bit to let contacts settle (important for contact-based success checks).
                for _ in range(8):
                    idle_act = expert._make_action(
                        arm_cmd=idle_arm,
                        gripper_cmd=None,
                        base_cmd=idle_base,
                        base_mode=-1.0,
                    )
                    env.step(idle_act)
                    actions.append(np.array(idle_act, dtype=np.float32).copy())
                    states.append(np.array(env.sim.get_state().flatten(), dtype=np.float64).copy())
                    if args.video_format != "none":
                        record_frame(frame_i)
                        frame_i += 1

                try:
                    store_ok = bool(env._check_store_success())  # noqa: SLF001 (expert script)
                except Exception:
                    store_ok = False

                dbg: dict[str, Any] = dict(
                    attempt=int(attempt_i),
                    insert_dist=float(ins),
                    obj_inside_mw_before_pick=bool(obj_inside_mw),
                    store_ok=bool(store_ok),
                )
                try:
                    dbg["obj_grasped_after_place"] = bool(OU.check_obj_grasped(env, "obj"))
                except Exception:
                    dbg["obj_grasped_after_place"] = None
                try:
                    dbg["obj_pos"] = _obj_body_pos(env, "obj").tolist()
                    dbg["plate_pos"] = _obj_body_pos(env, "microwave_plate").tolist()
                except Exception:
                    pass
                try:
                    dbg["obj_plate_contact"] = bool(
                        env.check_contact(env.objects["obj"], env.objects["microwave_plate"])
                    )
                except Exception:
                    dbg["obj_plate_contact"] = None
                try:
                    dbg["gripper_obj_far"] = bool(OU.gripper_obj_far(env, "obj"))
                except Exception:
                    dbg["gripper_obj_far"] = None
                try:
                    dbg["obj_inside_mw_after_place"] = bool(
                        OU.obj_inside_of(env, "obj", env.microwave, partial_check=True)
                    )
                except Exception:
                    dbg["obj_inside_mw_after_place"] = None

                stage_a_attempts.append(dbg)
                _json_dump(ep_dir / "stage_a_attempts.json", {"attempts": stage_a_attempts})

                if store_ok:
                    break

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
            # If the base cannot be actuated, constrain the shift target to keep the post-shift
            # retrieval reachable (similar to grounding v1 expert).
            if bool(args.shift_nearest_if_static_base) and (not enable_base_nav or expert.base_cmd_dim < 2):
                try:
                    cand = env._find_microwave_shift_counters()  # noqa: SLF001
                    mw_xy = np.array(getattr(env.microwave, "pos", [0.0, 0.0, 0.0]), dtype=np.float64)[:2]

                    def _dist(name: str) -> float:
                        fx = env.get_fixture(name, full_name_check=True)
                        if fx is None:
                            return float("inf")
                        pos = np.array(getattr(fx, "pos", [0.0, 0.0, 0.0]), dtype=np.float64)[:2]
                        return float(np.linalg.norm(pos - mw_xy))

                    reordered = sorted(list(cand), key=_dist)
                    env._mw_shift_counter_names = reordered  # noqa: SLF001
                    _json_dump(
                        ep_dir / "layout_shift_plan.json",
                        dict(
                            mode="nearest_first",
                            reason="static_base_or_base_nav_disabled",
                            candidates=cand,
                            reordered=reordered,
                        ),
                    )
                except Exception as e:
                    _json_dump(ep_dir / "layout_shift_plan.json", {"error": repr(e)})

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
            try:
                env.microwave.open_door(env=env)
            except Exception:
                pass

            for act in expert.pick_object_from_fixture(
                fixture=env.microwave,
                obj_name="obj",
                approach_dist=float(args.mw_pick_approach_dist),
                insert_dist=float(args.mw_pick_insert_dist),
                lift_z=float(args.mw_pick_lift_z),
                pre_z=float(args.mw_pick_pre_z),
            ):
                env.step(act)
                actions.append(np.array(act, dtype=np.float32).copy())
                states.append(np.array(env.sim.get_state().flatten(), dtype=np.float64).copy())
                if args.video_format != "none":
                    record_frame(frame_i)
                    frame_i += 1

            cont_pos = _obj_body_pos(env, "container")
            place_on_container = cont_pos.copy()
            try:
                obj_size = np.array(getattr(env.objects["obj"], "size", [0.04, 0.04, 0.04]), dtype=np.float64)
                cont_size = np.array(getattr(env.objects["container"], "size", [0.12, 0.12, 0.06]), dtype=np.float64)
                place_on_container[2] = float(cont_pos[2] + cont_size[2] / 2 + obj_size[2] / 2 + 0.005)
            except Exception:
                pass
            _json_dump(
                ep_dir / "stage_c_targets.json",
                {"container_pos": cont_pos.tolist(), "place_pos": place_on_container.tolist()},
            )

            for act in expert.place_object(place_pos=place_on_container):
                env.step(act)
                actions.append(np.array(act, dtype=np.float32).copy())
                states.append(np.array(env.sim.get_state().flatten(), dtype=np.float64).copy())
                if args.video_format != "none":
                    record_frame(frame_i)
                    frame_i += 1

            # Idle a bit to stabilize contacts
            for _ in range(10):
                idle_act = expert._make_action(arm_cmd=idle_arm, gripper_cmd=None, base_cmd=idle_base, base_mode=-1.0)
                env.step(idle_act)
                actions.append(np.array(idle_act, dtype=np.float32).copy())
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
