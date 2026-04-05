from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class IKPickPlaceConfig:
    # Motion parameters
    pos_gain: float = 4.0
    max_dpos: float = 0.05
    max_drot: float = 0.2

    # Heights
    pregrasp_height: float = 0.15
    grasp_height: float = 0.02
    lift_height: float = 0.20
    preplace_height: float = 0.18
    place_height: float = 0.03

    # Termination thresholds
    pos_tol: float = 0.015

    # Gripper probing
    gripper_probe_steps: int = 8

    # Optional pose fine-tuning (RobotWin-style)
    # These are local-frame XYZ offsets applied at grasp / place targets.
    # Set them if you see systematic bias (e.g., always grasping "too far forward").
    grasp_offset_local: tuple[float, float, float] = (0.0, 0.0, 0.0)
    place_offset_local: tuple[float, float, float] = (0.0, 0.0, 0.0)


class IKPickPlacePolicy:
    """
    A simple, controller-agnostic "IK-style" scripted policy for RoboCasa pick-place
    tasks that expose:
      - a grasp target object named "obj"
      - an optional receptacle object named "container"
      - or a sink fixture (for counter->sink tasks)

    It drives the robot via the env's controller interface (delta or absolute input),
    using repeated small steps until the EE reaches target positions.

    This is *not* a global motion planner; to get 100% dataset success, the
    collection script should wrap this in a retry loop.
    """

    def __init__(self, cfg: IKPickPlaceConfig | None = None):
        self.cfg = cfg or IKPickPlaceConfig()
        self._gripper_open_action: float | None = None
        self._gripper_close_action: float | None = None
        self._last_gripper_action: float | None = None

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(
        self,
        env: Any,
        *,
        max_steps: int = 500,
        render: bool = False,
    ) -> bool:
        """
        Executes a full pick-and-place attempt for the current env instance.

        Returns:
            bool: env._check_success() at the end
        """

        # Ensure we have the naming assumptions
        if "obj" not in env.objects:
            raise KeyError("Expected env.objects['obj'] to exist for pick-place policy")

        robot = env.robots[0]
        arm = self._select_arm(env)

        # Calibrate gripper actions once per env reset
        self._calibrate_gripper(env, arm)

        obj_pos = self._get_body_pos(env, "obj")
        ee_pos = self._get_eef_pos(env, arm)
        ee_quat_xyzw = self._get_eef_quat_xyzw(env, arm)
        ee_rot = self._get_eef_rot_mat(env, arm)

        # 1) Open gripper and move above object
        self._move_gripper(env, arm, open_gripper=True, render=render)
        pregrasp = obj_pos.copy()
        pregrasp[2] = max(pregrasp[2], ee_pos[2]) + self.cfg.pregrasp_height
        self._move_eef_to(
            env,
            arm,
            target_pos=pregrasp,
            target_quat_xyzw=ee_quat_xyzw,
            render=render,
            max_steps=max_steps // 4,
        )

        # 2) Descend to grasp
        grasp = obj_pos.copy()
        grasp[2] = obj_pos[2] + self.cfg.grasp_height
        grasp = grasp + ee_rot @ np.array(self.cfg.grasp_offset_local, dtype=float)
        self._move_eef_to(
            env,
            arm,
            target_pos=grasp,
            target_quat_xyzw=ee_quat_xyzw,
            render=render,
            max_steps=max_steps // 4,
        )

        # 3) Close gripper and lift
        self._move_gripper(env, arm, open_gripper=False, render=render)
        lift = grasp.copy()
        lift[2] = lift[2] + self.cfg.lift_height
        self._move_eef_to(
            env,
            arm,
            target_pos=lift,
            target_quat_xyzw=ee_quat_xyzw,
            render=render,
            max_steps=max_steps // 4,
        )

        # 4) Move above place target
        place_xy = self._infer_place_target_xy(env)
        preplace = np.array([place_xy[0], place_xy[1], lift[2]], dtype=float)
        preplace[2] = preplace[2] + self.cfg.preplace_height
        preplace = preplace + ee_rot @ np.array(self.cfg.place_offset_local, dtype=float)
        self._move_eef_to(
            env,
            arm,
            target_pos=preplace,
            target_quat_xyzw=ee_quat_xyzw,
            render=render,
            max_steps=max_steps // 4,
        )

        # 5) Descend and release
        place = preplace.copy()
        place[2] = self._infer_place_height(env) + self.cfg.place_height
        self._move_eef_to(
            env,
            arm,
            target_pos=place,
            target_quat_xyzw=ee_quat_xyzw,
            render=render,
            max_steps=max_steps // 4,
        )
        self._move_gripper(env, arm, open_gripper=True, render=render)

        # 6) Retract upward
        retract = place.copy()
        retract[2] = retract[2] + 0.12
        self._move_eef_to(
            env,
            arm,
            target_pos=retract,
            target_quat_xyzw=ee_quat_xyzw,
            render=render,
            max_steps=max_steps // 6,
        )

        # Give physics a moment to settle
        self._hold_still(env, arm, n_steps=10, render=render)

        return bool(env._check_success())

    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------

    def _select_arm(self, env: Any) -> str:
        robot = env.robots[0]
        arms = getattr(robot, "arms", ["right"])
        if len(arms) == 1:
            return arms[0]

        # Mimic user-provided heuristic: use right if object.x > 0 else left
        obj_pos = self._get_body_pos(env, "obj")
        return "right" if obj_pos[0] > 0 else "left"

    def _calibrate_gripper(self, env: Any, arm: str) -> None:
        """
        Tries to infer which scalar action opens vs closes the gripper.
        Falls back to (+1=open, -1=close) if probing isn't possible.
        """

        if self._gripper_open_action is not None and self._gripper_close_action is not None:
            return

        # Default assumption (robosuite convention in many configs)
        open_action = 1.0
        close_action = -1.0

        robot = env.robots[0]
        gripper = robot.gripper.get(arm, None) if hasattr(robot, "gripper") else None
        joint_names = getattr(gripper, "joints", None)
        if not joint_names:
            self._gripper_open_action = open_action
            self._gripper_close_action = close_action
            return

        def read_abs_qpos() -> float:
            vals = []
            for jn in joint_names:
                try:
                    vals.append(float(env.sim.data.get_joint_qpos(jn)))
                except Exception:
                    return float("nan")
            return float(np.mean(np.abs(vals))) if vals else float("nan")

        base_action = self._empty_action_dict(env, arm)
        dof = self._get_gripper_dof(env, arm)

        # Probe open
        for _ in range(self.cfg.gripper_probe_steps):
            a = dict(base_action)
            a[f"{arm}_gripper"] = np.repeat([open_action], dof).astype(float)
            env.step(self._action_dict_to_action(env, a))
        q_open = read_abs_qpos()

        # Probe close
        for _ in range(self.cfg.gripper_probe_steps):
            a = dict(base_action)
            a[f"{arm}_gripper"] = np.repeat([close_action], dof).astype(float)
            env.step(self._action_dict_to_action(env, a))
        q_close = read_abs_qpos()

        if np.isfinite(q_open) and np.isfinite(q_close) and q_open != q_close:
            # Larger finger joint magnitude generally corresponds to "open"
            if q_open > q_close:
                self._gripper_open_action, self._gripper_close_action = (
                    open_action,
                    close_action,
                )
            else:
                self._gripper_open_action, self._gripper_close_action = (
                    close_action,
                    open_action,
                )
        else:
            self._gripper_open_action = open_action
            self._gripper_close_action = close_action

        # After probing, remember "open" as the default commanded state
        self._last_gripper_action = self._gripper_open_action

    def _infer_place_target_xy(self, env: Any) -> np.ndarray:
        if "container" in env.objects:
            pos = self._get_body_pos(env, "container")
            return pos[:2]

        # Sink tasks: place into sink basin center (approx)
        if hasattr(env, "sink"):
            sink = env.sink
        else:
            # try fixture_refs / fixtures lookup
            sink = getattr(env, "fixtures", {}).get("sink", None)

        if sink is not None:
            try:
                regions = sink.get_reset_regions(env=env)
                # choose the first region
                reg = next(iter(regions.values()))
                offset = np.array(reg["offset"], dtype=float)
                # local->world via yaw rot
                import robosuite.utils.transform_utils as T  # type: ignore

                R = T.euler2mat(np.array([0.0, 0.0, float(getattr(sink, "rot", 0.0))]))
                world = np.array(sink.pos, dtype=float) + R @ offset
                return world[:2]
            except Exception:
                return np.array(sink.pos[:2], dtype=float)

        # Fallback: current EE xy
        robot = env.robots[0]
        arm = self._select_arm(env)
        return self._get_eef_pos(env, arm)[:2]

    def _infer_place_height(self, env: Any) -> float:
        if "container" in env.objects:
            pos = self._get_body_pos(env, "container")
            return float(pos[2])
        if hasattr(env, "sink"):
            return float(np.array(env.sink.pos)[2])
        return float(self._get_body_pos(env, "obj")[2])

    def _get_body_pos(self, env: Any, obj_name: str) -> np.ndarray:
        return np.array(env.sim.data.body_xpos[env.obj_body_id[obj_name]], dtype=float)

    def _get_eef_pos(self, env: Any, arm: str) -> np.ndarray:
        site_id = env.robots[0].eef_site_id[arm]
        return np.array(env.sim.data.site_xpos[site_id], dtype=float)

    def _get_eef_quat_xyzw(self, env: Any, arm: str) -> np.ndarray:
        site_id = env.robots[0].eef_site_id[arm]
        mat = np.array(env.sim.data.site_xmat[site_id]).reshape(3, 3)

        import robosuite.utils.transform_utils as T  # type: ignore

        quat_wxyz = T.mat2quat(mat)
        quat_xyzw = T.convert_quat(quat_wxyz, to="xyzw")
        return np.array(quat_xyzw, dtype=float)

    def _get_eef_rot_mat(self, env: Any, arm: str) -> np.ndarray:
        site_id = env.robots[0].eef_site_id[arm]
        return np.array(env.sim.data.site_xmat[site_id]).reshape(3, 3).astype(float)

    def _empty_action_dict(self, env: Any, arm: str) -> dict[str, Any]:
        action_dict: dict[str, Any] = {}

        # Base controller (if present)
        robot = env.robots[0]
        part_controllers = getattr(robot, "part_controllers", {})
        if "base" in part_controllers:
            action_dict["base_mode"] = -1
            action_dict["base"] = np.zeros(part_controllers["base"].control_dim)

        # Keep other arm zeros if bimanual
        arms = getattr(robot, "arms", [arm])
        for a in arms:
            if a in part_controllers:
                action_dict[a] = np.zeros(part_controllers[a].control_dim)

        # Gripper: keep last commanded value if available (important for lifting while grasped)
        dof = self._get_gripper_dof(env, arm)
        if self._last_gripper_action is None:
            action_dict[f"{arm}_gripper"] = np.zeros(dof, dtype=float)
        else:
            action_dict[f"{arm}_gripper"] = np.repeat([self._last_gripper_action], dof).astype(
                float
            )
        return action_dict

    def _action_dict_to_action(self, env: Any, action_dict: dict[str, Any]) -> np.ndarray:
        robot = env.robots[0]
        return robot.create_action_vector(action_dict)

    def _move_gripper(self, env: Any, arm: str, *, open_gripper: bool, render: bool) -> None:
        assert self._gripper_open_action is not None and self._gripper_close_action is not None
        g = self._gripper_open_action if open_gripper else self._gripper_close_action
        self._last_gripper_action = g
        base_action = self._empty_action_dict(env, arm)
        dof = self._get_gripper_dof(env, arm)
        for _ in range(12):
            a = dict(base_action)
            a[f"{arm}_gripper"] = np.repeat([g], dof).astype(float)
            env.step(self._action_dict_to_action(env, a))
            if render:
                env.render()

    def _hold_still(self, env: Any, arm: str, *, n_steps: int, render: bool) -> None:
        base_action = self._empty_action_dict(env, arm)
        action = self._action_dict_to_action(env, base_action)
        for _ in range(n_steps):
            env.step(action)
            if render:
                env.render()

    def _move_eef_to(
        self,
        env: Any,
        arm: str,
        *,
        target_pos: np.ndarray,
        target_quat_xyzw: np.ndarray,
        max_steps: int,
        render: bool,
    ) -> None:
        robot = env.robots[0]
        ctrl = robot.part_controllers[arm]

        base_action = self._empty_action_dict(env, arm)

        for _ in range(max_steps):
            cur = self._get_eef_pos(env, arm)
            err = target_pos - cur
            if np.linalg.norm(err) <= self.cfg.pos_tol:
                break

            action_dict = dict(base_action)

            if ctrl.input_type == "delta":
                dpos = np.clip(self.cfg.pos_gain * err, -self.cfg.max_dpos, self.cfg.max_dpos)
                # Some configs may be position-only (3D). Default to zeros for remaining dims.
                arm_action = np.zeros(ctrl.control_dim, dtype=float)
                arm_action[: min(3, ctrl.control_dim)] = dpos[: min(3, ctrl.control_dim)]
                if ctrl.control_dim >= 6:
                    # keep rotation deltas at 0
                    pass
                action_dict[arm] = arm_action
            elif ctrl.input_type == "absolute":
                # Try common conventions:
                #  - 7D pose: [x,y,z,qx,qy,qz,qw] (xyzw)
                #  - 6D pose: [x,y,z,ax,ay,az] (axis-angle)
                if ctrl.control_dim == 7:
                    arm_action = np.concatenate([target_pos, target_quat_xyzw], dtype=float)
                elif ctrl.control_dim == 6:
                    # keep orientation fixed (0 axis-angle)
                    arm_action = np.concatenate([target_pos, np.zeros(3)], dtype=float)
                else:
                    # fallback: zeros, at least keep stepping
                    arm_action = np.zeros(ctrl.control_dim, dtype=float)
                action_dict[arm] = arm_action
            else:
                # Unknown controller type
                action_dict[arm] = np.zeros(ctrl.control_dim, dtype=float)

            env.step(self._action_dict_to_action(env, action_dict))
            if render:
                env.render()

    def _get_gripper_dof(self, env: Any, arm: str) -> int:
        robot = env.robots[0]
        try:
            return int(robot.gripper[arm].dof)  # type: ignore[attr-defined]
        except Exception:
            return 1
