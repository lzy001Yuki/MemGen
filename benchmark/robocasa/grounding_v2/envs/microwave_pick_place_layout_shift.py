from __future__ import annotations

"""
Grounding v2: Microwave pick-place with layout shift + distractors.

This environment "assembles" existing atomic tasks into a single long-horizon
episode:

  (A) PickPlaceCounterToMicrowave
      - pick `obj` from a counter and place it onto a plate inside the microwave

  (Distractors) 2-3 NavigateKitchen-style subtasks
      - navigate the mobile base to a few (random) fixtures

  (B) Layout shift
      - move the microwave to a distant counter (runtime translation)

  (C) PickPlaceMicrowaveToCounter
      - retrieve `obj` from the moved microwave and place it onto a container on
        the counter

Notes:
  - The visual appearance / textures are untouched; we only change runtime body
    positions for the microwave at the shift moment.
  - Layout shift logic is adapted from `grounding/envs/cabinet_layout_shift_demo.py`.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

import robocasa.utils.env_utils as EnvUtils
import robocasa.utils.object_utils as OU
from robocasa.environments.kitchen.kitchen import Kitchen
from robocasa.models.fixtures import FixtureType

try:
    # robosuite dependency (available in a normal RoboCasa install)
    from robosuite.utils.mjcf_utils import array_to_string
    import robosuite.utils.transform_utils as T
except Exception as e:  # pragma: no cover
    array_to_string = None
    T = None
    _robosuite_import_error = e


@dataclass(frozen=True)
class _NavTarget:
    fixture_name: str
    target_pos: np.ndarray  # (3,)
    target_yaw: float


class GroundingV2MicrowavePickPlaceLayoutShift(Kitchen):
    """
    Composite long-horizon task described in the module docstring.
    """

    # Keep consistent with microwave pick/place atomics (layout 9: microwave far from counters)
    EXCLUDE_LAYOUTS = [9]

    # Stage identifiers
    STAGE_A_STORE = 0
    STAGE_DISTRACTORS = 1
    STAGE_SHIFT = 2
    STAGE_B_RETRIEVE = 3

    def __init__(
        self,
        obj_groups: str | tuple[str, ...] = "food",
        exclude_obj_groups: str | tuple[str, ...] | None = None,
        num_distractors: int = 3,
        distractor_min_dist: float = 0.8,
        *args,
        **kwargs,
    ):
        if num_distractors not in (2, 3):
            raise ValueError("num_distractors must be 2 or 3")

        self.obj_groups = obj_groups
        self.exclude_obj_groups = exclude_obj_groups
        self.num_distractors = int(num_distractors)
        self.distractor_min_dist = float(distractor_min_dist)

        # Per-episode runtime state (initialized in _setup_scene)
        self.stage: int = self.STAGE_A_STORE
        self._nav_targets: list[_NavTarget] = []
        self._nav_index: int = 0

        self.layout_shift_done: bool = False
        self.layout_shift_info: dict[str, Any] = {}
        self._mw_shift_counter_names: list[str] | None = None
        self._mw_shift_counter_name_chosen: str | None = None

        super().__init__(*args, **kwargs)

    # ---------------------------------------------------------------------
    # Kitchen references + object configs (adapted from atomic pick/place)
    # ---------------------------------------------------------------------
    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.microwave = self.register_fixture_ref(
            "microwave",
            dict(id=FixtureType.MICROWAVE),
        )
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, ref=self.microwave),
        )

        # Start near the microwave, matching the atomic tasks.
        self.init_robot_base_ref = self.microwave

        # Sample distractor navigation targets per episode.
        self._nav_targets = self._sample_nav_targets(num=self.num_distractors)
        self._nav_index = 0

        # candidate counters for a microwave layout shift (populated lazily)
        self._mw_shift_counter_names = None
        self._mw_shift_counter_name_chosen = None

    def _setup_scene(self):
        super()._setup_scene()

        # Reset runtime stage machine
        self.stage = self.STAGE_A_STORE
        self._nav_index = 0

        self.layout_shift_done = False
        self.layout_shift_info = {}

        # Make it easy to place/retrieve from microwave
        if getattr(self, "microwave", None) is not None:
            try:
                self.microwave.open_door(env=self)
            except Exception:
                pass

    def _get_obj_cfgs(self):
        cfgs: list[dict[str, Any]] = []

        # Target object on a counter near the microwave (PickPlaceCounterToMicrowave style)
        cfgs.append(
            dict(
                name="obj",
                obj_groups=self.obj_groups,
                exclude_obj_groups=self.exclude_obj_groups,
                graspable=True,
                microwavable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.microwave),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                ),
            )
        )

        # Plate inside microwave for Stage A success check
        cfgs.append(
            dict(
                name="microwave_plate",
                obj_groups=("plate"),
                placement=dict(
                    fixture=self.microwave,
                    size=(0.05, 0.05),
                    ensure_object_boundary_in_range=False,
                ),
            )
        )

        # Final receptacle container (we will reposition it onto the post-shift counter)
        cfgs.append(
            dict(
                name="container",
                obj_groups=("container"),
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.microwave),
                    size=(0.30, 0.30),
                    pos=("ref", 1.0),
                ),
            )
        )

        return cfgs

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang("obj")
        cont_lang = self.get_obj_lang("container")

        nav_lang_parts = []
        for i, t in enumerate(self._nav_targets):
            # `fixtures` contains nat_lang fields; fall back to fixture_name.
            fx = self.get_fixture(t.fixture_name, full_name_check=True)
            nav_lang_parts.append(getattr(fx, "nat_lang", t.fixture_name))

        nav_desc = ""
        if nav_lang_parts:
            nav_desc = " Then navigate to: " + ", ".join(nav_lang_parts) + "."

        ep_meta["lang"] = (
            f"(A) Pick the {obj_lang} from the counter and place it onto the plate inside the microwave."
            f"{nav_desc} "
            f"(B) After the microwave moves to a new location, pick the {obj_lang} from the microwave and place it on {cont_lang} on the counter."
        )
        ep_meta["grounding_v2"] = dict(
            stage=int(self.stage),
            num_distractors=int(self.num_distractors),
            distractor_targets=[
                dict(
                    fixture_name=t.fixture_name,
                    target_pos=t.target_pos.tolist(),
                    target_yaw=float(t.target_yaw),
                )
                for t in self._nav_targets
            ],
            layout_shift=dict(
                done=bool(self.layout_shift_done),
                info=self.layout_shift_info,
            ),
        )
        return ep_meta

    # ---------------------------------------------------------------------
    # Stage machine / success checks
    # ---------------------------------------------------------------------
    def _check_success(self):
        # Stage A: store into microwave (do NOT terminate episode)
        if self.stage == self.STAGE_A_STORE:
            if self._check_store_success():
                self.stage = self.STAGE_DISTRACTORS
                self._nav_index = 0
            return False

        # Stage distractors: sequential base navigation targets
        if self.stage == self.STAGE_DISTRACTORS:
            if self._check_nav_subtask_success():
                self._nav_index += 1
                if self._nav_index >= len(self._nav_targets):
                    self.stage = self.STAGE_SHIFT
            return False

        # Stage shift: apply once, then advance
        if self.stage == self.STAGE_SHIFT:
            if not self.layout_shift_done:
                self.apply_layout_shift(obj_names=("obj", "microwave_plate"))
                # After shift, we still want the microwave door open for retrieval.
                try:
                    self.microwave.open_door(env=self)
                except Exception:
                    pass
            self.stage = self.STAGE_B_RETRIEVE
            return False

        # Stage B: retrieve and place on container (terminate episode on success)
        if self.stage == self.STAGE_B_RETRIEVE:
            return self._check_retrieve_success()

        return False

    def _check_store_success(self) -> bool:
        """
        Matches `PickPlaceCounterToMicrowave._check_success` semantics, but uses:
          - obj: "obj"
          - plate: "microwave_plate"
        """
        obj = self.objects.get("obj", None)
        plate = self.objects.get("microwave_plate", None)
        if obj is None or plate is None or getattr(self, "microwave", None) is None:
            return False

        try:
            obj_plate_contact = self.check_contact(obj, plate)
            plate_micro_contact = self.check_contact(plate, self.microwave)
        except Exception:
            return False

        try:
            gripper_obj_far = OU.gripper_obj_far(self, "obj")
        except Exception:
            gripper_obj_far = False

        return bool(obj_plate_contact and plate_micro_contact and gripper_obj_far)

    def _check_retrieve_success(self) -> bool:
        """
        Matches `PickPlaceMicrowaveToCounter._check_success` semantics.
        """
        try:
            obj_in_container = OU.check_obj_in_receptacle(self, "obj", "container")
            gripper_obj_far = OU.gripper_obj_far(self, "obj")
            return bool(obj_in_container and gripper_obj_far)
        except Exception:
            return False

    def _check_nav_subtask_success(self) -> bool:
        """
        NavigateKitchen-style success for the current navigation target.
        """
        if not self._nav_targets:
            return True

        if self._nav_index >= len(self._nav_targets):
            return True

        target = self._nav_targets[self._nav_index]

        try:
            robot_id = self.sim.model.body_name2id("mobilebase0_base")
            base_pos = np.array(self.sim.data.body_xpos[robot_id], dtype=np.float64)
        except Exception:
            return False

        pos_check = float(np.linalg.norm(target.target_pos[:2] - base_pos[:2])) <= 0.20

        if T is None:
            # If robosuite is unavailable, degrade gracefully (pos only).
            return bool(pos_check)

        base_ori = T.mat2euler(np.array(self.sim.data.body_xmat[robot_id]).reshape((3, 3)))
        yaw_check = np.cos(float(target.target_yaw) - float(base_ori[2])) >= 0.98

        return bool(pos_check and yaw_check)

    # ---------------------------------------------------------------------
    # Distractor target sampling (adapted from NavigateKitchen)
    # ---------------------------------------------------------------------
    def _sample_nav_targets(self, num: int) -> list[_NavTarget]:
        fixtures = list(getattr(self, "fixtures", {}).values())
        if not fixtures:
            return []

        valid_classes = {
            "CoffeeMachine",
            "Toaster",
            "ToasterOven",
            "Stove",
            "Stovetop",
            "SingleCabinet",
            "HingeCabinet",
            "OpenCabinet",
            "Drawer",
            "Microwave",
            "Sink",
            "Hood",
            "Oven",
            "FridgeFrenchDoor",
            "FridgeSideBySide",
            "FridgeBottomFreezer",
            "Dishwasher",
            "ElectricKettle",
            "StandMixer",
        }

        # Exclude the microwave itself from distractors to keep semantics clean.
        def _is_valid(fx) -> bool:
            if fx is None:
                return False
            if fx is getattr(self, "microwave", None):
                return False
            cls = type(fx).__name__
            return cls in valid_classes and cls != "Accessory"

        candidates = [fx for fx in fixtures if _is_valid(fx)]
        if not candidates:
            return []

        # Greedy farthest sampling for diversity.
        chosen: list[Any] = []
        for _ in range(int(num)):
            if not candidates:
                break
            if not chosen:
                fx = candidates[int(self.rng.integers(len(candidates)))]
                chosen.append(fx)
                candidates.remove(fx)
                continue

            # pick a fixture that is not too close to existing chosen ones
            good = []
            for fx in candidates:
                try:
                    dists = [float(OU.fixture_pairwise_dist(fx, c)) for c in chosen]
                except Exception:
                    dists = [float(np.linalg.norm(np.array(fx.pos)[:2] - np.array(c.pos)[:2])) for c in chosen]
                if min(dists) >= self.distractor_min_dist:
                    good.append(fx)
            pool = good if good else candidates
            fx = pool[int(self.rng.integers(len(pool)))]
            chosen.append(fx)
            if fx in candidates:
                candidates.remove(fx)

        targets: list[_NavTarget] = []
        for fx in chosen:
            try:
                pos, ori = EnvUtils.compute_robot_base_placement_pose(self, fx)
                pos = np.array(pos, dtype=np.float64).reshape(3)
                yaw = float(np.array(ori, dtype=np.float64).reshape(-1)[2])
                targets.append(_NavTarget(fixture_name=fx.name, target_pos=pos, target_yaw=yaw))
            except Exception:
                continue
        return targets

    # ---------------------------------------------------------------------
    # Layout shift (adapted from grounding/envs/cabinet_layout_shift_demo.py)
    # ---------------------------------------------------------------------
    def apply_layout_shift(self, obj_names: tuple[str, ...] = ("obj",)):
        """
        Apply the microwave layout shift once, moving the microwave to a distant counter.

        Args:
            obj_names: movable object names to translate along with the microwave
        """
        if self.layout_shift_done:
            return self.layout_shift_info

        if array_to_string is None:  # pragma: no cover
            raise ImportError(
                "robosuite is required to run layout shift (missing array_to_string)."
            ) from _robosuite_import_error

        if getattr(self, "microwave", None) is None:
            raise RuntimeError("No microwave fixture found to move in this scene/layout.")

        if self._mw_shift_counter_names is None:
            self._mw_shift_counter_names = self._find_microwave_shift_counters()
        if not self._mw_shift_counter_names:
            raise RuntimeError(
                "Could not find a suitable counter to move the microwave onto; try a different layout/seed."
            )

        src_pos = np.array(self.microwave.pos, dtype=float)

        # Try candidate counters in order (farthest first) until sampling succeeds.
        last_err = None
        for counter_name in self._mw_shift_counter_names:
            counter = self.get_fixture(counter_name, full_name_check=True)
            if counter is None:
                continue
            try:
                new_pos = self._sample_fixture_pose_on_counter(
                    fixture=self.microwave,
                    counter=counter,
                    min_xy=self._get_fixture_footprint_xy(self.microwave),
                )
            except Exception as e:
                last_err = e
                continue

            self._set_fixture_pos(self.microwave, new_pos)
            delta = np.array(new_pos, dtype=float) - src_pos

            for name in obj_names:
                if name in self.objects:
                    self._shift_object(obj_name=name, delta=delta)

            # Place the final receptacle container onto the post-shift counter
            try:
                if "container" in self.objects:
                    self._teleport_object_onto_counter(obj_name="container", counter=counter)
            except Exception:
                pass

            self.sim.forward()

            self.layout_shift_done = True
            self._mw_shift_counter_name_chosen = str(counter.name)
            self.layout_shift_info = dict(
                strategy="move_microwave",
                microwave=self.microwave.name,
                counter=counter.name,
                delta=delta.tolist(),
            )
            return self.layout_shift_info

        raise RuntimeError(f"Failed to move microwave to any counter. Last error: {last_err}")

    def _set_fixture_pos(self, fixture, pos):
        pos = np.array(pos, dtype=float).reshape(3)
        body_id = self.sim.model.body_name2id(fixture.root_body)
        self.sim.model.body_pos[body_id] = pos

        # Keep fixture object's XML pose in sync for geometry-based checks.
        fixture._obj.set("pos", array_to_string(pos))

    def _get_fixture_footprint_xy(self, fixture):
        width = getattr(fixture, "width", None)
        depth = getattr(fixture, "depth", None)
        w = float(width) if width is not None else 0.35
        d = float(depth) if depth is not None else 0.30
        return (max(0.10, w * 1.05), max(0.10, d * 1.05))

    def _region_fits_xy(self, region_size, footprint_xy):
        sx, sy = float(region_size[0]), float(region_size[1])
        w, d = float(footprint_xy[0]), float(footprint_xy[1])
        return (w <= sx and d <= sy) or (w <= sy and d <= sx)

    def _infer_microwave_host_counter(self):
        try:
            all_counters = self.get_fixture(id=FixtureType.COUNTER, return_all=True) or []
        except Exception:
            all_counters = []

        mw_pos = np.array(getattr(self.microwave, "pos", [0.0, 0.0, 0.0]), dtype=float)
        for c in all_counters:
            if c is None:
                continue
            try:
                if OU.point_in_fixture(point=mw_pos, fixture=c, only_2d=True):
                    return c
            except Exception:
                continue
        return None

    def _find_microwave_shift_counters(self):
        if getattr(self, "microwave", None) is None:
            return []

        footprint_xy = self._get_fixture_footprint_xy(self.microwave)
        all_counters = self.get_fixture(id=FixtureType.COUNTER, return_all=True) or []
        src_counter = self._infer_microwave_host_counter()

        candidates: list[tuple[float, str]] = []
        for counter in all_counters:
            if counter is None:
                continue
            if src_counter is not None and counter is src_counter:
                continue
            try:
                regions = counter.get_reset_regions()
            except Exception:
                continue
            if not any(self._region_fits_xy(reg["size"], footprint_xy) for reg in regions.values()):
                continue
            dist = float(np.linalg.norm(np.array(counter.pos[:2]) - np.array(self.microwave.pos[:2])))
            candidates.append((dist, counter.name))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [name for _, name in candidates]

    def _sample_fixture_pose_on_counter(self, fixture, counter, min_xy):
        region = counter.sample_reset_region(min_size=min_xy)

        offset = np.array(region["offset"], dtype=float)
        region_size = np.array(region["size"], dtype=float)
        w, d = float(min_xy[0]), float(min_xy[1])
        margin = 0.02
        max_dx = max(0.0, (region_size[0] - w) / 2 - margin)
        max_dy = max(0.0, (region_size[1] - d) / 2 - margin)
        if max_dx > 0:
            offset[0] += float(self.rng.uniform(-max_dx, max_dx))
        if max_dy > 0:
            offset[1] += float(self.rng.uniform(-max_dy, max_dy))

        surface_pos = np.array(OU.get_pos_after_rel_offset(counter, offset), dtype=float)
        surface_pos[2] += 0.002

        # compute bottom-center offset in the fixture's local coordinates
        local_pts = np.array(fixture.get_ext_sites(all_points=True, relative=True), dtype=float)
        min_z = float(np.min(local_pts[:, 2]))
        bottom_pts = local_pts[np.abs(local_pts[:, 2] - min_z) < 1e-6]
        bottom_center_local = np.mean(bottom_pts, axis=0)

        # yaw-only rotation matrix
        yaw = float(getattr(fixture, "rot", 0.0))
        cy, sy = np.cos(yaw), np.sin(yaw)
        rot = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)

        fixture_origin = surface_pos - rot @ bottom_center_local
        return fixture_origin

    def _shift_object(self, obj_name, delta):
        obj = self.objects[obj_name]
        qpos = np.array(self.sim.data.get_joint_qpos(obj.joints[0]), dtype=float).copy()
        qpos[:3] += np.array(delta, dtype=float).reshape(3)
        with EnvUtils.no_collision(self.sim):
            self.sim.data.set_joint_qpos(obj.joints[0], qpos)
        self.sim.forward()

    def _teleport_object_onto_counter(self, obj_name: str, counter) -> None:
        obj = self.objects[obj_name]
        obj_size = np.array(getattr(obj, "size", [0.04, 0.04, 0.04]), dtype=np.float64)

        # Try a few samples and keep one that is not too close to the microwave center.
        min_xy_dist = float(getattr(self.microwave, "width", 0.35)) * 0.60
        for _ in range(30):
            try:
                try:
                    region = counter.sample_reset_region(env=self, min_size=obj_size)
                except TypeError:
                    region = counter.sample_reset_region(min_size=obj_size)
            except Exception:
                continue

            offset = np.array(region["offset"], dtype=np.float64)
            pos = np.array(OU.get_pos_after_rel_offset(counter, offset), dtype=np.float64)
            pos[2] += float(obj_size[2] / 2 + 0.005)

            try:
                if float(np.linalg.norm(pos[:2] - np.array(self.microwave.pos[:2], dtype=np.float64))) < min_xy_dist:
                    continue
            except Exception:
                pass

            qpos = np.array(self.sim.data.get_joint_qpos(obj.joints[0]), dtype=float).copy()
            qpos[:3] = pos.reshape(3)
            with EnvUtils.no_collision(self.sim):
                self.sim.data.set_joint_qpos(obj.joints[0], qpos)
            self.sim.forward()
            return

