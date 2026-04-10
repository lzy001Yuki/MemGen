from __future__ import annotations

"""
Grounding v2: Drawer pick-place with close / navigate / open sub-stages.

This environment "assembles" existing atomic tasks into a single long-horizon
episode:

  (A) PickPlaceCounterToDrawer
      - pick `obj` from a counter and place it into a drawer

  (B) CloseDrawer
      - fully close the drawer

  (C) NavigateKitchen-style subtask
      - navigate the mobile base to a (random) fixture

  (D) OpenDrawer
      - fully open the drawer

  (E) PickPlaceDrawerToCounter
      - retrieve `obj` from the drawer and place it onto the counter
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
    import robosuite.utils.transform_utils as T
except Exception as e:  # pragma: no cover
    T = None
    _robosuite_import_error = e  # noqa: F841


@dataclass(frozen=True)
class _NavTarget:
    fixture_name: str
    target_pos: np.ndarray  # (3,)
    target_yaw: float


class GroundingV2DrawerPickPlaceCloseNavigateOpen(Kitchen):
    """
    Composite long-horizon task described in the module docstring.
    """

    # Stage identifiers
    STAGE_A_STORE = 0
    STAGE_B_CLOSE = 1
    STAGE_C_NAVIGATE = 2
    STAGE_D_OPEN = 3
    STAGE_E_RETRIEVE = 4
    STAGE_DONE = 5

    STAGE_LABELS = {
        STAGE_A_STORE: "A_PickPlaceCounterToDrawer",
        STAGE_B_CLOSE: "B_CloseDrawer",
        STAGE_C_NAVIGATE: "C_NavigateKitchen",
        STAGE_D_OPEN: "D_OpenDrawer",
        STAGE_E_RETRIEVE: "E_PickPlaceDrawerToCounter",
        STAGE_DONE: "DONE",
    }

    def stage_label(self, stage: int | None = None) -> str:
        """
        Human-readable stage label for logging / wrappers.
        """
        if stage is None:
            stage = int(getattr(self, "stage", -1))
        return str(self.STAGE_LABELS.get(int(stage), f"stage_{int(stage)}"))

    def __init__(
        self,
        nav_min_dist: float = 1.0,
        *args,
        **kwargs,
    ):
        # Enable additional fixtures so navigation has more diverse targets.
        enable_fixtures = list(kwargs.get("enable_fixtures", []))
        enable_fixtures = enable_fixtures + [
            "electric_kettle",
            "stand_mixer",
            "toaster_oven",
        ]
        kwargs["enable_fixtures"] = enable_fixtures

        self.nav_min_dist = float(nav_min_dist)

        # Per-episode runtime state (initialized in _setup_scene / _setup_kitchen_references)
        self.stage: int = self.STAGE_A_STORE
        self._nav_target: _NavTarget | None = None

        super().__init__(*args, **kwargs)

    # ---------------------------------------------------------------------
    # Kitchen references + object configs (adapted from atomic pick/place)
    # ---------------------------------------------------------------------
    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()

        self.drawer = self.register_fixture_ref("drawer", dict(id=FixtureType.TOP_DRAWER))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.drawer)
        )

        # Start near the drawer, matching the atomic tasks.
        self.init_robot_base_ref = self.drawer

        # Sample a navigation target per episode.
        self._nav_target = self._sample_nav_target()

    def _setup_scene(self):
        super()._setup_scene()

        # Reset runtime stage machine
        self.stage = self.STAGE_A_STORE

        # Make it easy to place the object into the drawer in Stage A
        if getattr(self, "drawer", None) is not None:
            try:
                self.drawer.open_door(env=self)
            except Exception:
                pass

    def _get_obj_cfgs(self):
        cfgs: list[dict[str, Any]] = []

        # Target object on a counter near the drawer (PickPlaceCounterToDrawer style)
        cfgs.append(
            dict(
                name="obj",
                obj_groups=("tool", "utensil"),
                exclude_obj_groups=("reamer", "strainer", "cheese_grater"),
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.drawer),
                    size=(0.60, 0.30),
                    pos=("ref", -1.0),
                ),
            )
        )

        # Distractor on the same counter, consistent with the atomic task
        cfgs.append(
            dict(
                name="distr",
                exclude_obj_groups=("tool", "utensil"),
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.drawer),
                    size=(0.60, 0.30),
                    pos=("ref", -0.5),
                ),
            )
        )

        return cfgs

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang("obj")

        nav_desc = " (C) Navigate to another kitchen location."
        if self._nav_target is not None:
            fx = self.get_fixture(self._nav_target.fixture_name, full_name_check=True)
            nav_name = getattr(fx, "nat_lang", self._nav_target.fixture_name) if fx is not None else self._nav_target.fixture_name
            nav_desc = f" (C) Navigate to the {nav_name}."

        ep_meta["lang"] = (
            f"(A) Pick the {obj_lang} from the counter and place it in the drawer. "
            f"(B) Close the drawer."
            f"{nav_desc} "
            f"(D) Open the drawer. "
            f"(E) Pick the {obj_lang} from the drawer and place it on the counter."
        )
        ep_meta["grounding_v2"] = dict(
            stage=int(self.stage),
            nav_target=None
            if self._nav_target is None
            else dict(
                fixture_name=str(self._nav_target.fixture_name),
                target_pos=self._nav_target.target_pos.tolist(),
                target_yaw=float(self._nav_target.target_yaw),
            ),
        )
        return ep_meta

    # ---------------------------------------------------------------------
    # Stage machine / success checks
    # ---------------------------------------------------------------------
    def _check_success(self):
        # Stage A: store into drawer (do NOT terminate episode)
        if self.stage == self.STAGE_A_STORE:
            if self._check_store_success():
                self.stage = self.STAGE_B_CLOSE
            return False

        # Stage B: close drawer
        if self.stage == self.STAGE_B_CLOSE:
            if self._check_drawer_closed():
                self.stage = self.STAGE_C_NAVIGATE
            return False

        # Stage C: base navigation target
        if self.stage == self.STAGE_C_NAVIGATE:
            if self._check_nav_success():
                self.stage = self.STAGE_D_OPEN
            return False

        # Stage D: open drawer
        if self.stage == self.STAGE_D_OPEN:
            if self._check_drawer_open():
                self.stage = self.STAGE_E_RETRIEVE
            return False

        # Stage E: retrieve and place on counter (terminate episode on success)
        if self.stage == self.STAGE_E_RETRIEVE:
            ok = self._check_retrieve_success()
            if ok:
                self.stage = self.STAGE_DONE
            return bool(ok)

        if self.stage == self.STAGE_DONE:
            return True

        return False

    def _check_store_success(self) -> bool:
        """
        Matches `PickPlaceCounterToDrawer._check_success` semantics.
        """
        if "obj" not in getattr(self, "objects", {}):
            return False
        if getattr(self, "drawer", None) is None:
            return False

        try:
            in_drawer = OU.obj_inside_of(self, "obj", self.drawer) and not OU.check_obj_any_counter_contact(
                self, "obj"
            )
            gripper_obj_far = OU.gripper_obj_far(self, "obj")
            return bool(in_drawer and gripper_obj_far)
        except Exception:
            return False

    def _check_retrieve_success(self) -> bool:
        """
        Matches `PickPlaceDrawerToCounter._check_success` semantics.
        """
        if "obj" not in getattr(self, "objects", {}):
            return False

        try:
            on_counter = OU.check_obj_any_counter_contact(self, "obj")
            gripper_obj_far = OU.gripper_obj_far(self, "obj")
            return bool(on_counter and gripper_obj_far)
        except Exception:
            return False

    def _check_drawer_open(self) -> bool:
        if getattr(self, "drawer", None) is None:
            return False
        try:
            door_state = self.drawer.get_door_state(env=self)
        except Exception:
            return False
        if not door_state:
            return False
        return all(float(joint_p) >= 0.95 for joint_p in door_state.values())

    def _check_drawer_closed(self) -> bool:
        if getattr(self, "drawer", None) is None:
            return False
        try:
            door_state = self.drawer.get_door_state(env=self)
        except Exception:
            return False
        if not door_state:
            return False
        return all(float(joint_p) <= 0.05 for joint_p in door_state.values())

    def _check_nav_success(self) -> bool:
        """
        NavigateKitchen-style success for the single navigation target.
        """
        if self._nav_target is None:
            # If we failed to sample a navigation target, don't block episode completion.
            return True

        try:
            robot_id = self.sim.model.body_name2id("mobilebase0_base")
            base_pos = np.array(self.sim.data.body_xpos[robot_id], dtype=np.float64)
        except Exception:
            return False

        pos_check = float(np.linalg.norm(self._nav_target.target_pos[:2] - base_pos[:2])) <= 0.20

        if T is None:
            # If robosuite is unavailable, degrade gracefully (pos only).
            return bool(pos_check)

        base_ori = T.mat2euler(np.array(self.sim.data.body_xmat[robot_id]).reshape((3, 3)))
        yaw_check = np.cos(float(self._nav_target.target_yaw) - float(base_ori[2])) >= 0.98

        return bool(pos_check and yaw_check)

    # ---------------------------------------------------------------------
    # Navigation target sampling (adapted from NavigateKitchen)
    # ---------------------------------------------------------------------
    def _sample_nav_target(self) -> _NavTarget | None:
        fixtures = list(getattr(self, "fixtures", {}).values())
        if not fixtures:
            return None

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

        candidates = []
        for fx in fixtures:
            fx_class = type(fx).__name__
            if fx_class not in valid_classes or fx_class == "Accessory":
                continue

            # avoid sampling the task drawer itself as the navigation target
            if getattr(self, "drawer", None) is not None and getattr(fx, "name", None) == getattr(self.drawer, "name", None):
                continue

            candidates.append(fx)

        if not candidates:
            candidates = fixtures

        # Prefer fixtures sufficiently far from the drawer to encourage base movement
        if getattr(self, "drawer", None) is not None:
            far = []
            for fx in candidates:
                try:
                    dist = float(OU.fixture_pairwise_dist(fx, self.drawer))
                except Exception:
                    try:
                        dist = float(
                            np.linalg.norm(np.array(fx.pos, dtype=float)[:2] - np.array(self.drawer.pos, dtype=float)[:2])
                        )
                    except Exception:
                        dist = 0.0
                if dist >= self.nav_min_dist:
                    far.append(fx)
            if far:
                candidates = far

        if not candidates:
            return None

        # Try multiple candidates to maximize the odds of a valid base pose.
        for idx in self.rng.permutation(len(candidates)):
            fx = candidates[int(idx)]
            try:
                pos, ori = EnvUtils.compute_robot_base_placement_pose(self, fx)
                pos = np.array(pos, dtype=np.float64).reshape(3)
                yaw = float(np.array(ori, dtype=np.float64).reshape(-1)[2])
                return _NavTarget(fixture_name=str(fx.name), target_pos=pos, target_yaw=yaw)
            except Exception:
                continue

        return None
