from __future__ import annotations

from typing import Any

from robocasa.environments.kitchen.kitchen import *  # noqa: F403


class PickPlaceCounterToDrawerInOutOrderV1(Kitchen):  # noqa: F405
    """
    Tracking benchmark (v1): ordered in/out of a drawer.

    Episode:
      1) 2-3 objects start on the counter.
      2) Put them into the top drawer *in order*.
      3) Then take them out of the drawer *in the same order* and place them
         back on the counter.

    Order enforcement:
      - We record the *first timestep* each object becomes "inside drawer"
        (and not touching any counter).
      - We record the *first timestep* each object returns to the counter
        after being stored.
      - Success requires both sequences to be strictly increasing, and that
        retrieval starts after the last storage.
    """

    def __init__(
        self,
        num_objects: int = 3,
        obj_groups_seq: tuple[Any, ...] | None = None,
        *args,
        **kwargs,
    ):
        if num_objects not in (2, 3):
            raise ValueError(f"num_objects must be 2 or 3, got {num_objects}")

        self.num_objects = int(num_objects)
        self._obj_names = [f"obj{i + 1}" for i in range(self.num_objects)]
        self._obj_groups_seq = obj_groups_seq

        # Event times (first occurrence)
        self._stored_step: dict[str, int | None] = {n: None for n in self._obj_names}
        self._retrieved_step: dict[str, int | None] = {n: None for n in self._obj_names}

        super().__init__(*args, **kwargs)

    # ---------------------------------------------------------------------
    # RoboCasa hooks
    # ---------------------------------------------------------------------

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.drawer = self.register_fixture_ref("drawer", dict(id=FixtureType.TOP_DRAWER))  # noqa: F405
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, ref=self.drawer),  # noqa: F405
        )
        self.init_robot_base_ref = self.drawer

    def _setup_scene(self):
        super()._setup_scene()
        # Keep it open to focus on pick&place + ordering rather than door skills
        self.drawer.open_door(self, min=1.0, max=1.0)

    def _get_obj_cfgs(self):
        cfgs: list[dict[str, Any]] = []

        # Default: choose distinct groups so the language instruction is unambiguous.
        # (Keep them small-ish so they fit in a drawer.)
        if self._obj_groups_seq is None:
            if self.num_objects == 2:
                groups: list[Any] = ["fruit", "vegetable"]
            else:
                groups = ["fruit", "vegetable", ("tool", "utensil")]
        else:
            if len(self._obj_groups_seq) != self.num_objects:
                raise ValueError(
                    f"obj_groups_seq length must match num_objects={self.num_objects}, "
                    f"got {len(self._obj_groups_seq)}"
                )
            groups = list(self._obj_groups_seq)

        # Place objects on the counter near the drawer, spread along x.
        x_offsets = [-0.25, 0.25] if self.num_objects == 2 else [-0.30, 0.0, 0.30]

        for i, (obj_name, obj_groups, x) in enumerate(zip(self._obj_names, groups, x_offsets, strict=True)):
            obj_cfg: dict[str, Any] = dict(
                name=obj_name,
                obj_groups=obj_groups,
                graspable=True,
                init_robot_here=(i == 0),
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.drawer),
                    size=(0.60, 0.30),
                    pos=(x, -1.0),
                ),
            )

            # Keep parity with RoboCasa drawer atomic task: avoid a few rare weird tools.
            if obj_groups == ("tool", "utensil") or obj_groups == ("utensil", "tool"):
                obj_cfg["exclude_obj_groups"] = ("reamer", "strainer", "cheese_grater")

            # Reuse region to avoid collisions / keep sampling consistent
            if i > 0:
                obj_cfg["placement"]["reuse_region_from"] = self._obj_names[0]
                obj_cfg["placement"].pop("sample_region_kwargs", None)

            cfgs.append(obj_cfg)

        # Single distractor on counter (optional difficulty; keep small impact)
        used_groups_flat: list[str] = []
        for g in groups:
            if isinstance(g, (tuple, list)):
                used_groups_flat.extend([str(x) for x in g])
            else:
                used_groups_flat.append(str(g))
        exclude_groups = tuple(sorted(set(used_groups_flat)))
        cfgs.append(
            dict(
                name="distr",
                exclude_obj_groups=exclude_groups,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.drawer),
                    size=(0.60, 0.30),
                    pos=(0.0, 1.0),
                ),
            )
        )

        return cfgs

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        obj_langs = [self.get_obj_lang(obj_name=n) for n in self._obj_names]
        order_text = ", ".join(obj_langs)
        if len(obj_langs) == 2:
            pick_list = f"{obj_langs[0]} and {obj_langs[1]}"
        else:
            pick_list = f"{obj_langs[0]}, {obj_langs[1]}, and {obj_langs[2]}"

        ep_meta["lang"] = (
            f"Pick the {pick_list} from the counter and place them in the drawer in this order: "
            f"{order_text}. Then take them out of the drawer in the same order and place them back on the counter."
        )
        ep_meta["tracking_task"] = {
            "name": "drawer_inout_order_v1",
            "num_objects": self.num_objects,
            "objects": [{"name": n, "lang": l} for n, l in zip(self._obj_names, obj_langs, strict=True)],
        }
        return ep_meta

    def _reset_internal(self):
        super()._reset_internal()

        # Reset tracking
        self._stored_step = {n: None for n in self._obj_names}
        self._retrieved_step = {n: None for n in self._obj_names}

        # Prime the state so we don't accidentally treat initial placements as events
        self._update_event_times()

    def _post_action(self, action):
        reward, done, info = super()._post_action(action)
        self._update_event_times()
        return reward, done, info

    def _check_success(self):
        # Require every object to have been stored and retrieved
        stored_steps = [self._stored_step[n] for n in self._obj_names]
        retrieved_steps = [self._retrieved_step[n] for n in self._obj_names]

        if any(s is None for s in stored_steps):
            return False
        if any(r is None for r in retrieved_steps):
            return False

        stored = [int(s) for s in stored_steps if s is not None]
        retrieved = [int(r) for r in retrieved_steps if r is not None]

        # Must be in order
        if not all(stored[i] < stored[i + 1] for i in range(len(stored) - 1)):
            return False
        if not all(retrieved[i] < retrieved[i + 1] for i in range(len(retrieved) - 1)):
            return False

        # Retrieval should start after the last storage
        if retrieved[0] <= stored[-1]:
            return False

        # Final state: all objects on counter and gripper far
        if not all(OU.check_obj_any_counter_contact(self, n) for n in self._obj_names):  # noqa: F405
            return False
        if not all(OU.gripper_obj_far(self, n) for n in self._obj_names):  # noqa: F405
            return False

        return True

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _is_obj_stored(self, obj_name: str) -> bool:
        inside = OU.obj_inside_of(self, obj_name, self.drawer)  # noqa: F405
        touching_counter = OU.check_obj_any_counter_contact(self, obj_name)  # noqa: F405
        return bool(inside and not touching_counter)

    def _is_obj_on_counter(self, obj_name: str) -> bool:
        return bool(OU.check_obj_any_counter_contact(self, obj_name))  # noqa: F405

    def _update_event_times(self) -> None:
        # Record first time an object becomes stored
        for n in self._obj_names:
            if self._stored_step[n] is None and self._is_obj_stored(n):
                self._stored_step[n] = int(self.timestep)

        # Record first time an object returns to counter after being stored
        for n in self._obj_names:
            if self._stored_step[n] is None:
                continue
            if self._retrieved_step[n] is not None:
                continue
            if self._is_obj_on_counter(n) and not self._is_obj_stored(n):
                # Guard against pathological same-step bookkeeping
                if int(self.timestep) > int(self._stored_step[n] or -1):
                    self._retrieved_step[n] = int(self.timestep)

