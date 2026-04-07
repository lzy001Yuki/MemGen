from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TrackingTaskSpec:
    """
    A single long-horizon tracking task spec (one RoboCasa env instance).
    """

    name: str
    env_name: str
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    note: str | None = None


ENV_DRAWER_INOUT_ORDER_V1 = "PickPlaceCounterToDrawerInOutOrderV1"

TASK_DRAWER_INOUT_ORDER_2OBJ_V1 = TrackingTaskSpec(
    name="drawer_inout_order_2obj_v1",
    env_name=ENV_DRAWER_INOUT_ORDER_V1,
    env_kwargs=dict(
        layout_ids=1,
        style_ids=1,
        num_objects=2,
    ),
    note="2 objects: counter -> top drawer (in order), then drawer -> counter (same order).",
)

TASK_DRAWER_INOUT_ORDER_3OBJ_V1 = TrackingTaskSpec(
    name="drawer_inout_order_3obj_v1",
    env_name=ENV_DRAWER_INOUT_ORDER_V1,
    env_kwargs=dict(
        layout_ids=1,
        style_ids=1,
        num_objects=3,
    ),
    note="3 objects: counter -> top drawer (in order), then drawer -> counter (same order).",
)


def get_task_spec(name: str) -> TrackingTaskSpec:
    if name == TASK_DRAWER_INOUT_ORDER_2OBJ_V1.name:
        return TASK_DRAWER_INOUT_ORDER_2OBJ_V1
    if name == TASK_DRAWER_INOUT_ORDER_3OBJ_V1.name:
        return TASK_DRAWER_INOUT_ORDER_3OBJ_V1
    raise KeyError(f"Unknown tracking task: {name}")

