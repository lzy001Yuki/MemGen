from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class StageSpec:
    """
    A single stage (one RoboCasa env) inside a grounding_v3 task sequence.
    """

    name: str
    env_name: str
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    note: str | None = None


@dataclass(frozen=True)
class TaskSpec:
    """
    A multi-stage task composed from existing RoboCasa tasks.
    """

    name: str
    stages: list[StageSpec]
    note: str | None = None


# -----------------------------------------------------------------------------
# Task: Microwave relocation + distractors
# -----------------------------------------------------------------------------
#
# Timeline (per user request):
#   a) PickPlaceCounterToMicrowave
#   distractors) pick 2-3 atomic tasks from kitchen/atomic/
#   b) microwave position changes -> PickPlaceMicrowaveToCounter
#
# Implementation choice:
#   - keep distractors in the *same* layout/style as time-a
#   - force time-b to use a *different* layout id to ensure the microwave moves
#     (layout change is guaranteed to relocate the microwave fixture)
#
# You can override layout/style ids from the collection scripts if needed.

TASK_MICROWAVE_RELOCATION_V1 = TaskSpec(
    name="microwave_relocation_v1",
    stages=[
        StageSpec(
            name="t_a_counter_to_microwave",
            env_name="PickPlaceCounterToMicrowave",
            env_kwargs=dict(
                # fixed layout/style for reproducibility (override via scripts)
                layout_ids=1,
                style_ids=1,
            ),
            note="time-a: pick object from counter -> microwave",
        ),
        StageSpec(
            name="distractor_1_counter_to_sink",
            env_name="PickPlaceCounterToSink",
            env_kwargs=dict(
                layout_ids=1,
                style_ids=1,
            ),
            note="distractor: pick object from counter -> sink",
        ),
        StageSpec(
            name="distractor_2_stove_to_counter",
            env_name="PickPlaceStoveToCounter",
            env_kwargs=dict(
                layout_ids=1,
                style_ids=1,
            ),
            note="distractor: pick object from pan -> plate/bowl on counter",
        ),
        StageSpec(
            name="t_b_microwave_to_counter_relocated",
            env_name="PickPlaceMicrowaveToCounter",
            env_kwargs=dict(
                # layout differs from time-a to force microwave relocation
                layout_ids=2,
                style_ids=1,
            ),
            note="time-b: microwave moved (via layout change) -> pick from microwave -> counter",
        ),
    ],
    note=(
        "Assembled task sequence: Counter->Microwave, then 2 distractor atomic tasks, "
        "then Microwave->Counter with microwave relocated."
    ),
)


def get_task_spec(name: str) -> TaskSpec:
    """
    Simple task registry getter.
    """

    if name == TASK_MICROWAVE_RELOCATION_V1.name:
        return TASK_MICROWAVE_RELOCATION_V1
    raise KeyError(f"Unknown grounding_v3 task: {name}")

