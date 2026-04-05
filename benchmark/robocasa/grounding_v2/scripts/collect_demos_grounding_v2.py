"""
Thin wrapper around `robocasa.scripts.collect_demos` that ensures grounding_v2 envs
are imported (and therefore registered) before data collection starts.

Example:
  cd benchmark/robocasa
  python -m grounding_v2.scripts.collect_demos_grounding_v2 \
    --environment GroundingV2MicrowavePickPlaceLayoutShift \
    --split pretrain
"""

import runpy

# Ensure env class is imported + registered
from grounding_v2.envs.microwave_pick_place_layout_shift import (  # noqa: F401
    GroundingV2MicrowavePickPlaceLayoutShift,
)


def main() -> None:
    runpy.run_module("robocasa.scripts.collect_demos", run_name="__main__")


if __name__ == "__main__":
    main()

