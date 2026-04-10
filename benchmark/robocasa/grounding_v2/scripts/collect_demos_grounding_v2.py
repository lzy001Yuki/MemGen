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

# Ensure grounding_v2 env modules are imported + registered
import grounding_v2.envs  # noqa: F401


def main() -> None:
    runpy.run_module("robocasa.scripts.collect_demos", run_name="__main__")


if __name__ == "__main__":
    main()
