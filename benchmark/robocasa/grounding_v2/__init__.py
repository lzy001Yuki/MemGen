"""
grounding_v2
============

Task + data-collection utilities for the second iteration of the RoboCasa
grounding benchmark.

This package intentionally lives outside `robocasa.environments.*` so we can
prototype composite / long-horizon tasks without touching upstream task lists.

Importing this package registers the environments via RoboCasa / robosuite
metaclasses.
"""

from grounding_v2.envs.drawer_pick_place_close_navigate_open import (  # noqa: F401
    GroundingV2DrawerPickPlaceCloseNavigateOpen,
)
from grounding_v2.envs.microwave_pick_place_layout_shift import (  # noqa: F401
    GroundingV2MicrowavePickPlaceLayoutShift,
)
