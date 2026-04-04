"""
Long-Horizon Task System for Robocasa

This package provides a complete system for long-horizon manipulation tasks
in the robocasa kitchen environment, including:

- Four-phase task environment (place → distract → transform → retrieve)
- Automated data collection with programmatic control (Script A)
- Web-based teleoperation with headless rendering (Script B)
- Utility modules for collision detection, position validation, and data analysis

Quick Start:
    # Automated collection
    >>> from long_horizon_env import LongHorizonTask
    >>> env = LongHorizonTask(object_type="apple", container_type="microwave")
    >>> obs, info = env.reset()

    # Or use CLI
    $ python main.py collect --num_demos 10
    $ python main.py teleoperate --port 8000

For detailed usage, see README.md
"""

__version__ = "1.0.0"
__author__ = "AI Development Engineer"

# Export main classes
from .long_horizon_env import LongHorizonTask, TaskPhase
from .script_a_automated_recording import (
    TrajectoryBuffer,
    RobomimicHDF5Writer,
    ProgrammaticController,
    run_automated_collection,
)
from .utils import (
    CollisionChecker,
    PositionValidator,
    HDF5Analyzer,
    TrajectoryVisualizer,
)

__all__ = [
    # Environment
    "LongHorizonTask",
    "TaskPhase",
    # Recording
    "TrajectoryBuffer",
    "RobomimicHDF5Writer",
    "ProgrammaticController",
    "run_automated_collection",
    # Utilities
    "CollisionChecker",
    "PositionValidator",
    "HDF5Analyzer",
    "TrajectoryVisualizer",
]
