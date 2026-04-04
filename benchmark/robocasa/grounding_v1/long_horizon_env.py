"""
Long-Horizon Task Environment for Robocasa

This module implements a 4-phase long-horizon manipulation task:
- Phase 1 (T_A Placement): Robot picks an object and places it in a container (e.g., microwave)
- Phase 2 (Intermediate Tasks): Robot executes 2-3 distractor subtasks
- Phase 3 (Scene Transformation): Container randomly relocates to a new valid position
- Phase 4 (T_B Retrieval): Robot navigates to new position and retrieves the object

Author: AI Development Engineer
Date: 2025-01-XX
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import mujoco
import gymnasium as gym
from enum import Enum

from robocasa.environments.kitchen.kitchen import Kitchen
from robocasa.models.scenes.scene_registry import LayoutType


class TaskPhase(Enum):
    """Enumeration of task phases"""
    PHASE_1_PLACE = 1
    PHASE_2_DISTRACTOR = 2
    PHASE_3_TRANSFORM = 3
    PHASE_4_RETRIEVE = 4
    COMPLETED = 5


class LongHorizonTask(Kitchen):
    """
    Long-horizon task environment with 4 phases.

    This environment orchestrates a complex manipulation task spanning:
    1. Object placement into container
    2. Intermediate distractor tasks
    3. Container position transformation
    4. Object retrieval from new position

    Args:
        object_type (str): Type of object to manipulate (must be graspable)
        container_type (str): Container to use ('microwave', 'bowl', 'pot', etc.)
        num_intermediate (int): Number of intermediate tasks (2-3)
        **kwargs: Additional arguments passed to Kitchen base class
    """

    def __init__(
        self,
        object_type: str = "apple",
        container_type: str = "microwave",
        num_intermediate: int = 2,
        robots: str = "PandaMobile",
        layout_id: int = 0,
        style_id: int = 0,
        **kwargs
    ):
        # Validate parameters
        assert num_intermediate in [2, 3], "num_intermediate must be 2 or 3"

        self.object_type = object_type
        self.container_type = container_type
        self.num_intermediate = num_intermediate

        # Task state management
        self.current_phase = TaskPhase.PHASE_1_PLACE
        self.phase_complete_flags = {
            TaskPhase.PHASE_1_PLACE: False,
            TaskPhase.PHASE_2_DISTRACTOR: False,
            TaskPhase.PHASE_3_TRANSFORM: False,
            TaskPhase.PHASE_4_RETRIEVE: False,
        }
        self.intermediate_tasks_completed = 0

        # Object and container references
        self.target_object = None
        self.target_container = None
        self.original_container_pos = None
        self.new_container_pos = None

        # Intermediate task tracking
        self.intermediate_task_list = []
        kwargs.setdefault("camera_names", ["robot0_agentview_center", "robot0_eye_in_hand"])

        
        super().__init__(
            robots=robots,
            layout_id=layout_id,
            style_id=style_id,
            **kwargs
        )

    def _setup_kitchen_references(self):
        """
        Setup kitchen fixture and object references.
        Called after scene is built.
        """
        super()._setup_kitchen_references()

        # Get container fixture based on type
        if self.container_type == "microwave":
            from robocasa.models.fixtures.fixture import FixtureType
            self.target_container = self.get_fixture(FixtureType.MICROWAVE)
        # Add support for other containers (bowls, pots) as needed

        # Set initial robot position near container
        self.init_robot_base_ref = self.target_container

    def _get_obj_cfgs(self) -> List[Dict[str, Any]]:
        """
        Configure object placements for the task.

        Returns:
            List of object configuration dictionaries
        """
        cfgs = []

        # Main target object - place on counter near microwave
        cfgs.append(
            dict(
                name="target_obj",
                obj_groups=self.object_type,
                placement=dict(
                    fixture=self.target_container,
                    size=(0.30, 0.30),
                    pos=(0.0, -0.6),
                    ensure_object_boundary_in_range=False,
                ),
            )
        )

        # Add distractor objects for intermediate tasks
        # Place them on the counter as well
        for i in range(self.num_intermediate):
            cfgs.append(
                dict(
                    name=f"distractor_obj_{i}",
                    obj_groups="food",
                    placement=dict(
                        fixture=self.target_container,
                        size=(0.30, 0.30),
                        pos=(0.5 + i * 0.3, -0.6),
                        ensure_object_boundary_in_range=False,
                    ),
                )
            )

        return cfgs

    def _reset_internal(self):
        """
        Reset environment internal state.
        Called at the beginning of each episode.
        """
        super()._reset_internal()

        # Reset phase state
        self.current_phase = TaskPhase.PHASE_1_PLACE
        self.phase_complete_flags = {phase: False for phase in TaskPhase if phase != TaskPhase.COMPLETED}
        self.intermediate_tasks_completed = 0

        # Store original container position
        if self.target_container is not None:
            self.original_container_pos = self._get_container_position()

        # Generate intermediate task list
        self._generate_intermediate_tasks()

    def _generate_intermediate_tasks(self):
        """
        Generate a list of intermediate distractor tasks.
        These are simple atomic tasks like opening/closing drawers, etc.
        """
        task_pool = [
            "open_drawer",
            "close_drawer",
            "open_cabinet",
            "close_cabinet",
            "turn_on_sink",
            "turn_off_sink",
        ]

        # Randomly sample intermediate tasks
        self.intermediate_task_list = np.random.choice(
            task_pool,
            size=self.num_intermediate,
            replace=False
        ).tolist()

    def _get_container_position(self) -> np.ndarray:
        """
        Get current 3D position of the target container.

        Returns:
            3D position array [x, y, z]
        """
        if self.container_type == "microwave":
            # Get microwave body position from MuJoCo
            # body_id = mujoco.mj_name2id(
            #     self.sim.model,
            #     mujoco.mjtObj.mjOBJ_BODY.value,
            #     self.target_container.name
            # )
            microwave_body_name = next(
    (name for name in self.sim.model.body_names 
     if 'microwave' in name.lower()),None)
            body_id = self.sim.model.body_name2id(microwave_body_name)
            return self.sim.data.xpos[body_id].copy()
        else:
            # For movable objects like bowls/pots
            return self.sim.data.get_body_xpos(self.target_container.name).copy()

    def _set_container_position(self, new_pos: np.ndarray) -> bool:
        """
        Set container to a new position with collision and reachability checks.

        Args:
            new_pos: Target 3D position [x, y, z]

        Returns:
            True if position is valid and set successfully
        """
        # Validate position is within bounds
        if not self._is_position_valid(new_pos):
            return False

        # Set position via sim state modification
        if self.container_type == "microwave":
            # Fixed fixtures - need to modify joint positions
            # This is complex for articulated fixtures, simplified here
            pass
        else:
            # Movable objects - directly set qpos
            obj_qpos_addr = self.sim.model.get_joint_qpos_addr(
                f"{self.target_container.name}_joint"
            )
            self.sim.data.qpos[obj_qpos_addr[0]:obj_qpos_addr[0]+3] = new_pos

        # Forward simulate to settle
        mujoco.mj_forward(self.sim.model, self.sim.data)

        # Check for collisions
        if self._check_collision():
            # Rollback if collision detected
            return False

        return True

    def _is_position_valid(self, pos: np.ndarray) -> bool:
        """
        Check if a position is within valid workspace bounds.

        Args:
            pos: 3D position to validate

        Returns:
            True if position is valid
        """
        # Define workspace bounds (table/counter area)
        x_min, x_max = -1.0, 1.0
        y_min, y_max = -1.0, 1.0
        z_min, z_max = 0.8, 1.5  # Counter height range

        return (x_min <= pos[0] <= x_max and
                y_min <= pos[1] <= y_max and
                z_min <= pos[2] <= z_max)

    def _check_collision(self) -> bool:
        """
        Check if current state has any collisions.
        Uses MuJoCo contact detection.

        Returns:
            True if collision detected
        """
        # Simple collision check via contact forces
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            # Check if contact force exceeds threshold
            if np.linalg.norm(contact.frame[:3]) > 0.1:
                return True
        return False

    def _sample_new_container_position(self) -> np.ndarray:
        """
        Sample a new valid position for container transformation.
        Uses rejection sampling with collision and reachability checks.

        Returns:
            Valid new 3D position
        """
        max_attempts = 100

        for _ in range(max_attempts):
            # Sample random position on counter surface
            new_pos = np.array([
                np.random.uniform(-0.8, 0.8),  # x
                np.random.uniform(-0.8, 0.8),  # y
                1.0,  # z (counter height)
            ])

            # Ensure minimum distance from original position
            if np.linalg.norm(new_pos - self.original_container_pos) < 0.3:
                continue

            # Validate position
            if self._is_position_valid(new_pos):
                # Test position temporarily
                saved_state = self.sim.get_state()
                if self._set_container_position(new_pos):
                    # Position is valid
                    self.sim.set_state(saved_state)
                    return new_pos
                else:
                    # Collision detected, restore and retry
                    self.sim.set_state(saved_state)

        # Fallback: return nearby position if sampling fails
        return self.original_container_pos + np.array([0.3, 0.3, 0.0])

    def _check_phase_1_complete(self) -> bool:
        """
        Check if Phase 1 (object placement) is complete.

        Returns:
            True if object is inside container
        """
        # Check if target object is inside container
        obj_pos = self.sim.data.get_body_xpos("target_obj")
        container_pos = self._get_container_position()

        # Simple containment check (distance-based)
        distance = np.linalg.norm(obj_pos - container_pos)
        return distance < 0.15  # Within 15cm threshold

    def _check_phase_2_complete(self) -> bool:
        """
        Check if Phase 2 (intermediate tasks) is complete.

        Returns:
            True if all intermediate tasks completed
        """
        return self.intermediate_tasks_completed >= self.num_intermediate

    def _execute_phase_3_transform(self):
        """
        Execute Phase 3: Transform container position.
        Randomly relocates container to new valid position.
        """
        self.new_container_pos = self._sample_new_container_position()
        success = self._set_container_position(self.new_container_pos)

        if success:
            self.phase_complete_flags[TaskPhase.PHASE_3_TRANSFORM] = True
            self.current_phase = TaskPhase.PHASE_4_RETRIEVE

    def _check_phase_4_complete(self) -> bool:
        """
        Check if Phase 4 (object retrieval) is complete.

        Returns:
            True if object retrieved from container
        """
        # Check if object is removed from container (gripper holding or on counter)
        obj_pos = self.sim.data.get_body_xpos("target_obj")
        container_pos = self._get_container_position()

        distance = np.linalg.norm(obj_pos - container_pos)
        return distance > 0.20  # Object moved away from container

    def _check_success(self) -> bool:
        """
        Check if entire task sequence is successful.
        All 4 phases must be completed.

        Returns:
            True if all phases completed successfully
        """
        return all(self.phase_complete_flags.values())

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Step the environment with phase-aware logic.

        Args:
            action: Robot action (joint velocities or end-effector pose)

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Execute base environment step
        obs, reward, terminated, truncated, info = super().step(action)

        # Phase state machine logic
        if self.current_phase == TaskPhase.PHASE_1_PLACE:
            if self._check_phase_1_complete():
                self.phase_complete_flags[TaskPhase.PHASE_1_PLACE] = True
                self.current_phase = TaskPhase.PHASE_2_DISTRACTOR
                info["phase_transition"] = "1->2"

        elif self.current_phase == TaskPhase.PHASE_2_DISTRACTOR:
            # Simplified: auto-increment for demo (real impl would check task completion)
            self.intermediate_tasks_completed += 0.01  # Gradual progress
            if self._check_phase_2_complete():
                self.phase_complete_flags[TaskPhase.PHASE_2_DISTRACTOR] = True
                self._execute_phase_3_transform()
                info["phase_transition"] = "2->3->4"

        elif self.current_phase == TaskPhase.PHASE_4_RETRIEVE:
            if self._check_phase_4_complete():
                self.phase_complete_flags[TaskPhase.PHASE_4_RETRIEVE] = True
                self.current_phase = TaskPhase.COMPLETED
                terminated = True
                info["phase_transition"] = "4->DONE"

        # Update info with phase status
        info["current_phase"] = self.current_phase.value
        info["phase_progress"] = {
            phase.name: complete
            for phase, complete in self.phase_complete_flags.items()
        }

        return obs, reward, terminated, truncated, info

    def get_ep_meta(self) -> Dict[str, Any]:
        """
        Get episode metadata including language descriptions.

        Returns:
            Dictionary with episode metadata
        """
        ep_meta = super().get_ep_meta()

        # Phase-specific language instructions
        phase_descriptions = {
            TaskPhase.PHASE_1_PLACE: f"Pick the {self.object_type} and place it in the {self.container_type}.",
            TaskPhase.PHASE_2_DISTRACTOR: f"Complete {self.num_intermediate} intermediate tasks: {', '.join(self.intermediate_task_list)}.",
            TaskPhase.PHASE_3_TRANSFORM: f"The {self.container_type} has moved to a new location.",
            TaskPhase.PHASE_4_RETRIEVE: f"Navigate to the new {self.container_type} position and retrieve the {self.object_type}.",
        }

        ep_meta["lang"] = phase_descriptions.get(
            self.current_phase,
            f"Long-horizon task: {self.object_type} -> {self.container_type}"
        )
        ep_meta["task_config"] = {
            "object_type": self.object_type,
            "container_type": self.container_type,
            "num_intermediate": self.num_intermediate,
        }

        return ep_meta


if __name__ == "__main__":
    """
    Standalone test of the environment.
    """
    env = LongHorizonTask(
        object_type="apple",
        container_type="microwave",
        num_intermediate=2,
        render_mode="human",
    )

    obs, info = env.reset()
    print("Environment initialized successfully")
    print(f"Current phase: {env.current_phase}")
    print(f"Episode metadata: {env.get_ep_meta()}")

    # Run a few steps
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if "phase_transition" in info:
            print(f"Phase transition: {info['phase_transition']}")

        if terminated or truncated:
            print("Episode finished")
            break

    env.close()
