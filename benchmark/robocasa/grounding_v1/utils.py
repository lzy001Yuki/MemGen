"""
Utility Module for Long-Horizon Task

Contains helper functions for:
- Collision detection
- Position validation
- HDF5 data manipulation
- Trajectory analysis

Author: AI Development Engineer
Date: 2025-01-XX
"""

import numpy as np
import mujoco
import h5py
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path


class CollisionChecker:
    """
    Advanced collision detection using MuJoCo's contact detection.
    """

    def __init__(self, sim: mujoco.MjModel, contact_threshold: float = 0.1):
        """
        Initialize collision checker.

        Args:
            sim: MuJoCo simulation instance
            contact_threshold: Minimum contact force to consider collision (N)
        """
        self.sim = sim
        self.contact_threshold = contact_threshold

    def check_collision(self, excluded_geoms: Optional[List[str]] = None) -> bool:
        """
        Check if any collisions exist in current state.

        Args:
            excluded_geoms: List of geom names to exclude from collision check
                           (e.g., robot-floor contact is expected)

        Returns:
            True if unexpected collision detected
        """
        if excluded_geoms is None:
            excluded_geoms = ["floor", "table", "counter"]

        # Iterate through all contacts
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]

            # Get geom names
            geom1_id = contact.geom1
            geom2_id = contact.geom2

            geom1_name = mujoco.mj_id2name(
                self.sim.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                geom1_id
            )
            geom2_name = mujoco.mj_id2name(
                self.sim.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                geom2_id
            )

            # Skip if either geom is excluded
            if any(ex in geom1_name for ex in excluded_geoms):
                continue
            if any(ex in geom2_name for ex in excluded_geoms):
                continue

            # Check contact force magnitude
            contact_force = np.linalg.norm(contact.frame[:3])
            if contact_force > self.contact_threshold:
                return True

        return False

    def get_contact_info(self) -> List[Dict[str, Any]]:
        """
        Get detailed information about all active contacts.

        Returns:
            List of contact info dictionaries
        """
        contacts = []

        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]

            geom1_name = mujoco.mj_id2name(
                self.sim.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                contact.geom1
            )
            geom2_name = mujoco.mj_id2name(
                self.sim.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                contact.geom2
            )

            contacts.append({
                "geom1": geom1_name,
                "geom2": geom2_name,
                "position": contact.pos.copy(),
                "force": np.linalg.norm(contact.frame[:3]),
                "distance": contact.dist,
            })

        return contacts


class PositionValidator:
    """
    Validates positions for object/container placement.
    Checks workspace bounds, collisions, and reachability.
    """

    def __init__(
        self,
        workspace_bounds: Dict[str, Tuple[float, float]],
        collision_checker: Optional[CollisionChecker] = None
    ):
        """
        Initialize position validator.

        Args:
            workspace_bounds: Dictionary with 'x', 'y', 'z' bounds
            collision_checker: Optional collision checker instance
        """
        self.workspace_bounds = workspace_bounds
        self.collision_checker = collision_checker

    def is_valid_position(self, position: np.ndarray) -> bool:
        """
        Check if position is valid (within bounds and collision-free).

        Args:
            position: 3D position array [x, y, z]

        Returns:
            True if position is valid
        """
        # Check bounds
        if not self._check_bounds(position):
            return False

        # Check collision if checker available
        if self.collision_checker is not None:
            if self.collision_checker.check_collision():
                return False

        return True

    def _check_bounds(self, position: np.ndarray) -> bool:
        """
        Check if position is within workspace bounds.

        Args:
            position: 3D position array

        Returns:
            True if within bounds
        """
        x, y, z = position

        x_min, x_max = self.workspace_bounds.get('x', (-np.inf, np.inf))
        y_min, y_max = self.workspace_bounds.get('y', (-np.inf, np.inf))
        z_min, z_max = self.workspace_bounds.get('z', (-np.inf, np.inf))

        return (x_min <= x <= x_max and
                y_min <= y <= y_max and
                z_min <= z <= z_max)

    def sample_valid_position(
        self,
        num_attempts: int = 100,
        min_distance_from: Optional[np.ndarray] = None,
        min_distance: float = 0.3
    ) -> Optional[np.ndarray]:
        """
        Sample a random valid position using rejection sampling.

        Args:
            num_attempts: Maximum sampling attempts
            min_distance_from: Optional reference position for minimum distance constraint
            min_distance: Minimum distance from reference position (m)

        Returns:
            Valid position or None if sampling fails
        """
        for _ in range(num_attempts):
            # Sample random position within bounds
            x = np.random.uniform(*self.workspace_bounds['x'])
            y = np.random.uniform(*self.workspace_bounds['y'])
            z = np.random.uniform(*self.workspace_bounds['z'])

            position = np.array([x, y, z])

            # Check minimum distance constraint
            if min_distance_from is not None:
                if np.linalg.norm(position - min_distance_from) < min_distance:
                    continue

            # Validate position
            if self.is_valid_position(position):
                return position

        return None


class HDF5Analyzer:
    """
    Analyze and manipulate HDF5 trajectory files.
    """

    @staticmethod
    def load_trajectory(hdf5_path: str, demo_key: str = "demo_0") -> Dict[str, np.ndarray]:
        """
        Load a single trajectory from HDF5 file.

        Args:
            hdf5_path: Path to HDF5 file
            demo_key: Demonstration key (e.g., "demo_0")

        Returns:
            Dictionary with trajectory data
        """
        with h5py.File(hdf5_path, 'r') as f:
            demo_grp = f[f"data/{demo_key}"]

            traj = {
                "actions": demo_grp["actions"][:],
                "rewards": demo_grp["rewards"][:],
                "dones": demo_grp["dones"][:],
            }

            # Load observations
            obs_grp = demo_grp["obs"]
            if "flat" in obs_grp:
                traj["observations"] = obs_grp["flat"][:]
            else:
                traj["observations"] = {
                    key: obs_grp[key][:]
                    for key in obs_grp.keys()
                }

        return traj

    @staticmethod
    def get_trajectory_stats(hdf5_path: str) -> Dict[str, Any]:
        """
        Get statistics about trajectories in HDF5 file.

        Args:
            hdf5_path: Path to HDF5 file

        Returns:
            Dictionary with statistics
        """
        with h5py.File(hdf5_path, 'r') as f:
            num_demos = len(f["data"].keys())
            total_samples = 0
            demo_lengths = []

            for demo_key in f["data"].keys():
                length = len(f[f"data/{demo_key}/actions"])
                demo_lengths.append(length)
                total_samples += length

            stats = {
                "num_demos": num_demos,
                "total_samples": total_samples,
                "avg_demo_length": np.mean(demo_lengths),
                "min_demo_length": np.min(demo_lengths),
                "max_demo_length": np.max(demo_lengths),
                "demo_lengths": demo_lengths,
            }

            # Success rate if mask exists
            if "mask" in f:
                stats["success_rate"] = np.mean(f["mask"][:])

        return stats

    @staticmethod
    def merge_hdf5_files(
        input_paths: List[str],
        output_path: str,
        filter_successful: bool = True
    ):
        """
        Merge multiple HDF5 trajectory files into one.

        Args:
            input_paths: List of HDF5 file paths to merge
            output_path: Output merged HDF5 path
            filter_successful: Only include successful demonstrations
        """
        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_path, 'w') as f_out:
            data_grp = f_out.create_group("data")
            demo_idx = 0

            for input_path in input_paths:
                with h5py.File(input_path, 'r') as f_in:
                    # Get mask if exists
                    mask = f_in.get("mask", None)

                    for i, demo_key in enumerate(f_in["data"].keys()):
                        # Skip if not successful and filter enabled
                        if filter_successful and mask is not None:
                            if not mask[i]:
                                continue

                        # Copy demonstration
                        f_in.copy(
                            f"data/{demo_key}",
                            data_grp,
                            name=f"demo_{demo_idx}"
                        )
                        demo_idx += 1

            # Create mask (all successful since we filtered)
            mask = np.ones(demo_idx, dtype=bool)
            f_out.create_dataset("mask", data=mask)

            # Metadata
            f_out.attrs["num_demos"] = demo_idx
            f_out.attrs["merged_from"] = str(input_paths)

        print(f"✓ Merged {demo_idx} demonstrations to {output_path}")


class TrajectoryVisualizer:
    """
    Visualize trajectory data for debugging and analysis.
    """

    @staticmethod
    def plot_action_trajectory(
        actions: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot action trajectories over time.

        Args:
            actions: Action array (T, action_dim)
            save_path: Optional path to save plot
        """
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(actions.shape[1], 1, figsize=(12, 2 * actions.shape[1]))

            if actions.shape[1] == 1:
                axes = [axes]

            for i, ax in enumerate(axes):
                ax.plot(actions[:, i])
                ax.set_ylabel(f"Action {i}")
                ax.set_xlabel("Timestep")
                ax.grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150)
                print(f"✓ Saved plot to {save_path}")
            else:
                plt.show()

        except ImportError:
            print("Warning: matplotlib not installed, cannot plot trajectories")

    @staticmethod
    def print_trajectory_summary(traj: Dict[str, np.ndarray]):
        """
        Print summary statistics for a trajectory.

        Args:
            traj: Trajectory dictionary
        """
        print("=" * 60)
        print("Trajectory Summary")
        print("=" * 60)
        print(f"Length: {len(traj['actions'])} steps")
        print(f"Action space: {traj['actions'].shape[1]}-dimensional")
        print(f"Total reward: {np.sum(traj['rewards']):.3f}")
        print(f"Avg reward: {np.mean(traj['rewards']):.3f}")
        print(f"Success: {traj['dones'][-1]}")
        print("=" * 60)


if __name__ == "__main__":
    """
    Test utility functions.
    """
    print("Utility module loaded successfully")

    # Test position validator
    validator = PositionValidator(
        workspace_bounds={
            'x': (-1.0, 1.0),
            'y': (-1.0, 1.0),
            'z': (0.8, 1.5),
        }
    )

    test_pos = np.array([0.5, 0.5, 1.0])
    print(f"Position {test_pos} valid: {validator.is_valid_position(test_pos)}")

    # Test random sampling
    sampled_pos = validator.sample_valid_position()
    print(f"Sampled position: {sampled_pos}")
