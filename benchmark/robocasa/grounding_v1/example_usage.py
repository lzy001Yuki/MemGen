#!/usr/bin/env python3
"""
Example Usage Demonstrations

This script shows various usage patterns for the long-horizon task system.
Run individual examples by commenting/uncommenting sections.

Author: AI Development Engineer
Date: 2025-01-XX
"""

import numpy as np
from long_horizon_env import LongHorizonTask, TaskPhase
from script_a_automated_recording import (
    TrajectoryBuffer,
    RobomimicHDF5Writer,
    ProgrammaticController,
)
from utils import (
    CollisionChecker,
    PositionValidator,
    HDF5Analyzer,
    TrajectoryVisualizer,
)


def example_1_basic_environment():
    """
    Example 1: Create and interact with the environment
    """
    print("\n" + "=" * 70)
    print("Example 1: Basic Environment Usage")
    print("=" * 70)

    # Create environment
    env = LongHorizonTask(
        object_type="apple",
        container_type="microwave",
        num_intermediate=2,
    )

    # Reset and get initial state
    obs, info = env.reset()
    print(f"✓ Environment reset")
    print(f"  Current phase: {env.current_phase.name}")
    print(f"  Task description: {env.get_ep_meta()['lang']}")

    # Run for 50 steps
    for step in range(50):
        # Random action
        action = env.action_space.sample() * 0.1

        obs, reward, terminated, truncated, info = env.step(action)

        # Print phase transitions
        if "phase_transition" in info:
            print(f"  Step {step}: Phase transition → {info['phase_transition']}")

        if terminated or truncated:
            print(f"  Episode finished at step {step}")
            break

    env.close()
    print("✓ Environment closed\n")


def example_2_manual_recording():
    """
    Example 2: Manually record a trajectory
    """
    print("\n" + "=" * 70)
    print("Example 2: Manual Trajectory Recording")
    print("=" * 70)

    env = LongHorizonTask(
        object_type="banana",
        container_type="bowl",
        num_intermediate=2,
    )

    # Create trajectory buffer
    trajectory = TrajectoryBuffer()

    obs, info = env.reset()
    done = False
    step_count = 0

    print("Recording trajectory...")

    while not done and step_count < 100:
        # Generate action (replace with your policy)
        action = env.action_space.sample() * 0.05

        next_obs, reward, terminated, truncated, info = env.step(action)

        # Record transition
        trajectory.add_transition(
            obs=obs,
            action=action,
            reward=reward,
            done=terminated or truncated,
            info=info
        )

        obs = next_obs
        done = terminated or truncated
        step_count += 1

    print(f"✓ Recorded {trajectory.get_length()} steps")

    # Save to HDF5
    writer = RobomimicHDF5Writer(
        output_path="./data/example_demo.hdf5",
        env_meta=env.get_ep_meta()
    )
    writer.add_trajectory(trajectory)
    writer.write()

    env.close()
    print("✓ Trajectory saved to ./data/example_demo.hdf5\n")


def example_3_collision_detection():
    """
    Example 3: Use collision checker
    """
    print("\n" + "=" * 70)
    print("Example 3: Collision Detection")
    print("=" * 70)

    env = LongHorizonTask(object_type="apple", container_type="microwave")
    obs, info = env.reset()

    # Create collision checker
    collision_checker = CollisionChecker(
        sim=env.sim,
        contact_threshold=0.1
    )

    # Run some steps and check collisions
    for step in range(20):
        action = env.action_space.sample() * 0.1
        env.step(action)

        # Check for collisions
        has_collision = collision_checker.check_collision(
            excluded_geoms=["floor", "table", "counter"]
        )

        if has_collision:
            print(f"  Step {step}: Collision detected!")

            # Get detailed contact info
            contacts = collision_checker.get_contact_info()
            for contact in contacts[:3]:  # Show first 3 contacts
                print(f"    Contact: {contact['geom1']} ↔ {contact['geom2']}")
                print(f"    Force: {contact['force']:.3f} N")

    env.close()
    print("✓ Collision checking completed\n")


def example_4_position_validation():
    """
    Example 4: Validate and sample positions
    """
    print("\n" + "=" * 70)
    print("Example 4: Position Validation")
    print("=" * 70)

    # Create position validator
    validator = PositionValidator(
        workspace_bounds={
            'x': (-1.0, 1.0),
            'y': (-1.0, 1.0),
            'z': (0.8, 1.5),
        }
    )

    # Test specific positions
    test_positions = [
        np.array([0.5, 0.5, 1.0]),    # Valid
        np.array([2.0, 0.0, 1.0]),    # Out of bounds (x)
        np.array([0.0, 0.0, 0.5]),    # Out of bounds (z too low)
    ]

    for i, pos in enumerate(test_positions):
        valid = validator.is_valid_position(pos)
        print(f"  Position {i+1}: {pos} → {'✓ Valid' if valid else '✗ Invalid'}")

    # Sample random valid positions
    print("\n  Sampling 5 random valid positions:")
    for i in range(5):
        pos = validator.sample_valid_position(
            min_distance_from=np.array([0, 0, 1.0]),
            min_distance=0.3
        )
        if pos is not None:
            print(f"    Sample {i+1}: {pos}")
        else:
            print(f"    Sample {i+1}: Failed to find valid position")

    print()


def example_5_hdf5_analysis():
    """
    Example 5: Analyze HDF5 trajectory data
    """
    print("\n" + "=" * 70)
    print("Example 5: HDF5 Data Analysis")
    print("=" * 70)

    hdf5_path = "./data/example_demo.hdf5"

    # Check if file exists
    from pathlib import Path
    if not Path(hdf5_path).exists():
        print(f"  Skipping: {hdf5_path} not found")
        print("  Run example_2_manual_recording() first to create the file")
        return

    # Get statistics
    stats = HDF5Analyzer.get_trajectory_stats(hdf5_path)

    print(f"  File: {hdf5_path}")
    print(f"  Number of demos: {stats['num_demos']}")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Average demo length: {stats['avg_demo_length']:.1f} steps")

    # Load and analyze first trajectory
    traj = HDF5Analyzer.load_trajectory(hdf5_path, demo_key="demo_0")

    print(f"\n  First trajectory:")
    print(f"    Length: {len(traj['actions'])} steps")
    print(f"    Action dim: {traj['actions'].shape[1]}")
    print(f"    Total reward: {np.sum(traj['rewards']):.3f}")

    # Visualize (if matplotlib available)
    try:
        TrajectoryVisualizer.plot_action_trajectory(
            traj['actions'],
            save_path="./data/example_actions.png"
        )
    except ImportError:
        print("  (matplotlib not installed, skipping visualization)")

    print()


def example_6_programmatic_control():
    """
    Example 6: Use programmatic controller
    """
    print("\n" + "=" * 70)
    print("Example 6: Programmatic Controller")
    print("=" * 70)

    env = LongHorizonTask(object_type="apple", container_type="microwave")
    obs, info = env.reset()

    # Create controller
    controller = ProgrammaticController(env)

    # Generate action sequence for Phase 1
    print("  Generating actions for Phase 1 (pick and place)...")
    actions = controller.execute_phase_1()
    print(f"  ✓ Generated {len(actions)} actions")

    # Execute actions
    print("  Executing actions...")
    for i, action in enumerate(actions[:20]):  # Execute first 20
        obs, reward, terminated, truncated, info = env.step(action)

        if "phase_transition" in info:
            print(f"    Phase transition at step {i}: {info['phase_transition']}")

    env.close()
    print("✓ Programmatic control completed\n")


def run_all_examples():
    """
    Run all examples in sequence
    """
    print("\n" + "=" * 70)
    print("Running All Examples")
    print("=" * 70)

    examples = [
        example_1_basic_environment,
        example_2_manual_recording,
        example_3_collision_detection,
        example_4_position_validation,
        example_5_hdf5_analysis,
        example_6_programmatic_control,
    ]

    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"  ✗ Example failed: {type(e).__name__}: {e}\n")
            import traceback
            traceback.print_exc()

    print("=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    # You can run individual examples or all at once

    # Run single example:
    # example_1_basic_environment()
    # example_2_manual_recording()
    # example_3_collision_detection()
    # example_4_position_validation()
    # example_5_hdf5_analysis()
    # example_6_programmatic_control()

    # Or run all examples:
    run_all_examples()
