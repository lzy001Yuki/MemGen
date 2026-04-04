#!/usr/bin/env python3
"""
Main CLI Entry Point for Long-Horizon Task System

Provides unified interface for:
- Running automated data collection (Script A)
- Launching teleoperation server (Script B)
- Analyzing collected data
- Merging HDF5 datasets

Author: AI Development Engineer
Date: 2025-01-XX

Usage:
    python main.py collect --num_demos 10
    python main.py teleoperate --port 8000
    python main.py analyze --hdf5 data/demo.hdf5
    python main.py merge --inputs file1.hdf5 file2.hdf5 --output merged.hdf5
"""

import argparse
import sys
from pathlib import Path

# Import task modules
from long_horizon_env import LongHorizonTask
from script_a_automated_recording import run_automated_collection
import script_b_teleoperation_server as teleoperation
from utils import HDF5Analyzer, TrajectoryVisualizer


def command_collect(args):
    """
    Execute automated data collection (Script A).
    """
    print("\n🤖 Starting Automated Data Collection")
    print("=" * 70)

    output_path = run_automated_collection(
        object_type=args.object_type,
        container_type=args.container_type,
        num_intermediate=args.num_intermediate,
        num_demos=args.num_demos,
        output_dir=args.output_dir,
        render=args.render,
    )

    # Analyze collected data
    if args.analyze:
        print("\n📊 Analyzing collected data...")
        stats = HDF5Analyzer.get_trajectory_stats(output_path)
        print(f"  Demonstrations: {stats['num_demos']}")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Avg length: {stats['avg_demo_length']:.1f} steps")
        print(f"  Success rate: {stats.get('success_rate', 1.0) * 100:.1f}%")

    print("\n✅ Data collection completed!")


def command_teleoperate(args):
    """
    Launch teleoperation server (Script B).
    """
    print("\n🌐 Starting Teleoperation Server")
    print("=" * 70)
    print(f"Server will be available at: http://{args.host}:{args.port}")
    print(f"Open the URL in your browser to start teleoperating")
    print("=" * 70)

    # Update environment config
    teleoperation.env_config = {
        "object_type": args.object_type,
        "container_type": args.container_type,
        "num_intermediate": args.num_intermediate,
    }

    # Create data directory
    Path("./data/teleoperation").mkdir(parents=True, exist_ok=True)

    # Run server
    import uvicorn
    uvicorn.run(
        teleoperation.app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


def command_analyze(args):
    """
    Analyze HDF5 trajectory data.
    """
    print("\n📊 Analyzing HDF5 Trajectory Data")
    print("=" * 70)

    if not Path(args.hdf5).exists():
        print(f"❌ Error: File not found: {args.hdf5}")
        return

    # Get statistics
    stats = HDF5Analyzer.get_trajectory_stats(args.hdf5)

    print("\n📁 File Info:")
    print(f"  Path: {args.hdf5}")
    print(f"  Size: {Path(args.hdf5).stat().st_size / 1024:.2f} KB")

    print("\n📈 Trajectory Statistics:")
    print(f"  Number of demonstrations: {stats['num_demos']}")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Average demo length: {stats['avg_demo_length']:.1f} steps")
    print(f"  Min demo length: {stats['min_demo_length']} steps")
    print(f"  Max demo length: {stats['max_demo_length']} steps")

    if 'success_rate' in stats:
        print(f"  Success rate: {stats['success_rate'] * 100:.1f}%")

    # Show individual demo lengths
    print("\n📋 Individual Demo Lengths:")
    for i, length in enumerate(stats['demo_lengths'][:10]):  # Show first 10
        print(f"  demo_{i}: {length} steps")

    if len(stats['demo_lengths']) > 10:
        print(f"  ... ({len(stats['demo_lengths']) - 10} more)")

    # Visualize if requested
    if args.visualize:
        print("\n📊 Generating visualizations...")
        demo_key = args.demo or "demo_0"
        traj = HDF5Analyzer.load_trajectory(args.hdf5, demo_key)

        # Print trajectory summary
        TrajectoryVisualizer.print_trajectory_summary(traj)

        # Plot actions
        save_path = args.hdf5.replace('.hdf5', '_actions.png')
        TrajectoryVisualizer.plot_action_trajectory(traj['actions'], save_path)


def command_merge(args):
    """
    Merge multiple HDF5 files.
    """
    print("\n🔗 Merging HDF5 Files")
    print("=" * 70)

    # Validate input files
    for input_path in args.inputs:
        if not Path(input_path).exists():
            print(f"❌ Error: File not found: {input_path}")
            return

    print(f"Input files: {len(args.inputs)}")
    for path in args.inputs:
        print(f"  - {path}")

    print(f"\nOutput file: {args.output}")
    print(f"Filter successful only: {not args.include_failed}")

    # Merge
    HDF5Analyzer.merge_hdf5_files(
        input_paths=args.inputs,
        output_path=args.output,
        filter_successful=not args.include_failed
    )

    # Analyze merged file
    stats = HDF5Analyzer.get_trajectory_stats(args.output)
    print(f"\n✅ Merged file statistics:")
    print(f"  Total demonstrations: {stats['num_demos']}")
    print(f"  Total samples: {stats['total_samples']}")


def command_test(args):
    """
    Test environment setup.
    """
    print("\n🧪 Testing Environment Setup")
    print("=" * 70)

    try:
        # Create environment
        print("Creating environment...")
        env = LongHorizonTask(
            object_type=args.object_type,
            container_type=args.container_type,
            num_intermediate=args.num_intermediate,
        )

        # Reset
        print("Resetting environment...")
        obs, info = env.reset()

        # Get metadata
        meta = env.get_ep_meta()
        print("\n✅ Environment created successfully!")
        print(f"\nTask description: {meta['lang']}")
        print(f"Current phase: {env.current_phase.name}")
        print(f"Observation shape: {obs.shape if hasattr(obs, 'shape') else 'N/A'}")
        print(f"Action space: {env.action_space.shape}")

        # Run a few steps
        print("\nRunning 10 test steps...")
        for i in range(10):
            action = env.action_space.sample() * 0.1  # Small random actions
            obs, reward, terminated, truncated, info = env.step(action)

            if 'phase_transition' in info:
                print(f"  Step {i}: Phase transition - {info['phase_transition']}")

            if terminated or truncated:
                print(f"  Episode finished at step {i}")
                break

        env.close()
        print("\n✅ Test completed successfully!")

    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


def main():
    """
    Main CLI entry point.
    """
    parser = argparse.ArgumentParser(
        description="Long-Horizon Task System for Robocasa",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run automated data collection
  python main.py collect --num_demos 10 --render

  # Launch teleoperation server
  python main.py teleoperate --port 8000

  # Analyze collected data
  python main.py analyze --hdf5 data/demo.hdf5 --visualize

  # Merge multiple datasets
  python main.py merge --inputs file1.hdf5 file2.hdf5 --output merged.hdf5

  # Test environment
  python main.py test
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # ========== collect command ==========
    parser_collect = subparsers.add_parser("collect", help="Run automated data collection")
    parser_collect.add_argument("--object_type", type=str, default="apple")
    parser_collect.add_argument("--container_type", type=str, default="microwave")
    parser_collect.add_argument("--num_intermediate", type=int, default=2, choices=[2, 3])
    parser_collect.add_argument("--num_demos", type=int, default=10)
    parser_collect.add_argument("--output_dir", type=str, default="./data/automated")
    parser_collect.add_argument("--render", action="store_true", help="Enable rendering")
    parser_collect.add_argument("--analyze", action="store_true", help="Analyze after collection")
    parser_collect.set_defaults(func=command_collect)

    # ========== teleoperate command ==========
    parser_teleoperate = subparsers.add_parser("teleoperate", help="Launch teleoperation server")
    parser_teleoperate.add_argument("--object_type", type=str, default="apple")
    parser_teleoperate.add_argument("--container_type", type=str, default="microwave")
    parser_teleoperate.add_argument("--num_intermediate", type=int, default=2, choices=[2, 3])
    parser_teleoperate.add_argument("--host", type=str, default="0.0.0.0")
    parser_teleoperate.add_argument("--port", type=int, default=8000)
    parser_teleoperate.set_defaults(func=command_teleoperate)

    # ========== analyze command ==========
    parser_analyze = subparsers.add_parser("analyze", help="Analyze HDF5 trajectory data")
    parser_analyze.add_argument("--hdf5", type=str, required=True, help="Path to HDF5 file")
    parser_analyze.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser_analyze.add_argument("--demo", type=str, help="Demo key to visualize (e.g., demo_0)")
    parser_analyze.set_defaults(func=command_analyze)

    # ========== merge command ==========
    parser_merge = subparsers.add_parser("merge", help="Merge multiple HDF5 files")
    parser_merge.add_argument("--inputs", nargs="+", required=True, help="Input HDF5 files")
    parser_merge.add_argument("--output", type=str, required=True, help="Output HDF5 file")
    parser_merge.add_argument("--include_failed", action="store_true", help="Include failed demos")
    parser_merge.set_defaults(func=command_merge)

    # ========== test command ==========
    parser_test = subparsers.add_parser("test", help="Test environment setup")
    parser_test.add_argument("--object_type", type=str, default="apple")
    parser_test.add_argument("--container_type", type=str, default="microwave")
    parser_test.add_argument("--num_intermediate", type=int, default=2, choices=[2, 3])
    parser_test.set_defaults(func=command_test)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Execute command
    args.func(args)


if __name__ == "__main__":
    main()
