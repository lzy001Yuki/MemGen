"""
Script B: Teleoperation Server with Headless Rendering

This module implements a FastAPI-based WebSocket server for remote teleoperation:
- Headless rendering via mujoco.Renderer
- Real-time RGB/Depth streaming to web client
- Control command reception (joint velocities or end-effector pose)
- HDF5 trajectory recording aligned with human demonstrations

Author: AI Development Engineer
Date: 2025-01-XX
"""

import argparse
import asyncio
import base64
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime
from io import BytesIO

import mujoco
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from PIL import Image

from long_horizon_env import LongHorizonTask
from script_a_automated_recording import RobomimicHDF5Writer, TrajectoryBuffer


class HeadlessRenderer:
    """
    Headless rendering using mujoco.Renderer for offscreen image generation.
    """

    def __init__(self, model: mujoco.MjModel, width: int = 640, height: int = 480):
        """
        Initialize headless renderer.

        Args:
            model: MuJoCo model
            width: Image width in pixels
            height: Image height in pixels
        """
        self.model = model
        self.width = width
        self.height = height

        # Create offscreen renderer
        self.renderer = mujoco.Renderer(model, height=height, width=width)

        # Camera configuration
        self.camera_id = 0  # Default camera

    def render_rgb(self, data: mujoco.MjData) -> np.ndarray:
        """
        Render RGB image from current simulation state.

        Args:
            data: MuJoCo data instance

        Returns:
            RGB image array (H, W, 3) uint8
        """
        # Update scene
        self.renderer.update_scene(data, camera=self.camera_id)

        # Render RGB
        rgb = self.renderer.render()
        return rgb

    def render_depth(self, data: mujoco.MjData) -> np.ndarray:
        """
        Render depth image from current simulation state.

        Args:
            data: MuJoCo data instance

        Returns:
            Depth image array (H, W) float32
        """
        # Enable depth rendering
        self.renderer.enable_depth_rendering()

        # Update and render
        self.renderer.update_scene(data, camera=self.camera_id)
        depth = self.renderer.render()

        # Disable depth for next RGB render
        self.renderer.disable_depth_rendering()

        return depth

    def encode_image_base64(self, image: np.ndarray) -> str:
        """
        Encode image as base64 string for web transmission.

        Args:
            image: Image array (RGB or depth)

        Returns:
            Base64-encoded JPEG string
        """
        # Convert to PIL Image
        if image.dtype == np.float32:
            # Depth image - normalize to 0-255
            image = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)

        pil_img = Image.fromarray(image)

        # Encode as JPEG
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG", quality=85)
        buffer.seek(0)

        # Base64 encode
        img_str = base64.b64encode(buffer.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"


class TeleoperationSession:
    """
    Manages a single teleoperation session including:
    - Environment state
    - Recording state
    - Control command buffering
    """

    def __init__(
        self,
        env: LongHorizonTask,
        renderer: HeadlessRenderer,
        session_id: str
    ):
        """
        Initialize teleoperation session.

        Args:
            env: Task environment
            renderer: Headless renderer
            session_id: Unique session identifier
        """
        self.env = env
        self.renderer = renderer
        self.session_id = session_id

        # Recording state
        self.is_recording = False
        self.trajectory = TrajectoryBuffer()
        self.current_obs = None

        # Control state
        self.last_action = np.zeros(env.action_space.shape[0])

    def start_recording(self):
        """Start trajectory recording"""
        self.is_recording = True
        self.trajectory.reset()
        print(f"[{self.session_id}] Recording started")

    def stop_recording(self) -> TrajectoryBuffer:
        """
        Stop trajectory recording.

        Returns:
            Recorded trajectory buffer
        """
        self.is_recording = False
        print(f"[{self.session_id}] Recording stopped ({self.trajectory.get_length()} steps)")
        return self.trajectory

    def process_control_command(self, command: Dict) -> np.ndarray:
        """
        Process control command from web client.

        Args:
            command: Control command dictionary
                - type: "velocity" or "position"
                - values: list of floats

        Returns:
            Action array
        """
        cmd_type = command.get("type", "velocity")
        values = command.get("values", [0.0] * self.env.action_space.shape[0])

        # Convert to numpy array
        action = np.array(values, dtype=np.float32)

        # Clip to action space bounds
        action = np.clip(
            action,
            self.env.action_space.low,
            self.env.action_space.high
        )

        self.last_action = action
        return action

    def step(self, action: np.ndarray) -> Dict:
        """
        Execute one environment step.

        Args:
            action: Action array

        Returns:
            Step result dictionary with observation, reward, done, info
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Record if active
        if self.is_recording and self.current_obs is not None:
            self.trajectory.add_transition(
                self.current_obs,
                action,
                reward,
                terminated or truncated,
                info
            )

        self.current_obs = obs

        return {
            "observation": obs.tolist() if isinstance(obs, np.ndarray) else obs,
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": {k: v for k, v in info.items() if isinstance(v, (int, float, str, bool))},
        }

    def get_visual_frame(self) -> Dict[str, str]:
        """
        Get current visual frame (RGB + Depth).

        Returns:
            Dictionary with base64-encoded images
        """
        # Render RGB
        rgb = self.renderer.render_rgb(self.env.sim.data)
        rgb_b64 = self.renderer.encode_image_base64(rgb)

        # Render Depth
        depth = self.renderer.render_depth(self.env.sim.data)
        depth_b64 = self.renderer.encode_image_base64(depth)

        return {
            "rgb": rgb_b64,
            "depth": depth_b64,
        }


# FastAPI application
app = FastAPI(title="Robocasa Teleoperation Server")

# Global state
active_sessions: Dict[str, TeleoperationSession] = {}
env_config = {
    "object_type": "apple",
    "container_type": "microwave",
    "num_intermediate": 2,
}


@app.get("/")
async def get_client():
    """Serve the web client HTML"""
    html_path = Path(__file__).parent / "script_b_teleoperation_client.html"

    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    else:
        return HTMLResponse(content="""
        <html>
            <body>
                <h1>Robocasa Teleoperation Client</h1>
                <p>Client HTML not found. Please create script_b_teleoperation_client.html</p>
            </body>
        </html>
        """)


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for teleoperation session.

    Args:
        websocket: WebSocket connection
        session_id: Unique session identifier
    """
    await websocket.accept()
    print(f"Client connected: {session_id}")

    # Create environment and session
    env = LongHorizonTask(**env_config)
    obs, info = env.reset()

    renderer = HeadlessRenderer(
        model=env.sim.model,
        width=640,
        height=480
    )

    session = TeleoperationSession(env, renderer, session_id)
    session.current_obs = obs
    active_sessions[session_id] = session

    # Send initial state
    await websocket.send_json({
        "type": "init",
        "action_space_shape": env.action_space.shape[0],
        "episode_meta": env.get_ep_meta(),
    })

    try:
        # Main loop
        while True:
            # Receive client message
            try:
                message = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=0.05  # 50ms timeout for responsive rendering
                )
            except asyncio.TimeoutError:
                message = None

            # Process commands
            if message:
                msg_type = message.get("type")

                if msg_type == "control":
                    # Execute control command
                    action = session.process_control_command(message.get("command", {}))
                    step_result = session.step(action)

                    # Send step result
                    await websocket.send_json({
                        "type": "step_result",
                        "data": step_result,
                    })

                elif msg_type == "start_recording":
                    session.start_recording()
                    await websocket.send_json({
                        "type": "recording_status",
                        "recording": True,
                    })

                elif msg_type == "stop_recording":
                    trajectory = session.stop_recording()

                    # Save trajectory
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = f"./data/teleoperation/demo_{session_id}_{timestamp}.hdf5"

                    writer = RobomimicHDF5Writer(output_path, env.get_ep_meta())
                    writer.add_trajectory(trajectory)
                    writer.write()

                    await websocket.send_json({
                        "type": "recording_status",
                        "recording": False,
                        "saved_path": output_path,
                    })

                elif msg_type == "reset":
                    obs, info = env.reset()
                    session.current_obs = obs
                    await websocket.send_json({
                        "type": "reset_complete",
                    })

            # Send visual frame (20 FPS)
            await asyncio.sleep(0.05)
            frame = session.get_visual_frame()
            await websocket.send_json({
                "type": "frame",
                "data": frame,
            })

    except WebSocketDisconnect:
        print(f"Client disconnected: {session_id}")
    finally:
        # Cleanup
        env.close()
        if session_id in active_sessions:
            del active_sessions[session_id]


def main():
    """
    Main entry point for teleoperation server.
    """
    parser = argparse.ArgumentParser(
        description="Teleoperation Server for Long-Horizon Task"
    )
    parser.add_argument(
        "--object_type",
        type=str,
        default="apple",
        help="Object type to manipulate"
    )
    parser.add_argument(
        "--container_type",
        type=str,
        default="microwave",
        help="Container type"
    )
    parser.add_argument(
        "--num_intermediate",
        type=int,
        default=2,
        choices=[2, 3],
        help="Number of intermediate tasks"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port"
    )

    args = parser.parse_args()

    # Update global config
    global env_config
    env_config = {
        "object_type": args.object_type,
        "container_type": args.container_type,
        "num_intermediate": args.num_intermediate,
    }

    print("=" * 60)
    print("Teleoperation Server - Script B")
    print("=" * 60)
    print(f"Server URL: http://{args.host}:{args.port}")
    print(f"WebSocket: ws://{args.host}:{args.port}/ws/{{session_id}}")
    print("=" * 60)

    # Create data directory
    Path("./data/teleoperation").mkdir(parents=True, exist_ok=True)

    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
