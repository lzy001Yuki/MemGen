"""
Script B: Teleoperation Server (HTTP + MJPEG)

This server is designed to work reliably behind reverse proxies (e.g. VSCode / Jupyter
web proxy paths like ".../proxy/8000/") by avoiding WebSocket dependencies.

It follows the same connection pattern as `benchmark/robocasa/grounding/web/layout_shift_server.py`:
  - stdlib `http.server` (no FastAPI / Flask)
  - MJPEG streams for live frames
  - JSON POST endpoints for teleop / recording control

Endpoints (all are path-prefix friendly when accessed via relative URLs):
  - GET  /                     -> serves `script_b_teleoperation_client.html`
  - GET  /api/status            -> status + action space dim
  - POST /api/step              -> step env with action command
  - POST /api/reset             -> env.reset()
  - POST /api/start_recording   -> start recording
  - POST /api/stop_recording    -> stop & save HDF5
  - GET  /rgb.mjpg              -> MJPEG stream (RGB)
  - GET  /depth.mjpg            -> MJPEG stream (Depth visualized)

Typical usage:
  export MUJOCO_GL=egl
  python benchmark/robocasa/grounding_v1/script_b_teleoperation_server.py --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os

# --- Headless rendering / GL bootstrap -------------------------------------
# This server is meant for headless use. Robosuite/robocasa's offscreen renderer
# can use PyOpenGL, and on headless systems PyOpenGL may pick a GLX platform by
# default (no EGL), crashing with:
#   AttributeError: 'NoneType' object has no attribute 'eglQueryString'
#
# Make the backend explicit but still user-overridable.
os.environ.setdefault("MUJOCO_GL", "egl")
_mujoco_gl = (os.environ.get("MUJOCO_GL") or "").strip().lower()
if _mujoco_gl in ("egl", "osmesa"):
    os.environ.setdefault("PYOPENGL_PLATFORM", _mujoco_gl)
else:
    # If MUJOCO_GL is something else (or empty), EGL is the safest default for headless.
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

# For headless EGL, surfaceless is the most common choice; users can override.
if (os.environ.get("PYOPENGL_PLATFORM") or "").strip().lower() == "egl":
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import argparse
import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

from long_horizon_env import LongHorizonTask
from script_a_automated_recording import RobomimicHDF5Writer, TrajectoryBuffer


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")


def _encode_jpeg(image: np.ndarray, *, quality: int = 85) -> bytes:
    """
    Encode an image array as JPEG bytes.

    Supports:
      - RGB uint8 HxWx3
      - Depth float32/float64 HxW (normalized to uint8)
    """
    if image.ndim == 2:
        # Depth or grayscale
        if image.dtype != np.uint8:
            img_min = float(np.min(image))
            img_max = float(np.max(image))
            denom = (img_max - img_min) if (img_max - img_min) > 1e-8 else 1.0
            image = ((image - img_min) / denom * 255.0).astype(np.uint8)
        pil_img = Image.fromarray(image, mode="L")
    else:
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(image)

    buf = BytesIO()
    pil_img.save(buf, format="JPEG", quality=int(quality), optimize=True)
    return buf.getvalue()


class HeadlessRenderer:
    """
    Headless rendering using robosuite's sim.render.
    """

    def __init__(self, env, camera_names: list[str] | None = None, *, width: int = 640, height: int = 480):
        self.env = env
        self.camera_names = camera_names or ["robot0_agentview_center", "robot0_eye_in_hand"]
        self.width = int(width)
        self.height = int(height)

        # Best-effort: configure env to emit camera obs if supported.
        if hasattr(env, "camera_names"):
            try:
                env.camera_names = self.camera_names
                env.camera_heights = [self.height] * len(self.camera_names)
                env.camera_widths = [self.width] * len(self.camera_names)
            except Exception:
                pass

    def _clear_offscreen_render_context(self) -> None:
        """
        Clear RoboSuite's cached offscreen render context.

        When EGL context creation fails (or fails half-way), RoboSuite can leave
        `sim._render_context_offscreen` in a broken state which then causes
        subsequent renders to fail with errors like:
          - AttributeError: 'MjRenderContextOffscreen' object has no attribute 'con'
        """
        sim = getattr(self.env, "sim", None)
        if sim is None:
            return
        if not hasattr(sim, "_render_context_offscreen"):
            return
        try:
            ctx = getattr(sim, "_render_context_offscreen", None)
            if ctx is not None and hasattr(ctx, "free"):
                try:
                    ctx.free()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            setattr(sim, "_render_context_offscreen", None)
        except Exception:
            pass

    def _normalize_rgb_uint8(self, rgb: np.ndarray) -> np.ndarray:
        if rgb.dtype == np.uint8:
            return rgb
        rgb_f = np.array(rgb, dtype=np.float32, copy=False)
        mx = float(np.max(rgb_f)) if rgb_f.size else 0.0
        # Some backends return floats in [0, 1]
        if mx <= 1.5:
            rgb_f = np.clip(rgb_f, 0.0, 1.0) * 255.0
        else:
            rgb_f = np.clip(rgb_f, 0.0, 255.0)
        return rgb_f.astype(np.uint8)

    def _pick_camera(self, preferred: str | None) -> str:
        cam = preferred or self.camera_names[0]
        try:
            model = getattr(getattr(self.env, "sim", None), "model", None)
            if model is None:
                return cam
            # Most bindings expose camera_name2id; use it as validation.
            name2id = getattr(model, "camera_name2id", None)
            if callable(name2id):
                try:
                    name2id(cam)
                    return cam
                except Exception:
                    pass
            # Fallback: check camera_names list if present.
            names = getattr(model, "camera_names", None)
            if names:
                decoded = []
                for n in names:
                    if isinstance(n, bytes):
                        decoded.append(n.decode("utf-8", errors="ignore"))
                    else:
                        decoded.append(str(n))
                if cam in decoded:
                    return cam
                if decoded:
                    return decoded[0]
        except Exception:
            pass
        return cam

    def render_rgb(self, camera_name: str | None = None) -> np.ndarray:
        cam = self._pick_camera(camera_name)
        # Retry once if RoboSuite's cached context is broken.
        for attempt in range(2):
            try:
                # Use the same call style as layout_shift_server.py for compatibility.
                rgb = self.env.sim.render(height=self.height, width=self.width, camera_name=cam)[::-1]
                return self._normalize_rgb_uint8(rgb)
            except AttributeError as e:
                msg = repr(e)
                if attempt == 0 and "MjRenderContextOffscreen" in msg and "con" in msg:
                    self._clear_offscreen_render_context()
                    continue
                raise
            except Exception as e:
                # Common MuJoCo failure in headless / misconfigured GL backends.
                if attempt == 0 and "Default framebuffer is not complete" in repr(e):
                    self._clear_offscreen_render_context()
                    continue
                raise

    def render_depth(self, camera_name: str | None = None) -> np.ndarray:
        cam = self._pick_camera(camera_name)
        for attempt in range(2):
            try:
                out = self.env.sim.render(height=self.height, width=self.width, camera_name=cam, depth=True)
                if isinstance(out, tuple) and len(out) == 2:
                    _, depth = out
                else:
                    # Some versions may return only depth when depth=True (rare).
                    depth = out
                depth = np.flipud(depth)
                return np.array(depth, dtype=np.float32, copy=False)
            except AttributeError as e:
                msg = repr(e)
                if attempt == 0 and "MjRenderContextOffscreen" in msg and "con" in msg:
                    self._clear_offscreen_render_context()
                    continue
                raise
            except Exception as e:
                if attempt == 0 and "Default framebuffer is not complete" in repr(e):
                    self._clear_offscreen_render_context()
                    continue
                raise


@dataclass
class ServerConfig:
    object_type: str
    container_type: str
    num_intermediate: int
    host: str
    port: int
    stream_fps: float = 10.0
    jpeg_quality: int = 85
    width: int = 640
    height: int = 480
    out_dir: Path = Path("./data/teleoperation")


class TeleoperationAppState:
    def __init__(self, cfg: ServerConfig):
        self.cfg = cfg
        self.lock = threading.Lock()

        self.last_error: str | None = None
        self.last_saved: str | None = None
        self.step_i: int = 0
        self.last_reward: float | None = None
        self.last_info: dict[str, Any] = {}

        self.is_recording: bool = False
        self.trajectory = TrajectoryBuffer()

        self.env = LongHorizonTask(
            object_type=self.cfg.object_type,
            container_type=self.cfg.container_type,
            num_intermediate=self.cfg.num_intermediate,
            # Match layout_shift_server.py: don't rely on camera observations; we render
            # manually for MJPEG streaming.
            use_camera_obs=False,
            has_renderer=False,
            has_offscreen_renderer=True,
            render_gpu_device_id=-1,
            camera_heights=int(self.cfg.height),
            camera_widths=int(self.cfg.width),
        )
        reset_out = self.env.reset()
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            obs, info = reset_out
        else:
            obs, info = reset_out, {}
        self.current_obs = obs
        self.last_info = dict(info) if isinstance(info, dict) else {}

        self.renderer = HeadlessRenderer(self.env, width=self.cfg.width, height=self.cfg.height)
        # robosuite / robocasa typically exposes action bounds via `action_spec`
        # (low, high). Gymnasium exposes `action_space`.
        if hasattr(self.env, "action_spec"):
            low, high = self.env.action_spec  # type: ignore[attr-defined]
            self.action_low = np.array(low, dtype=np.float32).reshape(-1)
            self.action_high = np.array(high, dtype=np.float32).reshape(-1)
        elif hasattr(self.env, "action_space"):
            space = self.env.action_space  # type: ignore[attr-defined]
            self.action_low = np.array(space.low, dtype=np.float32).reshape(-1)
            self.action_high = np.array(space.high, dtype=np.float32).reshape(-1)
        else:
            raise AttributeError("env has neither `action_spec` nor `action_space`")

        self.action_dim = int(self.action_low.shape[0])
        self.last_action = np.zeros(self.action_dim, dtype=np.float32)

        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass

    def status(self) -> dict[str, Any]:
        phase = None
        try:
            phase = self.last_info.get("current_phase", None)
        except Exception:
            phase = None

        return dict(
            ok=(self.last_error is None),
            error=self.last_error,
            action_space_shape=int(self.action_dim),
            recording=bool(self.is_recording),
            saved_path=self.last_saved,
            step_i=int(self.step_i),
            last_reward=(float(self.last_reward) if self.last_reward is not None else None),
            current_phase=phase,
            env_config=dict(
                object_type=self.cfg.object_type,
                container_type=self.cfg.container_type,
                num_intermediate=int(self.cfg.num_intermediate),
            ),
        )

    def reset(self) -> dict[str, Any]:
        with self.lock:
            self.last_error = None
            self.last_saved = None
            reset_out = self.env.reset()
            if isinstance(reset_out, tuple) and len(reset_out) == 2:
                obs, info = reset_out
            else:
                obs, info = reset_out, {}
            self.current_obs = obs
            self.last_info = dict(info) if isinstance(info, dict) else {}
            self.step_i = 0
            self.last_reward = None
            self.last_action = np.zeros(self.action_dim, dtype=np.float32)
            return self.status()

    def start_recording(self) -> dict[str, Any]:
        with self.lock:
            self.last_error = None
            self.last_saved = None
            if self.is_recording:
                self.last_error = "already recording"
                return self.status()
            self.is_recording = True
            self.trajectory.reset()
            return self.status()

    def stop_recording(self) -> dict[str, Any]:
        with self.lock:
            self.last_error = None
            if not self.is_recording:
                self.last_error = "not recording"
                return self.status()

            self.is_recording = False
            trajectory = self.trajectory
            self.trajectory = TrajectoryBuffer()

            # Save trajectory (best-effort even if empty; return error if empty)
            if trajectory.is_empty():
                self.last_error = "trajectory is empty"
                return self.status()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(self.cfg.out_dir / f"demo_teleop_{timestamp}.hdf5")

            try:
                writer = RobomimicHDF5Writer(output_path, self.env.get_ep_meta())
                writer.add_trajectory(trajectory)
                writer.write()
                self.last_saved = output_path
            except Exception as e:
                self.last_error = f"save failed: {e!r}"

            return self.status()

    def step(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Step the env once.

        Expected payloads (client is flexible):
          - {"command": {"type": "velocity", "values": [...]}}
          - {"values": [...]}  (shorthand)
        """
        with self.lock:
            self.last_error = None
            self.last_saved = None

            try:
                command = payload.get("command", payload) if isinstance(payload, dict) else {}
                values = command.get("values", None) if isinstance(command, dict) else None
                if values is None:
                    values = [0.0] * self.action_dim
                action = np.array(values, dtype=np.float32).reshape(-1)
                if action.size < self.action_dim:
                    action = np.pad(action, (0, self.action_dim - action.size))
                action = action[: self.action_dim]
                action = np.clip(action, self.action_low, self.action_high)
            except Exception as e:
                self.last_error = f"bad action payload: {e!r}"
                return self.status()

            try:
                step_out = self.env.step(action)
            except Exception as e:
                self.last_error = f"env.step failed: {e!r}"
                return self.status()

            # robosuite API: (obs, reward, done, info)
            # gymnasium API: (obs, reward, terminated, truncated, info)
            obs = None
            reward = 0.0
            terminated = False
            truncated = False
            info: dict[str, Any] = {}
            try:
                if isinstance(step_out, tuple) and len(step_out) == 5:
                    obs, reward, terminated, truncated, info = step_out  # type: ignore[misc]
                elif isinstance(step_out, tuple) and len(step_out) == 4:
                    obs, reward, done, info = step_out  # type: ignore[misc]
                    terminated = bool(done)
                    truncated = False
                else:
                    raise ValueError(f"unexpected env.step return: {type(step_out)} / len={len(step_out) if isinstance(step_out, tuple) else 'n/a'}")
            except Exception as e:
                self.last_error = f"bad env.step return: {e!r}"
                return self.status()

            done = bool(terminated) or bool(truncated)

            # Record if active (record obs BEFORE the transition, same as original script_b)
            if self.is_recording and self.current_obs is not None:
                try:
                    self.trajectory.add_transition(self.current_obs, action, float(reward), done, info if isinstance(info, dict) else {})
                except Exception as e:
                    self.last_error = f"recording failed: {e!r}"

            self.current_obs = obs
            self.last_action = action
            self.last_reward = float(reward)
            self.last_info = dict(info) if isinstance(info, dict) else {}
            self.step_i += 1

            st = self.status()
            # Small step result blob for UI; keep it JSON-friendly.
            st["step_result"] = dict(
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                info={k: v for k, v in (info or {}).items() if isinstance(v, (int, float, str, bool))} if isinstance(info, dict) else {},
            )
            return st

    def snapshot_rgb_jpeg(self, *, quality: int | None = None) -> bytes:
        with self.lock:
            rgb = self.renderer.render_rgb()
        return _encode_jpeg(rgb, quality=(quality if quality is not None else self.cfg.jpeg_quality))

    def snapshot_depth_jpeg(self, *, quality: int | None = None) -> bytes:
        with self.lock:
            depth = self.renderer.render_depth()
        return _encode_jpeg(depth, quality=(quality if quality is not None else self.cfg.jpeg_quality))


class RequestHandler(BaseHTTPRequestHandler):
    server: "TeleoperationHTTPServer"  # type: ignore[assignment]

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D401
        # Keep logs readable in remote tmux output.
        return super().log_message(fmt, *args)

    def _send_bytes(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        self._send_bytes(status, "application/json; charset=utf-8", _json_bytes(payload))

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            html_path = Path(__file__).parent / "script_b_teleoperation_client.html"
            body = html_path.read_bytes() if html_path.exists() else b"missing client html"
            return self._send_bytes(HTTPStatus.OK, "text/html; charset=utf-8", body)

        if path == "/api/status":
            return self._send_json(HTTPStatus.OK, self.server.app.status())

        if path == "/snapshot_rgb.jpg":
            try:
                jpg = self.server.app.snapshot_rgb_jpeg()
                return self._send_bytes(HTTPStatus.OK, "image/jpeg", jpg)
            except Exception as e:
                return self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": repr(e)})

        if path == "/snapshot_depth.jpg":
            try:
                jpg = self.server.app.snapshot_depth_jpeg()
                return self._send_bytes(HTTPStatus.OK, "image/jpeg", jpg)
            except Exception as e:
                return self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": repr(e)})

        if path == "/rgb.mjpg":
            return self._handle_mjpeg_stream(parsed, kind="rgb")

        if path == "/depth.mjpg":
            return self._handle_mjpeg_stream(parsed, kind="depth")

        return self._send_json(HTTPStatus.NOT_FOUND, {"error": f"unknown path: {path}"})

    def _handle_mjpeg_stream(self, parsed, *, kind: str) -> None:
        qs = parse_qs(parsed.query or "")
        try:
            fps = float(qs.get("fps", [str(self.server.app.cfg.stream_fps)])[0])
        except Exception:
            fps = float(self.server.app.cfg.stream_fps)
        fps = max(0.1, min(60.0, fps))
        interval = 1.0 / fps

        try:
            quality = int(qs.get("quality", [str(self.server.app.cfg.jpeg_quality)])[0])
        except Exception:
            quality = int(self.server.app.cfg.jpeg_quality)
        quality = max(10, min(95, quality))

        boundary = "frame"
        self.send_response(HTTPStatus.OK)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
        self.end_headers()

        try:
            while True:
                start = time.time()
                try:
                    if kind == "rgb":
                        jpeg = self.server.app.snapshot_rgb_jpeg(quality=quality)
                    else:
                        jpeg = self.server.app.snapshot_depth_jpeg(quality=quality)
                except Exception as e:
                    with self.server.app.lock:
                        self.server.app.last_error = f"mjpeg({kind}) render failed: {e!r}"
                    return

                self.wfile.write(f"--{boundary}\r\n".encode("utf-8"))
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("utf-8"))
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()

                elapsed = time.time() - start
                sleep_t = interval - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)
        except BrokenPipeError:
            return
        except ConnectionResetError:
            return

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
        except Exception:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b""
        payload: dict[str, Any] = {}
        if raw:
            try:
                payload = json.loads(raw.decode("utf-8"))
                if not isinstance(payload, dict):
                    payload = {}
            except Exception:
                payload = {}

        if path == "/api/reset":
            return self._send_json(HTTPStatus.OK, self.server.app.reset())
        if path == "/api/start_recording":
            return self._send_json(HTTPStatus.OK, self.server.app.start_recording())
        if path == "/api/stop_recording":
            return self._send_json(HTTPStatus.OK, self.server.app.stop_recording())
        if path == "/api/step":
            return self._send_json(HTTPStatus.OK, self.server.app.step(payload))

        return self._send_json(HTTPStatus.NOT_FOUND, {"error": f"unknown path: {path}"})


class TeleoperationHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address, app: TeleoperationAppState):
        super().__init__(server_address, RequestHandler)
        self.app = app


def main() -> None:
    parser = argparse.ArgumentParser(description="Teleoperation Server (HTTP + MJPEG)")
    parser.add_argument("--object_type", type=str, default="apple", help="Object type to manipulate")
    parser.add_argument("--container_type", type=str, default="microwave", help="Container type")
    parser.add_argument("--num_intermediate", type=int, default=2, choices=[2, 3], help="Number of intermediate tasks")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--stream_fps", type=float, default=10.0, help="MJPEG stream FPS (default 10)")
    parser.add_argument("--jpeg_quality", type=int, default=85, help="JPEG quality (10-95)")
    parser.add_argument("--width", type=int, default=640, help="Render width")
    parser.add_argument("--height", type=int, default=480, help="Render height")
    parser.add_argument("--out_dir", type=str, default="./data/teleoperation", help="Output directory for demos")
    args = parser.parse_args()

    cfg = ServerConfig(
        object_type=args.object_type,
        container_type=args.container_type,
        num_intermediate=int(args.num_intermediate),
        host=args.host,
        port=int(args.port),
        stream_fps=float(args.stream_fps),
        jpeg_quality=int(args.jpeg_quality),
        width=int(args.width),
        height=int(args.height),
        out_dir=Path(args.out_dir),
    )

    print("=" * 60)
    print("Teleoperation Server - Script B (HTTP + MJPEG)")
    print("=" * 60)
    print(f"HTTP URL : http://{cfg.host}:{cfg.port}/")
    print(f"RGB MJPG : http://{cfg.host}:{cfg.port}/rgb.mjpg")
    print(f"Depth MJPG: http://{cfg.host}:{cfg.port}/depth.mjpg")
    print("=" * 60)

    try:
        app = TeleoperationAppState(cfg)
    except Exception:
        print("\n[Teleop] Environment init failed (offscreen render backend).")
        print("This is usually a headless OpenGL/EGL setup issue.")
        print("Recommended (EGL):")
        print("  export MUJOCO_GL=egl")
        print("  export PYOPENGL_PLATFORM=egl")
        print("  export EGL_PLATFORM=surfaceless")
        print("Fallback (CPU / OSMesa, requires OSMesa libs):")
        print("  export MUJOCO_GL=osmesa")
        print("  export PYOPENGL_PLATFORM=osmesa\n")
        raise
    server = TeleoperationHTTPServer((cfg.host, cfg.port), app)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            server.server_close()
        except Exception:
            pass
        app.close()


if __name__ == "__main__":
    main()
