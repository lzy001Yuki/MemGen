from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from robosuite.wrappers import Wrapper

try:
    import robosuite.macros as macros
    from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
except Exception:  # pragma: no cover
    macros = None
    IMAGE_CONVENTION_MAPPING = {"opengl": 1, "opencv": -1}


def _sanitize_for_path(text: str, *, max_len: int = 96) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(text)).strip("._-")
    if not s:
        s = "unnamed"
    return s[:max_len]


def _dedup_keep_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        it = str(it)
        if it in seen:
            continue
        seen.add(it)
        out.append(it)
    return out


class TaskCompletionRGBSaverWrapper(Wrapper):
    """
    Saves RGB images from all cameras when a staged / long-horizon task advances.

    Intended for composite environments that expose a `stage` int attribute (e.g.
    `GroundingV2DrawerPickPlaceCloseNavigateOpen`).
    """

    def __init__(
        self,
        env,
        *,
        enabled: bool = False,
        width: int = 256,
        height: int = 256,
        out_dir: str | os.PathLike[str] | None = None,
        subdir_name: str = "task_completion_rgb",
        verbose: bool = False,
    ):
        super().__init__(env)
        self.enabled = bool(enabled)
        self.width = int(width)
        self.height = int(height)
        self.out_dir = None if out_dir is None else Path(out_dir)
        self.subdir_name = str(subdir_name)
        self.verbose = bool(verbose)

        self._prev_stage: int | None = None
        self._saved_stage_ids: set[int] = set()
        self._fallback_ep_dir: Path | None = None

    def reset(self):
        obs = super().reset()
        self._prev_stage = self._get_stage()
        self._saved_stage_ids = set()
        self._fallback_ep_dir = None
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self._maybe_save_on_stage_change(source="step", obs=obs)
        return obs, reward, done, info

    def _check_success(self):
        # Some collectors call `_check_success()` outside `step()`. We hook it so
        # stage advances are still captured in debug / non-DataCollection settings.
        prev_stage = self._get_stage()
        result = self.env._check_success()
        self._maybe_save_on_stage_change(source="_check_success", prev_stage=prev_stage)
        return result

    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------
    def _get_stage(self) -> int | None:
        return getattr(self.unwrapped, "stage", None)

    def _stage_label(self, stage_id: int) -> str:
        stage_label_fn = getattr(self.unwrapped, "stage_label", None)
        if callable(stage_label_fn):
            try:
                return str(stage_label_fn(stage_id))
            except Exception:
                pass
        return f"stage_{int(stage_id)}"

    def _resolve_episode_dir(self) -> Path | None:
        """
        Prefer the DataCollectionWrapper's per-episode directory when present.
        Falls back to a timestamped directory under `out_dir` (or /tmp) otherwise.
        """
        ep_dir = getattr(self.env, "ep_directory", None)
        if ep_dir:
            return Path(ep_dir)

        if self._fallback_ep_dir is None:
            root = self.out_dir if self.out_dir is not None else Path("/tmp/robocasa_task_completion_rgb")
            root.mkdir(parents=True, exist_ok=True)
            t1, t2 = str(time.time()).split(".")
            self._fallback_ep_dir = root / f"ep_{t1}_{t2}"
            self._fallback_ep_dir.mkdir(parents=True, exist_ok=True)
        return self._fallback_ep_dir

    def _camera_names_to_save(self, obs: dict[str, Any] | None) -> list[str]:
        names: list[str] = []

        # 1) If camera observations exist, save those exact cameras (matches "observation rgb" semantics).
        if obs:
            for k, v in obs.items():
                if not isinstance(k, str) or not k.endswith("_image"):
                    continue
                if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[-1] in (3, 4):
                    names.append(k[: -len("_image")])

        # 2) Otherwise, save whatever camera list is configured on the env.
        cam_names = getattr(self.unwrapped, "camera_names", None)
        if isinstance(cam_names, (list, tuple)):
            names.extend([str(x) for x in cam_names])

        # 3) RoboCasa kitchen envs expose `_cam_configs` with robot camera keys.
        cam_cfgs = getattr(self.unwrapped, "_cam_configs", None)
        if isinstance(cam_cfgs, dict):
            names.extend([str(k) for k in cam_cfgs.keys()])

        return _dedup_keep_order([n for n in names if n])

    def _maybe_save_on_stage_change(
        self,
        *,
        source: str,
        obs: dict[str, Any] | None = None,
        prev_stage: int | None = None,
    ) -> None:
        if not self.enabled:
            self._prev_stage = self._get_stage()
            return

        curr_stage = self._get_stage()
        if curr_stage is None:
            return

        if prev_stage is None:
            prev_stage = self._prev_stage

        # Initialize lazy state if needed
        if prev_stage is None:
            self._prev_stage = int(curr_stage)
            return

        if int(curr_stage) == int(prev_stage):
            self._prev_stage = int(curr_stage)
            return

        completed_stage = int(prev_stage)
        self._prev_stage = int(curr_stage)

        # Save once per stage id per episode
        if completed_stage in self._saved_stage_ids:
            return
        self._saved_stage_ids.add(completed_stage)

        self._save_rgb_snapshot(stage_id=completed_stage, source=source, obs=obs)

    def _save_rgb_snapshot(self, *, stage_id: int, source: str, obs: dict[str, Any] | None) -> None:
        ep_dir = self._resolve_episode_dir()
        if ep_dir is None:
            return

        stage_label = self._stage_label(stage_id)
        stage_dir = ep_dir / self.subdir_name / f"{stage_id:02d}_{_sanitize_for_path(stage_label)}"
        stage_dir.mkdir(parents=True, exist_ok=True)

        rgb_payload: dict[str, np.ndarray] = {}

        # Save any already-present camera observations directly.
        saved_any = False
        if obs:
            for k, v in obs.items():
                if not isinstance(k, str) or not k.endswith("_image"):
                    continue
                if not isinstance(v, np.ndarray) or v.ndim != 3 or v.shape[-1] not in (3, 4):
                    continue
                cam_name = k[: -len("_image")]
                rgb = v[:, :, :3]
                out_path = stage_dir / f"{_sanitize_for_path(cam_name)}.png"
                self._imwrite_png(out_path, rgb)
                rgb_payload[str(cam_name)] = rgb
                saved_any = True

        # If no image observations exist, render manually from available cameras.
        if not saved_any:
            for cam_name in self._camera_names_to_save(obs=None):
                rgb = self._render_rgb(camera_name=cam_name, width=self.width, height=self.height)
                if rgb is None:
                    continue
                out_path = stage_dir / f"{_sanitize_for_path(cam_name)}.png"
                self._imwrite_png(out_path, rgb)
                rgb_payload[str(cam_name)] = rgb
                saved_any = True

        if rgb_payload:
            np.savez_compressed(str(stage_dir / "rgb_obs.npz"), **rgb_payload)

        if self.verbose:
            ts = getattr(self.unwrapped, "timestep", None)
            print(
                f"[TaskCompletionRGBSaverWrapper] saved={saved_any} stage={stage_id} label={stage_label} source={source} timestep={ts} dir={stage_dir}"
            )

    def _render_rgb(self, *, camera_name: str, width: int, height: int) -> np.ndarray | None:
        env = self.unwrapped
        try:
            img = env.sim.render(
                camera_name=str(camera_name),
                width=int(width),
                height=int(height),
                depth=False,
            )
        except Exception:
            return None

        if isinstance(img, (tuple, list)):
            img = img[0]

        convention = -1
        try:
            if macros is not None:
                convention = int(IMAGE_CONVENTION_MAPPING[getattr(macros, "IMAGE_CONVENTION", "opencv")])
        except Exception:
            convention = -1

        rgb = img[::convention]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return rgb

    def _imwrite_png(self, path: Path, rgb: np.ndarray) -> None:
        try:
            import imageio.v2 as imageio  # type: ignore
        except Exception:  # pragma: no cover
            import imageio  # type: ignore

        imageio.imwrite(str(path), rgb)
