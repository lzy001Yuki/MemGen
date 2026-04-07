# tracking

这个目录用于放 **第二个 benchmark：tracking / ordering**。

目标：在同一个长时序 episode 里，机器人需要跟踪多个物体的 **顺序**：

1. 从 counter 上依次拿起 2-3 个物体，按指定顺序放进某个地方（当前版本用 `TOP_DRAWER`）。
2. 再按同样顺序把它们从里面取出来，放回 counter。

## 1) Benchmark v1：`drawer_inout_order_v1`

环境定义：`tracking/envs/drawer_inout_order_v1.py`

特点：

- 单个 env 内部包含多个子目标（不是像 `grounding_v3` 那样每个 stage 一个独立 env）。
- 通过记录每个物体 **首次进入 drawer 的时间步**、以及 **首次回到 counter 的时间步** 来判定顺序是否正确。

任务配置（registry）：`tracking/tasks/drawer_inout_order_v1.py`

## 2) 快速用法（创建环境）

> 注意：本 repo 的 RoboCasa 是 vendored 在 `benchmark/robocasa/` 下的，默认不在 `PYTHONPATH`。  
> 可以复用 `grounding_v3.utils.add_robocasa_to_pythonpath()` 来添加路径。

示例：

```bash
python -c "from grounding_v3.utils import add_robocasa_to_pythonpath; add_robocasa_to_pythonpath(); import tracking.envs.drawer_inout_order_v1 as _; print('env registered')"
```

之后即可用 `robosuite.make(env_name=...)` 或复用 `grounding_v3.utils.make_robocasa_env()` 创建：

```bash
python - <<'PY'
from grounding_v3.utils import add_robocasa_to_pythonpath, make_robocasa_env
add_robocasa_to_pythonpath()
import tracking.envs.drawer_inout_order_v1  # registers env in robosuite
env, _ = make_robocasa_env(env_name="PickPlaceCounterToDrawerInOutOrderV1", controller="OSC_POSE", camera="robot0_frontview", renderer="mujoco", has_renderer=False, use_camera_obs=False, env_kwargs=dict(layout_ids=1, style_ids=1, num_objects=3))
env.reset()
print(env.get_ep_meta().get("lang"))
env.close()
PY
```

