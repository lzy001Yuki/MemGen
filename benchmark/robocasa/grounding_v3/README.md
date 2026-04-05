# grounding_v3

在 `grounding_v3/` 里定义“由现有 RoboCasa atomic task 组装而成”的新任务，并提供两套数据采集脚本：

- 人类演示（键盘 / SpaceMouse）
- 自动化（IK / controller-based scripted policy，带重试保证 100% 成功率：只保存成功轨迹）

## 1) 新任务：`microwave_relocation_v1`

定义文件：`grounding_v3/tasks/microwave_relocation_v1.py`

时间线（对应你的描述）：

1. **t=a**：`PickPlaceCounterToMicrowave`
2. **干扰任务**（从 `.../kitchen/atomic/` 选 2 个）：
   - `PickPlaceCounterToSink`
   - `PickPlaceStoveToCounter`
3. **t=b**：`PickPlaceMicrowaveToCounter`（layout id 改变来确保 microwave 位置改变）

> 说明：为了不改动 RoboCasa 内部的 scene/fixture 代码，这里用 `layout_ids` 从 1 切到 2 来“强制 microwave 位置变化”。  
> 如果你希望在同一个 layout 内“只移动 microwave fixture”，我也可以再加一个 runtime relocation 版本（直接改 `sim.model.body_pos` 并同步移动 microwave 内物体）。

## 2) 人类演示采集

脚本：`grounding_v3/scripts/collect_human_demos.py`

示例：

```bash
python grounding_v3/scripts/collect_human_demos.py \
  --task microwave_relocation_v1 \
  --out grounding_v3/data/human \
  --controller OSC_POSE \
  --renderer mjviewer \
  --camera robot0_frontview \
  --num-scenarios 1
```

每个 stage 会单独建目录并保存：

- `episodes/`（DataCollectionWrapper 输出的 `state_*.npz`、`model.xml` 等）
- `demo.hdf5`（把 `episodes/` 打包成单个 hdf5）
- `stage_meta.json`（stage 信息、是否成功等）
- `scenario_meta.json`（整个 sequence 的 meta）

## 3) IK 自动采集（100% 成功率）

脚本：`grounding_v3/scripts/collect_ik_demos.py`

示例（默认会重试直到成功，只保留成功轨迹）：

```bash
python grounding_v3/scripts/collect_ik_demos.py \
  --task microwave_relocation_v1 \
  --out grounding_v3/data/ik \
  --controller OSC_POSE \
  --num-scenarios 20 \
  --max-retries 50
```

可视化（调试用）：

```bash
python grounding_v3/scripts/collect_ik_demos.py --render --renderer mjviewer
```

## 4) 依赖说明

这些脚本依赖你本地环境已安装：

- `robosuite`
- `mujoco`
- RoboCasa（本 repo 里是 `benchmark/robocasa`，脚本会自动把它加到 `PYTHONPATH`）

