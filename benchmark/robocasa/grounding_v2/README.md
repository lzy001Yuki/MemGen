# grounding_v2 (Microwave layout-shift task)

本目录提供一个新的 long-horizon 任务 + 两种采集脚本（人类演示 / IK 专家）。

## 任务：`GroundingV2MicrowavePickPlaceLayoutShift`

环境实现见：
- `benchmark/robocasa/grounding_v2/envs/microwave_pick_place_layout_shift.py`

阶段（同一条 episode 内依次发生）：
1. **A**：Counter → Microwave（把 `obj` 放到微波炉里的盘子 `microwave_plate` 上）
2. **干扰任务**：2-3 个 NavigateKitchen 风格的导航子目标（移动底盘到指定 fixture）
3. **B（layout shift）**：运行时把 microwave 平移到一个远处 counter 上
4. **C**：Microwave → Counter（把 `obj` 放到 counter 上的 `container` 上）

## 采集：人类演示（viewer / teleop）

在 `benchmark/robocasa` 目录下运行：
```bash
python -m grounding_v2.scripts.collect_demos_grounding_v2 \
  --environment GroundingV2MicrowavePickPlaceLayoutShift \
  --split pretrain
```

其它参数（robot / controller / device / out_dir 等）与 `robocasa.scripts.collect_demos` 完全一致。

## 采集：IK / 专家脚本（headless）

```bash
python -m grounding_v2.scripts.collect_expert_demos_microwave_layout_shift \
  --num_episodes 1 \
  --split pretrain \
  --controller WHOLE_BODY_IK
```

默认会尝试移动底盘完成干扰导航与 post-shift 追踪；如需关闭底盘导航（不推荐）：
```bash
python -m grounding_v2.scripts.collect_expert_demos_microwave_layout_shift --disable_base_nav
```

