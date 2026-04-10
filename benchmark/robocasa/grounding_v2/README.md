# grounding_v2 (Composite long-horizon tasks)

本目录提供 grounding_v2 的 long-horizon 任务与采集脚本（人类演示 / IK 专家）。

## 任务：`GroundingV2MicrowavePickPlaceLayoutShift`

环境实现见：
- `benchmark/robocasa/grounding_v2/envs/microwave_pick_place_layout_shift.py`

阶段（同一条 episode 内依次发生）：
1. **A**：Counter → Microwave（把 `obj` 放到微波炉里的盘子 `microwave_plate` 上）
2. **干扰任务**：2-3 个 NavigateKitchen 风格的导航子目标（移动底盘到指定 fixture）
3. **B（layout shift）**：运行时把 microwave 平移到一个远处 counter 上
4. **C**：Microwave → Counter（把 `obj` 放到 counter 上的 `container` 上）

## 任务：`GroundingV2DrawerPickPlaceCloseNavigateOpen`

环境实现见：
- `benchmark/robocasa/grounding_v2/envs/drawer_pick_place_close_navigate_open.py`

阶段（同一条 episode 内依次发生）：
1. **A**：Counter → Drawer（把 `obj` 放到 drawer 里）
2. **B**：CloseDrawer（完全关上 drawer）
3. **C**：NavigateKitchen 风格导航子目标（移动底盘到随机 fixture）
4. **D**：OpenDrawer（完全打开 drawer）
5. **E**：Drawer → Counter（把 `obj` 从 drawer 取出放到 counter 上）

## 采集：人类演示（viewer / teleop）

在 `benchmark/robocasa` 目录下运行：
```bash
python -m grounding_v2.scripts.collect_demos_grounding_v2 \
  --environment GroundingV2MicrowavePickPlaceLayoutShift \
  --split pretrain
```

其它参数（robot / controller / device / out_dir 等）与 `robocasa.scripts.collect_demos` 完全一致。

如需在每个子任务完成时保存**所有相机**的 RGB 观测截图（用于调试 staged task），可显式开启：
```bash
python -m grounding_v2.scripts.collect_demos_grounding_v2 \
  --environment GroundingV2DrawerPickPlaceCloseNavigateOpen \
  --split pretrain \
  --save_task_completion_rgb
```
默认会把截图写到每个 episode 的目录下：`task_completion_rgb/`（可用 `--task_completion_rgb_width/height` 调整分辨率）。

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
