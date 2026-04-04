# Long-Horizon Task System for Robocasa

完整的长时程机器人操作任务系统,包含四阶段任务定义、自动化数据收集、Web 遥操作录制和数据分析工具。

## 📋 目录

- [系统概述](#系统概述)
- [任务定义](#任务定义)
- [安装指南](#安装指南)
- [快速开始](#快速开始)
- [模块说明](#模块说明)
- [使用示例](#使用示例)
- [数据格式](#数据格式)
- [故障排除](#故障排除)

---

## 系统概述

本系统实现了一个复杂的长时程机器人操作任务,分为四个连续阶段:

### 任务流程

```
Phase 1 (T_A 放置)
  ↓ 机器人抓取物体并放入容器(如微波炉)

Phase 2 (中间干扰任务)
  ↓ 执行 2-3 个独立子任务(开关抽屉、柜门等)

Phase 3 (场景变换)
  ↓ 容器随机移动到新位置(含碰撞检测)

Phase 4 (T_B 取出)
  ↓ 从新位置取出物体
```

### 核心特性

✅ **完整的任务定义**: 基于 gymnasium.Env 接口的四阶段任务环境
✅ **自动化录制**: 程序化控制器执行任务并记录轨迹
✅ **Web 遥操作**: 无头渲染 + WebSocket 实时控制和录制
✅ **标准数据格式**: 导出为 robomimic 兼容的 HDF5 格式
✅ **工具集**: 碰撞检测、位置验证、数据分析和可视化

---

## 任务定义

### 可用物体类型

基于 `robocasa/models/objects/kitchen_objects.py` 中的定义,可抓取物体包括:

**水果**: apple, banana, orange, lemon, mango, peach, pear, kiwi, strawberry...
**蔬菜**: carrot, tomato, cucumber, broccoli, onion, bell_pepper, potato...
**食品**: bread, cheese, egg, milk, yogurt, cereal, canned_food...
**容器**: bowl, cup, mug, pot, pan, plate (部分可移动)
**其他**: can, bottle, bar, chocolate, candy...

### 容器类型

- **microwave**: 固定式微波炉(带门)
- **bowl**: 可移动碗
- **pot**: 可移动锅
- **cup**: 可移动杯子

### CLI 参数

```bash
--object_type      物体名称(如 apple, banana)
--container_type   容器名称(如 microwave, bowl)
--num_intermediate 中间任务数量(2 或 3)
```

---

## 安装指南

### 环境要求

- Python >= 3.8
- robocasa (已安装)
- MuJoCo >= 2.3.0

### 依赖安装

所有代码依赖标准库和已有的 robocasa 环境,无需额外安装。如需 Web 遥操作功能:

```bash
pip install fastapi uvicorn websockets pillow
```

如需数据可视化:

```bash
pip install matplotlib
```

### 文件结构

```
benchmark/robocasa/grounding/
├── long_horizon_env.py              # 四阶段任务环境定义
├── script_a_automated_recording.py  # 自动化录制脚本
├── script_b_teleoperation_server.py # 遥操作服务端
├── script_b_teleoperation_client.html # 遥操作 Web 客户端
├── utils.py                         # 工具模块
├── main.py                          # 统一 CLI 入口
└── README.md                        # 本文档
```

---

## 快速开始

### 1. 测试环境

```bash
cd /home/yan/myProjects/GenMem/benchmark/robocasa/grounding
python main.py test --object_type apple --container_type microwave
```

### 2. 自动化数据收集 (Script A)

```bash
# 收集 10 条演示,使用 apple 和 microwave
python main.py collect \
  --object_type apple \
  --container_type microwave \
  --num_intermediate 2 \
  --num_demos 10 \
  --output_dir ./data/automated \
  --analyze

# 带可视化渲染
python main.py collect --num_demos 5 --render
```

### 3. Web 遥操作录制 (Script B)

```bash
# 启动遥操作服务器
python main.py teleoperate \
  --object_type apple \
  --container_type microwave \
  --port 8000

# 然后在浏览器打开: http://localhost:8000
```

#### Web 客户端功能

- **实时画面**: RGB + Depth 双摄像头视图(20 FPS)
- **虚拟摇杆**: 鼠标拖拽控制 XY 方向移动
- **关节滑块**: 精细控制各关节速度
- **键盘映射**:
  - `W/A/S/D`: XY 平面移动
  - `Q/E`: Z 轴上下移动
  - `G`: 夹爪开合
  - `R`: 开始/停止录制
  - `Space`: 重置环境
- **录制按钮**: 开始/停止按钮,自动保存 HDF5 文件

### 4. 数据分析

```bash
# 分析 HDF5 文件
python main.py analyze \
  --hdf5 ./data/automated/long_horizon_apple_microwave_20250101_120000.hdf5 \
  --visualize

# 合并多个 HDF5 文件
python main.py merge \
  --inputs file1.hdf5 file2.hdf5 file3.hdf5 \
  --output merged_dataset.hdf5
```

---

## 模块说明

### `long_horizon_env.py` - 任务环境

**类**: `LongHorizonTask(Kitchen)`

**关键方法**:
- `reset()`: 重置环境,初始化四阶段状态机
- `step(action)`: 执行动作,自动处理阶段转换
- `_check_phase_X_complete()`: 各阶段完成条件检测
- `_sample_new_container_position()`: Phase 3 位置采样(含碰撞检测)

**状态机**:
```python
Phase 1 → 检测物体是否在容器内 → Phase 2
Phase 2 → 完成 N 个中间任务 → Phase 3 (自动执行变换)
Phase 3 → 自动完成 → Phase 4
Phase 4 → 检测物体是否取出 → COMPLETED
```

### `script_a_automated_recording.py` - 自动化录制

**类**:
- `ProgrammaticController`: 程序化控制器
  - `execute_phase_1()`: 抓取并放置物体
  - `execute_phase_2()`: 执行中间任务
  - `execute_phase_4()`: 取出物体
- `TrajectoryBuffer`: 轨迹缓冲区
- `RobomimicHDF5Writer`: HDF5 导出工具

**执行流程**:
1. 创建环境和控制器
2. 为每个阶段生成动作序列
3. 执行并记录 state-action-reward
4. 导出为 robomimic HDF5 格式

### `script_b_teleoperation_server.py` - 遥操作服务端

**类**:
- `HeadlessRenderer`: 无头渲染器
  - 使用 `mujoco.Renderer` 进行离屏渲染
  - `render_rgb()`: 渲染 RGB 图像
  - `render_depth()`: 渲染深度图
  - `encode_image_base64()`: Base64 编码用于 Web 传输
- `TeleoperationSession`: 会话管理
  - 环境状态维护
  - 录制控制
  - 控制命令处理

**FastAPI 端点**:
- `GET /`: 返回 Web 客户端 HTML
- `WebSocket /ws/{session_id}`: 双向通信
  - 发送: 视频帧(RGB/Depth)
  - 接收: 控制命令、录制命令

**消息格式**:
```json
// 客户端 → 服务端
{
  "type": "control",
  "command": {
    "type": "velocity",
    "values": [0.1, 0.2, 0.0, ...]
  }
}

// 服务端 → 客户端
{
  "type": "frame",
  "data": {
    "rgb": "data:image/jpeg;base64,...",
    "depth": "data:image/jpeg;base64,..."
  }
}
```

### `utils.py` - 工具模块

**类**:
- `CollisionChecker`: MuJoCo 碰撞检测
- `PositionValidator`: 位置合法性验证(边界+碰撞)
- `HDF5Analyzer`: HDF5 文件分析和合并
- `TrajectoryVisualizer`: 轨迹可视化(matplotlib)

---

## 使用示例

### 示例 1: 自定义物体和容器

```bash
# 使用 banana 和 bowl
python main.py collect \
  --object_type banana \
  --container_type bowl \
  --num_intermediate 3 \
  --num_demos 20
```

### 示例 2: 批量收集不同配置

```bash
#!/bin/bash
# batch_collect.sh

OBJECTS=("apple" "banana" "orange")
CONTAINERS=("microwave" "bowl" "pot")

for obj in "${OBJECTS[@]}"; do
  for cont in "${CONTAINERS[@]}"; do
    python main.py collect \
      --object_type "$obj" \
      --container_type "$cont" \
      --num_demos 10 \
      --output_dir "./data/${obj}_${cont}"
  done
done
```

### 示例 3: 遥操作数据收集工作流

```bash
# 1. 启动服务器(在终端 1)
python main.py teleoperate --port 8000

# 2. 在浏览器打开 http://localhost:8000

# 3. 完成操作后分析数据(在终端 2)
python main.py analyze \
  --hdf5 ./data/teleoperation/demo_user_*.hdf5 \
  --visualize
```

### 示例 4: Python API 使用

```python
from long_horizon_env import LongHorizonTask, TaskPhase

# 创建环境
env = LongHorizonTask(
    object_type="apple",
    container_type="microwave",
    num_intermediate=2
)

obs, info = env.reset()

# 运行直到完成
done = False
while not done:
    # 使用你的策略生成动作
    action = your_policy(obs)

    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # 监测阶段转换
    if 'phase_transition' in info:
        print(f"Phase changed: {info['phase_transition']}")
        print(f"Current phase: {env.current_phase.name}")

env.close()
```

---

## 数据格式

### HDF5 结构 (robomimic 兼容)

```
demo_file.hdf5
│
├── data/
│   ├── demo_0/
│   │   ├── obs/
│   │   │   └── flat          # 观察数组 (T, obs_dim)
│   │   ├── actions           # 动作数组 (T, action_dim)
│   │   ├── rewards           # 奖励数组 (T,)
│   │   └── dones             # 终止标志 (T,)
│   ├── demo_1/
│   └── ...
│
└── mask                      # 成功标志 (N_demos,)
```

### 加载数据示例

```python
import h5py
import numpy as np

with h5py.File('demo.hdf5', 'r') as f:
    # 读取第一条演示
    demo_0 = f['data/demo_0']

    obs = demo_0['obs/flat'][:]      # (T, obs_dim)
    actions = demo_0['actions'][:]   # (T, action_dim)
    rewards = demo_0['rewards'][:]   # (T,)

    # 检查成功标志
    success = f['mask'][0]

    print(f"Demo length: {len(actions)}")
    print(f"Success: {success}")
```

---

## 故障排除

### 问题 1: ImportError - 无法导入 robocasa

**症状**: `ModuleNotFoundError: No module named 'robocasa'`

**解决**:
```bash
# 确保在 robocasa 环境中运行
cd /home/yan/myProjects/GenMem/benchmark/robocasa
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### 问题 2: MuJoCo 渲染错误

**症状**: `mujoco.FatalError: gladLoadGL error`

**解决**:
```bash
# 使用无头模式(不渲染)
python main.py collect --num_demos 10
# (不加 --render 标志)
```

### 问题 3: WebSocket 连接失败

**症状**: 浏览器显示 "Disconnected"

**解决**:
1. 检查服务器是否运行: `ps aux | grep teleoperation`
2. 检查端口占用: `lsof -i :8000`
3. 使用正确的 URL: `http://localhost:8000` (不是 `127.0.0.1`)

### 问题 4: 容器位置变换失败

**症状**: Phase 3 阶段卡住,容器未移动

**解决**:
- 检查 `_sample_new_container_position()` 中的边界设置
- 增加采样尝试次数 `max_attempts`
- 放宽碰撞阈值 `contact_threshold`

### 问题 5: HDF5 文件损坏

**症状**: `OSError: Unable to open file`

**解决**:
```python
import h5py

# 尝试恢复数据
with h5py.File('broken.hdf5', 'r', libver='latest') as f:
    # 如果失败,使用 h5repack 修复
    # h5repack broken.hdf5 fixed.hdf5
```

---

## 高级配置

### 自定义控制器

在 `script_a_automated_recording.py` 中修改 `ProgrammaticController`:

```python
class CustomController(ProgrammaticController):
    def _move_to_position(self, target_pos, offset=[0, 0, 0]):
        # 实现你的 IK 求解器或 MPC 控制器
        actions = my_ik_solver(target_pos + np.array(offset))
        return actions
```

### 自定义中间任务

在 `long_horizon_env.py` 中修改 `_generate_intermediate_tasks()`:

```python
def _generate_intermediate_tasks(self):
    # 添加自定义任务
    task_pool = [
        "open_drawer",
        "turn_on_stove",        # 新任务
        "press_microwave_button",  # 新任务
    ]
    self.intermediate_task_list = np.random.choice(
        task_pool, size=self.num_intermediate, replace=False
    ).tolist()
```

### 修改阶段完成条件

```python
def _check_phase_1_complete(self) -> bool:
    # 自定义完成检测逻辑
    obj_in_container = self._is_object_in_container("target_obj")
    gripper_far = self._gripper_far_from_object()

    return obj_in_container and gripper_far
```

---

## 性能优化

### 1. 加速数据收集

```python
# 使用多进程并行收集
from multiprocessing import Pool

def collect_demo(args):
    # ... 单个 demo 收集逻辑
    pass

with Pool(4) as p:  # 4 个并行进程
    p.map(collect_demo, demo_configs)
```

### 2. 优化 WebSocket 帧率

在 `script_b_teleoperation_server.py` 中调整:

```python
# 降低帧率以节省带宽(从 20 FPS 到 10 FPS)
await asyncio.sleep(0.1)  # 原来是 0.05
```

### 3. 减少 HDF5 文件大小

```python
# 使用压缩
demo_grp.create_dataset(
    "actions",
    data=traj["actions"],
    compression="gzip",
    compression_opts=4  # 压缩级别 1-9
)
```

---

## 引用和致谢

本系统基于以下开源项目构建:

- **robocasa**: Kitchen environment framework
- **robomimic**: Imitation learning library
- **MuJoCo**: Physics simulation engine

如使用本系统发表论文,请引用:

```bibtex
@misc{long_horizon_robocasa,
  author = {AI Development Engineer},
  title = {Long-Horizon Task System for Robocasa},
  year = {2025},
  url = {https://github.com/your-repo}
}
```

---

## 许可证

MIT License - 详见 LICENSE 文件

---

## 联系方式

如有问题或建议,请提交 Issue 或联系:
- Email: your-email@example.com
- GitHub: @your-username

**最后更新**: 2025-01-XX
