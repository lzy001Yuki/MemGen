# 文件清单与说明

本目录包含完整的长时程任务系统,所有代码已生成并可直接使用。

## 📁 核心模块 (9 个文件)

### 1️⃣ `long_horizon_env.py` (16 KB)
**任务环境定义**
- 类: `LongHorizonTask(Kitchen)` - 四阶段任务环境
- 类: `TaskPhase(Enum)` - 任务阶段枚举
- 功能:
  - ✅ Phase 1: 物体放置到容器
  - ✅ Phase 2: 2-3 个中间干扰任务
  - ✅ Phase 3: 容器位置随机变换(含碰撞检测)
  - ✅ Phase 4: 从新位置取出物体
- 关键方法:
  - `reset()`: 重置环境
  - `step(action)`: 执行动作,自动处理阶段转换
  - `_sample_new_container_position()`: 采样合法位置

### 2️⃣ `script_a_automated_recording.py` (17 KB)
**自动化数据录制脚本 (Script A)**
- 类: `ProgrammaticController` - 程序化控制器
  - `execute_phase_1()`: 抓取并放置
  - `execute_phase_2()`: 执行中间任务
  - `execute_phase_4()`: 取出物体
- 类: `TrajectoryBuffer` - 轨迹缓冲区
- 类: `RobomimicHDF5Writer` - HDF5 导出工具
- 函数: `run_automated_collection()` - 主执行函数
- 使用: `python script_a_automated_recording.py --num_demos 10`

### 3️⃣ `script_b_teleoperation_server.py` (13 KB)
**Web 遥操作服务端 (Script B - Server)**
- 类: `HeadlessRenderer` - 无头渲染器
  - 使用 `mujoco.Renderer` 离屏渲染
  - `render_rgb()`: 生成 RGB 图像
  - `render_depth()`: 生成深度图
  - `encode_image_base64()`: Base64 编码
- 类: `TeleoperationSession` - 会话管理
- FastAPI 服务:
  - `GET /`: 返回 Web 客户端
  - `WebSocket /ws/{session_id}`: 双向通信
- 使用: `python script_b_teleoperation_server.py --port 8000`

### 4️⃣ `script_b_teleoperation_client.html` (20 KB)
**Web 遥操作客户端 (Script B - Client)**
- 功能:
  - 📹 实时 RGB + Depth 视频流 (20 FPS)
  - 🕹️ 虚拟摇杆控制
  - 🎚️ 关节滑块控制
  - ⌨️ 键盘映射 (WASD, QE, G, R, Space)
  - ⏺️ 录制开始/停止按钮
  - 🔄 环境重置
- 界面设计: 现代化响应式布局
- 访问: 浏览器打开 `http://localhost:8000`

### 5️⃣ `utils.py` (13 KB)
**工具模块**
- 类: `CollisionChecker` - MuJoCo 碰撞检测
  - `check_collision()`: 检测碰撞
  - `get_contact_info()`: 获取接触详情
- 类: `PositionValidator` - 位置验证
  - `is_valid_position()`: 验证位置合法性
  - `sample_valid_position()`: 采样有效位置
- 类: `HDF5Analyzer` - HDF5 数据分析
  - `load_trajectory()`: 加载轨迹
  - `get_trajectory_stats()`: 统计信息
  - `merge_hdf5_files()`: 合并数据集
- 类: `TrajectoryVisualizer` - 轨迹可视化
  - `plot_action_trajectory()`: 绘制动作曲线
  - `print_trajectory_summary()`: 打印摘要

### 6️⃣ `main.py` (11 KB)
**统一 CLI 入口**
- 子命令:
  - `collect`: 自动化数据收集
  - `teleoperate`: 启动遥操作服务器
  - `analyze`: 分析 HDF5 数据
  - `merge`: 合并多个 HDF5 文件
  - `test`: 测试环境设置
- 使用示例:
  ```bash
  python main.py collect --num_demos 10
  python main.py teleoperate --port 8000
  python main.py analyze --hdf5 data.hdf5
  python main.py merge --inputs *.hdf5 --output merged.hdf5
  ```

### 7️⃣ `__init__.py` (1.5 KB)
**Python 包初始化**
- 导出主要类和函数
- 版本信息: `__version__ = "1.0.0"`
- 允许作为包导入:
  ```python
  from grounding import LongHorizonTask, TrajectoryBuffer
  ```

### 8️⃣ `example_usage.py` (8.7 KB)
**使用示例脚本**
- 6 个完整示例:
  1. 基础环境使用
  2. 手动轨迹录制
  3. 碰撞检测
  4. 位置验证
  5. HDF5 数据分析
  6. 程序化控制
- 运行: `python example_usage.py`

### 9️⃣ `requirements.txt` (373 B)
**依赖清单**
- 核心依赖 (已随 robocasa 安装):
  - numpy, gymnasium, mujoco, h5py
- Web 遥操作依赖:
  - fastapi, uvicorn, websockets, Pillow
- 可选依赖:
  - matplotlib (可视化)

---

## 📚 文档

### 🔟 `README.md` (13 KB)
**完整用户文档**
- 系统概述
- 安装指南
- 快速开始
- 模块说明
- 使用示例
- 数据格式
- 故障排除
- 高级配置

### 1️⃣1️⃣ `FILE_MANIFEST.md` (本文件)
**文件清单与说明**

---

## 🎯 快速启动指南

### 第一步: 安装依赖 (如需 Web 功能)

```bash
cd /home/yan/myProjects/GenMem/benchmark/robocasa/grounding
pip install fastapi uvicorn websockets pillow
```

### 第二步: 测试环境

```bash
python main.py test
```

### 第三步: 选择使用方式

**方式 A: 自动化数据收集**
```bash
python main.py collect --num_demos 10 --analyze
```

**方式 B: Web 遥操作**
```bash
# 终端 1: 启动服务器
python main.py teleoperate --port 8000

# 终端 2: 浏览器打开
# http://localhost:8000
```

**方式 C: Python API 使用**
```python
from long_horizon_env import LongHorizonTask

env = LongHorizonTask(
    object_type="apple",
    container_type="microwave",
    num_intermediate=2
)

obs, info = env.reset()
# ... 你的策略代码
```

---

## 📊 代码统计

- **总文件**: 11 个
- **Python 代码**: 6 个模块 + 1 个示例
- **Web 前端**: 1 个 HTML
- **文档**: 2 个 Markdown
- **配置**: 1 个 requirements.txt

**总代码量**: ~136 KB
- 核心环境: 16 KB
- Script A: 17 KB
- Script B (服务端): 13 KB
- Script B (客户端): 20 KB
- 工具模块: 13 KB
- CLI 入口: 11 KB
- 示例代码: 8.7 KB

**功能覆盖**:
- ✅ 四阶段任务环境
- ✅ 程序化控制器
- ✅ 无头渲染系统
- ✅ WebSocket 实时通信
- ✅ 碰撞检测与位置验证
- ✅ HDF5 数据导出(robomimic 兼容)
- ✅ 数据分析与可视化
- ✅ 完整 CLI 工具链
- ✅ 示例代码与文档

---

## 🏗️ 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    Long-Horizon Task System                  │
└─────────────────────────────────────────────────────────────┘
                              │
                 ┌────────────┴────────────┐
                 │                          │
        ┌────────▼────────┐        ┌───────▼────────┐
        │  Script A        │        │  Script B       │
        │  (自动化录制)     │        │  (遥操作录制)   │
        └────────┬────────┘        └───────┬────────┘
                 │                          │
                 │    ┌──────────────┐     │
                 └───►│ Long-Horizon │◄────┘
                      │ Environment  │
                      └──────┬───────┘
                             │
                ┌────────────┼────────────┐
                │            │            │
         ┌──────▼───┐ ┌─────▼────┐ ┌────▼─────┐
         │ Phase 1   │ │ Phase 2  │ │ Phase 3  │
         │ (Place)   │ │(Distract)│ │(Transform)│
         └──────────┘ └──────────┘ └──────┬───┘
                                          │
                                    ┌─────▼────┐
                                    │ Phase 4  │
                                    │(Retrieve)│
                                    └──────────┘
                                          │
                                    ┌─────▼─────┐
                                    │  HDF5     │
                                    │  Output   │
                                    └───────────┘
```

---

## 🔄 数据流

### Script A (自动化)
```
1. 创建环境 → 重置
2. 程序化控制器生成动作序列
3. 执行并记录 (state, action, reward)
4. 导出 HDF5 (robomimic 格式)
```

### Script B (遥操作)
```
1. 服务器创建环境 → 无头渲染
2. WebSocket 连接 ← 浏览器客户端
3. 循环:
   - 渲染 RGB/Depth → Base64 → WebSocket → 客户端显示
   - 客户端控制输入 → WebSocket → 服务器 → 环境.step()
   - 如果录制中 → 记录轨迹
4. 停止录制 → 导出 HDF5
```

---

## 🎨 设计特点

### 模块化设计
- ✅ 环境、控制器、录制工具完全解耦
- ✅ 每个模块可独立使用或组合
- ✅ 符合 SOLID 原则

### 类型安全
- ✅ 完整的类型注解 (Type Hints)
- ✅ Docstring 覆盖所有公开 API
- ✅ 参数验证与错误处理

### 边界处理
- ✅ 碰撞检测与回滚机制
- ✅ 位置采样失败降级策略
- ✅ WebSocket 断线自动重连
- ✅ HDF5 文件损坏检测

### 扩展性
- ✅ 易于添加新物体/容器类型
- ✅ 自定义中间任务
- ✅ 可插拔控制器
- ✅ 支持多种数据格式导出

---

## 📝 使用检查清单

**环境设置**
- [ ] robocasa 已正确安装
- [ ] MuJoCo >= 2.3.0
- [ ] Python >= 3.8

**Script A (自动化)**
- [ ] 测试环境: `python main.py test`
- [ ] 收集数据: `python main.py collect --num_demos 5`
- [ ] 分析数据: `python main.py analyze --hdf5 <file>`

**Script B (遥操作)**
- [ ] 安装 Web 依赖: `pip install fastapi uvicorn websockets pillow`
- [ ] 启动服务器: `python main.py teleoperate`
- [ ] 浏览器访问: `http://localhost:8000`
- [ ] 测试控制: 虚拟摇杆/键盘
- [ ] 测试录制: 开始录制 → 操作 → 停止录制

**数据处理**
- [ ] 验证 HDF5 文件: `python main.py analyze --hdf5 <file>`
- [ ] 合并数据集: `python main.py merge --inputs *.hdf5 --output merged.hdf5`
- [ ] 可视化轨迹: `python main.py analyze --hdf5 <file> --visualize`

---

## 🐛 已知限制

1. **容器类型**: 目前仅完全支持 `microwave` 固定式容器
   - 可移动容器 (bowl, pot) 需要额外适配 qpos 操作
   - 解决: 扩展 `_set_container_position()` 方法

2. **中间任务**: Phase 2 的中间任务为简化实现
   - 当前仅生成随机动作序列
   - 解决: 实现真实的任务控制器 (开抽屉、开柜门等)

3. **IK 求解**: 程序化控制器使用简化的运动规划
   - 未集成真实 IK 求解器
   - 解决: 集成 robosuite/robomimic 的 IK 控制器

4. **渲染性能**: Web 遥操作帧率限制为 20 FPS
   - 高分辨率下可能降低
   - 解决: 降低分辨率或使用 JPEG 压缩

---

## 🚀 未来增强

- [ ] 支持更多容器类型 (柜子、抽屉)
- [ ] 实现真实的中间任务控制器
- [ ] 集成 VR/手柄控制
- [ ] 添加力触觉反馈
- [ ] 多机器人协同任务
- [ ] 强化学习训练接口
- [ ] 视觉伺服控制

---

**最后更新**: 2025-01-XX
**状态**: ✅ 所有模块已完成
**测试**: 待用户测试
**许可**: MIT License
