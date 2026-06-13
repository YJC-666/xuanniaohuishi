# COREINNOVATION//开源计划

```
[ SYSTEM_STRATEGY ]: 保护核心算法底座，开源上层应用
```

>> GitHub 开源仓库: https://github.com/YJC-666/xuanniaohuishi

---

## PHASE_01_硬件开源 · [实体外设共创]

| 开放资产 | 社区价值 |
|---------|---------|
| 3D打印轻量化喷涂挂载模块的结构源文件 | 允许全球创客依据不同无人机底盘自行适配改装 |

## PHASE_02_接口开放 · [跨模态API共享]

| 开放资产 | 社区价值 |
|---------|---------|
| 提供"视觉大模型美学转译引擎"标准API | 邀请全球开发者与艺术家，接入多元化的创意表达 |

## PHASE_03_数据反哺 · [学术测试数据集]

| 开放资产 | 社区价值 |
|---------|---------|
| 共享"流体力学与飞行轨迹"物理反馈数据集 | 助力全球科研机构在"实体具身智能"领域的研究 |

---

```
STATUS: ECOSYSTEM_ACTIVE
```

> "用一段代码连接中美创客，让艺术的火种属于全世界！"

---

## 项目简介

**xuanniaohuishi** — 无人机精准喷涂系统。

基于 ROS 的自主无人机喷涂平台，集成 YOLOv8 实时目标检测、自适应网格路径规划与智能任务控制。

### 技术栈

- **框架**: ROS (Melodic/Noetic)
- **视觉**: YOLOv8 + OpenCV
- **路径规划**: 自适应网格(50cm×50cm)全覆盖遍历
- **UI**: PyQt5 地面站
- **坐标系**: 统一方格坐标系(B1 A9 ~ B7 A1)

### 快速开始

```bash
# 依赖安装
pip install torch ultralytics opencv-python pyyaml
sudo apt install ros-${ROS_DISTRO}-mavros ros-${ROS_DISTRO}-cv-bridge

# 启动
roslaunch xuanniaohuishi mission.launch
```

---

<p align="center">
  <sub>COREINNOVATION // 用代码连接世界</sub>
</p>