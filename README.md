# 轮腿机器人 URDF/MJCF 与 LQR 仿真

本项目完成了 `model/car.STEP -> URDF -> MJCF` 的模型核验，以及一个可复现的
10 状态离散 LQR 训练与 MuJoCo 闭环仿真流程。主仿真已按实车工程形式改为
C++ 三层架构；Python 代码保留用于增益训练和交叉回归。

## 当前结果

- URDF：关节树、坐标系、网格、质量、质心和惯量检查通过；总质量 25.530 g。
- MJCF：与 URDF 的无接触质量矩阵相对误差约 `5.9e-15`。
- 轮胎接触：视觉仍使用 CAD STL，碰撞改用 CAD 真值 `R=8 mm、宽=7 mm` 的圆柱；
  `home` 时两轮同时接触地面。
- LQR：24,000 个局部样本辨识，10 状态可控性秩 10/10，离散闭环谱半径
  `0.974106`。
- 仿真：开环约 0.05 s 触发倒下；闭环在 0、+0.57°、-0.57°初始扰动下均通过
  5 s 回归测试，稳态最大俯仰约 0.37-0.38°。

“站住”的判据同时要求：`|pitch| <= 5°`、机身高度相对 home 在 ±5 mm 内、
前进轨迹误差不超过 50 mm。因此躺倒、飞起或失控跑车不会被误判为成功。

## 目录

```text
model/car.STEP                 CAD 几何源
质量属性/*.txt                 SolidWorks 质量属性
urdf/car.urdf                  CAD 原坐标系权威 URDF
urdf/car_floating.urdf         floating-base URDF
urdf/car_zup.urdf              Z-up URDF
urdf/car.xml                   MuJoCo 仿真模型
urdf/meshes/                   link-local 米制 STL
LQR计算代码/parameter.py       解析模型参数
LQR计算代码/calculate.py       连续解析 LQR 框架
sim/state_map.py               MuJoCo 到 10 状态映射
sim/design_gain.py             局部辨识 + 离散 DARE（训练）
sim/lqr_gain.npz               已训练增益与元数据
sim/run_lqr.py                 闭环/开环仿真
sim/test_lqr.py                端到端回归测试
cpp/src/main.cpp               C++ 运行层：仿真、渲染、键鼠和控制循环
cpp/src/mujoco_interface.cpp   C++ 接口层：状态读取、扭矩输出
cpp/src/wheel_leg_controller.cpp C++ 控制层：LQR、速度和航向
cpp/include/robot_types.hpp    仿真/实车共用数据结构
```

## 环境与运行

已验证环境：Ubuntu 24.04 / Python 3.12 / MuJoCo 3.12 / NumPy 1.26 /
SciPy 1.11。

```bash
# 安装依赖
pip install mujoco numpy scipy matplotlib

# 1. 模型和状态映射核验
python3 sim/check_model.py
python3 sim/test_state_map.py

# 2. 重新辨识并训练 LQR（固定 seed，可复现）
python3 sim/design_gain.py

# 3. 开环与闭环
python3 sim/run_lqr.py --open-loop --seconds 2
python3 sim/run_lqr.py --seconds 10 --kick 0.01
python3 sim/run_lqr.py --view
python3 sim/run_lqr.py --plot
python3 sim/run_lqr.py --seconds 5 --vx 0.02
python3 sim/run_lqr.py --seconds 20 --target-yaw 5

# 4. 一键回归
python3 sim/test_lqr.py
```

### 交互控制

主程序使用 C++ 构建：

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j
./cpp/build/wheel_leg_sim
ctest --test-dir cpp/build --output-on-failure
```

运行 `./cpp/build/wheel_leg_sim`（Windows 也可双击 `启动LQR仿真.bat`）后，
先单击 MuJoCo 窗口使其获得键盘焦点：

- `W` / `S`：每次按键将前进速度增加 / 减少 10 mm/s，范围为 ±40 mm/s；
- `A` / `D`：每次按键将目标航向向左 / 向右调整 2°；
- `Space` 或 `X`：停止平移，并保持当前航向；
- `R`：回到 home 姿态并清空运动指令。

按键只在按下瞬间改变一次指令，长按不会连续累加到最大值。控制器对输出扭矩增加
了变化率限制；检测到俯仰超过 12°或高度误差超过 10 mm 后会关闭电机，按 `R`
才能重新启动。这可以防止跌倒后的饱和控制与硬碰撞把小车弹飞。

航向命令经过 0.2°误差限幅后再送入 LQR，避免原偏航高增益让电机在阶跃指令下
饱和。命令行的 `--vx` 会自动限幅到 ±0.04 m/s；`--target-yaw` 的单位是度。
详细的 C++ 分层与构建说明见 `cpp/README.md`。

Windows 本项目可直接通过 WSL 运行：

```powershell
wsl -d Ubuntu-24.04 -e bash -lc "cd '/mnt/d/Files/轮腿训练/newstart' && python3 sim/test_lqr.py"
```

## LQR 设计说明

状态顺序为：

```text
[x, x_dot, yaw, yaw_dot, leg_lower, leg_lower_dot,
 leg_upper, leg_upper_dot, body_pitch, body_pitch_dot]
```

直接在硬接触的几何姿态上调用有限差分会得到依赖差分步长的接触假模态，而且
`home` 只是几何初态，并非 `qacc=0` 的平衡点。`design_gain.py` 因此采用闭环
局部辨识：

1. 用保守的小增益采样控制器保持直立接触模态；
2. 对 4 个执行器注入小幅、固定随机种子的相关激励；
3. 拟合闭环离散模型，并按 `A_open = A_closed + B K_baseline` 恢复开环模型；
4. 验证可控性秩为 10；
5. 解离散代数 Riccati 方程，保存 `K`、参考状态、前馈控制和验证元数据。

最终控制律为：

```text
ctrl = clip(u_eq - K @ (x - x_cmd), -1, 1)
```

这里的输入直接是 MuJoCo 的归一化 `ctrl`，不会再混淆 N·m 与 actuator gear。

## 模型审查结论

URDF 本体是正确的：5 个 link、4 个 hinge、两条 40 mm 虚拟腿、两轮轴距
35 mm，质量和惯量均与 CAD 报告一致。注意：

- `car.urdf` 的“下”是 CAD `-Y`，MuJoCo 仿真应使用 `car.xml` 或
  `car_zup.urdf`。
- 不应直接用根 link 为 `body` 的 `car.urdf` 跑 MuJoCo，否则导入器会把根 link
  合入 world 并丢失其质量；使用 floating/Z-up 版本。
- 视觉网格不适合作为轮胎接触面。圆柱碰撞体消除了单轮承载和左右接触不对称，
  但不会改变显式指定的质量与惯量。
- 仿真训练使用 `1e-7 kg*m^2` 的等效轮电机转子惯量。该值是明确记录的仿真假设；
  接入实车前应换成电机转子惯量乘传动比平方后的实测值，并重新训练增益。

`LQR计算代码/calculate.py` 保留了题目给定的连续解析框架和 CAD 真值参数，适合
公式对照；实际接触仿真使用 `sim/design_gain.py` 的离散辨识模型，因为它包含轮胎
接触、采样周期、执行器 gear 和电机惯量的真实影响。
