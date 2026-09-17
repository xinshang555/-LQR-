# C++ MuJoCo 仿真

本目录按照实车工程使用的三层结构组织：

- `mujoco_interface.*`：唯一允许访问 `mjModel/mjData` 的接口层，读取状态并把
  控制器输出的物理扭矩转换为 MuJoCo `ctrl`；
- `main.cpp`：运行层，负责仿真时序、可视化、键鼠交互和模块调度；
- `wheel_leg_controller.*`：控制层，只依赖 `RobotState/UserCommand`，执行 LQR、
  前进和航向控制并输出 N·m；
- `robot_types.hpp`：仿真与实车共同使用的数据契约。

## 构建与运行

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j
./cpp/build/wheel_leg_sim
ctest --test-dir cpp/build --output-on-failure
```

CMake 会自动寻找 Python `mujoco` 包中的头文件和动态库，也可以显式传入
`-DMUJOCO_ROOT=/path/to/mujoco`。可视化依赖 GLFW 3 开发包。

键盘：`W/S` 调速、`A/D` 调航向、`Space/X` 停止保持、`R` 复位、`Esc` 退出。
鼠标左键旋转视角、右键平移、中键或滚轮缩放。

无窗口运行示例：

```bash
./cpp/build/wheel_leg_sim --headless --seconds 5 --vx 0.02
./cpp/build/wheel_leg_sim --headless --seconds 20 --target-yaw 5
```
