"""MuJoCo <-> LQR 状态映射层。

`LQR计算代码/calculate.py` 里的 K 是给一个 **10 维降阶模型** 用的:

    x_lqr = [x, x', y, y', tl, tl', tr, tr', f, f']

其中 (见 calculate.py 的说明与 parameter.py 的坐标约定)
    x   前后位移 (m)        y   高度 (m)
    tl  左腿虚拟腿角 (rad)   tr  右腿虚拟腿角 (rad)
    f   偏航角 (rad)

而 `urdf/car.xml` 是完整的多体模型,nq=11 / nv=10:

    qpos = [px, py, pz, qw, qx, qy, qz, hip_lower, wheel_lower, hip_upper, wheel_upper]
    qvel = [vx, vy, vz, wx, wy, wz, d_hip_lower, d_wheel_lower, d_hip_upper, d_wheel_upper]

**这两边的 tl / tr 不是同一个东西**:LQR 的 tl 是「虚拟腿角」(髋到轮的连线相对竖直的夹角),
而 MuJoCo 的 hip_*_joint 是「物理髋关节角」。因此不能把 qpos 直接塞进 K,
必须经过本模块换算。这正是"能跑但物理是错的"最容易发生的地方。

------------------------------------------------------------------------------
坐标/符号约定(全部由 probe 实测确定,不是猜的)
------------------------------------------------------------------------------
世界系:  +X 前方, +Y 侧向(左), +Z 上, 重力 (0,0,-9.81)
侧向轴(CAD Z)= 世界 **-Y**(4 个 hinge 的实测世界轴向都是 (0,-1,0))

髋关节角 hip_q 的实测效果(home 位姿,hip=0 时轮子在髋的正下方偏后 1.8 mm):

    hip_q   wheel-hip dx     dz
    -0.40     -17.22 mm   -36.10 mm
     0.00      -1.81 mm   -39.96 mm
    +0.40     +13.90 mm   -37.51 mm

→ hip_q 增大,轮子相对髋**向前**摆。
→ 定义虚拟腿角  tl = atan2(dx, -dz)  (dx,dz 为轮相对髋的世界位移),
  得到 -0.445 / -0.045 / +0.355 rad,与 hip_q 近似 1:1。
  注意 hip_q=0 时 tl = -0.045 rad ≠ 0:这是真实的装配零位偏置
  (腿不是严格竖直的),必须保留,否则稳态会有常值倾角。

------------------------------------------------------------------------------
平面化(2D)假设
------------------------------------------------------------------------------
本机两条腿沿 **Y(侧向)** 并排、在 **X-Z 矢状面** 内前后摆动,髋轴与轮轴都是侧向轴。
LQR 模型正是一个**平面倒立摆**,所以:

  * 前后平衡只由 X-Z 面内量决定;
  * 左右腿在 X-Z 面的运动通过机身**耦合**在一起;
  * 取 **两腿均值** 作为平面内的等效虚拟腿。

两腿的**差值**(打架/不同步)不混进这里,否则会把「两条腿不一致」误判成
「机身倾斜」。同步由 run_lqr.py 里的一个独立项闭环。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# qvel 索引(car.xml 实测)
DOF_BASE_POS = slice(0, 3)   # vx, vy, vz
DOF_BASE_ROT = slice(3, 6)   # wx, wy, wz
DOF_HIP_LOWER = 6
DOF_WHEEL_LOWER = 7
DOF_HIP_UPPER = 8
DOF_WHEEL_UPPER = 9

# qpos 索引
QP_BASE_POS = slice(0, 3)
QP_BASE_QUAT = slice(3, 7)
QP_HIP_LOWER = 7
QP_WHEEL_LOWER = 8
QP_HIP_UPPER = 9
QP_WHEEL_UPPER = 10

# 侧向轴 = 世界 -Y
LATERAL_AXIS = np.array([0.0, -1.0, 0.0])


@dataclass
class LegState:
    """单条腿的虚拟腿状态。"""
    lean: float        # 虚拟腿角 tl (rad):atan2(dx, -dz),正 = 轮在髋前方
    lean_rate: float   # 虚拟腿角速度 (rad/s)
    length: float      # 髋到轮的连线长度 (m)
    wheel_rate: float  # 轮子自转角速度 (rad/s)


def _quat_to_pitch(quat: np.ndarray) -> float:
    """从四元数提取绕侧向轴的机身俯仰角(rad)。正 = 前倾。

    侧向轴是世界 -Y,所以绕世界 +Y 的转角取负。
    """
    w, x, y, z = quat
    siny = 2.0 * (w * y + z * x)
    siny = max(-1.0, min(1.0, siny))
    return -math.asin(siny)


def leg_state(mujoco_data, lower: bool = True) -> LegState:
    """从 MuJoCo 数据里读出某条腿的虚拟腿状态。"""
    import mujoco

    m, d = mujoco_data.model, mujoco_data

    leg_name = "leg_lower" if lower else "leg_upper"
    wheel_name = "wheel_lower" if lower else "wheel_upper"
    dof_wheel = DOF_WHEEL_LOWER if lower else DOF_WHEEL_UPPER

    jl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, leg_name)
    jw = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, wheel_name)

    hip = d.xpos[jl]
    whl = d.xpos[jw]
    dv = whl - hip
    length = math.hypot(dv[0], dv[2])
    lean = math.atan2(dv[0], -dv[2])

    # 虚拟腿角速度 = 腿刚体绕侧向轴的角速度。
    #
    # 推导:轮子是腿的**子刚体**,所以「髋→轮」这个向量是随腿一起转的刚体向量。
    # 它的方向角(即虚拟腿角 lean)对时间的导数,就等于腿绕侧向轴的角速度。
    # 这是一个恒等式,不依赖腿长、也不依赖当前角度 —— 已被实测确认
    # (在 hip=-0.6..+0.6 全程,lean_rate / hip_rate 恒为 1.0000)。
    #
    # ⚠️ 不要用「两个刚体原点速度相减再投影」的做法。leg_lower 与 wheel_lower
    #    的原点不重合(相距 40 mm),mj_objectVelocity 报的是**各自原点处**的
    #    速度,相减得到的是两个不同点的相对速度,不是腿的转动。
    #    实测该做法会得到一个与角度无关的常数比例 0.548,静默地把速度反馈
    #    缩小到 55%,足以让整个控制器的阻尼失效。
    vel_leg = np.zeros(6)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, jl, vel_leg, 0)
    omega = vel_leg[0:3]  # 世界系角速度
    lean_rate = float(omega @ LATERAL_AXIS)  # 投影到侧向轴(世界 -Y)

    return LegState(lean=lean, lean_rate=lean_rate, length=length,
                    wheel_rate=float(d.qvel[dof_wheel]))


def leg_states(mujoco_data) -> tuple[LegState, LegState]:
    """返回 (lower, upper) 两条腿的状态。"""
    return leg_state(mujoco_data, True), leg_state(mujoco_data, False)


def lqr_state(mujoco_data, x_ref: float = 0.0) -> np.ndarray:
    """把 MuJoCo 状态映射成 LQR 的 10 维状态向量。

    [x, x', y, y', tl, tl', tr, tr', f, f']

    tl/tr 取两腿**均值**(平面等效虚拟腿);左右腿的差值由调用方单独闭环。
    y 取腿的**竖直投影长度**(与虚拟腿长直接对应,便于惩罚塌腿)。
    """
    import mujoco

    d = mujoco_data
    low, up = leg_states(d)

    x = float(d.qpos[0]) - x_ref
    x_dot = float(d.qvel[0])

    # 高度量:两腿竖直投影的均值
    y = 0.5 * (low.length * math.cos(low.lean) + up.length * math.cos(up.lean))
    y_dot = 0.5 * (low.length * -math.sin(low.lean) * low.lean_rate
                   + up.length * -math.sin(up.lean) * up.lean_rate)

    tl = 0.5 * (low.lean + up.lean)
    tl_dot = 0.5 * (low.lean_rate + up.lean_rate)
    tr, tr_dot = tl, tl_dot  # 平面模型里左右等效

    quat = np.array(d.qpos[QP_BASE_QUAT])
    w, qx, qy, qz = quat
    yaw = math.atan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    yaw_rate = float(d.qvel[5])

    return np.array([x, x_dot, y, y_dot, tl, tl_dot, tr, tr_dot, yaw, yaw_rate])


def body_pitch(mujoco_data) -> float:
    """机身在矢状面内的俯仰角(rad),正 = 前倾。"""
    return _quat_to_pitch(np.array(mujoco_data.qpos[QP_BASE_QUAT]))


def leg_offsets(mujoco_data) -> tuple[float, float]:
    """两腿虚拟腿角相对**零位偏置**的偏差 (d_lower, d_upper),rad。

    零位偏置 = home 位姿下 hip_q=0 时的虚拟腿角 ≈ -0.0451 rad。
    减去它以后,「0」才真正对应平衡姿态,否则控制律会一直顶着一个常值误差。
    """
    low, up = leg_states(mujoco_data)
    return low.lean - LEAN_OFFSET, up.lean - LEAN_OFFSET


def leg_angles_phys(mujoco_data) -> tuple[float, float]:
    """两个物理髋关节角 (lower, upper),rad。"""
    d = mujoco_data
    return float(d.qpos[QP_HIP_LOWER]), float(d.qpos[QP_HIP_UPPER])


def leg_rates_phys(mujoco_data) -> tuple[float, float]:
    d = mujoco_data
    return float(d.qvel[DOF_HIP_LOWER]), float(d.qvel[DOF_HIP_UPPER])


def wheel_rates(mujoco_data) -> tuple[float, float]:
    d = mujoco_data
    return float(d.qvel[DOF_WHEEL_LOWER]), float(d.qvel[DOF_WHEEL_UPPER])


# home 位姿(所有关节为 0)下的虚拟腿角偏置,实测。
# 由 leg_state() 在 home 位姿下算出 = atan2(-1.805e-3, 39.959e-3) = -0.04515 rad。
LEAN_OFFSET = math.atan2(-0.001804560, 0.039959274)
