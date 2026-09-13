"""MuJoCo <-> LQR 状态映射层。

------------------------------------------------------------------------------
⚠️ 重要更正(本文件早期版本把状态含义搞反了)
------------------------------------------------------------------------------
早期版本把 state[2] 当成"腿的竖直投影(高度)"、state[8] 当成偏航 —— 正好相反。
那导致得出"模型退化、选择矩阵 rank=7/10"的错误结论。正确含义由两条独立证据确认:

  证据 1(量纲)—— M1 的 5 个对角元:
        x   A = 2.63e-02 kg        -> 平移质量
        y   E = 3.98e-06 kg*m^2    -> 转动惯量
        tl  J = 2.57e-06 kg*m^2    -> 转动惯量
        tr  M = 2.57e-06 kg*m^2    -> 转动惯量
        f   Q = 3.84e-06 kg*m^2    -> 转动惯量
     5 个坐标里 4 个是转动、只有 1 个平移,所以 y 不可能是"高度"。

  证据 2(方程结构)——
        eq① (x)   无重力项            -> 水平平移,对称方向
        eq② (y)   无重力项            -> 对称方向;且被【差动力矩】(Tlw-Trw)
                                       驱动,系数 ±dz/r -> 绕竖直轴转动 = 偏航
        eq③④(腿) 重力矩 R*tl, S*tr    -> 腿虚拟倾角
        eq⑤ (f)   重力矩 -m_B*g*l_Bc*sin(theta_Bc)*f,惯量含 J_Bz
                                       -> 身体俯仰角

用正确映射重做选择矩阵 => rank = 10/10(满秩),10 个状态互相独立。
另:M1 的惯量矩阵与 MuJoCo 质量矩阵整体相对差 4.06%,差异可归因于
    参考原点不同(髋中心 vs mount 原点)与滚动约束项 J_wz/r^2。

------------------------------------------------------------------------------
状态向量(与 calculate.py 的约定一致)
------------------------------------------------------------------------------
    x_lqr = [x, x', y, y', tl, tl', tr, tr', f, f']

    x  : 前后平移 (m)
    y  : **偏航角** (rad)          绕世界竖直轴
    tl : **左腿虚拟倾角** (rad)     髋->轮 连线相对竖直,矢状面内
    tr : **右腿虚拟倾角** (rad)
    f  : **身体俯仰角** (rad)       机身矢状面内倾角

------------------------------------------------------------------------------
坐标/符号约定(全部实测确定)
------------------------------------------------------------------------------
世界系: +X 前, +Y 侧向, +Z 上, 重力 (0,0,-9.81)
侧向轴(髋轴/轮自转轴)= 世界 **-Y**
  依据:4 个 hinge 的世界轴向实测**全部**是 (0,-1,0);
        且轮子绕 (0,-1,0) 的惯量恰为 25.92 g·mm²(自转轴特征值)。

符号:
    身体俯仰 f   = -(绕世界 +Y 的转角)        -> 正 = 前倾
    偏航   y     = 世界 Z 角
    腿倾角 tl/tr = atan2(轮-髋 的 dx, -(dz))  -> 正 = 轮在髋前方

零位偏置:home 位姿下 hip=0 时 tl = tr = -0.04513 rad(腿并非严格竖直),
必须减去,否则稳态存在常值倾角误差。

左右腿分配:tl -> leg_lower, tr -> leg_upper。
依据:模型里 yaw 与 tl 的耦合为负、与 tr 为正(反号);
     MuJoCo 实测同样为 tl 负、tr 正,符号模式一致。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# qvel 索引(car.xml 实测)
DOF_BASE_POS = slice(0, 3)   # vx, vy, vz
DOF_BASE_ROT = slice(3, 6)   # wx, wy, wz (世界系角速度)
DOF_HIP_LOWER = 6
DOF_WHEEL_LOWER = 7
DOF_HIP_UPPER = 8
DOF_WHEEL_UPPER = 9

# qpos 索引(注意 qpos[3:7] 是四元数,铰链从 7 开始)
QP_BASE_POS = slice(0, 3)
QP_BASE_QUAT = slice(3, 7)
QP_HIP_LOWER = 7
QP_WHEEL_LOWER = 8
QP_HIP_UPPER = 9
QP_WHEEL_UPPER = 10

# 侧向轴 = 世界 -Y
LATERAL_AXIS = np.array([0.0, -1.0, 0.0])

# home 位姿(关节全 0)下的腿虚拟倾角,实测
LEAN_OFFSET = math.atan2(-0.001804560, 0.039959274)   # = -0.045129317 rad


@dataclass
class LegState:
    """单条腿的虚拟腿状态。"""
    lean: float        # 虚拟倾角 (rad),正 = 轮在髋前方
    lean_rate: float   # 倾角速度 (rad/s)
    length: float      # 髋->轮 距离 (m)
    wheel_rate: float  # 轮自转角速度 (rad/s)


def _body_omega(m, d, name: str) -> np.ndarray:
    """刚体的世界系角速度。"""
    import mujoco
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    res = np.zeros(6)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, i, res, 0)
    return res[0:3]


def _leg_lean(m, d, lower: bool) -> float:
    """腿虚拟倾角:髋->轮 连线相对竖直的夹角,矢状面内。"""
    import mujoco
    jl = mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_BODY, "leg_lower" if lower else "leg_upper")
    jw = mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_BODY, "wheel_lower" if lower else "wheel_upper")
    v = d.xpos[jw] - d.xpos[jl]
    return math.atan2(v[0], -v[2])


def leg_state(mujoco_data, lower: bool = True) -> LegState:
    """单条腿的虚拟腿状态。

    lean_rate 用腿刚体的世界角速度在侧向轴上的投影 —— 这是恒等式:
    轮是腿的子刚体,髋->轮 向量随腿一起转,所以其方向角变化率等于
    腿绕侧向轴的角速度。实测在 hip=-0.6..+0.6 全程误差 < 1.4e-15。
    """
    import mujoco
    m, d = mujoco_data.model, mujoco_data

    leg_name = "leg_lower" if lower else "leg_upper"
    wheel_name = "wheel_lower" if lower else "wheel_upper"

    jl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, leg_name)
    jw = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, wheel_name)

    dv = d.xpos[jw] - d.xpos[jl]
    length = math.hypot(dv[0], dv[2])
    lean = math.atan2(dv[0], -dv[2])

    om = _body_omega(m, d, leg_name)
    lean_rate = float(om @ LATERAL_AXIS)

    dof_wheel = DOF_WHEEL_LOWER if lower else DOF_WHEEL_UPPER
    return LegState(lean=lean, lean_rate=lean_rate, length=length,
                    wheel_rate=float(d.qvel[dof_wheel]))


def leg_states(mujoco_data) -> tuple[LegState, LegState]:
    """(左腿, 右腿) = (leg_lower, leg_upper) 的状态。"""
    return leg_state(mujoco_data, True), leg_state(mujoco_data, False)


def body_pitch(mujoco_data) -> float:
    """身体俯仰角 (rad),正 = 前倾。侧向轴是世界 -Y,故绕 +Y 的转角取负。"""
    w, x, y, z = mujoco_data.qpos[QP_BASE_QUAT]
    s = max(-1.0, min(1.0, 2.0 * (w * y + z * x)))
    return -math.asin(s)


def yaw(mujoco_data) -> float:
    """偏航角 (rad),绕世界竖直轴。"""
    w, x, y, z = mujoco_data.qpos[QP_BASE_QUAT]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def lqr_state(mujoco_data) -> np.ndarray:
    """映射成 LQR 的 10 维状态。

    [x, x', y(偏航), y', tl, tl', tr, tr', f(俯仰), f']

    注意:tl/tr 是**两条腿各自**的倾角,不做平均 —— 模型里它们是独立坐标,
    平均会丢掉差模信息(早期版本正是因此把秩压低了)。
    """
    m, d = mujoco_data.model, mujoco_data
    tl_state, tr_state = leg_states(mujoco_data)
    om_base = _body_omega(m, d, "mount")

    return np.array([
        d.qpos[0],                      # x   前后平移
        d.qvel[0],                      # x'
        yaw(d),                         # y   偏航角
        float(om_base[2]),              # y'  偏航角速度
        tl_state.lean,                  # tl  左腿倾角
        tl_state.lean_rate,             # tl'
        tr_state.lean,                  # tr  右腿倾角
        tr_state.lean_rate,             # tr'
        body_pitch(d),                  # f   身体俯仰
        -float(om_base[1]),             # f'  俯仰角速度
    ])


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


# 标称虚拟腿长(供仿真复用)= CAD 孔距
LEG_NOMINAL = 0.040
