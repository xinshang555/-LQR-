"""state_map 的回归测试。

覆盖
----
1. 选择矩阵满秩(rank = 10/10) —— 证明 10 个状态互相独立
2. 零位偏置 LEAN_OFFSET 精确
3. 腿倾角速度恒等式(lean_rate == 腿刚体角速度在侧向轴上的投影)
4. 两条腿独立(不被打包平均)
5. 各状态对输入的响应方向正确(偏航对差动力矩、俯仰对基座姿态)
6. 随机扰动下不产生 NaN / inf

历史:早期版本的状态映射把 y(偏航)与 f(身体俯仰)搞反,并把两腿平均成一个量,
导致选择矩阵 rank 只有 7,并据此错误地判定"模型退化"。本测试的 rank 检查
就是防止那类错误复现。
"""
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim"))

import mujoco  # noqa: E402
import state_map as sm  # noqa: E402

LABELS = ["x", "x'", "y(yaw)", "y'", "tl", "tl'", "tr", "tr'", "f(pitch)", "f'"]

m = mujoco.MjModel.from_xml_path(str(ROOT / "urdf" / "car.xml"))
d = mujoco.MjData(m)


def home():
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)


# ---------------------------------------------------------------- 1. 满秩
print("=" * 72)
print("1. 选择矩阵必须满秩(rank = 10/10)")
print("=" * 72)
nv, n = m.nv, 2 * m.nv
C = np.zeros((10, n))
eps = 1e-7


def pert(e, idx):
    """在切空间方向 idx 上加 e 的扰动。

    注意 qpos 布局:qpos[0:3]=平移, qpos[3:7]=四元数, qpos[7:11]=4 个铰链。
    早期版本写成 qpos[6+(idx-6)],会去扰动四元数分量而非髋关节 —— 这是个
    真实踩过的坑,故此处保留注释。
    """
    mujoco.mj_resetDataKeyframe(m, d, 0)
    if idx < nv:
        if idx < 3:
            d.qpos[idx] += e
        elif idx < 6:
            ax = np.zeros(3)
            ax[idx - 3] = e
            th = np.linalg.norm(ax)
            k = ax / th
            dq = np.array([math.cos(th / 2), *(math.sin(th / 2) * k)])
            w0, v0 = d.qpos[3], d.qpos[4:7]
            w1, v1 = dq[0], dq[1:4]
            d.qpos[3] = w1 * w0 - v1 @ v0
            d.qpos[4:7] = w1 * v0 + w0 * v1 + np.cross(v1, v0)
        else:
            d.qpos[7 + (idx - 6)] += e
    else:
        d.qvel[idx - nv] = e
    mujoco.mj_forward(m, d)


for j in range(n):
    pert(+eps, j)
    xp = sm.lqr_state(d)
    pert(-eps, j)
    xm = sm.lqr_state(d)
    C[:, j] = (xp - xm) / (2 * eps)

sv = np.linalg.svd(C, compute_uv=False)
rank = int(np.linalg.matrix_rank(C))
print(f"  rank(C) = {rank} / 10")
print(f"  奇异值  = {np.round(sv, 6)}")
assert rank == 10, f"选择矩阵秩不足({rank}/10),状态之间有线性相关"

# ------------------------------------------------------- 2. 零位偏置
print()
print("=" * 72)
print("2. 零位偏置 LEAN_OFFSET")
print("=" * 72)
home()
s = sm.lqr_state(d)
print(f"  LEAN_OFFSET = {sm.LEAN_OFFSET:+.9f} rad "
      f"({math.degrees(sm.LEAN_OFFSET):+.4f} deg)")
print(f"  home 处 tl - off = {s[4] - sm.LEAN_OFFSET:+.3e}")
print(f"  home 处 tr - off = {s[6] - sm.LEAN_OFFSET:+.3e}")
assert abs(s[4] - sm.LEAN_OFFSET) < 1e-12
assert abs(s[6] - sm.LEAN_OFFSET) < 1e-12
print("  home 位姿状态:", np.round(s, 9))

# ------------------------------------------- 3. 腿倾角速度恒等式
print()
print("=" * 72)
print("3. 腿倾角速度恒等式(身体不动时 tl' 应精确等于髋角速度)")
print("=" * 72)
worst = 0.0
for h in (-0.6, -0.3, 0.0, 0.3, 0.6):
    for w in (-2.0, 1.0, 3.0):
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.qpos[7] = h
        d.qpos[9] = h
        d.qvel[6] = w
        d.qvel[8] = w
        mujoco.mj_forward(m, d)
        st = sm.lqr_state(d)
        worst = max(worst, abs(st[5] - w), abs(st[7] - w))
print(f"  最大误差 = {worst:.3e}")
assert worst < 1e-9, f"倾角速度不精确,误差 {worst}"
print("  -> 精确(机器精度)")

# ------------------------------------------------- 4. 两腿独立
print()
print("=" * 72)
print("4. 两条腿必须独立(不能平均成一个量)")
print("=" * 72)
mujoco.mj_resetDataKeyframe(m, d, 0)
d.qpos[7] = 0.3
mujoco.mj_forward(m, d)
st = sm.lqr_state(d)
print(f"  只动 hip_lower=0.3 -> tl={st[4]:+.6f}  tr={st[6]:+.6f}")
assert abs(st[4] - st[6]) > 0.1, "两腿被打包成一个量了"
print("  -> tl 与 tr 独立")

# --------------------------------------- 5. 状态方向正确性
print()
print("=" * 72)
print("5. 状态方向正确性")
print("=" * 72)
# 5a. 绕世界 +Y 旋转基座 -> 身体俯仰应负(侧向轴是 -Y)
home()
ang = 0.2
d.qpos[3:7] = [math.cos(ang / 2), 0.0, math.sin(ang / 2), 0.0]
mujoco.mj_forward(m, d)
st = sm.lqr_state(d)
print(f"  绕世界 +Y 转 {ang} rad -> f(pitch) = {st[8]:+.6f} (应为负)")
assert st[8] < 0, "俯仰符号不符"
# 5b. 绕世界 Z 旋转基座 -> 偏航应等于该角度
home()
yz = 0.3
d.qpos[3:7] = [math.cos(yz / 2), 0.0, 0.0, math.sin(yz / 2)]
mujoco.mj_forward(m, d)
st = sm.lqr_state(d)
print(f"  绕世界 Z 转 {yz} rad -> y(yaw) = {st[2]:+.6f} (应等于该值)")
assert abs(st[2] - yz) < 1e-9, "偏航不符"
# 5c. 偏航角速度
home()
d.qvel[5] = 1.5
mujoco.mj_forward(m, d)
st = sm.lqr_state(d)
print(f"  基座 wz=1.5 -> y' = {st[3]:+.6f}")
assert abs(st[3] - 1.5) < 1e-9
# 5d. 俯仰角速度
home()
d.qvel[4] = 1.5
mujoco.mj_forward(m, d)
st = sm.lqr_state(d)
print(f"  基座 wy=1.5 -> f' = {st[9]:+.6f} (侧向轴 -Y,应为 -1.5)")
assert abs(st[9] + 1.5) < 1e-9
print("  -> 方向全部正确")

# ------------------------------------------------- 6. 随机扰动健壮性
print()
print("=" * 72)
print("6. 随机扰动下不得出现 NaN / inf")
print("=" * 72)
rng = np.random.default_rng(0)
for _ in range(100):
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qpos[:7] += rng.normal(0, 0.01, 7)
    d.qpos[7:11] += rng.normal(0, 0.1, 4)
    d.qvel[:] = rng.normal(0, 0.5, 10)
    mujoco.mj_forward(m, d)
    st = sm.lqr_state(d)
    assert st.shape == (10,), st.shape
    assert np.all(np.isfinite(st)), st
print("  100 组随机状态 -> 全部有限,形状 (10,)")

print()
print("=" * 72)
print("ALL STATE-MAP TESTS PASSED")
print("=" * 72)
