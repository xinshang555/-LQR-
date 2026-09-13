"""Test the MuJoCo<->LQR state mapping against the real model."""
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "urdf"))
sys.path.insert(0, str(ROOT / "sim"))

import mujoco  # noqa: E402
import state_map as sm  # noqa: E402

np.set_printoptions(precision=6, suppress=True, linewidth=200)

m = mujoco.MjModel.from_xml_path(str(ROOT / "urdf" / "car.xml"))
d = mujoco.MjData(m)

print("LEAN_OFFSET constant =", sm.LEAN_OFFSET)

mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)

print("\n--- at home keyframe ---")
print("hip qpos      :", d.qpos[7], d.qpos[9])
low, up = sm.leg_states(d)
print(f"lower: lean={low.lean:+.6f} len={low.length * 1000:.4f} mm")
print(f"upper: lean={up.lean:+.6f} len={up.length * 1000:.4f} mm")
print("=> LEAN_OFFSET should equal lower.lean; actual diff =",
      abs(sm.LEAN_OFFSET - low.lean))
assert abs(sm.LEAN_OFFSET - low.lean) < 1e-9, "LEAN_OFFSET mismatch!"
print("lean offsets after removing bias:", sm.leg_offsets(d))
print("body pitch    :", sm.body_pitch(d))
s = sm.lqr_state(d)
print("lqr_state     :", s)

print("\n--- sweep hip_lower, check tl tracks it ---")
for h in (-0.4, -0.2, 0.0, 0.2, 0.4):
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qpos[7] = h
    d.qpos[9] = h          # both legs together -> planar motion
    mujoco.mj_forward(m, d)
    s = sm.lqr_state(d)
    lo, up_ = sm.leg_states(d)
    print(f"  hip={h:+.2f}  tl={s[4]:+.6f}  y={s[2] * 1000:9.4f} mm  "
          f"lo.lean={lo.lean:+.6f} up.lean={up_.lean:+.6f}")

print("\n--- tilt body, check pitch & lean respond ---")
for ang in (-0.2, 0.0, 0.2):
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qpos[3:7] = [math.cos(ang / 2), 0.0, math.sin(ang / 2), 0.0]
    mujoco.mj_forward(m, d)
    s = sm.lqr_state(d)
    print(f"  bodyrotY={ang:+.2f}  pitch={sm.body_pitch(d):+.6f}  "
          f"tl={s[4]:+.6f}  y={s[2] * 1000:9.4f} mm  x={s[0] * 1000:+.4f} mm")

print("\n--- lean_rate must EQUAL the hip rate exactly (rigid-body identity) ---")
worst = 0.0
for h in (-0.6, -0.3, 0.0, 0.3, 0.6):
    for w in (-2.0, 1.0, 3.0):
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.qpos[7] = h
        d.qpos[9] = h
        d.qvel[6] = w
        d.qvel[8] = w
        mujoco.mj_forward(m, d)
        low, up = sm.leg_states(d)
        err = max(abs(low.lean_rate - w), abs(up.lean_rate - w))
        worst = max(worst, err)
        print(f"  hip={h:+.2f} rate={w:+.1f}  lean_rate=({low.lean_rate:+.8f}, "
              f"{up.lean_rate:+.8f})  err={err:.2e}")
assert worst < 1e-9, f"lean_rate identity violated, worst err {worst}"
print(f"  -> exact to {worst:.2e}")

print("\n--- body twist about lateral axis must show up in lean_rate ---")
mujoco.mj_resetDataKeyframe(m, d, 0)
d.qvel[4] = 1.5  # base angular velocity about world Y
mujoco.mj_forward(m, d)
low, _ = sm.leg_states(d)
print(f"  base wy=1.5 -> lean_rate={low.lean_rate:+.8f} (body carries legs rigidly)")
assert abs(low.lean_rate + 1.5) < 1e-8, "body twist not reflected (sign check)"
print("  sign: world +Y rate -> lean_rate negative (lateral axis is -Y) OK")

print("\n--- wheel rate passthrough ---")
mujoco.mj_resetDataKeyframe(m, d, 0)
d.qvel[7] = 4.0
d.qvel[9] = -3.0
mujoco.mj_forward(m, d)
low, up = sm.leg_states(d)
assert abs(low.wheel_rate - 4.0) < 1e-12 and abs(up.wheel_rate + 3.0) < 1e-12
print(f"  wheels = ({low.wheel_rate}, {up.wheel_rate}) OK")

print("\n--- lqr_state shape / finiteness under random perturbation ---")
rng = np.random.default_rng(0)
for trial in range(50):
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qpos[:7] += rng.normal(0, 0.01, 7)
    d.qpos[7:11] += rng.normal(0, 0.1, 4)
    d.qvel[:] = rng.normal(0, 0.5, 10)
    mujoco.mj_forward(m, d)
    s = sm.lqr_state(d)
    assert s.shape == (10,), s.shape
    assert np.all(np.isfinite(s)), s
print("  50 random states -> all finite, shape (10,) OK")

print("\nALL STATE-MAP TESTS PASSED")
