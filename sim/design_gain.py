"""从 MuJoCo 局部数据辨识模型并设计离散 LQR 增益。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim"))
import state_map as sm  # noqa: E402

XML = ROOT / "urdf" / "car.xml"
DEFAULT_GAIN = ROOT / "sim" / "lqr_gain.npz"
CONTROL_EVERY = 10
WHEEL_ARMATURE = 1.0e-7

# [pitch, pitch_rate, x, x_rate, hip, hip_rate]，仅用于安全采样。
BASELINE = np.array([1.26553, 0.03373, 4.63628, 0.31153, 25.27657, 0.12549])


def load_model() -> mujoco.MjModel:
    m = mujoco.MjModel.from_xml_path(str(XML))
    for name in ("wheel_lower_joint", "wheel_upper_joint"):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        m.dof_armature[m.jnt_dofadr[jid]] = WHEEL_ARMATURE
    return m


def baseline_gain() -> np.ndarray:
    """返回 u=-Kx 中的 K；状态顺序见 state_map.lqr_state。"""
    kp, kd, kx, kv, kh, dh = BASELINE
    K = np.zeros((4, 10))
    K[0, [4, 5, 8, 9]] = [kh, dh, -kh, -dh]
    K[2, [6, 7, 8, 9]] = [kh, dh, -kh, -dh]
    K[1, [0, 1, 8, 9]] = [kx, kv, -kp, -kd]
    K[3] = K[1]
    return K


def baseline_control(d: mujoco.MjData) -> np.ndarray:
    kp, kd, kx, kv, kh, dh = BASELINE
    pitch, pitch_rate = sm.body_pitch(d), -float(d.qvel[4])
    wheel = kp * pitch + kd * pitch_rate - kx * d.qpos[0] - kv * d.qvel[0]
    return np.array([
        -kh * d.qpos[7] - dh * d.qvel[6], wheel,
        -kh * d.qpos[9] - dh * d.qvel[8], wheel,
    ])


def reset(m: mujoco.MjModel, kick: float = 0.0) -> mujoco.MjData:
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    tangent = np.zeros(m.nv)
    tangent[4] = -kick
    mujoco.mj_integratePos(m, d.qpos, tangent, 1.0)
    mujoco.mj_forward(m, d)
    return d


def step_control(m: mujoco.MjModel, d: mujoco.MjData, ctrl: np.ndarray) -> None:
    d.ctrl[:] = np.clip(ctrl, -1.0, 1.0)
    for _ in range(CONTROL_EVERY):
        mujoco.mj_step(m, d)


def collect_samples(m: mujoco.MjModel, episodes: int, seed: int):
    rng = np.random.default_rng(seed)
    xs, us, ys = [], [], []
    for _ in range(episodes):
        d = reset(m, rng.uniform(-0.008, 0.008))
        noise = np.zeros(4)
        for _ in range(300):
            x = sm.lqr_state(d)
            noise = 0.8 * noise + rng.normal(0.0, [0.003, 0.002, 0.003, 0.002])
            step_control(m, d, baseline_control(d) + noise)
            y = sm.lqr_state(d)
            if (not np.isfinite(y).all() or abs(sm.body_pitch(d)) >= 0.2
                    or d.qpos[2] <= 0.035):
                break
            xs.append(x)
            us.append(noise.copy())
            ys.append(y)
    return np.asarray(xs), np.asarray(us), np.asarray(ys)


def identify_open_loop(X: np.ndarray, U: np.ndarray, Y: np.ndarray):
    """先辨识闭环，再用 A_open=A_closed+B*K_baseline 恢复开环。"""
    x_mean, y_mean = X.mean(axis=0), Y.mean(axis=0)
    Z = np.hstack([X - x_mean, U])
    scale = Z.std(axis=0)
    scale[scale < 1e-8] = 1.0
    Zn = Z / scale
    coef = np.linalg.solve(Zn.T @ Zn + 1e-6 * np.eye(14), Zn.T @ (Y - y_mean))
    coef = coef / scale[:, None]
    F_closed, B = coef[:10].T, coef[10:].T
    A = F_closed + B @ baseline_gain()
    prediction = y_mean + Z @ coef
    rmse = np.sqrt(np.mean((Y - prediction) ** 2, axis=0))
    ctrb = np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(10)])
    return A, B, x_mean, rmse, np.linalg.matrix_rank(ctrb, tol=1e-7)


def reference_control(x: np.ndarray) -> np.ndarray:
    offset = np.zeros(10)
    offset[[4, 6]] = sm.LEAN_OFFSET
    return -baseline_gain() @ (x - offset)


def design(episodes: int = 80, seed: int = 1234, r_weight: float = 0.1):
    m = load_model()
    X, U, Y = collect_samples(m, episodes, seed)
    if len(X) < 1000:
        raise RuntimeError(f"有效辨识样本过少：{len(X)}")
    A, B, reference, rmse, rank = identify_open_loop(X, U, Y)
    if rank != 10:
        raise RuntimeError(f"辨识模型不可控：rank={rank}/10")
    reference[0] = 0.0
    scales = np.array([0.02, 0.2, 0.1, 1.0, 0.05, 1.0,
                       0.05, 1.0, 0.05, 1.0])
    Q, R = np.diag(1.0 / scales**2), np.eye(4) * r_weight
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    rho = float(np.max(np.abs(np.linalg.eigvals(A - B @ K))))
    if rho >= 1.0:
        raise RuntimeError(f"离散闭环不稳定：谱半径={rho}")
    return {
        "K": K, "A": A, "B": B, "reference": reference,
        "u_eq": reference_control(reference), "rmse": rmse,
        "controllability_rank": np.array(rank), "spectral_radius": np.array(rho),
        "control_every": np.array(CONTROL_EVERY),
        "wheel_armature": np.array(WHEEL_ARMATURE), "seed": np.array(seed),
        "samples": np.array(len(X)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="辨识 MuJoCo 局部模型并设计离散 LQR")
    ap.add_argument("--episodes", type=int, default=80)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--r-weight", type=float, default=0.1)
    ap.add_argument("--output", type=Path, default=DEFAULT_GAIN)
    args = ap.parse_args()
    result = design(args.episodes, args.seed, args.r_weight)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **result)
    print(f"samples={int(result['samples'])}, controllability=10/10")
    print("RMSE:", np.array2string(result["rmse"], precision=3))
    print(f"closed-loop spectral radius={float(result['spectral_radius']):.6f}")
    print(f"saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
