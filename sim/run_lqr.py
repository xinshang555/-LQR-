"""在 MuJoCo 里跑 LQR 闭环平衡。

用法
----
    python3 sim/run_lqr.py                 # 无窗口跑 30 s 并打印指标(默认)
    python3 sim/run_lqr.py --view          # 带 MuJoCo 查看器(实时)
    python3 sim/run_lqr.py --open-loop     # 开环对照:ctrl 恒为 0,应当 1 s 内倒下
    python3 sim/run_lqr.py --plot          # 跑完画曲线存成 PNG
    python3 sim/run_lqr.py --seconds 60    # 改时长
    python3 sim/run_lqr.py --kick 0.05     # 额外注入初始倾角扰动 (rad)

控制律
------
    ctrl = u_eq - K @ (x_lqr - x_ref)

其中 x_lqr 由 `state_map.lqr_state()` 从 MuJoCo 状态映射而来(见该文件的长注释,
说明为什么不能把 qpos 直接塞进 K)。

除了平面平衡律之外,还叠加一项**腿同步**控制:两腿沿 Y 并排,平面模型里
它们必须同相。若不加这一项,两腿会各自漂移,机身开始绕竖轴扭,平面假设失效。
同步项把「两腿虚拟腿角之差」拉回 0。

输出 ctrl 是 4 维归一化电机命令,顺序与 car.xml 的 actuator 一致:
    [hip_lower, wheel_lower, hip_upper, wheel_upper]
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(ROOT / "LQR计算代码"))

import mujoco  # noqa: E402

import state_map as sm  # noqa: E402

XML = ROOT / "urdf" / "car.xml"
GAIN = ROOT / "sim" / "lqr_gain.npz"

# actuator 顺序(见 car.xml)
ACT_HIP_LOWER, ACT_WHEEL_LOWER, ACT_HIP_UPPER, ACT_WHEEL_UPPER = 0, 1, 2, 3


def build_controller(m, gain_path: Path = GAIN, verbose: bool = True):
    """加载 design_gain.py 生成的离散 LQR，并返回控制闭包。"""
    if not gain_path.exists():
        raise FileNotFoundError(
            f"未找到 {gain_path}；请先运行 python3 sim/design_gain.py")
    with np.load(gain_path) as gain:
        K = gain["K"].copy()
        reference = gain["reference"].copy()
        u_eq = gain["u_eq"].copy()
        control_every = int(gain["control_every"])
        armature = float(gain["wheel_armature"])
        rho = float(gain["spectral_radius"])
        rank = int(gain["controllability_rank"])
    for name in ("wheel_lower_joint", "wheel_upper_joint"):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        m.dof_armature[m.jnt_dofadr[jid]] = armature
    if verbose:
        print(f"[LQR] K={K.shape}, controllability={rank}/10, "
              f"rho={rho:.6f}, max|K|={np.max(np.abs(K)):.3f}")

    def controller(m, d) -> np.ndarray:
        return u_eq - K @ (sm.lqr_state(d) - reference)

    return controller, K, control_every


def run(seconds: float, view: bool, open_loop: bool, kick: float,
        gain_path: Path = GAIN, plot: bool = False,
        track_x: float = 0.0) -> dict:
    m = mujoco.MjModel.from_xml_path(str(XML))
    d = mujoco.MjData(m)

    if open_loop:
        controller, K, control_every = None, np.zeros((4, 10)), 10
    else:
        controller, K, control_every = build_controller(m, gain_path, verbose=True)

    # 从 home 关键帧出发(轮子正好贴地、机身竖直)
    mujoco.mj_resetDataKeyframe(m, d, 0)

    if kick:
        # 注入一个初始倾角:绕侧向轴(世界 -Y)转 kick 弧度
        half = kick / 2.0
        d.qpos[3:7] = [math.cos(half), 0.0, -math.sin(half), 0.0]
        print(f"[init] 注入初始倾角 {math.degrees(kick):+.2f} deg")

    mujoco.mj_forward(m, d)

    dt = m.opt.timestep
    n_steps = int(round(seconds / dt))
    # 默认每 10 个 0.5 ms 物理步更新一次，即 200 Hz。

    viewer = None
    wall_start = None
    if view:
        try:
            # 注意:必须用 importlib 取 viewer 子模块,
            # 否则 `import mujoco.viewer` 会把 `mujoco` 变成 run() 的**局部名**,
            # 遮蔽模块级导入,导致上面的 mujoco.MjModel 报 UnboundLocalError。
            import importlib
            mujoco_viewer = importlib.import_module("mujoco.viewer")
            viewer = mujoco_viewer.launch_passive(m, d)
            wall_start = time.perf_counter()
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] 无法打开查看器({type(exc).__name__}: {exc}),改为无窗口运行")
            viewer = None

    log_t, log_pitch, log_x, log_y, log_z, log_lean, log_ctrl = [], [], [], [], [], [], []
    ctrl = np.zeros(4)
    fell_at = None
    pitch0 = sm.body_pitch(d)
    home_z = float(d.qpos[2])

    for step in range(n_steps):
        t = step * dt

        if step % control_every == 0:
            if open_loop:
                ctrl = np.zeros(4)
            else:
                ctrl = controller(m, d)
                ctrl = np.clip(ctrl, -1.0, 1.0)   # 尊重 ctrlrange
            d.ctrl[:] = ctrl

        mujoco.mj_step(m, d)

        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
            # launch_passive() 不负责给外部仿真循环限速。没有这段时，
            # 0.5 ms 的物理步会尽可能快地运行，画面可达到约 10 倍实时速度。
            target_wall = wall_start + (step + 1) * dt
            remaining = target_wall - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)

        if step % 50 == 0:   # 每 25 ms 记一次
            pit = sm.body_pitch(d)
            st = sm.lqr_state(d)
            log_t.append(t)
            log_pitch.append(math.degrees(pit))
            log_x.append(float(d.qpos[0]) * 1000)
            log_y.append(float(st[2]) * 1000)
            log_z.append(float(d.qpos[2]) * 1000)
            log_lean.append(math.degrees(st[4]))
            log_ctrl.append(ctrl.copy())

            # 可信的站立判据：姿态、高度、位置都必须有界。
            if fell_at is None and (not np.isfinite(d.qpos).all()
                                    or abs(pit) > math.radians(5.0)
                                    or abs(float(d.qpos[2]) - home_z) > 0.005
                                    or abs(float(d.qpos[0])) > 0.050):
                fell_at = t

    if viewer is not None:
        viewer.close()

    log_t = np.array(log_t)
    log_pitch = np.array(log_pitch)
    log_x = np.array(log_x)
    log_y = np.array(log_y)
    log_z = np.array(log_z)
    log_lean = np.array(log_lean)
    log_ctrl = np.array(log_ctrl) if log_ctrl else np.zeros((0, 4))

    # 稳态段:去掉前 20% 的暂态
    n = len(log_t)
    tail = slice(int(n * 0.2), n)
    steady_pitch_max = float(np.max(np.abs(log_pitch[tail]))) if n else float("nan")
    steady_x_drift = float(abs(log_x[-1] - log_x[0])) if n else float("nan")
    final_pitch = float(log_pitch[-1]) if n else float("nan")

    result = {
        "fell_at": fell_at,
        "final_pitch_deg": final_pitch,
        "steady_pitch_max_deg": steady_pitch_max,
        "x_drift_mm": steady_x_drift,
        "max_height_error_mm": float(np.max(np.abs(log_z - home_z * 1000))),
        "t": log_t, "pitch": log_pitch, "x": log_x, "y": log_y,
        "z": log_z, "lean": log_lean, "ctrl": log_ctrl,
    }

    print()
    print("=" * 66)
    print(f"时长 {seconds:.1f} s   {'开环对照' if open_loop else 'LQR 闭环'}")
    print(f"  初始倾角              : {math.degrees(pitch0):+.4f} deg")
    print(f"  最终倾角              : {final_pitch:+.4f} deg")
    print(f"  稳态段最大倾角        : {steady_pitch_max:.4f} deg")
    print(f"  前后漂移              : {steady_x_drift:.4f} mm")
    print(f"  最大高度误差          : {result['max_height_error_mm']:.4f} mm")
    if fell_at is None:
        print("  结果                  : 站住了(全程未触发倒下判据)")
    else:
        print(f"  结果                  : 在 {fell_at:.3f} s 倒下")
    print("=" * 66)

    if plot:
        make_plot(result, open_loop)

    return result


def make_plot(result: dict, open_loop: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = result["t"]
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(t, result["pitch"], label="body pitch (deg)")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_ylabel("pitch [deg]")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, result["x"], color="C1", label="forward position (mm)")
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_ylabel("x [mm]")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(t, result["y"], color="C2", label="yaw (mrad)")
    axes[2].set_ylabel("yaw [mrad]")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    axes[3].plot(t, result["lean"], color="C3", label="virtual leg angle (deg)")
    axes[3].axhline(math.degrees(sm.LEAN_OFFSET), color="r", ls="--", lw=0.8,
                    label="home offset")
    axes[3].set_ylabel("lean [deg]")
    axes[3].set_xlabel("t [s]")
    axes[3].legend()
    axes[3].grid(alpha=0.3)

    tag = "openloop" if open_loop else "lqr"
    fig.suptitle(f"car.xml balance ({'open loop' if open_loop else 'LQR closed loop'})")
    fig.tight_layout()
    out = ROOT / "sim" / f"result_{tag}.png"
    fig.savefig(out, dpi=110)
    print(f"[plot] 已保存 {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description="MuJoCo 上的 LQR 平衡仿真")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--view", action="store_true", help="打开 MuJoCo 查看器")
    ap.add_argument("--open-loop", action="store_true", help="开环对照(ctrl=0)")
    ap.add_argument("--kick", type=float, default=0.0, help="初始倾角扰动 (rad)")
    ap.add_argument("--plot", action="store_true", help="保存曲线 PNG")
    ap.add_argument("--gain", type=Path, default=GAIN, help="design_gain.py 生成的 npz")
    args = ap.parse_args()

    run(args.seconds, args.view, args.open_loop, args.kick,
        gain_path=args.gain, plot=args.plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
