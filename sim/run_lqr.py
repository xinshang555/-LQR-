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
    u = -K @ x_lqr

其中 x_lqr 由 `state_map.lqr_state()` 从 MuJoCo 状态映射而来(见该文件的长注释,
说明为什么不能把 qpos 直接塞进 K)。

除了平面平衡律之外,还叠加一项**腿同步**控制:两腿沿 Y 并排,平面模型里
它们必须同相。若不加这一项,两腿会各自漂移,机身开始绕竖轴扭,平面假设失效。
同步项把「两腿虚拟腿角之差」拉回 0。

输出的 u 是 4 维,顺序与 car.xml 的 actuator 一致:
    [hip_lower, wheel_lower, hip_upper, wheel_upper]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(ROOT / "LQR计算代码"))

import mujoco  # noqa: E402

import calculate as lqr_calc  # noqa: E402
import parameter as P  # noqa: E402
import state_map as sm  # noqa: E402

XML = ROOT / "urdf" / "car.xml"

# actuator 顺序(见 car.xml)
ACT_HIP_LOWER, ACT_WHEEL_LOWER, ACT_HIP_UPPER, ACT_WHEEL_UPPER = 0, 1, 2, 3


def build_controller(leg_nominal: float = None, verbose: bool = True):
    """解出 K 并返回一个 (data) -> ctrl 的闭包。"""
    L = P.LEG_NOMINAL if leg_nominal is None else leg_nominal
    K = lqr_calc.calculate(L, L)
    if verbose:
        print(f"[LQR] 腿长 {L * 1000:.1f} mm, K 形状 {K.shape}, "
              f"max|K|={np.max(np.abs(K)):.3f}, 全部有限={np.all(np.isfinite(K))}")

    def controller(m, d) -> np.ndarray:
        x = sm.lqr_state(d)
        # 高度项与腿角项本来就有零位偏置:减去标称值,使平衡点对应 0 误差
        x[2] -= L * math.cos(sm.LEAN_OFFSET)   # 标称竖直投影长度
        x[4] -= sm.LEAN_OFFSET
        x[6] -= sm.LEAN_OFFSET

        u_planar = -K @ x

        # --- 腿同步项 ---
        # 两腿虚拟腿角之差与其变化率,单独用一个 PD 拉回 0。
        low, up = sm.leg_states(d)
        d_lean = low.lean - up.lean
        d_rate = low.lean_rate - up.lean_rate
        kp_sync, kd_sync = 2.0e-1, 2.0e-2
        u_sync = kp_sync * d_lean + kd_sync * d_rate

        ctrl = np.zeros(4)
        # K 的 4 个输入顺序是 [Tlw, Tll, Trw, Trl](左轮, 左腿, 右轮, 右腿)
        # 映射到 actuator:[hip_lower, wheel_lower, hip_upper, wheel_upper]
        #
        # "lower" 与 "upper" 两条腿在 X-Z 面内做**同向**运动(它们并排,
        # 机身带动它们一起前后倾),所以平面律对两条腿给**相同**的腿角指令;
        # 差别只在同步项上(反号)。
        ctrl[ACT_WHEEL_LOWER] = u_planar[0]          # Tlw
        ctrl[ACT_HIP_LOWER] = u_planar[1] + u_sync   # Tll
        ctrl[ACT_WHEEL_UPPER] = u_planar[2]          # Trw
        ctrl[ACT_HIP_UPPER] = u_planar[3] - u_sync   # Trl
        return ctrl

    return controller, K


def run(seconds: float, view: bool, open_loop: bool, kick: float,
        leg_nominal: float = None, plot: bool = False,
        track_x: float = 0.0) -> dict:
    m = mujoco.MjModel.from_xml_path(str(XML))
    d = mujoco.MjData(m)

    controller, K = build_controller(leg_nominal, verbose=not open_loop)

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
    # 控制频率:每 10 个物理步下一次控制 -> 100 Hz 控制、0.5 ms 物理
    control_every = 10

    viewer = None
    if view:
        try:
            # 注意:必须用 importlib 取 viewer 子模块,
            # 否则 `import mujoco.viewer` 会把 `mujoco` 变成 run() 的**局部名**,
            # 遮蔽模块级导入,导致上面的 mujoco.MjModel 报 UnboundLocalError。
            import importlib
            mujoco_viewer = importlib.import_module("mujoco.viewer")
            viewer = mujoco_viewer.launch_passive(m, d)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] 无法打开查看器({type(exc).__name__}: {exc}),改为无窗口运行")
            viewer = None

    log_t, log_pitch, log_x, log_y, log_lean, log_ctrl = [], [], [], [], [], []
    ctrl = np.zeros(4)
    fell_at = None
    pitch0 = sm.body_pitch(d)

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

        if step % 50 == 0:   # 每 25 ms 记一次
            pit = sm.body_pitch(d)
            st = sm.lqr_state(d)
            log_t.append(t)
            log_pitch.append(math.degrees(pit))
            log_x.append(float(d.qpos[0]) * 1000)
            log_y.append(float(st[2]) * 1000)
            log_lean.append(math.degrees(st[4]))
            log_ctrl.append(ctrl.copy())

            # 判定"倒下":机身倾角超过 60 度,或机身高度掉到 15 mm 以下
            if fell_at is None and (abs(pit) > math.radians(60.0)
                                    or float(d.qpos[2]) < 0.015):
                fell_at = t

    if viewer is not None:
        viewer.close()

    log_t = np.array(log_t)
    log_pitch = np.array(log_pitch)
    log_x = np.array(log_x)
    log_y = np.array(log_y)
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
        "t": log_t, "pitch": log_pitch, "x": log_x, "y": log_y,
        "lean": log_lean, "ctrl": log_ctrl,
    }

    print()
    print("=" * 66)
    print(f"时长 {seconds:.1f} s   {'开环对照' if open_loop else 'LQR 闭环'}")
    print(f"  初始倾角              : {math.degrees(pitch0):+.4f} deg")
    print(f"  最终倾角              : {final_pitch:+.4f} deg")
    print(f"  稳态段最大倾角        : {steady_pitch_max:.4f} deg")
    print(f"  前后漂移              : {steady_x_drift:.4f} mm")
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
    axes[0].plot(t, result["pitch"], label="机身俯仰角 (deg)")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_ylabel("pitch [deg]")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, result["x"], color="C1", label="前后位置 (mm)")
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_ylabel("x [mm]")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(t, result["y"], color="C2", label="腿竖直投影 (mm)")
    axes[2].set_ylabel("leg height [mm]")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    axes[3].plot(t, result["lean"], color="C3", label="虚拟腿角 (deg)")
    axes[3].axhline(math.degrees(sm.LEAN_OFFSET), color="r", ls="--", lw=0.8,
                    label="零位偏置")
    axes[3].set_ylabel("lean [deg]")
    axes[3].set_xlabel("t [s]")
    axes[3].legend()
    axes[3].grid(alpha=0.3)

    tag = "openloop" if open_loop else "lqr"
    fig.suptitle(f"car.xml 平衡仿真 ({'开环' if open_loop else 'LQR 闭环'})")
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
    args = ap.parse_args()

    run(args.seconds, args.view, args.open_loop, args.kick, plot=args.plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
