"""在 MuJoCo 里跑 LQR 闭环平衡和运动控制。

用法
----
    python3 sim/run_lqr.py                 # 无窗口跑 30 s 并打印指标(默认)
    python3 sim/run_lqr.py --view          # 带 MuJoCo 查看器(实时)
    python3 sim/run_lqr.py --open-loop     # 开环对照:ctrl 恒为 0,应当 1 s 内倒下
    python3 sim/run_lqr.py --plot          # 跑完画曲线存成 PNG
    python3 sim/run_lqr.py --seconds 60    # 改时长
    python3 sim/run_lqr.py --kick 0.05     # 额外注入初始倾角扰动 (rad)
    python3 sim/run_lqr.py --view          # W/S 前后,A/D 转向,空格停止,R 复位
    python3 sim/run_lqr.py --vx 0.01       # 无窗口:以 10 mm/s 前进
    python3 sim/run_lqr.py --target-yaw 5  # 无窗口:转到 5 deg 航向

控制律
------
    ctrl = u_eq - K @ (x_lqr - x_cmd)

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
from dataclasses import dataclass
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


def _wrap_angle(angle: float) -> float:
    """把角度包到 [-pi, pi)，避免跨越 +/-pi 时控制误差跳变。"""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class MotionCommand:
    """由键盘或命令行给出的平面运动指令。"""

    reference: np.ndarray
    forward_speed: float = 0.0
    speed_step: float = 0.01
    heading_step: float = math.radians(2.0)
    max_speed: float = 0.04
    heading_target: float | None = None
    reset_requested: bool = False
    hold_heading_requested: bool = False

    def __post_init__(self) -> None:
        self.forward_speed = float(np.clip(
            self.forward_speed, -self.max_speed, self.max_speed))
        if self.heading_target is None:
            self.heading_target = float(self.reference[2])

    def advance(self, dt: float) -> None:
        """积分速度指令，形成 LQR 的位置和偏航参考轨迹。"""
        self.reference[0] += self.forward_speed * dt
        self.reference[1] = self.forward_speed

    def stop(self, state: np.ndarray) -> None:
        """停止运动，并以当前位置/朝向作为新的保持点。"""
        self.forward_speed = 0.0
        self.reference[0] = state[0]
        self.reference[1] = 0.0
        self.reference[2] = state[2]
        self.reference[3] = 0.0
        # 操作者航向与 state_map 的世界 Z 偏航符号相反。
        self.heading_target = -state[2]

    def on_key(self, keycode: int) -> None:
        """MuJoCo passive viewer 的按键回调。按键可连续调节目标速度。"""
        key = chr(keycode).upper() if 0 <= keycode < 128 else ""
        if key == "W":
            self.forward_speed = min(self.max_speed,
                                     self.forward_speed + self.speed_step)
        elif key == "S":
            self.forward_speed = max(-self.max_speed,
                                     self.forward_speed - self.speed_step)
        elif key == "A":
            self.heading_target = _wrap_angle(
                self.heading_target + self.heading_step)
        elif key == "D":
            self.heading_target = _wrap_angle(
                self.heading_target - self.heading_step)
        elif keycode == 32 or key == "X":  # Space / X
            self.forward_speed = 0.0
            self.hold_heading_requested = True
        elif key == "R":
            self.reset_requested = True
        else:
            return
        print(f"[command] vx={self.forward_speed * 1000:+.0f} mm/s, "
              f"heading_target={math.degrees(self.heading_target):+.1f} deg"
              + (", reset" if self.reset_requested else ""))


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

    def controller(m, d, command_reference: np.ndarray | None = None) -> np.ndarray:
        target = reference if command_reference is None else command_reference
        error = sm.lqr_state(d) - target
        error[2] = _wrap_angle(error[2])
        return u_eq - K @ error

    return controller, K, control_every, reference


def run(seconds: float, view: bool, open_loop: bool, kick: float,
        gain_path: Path = GAIN, plot: bool = False,
        track_x: float = 0.0, drive_speed: float = 0.0,
        track_yaw: float = 0.0) -> dict:
    m = mujoco.MjModel.from_xml_path(str(XML))
    d = mujoco.MjData(m)

    if open_loop:
        controller, K, control_every = None, np.zeros((4, 10)), 10
        reference = np.zeros(10)
    else:
        controller, K, control_every, reference = build_controller(
            m, gain_path, verbose=True)

    command = MotionCommand(reference.copy(), forward_speed=drive_speed,
                            heading_target=track_yaw)
    command.reference[0] = track_x

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
            viewer = mujoco_viewer.launch_passive(m, d,
                                                   key_callback=command.on_key)
            wall_start = time.perf_counter()
            print("[keyboard] W/S 加减前后速度,A/D 调整目标航向,"
                  "Space/X 停止,R 复位")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] 无法打开查看器({type(exc).__name__}: {exc}),改为无窗口运行")
            viewer = None

    log_t, log_pitch, log_x, log_y, log_z, log_lean, log_ctrl = [], [], [], [], [], [], []
    log_x_ref, log_yaw_ref = [], []
    ctrl = np.zeros(4)
    fell_at = None
    pitch0 = sm.body_pitch(d)
    home_z = float(d.qpos[2])

    for step in range(n_steps):
        t = step * dt

        if command.reset_requested:
            mujoco.mj_resetDataKeyframe(m, d, 0)
            mujoco.mj_forward(m, d)
            command.reference[:] = reference
            command.forward_speed = 0.0
            command.heading_target = float(reference[2])
            command.reset_requested = False
            ctrl[:] = 0.0
            print("[command] 已复位到 home")

        if command.hold_heading_requested:
            command.stop(sm.lqr_state(d))
            command.hold_heading_requested = False

        if step % control_every == 0:
            if open_loop:
                ctrl = np.zeros(4)
            else:
                command.advance(control_every * dt)
                state = sm.lqr_state(d)
                # 辨识得到的偏航位置增益很大，直接给航向阶跃会令髋电机
                # 饱和。把每次送入 LQR 的航向误差限制在 0.2 deg，目标点
                # 随实际航向滚动前移，既保留四执行器协调转向又不冲击平衡。
                balance_reference = command.reference.copy()
                # 面向操作者定义左转为正；它与 state_map 的世界 Z 偏航
                # 正方向相反，因此控制航向为 -state[2]。
                control_heading = -state[2]
                heading_error = _wrap_angle(
                    command.heading_target - control_heading)
                limited_error = float(np.clip(
                    heading_error, -math.radians(0.2), math.radians(0.2)))
                balance_reference[2] = _wrap_angle(state[2] - limited_error)
                balance_reference[3] = 0.0
                ctrl = controller(m, d, balance_reference)
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
            log_x_ref.append(command.reference[0] * 1000)
            log_yaw_ref.append(math.degrees(command.heading_target))

            # 可信的站立判据：姿态、高度和轨迹误差都必须有界。
            if fell_at is None and (not np.isfinite(d.qpos).all()
                                    or abs(pit) > math.radians(5.0)
                                    or abs(float(d.qpos[2]) - home_z) > 0.005
                                    or abs(float(d.qpos[0])
                                           - command.reference[0]) > 0.050):
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
    log_x_ref = np.array(log_x_ref)
    log_yaw_ref = np.array(log_yaw_ref)

    # 稳态段:去掉前 20% 的暂态
    n = len(log_t)
    tail = slice(int(n * 0.2), n)
    steady_pitch_max = float(np.max(np.abs(log_pitch[tail]))) if n else float("nan")
    steady_x_drift = float(abs(log_x[-1] - log_x[0])) if n else float("nan")
    final_pitch = float(log_pitch[-1]) if n else float("nan")
    final_yaw = float(-log_y[-1] / 1000.0) if n else float("nan")

    result = {
        "fell_at": fell_at,
        "final_pitch_deg": final_pitch,
        "final_yaw_deg": math.degrees(final_yaw),
        "steady_pitch_max_deg": steady_pitch_max,
        "x_drift_mm": steady_x_drift,
        "max_height_error_mm": float(np.max(np.abs(log_z - home_z * 1000))),
        "t": log_t, "pitch": log_pitch, "x": log_x, "y": log_y,
        "z": log_z, "lean": log_lean, "ctrl": log_ctrl,
        "x_ref": log_x_ref, "yaw_ref_deg": log_yaw_ref,
    }

    print()
    print("=" * 66)
    print(f"时长 {seconds:.1f} s   {'开环对照' if open_loop else 'LQR 闭环'}")
    print(f"  初始倾角              : {math.degrees(pitch0):+.4f} deg")
    print(f"  最终倾角              : {final_pitch:+.4f} deg")
    print(f"  最终/目标航向         : {math.degrees(final_yaw):+.3f} / "
          f"{math.degrees(command.heading_target):+.3f} deg")
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
    axes[1].plot(t, result["x_ref"], color="C1", ls="--", label="command (mm)")
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_ylabel("x [mm]")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(t, -result["y"], color="C2", label="control heading (mrad)")
    axes[2].plot(t, np.radians(result["yaw_ref_deg"]) * 1000,
                 color="C2", ls="--", label="command (mrad)")
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
    ap.add_argument("--target-x", type=float, default=0.0,
                    help="初始前进位置目标 (m)")
    ap.add_argument("--vx", type=float, default=0.0,
                    help="前进速度指令 (m/s)，建议绝对值不超过 0.04")
    ap.add_argument("--target-yaw", type=float, default=0.0,
                    help="初始目标航向 (deg)")
    args = ap.parse_args()

    run(args.seconds, args.view, args.open_loop, args.kick,
        gain_path=args.gain, plot=args.plot, track_x=args.target_x,
        drive_speed=args.vx, track_yaw=math.radians(args.target_yaw))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
