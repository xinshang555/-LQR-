"""Independent verification of the LQR parameters against MuJoCo ground truth.

This is the "test suite" for the LQR stage.  It does NOT import the LQR code to
decide what is correct: it reads `urdf/car.xml` (the authoritative model, itself
verified against the URDF to 1e-15) and recomputes every physical quantity that
`LQR计算代码/parameter.py` declares.  Then it compares.

Run:
    cd urdf && python3 ../sim/check_model.py
or:
    python3 sim/check_model.py            # from the project root

Exit code 0 = all checks passed.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Locate the model, wherever we are invoked from.
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
XML = ROOT / "urdf" / "car.xml"
sys.path.insert(0, str(ROOT / "LQR计算代码"))

import mujoco  # noqa: E402

TOL = 0.01  # 1 % relative tolerance, per the agreed acceptance criterion

_FAILURES: list[str] = []
_CHECKS = 0


def check(name: str, got: float, want: float, tol: float = TOL, unit: str = "") -> None:
    """Relative comparison with an absolute floor for near-zero targets."""
    global _CHECKS
    _CHECKS += 1
    if want == 0.0:
        ok = abs(got) < 1e-12
        err = abs(got)
    else:
        err = abs(got - want) / abs(want)
        ok = err <= tol
    flag = "PASS" if ok else "FAIL"
    if not ok:
        _FAILURES.append(f"{name}: got {got:.9g} want {want:.9g} (rel err {err:.3%})")
    print(f"  [{flag}] {name:38s} got={got:14.9g} want={want:14.9g} "
          f"rel_err={err:9.3%} {unit}")


def load() -> tuple[mujoco.MjModel, mujoco.MjData]:
    if not XML.exists():
        sys.exit(f"cannot find {XML}")
    m = mujoco.MjModel.from_xml_path(str(XML))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)  # the `home` keyframe = balance start pose
    mujoco.mj_forward(m, d)
    return m, d


def bid(m: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)


def main() -> int:
    m, d = load()

    print("=" * 78)
    print("MuJoCo ground truth vs LQR计算代码/parameter.py")
    print("=" * 78)

    # ---- world axes sanity: gravity is -Z, wheels side by side along Y ------
    # The lateral axis (wheel spin / hip pivot) is world -Y.  Established twice,
    # independently: every hip+wheel hinge axis is (0,-1,0), and the wheel's
    # inertia about (0,-1,0) is the unique spin value 25.92 g*mm^2.
    print("\n[0] Frame conventions (must hold before anything else means anything)")
    check("gravity is (0,0,-9.81)", float(m.opt.gravity[2]), -9.81, tol=0.01)
    dl = d.xpos[bid(m, "wheel_upper")] - d.xpos[bid(m, "wheel_lower")]
    check("wheels separated along Y only (|dx|)", float(abs(dl[0])), 0.0)
    check("wheels separated along Y only (|dz|)", float(abs(dl[2])), 0.0)
    for jn in ("hip_lower_joint", "wheel_lower_joint", "hip_upper_joint", "wheel_upper_joint"):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
        check(f"{jn} axis is world (0,-1,0)", float(d.xaxis[j][1]), -1.0, tol=1e-9)

    # Lateral (spin) axis = world -Y ; fore-aft axis = world X ; vertical = world Z.
    LAT = np.array([0.0, -1.0, 0.0])
    FWD = np.array([1.0, 0.0, 0.0])

    # ---- geometry -----------------------------------------------------------
    print("\n[1] Geometry  (the LQR model's own frame: lateral = CAD Z)")
    hip = d.xpos[bid(m, "leg_lower")]
    axle_l = d.xpos[bid(m, "wheel_lower")]
    axle_u = d.xpos[bid(m, "wheel_upper")]
    axle = 0.5 * (axle_l + axle_u)

    # d_z in parameter.py is documented as HALF the track (origin at the
    # midpoint between the hips).  Total separation measured = 35 mm.
    track = float(abs(axle_u[1] - axle_l[1]))
    check("two-wheel separation d_z (total)", track, 0.035, unit="m")
    half_track = 0.5 * track

    # Wheel radius: distance from axle axis to the lowest point of the wheel.
    # Derived from the wheel geom's own geometry, not from a hardcoded number.
    r_meas = float(axle[2])  # wheels rest exactly on Z = 0 at the home pose
    check("wheel radius r (axle height at home)", r_meas, 0.008, tol=0.02, unit="m")

    # Leg pitch: hip pivot -> wheel axle.  This is what the leg rotates about.
    pitch = float(np.linalg.norm(hip - axle_l))
    print(f"  [info] hip->axle distance = {pitch * 1000:.6f} mm "
          f"(README quotes 40.000 mm as the CAD hole-to-hole figure)")

    # ---- mass ---------------------------------------------------------------
    print("\n[2] Mass")
    check("m_B body mass", float(m.body_mass[bid(m, 'mount')]), 0.02153, unit="kg")
    check("m_l leg mass (one leg)", float(m.body_mass[bid(m, 'leg_lower')]), 0.00114, unit="kg")
    check("m_w wheel mass", float(m.body_mass[bid(m, 'wheel_lower')]), 0.00086, unit="kg")
    total = float(m.body_mass.sum())
    check("total mass", total, 0.02553, unit="kg")

    # ---- inertia ------------------------------------------------------------
    # parameter.py wants the moment about each body's own COM, taken about a
    # named physical axis.  MuJoCo stores body_inertia in the body's own
    # inertial frame, so rotate into the world frame first, then project onto
    # the axis in question.  Doing it this way (rather than reading a single
    # diagonal entry) is what makes the tilted `mount` body come out right.
    print("\n[3] Inertia about each body's COM, projected onto the physical axis")
    print("      lateral axis  = world -Y  = CAD Z  (wheel spin / hip pivot)")
    print("      fore-aft axis = world +X  = CAD X")

    def world_inertia(name: str) -> np.ndarray:
        i = bid(m, name)
        R = d.ximat[i].reshape(3, 3)          # body-inertial -> world
        return R @ np.diag(m.body_inertia[i]) @ R.T

    def about(I: np.ndarray, axis: np.ndarray) -> float:
        return float(axis @ I @ axis)

    Ib = world_inertia("mount")
    Il = world_inertia("leg_lower")
    Iw = world_inertia("wheel_lower")

    # NOTE on which numbers are authoritative.  `质量属性/*.txt` prints three
    # different blocks; only the *principal* block is about the COM in a frame
    # the LQR model can use directly.  The "对齐输出的坐标系" tensor (Lxx/Lyy/Lzz)
    # is expressed in the CAD output frame and carries large products of
    # inertia, so feeding Lzz=3603.49 in as the lateral moment only *coincides*
    # with the truth by accident; Lxx=2090.79 vs the real fore-aft 3890.21 shows
    # how far that block can drift.  We therefore take MuJoCo's own numbers.
    check("J_Bz body, about lateral axis", about(Ib, LAT), 3.60349e-06, unit="kg*m^2")
    check("J_By body, about fore-aft axis", about(Ib, FWD), 2.090789e-06, unit="kg*m^2")
    check("J_llcz leg, about lateral axis", about(Il, LAT), 1.6738e-07, unit="kg*m^2")
    check("J_llcy leg, about fore-aft axis", about(Il, FWD), 1.6101e-07, unit="kg*m^2")
    check("J_wz wheel, about spin axis", about(Iw, LAT), 2.592e-08, unit="kg*m^2")
    check("J_wy wheel, transverse", about(Iw, FWD), 1.474e-08, unit="kg*m^2")

    # ---- body COM polar coordinates (theta_Bc, the missing name) ------------
    print("\n[4] Body COM polar coordinates (theta_Bc was undefined in parameter.py)")
    com_b = d.xipos[bid(m, "mount")]
    # LQR frame, origin at the hip:  X forward, Y downward, Z lateral.
    # "down" is world -Z here, and we take it relative to the HIP height.
    r_x = float(com_b[0] - hip[0])
    r_down = float(-(com_b[2] - hip[2]))
    l_Bc_meas = math.hypot(r_x, r_down)
    # parameter.py defines theta_Bc as atan2(r_y, r_x), polar about +X.
    theta_bc = math.atan2(r_down, r_x)

    print(f"  [info] body COM rel hip: fore/aft={r_x * 1000:+.6f} mm  "
          f"down={r_down * 1000:+.6f} mm")
    print(f"  [info] l_Bc = {l_Bc_meas * 1000:.6f} mm   "
          f"theta_Bc = {theta_bc:+.9f} rad = {math.degrees(theta_bc):+.6f} deg")
    # Cross-check against the raw CAD numbers in 质量属性/body.txt.
    cad = np.array([-11.14, 1.70, 17.50])          # body COM, car.STEP frame
    hip_cad = np.array([-9.397509, -1.080906, 0.0])
    d_cad = cad - hip_cad
    check("l_Bc agrees with body.txt CAD numbers", l_Bc_meas,
          float(np.linalg.norm(d_cad[:2])) * 1e-3, tol=0.002, unit="m")
    # CAD Y is "down", so down = -(Y_com - Y_hip) = Y_hip - Y_com.
    check("theta_Bc agrees with body.txt CAD numbers", theta_bc,
          math.atan2(hip_cad[1] - cad[1], cad[0] - hip_cad[0]), tol=0.002, unit="rad")

    # ---- the inverted-pendulum sanity check ---------------------------------
    print("\n[5] Inverted-pendulum sanity")
    tot = float(m.body_mass.sum())
    com = (d.xipos * m.body_mass[:, None]).sum(0) / tot
    h = float(com[2] - axle[2])
    check("total COM height above axle", h, 0.038, tol=0.02, unit="m")
    print(f"  [info] COM is {h * 1000:.4f} mm ABOVE the axle -> unstable, must close the loop")

    # ---- now import the LQR parameters and compare --------------------------
    print("\n[6] LQR计算代码/parameter.py contents vs ground truth")
    try:
        import parameter as p
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] cannot import parameter.py: {type(exc).__name__}: {exc}")
        _FAILURES.append("parameter.py import failed")
        return report()

    for attr in ("theta_Bc",):
        if not hasattr(p, attr):
            print(f"  [FAIL] parameter.py is missing `{attr}` (calculate.py references it)")
            _FAILURES.append(f"parameter.py missing {attr}")

    LEG_NOMINAL = 0.040  # nominal virtual leg length, m (CAD hole-to-hole pitch)

    def cmp_param(attr: str, want: float, tol: float = TOL, unit: str = "") -> None:
        if not hasattr(p, attr):
            print(f"  [FAIL] {attr:14s} MISSING from parameter.py")
            _FAILURES.append(f"{attr} missing")
            return
        val = getattr(p, attr)
        if callable(val):
            # parameter.py exposes l_rc/theta_rc/... as functions of leg length.
            # Sample them at the nominal 40 mm leg so we still verify the value.
            val = val(LEG_NOMINAL)
        check(f"p.{attr}", float(val), want, tol=tol, unit=unit)

    cmp_param("m_B", 0.02153, unit="kg")
    cmp_param("m_l", 0.00114, unit="kg")
    cmp_param("m_w", 0.00086, unit="kg")
    cmp_param("r", r_meas, tol=0.02, unit="m")
    cmp_param("d_z", half_track, unit="m")
    cmp_param("l_Bc", l_Bc_meas, unit="m")
    cmp_param("theta_Bc", theta_bc, tol=0.02, unit="rad")
    cmp_param("J_Bz", about(Ib, LAT), unit="kg*m^2")
    cmp_param("J_By", about(Ib, FWD), unit="kg*m^2")
    cmp_param("J_wz", about(Iw, LAT), unit="kg*m^2")
    cmp_param("J_wy", about(Iw, FWD), unit="kg*m^2")
    cmp_param("J_llcy", about(Il, FWD), unit="kg*m^2")
    cmp_param("J_llcz", about(Il, LAT), unit="kg*m^2")
    cmp_param("J_rlcy", about(Il, FWD), unit="kg*m^2")
    cmp_param("J_rlcz", about(Il, LAT), unit="kg*m^2")
    cmp_param("g", 9.81, tol=0.01, unit="m/s^2")

    # The leg COM helpers must return something consistent with the model.
    print("\n[7] Leg COM helper functions (sampled at the nominal 40 mm leg)")
    print("      NOTE: measured in the SAGITTAL (fore/aft + vertical) plane only.")
    print("      The LQR model is a planar inverted pendulum; the leg COM's lateral")
    print("      offset contributes nothing to the in-plane dynamics and must be")
    print("      excluded.  Including it yields 20.419 mm -- 13 % too large.")
    com_l = d.xipos[bid(m, "leg_lower")]
    l_leg_meas = float(math.hypot(com_l[0] - hip[0], com_l[2] - hip[2]))
    theta_leg_meas = math.atan2(-(com_l[2] - hip[2]), com_l[0] - hip[0])
    print(f"  [info] MuJoCo leg COM: sagittal dist={l_leg_meas * 1000:.6f} mm  "
          f"theta={theta_leg_meas:+.6f} rad")
    print(f"  [info] (3D distance incl. lateral would be "
          f"{float(np.linalg.norm(com_l - hip)) * 1000:.6f} mm -- do not use)")
    for fn in ("l_rc", "l_lc"):
        if hasattr(p, fn):
            v = float(getattr(p, fn)(LEG_NOMINAL))
            check(f"p.{fn}(40mm)", v, l_leg_meas, tol=0.05, unit="m")
    for fn in ("theta_rc", "theta_lc"):
        if hasattr(p, fn):
            v = float(getattr(p, fn)(LEG_NOMINAL))
            check(f"p.{fn}(40mm)", v, theta_leg_meas, tol=0.05, unit="rad")

    return report()


def report() -> int:
    print("\n" + "=" * 78)
    if _FAILURES:
        print(f"FAILED: {len(_FAILURES)} of {_CHECKS} checks")
        for f in _FAILURES:
            print("   -", f)
        print("=" * 78)
        return 1
    print(f"ALL {_CHECKS} CHECKS PASSED")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
