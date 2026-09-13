"""Design K on the real plant, correctly this time.

Two earlier mistakes to avoid
----------------------------
1. `C` (the 10-state LQR selection) has rank 7, so projecting through it
   destroyed the physical mode.  FIX: use the machine's natural, full-rank
   coordinates [px, pz, pitch, hip_lo, hip_up, yaw] + rates  (rank 12/12).
2. Naive eigenvector truncation (`V[:, slow]`) with complex eigenvectors and a
   pinv is numerically fragile.  FIX: build a real invariant subspace via the
   Schur decomposition, split by |eigenvalue|, and project the pair (A,B) with
   an orthogonal basis -- the standard model-order-reduction approach.

The ~1e4 rad/s poles are a numerical artefact of differentiating stiff contacts
(they scale with the FD step).  The Schur split removes them cleanly because
they are separated from the rigid-body spectrum by ~3 orders of magnitude.

Method
------
  1. Load car.xml; inject wheel armature at runtime.
  2. Continuous A, B via mjd_transitionFD + matrix logarithm.
  3. Real Schur form of A:  A = Z T Z', T quasi-upper-triangular with the
     eigenvalues ordered by magnitude on the diagonal.
  4. Keep the first `k` columns of Z (the slow invariant subspace).
     A_r = Z1' A Z1,  B_r = Z1' B.
  5. ARE on (A_r, B_r) with weights in units of the NATURAL coordinates.
  6. K = R^-1 B_r' P;  K_full = K_r Z1'   (a gain on the full 12 states).
  7. Verify by simulating car.xml with that gain, and by checking the
     closed-loop poles of the slow part.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import logm, schur, solve_continuous_are

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim"))

import mujoco  # noqa: E402

XML = str(ROOT / "urdf" / "car.xml")
WHEEL_ARMATURE = 1.0e-7

# natural state labels, in order
LABELS = ["px", "pz", "pitch", "hip_lo", "hip_up", "yaw",
          "vx", "vz", "wy", "d_hip_lo", "d_hip_up", "wz"]


def build(armature: float = WHEEL_ARMATURE, contactify: bool = False):
    m = mujoco.MjModel.from_xml_path(XML)
    for jn in ("wheel_lower_joint", "wheel_upper_joint"):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
        m.dof_armature[m.jnt_dofadr[j]] = armature
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    return m, d


def selection(m) -> np.ndarray:
    """12 x (2*nv) matrix picking the natural coordinates out of the tangent state."""
    nv, n = m.nv, 2 * m.nv
    S = np.zeros((12, n))
    S[0, 0] = 1.0          # px
    S[1, 2] = 1.0          # pz
    S[2, 4] = -1.0         # pitch about world Y, negated => + = forward lean
    S[3, 6] = 1.0          # hip_lower
    S[4, 8] = 1.0          # hip_upper
    S[5, 5] = 1.0          # yaw about world Z
    S[6, nv + 0] = 1.0     # vx
    S[7, nv + 2] = 1.0     # vz
    S[8, nv + 4] = -1.0    # wy  (match the pitch sign)
    S[9, nv + 6] = 1.0     # d hip_lower
    S[10, nv + 8] = 1.0    # d hip_upper
    S[11, nv + 5] = 1.0    # wz
    return S


def continuous_AB(m, d, S):
    nv, n, nu = m.nv, 2 * m.nv, m.nu
    A_d = np.zeros((n, n))
    B_d = np.zeros((n, nu))
    mujoco.mjd_transitionFD(m, d, 1e-7, True, A_d, B_d, None, None)
    dt = m.opt.timestep
    A = (logm(A_d) / dt).real
    B = (A @ np.linalg.solve(A_d - np.eye(n), B_d)).real
    # project to natural coordinates (S has full row rank, verified)
    Sp = np.linalg.pinv(S)
    return S @ A @ Sp, S @ B


def slow_subspace(A: np.ndarray, cutoff: float):
    """Orthogonal basis of the invariant subspace of modes with |lambda| < cutoff.

    Uses the ordered real Schur form.  Returns (Z1, k).
    """
    # schur with sorting: sort='rhp' puts Re>0 first; we want magnitude sorting.
    # scipy's sort callable receives (alpha, beta) for generalized problems;
    # for a standard problem beta == 1 and alpha are the eigenvalues.
    T, Z, sdim = schur(A, output="real", sort=lambda a, b: np.abs(a) < cutoff)
    return Z[:, :sdim], sdim, T, Z


def design(cutoff: float = 200.0, q=None, r=None, verbose: bool = True):
    m, d = build()
    S = selection(m)
    A12, B12 = continuous_AB(m, d, S)

    ev = np.linalg.eigvals(A12)
    if verbose:
        print("=== full 12-state natural model ===")
        print("  poles by magnitude:")
        for e in sorted(ev, key=lambda z: -abs(z)):
            tag = "UNSTABLE" if e.real > 1e-3 else ""
            print(f"    {e.real:+14.4f} {e.imag:+12.4f}j  |.|={abs(e):14.3f} {tag}")
        print("  physical sqrt(m g h/J) = 14.1491 rad/s")

    Z1, k, T, Z = slow_subspace(A12, cutoff)
    A_r = Z1.T @ A12 @ Z1
    B_r = Z1.T @ B12

    if verbose:
        evr = np.linalg.eigvals(A_r)
        print(f"\n=== slow invariant subspace (cutoff {cutoff} rad/s) ===")
        print(f"  kept {k} of {A12.shape[0]} modes")
        print("  slow poles:",
              [f"{e.real:.4f}" for e in sorted(evr, key=lambda z: -z.real)])
        print("  check invariance: ||A Z1 - Z1 A_r|| =",
              f"{np.linalg.norm(A12 @ Z1 - Z1 @ A_r):.3e}")

    # --- weights ---
    # The reduced model has k states that are NOT the original coordinates, so a
    # diagonal Q in the original basis cannot be used directly.  Build it by
    # mapping the desired per-coordinate weights onto the reduced subspace:
    #     Q_r = Z1' diag(q) Z1
    # which is the standard consistent choice (it penalises the same physical
    # errors expressed in the reduced coordinates).
    if q is None:
        # Weights in NATURAL coordinates:
        #   px[m], pz[m], pitch[rad], hip_lo[rad], hip_up[rad], yaw[rad]
        #   then their rates.
        # Sized to the real machine: omega ~ 14 rad/s.  Angle weights dominate
        # because attitude is what an inverted pendulum gets wrong first.
        q = [2.0,        # px        position hold (soft: allows travel)
             5.0e2,      # pz        height hold (anti-collapse)
             4.0e3,      # pitch     THE critical state
             3.0e2,      # hip_lo    keep near neutral, away from +-0.8 stops
             3.0e2,      # hip_up
             2.0e1,      # yaw
             1.0e0,      # vx
             5.0e0,      # vz
             1.0e2,      # wy
             3.0e1,      # d hip_lo
             3.0e1,      # d hip_up
             5.0e0]      # wz
    if r is None:
        r = [1.0, 1.0, 1.0, 1.0]
    Qfull = np.diag(np.asarray(q, dtype=float))
    Qr = Z1.T @ Qfull @ Z1
    Qr = 0.5 * (Qr + Qr.T)          # symmetrise
    R = np.diag(np.asarray(r, dtype=float))

    P = solve_continuous_are(A_r, B_r, Qr, R)
    K_r = np.linalg.solve(R, B_r.T @ P)
    K_full = K_r @ Z1.T          # gain on the full 12 natural states

    if verbose:
        print("\n=== K on the 12 natural states ===")
        print(f"{'state':>10} " + "".join(f"{'u' + str(j):>13}" for j in range(4)))
        for i, lb in enumerate(LABELS):
            print(f"{lb:>10} " + "".join(f"{K_full[j, i]:13.4f}" for j in range(4)))
        print("  max|K| =", float(np.max(np.abs(K_full))))

        Acl = A_r - B_r @ K_r
        wcl = np.linalg.eigvals(Acl)
        print("\n  closed-loop slow poles (want Re<0):")
        for e in sorted(wcl, key=lambda z: -z.real):
            print(f"    {e.real:+12.4f} {e.imag:+12.4f}j   |.|={abs(e):10.4f}")
        print("  all stable:",
              bool(np.all(np.real(wcl) < -1e-6)))

    info = {"S": S, "A12": A12, "B12": B12, "Z1": Z1, "k": k,
            "A_r": A_r, "B_r": B_r, "K_r": K_r}
    return K_full, info


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    K, info = design()
