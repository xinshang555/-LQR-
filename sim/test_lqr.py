"""LQR 端到端回归：开环必须失败，闭环必须通过严格站立判据。"""

import math

from run_lqr import run


def main() -> int:
    opened = run(2.0, False, True, 0.0)
    assert opened["fell_at"] is not None, "开环竟未触发倒下判据"
    for kick in (0.0, 0.01, -0.01):
        result = run(5.0, False, False, kick)
        assert result["fell_at"] is None, f"闭环倒下，kick={kick}"
        assert result["steady_pitch_max_deg"] < 1.0
        assert result["x_drift_mm"] < 20.0
        assert result["max_height_error_mm"] < 5.0

    drive = run(5.0, False, False, 0.0, drive_speed=0.04)
    assert drive["fell_at"] is None, "最大允许前进速度下倒下"
    assert abs(drive["x"][-1] - drive["x_ref"][-1]) < 20.0

    turn = run(12.0, False, False, 0.0, track_yaw=math.radians(2.0))
    assert turn["fell_at"] is None, "航向控制时倒下"
    assert turn["final_yaw_deg"] > 1.0, "航向指令未产生正确方向的转动"
    print("ALL LQR CLOSED-LOOP TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
