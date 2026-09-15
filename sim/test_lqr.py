"""LQR 端到端回归：开环必须失败，闭环必须通过严格站立判据。"""

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
    print("ALL LQR CLOSED-LOOP TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
