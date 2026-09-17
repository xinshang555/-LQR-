#pragma once

#include <array>

namespace wheel_leg {

struct RobotState {
  double x{};
  double x_dot{};
  double yaw{};
  double yaw_dot{};
  double leg_lower{};
  double leg_lower_dot{};
  double leg_upper{};
  double leg_upper_dot{};
  double body_pitch{};
  double body_pitch_dot{};
  double body_height{};

  std::array<double, 10> lqrVector() const {
    return {x, x_dot, yaw, yaw_dot, leg_lower, leg_lower_dot,
            leg_upper, leg_upper_dot, body_pitch, body_pitch_dot};
  }
};

struct UserCommand {
  double forward_speed{};   // m/s
  double heading{};         // rad; operator convention: left is positive
};

struct MotorCommand {
  // [hip_lower, wheel_lower, hip_upper, wheel_upper], physical N*m.
  std::array<double, 4> torque_nm{};
};

}  // namespace wheel_leg
