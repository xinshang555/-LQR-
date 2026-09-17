#pragma once

#include <array>

#include "robot_types.hpp"

namespace wheel_leg {

class WheelLegController {
 public:
  WheelLegController();

  MotorCommand update(const RobotState& state, double dt);
  void setCommand(const UserCommand& command);
  const UserCommand& command() const { return command_; }
  void changeForwardSpeed(double delta_mps);
  void changeHeading(double delta_rad);
  void stopAndHold(const RobotState& state);
  void reset();

 private:
  static double wrapAngle(double angle);

  UserCommand command_{};
  std::array<double, 10> reference_{};
  std::array<double, 4> previous_torque_nm_{};
};

}  // namespace wheel_leg
