#include "wheel_leg_controller.hpp"

#include <algorithm>
#include <cmath>

namespace wheel_leg {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kMotorGearNm = 0.004473;
constexpr double kMaxSpeed = 0.04;
constexpr double kHeadingErrorLimit = 0.2 * kPi / 180.0;

constexpr std::array<std::array<double, 10>, 4> kGain{{
    {{-6.57524291422409, -0.7414274080687234, 12.881908785099638,
      -0.04381595047725588, 0.45906245575312205, -0.019953907354693757,
      1.2431619875513626, -0.0010321764583203948, -1.8019378910377863,
      -0.07534710140861102}},
    {{-0.24160392245179682, 0.15006603381428435, 2.221900720431969,
      -0.010592587751525091, -1.0787698151469272, -0.053783800818914464,
      -0.049458601800721265, -0.0007876470734340268,
      -0.23705542482495431, -0.018869543828020276}},
    {{1.6260871936474923, -0.7855883724127963, -10.923893152594466,
      0.012897320215189474, 0.6017216593417894, -0.001677863947241375,
      -0.18825398360209197, -0.020743099743282426, -2.822117086839094,
      -0.07788170714028136}},
    {{1.112340671023295, 0.13873972506898166, -2.1391874597197167,
      0.016802308864981953, -0.15135998145311688, -0.0009342649934638206,
      -1.2067303544630243, -0.05383264656664655, -0.38966523711083456,
      -0.01926245007045564}},
}};

constexpr std::array<double, 10> kReference{
    0.0, 0.0012882624915544628, 0.00015299445903685304,
    8.96423812995808e-05, -0.041631341879127856, 0.003892432666807607,
    -0.041602520605331345, 0.003823980633248705, 0.005144469588426601,
    0.004994616439133764};

constexpr std::array<double, 4> kEquilibriumControl{
    0.041756044868186536, 0.006277616596739538,
    0.041036131969271146, 0.006277616596739538};

}  // namespace

WheelLegController::WheelLegController() { reset(); }

double WheelLegController::wrapAngle(double angle) {
  while (angle >= kPi) angle -= 2.0 * kPi;
  while (angle < -kPi) angle += 2.0 * kPi;
  return angle;
}

MotorCommand WheelLegController::update(const RobotState& state, double dt) {
  reference_[0] += command_.forward_speed * dt;
  reference_[1] = command_.forward_speed;

  const double control_heading = -state.yaw;
  const double heading_error = wrapAngle(command_.heading - control_heading);
  const double limited_error = std::clamp(
      heading_error, -kHeadingErrorLimit, kHeadingErrorLimit);
  reference_[2] = wrapAngle(state.yaw - limited_error);
  reference_[3] = 0.0;

  const auto measured = state.lqrVector();
  std::array<double, 10> error{};
  for (std::size_t i = 0; i < error.size(); ++i) {
    error[i] = measured[i] - reference_[i];
  }
  error[2] = wrapAngle(error[2]);

  MotorCommand output;
  for (std::size_t motor = 0; motor < output.torque_nm.size(); ++motor) {
    double normalized = kEquilibriumControl[motor];
    for (std::size_t state_index = 0; state_index < error.size(); ++state_index) {
      normalized -= kGain[motor][state_index] * error[state_index];
    }
    normalized = std::clamp(normalized, -1.0, 1.0);
    output.torque_nm[motor] = normalized * kMotorGearNm;
  }
  return output;
}

void WheelLegController::setCommand(const UserCommand& command) {
  command_ = command;
  command_.forward_speed = std::clamp(command_.forward_speed,
                                      -kMaxSpeed, kMaxSpeed);
  command_.heading = wrapAngle(command_.heading);
}

void WheelLegController::changeForwardSpeed(double delta_mps) {
  command_.forward_speed = std::clamp(command_.forward_speed + delta_mps,
                                      -kMaxSpeed, kMaxSpeed);
}

void WheelLegController::changeHeading(double delta_rad) {
  command_.heading = wrapAngle(command_.heading + delta_rad);
}

void WheelLegController::stopAndHold(const RobotState& state) {
  command_.forward_speed = 0.0;
  command_.heading = -state.yaw;
  reference_[0] = state.x;
  reference_[1] = 0.0;
}

void WheelLegController::reset() {
  reference_ = kReference;
  command_ = {};
}

}  // namespace wheel_leg
