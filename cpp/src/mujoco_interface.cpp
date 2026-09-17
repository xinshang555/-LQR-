#include "mujoco_interface.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace wheel_leg {
namespace {

constexpr int kBaseQuat = 3;
constexpr double kWheelArmature = 1.0e-7;

int requireId(const mjModel* model, mjtObj type, const char* name) {
  const int id = mj_name2id(model, type, name);
  if (id < 0) {
    throw std::runtime_error(std::string("MuJoCo object not found: ") + name);
  }
  return id;
}

}  // namespace

MujocoInterface::MujocoInterface(const std::string& model_path) {
  char error[1024]{};
  model_.reset(mj_loadXML(model_path.c_str(), nullptr, error, sizeof(error)));
  if (!model_) {
    throw std::runtime_error(std::string("Cannot load MuJoCo model: ") + error);
  }
  data_.reset(mj_makeData(model_.get()));
  if (!data_) {
    throw std::runtime_error("Cannot allocate mjData");
  }

  const std::array<const char*, 4> actuator_names = {
      "hip_lower_joint_motor", "wheel_lower_joint_motor",
      "hip_upper_joint_motor", "wheel_upper_joint_motor"};
  for (std::size_t i = 0; i < actuator_names.size(); ++i) {
    actuator_ids_[i] = requireId(model_.get(), mjOBJ_ACTUATOR, actuator_names[i]);
  }

  mount_body_ = requireId(model_.get(), mjOBJ_BODY, "mount");
  leg_lower_body_ = requireId(model_.get(), mjOBJ_BODY, "leg_lower");
  wheel_lower_body_ = requireId(model_.get(), mjOBJ_BODY, "wheel_lower");
  leg_upper_body_ = requireId(model_.get(), mjOBJ_BODY, "leg_upper");
  wheel_upper_body_ = requireId(model_.get(), mjOBJ_BODY, "wheel_upper");

  for (const char* joint_name : {"wheel_lower_joint", "wheel_upper_joint"}) {
    const int joint = requireId(model_.get(), mjOBJ_JOINT, joint_name);
    model_->dof_armature[model_->jnt_dofadr[joint]] = kWheelArmature;
  }
  resetHome();
}

MujocoInterface::~MujocoInterface() = default;

double MujocoInterface::clamp(double value, double low, double high) {
  return std::max(low, std::min(high, value));
}

std::array<double, 3> MujocoInterface::bodyAngularVelocity(int body_id) const {
  mjtNum velocity[6]{};
  mj_objectVelocity(model_.get(), data_.get(), mjOBJ_BODY, body_id, velocity, 0);
  return {velocity[0], velocity[1], velocity[2]};
}

double MujocoInterface::legLean(int leg_body_id, int wheel_body_id) const {
  const mjtNum* leg = data_->xpos + 3 * leg_body_id;
  const mjtNum* wheel = data_->xpos + 3 * wheel_body_id;
  return std::atan2(wheel[0] - leg[0], -(wheel[2] - leg[2]));
}

RobotState MujocoInterface::readState() const {
  const double w = data_->qpos[kBaseQuat + 0];
  const double x = data_->qpos[kBaseQuat + 1];
  const double y = data_->qpos[kBaseQuat + 2];
  const double z = data_->qpos[kBaseQuat + 3];
  const double pitch_sine = clamp(2.0 * (w * y + z * x), -1.0, 1.0);
  const auto mount_omega = bodyAngularVelocity(mount_body_);
  const auto lower_omega = bodyAngularVelocity(leg_lower_body_);
  const auto upper_omega = bodyAngularVelocity(leg_upper_body_);

  RobotState state;
  state.x = data_->qpos[0];
  state.x_dot = data_->qvel[0];
  state.yaw = std::atan2(2.0 * (w * z + x * y),
                         1.0 - 2.0 * (y * y + z * z));
  state.yaw_dot = mount_omega[2];
  state.leg_lower = legLean(leg_lower_body_, wheel_lower_body_);
  state.leg_lower_dot = -lower_omega[1];
  state.leg_upper = legLean(leg_upper_body_, wheel_upper_body_);
  state.leg_upper_dot = -upper_omega[1];
  state.body_pitch = -std::asin(pitch_sine);
  state.body_pitch_dot = -mount_omega[1];
  state.body_height = data_->qpos[2];
  return state;
}

void MujocoInterface::writeTorque(const MotorCommand& command) {
  for (std::size_t i = 0; i < actuator_ids_.size(); ++i) {
    const int actuator = actuator_ids_[i];
    const double gear = model_->actuator_gear[6 * actuator];
    if (std::abs(gear) < 1.0e-12) {
      throw std::runtime_error("Actuator gear must be non-zero");
    }
    const double low = model_->actuator_ctrlrange[2 * actuator];
    const double high = model_->actuator_ctrlrange[2 * actuator + 1];
    data_->ctrl[actuator] = clamp(command.torque_nm[i] / gear, low, high);
  }
}

void MujocoInterface::step() { mj_step(model_.get(), data_.get()); }

void MujocoInterface::resetHome() {
  if (model_->nkey < 1) {
    throw std::runtime_error("Model has no home keyframe");
  }
  mj_resetDataKeyframe(model_.get(), data_.get(), 0);
  mj_forward(model_.get(), data_.get());
}

void MujocoInterface::applyPitchKick(double angle_rad) {
  const double half = 0.5 * angle_rad;
  data_->qpos[3] = std::cos(half);
  data_->qpos[4] = 0.0;
  data_->qpos[5] = -std::sin(half);
  data_->qpos[6] = 0.0;
  mj_forward(model_.get(), data_.get());
}

double MujocoInterface::timestep() const { return model_->opt.timestep; }

double MujocoInterface::simulationTime() const { return data_->time; }

}  // namespace wheel_leg
