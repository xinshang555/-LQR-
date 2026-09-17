#pragma once

#include <array>
#include <memory>
#include <string>

#include <mujoco/mujoco.h>

#include "robot_types.hpp"

namespace wheel_leg {

class MujocoInterface {
 public:
  explicit MujocoInterface(const std::string& model_path);
  ~MujocoInterface();

  MujocoInterface(const MujocoInterface&) = delete;
  MujocoInterface& operator=(const MujocoInterface&) = delete;

  RobotState readState() const;
  void writeTorque(const MotorCommand& command);
  void step();
  void resetHome();

  double timestep() const;
  double simulationTime() const;
  mjModel* model() const { return model_.get(); }
  mjData* data() const { return data_.get(); }

 private:
  struct ModelDeleter {
    void operator()(mjModel* model) const { mj_deleteModel(model); }
  };
  struct DataDeleter {
    void operator()(mjData* data) const { mj_deleteData(data); }
  };

  static double clamp(double value, double low, double high);
  std::array<double, 3> bodyAngularVelocity(int body_id) const;
  double legLean(int leg_body_id, int wheel_body_id) const;

  std::unique_ptr<mjModel, ModelDeleter> model_;
  std::unique_ptr<mjData, DataDeleter> data_;
  std::array<int, 4> actuator_ids_{};
  int mount_body_{};
  int leg_lower_body_{};
  int wheel_lower_body_{};
  int leg_upper_body_{};
  int wheel_upper_body_{};
};

}  // namespace wheel_leg
