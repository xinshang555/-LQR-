#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

#include "mujoco_interface.hpp"
#include "wheel_leg_controller.hpp"

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr int kControlEvery = 10;
constexpr double kSpeedStep = 0.01;
constexpr double kHeadingStep = 2.0 * kPi / 180.0;

struct Options {
  std::string model_path = "urdf/car.xml";
  double seconds = 30.0;
  double speed = 0.0;
  double heading = 0.0;
  double kick = 0.0;
  bool headless = false;
};

struct App {
  wheel_leg::MujocoInterface* simulation{};
  wheel_leg::WheelLegController* controller{};
  mjvCamera* camera{};
  mjvScene* scene{};
  bool reset_requested{};
  bool left_down{};
  bool middle_down{};
  bool right_down{};
  double last_x{};
  double last_y{};
};

Options parseOptions(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&](const char* option) -> std::string {
      if (++i >= argc) throw std::runtime_error(std::string("Missing value for ") + option);
      return argv[i];
    };
    if (arg == "--model") options.model_path = value("--model");
    else if (arg == "--seconds") options.seconds = std::stod(value("--seconds"));
    else if (arg == "--vx") options.speed = std::stod(value("--vx"));
    else if (arg == "--target-yaw") options.heading = std::stod(value("--target-yaw")) * kPi / 180.0;
    else if (arg == "--kick-deg") options.kick = std::stod(value("--kick-deg")) * kPi / 180.0;
    else if (arg == "--headless") options.headless = true;
    else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: wheel_leg_sim [--model FILE] [--seconds N] [--headless]"
                   " [--vx MPS] [--target-yaw DEG] [--kick-deg DEG]\n"
                   "Keys: W/S speed, A/D heading, Space/X stop, R reset\n";
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }
  return options;
}

void printCommand(const App& app) {
  const auto& command = app.controller->command();
  std::cout << "[command] vx=" << std::showpos << std::fixed << std::setprecision(0)
            << command.forward_speed * 1000.0 << " mm/s, heading="
            << command.heading * 180.0 / kPi << " deg\n" << std::noshowpos;
}

void keyCallback(GLFWwindow* window, int key, int, int action, int) {
  // One physical key press means one command increment. Ignoring GLFW_REPEAT
  // prevents a short hold from silently requesting maximum speed/heading.
  if (action != GLFW_PRESS) return;
  auto* app = static_cast<App*>(glfwGetWindowUserPointer(window));
  if (!app) return;
  if (key == GLFW_KEY_W) app->controller->changeForwardSpeed(kSpeedStep);
  else if (key == GLFW_KEY_S) app->controller->changeForwardSpeed(-kSpeedStep);
  else if (key == GLFW_KEY_A) app->controller->changeHeading(kHeadingStep);
  else if (key == GLFW_KEY_D) app->controller->changeHeading(-kHeadingStep);
  else if (key == GLFW_KEY_SPACE || key == GLFW_KEY_X)
    app->controller->stopAndHold(app->simulation->readState());
  else if (key == GLFW_KEY_R) app->reset_requested = true;
  else if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(window, GLFW_TRUE);
  else return;
  printCommand(*app);
}

void mouseButtonCallback(GLFWwindow* window, int, int, int) {
  auto* app = static_cast<App*>(glfwGetWindowUserPointer(window));
  app->left_down = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
  app->middle_down = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
  app->right_down = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
  glfwGetCursorPos(window, &app->last_x, &app->last_y);
}

void cursorCallback(GLFWwindow* window, double x, double y) {
  auto* app = static_cast<App*>(glfwGetWindowUserPointer(window));
  if (!app->left_down && !app->middle_down && !app->right_down) return;
  const double dx = x - app->last_x;
  const double dy = y - app->last_y;
  app->last_x = x;
  app->last_y = y;
  int width = 1;
  int height = 1;
  glfwGetWindowSize(window, &width, &height);
  const bool shift = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                     glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
  mjtMouse action;
  if (app->right_down) action = shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
  else if (app->left_down) action = shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
  else action = mjMOUSE_ZOOM;
  mjv_moveCamera(app->simulation->model(), action, dx / height, dy / height,
                 app->camera);
}

void scrollCallback(GLFWwindow* window, double, double yoffset) {
  auto* app = static_cast<App*>(glfwGetWindowUserPointer(window));
  mjv_moveCamera(app->simulation->model(), mjMOUSE_ZOOM, 0.0, -0.05 * yoffset,
                 app->camera);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parseOptions(argc, argv);
    wheel_leg::MujocoInterface simulation(options.model_path);
    wheel_leg::WheelLegController controller;
    controller.setCommand({options.speed, options.heading});
    if (options.kick != 0.0) simulation.applyPitchKick(options.kick);

    GLFWwindow* window = nullptr;
    mjvCamera camera;
    mjvOption visual_options;
    mjvScene scene;
    mjrContext context;
    mjv_defaultCamera(&camera);
    mjv_defaultOption(&visual_options);
    mjv_defaultScene(&scene);
    mjr_defaultContext(&context);
    App app{&simulation, &controller, &camera, &scene};

    if (!options.headless) {
      if (!glfwInit()) throw std::runtime_error("Cannot initialize GLFW");
      window = glfwCreateWindow(1200, 800, "Wheel-leg MuJoCo C++", nullptr, nullptr);
      if (!window) throw std::runtime_error("Cannot create GLFW window");
      glfwMakeContextCurrent(window);
      glfwSwapInterval(1);
      glfwSetWindowUserPointer(window, &app);
      glfwSetKeyCallback(window, keyCallback);
      glfwSetMouseButtonCallback(window, mouseButtonCallback);
      glfwSetCursorPosCallback(window, cursorCallback);
      glfwSetScrollCallback(window, scrollCallback);
      mjv_makeScene(simulation.model(), &scene, 2000);
      mjr_makeContext(simulation.model(), &context, mjFONTSCALE_150);
      std::cout << "[keyboard] W/S speed, A/D heading, Space/X stop, R reset\n";
    }

    const double home_height = simulation.readState().body_height;
    bool fell = false;
    bool safety_latched = false;
    double max_pitch = 0.0;
    double max_height_error = 0.0;
    int step_count = 0;
    const auto wall_start = std::chrono::steady_clock::now();

    while (simulation.simulationTime() < options.seconds &&
           (!window || !glfwWindowShouldClose(window))) {
      if (app.reset_requested) {
        simulation.resetHome();
        controller.reset();
        safety_latched = false;
        app.reset_requested = false;
        step_count = 0;
        std::cout << "[command] reset to home\n";
      }
      if (step_count % kControlEvery == 0) {
        const auto state = simulation.readState();
        if (safety_latched) {
          simulation.writeTorque({});
        } else {
          simulation.writeTorque(controller.update(
              state, simulation.timestep() * kControlEvery));
        }
      }
      simulation.step();
      ++step_count;

      const auto state = simulation.readState();
      max_pitch = std::max(max_pitch, std::abs(state.body_pitch));
      max_height_error = std::max(max_height_error,
                                  std::abs(state.body_height - home_height));
      if (!std::isfinite(state.body_pitch) ||
          std::abs(state.body_pitch) > 5.0 * kPi / 180.0 ||
          std::abs(state.body_height - home_height) > 0.005) {
        fell = true;
      }
      if (!safety_latched &&
          (!std::isfinite(state.body_pitch) ||
           std::abs(state.body_pitch) > 12.0 * kPi / 180.0 ||
           std::abs(state.body_height - home_height) > 0.010)) {
        safety_latched = true;
        simulation.writeTorque({});
        std::cout << "[safety] fall detected; motors disabled, press R to reset\n";
      }

      if (window && step_count % 20 == 0) {
        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        const mjrRect viewport{0, 0, width, height};
        mjv_updateScene(simulation.model(), simulation.data(), &visual_options,
                        nullptr, &camera, mjCAT_ALL, &scene);
        mjr_render(viewport, &scene, &context);
        glfwSwapBuffers(window);
        glfwPollEvents();

        const auto target = wall_start + std::chrono::duration<double>(
            simulation.simulationTime());
        std::this_thread::sleep_until(target);
      }
    }

    const auto final_state = simulation.readState();
    std::cout << std::fixed << std::setprecision(4)
              << "final pitch: " << final_state.body_pitch * 180.0 / kPi << " deg\n"
              << "final/target heading: " << -final_state.yaw * 180.0 / kPi
              << " / " << controller.command().heading * 180.0 / kPi << " deg\n"
              << "max pitch: " << max_pitch * 180.0 / kPi << " deg\n"
              << "max height error: " << max_height_error * 1000.0 << " mm\n"
              << "result: " << (fell ? "FELL" : "STANDING") << '\n';

    if (window) {
      mjr_freeContext(&context);
      mjv_freeScene(&scene);
      glfwDestroyWindow(window);
      glfwTerminate();
    }
    return fell ? 2 : 0;
  } catch (const std::exception& error) {
    std::cerr << "[error] " << error.what() << '\n';
    return 1;
  }
}
