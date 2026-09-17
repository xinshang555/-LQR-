cmake_minimum_required(VERSION 3.16)
project(wheel_leg_mujoco LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python3 COMPONENTS Interpreter QUIET)
if(NOT MUJOCO_ROOT AND Python3_Interpreter_FOUND)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
            "import pathlib,mujoco; print(pathlib.Path(mujoco.__file__).resolve().parent)"
    OUTPUT_VARIABLE MUJOCO_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
endif()

if(Python3_Interpreter_FOUND)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
            "import pathlib,glfw; print(pathlib.Path(glfw.__file__).resolve().parent)"
    OUTPUT_VARIABLE PYTHON_GLFW_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
endif()

find_path(MUJOCO_INCLUDE_DIR mujoco/mujoco.h
          HINTS "${MUJOCO_ROOT}/include" ENV MUJOCO_ROOT
          PATH_SUFFIXES include)
find_library(MUJOCO_LIBRARY
             NAMES mujoco
             HINTS "${MUJOCO_ROOT}" "${MUJOCO_ROOT}/lib" ENV MUJOCO_ROOT
             PATH_SUFFIXES lib bin)
if(NOT MUJOCO_LIBRARY AND MUJOCO_ROOT)
  file(GLOB MUJOCO_SHARED_LIBRARIES
       "${MUJOCO_ROOT}/libmujoco.so*"
       "${MUJOCO_ROOT}/lib/libmujoco.so*"
       "${MUJOCO_ROOT}/bin/mujoco.dll")
  list(LENGTH MUJOCO_SHARED_LIBRARIES MUJOCO_LIBRARY_COUNT)
  if(MUJOCO_LIBRARY_COUNT GREATER 0)
    list(GET MUJOCO_SHARED_LIBRARIES 0 MUJOCO_LIBRARY)
  endif()
endif()

find_package(glfw3 CONFIG QUIET)
if(TARGET glfw)
  set(GLFW_TARGET glfw)
elseif(TARGET glfw3)
  set(GLFW_TARGET glfw3)
else()
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(GLFW QUIET IMPORTED_TARGET glfw3)
  endif()
  if(TARGET PkgConfig::GLFW)
    set(GLFW_TARGET PkgConfig::GLFW)
  else()
    find_library(GLFW_BUNDLED_LIBRARY NAMES glfw
      HINTS "${PYTHON_GLFW_ROOT}/x11" "${PYTHON_GLFW_ROOT}/wayland")
    if(NOT GLFW_BUNDLED_LIBRARY)
      message(FATAL_ERROR
        "GLFW not found. Install libglfw3-dev or the Python glfw package.")
    endif()
    add_library(glfw_bundled SHARED IMPORTED)
    set_target_properties(glfw_bundled PROPERTIES
      IMPORTED_LOCATION "${GLFW_BUNDLED_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
    set(GLFW_TARGET glfw_bundled)
  endif()
endif()

if(NOT MUJOCO_INCLUDE_DIR OR NOT MUJOCO_LIBRARY)
  message(FATAL_ERROR
    "MuJoCo C library not found. Install the Python mujoco package or set MUJOCO_ROOT.")
endif()

add_executable(wheel_leg_sim
  src/main.cpp
  src/mujoco_interface.cpp
  src/wheel_leg_controller.cpp)
target_include_directories(wheel_leg_sim PRIVATE include "${MUJOCO_INCLUDE_DIR}")
target_link_libraries(wheel_leg_sim PRIVATE "${MUJOCO_LIBRARY}" ${GLFW_TARGET})
target_compile_options(wheel_leg_sim PRIVATE -Wall -Wextra -Wpedantic)

set_target_properties(wheel_leg_sim PROPERTIES
  BUILD_RPATH "${MUJOCO_ROOT};${MUJOCO_ROOT}/lib;${PYTHON_GLFW_ROOT}/x11;${PYTHON_GLFW_ROOT}/wayland")

enable_testing()
add_test(NAME cpp_balance
  COMMAND wheel_leg_sim --headless --seconds 5)
set_tests_properties(cpp_balance PROPERTIES
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/..")
add_test(NAME cpp_drive
  COMMAND wheel_leg_sim --headless --seconds 5 --vx 0.04)
set_tests_properties(cpp_drive PROPERTIES
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/..")
add_test(NAME cpp_turn
  COMMAND wheel_leg_sim --headless --seconds 12 --target-yaw 2)
set_tests_properties(cpp_turn PROPERTIES
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/..")
