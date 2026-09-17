@echo off
setlocal
title MuJoCo LQR Simulation
pushd "%~dp0"

set "SIM_ARGS=%*"
if not defined SIM_ARGS set "SIM_ARGS=--seconds 86400"

echo Building and starting the C++ MuJoCo LQR viewer...
echo Controls: W/S speed, A/D heading, Space/X stop, R reset.
echo Close the MuJoCo window to stop.
echo.

wsl -d Ubuntu-24.04 --cd "%~dp0" -e bash -lc "cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release && cmake --build cpp/build -j2 && ./cpp/build/wheel_leg_sim %SIM_ARGS%"

echo.
if errorlevel 1 (
  echo [ERROR] The simulation failed. Keep this window open and take a screenshot.
) else (
  echo The simulation has ended normally.
)
pause
popd
endlocal
