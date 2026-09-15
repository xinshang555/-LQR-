@echo off
setlocal
title MuJoCo LQR Simulation
pushd "%~dp0"

echo Starting the MuJoCo LQR viewer...
echo Close the MuJoCo window to stop.
echo.

wsl -d Ubuntu-24.04 -e python3 sim/run_lqr.py --view --seconds 86400

echo.
if errorlevel 1 (
  echo [ERROR] The simulation failed. Keep this window open and take a screenshot.
) else (
  echo The simulation has ended normally.
)
pause
popd
endlocal
