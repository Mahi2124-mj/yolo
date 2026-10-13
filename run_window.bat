@echo off
REM Poketenashi Safety Monitor - DESKTOP window (OpenCV).
REM Use run.bat for the browser dashboard instead.
REM Usage:  run_window.bat                 (live camera from config.yaml)
REM         run_window.bat --source 0      (webcam test)
REM         run_window.bat --source clip.mp4
"%~dp0.venv\Scripts\python.exe" "%~dp0main.py" %*
