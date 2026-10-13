@echo off
REM Poketenashi Safety Monitor - BROWSER dashboard.
REM Runs the same pipeline as before, but the annotated view goes to a web page
REM instead of an OpenCV window. The browser opens once the API is answering.
REM
REM Usage:  run.bat                    (live camera from config.yaml)
REM         run.bat --source 0         (webcam test)
REM         run.bat --source clip.mp4  (replay a recording)
REM         run.bat --port 9000
REM
REM For the old desktop window, use run_window.bat.
REM
REM This is a live safety monitor, so it always runs SUPERVISED: watchdog.bat
REM restarts it when Windows kills it for memory or when its main loop hangs.
REM Everything it prints also goes to logs\monitor.log.
call "%~dp0watchdog.bat" %*
