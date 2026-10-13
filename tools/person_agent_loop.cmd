@echo off
rem Loop for tools.person_agent: if the supervisor itself ever stops, start it again.
cd /d C:\Users\DX-ADMIN\Pictures\yolo
:loop
.venv\Scripts\python.exe -m tools.person_agent
timeout /t 30 /nobreak >nul
goto loop
