@echo off
rem KE phone checker: build the data (resumes after a crash), then cross-validate. 12 Oct 2026.
cd /d C:\Users\DX-ADMIN\Pictures\yolo
set OPENBLAS_NUM_THREADS=1
:again
.venv\Scripts\python.exe tools\agent\ke_phone_data.py D:\poketenashi-data\ke_phone_data >> D:\poketenashi-data\agent\ke_phone_data.log 2>&1
findstr /b "total" D:\poketenashi-data\agent\ke_phone_data.log >nul
if errorlevel 1 ( timeout /t 60 /nobreak >nul & goto again )
.venv\Scripts\python.exe tools\agent\ke_phone_train.py cv D:\poketenashi-data\ke_phone_data > D:\poketenashi-data\agent\ke_phone_cv.log 2>&1
