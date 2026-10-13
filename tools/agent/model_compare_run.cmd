@echo off
rem Runs the model comparison until it finishes; restarts after a crash (it resumes from results.jsonl).
cd /d C:\Users\DX-ADMIN\Pictures\yolo
set OPENBLAS_NUM_THREADS=1
:again
.venv\Scripts\python.exe tools\agent\model_compare.py PO_review_v4\model_compare_20261012 >> D:\poketenashi-data\agent\model_compare_20261012.log 2>&1
if exist PO_review_v4\model_compare_20261012\summary.md goto done
timeout /t 60 /nobreak >nul
goto again
:done
