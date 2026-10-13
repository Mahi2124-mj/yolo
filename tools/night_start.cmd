@echo off
rem Overnight capture of the Toolroom camera until 09:00: raw 30-min recordings + every alert's evidence.
rem Started detached (via explorer) so it outlives any terminal. Output: D:\poketenashi-data\night\<date>\
cd /d C:\Users\DX-ADMIN\Pictures\yolo
start "night-record" /belownormal /min .venv\Scripts\python.exe -m tools.night_record --cam cam1 --until 09:00
start "night-evidence" /belownormal /min .venv\Scripts\python.exe -m tools.night_evidence --until 09:00
