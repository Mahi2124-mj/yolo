@echo off
rem 9 Oct 2026 (user): Toolroom recording for PO and the other rules, until 09:00.
rem Raw camera recording (30-min segments, light: ffmpeg stream copy) into
rem D:\poketenashi-data\toolroom_rec\<date>\raw  +  every Toolroom alert's clip and
rem poster copied to "Pictures\yolo\Toolroom alerts 9-10 Oct\<rule>\" with review.csv.
cd /d C:\Users\DX-ADMIN\Pictures\yolo
start "toolroom-record" /belownormal /min .venv\Scripts\python.exe -m tools.light_record --cam cam1 --until 09:00 --out D:/poketenashi-data/toolroom_rec --segment-min 30
start "toolroom-alerts" /belownormal /min .venv\Scripts\python.exe -m tools.night_poke --until 09:00 --rules PO,KE,NA,SHI,RUN,FALL,RESTRICTED --cams cam1 --out "Toolroom alerts 9-10 Oct"
