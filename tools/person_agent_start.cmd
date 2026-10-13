@echo off
rem 10 Oct 2026 (user): the person agent - every person seen gets a snapshot (written by
rem the monitor, config person_log) AND a clip. tools.person_agent keeps the raw
rem recorders (Toolroom + Help-desk, 10-min segments, stream copy) and the clip cutter
rem running, restarts any that stops, and deletes raw video older than 48 h.
rem Clips: D:\poketenashi-data\persons\<date>\<cam>\  Log: D:\poketenashi-data\persons_raw\agent.log
cd /d C:\Users\DX-ADMIN\Pictures\yolo
start "person-agent" /belownormal /min tools\person_agent_loop.cmd
