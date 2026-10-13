@echo off
setlocal EnableDelayedExpansion
REM Poketenashi Safety Monitor - supervised launcher.
REM
REM Windows kills this app when the machine's commit charge reaches its ceiling
REM (16 GB RAM + a FIXED 16 GB pagefile = a hard 31.3 GB). It is not crashing:
REM there is no traceback, the process is killed mid-frame, and the most it
REM manages to report is "Unable to allocate 8.54 MiB" for one camera frame.
REM 28 of those kills in the last 7 days. This wrapper brings the app back.
REM
REM It cannot stop the kills. The fix for that is to let Windows manage the
REM pagefile (sysdm.cpl -> Advanced -> Performance Settings -> Advanced ->
REM Virtual memory), which raises the ceiling to ~47 GB.
REM
REM A batch loop rather than a Python supervisor on purpose: a second
REM interpreter would cost tens of MB on a box that is already out of memory,
REM where cmd waits on the child for about 2 MB.
REM
REM Usage:  watchdog.bat                    (live camera from config.yaml)
REM         watchdog.bat --source clip.mp4  (replay a recording)
REM Stop:   close this window, or Ctrl+C then Y.

cd /d "%~dp0"

REM findstr splits an unquoted /r pattern on SPACES and treats each piece as a
REM separate search term. "^  port:" therefore also searched for "^", which
REM matches EVERY line - so the loop ran over the whole file and PORT ended up
REM as token 2 of the last line ("debug: false" -> false). /c: forces one
REM pattern. Symptom was "http://localhost:false" and a Get-NetTCPConnection
REM -LocalPort false error on every start.
for /f "tokens=2 delims=: " %%p in ('findstr /r /c:"^  port:" config.yaml') do set PORT=%%p
REM Refuse to carry on with a non-numeric port rather than passing it to
REM PowerShell and to the URL.
echo %PORT%| findstr /r /c:"^[0-9][0-9]*$" >nul || set PORT=
if "%PORT%"=="" set PORT=8080

set LOG=%~dp0watchdog.log
set RUNS=0
REM Tells web.py a supervisor is watching: a hung main loop then exits with code
REM 3 (stacks in logs\fatal.log) and gets restarted here, instead of sitting on a
REM frozen picture that still says LIVE.
set POKETENASHI_SUPERVISED=1

echo ============================================
echo   POKETENASHI Safety Monitor  (supervised)
echo   http://localhost:%PORT%
echo   restart log: %LOG%
echo   full log   : %~dp0logs\monitor.log
echo ============================================
echo.

REM Clear the port once, at the start, so a leftover process cannot make the
REM first bind fail. A restart does not need this: the app it is replacing is
REM already dead.
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"
set BROWSER_ARG=

:loop
set /a RUNS+=1
echo.
echo [watchdog] start #!RUNS!  %date% %time%
"%~dp0.venv\Scripts\python.exe" "%~dp0web.py" !BROWSER_ARG! %*
set CODE=!errorlevel!

REM Only the first start may open a browser. A restart must NOT open another
REM tab: Edge is one of the biggest memory consumers on this box, so a crash
REM loop that spawned tabs would feed the very problem it is recovering from.
set BROWSER_ARG=--no-browser

REM Exit code 0 means the app chose to stop - Ctrl+C, or a --source file that
REM played to the end. Restarting that would be a loop with no end condition.
if !CODE! equ 0 (
  echo [watchdog] clean exit - not restarting.
  goto end
)
if exist "%~dp0watchdog.stop" (
  echo [watchdog] watchdog.stop found - not restarting.
  del "%~dp0watchdog.stop"
  goto end
)

for /f %%m in ('powershell -NoProfile -Command "[math]::Round(((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB),2)"') do set FREEGB=%%m
set WHY=killed
if !CODE! equ 3 set WHY=hung - main loop stalled - stacks in logs\fatal.log -
if !CODE! equ 4 set WHY=restart requested - restart.request -
echo %date% %time%  !WHY! after start #!RUNS!, exit !CODE!, !FREEGB! GB free>>"%LOG%"
echo [watchdog] exit !CODE! (!FREEGB! GB commit free) - waiting for room.

REM Restarting straight into a full machine just gets killed again, and every
REM attempt drags ~2.5 GB of models in before dying - a crash loop that makes
REM the shortage worse. Wait for room instead.
REM
REM The test is free commit in GB, not a percentage: this box idles near 89%%
REM of its ceiling, so any percentage threshold either waits for ever or never
REM waits at all. The app needs ~2.5 GB, and killing it hands ~2.5 GB straight
REM back, so 3 GB free is both meaningful and normally true immediately.
REM
REM Give up waiting after 5 minutes and try anyway, so a bad reading can never
REM leave the camera unwatched indefinitely.
set /a WAITED=0
:waitmem
powershell -NoProfile -Command "exit ([int](((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB) -lt 3.0))"
if !errorlevel! equ 0 goto restart
if !WAITED! geq 300 (
  echo %date% %time%  only !FREEGB! GB free after !WAITED!s - restarting anyway>>"%LOG%"
  goto restart
)
REM ping, not timeout: timeout dies with "Input redirection is not supported"
REM whenever stdin is not a console, which silently turns this wait into a spin
REM - the restart log then claims a 5 minute wait that really took 20 seconds.
ping -n 16 127.0.0.1 >nul
set /a WAITED+=15
goto waitmem

:restart
echo [watchdog] restarting.
goto loop

:end
endlocal
