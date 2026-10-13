"""Person agent supervisor (10 Oct 2026). Keeps the raw recorders of Toolroom (cam1)
and Help-desk (cam4) and the clip cutter (tools.person_clips) running - if one
exits (camera outage, memory pressure: all three died at ~12:00 on 10 Oct) it is
started again after 20 s. Raw segments older than KEEP_H hours are deleted (the
person clips are already cut by then; Help-desk raw is ~45 GB a day).
Standard library only, a few MB. Log: D:/poketenashi-data/persons_raw/agent.log

    python -m tools.person_agent
"""
import datetime as dt
import glob
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
RAW = "D:/poketenashi-data/persons_raw"
KEEP_H = 48
JOBS = {
    "rec-cam1": ["-m", "tools.light_record", "--cam", "cam1", "--until", "23:59", "--out", RAW, "--segment-min", "10"],
    "rec-cam4": ["-m", "tools.light_record", "--cam", "cam4", "--until", "23:59", "--out", RAW, "--segment-min", "10"],
    "clips": ["-m", "tools.person_clips", "--until", "23:59"],
}
# 12 Oct: the KE review holds yolo11m (~2.4 GB) - it now runs ONCE every PERIODIC_MIN
# minutes and exits, instead of staying loaded (the PC ran out of memory)
PERIODIC = {"ke-review": ["-m", "tools.ke_review", "--once"],
            "po-check-clips": ["-m", "tools.po_check_clips", "--once"]}     # 13 Oct: PO check clips + page (file copies only)
PERIODIC_MIN = 30
BELOW_NORMAL = 0x00004000


def log(msg):
    os.makedirs(RAW, exist_ok=True)
    with open(os.path.join(RAW, "agent.log"), "a", encoding="utf-8") as f:
        f.write(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {msg}\n")


def start(name):
    p = subprocess.Popen([PY] + JOBS[name], cwd=ROOT, creationflags=BELOW_NORMAL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log(f"started {name} (pid {p.pid})")
    return p


def prune():
    cut = time.time() - KEEP_H * 3600
    n = 0
    for f in glob.glob(os.path.join(RAW, "*", "raw", "*.mp4")):
        try:
            if os.path.getmtime(f) < cut:
                os.remove(f)
                n += 1
        except OSError:
            pass
    if n:
        log(f"pruned {n} raw segments older than {KEEP_H} h")


def main():
    procs = {k: start(k) for k in JOBS}
    last_prune = 0.0
    periodic = {}                                   # name -> (process, start time)
    while True:
        time.sleep(20)
        for k, p in procs.items():
            if p.poll() is not None:
                log(f"{k} exited (code {p.returncode}) - restarting")
                procs[k] = start(k)
        for k, args in PERIODIC.items():
            p, t0 = periodic.get(k, (None, 0.0))
            if (p is None or p.poll() is not None) and time.time() - t0 >= PERIODIC_MIN * 60:
                q = subprocess.Popen([PY] + args, cwd=ROOT, creationflags=BELOW_NORMAL,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                periodic[k] = (q, time.time())
                log(f"periodic {k} (pid {q.pid})")
        if time.time() - last_prune > 3600:
            last_prune = time.time()
            prune()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:                       # leave a trace, then the cmd loop restarts us
        log(f"supervisor error: {type(e).__name__}: {e}")
        sys.exit(1)
