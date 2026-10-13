"""Watch the running monitor and print ONE line per problem.

Silent while everything is healthy. It runs next to the app - in a spare
console, or under an agent that turns each printed line into a notification -
and never touches the app itself: it only reads the app's log files and asks
the dashboard API how it is doing.

    python -m tools.health_watch
    python -m tools.health_watch --interval 10 --port 6002

What it reports (the same message at most once per --repeat-min):
  LOG      an error-looking line in logs/monitor.log (a traceback is one event)
  FATAL    new text in logs/fatal.log (a native crash, or a hang's stack dump)
  RESTART  a new line in watchdog.log (the app was killed or hung, and restarted)
  DOWN/UP  the dashboard stopped / started answering again
  CAMERA   the app says the camera is not delivering frames / recovered
  STALL    the camera is fine but no processed frame for a while
  SLOW     fps under --min-fps for several polls in a row / recovered
  MEMORY   commit headroom under --min-commit-gb: Windows starts killing
           processes when it reaches zero, and this app is the usual victim
  CLIENT   another machine (a TV, a phone) opened the dashboard - its IP and
           browser - or its page reported an error
"""
import argparse
import ctypes
import json
import os
import re
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Lines worth a human's attention. [ALERT] rows are the app doing its job, not
# a problem, and are excluded below.
ERRORISH = re.compile(
    r"traceback|error|exception|failed|failure|fatal|unable to|cannot|denied|"
    r"killed|out of memory|stalled|\[camera\]|\[watchdog\]|warning", re.IGNORECASE)
IGNORE = re.compile(r"\[ALERT\]|\[log\] MediaPipe telemetry noise|\[retention\] removed")


def emit(kind, msg):
    print(f"{time.strftime('%H:%M:%S')} {kind}: {msg}", flush=True)


class Limiter:
    """Same message (digits ignored) at most once per `window` seconds."""

    def __init__(self, window):
        self.window = window
        self.seen = {}          # signature -> [last_emit_time, suppressed_count]

    def __call__(self, kind, msg):
        sig = kind + re.sub(r"\d+", "#", msg)[:160]
        now = time.time()
        last = self.seen.get(sig)
        if last and now - last[0] < self.window:
            last[1] += 1
            return
        extra = f"  (+{last[1]} more like it since {time.strftime('%H:%M', time.localtime(last[0]))})" \
            if last and last[1] else ""
        self.seen[sig] = [now, 0]
        emit(kind, msg + extra)


class Tail:
    """New complete lines of a file since the last call. Opens and closes the
    file every time so it never blocks the writer's rotation (Windows refuses a
    rename while any reader holds the file)."""

    def __init__(self, path, from_start=False):
        self.path = path
        self.pos = 0 if from_start else self._size()
        self.partial = ""

    def _size(self):
        try:
            return os.path.getsize(self.path)
        except OSError:
            return 0

    def lines(self):
        size = self._size()
        if size < self.pos:              # rotated or truncated: start the new file
            self.pos, self.partial = 0, ""
        if size == self.pos:
            return []
        try:
            with open(self.path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(self.pos)
                data = f.read()
                self.pos = f.tell()
        except OSError:
            return []
        data = self.partial + data
        *done, self.partial = data.split("\n")
        return [d.rstrip("\r") for d in done]


def commit_free_gb():
    """Commit charge still available (what Windows kills processes over), GB."""
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    st = MEMORYSTATUSEX()
    st.dwLength = ctypes.sizeof(st)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
        return None
    return st.ullAvailPageFile / 1024 ** 3


def read_port(default=8080):
    try:
        import yaml
        with open(os.path.join(ROOT, "config.yaml"), encoding="utf-8") as f:
            return int(((yaml.safe_load(f) or {}).get("web") or {}).get("port", default))
    except Exception:
        return default


def main():
    ap = argparse.ArgumentParser(description="Report monitor problems, one line each.")
    ap.add_argument("--port", type=int, default=None, help="dashboard port (default: config)")
    ap.add_argument("--interval", type=float, default=15.0, help="seconds between API polls")
    ap.add_argument("--min-fps", type=float, default=1.0)
    ap.add_argument("--stall-sec", type=float, default=30.0)
    ap.add_argument("--min-commit-gb", type=float, default=2.0)
    ap.add_argument("--repeat-min", type=float, default=10.0,
                    help="minutes before the same message is reported again")
    ap.add_argument("--from-start", action="store_true",
                    help="also report what is already in the logs")
    args = ap.parse_args()

    port = args.port or read_port()
    url = f"http://127.0.0.1:{port}/api/state"
    say = Limiter(args.repeat_min * 60.0)
    logs = Tail(os.path.join(ROOT, "logs", "monitor.log"), args.from_start)
    fatal = Tail(os.path.join(ROOT, "logs", "fatal.log"), args.from_start)
    restarts = Tail(os.path.join(ROOT, "watchdog.log"), args.from_start)

    down_since = cam_down_since = stall_since = None
    fails = slow_polls = 0
    slow = mem_low = False
    tb = None                        # lines of a traceback being collected
    next_poll = 0.0
    emit("WATCH", f"watching {url} and {os.path.join(ROOT, 'logs')} "
                  f"(silent while healthy)")

    while True:
        try:
            # ---- the app's own log -------------------------------------
            for line in logs.lines():
                text = line[20:] if re.match(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d ", line) else line
                if tb is not None:
                    if text.startswith((" ", "\t")):
                        tb.append(text.strip())
                        continue
                    where = next((l for l in reversed(tb) if l.startswith("File ")), "")
                    say("LOG", f"Traceback -> {text.strip()}  [{where}]")
                    tb = None
                    continue
                if "Traceback (most recent call last)" in text:
                    tb = []
                    continue
                if text.startswith("[web]  client "):
                    # the app already limits these to one per client per 10 min
                    emit("CLIENT", text[len("[web]  client "):].strip()[:300])
                    continue
                if ERRORISH.search(text) and not IGNORE.search(text):
                    say("LOG", text.strip()[:300])
            for line in fatal.lines():
                if line.strip():
                    say("FATAL", line.strip()[:300])
            for line in restarts.lines():
                if line.strip():
                    emit("RESTART", line.strip()[:300])

            # ---- the dashboard API ----------------------------------------
            now = time.time()
            if now >= next_poll:
                next_poll = now + args.interval
                try:
                    with urllib.request.urlopen(url, timeout=5) as r:
                        s = json.loads(r.read().decode("utf-8"))
                    fails = 0
                    if down_since is not None:
                        emit("UP", f"dashboard answering again after {now - down_since:.0f}s")
                        down_since = None
                    if not s.get("camera_ok"):
                        if cam_down_since is None:
                            cam_down_since = now
                            emit("CAMERA", s.get("last_error") or "camera not delivering frames")
                    elif cam_down_since is not None:
                        emit("CAMERA", f"ok again after {now - cam_down_since:.0f}s")
                        cam_down_since = None
                    # The live view and the detector run separately: either can stall.
                    age = max(s.get("frame_age_sec") or 0.0, s.get("infer_age_sec") or 0.0)
                    if s.get("camera_ok") and age > args.stall_sec:
                        if stall_since is None:
                            stall_since = now
                            emit("STALL", f"no processed frame for {age:.0f}s although the "
                                          f"camera is up (view {s.get('frame_age_sec')}s, "
                                          f"detector {s.get('infer_age_sec')}s)")
                    elif stall_since is not None:
                        emit("STALL", f"frames processing again after {now - stall_since:.0f}s")
                        stall_since = None
                    fps = float(s.get("fps") or 0.0)
                    if s.get("camera_ok") and fps < args.min_fps:
                        slow_polls += 1
                        if slow_polls >= 3 and not slow:
                            slow = True
                            emit("SLOW", f"fps {fps:.1f} (< {args.min_fps}) for "
                                         f"{slow_polls} polls in a row")
                    else:
                        slow_polls = 0
                        if slow:
                            slow = False
                            emit("SLOW", f"recovered, fps {fps:.1f}")
                except Exception as e:
                    fails += 1
                    if fails >= 2 and down_since is None:
                        down_since = now
                        emit("DOWN", f"dashboard on :{port} not answering ({type(e).__name__}: {e})")

                free = commit_free_gb()
                if free is not None:
                    if free < args.min_commit_gb and not mem_low:
                        mem_low = True
                        emit("MEMORY", f"only {free:.1f} GB commit left - Windows kills "
                                       f"processes near 0 (set the pagefile to system-managed)")
                    elif mem_low and free > args.min_commit_gb + 0.5:
                        mem_low = False
                        emit("MEMORY", f"recovered, {free:.1f} GB commit free")
        except Exception as e:           # the watcher must outlive its own bugs
            say("WATCH", f"internal error {type(e).__name__}: {e}")
        time.sleep(1.0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
