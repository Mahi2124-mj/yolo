"""One place where everything the process prints ends up: logs/monitor.log.

Before this, the console window was the only record of what the monitor said,
it scrolled away within seconds, and most of it was noise:

  * MediaPipe's C++ layer tries to upload usage telemetry and, with no route to
    Google from this network, logs a four-line "Failed to send to clearcut"
    ERROR over and over. It is not this app's error and nothing can be done
    about it, but it looks exactly like a real one.
  * That native output goes straight to the C runtime's file descriptor 2, so
    swapping sys.stderr in Python never even sees it.

So file descriptors 1 and 2 are pointed at a pipe BEFORE the heavy libraries
load (install() has to be the first thing the entry point does), and one reader
thread:

  * drops the known-harmless native noise, and says so once,
  * appends every remaining line, timestamped, to the log (size-rotated),
  * forwards it to the real console on a separate thread through a bounded
    queue, so a console that stops accepting output can never stall the
    pipeline - lines are dropped from the screen instead, never from the log.

faulthandler writes to a second file, logs/fatal.log, that stays open for the
life of the process: when a native library aborts or the interpreter crashes,
the Python stack of every thread lands there even though the pipe dies with
the process.
"""
import atexit
import faulthandler
import os
import queue
import re
import sys
import threading
import time

# A native "Failed to send to clearcut" block: the first line names clearcut,
# the lines after it are its source-location trace.
_NOISE_HEAD = re.compile(r"clearcut", re.IGNORECASE)
_NOISE_TAIL = re.compile(r"^(\s|=== Source Location Trace|[\w./\\-]+\.(cc|h|cpp|c):\d+\s*$)")
# Written into the pipe at exit so the reader can say "everything before this
# is in the log" - there is no other way to know a pipe has been drained.
_FLUSH_MARK = "\x00logsetup-flush\x00"

_state = None


class _RotatingFile:
    """Append-only text file with size rotation that never raises.

    Rotation renames monitor.log -> monitor.log.1 -> ... Windows refuses the
    rename while another process holds the file open; then keep appending and
    try again a little later. Losing rotation for a while is harmless. Losing
    lines, or an exception inside the logger, is not.
    """

    def __init__(self, path, max_bytes, backups):
        self.path = path
        self.max_bytes = max_bytes
        self.backups = backups
        self.f = None
        self.size = 0
        self.next_rotate = max_bytes
        self.lock = threading.Lock()

    def _open(self):
        self.f = open(self.path, "a", encoding="utf-8", errors="replace")
        self.size = self.f.tell()
        self.next_rotate = max(self.max_bytes, self.size + 1)

    def write(self, text):
        with self.lock:
            try:
                if self.f is None:
                    self._open()
                self.f.write(text)
                self.f.flush()
                self.size += len(text)
                if self.size >= self.next_rotate:
                    self._rotate()
            except Exception:
                pass

    def _rotate(self):
        try:
            self.f.close()
            self.f = None
            for i in range(self.backups - 1, 0, -1):
                src = f"{self.path}.{i}"
                if os.path.exists(src):
                    os.replace(src, f"{self.path}.{i + 1}")
            os.replace(self.path, self.path + ".1")
        except OSError:
            pass
        finally:
            if self.f is None:
                self._open()
            if self.size >= self.max_bytes:            # the rename did not happen
                self.next_rotate = self.size + self.max_bytes // 10


def _reader(rfd, logf, conq, st):
    buf = b""
    skipping = False
    while True:
        try:
            chunk = os.read(rfd, 65536)
        except OSError:
            return
        if not chunk:
            return
        buf += chunk
        *lines, buf = buf.split(b"\n")
        for raw in lines:
            try:
                line = raw.rstrip(b"\r").decode("utf-8", "replace")
                if line == _FLUSH_MARK:
                    st["flushed"].set()
                    continue
                if _NOISE_HEAD.search(line) or (skipping and _NOISE_TAIL.match(line)):
                    skipping = True
                    st["suppressed"] += 1
                    if not st["announced"]:
                        st["announced"] = True
                        note = ("[log] MediaPipe telemetry noise ('Failed to send to "
                                "clearcut') is filtered from here on - it is not an app error")
                        logf.write(time.strftime("%Y-%m-%d %H:%M:%S ") + note + "\n")
                        _to_console(conq, st, note + "\n")
                    continue
                skipping = False
                logf.write(time.strftime("%Y-%m-%d %H:%M:%S ") + line + "\n")
                _to_console(conq, st, line + "\n")
            except Exception:
                continue


def _to_console(conq, st, text):
    if conq is None:
        return
    try:
        conq.put_nowait(text)
    except queue.Full:
        st["console_dropped"] += 1


def _console_writer(con, conq):
    while True:
        text = conq.get()
        try:
            con.write(text)
            con.flush()
        except Exception:
            pass
        finally:
            conq.task_done()


def install(log_dir="logs", name="monitor.log", max_mb=10, backups=5):
    """Route fd 1/2 through the filter and into `log_dir/name`. Idempotent.

    Call it before importing cv2, ultralytics or mediapipe: a library that
    grabbed the old stdout at import time keeps writing to a handle this closes.
    Returns the log path.
    """
    global _state
    if _state is not None:
        return _state["path"]
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.abspath(os.path.join(log_dir, name))
    logf = _RotatingFile(path, int(max_mb * 1024 * 1024), int(backups))

    fatal = None
    try:
        fatal = open(os.path.join(log_dir, "fatal.log"), "a", encoding="utf-8")
        faulthandler.enable(file=fatal, all_threads=True)
    except Exception:
        fatal = None

    for s in (sys.stdout, sys.stderr):
        try:
            s.flush()
        except Exception:
            pass

    # The real console, kept before fd 1 is repointed. Absent under pythonw.
    con = conq = None
    try:
        con_fd = os.dup(1)
        # open() on a console fd picks the UTF-16 console writer by itself, so
        # non-ASCII text shows correctly whatever the code page is.
        con = open(con_fd, "w", encoding="utf-8", errors="replace", buffering=1)
        conq = queue.Queue(maxsize=2000)
    except OSError:
        con = None

    rfd, wfd = os.pipe()
    os.dup2(wfd, 1)
    os.dup2(wfd, 2)
    os.close(wfd)
    if os.name == "nt":
        # Native code that asks Windows for its std handles (rather than using
        # the C runtime's fds) must get the pipe too.
        try:
            import ctypes
            import msvcrt
            k32 = ctypes.windll.kernel32
            k32.SetStdHandle(-11, msvcrt.get_osfhandle(1))    # STD_OUTPUT_HANDLE
            k32.SetStdHandle(-12, msvcrt.get_osfhandle(2))    # STD_ERROR_HANDLE
        except Exception:
            pass
    sys.stdout = open(1, "w", encoding="utf-8", errors="replace", buffering=1, closefd=False)
    sys.stderr = open(2, "w", encoding="utf-8", errors="replace", buffering=1, closefd=False)

    st = {"path": path, "logf": logf, "fatal": fatal, "suppressed": 0,
          "announced": False, "console_dropped": 0, "flushed": threading.Event(),
          "conq": conq}
    threading.Thread(target=_reader, args=(rfd, logf, conq, st),
                     daemon=True, name="log-reader").start()
    if con is not None:
        threading.Thread(target=_console_writer, args=(con, conq),
                         daemon=True, name="log-console").start()
    _state = st
    atexit.register(_drain)
    logf.write(time.strftime("%Y-%m-%d %H:%M:%S ")
               + f"[log] ---- process {os.getpid()} started: {' '.join(sys.argv)} ----\n")
    return path


def _drain(timeout=2.0):
    """At exit: wait until everything printed so far has reached the log."""
    st = _state
    if st is None:
        return
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        os.write(1, (_FLUSH_MARK + "\n").encode("utf-8"))
        st["flushed"].wait(timeout)
        deadline = time.time() + 1.0
        while st["conq"] is not None and not st["conq"].empty() and time.time() < deadline:
            time.sleep(0.02)
    except Exception:
        pass


def log_direct(msg):
    """Write straight to the log file, bypassing the pipe.

    For last words from a thread that is about to os._exit() the process - a
    line still sitting in the pipe at that moment would be lost.
    """
    st = _state
    if st is None:
        print(msg, flush=True)
        return
    st["logf"].write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")
    _to_console(st["conq"], st, msg + "\n")


def dump_stacks():
    """Every thread's Python stack into logs/fatal.log (for a hang report)."""
    st = _state
    target = st["fatal"] if st and st["fatal"] else sys.stderr
    try:
        target.write(time.strftime("\n==== %Y-%m-%d %H:%M:%S stack dump ====\n"))
        target.flush()
        faulthandler.dump_traceback(file=target, all_threads=True)
        target.flush()
    except Exception:
        pass
