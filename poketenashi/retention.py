"""Automatic clean-up of old evidence.

Evidence accumulates faster than anyone deletes it: this project's snapshots
folder reached 10,396 files and 7.4 GB before anybody looked. Clips are heavier
still, so retention is not optional - without it the disk fills and the app dies
on a write, which is a much worse failure than losing a three-day-old clip.

A daemon thread sweeps on an interval: anything in the media folders older than
`days` goes, and the JSONL log is rewritten without the rows that predate the
window. The log is rewritten via a temp file and one atomic replace, so a crash
mid-sweep leaves the original intact rather than a half-written log.
"""
import json
import os
import threading
import time

SWEEP_MIN_DEFAULT = 30.0        # minutes between sweeps


def _purge_dir(path, cutoff):
    """Delete files last modified before `cutoff`. Returns (files, bytes)."""
    n = freed = 0
    if not os.path.isdir(path):
        return 0, 0
    for name in os.listdir(path):
        f = os.path.join(path, name)
        try:
            st = os.stat(f)
            if not os.path.isfile(f) or st.st_mtime >= cutoff:
                continue
            size = st.st_size
            os.remove(f)
            n += 1
            freed += size
        except OSError:
            continue                # in use, or vanished under us - skip it
    return n, freed


def _replace(src, dst, tries=15, wait=0.1):
    """os.replace, retried. On Windows the rename fails while any other handle
    has the target open - the dashboard reads this log every few seconds - so a
    single attempt lost the whole sweep now and then."""
    for i in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(wait)


def _trim_log(path, cutoff, lock=None):
    """Drop rows older than `cutoff`. Returns how many were removed.

    `lock` is the one the alert writer holds while appending. Holding it from
    the read to the replace means a row appended in between cannot be lost.
    """
    if not os.path.isfile(path):
        return 0
    if lock is None:
        lock = threading.Lock()
    tmp = path + ".tmp"
    removed = kept = 0
    with lock:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as src, \
                 open(tmp, "w", encoding="utf-8") as dst:
                for line in src:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        t = time.mktime(time.strptime(
                            json.loads(line)["time"], "%Y-%m-%d %H:%M:%S"))
                    except Exception:
                        dst.write(line + "\n")      # unparseable: keep, don't lose data
                        kept += 1
                        continue
                    if t < cutoff:
                        removed += 1
                    else:
                        dst.write(line + "\n")
                        kept += 1
        except OSError:
            if os.path.exists(tmp):
                os.remove(tmp)
            return 0
        if removed:
            _replace(tmp, path)                 # atomic: never a half-written log
        else:
            os.remove(tmp)
    return removed


def sweep(days, dirs, log_path=None, log_lock=None):
    cutoff = time.time() - days * 86400.0
    files = freed = 0
    for d in dirs:
        n, b = _purge_dir(d, cutoff)
        files += n
        freed += b
    rows = _trim_log(log_path, cutoff, log_lock) if log_path else 0
    if files or rows:
        print(f"[retention] removed {files} files ({freed / 1024 / 1024:.0f} MB) "
              f"and {rows} log rows older than {days} day(s)", flush=True)
    return files, freed, rows


def preview(days, dirs=None, log_path=None):
    """What a sweep with `days` would delete right now: {files, bytes, log_rows}.
    Nothing is touched - the dashboard shows it before a change is confirmed."""
    dirs = _where["dirs"] if dirs is None else dirs
    log_path = _where["log"] if log_path is None else log_path
    out = {"files": 0, "bytes": 0, "log_rows": 0}
    if not days or days <= 0:
        return out
    cutoff = time.time() - days * 86400.0
    for d in dirs or []:
        if not os.path.isdir(d):
            continue
        for e in os.scandir(d):
            try:
                if e.is_file():
                    st = e.stat()
                    if st.st_mtime < cutoff:
                        out["files"] += 1
                        out["bytes"] += st.st_size
            except OSError:
                continue
    if log_path and os.path.isfile(log_path):
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                m = line.find('"time": "')
                if m < 0:
                    continue
                try:
                    t = time.mktime(time.strptime(line[m + 9:m + 28], "%Y-%m-%d %H:%M:%S"))
                except ValueError:
                    continue
                if t < cutoff:
                    out["log_rows"] += 1
    return out


# What the sweep loop uses - changed live from the dashboard's Settings page
# (configure()); the loop reads it on every round.
_conf = {"days": 2.0, "sweep_min": SWEEP_MIN_DEFAULT, "trim_log": True}
_where = {"dirs": [], "log": None}
_wake = threading.Event()


def configure(days=None, sweep_min=None, trim_log=None):
    """Change the retention live; the next sweep (started right away) uses it."""
    if days is not None:
        _conf["days"] = float(days)
    if sweep_min is not None:
        _conf["sweep_min"] = float(sweep_min)
    if trim_log is not None:
        _conf["trim_log"] = bool(trim_log)
    _wake.set()
    return dict(_conf)


def settings():
    return dict(_conf)


def start(days, dirs, log_path=None, sweep_min=SWEEP_MIN_DEFAULT, log_lock=None, trim_log=True):
    """Sweep now, then every `sweep_min` minutes on a daemon thread. days <= 0
    switches deleting off (until configure() switches it back on).

    `log_path` is always the log; `trim_log` says whether old rows are dropped
    from it. (Passing no path to mean "don't trim" made the Settings switch
    useless: turning trimming on later had no file to trim.)"""
    _conf.update(days=float(days or 0), sweep_min=float(sweep_min),
                 trim_log=bool(trim_log) and log_path is not None)
    _where.update(dirs=list(dirs), log=log_path)

    def _loop():
        while True:
            d = _conf["days"]
            if d and d > 0:
                try:
                    sweep(d, dirs, log_path if _conf["trim_log"] else None, log_lock)
                except Exception as exc:            # a sweep must never kill the app
                    print("[retention] sweep failed:", exc, flush=True)
            _wake.wait(max(60.0, _conf["sweep_min"] * 60.0))
            _wake.clear()

    t = threading.Thread(target=_loop, daemon=True, name="retention")
    t.start()
    if days and days > 0:
        print(f"[retention] keeping {days} day(s); sweeping every {sweep_min:.0f} min", flush=True)
    else:
        print("[retention] disabled (retention.days is 0)", flush=True)
    return t
