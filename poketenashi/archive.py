"""Evidence archive: every alert's clip, poster and log row, kept for good.

The 2-day retention (poketenashi/retention.py) clears the live clips folder;
the user wants every rule break kept, 24x7 (7 Oct 2026). A thread of the
monitor itself (so it runs whenever the monitor runs - the watchdog keeps it
alive) copies each alert's files into <dir>/<YYYYMMDD of the alert>/ and
appends its row to that day's alerts.jsonl. Once a minute; a clip that is still
being encoded is picked up on a later pass. ~150 MB a day.

It opens violations.jsonl only to read it, for milliseconds: retention renames
that file, which a long-held handle would block on Windows.
"""
import json
import os
import shutil
import threading
import time

from . import late_names


def _row_time(r):
    try:
        return time.mktime(time.strptime(str(r.get("time")), "%Y-%m-%d %H:%M:%S"))
    except ValueError:
        return 0.0


class Archiver(threading.Thread):
    def __init__(self, log_file, clip_dir, out_dir, every_sec=60.0, settle_sec=20.0):
        super().__init__(name="evidence-archive", daemon=True)
        self.log_file, self.clip_dir, self.out_dir = log_file, clip_dir, out_dir
        self.every, self.settle = float(every_sec), float(settle_sec)
        self._seen = {}            # day -> set of row keys already archived
        self._pending = set()      # (day, file name) waiting for the clip to land

    def _day_keys(self, day):
        if day not in self._seen:
            keys = set()
            p = os.path.join(self.out_dir, day, "alerts.jsonl")
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    for l in f:
                        try:
                            r = json.loads(l)
                            if r.pop("person_late", False):
                                r.pop("person", None)          # the key is the row as logged
                            keys.add(json.dumps(r, sort_keys=True))
                        except ValueError:
                            pass
            self._seen[day] = keys
        return self._seen[day]

    def pass_once(self):
        try:
            with open(self.log_file, encoding="utf-8") as f:
                lines = f.readlines()
        except OSError:
            return 0
        added = 0
        for l in lines:
            try:
                r = json.loads(l)
            except ValueError:
                continue
            day = str(r.get("time", ""))[:10].replace("-", "")
            if len(day) != 8:
                continue
            k = json.dumps(r, sort_keys=True)
            keys = self._day_keys(day)
            if k in keys:
                continue
            # 9 Oct: give face ID its late name first (late_names); the key above
            # stays the row as logged, so the row is archived exactly once
            if not r.get("person") and time.time() - _row_time(r) < self.settle:
                continue
            if not r.get("person") and late_names.fill(
                    r, late_names.load(late_names.path_for(self.log_file))).get("person"):
                r["person_late"] = True                # not in the log row: see _day_keys
            os.makedirs(os.path.join(self.out_dir, day), exist_ok=True)
            with open(os.path.join(self.out_dir, day, "alerts.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(r) + "\n")
            keys.add(k)
            added += 1
            for name in (r.get("clip"), r.get("poster"), r.get("snapshot")):
                if name:
                    self._pending.add((day, name))
        for day, name in list(self._pending):
            dst = os.path.join(self.out_dir, day, name)
            src = os.path.join(self.clip_dir, name)
            if os.path.exists(dst):
                self._pending.discard((day, name))
            elif os.path.exists(src) and time.time() - os.path.getmtime(src) > self.settle:
                shutil.copy2(src, dst)
                self._pending.discard((day, name))
        # forget the keys of days the log no longer holds (retention trimmed
        # them - they cannot come back), so memory does not grow for ever
        log_days = set()
        for l in lines:
            log_days.add(l[10:20].replace("-", "") if l.startswith('{"time": "') else "")
        for d in [d for d in self._seen if d not in log_days]:
            del self._seen[d]
        return added

    def run(self):
        print(f"[archive] keeping every alert's clip, poster and row in {self.out_dir}\\<date>", flush=True)
        while True:
            try:
                self.pass_once()
            except Exception as e:             # never take the monitor down
                print(f"[archive] pass failed: {type(e).__name__}: {e}", flush=True)
            time.sleep(self.every)
