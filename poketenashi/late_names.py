"""Face-ID names that arrived after their alert row was written (9 Oct 2026, user:
"the name in the alert at the exact time; if it is ever missed, add it later").

web.py appends one JSON line per such alert to logs/alert_names.jsonl (next to
the alert log). The dashboard (/api/violations) and the evidence archive fill a
row's missing `person` from it. The alert log itself is never rewritten: the
retention renames it and the archive keys its rows by their content.
"""
import json
import os
import threading

_lock = threading.Lock()
_cache = {}          # path -> ((mtime_ns, size), {key: person})


def path_for(log_path):
    """logs/alert_names.jsonl beside the alert log (violations.jsonl)."""
    return os.path.join(os.path.dirname(os.path.abspath(log_path)), "logs", "alert_names.jsonl")


def _keys(r):
    ks = [f"f:{r[k]}" for k in ("poster", "snapshot") if r.get(k)]
    ks.append("t:{}|{}|{}|{}".format(r.get("time"), r.get("camera") or "cam1", r.get("track_id"), r.get("rule")))
    return ks


def load(path):
    try:
        st = os.stat(path)
    except OSError:
        return {}
    sig = (st.st_mtime_ns, st.st_size)
    with _lock:
        hit = _cache.get(path)
        if hit and hit[0] == sig:
            return hit[1]
        names = {}
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    if r.get("person"):
                        for k in _keys(r):
                            names[k] = r["person"]
        except OSError:
            return {}
        _cache[path] = (sig, names)
        return names


def fill(rec, names):
    """Give `rec` (an alert row) its late name when it has none. Returns rec."""
    if names and not rec.get("person"):
        for k in _keys(rec):
            if k in names:
                rec["person"] = names[k]
                break
    return rec
