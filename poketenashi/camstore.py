"""The cameras the monitor runs: monitor_cameras.json, managed from the dashboard.

config.yaml keeps the global settings (rules, models, alerts ...). The camera
list - name, source, whether the rules run on it, and each camera's own zones -
lives in monitor_cameras.json next to it, written only by the app: validated,
backed up to config_backups/ and replaced atomically on every change. (Not
"cameras.json": a camera inventory of that name, in another format, already sat
in this folder - it is left alone.)

On the first run the file is created from config.yaml (camera.source and
zones) as camera "cam1", so an existing install keeps working unchanged. cam1's
zones are also kept in config.yaml, which the replay/record tools read.

A source is an RTSP URL (usually with the camera's password in it), a webcam
index or a video file. It is never sent back to a browser in full - the
dashboard gets mask_url() of it.
"""
import copy
import json
import os
import re
import shutil
import threading
import time

from .camera import mask_url

ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,23}$")
NAME_MAX = 40
MAX_CAMERAS = 8
KEEP_BACKUPS = 30


def _clean_name(name):
    name = " ".join(str(name or "").split())[:NAME_MAX]
    return name


def _check_source(src):
    src = str(src or "").strip()
    if not src:
        return None, "the camera address (RTSP URL) is empty"
    if len(src) > 400 or any(c in src for c in "\r\n\t"):
        return None, "the camera address is not valid"
    low = src.lower()
    if not (low.startswith(("rtsp://", "rtsps://", "http://", "https://")) or src.isdigit()
            or os.path.isfile(src)):
        return None, "use an rtsp:// (or http://) camera address, a webcam number or a video file"
    return src, None


def _describe(src):
    """(kind, host, port, path) of a source, for the dashboard - never the login.
    kind: rtsp | http | webcam | file."""
    from urllib.parse import urlsplit
    s = str(src or "").strip()
    low = s.lower()
    if low.startswith(("rtsp://", "rtsps://", "http://", "https://")):
        try:
            u = urlsplit(s)
            port = u.port or (554 if low.startswith("rtsp") else (443 if low.startswith("https") else 80))
            path = (u.path or "/") + (("?" + u.query) if u.query else "")
            return ("rtsp" if low.startswith("rtsp") else "http"), (u.hostname or ""), port, path
        except ValueError:
            return "rtsp" if low.startswith("rtsp") else "http", "", None, ""
    if s.isdigit():
        return "webcam", "", None, s
    return "file", "", None, os.path.basename(s)


def _valid(c):
    return (isinstance(c, dict) and isinstance(c.get("id"), str) and ID_RE.match(c["id"])
            and isinstance(c.get("source"), str) and c["source"].strip())


class CameraStore:
    def __init__(self, path, cfg):
        self.path = path
        self.lock = threading.Lock()
        seed = [{
            "id": "cam1",
            "name": "Camera 1",
            "source": str((cfg.get("camera") or {}).get("source", "")),
            "enabled": True,
            "detect": True,
            "zones": copy.deepcopy(cfg.get("zones") or {}),
        }]
        if not os.path.exists(path):
            self.cams = seed
            self._write()
            print(f"[cameras] created {os.path.basename(path)} with the camera from config.yaml",
                  flush=True)
            return
        cams, why = [], None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            raw = data.get("cameras") if isinstance(data, dict) else None
            cams = [c for c in (raw or []) if _valid(c)]
            if raw and len(cams) < len(raw):
                why = f"{len(raw) - len(cams)} entr{'y' if len(raw) - len(cams) == 1 else 'ies'} not understood"
        except Exception as e:
            why = f"unreadable ({type(e).__name__})"
        if why:
            print(f"[cameras] {os.path.basename(path)}: {why} - "
                  + ("skipped" if cams else "running the camera from config.yaml"), flush=True)
        # A broken file must not stop the monitor: fall back to config.yaml's
        # camera. Nothing is written until the next change, and that change backs
        # the old file up first.
        self.cams = cams or seed

    # -- reading -------------------------------------------------------------
    def all(self):
        with self.lock:
            return copy.deepcopy(self.cams)

    def get(self, cid):
        with self.lock:
            for c in self.cams:
                if c["id"] == cid:
                    return copy.deepcopy(c)
        return None

    def public(self):
        """The list as the dashboard may see it: no passwords."""
        out = []
        for c in self.all():
            src = c.get("source", "")
            kind, host, port, path = _describe(src)
            out.append({"id": c["id"], "name": c.get("name") or c["id"],
                        "source": mask_url(src), "kind": kind, "host": host, "port": port,
                        "path": path, "has_login": kind == "rtsp" and "@" in src.split("/")[2]
                        if src.count("/") >= 2 else False,
                        "enabled": bool(c.get("enabled", True)),
                        "detect": bool(c.get("detect", True)),
                        "zones": sorted(k for k, v in (c.get("zones") or {}).items() if v),
                        # the rules this camera checks (None = every rule switched
                        # on in config.yaml - a new camera starts with all of them)
                        "rules_enabled": ((c.get("rules") or {}).get("enabled"))})
        return out

    # -- changing ------------------------------------------------------------
    def add(self, name, source, detect=True, enabled=True):
        src, err = _check_source(source)
        if err:
            return None, err
        with self.lock:
            if len(self.cams) >= MAX_CAMERAS:
                return None, f"at most {MAX_CAMERAS} cameras"
            n = 1
            ids = {c["id"] for c in self.cams}
            while f"cam{n}" in ids:
                n += 1
            cam = {"id": f"cam{n}", "name": _clean_name(name) or f"Camera {n}", "source": src,
                   "enabled": bool(enabled), "detect": bool(detect), "zones": {}}
            self.cams.append(cam)
            self._write()
            return copy.deepcopy(cam), None

    def update(self, cid, name=None, source=None, detect=None, enabled=None, rules_enabled=None):
        """Change a camera. `source` None or empty keeps the current one (the
        browser never had the full URL, so it cannot send it back).
        `rules_enabled`: the rule codes this camera checks (its own
        rules.enabled, laid over config.yaml's by web.py)."""
        with self.lock:
            cam = next((c for c in self.cams if c["id"] == cid), None)
            if cam is None:
                return None, "no such camera"
            if name is not None:
                cam["name"] = _clean_name(name) or cam["name"]
            if source:
                src, err = _check_source(source)
                if err:
                    return None, err
                cam["source"] = src
            if detect is not None:
                cam["detect"] = bool(detect)
            if enabled is not None:
                cam["enabled"] = bool(enabled)
            if rules_enabled is not None:
                cam.setdefault("rules", {})["enabled"] = [str(r) for r in rules_enabled]
            self._write()
            return copy.deepcopy(cam), None

    def remove(self, cid):
        with self.lock:
            if len(self.cams) <= 1:
                return "the last camera cannot be removed"
            before = len(self.cams)
            self.cams = [c for c in self.cams if c["id"] != cid]
            if len(self.cams) == before:
                return "no such camera"
            self._write()
            return None

    def set_zones(self, cid, zones):
        """Merge `zones` ({name: points}) into the camera's zones; returns them."""
        with self.lock:
            cam = next((c for c in self.cams if c["id"] == cid), None)
            if cam is None:
                return None
            merged = dict(cam.get("zones") or {})
            merged.update(zones)
            cam["zones"] = merged
            self._write()
            return copy.deepcopy(merged)

    def set_go_dir(self, cid, go_dir):
        """The way the painted feet on each STOP-LOOK-GO poster point (8 Oct):
        {checkpoint zone: [dx, dy] or None} -> the camera's rules.SHI.go_dir.
        Returns the camera, or None."""
        with self.lock:
            cam = next((c for c in self.cams if c["id"] == cid), None)
            if cam is None:
                return None
            shi = cam.setdefault("rules", {}).setdefault("SHI", {})
            gd = dict(shi.get("go_dir") or {})
            for k, v in go_dir.items():
                if v is None:
                    gd.pop(k, None)
                else:
                    gd[k] = [int(v[0]), int(v[1])]
            shi["go_dir"] = gd
            self._write()
            return copy.deepcopy(cam)

    # -- file ----------------------------------------------------------------
    def _write(self):
        if os.path.exists(self.path):
            d = os.path.join(os.path.dirname(os.path.abspath(self.path)), "config_backups")
            os.makedirs(d, exist_ok=True)
            shutil.copy2(self.path, os.path.join(d, time.strftime("monitor_cameras_%Y%m%d_%H%M%S.json")))
            olds = sorted(f for f in os.listdir(d)
                          if f.startswith("monitor_cameras_") and f.endswith(".json"))
            for f in olds[:-KEEP_BACKUPS]:
                try:
                    os.remove(os.path.join(d, f))
                except OSError:
                    pass
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"cameras": self.cams}, f, indent=1)
        os.replace(tmp, self.path)
