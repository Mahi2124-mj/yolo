"""Dashboard users: who may change cameras, zones and settings.

users.json holds name, role, the cameras they may see and a salted
PBKDF2-SHA256 hash per user - never a password. Roles:
    admin   everything: cameras, zones, settings, users; every camera
    viewer  looks only, and only at their cameras ("cams": ids, or ["*"] = all)

Since 7 Oct 2026 looking needs a login too (the user: each person sees the
cameras given to them), or a display link (Displays: a TV screen that shows
its cameras without anyone logging in on it).

Logins are random session tokens in an HttpOnly cookie, kept in sessions.json
next to users.json - only their SHA-256, never the token - so a restart no
longer logs everyone out. Until the first user exists the monitor PC itself
(localhost) may create an admin; nothing on the network can.
"""
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time

ITERATIONS = 200_000
SESSION_SEC = 7 * 24 * 3600        # a login lasts a week of not being used
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{1,31}$")
ROLES = ("admin", "viewer")
MIN_PASSWORD = 6
LOCK_AFTER = 5                 # failed logins from one address ...
LOCK_SEC = 120                 # ... lock it out this long


def _hash(password, salt, iterations=ITERATIONS):
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt),
                               iterations).hex()


class Users:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.users = []
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    self.users = json.load(f).get("users") or []
            except Exception as e:
                print(f"[auth] could not read {path}: {e}", flush=True)
        self._fails = {}           # ip -> [count, first_fail_time]

    def any(self):
        with self.lock:
            return bool(self.users)

    def list(self):
        with self.lock:
            return [{"name": u["name"], "role": u["role"], "created": u.get("created"),
                     "cams": list(u.get("cams") or ["*"])} for u in self.users]

    def cams_of(self, name):
        """The camera ids `name` may see: None = all of them (an admin, or
        ["*"]), else the list ([] = none)."""
        with self.lock:
            u = self._find(name)
            if u is None:
                return []
            if u["role"] == "admin":
                return None
            cams = list(u.get("cams") or ["*"])
            return None if "*" in cams else cams

    def _find(self, name):
        return next((u for u in self.users if u["name"].lower() == str(name).lower()), None)

    # -- login ---------------------------------------------------------------
    def locked(self, ip):
        f = self._fails.get(ip)
        return bool(f and f[0] >= LOCK_AFTER and time.time() - f[1] < LOCK_SEC)

    def verify(self, name, password, ip):
        """The user's (name, role), or None. Counts failures per address."""
        if self.locked(ip):
            return None
        with self.lock:
            u = self._find(name)
            ok = False
            if u is not None:
                ok = hmac.compare_digest(_hash(str(password), u["salt"], u.get("iter", ITERATIONS)),
                                         u["hash"])
            else:
                _hash(str(password), "00" * 16)          # same time whether or not the name exists
        if ok:
            self._fails.pop(ip, None)
            return u["name"], u["role"]
        f = self._fails.get(ip)
        if not f or time.time() - f[1] >= LOCK_SEC:
            f = [0, time.time()]
        f[0] += 1
        self._fails[ip] = f
        return None

    # -- managing ------------------------------------------------------------
    @staticmethod
    def _check(name, password, role):
        if not NAME_RE.match(str(name or "")):
            return "user name: 2-32 letters, digits, '.', '_' or '-'"
        if password is not None and len(str(password)) < MIN_PASSWORD:
            return f"password: at least {MIN_PASSWORD} characters"
        if role is not None and role not in ROLES:
            return "role must be admin or viewer"
        return None

    @staticmethod
    def _cams(cams):
        """A clean camera list, None (not given) or False (bad)."""
        if cams is None:
            return None
        if not isinstance(cams, list) or not all(isinstance(c, str) and re.match(r"^[A-Za-z0-9_*-]{1,32}$", c)
                                                 for c in cams):
            return False
        return ["*"] if "*" in cams else sorted(set(cams))

    def add(self, name, password, role, cams=None):
        err = self._check(name, password, role)
        if err:
            return err
        cams = self._cams(cams if cams is not None else ["*"])
        if cams is False:
            return "bad camera list"
        with self.lock:
            if self._find(name):
                return "that user already exists"
            salt = secrets.token_hex(16)
            self.users.append({"name": name, "role": role, "salt": salt, "iter": ITERATIONS,
                               "hash": _hash(str(password), salt), "cams": cams,
                               "created": time.strftime("%Y-%m-%d %H:%M")})
            self._write()
        return None

    def update(self, name, password=None, role=None, cams=None):
        err = self._check(name, password, role)
        if err:
            return err
        cams = self._cams(cams)
        if cams is False:
            return "bad camera list"
        with self.lock:
            u = self._find(name)
            if u is None:
                return "no such user"
            if role is not None and role != "admin" and u["role"] == "admin" \
                    and sum(1 for x in self.users if x["role"] == "admin") <= 1:
                return "the last admin cannot be made a viewer"
            if password is not None:
                u["salt"] = secrets.token_hex(16)
                u["iter"] = ITERATIONS
                u["hash"] = _hash(str(password), u["salt"])
            if role is not None:
                u["role"] = role
            if cams is not None:
                u["cams"] = cams
            self._write()
        return None

    def remove(self, name):
        with self.lock:
            u = self._find(name)
            if u is None:
                return "no such user"
            if u["role"] == "admin" and sum(1 for x in self.users if x["role"] == "admin") <= 1:
                return "the last admin cannot be removed"
            self.users = [x for x in self.users if x is not u]
            self._write()
        return None

    def _write(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"users": self.users}, f, indent=1)
        os.replace(tmp, self.path)


def _th(tok):
    return hashlib.sha256(str(tok).encode("utf-8")).hexdigest()


class Sessions:
    """Logins, kept across restarts in `path` (sessions.json) by their SHA-256:
    the file never holds a usable token. A session lasts SESSION_SEC since it
    was last used. Without a path (tests) they live in memory only."""

    def __init__(self, path=None):
        self.path = path
        self._s = {}                       # sha256(token) -> [name, role, expires]
        self.lock = threading.Lock()
        self._saved = 0.0
        if path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    now = time.time()
                    self._s = {k: list(v) for k, v in (json.load(f).get("sessions") or {}).items()
                               if isinstance(v, list) and len(v) == 3 and v[2] > now}
            except Exception as e:
                print(f"[auth] could not read {path}: {e} - everyone logs in again", flush=True)

    def _write(self, force=True):
        if not self.path or (not force and time.time() - self._saved < 300):
            return
        self._saved = time.time()
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"sessions": self._s}, f)
            os.replace(tmp, self.path)
        except Exception as e:
            print(f"[auth] sessions not saved: {e}", flush=True)

    def new(self, name, role):
        tok = secrets.token_urlsafe(32)
        with self.lock:
            now = time.time()
            for k in [k for k, v in self._s.items() if v[2] < now]:
                del self._s[k]
            self._s[_th(tok)] = [name, role, now + SESSION_SEC]
            self._write()
        return tok

    def get(self, tok):
        if not tok:
            return None
        with self.lock:
            k = _th(tok)
            v = self._s.get(k)
            if v is None or v[2] < time.time():
                if self._s.pop(k, None) is not None:
                    self._write()
                return None
            v[2] = time.time() + SESSION_SEC           # used: a week from now
            self._write(force=False)
            return v[0], v[1]

    def drop(self, tok):
        with self.lock:
            if self._s.pop(_th(tok), None) is not None:
                self._write()

    def drop_user(self, name, keep=None):
        """Log a user out everywhere - removed, demoted, or their password
        changed - except the session `keep` (the one that changed it)."""
        with self.lock:
            kk = _th(keep) if keep else None
            for k in [k for k, v in self._s.items()
                      if v[0].lower() == str(name).lower() and k != kk]:
                del self._s[k]
            self._write()


class Displays:
    """Screens that show cameras without anyone logging in on them - the TV in
    the Toolroom shows the Toolroom camera, the one at Dx shows Dx (user, 7
    Oct). displays.json: [{id, name, cams, token, created}]. The token is the
    secret part of the screen's link (/tv?d=<token>), which sets a cookie on
    that screen; a new link (regenerate) locks the old one out."""

    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.items = []
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    self.items = json.load(f).get("displays") or []
            except Exception as e:
                print(f"[auth] could not read {path}: {e}", flush=True)

    def _write(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"displays": self.items}, f, indent=1)
        os.replace(tmp, self.path)

    def list(self):
        with self.lock:
            return [dict(d) for d in self.items]

    def by_token(self, tok):
        if not tok:
            return None
        with self.lock:
            for d in self.items:
                if hmac.compare_digest(str(d.get("token", "")), str(tok)):
                    return dict(d)
        return None

    def add(self, name, cams):
        name = " ".join(str(name or "").split())[:40]
        if not name:
            return None, "give the screen a name"
        cams = Users._cams(cams)
        if not cams:
            return None, "pick at least one camera"
        with self.lock:
            n = 1
            while any(d["id"] == f"tv{n}" for d in self.items):
                n += 1
            d = {"id": f"tv{n}", "name": name, "cams": cams, "token": secrets.token_urlsafe(18),
                 "created": time.strftime("%Y-%m-%d %H:%M")}
            self.items.append(d)
            self._write()
        return dict(d), None

    def update(self, did, name=None, cams=None, regenerate=False):
        with self.lock:
            d = next((x for x in self.items if x["id"] == did), None)
            if d is None:
                return None, "no such screen"
            if name is not None:
                name = " ".join(str(name).split())[:40]
                if not name:
                    return None, "give the screen a name"
                d["name"] = name
            if cams is not None:
                cams = Users._cams(cams)
                if not cams:
                    return None, "pick at least one camera"
                d["cams"] = cams
            if regenerate:
                d["token"] = secrets.token_urlsafe(18)
            self._write()
            return dict(d), None

    def remove(self, did):
        with self.lock:
            n = len(self.items)
            self.items = [x for x in self.items if x["id"] != did]
            if len(self.items) == n:
                return "no such screen"
            self._write()
        return None
