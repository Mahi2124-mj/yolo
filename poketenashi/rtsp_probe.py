"""Ask an RTSP camera which stream paths it serves - without opening a video.

Every brand puts its stream at a different path after host:port (Dahua
/cam/realmonitor..., Hikvision /Streaming/Channels/101, older OEM firmware
/h264/ch1/main/av_stream ...). A wrong path looks exactly like a wrong password
to OpenCV ("the stream did not open"), so this speaks RTSP itself: one DESCRIBE
per path, with the camera's Digest/Basic login, and reads the answer code:

    200  the stream exists           401  login refused (user / password)
    404  no stream at this path      other: what the camera said

The login is checked once, on the path the user gave, before any other path is
tried: a camera that locks an account after a few failed logins never sees more
than one. Credentials stay in this process - results carry paths only.
"""
import base64
import hashlib
import os
import re
import socket
import time
from urllib.parse import unquote, urlsplit, urlunsplit

# (label, path) - main stream first, then sub stream, per family.
CANDIDATE_PATHS = [
    ("Dahua / Amcrest / CP Plus - main", "/cam/realmonitor?channel=1&subtype=0"),
    ("Dahua / Amcrest / CP Plus - sub", "/cam/realmonitor?channel=1&subtype=1"),
    ("Hikvision - main", "/Streaming/Channels/101"),
    ("Hikvision - sub", "/Streaming/Channels/102"),
    ("Generic H.264 (older OEM) - main", "/h264/ch1/main/av_stream"),
    ("Generic H.264 (older OEM) - sub", "/h264/ch1/sub/av_stream"),
    ("Uniview - main", "/unicast/c1/s0/live"),
    ("Uniview - sub", "/unicast/c1/s1/live"),
    ("Panasonic", "/MediaInput/h264"),
    ("Axis", "/axis-media/media.amp"),
    ("Generic /live", "/live"),
    ("Generic /stream1", "/stream1"),
]


def _md5(s):
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _split(url):
    u = urlsplit(url)
    if u.scheme.lower() not in ("rtsp", "rtsps") or not u.hostname:
        raise ValueError("not an rtsp:// address")
    host, port = u.hostname, u.port or 554
    netloc = host + (f":{u.port}" if u.port else "")
    return u, host, port, netloc


def with_path(url, path):
    """`url` with its path (and query) replaced - login, host and port kept."""
    u = urlsplit(url)
    p, _, q = path.partition("?")
    return urlunsplit((u.scheme, u.netloc, p or "/", q, ""))


def _request(host, port, uri, cseq, auth, timeout):
    req = (f"DESCRIBE {uri} RTSP/1.0\r\nCSeq: {cseq}\r\nAccept: application/sdp\r\n"
           f"User-Agent: Poketenashi-probe\r\n")
    if auth:
        req += f"Authorization: {auth}\r\n"
    req += "\r\n"
    with socket.create_connection((host, port), timeout=timeout) as s:
        s.settimeout(timeout)
        s.sendall(req.encode("utf-8"))
        data = b""
        t0 = time.time()
        while b"\r\n\r\n" not in data and time.time() - t0 < timeout and len(data) < 65536:
            chunk = s.recv(4096)
            if not chunk:
                break
            data += chunk
    head = data.split(b"\r\n\r\n", 1)[0].decode("utf-8", "replace")
    m = re.match(r"RTSP/\d\.\d\s+(\d{3})\s*(.*)", head)
    if not m:
        raise ConnectionError("the device did not answer as an RTSP camera")
    return int(m.group(1)), m.group(2).strip(), head


def _auth_header(head, user, password, uri):
    m = re.search(r"^WWW-Authenticate:\s*Digest\s+(.+)$", head, re.I | re.M)
    if m:
        p = dict(re.findall(r'(\w+)\s*=\s*"?([^",]*)"?', m.group(1)))
        realm, nonce = p.get("realm", ""), p.get("nonce", "")
        ha1 = _md5(f"{user}:{realm}:{password}")
        ha2 = _md5(f"DESCRIBE:{uri}")
        parts = [f'username="{user}"', f'realm="{realm}"', f'nonce="{nonce}"', f'uri="{uri}"']
        qop = p.get("qop", "")
        if "auth" in [q.strip() for q in qop.split(",")]:
            nc, cnonce = "00000001", os.urandom(8).hex()
            parts += [f'response="{_md5(f"{ha1}:{nonce}:{nc}:{cnonce}:auth:{ha2}")}"',
                      "qop=auth", f"nc={nc}", f'cnonce="{cnonce}"']
        else:
            parts.append(f'response="{_md5(f"{ha1}:{nonce}:{ha2}")}"')
        if p.get("opaque"):
            parts.append(f'opaque="{p["opaque"]}"')
        return "Digest " + ", ".join(parts)
    if re.search(r"^WWW-Authenticate:\s*Basic", head, re.I | re.M):
        return "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()
    return None


def describe(url, timeout=4.0):
    """(code, reason) of an RTSP DESCRIBE of `url`, logging in if asked to."""
    u, host, port, netloc = _split(url)
    uri = urlunsplit((u.scheme, netloc, u.path or "/", u.query, ""))
    code, reason, head = _request(host, port, uri, 1, None, timeout)
    if code == 401 and (u.username or u.password):
        auth = _auth_header(head, unquote(u.username or ""), unquote(u.password or ""), uri)
        if auth:
            # A new connection: some cameras hang up after the 401.
            code, reason, head = _request(host, port, uri, 2, auth, timeout)
    return code, reason


def explain(code, reason=""):
    if code == 200:
        return "stream found"
    if code == 401:
        return "login refused - check the user name and password"
    if code in (404, 451):           # 451 "Parameter Not Understood": some firmware's 404
        return "no stream at this path on this camera"
    if code == 403:
        return "the camera forbids this user"
    if code == 454:
        return "the camera does not know this session/path"
    return f"camera answered {code} {reason}".strip()


def find_paths(url, timeout=4.0):
    """Which known paths this camera serves, with the login in `url`.

    {"login": "ok"|"refused"|"none", "given": {...}, "found": [{label, path}],
     "tried": [{label, path, code, why}], "error": str|None}
    """
    out = {"login": None, "given": None, "found": [], "tried": [], "error": None}
    try:
        u, host, port, netloc = _split(url)
    except ValueError as e:
        out["error"] = str(e)
        return out
    given_path = (u.path or "/") + (("?" + u.query) if u.query else "")
    try:
        code, reason = describe(url, timeout)
    except ConnectionRefusedError:
        out["error"] = (f"{host} answers, but nothing listens on port {port} - check the RTSP port "
                        f"(usually 554).")
        return out
    except ConnectionError as e:          # answered, but not as RTSP
        out["error"] = f"{host}:{port} answers, but not as a camera stream ({e})."
        return out
    except OSError:
        out["error"] = (f"Nothing answers at {host}:{port} - check the IP address, that the camera "
                        f"has power, and that it is on this PC's network.")
        return out
    out["given"] = {"path": given_path, "code": code, "why": explain(code, reason)}
    if code == 401:
        out["login"] = "refused"
        return out                           # do not try more paths with a bad login
    out["login"] = "ok" if (u.username or u.password) else "none"
    if code == 200:
        out["found"].append({"label": "the address you gave", "path": given_path})
    for label, path in CANDIDATE_PATHS:
        if path == given_path:
            continue
        try:
            c, r = describe(with_path(url, path), timeout)
        except (OSError, ConnectionError) as e:
            c, r = 0, type(e).__name__
        out["tried"].append({"label": label, "path": path, "code": c, "why": explain(c, r)})
        if c == 200:
            out["found"].append({"label": label, "path": path})
        elif c == 401:
            break                            # the login stopped working: stop here
    return out
