"""Save zone polygons into config.yaml without touching anything else.

config.yaml is more comment than value (each threshold records why it is what
it is), so it is never round-tripped through a YAML dump - that drops every
comment. Each zone is ONE flow-style line under `zones:`; saving rewrites
exactly those lines (the approach of tools/zone_drawer.py), after copying the
file to config_backups/, and only if the result still parses.
"""
import io
import os
import re
import shutil
import time

import yaml

NAME_RE = re.compile(r"^[a-z_][a-z0-9_]{0,31}$")
KEEP_BACKUPS = 30
MAX_ZONES = 16
MAX_POINTS = 60

# What each zone is for - shown in the zone editor.
HELP = {
    "gangway": "The aisle. PO, KE, RUN and FALL judge only people whose feet are inside "
               "it, and NA measures crossing angles on it: draw its 4 corners (far-left, "
               "far-right, near-right, near-left).",
    "crossing": "T-junction: the side gangways. Walking left/right here, or turning into "
                "the aisle from here, is not NA.",
    "stairs": "Stairs. TE (hold the handrail) needs this and 'handrail'.",
    "handrail": "The handrail along the stairs (TE).",
    "checkpoint": "SHI: the spot where people must stop and look around.",
    "restricted": "Restricted area: anyone inside is flagged.",
    "walkway_center": "Old NA zone (walking in the centre). Use 'gangway' instead.",
}


def validate(zones, w, h):
    """({name: [[x, y], ...]}, None) cleaned and clamped to the frame, or
    (None, reason). An empty list switches a zone off."""
    if not isinstance(zones, dict) or not zones:
        return None, "no zones given"
    if len(zones) > MAX_ZONES:
        return None, "too many zones"
    out = {}
    for name, pts in zones.items():
        if not isinstance(name, str) or not NAME_RE.match(name):
            return None, f"bad zone name {name!r} - use lowercase letters, digits and _"
        pts = pts or []
        if not isinstance(pts, list) or len(pts) > MAX_POINTS:
            return None, f"{name}: bad point list"
        if 0 < len(pts) < 3:
            return None, f"{name}: a zone needs at least 3 points (or none, to switch it off)"
        clean = []
        for p in pts:
            if not isinstance(p, (list, tuple)) or len(p) != 2:
                return None, f"{name}: bad point {p!r}"
            try:
                x, y = float(p[0]), float(p[1])
            except (TypeError, ValueError):
                return None, f"{name}: bad point {p!r}"
            clean.append([int(round(min(max(x, 0.0), w))), int(round(min(max(y, 0.0), h)))])
        out[name] = clean
    return out, None


def backup(path):
    """Copy config.yaml to config_backups/config_<time>.yaml; keep the newest ones."""
    d = os.path.join(os.path.dirname(os.path.abspath(path)), "config_backups")
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, time.strftime("config_%Y%m%d_%H%M%S.yaml"))
    shutil.copy2(path, dst)
    olds = sorted(f for f in os.listdir(d) if f.startswith("config_") and f.endswith(".yaml"))
    for f in olds[:-KEEP_BACKUPS]:
        try:
            os.remove(os.path.join(d, f))
        except OSError:
            pass
    return dst


def save_zones(path, zones):
    """Rewrite (or add) one `  name: [[x, y], ...]` line per zone under `zones:`.
    Everything else in the file stays byte for byte."""
    raw = io.open(path, encoding="utf-8", newline="").read()
    nl = "\r\n" if "\r\n" in raw else "\n"
    lines = [l.rstrip("\r") for l in raw.split(nl)]
    z0 = next(i for i, l in enumerate(lines) if l.startswith("zones:"))
    end = z0 + 1                   # the block: indented, blank or comment lines
    while end < len(lines) and (lines[end].startswith("  ") or not lines[end].strip()
                                or lines[end].lstrip().startswith("#")):
        end += 1
    while end > z0 + 1 and not lines[end - 1].strip():
        end -= 1                   # a blank line before the next section is not ours
    for name, pts in zones.items():
        flow = "  %s: %s" % (name, [[int(x), int(y)] for x, y in pts])
        idx = next((i for i in range(z0 + 1, end)
                    if re.match(r"^  %s:" % re.escape(name), lines[i])), None)
        if idx is None:
            lines.insert(end, flow)
            end += 1
            continue
        j = idx + 1                # a value written as a block list, if any
        while j < end and lines[j].lstrip().startswith("- "):
            j += 1
        comment = "   " + lines[idx][lines[idx].index("#"):] if "#" in lines[idx] else ""
        lines[idx:j] = [flow + comment]
        end -= j - idx - 1
    text = nl.join(lines)
    parsed = (yaml.safe_load(text) or {}).get("zones") or {}
    for name, pts in zones.items():            # never write a file that reads back wrong
        if [list(map(int, p)) for p in (parsed.get(name) or [])] != [list(p) for p in pts]:
            raise ValueError(f"zone {name} did not read back as written - not saved")
    tmp = path + ".tmp"
    io.open(tmp, "w", encoding="utf-8", newline="").write(text)
    os.replace(tmp, path)
