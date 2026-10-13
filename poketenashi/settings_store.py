"""Change single settings in config.yaml from the dashboard, keeping the file.

Like zone_store: config.yaml is never round-tripped through a YAML dump (that
would drop every comment). set_values() rewrites only the `key: value` lines it
is asked to, inside their section block - top level ("retention") or nested
("rules.PO", two spaces deeper per level) - keeps any trailing comment, backs
the file up first and writes it only if it reads back with the new values.
"""
import io
import os
import re

import yaml

from .zone_store import backup


def _fmt(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _indent(line):
    return len(line) - len(line.lstrip(" "))


def _block(lines, keys):
    """(first, end, indent) of the lines inside the block `keys` (e.g. ["rules",
    "PO"]): first..end-1 are its children, indented `indent` spaces. None if the
    block is missing."""
    lo, hi, ind = 0, len(lines), 0
    for k in keys:
        pat = re.compile(r"^%s%s:\s*(#.*)?$" % (" " * ind, re.escape(k)))
        i = next((i for i in range(lo, hi) if pat.match(lines[i])), None)
        if i is None:
            return None
        j = i + 1
        while j < hi and (not lines[j].strip() or lines[j].lstrip().startswith("#")
                          or _indent(lines[j]) > ind):
            j += 1
        while j > i + 1 and (not lines[j - 1].strip() or lines[j - 1].lstrip().startswith("#")) \
                and _indent(lines[j - 1]) <= ind:
            j -= 1                                   # comments/blanks after the block
        lo, hi, ind = i + 1, j, ind + 2
    return lo, hi, ind


def set_values(path, section, values):
    """values: {key: scalar} for `section` ("retention" or "rules.PO").
    Returns the backup file name."""
    raw = io.open(path, encoding="utf-8", newline="").read()
    nl = "\r\n" if "\r\n" in raw else "\n"
    lines = [l.rstrip("\r") for l in raw.split(nl)]
    keys = section.split(".")
    blk = _block(lines, keys)
    if blk is None:
        if len(keys) > 1:
            raise ValueError(f"section {section} not found in {os.path.basename(path)}")
        lines += ["", f"{section}:"]
        blk = (len(lines), len(lines), 2)
    first, end, ind = blk
    pad = " " * ind
    for key, val in values.items():
        idx = next((i for i in range(first, end)
                    if re.match(r"^%s%s:" % (pad, re.escape(key)), lines[i])), None)
        new = f"{pad}{key}: {_fmt(val)}"
        if idx is None:
            lines.insert(end, new)
            end += 1
        else:
            m = re.search(r"\s+#.*$", lines[idx])
            lines[idx] = new + (m.group(0) if m else "")
    text = nl.join(lines)
    parsed = yaml.safe_load(text) or {}
    for k in keys:
        parsed = (parsed or {}).get(k) or {}
    for key, val in values.items():
        if parsed.get(key) != val and not (isinstance(val, float) and parsed.get(key) == int(val)):
            raise ValueError(f"{section}.{key} did not read back as written - not saved")
    bak = backup(path)
    tmp = path + ".tmp"
    io.open(tmp, "w", encoding="utf-8", newline="").write(text)
    os.replace(tmp, path)
    return os.path.basename(bak)


def set_camera_source(path, source):
    """Keep config.yaml's camera.source in step with cam1 (the record/replay
    tools read it). Only the active `  source:` line under `camera:` changes."""
    raw = io.open(path, encoding="utf-8", newline="").read()
    nl = "\r\n" if "\r\n" in raw else "\n"
    lines = [l.rstrip("\r") for l in raw.split(nl)]
    c0 = next((i for i, l in enumerate(lines) if l.startswith("camera:")), None)
    if c0 is None:
        return None
    i = c0 + 1
    while i < len(lines) and (lines[i].startswith("  ") or not lines[i].strip()):
        if re.match(r"^  source:", lines[i]):
            lines[i] = "  source: " + yaml.safe_dump(source, default_style='"').strip()
            text = nl.join(lines)
            if (yaml.safe_load(text) or {}).get("camera", {}).get("source") != source:
                raise ValueError("camera.source did not read back as written - not saved")
            bak = backup(path)
            tmp = path + ".tmp"
            io.open(tmp, "w", encoding="utf-8", newline="").write(text)
            os.replace(tmp, path)
            return os.path.basename(bak)
        i += 1
    return None
