"""Guard test for a PO parameter change (the nightly review agent, 8 Oct 2026).

A change may go live ONLY if, compared with the live settings, on every labelled set:
  * no real PO that live catches is lost (t25, lt, dx, dx6, live6, live7 "real"), and
  * false alerts do not go up anywhere, and go down somewhere.
Usage:
  python tools/agent/guard.py '<json of rules.PO overrides>'   -> PASS / FAIL + table
Exit code 0 = PASS, 1 = FAIL, 3 = not run (not enough memory).
Runs only when the PC has >= 2.5 GB free commit (the live monitor comes first)."""
import ctypes, json, os, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
SETS = "t25,lt,dx,dx6,live6,live7,raw"


class MS(ctypes.Structure):
    _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [(k, ctypes.c_ulonglong) for k in ("tp", "ap", "tpf", "apf", "tv", "av", "ae")]


def free_gb():
    m = MS(); m.l = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.apf / 2 ** 30


def run(name, po):
    var = {"name": name, "po": po}
    d = tempfile.mkdtemp(prefix="guard_")
    vp, rp = os.path.join(d, "var.json"), os.path.join(d, "res.json")
    json.dump(var, open(vp, "w"))
    r = subprocess.run([PY, os.path.join(HERE, "eval_po.py"), vp, rp, SETS], cwd=ROOT, capture_output=True, text=True, errors="replace")
    if not os.path.exists(rp):
        raise RuntimeError("replay failed: " + (r.stderr or "")[-400:])
    S = json.load(open(rp))["sets"]
    out = {}
    for s, v in S.items():
        out[s] = {"real_hit": sorted(e["name"] for e in v["real"] if e.get("hit") is not None),
                  "real_n": len(v["real"]), "false": len(v["false"])}
    return out


def main():
    po = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    if free_gb() < 2.5:
        print(f"NOT RUN: only {free_gb():.1f} GB free commit (need 2.5) - the live monitor comes first")
        sys.exit(3)
    base, cand = run("live", {}), run("candidate", po)
    ok, better = True, False
    print(f"{'set':6s} {'real live':>10s} {'real new':>9s} {'false live':>11s} {'false new':>10s}")
    for s in base:
        b, c = base[s], cand[s]
        lost = sorted(set(b["real_hit"]) - set(c["real_hit"]))
        print(f"{s:6s} {len(b['real_hit']):>4d}/{b['real_n']:<5d} {len(c['real_hit']):>4d}/{c['real_n']:<4d} {b['false']:>11d} {c['false']:>10d}"
              + (f"   LOST: {lost}" if lost else ""))
        if lost or c["false"] > b["false"]:
            ok = False
        if c["false"] < b["false"] or len(c["real_hit"]) > len(b["real_hit"]):
            better = True
    verdict = ok and better
    print("PASS" if verdict else "FAIL" + ("" if ok else " (loses a real PO or adds a false alert)") + ("" if better else " (no improvement)"))
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    main()
