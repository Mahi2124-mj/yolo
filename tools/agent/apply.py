"""Apply / roll back a rules.PO parameter change - ONLY after guard.py said PASS.
  python tools/agent/apply.py apply '<json overrides>' "<reason>"
  python tools/agent/apply.py rollback          (restores the last backup)
Changes only keys under rules.PO in config.yaml (text edit of those lines, comments
kept), keeps a backup in _backup/agent/<stamp>/config.yaml, writes the audit log and
agent/changes.jsonl, then asks for a clean restart (restart.request)."""
import datetime as dt, glob, json, os, re, shutil, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CFG = os.path.join(ROOT, "config.yaml")
BAK = os.path.join(ROOT, "_backup", "agent")
LOG = "D:/poketenashi-data/agent/changes.jsonl"
ALLOWED = {"evidence_hi", "evidence_lo", "evidence_sustain_sec", "evidence_unknown_max_sec", "evidence_gap_sec",
           "evidence_min_forearm_drop", "evidence_release_out_sec", "evidence_latch_max_sec", "evidence_ir_max_sat",
           "hold_sec", "evidence_min_torso"}
BOUNDS = {"evidence_hi": (0.6, 0.95), "evidence_lo": (0.1, 0.5), "evidence_sustain_sec": (0.5, 2.0),
          "evidence_unknown_max_sec": (1.0, 5.0), "evidence_gap_sec": (0.5, 3.0), "evidence_min_forearm_drop": (0.0, 0.4),
          "evidence_release_out_sec": (0.5, 2.0), "evidence_latch_max_sec": (2.0, 8.0), "evidence_ir_max_sat": (15, 45),
          "hold_sec": (0.5, 3.0), "evidence_min_torso": (50, 110)}


def audit(detail):
    with open(os.path.join(ROOT, "logs", "audit.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "user": "agent", "ip": "127.0.0.1",
                            "action": "settings", "detail": detail}) + "\n")


def apply(over, reason):
    bad = [k for k in over if k not in ALLOWED or not (BOUNDS[k][0] <= float(over[k]) <= BOUNDS[k][1])]
    if bad:
        sys.exit(f"refused: not allowed or out of bounds: {bad}")
    s = open(CFG, encoding="utf-8", newline="").read()
    a = s.index("\n  PO:")
    b = a + 1 + re.search(r"\n  [A-Za-z]", s[a + 1:]).start()        # next rule block
    block = s[a:b]
    for k, v in over.items():
        new, n = re.subn(rf"(\n    {k}:\s*)([^\s#]+)", lambda m: m.group(1) + str(v), block, count=1)
        if n == 0:
            new = block.rstrip("\r\n") + f"\n    {k}: {v}\n"
        block = new
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.join(BAK, stamp), exist_ok=True)
    shutil.copy2(CFG, os.path.join(BAK, stamp, "config.yaml"))
    open(CFG, "w", encoding="utf-8", newline="").write(s[:a] + block + s[b:])
    import yaml
    po = yaml.safe_load(open(CFG, encoding="utf-8"))["rules"]["PO"]
    assert all(str(po[k]) == str(v) for k, v in over.items()), "config check failed"
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps({"time": stamp, "change": over, "reason": reason, "backup": os.path.join(BAK, stamp)}) + "\n")
    audit(f"review agent: PO {over} ({reason}); backup _backup/agent/{stamp}")
    open(os.path.join(ROOT, "restart.request"), "w").write("review agent " + stamp)
    print("applied", over, "backup", stamp)


def rollback():
    last = sorted(glob.glob(os.path.join(BAK, "*")))
    if not last:
        sys.exit("no backup")
    shutil.copy2(os.path.join(last[-1], "config.yaml"), CFG)
    audit(f"review agent: rolled back to _backup/agent/{os.path.basename(last[-1])}")
    open(os.path.join(ROOT, "restart.request"), "w").write("review agent rollback")
    shutil.move(last[-1], last[-1] + "_rolled_back")
    print("rolled back to", last[-1])


if __name__ == "__main__":
    if sys.argv[1] == "apply":
        apply(json.loads(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "")
    elif sys.argv[1] == "rollback":
        rollback()
