"""Deploy / roll back a PO model for the nightly train agent (10 Oct 2026).

  po_deploy.py deploy <candidate.pt> "<reason>"   copy to models/po_arm_auto_<stamp>.pt,
                                                  set rules.PO.model_path, restart
  po_deploy.py rollback                           the model_path before the last deploy

Only rules.PO.model_path changes (one line of config.yaml); a backup of config.yaml
goes to _backup/agent_model/<stamp>/ and every change to
D:/poketenashi-data/agent/model_changes.jsonl. The monitor restarts through
restart.request (watchdog). The master cache rebuilds by itself for the new model."""
import datetime as dt
import json
import os
import re
import shutil
import sys

ROOT = r"C:\Users\DX-ADMIN\Pictures\yolo"
LOG = r"D:\poketenashi-data\agent\model_changes.jsonl"


def _set_model(path_rel, note):
    p = os.path.join(ROOT, "config.yaml")
    s = open(p, encoding="utf-8", newline="").read()
    m = re.search(r"^(    model_path: )(\S+)(.*)$", s, re.M)
    if not m:
        sys.exit("model_path line not found")
    old = m.group(2)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    bdir = os.path.join(ROOT, "_backup", "agent_model", stamp)
    os.makedirs(bdir, exist_ok=True)
    shutil.copy2(p, os.path.join(bdir, "config.yaml"))
    s = s[:m.start()] + f"{m.group(1)}{path_rel}   # {note}" + s[m.end():]
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(s)
    import yaml
    assert yaml.safe_load(open(p, encoding="utf-8"))["rules"]["PO"]["model_path"] == path_rel
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": dt.datetime.now().isoformat(timespec="seconds"), "old": old, "new": path_rel,
                            "note": note, "backup": bdir}) + "\n")
    open(os.path.join(ROOT, "restart.request"), "w").close()
    print(f"model_path {old} -> {path_rel}; restart requested; backup {bdir}")


def main():
    if sys.argv[1] == "deploy":
        cand, reason = sys.argv[2], sys.argv[3]
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
        rel = f"models/po_arm_auto_{stamp}.pt"
        shutil.copy2(cand, os.path.join(ROOT, rel))
        _set_model(rel, f"train agent {stamp}: {reason}")
    elif sys.argv[1] == "rollback":
        last = [json.loads(l) for l in open(LOG, encoding="utf-8")][-1]
        _set_model(last["old"], f"train agent rollback of {last['new']}")


if __name__ == "__main__":
    main()
