"""PO model training for the nightly train agent (10 Oct 2026, the user: "an agent that
monitors and trains the model").

  po_train.py arms   <out_dir>            arm crops from the user's labels
  po_train.py train  <out_dir> <out.pt>   candidate model (v1 fine-tuned)
  po_train.py heldout <out_dir> <cand.pt> held-out check: live pair vs candidate pair

Labels = ONLY the user's folders PO_review_v4/OK_PO_real (PO) and NG_PO_false (not PO).
OK picture: the hand at the hip the models rate most pocket-like is the positive;
NG picture: every hand at the hip is a negative. Files the live model was NOT trained on
(not in MANIFEST) are HELD OUT and are the test; after a deploy they are added
(po_train.py commit <out_dir>).
Live decision = min(main model, v1) per hand (rules.PO.model2_path) - the check
compares the live pair with (candidate, v1). Colour-blind training (po_train_lib).
Runs at below-normal priority, 2 threads."""
import json
import os
import sys
import time

import numpy as np

ROOT = r"C:\Users\DX-ADMIN\Pictures\yolo"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lowprio  # noqa: E402,F401

MANIFEST = "D:/poketenashi-data/agent/train/trained_files.json"   # files the live model was trained on


def _manifest():
    try:
        return set(json.load(open(MANIFEST, encoding="utf-8")))
    except Exception:
        return set()
EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".mp4")


def _models():
    import yaml
    from poketenashi import po_model
    c = yaml.safe_load(open(os.path.join(ROOT, "config.yaml"), encoding="utf-8"))["rules"]["PO"]
    ab = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)
    main = po_model.PocketModel(ab(c["model_path"]))
    v1 = po_model.PocketModel(ab(c["model2_path"])) if c.get("model2_path") else None   # live pair or single model
    return main, v1, float(c.get("evidence_hi", 0.35))


def arms(out):
    import glob
    import cv2
    import torch
    torch.set_num_threads(2)
    from ultralytics import YOLO
    from poketenashi import po_model
    from poketenashi.po_refs import _read_images
    main, v1, _ = _models()
    pose = YOLO(os.path.join(ROOT, "yolo11n-pose.pt"))
    os.makedirs(os.path.join(out, "img"), exist_ok=True)
    idx = open(os.path.join(out, "index.jsonl"), "w", encoding="utf-8")
    seen, n = _manifest(), {"pocket": 0, "out": 0}
    # 13 Oct: PO_OK_DIR = the cleaned OK folder (only the red-boxed person, hand in pocket)
    for lab, d in (("pocket", os.environ.get("PO_OK_DIR", "PO_review_v4/OK_PO_real")),
                   ("out", os.environ.get("PO_NG_DIR", "PO_review_v4/NG_PO_false"))):
        for f in sorted(glob.glob(os.path.join(ROOT, d, "**", "*"), recursive=True)):
            if not f.lower().endswith(EXTS):
                continue
            test = os.path.basename(f) not in seen
            for k, img in enumerate(_read_images(f)):
                if max(img.shape[:2]) < 120:
                    continue
                r = pose.predict(img, imgsz=320 if max(img.shape[:2]) <= 700 else 960, conf=0.3, verbose=False, device="cpu")[0]
                if r.keypoints is None or len(r.boxes) == 0:
                    continue
                b = r.boxes.xyxy.numpy()
                kp = r.keypoints.data.numpy()[int(np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])))]
                if min(kp[[5, 6, 11, 12], 2]) < 0.3:
                    continue
                torso = float(np.linalg.norm(kp[[5, 6], :2].mean(0) - kp[[11, 12], :2].mean(0)))
                if torso < 25:
                    continue
                cs = []
                for side, wi, hi in (("l", 9, 11), ("r", 10, 12)):
                    W, H = kp[wi, :2], kp[hi, :2]
                    if kp[wi, 2] < .3 or abs(W[1] - H[1]) > .5 * torso or abs(W[0] - H[0]) > .6 * torso:
                        continue
                    c = po_model.arm_crop(img, W, torso)
                    if c is not None:
                        cs.append((side, c, max(main.score([c])[0], v1.score([c])[0] if v1 else 0.0)))
                if not cs:
                    continue
                if lab == "pocket":
                    cs = [max(cs, key=lambda x: x[2])]
                    if cs[0][2] < 0.05:
                        continue
                for side, c, p in cs:
                    i = f"a{sum(n.values())}_{side}"
                    cv2.imwrite(os.path.join(out, "img", i + ".jpg"), c)
                    idx.write(json.dumps({"id": i, "file": os.path.basename(f), "label": lab, "test": test, "p": round(p, 3)}) + "\n")
                    n[lab] += 1
    print("arms", n, flush=True)


def _user(out, test):
    import cv2
    ims, ys, names = [], [], []
    for l in open(os.path.join(out, "index.jsonl"), encoding="utf-8"):
        r = json.loads(l)
        if r["test"] != test:
            continue
        im = cv2.imread(os.path.join(out, "img", r["id"] + ".jpg"))
        if im is not None:
            ims.append(im); ys.append(r["label"] == "pocket"); names.append(r["file"])
    return ims, np.array(ys, np.float32), names


def train(out, out_pt, epochs=5):
    import random
    import cv2
    import torch
    import po_train_lib as T
    torch.set_num_threads(2)
    imgs, y, srcs = T.load()
    imgs = [cv2.resize(i, (T.SZ + 16, T.SZ + 16)) for i in imgs]
    u_ims, u_y, _ = _user(out, test=False)
    for im, v in zip(u_ims, u_y):
        imgs.append(cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (T.SZ + 16, T.SZ + 16)))
        y = np.append(y, np.float32(v)); srcs.append("user")
    new = {"dx8a", "dx8b", "dx7", "live7a", "live7b", "dx6", "dx", "live", "user"}
    idx_new = [i for i, s in enumerate(srcs) if s in new]
    idx_old = [i for i, s in enumerate(srcs) if s not in new]
    random.seed(0)
    idx_old = random.sample(idx_old, min(len(idx_old), int(0.6 * len(idx_new))))
    keep = idx_new + idx_old + [i for i, s in enumerate(srcs) if s == "user"] \
        + [i for i, s in enumerate(srcs) if s == "user" and y[i] == 1] * 2
    print("train crops", len(keep), "pocket", int(y[keep].sum()), flush=True)
    net = T.make(False)
    net.load_state_dict(torch.load(os.path.join(ROOT, "models", "po_arm_mnv3.pt"), map_location="cpu"))
    pos = float(y[keep].sum())
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(min(3.0, (len(keep) - pos) / max(1.0, pos))))
    opt = torch.optim.AdamW(net.parameters(), lr=1.5e-4, weight_decay=1e-4)
    for e in range(epochs):
        net.train(); random.shuffle(keep)
        for i in range(0, len(keep), 32):
            b = keep[i:i + 32]
            opt.zero_grad()
            loss = lossf(net(T.to_t([T.aug(imgs[j]) for j in b])).squeeze(1), torch.from_numpy(y[b]))
            loss.backward(); opt.step()
        print("epoch", e + 1, flush=True)
    torch.save(net.eval().state_dict(), out_pt)
    print("saved", out_pt, flush=True)


def heldout(out, cand_pt):
    """Per held-out FILE (a person): max over its hands of min(model, v1).
    Prints AUC and counts at the live threshold for the live pair and the candidate pair."""
    from poketenashi import po_model
    main, v1, hi = _models()
    cand = po_model.PocketModel(cand_pt)
    ims, ys, names = _user(out, test=True)
    if not ims:
        print(json.dumps({"held_out": 0})); return
    per = {}
    for im, v, nm in zip(ims, ys, names):
        a, c = main.score([im])[0], cand.score([im])[0]
        b = v1.score([im])[0] if v1 else 1.0          # second model of the live pair, if any
        d = per.setdefault(nm, {"y": v, "live": 0.0, "cand": 0.0})
        d["live"] = max(d["live"], min(a, b)); d["cand"] = max(d["cand"], min(c, b))
    res = {"held_out_files": len(per)}
    for k in ("live", "cand"):
        p = np.array([d[k] for d in per.values() if d["y"] == 1]); n = np.array([d[k] for d in per.values() if d["y"] == 0])
        auc = float(np.mean((p[:, None] > n[None, :]) + 0.5 * (p[:, None] == n[None, :]))) if len(p) and len(n) else None
        res[k] = {"auc": auc, "ok_hit": int((p >= hi).sum()), "ok": len(p), "ng_hit": int((n >= hi).sum()), "ng": len(n)}
    print(json.dumps(res), flush=True)


def commit(out):
    """After a deploy: the held-out files were trained on now -> into MANIFEST."""
    names = _manifest() | {json.loads(l)["file"] for l in open(os.path.join(out, "index.jsonl"), encoding="utf-8")}
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    json.dump(sorted(names), open(MANIFEST, "w", encoding="utf-8"))
    print("manifest", len(names), "files")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "commit":
        commit(sys.argv[2])
    if cmd == "arms":
        arms(sys.argv[2])
    elif cmd == "train":
        train(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 5)
    elif cmd == "heldout":
        heldout(sys.argv[2], sys.argv[3])
