"""KE phone checker: is the object model's "cell phone" box a real phone in use?
(12 Oct 2026). MobileNetV3-Small, binary, on the candidate crops of ke_phone_data.py.

  ke_phone_train.py cv <data_dir>            5-fold CV split BY ALERT: alert-level OK caught /
                                             NG stopped, object model alone vs + checker
  ke_phone_train.py train <data_dir> <out.pt>

Alert decision (as live): the alert fires if ANY candidate has detector conf >= DET_MIN
(and, with the checker, checker P >= thr). Colour-blind augmentation (no clothing cue)."""
import json
import os
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lowprio  # noqa: E402,F401

SZ = 112
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
DET_MIN = 0.40
torch.set_num_threads(2)


def load(d):
    rows = [json.loads(l) for l in open(os.path.join(d, "index.jsonl"), encoding="utf-8")]
    ims = []
    for r in rows:
        im = cv2.imread(os.path.join(d, "img", r["id"] + ".jpg"))
        ims.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    return rows, ims


def aug(im):
    im = cv2.resize(im, (SZ + 12, SZ + 12))
    y, x = random.randint(0, 12), random.randint(0, 12)
    im = np.ascontiguousarray(im[y:y + SZ, x:x + SZ])
    if random.random() < 0.5:
        im = np.ascontiguousarray(im[:, ::-1])
    r = random.random()
    if r < 0.35:
        g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY); im = np.stack([g, g, g], 2)
    elif r < 0.7:
        hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV).astype(np.int32)
        hsv[..., 0] = (hsv[..., 0] + random.randint(0, 179)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(0.3, 1.6), 0, 255)
        im = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    im = im.astype(np.float32) / 255.0
    im = np.clip(((im - .5) * random.uniform(.7, 1.3) + .5) * random.uniform(.7, 1.3), 0, 1)
    if random.random() < 0.3:
        im = cv2.GaussianBlur(im, (3, 3), 0)
    return im


def prep(im):
    return cv2.resize(im, (SZ, SZ)).astype(np.float32) / 255.0


def to_t(b):
    a = (np.stack(b) - MEAN) / STD
    return torch.from_numpy(a.transpose(0, 3, 1, 2).copy())


def make():
    net = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.DEFAULT)
    net.classifier[3] = nn.Linear(net.classifier[3].in_features, 1)
    return net


def train(ims, y, idx, epochs=12):
    torch.manual_seed(0); random.seed(0)
    net = make()
    pos = float(y[idx].sum())
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(min(4.0, (len(idx) - pos) / max(1.0, pos))))
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    for _ in range(epochs):
        net.train(); o = list(idx); random.shuffle(o)
        for i in range(0, len(o), 32):
            b = o[i:i + 32]; opt.zero_grad()
            l = lossf(net(to_t([aug(ims[j]) for j in b])).squeeze(1), torch.from_numpy(y[b])); l.backward(); opt.step()
    return net.eval()


@torch.inference_mode()
def predict(net, ims, idx):
    out = []
    for i in range(0, len(idx), 64):
        b = idx[i:i + 64]
        p1 = torch.sigmoid(net(to_t([prep(ims[j]) for j in b])).squeeze(1))
        p2 = torch.sigmoid(net(to_t([prep(np.ascontiguousarray(ims[j][:, ::-1])) for j in b])).squeeze(1))
        out.append(((p1 + p2) / 2).numpy())
    return np.concatenate(out) if out else np.zeros(0)


def alert_table(rows, p, thr):
    A = {}
    for r, pp in zip(rows, p):
        a = A.setdefault(r["alert"], {"label": r["label"], "det": False, "both": False})
        if r["conf"] >= DET_MIN:
            a["det"] = True
            if pp >= thr:
                a["both"] = True
    return A


def cv(d):
    rows, ims = load(d)
    y = np.array([r["label"] == "OK" for r in rows], np.float32)
    alerts = sorted({r["alert"] for r in rows}); random.seed(1); random.shuffle(alerts)
    fold = {a: i % 5 for i, a in enumerate(alerts)}
    P = np.zeros(len(rows))
    for f in range(5):
        te = [i for i, r in enumerate(rows) if fold[r["alert"]] == f]
        tr = [i for i, r in enumerate(rows) if fold[r["alert"]] != f]
        P[te] = predict(train(ims, y, tr), ims, te)
        print("fold", f + 1, "done", flush=True)
    print(f"candidates: {len(rows)} (phone-labelled {int(y.sum())}) from {len(alerts)} alerts")
    for thr in (0.3, 0.5, 0.7):
        A = alert_table(rows, P, thr)
        ok = [a for a in A.values() if a["label"] == "OK"]; ng = [a for a in A.values() if a["label"] == "NG"]
        print(f"thr {thr}: object model alone - OK caught {sum(a['det'] for a in ok)}/{len(ok)}, NG fired {sum(a['det'] for a in ng)}/{len(ng)}"
              f" | + checker - OK caught {sum(a['both'] for a in ok)}/{len(ok)}, NG fired {sum(a['both'] for a in ng)}/{len(ng)}")
    json.dump({"p": P.tolist()}, open(os.path.join(d, "cv_scores.json"), "w"))


def train_full(d, out):
    rows, ims = load(d)
    y = np.array([r["label"] == "OK" for r in rows], np.float32)
    torch.save(train(ims, y, list(range(len(rows)))).state_dict(), out)
    print("saved", out)


if __name__ == "__main__":
    if sys.argv[1] == "cv":
        cv(sys.argv[2])
    else:
        train_full(sys.argv[2], sys.argv[3])
