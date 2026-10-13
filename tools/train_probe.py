"""Train a small classifier on labelled sampler crops (phone / pocket), honestly.

    python -m tools.train_probe <po|ke> --samples DIR [DIR ...] --labels labels.json
           [--backbone mobilenet|resnet50] [--save models/ke_phone_probe.pt]

A pretrained ImageNet backbone (already on this PC, nothing downloaded) turns
each crop into a feature vector; a logistic-regression head is trained on top.
Accuracy is measured with GROUPED cross-validation - all crops of one person
(clip + track) are either all in training or all in the test fold - because
crops of the same person are near-duplicates, and mixing them makes any model
look perfect.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

POS = {"ke": "phone", "po": "pocket"}
NEG = {"ke": "no_phone", "po": "out"}


def backbone(name, device):
    import torchvision.models as tvm
    if name == "resnet50":
        w = tvm.ResNet50_Weights.DEFAULT
        net = tvm.resnet50(weights=w)
        net.fc = torch.nn.Identity()
    else:
        w = tvm.MobileNet_V3_Small_Weights.DEFAULT
        net = tvm.mobilenet_v3_small(weights=w)
        net.classifier = torch.nn.Identity()
    return net.eval().to(device), w.transforms()


@torch.inference_mode()
def embed(net, tf, paths, device, bs=32):
    out = []
    for i in range(0, len(paths), bs):
        ts = []
        for p in paths[i:i + bs]:
            im = cv2.imread(p)
            rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            ts.append(tf(torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)))
        v = net(torch.stack(ts).to(device)).float()
        out.append((v / (v.norm(dim=1, keepdim=True) + 1e-6)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 1), np.float32)


def fit(X, y, l2=1e-3, epochs=400):
    Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    pos = float(yt.sum())
    wpos = (len(yt) - pos) / max(1.0, pos)              # balance the classes
    lin = torch.nn.Linear(X.shape[1], 1)
    opt = torch.optim.LBFGS(lin.parameters(), lr=0.5, max_iter=epochs)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(wpos))

    def closure():
        opt.zero_grad()
        loss = lossf(lin(Xt).squeeze(1), yt) + l2 * (lin.weight ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return lin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["po", "ke"])
    ap.add_argument("--samples", nargs="+", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--backbone", default="mobilenet", choices=["mobilenet", "resnet50"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--save", default=None)
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    labels = json.load(open(a.labels, encoding="utf-8"))
    paths, y, groups = [], [], []
    for d in a.samples:
        idx = os.path.join(d, a.kind, "index.jsonl")
        if not os.path.exists(idx):
            continue
        for line in open(idx, encoding="utf-8"):
            r = json.loads(line)
            lab = labels.get(r["id"])
            if lab not in (POS[a.kind], NEG[a.kind]):
                continue
            p = os.path.join(d, a.kind, r["day"], r["id"] + "_crop.jpg")
            if os.path.exists(p):
                paths.append(p)
                y.append(1 if lab == POS[a.kind] else 0)
                groups.append(f"{d}|{r['day']}|{r['track']}|{r['id'].split('_')[0][:7]}")
    y = np.array(y)
    print(f"{a.kind}: {len(y)} labelled crops ({int(y.sum())} {POS[a.kind]}, "
          f"{int((1 - y).sum())} {NEG[a.kind]}), {len(set(groups))} people")
    if y.sum() < 5 or (1 - y).sum() < 5:
        print("not enough of both classes to train yet")
        return
    device = "cpu"
    net, tf = backbone(a.backbone, device)
    X = embed(net, tf, paths, device)
    ug = sorted(set(groups))
    rng = np.random.RandomState(0)
    rng.shuffle(ug)
    fold_of = {g: i % a.folds for i, g in enumerate(ug)}
    scores = np.zeros(len(y))
    for k in range(a.folds):
        te = np.array([fold_of[g] == k for g in groups])
        if te.sum() == 0 or y[~te].sum() == 0 or (1 - y[~te]).sum() == 0:
            continue
        lin = fit(X[~te], y[~te])
        with torch.no_grad():
            scores[te] = torch.sigmoid(lin(torch.tensor(X[te])).squeeze(1)).numpy()
    print(f"grouped {a.folds}-fold cross-validation ({a.backbone}):")
    for t in (0.3, 0.5, 0.7, 0.8, 0.9):
        pred = scores >= t
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        print(f"  threshold {t:.1f}: precision {prec:5.1%}  recall {rec:5.1%}  (tp {tp}, fp {fp}, fn {fn})")
    if a.save:
        lin = fit(X, y)
        torch.save({"backbone": a.backbone, "weight": lin.weight.detach(), "bias": lin.bias.detach(),
                    "kind": a.kind, "n": int(len(y))}, a.save)
        print("saved", a.save)


if __name__ == "__main__":
    main()
