"""Arm model v3: same network and crop as poketenashi/po_model.py (MobileNetV3-Small,
arm_crop 160 -> 144), fine-tuned from ImageNet on
  po_arms           25 Sep tester (t25), 28 Sep live test (lt), 24 Sep aisle (la)
  po_sample_arms    live sampler arms 26 Sep - 3 Oct (raw; the 3 Oct raw replay windows left out)
  po_dx_arms        5 Oct Dx staged test (user-checked labels): pockets + hands behind the back,
                    hanging still, swinging
  po_dx6_arms       6 Oct Dx staged test: hand IN the pocket vs resting / hanging OUTSIDE on the
                    jeans - same tester, same clothes
  po_live_fp_arms   sampler arms of the people behind the 4-6 Oct false live alerts
COLOUR-BLIND (user, 3 Oct: no clothing-colour cue): random grayscale, hue rotation, channel
shuffle, strong saturation jitter. No oversampling of the few positives (v2 learned
"navy uniform = pocket" from 22 oversampled live pockets).

Usage: train_v3.py <out.pt> [excl=src1,src2] [epochs=8]
       train_v3.py cv           held-out AUC per source group, v1 vs v3
"""
import lowprio  # noqa
import json, os, random, sys, time
import cv2, numpy as np, torch, torch.nn as nn, torchvision.models as tvm

SZ = 144
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
torch.set_num_threads(3)
D = r"D:\poketenashi-data"
CFG = {"colour": "full", "pos_cap": 4.0, "init": None, "lr": 3e-4, "lr_head": 1e-3}


def load():
    recs = []
    lab = json.load(open(os.path.join(D, "po_arms", "labels.json")))
    for l in open(os.path.join(D, "po_arms", "index.jsonl"), encoding="utf-8"):
        r = json.loads(l)
        if lab.get(r["id"]) in ("pocket", "out"):
            recs.append((os.path.join(D, "po_arms", "img", r["id"] + ".jpg"), lab[r["id"]] == "pocket", r["src"]))
    llab = json.load(open(os.path.join(D, "po_sample_arms", "arm_labels.json")))
    for l in open(os.path.join(D, "po_sample_arms", "index.jsonl"), encoding="utf-8"):
        r = json.loads(l)
        tm = r["time"][:16]
        if "2026-10-03 12:29" <= tm <= "2026-10-03 13:30" or "2026-10-03 15:26" <= tm <= "2026-10-03 15:42":
            continue                                    # the raw replay recordings: never trained on
        if llab.get(r["id"]) in ("pocket", "out") and r.get("inside", 0) >= 0.7:
            recs.append((os.path.join(D, "po_sample_arms", "img", r["id"] + ".jpg"), llab[r["id"]] == "pocket", "sample"))
    for sub, src in (("po_dx_arms", "dx"), ("po_dx6_arms", "dx6")):
        for l in open(os.path.join(D, sub, "index.jsonl"), encoding="utf-8"):
            r = json.loads(l)
            recs.append((os.path.join(D, sub, "img", r["id"] + ".jpg"), r["label"] == "pocket", src))
    for l in open(os.path.join(D, "po_live_fp_arms", "index.jsonl"), encoding="utf-8"):
        r = json.loads(l)
        if r["cand"] and r["inside"] >= 0.7:
            recs.append((os.path.join(D, "po_live_fp_arms", "img", r["id"] + ".jpg"), False, "live"))
    # 7 Oct: the 32 live PO alerts 6-7 Oct, labelled by the user (PO_review): src live7a = 6 Oct, live7b = 7 Oct
    l7 = os.path.join(D, "po_live7_arms", "index.jsonl")
    if os.path.exists(l7):
        for l in open(l7, encoding="utf-8"):
            r = json.loads(l)
            recs.append((os.path.join(D, "po_live7_arms", "img", r["id"] + ".jpg"), r["label"] == "pocket",
                         "live7a" if r["fold"] == "2026-10-06" else "live7b"))
    # 7 Oct evening: the raw Dx PO test (16:23-16:30), arms labelled by eye (only the unambiguous ones)
    d7 = os.path.join(D, "po_dx7_arms", "index.jsonl")
    if os.path.exists(d7):
        for l in open(d7, encoding="utf-8"):
            r = json.loads(l)
            recs.append((os.path.join(D, "po_dx7_arms", "img", r["id"] + ".jpg"), r["label"] == "pocket", "dx7"))
    # 8 Oct: user-confirmed PO test on Dx (other people, navy polo, dark trousers): dx8a = 14:49-14:50, dx8b = 14:50:47 / 14:51:12
    d8 = os.path.join(D, "po_dx8_arms", "index.jsonl")
    if os.path.exists(d8):
        for l in open(d8, encoding="utf-8"):
            r = json.loads(l)
            src = "dx8b" if r["clip"] in ("dxraw_145047_OK", "dxraw_145112_OK") else "dx8a"
            recs.append((os.path.join(D, "po_dx8_arms", "img", r["id"] + ".jpg"), r["label"] == "pocket", src))
    imgs, ys, srcs = [], [], []
    for p, y, s in recs:
        im = cv2.imread(p)
        if im is not None:
            imgs.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            ys.append(float(y))
            srcs.append(s)
    return imgs, np.array(ys, np.float32), srcs


def aug(im):
    h, w = im.shape[:2]
    s = random.uniform(0.80, 1.0)
    ch, cw = int(h * s), int(w * s)
    y0, x0 = random.randint(0, h - ch), random.randint(0, w - cw)
    im = np.ascontiguousarray(im[y0:y0 + ch, x0:x0 + cw])
    if random.random() < 0.5:
        im = np.ascontiguousarray(im[:, ::-1])
    if random.random() < 0.3:
        f = random.uniform(0.4, 0.75)
        im = cv2.resize(im, (max(8, int(im.shape[1] * f)), max(8, int(im.shape[0] * f))))
    im = cv2.resize(im, (SZ, SZ))
    # --- colour-blind: the hand's SHAPE, never the cloth's colour
    r = random.random() if CFG["colour"] in ("full", "gray") else 1.0
    if CFG["colour"] == "old":
        sat = random.uniform(0.7, 1.3)
        g = im.mean(axis=2, keepdims=True).astype(np.float32)
        im = np.clip(g + (im.astype(np.float32) - g) * sat, 0, 255).astype(np.uint8)
    elif r < 0.3:
        g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        im = np.stack([g, g, g], axis=2)
    elif r < 0.65 and CFG["colour"] == "full":
        hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV).astype(np.int32)
        hsv[:, :, 0] = (hsv[:, :, 0] + random.randint(0, 179)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.3, 1.6), 0, 255)
        im = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    if CFG["colour"] == "full" and random.random() < 0.3:
        im = im[:, :, np.random.permutation(3)]
    im = im.astype(np.float32) / 255.0
    b, c = random.uniform(0.7, 1.3), random.uniform(0.7, 1.3)
    im = np.clip(((im - 0.5) * c + 0.5) * b, 0, 1)
    if random.random() < 0.3:
        im = cv2.GaussianBlur(im, (3, 3), 0)
    return im.astype(np.float32)


def prep(im):
    return cv2.resize(im, (SZ, SZ)).astype(np.float32) / 255.0


def to_t(batch):
    a = (np.stack(batch) - MEAN) / STD
    return torch.from_numpy(a.transpose(0, 3, 1, 2).astype(np.float32).copy())


def make(pretrained=True):
    net = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
    net.classifier[3] = nn.Linear(net.classifier[3].in_features, 1)
    return net


def train(imgs, y, idx, epochs):
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    net = make(CFG["init"] is None)
    if CFG["init"]:
        net.load_state_dict(torch.load(CFG["init"], map_location="cpu"))
    pos = float(y[idx].sum())
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(min(CFG["pos_cap"], (len(idx) - pos) / max(1.0, pos))))
    opt = torch.optim.AdamW([{"params": net.features.parameters(), "lr": CFG["lr"]},
                             {"params": net.classifier.parameters(), "lr": CFG["lr_head"]}], weight_decay=1e-4)
    for ep in range(epochs):
        net.train()
        order = list(idx)
        random.shuffle(order)
        for i in range(0, len(order), 32):
            b = order[i:i + 32]
            opt.zero_grad()
            loss = lossf(net(to_t([aug(imgs[j]) for j in b])).squeeze(1), torch.from_numpy(y[b]))
            loss.backward()
            opt.step()
    return net.eval()


@torch.inference_mode()
def predict(net, imgs, idx):
    out = []
    for i in range(0, len(idx), 64):
        b = idx[i:i + 64]
        s1 = torch.sigmoid(net(to_t([prep(imgs[j]) for j in b])).squeeze(1))
        s2 = torch.sigmoid(net(to_t([prep(np.ascontiguousarray(imgs[j][:, ::-1])) for j in b])).squeeze(1))
        out.append(((s1 + s2) / 2).numpy())
    return np.concatenate(out) if out else np.zeros(0)


def auc(p, n):
    return float(np.mean((p[:, None] > n[None, :]) + 0.5 * (p[:, None] == n[None, :]))) if len(p) and len(n) else float("nan")


if __name__ == "__main__":
    t0 = time.time()
    imgs, y, srcs = load()
    from collections import Counter
    print("crops", len(y), "pocket", int(y.sum()), dict(Counter((s, int(v)) for s, v in zip(srcs, y))), flush=True)
    if sys.argv[1] == "cv":
        v1 = make(False)
        v1.load_state_dict(torch.load(r"C:\Users\DX-ADMIN\Pictures\yolo\models\po_arm_mnv3.pt", map_location="cpu"))
        v1.eval()
        for held in (["dx", "dx6"], ["live"]):
            te = [i for i, s in enumerate(srcs) if s in held]
            tr = [i for i, s in enumerate(srcs) if s not in held]
            p3 = predict(train(imgs, y, tr, 8), imgs, te)
            p1 = predict(v1, imgs, te)
            yt = y[te]
            for name, p in (("v1", p1), ("v3", p3)):
                pp, nn_ = p[yt == 1], p[yt == 0]
                print(f"held out {held}: {name} AUC {auc(pp, nn_):.3f}  pocket>=0.8 {np.mean(pp >= 0.8) if len(pp) else float('nan'):.2f}"
                      f"  out>=0.8 {np.mean(nn_ >= 0.8):.3f}   ({time.time() - t0:.0f}s)", flush=True)
            if held == ["dx", "dx6"]:
                # the case the live rule gets wrong: hand OUTSIDE on the jeans (6 Oct)
                sub = [k for k, i in enumerate(te) if srcs[i] == "dx6"]
                for name, p in (("v1", p1), ("v3", p3)):
                    q = p[sub]; yy = yt[sub]
                    print(f"   6 Oct only {name}: pocket>=0.8 {np.mean(q[yy == 1] >= 0.8):.2f}  out>=0.8 {np.mean(q[yy == 0] >= 0.8):.3f}")
        sys.exit(0)
    out = sys.argv[1]
    excl = next((a.split("=", 1)[1].split(",") for a in sys.argv[2:] if a.startswith("excl=")), [])
    ep = int(next((a.split("=", 1)[1] for a in sys.argv[2:] if a.startswith("epochs=")), 8))
    tr = [i for i, s in enumerate(srcs) if s not in excl]
    torch.save(train(imgs, y, tr, ep).state_dict(), out)
    print("saved", out, len(tr), "crops, excluded", excl, f"({time.time() - t0:.0f}s)", flush=True)
