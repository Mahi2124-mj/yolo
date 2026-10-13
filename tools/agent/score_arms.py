"""Score labelled arm crops with several models; per (clip, track, side) the share
of crops >= 0.8 (the live POCKET threshold). Usage: score_arms.py <index.jsonl> <img dir> <model.pt>..."""
import json, os, sys
import cv2, numpy as np, torch
torch.set_num_threads(1)
sys.path.insert(0, r"C:\Users\DX-ADMIN\Pictures\yolo")
from poketenashi.po_model import PocketModel
idx, img = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(idx)]
for mp in sys.argv[3:]:
    m = PocketModel(mp, threads=1)
    by = {}
    for r in rows:
        c = cv2.imread(os.path.join(img, r["id"] + ".jpg"))
        p = float(m.score([c])[0])
        by.setdefault((r["clip"][6:12], r["track"], r["side"], r["label"]), []).append(p)
    print("==", os.path.basename(mp))
    for k, v in sorted(by.items()):
        print(f"  {k[0]} t{k[1]}{k[2]} {k[3]:6s} >=0.8: {sum(x >= 0.8 for x in v)}/{len(v)}  median {np.median(v):.2f}")
