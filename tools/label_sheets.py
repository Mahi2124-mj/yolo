"""Numbered contact sheets of sampler crops, for labelling by eye.

    python -m tools.label_sheets <samples_dir> <po|ke> <out_dir> [--pick fired|quiet|all|scored]
           [--limit N] [--per-sheet 30] [--seed 0] [--skip-labelled <labels.json>]

Writes <out_dir>/<kind>_sheet_NN.jpg and <kind>_sheet_index.json (tile number ->
sample id). Labels go into a JSON file {sample_id: label}; tools/train_probe.py
reads them.
"""
import argparse
import glob
import json
import os
import random

import cv2
import numpy as np


def load_index(samples_dir, kind):
    recs = []
    p = os.path.join(samples_dir, kind, "index.jsonl")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
    return recs


def crop_path(samples_dir, kind, rec, part="crop"):
    return os.path.join(samples_dir, kind, rec["day"], f"{rec['id']}_{part}.jpg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("samples_dir")
    ap.add_argument("kind", choices=["po", "ke"])
    ap.add_argument("out_dir")
    ap.add_argument("--pick", default="all",
                    choices=["fired", "quiet", "all", "scored", "pose", "pocketish"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--per-sheet", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-labelled", default=None)
    a = ap.parse_args()
    recs = load_index(a.samples_dir, a.kind)
    if a.pick == "fired":
        recs = [r for r in recs if r.get("fired")]
    elif a.pick == "quiet":
        recs = [r for r in recs if not r.get("fired")]
    elif a.pick == "scored":
        recs = [r for r in recs if r.get("fired") or r.get("best", 0) > 0]
    elif a.pick == "pose":                  # KE: phone-use or call pose, or fired
        recs = [r for r in recs if r.get("use_pose") or r.get("call_pose") or r.get("fired")]
    elif a.pick == "pocketish":             # PO: a still wrist parked at the hip, or fired
        recs = [r for r in recs if r.get("pocketish") or r.get("fired")]
    if a.skip_labelled and os.path.exists(a.skip_labelled):
        done = json.load(open(a.skip_labelled, encoding="utf-8"))
        recs = [r for r in recs if r["id"] not in done]
    random.Random(a.seed).shuffle(recs)
    if a.limit:
        recs = recs[:a.limit]
    os.makedirs(a.out_dir, exist_ok=True)
    tile = 200 if a.kind == "ke" else 170
    cols = 6
    index = {}
    sheets = 0
    for s0 in range(0, len(recs), a.per_sheet):
        tiles = []
        for j, r in enumerate(recs[s0:s0 + a.per_sheet]):
            n = s0 + j
            im = cv2.imread(crop_path(a.samples_dir, a.kind, r))
            t = np.zeros((tile + 18, tile, 3), np.uint8)
            if im is not None:
                s = tile / max(im.shape[:2])
                im = cv2.resize(im, (max(1, int(im.shape[1] * s)), max(1, int(im.shape[0] * s))),
                                interpolation=cv2.INTER_CUBIC)
                t[:im.shape[0], :im.shape[1]] = im
            tag = f"{n}" + (" F" if r.get("fired") else "") + (f" p{r['best']:.2f}" if r.get("best") else "")
            cv2.putText(t, tag, (3, tile + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            tiles.append(t)
            index[n] = r["id"]
        while len(tiles) % cols:
            tiles.append(np.zeros((tile + 18, tile, 3), np.uint8))
        grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
        cv2.imwrite(os.path.join(a.out_dir, f"{a.kind}_sheet_{sheets:02d}.jpg"), grid,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        sheets += 1
    json.dump(index, open(os.path.join(a.out_dir, f"{a.kind}_sheet_index.json"), "w"), indent=0)
    print(f"{a.kind}: {len(recs)} samples on {sheets} sheets in {a.out_dir}")


if __name__ == "__main__":
    main()
