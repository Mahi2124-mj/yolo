"""Rebuild the PO master library (.npz) from the saved crop folders.

Use this if you added/removed images by hand under masters/po/pocket and
masters/po/out, or want to regenerate embeddings (e.g. after a model change).

    python -m tools.build_masters
"""
import os
import glob

import cv2
import yaml

from poketenashi.embedder import Embedder
from poketenashi.master_match import MasterLibrary


def main():
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    out_file = cfg["rules"]["PO"].get("master_file", "masters/po_masters.npz")

    emb = Embedder(device=cfg.get("models", {}).get("device") or None)
    lib = MasterLibrary(out_file)
    lib.emb = lib.emb[:0] if lib.emb.size else lib.emb   # clear
    lib.labels = []

    total = 0
    for label in ("pocket", "out"):
        files = glob.glob(f"masters/po/{label}/*.jpg")
        for f in files:
            img = cv2.imread(f)
            v = emb.embed(img)
            if v is not None:
                lib.add(v, label)
                total += 1
        print(f"{label}: {len(files)} images")

    lib.save()
    print(f"[build] wrote {total} masters to {out_file}")


if __name__ == "__main__":
    main()
