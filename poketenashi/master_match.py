"""A tiny labelled "master" library of reference embeddings + similarity voting.

Stores embeddings and their labels ('pocket' / 'out') in a .npz file. At runtime
a query embedding is compared to all masters by cosine similarity; the k nearest
vote (similarity-weighted) to decide the label and a confidence.
"""
import os

import numpy as np


class MasterLibrary:
    def __init__(self, path):
        self.path = path
        self.emb = np.zeros((0, 0), dtype=np.float32)
        self.labels = []
        self.load()

    def load(self):
        if os.path.exists(self.path):
            d = np.load(self.path, allow_pickle=True)
            self.emb = d["emb"].astype(np.float32)
            self.labels = [str(x) for x in d["labels"]]

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        np.savez(self.path, emb=self.emb, labels=np.array(self.labels))

    def add(self, vec, label):
        vec = np.asarray(vec, dtype=np.float32).reshape(1, -1)
        self.emb = vec if self.emb.shape[0] == 0 else np.vstack([self.emb, vec])
        self.labels.append(label)

    def pop(self, n=1):
        """Remove the last n samples (for undo). Returns how many were removed."""
        n = min(n, len(self.labels))
        if n:
            self.emb = self.emb[:-n]
            del self.labels[-n:]
        return n

    def count(self, label):
        return self.labels.count(label)

    def __len__(self):
        return len(self.labels)

    def predict(self, vec, k=5):
        """Return (label, confidence) for a query embedding, or (None, 0.0)."""
        if vec is None or len(self) == 0:
            return None, 0.0
        sims = self.emb @ np.asarray(vec, dtype=np.float32)   # cosine (all normalised)
        k = min(k, len(self))
        idx = np.argsort(-sims)[:k]
        pocket = float(sum(max(sims[i], 0.0) for i in idx if self.labels[i] == "pocket"))
        out = float(sum(max(sims[i], 0.0) for i in idx if self.labels[i] == "out"))
        if pocket + out <= 1e-6:
            return "out", 0.0
        label = "pocket" if pocket > out else "out"
        conf = (pocket if label == "pocket" else out) / (pocket + out)
        return label, conf
