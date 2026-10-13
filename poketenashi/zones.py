"""Polygon zones used by the location-dependent rules (TE, NA, SHI).

Zones are site-specific, so they live in config.yaml and are drawn with
`python -m tools.zone_drawer`. An empty/undefined zone disables its rule.
"""
import numpy as np
import cv2


class Zones:
    def __init__(self, zones_cfg):
        self.polys = {}
        for name, pts in (zones_cfg or {}).items():
            if pts and len(pts) >= 3:
                self.polys[name] = np.array(pts, dtype=np.int32)

    def update(self, zones_cfg):
        """Swap in new polygons (from the zone editor). One reference swap, so a
        reader on another thread sees the old set or the new one, never half."""
        polys = {}
        for name, pts in (zones_cfg or {}).items():
            if pts and len(pts) >= 3:
                polys[name] = np.array(pts, dtype=np.int32)
        self.polys = polys

    def parts(self, name):
        """The polygons of zone `name`: `name` itself plus any `name_<part>`
        (e.g. checkpoint_left + checkpoint_right on the Material Gate camera -
        a zone in two places, one polygon each)."""
        polys = self.polys
        return [p for k, p in polys.items() if k == name or k.startswith(name + "_")]

    def has(self, name):
        return bool(self.parts(name))

    def part_of(self, name, point):
        """The key of the part of zone `name` that holds `point` (e.g.
        'checkpoint_left'), or None."""
        if point is None:
            return None
        p = (float(point[0]), float(point[1]))
        for k, poly in self.polys.items():
            if (k == name or k.startswith(name + "_")) and cv2.pointPolygonTest(poly, p, False) >= 0:
                return k
        return None

    def contains(self, name, point):
        if point is None:
            return False
        p = (float(point[0]), float(point[1]))
        return any(cv2.pointPolygonTest(poly, p, False) >= 0 for poly in self.parts(name))

    def signed_dist(self, name, point):
        """+inside / -outside distance to the polygon edge, or None if no zone
        (several parts: the nearest / deepest one)."""
        parts = self.parts(name)
        if not parts or point is None:
            return None
        p = (float(point[0]), float(point[1]))
        return max(cv2.pointPolygonTest(poly, p, True) for poly in parts)
