"""Headless smoke test: runs the full pipeline on a sample image (no camera, no
GUI window). Confirms models load, detection/pose/tracking work, and the rule
engine runs end-to-end.

    python -m tools.test_pipeline
"""
import yaml
import cv2
from ultralytics.utils import ASSETS

from poketenashi.detector import Detector
from poketenashi.zones import Zones
from poketenashi.rules import RuleEngine


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    img = cv2.imread(str(ASSETS / "bus.jpg"))  # ships with ultralytics
    if img is None:
        print("could not load sample image")
        return

    det = Detector(cfg)
    rules = RuleEngine(cfg, Zones(cfg.get("zones")))

    persons = phones = violations = None
    for _ in range(6):  # a few frames so persistence streaks can build up
        persons, phones = det(img)
        violations = rules.check(persons, phones, img)

    print("=== pipeline smoke test ===")
    print(f"persons detected : {len(persons)}")
    print(f"phones detected  : {len(phones)}")
    print(f"violations       : {violations}")
    for p in persons[:3]:
        print(f"  ID{p.track_id}  conf={p.conf:.2f}  torso={p.torso_len():.0f}px")
    print("OK - pipeline ran without errors.")


if __name__ == "__main__":
    main()
