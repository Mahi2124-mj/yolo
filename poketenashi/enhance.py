"""Display-side image enhancement.

An indoor factory aisle is a hard scene to look at: bright daylight through the
shutters at one end, deep shade under the racking at the other. A global
brightness/contrast lift blows out the bright half to fix the dark half. CLAHE
works on small tiles instead, so each region gets the contrast it needs - which
is exactly the uneven-lighting case here.

Applied to the L channel of LAB, never to RGB directly: equalising the colour
channels separately shifts hues, and a person's shirt changing colour is not an
improvement.

DISPLAY ONLY. The models keep seeing the original frame. Enhancement changes the
pixel statistics the detector was trained on, so feeding it in could shift every
rule's behaviour in ways nobody measured - a much bigger change than "make the
picture easier to read", which is what was actually asked for.

Measured on this machine:
    1280x720   CLAHE  5.2 ms   + sharpen  8.0 ms
    1920x1080  CLAHE 12.0 ms   + sharpen 15.4 ms
"""
import cv2


def clahe(frame, clip=2.0, grid=8):
    """Contrast-limited adaptive histogram equalisation on lightness only."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=float(clip),
                        tileGridSize=(int(grid), int(grid))).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def sharpen(frame, amount=0.5, radius=1.4):
    """Unsharp mask - subtract a blurred copy to bring edges back.

    Kept mild on purpose: a hard sharpen on a compressed RTSP frame amplifies the
    codec's blocking artefacts more than it recovers real detail.
    """
    blur = cv2.GaussianBlur(frame, (0, 0), float(radius))
    return cv2.addWeighted(frame, 1.0 + amount, blur, -amount, 0)


def apply(frame, cfg):
    """Run whatever the output config asks for. Returns the frame unchanged when off."""
    out = cfg or {}
    if out.get("enhance", False):
        frame = clahe(frame, out.get("enhance_clip", 2.0), out.get("enhance_grid", 8))
    if out.get("enhance_sharpen", False):
        frame = sharpen(frame, out.get("sharpen_amount", 0.5))
    return frame
