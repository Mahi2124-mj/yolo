# PO and KE — what they are, what was measured, what is still wrong

Context prompt for the Poketenashi safety monitor (Toyota Boshoku Device India).
Everything below is measured on the live camera unless it says otherwise.

---

## The setup

One RTSP camera, `192.168.30.241`, overhead-oblique over a desk area on the left
and a gangway (the lane between the white lines) on the right. Rules run in two
separate processes:

| instance | port | rules | fps |
|---|---|---|---|
| main | 6002 | TE, NA, SHI, RUN, FALL, RESTRICTED | ~15–18 |
| PO + KE | 6003 | PO, KE | ~3 |

They were split because **KE costs 343 ms of a 411 ms frame and PO another 86**.
With both in the main pipeline everything ran at 4.9 fps; without them the rest
run at 15–18. Each instance keeps its own evidence
(`violations_po_ke.jsonl`, `clips_po_ke/`) — two processes appending to one log
interleave their writes.

Launch: `run.bat` and `run_po_ke.bat`.

---

## PO — hands in pockets

**Decides** via fingertips (primary) or wrist-to-hip geometry (fallback). A
visible hand near the pocket means *not* a pocket; a wrist in the pocket region
with no visible hand means pocket. MediaPipe supplies the fingertips.

**Gate:** `po_is_walking()` — torso lengths per second, with hysteresis
(`walk_on_tps: 1.0`, `walk_off_tps: 0.5`, `walk_window_sec: 1.0`).

**Measured history**

| configuration | rate |
|---|---|
| no gate at all | 378 alerts / 17 min |
| walking gate (current) | 7 alerts / 15 min |

**A dead end worth knowing about.** The site definition is "gangway plus the
table areas", which is the whole camera view, so a *zone* gate filters nothing.
The obvious substitute was a posture gate — exclude seated people. It does not
work on this camera: measured over 6950 samples, seated workers' knee-below-hip
drop spans **−0.21 to 0.77 torso lengths with a median of 0.34**, and the
threshold was 0.35. It is a coin flip. The camera is oblique, so a seated
person's thighs project downward by an amount that depends entirely on which way
they face. That code was written, measured, and reverted. Do not rebuild it
without a different feature.

---

## KE — phone in use

**Two independent paths, both requiring an object, not just a pose:**

- **PATH A, ear call** — `phone_call_pose()` (wrist beside a head keypoint, to
  the side of the nose, not dropped below the shoulders, upright) AND a phone
  either assigned right now or seen within `KE_EAR_PHONE_MEMORY_SEC`. Builds a
  streak; fires at `KE_EAR_CONFIRM` frames.
- **PATH B, front hand** — `check_phone_in_use()`: a detected phone within
  `KE_WRIST_NEAR` (0.60 torso) of a wrist. Fires at `KE_FRONT_CONFIRM` frames.

**Gate:** `(walking or in_gangway) and _upright()`.

**Phone detection** is a second model, `yolov8s @ imgsz 1280, conf 0.40`, run
every 2nd frame, filtered by hand proximity and nearest-wrist ownership; plus an
ear-ROI crop pass at `conf 0.20` on an upscaled crop, for the phone hidden
behind a hand.

---

## The measurements that matter

**A real phone and a steel bottle score the same.**

| object | confidence from `yolov8s` |
|---|---|
| a real phone on this camera | 0.45 – 0.50 |
| a steel water bottle on the desk | **0.46 – 0.51** |

Asking the model unrestricted (all 80 classes) does not help: it still answers
"cell phone" and nothing competes. **So `phone_conf` cannot separate them.**
Raising it above 0.51 kills real phones too. What does separate them is motion —
the bottle's box moved one pixel between consecutive frames.

**The ear streak used to be a ratchet.** Build `+1.0` per hit, decay `−0.5` per
miss, cap at `CONFIRM + 12`. Break-even was a **33 % duty cycle**: anyone whose
hand was near their head a third of the time climbed to the cap and stayed
there, and it took 24 further frames to fall below the firing line. Rubbing an
eye, adjusting a cap and resting a hand on the face all fired KE with **no phone
within 1.2 to 7.6 torso lengths, and none at all at the production threshold**.

**A frame-based threshold is not a time-based one.** `KE_EAR_CONFIRM = 6` frames
meant 1.2 s at 5 fps and 0.3 s at 20 fps. This pipeline has run anywhere from
3 to 18 fps depending on load.

**Chest-held phone is missed.** In a controlled recording the phone was detected
at 0.45–0.50 with the wrist 0.13 torso away, and KE never fired: the phone was
only assigned on ~15 % of frames, so the front streak never confirmed. This is a
false NEGATIVE and it is still open.

---

## What is fixed (all verified)

| fix | before | after |
|---|---|---|
| `KE_EAR_PHONE_MEMORY_SEC` | 8.0 s | 1.5 s |
| `KE_EAR_DECAY` | 0.5 | 1.0 (equal to the build) |
| ear streak cap | `CONFIRM + 12` | `CONFIRM + 2` |
| red box hold | 3 s stopwatch from the last fire | holds while a phone is still detected |
| clip pre-roll | 12 s | 3.7 s |
| clip playback | 3× fast-forward | real speed |
| stream lag | 1–3 s and growing | 0–1 s, stable |
| dashboard fps | EMA of `1/dt` (read 13 while the client got 5.6) | frames ÷ elapsed |

Verification of the KE changes, on synthetic sequences:

```
real call, phone visible, pose 95%   ->  97% red   (must stay high)
real call, pose only 70%             ->  94% red   (must stay high)
eye-rub, one stale phone sighting    ->   0%
hand on face, pose 60%               ->   0%
no phone at all, pose 100%           ->   0%
```

And on 5 minutes of full-resolution footage, with `--why`:

```
KE fired on 6 frames, 2 people, 1.2 per video-minute
  a phone was assigned AT THAT MOMENT : 6 of 6 (100%)
  seconds since a phone was last seen : median 0.00
  paths: ear-only 0, front-only 6
```

The ear path fired **zero** times — its streak never reached the confirm
threshold. Every remaining fire came from the front path with a genuine
detection at that instant. That detection was the bottle.

---

## What is still open

1. **A steel bottle on the desk is detected as a phone** (0.46–0.51). Confidence
   cannot fix it. The discriminator that does work is that it never moves — a
   parked-object test was built and measured (8/8 synthetic cases, and it took
   the false positives on a real clip from 5 frames to 0) and then reverted on
   request. Reinstating just that test is the open proposal.
2. **Chest-held phone is missed** — see above.
3. **A PC monitor is given a person ID** — reported, not yet reproduced with
   evidence.
4. **The camera clock is 12 h 04 m behind** the PC. It is the camera's own clock
   (its HTTP `Date` header agrees with the burned-in overlay), it is not stable
   (the offset was 1 h 57 m two days earlier), and it needs NTP, not a one-off
   set. Nothing in the app reads it — every alert, clip and evidence timestamp
   comes from the PC clock and is correct.
5. **Machine memory.** 15.7 GB RAM plus a *fixed* 16 GB pagefile is a hard
   31.3 GB commit ceiling. 28 low-memory kills in one week; the app once failed
   to allocate 8.5 MB for a single camera frame. The fix is to let Windows
   manage the pagefile.

---

## Tools built for this

- `tools/record.py` — records raw camera footage at the **same width the live
  pipeline uses**. Evidence clips are useless as test input: they are
  downscaled, already have boxes painted on, and are only a few seconds long.
- `tools/replay.py` — runs recorded footage through the real pipeline. Nothing
  is simulated; the one substitution is the clock, which advances at video rate,
  because a frame costs over a second of CPU while representing 0.08 s of video,
  so a "2 second" window otherwise covers two frames and silently never fires.
  `--set` overrides any config key or `ke.py` constant, `--json` + `--diff`
  compares two runs on identical footage, and `--why` reports the evidence
  behind every KE fire.
- `tools/scan_clips.py` — finds the violation frame inside an evidence clip.
  Written after five failed attempts at the same job: earlier versions counted
  the room's fire-exit sign, the red-and-white barrier rope, JPEG noise, and the
  app's own red badge in the corner, which is present in every frame and made
  clips report "42 of 42 frames have a violation". It looks for a red rectangle
  *outline* instead — person-shaped and thin-stroked.

---

## Working rules for anyone picking this up

- **Measure on full-resolution footage, not on clips.** Clips are written at
  1280 and the pipeline runs at 1920; the ear-ROI crop is 1.5× larger in
  production and detects differently. Two wrong conclusions came from ignoring
  this.
- **A rate is frames ÷ elapsed.** An EMA of `1/dt` reads high and misled this
  investigation for hours.
- **Both directions, always.** Driving false positives to zero is trivial —
  switch the rule off. Every threshold change needs a real-phone sample checked
  alongside.
- **Two live 15-minute windows are not comparable.** Different people, different
  traffic. Use `replay.py --diff` on one recording.

---

## Deployment changes (not application code)

### Windows paging file — the cause of the repeated kills

The box has **15.7 GB RAM and a manually fixed 16 GB pagefile**, so the commit
limit is a hard **31.3 GB** that Windows cannot grow. Measured consequences:

- **28** `Resource-Exhaustion-Detector` events in one week
- the app killed mid-frame with no traceback - the most it managed to report was
  `numpy._core._exceptions._ArrayMemoryError: Unable to allocate 8.54 MiB` for a
  single camera frame
- PowerShell and Explorer dying at the same moments, which is the same cause

Largest consumers when it happened: Claude 7.6 GB, Edge + WebView 2.2 GB,
PowerBI gateways 1.0 GB, the monitor ~2.5 GB.

**Recommendation.** Set the paging file to *Automatically manage paging file
size for all drives*:

> `Win+R` -> `sysdm.cpl` -> Advanced -> Performance **Settings** -> Advanced ->
> Virtual memory **Change** -> tick **Automatically manage** -> OK -> reboot

C: has 260 GB free, so this raises the ceiling from 31.3 GB to roughly 47 GB.
This is a machine setting; no application change fixes it, and application
tuning cannot: `torch_threads` was measured at 4, 8, 12 and 16 threads and made
no difference to speed (314 ms vs 327 ms), so there is nothing to trade.

Running two instances costs ~5 GB of the ceiling. That is affordable today
(22.6 / 31.3 GB) but it is the first thing to reconsider if the kills return
before the pagefile is fixed.

### Camera clock — enable NTP

The camera's own clock is **12 h 04 m behind** the PC. It is the camera, not the
overlay: its HTTP `Date` header agrees with the timestamp burned into the video.
It is also **not stable** - the offset was 1 h 57 m two days earlier, so a
one-off manual set will drift again. That points at NTP being off and possibly a
dead RTC battery.

**Nothing in the application reads it.** Every alert time, clip name and
evidence timestamp comes from the PC clock and is correct - verified: PC
14:01:46, newest alert 14:01:14, newest clip file 14:01:21. The only wrong thing
is the cosmetic overlay burned into the picture, which matters for anyone
reading a clip as evidence.

**Recommendation.** Enable NTP in the camera's own web interface at
`http://192.168.30.241`. Note the device is XiongMai-based (`Server: APPWebs/`),
not Hikvision - it returns 404 for ISAPI, has no ONVIF endpoint, and its
configuration port is 34567 (the proprietary DVRIP protocol). Setting the clock
therefore needs the vendor UI or the vendor app; there is no clean HTTP API for
it. If the site has no internet, the monitoring PC can serve NTP on the LAN.

**Rule to keep:** no application code may depend on the camera clock. Anything
needing a timestamp uses the PC clock.
