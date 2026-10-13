"""Every rule's identity, in ONE place.

A rule has a code, an operator-facing label, and - usually - a zone it reads.
This module owns all three. Nothing else names a rule:

  * the alert log writes the label (poketenashi/alerts.py)
  * /api/state reports each rule, its label and whether it can run, with the
    reason when it cannot (poketenashi/webserver.py)
  * the rule engine asks ZONES before it tries to judge anyone, and prints the
    same reason the dashboard shows (poketenashi/rules.py)
  * the dashboard and the TV page draw whatever the API sends, instead of
    carrying their own copy of the names

Why it matters: config.yaml switches rules on, but whether one can actually run
also depends on the zones drawn for THIS camera. TE needs the stairs and the
handrail, NA runs from the 'gangway' (or the older 'walkway_center'), SHI needs
a 'checkpoint' and RESTRICTED a 'restricted' area - all of which are empty in
the shipped config. Before this module the API and the dashboards still listed
all eight rules as if they were working: three of them had never judged
anything, and on 2026-10-05 NA itself went dark for a window while its gangway
zone was momentarily empty, silently. A rule that cannot fire must say so.
"""

# The rules that exist, in the order the dashboard and the log show them.
CODES = ("PO", "KE", "TE", "NA", "SHI", "RUN", "FALL", "RESTRICTED")

# Operator-facing text, as the company rule sheet words it. NA is a DIAGONAL
# crossing of the walkway (see poketenashi/na.py and rules.NA.mode); it was
# called "Walking in center (use sides)" until 6 Oct 2026, which described the
# rule the monitor stopped running on 3 Oct - on the rule that is two thirds of
# every alert.
LABELS = {
    "PO": "Hands in pockets",
    "KE": "Phone while walking",
    "TE": "Not holding the handrail",
    "NA": "Crossing the walkway diagonally",
    "SHI": "No safety confirmation at checkpoint",
    "RUN": "Running",
    "FALL": "Fall detected",
    "RESTRICTED": "In a restricted area",
}

# The zones a rule reads before it can judge anyone. "all" -> every zone listed
# must be drawn (TE compares a wrist against the stairs AND the handrail);
# "any" -> one of them is enough (NA works on the aisle zone, and on the older
# walkway_center that came before it). A rule that is not listed has no zone.
ZONES = {
    "TE": {"all": ("stairs", "handrail")},
    "NA": {"any": ("gangway", "walkway_center")},
    "SHI": {"any": ("checkpoint",)},
    "RESTRICTED": {"any": ("restricted",)},
}


def missing_zones(code, has):
    """The zones `code` still needs drawn - [] when it has all it needs.

    `has(name)` is True when that zone is drawn on the camera being judged.
    """
    need = ZONES.get(code)
    if not need:
        return []
    if "all" in need:
        return [z for z in need["all"] if not has(z)]
    return [] if any(has(z) for z in need["any"]) else list(need["any"])


def why(code, has):
    """Plain-language reason a rule cannot judge anyone, or None when it can."""
    need = ZONES.get(code)
    if not need:
        return None
    if "all" in need:
        miss = [z for z in need["all"] if not has(z)]
        if not miss:
            return None
        if len(miss) == 1:
            return f"needs the {miss[0]} zone drawn"
        return "needs the %s zones drawn" % " and ".join(miss)
    alts = need["any"]
    if any(has(z) for z in alts):
        return None
    if len(alts) == 1:
        return f"needs the {alts[0]} zone drawn"
    return "needs the %s zone drawn (or %s)" % (alts[0], " or ".join(alts[1:]))


def status(enabled, has, blocked=None):
    """[{rule, label, active, reason}] - what this camera is really running.

    enabled: the rule codes switched on in config.yaml.
    has:     the has-zone test of the camera being judged.
    blocked: a reason that stops every rule on this camera - it is switched off,
             or runs live view only. None when the camera runs its rules.
    """
    on = set(enabled or ())
    out = []
    for code in CODES:
        if blocked:
            reason = blocked
        elif code not in on:
            reason = "switched off in config.yaml"
        else:
            reason = why(code, has)
        out.append({"rule": code, "label": LABELS[code],
                    "active": reason is None, "reason": reason})
    return out
