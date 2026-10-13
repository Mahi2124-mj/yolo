"""Pin the rule surface - the one thing nothing else enforces.

A rule has one code, one operator-facing label and, usually, zones it needs
drawn. poketenashi/ruleinfo.py owns all three; the rule engine, the alert log,
the API and both dashboards must all read from it. Nothing in the running system
fails when that stops being true - a label is restated on the fire path, a rule
drops out of the catalogue, or a rule's dispatch code drifts from CODES - until
somebody reads the diff. This is that guard.

It matters most for the rules the live surface cannot reach: TE, SHI and
RESTRICTED never fire under the shipped config (their zones are empty), so this
check is the only evidence that they are still wired to the catalogue at all.

    python -m tools.test_rule_surface
"""
import ast
import inspect
import json
import os
import re
import sys

from poketenashi import alerts, ruleinfo, webserver
from poketenashi.rules import RuleEngine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Constants so a mutation test can repoint them at a scratch copy.
FIRE_PATH = os.path.join(ROOT, "poketenashi", "rules.py")
DASHBOARDS = [os.path.join(ROOT, "web", n) for n in ("app.js", "index.html", "tv.html")]
SNAP_DIR = os.path.join(ROOT, "snapshots")

_failed = []


def check(what, ok, detail=""):
    print(("ok    " if ok else "FAIL  ") + what +
          (f"  -> {detail}" if detail and not ok else ""))
    if not ok:
        _failed.append(what)


def string_literals(path):
    """The string constants a module really uses - docstrings left out.

    A rule's label inside a docstring or comment is prose, not identity; only a
    literal the code could hand to a reader counts.
    """
    tree = ast.parse(open(path, encoding="utf-8").read())
    docs = set()
    for node in ast.walk(tree):
        kinds = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        if isinstance(node, kinds) and node.body:
            first = node.body[0]
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                docs.add(id(first.value))
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in docs]


def dispatch_codes():
    """The rule codes check() actually dispatches on."""
    return set(re.findall(r'"([A-Z]+)" in self\.enabled',
                          inspect.getsource(RuleEngine.check)))


def main():
    codes = list(ruleinfo.CODES)
    labels = set(ruleinfo.LABELS.values())

    # 1. The engine can dispatch exactly the catalogued rules, and no others -
    #    the guarantee the rule engine itself offers no evidence for when a zone
    #    keeps a rule dark for days.
    dispatch = dispatch_codes()
    check("engine dispatch == ruleinfo.CODES",
          dispatch == set(codes), f"engine={sorted(dispatch)} catalogue={sorted(codes)}")
    unnamed = [c for c in codes if not callable(getattr(RuleEngine, "rule_" + c, None))]
    check("every code has a rule_<CODE> method", not unnamed, str(unnamed))

    # 2. The catalogue is self-consistent: a code with no label, or a zone entry
    #    for a rule that does not exist, is unusable.
    check("labels cover every code", set(ruleinfo.LABELS) == set(codes),
          f"labels={sorted(ruleinfo.LABELS)} codes={sorted(codes)}")
    check("zone rules all exist", set(ruleinfo.ZONES) <= set(codes),
          str(sorted(set(ruleinfo.ZONES) - set(codes))))

    # 3. One owner per label. The log reads ruleinfo directly; a label restated
    #    on the fire path is a second owner, and one in a dashboard is a copy of
    #    the names that will drift exactly the way "Walking in center" did.
    check("alerts.RULE_NAMES is ruleinfo.LABELS", alerts.RULE_NAMES is ruleinfo.LABELS)
    fire_hits = sorted(l for l in string_literals(FIRE_PATH) if l in labels)
    check("no rule label on the fire path", not fire_hits, str(fire_hits))
    web_hits = []
    for path in DASHBOARDS:
        text = open(path, encoding="utf-8").read()
        name = os.path.basename(path)
        web_hits += [f"{name}:{l}" for l in labels
                     if re.search(r"""["'`]%s["'`]""" % re.escape(l), text)]
    check("no rule label copied into the dashboards", not web_hits, str(web_hits))

    # 4. /api/state reports each rule with the catalogue label - and the real
    #    reason, not a claim, for the ones that cannot run.
    all_zones = {"enabled": True, "detect": True,
                 "zones": ["gangway", "stairs", "handrail", "checkpoint", "restricted"]}
    every = webserver.rule_status(codes, all_zones)
    check("rule_status lists every code, in catalogue order",
          [r["rule"] for r in every] == codes)
    check("rule_status labels come from the catalogue",
          all(r["label"] == ruleinfo.LABELS[r["rule"]] for r in every))
    check("every rule runs once its zones are drawn",
          all(r["active"] for r in every),
          str([(r["rule"], r["reason"]) for r in every if not r["active"]]))
    shipped = webserver.rule_status(["PO", "KE", "NA", "RUN", "FALL"],
                                    {"enabled": True, "detect": True, "zones": ["gangway"]})
    dark = [r["rule"] for r in shipped if not r["active"]]
    check("shipped config: TE, SHI and RESTRICTED are inactive", dark == ["TE", "SHI", "RESTRICTED"],
          str(dark))
    check("every inactive rule says why",
          all(r["reason"] for r in shipped if not r["active"]))

    # 5. A served row shows the catalogue label, even one written before a
    #    rename: /api/violations reads the row's code, never its frozen text.
    legacy = webserver._parse(json.dumps(
        {"time": "2026-10-05 15:11:16", "track_id": 4409, "rule": "NA",
         "desc": "Walking in center (use sides)"}), SNAP_DIR)
    check("retained rows are served the current label",
          bool(legacy) and legacy["desc"] == ruleinfo.LABELS["NA"],
          str(legacy and legacy.get("desc")))
    for code in codes:
        rec = webserver._parse(json.dumps(
            {"time": "2026-10-06 00:00:00", "track_id": 1, "rule": code, "desc": "stale"}),
            SNAP_DIR)
        check(f"served {code} row carries the catalogue label",
              bool(rec) and rec["desc"] == ruleinfo.LABELS[code],
              str(rec and rec.get("desc")))

    print()
    if _failed:
        print(f"{len(_failed)} FAILED: " + ", ".join(_failed))
        return 1
    print("all rule-surface contracts hold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
