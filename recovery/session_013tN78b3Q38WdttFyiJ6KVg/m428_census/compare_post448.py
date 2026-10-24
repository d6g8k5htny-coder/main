"""Compare the #428 mutation census before and after #448.

Usage: python3 -I compare_post448.py OLD.tsv NEW.tsv CLS.json OUT.json
OLD is the run at bccbd346 (test blob ba731031); NEW the run at main efc2fe90 (test blob 87589a82).
CLS maps each earlier survivor to its class: A (admission double-guarded by the semantic hash),
B (changes admission), C (equivalent), U (edge case).
"""
import collections
import json
import sys

old_p, new_p, cls_p, out_p = sys.argv[1:5]


def load(path):
    rows = {}
    for line in open(path):
        parts = line.rstrip("\n").split("\t")
        rows[parts[0]] = {"desc": parts[1], "verdict": parts[2], "normal": parts[3] if len(parts) > 3 else "",
                          "optimized": parts[4] if len(parts) > 4 else ""}
    return rows


old, new, cls = load(old_p), load(new_p), json.load(open(cls_p))
assert set(old) == set(new), sorted(set(old) ^ set(new))
assert old["BASE"]["verdict"] == "SURVIVED" and new["BASE"]["verdict"] == "SURVIVED", "base must pass"
ids = [k for k in old if k != "BASE"]
transitions = collections.Counter((old[k]["verdict"], new[k]["verdict"]) for k in ids)
newly_killed = sorted(k for k in ids if old[k]["verdict"] == "SURVIVED" and new[k]["verdict"] == "KILLED")
newly_survived = sorted(k for k in ids if old[k]["verdict"] == "KILLED" and new[k]["verdict"] != "KILLED")
mixed = sorted(k for k in ids if new[k]["verdict"] == "MIXED")
still = sorted(k for k in ids if new[k]["verdict"] == "SURVIVED")
by_class = collections.defaultdict(list)
for k in still:
    by_class[cls.get(k, "?")].append(k)
killed_by_class = collections.Counter(cls.get(k, "?") for k in newly_killed)
result = {
    "mutants": len(ids),
    "killed_before": sum(1 for k in ids if old[k]["verdict"] == "KILLED"),
    "killed_after": sum(1 for k in ids if new[k]["verdict"] == "KILLED"),
    "transitions": {"%s->%s" % t: n for t, n in sorted(transitions.items())},
    "newly_killed": {k: {"class": cls.get(k), "desc": new[k]["desc"], "normal": new[k]["normal"],
                         "optimized": new[k]["optimized"]} for k in newly_killed},
    "newly_killed_by_class": dict(killed_by_class),
    "regressions": newly_survived,
    "mixed": mixed,
    "still_surviving_by_class": {c: sorted(v) for c, v in sorted(by_class.items())},
    "still_surviving_B": {k: new[k]["desc"] for k in sorted(by_class.get("B", []))},
    "base": {"old": old["BASE"], "new": new["BASE"]},
}
json.dump(result, open(out_p, "w"), indent=1, sort_keys=True)
print(json.dumps({k: result[k] for k in ("mutants", "killed_before", "killed_after", "transitions",
                                          "newly_killed_by_class", "regressions", "mixed")}, sort_keys=True))
print("newly killed:", newly_killed)
print("still surviving by class:", {c: len(v) for c, v in result["still_surviving_by_class"].items()})
print("B still surviving:", list(result["still_surviving_B"]))
