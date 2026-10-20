"""Exact-delta readback for main PR 453 after the F1/F2 (and optional N1) AMEND.

Usage: python3 -I readback453.py NEW_HEAD   (run from /home/user/main)
Builds the expected text by applying exactly F1, F2 and, as a variant, N1 to
the d07fde42 file, and compares it with the new head's file, normalizing only
whitespace runs (the PR's line wrapping may differ). Any other change makes
both comparisons fail.
"""
import json, re, subprocess, sys

PATH = "governance/OP-WORKFLOW-20260930.md"
OLD = "d07fde42b83b9a3599f5683450e297412c8a3619"


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


new_head = sys.argv[1]
parents = git("rev-list", "--parents", "-n1", new_head).split()[1:]
changed = git("diff", "--name-only", OLD, new_head).split()
old_text = norm(git("show", f"{OLD}:{PATH}"))
new_text = norm(git("show", f"{new_head}:{PATH}"))

f1_old = "The route applies when a usage-limit failure is established"
f1_new = "The route applies only when a usage-limit failure is established"
f2_old = ("For a base-only refresh, an exact-current-head composition assessment confirms unchanged "
          "reviewed content, compatible base changes and preserved review qualifications; an eligible "
          "prior reviewer may perform that assessment, and a different model family is not "
          "automatically required.")
f2_new = ("For a base-only refresh, require an exact-current-head composition assessment confirming "
          "unchanged reviewed content, compatible base changes and preserved review qualifications. An "
          "eligible prior reviewer may perform that assessment; a different model family is not "
          "automatically required.")
n1_anchor = ("It does not replace required CI checks or platform-enforced approvals and earns no "
             "scientific or organizational-independence credit.")

assert old_text.count(f1_old) == 1 and old_text.count(f2_old) == 1 and old_text.count(n1_anchor) == 1
base = old_text.replace(f1_old, f1_new).replace(f2_old, f2_new)
variants = {
    "F1+F2": base,
    "F1+F2+N1": base.replace(n1_anchor, n1_anchor + " It is not a review."),
}
print(f"new head {new_head}")
print(f"parents: {' '.join(parents)} (expect exactly {OLD})")
print(f"files changed vs d07fde42: {changed} (expect ['{PATH}'])")
for name, text in variants.items():
    print(f"new text == d07fde42 + {name}: {text == new_text}")
owner = json.loads(subprocess.run(["gh", "api", "repos/d6g8k5htny-coder/main/issues/comments/6280206603"],
                                  capture_output=True, text=True, check=True).stdout)["body"]
for q in re.findall(r'"([^"]+)"', owner):
    if q != "the route replaces an unavailable check":
        print(f"owner quote present: {norm(q) in new_text} :: {norm(q)[:90]}")
print("owner base-only refresh sentences present:", norm(f2_new) in new_text)
print("'only when' present:", f1_new in new_text, "| 'applies when' absent:", f1_old not in new_text)
