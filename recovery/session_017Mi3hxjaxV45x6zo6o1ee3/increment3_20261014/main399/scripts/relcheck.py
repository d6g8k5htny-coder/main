"""Scope and release-key-only check for a site candidate (017 review tooling).

usage: python3 -I relcheck.py REPO BASE HEAD [SEMANTIC_PATH ...]

Lists every path changed between BASE and HEAD. Paths not named as semantic
must be modifications whose bytes are equal to the base after removing the
site-release key with the HEAD's own tools/site_asset_release.py transform
(the method used in the main#390 review). Reports release-key sets and any
violation; exit status 1 on any violation.
"""
import collections
import importlib.util
import re
import subprocess
import sys
import tempfile

repo, base, head = sys.argv[1], sys.argv[2], sys.argv[3]
semantic = set(sys.argv[4:])


def git(*args):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, check=True).stdout


def show(rev, path):
    return git("show", f"{rev}:{path}")


tool = "tools/site_asset_release.py"
with tempfile.NamedTemporaryFile("wb", suffix=".py", delete=False) as fh:
    fh.write(show(head, tool))
spec = importlib.util.spec_from_file_location("sar_head", fh.name)
sar = importlib.util.module_from_spec(spec)
sys.modules["sar_head"] = sar
spec.loader.exec_module(sar)

problems = []
if show(base, tool) != show(head, tool):
    problems.append(f"{tool} itself differs between base and head")

rows = [line.split("\t") for line in git("diff", "--no-renames", "--name-status", base, head).decode().splitlines() if line]
kinds, keys_old, keys_new = collections.Counter(), set(), set()
seen_semantic, others = [], 0
for status, path in rows:
    if path in semantic:
        seen_semantic.append((status, path))
        continue
    others += 1
    if status != "M":
        problems.append(f"{path}: status {status} outside the semantic set")
        continue
    suffix = "." + path.rsplit(".", 1)[-1] if "." in path else ""
    kinds[suffix] += 1
    a, b = show(base, path), show(head, path)
    if suffix in sar.EDITABLE:
        ta, tb = a.decode(), b.decode()
        if sar.transform(ta, suffix, sar.without_release) != sar.transform(tb, suffix, sar.without_release):
            problems.append(f"{path}: differs after release-key normalization")
        keys_old |= set(re.findall(r"site-release=([0-9a-f]{64})", ta))
        keys_new |= set(re.findall(r"site-release=([0-9a-f]{64})", tb))
    elif a != b:
        problems.append(f"{path}: non-editable bytes differ")

print(f"base {base} head {head}")
print(f"changed paths: {len(rows)}; semantic: {len(seen_semantic)}; other: {others} {dict(kinds)}")
for status, path in seen_semantic:
    print(f"  semantic {status} {path}")
print("release keys base:", sorted(k[:12] for k in keys_old), "head:", sorted(k[:12] for k in keys_new))
print("violations:", problems if problems else "none")
sys.exit(1 if problems else 0)
