"""P0 baseline: the cd32303 museum.json packet index is reproduced from the declared snapshot.
(a) git ls-tree of the declared main commit 71400b94 lists exactly the incoming/ packets;
(b) museum_data.build() from the git object store (git show 71400b94:<path>, read-only) and from plain
    verified fixtures both reproduce cd32303 docs/site/museum.json byte-for-byte."""
import json, subprocess, sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import load, CD, MAIN_GIT, FIX
md = load()
snap = md.SOURCES["STATUS.md"]["commit"]
print("declared main snapshot (STATUS.md / packet commit):", snap)
print("MAIN_COMMIT constant:", md.MAIN_COMMIT, "| referenced elsewhere in museum_data.py:",
      (CD / "tools/museum_data.py").read_text().count("MAIN_COMMIT") - 1, "times")
tree = subprocess.run(["git", "-C", str(MAIN_GIT), "ls-tree", "-d", "--name-only", snap, "incoming/"],
                      capture_output=True, text=True, check=True).stdout.split()
print("packets in tree at snapshot:", tree)
print("packets in museum.json     :", ["incoming/" + p["id"] for p in json.loads((CD / "docs/site/museum.json").read_text())["packets"]])
expected = (CD / "docs/site/museum.json").read_bytes()
for label, main_root in (("git objects @/home/user/main", MAIN_GIT), ("plain fixture", FIX / "main")):
    got = md.dump(md.build(main_root, FIX / "Math-"))
    print(f"build(main_root={label}) == cd32303 museum.json:", got == expected)
