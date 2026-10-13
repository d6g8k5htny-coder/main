"""V1 (static, cd32303 bytes): MAIN_COMMIT use, membership code paths, literal summary line,
and the declared snapshot's incoming/ tree vs the committed museum.json packet list."""
import json, re, subprocess, sys
from pathlib import Path
H = Path(__file__).resolve().parent; CD = H.parents[2] / "cd32303"; OBJ = H / "objstore.git"
src = (CD / "tools/museum_data.py").read_text()
print("MAIN_COMMIT occurrences in tools/museum_data.py:",
      [i + 1 for i, l in enumerate(src.splitlines()) if "MAIN_COMMIT" in l])
files = ["tools/museum_data.py", "tools/museum_check.py", "docs/site/museum.mjs", "docs/site/geometry.mjs",
         "tests/test_museum_data.py", "tests/test_museum_frontend.mjs", "tests/test_museum_geometry.mjs"]
pat = re.compile(r"ls-tree|ls_tree|iterdir|listdir|\bglob\b|\.glob\(|readdir|scandir|os\.walk|rglob|incoming/\*")
hits = [f"{f}:{i+1}" for f in files for i, l in enumerate((CD / f).read_text().splitlines()) if pat.search(l)]
print("tree/directory enumeration calls in the 7 museum files:", hits)
lines = src.splitlines()
print("museum_data.py:454-460 packet_descriptors body:"); [print("   ", lines[i]) for i in range(453, 460)]
print("museum_data.py:553:", lines[552].strip())
mjs = (CD / "docs/site/museum.mjs").read_text().splitlines()
print("museum.mjs:102:", mjs[101].strip()); print("museum.mjs:104 (id literal):", mjs[103].strip()[:120])
print("museum.mjs:191:", mjs[190].strip())
snap = json.loads((CD / "docs/site/museum.json").read_text())
c = snap["status_source"]["commit"]
tree = subprocess.run(["git", f"--git-dir={OBJ}", "ls-tree", "-d", "--name-only", c, "incoming/"],
                      capture_output=True, text=True, check=True, env={"GIT_OPTIONAL_LOCKS": "0", "PATH": "/usr/bin:/bin"}).stdout.split()
print(f"declared snapshot {c[:8]} incoming/ tree:", tree)
print("committed museum.json packets:", [p["id"] + "@" + p["result"]["commit"][:8] for p in snap["packets"]])
