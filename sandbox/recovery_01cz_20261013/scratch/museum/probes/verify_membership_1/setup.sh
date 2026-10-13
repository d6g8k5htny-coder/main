#!/bin/sh
# setup.sh: (1) mkrepo <dir>: a scratch Git repository whose working tree is a byte copy of the extracted
# cd32303 tree and whose object store borrows READ-ONLY from /home/user/main/.git/objects (alternates),
# with HEAD = cd32303. This is what actions/checkout fetch-depth:0 (public-shop.yml) gives CI: all main
# commits present, working tree = PR head. Nothing is written to /home/user/main.
# (2) mathfix/: Math- source bytes copied from the membership lens's fixture, each re-verified here
# against the cd32303 SOURCES pin with museum_data.verify_bytes (size, git blob id, sha256).
set -e
V=$(cd "$(dirname "$0")" && pwd); CD=$V/../../cd32303
mkrepo() {
  rm -rf "$1"; mkdir -p "$1"; git init -q "$1"
  echo /home/user/main/.git/objects > "$1/.git/objects/info/alternates"
  git -C "$1" update-ref refs/heads/pr cd32303bcf0fe4a91a1a1f2aad2b5b6eb45088d6
  git -C "$1" symbolic-ref HEAD refs/heads/pr
  (cd "$CD" && tar --exclude=__pycache__ -cf - .) | (cd "$1" && tar -xf -)
  git -C "$1" read-tree HEAD
  echo "repo $1: HEAD=$(git -C "$1" rev-parse --short HEAD) tracked-file diffs vs HEAD: $(git -C "$1" diff --name-only | wc -l)"
}
[ "$1" = mkrepo ] && { mkrepo "$2"; exit 0; }
rm -rf "$V/mathfix"; mkdir -p "$V/mathfix"
cp -a "$V/../membership/fixture/Math-/." "$V/mathfix/"
python3 -I - "$CD" "$V/mathfix" <<'PY'
import importlib.util, sys
from pathlib import Path
cd, fx = Path(sys.argv[1]), Path(sys.argv[2])
spec = importlib.util.spec_from_file_location("museum_data", cd / "tools/museum_data.py")
md = importlib.util.module_from_spec(spec); spec.loader.exec_module(md)
n = 0
for p, ident in md.SOURCES.items():
    if ident["repository"].endswith("/Math-"):
        md.verify_bytes((fx / ident["path"]).read_bytes(), ident); n += 1
print("mathfix: all", n, "Math- sources pass verify_bytes against the cd32303 pins")
PY
mkrepo "$V/ck"
