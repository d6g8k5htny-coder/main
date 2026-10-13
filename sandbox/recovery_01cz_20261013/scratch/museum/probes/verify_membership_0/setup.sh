#!/bin/sh
# setup.sh: (1) copy the exact cd32303 tree to ./wt_base (no __pycache__), (2) give it a hand-written
# non-bare .git whose ONLY object source is objects/info/alternates -> /home/user/main/.git/objects
# (read-only use; git never writes to an alternate). This mirrors the CI checkout of public-shop.yml
# (actions/checkout fetch-depth: 0 => all main history present; working tree = PR head), with git
# DISCOVERED from cwd (no GIT_DIR in the environment). (3) build ./fixture/Math- by copying the other
# lens's plain files and re-verifying EVERY one with cd32303 museum_data.verify_bytes (size, blob, sha256).
set -e
H=$(cd "$(dirname "$0")" && pwd); CD="$H/../../cd32303"
rm -rf "$H/wt_base"; mkdir "$H/wt_base"; (cd "$CD" && tar --exclude=__pycache__ -cf - .) | (cd "$H/wt_base" && tar -xf -)
G="$H/wt_base/.git"; mkdir -p "$G/objects/info" "$G/refs/heads" "$G/refs/tags"
printf 'ref: refs/heads/probe\n' > "$G/HEAD"
printf '[core]\n\trepositoryformatversion = 0\n\tbare = false\n' > "$G/config"
printf '/home/user/main/.git/objects\n' > "$G/objects/info/alternates"
rm -rf "$H/fixture"; mkdir -p "$H/fixture"; cp -a "$H/../membership/fixture/Math-" "$H/fixture/Math-"
python3 -I -B - "$CD" "$H/fixture/Math-" <<'PY'
import importlib.util, sys
from pathlib import Path
cd, fx = Path(sys.argv[1]), Path(sys.argv[2])
spec = importlib.util.spec_from_file_location("md", cd / "tools/museum_data.py"); md = importlib.util.module_from_spec(spec); spec.loader.exec_module(md)
n = 0
for p, ident in md.SOURCES.items():
    if ident["repository"].endswith("/Math-"):
        md.verify_bytes((fx / p).read_bytes(), ident); n += 1
print("Math- fixture re-verified with cd32303 verify_bytes:", n, "files")
PY
cd "$H/wt_base" && GIT_OPTIONAL_LOCKS=0 git rev-parse --git-dir && GIT_OPTIONAL_LOCKS=0 git cat-file -t cd32303bcf0fe4a91a1a1f2aad2b5b6eb45088d6
