#!/bin/sh
# mkcopy.sh <name> <new-main-commit>: copy the cd32303 tree to ./<name>, change ONLY the main-repository
# commit pin (71400b94 -> <new-main-commit>) in tools/museum_data.py, i.e. what a "reviewed engineering
# refresh" of the snapshot does, and regenerate docs/site/museum.json with the PR's own generator.
# Main objects are read with GIT_DIR=$H/objstore.git (read-only `git show`); Math- from ./fixture.
set -e
H=$(cd "$(dirname "$0")" && pwd); CD="$H/../../cd32303"; N="$H/$1"
rm -rf "$N"; cp -a "$CD" "$N"; rm -rf "$N/tools/__pycache__"
sed -i "s/71400b94f6cb354a8cf7aba73ffede2138a64efa/$2/g" "$N/tools/museum_data.py"
export GIT_DIR=$H/objstore.git GIT_OPTIONAL_LOCKS=0 PATH=/opt/node22/bin:$PATH
python3 -B -E -s "$N/tools/museum_data.py" --main-root "$N" --math-root "$H/fixture/Math-"
