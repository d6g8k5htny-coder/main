#!/bin/sh
# Q3-F2 verifier-2 setup (no network, no state change outside this directory).
#  - objstore.git: hand-made bare repo whose only object source is an alternates link to the
#    read-only /home/user/main object database (anything git writes lands here, never in main).
#  - fixture/: plain copies of every cd32303 SOURCES entry, each re-verified with the cd32303
#    museum_data.verify_bytes (size, git blob id, sha256) before use.
set -e
H=$(cd "$(dirname "$0")" && pwd); CD="$H/../../../cd32303"
rm -rf "$H/objstore.git" "$H/fixture"
mkdir -p "$H/objstore.git/objects/info" "$H/objstore.git/refs/heads" "$H/objstore.git/refs/tags"
printf 'ref: refs/heads/none\n' > "$H/objstore.git/HEAD"
printf '[core]\n\trepositoryformatversion = 0\n\tbare = true\n' > "$H/objstore.git/config"
echo /home/user/main/.git/objects > "$H/objstore.git/objects/info/alternates"
cp -a "$H/../../membership/fixture" "$H/fixture"
python3 -I -B "$H/verify_fixture.py" "$CD" "$H/fixture"
