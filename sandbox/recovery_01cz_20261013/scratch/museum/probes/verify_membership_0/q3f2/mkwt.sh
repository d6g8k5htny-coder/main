#!/bin/sh
# mkwt.sh <name> [<new-main-commit> [all|datapy]]: copy the exact cd32303 tree to ./<name> (no __pycache__) and
# give it a hand-written .git whose only object source is an ALTERNATE to /home/user/main/.git/objects
# (git never writes to an alternate), so `git show <commit>:<path>` run by read_source from cwd works as in
# CI (fetch-depth: 0). Optionally replace the main pin 71400b94 by <new-main-commit> in
#   all   : every file of tools/ tests/ docs/ that names it (a complete "reviewed engineering refresh"),
#   datapy: tools/museum_data.py only.
set -e
H=$(cd "$(dirname "$0")" && pwd); CD="$H/../../../cd32303"; N="$H/$1"
rm -rf "$N"; mkdir "$N"; (cd "$CD" && tar --exclude=__pycache__ -cf - .) | (cd "$N" && tar -xf -)
G="$N/.git"; mkdir -p "$G/objects/info" "$G/refs/heads"
printf 'ref: refs/heads/probe\n' > "$G/HEAD"; printf '[core]\n\trepositoryformatversion = 0\n\tbare = false\n' > "$G/config"
printf '/home/user/main/.git/objects\n' > "$G/objects/info/alternates"
if [ -n "$2" ]; then
  if [ "$3" = datapy ]; then F=tools/museum_data.py; else F=$(cd "$N" && grep -rl 71400b94f6cb354a8cf7aba73ffede2138a64efa tools tests docs | sort); fi
  (cd "$N" && sed -i "s/71400b94f6cb354a8cf7aba73ffede2138a64efa/$2/g" $F); echo "pin replaced in: $(echo $F | tr '\n' ' ')"
fi
