#!/bin/sh
# P3: nothing checks that the declared main snapshot is on the default branch ("landed").
# Pin main to 3b8c4545 = origin/chatgpt/never-merge-formal-negative-20260927, a commit titled
# "TEST ONLY: ... NEVER MERGE" that is not an ancestor of main, yet holds STATUS.md / SARD review / notebook /
# the packet byte-identical to the pins. Generator, CI gate and strict commit:path render all pass, and
# the page labels the packet "pinned landed submissions".
H=$(cd "$(dirname "$0")" && pwd); C=3b8c45452c01f669a7245c472acdb8cc4af5b2f6
"$H/mkcopy.sh" wt_p3 $C
export GIT_DIR=$H/objstore.git GIT_OPTIONAL_LOCKS=0 PATH=/opt/node22/bin:$PATH
git log --format='--- %h %s' -1 $C
env -u GIT_DIR git -C /home/user/main branch -r --contains $C | sed 's/^/    contained in: /'
git merge-base --is-ancestor $C 513261bb && echo "    ancestor of main 513261bb: yes" || echo "    ancestor of main 513261bb: NO"
git merge-base --is-ancestor $C d77a85a7 && echo "    ancestor of #151 merge d77a85a7: yes" || echo "    ancestor of #151 merge d77a85a7: NO"
cd "$H/wt_p3" && python3 -B -E -s tools/museum_check.py --math-root "$H/fixture/Math-" > "$H/out_p3_check.txt" 2>&1; rc=$?
grep -E '^# (pass|fail)|^Museum' "$H/out_p3_check.txt"; echo "museum_check exit=$rc"
cd "$H"; node render.mjs wt_p3/docs/site/museum.json strict | grep -v 'packet-cards'
grep -o 'These are pinned landed submissions[^<]*' ../../cd32303/docs/site/museum.html
echo "--- ancestry / default-branch checks in the three files:"
grep -cE "merge-base|is-ancestor|rev-list|first-parent|refs/heads|origin/main" ../../cd32303/tools/museum_data.py ../../cd32303/tools/museum_check.py ../../cd32303/docs/site/museum.mjs
