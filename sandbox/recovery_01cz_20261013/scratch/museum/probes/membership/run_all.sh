#!/bin/sh
# Q3 membership lens: run every probe against the cd32303 bytes. No network; git is used read-only
# (git show / cat-file / ls-tree / merge-base). Main objects are read through ./objstore.git, a bare store
# whose objects/info/alternates points at /home/user/main/.git/objects, so nothing a probe or a PR test
# runs can write into /home/user/main. Writes only here.
H=$(cd "$(dirname "$0")" && pwd); cd "$H"
echo "##### build_fixture"; rm -rf fixture; python3 -I build_fixture.py ../../cd32303 /home/user/Math- /home/user/math- /home/user/main
echo "##### P0 baseline"; python3 -I p0_baseline.py
echo "##### P1 unit"; python3 -I p1_fallback_unit.py
echo "##### P1 CI"; ./p1_ci_false_green.sh
echo "##### P1b render (control 71400b94)"; for m in harness strict; do /opt/node22/bin/node render.mjs ../../cd32303/docs/site/museum.json $m | grep -v packet-cards; done
echo "##### P1b render (bd9fc9d6)"; for m in harness strict; do /opt/node22/bin/node render.mjs wt_p1/docs/site/museum.json $m; done
echo "##### P1c"; ./p1c_env_dependent_source.sh
echo "##### P2"; ./p2_omission.sh
echo "##### P2c"; ./p2c_empty_list_ci.sh
echo "##### P3"; ./p3_unlanded_snapshot.sh
echo "##### P4"; ./p4_status_moving_ref.sh
echo "##### P5"; ./p5_test_writes_inherited_gitdir.sh
