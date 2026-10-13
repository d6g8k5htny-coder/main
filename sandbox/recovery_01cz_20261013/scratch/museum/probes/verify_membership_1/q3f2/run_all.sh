#!/bin/sh
H=$(cd "$(dirname "$0")" && pwd); cd "$H"
for s in "./setup.sh" "python3 -I -B v1_static.py" "./v0_baseline_render.sh" "./v2_full_refresh.sh" "./v3_v4_empty.sh" "./v5_narrow_refresh_unittest.sh"; do echo "\$ $s"; sh -c "$s" 2>&1; done
