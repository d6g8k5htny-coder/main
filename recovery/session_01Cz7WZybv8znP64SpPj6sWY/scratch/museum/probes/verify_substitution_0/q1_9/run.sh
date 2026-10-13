#!/bin/sh
# Q1-9 verifier: run from anywhere. No network (proxies pointed at a closed port; fetch/urlopen stubbed).
set -e
D=<SCRATCH>/museum/probes/verify_substitution_0/q1_9
export PATH=/opt/node22/bin:$PATH HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 https_proxy=http://127.0.0.1:9 http_proxy=http://127.0.0.1:9
cd /tmp
python3 -I -B "$D/v_q1_9.py"
W=$(mktemp -d "$D/render_XXXX"); python3 -I -B "$D/v_q1_9_build_manifest.py" "$W/museum.json" "$W/math"
node "$D/v_q1_9_render.mjs" "$W/museum.json" "$W/math"
rm -rf "$W"
