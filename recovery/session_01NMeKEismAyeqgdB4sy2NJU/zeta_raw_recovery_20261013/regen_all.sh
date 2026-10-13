#!/bin/bash
# Regenerate the six raw payloads of experiments/zeta_fold_exploration (main 4675c5d3) with the recorded commands,
# single-threaded OpenBLAS (the configuration that reproduced zeta_A byte for byte), and compare with MANIFEST.json.
cd "$(dirname "$0")"
export ZETA_FOLD_RAW=$PWD/raw_exact OPENBLAS_NUM_THREADS=1
: > regen_all.log
run () { local t0=$(date +%s); echo "$(date -u +%H:%M:%SZ) START $*" >> regen_all.log; ./venv/bin/python -B "$@" >> regen_all.out 2>&1; echo "$(date -u +%H:%M:%SZ) END rc=$? $(( $(date +%s) - t0 ))s $*" >> regen_all.log; }
run pkt/stats/rs_zeta.py A 1e4 1e5
run pkt/stats/rs_zeta.py B 1e5 1e6
run pkt/stats/gauss_proc.py 500 4
run pkt/stats/cue.py 12 86016 4
run pkt/stats/cue.py 128 8064 4
run pkt/stats/analyze.py
./venv/bin/python -I - >> regen_all.log <<'PY'
import json, hashlib, os
m = json.load(open('MANIFEST_zeta.json'))
for f in m['raw_files']:
    name = os.path.basename(f['path']); p = os.path.join('raw_exact', name)
    if not os.path.exists(p):
        print('MISSING', name); continue
    b = open(p, 'rb').read(); h = hashlib.sha256(b).hexdigest()
    print('%s %s %s bytes=%d (expected %d) sha=%s' % ('MATCH' if h == f['sha256'] and len(b) == f['bytes'] else 'DIFF ', name, '', len(b), f['bytes'], h[:16]))
PY
echo DONE >> regen_all.log
