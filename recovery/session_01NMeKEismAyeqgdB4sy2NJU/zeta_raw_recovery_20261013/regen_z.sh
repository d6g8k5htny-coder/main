#!/bin/bash
cd "$(dirname "$0")"
export ZETA_FOLD_RAW=$PWD/raw_exact OPENBLAS_NUM_THREADS=1
for a in "600 900 0.02" "5000 5200 0.015"; do t0=$(date +%s); ./venv/bin/python -B pkt/stats/sample_Z.py $a >> regen_z.out 2>&1; echo "$(date -u +%H:%M:%SZ) rc=$? $(( $(date +%s)-t0 ))s sample_Z.py $a" >> regen_z.log; done
sha256sum raw_exact/Z_grid_*.npz >> regen_z.log; echo DONE >> regen_z.log
