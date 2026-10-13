#!/bin/bash
# search the CPU-dispatch space for a byte-identical zeta_A.npz (expected sha256 f365833e...)
cd "$(dirname "$0")"
EXP=f365833e32edbf3545d0094367c918a54be88bec1c0f6b34cd1d786ce538d26c
: > search_A.log
for th in 1 2 4 8; do
 for core in "" Haswell SkylakeX Cooperlake SapphireRapids Zen Sandybridge Nehalem Prescott; do
  for dis in "" "AVX512_SPR" "AVX512_ICL AVX512_SPR" "X86_V4 AVX512_ICL AVX512_SPR" "X86_V3 X86_V4 AVX512_ICL AVX512_SPR"; do
    d=$(mktemp -d -p .); 
    env ZETA_FOLD_RAW=$PWD/$d OPENBLAS_NUM_THREADS=$th ${core:+OPENBLAS_CORETYPE=$core} NPY_DISABLE_CPU_FEATURES="$dis" ./venv/bin/python -B pkt/stats/rs_zeta.py A 1e4 1e5 >/dev/null 2>$d/err
    h=$(sha256sum $d/zeta_A.npz 2>/dev/null | cut -d' ' -f1); s=$(stat -c %s $d/zeta_A.npz 2>/dev/null)
    echo "th=$th core=${core:-default} dis=[$dis] size=$s sha=${h:0:16}" >> search_A.log
    if [ "$h" = "$EXP" ]; then echo "MATCH th=$th core=${core:-default} dis=[$dis]" >> search_A.log; cp $d/zeta_A.npz zeta_A_MATCH.npz; rm -rf $d; exit 0; fi
    rm -rf $d
  done
 done
done
echo "NO MATCH" >> search_A.log
