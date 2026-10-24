#!/bin/bash
S=${SCRATCH:?set SCRATCH to the working directory}/ci-pre
CLONE=${CLONE:?set CLONE to a Math- clone with pr/<N> refs}
BASE=46f2a289de2dbdb6d3cf3b03986eeb2f3f681975
OUT=$S/full
mkdir -p $OUT
PRS="312 316 328 330 336 344 348 351 353 354 361 364 366 367 369 370 372 379 380 385 389 391 397 405 414 415 424 425 426 427 430 444 451 455"
cd $S
SHIM_DIR=$S/../rr-mut/shim WORKERS=3 python3 -B prescreen.py $CLONE $BASE $OUT $PRS
python3 -B -c "
import json,glob
u=set()
for f in glob.glob('$OUT/target-*.json'):
    d=json.load(open(f)); u|=set(d.get('selected',[]))
json.dump(sorted(u), open('$OUT/base_union.json','w')); print('union', len(u))"
SHIM_DIR=$S/../rr-mut/shim WORKERS=1 python3 -B prescreen.py $CLONE $BASE $OUT 0
echo ALL_DONE
