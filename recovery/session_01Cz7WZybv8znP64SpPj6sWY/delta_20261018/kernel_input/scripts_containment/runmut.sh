#!/bin/bash
cd <SCRATCH>/kern/work_containment
printf '%s\n' fixtures/point_certificate.TEST_ONLY.json fixtures/interval_certificate.TEST_ONLY.json out_sweep/105.json out_sweep/109.json out_sweep/201.json out_sweep/210.json out_sweep/104.json out_sweep/290.json out_sweep/323.json out_sweep/326.json out_sweep/328.json out_sweep/350.json out_sweep/359.json | xargs -P 4 -I{} sh -c 'b=$(basename {} .json); python3 -I -u ../scripts_containment/mutate.py . mut/$b {} > mut/$b.log 2>&1'
echo ALLDONE
