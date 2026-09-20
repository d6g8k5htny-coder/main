#!/bin/sh
# finishing: wait for -O transcript + identity stamp, excerpt, copy artifacts, HASHES.txt
D=/mnt/agents/output/K3_SIDE24_LB/W8_lambda
W=/mnt/agents/output/19fcef2e-c1c2-8c6c-8000-0f5a243156d9/work
while [ ! -f $D/transcripts/identity_stamp.txt ]; do sleep 20; done
sleep 5
grep -E "region |auxiliary|quotient stability|B3max|scanned sum|E_cert|LAMBDA1|h_kill|margins|sha256|CERTIFICATE|rungs|rung" $D/transcripts/transcript_normal.txt > $D/transcripts/S6_EXCERPT.txt
cp $W/verify_lambda_grid_v2.py $D/
cp $W/mutate_lambda_grid.py $D/
cd $D
sha256sum KIMI-DER-027c_NONCLOSURE.md verify_lambda_grid_v2.py mutate_lambda_grid.py mutation_receipts.txt transcripts/transcript_normal.txt transcripts/transcript_O.txt transcripts/identity_stamp.txt transcripts/S6_EXCERPT.txt > HASHES.txt
echo FINISHED >> HASHES.txt
