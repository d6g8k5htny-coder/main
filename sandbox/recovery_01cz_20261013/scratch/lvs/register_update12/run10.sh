#!/bin/sh
# usage: run10.sh   (env RS7_DOC optional)
S=<SCRATCH>
exec python3 -B $S/lvs/register_update12/render_spec10.py 423 38149888699 4363b6928c62da242952631c6b0357d096c3b252 cdeb5c73aa03124ed0db16fa14d25fb03a8fa842457fce1e0c6565dbc5ee7ca8 \
  --round4-pr 434 --round4-run 38180705952 --round4-head 48bfcbae7834c2d6f4d3123d85c69d0c5afedfa6 --round4-merge 9833d6d9f9facc335e1a56e243c3144adefe6225 \
  --round4-sha 87309efad9b804c53c543b60fdb466de1f797971417363cc2e8fa1e10eac26bd \
  --round5-pr 439 --round5-run 38195810463 --round5-head c6ce025652f76be6e580fa24947380253450cd3b --round5-merge 1320c5b217e491d301ea140e5cc25b07ff5a21b7 \
  --round5-sha d8abd42be0facd4248647c3c7eb67a264bd19b9b47c6128733b5b5fc93b0a6ff
