#!/bin/sh
# Verifier-1 probes for Q3-F2, against the cd32303 bytes (read-only), no network, no state-changing git.
H=$(cd "$(dirname "$0")" && pwd); cd "$H"
echo "##### V0 static + snapshot facts"; python3 -I -B v0_static_and_snapshot.py
echo "##### V1 control: unmodified cd32303, whole CI museum step"; ./mkwt.sh wt_pin >/dev/null && ./v1_ci_step.sh wt_pin
echo "##### V1(A) complete refresh 71400b94 -> a12c178c, whole CI museum step"; ./mkwt.sh wt_a12_all a12c178c0f857a130cf434e9efd44233a038195b all && ./v1_ci_step.sh wt_a12_all
echo "##### V1(B) partial refresh (tools/museum_data.py only), whole CI museum step"; ./mkwt.sh wt_a12_datapy a12c178c0f857a130cf434e9efd44233a038195b datapy && ./v1_ci_step.sh wt_a12_datapy
echo "##### V2(A) strict render of the complete-refresh manifest"; /opt/node22/bin/node v2_render.mjs "$H/wt_a12_all" "$H/wt_a12_all/docs/site/museum.json"
python3 -I -c "import json,sys;d=json.load(open(sys.argv[1]));d['packets']=[];open(sys.argv[2],'w').write(json.dumps(d))" "$H/../../../cd32303/docs/site/museum.json" "$H/cd32303_packets_empty.json"
echo "##### V2(C) strict render of the real cd32303 manifest with packets:[]"; /opt/node22/bin/node v2_render.mjs "$H/wt_pin" "$H/cd32303_packets_empty.json"
echo "##### V3 calibration (what CI catches)"; ./v3_calibration.sh
