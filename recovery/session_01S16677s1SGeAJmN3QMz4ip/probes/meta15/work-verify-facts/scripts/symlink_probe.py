import sys, json, subprocess, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe import make_ws
SRC=Path(sys.argv[1]); PY=sys.argv[2]
outside = Path(__file__).resolve().parents[1]/'ws'/'outside'; outside.mkdir(parents=True, exist_ok=True)
(outside/'o.json').write_text('{"scientific_status_authority":true}')
(outside/'d').mkdir(exist_ok=True); (outside/'d'/'in.json').write_text('{"scientific_status_authority":true}')
def run(root, label):
    p=subprocess.run([PY,'-B','-S',str(SRC/'tools/architecture_conformance.py'),'--workspace',str(root)],capture_output=True,text=True)
    print(label, p.returncode, json.loads(p.stdout)['violations'] if p.stdout else p.stderr[-100:])
r=make_ws({'keep.json':'{}'}); (r/'meta-framework/generated/link.json').symlink_to(outside/'o.json'); run(r,'file symlink outside:')
r=make_ws({'keep.json':'{}'}); (r/'meta-framework/generated/ld').symlink_to(outside/'d', target_is_directory=True); run(r,'dir symlink:')
r=make_ws({'keep.json':'{}'}); (r/'meta-framework/generated/dir.json').mkdir(); run(r,'dir named dir.json:')
r=make_ws({}); (r/'meta-framework/generated').write_text('{"scientific_status_authority":true}'); run(r,'regular file generated:')
