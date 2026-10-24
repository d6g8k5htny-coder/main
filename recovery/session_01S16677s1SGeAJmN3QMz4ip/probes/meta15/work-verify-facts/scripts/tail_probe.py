import sys, subprocess, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe
SRC=Path(sys.argv[1]); PY=sys.argv[2]
for n in [int(x) for x in sys.argv[3].split(',')]:
    r = probe.make_ws({'x.json': '{"a":'*n + '{"scientific_status_authority":true}' + '}'*n})
    for opt in ([], ['-O']):
        p = subprocess.run([PY,'-B','-S',*opt,str(SRC/'tools/architecture_conformance.py'),'--workspace',str(r)],capture_output=True,text=True)
        print(n, opt, p.returncode, len(p.stdout), repr(p.stderr.strip().splitlines()[-1]) if p.stderr else '')
