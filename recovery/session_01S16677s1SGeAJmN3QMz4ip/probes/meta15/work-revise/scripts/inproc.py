import sys,tempfile
sys.path.insert(0,sys.argv[1]); sys.path.insert(0,sys.argv[0].rsplit('/',1)[0])
from common import ws,deep,M
from pathlib import Path
from tools.architecture_conformance import check_workspace
for n in [int(x) for x in sys.argv[2:]]:
    with tempfile.TemporaryDirectory() as t:
        ws(Path(t),deep(n,'{'+M+':true}'))
        try:
            r=check_workspace(Path(t)); print(sys.version.split()[0],'n',n,'API',[v.split(':')[0] for v in r['violations']])
        except RecursionError as e: print(sys.version.split()[0],'n',n,'API raises RecursionError',e)
