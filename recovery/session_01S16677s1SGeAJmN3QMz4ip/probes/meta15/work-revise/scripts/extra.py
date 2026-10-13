import sys,tempfile; sys.path.insert(0,sys.argv[0].rsplit('/',1)[0]); sys.path.insert(0,'src')
from common import *
from tools.architecture_conformance import check_workspace
PY=sys.argv[1]
K='scientific_status_authority'
cases={'two_true':'{"a":{"%s":true},"b":[{"%s":true}]}'%(K,K),
 'two_mal':'{"a":{"%s":1},"b":{"%s":null}}'%(K,K),
 'case_true':'{"Scientific_Status_Authority":true}','case_1':'{"Scientific_Status_Authority":1}',
 'space_true':'{"scientific_status_authority ":true}','space_1':'{"scientific_status_authority ":1}'}
for k,t in cases.items():
    a=run('src',PY,t,False);b=run('src',PY,t,True)
    with tempfile.TemporaryDirectory() as d:
        ws(Path(d),t); api=check_workspace(Path(d))['violations']
    print(PY,k,'API',api,'| CLI N',short(a),'O',short(b),'stdout==api',a[3]==json.dumps(check_workspace(Path(d)) if False else None) or '')
