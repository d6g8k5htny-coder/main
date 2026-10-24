import json,subprocess,sys,tempfile
from pathlib import Path
sys.path.insert(0,sys.argv[3]); from bis import ws
py,script=sys.argv[1],sys.argv[2]
deep='{"a":'*1000+'{}'+'}'*1000
for label,doc in (('deep-first','{"a":'+deep+',"b":{"scientific_status_authority":true}}'),('true-first','{"b":{"scientific_status_authority":true},"a":'+deep+'}')):
    with tempfile.TemporaryDirectory() as t:
        root=Path(t);ws(root)
        (root/'meta-framework/generated/x.json').write_text(doc)
        r=subprocess.run([py,'-B','-S',script,'--workspace',str(root)],capture_output=True,text=True)
        v=json.loads(r.stdout)['violations'] if r.stdout else None
        print(py,label,r.returncode,(v[0][:45] if v else v),r.stderr.strip().splitlines()[-1][:50] if r.stderr else '')
