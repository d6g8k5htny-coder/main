import sys,tempfile,json
sys.path.insert(0,sys.argv[1]); sys.path.insert(0,sys.argv[0].rsplit('/',1)[0])
from common import ws,REPOS
from pathlib import Path
from tools.architecture_conformance import check_workspace
K='"scientific_status_authority"'
A='generated scientific-status authority: meta-framework/generated/x.json'
Mv='generated scientific-status marker type invalid: meta-framework/generated/x.json'
old={}
vals=['0','1','2','0.0','-0.0','1.0','NaN','""','"false"','"true"','"0"','"x"','null','[]','[true]','{}','{"a":1}']
for v in vals:
    old['root '+v]=('{'+K+':'+v+'}',[Mv]); old['nested '+v]=('{"nested":[{'+K+':'+v+'}]}',[Mv])
old['true root']=('{'+K+':true}',[A]); old['true nested']=('{"a":{'+K+':true}}',[A]); old['true rootarr']=('[{'+K+':true}]',[A]); old['true arrdict']=('{"i":[{'+K+':true}]}',[A])
old['obj marker holding true']=('{'+K+':{'+K+':true}}',[A,Mv]); old['list marker holding true']=('{'+K+':[{'+K+':true}]}',[A,Mv])
old['sib mal-true']=('{"a":{'+K+':1},"b":{'+K+':true}}',[A,Mv]); old['sib true-mal']=('{"b":{'+K+':true},"a":{'+K+':1}}',[A,Mv])
old['depth900 true']=('{"a":'*900+'{'+K+':true}'+'}'*900,[A]); old['depth900 mal']=('{"a":'*900+'{'+K+':1}'+'}'*900,[Mv])
old['pos absent']=('{"k":"v"}',[]); old['pos false root']=('{'+K+':false}',[]); old['pos false nested']=('{"a":[{'+K+':false}]}',[]); old['pos ordinary']=('{"a":{"b":[1,2,{"c":null}]}}',[]); old['pos root true']=('true',[])
old['pos case false']=('{"Scientific_Status_Authority":false}',[]); old['pos space false']=('{"scientific_status_authority ":false}',[])
new={'two true':('{"a":{'+K+':true},"b":[{'+K+':true}]}',[A]),'two mal':('{"a":{'+K+':1},"b":{'+K+':null}}',[Mv]),
 'case true':('{"Scientific_Status_Authority":true}',[]),'case 1':('{"Scientific_Status_Authority":1}',[]),
 'space true':('{"scientific_status_authority ":true}',[]),'space 1':('{"scientific_status_authority ":1}',[])}
def check(cases):
    fails=[]
    for name,(text,exp) in cases.items():
        with tempfile.TemporaryDirectory() as d:
            ws(Path(d),text)
            try: got=check_workspace(Path(d))['violations']
            except RecursionError: got='RecursionError'
        if got!=exp: fails.append(name)
    return fails
print(sys.version.split()[0],sys.argv[1],'old A6 fails:',check(old),'| new cases fail:',check(new))
