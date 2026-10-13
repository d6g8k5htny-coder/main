import json,os,urllib.request,sys
from gh import get
R='/repos/d6g8k5htny-coder/Math-'
H='3a35410ded9ab042ea9df37ebf9d17712bbd0543'
pr,_=get(f'{R}/pulls/371')
m,_=get(f'{R}/git/ref/heads/main')
cr,_=get(f'{R}/commits/{H}/check-runs?per_page=100')
st={c['name']:(c['status'],c['conclusion']) for c in cr['check_runs']}
rc,_=get(f'{R}/pulls/371/comments?per_page=100')
print('head',pr['head']['sha'],'main',m['object']['sha'],'mergeable',pr['mergeable'],pr['mergeable_state'],'inline',len(rc))
print(st)
assert pr['head']['sha']==H and pr['state']=='open' and not pr['draft']
assert m['object']['sha']=='a0323b94ae54a78f3d2878dc9c358dc40cdfe1dc', 'main moved: reconcile first'
assert st.get('math-downstream-gates')==('completed','success')
assert all(v==('completed','success') for v in st.values()), 'a check is not success'
assert len(rc)==0
if '--go' not in sys.argv: print('preflight OK; not merging'); sys.exit(0)
body={'sha':H,'merge_method':'merge',
 'commit_title':'Merge pull request #371 from d6g8k5htny-coder/chatgpt/d2-sharp-scalar-publication-20261006-r5',
 'commit_message':'D2: publish Part B sharper scalar floor with exact source-bound controls\n\nExpected-head merge of carried head 3a35410d (parents ce96fa19, a0323b94) by the integrator\nactivated in Math-#371 comment 6129539436; math-downstream-gates run 38286539699 success.\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01S16677s1SGeAJmN3QMz4ip'}
req=urllib.request.Request('https://api.github.com'+R+'/pulls/371/merge',data=json.dumps(body).encode(),method='PUT',headers={'Authorization':'Bearer '+os.environ['GH_TOKEN'],'Accept':'application/vnd.github+json','Content-Type':'application/json'})
try:
    r=urllib.request.urlopen(req,timeout=60); print(r.status,r.read().decode())
except urllib.error.HTTPError as e:
    print('HTTP',e.code,e.read().decode()[:1000])
