import sys, json
sys.path.insert(0,'/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad')
from gh import get
out=[]
page=1
while True:
    data,h=get(f'/repos/d6g8k5htny-coder/Math-/issues/363/comments?per_page=100&page={page}')
    out+=data
    if len(data)<100: break
    page+=1
json.dump(out,open('/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/crawl/math363verify/c363.json','w'))
print(len(out))
for c in out:
    print(c['id'],c['created_at'],c['updated_at'],len(c['body']), c['body'][:120].replace('\n',' '))
