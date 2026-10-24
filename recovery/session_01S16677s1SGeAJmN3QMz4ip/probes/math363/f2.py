import sys, json
sys.path.insert(0,'/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad')
from gh import get
D='/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/crawl/math363verify/'
pr,h=get('/repos/d6g8k5htny-coder/Math-/pulls/363')
print('PR', pr['state'], pr['draft'], pr['head']['sha'], pr['base']['sha'], pr['mergeable_state'], pr['merged'], pr['updated_at'])
print('REMAIN', h.get('X-RateLimit-Remaining') if hasattr(h,'get') else h)
for p in ('/repos/d6g8k5htny-coder/main/issues/307/comments?since=2026-10-11T18:00:00Z&per_page=100',
          '/repos/d6g8k5htny-coder/main/issues/275/comments?since=2026-10-11T18:00:00Z&per_page=100',
          '/repos/d6g8k5htny-coder/Math-/issues/363/timeline?per_page=100'):
    d,h=get(p)
    key=p.split('/issues/')[1].split('?')[0].replace('/','_')
    json.dump(d,open(D+key+'.json','w'))
    print('====',p,len(d))
    for c in d:
        if 'timeline' in p:
            print(c.get('event'),c.get('created_at') or c.get('submitted_at') or (c.get('committer') or {}).get('date'), c.get('sha') or c.get('id'))
        else:
            print(c['id'],c['created_at'],c['updated_at'],c['body'][:200].replace('\n',' '))
