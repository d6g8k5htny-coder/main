import importlib.util, json, pathlib, subprocess, sys, tempfile
from unittest import mock
src = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('rs', src); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
# Control 1: keyed route, duplicate id across pages with total_count matching -> pages() accepts
def get(p):
    if 'page=1&' in p: return {'total_count': 101, 'workflow_runs': [{'id': i} for i in range(1, 101)]}
    return {'total_count': 101, 'workflow_runs': [{'id': 100}]}   # id 100 repeated, id 101 never seen
rows = m.pages(get, 'repos/x/actions/runs?head_sha=a', 'workflow_runs')
print('C1 keyed dup-across-pages accepted by pages():', len(rows), 'rows,', len({r['id'] for r in rows}), 'unique ids')
# Control 2: duplicate valid PR number through mocked collect()
with tempfile.TemporaryDirectory() as d:
    root = pathlib.Path(d); fx = root/'fx'; fx.mkdir()
    g = lambda *a: subprocess.check_output(['git','-C',str(fx),*a],stderr=subprocess.DEVNULL).decode().strip()
    g('init','-q'); g('config','user.email','t@example.invalid'); g('config','user.name','t')
    (fx/'a.txt').write_text('a'); g('add','.'); g('commit','-qm','x'); sha = g('rev-parse','HEAD')
    calls = {'pulls/7': 0}
    class API:
        def get(self, path):
            res = '/'.join(path.split('/')[3:]).split('?')[0]; name = path.split('/')[2]
            if not res: return {'private': False, 'full_name': f'{m.OWNER}/{name}', 'default_branch': 'main'}
            if res == 'commits/main': return {'sha': sha}
            if res == 'pulls': return [{'number': 7, 'v': 'first'}, {'number': 7, 'v': 'second'}]
            if res == 'pulls/7':
                calls['pulls/7'] += 1
                return {'number': 7, 'title': f'call{calls["pulls/7"]}', 'head': {'sha': sha, 'repo': {'full_name': f'{m.OWNER}/{name}'}},
                        'base': {'sha': sha}, 'user': {'login': 'u'}}
            if res.endswith('check-runs'): return {'check_runs': [], 'total_count': 0}
            if res == 'actions/runs': return {'workflow_runs': [], 'total_count': 0}
            if res.endswith('/status'): return {'state': 'success'}
            return []
    out = root/'out'
    with mock.patch.object(m,'REPOS',('main',)), mock.patch.object(m,'Client',API), mock.patch.object(m,'repo_url',return_value=str(fx)):
        r = m.collect(out)
    repo = r['repositories'][0]
    print('C2 duplicate PR#7: complete=', r['complete'], 'errors=', len(r['errors']),
          'pull_requests entries=', len(repo['pull_requests']), 'detail fetches=', calls['pulls/7'],
          'pr-7/detail.json title=', json.loads((out/'main'/'pr-7'/'detail.json').read_text())['title'],
          'open-prs.json rows=', len(json.loads((out/'main'/'open-prs.json').read_text())))
