#!/usr/bin/env python3
"""Read-only collector for the hosted runs on Math-#447 head 077a168b: run/job outcomes, artifacts (name, id, size, GitHub digest),
and the actual tested merge commit (from each job's checkout log 'HEAD is now at'), written to runs447.json. Prints the binding
sentence for the PR body and the #307 post. Scientific effect NONE: a run is a record of a computation, never a review or an
acceptance. usage: python3 bind_runs447.py"""
import json, re, subprocess, sys
from pathlib import Path
S = Path('<SCRATCH>')
import os
REPO = 'd6g8k5htny-coder/Math-'; HEAD = os.environ.get('R7_HEAD', '077a168ba9fe3a0888c2d1988f85511c30ca9484'); BASE_HEAD = '9fb0a4a06900da5cc3f9126f30e117a747df1587'
def api(path):
    r = subprocess.run(['gh', 'api', path], capture_output=True, text=True); assert r.returncode == 0, (path, r.stderr[:300]); return r.stdout
runs = json.loads(api(f'repos/{REPO}/actions/runs?head_sha={HEAD}&per_page=20'))['workflow_runs']
out = []
for r in sorted(runs, key=lambda x: x['id']):
    jobs = json.loads(api(f'repos/{REPO}/actions/runs/{r["id"]}/jobs'))['jobs']
    arts = json.loads(api(f'repos/{REPO}/actions/runs/{r["id"]}/artifacts'))['artifacts']
    merges = set()
    for j in jobs:
        if j['status'] != 'completed': continue
        log = subprocess.run(['gh', 'api', f'repos/{REPO}/actions/jobs/{j["id"]}/logs'], capture_output=True, text=True, errors='replace').stdout
        for m in re.finditer(r'HEAD is now at ([0-9a-f]{7,40})', log): merges.add(m.group(1))
        for m in re.finditer(r'checked_commit[^0-9a-f]*([0-9a-f]{40})', log): merges.add(m.group(1))
    out.append({'run': r['id'], 'name': r['name'], 'event': r['event'], 'status': r['status'], 'conclusion': r['conclusion'], 'attempt': r['run_attempt'],
                'head_sha': r['head_sha'], 'url': r['html_url'], 'jobs': [{'id': j['id'], 'name': j['name'], 'status': j['status'], 'conclusion': j['conclusion']} for j in jobs],
                'artifacts': [{'name': a['name'], 'id': a['id'], 'size': a['size_in_bytes'], 'digest': a.get('digest')} for a in arts],
                'merge_candidates_from_logs': sorted(merges)})
(S / f'lvs/r7/runs447_{HEAD[:8]}.json').write_text(json.dumps(out, indent=1) + '\n')
for o in out:
    print(o['run'], o['name'][:48], o['status'], o['conclusion'], 'jobs', [(j['name'][:24], j['conclusion']) for j in o['jobs']], 'arts', [(a['name'], a['digest']) for a in o['artifacts']], 'merge', o['merge_candidates_from_logs'])
