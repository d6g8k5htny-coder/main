"""Verify the landed main#466 merge against the qualified head and the tested merge (stdlib only; uses `gh api` and local git).

Usage: python3 -I verify_landed.py
"""
import json, subprocess

REPO = 'repos/d6g8k5htny-coder/main'
H = '3dd0be245634f3b9f75bd90709993183c98b7f1c'            # qualified (refreshed) head
MAIN_BEFORE = '4c7354e5e629e8ded3cdc9b7ad33b47c22bfe49d'  # main before the merge
TESTED_TREE = '7af68e5ba6b8cb2cd32fe584c332f5df85a43434'  # head tree = local merge-tree of 8ba9bf2a and 4c7354e5
REVIEWED = {'docs/WORKSPACE.md': '1da250d10cddadf722ec2f7510b637e4d1f0b842'}
EXPECTED_DIFF = ['M\tdocs/WORKSPACE.md']


def gh(path):
    return json.loads(subprocess.run(['gh', 'api', path], capture_output=True, text=True, check=True).stdout)


def git(*a):
    return subprocess.run(['git', '-C', '/home/user/main', *a], capture_output=True, text=True, check=True).stdout.strip()


p = gh(f'{REPO}/pulls/466')
print('merged', p['merged'], 'merged_at', p['merged_at'], 'merge_commit', p['merge_commit_sha'])
M = p['merge_commit_sha']
c = gh(f'{REPO}/commits/{M}')
parents = [x['sha'] for x in c['parents']]
tree = c['commit']['tree']['sha']
main = gh(f'{REPO}/branches/main')['commit']['sha']
print('main', main, '| parents', parents, '| tree', tree, '| message', c['commit']['message'].splitlines()[0])
git('fetch', '-q', 'origin', 'main')
blobs = {path: git('rev-parse', f'{M}:{path}') for path in REVIEWED}
diff = sorted(git('diff', '--name-status', MAIN_BEFORE, M).splitlines())
checks = {
    'merge commit is main tip': main == M,
    'parents = [main before, qualified head]': parents == [MAIN_BEFORE, H],
    'landed tree = tested tree 7af68e5b': tree == TESTED_TREE,
    'docs/WORKSPACE.md carries the qualified blob': blobs == REVIEWED,
    'diff vs main-before = exactly docs/WORKSPACE.md': diff == EXPECTED_DIFF,
}
for k, v in checks.items():
    print(('PASS ' if v else 'FAIL ') + k)
print('ALL PASS' if all(checks.values()) else 'SOME FAIL')
