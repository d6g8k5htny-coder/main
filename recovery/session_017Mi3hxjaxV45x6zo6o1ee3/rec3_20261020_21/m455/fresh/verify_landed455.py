"""Verify the landed main#455 merge against the qualified head (stdlib only; uses `gh api` and local git).

Usage: python3 -I verify_landed455.py <merged_head> <main_before> <tested_tree>
"""
import json, subprocess, sys
H = sys.argv[1]            # expected PR head that was merged
BASE_BEFORE = sys.argv[2]  # main before the merge
EXP_TREE = sys.argv[3]     # tested-merge tree (= refreshed head tree)
PATH = 'docs/FORMAL_VERIFICATION.md'
REVIEWED_BLOB = 'f47fec155cb27313550012f0762e369a1df113ef'
def gh(path):
    return json.loads(subprocess.run(['gh', 'api', path], capture_output=True, text=True, check=True).stdout)
def git(*a):
    return subprocess.run(['git', '-C', '/home/user/main', *a], capture_output=True, text=True, check=True).stdout.strip()
p = gh('repos/d6g8k5htny-coder/main/pulls/455')
print('merged', p['merged'], 'merged_at', p['merged_at'], 'merged_by', (p.get('merged_by') or {}).get('login'), 'merge_commit', p['merge_commit_sha'])
M = p['merge_commit_sha']
c = gh(f'repos/d6g8k5htny-coder/main/commits/{M}')
parents = [x['sha'] for x in c['parents']]
tree = c['commit']['tree']['sha']
main = gh('repos/d6g8k5htny-coder/main/branches/main')['commit']['sha']
print('main', main, '| parents', parents, '| tree', tree, '| message', c['commit']['message'].splitlines()[0])
git('fetch', '-q', 'origin', 'main')
blob = git('rev-parse', f'{M}:{PATH}')
diff = git('diff', '--name-status', BASE_BEFORE, M).splitlines()
headtree = gh(f'repos/d6g8k5htny-coder/main/commits/{H}')['commit']['tree']['sha']
checks = {
    'merge commit is main tip': main == M,
    'parents = [main before, merged head]': parents == [BASE_BEFORE, H],
    'landed tree = tested-merge tree = head tree': tree == EXP_TREE == headtree,
    f'{PATH} blob = reviewed f47fec15': blob == REVIEWED_BLOB,
    'diff vs main-before = exactly that one modified path': diff == [f'M\t{PATH}'],
}
for k, v in checks.items():
    print(('PASS ' if v else 'FAIL ') + k)
print('ALL PASS' if all(checks.values()) else 'SOME FAIL')
