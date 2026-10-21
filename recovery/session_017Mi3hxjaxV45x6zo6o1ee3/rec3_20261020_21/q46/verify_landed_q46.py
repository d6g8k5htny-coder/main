"""Verify the landed query-#46 merge against the qualified head and the tested merge (stdlib only; uses `gh api` and local git).

Usage: python3 -I verify_landed_q46.py
"""
import json, subprocess

REPO = 'repos/d6g8k5htny-coder/query-'
H = '9a7d71456fa47e39e6c5834ce2c7b20ae23d10b4'            # qualified head (not refreshed)
MAIN_BEFORE = '81efa995dd6c3542d1acc1ea683a25055fd2a130'  # query- main before the merge
TESTED_TREE = '411d6a3b94d4ff0a795888d810c0f913dfc0e6f3'  # local merge-tree = GitHub test-merge 75916e26 tree
REVIEWED = {
    'setup.py': 'df8e02aeaf30cd96154212e99e44fc72403d10d1',
    'tests/test_install_provenance.py': '7eb8d632343249799a5efd7851418bf08785f0d9',
    'pyproject.toml': '373ba38435e2a72a72e8777245586f91ba6b8c87',
    'README.md': 'e4249794f1de13369d9c1cdc7cb56bae387d0005',
}
EXPECTED_DIFF = sorted(['M\tREADME.md', 'M\tpyproject.toml', 'A\tsetup.py', 'A\ttests/test_install_provenance.py'])


def gh(path):
    return json.loads(subprocess.run(['gh', 'api', path], capture_output=True, text=True, check=True).stdout)


def git(*a):
    return subprocess.run(['git', '-C', '/home/user/query-', *a], capture_output=True, text=True, check=True).stdout.strip()


p = gh(f'{REPO}/pulls/46')
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
    'merge commit is query- main tip': main == M,
    'parents = [main before, qualified head]': parents == [MAIN_BEFORE, H],
    'landed tree = tested merge tree 411d6a3b': tree == TESTED_TREE,
    'four paths carry the reviewed blobs': blobs == REVIEWED,
    'diff vs main-before = exactly the four reviewed paths': diff == EXPECTED_DIFF,
}
for k, v in checks.items():
    print(('PASS ' if v else 'FAIL ') + k)
print('ALL PASS' if all(checks.values()) else 'SOME FAIL')
