"""Verify the landed query-#46 merge against the qualified head and the tested merge (stdlib only; uses `gh api` and local git).

Usage: python3 -I verify_landed.py
"""
import json, subprocess

REPO = 'repos/d6g8k5htny-coder/query-'
H = '72c3e3f9ce4ec67bf8d881b15fb3d0d9bfe37152'            # qualified head
MAIN_BEFORE = '81efa995dd6c3542d1acc1ea683a25055fd2a130'  # query- main before the merge
TESTED_TREE = '5678e34c95709058c22ab28f26dc4c268ac8b12e'  # local merge-tree = GitHub test-merge 670020b2 tree
REVIEWED = {
    'README.md': 'a66a0a38af6852ec0b770b1378e9a1ec1f18da67',
    'pyproject.toml': '373ba38435e2a72a72e8777245586f91ba6b8c87',
    'setup.py': 'ba60e26b9d85bd46aff2b92aeca5b327d5a969b2',
    'tests/test_install_provenance.py': 'a742a9ef8c4c9090708103c36b2f7a22c29556ae',
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
    'landed tree = tested merge tree 5678e34c': tree == TESTED_TREE,
    'four paths carry the reviewed blobs': blobs == REVIEWED,
    'diff vs main-before = exactly the four reviewed paths': diff == EXPECTED_DIFF,
}
for k, v in checks.items():
    print(('PASS ' if v else 'FAIL ') + k)
print('ALL PASS' if all(checks.values()) else 'SOME FAIL')
