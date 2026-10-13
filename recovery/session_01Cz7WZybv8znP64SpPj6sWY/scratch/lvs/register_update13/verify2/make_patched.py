"""Make dry-run copies of render_docs13.py and edit_proof_docs13.py that read the docs, module and register from a
scratch mirror (docs restored to 4afcca6a) and answer only the HEAD / status / ls-files questions with the round-6
pre-commit state; every other git call goes read-only to the worktree with GIT_OPTIONAL_LOCKS=0. No other change."""
from pathlib import Path
H = Path('<SCRATCH>/lvs/register_update13')
V = H / 'verify2'
M = V / 'mirror'

def patch(src, dst, reps):
    t = src.read_text(encoding='utf-8')
    for old, new in reps:
        assert t.count(old) == 1, (src.name, old)
        t = t.replace(old, new)
    dst.write_text(t, encoding='utf-8')

patch(H / 'render_docs13.py', V / 'render_docs13_dry.py', [
    ("PK = REPO / PACKAGE\n", f"PK = Path('{M}') / PACKAGE   # DRY: mirror\n"),
    ("OUT = HERE / 'sdocs'\n", f"OUT = Path('{V}/sdocs_dry')   # DRY\n"),
    ("def git(*args):\n    return subprocess.run(['git', '-C', str(REPO), *args], check=True, capture_output=True, text=True, encoding='utf-8').stdout\n",
     "def git(*args):\n"
     "    if args == ('rev-parse', 'HEAD'):\n        return EXPECTED_HEAD + '\\n'   # DRY: pre-commit HEAD\n"
     "    if args[:2] == ('status', '--porcelain'):\n        return f' M {PACKAGE}/REGISTER.json\\n'   # DRY: pre-commit status\n"
     "    return subprocess.run(['git', '-C', str(REPO), *args], check=True, capture_output=True, text=True, encoding='utf-8',\n"
     "                          env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0'}).stdout\n"),
])
patch(H / 'edit_proof_docs13.py', V / 'edit_proof_docs13_dry.py', [
    ("PACKET = CLONE / PACKET_REL\n", f"PACKET = Path('{M}') / PACKET_REL   # DRY: mirror\n"),
    ("def git(*args):\n    return subprocess.run(['git', '-C', str(CLONE), *args], check=True, capture_output=True, text=True, encoding='utf-8').stdout\n",
     "def git(*args):\n"
     "    if args == ('rev-parse', 'HEAD'):\n        return HEAD + '\\n'   # DRY: pre-commit HEAD\n"
     "    if args == ('ls-files', f'{PACKET_REL}/{MODULE}'):\n        return ''   # DRY: pre-commit, module untracked\n"
     "    return subprocess.run(['git', '-C', str(CLONE), *args], check=True, capture_output=True, text=True, encoding='utf-8',\n"
     "                          env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0'}).stdout\n"),
])
print('ok')
