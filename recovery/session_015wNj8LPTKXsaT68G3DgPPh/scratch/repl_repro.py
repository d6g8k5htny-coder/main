import importlib.util, pathlib, tempfile, subprocess, hashlib, os
p = pathlib.Path("frontiers/ordinary_short_bar_occurrence_all_d_20261001/verify_sources.py").resolve()
s = importlib.util.spec_from_file_location("v", p); v = importlib.util.module_from_spec(s); s.loader.exec_module(v)
with tempfile.TemporaryDirectory() as d:
    root = pathlib.Path(d); v.ROOT = root
    env = dict(os.environ, GIT_AUTHOR_NAME="F", GIT_COMMITTER_NAME="F", GIT_AUTHOR_EMAIL="f@example.invalid", GIT_COMMITTER_EMAIL="f@example.invalid")
    env.pop("GIT_NO_REPLACE_OBJECTS", None)
    def git(*a, data=None):
        return subprocess.run(["git", *a], cwd=root, env=env, input=data, capture_output=True, check=True).stdout.strip().decode()
    git("init", "-q"); empty = git("mktree", data=b""); wrong = git("commit-tree", empty, data=b"empty original\n")
    (root/"P.md").write_bytes(b"fixture only\n"); git("add", "P.md"); tree = git("write-tree"); good = git("commit-tree", tree, data=b"source\n")
    blob = git("rev-parse", good+":P.md"); data = (root/"P.md").read_bytes()
    e = dict(key="P", commit=wrong, path="P.md", blob=blob, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    try: v.historical(e)
    except ValueError: print("CONTROL_REJECTED")
    git("replace", wrong, good)
    try:
        v.historical(e); print("BUG_REPLACEMENT_ACCEPTED")
    except ValueError: print("FIXED_REPLACEMENT_REJECTED")
