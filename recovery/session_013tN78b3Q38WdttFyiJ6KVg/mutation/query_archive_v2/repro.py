"""Reproduce query-#47's three portability findings against ArchiveRefusalPins, and check a fix.

Usage: QUERY_REPO=<query- clone> python3 -I repro.py TREE [CLASS_FILE]
TREE is a checkout of query-#47 (head 4fb15c97); each run uses a fresh detached worktree of its HEAD. If CLASS_FILE is given, the ArchiveRefusalPins class
in a copy of TREE's tests/test_source_release.py is replaced by that file's text first.
Each configuration runs `python3 -B -S [-O] -m unittest tests.test_source_release` in a fresh copy:
  baseline     : this container's environment
  ignores      : a global excludes file ignoring __pycache__/, *.pyc and Sandbox/ (finding 1)
  no_sha256    : a git wrapper that refuses --object-format=sha256, like a Git without SHA-256 (finding 2)
  ancestor_repo: TMPDIR inside an unrelated Git worktree (finding 3)
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TREE = Path(sys.argv[1]).resolve()
CLASS_FILE = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else None
TEST = "tests/test_source_release.py"
REAL_GIT = shutil.which("git")


REPO = Path(os.environ.get("QUERY_REPO", TREE)).resolve()
HEAD = subprocess.run([REAL_GIT, "-C", str(TREE), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


def tree_copy(tmp):
    # Some existing tests run `git ls-files` on the checkout, so each copy is a detached worktree of HEAD.
    root = Path(tmp) / "t"
    subprocess.run([REAL_GIT, "-C", str(REPO), "worktree", "add", "-q", "--detach", str(root), HEAD], check=True)
    if CLASS_FILE:
        text = (root / TEST).read_text()
        start = text.index("\n\nclass ArchiveRefusalPins(unittest.TestCase):")
        nxt = re.search(r"\n\n\nclass |\n\nclass ", text[start + 10:])
        end = start + 10 + nxt.start() if nxt else len(text)
        new = text[:start] + "\n\n" + CLASS_FILE.read_text().strip("\n") + "\n" + text[end:]
        (root / TEST).write_text(new)
    return root


def configure(name, tmp):
    env = dict(os.environ)
    home = Path(tmp) / "home"
    home.mkdir()
    gcfg = home / "gitconfig"
    gcfg.write_text("")
    env["GIT_CONFIG_GLOBAL"] = str(gcfg)  # isolate from this container's global config in every case
    if name == "ignores":
        ign = home / "ignore"
        ign.write_text("__pycache__/\n*.pyc\nSandbox/\n")
        gcfg.write_text("[core]\n\texcludesFile = %s\n" % ign)
    elif name == "no_sha256":
        shim = Path(tmp) / "shim"
        shim.mkdir()
        (shim / "git").write_text(
            "#!/bin/sh\nfor a; do case \"$a\" in --object-format=sha256) "
            "echo 'fatal: unknown option: --object-format=sha256' >&2; exit 129;; esac; done\n"
            "exec %s \"$@\"\n" % REAL_GIT)
        (shim / "git").chmod(0o755)
        env["PATH"] = str(shim) + os.pathsep + env["PATH"]
    elif name == "ancestor_repo":
        outer = Path(tmp) / "outer-worktree"
        outer.mkdir()
        subprocess.run([REAL_GIT, "init", "-q", str(outer)], check=True, env=env)
        (outer / "tmp").mkdir()
        env["TMPDIR"] = str(outer / "tmp")
    return env


def run(root, env, opt):
    cmd = [sys.executable, "-B", "-S"] + (["-O"] if opt else []) + ["-m", "unittest", "tests.test_source_release"]
    r = subprocess.run(cmd, cwd=root, capture_output=True, text=True, env=env, timeout=900)
    tail = [l for l in r.stderr.splitlines() if l.startswith(("Ran ", "OK", "FAILED"))]
    bad = sorted(set(re.findall(r"^(?:FAIL|ERROR): (\w+) \(tests\.test_source_release\.(\w+)", r.stderr, re.M)))
    return r.returncode, " ".join(tail), ["%s.%s" % (c, t) for t, c in bad]


for name in ("baseline", "ignores", "no_sha256", "ancestor_repo"):
    with tempfile.TemporaryDirectory() as tmp:
        root = tree_copy(tmp)
        env = configure(name, tmp)
        for opt in (False, True):
            rc, tail, bad = run(root, env, opt)
            print("%-14s %-3s rc=%d %s %s" % (name, "-O" if opt else "", rc, tail, bad), flush=True)
        subprocess.run([REAL_GIT, "-C", str(REPO), "worktree", "remove", "--force", str(root)], check=True)
