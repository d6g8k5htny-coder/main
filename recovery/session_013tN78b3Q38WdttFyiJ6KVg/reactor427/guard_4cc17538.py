import hashlib
import json
from pathlib import Path
import os
import subprocess

closure_path = "experiments/reactor_v0/closure.json"
closure_file = Path(closure_path)
package = Path("experiments/reactor_v0")
package_entries = list(package.rglob("*"))
if (closure_file.is_symlink() or package.is_symlink() or
        any(path.is_symlink() for path in package_entries)):
    raise SystemExit("frozen package may not contain symlinks")
base = os.environ["BASE_SHA"]
base_commit = subprocess.run(
    ["git", "cat-file", "-e", base + "^{commit}"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    check=False,
)
if base_commit.returncode != 0:
    raise SystemExit("pull-request base is not an available commit")
base_path = subprocess.run(
    ["git", "ls-tree", "-z", base, "--", closure_path],
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
    check=False,
)
if base_path.returncode != 0:
    raise SystemExit("unable to inspect frozen closure at pull-request base")
if not base_path.stdout:
    print("Base has no Reactor A freeze; introduction is permitted.")
    raise SystemExit(0)
prior = subprocess.run(
    ["git", "show", base + ":" + closure_path],
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
    check=False,
)
if prior.returncode != 0:
    raise SystemExit("unable to read frozen closure at pull-request base")
current = closure_file.read_bytes()
if current != prior.stdout:
    raise SystemExit("frozen closure.json differs from pull-request base")
closure = json.loads(prior.stdout)
expected_package = {
    Path(path).relative_to("experiments/reactor_v0").as_posix()
    for path in closure
    if path.startswith("experiments/reactor_v0/")
} | {"closure.json"}
actual_package = {
    path.relative_to(package).as_posix()
    for path in package_entries
    if not path.is_dir()
}
if actual_package != expected_package:
    raise SystemExit("frozen package file set differs from pull-request base")
for relative, expected in closure.items():
    path = Path(relative)
    if path.is_symlink():
        raise SystemExit("frozen closure member may not be a symlink: " + relative)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit("frozen byte mismatch: " + relative)
