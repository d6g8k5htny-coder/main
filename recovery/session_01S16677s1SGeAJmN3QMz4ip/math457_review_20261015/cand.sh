set -euo pipefail
log_dir="$(mktemp -d)"
trap 'rm -rf "$log_dir"' EXIT
python -B -S -m unittest discover -s tests -v 2>&1 | tee "$log_dir/normal.log"
grep -Eq '^Ran [1-9][0-9]* tests? in [0-9]+([.][0-9]+)?s$' "$log_dir/normal.log"
python -B -O -S -m unittest discover -s tests -v 2>&1 | tee "$log_dir/optimized.log"
grep -Eq '^Ran [1-9][0-9]* tests? in [0-9]+([.][0-9]+)?s$' "$log_dir/optimized.log"
