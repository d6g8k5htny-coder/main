set -euo pipefail
python3 -B -E -S experiments/universality/verify_published.py experiments/universality/results/exploratory50 2>&1 | tee "$RUNNER_TEMP/universality-published-verify.log"
