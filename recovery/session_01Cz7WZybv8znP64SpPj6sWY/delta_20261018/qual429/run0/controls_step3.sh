set -euo pipefail
python3 -B -E -S experiments/universality/run_pilot.py --output "$RUNNER_TEMP/universality-exploration" 2>&1 | tee "$RUNNER_TEMP/universality-exploration-run.log"
python3 -B -E -S experiments/universality/run_pilot.py --verify "$RUNNER_TEMP/universality-exploration/observations.json" 2>&1 | tee "$RUNNER_TEMP/universality-exploration-verify.log"
