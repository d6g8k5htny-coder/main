# Reproduction commands

The current replay environment was Python 3.13.5 with the dependencies in `requirements.txt`. The received Kimi environment lock is separately preserved. Matching outputs were reproduced across those environments; environment identity is not claimed.

From this package directory, run:

    python tools/verify_bundle.py
    python tools/check_findings.py
    python -O tools/check_findings.py
    python tools/replay_received.py --group quick --output /your/new/scratch_directory

`--group all` replays seven received programs in both modes; timeout defaults to 240 seconds PER PROGRAM. The findings checker tests this audit's exact arithmetic/custody findings, NOT the original mathematical theorem. The source replay can PASS even while the mathematical defects described in the audit remain.

No command writes Drive. The replay runner copies received sources to a temporary tree, writes receipts outside this sealed package, does not run the old superseded sine-omission script, and does not launch W8's expensive full sweep. W8 input/preflight instructions are in the Drive supplement README.
