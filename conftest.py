"""Keep the frozen command-style timestamp harness out of automatic collection.

It consumes a source-file argument at import time. Preserve the packet bytes and
run its eight controls using the standalone commands documented in its README.
Only this exact path is excluded; neighboring tests remain discoverable.
"""

import sys

# Frozen packages reject generated files. Prevent new import-time bytecode after
# this root configuration loads; do not delete or exempt pre-existing caches.
# Standalone and child Python commands still need their explicit -B flags.
sys.dont_write_bytecode = True

collect_ignore = [
    "experiments/board_capacity_probe_20261022/test_timestamp_offsets.py",
]
