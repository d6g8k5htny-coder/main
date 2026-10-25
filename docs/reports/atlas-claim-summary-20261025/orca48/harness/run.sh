#!/usr/bin/env bash
set -euo pipefail
cd /workspace/scratch/atlas-summary-at
source ./env.sh
exec dbus-run-session -- python3 ./session.py "$@"
