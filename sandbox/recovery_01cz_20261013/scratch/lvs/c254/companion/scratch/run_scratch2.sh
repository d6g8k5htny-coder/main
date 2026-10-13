#!/usr/bin/env bash
set -u
export PATH=$HOME/.elan/bin:$PATH
S=<SCRATCH>
cd "$S/repos/math_c254/formal" || exit 99
exec lake env bash "$S/lvs/c254/companion/scratch/inner.sh" "$@"
