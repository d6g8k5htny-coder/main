#!/usr/bin/env bash
# Source only from the task's run.sh. Preserve HOME, proxies, and CA variables.
AT_ROOT=/workspace/scratch/atlas-summary-at
export PATH="$AT_ROOT/runtime/usr/bin:$PATH"
export LD_LIBRARY_PATH="$AT_ROOT/runtime/usr/lib/x86_64-linux-gnu:$AT_ROOT/runtime/usr/lib/x86_64-linux-gnu/pulseaudio:$AT_ROOT/runtime/usr/lib/pulse-17.0+dfsg1/modules${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$AT_ROOT/runtime/usr/lib/python3/dist-packages${PYTHONPATH:+:$PYTHONPATH}"
export GI_TYPELIB_PATH="$AT_ROOT/runtime/usr/lib/x86_64-linux-gnu/girepository-1.0${GI_TYPELIB_PATH:+:$GI_TYPELIB_PATH}"
export XDG_DATA_DIRS="$AT_ROOT/runtime/usr/share:${XDG_DATA_DIRS:-/usr/local/share:/usr/share}"
export ESPEAK_DATA_PATH="$AT_ROOT/runtime/usr/lib/x86_64-linux-gnu/espeak-ng-data"
export SPEECHD_PLUGIN_DIR="$AT_ROOT/runtime/usr/lib/x86_64-linux-gnu/speech-dispatcher"
export GTK_MODULES=atk-bridge
export NO_AT_BRIDGE=0
export ACCESSIBILITY_ENABLED=1
export LANG=C.UTF-8
