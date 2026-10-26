#!/usr/bin/env bash
set -euo pipefail
cd /workspace/scratch/atlas-summary-at
mkdir -p apt/lists/partial apt/archives/partial runtime
apt_options=(-o Dir::State::lists="$PWD/apt/lists" -o Dir::Cache::archives="$PWD/apt/archives" -o Debug::NoLocking=true)
apt-get "${apt_options[@]}" update > apt-update-local.log 2>&1
apt-get "${apt_options[@]}" --download-only --no-install-recommends -y install orca xvfb xdotool speech-dispatcher speech-dispatcher-espeak-ng espeak-ng pulseaudio python3-pyatspi x11-utils > apt-download.log 2>&1
for package in apt/archives/*.deb; do dpkg-deb -x "$package" runtime; done
sha256sum apt/archives/*.deb > packages.sha256
for package in apt/archives/*.deb; do dpkg-deb -W "$package"; done > packages.tsv
