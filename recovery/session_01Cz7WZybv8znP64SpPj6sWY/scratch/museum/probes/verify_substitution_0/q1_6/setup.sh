#!/bin/sh
# Offline fixture for the Q1 probes. Read-only git object reads; no network.
# PROOF_INDEX.md@d6628da: the exact bytes embedded in cd32303 tests/test_museum_data.py (identity checked).
# Other Math-@d6628da sources: blob objects from the local shallow Math- clone (sha256 checked).
# EC-014 export: blob object from the local main clone. README.md@d6628da is NOT available offline.
set -e
H=<SCRATCH>/museum/probes/verify_substitution_0/q1_6
T=<SCRATCH>/museum/cd32303
cd "$H"; rm -rf fixture
python3 -I extract_index.py "$T/tests/test_museum_data.py" PROOF_INDEX.md
python3 -I locate_sources.py "$T/tools/museum_data.py" /home/user/Math- ./fixture | grep -c '^OK' | sed 's/^/Math- sources byte-verified from local objects: /'
cp PROOF_INDEX.md fixture/Math-/PROOF_INDEX.md
mkdir -p fixture/Math-/imports/hardening_ebedb780/EC-014
git -C /home/user/main cat-file -p 7189b4d1538866594c6d8c94f09e247a2cd65abe > fixture/Math-/imports/hardening_ebedb780/EC-014/proof.md.export.txt
git -C /home/user/Math- show 10e1f191c7d9f2755ca971ddbba87874d2475619:README.md > readme_10e1f191.md
sha256sum fixture/Math-/imports/hardening_ebedb780/EC-014/proof.md.export.txt readme_10e1f191.md
