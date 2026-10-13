#!/bin/sh
# Re-run every fail-open probe against the extracted cd32303 bytes. Offline. Writes ALL_OUTPUT.txt.
# Fixture: ../pointers/fixture/verified-inputs.json is re-verified here against the cd32303 SOURCES pins before use.
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"; export PATH=/opt/node22/bin:$PATH; unset MUSEUM_TREE
{
echo "##### verify_fixture"; python3 -I verify_fixture.py ../../cd32303 ../pointers/fixture/verified-inputs.json ./fixture
echo "##### p0_control"; node p0_control.mjs
echo "##### p1_availability_unbound"; node p1_availability_unbound.mjs
echo "##### p2_pointer_only_truthy"; node p2_pointer_only_truthy.mjs
echo "##### p3_mutant_impact"; python3 -I p3_mutant_impact.py
echo "##### p4_status_quote_default"; python3 -I p4_status_quote_default.py ../../cd32303 ./fixture
echo "##### p5_replay_unbound"; node p5_replay_unbound.mjs
echo "##### p6_generator_path_ci_green"; python3 -I p6_generator_path_ci_green.py
echo "##### p7_banner_and_count"; node p7_banner_and_count.mjs
echo "##### p8_museum_check_skip_is_pass"; sh p8_museum_check_skip_is_pass.sh
echo "##### p9_main_persistence (informational, current main 513261bb)"; git -C /home/user/main cat-file blob 3aeb0412f7a9d4357b91977179bf2d608826412c > work/main_status_f2e432ea.md; node p9_main_persistence.mjs
echo "##### mutants"; python3 -I mutants.py
} 2>&1 | tee ALL_OUTPUT.txt
