#!/usr/bin/env bash

echo '=================== Q1-1'
( cd <SCRATCH>/museum && env -u GIT_DIR -u GIT_WORK_TREE GIT_CEILING_DIRECTORIES=/tmp HOME=<SCRATCH>/museum/synth/home timeout 600 bash -c "cd <SCRATCH>/museum/probes/substitution && sh setup.sh && python3 -I -B p1_review_role.py && /opt/node22/bin/node p1_review_role.mjs && /opt/node22/bin/node p1b_suffix_match.mjs" ) > <SCRATCH>/museum/synth/out_Q1-1.txt 2>&1; echo "exit=$?"

echo '=================== Q1-2'
( cd <SCRATCH>/museum && env -u GIT_DIR -u GIT_WORK_TREE GIT_CEILING_DIRECTORIES=/tmp HOME=<SCRATCH>/museum/synth/home timeout 600 bash -c "cd <SCRATCH>/museum/probes/substitution && sh setup.sh && python3 -I -B p2_review_snapshot.py && /opt/node22/bin/node p2_review_snapshot.mjs" ) > <SCRATCH>/museum/synth/out_Q1-2.txt 2>&1; echo "exit=$?"

echo '=================== Q1-3'
( cd <SCRATCH>/museum && env -u GIT_DIR -u GIT_WORK_TREE GIT_CEILING_DIRECTORIES=/tmp HOME=<SCRATCH>/museum/synth/home timeout 600 bash -c "cd <SCRATCH>/museum/probes/substitution && sh setup.sh && /opt/node22/bin/node p3_pointer_substring.mjs" ) > <SCRATCH>/museum/synth/out_Q1-3.txt 2>&1; echo "exit=$?"

echo '=================== Q1-4'
( cd <SCRATCH>/museum && env -u GIT_DIR -u GIT_WORK_TREE GIT_CEILING_DIRECTORIES=/tmp HOME=<SCRATCH>/museum/synth/home timeout 600 bash -c "cd <SCRATCH>/museum/probes/substitution && sh setup.sh && /opt/node22/bin/node p4_exhibit_binding.mjs" ) > <SCRATCH>/museum/synth/out_Q1-4.txt 2>&1; echo "exit=$?"

echo '=================== Q1-5'
( cd <SCRATCH>/museum && env -u GIT_DIR -u GIT_WORK_TREE GIT_CEILING_DIRECTORIES=/tmp HOME=<SCRATCH>/museum/synth/home timeout 600 bash -c "cd <SCRATCH>/museum/probes/substitution && sh setup.sh && python3 -I -B p5_replay_binding.py && /opt/node22/bin/node p5_replay_binding.mjs" ) > <SCRATCH>/museum/synth/out_Q1-5.txt 2>&1; echo "exit=$?"

echo '=================== Q1-6'
( cd <SCRATCH>/museum && env -u GIT_DIR -u GIT_WORK_TREE GIT_CEILING_DIRECTORIES=/tmp HOME=<SCRATCH>/museum/synth/home timeout 600 bash -c "cd <SCRATCH>/museum/probes/substitution && sh setup.sh && python3 -I -B p6_open_row_position.py && /opt/node22/bin/node p6_open_row_position.mjs" ) > <SCRATCH>/museum/synth/out_Q1-6.txt 2>&1; echo "exit=$?"

echo '=================== Q1-7'
( cd <SCRATCH>/museum && env -u GIT_DIR -u GIT_WORK_TREE GIT_CEILING_DIRECTORIES=/tmp HOME=<SCRATCH>/museum/synth/home timeout 600 bash -c "cd <SCRATCH>/museum/probes/substitution && sh setup.sh && /opt/node22/bin/node p7_labels.mjs" ) > <SCRATCH>/museum/synth/out_Q1-7.txt 2>&1; echo "exit=$?"

echo '=================== Q1-8'
( cd <SCRATCH>/museum && env -u GIT_DIR -u GIT_WORK_TREE GIT_CEILING_DIRECTORIES=/tmp HOME=<SCRATCH>/museum/synth/home timeout 600 bash -c "cd <SCRATCH>/museum/probes/substitution && sh setup.sh && /opt/node22/bin/node p7_labels.mjs" ) > <SCRATCH>/museum/synth/out_Q1-8.txt 2>&1; echo "exit=$?"

echo '=================== Q1-9'
( cd <SCRATCH>/museum && env -u GIT_DIR -u GIT_WORK_TREE GIT_CEILING_DIRECTORIES=/tmp HOME=<SCRATCH>/museum/synth/home timeout 600 bash -c "cd <SCRATCH>/museum/probes/substitution && sh setup.sh && python3 -I -B p8_landing_axis.py" ) > <SCRATCH>/museum/synth/out_Q1-9.txt 2>&1; echo "exit=$?"

echo '=================== Q2-F1'
( cd <SCRATCH>/museum && env -u GIT_DIR -u GIT_WORK_TREE GIT_CEILING_DIRECTORIES=/tmp HOME=<SCRATCH>/museum/synth/home timeout 600 bash -c "cd <SCRATCH>/museum/probes/pointers && /opt/node22/bin/node p1_review_url_on_byte_verified_review.mjs && python3 -I ci_sim.py control && python3 -I ci_sim.py F1_review_url_on_byte_verified && /opt/node22/bin/node render_sim.mjs F1_review_url_on_byte_verified d5-all-height-annulus && python3 -I p6b_handedit_control.py" ) > <SCRATCH>/museum/synth/out_Q2-F1.txt 2>&1; echo "exit=$?"

echo '=================== Q2-F2'
( cd <SCRATCH>/museum && env -u GIT_DIR -u GIT_WORK_TREE GIT_CEILING_DIRECTORIES=/tmp HOME=<SCRATCH>/museum/synth/home timeout 600 bash -c "cd <SCRATCH>/museum/probes/pointers && /opt/node22/bin/node p2_pointer_substring_binding.mjs && python3 -I p2b_python_any_link_target.py && python3 -I ci_sim.py F2_d1_pointer_prefix && /opt/node22/bin/node render_sim.mjs F2_d1_pointer_prefix d1-parent-selection-open" ) > <SCRATCH>/museum/synth/out_Q2-F2.txt 2>&1; echo "exit=$?"

echo '=================== Q2-F3'
( cd <SCRATCH>/museum && env -u GIT_DIR -u GIT_WORK_TREE GIT_CEILING_DIRECTORIES=/tmp HOME=<SCRATCH>/museum/synth/home timeout 600 bash -c "cd <SCRATCH>/museum/probes/pointers && python3 -I ci_sim.py F3_notice_claims_byte_frozen && /opt/node22/bin/node render_sim.mjs F3_notice_claims_byte_frozen d2-lifetime-remainder" ) > <SCRATCH>/museum/synth/out_Q2-F3.txt 2>&1; echo "exit=$?"

echo '=================== Q2-F4'
( cd <SCRATCH>/museum && env -u GIT_DIR -u GIT_WORK_TREE GIT_CEILING_DIRECTORIES=/tmp HOME=<SCRATCH>/museum/synth/home timeout 600 bash -c "cd <SCRATCH>/museum/probes/pointers && /opt/node22/bin/node p3_byte_verified_review_identity_unbound.mjs && /opt/node22/bin/node p6_controls.mjs" ) > <SCRATCH>/museum/synth/out_Q2-F4.txt 2>&1; echo "exit=$?"

echo '=================== Q2-F5'
( cd <SCRATCH>/museum && env -u GIT_DIR -u GIT_WORK_TREE GIT_CEILING_DIRECTORIES=/tmp HOME=<SCRATCH>/museum/synth/home timeout 600 bash -c "cd <SCRATCH>/museum/probes/pointers && /opt/node22/bin/node p4_comment_edit_invisible.mjs" ) > <SCRATCH>/museum/synth/out_Q2-F5.txt 2>&1; echo "exit=$?"

echo '=================== Q2-F6'
( cd <SCRATCH>/museum && env -u GIT_DIR -u GIT_WORK_TREE GIT_CEILING_DIRECTORIES=/tmp HOME=<SCRATCH>/museum/synth/home timeout 600 bash -c "cd <SCRATCH>/museum/probes/pointers && /opt/node22/bin/node p5_pointer_labels_and_selection.mjs" ) > <SCRATCH>/museum/synth/out_Q2-F6.txt 2>&1; echo "exit=$?"

echo '=================== Q3-F1'
( cd <SCRATCH>/museum && env -u GIT_DIR -u GIT_WORK_TREE GIT_CEILING_DIRECTORIES=/tmp HOME=<SCRATCH>/museum/synth/home timeout 600 bash -c "cd <SCRATCH>/museum/probes/membership && python3 -I build_fixture.py ../../cd32303 /home/user/Math- /home/user/math- /home/user/main && python3 -I p1_fallback_unit.py && ./p1_ci_false_green.sh && for m in harness strict; do /opt/node22/bin/node render.mjs wt_p1/docs/site/museum.json $m; done && ./p1c_env_dependent_source.sh" ) > <SCRATCH>/museum/synth/out_Q3-F1.txt 2>&1; echo "exit=$?"

echo '=================== Q3-F2'
( cd <SCRATCH>/museum && env -u GIT_DIR -u GIT_WORK_TREE GIT_CEILING_DIRECTORIES=/tmp HOME=<SCRATCH>/museum/synth/home timeout 600 bash -c "cd <SCRATCH>/museum/probes/membership && python3 -I p0_baseline.py && ./p2_omission.sh && ./p2c_empty_list_ci.sh" ) > <SCRATCH>/museum/synth/out_Q3-F2.txt 2>&1; echo "exit=$?"

echo '=================== Q3-F3'
( cd <SCRATCH>/museum && env -u GIT_DIR -u GIT_WORK_TREE GIT_CEILING_DIRECTORIES=/tmp HOME=<SCRATCH>/museum/synth/home timeout 600 bash -c "cd <SCRATCH>/museum/probes/membership && ./p3_unlanded_snapshot.sh" ) > <SCRATCH>/museum/synth/out_Q3-F3.txt 2>&1; echo "exit=$?"

echo '=================== Q3-F4'
( cd <SCRATCH>/museum && env -u GIT_DIR -u GIT_WORK_TREE GIT_CEILING_DIRECTORIES=/tmp HOME=<SCRATCH>/museum/synth/home timeout 600 bash -c "cd <SCRATCH>/museum/probes/membership && ./p4_status_moving_ref.sh" ) > <SCRATCH>/museum/synth/out_Q3-F4.txt 2>&1; echo "exit=$?"

echo '=================== Q3-F5'
( cd <SCRATCH>/museum && env -u GIT_DIR -u GIT_WORK_TREE GIT_CEILING_DIRECTORIES=/tmp HOME=<SCRATCH>/museum/synth/home timeout 600 bash -c "cd <SCRATCH>/museum/probes/membership && ./p5_test_writes_inherited_gitdir.sh" ) > <SCRATCH>/museum/synth/out_Q3-F5.txt 2>&1; echo "exit=$?"

echo '=================== Q4-1'
( cd <SCRATCH>/museum && env -u GIT_DIR -u GIT_WORK_TREE GIT_CEILING_DIRECTORIES=/tmp HOME=<SCRATCH>/museum/synth/home timeout 600 bash -c "/opt/node22/bin/node <SCRATCH>/museum/probes/limits/p1_nonok_body_not_released.mjs <SCRATCH>/museum/cd32303" ) > <SCRATCH>/museum/synth/out_Q4-1.txt 2>&1; echo "exit=$?"

echo '=================== Q4-2'
( cd <SCRATCH>/museum && env -u GIT_DIR -u GIT_WORK_TREE GIT_CEILING_DIRECTORIES=/tmp HOME=<SCRATCH>/museum/synth/home timeout 600 bash -c "/opt/node22/bin/node <SCRATCH>/museum/probes/limits/p2_verifyclaim_default_unbounded.mjs <SCRATCH>/museum/cd32303" ) > <SCRATCH>/museum/synth/out_Q4-2.txt 2>&1; echo "exit=$?"

echo '=================== Q4-3'
( cd <SCRATCH>/museum && env -u GIT_DIR -u GIT_WORK_TREE GIT_CEILING_DIRECTORIES=/tmp HOME=<SCRATCH>/museum/synth/home timeout 600 bash -c "/opt/node22/bin/node <SCRATCH>/museum/probes/limits/p6_bound_wiring_mutants.mjs <SCRATCH>/museum/cd32303" ) > <SCRATCH>/museum/synth/out_Q4-3.txt 2>&1; echo "exit=$?"

echo '=================== Q4-4'
( cd <SCRATCH>/museum && env -u GIT_DIR -u GIT_WORK_TREE GIT_CEILING_DIRECTORIES=/tmp HOME=<SCRATCH>/museum/synth/home timeout 600 bash -c "/opt/node22/bin/node <SCRATCH>/museum/probes/limits/p3_manifest_cap_tested_on_reserialized_bytes.mjs <SCRATCH>/museum/cd32303" ) > <SCRATCH>/museum/synth/out_Q4-4.txt 2>&1; echo "exit=$?"

echo '=================== Q4-5'
( cd <SCRATCH>/museum && env -u GIT_DIR -u GIT_WORK_TREE GIT_CEILING_DIRECTORIES=/tmp HOME=<SCRATCH>/museum/synth/home timeout 600 bash -c "python3 -I -B <SCRATCH>/museum/probes/limits/p4_urlopen_timeout_is_per_recv.py <SCRATCH>/museum/cd32303" ) > <SCRATCH>/museum/synth/out_Q4-5.txt 2>&1; echo "exit=$?"

echo '=================== Q4-6'
( cd <SCRATCH>/museum && env -u GIT_DIR -u GIT_WORK_TREE GIT_CEILING_DIRECTORIES=/tmp HOME=<SCRATCH>/museum/synth/home timeout 600 bash -c "python3 -I -B <SCRATCH>/museum/probes/limits/p5_local_fallback_uncapped.py <SCRATCH>/museum/cd32303" ) > <SCRATCH>/museum/synth/out_Q4-6.txt 2>&1; echo "exit=$?"

echo '=================== Q4-7'
( cd <SCRATCH>/museum && env -u GIT_DIR -u GIT_WORK_TREE GIT_CEILING_DIRECTORIES=/tmp HOME=<SCRATCH>/museum/synth/home timeout 600 bash -c "/opt/node22/bin/node <SCRATCH>/museum/probes/limits/p8_geometry_loader_no_deadline.mjs <SCRATCH>/museum/cd32303" ) > <SCRATCH>/museum/synth/out_Q4-7.txt 2>&1; echo "exit=$?"

echo '=================== FO-1'
( cd <SCRATCH>/museum && env -u GIT_DIR -u GIT_WORK_TREE GIT_CEILING_DIRECTORIES=/tmp HOME=<SCRATCH>/museum/synth/home timeout 600 bash -c "cd <SCRATCH>/museum/probes/failopen && PATH=/opt/node22/bin:$PATH node p1_availability_unbound.mjs && PATH=/opt/node22/bin:$PATH node p5_replay_unbound.mjs && python3 -I p6_generator_path_ci_green.py" ) > <SCRATCH>/museum/synth/out_FO-1.txt 2>&1; echo "exit=$?"

echo '=================== FO-2'
( cd <SCRATCH>/museum && env -u GIT_DIR -u GIT_WORK_TREE GIT_CEILING_DIRECTORIES=/tmp HOME=<SCRATCH>/museum/synth/home timeout 600 bash -c "cd <SCRATCH>/museum/probes/failopen && PATH=/opt/node22/bin:$PATH node p2_pointer_only_truthy.mjs" ) > <SCRATCH>/museum/synth/out_FO-2.txt 2>&1; echo "exit=$?"

echo '=================== FO-3'
( cd <SCRATCH>/museum && env -u GIT_DIR -u GIT_WORK_TREE GIT_CEILING_DIRECTORIES=/tmp HOME=<SCRATCH>/museum/synth/home timeout 600 bash -c "cd <SCRATCH>/museum/probes/failopen && python3 -I mutants.py M31 M32 M33 M21 M20 M22 M23 M16 && python3 -I p3_mutant_impact.py && PATH=/opt/node22/bin:$PATH node p7_banner_and_count.mjs" ) > <SCRATCH>/museum/synth/out_FO-3.txt 2>&1; echo "exit=$?"

echo '=================== FO-4'
( cd <SCRATCH>/museum && env -u GIT_DIR -u GIT_WORK_TREE GIT_CEILING_DIRECTORIES=/tmp HOME=<SCRATCH>/museum/synth/home timeout 600 bash -c "cd <SCRATCH>/museum/probes/failopen && python3 -I mutants.py M01 M02 M03 M04 M05 M06 M07 M08 M09 M10 M12 M13 M14 M15 M18 M19 M25 M26 M27 && python3 -I p3_mutant_impact.py" ) > <SCRATCH>/museum/synth/out_FO-4.txt 2>&1; echo "exit=$?"

echo '=================== FO-5'
( cd <SCRATCH>/museum && env -u GIT_DIR -u GIT_WORK_TREE GIT_CEILING_DIRECTORIES=/tmp HOME=<SCRATCH>/museum/synth/home timeout 600 bash -c "cd <SCRATCH>/museum/probes/failopen && python3 -I p4_status_quote_default.py ../../cd32303 ./fixture" ) > <SCRATCH>/museum/synth/out_FO-5.txt 2>&1; echo "exit=$?"

echo '=================== FO-6'
( cd <SCRATCH>/museum && env -u GIT_DIR -u GIT_WORK_TREE GIT_CEILING_DIRECTORIES=/tmp HOME=<SCRATCH>/museum/synth/home timeout 600 bash -c "cd <SCRATCH>/museum/probes/failopen && python3 -I mutants.py P01 P02 P03 P04 P05 P06 P07 P08 P09 P10 P11" ) > <SCRATCH>/museum/synth/out_FO-6.txt 2>&1; echo "exit=$?"

echo '=================== FO-7'
( cd <SCRATCH>/museum && env -u GIT_DIR -u GIT_WORK_TREE GIT_CEILING_DIRECTORIES=/tmp HOME=<SCRATCH>/museum/synth/home timeout 600 bash -c "cd <SCRATCH>/museum/probes/failopen && sh p8_museum_check_skip_is_pass.sh && python3 -I mutants.py C01 C02 C03" ) > <SCRATCH>/museum/synth/out_FO-7.txt 2>&1; echo "exit=$?"

echo '=================== FO-8'
( cd <SCRATCH>/museum && env -u GIT_DIR -u GIT_WORK_TREE GIT_CEILING_DIRECTORIES=/tmp HOME=<SCRATCH>/museum/synth/home timeout 600 bash -c "cd <SCRATCH>/museum/probes/failopen && PATH=/opt/node22/bin:$PATH node p7_banner_and_count.mjs" ) > <SCRATCH>/museum/synth/out_FO-8.txt 2>&1; echo "exit=$?"
