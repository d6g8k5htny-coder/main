#!/bin/bash
S=/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad
PRISTINE=$S/wt343/experiments/universality/two_parameter_cusp_population
COPYPKG=$S/p343patch2/copy/experiments/universality/two_parameter_cusp_population
MUT=$S/p343336/agents/343-tests-mutation/mutants
OUT=$S/p343patch2/mutants; mkdir -p $OUT
NEW_TESTS="test_power_enclosures_bracket_irrational_radicands test_direct_row_replays_its_component_composition test_failed_direct_gate_is_not_certified test_saturation_is_reached_by_the_closed_form_not_only_the_literal test_coefficient_inconclusive_strictly_between_outer_and_inner test_void_width_gate_at_its_declared_limit test_overall_priority_is_the_declared_one test_negative_controls_evaluate_every_declared_gate test_negative_controls_fail_when_the_selector_is_fixed_left test_consistent_routes_and_correct_coefficient_pass test_wrong_coefficient_is_falsified_at_row_eight_only test_disjoint_routes_are_route_consistency_failures test_row_eight_mass_gate_withholds_the_void_test test_wide_void_interval_is_inconclusive_even_with_a_passing_coefficient test_falsified_attempt_exits_one_and_is_no_full_campaign_under_reduced_budget test_source_edited_during_evaluation_is_source_drift test_void_interval_replays_its_outward_log_exp_composition test_poisson_target_replays_its_outward_exp_composition test_overall_requires_every_status_to_pass"
for d in "$@"; do
  mods=""
  for f in population_certificate.py run_population_certificate.py; do
    cmp -s $MUT/$d/experiments/universality/two_parameter_cusp_population/$f $PRISTINE/$f || mods="$mods $f"
  done
  for f in $mods; do cp $MUT/$d/experiments/universality/two_parameter_cusp_population/$f $COPYPKG/$f; done
  (cd $S/p343patch2/copy && python3.12 -B -E -S -m unittest discover -s experiments/universality/two_parameter_cusp_population -p 'test_population_certificate.py' -v) > $OUT/$d.log 2>&1
  rc=$?
  failed=$(grep -E '^(FAIL|ERROR): ' $OUT/$d.log | sed -E 's/^(FAIL|ERROR): (test_[A-Za-z0-9_]+).*/\2/' | sort -u)
  newfail=""; oldfail=""
  for t in $failed; do if echo " $NEW_TESTS " | grep -q " $t "; then newfail="$newfail $t"; else oldfail="$oldfail $t"; fi; done
  ran=$(grep -E '^Ran [0-9]+ tests' $OUT/$d.log | head -1)
  if [ -n "$newfail" ]; then v=KILLED_BY_NEW; elif [ -n "$oldfail" ]; then v=KILLED_BY_EXISTING_ONLY; else v=SURVIVES; fi
  # restore pristine and verify byte identity
  for f in $mods; do cp $PRISTINE/$f $COPYPKG/$f; done
  restore=OK
  for f in population_certificate.py run_population_certificate.py; do
    a=$(sha256sum $PRISTINE/$f | cut -d' ' -f1); b=$(sha256sum $COPYPKG/$f | cut -d' ' -f1); [ "$a" = "$b" ] || restore="MISMATCH:$f"
  done
  printf '%-24s %-44s modifies=[%s] exit=%s %s restore=%s\n    new_failing:%s\n    existing_failing:%s\n' "$v" "$d" "${mods# }" "$rc" "$ran" "$restore" "${newfail:- (none)}" "${oldfail:- (none)}"
done
