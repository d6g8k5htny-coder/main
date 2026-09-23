#!/usr/bin/env python3
"""Machine checks for the C091 proof-gate reduction."""
from __future__ import annotations
import json
from pathlib import Path

BASE=Path('/mnt/data')

def main():
    contract=json.loads((BASE/'q0_c091_canonical_contract.json').read_text())
    gates=json.loads((BASE/'q0_c091_gate_reduction_report.json').read_text())
    torus=json.loads((BASE/'periodized_bf_matrix_transfer_report.json').read_text())
    corridor=json.loads((BASE/'gaussian_corridor_certificate_report.json').read_text())
    gap=json.loads((BASE/'bf_two_critical_value_gap_report.json').read_text())
    palm=json.loads((BASE/'palm_weighted_gaussian_chernoff_report.json').read_text())

    allowed=[r['allowed_Crep_times_Cmark_after_Cdec_4p051'] for r in gates['bonferroni']['two_scale_rows']]
    counter=corridor['adjudication']['number_of_tested_skeletons_where_product_underestimates']
    checks={
      'torus_local_pass':torus['local_transfer']['status']=='PASS',
      'torus_far_pass':torus['far_transfer']['status']=='PASS',
      'torus_far_floor_positive':torus['far_transfer']['floor_after_transfer']>0,
      'one_sided_curvature_binding_gate_64':abs(gates['upper_shape']['intervals']['[0.025,0.05]']['M_minus_nominal_0p99']-64)<1e-9,
      'marked_bonferroni_uniform_target_80_fits_all_rungs':min(allowed)>80,
      'h4_product_has_counterexamples':counter>0,
      'h4_product_contract_killed':contract['precedence']['supersedes_or_quarantines'][0]['status']=='KILLED',
      'planar_gap_varZ_limit':abs(gap['limits']['Var_Z']-1/24)<1e-15,
      'Palm_weighted_chernoff_demo_valid':palm['demo']['certificate_exceeds_monte_carlo'],
      'finite_lower_remains_blocked':contract['claims']['LOWER_FINITE_084']['status']=='BLOCKED',
      'continuous_upper_remains_blocked':contract['claims']['UPPER_CONTINUOUS_101_BANDED']['status']=='BLOCKED'
    }
    report={
      'contract_id':contract['contract_id'],
      'checks':checks,
      'all_pass':all(checks.values()),
      'minimum_allowed_Crep_times_Cmark':min(allowed),
      'H4_product_counterexample_count':counter,
      'torus_far_error_over_floor':torus['far_transfer']['schur_error_over_floor'],
      'live_decimal_theorem_promoted':False
    }
    (BASE/'q0_c091_contract_check_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    if not report['all_pass']: raise SystemExit(1)
if __name__=='__main__': main()
