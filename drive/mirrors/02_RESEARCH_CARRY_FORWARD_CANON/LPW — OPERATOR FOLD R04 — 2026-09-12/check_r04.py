#!/usr/bin/env python3
"""Read-only exact arithmetic, source identity and evidence-state guards.

This is an author-side regression checker. It does not validate absent Kimi
certificates, execute imported scripts, or prove the complete LPW theorem.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
QUAL_SHA = 'cf58f72eb0399c626160c143b50f674ff3bd5c449225211edd6fd378a374e1a5'
FALLBACK_SHA = '18a6e27cb945c7fc80975e20f775724989de0d368ce7a6f2ee29f11bea165d70'
MUTATIONS = ('coefficient_upward', 'radius_too_large', 'floor_upward',
             'odd_derivative_zero', 'permission_as_proof', 'raw_archive_invented',
             'h_b3_discharged', 'two_dim_three_dim_merge')

class Rejected(ValueError):
    pass

def strict_json(path: Path):
    def pairs(xs):
        d = {}
        for k, v in xs:
            if k in d:
                raise Rejected('DUPLICATE_KEY:' + k)
            d[k] = v
        return d
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=pairs)

def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mutation', choices=MUTATIONS)
    opt = parser.parse_args()
    tests = []
    def ck(label: str, condition: bool):
        if not condition:
            raise Rejected(label)
        tests.append({'check': label, 'result': 'PASS'})
    try:
        state = strict_json(ROOT / 'CURRENT_STATE.json')
        ledger = strict_json(ROOT / 'DRIVE_SOURCE_HASH_LEDGER.json')
        ck('source_count_13', ledger['count'] == len(ledger['sources']) == 13)
        for item in ledger['sources']:
            p = ROOT / item['path']
            ck('source_bytes:' + p.name, p.is_file() and p.stat().st_size == item['bytes'])
            ck('source_sha256:' + p.name, digest(p) == item['sha256'])
        ck('qualitative_exact_target', digest(ROOT/'drive_sources/targets/02_LOCAL_PATH_LOWER_BOUND_CANDIDATE.md') == QUAL_SHA)
        ck('fallback_exact_target', digest(ROOT/'03_EXPLICIT_CONSTANTS_CANDIDATE.md') == FALLBACK_SHA)
        gp = next((ROOT/'drive_sources/core').glob('GP-DER-118*'))
        data = gp.read_bytes()
        start, end = b'BEGIN_FROZEN_THEOREM_BODY\n', b'END_FROZEN_THEOREM_BODY'
        ck('gp_unique_begin', data.splitlines().count(start.rstrip(b'\n')) == 1)
        ck('gp_unique_end', data.splitlines().count(end) == 1)
        body = data.split(start,1)[1].split(end,1)[0].rstrip(b'\n')+b'\n'
        ck('gp_body_bytes', len(body) == 29293)
        ck('gp_body_hash', hashlib.sha256(body).hexdigest() == ledger['GP_DER118_frozen_body_sha256'])
        for item in state['received_current_files']:
            p = ROOT/'incoming'/item['name']
            ck('pdf_bytes:' + p.name, p.stat().st_size == item['bytes'])
            ck('pdf_hash:' + p.name, digest(p) == item['sha256'])
        prior = ROOT/'prior_return03'
        count = 0
        for line in (prior/'MANIFEST.sha256').read_text().splitlines():
            if not line.strip(): continue
            h, rel = line.split(maxsplit=1)
            rel = rel.lstrip(' *')
            p = prior/rel
            if not p.is_file() or digest(p)!=h:
                raise Rejected('PRIOR_MANIFEST:'+rel)
            count += 1
        ck('prior_manifest_49_exact', count == 49)
        d = F(1,1024)
        ck('delta_fourth_exact', d**4 == F(1,2**40))
        c_calc = 260 * F(1,10**21) * d**4 / F(3790000000000)
        claimed = F(6239, 10**47)
        if opt.mutation == 'coefficient_upward': claimed = F(6240,10**47)
        ck('reported_coefficient_below_conditional_scalar_floor', claimed <= c_calc)
        B4 = F(471575,100)
        ck('required_K_exact', 2*B4 == F(18863,2))
        K = F(9432)
        r = F(1,2414592)
        if opt.mutation == 'radius_too_large': r = F(1,2000000)
        ck('K_dominates_reported_B4', K >= 2*B4 and K >= 1)
        ck('radius_respects_K', 256*K*r <= 1)
        ck('reported_radius_below_covariance_interval', r <= F(1,100000))
        ck('geometry_tolerance', 16*d + 8*K*r < F(1,16))
        ck('canonical_K_radius_identity', 256*K == 2414592)
        ck('endpoint_floor_is_below_one_eighth', F(31,250) < F(1,8))
        ck('fallback_covariance_arithmetic', F(31,250)-F(32,1600)==F(13,125))
        ck('fallback_covariance_lower_tenth', F(13,125)>F(1,10))
        endpoint = F(31,250)
        modulus = F('19.071156')
        radius = F(1,100000)
        floor = F('0.1238092884')
        if opt.mutation == 'floor_upward': floor = F('0.1238092885')
        ck('reported_floor_rounds_down', floor <= endpoint-modulus*radius)
        ck('exact_modulus_arithmetic', endpoint-modulus*radius == F('0.12380928844'))
        # K(t)=cos(t), obtained from independent real cosine and sine modes.
        # Cov(f(pi/2), f'(0)) = 1. Its odd endpoint covariance at t=0 is 0.
        f_at_zero, f_at_pi_over_2, derivative_at_zero = (F(1),F(0)), (F(0),F(1)), (F(0),F(1))
        offorigin = sum(a*b for a,b in zip(f_at_pi_over_2,derivative_at_zero))
        if opt.mutation == 'odd_derivative_zero': offorigin = F(0)
        ck('nonzero_separation_odd_covariance_retained', offorigin == 1)
        ck('toy_covariance_endpoint_zero', sum(a*b for a,b in zip(f_at_zero,derivative_at_zero)) == 0)
        # Integral Taylor kernel h^2/2, with h=r/2, gives 11*r^2/8.
        rr = F(1,1600); h = rr/2
        ck('average_y_integral_bound', 11*h*h/2 == 11*rr*rr/8)
        ck('extra_positive_remainder_cannot_disappear', 11*rr**2/8 + 11*rr**3/48 > 11*rr**2/8)
        ck('fixed_term_cubic_ratio_grows_under_halving', F(1,10**40)/(rr/2)**3 == 8*F(1,10**40)/rr**3)
        # Explicit fallback stays exact, no float conversion.
        ck('fallback_c_positive', F(1,10**1235)>0)
        ck('fallback_coefficient_arithmetic', 260*F(1,10**1129)*d**4/F(10**96)>F(1,10**1235))
        ck('fallback_radius', 256*2*10**24*F(1,10**28)<1)
        if opt.mutation == 'permission_as_proof': state['class3_technical_conditions_waived']=True
        if opt.mutation == 'raw_archive_invented': state['missing_deliveries'][0]['received']=True
        claims = {x['id']:x for x in state['claims']}
        if opt.mutation == 'h_b3_discharged': claims['W8-V3']['H_B3_discharged']=True
        if opt.mutation == 'two_dim_three_dim_merge': claims['MATCHING-2D-UPPER']['two_sided_theta_admitted']=True
        ck('owner_permission_received', state['operator_authorized'] is True)
        ck('no_repeat_owner_gate', state['operator_approval_pending'] is False)
        ck('permission_not_proof_waiver', state['class3_technical_conditions_waived'] is False)
        ck('raw_archive_not_invented', state['missing_deliveries'][0]['received'] is False)
        ck('h_b3_remains_premise', claims['W8-V3']['H_B3_discharged'] is False)
        ck('two_dimensional_upper_not_invented', claims['MATCHING-2D-UPPER']['two_sided_theta_admitted'] is False)
        ck('sharp_certificate_not_locally_certified', claims['LPW-CONSTANT-SHARP']['certified_locally'] is False)
        ck('one_external_provider_family', state['independent_review_provider_families_received']==['Kimi'])
        ck('five_task_branches_not_five_families', state['reported_implementation_branches']==5)
        ck('p01_not_promoted', claims['P0.1']['formal_class3_promotion']=='HOLD')
        ck('original_3d_unchanged', claims['SIDE24-3D']['dimension']==3 and not claims['SIDE24-3D']['changed'])
        ck('k3_not_rehabilitated', claims['K3-THM-001']['status']=='REFUTED_AS_WRITTEN_NONCONTROLLING')
        ck('external_release_not_inferred', state['external_release_authorized'] is False)
        ck('no_legacy_overwrite', state['legacy_control_overwrites']==0)
        declared='04bcbdf2013d83b149a8933a332a078c303af347335ee40f154f37ba195ad448'
        ck('declared_verdict_hash_syntactically_valid_only', len(declared)==64 and all(c in '0123456789abcdef' for c in declared))
        result = {'status':'PASS_AT_ARITHMETIC_IDENTITY_AND_STATE_GUARD_SCOPE','checks_passed':len(tests),'checks':tests,
                  'conditional_sharp_coefficient':{'numerator':str(c_calc.numerator),'denominator':str(c_calc.denominator)},
                  'claimed_inputs_certified_by_this_script':False,'raw_kimi_code_executed':False,'formal_theorem_proved_by_checker':False}
        print(json.dumps(result,indent=2,sort_keys=True))
        return 0
    except (Rejected, OSError, ValueError, KeyError) as exc:
        print(json.dumps({'status':'REJECTED','reason':str(exc),'passed_before_rejection':len(tests),'mutation':opt.mutation},sort_keys=True))
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
