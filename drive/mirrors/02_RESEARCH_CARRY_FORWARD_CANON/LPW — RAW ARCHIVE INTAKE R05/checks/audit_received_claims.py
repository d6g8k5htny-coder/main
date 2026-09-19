#!/usr/bin/env python3
"""Exact checks of received-claim discrepancies; deliberate mismatches are findings."""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal, localcontext
import hashlib,json,re,zipfile
P=Path(__file__).resolve().parents[1]
checks=[]
def ck(name,ok):
    if not ok: raise SystemExit('FAIL: '+name)
    checks.append(name)
def H(b):return hashlib.sha256(b).hexdigest()
B=3790446482793;c=F(260,B*2**40*10**21);reported=F('6.239e-44');safe=F('6.238e-44')
ck('printed_decimal_exceeds_exact_chain',reported>c)
ck('corrected_decimal_below_exact_chain',safe<=c)
ck('old_rounded_denominator_is_smaller',F('3.79e12')<B)
ck('exact_power_floor_1e_minus44_is_safe',F('1e-44')<c)
ck('radius_arithmetic',256*9432==2414592)
ck('geometric_tolerance',16*F(1,1024)+8*9432*F(1,2414592)==F(3,64)<F(1,16))
v=(P/'received_raw/review/KIMI_LPW_REVIEW_VERDICT.md').read_bytes()
mark=b'SHA-256 of this report body (text after this line is excluded):'
ck('one_unique_body_marker',v.count(mark)==1)
pos=v.index(mark);decl=re.search(rb'[0-9a-f]{64}',v[pos+len(mark):]).group().decode()
ck('whole_verdict_hash_correct',H(v)=='04bcbdf2013d83b149a8933a332a078c303af347335ee40f154f37ba195ad448')
ck('declared_prefix_rule_mismatch',H(v[:pos])!=decl)
ck('separator_exact',v[pos-6:pos]==b'\n---\n\n')
ck('separator_excluded_hash_matches',H(v[:pos-6])==decl)
w=P/'received_raw/w8/transcripts';a=(w/'transcript_v3_normal.txt').read_bytes();b=(w/'transcript_v3_O.txt').read_bytes()
ck('W8_transcripts_identical',a==b)
ck('W8_final_failclosed',a.rstrip().endswith(b'FAIL-CLOSED TRIGGER: used core spacing violates kill condition at floor margin'))
ck('W8_refined_rung_pass_evidence_present',b'1.8846416' in a)
ck('W8_no_executed_completion',b'CERTIFICATE COMPLETE' not in a)
f=(P/'received_raw/constant/falsify.py').read_text()
ck('vacuous_guard_reproduced','or True' in f)
raw=(P/'received_raw/constant/LPW_CONSTANT_REPORT.md').read_text()
ck('false_moment_identity_present','12 + 16/pi' in raw and '|xi_k|+|eta_k|' in raw)
# Exact coefficient ledger for the fourth moment of |xi|+|eta|:
# E|xi|=sqrt(2/pi), E|xi|^3=2sqrt(2/pi), E xi^2=1, E xi^4=3.
constant_term=3+3+6;pi_term=4*2*2+4*2*2
ck('absolute_sum_moment_coefficients',constant_term==12 and pi_term==32)
ck('Rayleigh_fourth_moment',3+2*1+3==8)
with localcontext() as ctx:
 ctx.prec=60;cv=Decimal(c.numerator)/Decimal(c.denominator)
 scalar={'exact_c':str(c),'c_display':str(cv),'reported_decimal':str(Decimal('6.239e-44')),'safe_decimal':'6.238e-44','exact_r0':'1/2414592'}
print(json.dumps({'status':'PASS_EXPECTED_FINDINGS_REPRODUCED','check_count':len(checks),'checks':checks,'scalar':scalar,'body':{'marker_offset':pos,'exact_prefix_hash':H(v[:pos]),'prefix_minus_separator_hash':H(v[:pos-6]),'declared_hash':decl},'limitations':['checks localize received proof/custody defects, not a counterexample to the true LPW probability','Rayleigh analytic repair reviewed separately','W8 transcript inspection, not W8 re-execution']},indent=2,sort_keys=True))
