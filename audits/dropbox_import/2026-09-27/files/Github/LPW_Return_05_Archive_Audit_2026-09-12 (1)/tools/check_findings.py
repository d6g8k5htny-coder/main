#!/usr/bin/env python3
"""Exact regression checks of Return-05 findings, NOT a theorem verifier."""
from pathlib import Path
from fractions import Fraction
from decimal import Decimal, localcontext
import argparse, hashlib, json, re
import sympy as s

ROOT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--mutation',choices=['moment_cross_term','headline_rounding','body_boundary','cross_dimension']);a=ap.parse_args()
checks=[]
def ck(name,cond):
 if not bool(cond):
  print(json.dumps({'status':'FAIL_CLOSED','failed':name,'scope':'Return-05 finding regression'}));raise SystemExit(1)
 checks.append(name)
def sha(b):return hashlib.sha256(b).hexdigest()

cdir=ROOT/'raw/constant/LPW_CONSTANT'
source=(cdir/'lpw_constant.py').read_bytes()
ck('exact_received_certifier_identity',sha(source)=='e258322cfbb71dbb8665c2d0c0517399dca5ca5913e9ddf70eb7247ba74d3035')
ck('source_contains_disputed_moment',b'mu4 = 12 + 16/pi + ROUND' in source)
D=3790446482793
cr=Fraction(260,D*2**40*10**21);pub=Fraction(6239,10**47);safe=Fraction(6238,10**47)
ck('delivered_integer_exceeds_rounded_input',D>3790000000000)
ck('reported_decimal_exceeds_printed_rational',pub>cr)
ck('safe_display_6_238_below_rational',safe<cr)
ck('coarse_10_neg44_floor_is_arithmetically_below_rational',Fraction(1,10**44)<cr)
ck('rounded_input_produces_different_fraction',Fraction(260,3790000000000*2**40*10**21)>pub)
selected=pub if a.mutation=='headline_rounding' else safe
ck('successor_display_guard',selected<=cr)
# Derive rather than assume the cross terms of the fourth moment.
m=[s.Integer(1),s.sqrt(2/s.pi),s.Integer(1),2*s.sqrt(2/s.pi),s.Integer(3)]
expansion=s.simplify(sum(s.binomial(4,k)*m[k]*m[4-k] for k in range(5)))
ck('half_normal_fourth_moment_expansion',s.simplify(expansion-(12+32/s.pi))==0)
ck('old_moment_missing_16_over_pi',s.simplify(expansion-(12+16/s.pi))==16/s.pi)
proposed=12+(16 if a.mutation=='moment_cross_term' else 32)/s.pi
ck('replacement_half_normal_moment_guard',s.simplify(expansion-proposed)==0)
ck('Rayleigh_fourth_moment',3+2*1+3==8)
ck('Rayleigh_second_moment',1+1==2)
ck('Rayleigh_fourth_cap_below_old_numeric_multiplier',8<12)
# Radius and geometric arithmetic are unaffected by coefficient repair.
r=Fraction(1,2414592);K=9432;delta=Fraction(1,1024)
ck('radius_denominator',2414592==256*K)
ck('radius_within_covariance_interval',r<Fraction(1,100000))
ck('exact_radius_product',256*K*r==1)
ck('original_geometric_margin',16*delta+8*K*r==Fraction(3,64)<Fraction(1,16))
ck('fallback_domain_is_subset_not_extension',Fraction(1,10**28)<r)
ck('path_clearance',Fraction(1,6)-Fraction(99,1280)-Fraction(1,16)==Fraction(103,3840)>0)
# Body boundary discrepancy: do not silently turn a diagnostic alternative into the declared rule.
vr=(ROOT/'raw/review/KIMI_LPW_REVIEW_VERDICT.md').read_bytes()
ck('whole_verdict_identity',len(vr)==9219 and sha(vr)=='04bcbdf2013d83b149a8933a332a078c303af347335ee40f154f37ba195ad448')
marker=b'SHA-256 of this report body (text after this line is excluded):'
ck('unique_verdict_marker',vr.count(marker)==1)
k=vr.index(marker);ck('marker_at_9090',k==9090)
ck('literal_prefix_hash',sha(vr[:k])=='17863be2e9fedc4ddc6517562c501a609fa4fa3579f8dad1348fb67f68a9c5bd')
ck('six_bytes_explain_mismatch',vr[9084:9090]==b'\n---\n\n')
ck('declared_digest_matches_shorter_prefix',sha(vr[:9084])=='d399e28378d177f88d08032c81f1573f35c9af2b600b9b775ebfe40f555cbbad')
selected_end=9084 if a.mutation=='body_boundary' else 9090
ck('declared_exact_prefix_rule_guard',selected_end==k)
# Raw current manifest checks (strict sizes, hashes, duplicate lines, coverage).
manifest_paths=[ROOT/'raw/constant/LPW_CONSTANT/MANIFEST.sha256',ROOT/'raw/review/MANIFEST.sha256',ROOT/'raw/qc/QC_RETURN03_REVIEW/MANIFEST.sha256',ROOT/'raw/w8/W8_lambda/MANIFEST.sha256']
n=0
for mf in manifest_paths:
 names=set()
 for l in mf.read_text().splitlines():
  if not l.strip():continue
  mt=re.fullmatch(r'([0-9a-f]{64}) (\d+) (.+)',l)
  ck('manifest_line_format_'+str(n),mt is not None)
  h,bs,fn=mt.groups();p=mf.parent/fn
  ck('manifest_member_'+str(n),fn not in names and p.is_file() and len(p.read_bytes())==int(bs) and sha(p.read_bytes())==h)
  names.add(fn);n+=1
 files={str(p.relative_to(mf.parent)) for p in mf.parent.rglob('*') if p.is_file() and p!=mf}
 ck('full_manifest_coverage_'+mf.parent.name,files==names)
ck('current_payload_count',n==82)
for sub,n1,n2 in [('review','out_normal.json','out_opt.json'),('qc/QC_RETURN03_REVIEW','OUTPUT_NORMAL.json','OUTPUT_OPTIMIZED.json'),('constant/LPW_CONSTANT','out_normal.txt','out_O.txt'),('w8/W8_lambda/transcripts','transcript_v3_normal.txt','transcript_v3_O.txt')]:
 ck('delivered_twin_identity_'+sub,(ROOT/'raw'/sub/n1).read_bytes()==(ROOT/'raw'/sub/n2).read_bytes())
q=json.loads((ROOT/'raw/qc/QC_RETURN03_REVIEW/OUTPUT_NORMAL.json').read_text())
ck('QC_check_count',q['check_count']==335)
w=(ROOT/'raw/w8/W8_lambda/transcripts/transcript_v3_normal.txt').read_text()
ck('W8_terminal_failure_preserved','FAIL-CLOSED TRIGGER: used core spacing violates kill condition at floor margin' in w)
ck('W8_source_requires_recovered_hash','0003756d4075bbfa881edfeac68d6cca7242c54cee2c551f2814fd57759b4c18' in (ROOT/'raw/w8/W8_lambda/verify_lambda_grid_v3.py').read_text())
ck('recovered_Drive_input_identity',sha((ROOT/'drive_supplements/c027 sweep core40.json').read_bytes())=='0003756d4075bbfa881edfeac68d6cca7242c54cee2c551f2814fd57759b4c18')
updim=3 if a.mutation=='cross_dimension' else 2
ck('matching_2D_composition_guard',updim==2)
with localcontext() as ctx:
 ctx.prec=80;dec=str(Decimal(cr.numerator)/Decimal(cr.denominator));gap=str(Decimal((pub-cr).numerator)/Decimal((pub-cr).denominator))
print(json.dumps({'status':'PASS_FINDING_REGRESSION_ONLY','check_count':len(checks),'checks':checks,'manifest_payloads':n,'exact_rational':{'numerator':cr.numerator,'denominator':cr.denominator,'decimal':dec},'advertised_decimal_minus_rational':gap,'half_normal_fourth_moment':str(expansion),'mathematical_promotion':False,'drive_writes':0},indent=2,sort_keys=True))
