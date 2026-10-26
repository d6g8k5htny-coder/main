"""Exact counterexample search in one reviewed Gaussian moment family.

A finite search can find a genuine refuting point. Failure to find one never
establishes a universal bound. All probability identities remain scoped to the
explicit generic law checked by moment_checker, with no actual-field alignment.
"""
from fractions import Fraction
from frontier_ir import digest
from moment_checker import verify_certificate

SCOPE='generic_independent_gaussian_variance_family_only'


def _rational(value):
    if (not isinstance(value,list) or len(value)!=2 or any(type(x) is not int or x.bit_length()>128 for x in value)
            or value[1]<=0):
        raise ValueError('a bounded integer numerator and positive denominator are required')
    return Fraction(*value)


def _pair(value):return [value.numerator,value.denominator]


def probe_bound(certificate,power=4,claimed_upper_bound=None):
    if type(power) is not int or power not in (2,4):raise ValueError('only reviewed second/fourth moments are supported')
    if not verify_certificate(certificate).get('valid'):raise ValueError('moment certificate is not valid for the reviewed law')
    bound=_rational([131,1] if claimed_upper_bound is None else claimed_upper_bound)
    coefficients=certificate['moments'][str(power)]
    report={'schema':'ul-v19-exact-counterexample-1','scope':SCOPE,'power':power,
        'claimed_upper_bound':_pair(bound),'certificate_sha256':certificate['certificate_sha256'],
        'question':'For every s in [0,1], is the stated moment at most the claimed upper bound?',
        'status':'inconclusive','points_checked':0,'actual_field_applicable':False,'full_theorem_certificates':[]}
    for point in (Fraction(0),Fraction(1,2),Fraction(1)):
        value=sum(Fraction(c)*point**i for i,c in enumerate(coefficients))
        report['points_checked']+=1
        if value>bound:
            report.update(status='counterexample_found',point=_pair(point),value=_pair(value),
                violation=_pair(value-bound),evidence_kind='exact_generic_law_point_witness')
            break
    report['report_sha256']=digest(report)
    return report


def verify_counterexample(report,certificate):
    try:
        if not isinstance(report,dict):raise ValueError('report must be an object')
        expected=probe_bound(certificate,report['power'],report['claimed_upper_bound'])
        if report!=expected:raise ValueError('report, scope, law, point or arithmetic mismatch')
        if expected['status']!='counterexample_found':
            return {'valid':False,'state':'inconclusive','reason':'No witness found; universal truth is not inferred.',
                'actual_field_applicable':False,'full_theorem_certificates':[]}
        return {'valid':True,'state':'exact_generic_law_counterexample','scope':SCOPE,
                'point':report['point'],'value':report['value'],'violation':report['violation'],
                'actual_field_applicable':False,'full_theorem_certificates':[]}
    except (ValueError,TypeError,KeyError,OverflowError,RecursionError) as error:
        return {'valid':False,'state':'invalid','reason':str(error),'actual_field_applicable':False,
                'full_theorem_certificates':[]}
