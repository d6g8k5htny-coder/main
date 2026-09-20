#!/usr/bin/env python3
"""Reproduce candidate envelope constants. No review credit or status promotion."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from fractions import Fraction as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.manifest_integrity_check import strict_object, reject_constant
from research.interval import Interval
from research.bands.hermite_gaussian import hermite_gaussian
from research.bands.lattice import tail_bound

DEFAULT = ROOT / 'research/bands/candidates/hermite_gaussian_20260919.json'


def build():
    cases = []
    for total in range(11):
        for a in range(total + 1):
            b = total-a
            kernel = hermite_gaussian(a,b)
            tail = tail_bound(kernel.envelope,Interval(-17,17),Interval.exact(0),prec=30)
            cases.append({'a':a,'b':b,'A':str(kernel.envelope.A),'B':'1/4',
                          'tail_upper':str(tail),'tail_lt_1e_60':tail < F(1,10**60)})
    paths=['tools/hermite_envelope_report.py','research/interval/core.py','research/interval/transcendental.py','research/bands/hermite_gaussian.py','research/bands/lattice.py',
           'engine/rn_engine/frozen/K3_SIDE24_LB/UPPER2D/D3_percolation/d3_rn_unif.py','docs/HERMITE_GAUSSIAN_ENVELOPE.md']
    return {'schema_version':1,'technical_status':'AUTHOR_CANDIDATE_NOT_REVIEWED',
            'independence_credit':0,'envelope_reviewed':False,'scientific_status_changes':[],
            'inputs':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
            'geometry':{'dx':['-17','17'],'dy':['0','0'],'period':'24','n_trunc':1,'prec':30},
            'scope':'Global plane derivative envelope; conditional lattice tails on this axial box only. No six-pin, full 24-jet, r-band or obligation result.',
            'cases':cases}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    report=build()
    if args.check is not None:
        try:
            if json.loads(args.check.read_text(),object_pairs_hook=strict_object,parse_constant=reject_constant) != report:raise ValueError('candidate report drift')
        except (OSError,ValueError) as error:
            print(f'FAIL: {error}',file=sys.stderr);return 1
        print('ok: 66 exact derivative-envelope cases; review pending, no status change')
    elif args.output is not None:
        args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
        print('wrote 66 candidate cases; review pending')
    else:print(json.dumps(report,indent=2,sort_keys=True))
    return 0


if __name__=='__main__':raise SystemExit(main())
