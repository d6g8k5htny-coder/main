"""Source wiring and exact finite controls; these do not execute Lean."""
from fractions import Fraction as F
import hashlib
import itertools
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULE = 'ResearchFormalCoreR1/D2Schur.lean'
NAMES = [
    'd2_schur_numerator', 'd2_det_axis', 'd2_det_diagonal',
    'd2_det_interpolation', 'd2_schur_ratio', 'd2_numerator_pos',
    'd2_det_pos', 'd2_schur_pos', 'd2_det_le_endpoint_max',
    'd2_schur_endpoint_lower', 'd2_tau_cumulant_eq', 'd2_tau_axis',
    'd2_tau_diagonal', 'd2_tau_pos', 'd2_gaussian_control',
    'd2_degenerate_control',
]


def det3(a):
    return sum(((-1 if (i, j, k) in ((0, 2, 1), (1, 0, 2), (2, 1, 0)) else 1)
                * a[0][i]*a[1][j]*a[2][k]
                for i,j,k in itertools.permutations(range(3))), F(0))


class D2SchurTests(unittest.TestCase):
    def source(self):
        self.assertTrue((ROOT/MODULE).is_file(), 'D2Schur Lean implementation absent')
        return (ROOT/MODULE).read_text()

    def test_exact_declarations(self):
        self.assertEqual(re.findall(r'^theorem\s+([A-Za-z_][A-Za-z0-9_]*)', self.source(), re.M), NAMES)

    def test_no_proof_holes_or_native_shortcut(self):
        text = self.source()
        for forbidden in ('sorry', 'admit', 'native_decide', 'axiom ', 'unsafe '):
            self.assertNotIn(forbidden, text)
        self.assertIn('(hD : d2DetV m2 m4 q ≠ 0)', text)
        self.assertIn('(hm4 : m2 ^ 2 < m4)', text)
        self.assertIn('(hDelta : 0 < d2Delta m2 m4 m6)', text)

    def test_manifest_exact_extension(self):
        self.assertTrue((ROOT/'manifest.json').is_file(), 'manifest absent')
        m = json.loads((ROOT/'manifest.json').read_text())
        self.assertEqual(m['targets'][47:], ['ResearchFormalCoreR1.'+n for n in NAMES])
        self.assertEqual(len(m['targets']), 63)
        self.assertEqual(m['source_modules'][-1], MODULE)
        self.assertEqual(m['alignment_status'], 'PENDING_INDEPENDENT_REVIEW')
        self.assertEqual(m['scientific_effect'], 'NONE')
        for path in (MODULE, 'D2_SCHUR.md', 'tests/test_d2_schur.py'):
            self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(), m['files'][path])
        self.assertEqual(m['files']['gate.py'],
                         'ff8a5374597772b2d37054fd722c2243c4b552a63802c69f7c605519832be6c4')
        self.assertEqual(m['files']['ResearchFormalCoreR1/WeightPerturbation.lean'],
                         '50ce15709a133da12a0b4333cc653d26c5fd8a412976c975a7c2f3b25ed5c00f')

    def test_import_and_blueprint_wiring(self):
        self.assertTrue((ROOT/'ResearchFormalCoreR1.lean').is_file(), 'root absent')
        self.assertIn('import ResearchFormalCoreR1.D2Schur\n', (ROOT/'ResearchFormalCoreR1.lean').read_text())
        text = (ROOT/'blueprint/src/content.tex').read_text()
        self.assertIn(r'\lean{ResearchFormalCoreR1.d2_schur_endpoint_lower}', text)
        self.assertNotIn(r'\leanok', text)

    def test_independent_spectral_gram_rotations(self):
        laws = [((F(1), F(1)),),
                ((F(0), F(1,2)), (F(1), F(1,2))),
                ((F(1), F(1,2)), (F(2), F(1,2))),
                ((F(0), F(1,3)), (F(1), F(1,3)), (F(3), F(1,3))),
                ((F(1,2), F(2,3)), (F(5,2), F(1,3)))]
        count = 0
        for radial in laws:
            atoms = [(sign*r, p/2) for r,p in radial for sign in (-1,1)]
            moment = lambda n: sum((p*x**n for x,p in atoms), F(0))
            m2,m4,m6 = (moment(n) for n in (2,4,6))
            for t in (F(0), F(1,5), F(1,3), F(1,2), F(2,3), F(1), F(2)):
                c,s=(1-t*t)/(1+t*t), 2*t/(1+t*t)
                q=c*c*s*s; k=m4-3*m2*m2
                gram=[[F(0) for _ in range(3)] for _ in range(3)]
                for (x,px),(y,py) in itertools.product(atoms, repeat=2):
                    u,w=c*x+s*y,-s*x+c*y
                    vals=(u*u,u*w,w*w)
                    for i,j in itertools.product(range(3),repeat=2):
                        gram[i][j] += px*py*vals[i]*vals[j]
                D=gram[0][0]*gram[1][1]-gram[0][1]**2
                formula=m2*m2*m4+k*(m4+m2*m2)*q
                numerator=m2*m2*(m4*m4-m2**4)
                self.assertEqual(D,formula)
                self.assertEqual(det3(gram),numerator)
                if D>0:
                    cross=(gram[0][2],gram[1][2])
                    schur=gram[2][2]-(gram[1][1]*cross[0]**2-2*gram[0][1]*cross[0]*cross[1]+gram[0][0]*cross[1]**2)/D
                    self.assertEqual(schur,numerator/D)
                count+=1
        self.assertEqual(count,35)

    def test_all_endpoint_inequalities_on_rational_controls(self):
        count=0
        for m2 in (F(1,3),F(1),F(2)):
            for excess,delta in itertools.product((F(1,7),F(1),F(4)),repeat=2):
                m4=m2*m2+excess; m6=m4*m4/m2+delta
                A=m2*m2*m4; B=(m4-m2*m2)*(m4+3*m2*m2)/4
                N=m2*m2*(m4*m4-m2**4)
                for q in (F(0),F(1,32),F(1,8),F(7,32),F(1,4)):
                    D=A+(m4-3*m2*m2)*(m4+m2*m2)*q
                    T=delta+q*(9*m2*excess-3*delta)
                    self.assertEqual(D,(1-4*q)*A+4*q*B)
                    self.assertGreater(D,0); self.assertGreater(T,0)
                    self.assertLessEqual(D,max(A,B))
                    self.assertLessEqual(N/max(A,B),N/D)
                    count+=1
        self.assertEqual(count,135)

    def test_gaussian_reference_and_required_denominator_guard(self):
        for q in (F(0),F(1,8),F(1,4)):
            self.assertEqual(F(1)**2*(F(3)**2-F(1)**4)/3,F(8,3))
        m2=m4=F(1); q=F(1,4); k=m4-3*m2*m2
        alpha=m4-2*k*q; gamma=m2*m2+2*k*q
        D=m2*m2*m4+k*(m4+m2*m2)*q
        self.assertEqual((D,alpha,gamma),(F(0),F(2),F(0)))
        # Lean's total division gives 0/0=0; at this singular input the
        # subtractive expression is 2 but the numerator/determinant ratio is 0.
        self.assertNotEqual(alpha,F(0))

    def test_wrong_sign_and_missing_soft_factor_are_detected(self):
        m2,m4,q=F(1),F(2),F(1,8)
        D=m2*m2*m4+(m4-3*m2*m2)*(m4+m2*m2)*q
        wrong=m2*m2*m4-(m4-3*m2*m2)*(m4+m2*m2)*q
        self.assertNotEqual(D,wrong)
        N=m2*m2*(m4*m4-m2**4)
        self.assertNotEqual(N,m2*m2*m4*m4)
        self.assertEqual((m4-m2*m2)*(m4+3*m2*m2)/4,F(5,4))


if __name__=='__main__':
    unittest.main()
