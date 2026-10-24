"""Lens side-24-correction-and-numerics: high-precision cross-check of chain side-24
statements against side24_v1 PROOF/ENCLOSURE.  Read-only on sources."""
import re, sys, json
from fractions import Fraction as Fr
import mpmath as mp

mp.mp.dps = 420
S9 = '<SCRATCH>/s9'
NUM = re.compile(r'-?\d+\.\d+(?:e-?\d+)?')


def line(name, n):
    with open(f'{S9}/named/{name}.txt', encoding='utf-8-sig') as fh:
        return fh.read().split('\n')[n - 1]


def nums(name, n):
    return NUM.findall(line(name, n))


def sigdigits_agree(a, b):
    """number of leading significant decimal digits on which strings a,b agree (mantissa only)."""
    def mant(s):
        s = s.lstrip('-').split('e')[0].replace('.', '').lstrip('0')
        return s
    ma, mb = mant(a), mant(b)
    k = 0
    for x, y in zip(ma, mb):
        if x != y:
            break
        k += 1
    return k, len(ma), len(mb)


out = {}
pi = mp.pi
g16 = mp.gamma(mp.mpf(1) / 6)
g76 = mp.gamma(mp.mpf(7) / 6)
s6 = mp.sqrt(6)
# chain closed forms
c059 = 2 ** (mp.mpf(1) / 6) * 3 ** (mp.mpf(1) / 3) * (29 * s6 - 36) * g16 / (432 * pi ** mp.mpf(2.5))  # LS-DER-059 (1)
c077 = 2 ** (mp.mpf(-23) / 6) * 3 ** (mp.mpf(-8) / 3) * (29 * s6 - 36) * g16 * pi ** mp.mpf(-2.5)   # LS-CLS-077 sec 2
# PROOF (1), d=3
D2 = mp.mpf(29) / 6 - s6
cref = g76 * (mp.mpf(3) / 2) ** (mp.mpf(1) / 3) * D2 / (2 * mp.sqrt(3) * pi ** 2 * mp.sqrt(pi))
out['c059-cref'] = mp.nstr(c059 - cref, 5)
out['c077-cref'] = mp.nstr(c077 - cref, 5)

# P_3(24)
L = 24
P3_055 = Fr(-L * L * (10 * L ** 4 - 147 * L ** 2 + 315), 105)
d = 3
P3_gen = Fr(-d * L ** 2) + Fr((d + 4) * L ** 4, d + 2) - Fr(10 * L ** 6, 3 * (d + 2) * (d + 4))
out['P3(24) from 055 (27)'] = str(P3_055)
out['P3(24) from 055 (23) d=3'] = str(P3_gen)
out['equals -620813376/35'] = (P3_055 == Fr(-620813376, 35) == P3_gen)
out['-576*3233403/105 check (CL-AUD-062 l.96)'] = (Fr(-576 * 3233403, 105) == P3_055)

q = mp.exp(-288)
R1 = mp.mpf(P3_055.numerator) / P3_055.denominator * q
out['R1 = P3 e^-288'] = mp.nstr(R1, 50)
c24 = cref * (1 + R1)
c24_lo = cref * (1 + R1 - mp.mpf(10) ** -180)
c24_hi = cref * (1 + R1 + mp.mpf(10) ** -180)

# PROOF / ENCLOSURE d=3 interval
enc = json.load(open(f'{S9}/side24/ENCLOSURE.json'))
lo3 = mp.mpf(enc['dimensions']['3']['lower']); hi3 = mp.mpf(enc['dimensions']['3']['upper'])
out['ENCLOSURE d3'] = (enc['dimensions']['3']['lower'], enc['dimensions']['3']['upper'])
out['cref in ENCLOSURE'] = bool(lo3 < cref < hi3)
out['chain c24 band (R1 +- 1e-180) in ENCLOSURE'] = bool(lo3 < c24_lo and c24_hi < hi3)
out['cref - lower'] = mp.nstr(cref - lo3, 6)
out['upper - cref'] = mp.nstr(hi3 - cref, 6)
out['PROOF band cref*(1 +- 1e-106) half-width'] = mp.nstr(cref * mp.mpf(10) ** -106, 6)
out['chain |c24-cref|'] = mp.nstr(abs(c24 - cref), 12)
out['|R1| < 1e-106 (PROOF (4) consistent)'] = bool(abs(R1) < mp.mpf(10) ** -106)
out['|R1| / 1e-106'] = mp.nstr(abs(R1) / mp.mpf(10) ** -106, 6)
# decimal places shared c_ref vs chain c24
sref = mp.nstr(cref, 200, strip_zeros=False)
s24 = mp.nstr(c24, 200, strip_zeros=False)
k = 0
for x, y in zip(sref, s24):
    if x != y:
        break
    k += 1
out['cref 70 sig digits'] = mp.nstr(cref, 70)
out['chain c24 (eps dropped) 70 sig digits'] = mp.nstr(c24, 70)
out['cref vs c24: identical leading characters'] = k
out['cref vs c24: first differing decimal place'] = k - 1  # '0.' prefix -> index of place
out['ENCLOSURE width'] = mp.nstr(hi3 - lo3, 5)

# PROOF image constants
E = mp.mpf(21175738586478) * mp.mpf(10) ** -125
out['PROOF E (2)'] = mp.nstr(E, 8)
out['1458*(76*24^6+15)'] = 1458 * (76 * 24 ** 6 + 15)
out['60E < 1e-108'] = bool(60 * E < mp.mpf(10) ** -108)
out['actual nearest-image derivative scale 24^6 e^-288'] = mp.nstr(mp.mpf(24) ** 6 * q, 6)
out['e^-288'] = mp.nstr(q, 30)

# chain LS-DER-059 checks
out['059 (8) 6*720*25^6 e^-288'] = mp.nstr(6 * 720 * mp.mpf(25) ** 6 * q, 6)
out['059 (9) 6 e^-288 (leading Z-1)'] = mp.nstr(6 * q, 6)
z24 = mp.nsum(lambda n: mp.exp(-288 * n * n), [-mp.inf, mp.inf]) ** 3
out['059 (9) exact Z_24-1'] = mp.nstr(z24 - 1, 8)
out['059 (18) R1 string agree sig digits'] = sigdigits_agree(nums('LS-DER-059-v1.0', 322)[0] + 'e-118', mp.nstr(R1, 80))
c059s = nums('LS-DER-059-v1.0', 175)[0]
out['059 (1) printed c_inf agree sig digits'] = sigdigits_agree(c059s, mp.nstr(cref, 80))
abs25 = nums('LS-DER-059-v1.0', 390)[0]
out['059 (25) abs leading agree'] = sigdigits_agree(abs25, mp.nstr(R1 * cref, 80))
abs27 = nums('LS-DER-059-v1.0', 407)[0]
out['059 (27) 24^3 abs agree'] = sigdigits_agree(abs27, mp.nstr(24 ** 3 * R1 * cref, 80))
out['059 (22) R1 in (-1.4863,-1.4861)e-118'] = bool(mp.mpf('-1.4863e-118') < R1 < mp.mpf('-1.4861e-118'))
out['059 (24) R1*c in (-6.210,-6.207)e-120'] = bool(mp.mpf('-6.210e-120') < R1 * cref < mp.mpf('-6.207e-120'))

# LS-DER-061 strings
r061_12 = nums('LS-DER-061-v1.0', 195)[0]
a061_13 = nums('LS-DER-061-v1.0', 204)[0]
c061_14 = nums('LS-DER-061-v1.0', 211)[0]
c061_15 = nums('LS-DER-061-v1.0', 218)[0]
f061_17 = nums('LS-DER-061-v1.0', 238)[0]
r063_1 = nums('LS-DER-063-v1.0', 79)[0]
a063_2 = nums('LS-DER-063-v1.0', 86)[0]
f063_3 = nums('LS-DER-063-v1.0', 92)[0]
aud280 = nums('CL-AUD-LSDER062-20260728-01', 139)[0]
aud162 = nums('CL-AUD-LSDER062-20260728-01', 45)[0]
out['061 (14) c_inf vs closed form: sig digits agree / printed'] = sigdigits_agree(c061_14, mp.nstr(cref, 300))
out['061 (15)-(14) vs (13)'] = mp.nstr(mp.mpf(c061_15) - mp.mpf(c061_14) - mp.mpf(a061_13), 6)
out['061 (15) vs chain c24 (eps dropped): abs diff'] = mp.nstr(mp.mpf(c061_15) - c24, 6)
out['061 (15) sig digits agree with c_inf(1+R1)'] = sigdigits_agree(c061_15, mp.nstr(c24, 300))
out['061 (12) vs 063 (1) agree sig digits'] = sigdigits_agree(r061_12, r063_1)
out['063 (1) vs CL-AUD-062 280-digit agree'] = sigdigits_agree(r063_1, aud280)
out['061 (12) vs CL-AUD-062 162-digit'] = sigdigits_agree(r061_12, aud162)
out['063 (2) vs (1)*c_inf'] = sigdigits_agree(a063_2, mp.nstr(mp.mpf(r063_1) * cref, 300))
out['063 (3) vs (1)-R1'] = sigdigits_agree(f063_3, mp.nstr(mp.mpf(r063_1) - R1, 150))
out['061 (17) vs 063 (3) agree sig digits'] = sigdigits_agree(f061_17, f063_3)
out['063 (1) - R1'] = mp.nstr(mp.mpf(r063_1) - R1, 20)
out['063 (1) relative eps/R1'] = mp.nstr((mp.mpf(r063_1) - R1) / R1, 6)
# LS-DER-062 exp enclosure and intervals
el = mp.mpf(nums('LS-DER-062-v1.0', 217)[0]); eh = mp.mpf(nums('LS-DER-062-v1.0', 221)[0])
out['062 (10) e^-288 in [el,eh]'] = bool(el < q < eh)
Rm = mp.mpf(nums('LS-DER-062-v1.0', 247)[0]); Rp = mp.mpf(nums('LS-DER-062-v1.0', 251)[0])
out['062 (12) R1 in [R1-,R1+]'] = bool(Rm < R1 < Rp)
Cm = mp.mpf(nums('LS-DER-062-v1.0', 283)[0]); Cp = mp.mpf(nums('LS-DER-062-v1.0', 287)[0])
out['062 (17) C+ - C- (width)'] = mp.nstr(Cp - Cm, 8)
out['063 (1) in 062 (17)'] = bool(Cm < mp.mpf(r063_1) < Cp)
out['061 (12) in 062 (17)'] = bool(Cm < mp.mpf(r061_12) < Cp)
out['063 (1) in R1 +- 1e-180'] = bool(abs(mp.mpf(r063_1) - R1) < mp.mpf(10) ** -180)
# exact rational re-derivation of 062 (9)-(10)
N = 900
SN = sum(Fr(288 ** n, 1) / Fr(mp.factorial(n).__int__() if False else 1) for n in range(0))  # placeholder
from math import factorial
SN = Fr(0)
t = Fr(1)
for n in range(N + 1):
    if n > 0:
        t = t * 288 / n
    SN += t
tN1 = t * 288 / (N + 1)
rho = Fr(288, N + 2)
lo_e = 1 / (SN + tN1 / (1 - rho)); hi_e = 1 / SN
out['062 (9) exact rational enclosure reproduces (10) lower to 200 sig'] = sigdigits_agree(mp.nstr(mp.mpf(lo_e.numerator) / lo_e.denominator, 215), nums('LS-DER-062-v1.0', 217)[0])
out['062 (9) exact rational enclosure reproduces (10) upper to 200 sig'] = sigdigits_agree(mp.nstr(mp.mpf(hi_e.numerator) / hi_e.denominator, 215), nums('LS-DER-062-v1.0', 221)[0])

# Relative-vs-absolute: 063 (4)
B3 = 3 * g76 / (2 ** (mp.mpf(4) / 3) * pi ** mp.mpf(3.5))
out['063 (4) B3/c_inf identity'] = mp.nstr(B3 / cref - 2 ** mp.mpf(1.5) * 3 ** (mp.mpf(8) / 3) / ((29 * s6 - 36) * pi), 5)

for kk, v in out.items():
    print(f'{kk}: {v}')
