# Reported sharper constant — acceptance contract and scalar consistency

Status: REPORTED CERTIFICATE / NOT RECEIVED LOCALLY. The user's relay is the source for m>=10^-21, B3<=3.79e12, B4<=4715.75, radius 1/2414592, and c>=6.239e-44. No full certificate appears in this turn's files.

The scalar arithmetic is consistent with the original LPW sufficient bounds. The included exact-rational checker evaluates:

    c_calc = 260 * 10^-21 * 2^-40 / 3790000000000
           > 6.239 * 10^-44.
    K_required = 2*4715.75 = 9431.5.
    256*K_required = 2414464 < 2414592.

The radius is therefore conservative (not inconsistent) relative to the reported B4. Choosing K=9432 makes its denominator exactly 256K. At that choice, the original tolerance is 16/1024+8K/(256K)=3/64<1/16. The original r<=1 condition is automatic.

Similarly, if a uniform modulus 19.071156*R and endpoint floor 0.124 are valid on R=10^-5, the derived floor is 0.12380928844; reporting the smaller 0.1238092884 has the correct direction. The modulus itself still needs the exact proof/certificate; arithmetic does not certify it.

The coefficient gain over 10^-1235 is about 1192 decimal orders, not an exact '1200 orders' identity. More important than the gain is that the theorem still targets a positive lower bound, not the actual failure probability or sharp limiting constant. At the reported radius its displayed lower wall is of order 10^-63, not an estimate of observed frequency.

## Mandatory delivered objects

- Exact theorem statement and changed compact set, with full target hash and version.
- Proof of thin-box containment for every r in the claimed interval.
- m defined and enclosed uniformly over that same interval and compact set.
- B3 as a six-pin conditional fourth moment of max(1,C3 norm); B4 as a ten-coordinate conditional M4 first-moment bound; no swap of norm, event, or law.
- Complete normalized covariance basis, exact cos+sin derivative formula, tails and a continuum modulus; no reuse of the erroneous RB rung table.
- Nonstationary regression residual bound and correct Markov-then-jet integration.
- Directed arithmetic or exact rational enclosure, environment/tool versions, original script hashes, normal/-O output and exits.
- All four contract mutation specimens plus actual rejection receipts.
- A reviewed analytic justification for every non-arithmetic bound.

## Admission rule

The owner gate is already satisfied. Admit the stronger explicit pair only after the above evidence is received, matched to the theorem hash, replayed as appropriate, and all applicable technical review predicates pass. A missing certificate remains missing regardless of the generosity of the operator's authorization.
