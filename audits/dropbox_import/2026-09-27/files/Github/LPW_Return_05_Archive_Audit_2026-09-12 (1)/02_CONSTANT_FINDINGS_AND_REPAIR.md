# LPW-05 quantitative audit — exact defects and a proposed repair

**Source under audit:** `raw/constant/LPW_CONSTANT/`, especially `lpw_constant.py` SHA e258322c..., `LPW_CONSTANT_REPORT.md` SHA 289d9f40..., and the lead review SHA 940057a5.... Received sources are unmodified. Numbered convenience copies are in `source_excerpts/`; their line labels are navigation only.

**Disposition:** The received certifier reproduces, but the published strongest-constant package is AMEND REQUIRED. This is NOT a counterexample to the actual pairing probability or a refutation of the qualitative LPW construction. The conservative fallback has a separate argument and review.

## F1 — the exact rational is below the advertised decimal

The delivered integer is B3_rat = 3790446482793. The exact expression in the source is

    c_rat = 260 / (3790446482793 * 2^40 * 10^21)
          = 6.238542702935587792943041084518910355631425081279984...e-44.

Consequently c_rat < 6.239e-44. The displayed implication from this fraction to c >= 6.239e-44 is not valid. A downward-safe decimal for THIS fraction is 6.238e-44, conditional on validity of its mathematical inputs.

The repeated value 6.239277637492383...e-44 is instead produced by replacing the delivered denominator with 3790000000000 (3.79e12). That is a smaller denominator, not an upper bound for the delivered B3_rat. Rounding an upper denominator downward cannot support a lower-bound computation.

**Author-side correction:** Return-04 checked the user-relayed rounded input B3 <= 3.79e12 and its corresponding quotient. That check does not verify the newly delivered, different integer denominator. Any interpretation of the earlier response as certifying the present raw fraction's advertised decimal is withdrawn here. The exact integer controls.

The executable's own final floor is 10^-44 and is arithmetically below c_rat. It does not test the manuscript's sharper 6.239e-44. `falsify.py` likewise tests the power-of-ten floor, not the published decimal. It also contains `ck(Fr(10)**(Nc+1) > c_lo or True, "")`, an inert check; remove it or replace it with a meaningful predicate.

**Required guard:** parse the final displayed coefficient and exact rational as Fractions and require displayed_lower <= exact_lower. Reject drift in the display, receipt, report, and review, not only in the main numeric program.

## F2 — the declared half-normal fourth moment is wrong

The code and report define zeta = |X| + |Y| for independent standard normal X,Y and use

    E[zeta^4] = 12 + 16/pi.

But the exact moments are E|X|=sqrt(2/pi), E|X|^2=1, E|X|^3=2sqrt(2/pi), E|X|^4=3. Therefore

    E[(|X|+|Y|)^4]
      = 2 E|X|^4 + 8 E|X|^3 E|Y| + 6 E|X|^2 E|Y|^2
      = 12 + 32/pi.

The missing amount is 16/pi > 0. The false equality is used in the claimed Fourier C3 fourth-moment bound, which feeds B3 and hence the final coefficient. The delivered mutation suite does not attack this moment identity, so its PASS does not resolve it.

This shows a defect in the supplied derivation. It does not by itself prove that the final B3 number is below the true field moment: the majorant could have additional slack. Any rescue must supply the actual missing argument rather than assert that the wrong equality is harmless.

## Proposed analytic repair: replace the majorant, not the theorem

Here is a short, self-contained alternative lemma. It is NEW AUTHOR-SIDE REPAIR MATERIAL requiring Kimi's explicit review; it is not silently inserted into a frozen report.

Let p_k be the exact normalized symmetric lattice weights, sum p_k=1. Take independent standard normal X_k,Y_k indexed by the full lattice and define

    f(x) = sum_k sqrt(p_k) [X_k cos(k.x) + Y_k sin(k.x)].

Its covariance is exactly sum_k p_k cos(k.(x-y)), the prescribed real Gaussian field covariance. The use of the full lattice with independent coefficient pairs already has the correct normalization: no extra factor of two is inserted. The zero-frequency Y term vanishes, which only makes the following bound more conservative.

Set R_k=sqrt(X_k^2+Y_k^2). Every phase-rotated derivative contribution is bounded in absolute value by |k^alpha|R_k. Thus, with the source's SAME S_p=sum_k sqrt(p_k)(1+|k|)^p,

    ||f||_{C^p} <= sum_k sqrt(p_k)(1+|k|)^p R_k.

The Gaussian moments give E[R_k^4]=3+2+3=8 and E[R_k] <= sqrt(E[R_k^2])=sqrt(2). Minkowski therefore gives

    E||f||_{C3}^4 <= 8 S_3^4,
    E||f||_{C4}   <= sqrt(2) S_4.

Since 8 < 12+16/pi, and sqrt(2) < 2sqrt(2/pi) because pi<4, the old numeric multipliers can be retained as CONSERVATIVE UPPER CAPS for this new Rayleigh majorant. They must no longer be called the exact moments of |X|+|Y|. This alternative may preserve the old B3/B4 upper values after the rest of the chain is validated, or it can sharpen them if the smaller caps are used.

Two valid repair paths:

A. Keep the original half-normal-sum majorant, replace its moment by 12+32/pi, and recompute B3 and c.
B. Adopt the explicit Rayleigh-majorant lemma above, revise the code/report semantics, and re-enclose all numerical caps and the final printed coefficient. This is the preferred minimal repair.

Do not choose between these silently. Issue a new version, retain the predecessor, and bind a new review verdict to the successor's exact bytes.

## F3 — certification scope versus point-arithmetic replay

The main program uses ordinary high-precision mpmath `mpf` operations and a blanket ROUND=1e-45; it is not a directed-rounding interval program. That does not automatically make the bounds false: rigorous forward-error analysis is a legitimate alternative. But byte-identical replay is not that analysis.

The header's rough magnitude/operation-count justification does not by itself discharge all propagation through exponentiation, logarithms, square roots, matrix operations, and integer ceiling boundaries. Some final quantities exceed the rough magnitude described there. This intake has NOT reconstructed a complete forward-error certificate for every claimed enclosure.

For the repaired package, either (i) return a directed-rounding/rational enclosure implementation for the few consumed final scalar caps, or (ii) give a complete proved error ledger, including amplified errors and safe distances to every integer/decimal rounding boundary. Avoid claiming that every output is interval-enclosed merely because an analytic all-r modulus was proved. Mathematical uniformity and numeric rounding assurance are distinct.

This is an OPEN CERTIFICATION OBLIGATION, not a claimed observed violation of lambda, m, B3, or B4.

## F4 — the unconditioned J covariance is not diagonal

The lead review says all off-diagonals of Cov(J) vanish by parity at the origin. For J=(f_yy,f_xxy/2,f_xyy/2,f_yyy/6),

    Cov(f_xxy/2, f_yyy/6) = a2*a4/12 != 0.

The actual code's Gershgorin row sums include this term. Thus the current code handles it; the review prose should be corrected. Do not confuse the unconditioned Cov(J) with the conditional endpoint Schur complement, whose stated diagonal form is a different object.

## F5 — radius and fallback scope

The integer radius 1/(256*9432)=1/2414592 is algebraically correct. It corresponds to the conservative cap K=9432 and gives 16delta+8Kr0=3/64<1/16. It is smaller than 1e-5.

The interval (0,10^-28] is a subset of (0,1/2414592]. The fallback does NOT extend the improved theorem to smaller radii: the improved statement already has no positive lower cutoff. The fallback is valuable as a separately justified, independently reviewable route—not extra r-domain coverage. Correct this sentence in the lead review.

## Acceptance target for the successor

1. Exact majorant, moment identity, Fourier normalization, tails and residual norm specified.
2. m, covariance floors and B3/B4 on the same changed compact set and full r interval.
3. Complete numeric assurance, not a point replay claim.
4. Final decimal generated from an exact rational LOWER certificate, with a machine guard.
5. New negative controls: wrong fourth-moment cross term; upward decimal display; shrunken box failing containment; sine omission at nonzero separation; body-hash marker ambiguity.
6. Same-source replay in a fresh directory, new hashes, and an explicit mathematical review of the changed lemma and its consumers.

No new owner vote is required. Unresolved evidence is the gate.
