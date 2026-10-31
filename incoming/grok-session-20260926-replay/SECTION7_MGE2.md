# A7 m≥2 integral (7.4) — written comparison

Scientific effect: NONE. Does not accept Theorem A.

Parent §7 after the landed erratum. Cap pairing identities already on Math- default (`5ed3b455`).

## Measure

Symmetric-matrix Lebesgue measure in eigenvalue coordinates is a constant times

    product_{i<j} |λ_j-λ_i| × angular volume.

This is a change of variables, not a GOE claim. On 0 ≤ λ_1 ≤ ⋯ ≤ λ_m = Λ the Vandermonde is at most Λ^{m(m-1)/2}. Angular space is compact. After (3.5) and angular integration the majorant is

    C exp(-c ∑ λ_i²) Λ^{m(m-1)/2} dλ_1⋯dλ_m.     (7.2)

## Weight

From (6.2), r ≤ 1, h ≤ K U, U = J + Λ ≥ 1:

    W_r ≤ C r² U² · λ_1(λ_1 + E r U) · ∏_{j=2}^m λ_j(λ_j + r h).

The product over j ≥ 2 and the leftover powers of U are ≤ C U^{2m} on the ordered positive orthant. Depth failure is λ_1 ≤ D r U².

## λ_1 integral

Extend the integration interval to [0, D r U²]:

    ∫_0^{D r U²} λ_1(λ_1 + E r U) dλ_1
      = r³ [(D³/3) U⁶ + (E D²/2) U⁵]
      ≤ C r³ U⁶.                                        (7.3)

No Gaussian factor in λ_1 is needed for an upper bound. λ_2 is not cut away from zero.

## Resulting power

    E[W 1_{depth}] ≤ C r⁵ E ∫ (J+Λ)^{2m+6} Λ^{m(m-1)/2}
                         exp(-c ∑_{j≥2} λ_j²) dλ_2⋯dλ_m.

The integral is finite by Gaussian-polynomial tails and uniform J-moments of order 2m+6. Hence E[W 1_{depth}] = O(r⁵). After an A3 floor Z = Θ(r²) this is O(r³).

Corank ≥ 2 sits inside the same integral: only nonnegative powers of Λ and λ_j.

## What this does not close

- (3.5) and (4.3) as independently reviewed inputs
- A3 floor as a reviewed theorem
- m=1 far branch (separate note; smaller power)
- cap pairing (embedded 2r chart + Morse/§8)
- Theorem A
