# PR-AP-007 — Walker's base-55 set remains optimal in an enlarged adaptive menu

2026-09-16. Complete author-side proof with the specified rational finite certificates; external review and novelty unresolved. This is NOT an extremality claim among all four-AP-free sets or all base-55 alphabets.

## Exact menu

At each node independently allow either:

1. Any admissible alphabet containing zero in ANY radix 2 through 30, as in PR-AP-003.
2. The single published base-55 alphabet D55 in PR-AP-005, or any of its affine images u(D55-v) mod55 with gcd(u,55)=1 and v in D55. There are 42 distinct such alphabets, including D55.
3. Any one of the seven additional explicitly transcribed alphabets from Walker's Table 1 in `code/walker_table_probe.py` (bases 97,105,157,177,191,193; two distinct base-193 alphabets).

Only these supplied larger-base alphabets are allowed. We did NOT enumerate every alphabet in those larger bases. The local cyclic exclusion remains the strong condition forbidding every nonzero difference, including short-period repeated residues.

Let B55 denote the regular sum of the stationary D55 construction. The optimal regular value in this WHOLE adaptive menu is exactly B55.

## Proof

Each allowed action has |D|/b<=2/3 and bounded immediate reward; there are finitely many actions. The Bellman operator is therefore a monotone sup-norm contraction. It suffices to show

    L_action B55 <= B55

for every action, with equality for D55. Its stationary fixed point will then be the global optimal value, allowing arbitrary depth and history-dependent policy.

The proof separates the actions to avoid a huge duplicate search.

The `walker_bridge.json` certificate proves on ALL shifts 0<=a<=1:

    0 <= B55(a)-B0(a) <= 1/25,

where B0 is the base-11 regular sum. Its 128 mesh intervals are covered by the analytic 2-Lipschitz bound on the difference and exact rational endpoint enclosures, not by an unqualified sampled extrapolation.

PR-AP-003's certificate proves for every radix other than 11 and22, throughout 2..30,

    L_action B0 <= B0-1/16.

Therefore

    L_action B55 <= B0-1/16+(2/3)(1/25) < B0 <= B55.

At bases11 and22, rank domination reduces to the two canonical actions. The finer 1024-mesh rational certificates in `walker_local_exclusion.json` prove positive uniform gaps. The reported certified gaps are greater than 0.0021535 and 0.0001706, respectively.

For the seven other published alphabets, `walker_extra_exclusion.json` proves positive uniform gaps; its weakest gap exceeds0.0125. For the 41 nonidentity affine base55 alphabets, `walker_affine_exclusion.json` proves positive uniform gaps; its weakest gap exceeds0.0358. Each certificate uses the same exact centered-moment enclosure plus the 2-Lipschitz extension over every full cell.

The remaining D55 action satisfies L55 B55=B55 identically by the digit recurrence. This proves the Bellman fixed-point assertion and hence global optimality on the exact menu. QED.

## Consequence and limitation

The entire adaptive menu has optimal harmonic sum in

    [4439753369254540648,4439753369254540649]/10^18,

attained by Walker's known stationary construction. Thus simply mixing these published alphabets, even with unlimited history dependence, cannot improve that benchmark. The proof replaces an unbounded-depth search by a finite family of local inequalities and their analytic interval extension.

A better harmonic example could still use a new alphabet at a larger base, a non-modular carry-aware grammar, or a nondigital construction. The global uniform upper bound asked by Erdos #3 is not established.
