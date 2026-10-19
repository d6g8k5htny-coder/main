# C8 own findings on A4 PROOF.md (abed6f94...), recorded BEFORE reading REVIEW.md
Recorded: (see file mtime / sha256 below)

Checked, no defect:
- §2 (4): third/fourth/fifth ridge derivatives — sympy ridge.py: g3=T, g4=Q4, g5=Q5 exactly; t'' = -gamma/a.
- §2 orientation: (u,v)->(-u,-v) flips Q5 and T, fixes Q4 -> unique Q5>0 frame; once-counting.
- §3 Gamma positive definite from dense span; regression kernel f_z; (9) = standard Kac-Rice with jet disintegration.
  Import: G gives the pair identity in full (18) and the one-point identity "analogously" (G l.382, l.450) -> supported, briefly stated.
- §4 (10) |a+lam| (sympy, any theta); SO(2) on raw Sym^3,4,5 det 1 (sympy); block-triangular frame Jacobian; (13) 36 r^5; (14).
- §5 ridge existence/uniqueness, (18), (19)-(21) remainders incl. d_r and d_xi d_r bounds.
- §6 (22) factorisation, heights 0,-19/30,-4/15,-9/10; open conditions; exclusion of extra roots on [-rho,0), [C2 r, rho] (inequality 1/24-1/96-1/48=1/96 >0), scaled interval; (24); inertia.
- §7 enclosure D_r: left/side/right faces <= d_r; interior <= b; ridge path min = d_r; threshold exactly d_r for every cutoff field; elder identification on Morse event; nearer saddle kills the s2 maximum.
- §8 (25) from (9)+(10)-(14); Delta' >0; inverse r_ell; (26); (27); limit (28) incl. L0^(-9/5) and h; positivity; (29) O(ell); cumulative 5/9.
- §9 A_k ledger: (k-1)(k+2)/(2(k+1)) = k/2-1/(k+1); 2/3, 5/4, 9/5.
- Mark extension: w_B Borel on all fields; equals N_A4 a.s.; H m(z)=0 so the cutoff is exactly on omega.

Numeric controls (illustration, not proof):
- controls.py: box conditions hold at 27 points for eps=0.01, fail for eps=0.05 (L0 down to 0.37): U is genuinely small.
- persist2/3/4.py: grid superlevel-H0 persistence with random a,gamma,B,eta and a remote decoy higher than F(M):
  death = farther-saddle ridge value in every case inside the asymptotic regime (r=0.1 seeds 1,3,4,6; r=0.01 seeds 1,2,5; r=0.003 seeds 2,5).
  Outside it (r=0.1 seeds 2,5: no saddle near 3r; r=0.03 seeds 2,5: s3 saddle above s1) the max dies at s1 — allowed by "r<r0".
  L_r for seed 2: 0.71 (r=0.01), 0.85 (r=0.003) -> 0.9; O(r) constant ~20: r0 is small for the full zeta box.

Non-blocking notes:
N1 U is small (eps=0.01 ok, 0.05 not, by corner grid); matters for numerical evaluation of C_A4.
N2 O(r) constants depend strongly on zeta and a; r0 well below 0.03 in field units for the full zeta box (control above). Consistent with README "not finite-scale power laws".
N3 The one-point all-Borel identity is G's "analogous point identity"; (9)'s full-jet disintegration is a short standard step not written out in G.
N4 Agree a sentence defining w_B before (25) helps (extension note).
Verdict (own, pre-comparison): PASS_SCOPED, no must-fix.
