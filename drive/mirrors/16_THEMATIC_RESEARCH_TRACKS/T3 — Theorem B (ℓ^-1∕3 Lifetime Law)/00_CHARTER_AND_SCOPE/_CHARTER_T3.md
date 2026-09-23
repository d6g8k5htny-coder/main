U-ID: U011 | Claude / C047R | 2026-07-20

# WORKSPACE CHARTER — T3: Theorem B (near-diagonal lifetime law)
**Item.** ν_B(ℓ) = C*·ℓ^(−1/3)·(1+o(1)); conditional-Morse–Smale form proven-here (C104); exponent exact by Jacobian arithmetic d/p − 1 = −1/3. **Fences:** no numerical C* claimed; no asymptotic exponent value inferred from experiments; the 0.946 constant is a rung-tagged measurement only (root PAIR_DEFECT_DIRECT_R0025_B1200).

**Complete minimal context (load ONLY these):**
- Manuscript: 03_NEAR_DIAGONAL_LIFETIME_LAW.md (in this folder; sha 50dd5b9d). Verified anchors: exact Jacobian r dr = (6^{2/3}/3)κ^{−2/3}ℓ^{−1/3}dℓ (re-verified U005); r³·r⁻⁵·r² = r⁰ cancellation; §12's honest experiment history (failed first proxy 0.1487 vs 2/3; replacement α̂ = −0.4814 CI [−0.7115, −0.2437] contains −1/3 primary; α̂ = −0.5005 [−0.6385, −0.3791] excludes at secondary cutoff).
- Q0_MASTER.md lines 2023–2354 (C104 Theorem B package); 2355–2484 (C103 held-out simulation adjudication); 2485–2658 (C104 continuum-validation adjudication).
- Evidence: C104_COUPLED_PERSISTENCE.json (Drive 1ty3rZNyT6UFMOthPZBMbx6_jZ4sxZajn, 2.4 MB — the coupled-refinement data).
- Shared dependency with T1: the free-jet law diag(2,2,2,6) and E[f_yy|pins] = −b are machine-certified in C047R_verification_battery_01.py (1oUx_WO82T_3xp3jQX9FlbfTW0Sb9apk-).

**Open questions / next actions:**
1. **Queued deep dive:** C104 package + C103/C104 adjudications audited against the 2.4 MB evidence file (stable-bar criterion 2ε_h/4ε_h; truncation; bootstrap design).
2. The adjacency-contact lemma (File 03 §6.2's four-step program) — shared crux with T1's normal-form item and T4's gate.
3. κ-tail domination with one constant set (File 03 §9's status caveat).

**Rule.** Work in this folder cites this charter's context list; amendments by new charter version only.
