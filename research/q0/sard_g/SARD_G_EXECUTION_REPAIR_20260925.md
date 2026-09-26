# SARD-G execution repair — countable charts without selection and endpoint-support localization

**Object:** SARD-G-EXEC-20260925-v1
**Author:** OpenAI / ChatGPT
**Disposition:** AUTHOR-SIDE REPAIR / NONCONTROLLING / INDEPENDENT REVIEW REQUIRED
**Scientific effect:** NONE. The historical transversality manuscript and all status registers remain unchanged.
**Predicate successor:** chart-membership repair of commit d49be07074bd0696cc034757379771eae3c852de after the A1/A6 amendment. Sections 4–6 are unchanged. A fresh nonauthor re-check of A1 and A6 is required before the full-Gaussian C103 corollary can move.

## 1. Exact repair target

The source manuscript is the 19,242-byte Gaussian-transversality working source, Git blob a0f4aa6ab6d0e21e7a24b4a6e1aa4fdebc40efdc. Specialist review KIMI-AUD-022, Git blob e968385a3bf4e8318a9ab6cb4f57818828f01b18, judged the architecture correct but left two execution-level clarifications:

- Clarification A: complete the countable chart / measurable-selection bookkeeping.
- Clarification C: justify the endpoint contribution instead of relying on undisplayed symbolic two-jet coefficients.

This successor removes both avoidable claims. It does not select a first chart, and it does not assert that endpoint launch variation is literally a finite combination of endpoint two-jets. The chart predicate below is the later A1/A6 repair: membership requires the designated saddle to be the unique critical point of its closed outer box, with a quantitative gradient exclusion outside its continuation neighborhood.

## 2. Functional-analytic setting

Let X=C^2(T_L^2) with its usual separable Banach topology; the periodized Bargmann–Fock sample paths are almost surely smoother than X. Let mu be their centered Gaussian law and H its Cameron–Martin space.

Finite-jet nondegeneracy and the usual Bulinskaya/Kac–Rice argument give a full-mu set Omega_gen on which critical points are nondegenerate and critical values are distinct. The present repair concerns the additional event that two distinct index-one saddles of f in Omega_gen are connected by a gradient trajectory.

## 3. Countable chart family — no measurable selection

Fix once and for all countable rational data:

- nested disjoint rational coordinate boxes B_p^- compactly inside B_p^+ and B_q^- compactly inside B_q^+;
- rational continuation neighborhoods N_p compactly inside B_p^- and N_q compactly inside B_q^-;
- rational hyperbolicity margins delta>0 and rational gradient-exclusion margins eta>0 on the compact complements closure(B_p^+ minus N_p) and closure(B_q^+ minus N_q);
- one of the two local stable/unstable branch labels at each saddle, encoded by which member of a rational local-section pair the branch crosses;
- a polygonal tube with rational vertices and rational width;
- a rational transverse line segment Sigma inside the tube;
- rational lower bounds on speed and crossing angle and a rational upper bound on travel time.

For a tuple chi of these data, define U_chi subset X to be the set of fields for which:

1. the closed outer box closure(B_p^+) contains exactly one critical point; that point lies in the continuation neighborhood N_p, is index one, and its Hessian spectral gap exceeds delta. On the compact complement closure(B_p^+ minus N_p), |grad f| is at least eta. The same holds for q, with N_q. Every critical point of the tracked outer box is therefore the designated saddle inside its continuation neighborhood;
2. the declared local branches exist and meet their declared local sections uniquely and transversely;
3. the forward p-branch and backward q-branch remain inside the declared tube until their unique first crossing of Sigma;
4. speed, crossing-angle and travel-time margins satisfy the declared strict inequalities.

All conditions use strict quantitative margins. For condition 1, let K_p = closure(B_p^+ minus N_p). The bound |grad f| >= eta on the compact set K_p is open in the C^1 topology, and it forces every critical point in closure(B_p^+) to lie in N_p. The boundary of N_p is contained in K_p, so that unique critical point is interior to N_p. Its index and spectral gap persist under a small C^2 perturbation. Uniqueness persists as well: on closure(N_p) minus a small ball about the saddle, |grad f| has a positive minimum, because the saddle is the only zero in the compact outer box, and a sufficiently small C^1 perturbation creates no new zero on that compact remainder or on K_p. The implicit-function theorem keeps exactly one nondegenerate zero inside the ball. A degenerate critical point outside N_p violates the gradient margin, so it is not a member of U_chi and cannot split into an additional index-one saddle while remaining in U_chi. The same holds at q. Conditions 2–4 are open by the implicit-function theorem, the parameter-dependent local stable/unstable-manifold theorem, and continuous dependence of ODE solutions and transverse hitting times. Therefore U_chi is open in X. On U_chi the tracked critical points, local branch crossings and first crossings of Sigma depend continuously on f; after fixing the signed coordinate sigma on Sigma,

    D_chi(f) = sigma(z_u(f)) - sigma(z_s(f))

is C^1 on U_chi.

Every genuine nondegenerate saddle–saddle connection belongs to at least one U_chi: choose nested endpoint boxes so the saddle is the only critical point in the outer box, then choose a rational continuation neighborhood N around that saddle and inside the inner box. Compactness gives |grad f| > 0 on closure(B^+ minus N); choose a smaller rational margin eta. Then choose local sections, a compact regular segment of the connection, a narrow tube and an interior transverse section. Nondegeneracy and compactness provide positive hyperbolicity, gradient-exclusion, speed, separation and transversality margins; choose smaller rational margins and rational approximations of the geometric data. Therefore

    Connection ∩ Omega_gen  subset  union_chi { f in U_chi : D_chi(f)=0 },

with a countable union.

**No 'first rational chart' is selected.** Consequently no measurable-selection theorem is needed for this step. Each chart event is Borel because U_chi is open and D_chi is continuous.

## 4. Endpoint variation — localization, not two-jet atoms

Choose the endpoint boxes/local sections so their closures are disjoint from the compact interior orbit segment used in the chart. The local p-branch launch map is determined by the vector field inside the chosen p-neighborhood up to its local section; similarly for q. Parameter-dependent invariant-manifold theory makes these launch maps C^1 on U_chi.

Let L_p and L_q be the derivatives of the two launch/crossing contributions to dD_chi. If a perturbation h vanishes on the p-neighborhood, then grad h vanishes there, the local vector field is unchanged there, and the p saddle/local branch/local-section launch are unchanged. Hence L_p(h)=0. Thus L_p is a continuous linear functional supported in the closure of the p-neighborhood; likewise L_q is supported near q.

As continuous linear functionals on C^2, L_p and L_q are finite-order distributions supported in those endpoint neighborhoods. This is all the later nonvanishing argument needs. There is no need to identify them with a finite list of coefficients multiplying h(p), grad h(p), or H_h(p).

Writing the interior adjoint contribution on the compact regular segment as

    L_curve(h) = integral a(t) n(t) · grad h(gamma(t)) dt,

we have

    dD_chi = L_curve + L_p + L_q

as a compactly supported finite-order distribution on the torus.

## 5. Nonvanishing on the Cameron–Martin space

For the periodized Bargmann–Fock covariance every lattice Fourier weight is strictly positive. Therefore convolution/kernel embedding by K_L is injective on finite-order distributions on the torus: if K_L*T=0 then every Fourier coefficient of T is zero.

Suppose dD_chi vanished on H. Its RKHS Riesz representative would be zero, so injectivity would force the distribution T=L_curve+L_p+L_q to be zero.

Choose a smooth test function psi supported in a small neighborhood of an interior point of the compact orbit segment and disjoint from both endpoint neighborhoods. Then L_p(psi)=L_q(psi)=0. The adjoint density a(t)n(t) is continuous and nonzero on the regular segment. In local flow-box coordinates choose psi so that its transverse derivative has fixed sign on a sufficiently short subsegment. Then L_curve(psi) != 0, contradiction.

Hence dD_chi restricted to H is a nonzero continuous linear functional at every charted connection.

## 6. Countable Cameron–Martin directions with genuine 1D decomposition

Choose a countable dense family of continuous linear functionals ell_j in X* whose covariance images Q ell_j are dense in H; this is possible because H is the closure of Q(X*) and H is separable. Normalize

    h_j = Q ell_j / sqrt( Var[ell_j(f)] ).

Then h_j is dense in the unit sphere of H (after discarding zero members). For each j define

    xi_j = ell_j(f) / sqrt( Var[ell_j(f)] ).

so xi_j is standard Gaussian and

    f = g_j + xi_j h_j

where g_j=f-xi_j h_j is Gaussian and independent of xi_j. This is the one-dimensional Gaussian decomposition actually used in the slicing argument.

At any charted zero, dD_chi is nonzero on H, so some h_j satisfies dD_chi[h_j] != 0.

Thus

    Connection ∩ Omega_gen
      subset union_{chi,j} { f in U_chi : D_chi(f)=0 and dD_chi(f)[h_j] != 0 }.

The union is countable.

## 7. Slicing on chart-validity intervals

Fix chi,j and condition on g_j. The set

    I_{g,chi} = { xi in R : g_j + xi h_j in U_chi }

is open because the repaired U_chi is open. Membership uses condition 1 of Section 3: the designated saddle is the unique critical point of the closed outer box, and |grad f| has strict positive lower bound eta on the compact complement of its continuation neighborhood. Hence I_{g,chi} is a countable disjoint union of open intervals. On each interval

    F_g(xi)=D_chi(g_j+xi h_j)

is C^1.

Every zero counted in the event above satisfies F_g'(xi) != 0, hence is isolated. A discrete subset of an interval is countable, and a countable union of such sets is countable. The conditional law of xi_j has a smooth density, so the conditional probability of the regular zero set is zero. Integrating over the independent residual g_j gives probability zero for the fixed chi,j event. Countable union over chi,j gives

    mu(Connection ∩ Omega_gen)=0.

Because mu(Omega_gen)=1, the saddle–saddle connection event has probability zero.

## 8. Why Section 11 no longer needs an analytic-set projection

The proof above never asserts that every field on a one-dimensional shift line is Morse. A chart U_chi is used only on its open validity intervals, and those intervals are open because U_chi is the repaired predicate of Section 3. A hidden degenerate critical point outside the continuation neighborhood violates the gradient margin, so it is not a chart member and is not sliced. Degenerate/tangent/boundary parameters simply lie outside that chart interval. Every nondegenerate connection in Omega_gen is captured by some chart with positive margins, and every regular chart zero is handled by the interval slicing argument.

Thus the global a.s. finite-jet statement mu(Omega_gen)=1 is sufficient. No projection of the parametric degeneracy set to the xi-axis is needed for this connection event.

## 9. Scope

This repairs only SARD-G's execution-level charting/slicing and endpoint-support interfaces. It does not prove the finite-jet nondegeneracy input, Q0-C103's Kac–Rice regional estimates, Theorem B, RN/JETMOD, or any 3D theorem.

Independent review should separately check:

A1 openness/countability/coverage of the repaired U_chi, including the continuation-neighborhood gradient exclusion;
A2 parameter-dependent local invariant-manifold and first-crossing regularity;
A3 endpoint-support localization and distribution order;
A4 RKHS injectivity for these supported distributions;
A5 dense covariance-image directions and the exact Gaussian decomposition;
A6 interval slicing on the repaired open predicate and elimination of the old exceptional-parameter projection.

Any failed item leaves the full-Gaussian C103 corollary on HOLD. The predicate repair in Sections 3, 7, 8, and 10 is author-side only. It does not move that corollary. A fresh nonauthor re-check of A1 and A6 is required before the corollary can move. Sections 4–6 were not rewritten.

## 10. Negative control — old predicate fails openness; repaired predicate excludes the example

Let T^2 = R^2/Z^2 and

    f_eps(x,y) = (-5+eps) cos(2 pi x) - cos(4 pi x) + cos(6 pi x) + cos(2 pi y).

Index one means unstable dimension one for x-dot = grad f, equivalently exactly one positive Hessian eigenvalue. The reproducible check is

    python3 research/q0/sard_g/chart_predicate_negative_control.py

On the predecessor chart, the inner cover rectangle is x in (0.30, 1.08), y in (0.82, 1.18), and the outer cover rectangle is x in (0.22, 1.16), y in (0.74, 1.26). Rational margins are delta = 1 and eta = 1. At eps = 0 the field has exactly one index-one critical point in the inner box, namely (1/2, 0), with spectral gap 39.478418, and the sampled annulus minimum of |grad f| is 5.691911. The predecessor predicate therefore accepts f_0. That accepted field still has the degenerate critical point (0, 0) inside the inner box. At eps = 10^{-3} the C^2 size of the perturbation eps cos(2 pi x) is 0.039478, and the inner box contains three index-one points, including new ones at x = 0.001592 and x = 0.998408. The predecessor predicate rejects f_eps. Acceptance of f_0 and rejection of this nearby field is the openness failure.

The repaired predicate on the same outer box, with continuation neighborhood x in (0.45, 0.55), y in (0.92, 1.08) about the designated saddle, sees four critical points in the closed outer box at eps = 0. It excludes f_0. The degenerate point (0, 0) is the witness outside that continuation neighborhood.

The repaired predicate is not empty along this family. On the tight outer rectangle x in (0.42, 0.58), y in (0.88, 1.12), with continuation neighborhood x in (0.47, 0.53), y in (0.94, 1.06), both f_0 and f_eps have the saddle (1/2, 0) as their unique critical point, with gap 39.478418 and sampled complement minimum of |grad f| equal to 2.322725, so both satisfy the repaired predicate. The control checks the chart predicate only. It preserves countability, endpoint-support localization, RKHS injectivity, the one-dimensional Gaussian decomposition, and the removal of the analytic-set projection. Scientific effect: NONE.
