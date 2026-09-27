# SARD-G execution repair — countable charts without selection and endpoint-support localization

**Object:** SARD-G-EXEC-20260925-v1
**Author:** OpenAI / ChatGPT
**Disposition:** AUTHOR-SIDE REPAIR / NONCONTROLLING / INDEPENDENT REVIEW REQUIRED
**Scientific effect:** NONE. The historical transversality manuscript and all status registers remain unchanged.
**Predicate successor:** chart-membership repair of commit d49be07074bd0696cc034757379771eae3c852de after the A1/A6 amendment, then of commit a1fc9581291b6f81c76c903c96a13955b51f8d54 after the geometric-boundary amendment (relative-interior section hits with clearance margins; Section 3 conditions 2–3). Sections 4–6 are unchanged. A fresh nonauthor re-check of A1 and A6 is required before the full-Gaussian C103 corollary can move.

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
- rational lower bounds on speed and crossing angle and a rational upper bound on travel time;
- a rational clearance margin rho>0 and a rational time margin tau>0 for the section hits.

For a tuple chi of these data, define U_chi subset X to be the set of fields for which:

1. the closed outer box closure(B_p^+) contains exactly one critical point; that point lies in the continuation neighborhood N_p, is index one, and its Hessian spectral gap exceeds delta. On the compact complement closure(B_p^+ minus N_p), min |grad f| strictly exceeds eta. The same holds for q, with N_q. Every critical point of the tracked outer box is therefore the designated saddle inside its continuation neighborhood;
2. the declared local branches exist and meet their declared local sections transversely at points whose distance from each endpoint of the section exceeds rho; before that hit each branch stays at distance greater than rho from the closed local section, so there is no earlier or tangential contact;
3. the forward p-branch, followed from its local-section hit, reaches Sigma at a finite time T_p below the declared travel-time bound with a transverse crossing at a point whose distance from both endpoints of Sigma exceeds rho; on [0, T_p - tau] the arc stays at distance greater than rho from the closed segment Sigma; on [0, T_p] the arc stays at distance greater than rho from the boundary of the tube. The same holds for the backward q-branch with its time T_q;
4. speed along the arcs, crossing angles at Sigma and at the local sections, and travel times satisfy the declared strict inequalities.

All conditions use strict quantitative margins. For condition 1, let K_p = closure(B_p^+ minus N_p) and m_p(f) = min_{K_p} |grad f|. Since |m_p(f)-m_p(g)| <= ||grad f-grad g||_infinity, the strict bound m_p(f) > eta is open in the C^1 topology: a perturbation with gradient norm smaller than m_p(f)-eta preserves it. This bound forces every critical point in closure(B_p^+) to lie in N_p. The boundary of N_p is contained in K_p, so that unique critical point is interior to N_p. Its index and spectral gap persist under a small C^2 perturbation. Uniqueness persists as well: on closure(N_p) minus a small ball about the saddle, |grad f| has a positive minimum, because the saddle is the only zero in the compact outer box, and a sufficiently small C^1 perturbation creates no new zero on that compact remainder or on K_p. The implicit-function theorem keeps exactly one nondegenerate zero inside the ball. A degenerate critical point outside N_p violates the gradient margin, so it is not a member of U_chi and cannot split into an additional index-one saddle while remaining in U_chi. The same holds at q. Conditions 2–4 are open, and the clearance clauses are what makes them so. The tracked saddle, its local branches and their local-section hits depend C^1 on f by the implicit-function theorem and the parameter-dependent local stable/unstable-manifold theorem. Fix f in U_chi with p-arc gamma on [0, T_p]. By continuous dependence of solutions on compact time intervals, jointly in the initial point and in the C^1 field, the arcs of every g close to f in X stay uniformly close to gamma on [0, T_p + tau]. Each geometric requirement is a strict inequality between continuous quantities evaluated on a compact set: the distance from the arc on [0, T_p - tau] to the closed segment Sigma, the distance from the arc on [0, T_p] to the tube boundary, the distance from the crossing point to the endpoints of Sigma, the speed minimum, and the crossing angle. Strict inequalities of this kind persist under uniform perturbation. The clause is needed at the crossing itself: transversality alone gives a nearby transverse crossing of the supporting line of Sigma, not of the segment, and a field whose crossing sits at an endpoint of Sigma has nearby fields that miss the segment entirely (Section 10). With the clause, the arc of g on [0, T_p - tau] stays at distance greater than rho/2 from Sigma, so g crosses nothing there; on [T_p - tau, T_p + tau] transversality at gamma(T_p) gives exactly one crossing of the supporting line, at a point within rho/2 of gamma(T_p), hence in the relative interior of Sigma at distance greater than rho/2 from its endpoints, with crossing time C^1 in g. That crossing is therefore the unique first crossing of Sigma by the g-arc, and it is the continuation of the f-crossing. The same argument applies to the local-section hits of condition 2 and to the backward q-branch. Therefore U_chi is open in X. On U_chi the tracked critical points, local branch crossings and first crossings of Sigma depend continuously on f; after fixing the signed coordinate sigma on Sigma,

    D_chi(f) = sigma(z_u(f)) - sigma(z_s(f))

is C^1 on U_chi.

Every genuine nondegenerate saddle–saddle connection belongs to at least one U_chi: choose nested endpoint boxes so the saddle is the only critical point in the outer box, then choose a rational continuation neighborhood N around that saddle and inside the inner box. Compactness gives |grad f| > 0 on closure(B^+ minus N); choose a smaller rational margin eta. Then choose local sections, a compact regular segment of the connection, a narrow tube and an interior transverse section. Nondegeneracy and compactness provide positive hyperbolicity, gradient-exclusion, speed, separation and transversality margins. The connection crosses the chosen Sigma transversely at one point of its relative interior, so its distance from the endpoints of Sigma is positive; the compact arc up to any time before that crossing is disjoint from the closed segment Sigma, so its distance from Sigma is positive; and the compact travelled arc lies in the open tube at positive distance from the tube boundary. The same holds at the local sections. Choose rational rho and tau below these positive values, smaller rational margins throughout, and rational approximations of the geometric data. Therefore

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

is open because the repaired U_chi is open. Membership uses conditions 1–4 of Section 3: the designated saddle is the unique critical point of the closed outer box with min |grad f| > eta on the compact complement of its continuation neighborhood, and the branch arcs hit their sections in the relative interior with the strict clearance margins rho and tau. Hence I_{g,chi} is a countable disjoint union of open intervals. On each interval

    F_g(xi)=D_chi(g_j+xi h_j)

is C^1.

Every zero counted in the event above satisfies F_g'(xi) != 0, hence is isolated. A discrete subset of an interval is countable, and a countable union of such sets is countable. The conditional law of xi_j has a smooth density, so the conditional probability of the regular zero set is zero. Integrating over the independent residual g_j gives probability zero for the fixed chi,j event. Countable union over chi,j gives

    mu(Connection ∩ Omega_gen)=0.

Because mu(Omega_gen)=1, the saddle–saddle connection event has probability zero.

## 8. Why Section 11 no longer needs an analytic-set projection

The proof above never asserts that every field on a one-dimensional shift line is Morse. A chart U_chi is used only on its open validity intervals, and those intervals are open because U_chi is the repaired predicate of Section 3. A hidden degenerate critical point outside the continuation neighborhood violates the strict gradient margin min |grad f| > eta, so it is not a chart member and is not sliced. Equality at eta is excluded as well. Degenerate/tangent/boundary parameters simply lie outside that chart interval. Every nondegenerate connection in Omega_gen is captured by some chart with positive margins, and every regular chart zero is handled by the interval slicing argument.

Thus the global a.s. finite-jet statement mu(Omega_gen)=1 is sufficient. No projection of the parametric degeneracy set to the xi-axis is needed for this connection event.

## 9. Scope

This repairs only SARD-G's execution-level charting/slicing and endpoint-support interfaces. It does not prove the finite-jet nondegeneracy input, Q0-C103's Kac–Rice regional estimates, Theorem B, RN/JETMOD, or any 3D theorem.

Independent review should separately check:

A1 openness/countability/coverage of the repaired U_chi, including the continuation-neighborhood gradient exclusion and the relative-interior section hits with clearance margins rho and tau;
A2 parameter-dependent local invariant-manifold and first-crossing regularity;
A3 endpoint-support localization and distribution order;
A4 RKHS injectivity for these supported distributions;
A5 dense covariance-image directions and the exact Gaussian decomposition;
A6 interval slicing on the repaired open predicate and elimination of the old exceptional-parameter projection.

Any failed item leaves the full-Gaussian C103 corollary on HOLD. The predicate repair in Sections 3, 7, 8, and 10, including the geometric-boundary amendment, is author-side only. It does not move that corollary. A fresh nonauthor re-check of A1 and A6 is required before the corollary can move. Sections 4–6 were not rewritten.

## 10. Negative control — old predicate fails openness; repaired predicate excludes the example

Let T^2 = R^2/Z^2 and

    f_eps(x,y) = (-5+eps) cos(2 pi x) - cos(4 pi x) + cos(6 pi x) + cos(2 pi y).

Index one means unstable dimension one for x-dot = grad f, equivalently exactly one positive Hessian eigenvalue. The reproducible check is

    python3 research/q0/sard_g/chart_predicate_negative_control.py

On the predecessor chart, the inner cover rectangle is x in (0.30, 1.08), y in (0.82, 1.18), and the outer cover rectangle is x in (0.22, 1.16), y in (0.74, 1.26). Rational margins are delta = 1 and eta = 1. At eps = 0 the field has exactly one index-one critical point in the inner box, namely (1/2, 0), with spectral gap 39.478418, and the sampled annulus minimum of |grad f| is 5.691911 (a NON-CERTIFYING grid value; the true compact minimum is 2 pi sin(2 pi 0.18) = 5.685196, attained on y = 0.82, so the sampled number is not a lower bound; harmless at eta = 1). The predecessor predicate therefore accepts f_0. That accepted field still has the degenerate critical point (0, 0) inside the inner box. At eps = 10^{-3} the C^2 size of the perturbation eps cos(2 pi x) is 0.039478, and the inner box contains three index-one points, including new ones at x = 0.001592 and x = 0.998408. The predecessor predicate rejects f_eps. Acceptance of f_0 and rejection of this nearby field is the openness failure.

The repaired predicate on the same outer box, with continuation neighborhood x in (0.45, 0.55), y in (0.92, 1.08) about the designated saddle, sees four critical points in the closed outer box at eps = 0. It excludes f_0. The degenerate point (0, 0) is the witness outside that continuation neighborhood.

The repaired predicate is not empty along this family. On the tight outer rectangle x in (0.42, 0.58), y in (0.88, 1.12), with continuation neighborhood x in (0.47, 0.53), y in (0.94, 1.06), both f_0 and f_eps have the saddle (1/2, 0) as their unique critical point, with gap 39.478418 and sampled complement minimum of |grad f| equal to 2.322725 (NON-CERTIFYING grid value; the true compact minimum is 2 pi sin(2 pi 0.06) = 2.312995 on y = 0.94), so both satisfy the repaired predicate. The control checks the chart predicate only. It preserves countability, endpoint-support localization, RKHS injectivity, the one-dimensional Gaussian decomposition, and the removal of the analytic-set projection. Scientific effect: NONE.

**Equality-boundary control (analytic).** The added counterexample is from the [OpenAI A1/A6 review](https://github.com/d6g8k5htny-coder/main/blob/cc18155abf912d3b8f25ac133ea42bb327287709/reviews/sard_g_a1_a6_20260926/REVIEW.md), a source-exposed same-provider contribution with zero organizational-independence credit. Strictness at the fixed threshold is necessary even without a hidden degeneracy. On the unit torus take F(x,y) = sin(2 pi x)(2-cos(2 pi y)). Its only critical points have x in {1/4,3/4}, y in {0,1/2}; they are nondegenerate and have distinct values -3,-1,1,3. The points p=(3/4,0) and q=(1/4,0) are saddles with Hessians diag(a,-a) and diag(-a,a), where a=(2 pi)^2. The invariant line y=0 contains a p-to-q connection: in cover coordinates 3/4<x<5/4, dx/dt=2 pi cos(2 pi x)>0.

Use endpoint boxes centered at p and the lift (5/4,0) of q, with rational outer, inner and continuation halfwidths 1/20, 3/100 and 1/100. Let K be the union of their compact continuation complements and m=min_K |grad F|. Each outer box contains only its designated saddle, so m>0 and is attained. Set f=F/m and eta=1. Then min_K |grad f|=1: the non-strict predecessor accepts the exclusion bound, with equality at at least one endpoint. Choose a rational spectral margin below a/m and rational local sections, tube and time/speed/angle bounds with strict slack along the compact horizontal connection. Positive scaling f_epsilon=(1-epsilon)f preserves critical points, branch geometry and first-crossing locations; for sufficiently small epsilon>0 it also preserves those other strict margins. But min_K |grad f_epsilon|=1-epsilon<1, while ||f_epsilon-f||_{C^2}=epsilon||f||_{C^2} tends to zero. Thus the non-strict predicate is not open. The strict predicate excludes the equality field from this fixed chart; choosing a smaller rational eta preserves connection coverage.

The script separately checks the repaired membership decision with exact binary minima 0.5, 1.0 and 1.5 at eta=1, expecting rejection, rejection and acceptance. It substitutes only the minimum returned by the sampling routine, leaving the other tight-saddle checks active. This is a predicate-logic check, not a certified gradient minimum for the sampled field. The analytic scaling argument above supplies the boundary counterexample independently of sampling. Both controls remain author-side and confer no mathematical acceptance or status change.

**Geometric-boundary control (analytic).** The example is from the [OpenAI successor A1/A6 review](https://github.com/d6g8k5htny-coder/main/blob/e8ec7afcf3933d2fe2f6cb7306b8ff05ed518b81/reviews/sard_g_successor_a1_a6_20260926/REVIEW.md) (Git blob 9fc61b47c2b4c4bcc3898a6c61a6dcf9cbe1d136, 25,057 bytes), a source-exposed same-provider contribution with zero organizational-independence credit; the PR-thread review of 2026-09-26 (Anthropic / Claude) stated the same defect and the repair adopted here. Take again F(x,y) = sin(2 pi x)(2-cos(2 pi y)) with its p-to-q connection along y = 0, boxes and eta = 1/20 as in the equality-boundary control, an open tube about the horizontal segment, and the closed section Sigma = {(1,y) : 0 <= y <= 1/200}. The connection crosses the supporting line x = 1 transversely at (1,0), which is an endpoint of Sigma, with every critical-point, spectral-gap, gradient-exclusion, tube, speed, angle and travel-time margin strictly satisfied. The translates F_eps(x,y) = F(x,y+eps) converge to F in C^2 and preserve all of those margins, but their connection crosses the line x = 1 at (1,-eps), outside Sigma. Under the predecessor predicate of commit a1fc9581291b6f81c76c903c96a13955b51f8d54, which asked only for a unique transverse first crossing of Sigma, F is a member and F_eps is not, so that U_chi was not open. Under the present condition 3 the crossing point of F has distance 0 from an endpoint of Sigma, which exceeds no rho > 0, so F is not a member of any chart carrying this Sigma; the same connection is charted instead by a section whose relative interior contains (1,0), for instance Sigma' = {(1,y) : -1/200 <= y <= 1/200} with rho = 1/400 and a tau below the horizontal travel time over a distance 1/400, and then F and all translates with |eps| < 1/400 are members together. The same clause excludes a tangential contact with a closed section before the first crossing. This control is analytic; the script does not exercise it.
