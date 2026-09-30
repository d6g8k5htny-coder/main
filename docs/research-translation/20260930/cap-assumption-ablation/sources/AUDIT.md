# D1 external-audit response: the separating cap really gives a global elder pair

Object: OA-D1-CAP-AUDIT-20260930-v1. Author/executor: OpenAI / GPT-6 Astra Pro.
Requested by Dylan in the current conversation; coordination: Math- issue #193.
Source cut: `3e0a91b69892b9717564c592e7a966492a81566a`.

**Disposition:** source/author-exposed reconstruction and new explanatory corollaries;
new nonauthor review requested. Not an independent-human or institutional review.
Scientific effect NONE: no original proof, verdict, GRAPH, STATUS, prize, premise
or formal-scope change. The executor is exposed to the original OpenAI proof chain.

## 1. What this audit resolves, and what it does not

The external critique identifies the right load-bearing question: why does a local
Hessian event identify the GLOBAL ordinary superlevel H0 elder partner? Sections
2–7 reconstruct that implication without relying on an ACCEPT label. The crucial
information is stronger than endpoint jets: a closed neighborhood has no exit above
the saddle height, and a separately constructed path reaches an older point at
exactly that height. Remote excursions, loops and reentry cannot avoid the first exit.
No deterministic gap was found in CAP §§1–5 under their full hypotheses.

This is not a fresh audit of all Gaussian estimates or of the entire D1 theorem.
The matrix boundary layer, pinned genericity, marked Kac–Rice extension and source
coefficient identification remain their own analytic interfaces. `SOURCES.json`
binds the exact source set; an identity match never establishes its truth.

The critique also contains a statement that is correct for D1 alone but outdated
for the program: a separate, source-bound reviewed D2 theorem already supplies an
unrestricted **O(1) additive remainder**, not the stronger compact O(ell^(2/3))
difference. Section8 separates this update from genuine open validation and
numerical-constant requirements. Institutional external review and a complete Lean
formalization are not provided by this response and are not marked resolved.

## 2. The topological lemma, independent of Gaussian probability

Let X be a compact smooth manifold, f continuous, M in the interior of a compact
set C, b=f(M), and s<b. Assume:

(C1) f<=b throughout C;
(C2) f<=s on the entire boundary of C;
(C3) there is a continuous path from M to z with f(z)>b and f>=s everywhere on it.

Define the connection value

    d_f(M)=sup_{gamma(0)=M, f(gamma(1))>b} min_{t in[0,1]} f(gamma(t)).  (A1)

Then **d_f(M)=s**.

Proof of the upper bound: every older endpoint f(z)>b is outside C by C1. Since M
is interior, a continuous path to that endpoint has a first boundary exit. Its
value is <=s by C2, so the path minimum is <=s. Later excursions or reentry cannot
remove that already-encountered value. Taking the supremum proves d_f(M)<=s.
C3 proves the opposite inequality and also shows that the path family is nonempty.
This argument needs no smooth boundary, no absence of remote critical points,
no stochastic independence, and no identification of a ridge with a flow line.

For a Morse f with distinct critical values, this is precisely the ordinary
superlevel-component elder death level. One can see it directly from the rule:
for s<h<b the component of M in {f>h} cannot contain a point above b, since an
open connected manifold subset is path connected and that would contradict the
upper bound. Thus no older maximum has merged with M's class at those levels.
For every h<s, the path C3 is contained in {f>h} and reaches a point above b, so
the component now contains an older maximum. A component containing such a point
has a maximum of value greater than b: maximize f on its compact closure. Since
the maximum is above h, it lies in the component interior, not its level-h boundary.
Therefore death occurs at level s, not above or below it. The nonempty older path
excludes the essential global-maximum class. If a unique critical point S has
critical value s, that point is the elder death saddle.

The argument does NOT say that a path with minimum s by itself determines death:
it proves only d_f(M)>=s. Nor does an exit barrier alone guarantee finite death:
without an older endpoint the class can be essential. Both directions are necessary.

## 3. The exact local hypotheses in D1

Write local coordinates (x,y) in R x R^(d-1), with d>=2. For r>0 the full cylinder is

    D=[-2r,2r] x closed_ball(0,2r).

For the torus X=R^d/(LZ^d), impose `r<L/(4sqrt(2))`; then D lies inside the
injectivity ball and is an embedded chart. Fix k>0 and a C4 function near D with
critical pins M=(-r/2,0), S=(r/2,0), f(M)=b, f(S)=s=b-kr^3.
For j=3,4 use exactly the partial-block operator norms from CAP:

    M_j=max_{a+c=j} sup_D ||partial_x^a D_y^c f||_op,
    lambda=lambda_min(-D_y^2 f(M)).

The event used in P §7 is

    G_r: lambda > (4/(3k))r M_3^2,     rM_4 <= 3k/10.     (A2)

These are bounds on the WHOLE CYLINDER, not just the Hessians or jets at M and S.
Using arbitrary-coordinate entry maxima without the appropriate dimension factors
would not be the same event. The transverse Hessian inequality is a matrix-order
inequality, not an entrywise scalar assertion.

Normalize by f_tilde=f/(6k). Its gap is r^3/6. Put m=M_3/(6k), n=M_4/(6k), and
lambda_tilde=lambda/(6k). The hypotheses become

    lambda_tilde>8r m^2,          rn<=1/20.                (A3)

In Sections4–6 all derivatives/heights refer to f_tilde. Write b_tilde=b/(6k),
s_tilde=b_tilde-r^3/6, a=-r/2 and c=r/2.

## 4. Reconstructing the transverse ridge without a vector Rolle shortcut

The two critical pins yield the exact Hermite identity

 f_tilde(c,0)-f_tilde(a,0)
    =-1/2 integral_a^c (x-a)(c-x) f_tilde_xxx(x,0) dx.     (A4)

The positive kernel has integral r^3/6. Its weighted derivative average is therefore
2; in particular m>=2, and f_tilde_xxx=2 somewhere in the pin interval. Each point
of D is at product distance <5r from that point and from M. The block norms give

 f_tilde_xxx>=2-5rn>=7/4,
 D_y^2 f_tilde <= -delta I,
 delta=lambda_tilde-5rm>rm(8m-5).                         (A5)

Let K=8m-5>=11 and w(x)=grad_y f_tilde(x,0). Both endpoint values of w are zero.
Apply the two-node scalar interpolation remainder to each unit-vector projection:

 ||w(x)|| <= (m/2)|x^2-r^2/4| <=(15/8)mr^2<2mr^2.         (A6)

This also holds outside the pin interval: use the convex hull of x,a,c when
applying the scalar interpolation remainder. For its derivative, use the average:

 w'(x)=r^-1 integral_a^c [w'(x)-w'(t)]dt,
 ||w'(x)|| <=(m/r) integral_a^c |x-t|dt <=2mr.             (A7)

There is no assertion of one point where all components of w' vanish. Indeed the
exact vector w=(x^2-1/4, x^3-x/4) has both endpoint zeros but no common zero of w'.

On ||y||=2r, strong concavity gives

 grad_y f_tilde(x,y).y <= ||w(x)|| ||y||-delta||y||^2<0.

The maximum in each compact transverse ball is interior and unique. Call it h(x).
The implicit-function theorem gives h in C3, h(a)=h(c)=0 and

 ||h||<2r/K,
 ||h'||<(2/K)+(2/K^2)<=24/121,
 m||h'|| <=(1+6/K+5/K^2)/4<=48/121.                       (A8)

For example h'=-(D_y^2 f_tilde)^(-1) partial_x grad_y f_tilde; (A7) plus the
mixed derivative bound at h gives the second inequality. All estimates use vector
operator norms and have no hidden dimension-dependent coordinate sum.

## 5. The reduced derivative and EVERY boundary face

Let g(x)=f_tilde(x,h(x)) and F=g'. Put v=(1,h'). Differentiating the transverse
critical equation gives (H_f_tilde v)_y=0. In the third derivative of g, the
h'' term pairs this Hessian vector with a purely transverse vector and is zero;
the h''' term pairs grad_y f_tilde=0. Therefore exactly

 F''=f_xxx+3f_xxy[h']+3f_xyy[h',h']+f_yyy[h',h',h'].       (A9)

This does not make h a gradient-flow trajectory. The adverse part in (A9) is at
most (48/121)[3+3(24/121)+(24/121)^2]=2554128/1771561. Hence

 F''>7/4-2554128/1771561=2184415/7086244>1/4.              (A10)

Since F(a)=F(c)=0, strict convexity gives F<0 between the pins and F>0 outside.
It also gives F'(a)<0<F'(c). All critical points in D lie on h, so these are the
only two. The transverse block is negative; its Schur complement at a/c is
F'(a)/F'(c). Thus M is a maximum and S has exactly one positive direction.

Now take the CLOSED cap

    C=[-2r,r/2] x closed_ball(0,2r).

Strong transverse concavity yields f_tilde(x,y)<=g(x), and the sign of F gives
g<=b_tilde on this cap. This establishes C1, not just a bound on its boundary.

For the longitudinal faces subtract (x-a)(x-c)/8 from F. The difference is convex,
is zero at a,c, and is nonnegative outside [a,c]. Integrating on the two exterior
intervals yields

 g(-2r)<=b_tilde-9r^3/32=s_tilde-11r^3/96,
 g( 2r)>=s_tilde+9r^3/32=b_tilde+11r^3/96.                (A11)

For the lateral boundary ||y||=2r, (A5)–(A8) give delta>22r and
||y-h||>20r/11. Hence, for every x in the cap,

 f_tilde(x,y)<=g(x)-(delta/2)||y-h||^2
             <b_tilde-(4400/121)r^3<s_tilde.             (A12)

This covers the full spherical side, its intersections with both end faces,
and all corners in d=2. On the remaining cap face x=c, h(c)=0, so

 f_tilde(c,y)<=s_tilde-(delta/2)||y||^2,

with equality only at S. On x=-2r use (A11). Thus the ENTIRE cap boundary is
at most s_tilde, and only S attains s_tilde. This establishes C2.

Finally the ridge path x from a to2r starts at M, decreases to S, then increases
to z=(2r,h(2r)). Its minimum is exactly s_tilde and (A11) gives f_tilde(z)>b_tilde.
This establishes C3. A path through S at level s is not an above-s preemption path.

Multiplication by6k returns to the original function. The axial drop/rise is at
least (27/16)kr^3; the older-point excess is at least (11/16)kr^3; the lateral drop
is greater than (26400/121)kr^3. Heights b,s and all critical points retain their
order. By Section2 the global elder partner is S on the Morse/distinct-value
locus. This is the desired implication (A2), in all fixed d>=2.

The optional ascending-branch conclusion also follows: an ascent-unstable vector
at S cannot lie in the negative transverse hyperplane. One half-branch enters C
at height>s and cannot leave it. Its omega-limit lies among the finitely many
critical points in C; the strictly increasing gradient Lyapunov function excludes
S, so the limit is M. The other half-branch cannot cross the boundary at height>s
or approach the interior point M. The same reasoning works with a smooth positive
definite Riemannian metric; no Morse–Smale property is used. The branch claim is
not needed for the elder maximin proof.

## 6. Two useful audit consequences

### Exterior invariance

Let f satisfy the preceding cylinder hypotheses. Any other compact Morse extension
f_hat that agrees with f on a neighborhood of D and has distinct critical values
also pairs M with S, regardless of its values outside D. The proof uses exactly
the same cap C, boundary and older-reaching ridge segment. Thus modifying remote
maxima, creating remote saddles or providing an exterior high corridor cannot
preempt the pair while these protected data remain unchanged. This is a corollary
of the first-exit proof, not a probabilistic decorrelation assertion.

### Endpoint jets alone are insufficient: an exact local counterexample

For r=k=1, in two variables put

 q(x)=2x^3-(3/2)x-1/2,
 f0(x,y)=q(x)-128y^2,
 f1(x,y)=f0(x,y)+65536y^4+y^5.                            (A13)

Both pins M=(-1/2,0), S=(1/2,0) have the same heights0,-1 and identical jets through
order3. Both have the same max/saddle Hessian types. The unmodified f0 has M3=12,
M4=0 and lambda=256>192, so its local cylinder satisfies (A2).

For f1 the vertical path x=-1/2,0<=y<=1/16 has values

 65536(y^2-1/1024)^2-1/16+y^5 >=-1/16>-1,

and its endpoint has value 1/2+1/16^5>0. It connects M to an older point above the
candidate saddle level. Therefore S cannot be its elder partner in any extension
preserving this path. This is an exact continuous path inequality, not an endpoint
sampling argument. The example does not contradict CAP: f1_yyyy(M)=24*65536, so it
violates the required whole-neighborhood fourth-derivative bound severely. It
shows why local pictures, pins or third jets alone cannot replace (A2).

## 7. How the deterministic audit plugs into D1, and the corrected reading set

P §7 uses exactly (A2). P §8 establishes Morse/distinct-value genericity for each
fixed conditional law and transfers it to Q^W by absolute continuity; no common
null set for all uncountably many parameters is asserted. The cap proof therefore
implies pointwise on that full-measure locus

 1{S not the ordinary elder partner of M} <= 1{G_r fails}.

Its expectation under the ORIGINAL full weight is bounded by Q^W(G_r^c). The
separate matrix calculation in P §§3–7 gives E_Q[W 1{G_r^c}]=O(r^5), while the
full Z_r is of order r^2; that supplies the O(r^3) probability bound. Today's
deterministic proof does not independently re-establish those stochastic inputs.
No regional D5 witness-collision mechanism is used to prove this implication.

The citable parent is NOT its immutable uncorrected text in isolation:

- P §5 must use E1: `D_r=diag(r^(-1/2),I)`. For H=[[r alpha,r beta^t],[r beta,A]],
  `D_r H D_r=[[alpha,sqrt(r) beta^t],[sqrt(r) beta,A]]` and det(D_r H D_r)=det(H)/r.
  The old diag(sqrt(r),I) would give r^2 alpha, not alpha.
- REC §5 W1 supplies the correct joint-convergence reading for the type argument;
  it does not pretend arbitrary conditioned fields live in an unspecified common
  almost-sure coupling.
- P §9 is replaced by E2's Borel marked Kac–Rice proof; an elder indicator is not
  assumed continuous. This is a second essential amendment, not just typography.
- REC supplies the explicit embedding radius r<L/(4sqrt(2)).

The exact five-part P/CAP/E1/E2/REC identity set is mandatory for this audit's
connection to D1. No original file is edited to conceal its correction history.

## 8. Resolution of the remaining critique items

| Audit concern | Resolution at the inspected source cut |
|---|---|
| Global topology might not follow from local geometry | The full cap implication is reconstructed in §§2–5; both a global upper barrier and an older-reaching lower witness are proved. No uncovered deterministic gap found under the exact hypotheses. New reconstruction still requests nonauthor review. |
| The proof was corrected | Correct. Retain E1, E2 and REC/W1 as part of the mathematical object; tests explicitly reject the wrong congruence. A correction is not evidence of infallibility, but the corrected statement is the one to assess. |
| No unrestricted quantitative remainder | True for D1 Theorem C by itself; outdated as a statement about the program. D2 Theorem R gives `nu_cand=c ell^(-1/3)+O(1)`, `0<=nu_cand-nu_eld<=O(1)`, and `nu_eld=c ell^(-1/3)+O(1)`. The existing full-depth Claude review records R1–R6 ACCEPT in addition to the earlier xAI record. This is relative error O(ell^(1/3)), not the compact O(ell^(2/3)) difference. C and ell_* remain existential. |
| No convergence/second coefficient of that remainder | Not supplied by Theorem R. PR191 proposes a stronger remainder statement; it is separate active candidate work, not adopted or re-proved in this audit. |
| No number for arbitrary c_(d,L), C or r_* | An explicit positive finite integral can specify a theorem constant without a numerical value. Evaluated constants and usable finite-radius error bounds are additional quantitative tasks, not logical prerequisites for the existential statement. This audit does not produce them. |
| SIDE24 enclosures may overstate evidence | Keep their precise meaning: arithmetic intervals for P(15.2), with a proved covariance-perturbation argument in their own source. They do not bound theorem correctness or finite-lifetime asymptotic error. Their identification with persistence still depends on the reconciled D1 chain. The full interval program is not re-executed in this audit. |
| The d=3 Hessian-cone constraint | Correctly retained in SIDE24: for A=[[s+x,y],[y,s-x]], Var(s)=5/3 and Var(x)=Var(y)=1, integrating the NEGATIVE-DEFINITE cone gives 29/6-sqrt(6), not half the untruncated moment29/6. This is a different conditional matrix from a GOE substitution. |
| D5/witness-collision remains open | D1's dependency route is P/CAP/E1/E2/REC, not D5. The merged C6 residual reconciliation separately supplies leading pair-mass localization and is explicitly declarative (`executed:false`). It does not authorize relabeling every historical shrinking-region mechanism or changing the live node. Open status and resolved analytic subclaims must not be conflated. |
| Reviews are not institutionally independent | Correct and unresolved as an external-validation request. Shared account, actual provider, session, source exposure and author role remain disclosed. Delegated AI review is not a human mathematician's review. Today's executor does not add an organizationally independent vote. |
| Full formal proof is absent | Correct. `formal/SCOPE.md` covers13 scalar companions, explicitly not D1 elder selection, Kac–Rice or SIDE24 coefficient verification. No local Lean toolchain was available and no D1 kernel proof was produced. Green hosted formal checks retain their exact narrow scope. |
| Tests/hashes are not proof | Correct. The continuum proof is written above. The executable checks are finite rational identities, a continuous polynomial falsifier and finite graph analogues; they are neither Gaussian simulation nor Lean verification of that continuum proof. |

The two cited SIDE24 intervals remain

 0.07340691930603427103 < c_(2,24) < 0.07340691930603427104,
 0.04177593184059834334 < c_(3,24) < 0.04177593184059834335.

They are repeated as the exact scoped source claim, not as a newly executed
certificate. In particular a 20-decimal coefficient enclosure does NOT mean
20-decimal accuracy of nu(ell) at a chosen positive ell.

## 9. Verification and a concrete external-referee route

`cap_audit.py` and `test_cap_audit.py` recompute the rational margins, pin scaling,
vector average counterexample, the vector ridge identity in polynomial examples,
the corrected congruence, and every derivative through order3 of (A13) at both pins.
Two independent graph algorithms agree over1,728 exterior modifications behind
a fixed separator. Three negative examples show that a path alone, a barrier alone,
or an equal-birth endpoint does not prove the required global pairing.
These are checks of their stated finite content only. The original source proof
and review dispositions are not changed by their output.

The appropriate next external referee can focus on five concrete obligations,
without navigating the full workspace: verify (A5)–(A10) for vector transverse
coordinates; check every boundary face in (A11)–(A12); verify the first-exit and
ordinary H0 argument in §2; verify P's full-weight/genericity interface in §7;
and independently inspect the Gaussian/Kac–Rice inputs not audited anew here.
An ACCEPT or AMEND should identify exact equations, assumptions and file hashes.
A request is not a delivered review. No journal submission, human acceptance or
external-person contact is claimed by this packet.

This completes a written answer to the critique and a deterministic cap audit,
not all possible future validation. Historical failures and residual open claims
remain visible. The new audit has scientific effect NONE and is not self-merged.
