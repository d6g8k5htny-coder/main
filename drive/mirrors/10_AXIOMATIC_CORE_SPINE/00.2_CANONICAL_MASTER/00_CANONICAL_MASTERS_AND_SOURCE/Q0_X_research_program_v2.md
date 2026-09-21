# Q0-X v2: Infinite-Volume Pairing Theory and Verified-Reasoning Architecture

**Status:** research derivation, July 14, 2026  
**Source discipline:** the live q0 Rate Program is Master v3.2. Statements below are labeled as
Established, Derived, Conditional Theorem, Hypothesis, or Open Obligation.

## Revision-v2 provenance corrections

1. The UB-G exterior enhancement is denoted here by
   \(E_{\mathrm{sup6}}^{\mathrm{UB-G}}=1.014\). The archive's bare
   \(E_{\mathrm{sup}}^{\mathrm{TC}}=1.0235\) belongs to terminal counting and is a
   different constant paired with the maximum intensity. The v1 number was mechanism-correct
   but its symbol collided with the terminal-counting notation.
2. The current canonical fold normalization is \(\ell=r^3/6\). Formulas containing
   \(\ell=\kappa r^3/6\) are explicitly marked generalizations with a fold-curvature mark
   \(\kappa\); they are not quotations of the frozen theorem.
3. STAB-UB now uses a **selected random-set Campbell bound**. An unconditional one-point
   saddle-intensity bound cannot be inserted into a random influence region that depends on the
   same field, while a bound for every arbitrary field-dependent selector would be unrealistically
   strong.

---

## 1. Starting point

The live calibrated result is

\[
0.8411\,r^3\le 1-q(r,1.2)\le 4.3\,r^3,\qquad 0<r\le 0.025,
\]

with exact/derived structure and station-certified constants. The current upper proof counts all
exterior window saddles and therefore contains an area term proportional to \(L^2\). It deliberately
does not use adjacency.

The SARD-G mechanism has four abstract parts:

1. a charted mismatch functional \(D\);
2. an \(H\)-directional derivative represented by an RKHS element \(G\);
3. injectivity/support separation proving \(G\ne0\);
4. finite-dimensional disintegration over a countable family of witness directions.

The framework below extends these two structures.

---

# Part I. Infinite-volume and renormalization

## 2. Two competing large-volume mechanisms

Let \(q_L(r,b)\) denote the pairing probability on a torus of side \(L\).

### Mechanism A: global-window hazard

If every window saddle in the whole torus is treated as an interceptor, then the raw exterior mean is

\[
\mathbb E N_{\mathrm{ext}}
\approx
\frac{\rho_{\mathrm{sad}}(b)E_{\mathrm{sup6}}^{\mathrm{UB-G}}}{6}
L^2r^3.
\]

At \(b=1.2\), using the UB-G window-saddle values,

\[
\alpha_{\mathrm{raw}}
=
\frac{0.030449\cdot1.014}{6}
\approx 0.00514588.
\]

After the current typing-floor division \(0.80\),

\[
\alpha_{\mathrm{typed}}\approx0.00643235.
\]

The corresponding finite-size variable is

\[
g=\alpha L^2r^3.
\]

If relevant interceptors converge to a Poisson process, then a candidate crossover law is

\[
q_L(r,b)\approx q_{\mathrm{loc}}(r,b)e^{-g}.
\]

The invariant scaling is

\[
L\mapsto sL,\qquad r\mapsto s^{-2/3}r,
\]

or equivalently \(Lr^{3/2}\) fixed.

This is a **conditional finite-size ansatz**, not a theorem.

### Mechanism B: adjacency stabilization

The current \(L^2\) factor may be a proof artifact. A window saddle can preempt the candidate only
when it lies in the relevant merge-tree/influence cluster. Let \(\mathcal I_r(M)\) be the smallest
region whose critical data determine the death partner of the pinned maximum \(M\).

If

\[
\sup_{0<r<r_0}\mathbb E^{\mathrm{Palm}}|\mathcal I_r(M)|<\infty,
\]

then the exterior contribution is \(O(r^3)\) uniformly in \(L\).

This is the preferred route at fixed positive level \(b=1.2\).

---

## 3. Conditional theorem: stabilized UB-G

### Theorem STAB-UB

Assume, uniformly for \(0<r<r_0\):

1. **Adjacency localization:** every exterior preemption event implies the existence of a
   window saddle in a random influence region \(\mathcal I_r(M)\).
2. **Finite susceptibility:**
   \[
   \mathbb E^{\mathrm{Palm}}|\mathcal I_r(M)|\le A_b<\infty.
   \]
3. **Selected random-set Campbell bound:** let \(\Xi^{\mathrm{win}}_f\) be the point measure
   of height-window saddles. For the actual certified influence selector
   \(h_r(f,x)=\mathbf 1\{x\in\mathcal I_r(f)\}\),
   \[
   \mathbb E^{\mathrm{pair}}_{r,b}
   \left[\int h_r(f,x)\,\Xi^{\mathrm{win}}_f(dx)\right]
   \le
   \bar\rho_b\,\ell\,
   \mathbb E^{\mathrm{pair}}_{r,b}
   \left[\int h_r(f,x)\,dx\right],
   \qquad \ell=\kappa r^3/6.
   \]
   A sufficient route is to construct a sigma-field making \(\mathcal I_r\) measurable and
   certify a conditional compensator bound for window saddles. No assertion is made for arbitrary
   selectors that could localize directly on hazard atoms.
4. The current near, collar, and direct-gain terms retain their \(O(r^3)\) bounds.

Then

\[
1-q_L(r,b)
\le
\left(
C_{\mathrm{near}}
+
\frac{\bar\rho_b A_b\kappa}{6}
+
C_{\mathrm{collar}}
+
C_{\Gamma}
\right)r^3
\]

with a constant independent of \(L\). Consequently \(q_\infty(r,b)\) exists along any
finite-volume exhaustion for which the influence regions couple consistently, and

\[
1-q_\infty(r,b)=O(r^3).
\]

### Proof

Let \(N_{\mathrm{win}}(\mathcal I_r)\) count window saddles in the influence region.
Adjacency localization and Markov give

\[
\mathbb P(\Pi)\le
\mathbb P(N_{\mathrm{win}}(\mathcal I_r)\ge1)
\le
\mathbb E N_{\mathrm{win}}(\mathcal I_r).
\]

Apply joint Campbell domination with
\(h(f,x)=\mathbf 1\{x\in\mathcal I_r(f)\}\). Then

\[
\mathbb E N_{\mathrm{win}}(\mathcal I_r)
\le
\bar\rho_b\ell\,\mathbb E|\mathcal I_r|
\le
\frac{\bar\rho_b A_b\kappa}{6}r^3.
\]

Add the existing local terms. ∎

### Two separate live obligations

1. **PALM-SUSC:** prove the finite-susceptibility bound under the pair-Palm law.
2. **PALM-CAMP:** prove the selected random-set Campbell inequality—or a sufficient conditional
   compensator theorem—for the influence selector. Uniform deterministic-set Kac--Rice intensity
   alone does not discharge this.

For the planar Bargmann--Fock field, positive superlevel sets are subcritical because the critical
level is zero, and crossing probabilities above zero decay exponentially. At \(b=1.2\) and
\(\ell\ll1\), the relevant level remains uniformly positive. The missing bridge is to show that
finite local pinning preserves an exponential cluster-radius tail uniformly as \(r\downarrow0\).

A concrete route is:

1. express the conditioned field as a deterministic mean plus a finite-codimensional Gaussian
   residual;
2. use the GCJA/divided-difference frame to obtain \(r\)-uniform control of the conditioning
   functions;
3. prove that the mean and covariance perturbations decay Gaussianly with distance;
4. couple outside a fixed neighborhood to an unconditioned BF field with a small positive
   sprinkling of the level;
5. import exponential subcritical crossing decay;
6. show that the merge-tree influence region is contained in a bounded enlargement of the
   relevant positive superlevel cluster.

---

## 4. The true renormalization frontier is the critical-height limit

At fixed \(b>0\), exponential cluster decay predicts a finite influence susceptibility. Genuine
critical scaling should appear as \(b\downarrow0\), where the BF excursion set approaches its
percolation threshold.

Define

\[
\chi(b)=\mathbb E|\mathcal C_b(0)|,
\]

for the relevant positive excursion cluster, with an appropriate Palm version in the actual proof.
A natural effective coupling is

\[
g_{\mathrm{eff}}(r,b)=r^3\chi(b).
\]

The conjectural scaling form is

\[
1-q_\infty(r,b)
=
r^3 C_{\mathrm{loc}}(b)
+
\Psi(r^3\chi(b))
+
o(r^3).
\]

At fixed \(b=1.2\), \(g_{\mathrm{eff}}\to0\). Near criticality, \(\chi(b)\) may diverge and produce a
nontrivial crossover. Exact critical exponents should not be assumed without proof.

For LLMs, \(\chi\) becomes the expected size of the minimal dependency/evidence cone required to
validate one claim. Modular certificates aim to keep this susceptibility bounded as context grows.

---

## 5. New corollary: density of pairing defects

Suppose, uniformly over a compact mark set,

\[
\nu_{\mathrm{cand}}(\ell,\kappa)
=
A(\kappa)\ell^{-1/3}(1+o(1))
\]

and

\[
1-q(r,\kappa)
=
C(\kappa)r^3(1+o(1)).
\]

Since \(r^3=6\ell/\kappa\),

\[
\nu_{\mathrm{def}}(\ell,\kappa)
=
\nu_{\mathrm{cand}}(\ell,\kappa)(1-q)
=
\frac{6A(\kappa)C(\kappa)}{\kappa}
\ell^{2/3}(1+o(1)).
\]

Thus the density of incorrectly paired near-diagonal candidates **vanishes**, even though the total
bar density diverges like \(\ell^{-1/3}\).

The cumulative defect count obeys

\[
N_{\mathrm{def}}(0,\varepsilon)
\sim
\frac{18}{5}
\int\frac{A(\kappa)C(\kappa)}{\kappa}\,d\kappa
\;\varepsilon^{5/3}.
\]

Meanwhile total near-diagonal bars scale as \(\varepsilon^{2/3}\), so the defective fraction is
linear in \(\varepsilon\).

In area \(L^2\):

- the smallest ordinary near-diagonal bar is predicted at scale \(\ell\asymp L^{-3}\);
- the first local pairing defect is predicted at scale \(\ell\asymp L^{-6/5}\);
- a naive global-interceptor crossover would occur at \(\ell\asymp L^{-2}\).

Whether the \(L^{-2}\) scale survives after adjacency localization is an immediate simulation test.

---

# Part II. Generalized quantitative SARD theory

## 6. Abstract null-set theorem

### Theorem Q-SARD-0

Let \(E\) be a separable Banach space with Borel probability measure \(\mu\), and let
\(H\subset E\) be a separable linear space of admissible directions. Let
\(\{D_n:U_n\to\mathbb R^{k_n}\}_{n\ge1}\) be a countable chart family.

Assume:

1. each \(D_n\) is measurable and continuously differentiable along \(H\);
2. at every \(x\in U_n\) with \(D_n(x)=0\),
   \[
   d_HD_n(x):H\to\mathbb R^{k_n}
   \]
   is surjective;
3. for every finite-dimensional witness space \(V\subset H\) generated by a fixed countable dense
   subset, \(\mu\) admits a disintegration along affine \(V\)-fibers whose conditional measures are
   absolutely continuous with respect to Lebesgue measure on \(V\).

Then

\[
\mu\left(\bigcup_n D_n^{-1}(0)\right)=0.
\]

### Proof sketch

At each zero, surjectivity is witnessed by a \(k_n\)-tuple from the countable dense family.
Continuity gives a countable cover on which the corresponding determinant stays nonzero. On every
affine witness fiber, the implicit-function theorem makes the zero set a codimension-\(k_n\)
submanifold, hence Lebesgue-null. Fiberwise absolute continuity and Fubini give zero conditional
probability. Countable union completes the proof.

Gaussian Cameron--Martin disintegration is one sufficient instance. Non-Gaussian measures are
covered whenever the finite-dimensional conditional absolute-continuity assumption is verified,
including many measures absolutely continuous with respect to a suitable Gaussian reference.

---

## 7. Quantitative tube theorem

Almost-sure avoidance is insufficient for hallucination control. We need rates.

### Theorem Q-SARD-\(\varepsilon\)

Fix a \(k\)-dimensional witness space \(V\). On each relevant fiber assume:

1. the conditional density is bounded by \(M\);
2. on the false-acceptance region \(\|D\|\le\varepsilon\),
   \[
   |\det D_VD|\ge \sigma^k;
   \]
3. every output value has at most \(N\) preimages in the chart.

Then

\[
\mathbb P(\|D\|\le\varepsilon)
\le
N M\,\mathrm{Vol}(B_k)
\left(\frac{\varepsilon}{\sigma}\right)^k.
\]

This follows from the area formula, fiberwise, followed by disintegration.

The dependence

\[
\left(\frac{\varepsilon}{\sigma}\right)^k
\]

is exponential in the number \(k\) of transverse mismatch channels. It does not require statistical
independence; it requires a certified rank-\(k\) Jacobian and density/multiplicity control.

For \(N M\mathrm{Vol}(B_k)\le C\), total graph size \(m\), and target failure probability \(\delta\),
it is enough to choose

\[
k\ge
\frac{\log(Cm/\delta)}
{\log(\sigma/\varepsilon)}.
\]

Thus logarithmically many well-conditioned witnesses can suppress the union-bound failure
probability across a large reasoning graph.

---

## 8. Discrete-token boundary term

Token sequences live in a discrete space, so a Sard theorem does not apply directly to final token
strings. Apply it to a continuous latent/noise representation \(Z\), and let \(\pi(Z)\) be the
discrete decoder.

If the selected-token logit margin is at least \(m\) and the logit map is \(L_\pi\)-Lipschitz, then
\(\pi\) is locally constant on a ball of radius \(m/(2L_\pi)\). A complete bound therefore has the
form

\[
\mathbb P(\text{false accept})
\le
\mathbb P(\text{small decoding margin})
+
C\left(\frac{\varepsilon}{\sigma}\right)^k
+
\delta_{\mathrm{approx}}
+
\delta_{\mathrm{cal}}.
\]

The small-margin term is indispensable. Deterministic greedy decoding without a continuous
perturbation model does not receive a measure-theoretic Sard guarantee.

---


# Part II-B. Uniform conditioned-law stability

## 9. The common frontier behind STAB-UB and Q-SARD

The geometric and language-model problems share a single mathematical bottleneck: a useful local
estimate must remain uniform after conditioning on the very configuration that makes failure
plausible.

Let \(\{\mu_\theta\}_{\theta\in\Theta}\) be a family of conditioned laws. A bad accepted event
\(B_{\theta,\varepsilon}\) is assumed to satisfy

\[
B_{\theta,\varepsilon}
\subset
B_{\theta}^{\mathrm{haz}}
\cup
B_{\theta,\varepsilon}^{\mathrm{tube}}
\cup
R_\theta,
\qquad
\mu_\theta(R_\theta)\le\delta_\theta.
\]

### Hazard channel

Let \(\Xi_\theta\) be a random hazard measure and let
\(h_\theta(\omega,x)\in[0,1]\) select the random influence region. Assume:

\[
B_\theta^{\mathrm{haz}}
\subset
\left\{\int h_\theta\,d\Xi_\theta\ge1\right\},
\]

and a Campbell inequality for every selector in a declared certified class

\[
\mathbb E_{\mu_\theta}
\left[\int h\,d\Xi_\theta\right]
\le
\lambda_\theta
\mathbb E_{\mu_\theta}
\left[\int h\,dx\right]
\]

for every selector in the certified class. The class must be fixed independently of the realized
hazard atoms and must contain the actual influence selector. Define the geometric susceptibility

\[
\chi_\theta
=
\mathbb E_{\mu_\theta}\int h_\theta\,dx.
\]

Then

\[
\mu_\theta(B_\theta^{\mathrm{haz}})
\le
\lambda_\theta\chi_\theta.
\]

### Transversality channel

Let \(D_\theta\) be a \(k\)-dimensional mismatch map on a finite-dimensional witness fiber.
Suppose its conditional density is bounded by \(M_\theta\), its chart multiplicity by \(N_\theta\),
and its witness Jacobian has singular-value floors

\[
s_{\theta,1},\ldots,s_{\theta,k}>0
\]

throughout the false-acceptance tube. The area formula gives

\[
\mu_\theta(B_{\theta,\varepsilon}^{\mathrm{tube}})
\le
N_\theta M_\theta\operatorname{Vol}(B_k)
\frac{\varepsilon^k}
{\prod_{i=1}^k s_{\theta,i}}.
\]

Define the transversality susceptibility

\[
\tau_\theta
=
\left(\prod_{i=1}^k s_{\theta,i}\right)^{-1}.
\]

### Uniform conditioned-law bound

Combining the two channels,

\[
\boxed{
\mu_\theta(B_{\theta,\varepsilon})
\le
\lambda_\theta\chi_\theta
+
N_\theta M_\theta\operatorname{Vol}(B_k)
\,\tau_\theta\,\varepsilon^k
+
\delta_\theta.
}
\]

This bound is elementary once its assumptions are certified. The difficult frontier is uniformity:

\[
\sup_{\theta\in\Theta}\chi_\theta<\infty,
\qquad
\sup_{\theta\in\Theta}\tau_\theta<\infty,
\]

together with uniform Campbell, density, multiplicity, and coverage bounds.

The two principal failure modes are therefore dual:

\[
\boxed{
\text{geometric criticality: }\chi_\theta\to\infty,
\qquad
\text{transversality criticality: }\sigma_{\min}(dD_\theta)\to0.
}
\]

For the Bargmann--Fock program, the expected critical direction is \(b\downarrow0\), where the
influence region may delocalize. For a verifier, the dangerous regime is the accepted
locally-consistent error set, where mismatch directions may become nearly linearly dependent.

This is the shared theorem-level object that the analogy alone does not expose.

## 10. Effective-rank degradation

The isotropic shorthand

\[
C(\varepsilon/\sigma)^k
\]

is safe only when all \(k\) singular values have a common positive floor. The anisotropic bound is

\[
C\frac{\varepsilon^k}{s_1\cdots s_k}.
\]

If one singular value falls below the tube scale, that direction no longer contributes a robust
power of \(\varepsilon\). A practical diagnostic is

\[
k_{\mathrm{eff}}(\varepsilon)
=
\#\{i:s_i\ge c\varepsilon\},
\]

for a declared safety factor \(c\). This is a diagnostic rather than a replacement theorem, but it
correctly predicts rank-collapse controls.

A Monte Carlo smoke test performed for this revision gave log--log slopes

\[
0.998,\quad2.021,\quad2.975
\]

for full-rank dimensions \(k=1,2,3\), and slope \(1.960\) for a nominally three-dimensional map with
one singular value \(10^{-3}\) over a tube ladder larger than that scale. These tests validate the
expected exponent mechanism; they do not certify any real model's witness Jacobian.

---

# Part III. Persistence on reasoning graphs

## 11. Reasoning persistence complex

Represent an answer by atomic claims \(V\), evidence nodes \(E\), and directed support edges
\(u\to v\) with strength \(s_{uv}\in[0,1]\).

For threshold \(\tau\), retain edges with \(s_{uv}\ge\tau\). The grounding death level of claim \(v\)
is

\[
d(v)=\max_{v\leadsto E}\min_{e\in\text{path}}s_e.
\]

This is the maximum-bottleneck or widest-path value. If \(b(v)\) is the model's claim confidence,
define grounding persistence

\[
\ell(v)=\bigl(b(v)-d(v)\bigr)_+.
\]

A high-confidence claim with weak evidence has large \(\ell(v)\).

### Exact compression theorem

A maximum-bottleneck predecessor forest rooted at the evidence nodes preserves every \(d(v)\).
Therefore all \(H_0\) grounding thresholds can be represented with at most

\[
|V|-|E|
\]

support edges, regardless of the original graph density.

This is an exact token/computation reduction for the \(H_0\) layer.

### Pairing defect

Let \(g(v)\) be the local/geometric partner selected by attention, similarity, or nearest evidence.
Let \(p(v)\) be the persistence partner in the maximum-bottleneck forest. Define

\[
\Delta_{\mathrm{pair}}(v)=\mathbf 1\{g(v)\ne p(v)\}.
\]

The empirical reasoning analogue of the q0 law is

\[
q_{\mathrm{reason}}(\ell)
=
\mathbb P(g(v)=p(v)\mid \ell(v)\approx\ell).
\]

A cubic-fold transfer hypothesis predicts \(1-q_{\mathrm{reason}}(\ell)=O(\ell)\), but the exponent
must be measured rather than assumed.

Directed cycles/SCCs detect circular support. Clique-complex \(H_1\) can be added when higher-order
redundancy matters, but it is not needed for the exact \(H_0\) grounding certificate.

---

# Part IV. Kernel mismatch detection

## 12. Signed evidence residual

Let \(z_i\) be latent embeddings for claims/evidence and let \(a_i\) be signed residual weights
(claim mass minus verified incoming support). Define

\[
T=\sum_i a_i\delta_{z_i}.
\]

For a kernel \(K\), the mismatch score is

\[
\mathcal M_K^2(T)
=
\left\|\int K(\cdot,z)\,dT(z)\right\|_{\mathcal H_K}^2
=
\sum_{i,j}a_ia_jK(z_i,z_j).
\]

A universal kernel is injective on finite signed measures, so \(\mathcal M_K(T)=0\) only when the
signed residual vanishes.

## 13. Anchored trainable-kernel theorem

Let \(K_0\) be universal and let \(K_\theta\) be any positive-semidefinite learned kernel. For
\(\lambda>0\), define

\[
K_{\mathrm{safe}}=\lambda K_0+K_\theta.
\]

Then for every nonzero signed measure \(T\),

\[
\|T\|_{K_{\mathrm{safe}}}^2
=
\lambda\|T\|_{K_0}^2+\|T\|_{K_\theta}^2
>0.
\]

Thus training cannot destroy injectivity as long as the fixed universal anchor retains positive
weight. A multiscale Gaussian-RBF mixture is a practical anchor.

Random Fourier features provide a streaming approximation. If pairwise kernel error is at most
\(\eta\), then for signed weights \(a\),

\[
|\widehat{\mathcal M}^2-\mathcal M^2|
\le
\eta\|a\|_1^2.
\]

This error belongs explicitly in the false-acceptance ledger.

---

# Part V. Scalable proof-carrying generation

## 14. Proof object

Every externally visible claim should carry:

- claim ID and atomic statement;
- dependency IDs;
- evidence pointers;
- epistemic grade;
- verifier and tolerance;
- version;
- Merkle hash;
- supersession status.

Exact structural gates are:

1. dependency closure;
2. acyclicity;
3. no reachable superseded node;
4. evidence closure;
5. grade/claim-language compatibility;
6. hash integrity.

These are linear-time graph checks.

## 15. Module certificates

For a verified subgraph \(S\), store only:

- boundary assumptions;
- boundary conclusions;
- verifier result;
- Merkle root;
- risk budget.

As long as the root hash is unchanged, the interior need not be reinserted into context. A change
invalidates exactly the reverse-reachable descendants. This is mathematical incremental compilation.

For grounding graphs, retain the maximum-bottleneck forest plus a selected set of persistent cycle
generators. For proof DAGs, retain module boundaries and content hashes.

---

# Part VI. Measure-theoretic output control

## 16. Sequence measure

An autoregressive model defines a probability measure on finite token sequences through its
conditional kernels. Let \(H\) denote the event that at least one accepted atomic claim is
hallucinated, and let \(A_\lambda\) be the verifier acceptance event at threshold \(\lambda\).

The operational goal is not a bare confidence score but a calibrated bound on a declared loss, for
example:

- fraction of unsupported claims;
- probability of any unsupported load-bearing claim;
- graph distance from a valid proof;
- topological grounding loss.

A held-out calibration set can choose \(\lambda\) using conformal risk control for a monotone loss
under exchangeability. This statistical layer does not replace the structural or Q-SARD layers.

A complete ledger is

\[
R_{\mathrm{total}}
\le
R_{\mathrm{evidence}}
+
R_{\mathrm{Q\text{-}SARD}}
+
R_{\mathrm{kernel\ approximation}}
+
R_{\mathrm{decoder\ boundary}}
+
R_{\mathrm{calibration}}
+
R_{\mathrm{distribution\ shift}}.
\]

---

# Part VII. Integrated conditional guarantee

## 17. Verified-reasoning rate theorem

Consider \(m\) reasoning modules. Assume:

1. exact structural graph gates pass;
2. every wrong accepted module induces a \(k\)-dimensional mismatch map satisfying
   Q-SARD-\(\varepsilon\) with common ratio \(\rho=\varepsilon/\sigma<1\);
3. the total chart/density/multiplicity prefactor per module is at most \(C\);
4. kernel approximation, decoding-boundary, evidence, and calibration errors sum to
   \(\delta_0\).

Then

\[
\mathbb P(\text{any wrong module accepted})
\le
mC\rho^k+\delta_0.
\]

Choosing

\[
k\ge
\left\lceil
\frac{\log(mC/(\delta-\delta_0))}
{\log(1/\rho)}
\right\rceil
\]

gives total failure probability at most \(\delta\).

This is the precise sense in which the framework can obtain exponential error suppression while
using only logarithmically many transverse, well-conditioned mismatch coordinates.

It is a conditional theorem. The hard empirical/mathematical work is to certify the coverage,
Jacobian, density, and approximation assumptions on real model latent paths.

---

# Part VIII. Immediate experiments and falsifiers


## 18A. Stage-0 periodized-BF susceptibility probe

Before implementing the full pair-Palm \(q_L\) experiment, a cheaper prerequisite test was run on
the exact discrete periodized-BF spectrum.

Parameters:

\[
L\in\{12,24,48\},
\qquad
b\in\{1.2,0.8,0.4,0.2\},
\qquad
\Delta x=0.25,
\]

with 300 samples per cell. Each field was conditioned only on

\[
f(0)=b+0.25,
\]

and the area of the four-neighbor periodic excursion cluster
\(\{f\ge b\}\) containing the pin was measured.

This is **not** a pair-Palm experiment: it has no gradient constraints, Hessian typing,
determinant Palm weights, nearby saddle pin, or direct estimate of \(q_L\). It probes only whether
the most basic conditioned excursion susceptibility appears to stabilize with volume.

Observed mean cluster areas at \(L=12,24,48\) were:

\[
\begin{array}{c|ccc}
b & L=12 & L=24 & L=48\\
\hline
1.2 & 6.934 & 6.742 & 7.404\\
0.8 & 13.566 & 14.928 & 16.290\\
0.4 & 33.814 & 61.973 & 67.448\\
0.2 & 48.307 & 130.194 & 230.328
\end{array}
\]

Thus the \(L=48\) to \(L=24\) mean-area ratios were approximately

\[
1.10,\quad1.09,\quad1.09,\quad1.77.
\]

The coarse evidence supports stabilization at \(b=1.2\) and \(0.8\), probable but slower
stabilization at \(b=0.4\), and unresolved growth at \(b=0.2\). This is consistent with relocating
the scaling frontier toward the critical-height regime. It is only a precondition probe and cannot
establish STAB-UB, PALM-SUSC, PALM-CAMP, or the behavior of \(q_L\).

The next simulation layer must condition on the maximum--saddle jet constraints, apply determinant
Palm weights, and compute the actual persistence partner. Directly resolving the theorem scale
\(r\le0.025\) on a uniform grid is impractical; the serious implementation should combine spectral
conditional sampling with adaptive critical-point localization and merge-tree construction.

## 18. BF infinite-volume experiment

Generate periodized BF fields for

\[
L\in\{24,48,96,192\},\qquad
b\in\{1.2,0.8,0.4,0.2\},
\]

and multiple \(r\)-rungs.

Measure:

1. \(q_L(r,b)\);
2. total exterior window saddles;
3. adjacency-qualified window saddles;
4. area/radius of the merge-tree influence cluster;
5. finite-volume boundary-touch probability.

Competing predictions:

- **global hazard:** \(-\log q_L\) collapses against \(L^2r^3\);
- **stabilization:** \((1-q_L)/r^3\) converges to an \(L\)-independent constant for fixed \(b>0\);
- **critical crossover:** stabilization degrades only as \(b\downarrow0\).

A failure of uniform Palm-cluster tails kills STAB-UB in its present form.

## 19. Q-SARD exponent experiment

For synthetic latent-path mismatch maps of rank \(k=1,\ldots,8\), estimate

\[
\mathbb P(\|D\|\le\varepsilon)
\]

over an \(\varepsilon\)-ladder. The predicted log-log slope is \(k\). Repeat under:

- Gaussian latent noise;
- Gaussian-density tilts \(d\mu\propto e^{-V}d\gamma\);
- non-Gaussian quasi-invariant path measures;
- rank-deficient controls.

Failure of the slope at certified rank indicates multiplicity, density, or conditioning violations.

## 20. Reasoning-graph experiment

Use externally visible atomic-claim traces, not hidden chain-of-thought. Benchmark factual QA,
multi-hop graph QA, mathematical proof checking, and document-grounded generation.

Compare:

1. entropy/perplexity;
2. ordinary graph statistics;
3. H0 grounding persistence;
4. persistent cycles/SCCs;
5. anchored RKHS mismatch;
6. combined proof-carrying verifier.

Primary outcomes:

- unsupported-claim AUROC/AUPRC;
- false acceptance at fixed coverage;
- tokens and latency;
- graph compression ratio;
- calibration error under domain shift.

## 21. Token-efficiency experiment

Ablate:

- full support graph vs bottleneck forest;
- raw prior proof text vs Merkle module certificate;
- unanchored learned kernel vs anchored kernel;
- exhaustive verification vs risk-ranked dominator verification;
- one mismatch channel vs \(k\)-channel Q-SARD design.

The framework is successful only if accuracy improves at equal or lower total inference tokens and
the risk ledger remains calibrated.

---

# References used in this derivation

- q0 Rate Program Master v3.2, Parts I--V, Inventory/Trace, and Claim Language Annex.
- A. Rivera and H. Vanneuville, *The critical threshold for Bargmann--Fock percolation*,
  arXiv:1711.05012.
- S. Muirhead and H. Vanneuville, *The sharp phase transition for level set percolation of smooth
  planar Gaussian fields*, arXiv:1806.11545.
- A. Lerario and M. Stecconi, *Differential Topology of Gaussian Random Fields*,
  arXiv:1902.03805.
- B. K. Sriperumbudur, K. Fukumizu, and G. R. G. Lanckriet,
  *Universality, Characteristic Kernels and RKHS Embedding of Measures*,
  arXiv:1003.0887 / JMLR.
- M. Erraoui, M. Röckner, and J. L. da Silva,
  *Cameron--Martin Type Theorem for a Class of non-Gaussian Measures*,
  arXiv:2312.15695.
- A. N. Angelopoulos et al., *Conformal Risk Control*, arXiv:2208.02814.
- Recent empirical TDA/graph-spectral work on LLM reasoning:
  arXiv:2512.19135, arXiv:2510.20665, arXiv:2510.19117, arXiv:2605.26362.
