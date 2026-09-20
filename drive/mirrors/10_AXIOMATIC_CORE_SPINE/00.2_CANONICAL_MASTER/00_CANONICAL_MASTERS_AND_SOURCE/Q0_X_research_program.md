# Q0-X: Infinite-Volume Pairing Theory and Verified-Reasoning Architecture

**Status:** research derivation, July 14, 2026  
**Source discipline:** the live q0 Rate Program is Master v3.2. Statements below are labeled as
Established, Derived, Conditional Theorem, Hypothesis, or Open Obligation.

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
\frac{\rho_{\mathrm{sad}}(b)E_{\mathrm{sup}}}{6}
L^2r^3.
\]

At \(b=1.2\), using the current program values,

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
3. **Window Kac--Rice bound:** the pair-Palm conditional saddle intensity integrated over the
   height window of width \(\ell=\kappa r^3/6\) is at most
   \[
   \bar\rho_b\,\ell
   \]
   per unit area.
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

Conditional Kac--Rice and Tonelli give

\[
\mathbb E N_{\mathrm{win}}(\mathcal I_r)
\le
\bar\rho_b\ell\,\mathbb E|\mathcal I_r|
\le
\frac{\bar\rho_b A_b\kappa}{6}r^3.
\]

Add the existing local terms. ∎

### Main obligation

Prove the finite-susceptibility bound under the pair-Palm law.

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

# Part III. Persistence on reasoning graphs

## 9. Reasoning persistence complex

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

## 10. Signed evidence residual

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

## 11. Anchored trainable-kernel theorem

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

## 12. Proof object

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

## 13. Module certificates

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

## 14. Sequence measure

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

## 15. Verified-reasoning rate theorem

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
using only logarithmically many independent, well-conditioned mismatch coordinates.

It is a conditional theorem. The hard empirical/mathematical work is to certify the coverage,
Jacobian, density, and approximation assumptions on real model latent paths.

---

# Part VIII. Immediate experiments and falsifiers

## 16. BF infinite-volume experiment

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

## 17. Q-SARD exponent experiment

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

## 18. Reasoning-graph experiment

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

## 19. Token-efficiency experiment

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
