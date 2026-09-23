**Unified Master Source — Part 1**  
  
**C006 / q₀ Program Reconstruction: Canonical Architecture, Status, and Source-of-Truth Skeleton**  
  
**0. Purpose of This Master Source**  
  
This document is the beginning of a single consolidated source of truth for the C006 / q₀ theorem program.  
  
Its job is to merge the uploaded architecture files, theorem packages, lemma packages, audits, literature-verification passes, calibration records, kill logs, and proof-roadmap documents into one logically ordered research monograph.  
  
This master source will not treat the archive as a pile of independent drafts. It will treat it as one evolving mathematical object with:  
  
1. one notation system,  
2. one hypothesis ledger,  
3. one theorem dependency graph,  
4. one proof-status taxonomy,  
5. one kill registry,  
6. one calibration ledger,  
7. one literature boundary map,  
8. one open-obligation register,  
9. one reconstruction protocol for future AI or human researchers.  
  
The current archive is not merely about proving a theorem. It is also a stress test of an AI-assisted research architecture: can a frontier mathematical program survive literature gaps, failed derivations, changing empirical evidence, retractions, pre-registered falsification cycles, and multiple model passes without corrupting its internal logic?  
  
The answer so far is: yes, provisionally, because uncertainty has been compressed into named, enumerable proof obligations instead of left ambient.  
  
The archive’s core mathematical target is:  
  
q_0 = 1,  
  
the elder-rule selection constant for near-diagonal H_0** persistence pairing of smooth stationary Gaussian fields, together with a near-diagonal birth-death density law of the form**  
  
\text{density scale} \sim \ell^{-1/3},  
  
under the stated Gaussian-field hypotheses and the unresolved but isolated technical conditions.  
  
This master source begins by freezing the canonical map of the program.  
  
⸻  
  
**1. Program Identity**  
  
**1.1 Name**  
  
Canonical name:  
  
**The C006 / q₀ Program for Near-Diagonal **H_0** Persistence of Smooth Stationary Gaussian Fields**  
  
Short forms:  
  
* q₀ program  
* C006 architecture  
* elder-rule selection program  
* near-diagonal persistence-density program  
* Gaussian critical-pair persistence program  
  
**1.2 Mathematical domain**  
  
The program sits at the intersection of:  
  
* smooth Gaussian random fields,  
* Morse theory,  
* Morse–Smale gradient dynamics,  
* persistent homology,  
* H_0 elder-rule pairing,  
* Kac–Rice formulas,  
* Palm conditioning,  
* critical-point point processes,  
* near-collision asymptotics,  
* Gaussian regression,  
* conditional jet asymptotics,  
* Hermite interpolation,  
* spectral-moment geometry.  
  
**1.3 Target object**  
  
Let f** be a centered smooth stationary Gaussian field on a two-dimensional flat torus or locally on **\mathbb{R}^2**. Consider superlevel-set persistence:**  
  
E_t = \{x : f(x) \ge t\}.  
  
For H_0(E_t)**, connected components are born at local maxima and die when they merge through saddles. The elder rule decides which component survives and which component dies.**  
  
The central local question is:  
  
Given a nearby maximum-saddle critical pair at small spatial separation r**, what is the probability that the death saddle is paired with the nearby maximum rather than with some other older component?**  
  
This probability is denoted schematically by  
  
q(r,b,\ell),  
  
where:  
  
* r is the pair separation,  
* b is the base height,  
* \ell is the persistence lifetime / height gap,  
* q is the elder-rule local selection probability.  
  
The main selection constant is  
  
q_0 = \lim_{r \to 0} q(r,b,\ell),  
  
in the correct near-diagonal scaling regime.  
  
The flagship claim is:  
  
q_0 = 1.  
  
Interpretation: in the near-diagonal limit, a sufficiently close maximum-saddle fold pair is paired locally with probability tending to one.  
  
⸻  
  
**2. Archive-Level Status**  
  
The program is not finished. It is also not speculative in the loose sense.  
  
The correct status is:  
  
\textbf{PROVEN-MODULO named obligations}.  
  
The archive has already killed several earlier incorrect routes. The surviving structure is stronger precisely because failed claims were logged rather than silently absorbed.  
  
The most important current status claims are:  
  
**2.1 Theorem A status**  
  
**Theorem A: elder-rule selection constant **q_0 = 1**.**  
  
Status:  
  
\textbf{PROVEN-MODULO}  
  
depending on the remaining inner-zone / near-degenerate third-point control, now substantially sharpened by the Flat-Saddle Tail Lemma.  
  
The theorem does not require the exact asymptotic constant of the inner-zone count. It requires that the obstruction probability tends to zero. The updated Flat-Saddle Tail Lemma gives an upper bound strong enough for this purpose.  
  
**2.2 Theorem B status**  
  
**Theorem B: near-diagonal **H_0** bar-density law with **\ell^{-1/3}**-type scaling.**  
  
Status:  
  
\textbf{PROVEN-MODULO Theorem A and associated pair-density inputs}.  
  
The theorem depends structurally on Theorem A plus a critical-pair intensity degeneration law and a localization argument.  
  
**2.3 GCJA theorem status**  
  
**Graded Conditional-Jet Asymptotics theorem.**  
  
Status:  
  
\textbf{theorem/proof package; discharged at lemma level for the stated design class}.  
  
This package converts the earlier symbolic-conditioning and filtration claims into a formal theorem. It supports the scaling laws for conditional covariance, rank collapse, slaving, parity, positivity, and design-surgery behavior.  
  
**2.4 Flat-Saddle Tail Lemma status**  
  
Status:  
  
\textbf{derived upper bound; not yet a two-sided law}.  
  
The lemma corrects an earlier super-exponential claim and a later r^4**-assembly claim. The current upper bound is:**  
  
\mathbb{E}[N_{\mathrm{inner}}] \le C r^5(1+o(1))  
  
under the stated assumptions and uniformly over the relevant Palm base compact.  
  
This is enough for Theorem A because it tends to zero.  
  
The matching lower bound is still conjectural.  
  
⸻  
  
**3. The Central Story of the Research Program**  
  
The archive began with an attempted percolation-style or geometric intuition for why local maximum-saddle pairs should control near-diagonal H_0** persistence. That early picture contained load-bearing errors.**  
  
The program evolved by repeatedly converting vague geometric statements into finite, auditable mathematical obligations.  
  
The important evolution was not merely theorem refinement. It was uncertainty relocation.  
  
At the start, uncertainty was diffuse:  
  
* Is the elder-rule local?  
* Is the near-diagonal law real?  
* Does literature already contain the result?  
* Is the Gaussian conditioning correct?  
* Can third critical points enter the inner zone?  
* Does Palm conditioning preserve the expected suppression?  
* Does the determinant bias change the asymptotics?  
* Are there hidden Morse–Smale failures?  
* Is isotropy necessary?  
* Is the r^4 or r^5 count correct?  
* Is the \ell^{-1/3} exponent stable?  
  
By the current checkpoint, the uncertainty has been compressed into a smaller set of named obligations:  
  
1. exact symbolic/nondegeneracy checks for certain conditional covariance structures,  
2. positivity of finite-dimensional Gaussian integrals,  
3. completion of a full Morse–Smale almost-sure proof or accepted conditional lemma,  
4. lower-bound version of the Flat-Saddle Tail Lemma if an exact two-sided law is desired,  
5. thermodynamic-limit extension L \to \infty, currently excluded,  
6. final manuscript-level unification and notation cleanup.  
  
This is the main architectural success: the program may still fail at named points, but it is no longer vulnerable to undefined ambient uncertainty.  
  
⸻  
  
**4. Canonical Source Inventory**  
  
The archive currently contains these major source classes.  
  
**4.1 Constitutional / overview sources**  
  
**File 1 — Architectural Overview and Core Theorems**  
  
Role: constitution of the program.  
  
Owns:  
  
* global notation,  
* hypotheses,  
* main theorem statements,  
* novelty claims,  
* dependency graph,  
* kill registry,  
* calibration summary,  
* boundary of validity,  
* proof architecture.  
  
This is the root source.  
  
**Backstory source**  
  
Role: historical and architectural narrative.  
  
Owns:  
  
* how the program emerged,  
* why modular files were created,  
* how AI-assisted verification is being used,  
* why the current archive is a save point,  
* what each file contributes,  
* why uncertainty concentration matters.  
  
This source is not a proof document, but it is essential for reconstruction.  
  
⸻  
  
**4.2 Theorem package sources**  
  
**GCJA Theorem Package**  
  
Role: formal theorem/proof package for collapsing Gaussian observation designs.  
  
Owns:  
  
* spectral representation notation,  
* staircase design definition,  
* visible/unseen jet space decomposition,  
* graded conditional covariance asymptotics,  
* Hermite interpolation lemmas,  
* Hilbert-space projection lemmas,  
* multiplier maps,  
* leading-stratum analysis,  
* corollaries recovering previous symbolic observations.  
  
This is one of the most important infrastructure files.  
  
⸻  
  
**4.3 Proof architecture and filtration sources**  
  
**S3 Proof Architecture ND**  
  
Role: structural proof architecture for the nondegeneracy / filtration mechanism.  
  
Owns:  
  
* explanation that \Sigma_2(0) is not accidental,  
* reduction of symbolic covariance structure to a filtration theorem,  
* general-kernel version of the ND′ principle,  
* identification of which assumptions are actually used,  
* separation of Gaussian, stationarity, smoothness, isotropy, and Bargmann–Fock-specific ingredients.  
  
Key architectural point:  
  
**Isotropy is not structurally load-bearing for the filtration mechanism.**  
  
It simplifies constants and presentation, but the underlying design is a moment-problem invariant.  
  
**S4 Filtration On Trial**  
  
Role: adversarial stress test of the filtration hypothesis.  
  
Owns:  
  
* Gaussian vs non-Gaussian boundaries,  
* stationary vs non-stationary boundaries,  
* isotropic vs anisotropic boundaries,  
* full-rank vs degenerate spectral support boundaries,  
* Hermite interpolation interpretation,  
* candidate theorem formulation,  
* exact out-of-domain tests,  
* proposition-level ledger.  
  
This file functions as a falsification attempt against the filtration principle.  
  
⸻  
  
**4.4 Morse-theoretic source**  
  
**File 2 — Almost-Sure Morse–Smale Property**  
  
Role: literature verification and proof strategy for the Morse–Smale layer.  
  
Owns:  
  
* distinction between almost-sure Morse and almost-sure Morse–Smale,  
* identification of a real folklore gap,  
* reduction on \mathbb{T}^2,  
* saddle-connection obstruction,  
* parametric transversality strategy,  
* Cameron–Martin support strategy,  
* proof roadmap.  
  
Key finding:  
  
Almost-sure Morse and distinct critical values are literature-supported under appropriate nondegeneracy hypotheses, but a fully cited almost-sure no-saddle-connection / Morse–Smale result for Gaussian fields is not located as an off-the-shelf theorem.  
  
This is a genuine gap that must either be proved or carried as a named assumption.  
  
⸻  
  
**4.5 Three-point / Kac–Rice sources**  
  
**File 3 — Three-Point Kac–Rice Literature Verification**  
  
Role: literature boundary map for the three-point/two-scale expansion.  
  
Owns:  
  
* verification of existing k-point Kac–Rice machinery,  
* Ladgham–Lachièze-Rey divided-difference blow-up relevance,  
* Azaïs–Delmas conditional determinant relevance,  
* Beliaev–McAuley–Muirhead three-point intensity relevance,  
* Ancona–Letendre multijet desingularization relevance,  
* identification that the exact two-scale three-point expansion needed here appears novel.  
  
Key finding:  
  
The literature supports pieces of the machinery, but no located work executes the specific three-point, two-scale degeneration required by Lemma I.  
  
**File 3.1 — ThreePoint TwoScale KacRice**  
  
Role: constructive derivation source for the actual three-point/two-scale machinery.  
  
Owns:  
  
* working frame,  
* nested divided differences,  
* 15-dimensional Gaussian conditioning design,  
* exponent ledger,  
* value-window counting,  
* localization inputs,  
* unresolved nondegeneracy condition.  
  
This is one of the central proof-engine files.  
  
⸻  
  
**4.6 Pair-Palm / pinning source**  
  
**File 4 — Pinning Lemma Pair Palm**  
  
Role: pair-conditioning and Palm-law source.  
  
Owns:  
  
* critical-pair Palm conditioning,  
* fold-pair normal form,  
* regression consequences of pinning a nearby maximum-saddle pair,  
* deterministic on-axis behavior,  
* pair frame,  
* adjacency weighting,  
* interface with File 3.  
  
Core discovery:  
  
Conditioning on a nearby maximum-saddle pair pins the line-direction behavior strongly enough to impose a cubic / fold structure. This is what makes the elder-rule local-selection theorem possible.  
  
⸻  
  
**4.7 Near-diagonal density sources**  
  
**File 5 — Citation Verification**  
  
Role: citation and antecedent verification for the near-diagonal law.  
  
Owns:  
  
* Feldbrugge persistence integral check,  
* Rychlik rainflow check,  
* near-diagonal persistence-density antecedent search,  
* distinction between 1D and d \ge 2,  
* novelty audit.  
  
Key finding:  
  
One-dimensional exact or semi-exact pairing/rainflow laws exist, but they do not supply the desired d \ge 2** smooth Gaussian near-diagonal **H_0** law.**  
  
**File 5.1 — NearDiagonal Law Derivation**  
  
Role: derivation of the near-diagonal law.  
  
Owns:  
  
* execution of Theorem B architecture,  
* factorization into pair intensity and selection probability,  
* \ell^{-1/3} scaling law,  
* comparison with one-dimensional cases,  
* interface ledger.  
  
⸻  
  
**4.8 Anisotropy and literature foundation source**  
  
**File 6 — Literature Foundations Anisotropy Bridge**  
  
Role: literature provenance and anisotropy upgrade.  
  
Owns:  
  
* weakest-hypothesis analysis,  
* anisotropic extension,  
* distinction between isotropy-dependent constants and structurally invariant mechanisms,  
* bridge from isotropic calibration to anisotropic theorem scope.  
  
Key point:  
  
The program’s core selection architecture is not intrinsically isotropic. Isotropy is convenient, but the deeper mechanism is tied to nondegenerate spectral moment geometry and Gaussian conditional-jet structure.  
  
⸻  
  
**4.9 Updated audit and correction sources**  
  
**O1 Palm Mean Regression Audit**  
  
Role: adversarial closure package for Palm mean/covariance scaling.  
  
Owns:  
  
* correction of conditional mean scaling,  
* GCJA alignment,  
* Palm regression checks,  
* explicit argmin sets,  
* covariance scale verification.  
  
**Flat-Saddle Tail Lemma**  
  
Role: correction and replacement for the inner-zone suppression story.  
  
Owns:  
  
* retraction of super-exponential averaged suppression,  
* correction of r^4 claim to r^5 upper bound,  
* explanation of determinant bias near flat saddles,  
* polynomial tail mechanism,  
* preservation of Theorem A despite correction.  
  
**C006 Arm1c Snapshot**  
  
Role: pre-registered empirical falsification snapshot.  
  
Owns:  
  
* immutable falsification condition,  
* bootstrap confidence interval rule,  
* observed median-ratio protocol,  
* deterministic FALSIFIED / SURVIVES output rule.  
  
**S2 C006 Merged PreFreeze Spec**  
  
Role: draft supporting pre-freeze specification.  
  
Owns:  
  
* pre-freeze architecture notes,  
* C006 target definitions,  
* references to File 3 Remark 8.1-style empirical adjudication.  
  
⸻  
  
**5. Canonical Theorem Statements**  
  
The exact final manuscript wording will be reconstructed later. For now, the source-of-truth mathematical content is as follows.  
  
**5.1 Theorem A — Elder-Rule Selection Constant**  
  
Let f** be a centered stationary sufficiently smooth Gaussian field on the flat two-torus or locally on **\mathbb{R}^2**, satisfying the spectral moment and nondegeneracy assumptions listed in the hypothesis ledger.**  
  
Consider a near-diagonal maximum-saddle critical pair (x_M,x_S)** with separation**  
  
r = |x_M - x_S| \to 0,  
  
height gap / persistence scale  
  
\ell = f(x_M)-f(x_S),  
  
and base height b** restricted to a compact regular window.**  
  
Let q(r,b,\ell)** denote the conditional probability that the elder-rule death at **x_S** is paired with the nearby maximum **x_M**, rather than with a competing older component.**  
  
Then, in the near-fold scaling regime,  
  
q(r,b,\ell) \to 1  
  
uniformly over the allowed compact height base.  
  
Equivalently,  
  
q_0 = 1.  
  
Status:  
  
\textbf{PROVEN-MODULO inner-zone and Morse-geometric obligations}.  
  
The latest Flat-Saddle Tail Lemma strengthens the inner-zone control enough for the qualitative limit q_0=1**, although it changes the expected count law from a previously claimed super-exponential suppression to a polynomial suppression.**  
  
⸻  
  
**5.2 Theorem B — Near-Diagonal **H_0** Bar-Density Law**  
  
Let \ell** denote a small **H_0** persistence lifetime / birth-death height gap.**  
  
Under the same Gaussian smoothness and nondegeneracy hypotheses, and assuming Theorem A and the critical-pair intensity degeneration input, the near-diagonal H_0** bar-density obeys an **\ell^{-1/3}**-type law.**  
  
Schematic form:  
  
\rho_{H_0}(\ell) \sim C \ell^{-1/3}  
  
or, depending on normalization and integrated density convention,  
  
N(\ell \le \varepsilon) \sim C' \varepsilon^{2/3}.  
  
The exact constant depends on the spectral moment geometry and the chosen normalization.  
  
Status:  
  
\textbf{PROVEN-MODULO Theorem A and pair-intensity inputs}.  
  
The exponent is the central object. Constants are secondary and require final normalization.  
  
⸻  
  
**5.3 GCJA Theorem — Graded Conditional-Jet Asymptotics**  
  
Let f** be a centered stationary Gaussian field with spectral measure **\rho**, finite moments through the required order, and a nondegeneracy condition excluding spectral concentration on low-degree polynomial zero sets.**  
  
Consider a collapsing observation design D(p;d)** consisting of jets**  
  
\partial_t^m \partial_s^k f(c_i r,0),  
  
with p** nodes **c_i r** collapsing as **r \to 0**, and staircase jet orders **d_k**.**  
  
Then the conditional law of unseen jets at a nearby evaluation point admits a graded asymptotic expansion determined by:  
  
1. a visible polynomial jet space,  
2. an unseen stratum,  
3. Hermite interpolation error orders,  
4. spectral projection onto the visible complement,  
5. deterministic multiplier polynomials,  
6. a finite exponent ladder.  
  
The leading conditional covariance is the Gram pushforward of the leading unseen stratum.  
  
Status:  
  
\textbf{proved in theorem package under stated hypotheses}.  
  
Program role:  
  
This theorem explains why the conditional covariance scales are not accidental outputs of symbolic algebra. They are forced by the geometry of collapsing observation designs.  
  
⸻  
  
**5.4 Flat-Saddle Tail Lemma**  
  
Under the fold-pair Palm law at separation r**, base height **b**, and gap **\ell**, define the fold modulus**  
  
\kappa := \frac{6\ell}{r^3}.  
  
Let N_{\mathrm{win}}** be the number of third critical points in the inner window**  
  
|y-x_S| \le C r^2.  
  
Then, under the stated assumptions,  
  
\mathbb{E}[N_{\mathrm{win}}] \le C' r^5 (1+o(1)).  
  
The contributing configurations are not generic fold-pair configurations. They are near-flat configurations where the corrected transverse curvature is small:  
  
\widetilde{\mu}_S \asymp r^2.  
  
Status:  
  
\textbf{derived upper bound; matching lower bound open}.  
  
Critical correction:  
  
Earlier conditional-at-fixed-pair reasoning suggested super-exponential suppression. That was valid only for generic fixed pair data. After averaging over the Palm law with determinant bias, near-flat saddles dominate the averaged tail and convert exponential suppression into polynomial suppression.  
  
⸻  
  
**6. Hypothesis Ledger**  
  
This section records the assumptions in canonical form. Later installments will refine notation.  
  
**6.1 Gaussianity**  
  
The field f** is centered Gaussian.**  
  
This is load-bearing because:  
  
* conditional laws remain Gaussian,  
* conditioning equals Hilbert-space projection,  
* Kac–Rice densities are computable through Gaussian covariance matrices,  
* Palm conditioning is finite-dimensional Gaussian regression plus determinant weighting,  
* GCJA relies on Gaussian projection geometry.  
  
Non-Gaussian analogues may exist, but they are outside the current theorem claim.  
  
**6.2 Stationarity**  
  
The field is stationary:  
  
C(x,y)=C(x-y).  
  
This is load-bearing for:  
  
* spectral representation,  
* translation-invariant local coordinates,  
* uniform Kac–Rice expansions,  
* moment tensor normalization,  
* local fold chart stability.  
  
**6.3 Smoothness / spectral moments**  
  
The covariance has enough finite spectral moments.  
  
The GCJA theorem package uses moment conditions of the form:  
  
\lambda_{2m} = \int |k|^{[2m](x-apple-data-detectors://embedded-result/19577)}\,d\rho(k) < \infty  
  
up to the order required by the observation design.  
  
For the full (p,j)=(2,2)** design, the package references moment orders up to **12** for leading conclusions and **14** for error control.**  
  
Programmatically:  
  
* \Lambda_2 controls gradients,  
* \Lambda_4 controls Hessians,  
* \Lambda_6, \Lambda_8, and higher moments enter regression and error terms,  
* higher moment assumptions should be weakened wherever possible but not silently.  
  
**6.4 Nondegeneracy**  
  
A spectral nondegeneracy condition is required.  
  
Canonical GCJA form:  
  
No nonzero polynomial of degree up to the required bound vanishes \rho**-almost everywhere.**  
  
Interpretation:  
  
The spectral measure must not collapse onto a set that destroys the needed jet covariance ranks.  
  
For practical sufficient conditions:  
  
* spectral density positive on an open set is enough,  
* torus lattice versions require corresponding finite-dimensional nondegeneracy conditions.  
  
**6.5 Morse / Morse–Smale structure**  
  
The field must almost surely be Morse, have distinct critical values, and have no relevant saddle-connection degeneracies so that persistence pairing is well-defined through Morse–Smale gradient structure.  
  
Status:  
  
* almost-sure Morse: literature-supported under standard nondegeneracy,  
* distinct critical values: literature-supported under appropriate joint density bounds,  
* no saddle-saddle / no problematic saddle connections: genuine proof obligation or named assumption.  
  
**6.6 Compact height windows**  
  
Many uniformity statements are restricted to compact height windows.  
  
This avoids uncontrolled tail behavior at extreme heights.  
  
**6.7 Fixed torus size**  
  
The current theorem architecture is fixed-L**.**  
  
The thermodynamic limit  
  
L \to \infty  
  
before or jointly with  
  
r \to 0  
  
is deliberately excluded from the current theorem.  
  
This matters because global percolation effects can re-enter in the infinite-volume limit.  
  
⸻  
  
**7. Canonical Notation**  
  
This section begins the unified notation dictionary.  
  
**7.1 Field and covariance**  
  
f : \mathbb{T}^2_L \to \mathbb{R}  
  
or locally  
  
f : \mathbb{R}^2 \to \mathbb{R}.  
  
Mean:  
  
\mathbb{E}f(x)=0.  
  
Covariance:  
  
C(h)=\mathbb{E}[f(x)f(x+h)].  
  
Spectral representation:  
  
C(x)=\int e^{i\langle k,x\rangle}\,d\rho(k).  
  
**7.2 Critical points**  
  
Gradient:  
  
\nabla f = (f_1,f_2).  
  
Hessian:  
  
D^2 f =  
\begin{pmatrix}  
f_{11} & f_{12}\\  
f_{12} & f_{22}  
\end{pmatrix}.  
  
A local maximum x_M** satisfies:**  
  
\nabla f(x_M)=0,\qquad D^2f(x_M)<0.  
  
A saddle x_S** satisfies:**  
  
\nabla f(x_S)=0,\qquad \det D^2f(x_S)<0.  
  
**7.3 Pair variables**  
  
Maximum:  
  
x_M.  
  
Saddle:  
  
x_S.  
  
Separation:  
  
r=|x_M-x_S|.  
  
Base height:  
  
b.  
  
Gap / lifetime:  
  
\ell=f(x_M)-f(x_S).  
  
Fold modulus:  
  
\kappa = \frac{6\ell}{r^3}.  
  
**7.4 Pair frame**  
  
Use local coordinates (t,s)**, where:**  
  
* t is the axis along the maximum-saddle pair,  
* s is the transverse axis.  
  
The pair is usually normalized so that the two critical points lie on or near the t**-axis.**  
  
**7.5 Hessian components in pair frame**  
  
At the saddle and maximum:  
  
f_{tt},\quad f_{ts},\quad f_{ss}.  
  
The transverse curvature variable is defined schematically as  
  
\mu := -f_{ss}(x_S).  
  
The corrected transverse curvature used in the Flat-Saddle Tail Lemma is  
  
\widetilde{\mu}_S = \mu + r\nu_S^2/\kappa,  
  
with a corresponding \widetilde{\mu}_M** at the maximum.**  
  
The important relation is:  
  
\widetilde{\mu}_M = \widetilde{\mu}_S + r\widetilde{\eta}+O(r^2).  
  
**7.6 Persistence selection probability**  
  
q(r,b,\ell)  
  
denotes the probability that a near-fold maximum-saddle pair is the actual elder-rule pair.  
  
The limiting selection constant:  
  
q_0 = \lim q(r,b,\ell).  
  
The theorem target:  
  
q_0=1.  
  
⸻  
  
**8. Dependency Graph**  
  
The canonical dependency graph is:  
  
\text{Theorem B}  
\Leftarrow  
\text{Theorem A}  
+  
\text{pair-density degeneration}  
+  
\text{localization/change-of-variables}.  
  
\text{Theorem A}  
\Leftarrow  
\text{Morse–Smale layer}  
+  
\text{Pinning Lemma}  
+  
\text{Three-point Kac–Rice}  
+  
\text{Flat-Saddle Tail control}  
+  
\text{annulus/far/loop exclusion}.  
  
\text{Three-point Kac–Rice}  
\Leftarrow  
\text{GCJA}  
+  
\text{divided differences}  
+  
\text{conditional determinant bounds}  
+  
\text{value-window counting}  
+  
\text{nondegeneracy condition}.  
  
\text{Flat-Saddle Tail Lemma}  
\Leftarrow  
\text{Pinning Lemma}  
+  
\text{GCJA covariance scaling}  
+  
\text{Palm determinant bias}  
+  
\text{corrected transverse-curvature tail integration}.  
  
\text{Pinning Lemma}  
\Leftarrow  
\text{Gaussian regression}  
+  
\text{critical-pair Palm conditioning}  
+  
\text{fold normal form}.  
  
\text{Morse–Smale layer}  
\Leftarrow  
\text{almost-sure Morse}  
+  
\text{distinct critical values}  
+  
\text{no saddle-connection degeneracy}.  
  
⸻  
  
**9. Kill Registry**  
  
The kill registry is binding. A killed claim cannot be reused unless explicitly repaired.  
  
**K1–K4**  
  
Earlier kill entries exist in the archive and must be preserved in full during later reconstruction.  
  
Their exact wording will be imported in a later installment.  
  
**R3**  
  
Killed claim:  
  
\mathbb{E}[N_{\mathrm{inner}}(|y-x_S|\le Cr^2)]  
=  
O(e^{-c/r^4})  
  
under the fold-pair Palm law.  
  
Reason:  
  
This is valid only conditionally at fixed generic pair data. It fails after averaging over the Palm law because the determinant-biased distribution contains near-flat saddles where the suppression constant degenerates.  
  
Correct replacement:  
  
A polynomial upper bound driven by the flat-saddle tail.  
  
**R3b**  
  
Superseded claim:  
  
\mathbb{E}[N_{\mathrm{inner}}]\asymp r^4.  
  
Reason:  
  
The calculation treated the maximum-side determinant as an independent O(\kappa r)** factor in the relevant tail region. Smoothness instead forces**  
  
f_{ss}(x_M)=f_{ss}(x_S)+O_P(r),  
  
so on the contributing set the maximum-side determinant is dragged down by an additional power.  
  
Corrected upper bound:  
  
\mathbb{E}[N_{\mathrm{inner}}]\le C r^5(1+o(1)).  
  
Downstream effect:  
  
Theorem A survives because it only needs the inner obstruction to vanish.  
  
⸻  
  
**10. The Most Important Correction So Far**  
  
The most important update in the current archive is the Flat-Saddle Tail correction.  
  
Earlier reasoning separated two ideas incorrectly:  
  
1. conditional suppression at fixed generic pair data,  
2. averaged suppression under the Palm law.  
  
At fixed generic pair data, the third-point 1-jet has a conditional mean much larger than its standard deviation in the inner zone. This gives exponential suppression.  
  
But the Palm law does not sample only generic pair data. It includes determinant weighting:  
  
W = |\det H_M||\det H_S| \cdot 1_{\mathrm{adj}}.  
  
Near-flat saddles have small transverse curvature. The determinant weight penalizes them only polynomially, not exponentially. These rare near-flat configurations dominate the averaged inner-zone contribution.  
  
Thus:  
  
* conditional generic law: exponential suppression,  
* Palm-averaged law: polynomial suppression.  
  
This correction is not a failure of the theorem. It is a success of the architecture because the invalid route was isolated, killed, and replaced.  
  
The new mechanism is:  
  
\widetilde{\mu}_S \asymp r^2  
  
on the contributing set.  
  
Then:  
  
|\det H_S| \asymp r^3,  
  
and  
  
|\det H_M| \asymp r^2  
  
on the same tail set, due to smoothness coupling.  
  
This gives the updated power ledger leading to the r^5** upper bound.**  
  
⸻  
  
**11. Calibration and Falsification Interface**  
  
The program does not rely only on proof sketches. It also uses calibration cycles.  
  
The important design principle is:  
  
Predictions are snapshotted before evidence collection.  
  
A valid calibration cycle contains:  
  
1. prediction snapshot,  
2. evidence package,  
3. calibration report,  
4. rule-update decision.  
  
The C006 Arm1c snapshot is tied to the flat-saddle tail mechanism.  
  
The falsification protocol computes a ratio of medians between:  
  
* window pairs,  
* matched controls.  
  
It then bootstraps a confidence interval and applies a deterministic rule.  
  
The key empirical question is whether the near-flat / flat-saddle mechanism is visible in the sampled |f_{ss}(x_S)|** distribution.**  
  
The empirical result cannot prove the theorem, but it can falsify or support the proposed mechanism.  
  
This is important because the program has a pre-declared branch structure:  
  
* if the nondegeneracy condition holds cleanly, proceed on the main proof route;  
* if degeneracy is integrable, modify the exponent bookkeeping;  
* if collapse occurs on an open set, the current inner-zone proof route fails and must be rebuilt.  
  
⸻  
  
**12. Literature Boundary**  
  
The current archive’s novelty claim is not “nothing related exists.”  
  
The correct claim is narrower and stronger:  
  
No located source proves the exact elder-rule selection law q_0=1** for smooth stationary Gaussian fields in dimension **d\ge2**, and no located source derives the corresponding near-diagonal **H_0** persistence bar-density exponent for the smooth field setting.**  
  
Existing literature supplies important components:  
  
* Kac–Rice formulas,  
* critical-point intensity formulas,  
* Gaussian-field Morse results,  
* conditional determinant calculations,  
* divided-difference blow-up techniques,  
* 1D rainflow / persistence pairing analogues,  
* random topology universality results,  
* point-process persistence analogues,  
* multijet desingularization.  
  
But the archive identifies the missing synthesis:  
  
\text{elder-rule topology}  
+  
\text{critical-pair Palm conditioning}  
+  
\text{three-point two-scale degeneration}  
+  
\text{near-diagonal persistence density}  
  
for smooth Gaussian fields in d\ge2**.**  
  
This is the mathematical gap the program targets.  
  
⸻  
  
**13. What Is Already Structurally Strong**  
  
The strongest parts of the program are:  
  
1. the modular dependency graph,  
2. the explicit proof-obligation ledger,  
3. the GCJA theorem package,  
4. the corrected Flat-Saddle Tail mechanism,  
5. the distinction between conditional and Palm-averaged laws,  
6. the literature boundary map,  
7. the falsification protocol,  
8. the kill registry,  
9. the preservation of failed derivations,  
10. the separation of theorem claims from empirical calibration.  
  
The architecture is unusually robust because it does not erase mistakes. It uses mistakes as structural information.  
  
⸻  
  
**14. What Remains Weak or Open**  
  
The current weak points are:  
  
1. full manuscript-level proof of the Morse–Smale Gaussian-field result,  
2. final symbolic verification of the nondegeneracy condition where still carried,  
3. exact lower bound matching the Flat-Saddle Tail upper bound,  
4. final constant normalization in the near-diagonal density law,  
5. rigorous thermodynamic-limit extension,  
6. final unification of duplicate files and superseded text,  
7. final proof that all uses of isotropy have been removed or explicitly retained,  
8. final referee-grade citation cleanup.  
  
None of these are vague. Each can be assigned to a named section and resolved independently.  
  
⸻  
  
**15. Master Monograph Target Structure**  
  
The final unified document should have the following table of contents.  
  
**Volume I — Architecture and Theorems**  
  
1. Program overview  
2. Historical evolution and kill registry  
3. Global notation  
4. Hypotheses  
5. Superlevel H_0 persistence and elder rule  
6. Main theorem statements  
7. Dependency graph  
8. Boundary of validity  
9. Calibration interface  
  
**Volume II — Deterministic Morse and Persistence Layer**  
  
1. Morse theory background  
2. Morse–Smale requirements  
3. Almost-sure Morse property  
4. Distinct critical values  
5. Saddle-connection exclusion  
6. Elder-rule deterministic reduction  
7. Loop and crater obstructions  
8. Local-vs-global pairing reduction  
  
**Volume III — Gaussian Critical-Pair Machinery**  
  
1. Spectral representation  
2. Gaussian regression  
3. Kac–Rice formulas  
4. Palm conditioning  
5. Critical-pair Palm law  
6. Fold coordinates  
7. Pinning Lemma  
8. Pair determinant weights  
9. Adjacency indicator  
  
**Volume IV — Conditional-Jet Asymptotics**  
  
1. Collapsing observation designs  
2. Staircase designs  
3. Visible and unseen jet spaces  
4. Hermite unisolvence  
5. Divided differences  
6. Projection perturbation  
7. GCJA theorem  
8. Corollaries  
9. Design surgery  
10. Anisotropy bridge  
  
**Volume V — Three-Point Inner-Zone Analysis**  
  
1. Three-point Kac–Rice setup  
2. Two-scale degeneration  
3. Working frame  
4. Conditional covariance scaling  
5. Value-window counting  
6. Inner-zone obstruction  
7. Nondegeneracy condition  
8. Flat-saddle tail correction  
9. r^5 upper bound  
10. Lower-bound conjecture  
  
**Volume VI — Near-Diagonal Density Law**  
  
1. Pair-intensity degeneration  
2. Change of variables  
3. Selection factorization  
4. Theorem A assembly  
5. Theorem B assembly  
6. \ell^{-1/3} exponent  
7. 1D comparison  
8. Constants and normalization  
9. Extension limits  
  
**Volume VII — Literature and Verification**  
  
1. Literature map  
2. What is known  
3. What is adaptable  
4. What is absent  
5. Citation verification  
6. Calibration cycles  
7. C006 falsification protocol  
8. Reproducibility ledger  
9. Machine-readable appendices  
  
⸻  
  
**16. Immediate Next Reconstruction Step**  
  
The next installment should fill in the canonical **Global Definitions and Hypotheses chapter in full.**  
  
That chapter must normalize:  
  
* f,  
* C,  
* \rho,  
* \lambda_{[2m](x-apple-data-detectors://embedded-result/32124)},  
* \Lambda_2,\Lambda_4,\Lambda_6,\ldots,  
* x_M,x_S,  
* r,b,\ell,\kappa,  
* pair frame (t,s),  
* Hessian components,  
* Palm law,  
* determinant weight,  
* adjacency indicator,  
* q(r,b,\ell),  
* q_0,  
* inner / annulus / far zones,  
* N_{\mathrm{inner}},  
* corrected transverse curvature,  
* visible/unseen jet spaces,  
* design D(p;d),  
* GCJA ladder orders,  
* theorem status labels.  
  
## This is the correct next move because every later chapter depends on stable notation.  
  
**Unified Master Source — Part 2**  
  
**Global Definitions, Hypotheses, and Canonical Notation**  
  
⸻  
  
**17. Standing Space**  
  
The base domain is the flat two-torus  
  
\mathbb{T}_L^2=\mathbb{R}^2/(L\mathbb{Z})^2,  
  
with side length  
  
L\ge 1.  
  
Throughout the main theorem sequence,  
  
L \text{ is fixed}.  
  
This is not cosmetic. The fixed-L** condition prevents infinite-volume percolation effects from entering the local near-diagonal limit.**  
  
The metric is the quotient flat metric  
  
d(x,y).  
  
Lebesgue measure is denoted  
  
dx,  
  
with total volume  
  
|\mathbb{T}_L^2|=L^2.  
  
Local calculations may be transferred to \mathbb{R}^2**, but the theorem statement lives on **\mathbb{T}_L^2** unless explicitly stated otherwise.**  
  
⸻  
  
**18. Gaussian Field**  
  
The field is  
  
f:\mathbb{T}_L^2\to\mathbb{R}.  
  
It is a centered stationary Gaussian random field.  
  
Centered means  
  
\mathbb{E}f(x)=0.  
  
Stationary means its covariance depends only on displacement:  
  
C(x)=\mathbb{E}[f(y)f(y+x)].  
  
Unit variance means  
  
C(0)=1.  
  
The field law is denoted  
  
\mathbb{P},  
  
and expectation is denoted  
  
\mathbb{E}.  
  
⸻  
  
**19. Spectral Representation**  
  
On the torus, the covariance has spectral representation  
  
C(x)=\sum_{k\in(2\pi/L)\mathbb{Z}^2}\rho(k)e^{i\langle k,x\rangle},  
  
where \rho** is a symmetric spectral measure on the dual lattice.**  
  
Symmetric means  
  
\rho(k)=\rho(-k).  
  
This guarantees that the field is real-valued.  
  
In continuum idealizations, one writes  
  
C(x)=\int_{\mathbb{R}^2}e^{i\langle k,x\rangle}\,d\rho(k).  
  
The theorem architecture is measure-theoretic in \rho**, so the torus and continuum forms use the same logic, with lattice sums replacing integrals.**  
  
⸻  
  
**20. No Isotropy Assumption**  
  
The field is not assumed isotropic.  
  
That means C(x)** is not required to depend only on **|x|**.**  
  
Equivalently, the spectral measure \rho** is not required to be rotationally invariant.**  
  
Earlier isotropic notation such as scalar moments \lambda_2,\lambda_4,\lambda_6** may be used for calibration comparison, but the theorem’s canonical form is anisotropic.**  
  
The anisotropic theorem uses spectral moment tensors.  
  
Isotropy is retained only when useful for:  
  
1. numerical calibration,  
2. simplified constants,  
3. comparison with older drafts,  
4. sanity-check examples such as Bargmann–Fock fields.  
  
It is not structurally load-bearing for the final q_0=1** architecture.**  
  
⸻  
  
**21. Derivative Notation**  
  
Coordinates are  
  
x=(x_1,x_2).  
  
First derivatives:  
  
f_i=\partial_{x_i}f.  
  
Gradient:  
  
\nabla f=(f_1,f_2).  
  
Second derivatives:  
  
f_{ij}=\partial_{x_i}\partial_{x_j}f.  
  
Hessian:  
  
D^2f=  
\begin{pmatrix}  
f_{11} & f_{12}\\  
f_{12} & f_{22}  
\end{pmatrix}.  
  
Third derivative tensor:  
  
D^3f.  
  
For a unit vector v**,**  
  
f_{vvv}=D^3f[v,v,v].  
  
In pair coordinates (t,s)**, the corresponding derivatives are:**  
  
f_t,\quad f_s,\quad f_{tt},\quad f_{ts},\quad f_{ss},\quad f_{ttt},\quad f_{tss},\quad \ldots  
  
where:  
  
* t is the longitudinal pair axis,  
* s is the transverse axis.  
  
⸻  
  
**22. Spectral Moment Tensors**  
  
The second spectral moment tensor is  
  
\Lambda_2=\mathrm{Cov}(\nabla f).  
  
Higher tensors are defined analogously:  
  
\Lambda_4=\mathrm{Cov}(D^2f),  
  
\Lambda_6=\mathrm{Cov}(D^3f),  
  
\Lambda_8=\mathrm{Cov}(D^4f).  
  
More generally, spectral moments are controlled by  
  
\int |k|^{2m}\,d\rho(k)<\infty.  
  
In isotropic scalar notation one sometimes writes  
  
\lambda_{2m}=\int |k|^{[2m](x-apple-data-detectors://embedded-result/35809)}\,d\rho(k).  
  
The global theorem sequence uses tensor notation because isotropy is not assumed.  
  
⸻  
  
**23. Critical Points**  
  
The critical set is  
  
\mathrm{Crit}(f)=\{x:\nabla f(x)=0\}.  
  
Under the main hypotheses, f** is almost surely Morse.**  
  
A critical point is Morse if the Hessian is nonsingular.  
  
The Morse index is the number of negative Hessian eigenvalues.  
  
In dimension two:  
  
* index 0: local minimum,  
* index 1: saddle,  
* index 2: local maximum.  
  
For superlevel H_0** persistence, the relevant births occur at local maxima and deaths occur at saddles.**  
  
⸻  
  
**24. Superlevel Filtration**  
  
The superlevel set at height t** is**  
  
E_t=\{x:f(x)\ge t\}.  
  
The filtration is descending in t**.**  
  
As t** decreases:**  
  
1. a new component is born at each local maximum,  
2. components merge at index-one saddles,  
3. the elder rule decides which component persists.  
  
The zeroth persistent homology is  
  
H_0(E_t).  
  
All H_0** statements are coefficient-independent.**  
  
⸻  
  
**25. Elder Rule**  
  
At a merge saddle s**, two superlevel components become connected.**  
  
Each component has a birth maximum.  
  
The component with the larger birth value is older.  
  
The component with the smaller birth value is younger.  
  
The elder rule says:  
  
\text{the younger component dies}.  
  
Thus, if a local maximum m** dies at saddle **s**, its **H_0** bar is**  
  
[f(s),f(m)].  
  
The death saddle of m** is denoted**  
  
D(m).  
  
The lifetime of the bar born at m** is**  
  
\ell(m)=f(m)-f(D(m)).  
  
The persistence diagram is  
  
\mathrm{dgm}(f)=\{(f(m),f(D(m))):m\text{ non-essential local maximum}\}.  
  
On the torus, exactly one H_0** component is essential per realization.**  
  
⸻  
  
**26. Maximum-Saddle Pair**  
  
A candidate local persistence pair consists of:  
  
(m,s),  
  
where:  
  
* m is a local maximum,  
* s is an index-one saddle.  
  
The spatial separation is  
  
r=d(m,s).  
  
The birth height is  
  
b=f(m).  
  
The death height is  
  
f(s)=b-\ell.  
  
The lifetime is  
  
\ell=f(m)-f(s)>0.  
  
The main near-diagonal regime is  
  
r\to0,  
\qquad  
\ell\to0,  
  
with fold scaling  
  
\ell\asymp r^3.  
  
⸻  
  
**27. Gradient Adjacency**  
  
A maximum-saddle pair (m,s)** is gradient-adjacent if one ascending separatrix of **s** terminates at **m**.**  
  
Equivalently, under the gradient flow of f**, one unstable branch of the saddle ascends into the maximum.**  
  
This is a geometric adjacency condition, not yet a homological pairing condition.  
  
The main theorem proves that near the diagonal, this geometric adjacency becomes homological pairing with probability tending to one.  
  
⸻  
  
**28. Fold Pair**  
  
A pair (m,s)** is an **r**-fold pair at height **b** if:**  
  
1. m is a local maximum,  
2. s is an index-one saddle,  
3. d(m,s)=r,  
4. f(m)=b,  
5. f(s)=b-\ell,  
6. \ell>0,  
7. s is gradient-adjacent to m,  
8. the pair satisfies the local fold normal form.  
  
The fold normal form locks the height gap to the spatial separation:  
  
\ell=\frac{\kappa}{6}r^3(1+o(1)).  
  
Here  
  
\kappa=|f_{vvv}(x_0)|  
  
along the soft direction at the midpoint  
  
x_0=\frac{m+s}{2}.  
  
Equivalently,  
  
\kappa=\frac{6\ell}{r^3}.  
  
The fold modulus \kappa** is positive and remains in a compact nondegenerate window in the main Palm base.**  
  
⸻  
  
**29. Selection Probability**  
  
For small r>0**, compact height window **B\subset\mathbb{R}**, and fold-pair Palm law at height **b\in B**, define**  
  
q(r,b)  
=  
\mathbb{P}(D(m)=s\mid (m,s)\text{ is an }r\text{-fold pair at height }b).  
  
This is the probability that the local gradient-adjacent fold saddle is also the actual elder-rule death saddle of the maximum.  
  
The main constant is  
  
q_0=\lim_{r\to0}q(r,b).  
  
Theorem A asserts  
  
q_0=1  
  
uniformly for b** in compact sets.**  
  
No height threshold appears in the final theorem.  
  
No covariance sign condition appears.  
  
No isotropy condition appears.  
  
⸻  
  
**30. Near-Diagonal Bar Count**  
  
For a compact birth-height window B**, define**  
  
N(\ell_1,\ell_2;B)  
  
as the expected number of non-essential H_0** bars with lifetime in**  
  
[\ell_1,\ell_2]  
  
and birth height in B**.**  
  
When a density exists, write  
  
\nu(\ell)  
  
for the expected bar-count density per unit lifetime as  
  
\ell\to0.  
  
Theorem B asserts the near-diagonal law  
  
\nu(\ell)=C_*\ell^{-1/3}(1+o(1)).  
  
The constant C_*** is positive and finite under the theorem hypotheses.**  
  
⸻  
  
**31. Pair Intensity**  
  
Let  
  
\rho_{\mathrm{pair}}(r;b)  
  
denote the Palm intensity of r**-fold maximum-saddle pairs at birth height **b**.**  
  
The relevant two-point critical-pair input is the contact neutrality in dimension two:  
  
\rho_{\mathrm{pair}}(r;b)\to \rho_0(b)\in(0,\infty)  
  
as  
  
r\to0.  
  
This is the pair-intensity nondegeneration input, also called Proposition B2 or link L2.  
  
It is one of the main analytic bridges from critical-point process geometry to persistence-density asymptotics.  
  
⸻  
  
**32. Change of Variables Producing **\ell^{-1/3}  
  
The fold relation is  
  
\ell=\frac{\kappa}{6}r^3.  
  
Thus  
  
r=\left(\frac{6\ell}{\kappa}\right)^{1/3}.  
  
Differentiating,  
  
dr  
=  
\frac{1}{3}\left(\frac{6}{\kappa}\right)^{1/3}\ell^{-2/3}d\ell.  
  
The two-dimensional radial measure contributes  
  
r\,dr.  
  
Therefore,  
  
r\,dr  
\sim  
\ell^{1/3}\ell^{-2/3}d\ell  
=  
\ell^{-1/3}d\ell.  
  
This is the origin of the \ell^{-1/3}** exponent.**  
  
The exponent does not come from elder-rule selection.  
  
The elder-rule selection contributes a factor  
  
q_0=1.  
  
Thus the near-diagonal exponent is geometric-critical, while the selection theorem proves that no additional nontrivial selection constant suppresses it.  
  
⸻  
  
**33. Main Hypotheses**  
  
The final manuscript hypothesis set is intentionally small.  
  
The canonical global assumptions are:  
  
(H1+),\qquad (H2').  
  
Earlier drafts carried additional hypotheses, but the current architecture demotes them into lemmas or proof obligations.  
  
⸻  
  
**33.1 Hypothesis **H1+**: Smooth Stationary Gaussian Field**  
  
(H1+)  
  
The field f** is a centered stationary Gaussian field on **\mathbb{T}_L^2** with:**  
  
1. unit variance,  
2. C^3 sample paths,  
3. nondegenerate \Lambda_2,  
4. nondegenerate \Lambda_4,  
5. nondegenerate \Lambda_6,  
6. finite \Lambda_8.  
  
Equivalently, in scalar spectral shorthand,  
  
\sum_k |k|^8\rho(k)<\infty  
  
in the torus case.  
  
The role of \Lambda_8<\infty** is not universal. It is specifically used for sharp residual scaling in the annulus estimate.**  
  
If \Lambda_8** fails but the field has effective spectral smoothness**  
  
\nu\in(3,4),  
  
the theorem is expected to survive with degraded correction exponents, but those constants are not part of the main theorem.  
  
⸻  
  
**33.2 Hypothesis **H2'**: Finite-Jet Nondegeneracy**  
  
(H2')  
  
The spectral measure satisfies the finite-jet nondegeneracy condition:  
  
For every finite family of derivative functionals  
  
\{D^{\alpha_i}f(y_i)\},  
\qquad |\alpha_i|\le4,  
  
at finitely many distinct points y_i**, the joint Gaussian law is nondegenerate.**  
  
Collision degeneracies are controlled by Taylor order.  
  
A sufficient continuum condition is:  
  
\rho \text{ has density positive on an open set}.  
  
On the torus, the operative condition is not literal open-set density but the corresponding finite-dimensional nondegeneracy on the dual lattice.  
  
Equivalent Hilbert-space version:  
  
If a finite linear combination of derivative-evaluation functionals has spectral representative  
  
Q(k),  
  
then  
  
\int |Q(k)|^2\,d\rho(k)>0  
  
unless  
  
Q\equiv0  
  
as the relevant polynomial-exponential expression.  
  
This prevents hidden rank collapse in Gaussian regression.  
  
⸻  
  
**34. What Is Not Assumed**  
  
The theorem does not assume:  
  
1. isotropy,  
2. covariance positivity,  
3. FKG inequality,  
4. percolation monotonicity,  
5. sign condition on covariance,  
6. height threshold for b,  
7. thermodynamic limit,  
8. non-Gaussian universality,  
9. exact two-sided flat-saddle tail law,  
10. exact finite-r equality q(r,b)=1.  
  
The theorem is asymptotic:  
  
q(r,b)\to1.  
  
⸻  
  
**35. Smoothness Boundary**  
  
The current theorem is stated for C^3**-level fold geometry plus the stronger moment assumptions needed by the proof.**  
  
The archive contains a conjectural rougher-boundary extension:  
  
For effective smoothness  
  
3<\nu<4,  
  
the pinned residuals should scale with \nu**-dependent exponents, preserving the qualitative theorem but changing rates.**  
  
Below C^3**, the fold normal form itself is no longer available in the same form.**  
  
Therefore, the current theorem claims nothing below C^3**.**  
  
⸻  
  
**36. Height Window**  
  
Let  
  
B\subset\mathbb{R}  
  
be compact.  
  
All uniformity in b** is taken over compact **B**.**  
  
The theorem does not assert uniformity as  
  
|b|\to\infty.  
  
This restriction is necessary because Gaussian tail conditioning may alter constants and uniform regression bounds at extreme heights.  
  
⸻  
  
**37. Pair Frame**  
  
For a fold pair (m,s)**, define the pair axis.**  
  
Let t** be the longitudinal coordinate along the direction from saddle to maximum or maximum to saddle, depending on normalization.**  
  
Let s** be the transverse coordinate.**  
  
The notation is overloaded: s** can denote the saddle point and also the transverse coordinate. To avoid ambiguity in final manuscript form, the saddle point should be denoted**  
  
x_S  
  
and the transverse coordinate should be denoted  
  
u  
  
or  
  
\eta_\perp  
  
if needed.  
  
For now, pair-frame derivatives are written:  
  
f_t,\quad f_s,\quad f_{tt},\quad f_{ts},\quad f_{ss}.  
  
In final notation cleanup, use:  
  
x_M=\text{maximum},  
\qquad  
x_S=\text{saddle}.  
  
⸻  
  
**38. Pair-Conditioned Linear Event**  
  
The fold-pair Palm law first pins a linear event:  
  
E_{\mathrm{lin}}.  
  
This includes values and gradients at the maximum and saddle:  
  
f(x_M)=b,  
  
f(x_S)=b-\ell,  
  
\nabla f(x_M)=0,  
  
\nabla f(x_S)=0.  
  
The residual pair data are the six Hessian entries:  
  
D^2f(x_M),\qquad D^2f(x_S).  
  
The Palm density then weights by the determinant magnitudes and index/adjacency indicators.  
  
⸻  
  
**39. Palm Weight**  
  
The critical-pair Palm weight is  
  
W=|\det H_M|\,|\det H_S|\,1_{\mathrm{adj}},  
  
where  
  
H_M=D^2f(x_M),  
\qquad  
H_S=D^2f(x_S),  
  
and  
  
1_{\mathrm{adj}}  
  
is the gradient-adjacency indicator.  
  
For upper bounds, one may often use  
  
1_{\mathrm{adj}}\le1.  
  
But for exact constants and lower bounds, adjacency must be handled explicitly.  
  
This distinction matters in the Flat-Saddle Tail Lemma: dropping adjacency is valid for an upper bound, not for a two-sided law.  
  
⸻  
  
**40. Fold-Pair Regression Facts**  
  
Under E_{\mathrm{lin}}**, the pair-frame Hessian entries satisfy the following canonical scaling laws.**  
  
At the saddle:  
  
f_{tt}(x_S)=+\kappa r(1+O_P(r)).  
  
At the maximum:  
  
f_{tt}(x_M)=-\kappa r(1+O_P(r)).  
  
Mixed derivatives:  
  
f_{ts}(x_S)=r\nu_S,  
  
f_{ts}(x_M)=r\nu_M,  
  
with  
  
\nu_S,\nu_M=O_P(1).  
  
Transverse curvature at the saddle:  
  
f_{ss}(x_S)=-\mu.  
  
The conditional law of \mu** has nondegenerate limiting variance and a bounded positive density near relevant values.**  
  
Smoothness couples transverse curvature at the maximum and saddle:  
  
f_{ss}(x_M)=f_{ss}(x_S)-r\eta+O_P(r^2),  
  
where  
  
\eta=O_P(1)  
  
has nondegenerate conditional behavior.  
  
⸻  
  
**41. Corrected Transverse Curvature**  
  
Define  
  
\mu=-f_{ss}(x_S).  
  
The corrected saddle-side transverse curvature is  
  
\widetilde{\mu}_S=\mu+\frac{r\nu_S^2}{\kappa}.  
  
The corrected maximum-side transverse curvature is  
  
\widetilde{\mu}_M=\mu+r\left(\eta-\frac{\nu_M^2}{\kappa}\right)+O_P(r^2).  
  
Equivalently,  
  
\widetilde{\mu}_M  
=  
\widetilde{\mu}_S+r\widetilde{\eta}+O_P(r^2),  
  
where  
  
\widetilde{\eta}  
=  
\eta-\frac{\nu_M^2}{\kappa}+\frac{\nu_S^2}{\kappa}.  
  
The determinant approximations are:  
  
\det H_S  
=  
-\kappa r\,\widetilde{\mu}_S(1+O_P(r)),  
  
\det H_M  
=  
+\kappa r\,\widetilde{\mu}_M(1+O_P(r)).  
  
This corrected curvature is the key variable in the Flat-Saddle Tail Lemma.  
  
⸻  
  
**42. Flat-Saddle Set**  
  
The flat-saddle contribution occurs when  
  
\widetilde{\mu}_S=O(r^2).  
  
On this event,  
  
|\det H_S|\asymp r^3.  
  
Because of smoothness coupling,  
  
\widetilde{\mu}_M  
=  
\widetilde{\mu}_S+r\widetilde{\eta}+O_P(r^2),  
  
so typically  
  
\widetilde{\mu}_M\asymp r.  
  
Therefore,  
  
|\det H_M|\asymp r^2.  
  
This gives the determinant depletion:  
  
|\det H_S||\det H_M|\asymp r^5  
  
on the contributing flat-saddle tail set.  
  
The earlier r^4** claim failed because it did not account correctly for the maximum-side determinant being dragged down by smoothness.**  
  
⸻  
  
**43. Inner Window**  
  
The inner window around the saddle is  
  
|y-x_S|\le Cr^2.  
  
Use rescaled coordinates:  
  
y=x_S+r^2z,  
  
with  
  
|z|\le C.  
  
The expected number of third critical points in this window is  
  
N_{\mathrm{inner}}.  
  
The current upper bound is  
  
\mathbb{E}N_{\mathrm{inner}}\le C'r^5(1+o(1)).  
  
This is enough for Theorem A because  
  
r^5\to0.  
  
⸻  
  
**44. Annulus and Far Zones**  
  
The obstruction analysis separates space near the candidate pair into zones.  
  
Canonical zoning:  
  
1. inner zone:  
    |y-x_S|\le Cr^2;  
1. annulus zone:  
    Cr^2<|y-x_S|\le c r;  
1. far zone:  
    |y-x_S|>c r.  
  
Exact constants are not universal.  
  
The logic is:  
  
* inner zone requires two-scale three-point Kac–Rice and flat-saddle tail control,  
* annulus requires pinned-residual suppression,  
* far zone requires global/local separation and Gaussian-regression bounds,  
* loop/crater obstructions require deterministic Morse–Smale topology.  
  
⸻  
  
**45. Graded Conditional-Jet Asymptotics: Global Notation**  
  
The GCJA theorem uses a centered stationary Gaussian field on \mathbb{R}^2** or **\mathbb{T}_L^2**.**  
  
Let \rho** be the spectral measure.**  
  
A real linear functional  
  
\Lambda=\sum c_{\alpha,x}\partial^\alpha f(x)  
  
corresponds to a spectral representative  
  
\lambda(k)=\sum c_{\alpha,x}(ik)^\alpha e^{i\langle k,x\rangle}.  
  
The covariance identity is  
  
\mathbb{E}[\Lambda\Lambda']  
=  
\int \lambda(k)\overline{\lambda'(k)}\,d\rho(k).  
  
This is the Hilbert-space backbone of the entire conditional-regression architecture.  
  
⸻  
  
**46. Staircase Design**  
  
Fix:  
  
p\ge2  
  
distinct real nodes  
  
c_1<\cdots<c_p.  
  
Fix a staircase  
  
d=(d_0\ge d_1\ge\cdots\ge d_J\ge0).  
  
At scale r**, the observation design is**  
  
D(p;d).  
  
It observes  
  
O_r=  
\{  
\partial_t^m\partial_s^k f(c_i r,0):  
k\le J,\,  
m\le d_k,\,  
i\le p  
\}.  
  
The Gaussian span is  
  
S_r=\mathrm{span}(O_r)\subset L^2(\rho).  
  
Its dimension is  
  
n_D=\sum_k p(d_k+1)  
  
for small r**, assuming nondegeneracy.**  
  
The full design D(p,j)** is**  
  
d_k=j-k,  
\qquad  
J=j.  
  
⸻  
  
**47. Visible Space**  
  
For each transverse level k**, define**  
  
q_k=p(d_k+1)  
  
for  
  
k\le J,  
  
and  
  
q_k=0  
  
for  
  
k>J.  
  
The monomial functional is  
  
\phi_{m,k}(k')=(ik'_t)^m(ik'_s)^k.  
  
The visible space is  
  
V(D)=\mathrm{span}\{\phi_{m,k}:k\le J,\ m\le q_k-1\}.  
  
The unseen space is the complement of the visible polynomial jet directions.  
  
An unseen monomial is  
  
\psi_\sigma=\phi_{m,k}  
  
where  
  
\sigma=(m,k)  
  
is unseen, meaning  
  
k>J  
  
or  
  
m\ge q_k.  
  
The key design principle is:  
  
\#V(D)=n_D.  
  
There is no wasted observation rank.  
  
⸻  
  
**48. Full-Design Weighted Ball**  
  
For the full design D(p,j)**,**  
  
V=\{(m,k):m+pk\le p(j+1)-1\}.  
  
This is a weighted ball with weights:  
  
m+pk.  
  
The minimal unseen stratum has weighted degree  
  
p(j+1).  
  
This is the abstract filtration mechanism behind the specific covariance-scaling miracles observed in earlier symbolic computations.  
  
⸻  
  
**49. Evaluation Scaling**  
  
Let the evaluation point be  
  
y=(c_pr+r^\gamma z_t,\ r^\gamma z_s),  
  
where:  
  
\gamma>0,  
  
and  
  
z=(z_t,z_s)  
  
lies in a fixed compact set.  
  
For an observable  
  
\phi_y^a=\partial^{(a_t,a_s)}f(y),  
\qquad |a|\le1,  
  
and unseen monomial  
  
\psi_\sigma,  
\qquad \sigma=(m,k),  
  
define the deterministic ladder order  
  
L_a(\sigma;\gamma).  
  
This is the exact power of r** carried by **\psi_\sigma**’s coefficient in the residual expansion of **\phi_y^a**.**  
  
The leading order for observable a** is**  
  
\ell_a=\min_\sigma L_a(\sigma;\gamma).  
  
The leading stratum is  
  
\Sigma_a=\mathrm{argmin}_\sigma L_a(\sigma;\gamma).  
  
The finite exponent gap is  
  
\delta(D,\gamma)>0  
  
unless \gamma** lies at a tie value. At tie values, the tied strata jointly enter the limit.**  
  
⸻  
  
**50. Gram Pushforward**  
  
Let  
  
Q_V  
  
be orthogonal projection onto  
  
V(D)^\perp  
  
in  
  
L^2(\rho).  
  
Define  
  
G_{U|V}(\sigma,\tau)  
=  
\langle Q_V\psi_\sigma,Q_V\psi_\tau\rangle.  
  
This is the conditional covariance of unseen monomial functionals at the origin after conditioning on the visible space.  
  
The multiplier map  
  
m^a:\Sigma_a\to\mathbb{R}[z]  
  
assigns deterministic coefficient polynomials to leading unseen monomials.  
  
The GCJA limiting covariance object is  
  
\Sigma_\infty^{ab}(z)  
=  
\sum_{\sigma,\tau}  
m_\sigma^a(z)m_\tau^b(z)  
G_{U|V}(\sigma,\tau).  
  
⸻  
  
**51. GCJA Hypotheses**  
  
The GCJA theorem uses:  
  
(H-f),\quad (H-mom),\quad (H-nd).  
  
**51.1 **H-f  
  
The field is centered, stationary, Gaussian, with spectral measure \rho**.**  
  
**51.2 **H-mom  
  
The required spectral moment order is  
  
\lambda_{2m}<\infty  
  
for all  
  
2m\le M(D),  
  
where  
  
M(D)=2\max_k(q_k+k)+2.  
  
For the full (p,j)=(2,2)** design,**  
  
M=14.  
  
The leading-order conclusions require only moments through 12**; the extra **+2** provides an error margin.**  
  
**51.3 **H-nd  
  
Let  
  
N=\max_k(q_k+k).  
  
The spectral measure \rho** is **N**-nondegenerate:**  
  
No nonzero polynomial of degree  
  
\le 2N  
  
vanishes \rho**-almost everywhere.**  
  
This is implied if the spectral measure charges an open set in continuum settings.  
  
On the torus, this is supplied by H2'**.**  
  
⸻  
  
**52. Theorem Status Labels**  
  
The archive uses explicit status labels.  
  
Canonical labels:  
  
**52.1 PROVEN-HERE**  
  
A proof is included in the current manuscript sequence.  
  
**52.2 PROVEN-MODULO**  
  
The statement is proved conditional on named lemmas, proof obligations, or verification packages.  
  
This is not a vague hedge. It means the dependencies are finite and named.  
  
**52.3 DERIVED**  
  
A calculation or bound is executed in the archive, but may not yet be packaged as a final theorem.  
  
**52.4 LITERATURE-SUPPORTED**  
  
The claim is supported by cited literature, but the exact theorem may need adaptation.  
  
**52.5 PLAUSIBLE**  
  
The claim is coherent and supported by mechanism-level reasoning, but not proof-complete.  
  
**52.6 CONJECTURE**  
  
The statement is not proven.  
  
**52.7 OPEN**  
  
The statement is explicitly unresolved.  
  
**52.8 KILLED**  
  
The statement is false, invalid, superseded, or unusable.  
  
Killed claims cannot be reused unless repaired and relabeled.  
  
⸻  
  
**53. Main Theorem Status Ledger**  
  
**53.1 Theorem A**  
  
q_0=1.  
  
Status:  
  
\text{PROVEN-MODULO}.  
  
Dependencies:  
  
1. Pinning Lemma P,  
2. inner two-scale blow-up nondegeneracy Lemma I,  
3. Morse–Smale genericity Sublemma R0,  
4. Reduction Lemma R,  
5. far-field Lemma G,  
6. annulus Lemma A',  
7. band Lemmas B0,B1,  
8. Flat-Saddle Tail upper bound.  
  
The theorem needs only  
  
\mathbb{E}N_{\mathrm{inner}}\to0,  
  
not the exact exponent.  
  
**53.2 Theorem B**  
  
\nu(\ell)=C_*\ell^{-1/3}(1+o(1)).  
  
Status:  
  
\text{PROVEN-MODULO}.  
  
Dependencies:  
  
1. Theorem A,  
2. Proposition B2 / L2 pair-intensity nondegeneration,  
3. fold change of variables,  
4. compact height-window integration,  
5. constant positivity and finiteness.  
  
**53.3 GCJA**  
  
Status:  
  
\text{PROVEN-HERE / theorem package}.  
  
It supplies the filtration foundation for conditional covariance asymptotics.  
  
**53.4 Flat-Saddle Tail Lemma**  
  
Status:  
  
\text{DERIVED upper bound}.  
  
Open:  
  
matching lower bound.  
  
⸻  
  
**54. Reduction Architecture Notation**  
  
The obstruction to local elder-rule pairing is decomposed into events.  
  
Let  
  
\mathcal{F}_{r,b}  
  
denote the fold-pair Palm conditioning event.  
  
The failure event is  
  
\{D(m)\ne s\}.  
  
The proof decomposes:  
  
\{D(m)\ne s\}  
\subset  
\mathcal{E}_{\mathrm{inner}}  
\cup  
\mathcal{E}_{\mathrm{annulus}}  
\cup  
\mathcal{E}_{\mathrm{far}}  
\cup  
\mathcal{E}_{\mathrm{loop}}  
\cup  
\mathcal{E}_{\mathrm{band}}.  
  
Each term must vanish as  
  
r\to0.  
  
The resulting quantitative form is schematically  
  
1-q(r,b)  
\le  
C(L,B)[r^3+r^\beta]+C e^{-c/r^2}.  
  
Here:  
  
* r^3 comes from one class of geometric/global obstructions,  
* r^\beta comes from the inner two-scale lemma,  
* exponential terms come from Gaussian-regression suppression in outer zones.  
  
The exact value of \beta** is not essential for **q_0=1**.**  
  
Any  
  
\beta>0  
  
suffices.  
  
⸻  
  
**55. Uniformity Convention**  
  
Unless stated otherwise, all constants  
  
C,c>0  
  
may change line to line.  
  
They are uniform for:  
  
b\in B  
  
with B** compact, and**  
  
0<r\le r_0.  
  
Constants may depend on:  
  
1. L,  
2. B,  
3. spectral moment bounds,  
4. nondegeneracy constants,  
5. compact \kappa-window,  
6. chosen local coordinate charts.  
  
They do not depend on r** or **b** within the stated window.**  
  
⸻  
  
**56. Asymptotic Notation**  
  
a(r)=O(b(r))  
  
means  
  
|a(r)|\le C|b(r)|  
  
for small r**.**  
  
a(r)=o(b(r))  
  
means  
  
\frac{a(r)}{b(r)}\to0.  
  
a(r)\sim b(r)  
  
means  
  
\frac{a(r)}{b(r)}\to1.  
  
a(r)=\Theta(b(r))  
  
means both  
  
a(r)=O(b(r))  
  
and  
  
b(r)=O(a(r)).  
  
⸻  
  
**57. Calibration Variable**  
  
In isotropic numerical calibration, a dimensionless separation variable appears:  
  
u=r\sqrt{\lambda_4/\lambda_2}.  
  
In anisotropic settings, this becomes direction-dependent.  
  
Therefore u** is not part of the theorem’s core notation.**  
  
It is retained only for empirical comparison with previous calibration cycles.  
  
⸻  
  
**58. Canonical Logical Meaning of **q_0=1  
  
The statement  
  
q_0=1  
  
does not mean every nearby maximum-saddle pair is exactly paired at finite r**.**  
  
It means:  
  
Given a fold-pair Palm law at separation r**, the probability of elder-rule mismatch tends to zero.**  
  
Formally:  
  
\lim_{r\to0}  
\mathbb{P}(D(m)\ne s\mid (m,s)\text{ is an }r\text{-fold pair at height }b)  
=0.  
  
Uniformly:  
  
\sup_{b\in B}  
\mathbb{P}(D(m)\ne s\mid (m,s)\text{ is an }r\text{-fold pair at height }b)  
\to0.  
  
Thus:  
  
\lim_{r\to0}q(r,b)=1.  
  
The phrase “selection constant” refers to the limiting multiplicative factor in the near-diagonal bar-density law.  
  
The theorem says that factor is exactly one.  
  
⸻  
  
**59. Canonical Meaning of the Near-Diagonal Law**  
  
The law  
  
\nu(\ell)=C_*\ell^{-1/3}(1+o(1))  
  
means that the expected H_0** persistence diagram has a singular density near the diagonal.**  
  
Small lifetimes are abundant.  
  
The singularity exponent comes from the fold relation  
  
\ell\asymp r^3  
  
combined with two-dimensional spatial measure  
  
r\,dr.  
  
The law is not claiming that individual lifetimes follow an independent distribution.  
  
It is an intensity statement for the persistence diagram.  
  
⸻  
  
**60. Next Reconstruction Step**  
  
The next installment should reconstruct the deterministic topological layer:  
  
1. Morse functions,  
2. superlevel components,  
3. elder-rule pairing,  
4. gradient adjacency,  
5. Morse–Smale requirement,  
6. why local geometric adjacency is not automatically homological pairing,  
7. obstruction decomposition,  
8. Reduction Lemma structure,  
9. loop/crater/far/annulus/inner events,  
10. exact role of fixed L.  
  
## This is the correct next layer because Theorem A is not merely Gaussian. It is a deterministic Morse-theoretic statement plus probabilistic estimates.  
  
**Unified Master Source — Part 3**  
  
**Deterministic Topological Layer: Morse Theory, Elder Rule, Local Pairing, and Obstruction Decomposition**  
  
⸻  
  
**61. Purpose of the Deterministic Layer**  
  
The theorem  
  
q_0=1  
  
is not purely a Gaussian regression theorem.  
  
It has two distinct halves:  
  
1. a deterministic Morse-topological half,  
2. a probabilistic Gaussian-estimate half.  
  
The Gaussian half estimates how likely certain bad configurations are.  
  
The deterministic half explains which configurations are actually bad.  
  
This distinction is essential.  
  
A nearby maximum-saddle pair may be geometrically adjacent, but geometric adjacency alone does not automatically imply that the pair is the actual H_0** persistence pair. Persistent homology depends on component history, not merely on local gradient geometry.**  
  
Therefore the program needs a deterministic reduction:  
  
\text{elder-rule mismatch}  
\Rightarrow  
\text{one of finitely many geometric/probabilistic obstruction events occurs}.  
  
Once the failure event is decomposed into obstruction events, Gaussian machinery can estimate each obstruction probability.  
  
This deterministic reduction is the bridge between:  
  
\text{Morse geometry}  
  
and  
  
\text{Kac–Rice / Palm / critical-point counting}.  
  
⸻  
  
**62. Morse Functions on the Torus**  
  
Let  
  
M=\mathbb{T}_L^2.  
  
A smooth function  
  
f:M\to\mathbb{R}  
  
is Morse if every critical point is nondegenerate.  
  
That is, for each  
  
x\in M  
  
with  
  
\nabla f(x)=0,  
  
the Hessian  
  
D^2f(x)  
  
is invertible.  
  
On a compact surface, a Morse function has finitely many critical points.  
  
In dimension two, each critical point has one of three types:  
  
1. local minimum,  
2. saddle,  
3. local maximum.  
  
For superlevel H_0** persistence, the relevant critical points are:**  
  
* local maxima: births,  
* saddles: deaths/mergers.  
  
Local minima affect higher global structure but not finite non-essential H_0** deaths in the superlevel filtration.**  
  
⸻  
  
**63. Distinct Critical Values**  
  
A Morse function has distinct critical values if  
  
x\ne y,\quad \nabla f(x)=\nabla f(y)=0  
  
implies  
  
f(x)\ne f(y).  
  
This condition prevents tie-breaking ambiguity in persistence.  
  
Without distinct critical values, two births or deaths can occur at the same height, and the elder rule may require arbitrary conventions.  
  
The q₀ theorem assumes or proves, through Gaussian genericity, that critical values are almost surely distinct.  
  
Thus, in the deterministic layer:  
  
\text{all maxima have strictly ordered birth heights}.  
  
This makes “older” and “younger” unambiguous.  
  
⸻  
  
**64. Gradient Flow**  
  
The gradient flow is  
  
\dot{x}=\nabla f(x)  
  
for upward flow, or  
  
\dot{x}=-\nabla f(x)  
  
for downward flow.  
  
For superlevel components, upward flow from saddles toward maxima is the natural picture.  
  
A saddle in dimension two has:  
  
* two ascending separatrix branches,  
* two descending separatrix branches.  
  
The ascending branches terminate at local maxima under Morse–Smale genericity.  
  
The descending branches terminate at local minima.  
  
The unstable/stable terminology depends on whether one uses upward or downward flow. To avoid ambiguity, this master source uses directional language:  
  
* ascending branch: follows increasing f,  
* descending branch: follows decreasing f.  
  
⸻  
  
**65. Morse–Smale Condition**  
  
A Morse function is Morse–Smale if stable and unstable manifolds of critical points intersect transversely.  
  
On a compact two-dimensional surface, for gradient flows, the relevant obstruction is the presence of saddle-saddle connections.  
  
A saddle-saddle connection is a separatrix branch leaving one saddle and landing at another saddle.  
  
For the program’s purposes, the operational condition is:  
  
\text{Morse}+\text{distinct critical values}+\text{no saddle-saddle separatrix connections}.  
  
This ensures the gradient graph has the expected combinatorial structure.  
  
The archive identifies this as a genuine issue:  
  
* almost-sure Morse is standard under Gaussian nondegeneracy,  
* almost-sure distinct critical values is standard under joint-density conditions,  
* almost-sure Morse–Smale / no saddle connections is not available as a universally cited off-the-shelf theorem for the exact random-field class.  
  
Therefore the final theorem must either:  
  
1. prove no saddle connections under the program’s hypotheses, or  
2. explicitly carry Morse–Smale genericity as a named assumption.  
  
The cleanest final manuscript strategy is:  
  
\textbf{Assumption MS: } f \text{ is almost surely Morse–Smale.}  
  
Then provide a separate appendix proving or partially proving Assumption MS under stronger spectral support hypotheses.  
  
⸻  
  
**66. Superlevel Components**  
  
For each height  
  
t\in\mathbb{R},  
  
define the superlevel set  
  
E_t=\{x\in M:f(x)\ge t\}.  
  
As t** decreases from **+\infty** to **-\infty**:**  
  
* no components exist initially,  
* each local maximum creates a new connected component,  
* each saddle either merges two components or changes topology,  
* one final component persists forever.  
  
For H_0**, only merge saddles matter.**  
  
At a merge saddle, two previously distinct superlevel components become connected.  
  
⸻  
  
**67. Local Saddle Picture**  
  
Near an index-one saddle x_S**, the Morse lemma gives local coordinates **(u,v)** such that**  
  
f(u,v)=f(x_S)+u^2-v^2  
  
or the negative-equivalent convention.  
  
For superlevel sets just above the saddle level, there are two local arms.  
  
For superlevel sets just below the saddle level, the arms connect.  
  
Thus a saddle is locally a merger.  
  
However, whether the saddle creates an H_0** death depends on the global components attached to those two local arms.**  
  
If the two arms already belong to the same component globally, the saddle may instead create or close a loop. This is a loop/crater obstruction.  
  
Therefore local saddle geometry is not enough. One must track global component identity.  
  
⸻  
  
**68. Elder Rule in Deterministic Terms**  
  
Let x_S** be a merge saddle.**  
  
Let the two superlevel components just above level  
  
f(x_S)  
  
be  
  
C_1,\quad C_2.  
  
Let their birth maxima be  
  
m_1,\quad m_2.  
  
Assume distinct critical values.  
  
If  
  
f(m_1)>f(m_2),  
  
then C_1** is older and **C_2** is younger.**  
  
The elder rule pairs the younger maximum m_2** with the saddle **x_S**.**  
  
Thus  
  
D(m_2)=x_S.  
  
The older component persists.  
  
The persistence lifetime is  
  
f(m_2)-f(x_S).  
  
⸻  
  
**69. Candidate Pair Versus Actual Persistence Pair**  
  
Let (x_M,x_S)** be a nearby maximum-saddle pair.**  
  
The pair is a candidate local pair if:  
  
1. x_M is a local maximum,  
2. x_S is a saddle,  
3. one ascending branch from x_S terminates at x_M,  
4. d(x_M,x_S)=r\ll1,  
5. f(x_M)-f(x_S)=\ell\asymp r^3.  
  
The pair is an actual persistence pair if:  
  
D(x_M)=x_S.  
  
The whole q₀ theorem is about proving:  
  
\mathbb{P}(\text{candidate local pair is not actual persistence pair})\to0.  
  
Equivalently:  
  
\mathbb{P}(D(x_M)\ne x_S\mid\text{candidate fold pair})\to0.  
  
⸻  
  
**70. Why Local Adjacency Can Fail to Imply Persistence Pairing**  
  
There are several possible failure mechanisms.  
  
**70.1 Older-neighbor failure**  
  
The candidate maximum x_M** might be attached through **x_S** to a component born at a higher maximum.**  
  
If x_M** is not the younger branch at the merge, it will not die at **x_S**.**  
  
**70.2 Competing near maximum**  
  
A third maximum very near the saddle may have height close enough to interfere with the local elder-rule decision.  
  
**70.3 Competing near saddle**  
  
A third saddle may create a prior or simultaneous connection changing which components exist at the candidate level.  
  
**70.4 Loop/crater obstruction**  
  
The two local arms of the saddle may already belong to the same component, so the saddle does not merge two distinct components.  
  
**70.5 Global reconnection**  
  
A far-away path above the saddle level may connect the local arms before the candidate saddle is crossed.  
  
**70.6 Annulus obstruction**  
  
A critical point at intermediate distance may mediate a nonlocal connection while still being close enough to evade far-field estimates.  
  
**70.7 Inner-zone obstruction**  
  
A third critical point inside the r^2**-scale neighborhood of the candidate saddle may alter the local normal-form topology.**  
  
The reduction lemma must show that all mismatch events are contained in this finite list.  
  
⸻  
  
**71. Deterministic Reduction Principle**  
  
The deterministic reduction principle is:  
  
If the candidate fold pair (x_M,x_S)** is not the actual persistence pair, then there exists an alternate critical object or alternate superlevel path witnessing that failure.**  
  
That witness must fall into one of three spatial regimes:  
  
1. inner,  
2. annulus/intermediate,  
3. far/global.  
  
Thus:  
  
\{D(x_M)\ne x_S\}  
\subset  
\mathcal{O}_{\mathrm{inner}}  
\cup  
\mathcal{O}_{\mathrm{ann}}  
\cup  
\mathcal{O}_{\mathrm{far}}  
\cup  
\mathcal{O}_{\mathrm{loop}}  
\cup  
\mathcal{O}_{\mathrm{band}}.  
  
This is the logical heart of the deterministic layer.  
  
The probabilistic proof then estimates each term.  
  
⸻  
  
**72. Local Pair Geometry**  
  
In the fold-pair regime, place the saddle and maximum along the t**-axis.**  
  
A convenient normalization is:  
  
x_S=(-r/2,0),  
\qquad  
x_M=(r/2,0),  
  
or the reverse orientation.  
  
Along the longitudinal axis, the field resembles a cubic fold.  
  
A schematic normal form is:  
  
f(t,0)=b-\frac{\ell}{2}+\frac{\kappa}{6}\left(t^3-\frac{3r^2}{4}t\right)+\text{higher order}.  
  
The derivative along t** has two nearby zeros:**  
  
t=-r/2,\qquad t=r/2.  
  
The second derivative changes sign between them:  
  
* one point is saddle-like in the longitudinal direction,  
* the other is maximum-like in the longitudinal direction.  
  
The height gap is:  
  
\ell\asymp \kappa r^3.  
  
This cubic geometry is why near-diagonal lifetime \ell** corresponds to separation **r\sim \ell^{1/3}**.**  
  
⸻  
  
**73. Local Corridor**  
  
The candidate maximum and saddle determine a small local corridor.  
  
The corridor is a tubular region around the ascending separatrix from x_S** to **x_M**.**  
  
At the candidate saddle height  
  
f(x_S),  
  
the local component born at x_M** is supposed to merge through this corridor.**  
  
The local theorem says that with high conditional probability, no other critical point or high-level connection inside this corridor changes the component identity.  
  
⸻  
  
**74. Band Event**  
  
Define the height band:  
  
I_{\mathrm{band}}=[f(x_S),f(x_M)].  
  
Its width is  
  
|I_{\mathrm{band}}|=\ell\asymp r^3.  
  
Any critical point with value inside this band can potentially interfere with the persistence pairing.  
  
Thus define the band obstruction:  
  
\mathcal{O}_{\mathrm{band}}  
=  
\{\exists\text{ critical point }y\notin\{x_M,x_S\}:f(y)\in I_{\mathrm{band}}\text{ and }y\text{ is relevant}\}.  
  
The word “relevant” must be made precise by spatial zone.  
  
For far-field estimates, relevance is often bounded crudely by counting all critical points in the band.  
  
Since the band width is O(r^3)**, expected far-field band counts are typically **O(r^3)** on fixed volume.**  
  
This is one source of the r^3**-type error in the selection theorem.**  
  
⸻  
  
**75. Far-Zone Event**  
  
Let  
  
\delta>0  
  
be fixed small but independent of r**.**  
  
The far zone is  
  
\mathcal{Z}_{\mathrm{far}}  
=  
\{y:d(y,x_S)\ge\delta\}.  
  
The far obstruction is:  
  
\mathcal{O}_{\mathrm{far}}  
=  
\{\exists\text{ far critical point/path that changes the elder-rule decision in the band}\}.  
  
Because the torus size L** is fixed, the far zone has finite volume.**  
  
The Gaussian field at far points remains nondegenerate under pair conditioning.  
  
Thus Kac–Rice plus band-width counting gives a small probability, typically of order:  
  
O(\ell)=O(r^3).  
  
This is why fixed L** is crucial.**  
  
If L\to\infty**, the far-zone volume grows and this estimate may fail.**  
  
⸻  
  
**76. Annulus Event**  
  
The annulus zone surrounds the candidate pair but is not at the deepest r^2** inner scale.**  
  
A representative annulus is:  
  
Cr^2<d(y,x_S)<\delta.  
  
Some drafts further split it as:  
  
3r\le d(y,x_0)\le \delta  
  
and a smaller intermediate band.  
  
The annulus is difficult because:  
  
* the third point is close enough to feel the pair conditioning,  
* but not close enough for the full inner r^2 blow-up,  
* competing terms in the conditional-jet expansion may tie,  
* the GCJA exponent ladder reaches a phase transition near \gamma=1.  
  
The annulus event is:  
  
\mathcal{O}_{\mathrm{ann}}  
=  
\{\exists\text{ interfering critical point/path in the annulus with value in the critical band}\}.  
  
Status:  
  
\text{OPEN / PROVEN-MODULO depending on subregion}.  
  
The annulus is one of the remaining genuine technical obligations.  
  
⸻  
  
**77. Inner-Zone Event**  
  
The inner zone is:  
  
\mathcal{Z}_{\mathrm{inner}}  
=  
\{y:|y-x_S|\le Cr^2\}.  
  
This is the most singular spatial scale.  
  
A third critical point in this window could destroy the clean two-critical-point fold picture.  
  
Define:  
  
\mathcal{O}_{\mathrm{inner}}  
=  
\{\exists y\in\mathcal{Z}_{\mathrm{inner}}\setminus\{x_S\}: \nabla f(y)=0  
\text{ and } y\text{ is topologically interfering}\}.  
  
For upper bounds, it is enough to count all third critical points:  
  
\mathbb{P}(\mathcal{O}_{\mathrm{inner}})  
\le  
\mathbb{E}N_{\mathrm{inner}}.  
  
The current corrected bound is:  
  
\mathbb{E}N_{\mathrm{inner}}\le C r^5(1+o(1)).  
  
This is stronger than needed for q_0=1**.**  
  
⸻  
  
**78. Loop / Crater Event**  
  
A loop or crater obstruction occurs when the local saddle does not merge the component born at x_M** with a genuinely different component.**  
  
Instead, the two local branches above the saddle level may already be connected elsewhere at a height above  
  
f(x_S).  
  
Then crossing x_S** does not kill the **x_M**-component in the expected way.**  
  
This event is global/topological rather than purely local analytic.  
  
It can be witnessed by a superlevel path:  
  
\gamma:[0,1]\to E_{f(x_S)+\epsilon}  
  
connecting the two arms of the saddle without passing through x_S**.**  
  
For the near-diagonal fold pair, such a path would have to involve either:  
  
1. a critical point in the narrow height band,  
2. a far high-level connection,  
3. an annulus/inner critical obstruction,  
4. a saddle-connection degeneracy.  
  
Thus loop/crater failures are absorbed into the same obstruction decomposition.  
  
⸻  
  
**79. Saddle-Connection Degeneracy**  
  
If the gradient flow has a saddle-saddle connection, the combinatorial merge tree may be unstable.  
  
A saddle branch can land at another saddle, and small perturbations can change the pairing graph.  
  
In deterministic Morse theory, this is a codimension-one phenomenon.  
  
For a generic deterministic function, it is avoided.  
  
For Gaussian random fields, the archive records that a full almost-sure proof is not automatically supplied by existing local Kac–Rice arguments.  
  
Therefore saddle connections are handled by:  
  
\text{Assumption MS}  
  
or by a future global-flow proof.  
  
The q₀ program should not pretend that finite-jet nondegeneracy alone excludes saddle connections. It does not.  
  
⸻  
  
**80. Reduction Lemma — Canonical Form**  
  
The Reduction Lemma should be stated as follows.  
  
**Lemma R — Deterministic Reduction to Obstruction Events**  
  
Let f** be a **C^3** Morse–Smale function on **\mathbb{T}_L^2** with distinct critical values.**  
  
Let (x_M,x_S)** be a gradient-adjacent maximum-saddle fold pair satisfying:**  
  
d(x_M,x_S)=r,  
  
f(x_M)-f(x_S)=\ell\asymp r^3,  
  
and the local fold-normal-form conditions.  
  
Fix a compact height window and a sufficiently small local chart.  
  
Then, for all sufficiently small r**,**  
  
D(x_M)\ne x_S  
  
implies at least one of the following events:  
  
1. an inner-zone critical obstruction occurs,  
2. an annulus-zone critical obstruction occurs,  
3. a far-zone critical obstruction occurs,  
4. a height-band critical obstruction occurs,  
5. a loop/crater obstruction occurs,  
6. a Morse–Smale degeneracy occurs.  
  
Equivalently,  
  
\{D(x_M)\ne x_S\}  
\subset  
\mathcal{O}_{\mathrm{inner}}  
\cup  
\mathcal{O}_{\mathrm{ann}}  
\cup  
\mathcal{O}_{\mathrm{far}}  
\cup  
\mathcal{O}_{\mathrm{band}}  
\cup  
\mathcal{O}_{\mathrm{loop}}  
\cup  
\mathcal{O}_{\mathrm{MS}}.  
  
If f** is Morse–Smale, then**  
  
\mathcal{O}_{\mathrm{MS}}=\emptyset.  
  
If the loop/crater obstruction is defined through critical witnesses, it can be absorbed into the inner/annulus/far/band events.  
  
Therefore a reduced form is:  
  
\{D(x_M)\ne x_S\}  
\subset  
\mathcal{O}_{\mathrm{inner}}  
\cup  
\mathcal{O}_{\mathrm{ann}}  
\cup  
\mathcal{O}_{\mathrm{far}}  
\cup  
\mathcal{O}_{\mathrm{band}}.  
  
⸻  
  
**81. Proof Skeleton for the Reduction Lemma**  
  
The proof proceeds by contradiction.  
  
Assume:  
  
D(x_M)\ne x_S.  
  
Then at the saddle height, either:  
  
1. x_S does not merge the component born at x_M, or  
2. it merges it but x_M is not the younger branch, or  
3. the component born at x_M died earlier at another saddle.  
  
Each alternative produces a witness.  
  
**81.1 Earlier-death case**  
  
If x_M** died earlier, then there exists another saddle**  
  
x_{S'}  
  
such that  
  
D(x_M)=x_{S'}  
  
and  
  
f(x_S)<f(x_{S'})<f(x_M).  
  
Thus x_{S'}** lies in the height band.**  
  
Depending on its location, it belongs to inner, annulus, or far zone.  
  
**81.2 Older-branch case**  
  
If x_S** merges two components but **x_M** belongs to the older branch, then the other branch contains a maximum**  
  
x_{M'}  
  
with  
  
f(x_{M'})<f(x_M)  
  
if x_M** survives, or with a different ordering depending on orientation.**  
  
In either case, component identity changes through a critical event in the height band or a pre-existing connection. That yields a witness maximum/saddle/path.  
  
**81.3 Loop case**  
  
If the two arms of the saddle already belong to the same component above f(x_S)**, then there exists a path in the superlevel set connecting them before crossing **x_S**.**  
  
The creation of such a path must be supported by prior critical events or by a global high-level connection. Under Morse–Smale and distinct critical values, this can be localized to a finite critical witness.  
  
**81.4 No-witness contradiction**  
  
If no inner, annulus, far, or band witness exists, then the local component born at x_M** remains isolated until it crosses **x_S**, and **x_S** is the first saddle at which it can merge.**  
  
Since x_M** is the only local birth in the fold cell and no other higher maximum attaches, the elder rule must pair **x_M** with **x_S**.**  
  
Contradiction.  
  
Therefore the mismatch implies an obstruction.  
  
⸻  
  
**82. Why the Height Band Is Width **r^3  
  
The fold relation is  
  
\ell=f(x_M)-f(x_S)=\frac{\kappa}{6}r^3(1+o(1)).  
  
Thus any critical value capable of changing the pairing of x_M** before **x_S** must lie between:**  
  
f(x_S)  
  
and  
  
f(x_M).  
  
This interval has width:  
  
O(r^3).  
  
Hence global critical-count obstructions away from the pair are small because they require critical values to fall into a shrinking band.  
  
This is the simplest probabilistic reason q_0=1** is plausible.**  
  
⸻  
  
**83. Why the Inner Scale Is **r^2  
  
The inner scale  
  
|y-x_S|\sim r^2  
  
is not arbitrary.  
  
It is the scale at which the pinned fold geometry and third-point critical equations balance.  
  
Near the saddle, the longitudinal derivative has a small slope controlled by the fold-pair separation r**. The transverse and mixed terms under Palm conditioning have their own degenerating scales. When a third critical point is placed at distance **r^\gamma**, the conditional mean and covariance of the gradient residual change with **\gamma**.**  
  
The GCJA filtration identifies the critical inner scale as the point where the unseen conditional-jet stratum begins to contribute in a new way.  
  
For the (p,j)=(2,2)** pair design, the inner **r^2** scale is where the nested divided-difference structure becomes necessary.**  
  
Thus:  
  
r^2  
  
is the canonical third-point collision scale.  
  
⸻  
  
**84. Why the Annulus Is Hard**  
  
The far zone is controlled by ordinary nondegenerate Kac–Rice estimates.  
  
The inner zone is controlled by a specialized two-scale blow-up.  
  
The annulus sits between them.  
  
At annulus scales, the third point is close enough that pair conditioning matters, but not close enough for the fully normalized inner chart to dominate cleanly.  
  
The GCJA package identifies a phase transition at a boundary value of the scaling parameter \gamma**. At that boundary, competing unseen strata tie.**  
  
This creates the annulus obligation:  
  
O\text{-ND2}.  
  
The annulus is therefore not a leftover detail. It is a structurally real boundary between two asymptotic regimes.  
  
⸻  
  
**85. Probabilistic Translation of the Reduction Lemma**  
  
Condition on a fold pair.  
  
Then:  
  
1-q(r,b)  
=  
\mathbb{P}(D(x_M)\ne x_S\mid\mathcal{F}_{r,b}).  
  
By the Reduction Lemma:  
  
1-q(r,b)  
\le  
P_{\mathrm{inner}}(r,b)  
+  
P_{\mathrm{ann}}(r,b)  
+  
P_{\mathrm{far}}(r,b)  
+  
P_{\mathrm{band}}(r,b)  
+  
P_{\mathrm{loop}}(r,b).  
  
The program then proves:  
  
P_{\mathrm{inner}}(r,b)\to0,  
  
P_{\mathrm{ann}}(r,b)\to0,  
  
P_{\mathrm{far}}(r,b)\to0,  
  
P_{\mathrm{band}}(r,b)\to0,  
  
P_{\mathrm{loop}}(r,b)\to0.  
  
Therefore:  
  
1-q(r,b)\to0.  
  
Hence:  
  
q_0=1.  
  
⸻  
  
**86. Expected Bound Form**  
  
The target quantitative form is:  
  
1-q(r,b)  
\le  
C_1 r^{\alpha_{\mathrm{inner}}}  
+  
C_2 r^{\alpha_{\mathrm{ann}}}  
+  
C_3 r^3  
+  
C_4e^{-c/r^\theta}  
+  
o(1),  
  
with all exponents positive.  
  
The exact exponent values are not required for Theorem A.  
  
Current expected ledger:  
  
\alpha_{\mathrm{inner}}=5  
  
as an upper-bound exponent for the inner critical-count contribution.  
  
The far/band contribution is typically:  
  
O(r^3).  
  
The annulus exponent remains a proof obligation, but any positive exponent suffices for the selection constant.  
  
⸻  
  
**87. Essential Versus Non-Essential Bars**  
  
On the torus, one H_0** class persists forever.**  
  
This is the essential class.  
  
The near-diagonal law concerns finite bars, i.e. non-essential components that die at saddles.  
  
The global maximum gives the essential bar in the superlevel filtration.  
  
Near-diagonal bars correspond to local maxima that are close in height to their death saddles.  
  
Because \ell\to0**, these are local fold-pair events, not global maxima.**  
  
The probability that the global maximum participates in a near-diagonal r\to0** fold pair is negligible in the fixed-**L** local intensity calculation, and in any case it can be excluded by treating essential bars separately.**  
  
⸻  
  
**88. Merge Tree Formulation**  
  
The deterministic layer can also be expressed using the merge tree.  
  
For a Morse function f**, the superlevel-set merge tree has:**  
  
* leaves corresponding to local maxima,  
* internal nodes corresponding to merge saddles,  
* root corresponding to the final essential component.  
  
Each non-root leaf is paired with the internal node where its branch merges into an older branch.  
  
The elder rule is exactly the merge-tree pruning rule.  
  
The q₀ theorem says:  
  
In the near-diagonal fold regime, the local maximum-saddle edge in the gradient adjacency graph becomes the corresponding leaf-parent edge in the merge tree with probability tending to one.  
  
This distinction is important:  
  
\text{gradient graph edge}  
\neq  
\text{merge-tree edge}  
  
in general.  
  
The theorem proves asymptotic equality for near-fold edges.  
  
⸻  
  
**89. Gradient Graph Versus Merge Tree**  
  
The gradient graph, or Morse–Smale 1-skeleton, contains separatrix connections between saddles and extrema.  
  
The merge tree records component mergers in the filtration.  
  
A saddle may have ascending branches to two maxima. Those maxima are gradient-adjacent to the saddle.  
  
But only one of those maxima dies at that saddle under the elder rule.  
  
Therefore:  
  
\text{two gradient-adjacent maxima}  
\quad\rightarrow\quad  
\text{one persistence death}.  
  
The elder rule chooses the lower maximum.  
  
For a near-fold maximum-saddle pair, the candidate maximum is close in both space and height. The theorem says that it is overwhelmingly the lower local branch and therefore the one killed at the saddle.  
  
The rare failures are precisely the obstruction events.  
  
⸻  
  
**90. Deterministic Local Cell**  
  
Around a fold pair, define a local cell  
  
U_r  
  
of radius  
  
O(r)  
  
or smaller around the pair.  
  
Inside U_r**, the field has exactly the candidate maximum and saddle if the inner/annulus obstructions are absent.**  
  
The local superlevel component born at x_M** is then confined to the local lobe until it reaches the saddle level.**  
  
If no high-level path exits U_r** and reconnects elsewhere, then the saddle **x_S** is the first possible death.**  
  
Thus local cell isolation plus no-band-critical-points implies persistence pairing.  
  
⸻  
  
**91. Role of Compactness**  
  
The torus is compact, so:  
  
1. the number of critical points is finite almost surely,  
2. the merge tree is finite,  
3. all superlevel component changes occur at finitely many heights,  
4. far-zone critical counts are finite,  
5. global reconnection events can be reduced to finite critical witnesses.  
  
This compactness is another reason fixed L** is part of the theorem.**  
  
In noncompact \mathbb{R}^2**, the merge tree of the whole field is not automatically finite. One must use windowed persistence or intensity measures.**  
  
The current theorem avoids that complication.  
  
⸻  
  
**92. Deterministic Proof Obligation Remaining**  
  
The deterministic layer has one serious unresolved issue:  
  
\text{almost-sure Morse–Smale property for the Gaussian field class}.  
  
Finite-jet nondegeneracy proves local critical nondegeneracy. It does not automatically exclude global saddle connections.  
  
The final program should split this into:  
  
**92.1 Main theorem with Assumption MS**  
  
State Theorem A conditional on almost-sure Morse–Smale.  
  
This keeps the main proof clean.  
  
**92.2 Appendix theorem MS-Gaussian**  
  
Attempt to prove Assumption MS under stronger spectral assumptions.  
  
A possible route:  
  
1. prove the field has full support in C^k around any deterministic perturbation in its Cameron–Martin space,  
2. show saddle-connection events can be destroyed by an arbitrarily small admissible perturbation,  
3. use a parametric transversality / Sard argument over finite saddle-pair branches,  
4. control countability through compactness and isolated critical points,  
5. conclude the saddle-connection event has probability zero.  
  
This appendix is technically nontrivial and should not be hidden.  
  
⸻  
  
**93. Why Finite-Jet Methods Cannot Prove Morse–Smale Alone**  
  
A saddle connection depends on the global integral curve of the gradient flow.  
  
It is not determined by any finite jet at the saddle endpoints.  
  
Two functions can share arbitrarily high finite jets at critical points but differ in whether a separatrix reaches another saddle.  
  
Therefore Kac–Rice finite-dimensional density methods are insufficient by themselves.  
  
This is why the archive correctly treats Morse–Smale as a separate global-flow problem.  
  
⸻  
  
**94. Deterministic Final Assembly of Theorem A**  
  
Once the Gaussian estimates are available, the deterministic assembly is short.  
  
Assume:  
  
P_{\mathrm{inner}}(r,b)\le C r^{\alpha_1},  
  
P_{\mathrm{ann}}(r,b)\le C r^{\alpha_2},  
  
P_{\mathrm{far}}(r,b)\le C r^3,  
  
P_{\mathrm{band}}(r,b)\le C r^3,  
  
P_{\mathrm{loop}}(r,b)\le C r^{\alpha_3}.  
  
with  
  
\alpha_i>0.  
  
Then  
  
1-q(r,b)  
\le  
C(r^{\alpha_1}+r^{\alpha_2}+r^{\alpha_3}+r^3).  
  
Therefore:  
  
\lim_{r\to0}q(r,b)=1.  
  
Uniformity over compact B** follows if all constants are uniform over **b\in B**.**  
  
Thus:  
  
q_0=1.  
  
⸻  
  
**95. Deterministic Final Assembly of Theorem B**  
  
Theorem B does not require new deterministic topology beyond Theorem A.  
  
It uses the factorization:  
  
\text{bar intensity}  
=  
\text{candidate fold-pair intensity}  
\times  
\text{selection probability}.  
  
As  
  
r\to0,  
  
the selection probability tends to  
  
q_0=1.  
  
Therefore the near-diagonal bar-density exponent is inherited directly from the candidate pair intensity and the fold change of variables.  
  
No hidden topological constant remains.  
  
⸻  
  
**96. Failure Modes of the Deterministic Layer**  
  
The deterministic layer would fail if any of the following occurred:  
  
1. a nearby maximum-saddle fold pair could fail to pair without any critical/path witness,  
2. the loop/crater obstruction could occur without reducing to a band or zone witness,  
3. saddle connections occurred with positive probability and changed pairing non-negligibly,  
4. the local fold chart failed to isolate the candidate lobe,  
5. compactness failed due to an unhandled thermodynamic limit,  
6. essential bars contaminated the local near-diagonal intensity,  
7. distinct critical values failed with positive probability.  
  
The current architecture handles all except the full Gaussian Morse–Smale proof, which remains named.  
  
⸻  
  
**97. Clean Final Theorem Form With Deterministic Assumption**  
  
The cleanest final statement is:  
  
**Theorem A — Conditional Form**  
  
Let f** be a centered stationary Gaussian field on **\mathbb{T}_L^2** satisfying **H1+**, **H2'**, and Assumption MS. Let **B\subset\mathbb{R}** be compact. Under the fold-pair Palm law at separation **r** and birth height **b\in B**, let **q(r,b)** be the probability that the candidate maximum-saddle fold pair is the actual **H_0** elder-rule persistence pair.**  
  
Then  
  
\sup_{b\in B}|q(r,b)-1|\to0  
  
as  
  
r\to0.  
  
Equivalently,  
  
q_0=1.  
  
This statement is honest, compact, and proof-ready.  
  
⸻  
  
**98. Stronger Final Theorem Form Without Deterministic Assumption**  
  
The stronger version is:  
  
**Theorem A′ — Full Gaussian Form**  
  
Under H1+** and **H2'**, the field **f** is almost surely Morse–Smale, and therefore Theorem A holds without Assumption MS.**  
  
Status:  
  
\text{OPEN / requires global-flow proof}.  
  
This should not be stated as proven until the Morse–Smale appendix is complete.  
  
⸻  
  
**99. Recommended Manuscript Strategy**  
  
The final monograph should use a two-tier statement.  
  
First:  
  
\textbf{Theorem A: } q_0=1 \text{ conditional on Assumption MS.}  
  
Second:  
  
\textbf{Conjecture / Appendix Theorem MS: } H1+ + H2' \Rightarrow \text{Assumption MS}.  
  
If Appendix MS is completed, Theorem A can be upgraded.  
  
This avoids overstating literature support.  
  
⸻  
  
**100. Next Reconstruction Step**  
  
The next installment should reconstruct the Gaussian critical-pair layer:  
  
1. Kac–Rice formula,  
2. two-point critical-pair Palm conditioning,  
3. fold-pair linear event,  
4. determinant weighting,  
5. pair-frame normalization,  
6. Gaussian regression under pinned maximum-saddle values,  
7. Hessian scaling,  
8. corrected transverse curvature,  
9. Pinning Lemma,  
10. relation between fold geometry and Palm density.  
  
## This is the probabilistic engine that turns the deterministic obstruction decomposition into quantitative bounds.  
  
**Unified Master Source — Part 4**  
  
**Gaussian Critical-Pair Layer: Kac–Rice, Pair Palm Conditioning, Pinning, and Fold Regression**  
  
⸻  
  
**101. Purpose of the Gaussian Critical-Pair Layer**  
  
The deterministic layer reduced elder-rule failure to obstruction events.  
  
The Gaussian critical-pair layer supplies the probability law under which those obstruction events are estimated.  
  
The central object is not the unconditional Gaussian field. It is the Gaussian field conditioned to contain a nearby maximum-saddle candidate pair.  
  
This conditioning is singular as  
  
r\to0.  
  
The pair constraints collapse onto one point, so ordinary fixed-point Kac–Rice conditioning is insufficient. The correct object is a degenerating two-point Palm law, renormalized through divided differences.  
  
The key outputs of this layer are:  
  
1. the two-point Kac–Rice pair density,  
2. the pair-Palm representation,  
3. the determinant-weighted Hessian law,  
4. the fold-pair normal form,  
5. the cubic pinning theorem,  
6. the corrected transverse-curvature variables,  
7. the Palm density input used by inner, annulus, and far-zone estimates.  
  
⸻  
  
**102. Critical-Point Kac–Rice Formula**  
  
Let  
  
f:\mathbb{T}_L^2\to\mathbb{R}  
  
be a smooth centered stationary Gaussian field.  
  
For a Borel set  
  
A\subset\mathbb{T}_L^2  
  
and a critical-point class \mathcal{C}**, the expected number of critical points in **A** is given by a Kac–Rice formula:**  
  
\mathbb{E}N_{\mathcal{C}}(A)  
=  
\int_A  
p_{\nabla f(x)}(0)  
\,  
\mathbb{E}\left[  
|\det D^2f(x)|  
1_{\mathcal{C}}(D^2f(x),f(x))  
\mid  
\nabla f(x)=0  
\right]  
dx.  
  
For local maxima:  
  
\mathcal{C}=\{D^2f(x)<0\}.  
  
For saddles:  
  
\mathcal{C}=\{\det D^2f(x)<0\}.  
  
The determinant factor  
  
|\det D^2f(x)|  
  
is not optional. It is the Jacobian of the gradient map.  
  
⸻  
  
**103. Two-Point Critical Kac–Rice Formula**  
  
For two distinct points  
  
x\ne y,  
  
the joint intensity of critical points at x** and **y** is**  
  
\rho_2(x,y)  
=  
p_{\nabla f(x),\nabla f(y)}(0,0)  
\,  
\mathbb{E}  
\left[  
|\det H_x|\,|\det H_y|  
\,1_{\mathcal{C}_x}\,1_{\mathcal{C}_y}  
\mid  
\nabla f(x)=0,\nabla f(y)=0  
\right],  
  
where  
  
H_x=D^2f(x),  
\qquad  
H_y=D^2f(y).  
  
For a maximum-saddle pair,  
  
1_{\mathcal{C}_x}=1_{\max},  
\qquad  
1_{\mathcal{C}_y}=1_{\mathrm{saddle}}.  
  
If height marks are included, one inserts delta densities or conditional densities for  
  
f(x)=b,  
\qquad  
f(y)=b-\ell.  
  
The pair density for a marked maximum-saddle pair is therefore schematically:  
  
\rho_{\mathrm{MS}}(x,y;b,\ell)  
=  
p_{\Phi_{x,y}}(b,b-\ell,0,0)  
\,  
\mathbb{E}  
\left[  
|\det H_x|\,|\det H_y|  
1_{\max}(H_x)1_{\mathrm{saddle}}(H_y)  
\mid  
\Phi_{x,y}  
\right],  
  
where  
  
\Phi_{x,y}  
=  
(f(x),f(y),\nabla f(x),\nabla f(y)).  
  
⸻  
  
**104. Pair Coordinates**  
  
For near-diagonal analysis, set the midpoint to zero:  
  
x_0=0.  
  
Use a unit vector  
  
e_t  
  
along the pair axis and a perpendicular unit vector  
  
e_s.  
  
Place the two candidate critical points at  
  
x_M=-\frac r2 e_t,  
  
x_S=+\frac r2 e_t.  
  
Some earlier drafts use the opposite orientation. The sign convention is not load-bearing, but the final manuscript must choose one and hold it fixed.  
  
Coordinates are:  
  
(t,s),  
  
with  
  
t=\langle x,e_t\rangle,  
\qquad  
s=\langle x,e_s\rangle.  
  
The pair values are:  
  
f(x_M)=b,  
  
f(x_S)=b-\ell.  
  
The fold scaling is:  
  
\ell=\frac{\kappa}{6}r^3.  
  
Thus:  
  
\kappa=\frac{6\ell}{r^3}.  
  
⸻  
  
**105. Raw Pair Constraint Vector**  
  
Define the six raw pair functionals:  
  
\Phi_r(f)  
=  
\left(  
f(x_M),  
f(x_S),  
f_t(x_M),  
f_s(x_M),  
f_t(x_S),  
f_s(x_S)  
\right).  
  
The fold-pair linear constraint is:  
  
\Phi_r(f)  
=  
(b,b-\ell,0,0,0,0).  
  
For every fixed  
  
r>0,  
  
the covariance matrix of \Phi_r** is invertible under finite-point nondegeneracy.**  
  
But as  
  
r\to0,  
  
\mathrm{Cov}(\Phi_r)  
  
degenerates.  
  
This is the core problem.  
  
A raw fixed-frame conditioning formula produces exploding inverse covariances. The solution is to replace \Phi_r** by a renormalized divided-difference frame.**  
  
⸻  
  
**106. Corrected Divided-Difference Pair Frame**  
  
The correct normalized frame is not the naive value difference divided by r^3**.**  
  
The naive coordinate  
  
\frac{f(x_S)-f(x_M)}{r^3}  
  
contains a divergent contribution from the first derivative unless the gradient constraints are already imposed.  
  
For off-surface regression theory, it must be corrected.  
  
Define:  
  
V^+=\frac{f(x_S)+f(x_M)}2.  
  
Define the corrected cubic value-difference coordinate:  
  
V^-_{\mathrm{corr}}  
=  
\frac{  
f(x_S)-f(x_M)  
-  
r\frac{f_t(x_S)+f_t(x_M)}2  
}{r^3}.  
  
Define gradient sum and difference coordinates:  
  
G_t^+=\frac{f_t(x_S)+f_t(x_M)}2,  
  
G_t^-=\frac{f_t(x_S)-f_t(x_M)}r,  
  
G_s^+=\frac{f_s(x_S)+f_s(x_M)}2,  
  
G_s^-=\frac{f_s(x_S)-f_s(x_M)}r.  
  
The corrected pair frame is:  
  
\Psi_r^{\mathrm{corr}}  
=  
\left(  
V^+,  
V^-_{\mathrm{corr}},  
G_t^+,  
G_t^-,  
G_s^+,  
G_s^-  
\right).  
  
On the pair constraint surface, where  
  
f_t(x_M)=f_t(x_S)=0,  
  
the correction term vanishes, so  
  
V^-_{\mathrm{corr}}  
=  
\frac{f(x_S)-f(x_M)}{r^3}  
=  
-\frac{\ell}{r^3}.  
  
Thus the corrected frame agrees with the intuitive frame on the Palm event, while remaining valid off the event.  
  
⸻  
  
**107. Limiting Pair Frame**  
  
Taylor expansion at the midpoint gives:  
  
V^+\to f(0).  
  
The corrected value coordinate satisfies:  
  
V^-_{\mathrm{corr}}\to -\frac{1}{12}f_{ttt}(0).  
  
The longitudinal gradient sum satisfies:  
  
G_t^+\to f_t(0).  
  
The longitudinal gradient difference satisfies:  
  
G_t^-\to f_{tt}(0).  
  
Up to sign depending on orientation, the transverse coordinates satisfy:  
  
G_s^+\to f_s(0),  
  
G_s^-\to f_{ts}(0).  
  
Therefore the limiting conditioning frame is the six-dimensional jet block  
  
\left(  
f(0),  
f_{ttt}(0),  
f_t(0),  
f_{tt}(0),  
f_s(0),  
f_{ts}(0)  
\right),  
  
up to fixed nonzero constants and signs.  
  
Under finite-jet nondegeneracy, this limiting covariance matrix is invertible.  
  
Hence the corrected frame has uniformly bounded condition number for small r**.**  
  
This is the key technical fact that makes uniform pair-Palm conditioning possible.  
  
⸻  
  
**108. Pair-Palm Regression Decomposition**  
  
Let  
  
\Psi_r=\Psi_r^{\mathrm{corr}}.  
  
For any finite family of field values or derivatives Y**, Gaussian regression gives:**  
  
Y\mid \Psi_r=\psi  
\sim  
\mathcal{N}  
\left(  
C_{Y\Psi}\Sigma_\Psi^{-1}\psi,  
\;  
\Sigma_Y-C_{Y\Psi}\Sigma_\Psi^{-1}C_{\Psi Y}  
\right).  
  
Equivalently, there is a decomposition  
  
f(y)=m_r(y;\psi)+g_r(y),  
  
where:  
  
m_r(y;\psi)  
=  
C_{f(y),\Psi_r}\Sigma_\Psi(r)^{-1}\psi,  
  
and g_r** is a centered Gaussian residual field independent of **\Psi_r**.**  
  
The residual covariance is the Schur complement:  
  
C_g^{(r)}(y,y')  
=  
C(y-y')  
-  
C_{f(y),\Psi_r}  
\Sigma_\Psi(r)^{-1}  
C_{\Psi_r,f(y')}.  
  
This is the pair-Palm Gaussian kernel before Hessian determinant weighting.  
  
⸻  
  
**109. From Conditional Gaussian Law to Critical-Pair Palm Law**  
  
Conditioning on the six linear constraints produces the Gaussian residual law.  
  
But the critical-pair Palm law is not just this Gaussian conditional law.  
  
It includes the Kac–Rice determinant weight:  
  
W=  
|\det H_M|\,|\det H_S|\,  
1_{\max}(H_M)\,  
1_{\mathrm{saddle}}(H_S)\,  
1_{\mathrm{adj}}.  
  
Here:  
  
H_M=D^2f(x_M),  
\qquad  
H_S=D^2f(x_S).  
  
Thus the pair-Palm expectation of a test functional F** is:**  
  
\mathbb{E}_{\mathrm{Palm}}[F]  
=  
\frac{  
\mathbb{E}\left[  
F  
|\det H_M|\,|\det H_S|\,  
1_{\max}1_{\mathrm{saddle}}1_{\mathrm{adj}}  
\mid  
\Phi_r=(b,b-\ell,0,0,0,0)  
\right]  
}{  
\mathbb{E}\left[  
|\det H_M|\,|\det H_S|\,  
1_{\max}1_{\mathrm{saddle}}1_{\mathrm{adj}}  
\mid  
\Phi_r=(b,b-\ell,0,0,0,0)  
\right]  
}.  
  
For upper bounds, one may replace  
  
1_{\max}1_{\mathrm{saddle}}1_{\mathrm{adj}}  
\le 1.  
  
For exact constants, this replacement is not valid.  
  
⸻  
  
**110. Fold Normal Form from Pair Constraints**  
  
On the pair constraint surface, the field along the pair axis is forced to have a cubic fold.  
  
Write the Taylor expansion at the midpoint:  
  
f(t,s)  
=  
\sum_{j+k\le3}  
\frac{c_{jk}}{j!k!}t^j s^k  
+  
R_4(t,s).  
  
The constraints  
  
f_t(\pm r/2,0)=0  
  
force:  
  
c_{10}  
=  
-\frac{r^2}{8}c_{30}  
+  
O(r^4),  
  
c_{20}  
=  
O(r^2).  
  
The value gap constraint forces:  
  
c_{30}  
=  
\frac{12\ell}{r^3}  
+  
O(r^2).  
  
Using  
  
\ell=\frac{\kappa}{6}r^3,  
  
this becomes:  
  
c_{30}=2\kappa+O(r^2).  
  
Thus the longitudinal cubic coefficient is pinned.  
  
The resulting leading on-axis polynomial is:  
  
f(t,0)  
=  
b'  
+  
\kappa\left(\frac{t^3}{3}-\frac{r^2}{4}t\right)  
+  
\text{higher-order correction},  
  
with critical points at  
  
t=\pm r/2.  
  
The height gap is:  
  
\frac{\kappa r^3}{6}.  
  
This closes the fold consistency check.  
  
⸻  
  
**111. Pinning Theorem — Canonical Statement**  
  
**Theorem P3 — Cubic Pinning**  
  
Under H1+**, condition on the corrected pair frame satisfying**  
  
\Psi_r^{\mathrm{corr}}  
=  
\psi  
  
equivalently, on the pair constraints  
  
f(x_M)=b,\quad f(x_S)=b-\ell,  
\quad  
\nabla f(x_M)=\nabla f(x_S)=0.  
  
In pair coordinates centered at the midpoint, the Taylor expansion of f** satisfies:**  
  
1. the on-axis cubic coefficient is pinned:  
  
c_{30}=2\kappa+O_P(r^2),  
  
1. the low-order axis coefficients are forced into fold position:  
  
c_{10}= -\frac{r^2}{8}c_{30}+O_P(r^4),  
  
c_{20}=O_P(r^2),  
  
1. the transverse-gradient constraints force corresponding mixed low-order coefficients,  
2. the directions c_{12} and c_{03} remain free at leading order,  
3. the conditioned residual on the axis has standard deviation  
  
\Theta(\tau^4)  
  
at distance \tau**, not **\Theta(\tau^3)**.**  
  
This theorem is one of the main proof-engine results.  
  
It says the nearby maximum-saddle pair does not merely mark two critical points. It pins an entire cubic fold geometry.  
  
⸻  
  
**112. Free and Pinned Jet Directions**  
  
The pair constraints pin:  
  
1,\quad t,\quad s,\quad t^2,\quad ts,\quad t^3  
  
directions in the midpoint jet algebra, up to correction orders.  
  
The leading free cubic directions include:  
  
ts^2,  
\qquad  
s^3.  
  
This matters because third-point obstruction estimates depend on which conditional residual directions remain random.  
  
The axis-cubic fluctuation is removed. Transverse cubic fluctuations remain.  
  
Thus the conditional field near the fold has a highly anisotropic residual structure.  
  
⸻  
  
**113. Conditional Residual Scaling**  
  
Let  
  
y=x_0+\tau z  
  
near the midpoint or saddle.  
  
The pair-conditioned residual has scale depending on the direction.  
  
On the pair axis:  
  
\mathrm{sd}(g_r(y))=\Theta(\tau^4).  
  
For gradient components, the corresponding residual scales are typically:  
  
\mathrm{sd}(\partial_t g_r(y))=\Theta(\tau^3),  
  
\mathrm{sd}(\partial_s g_r(y))=\Theta(\tau^3),  
  
depending on the exact chart and component.  
  
The mean-to-standard-deviation ratio in the annulus becomes large because the deterministic pinned mean dominates residual fluctuations.  
  
This is the mechanism that suppresses annulus critical points.  
  
⸻  
  
**114. Hessian Variables Under Pair Conditioning**  
  
At the saddle:  
  
H_S=  
\begin{pmatrix}  
f_{tt}(x_S)&f_{ts}(x_S)\\  
f_{ts}(x_S)&f_{ss}(x_S)  
\end{pmatrix}.  
  
At the maximum:  
  
H_M=  
\begin{pmatrix}  
f_{tt}(x_M)&f_{ts}(x_M)\\  
f_{ts}(x_M)&f_{ss}(x_M)  
\end{pmatrix}.  
  
Fold pinning gives longitudinal curvature:  
  
f_{tt}(x_S)=+\kappa r+O_P(r^2),  
  
f_{tt}(x_M)=-\kappa r+O_P(r^2).  
  
Mixed derivatives scale as:  
  
f_{ts}(x_S)=r\nu_S+O_P(r^2),  
  
f_{ts}(x_M)=r\nu_M+O_P(r^2),  
  
with  
  
\nu_S,\nu_M=O_P(1).  
  
The transverse saddle curvature is written:  
  
f_{ss}(x_S)=-\mu.  
  
Smoothness gives:  
  
f_{ss}(x_M)=f_{ss}(x_S)+r\eta+O_P(r^2),  
  
with  
  
\eta=O_P(1).  
  
Signs may change under orientation convention, but the scaling is invariant.  
  
⸻  
  
**115. Determinants Under Pair Conditioning**  
  
At the saddle:  
  
\det H_S  
=  
f_{tt}(x_S)f_{ss}(x_S)-f_{ts}(x_S)^2.  
  
Using the fold scalings:  
  
\det H_S  
=  
(\kappa r)(-\mu)-(r\nu_S)^2+O_P(r^3).  
  
Thus:  
  
\det H_S  
=  
-\kappa r  
\left(  
\mu+\frac{r\nu_S^2}{\kappa}  
\right)  
+  
O_P(r^3).  
  
At the maximum:  
  
\det H_M  
=  
f_{tt}(x_M)f_{ss}(x_M)-f_{ts}(x_M)^2.  
  
Using  
  
f_{tt}(x_M)=-\kappa r,  
  
and the transverse-curvature relation, one obtains:  
  
\det H_M  
=  
+\kappa r  
\left(  
\widetilde{\mu}_M  
\right)  
+  
O_P(r^3),  
  
with \widetilde{\mu}_M** defined below.**  
  
⸻  
  
**116. Corrected Transverse Curvatures**  
  
Define the corrected saddle-side transverse curvature:  
  
\widetilde{\mu}_S  
=  
\mu+\frac{r\nu_S^2}{\kappa}.  
  
Then:  
  
\det H_S  
=  
-\kappa r\,\widetilde{\mu}_S  
+  
O_P(r^3).  
  
Define the corrected maximum-side transverse curvature:  
  
\widetilde{\mu}_M  
=  
\mu+r\left(\eta-\frac{\nu_M^2}{\kappa}\right)+O_P(r^2).  
  
Then:  
  
\det H_M  
=  
+\kappa r\,\widetilde{\mu}_M  
+  
O_P(r^3).  
  
The two corrected curvatures are coupled:  
  
\widetilde{\mu}_M  
=  
\widetilde{\mu}_S  
+  
r\widetilde{\eta}  
+  
O_P(r^2),  
  
where  
  
\widetilde{\eta}  
=  
\eta-\frac{\nu_M^2}{\kappa}+\frac{\nu_S^2}{\kappa}.  
  
This relation is central to the Flat-Saddle Tail correction.  
  
⸻  
  
**117. Maximum and Saddle Index Conditions**  
  
For the saddle:  
  
\det H_S<0.  
  
Given  
  
f_{tt}(x_S)\sim+\kappa r>0,  
  
this requires transverse curvature negative enough, equivalently:  
  
\widetilde{\mu}_S>0  
  
to leading order.  
  
For the maximum:  
  
H_M<0.  
  
Given  
  
f_{tt}(x_M)\sim-\kappa r<0,  
  
this requires:  
  
\det H_M>0,  
  
equivalently:  
  
\widetilde{\mu}_M>0  
  
to leading order.  
  
Thus the fold maximum-saddle event imposes:  
  
\widetilde{\mu}_S>0,  
\qquad  
\widetilde{\mu}_M>0,  
  
up to lower-order corrections.  
  
This is why the flat-saddle tail concerns small positive \widetilde{\mu}_S**.**  
  
⸻  
  
**118. Determinant Weight in Corrected Variables**  
  
The Kac–Rice determinant product becomes:  
  
|\det H_S|\,|\det H_M|  
\approx  
\kappa^2r^2  
\,  
\widetilde{\mu}_S  
\widetilde{\mu}_M  
  
on the valid maximum-saddle region.  
  
Using the coupling:  
  
\widetilde{\mu}_M  
=  
\widetilde{\mu}_S+r\widetilde{\eta}+O_P(r^2).  
  
For generic configurations:  
  
\widetilde{\mu}_S\asymp1,  
\qquad  
\widetilde{\mu}_M\asymp1,  
  
so:  
  
|\det H_S|\,|\det H_M|  
\asymp r^2.  
  
For flat-saddle configurations:  
  
\widetilde{\mu}_S\asymp r^2.  
  
Then:  
  
|\det H_S|\asymp r^3.  
  
Meanwhile smoothness coupling gives typically:  
  
\widetilde{\mu}_M\asymp r,  
  
so:  
  
|\det H_M|\asymp r^2.  
  
Therefore:  
  
|\det H_S|\,|\det H_M|\asymp r^5.  
  
This is the corrected power ledger.  
  
⸻  
  
**119. The Earlier Determinant Error**  
  
A superseded argument treated the maximum-side determinant as if it remained of generic size  
  
|\det H_M|\asymp r.  
  
Together with  
  
|\det H_S|\asymp r^3,  
  
this gave:  
  
|\det H_S|\,|\det H_M|\asymp r^4.  
  
That was incorrect.  
  
The two transverse curvatures are not independent. Smoothness couples them across a distance r**. When the saddle-side corrected curvature is forced down to **r^2**, the maximum-side corrected curvature is also depleted to scale **r**, not order one.**  
  
Thus the correct determinant product is:  
  
r^5,  
  
not  
  
r^4.  
  
This correction preserves the theorem because both powers vanish.  
  
⸻  
  
**120. Generic Pair Versus Palm-Averaged Pair**  
  
A major conceptual distinction:  
  
At fixed generic pair data, third-point criticality in the inner window may be exponentially suppressed.  
  
But the Palm law averages over pair data with determinant weighting.  
  
Rare near-flat saddle configurations have much weaker suppression and dominate the averaged expectation.  
  
Thus:  
  
\text{fixed generic pair}  
\Rightarrow  
\text{exponential suppression},  
  
while  
  
\text{Palm average}  
\Rightarrow  
\text{polynomial suppression}.  
  
This is the central lesson of the Flat-Saddle Tail correction.  
  
⸻  
  
**121. Pair-Palm Density Normalization**  
  
The denominator in the pair-Palm law is the marked maximum-saddle pair intensity.  
  
Schematically:  
  
Z_{\mathrm{pair}}(r,b,\ell)  
=  
p_{\Phi_r}(b,b-\ell,0,0,0,0)  
\,  
\mathbb{E}  
[  
|\det H_M||\det H_S|  
1_{\max}1_{\mathrm{saddle}}1_{\mathrm{adj}}  
\mid  
\Phi_r  
].  
  
After transformation to the corrected divided-difference frame, the Gaussian density contribution has a finite nonzero limit after the appropriate Jacobian is accounted for.  
  
The determinant expectation contributes the leading pair-intensity scale.  
  
The architecture’s pair-density claim is that the full candidate fold-pair intensity remains asymptotically finite and positive in the radial separation variable used for the near-diagonal law.  
  
This is the “contact neutrality” input:  
  
\rho_{\mathrm{pair}}(r;b)\to\rho_0(b)\in(0,\infty).  
  
⸻  
  
**122. Pair Intensity and the Fold Modulus**  
  
The fold modulus is:  
  
\kappa=\frac{6\ell}{r^3}.  
  
In the pair-Palm description, \kappa** is not an external constant. It is the normalized height gap.**  
  
The near-diagonal density integrates over \kappa** in a compact positive window or over its limiting density.**  
  
The constant in Theorem B has schematic form:  
  
C_*  
=  
\int_B  
\int_0^\infty  
\mathcal{J}(b,\kappa)  
q_0(b,\kappa)  
\,d\kappa\,db,  
  
where:  
  
q_0(b,\kappa)=1.  
  
The Jacobian factor from  
  
\ell=\frac{\kappa r^3}{6}  
  
supplies the \ell^{-1/3}** singularity.**  
  
⸻  
  
**123. Conditional Law of the Stiff Curvature**  
  
The transverse curvature variable  
  
\mu=-f_{ss}(x_S)  
  
is not pinned to a deterministic value by the pair constraints.  
  
It remains an O(1)** Gaussian-type variable after conditioning, with a nondegenerate density.**  
  
The maximum-saddle index event restricts it to the positive admissible region after corrected terms.  
  
Near  
  
\widetilde{\mu}_S=0,  
  
the conditional density remains bounded rather than exponentially small.  
  
This is why the flat-saddle tail contributes polynomially.  
  
⸻  
  
**124. Adjacency Indicator**  
  
The adjacency indicator  
  
1_{\mathrm{adj}}  
  
asserts that the candidate maximum and saddle are connected by a gradient separatrix.  
  
The pair-Palm law used in the theorem should include this indicator for exact selection probability.  
  
However, many upper bounds use:  
  
1_{\mathrm{adj}}\le1.  
  
This is legitimate when proving obstruction probabilities vanish.  
  
It is not legitimate for computing exact constants unless one proves the adjacency probability has a finite nonzero limit and incorporates it.  
  
Therefore the final monograph should separate:  
  
1. upper-bound Palm law without adjacency,  
2. exact candidate-pair intensity with adjacency,  
3. selection probability conditional on adjacency.  
  
⸻  
  
**125. Critical-Pair Palm Law — Canonical Definition**  
  
Let  
  
\mathcal{F}_{r,b,\kappa}  
  
denote the event/conditioning kernel:  
  
f(x_M)=b,  
  
f(x_S)=b-\frac{\kappa r^3}{6},  
  
\nabla f(x_M)=0,  
  
\nabla f(x_S)=0,  
  
with Hessian signs maximum/saddle and adjacency imposed through Kac–Rice weighting.  
  
For a test functional F**, define:**  
  
\mathbb{E}_{r,b,\kappa}^{\mathrm{MS}}[F]  
=  
\frac{  
\mathbb{E}\left[  
F  
|\det H_M||\det H_S|  
1_{\max}1_{\mathrm{saddle}}1_{\mathrm{adj}}  
\mid  
\Phi_r=(b,b-\kappa r^3/6,0,0,0,0)  
\right]  
}{  
\mathbb{E}\left[  
|\det H_M||\det H_S|  
1_{\max}1_{\mathrm{saddle}}1_{\mathrm{adj}}  
\mid  
\Phi_r=(b,b-\kappa r^3/6,0,0,0,0)  
\right]  
}.  
  
Then:  
  
q(r,b,\kappa)  
=  
\mathbb{P}_{r,b,\kappa}^{\mathrm{MS}}  
(D(x_M)=x_S).  
  
The theorem says:  
  
\sup_{b\in B,\kappa\in K}  
|q(r,b,\kappa)-1|  
\to0  
  
for compact admissible  
  
B\subset\mathbb{R},  
\qquad  
K\subset(0,\infty).  
  
The \kappa**-uniform version is stronger and cleaner than an **\ell**-only statement.**  
  
⸻  
  
**126. Mean Field Under Pair Conditioning**  
  
The conditional mean is:  
  
m_r(y)=  
\mathbb{E}[f(y)\mid\Phi_r=(b,b-\ell,0,0,0,0)].  
  
In the corrected frame, it has a stable limit as r\to0**.**  
  
Near the pair, it decomposes into:  
  
m_r(t,s)  
=  
b  
+  
\kappa\left(\frac{t^3}{3}-\frac{r^2}{4}t\right)  
-  
\frac{\mu}{2}s^2  
+  
\text{mixed pinned corrections}  
+  
\text{higher-order terms}.  
  
Here \mu** may be treated either as a Hessian mark or as a residual random coordinate depending on the conditioning level.**  
  
The main deterministic fold comes from the (t^3,t)** terms.**  
  
The transverse confinement comes from -\mu s^2/2**.**  
  
⸻  
  
**127. Hessian-Extended Palm Conditioning**  
  
Some estimates condition not only on values and gradients but also on parts of the Hessian.  
  
For instance, one may condition on:  
  
\mu,\nu_S,\nu_M,\eta  
  
or related Hessian/third-jet variables.  
  
This creates a refined Palm base.  
  
The workflow is:  
  
1. condition on the six linear pair constraints,  
2. expose Hessian variables,  
3. weight by determinant product and index indicators,  
4. estimate third-point probabilities conditional on this base,  
5. integrate over the Palm-weighted base.  
  
The flat-saddle correction arises precisely at step 5.  
  
At step 4, generic bases yield exponential suppression.  
  
At step 5, near-flat bases dominate the averaged integral.  
  
⸻  
  
**128. Gaussian Regression Identity as Projection**  
  
The conditional variance formula can be written:  
  
\mathrm{Var}(Y\mid X)  
=  
\inf_a  
\mathrm{Var}(Y-a\cdot X).  
  
This means conditional variance is squared distance in the Gaussian Hilbert space from Y** to the span of the conditioned variables.**  
  
The divided-difference frame is therefore not just a computational trick. It identifies the limiting span of the conditioning event.  
  
The GCJA theorem generalizes this principle:  
  
collapsing observation designs converge to visible polynomial jet spaces, and residual variances are distances to those spaces.  
  
⸻  
  
**129. Why the Corrected Frame Matters**  
  
The correction in  
  
V^-_{\mathrm{corr}}  
  
prevents a false divergence.  
  
Without subtracting  
  
r\frac{f_t(x_S)+f_t(x_M)}2,  
  
the value-difference coordinate contains a lower-order gradient contribution.  
  
On the constraint surface, the gradients vanish, so earlier on-surface calculations could accidentally produce correct answers.  
  
But conditional density theory must work off the surface.  
  
Therefore the corrected frame is mandatory for a rigorous pair-Palm representation.  
  
This is an important audit entry: the architecture found and repaired a subtle coordinate error before it corrupted the proof.  
  
⸻  
  
**130. Pair-Palm Novelty Boundary**  
  
The ingredients are classical:  
  
* Gaussian regression,  
* Kac–Rice,  
* Schur complements,  
* Palm conditioning,  
* determinant weights.  
  
The new part is the uniform degenerating pair version:  
  
r\to0  
  
with two critical points collapsing, six simultaneous constraints, height gap O(r^3)**, and determinant-weighted Hessian integration.**  
  
Existing fixed-point Kac–Rice theory does not automatically supply this.  
  
The corrected divided-difference frame is the mechanism that makes the degenerating limit uniform.  
  
⸻  
  
**131. Interface With Three-Point Analysis**  
  
The pair-Palm law conditions on two critical points.  
  
The inner-zone obstruction involves a third critical point  
  
y.  
  
Thus the three-point Kac–Rice object is:  
  
(x_M,x_S,y).  
  
The third point approaches x_S** at scale:**  
  
|y-x_S|\sim r^2.  
  
The conditional distribution of  
  
\nabla f(y)  
  
given the pair constraints is governed by the pair-Palm residual covariance.  
  
The probability that  
  
\nabla f(y)=0  
  
depends on:  
  
1. the conditional mean of \nabla f(y),  
2. the conditional covariance of \nabla f(y),  
3. the determinant expectation at y,  
4. the pair determinant weight,  
5. integration over y,  
6. integration over the Palm base.  
  
This is where GCJA and the Flat-Saddle Tail Lemma connect.  
  
⸻  
  
**132. Interface With Annulus Estimates**  
  
In the annulus, the third point satisfies:  
  
r^2\ll |y-x_S|\ll 1  
  
or comparable intermediate bounds.  
  
The pair-conditioned mean of \nabla f(y)** is typically large relative to its conditional standard deviation.**  
  
This produces suppression of third critical points.  
  
The pinning theorem’s key contribution is that on-axis residual fluctuations are reduced to order \tau^4**, improving the mean-to-noise ratio and killing the old annulus floor.**  
  
The archive records that this discharges an earlier failure mode.  
  
⸻  
  
**133. Interface With Far-Zone Estimates**  
  
For y** bounded away from the pair, the pair conditioning remains nondegenerate and ordinary Kac–Rice estimates apply.**  
  
The only small factor is the height band width:  
  
\ell=O(r^3).  
  
Thus far-zone interference probabilities are typically:  
  
O(r^3)  
  
on fixed torus volume.  
  
The pair-Palm layer ensures the constants in this estimate remain uniform under the fold-pair conditioning.  
  
⸻  
  
**134. Pair-Palm Layer Failure Modes**  
  
This layer would fail if:  
  
1. the corrected divided-difference frame were not uniformly invertible,  
2. the limiting six-jet covariance were degenerate,  
3. determinant weighting introduced a non-integrable singularity,  
4. the fold modulus \kappa had mass at zero in the theorem window,  
5. Hessian signs destroyed the claimed finite positive pair intensity,  
6. adjacency probability vanished in the near-fold limit,  
7. Palm averaging amplified inner obstructions to order one.  
  
The archive’s current state:  
  
* uniform frame invertibility is handled by the Pinning Lemma package,  
* determinant weighting is handled at upper-bound level,  
* flat-saddle amplification is corrected to r^5, still vanishing,  
* exact adjacency constants remain a manuscript-cleanup issue,  
* exact lower bounds are open but not needed for q_0=1.  
  
⸻  
  
**135. Canonical Pair-Palm Lemma Package**  
  
The final monograph should package this layer as the following lemmas.  
  
**Lemma P1 — Corrected Frame Invertibility**  
  
The corrected pair frame  
  
\Psi_r^{\mathrm{corr}}  
  
has covariance converging to a nondegenerate six-jet covariance matrix. Therefore its covariance eigenvalues are uniformly bounded above and below for small r**.**  
  
**Lemma P2 — Pair-Palm Regression Kernel**  
  
Conditioned on the pair constraints, the field decomposes as:  
  
f=m_r+g_r,  
  
where m_r** is the explicit Gaussian regression mean and **g_r** is a centered residual independent of the corrected pair frame.**  
  
**Lemma P3 — Cubic Pinning**  
  
On the fold-pair constraint surface,  
  
c_{30}=2\kappa+O_P(r^2),  
  
and the on-axis residual standard deviation is \Theta(\tau^4)**.**  
  
**Corollary P4 — Working Scaling Consequences**  
  
The conditional mean/residual ratios in the annulus and inner-transition regions have the powers required by Lemmas A'**, **B1**, and **I**.**  
  
**Proposition P5 — Rough-Smoothness Variant**  
  
If \Lambda_8=\infty** but effective spectral smoothness is **\nu\in(3,4)**, the same architecture survives with degraded exponents.**  
  
Status:  
  
\text{secondary / not part of main theorem}.  
  
⸻  
  
**136. Main Output of Part 4**  
  
The Gaussian critical-pair layer proves that the fold-pair Palm law is well-defined uniformly as r\to0**, and that under this law the field is forced into a cubic local normal form with controlled residuals.**  
  
This is the probabilistic engine of the program.  
  
## The next layer must explain the GCJA theorem, because it is the general mechanism behind the conditional residual powers used here and in the three-point analysis.  
  
**Unified Master Source — Part 5**  
  
**Graded Conditional-Jet Asymptotics: The Filtration Engine Behind the Program**  
  
⸻  
  
**137. Purpose of the GCJA Layer**  
  
The Graded Conditional-Jet Asymptotics theorem, abbreviated  
  
\mathrm{GCJA},  
  
is the structural engine behind the conditional covariance powers used throughout the C006 / q₀ program.  
  
Before this theorem, many power laws in the archive appeared as separate symbolic facts:  
  
* covariance rank collapse,  
* slaving of certain derivatives,  
* disappearance of certain residual directions,  
* parity restrictions,  
* anisotropic covariance survival,  
* \Sigma_2(0)-type limiting covariance matrices,  
* r^\gamma-family scaling transitions,  
* design-surgery behavior,  
* suppression of annulus critical points,  
* inner-zone two-scale covariance degeneration.  
  
The GCJA theorem unifies these facts.  
  
It says they are not accidents of Bargmann–Fock covariance, isotropy, or symbolic algebra. They follow from a deterministic interpolation filtration combined with Gaussian Hilbert-space projection.  
  
The theorem’s core message is:  
  
\text{collapsing observations reveal a finite polynomial jet space;}  
  
\text{unseen polynomial jets determine the residual covariance;}  
  
\text{their graded order determines every asymptotic power of } r.  
  
This is the reason the q₀ program can be upgraded from a collection of local computations to a general theorem architecture.  
  
⸻  
  
**138. The Problem GCJA Solves**  
  
Suppose a Gaussian field is observed at several nearby points:  
  
(c_i r,0),  
\qquad  
i=1,\ldots,p,  
  
and possibly several derivatives are observed at each point.  
  
As  
  
r\to0,  
  
these observation points collapse.  
  
Naively, the covariance matrix of the observations becomes singular.  
  
A standard Gaussian conditioning formula  
  
\Sigma_{Y|X}  
=  
\Sigma_Y-\Sigma_{YX}\Sigma_X^{-1}\Sigma_{XY}  
  
becomes hard to interpret because  
  
\Sigma_X^{-1}  
  
develops large entries.  
  
The GCJA theorem resolves this by replacing raw point observations with their limiting divided-difference content.  
  
The collapsed observations are equivalent, asymptotically, to observing a finite list of derivatives at the collision point.  
  
Those observed derivatives span a visible polynomial jet space.  
  
Everything not in that visible space remains random.  
  
The leading residual covariance is the covariance of the lowest-order unseen jets.  
  
⸻  
  
**139. Gaussian Hilbert-Space Viewpoint**  
  
Let  
  
f  
  
be a centered stationary Gaussian field with spectral measure  
  
\rho.  
  
A finite linear functional  
  
\Lambda  
=  
\sum_{\alpha,x}c_{\alpha,x}\partial^\alpha f(x)  
  
corresponds to the spectral representative  
  
\lambda(k)  
=  
\sum_{\alpha,x}c_{\alpha,x}(ik)^\alpha e^{i\langle k,x\rangle}.  
  
The covariance is the spectral inner product:  
  
\mathbb{E}[\Lambda\Lambda']  
=  
\int \lambda(k)\overline{\lambda'(k)}\,d\rho(k).  
  
Thus all finite Gaussian conditioning can be interpreted in the Hilbert space  
  
L^2(\rho).  
  
Conditioning on observations means projecting onto their span.  
  
Residual covariance means squared distance to that span.  
  
This gives the exact identity:  
  
\mathrm{Var}(Y\mid X)  
=  
\|Q_X y\|_{L^2(\rho)}^2,  
  
where:  
  
* y is the spectral representative of Y,  
* Q_X is orthogonal projection onto the orthogonal complement of the span of X.  
  
GCJA is the asymptotic version of this projection identity when the observed span collapses.  
  
⸻  
  
**140. Standing GCJA Hypotheses**  
  
The theorem uses three core hypotheses.  
  
**140.1 **H_f**: stationary Gaussian field**  
  
The field is centered, stationary, Gaussian on \mathbb{R}^2** or **\mathbb{T}_L^2**, with covariance**  
  
C(x)  
=  
\int e^{i\langle k,x\rangle}\,d\rho(k).  
  
The spectral measure is finite and symmetric:  
  
\rho(A)=\rho(-A).  
  
**140.2 **H_{\mathrm{mom}}**: moment condition**  
  
For the chosen observation design D**, the spectral measure has enough finite moments:**  
  
\lambda_{2m}  
=  
\int |k|^{2m}\,d\rho(k)  
<  
\infty  
  
for all  
  
2m\le M(D).  
  
The theorem package defines  
  
M(D)  
=  
2\max_k(q_k+k)+2.  
  
For the full design  
  
(p,j)=(2,2),  
  
one obtains  
  
M=14.  
  
The leading asymptotic conclusions need moments only through order 12**; the extra two derivatives provide an error margin.**  
  
**140.3 **H_{\mathrm{nd}}**: finite polynomial nondegeneracy**  
  
Let  
  
N=\max_k(q_k+k).  
  
The spectral measure is N**-nondegenerate if no nonzero polynomial of degree**  
  
\le 2N  
  
vanishes \rho**-almost everywhere.**  
  
A sufficient continuum condition is:  
  
\rho \text{ charges an open set}.  
  
On the torus, the analogous condition is supplied by the finite-jet nondegeneracy hypothesis H2'**.**  
  
This assumption is what prevents hidden spectral collapse.  
  
⸻  
  
**141. Staircase Observation Design**  
  
Fix:  
  
p\ge2  
  
distinct real nodes  
  
c_1<c_2<\cdots<c_p.  
  
Fix a staircase of derivative orders:  
  
d=(d_0\ge d_1\ge\cdots\ge d_J\ge0).  
  
At scale r**, the design observes**  
  
O_r  
=  
\left\{  
\partial_t^m\partial_s^k f(c_i r,0):  
k\le J,\,  
m\le d_k,\,  
i=1,\ldots,p  
\right\}.  
  
For each transverse level k**, there are**  
  
p(d_k+1)  
  
longitudinal Hermite data.  
  
Define:  
  
q_k=p(d_k+1),  
\qquad k\le J.  
  
For  
  
k>J,  
  
define:  
  
q_k=0.  
  
The observation span is:  
  
S_r=\mathrm{span}(O_r)\subset L^2(\rho).  
  
Its dimension is:  
  
n_D=\sum_k p(d_k+1).  
  
The full design D(p,j)** is the staircase:**  
  
d_k=j-k,  
\qquad  
J=j.  
  
⸻  
  
**142. Visible Space**  
  
For monomial functionals, define  
  
\phi_{m,k}(K)  
=  
(iK_t)^m(iK_s)^k.  
  
The visible space is  
  
V(D)  
=  
\mathrm{span}  
\left\{  
\phi_{m,k}:  
k\le J,\,  
m\le q_k-1  
\right\}.  
  
Thus, at transverse level k**, the collapsing observations reveal all longitudinal derivatives up to order**  
  
q_k-1.  
  
The number of visible monomials is:  
  
\#V(D)=\sum_k q_k=n_D.  
  
Therefore there is no wasted observation rank.  
  
Every observed degree of freedom becomes exactly one visible polynomial jet direction.  
  
This is the algebraic reason the limiting covariance is finite and structured rather than chaotic.  
  
⸻  
  
**143. Unseen Space**  
  
A monomial  
  
\psi_\sigma=\phi_{m,k}  
  
with  
  
\sigma=(m,k)  
  
is unseen if:  
  
k>J  
  
or  
  
m\ge q_k.  
  
The unseen set is denoted:  
  
U(D).  
  
The residual covariance after conditioning is controlled by the projection of these unseen monomials onto  
  
V(D)^\perp.  
  
That is:  
  
Q_V\psi_\sigma,  
  
where  
  
Q_V  
  
is orthogonal projection onto the orthogonal complement of V(D)** in **L^2(\rho)**.**  
  
The unseen monomials are not assumed orthogonal. Their conditional covariance is the Gram matrix:  
  
G_{U|V}(\sigma,\tau)  
=  
\langle Q_V\psi_\sigma,Q_V\psi_\tau\rangle_{L^2(\rho)}.  
  
⸻  
  
**144. Full Design as Weighted Ball**  
  
For the full design  
  
D(p,j),  
  
where  
  
d_k=j-k,  
  
the visible monomial set is:  
  
V  
=  
\{(m,k):m+pk\le p(j+1)-1\}.  
  
This is a weighted ball with weights:  
  
m+pk.  
  
The minimal unseen stratum has weighted degree:  
  
p(j+1).  
  
This is the filtration law.  
  
For the important case  
  
(p,j)=(2,2),  
  
the visible condition is:  
  
m+2k\le 5.  
  
The minimal unseen stratum has weighted degree:  
  
6.  
  
This single fact explains many of the archive’s earlier symbolic covariance collapses.  
  
⸻  
  
**145. Hermite Interpolation Mechanism**  
  
The deterministic core of GCJA is Hermite interpolation on a line.  
  
For one transverse level k**, the observations are:**  
  
\partial_t^m\partial_s^k f(c_i r,0),  
\qquad  
m\le d_k,\quad i=1,\ldots,p.  
  
These are the values of a one-dimensional function and its derivatives at p** nodes.**  
  
Hermite interpolation says these data determine the Taylor coefficients at the origin up to longitudinal order:  
  
q_k-1=p(d_k+1)-1.  
  
Thus, at transverse level k**, the collapsing observations reveal:**  
  
\partial_t^m\partial_s^k f(0,0),  
\qquad  
m=0,\ldots,q_k-1.  
  
The first unrevealed longitudinal derivative at level k** is:**  
  
\partial_t^{q_k}\partial_s^k f(0,0).  
  
This is the start of the unseen stratum at that transverse level.  
  
⸻  
  
**146. Hermite Unisolvence Lemma**  
  
The deterministic lemma is:  
  
Given distinct nodes  
  
x_1,\ldots,x_p  
  
and jet order d**, for any data**  
  
g_{i,\ell},  
\qquad  
0\le \ell\le d,  
  
there is a unique polynomial H** of degree**  
  
\le p(d+1)-1  
  
such that  
  
H^{(\ell)}(x_i)=g_{i,\ell}  
  
for all i,\ell**.**  
  
Proof mechanism:  
  
If a polynomial of degree at most  
  
p(d+1)-1  
  
has zeros of multiplicity  
  
d+1  
  
at each of p** distinct nodes, then it has at least**  
  
p(d+1)  
  
zeros counted with multiplicity.  
  
That is impossible unless the polynomial is identically zero.  
  
Therefore the interpolation map is injective between spaces of equal finite dimension, hence bijective.  
  
This elementary fact is the deterministic foundation of the filtration theorem.  
  
⸻  
  
**147. Divided-Difference Rate Lemma**  
  
The next deterministic lemma quantifies convergence.  
  
At nodes  
  
c_i r,  
  
with jet order d**, let**  
  
q=p(d+1).  
  
For each  
  
m\le q-1,  
  
there exist weights  
  
w_{i,\ell}(r)  
=  
r^{m-\ell}\widehat{w}_{i,\ell}  
  
such that:  
  
\sum_{i,\ell}w_{i,\ell}(r)g^{(\ell)}(c_i r)  
  
approximates  
  
g^{(m)}(0)  
  
with error  
  
O(r^{q-m})  
  
controlled by  
  
\sup |g^{(q)}|.  
  
The scaling  
  
r^{m-\ell}  
  
is forced by homogeneity.  
  
This lemma is the rigorous replacement for informal “take divided differences” reasoning.  
  
⸻  
  
**148. Evaluation Near a Terminal Node**  
  
A crucial refinement occurs when the evaluation point lies near the last node:  
  
x^*=c_p r+r^\gamma z.  
  
Then the Hermite interpolation error factorizes.  
  
If the interpolant matches derivatives up to order d** at **c_p r**, then the error contains powers of**  
  
|x^*-c_p r|.  
  
Specifically, for derivative order a\le d**, the error has factors:**  
  
r^{(p-1)(d+1)}  
  
from the other p-1** nodes and**  
  
r^{\gamma(d+1-a)}  
  
from proximity to the terminal node.  
  
This gives the key suppression factor:  
  
r^{(p-1)(d+1)+\gamma(d+1-a)}.  
  
This is one of the central formulas behind annulus and inner-zone exponents.  
  
It explains why a third point near one of the conditioned critical points sees stronger residual suppression than a generic point at the same global scale.  
  
⸻  
  
**149. Projection Perturbation Lemma**  
  
Let  
  
e_i(r)\to e_i(0)  
  
in a Hilbert space H**, and suppose the limiting Gram matrix**  
  
G(0)=(\langle e_i(0),e_j(0)\rangle)  
  
is positive definite.  
  
Let P_r** project onto**  
  
\mathrm{span}\{e_i(r)\}.  
  
Then:  
  
P_r\psi\to P_0\psi  
  
for every fixed \psi**.**  
  
Moreover, the convergence rate is controlled by:  
  
\|G(r)-G(0)\|  
+  
\sum_i\|e_i(r)-e_i(0)\|.  
  
This proves that the collapsing observation span S_r** converges to the visible space **V(D)** in the appropriate finite-dimensional projection sense.**  
  
Thus conditional covariance under S_r** converges to conditional covariance under **V(D)**.**  
  
⸻  
  
**150. Evaluation Scaling**  
  
The theorem evaluates field derivatives at points of the form:  
  
y  
=  
(c_p r+r^\gamma z_t,\ r^\gamma z_s),  
  
where:  
  
\gamma>0,  
  
and z** lies in a compact set.**  
  
The parameter \gamma** identifies the spatial regime:**  
  
* \gamma=1: O(r) scale,  
* \gamma=2: O(r^2) inner scale,  
* 1<\gamma<2: intermediate annulus scales,  
* \gamma<1: farther but still collapsing scales.  
  
For an observable  
  
\phi_y^a  
=  
\partial_t^{a_t}\partial_s^{a_s}f(y),  
\qquad |a|\le1,  
  
one expands its spectral representative around the visible space.  
  
The residual is a sum over unseen monomials.  
  
Each unseen monomial carries a deterministic power of r**.**  
  
⸻  
  
**151. Ladder Orders**  
  
For each observable a** and unseen monomial**  
  
\sigma=(m,k),  
  
define the ladder order:  
  
L_a(\sigma;\gamma).  
  
This is the exact exponent of r** multiplying the unseen monomial contribution in the residual expansion.**  
  
The leading order for observable a** is:**  
  
\ell_a  
=  
\min_{\sigma\in U(D)}  
L_a(\sigma;\gamma).  
  
The leading stratum is:  
  
\Sigma_a  
=  
\{\sigma\in U(D):L_a(\sigma;\gamma)=\ell_a\}.  
  
The exponent gap is:  
  
\delta(D,\gamma)  
=  
\min\{L_a(\sigma;\gamma)-\ell_a:  
\sigma\notin\Sigma_a\},  
  
with ties handled by including all tied monomials in \Sigma_a**.**  
  
This produces a finite ladder of powers.  
  
The residual covariance is dominated by the leading stratum.  
  
⸻  
  
**152. Multiplier Map**  
  
Each leading unseen monomial contributes with a deterministic coefficient polynomial in z**.**  
  
This defines the multiplier map:  
  
m^a:\Sigma_a\to\mathbb{R}[z].  
  
For each leading monomial \sigma**, the corresponding residual term is:**  
  
r^{\ell_a}m_\sigma^a(z)Q_V\psi_\sigma.  
  
Thus the rescaled residual observable satisfies:  
  
r^{-\ell_a}  
Q_{S_r}\phi_y^a  
\to  
\sum_{\sigma\in\Sigma_a}  
m_\sigma^a(z)Q_V\psi_\sigma.  
  
The deterministic multiplier map records where the third point is located inside the blown-up chart.  
  
⸻  
  
**153. GCJA Theorem — Canonical Statement**  
  
Let f** be a centered stationary Gaussian field satisfying **H_f**, **H_{\mathrm{mom}}**, and **H_{\mathrm{nd}}**. Let **D(p;d)** be a staircase observation design and let **S_r** be its collapsing observation span.**  
  
For observables  
  
\phi_y^a=\partial^a f(y),  
\qquad |a|\le1,  
  
evaluated at  
  
y=(c_pr+r^\gamma z_t,\ r^\gamma z_s),  
  
with z** in a fixed compact set, define:**  
  
\ell_a=\min_\sigma L_a(\sigma;\gamma),  
  
and leading stratum  
  
\Sigma_a.  
  
Then:  
  
r^{-\ell_a}Q_{S_r}\phi_y^a  
\to  
\sum_{\sigma\in\Sigma_a}  
m_\sigma^a(z)Q_V\psi_\sigma  
  
in L^2(\rho)**, uniformly for **z** in compact sets.**  
  
Consequently, the conditional covariance satisfies:  
  
r^{-(\ell_a+\ell_b)}  
\mathrm{Cov}  
\left(  
\phi_y^a,\phi_y^b  
\mid S_r  
\right)  
\to  
\Sigma_\infty^{ab}(z),  
  
where  
  
\Sigma_\infty^{ab}(z)  
=  
\sum_{\sigma,\tau}  
m_\sigma^a(z)m_\tau^b(z)  
G_{U|V}(\sigma,\tau).  
  
The error is  
  
O(r^\delta)  
  
away from ladder-tie degeneracies, with tied strata included where necessary.  
  
This is the core theorem.  
  
⸻  
  
**154. Positivity of the Limit**  
  
The limiting covariance matrix  
  
\Sigma_\infty(z)  
  
is positive semidefinite automatically because it is a Gram pushforward.  
  
It is positive definite if the leading multiplier combination does not land inside the visible space.  
  
That is, for any nonzero coefficient vector a**,**  
  
\sum_{\sigma\in\Sigma}c_\sigma(z)Q_V\psi_\sigma\ne0  
  
in L^2(\rho)**.**  
  
Under H_{\mathrm{nd}}**, this reduces to a polynomial nonvanishing condition.**  
  
For the specific designs used in the q₀ program, positivity is checked by showing the relevant leading monomials cannot be swallowed by the visible space and do not cancel identically.  
  
This discharges the earlier “positivity obligation.”  
  
⸻  
  
**155. Rank Collapse and Slaving**  
  
The GCJA theorem explains rank collapse.  
  
If an observable’s leading stratum is empty at a certain scale or its leading residual is determined by another observable’s leading residual, the limiting covariance matrix has reduced rank.  
  
This is not a pathology. It is slaving.  
  
A derivative is “slaved” when the collapsing observations determine it to a higher order than expected, leaving only a dependent residual direction.  
  
Rank collapse in earlier computations was therefore not a numerical accident.  
  
It was a visible-space consequence.  
  
The theorem predicts exactly which components remain random and which are slaved.  
  
⸻  
  
**156. Parity**  
  
Stationary Gaussian fields with symmetric spectral measure have parity constraints.  
  
Monomials with odd/even combinations may be orthogonal under symmetric \rho**, depending on the coordinate reflection structure.**  
  
GCJA does not require isotropy, but it naturally records parity through the spectral inner product:  
  
\langle \phi_{m,k},\phi_{m',k'}\rangle  
=  
\int  
(iK_t)^m(iK_s)^k  
\overline{(iK_t)^{m'}(iK_s)^{k'}}  
\,d\rho(K).  
  
If the integrand is odd under a symmetry of \rho**, the covariance vanishes.**  
  
Thus parity zeros come from spectral symmetry, not from rotational invariance.  
  
This is one reason anisotropy does not destroy the theorem.  
  
⸻  
  
**157. Why Isotropy Is Not Needed**  
  
The GCJA theorem is measure-theoretic in \rho**.**  
  
It only needs:  
  
1. finite moments,  
2. polynomial nondegeneracy,  
3. symmetry for real-valuedness,  
4. stationarity for spectral representation.  
  
It does not need:  
  
\rho(RA)=\rho(A)  
  
for rotations R**.**  
  
Isotropy simplifies scalar constants such as:  
  
\lambda_2,\lambda_4,\lambda_6,  
  
but the filtration itself is algebraic:  
  
\text{Hermite interpolation}  
+  
\text{spectral projection}.  
  
Therefore anisotropic fields obey the same visible/unseen jet law, with tensor-valued constants.  
  
This is the technical basis for the anisotropy bridge.  
  
⸻  
  
**158. Design Surgery**  
  
The GCJA package also explains design surgery.  
  
Changing the observation design changes q_k**, hence changes the visible space:**  
  
V(D).  
  
Adding observations expands V(D)**.**  
  
Removing observations shrinks V(D)**.**  
  
The minimal unseen stratum changes accordingly.  
  
Therefore all residual covariance exponents can be predicted before doing symbolic computation.  
  
This is important for the q₀ program because different obstruction regions effectively see different designs:  
  
* pair-only design,  
* pair plus Hessian-mark design,  
* three-point design,  
* value-window design,  
* gradient-only design.  
  
The theorem gives a general language for comparing them.  
  
⸻  
  
**159. The Special **(p,j)=(2,2)** Design**  
  
The pair maximum-saddle problem often corresponds to  
  
p=2,  
\qquad  
j=2.  
  
Then:  
  
d_0=2,  
\qquad  
d_1=1,  
\qquad  
d_2=0.  
  
The observed data include, schematically:  
  
* values and first/second longitudinal derivatives at two nodes for transverse level 0,  
* transverse first derivatives and their longitudinal derivatives for transverse level 1,  
* transverse second derivatives for transverse level 2.  
  
The visible condition is:  
  
m+2k\le5.  
  
The minimal unseen weighted degree is:  
  
6.  
  
Candidate minimal unseen monomials include:  
  
(t^6),  
\quad  
(t^4s),  
\quad  
(t^2s^2),  
\quad  
(s^3),  
  
depending on the exact observable and derivative order.  
  
This weighted-degree structure is the source of the \Sigma_2(0)** and inner-zone covariance patterns.**  
  
⸻  
  
**160. Relation to Pair Pinning**  
  
The pair-pinning lemma is a low-dimensional consequence of GCJA.  
  
Conditioning on two nearby critical points and their height gap forces the visible jet space to include the low-order longitudinal fold directions.  
  
The residual on the axis begins at higher order.  
  
This is why the on-axis residual standard deviation scales like:  
  
\Theta(\tau^4)  
  
rather than  
  
\Theta(\tau^3).  
  
GCJA explains this as:  
  
\text{the cubic direction is visible/pinned;}  
  
\text{the first unseen axis direction occurs one degree later.}  
  
Thus the annulus suppression is not a coincidence.  
  
⸻  
  
**161. Relation to Three-Point Kac–Rice**  
  
The inner-zone three-point problem requires conditioning on two critical points and evaluating the gradient at a third point  
  
y=x_S+r^2z.  
  
This corresponds to  
  
\gamma=2.  
  
GCJA predicts the residual covariance scale of  
  
\nabla f(y)  
  
under the pair conditioning.  
  
It also predicts which jet directions dominate.  
  
That information enters the Kac–Rice density:  
  
p_{\nabla f(y)\mid \mathrm{pair}}(0)  
=  
\frac{1}{2\pi\sqrt{\det\Sigma_{\nabla}(y)}}  
\exp  
\left(  
-\frac12  
m_\nabla(y)^T\Sigma_\nabla(y)^{-1}m_\nabla(y)  
\right).  
  
Without GCJA, the determinant and exponent scaling would have to be guessed or repeatedly computed symbolically.  
  
With GCJA, the powers are read from the filtration.  
  
⸻  
  
**162. Relation to Flat-Saddle Tail**  
  
For generic Palm base variables, GCJA plus the pinned mean gives a large exponent:  
  
m_\nabla^T\Sigma_\nabla^{-1}m_\nabla  
\gg1,  
  
producing exponential suppression.  
  
But if the corrected transverse curvature is flat:  
  
\widetilde{\mu}_S\asymp r^2,  
  
the deterministic mean term collapses.  
  
Then the exponential suppression disappears or weakens, and the Palm-average integral is dominated by the polynomial determinant weight near the flat-saddle set.  
  
Thus GCJA supplies the conditional covariance scaling, but the Flat-Saddle Tail Lemma supplies the Palm-base integration.  
  
Both are necessary.  
  
⸻  
  
**163. Relation to Annulus Estimates**  
  
Let the third point be at scale:  
  
|y-x_S|\sim r^\gamma,  
\qquad  
1<\gamma<2.  
  
GCJA gives the conditional residual variance as a function of \gamma**.**  
  
The pinned deterministic mean generally scales at a lower power than the residual standard deviation.  
  
Therefore the Gaussian density at zero is exponentially small:  
  
\exp(-c r^{-\theta(\gamma)}).  
  
This kills annulus obstructions.  
  
At tie values of \gamma**, multiple strata contribute, but the theorem still gives a finite limiting covariance from the tied stratum.**  
  
The annulus proof obligation reduces to verifying that the mean does not vanish on a dangerous open set of Palm bases.  
  
⸻  
  
**164. Relation to S3 Proof Architecture**  
  
The S3 proof architecture argued that \Sigma_2(0)** and related covariance matrices were not special coincidences.**  
  
GCJA formalizes that claim.  
  
S3’s central insight becomes:  
  
\text{nondegeneracy is filtration-theoretic, not formula-specific.}  
  
The covariance object \Sigma_2(0)** is a Gram matrix of unseen monomials after projection away from the visible space.**  
  
Therefore:  
  
* its rank,  
* its positivity,  
* its anisotropic robustness,  
* its dependence on spectral moments,  
  
are all determined by V(D)** and **G_{U|V}**.**  
  
This converts S3 from a heuristic architecture into a theorem-backed layer.  
  
⸻  
  
**165. Relation to S4 Filtration Stress Test**  
  
S4 tested the filtration principle against possible failure modes:  
  
1. non-Gaussian fields,  
2. non-stationary fields,  
3. anisotropic fields,  
4. degenerate spectral support,  
5. rank-deficient designs,  
6. symbolic covariance accidents,  
7. perturbations of observation design.  
  
GCJA resolves the stress test as follows:  
  
* Gaussianity is required for projection-based conditioning.  
* Stationarity is required for spectral monomial representation.  
* Isotropy is not required.  
* Degenerate spectral support is excluded by H_{\mathrm{nd}}.  
* Rank-deficient designs fail when Hermite unisolvence fails.  
* Symbolic covariance patterns are explained by visible/unseen decomposition.  
* Design perturbations are handled by changing V(D).  
  
Thus S4’s candidate theorem becomes a formal theorem under explicit assumptions.  
  
⸻  
  
**166. The Filtration Invariant**  
  
The central invariant is:  
  
\mathcal{F}_D(m,k)  
=  
m+pk  
  
for full designs.  
  
For staircase designs, the invariant is level-dependent:  
  
m<q_k  
  
visible at transverse level k**.**  
  
This filtration determines:  
  
1. which jets are observed,  
2. which jets are unseen,  
3. where the leading residual starts,  
4. which powers of r appear,  
5. how anisotropic covariance enters,  
6. how rank collapse occurs.  
  
Thus, the q₀ program’s conditional covariance exponents are ultimately finite combinatorial data.  
  
⸻  
  
**167. Explicit Interpretation of “Visible”**  
  
A jet direction is visible if the collapsing observations determine it to leading order.  
  
Example:  
  
If values are observed at two nodes  
  
-r/2,\quad r/2,  
  
then the average approximates  
  
f(0),  
  
and the difference divided by r** approximates**  
  
f_t(0).  
  
Thus  
  
f(0),\quad f_t(0)  
  
are visible.  
  
If gradients are also observed at the two nodes, then higher derivatives become visible through Hermite interpolation.  
  
The visible space is exactly the space of polynomial jet directions recoverable from the collapsing data.  
  
⸻  
  
**168. Explicit Interpretation of “Unseen”**  
  
A jet direction is unseen if no linear combination of the observations recovers it at leading order.  
  
Example:  
  
If only two function values are observed, then  
  
f_{tt}(0)  
  
is unseen.  
  
The conditional residual of evaluating the field near the nodes begins with the f_{tt}** direction.**  
  
If two values and two derivatives are observed, then  
  
f_{tt}(0),f_{ttt}(0)  
  
may also become visible, and the residual begins at higher order.  
  
This is why adding gradient constraints dramatically changes residual powers.  
  
⸻  
  
**169. Inner-Zone Example**  
  
Suppose the pair constraints make the field and gradient visible at two nearby points.  
  
Then the field near the saddle is not an unconstrained Gaussian function.  
  
It is already forced to have:  
  
* the right value,  
* zero gradient,  
* fold-compatible longitudinal curvature,  
* pinned cubic gap.  
  
A third point at distance r^2** sees a conditional gradient whose mean and variance are both small but at different powers.**  
  
The inner-zone Kac–Rice density depends on this power comparison.  
  
GCJA supplies the variance powers.  
  
The Palm-base variables supply the mean powers.  
  
The determinant weights supply the integration powers.  
  
The Flat-Saddle Tail Lemma combines them.  
  
⸻  
  
**170. What GCJA Does Not Prove**  
  
GCJA does not prove:  
  
1. elder-rule pairing,  
2. Morse–Smale genericity,  
3. Kac–Rice integrability by itself,  
4. determinant-weighted Palm bounds by itself,  
5. flat-saddle tail lower bounds,  
6. near-diagonal persistence-density constants,  
7. thermodynamic-limit behavior,  
8. non-Gaussian universality.  
  
It is a conditional covariance theorem.  
  
Its role is necessary but not sufficient.  
  
⸻  
  
**171. GCJA Failure Modes**  
  
GCJA can fail if:  
  
1. the field is not Gaussian,  
2. the field is not stationary,  
3. spectral moments are insufficient,  
4. the spectral measure is polynomially degenerate,  
5. observation nodes collide in a non-Hermite-unisolvent way,  
6. the evaluation point enters a scale not covered by the finite ladder,  
7. the design is mis-specified,  
8. tied strata are ignored rather than jointly included.  
  
The archive’s design handles these by explicit hypotheses.  
  
⸻  
  
**172. Why GCJA Is a Major Upgrade**  
  
The program originally depended on several isolated calculations.  
  
GCJA converts these into a reusable theorem.  
  
This creates four upgrades:  
  
**172.1 Proof compression**  
  
Many covariance computations become corollaries.  
  
**172.2 Error detection**  
  
If a claimed power does not match the visible/unseen filtration, it is suspect.  
  
**172.3 Anisotropy extension**  
  
The theorem works with general spectral measure \rho**, so isotropic assumptions can be removed.**  
  
**172.4 Design exploration**  
  
New conditioning schemes can be analyzed by computing V(D)** and the leading unseen stratum.**  
  
This is the kind of theorem infrastructure needed for frontier mathematical work.  
  
⸻  
  
**173. Recommended Final Manuscript Packaging**  
  
The GCJA chapter should be structured as:  
  
1. spectral Hilbert-space setup,  
2. staircase design definitions,  
3. visible and unseen spaces,  
4. Hermite interpolation lemmas,  
5. projection perturbation lemma,  
6. residual expansion theorem,  
7. covariance convergence theorem,  
8. positivity criterion,  
9. full-design weighted-ball corollary,  
10. (p,j)=(2,2) corollary,  
11. pair-pinning corollary,  
12. three-point inner-zone corollary,  
13. annulus suppression corollary,  
14. anisotropy corollary,  
15. design-surgery corollary.  
  
This chapter should appear before the three-point Kac–Rice proof because it supplies the covariance scaling used there.  
  
⸻  
  
**174. Canonical GCJA Corollaries Needed by q₀**  
  
The q₀ program needs the following explicit corollaries.  
  
**Corollary G1 — Uniform pair-frame invertibility**  
  
The corrected pair observation frame has covariance uniformly bounded away from singularity after divided-difference normalization.  
  
**Corollary G2 — Cubic pinning**  
  
The conditioned field under a nearby maximum-saddle pair has a pinned cubic longitudinal fold.  
  
**Corollary G3 — On-axis residual suppression**  
  
Along the pair axis, residual field fluctuations start one order later than naïve Taylor expansion would suggest.  
  
**Corollary G4 — Inner-zone gradient covariance**  
  
At  
  
y=x_S+r^2z,  
  
the conditional covariance of  
  
\nabla f(y)  
  
has the finite-rank limiting form predicted by the leading unseen stratum.  
  
**Corollary G5 — Annulus residual dominance**  
  
For intermediate scales  
  
r^2\ll |y-x_S|\ll r,  
  
the conditional mean dominates residual fluctuations except near named degeneracy strata.  
  
**Corollary G6 — Anisotropic robustness**  
  
All previous isotropic covariance patterns persist with tensor constants under H_{\mathrm{nd}}**.**  
  
⸻  
  
**175. Connection to Theorem A**  
  
Theorem A requires:  
  
P_{\mathrm{inner}}\to0,  
  
P_{\mathrm{ann}}\to0.  
  
Both require conditional covariance estimates under pair Palm conditioning.  
  
GCJA supplies those covariance estimates.  
  
Then:  
  
* inner zone uses GCJA + Flat-Saddle Tail,  
* annulus uses GCJA + pinned mean separation,  
* far zone uses ordinary Kac–Rice,  
* deterministic layer assembles them.  
  
Thus the dependency is:  
  
\mathrm{GCJA}  
\Rightarrow  
\text{Pinning}  
\Rightarrow  
\text{Inner/Annulus estimates}  
\Rightarrow  
q_0=1.  
  
⸻  
  
**176. Connection to Theorem B**  
  
Theorem B’s \ell^{-1/3}** exponent does not directly depend on GCJA.**  
  
However, Theorem B depends on Theorem A.  
  
Since Theorem A depends on GCJA, the full near-diagonal law depends indirectly on GCJA.  
  
The logical chain is:  
  
\mathrm{GCJA}  
\Rightarrow  
q_0=1  
\Rightarrow  
\nu(\ell)\sim C_*\ell^{-1/3}.  
  
Without GCJA, the selection constant remains unproven.  
  
⸻  
  
**177. Epistemic Status of GCJA**  
  
Current status:  
  
\textbf{PROVEN-HERE / theorem package}  
  
under the stated hypotheses.  
  
Remaining tasks are not conceptual but manuscript-level:  
  
1. align notation with the main theorem,  
2. verify every referenced design matches the q₀ pair design,  
3. extract exact corollaries used by File 3 and File 4,  
4. eliminate duplicate names,  
5. ensure moment orders are not stronger than needed,  
6. state torus-lattice nondegeneracy cleanly,  
7. include tie-stratum handling explicitly.  
  
The theorem should be treated as one of the strongest components of the archive.  
  
⸻  
  
**178. Immediate Next Reconstruction Step**  
  
The next installment should reconstruct the three-point Kac–Rice and two-scale degeneration layer.  
  
That layer must explain:  
  
1. why a third critical point is the dangerous obstruction,  
2. how the three-point Kac–Rice density is written,  
3. why the r^2 inner scale appears,  
4. how pair conditioning changes the third-point gradient law,  
5. how determinant factors enter,  
6. how value-window constraints enter,  
7. what the nondegeneracy condition is,  
8. how the flat-saddle tail corrects the averaged estimate,  
9. what remains open.  
  
## This is the layer where GCJA becomes a concrete obstruction bound.  
  
**Unified Master Source — Part 6**  
  
**Three-Point Kac–Rice and Two-Scale Degeneration: Inner-Zone Obstruction Machinery**  
  
⸻  
  
**179. Purpose of the Three-Point Layer**  
  
The three-point layer is the technical core of the local obstruction proof.  
  
The deterministic layer says:  
  
D(x_M)\ne x_S  
  
can occur only if some obstruction exists.  
  
The Gaussian pair-Palm layer gives the law of the field conditioned on the nearby maximum-saddle pair.  
  
The GCJA layer gives the conditional covariance powers.  
  
The three-point Kac–Rice layer combines these into an actual estimate for the probability that a third critical point appears near the candidate fold pair.  
  
The most dangerous third point is not far away.  
  
The most dangerous third point is at the inner scale:  
  
|y-x_S|\sim r^2.  
  
This is where the local fold degeneracy, Palm conditioning, and third-point critical equations interact most singularly.  
  
The goal of this layer is to prove:  
  
\mathbb{E}_{\mathrm{Palm}}N_{\mathrm{inner}}\to0.  
  
The current corrected result is stronger:  
  
\mathbb{E}_{\mathrm{Palm}}N_{\mathrm{inner}}  
\le  
C r^5(1+o(1)).  
  
This upper bound is enough to preserve Theorem A.  
  
⸻  
  
**180. Why a Third Critical Point Matters**  
  
The candidate fold pair consists of exactly two nearby critical points:  
  
x_M,\quad x_S.  
  
In the clean local picture:  
  
* x_M is born,  
* x_S kills it,  
* no other critical point enters the fold cell.  
  
A third critical point near x_S** can disrupt this picture.**  
  
It can:  
  
1. create another maximum nearby,  
2. create another saddle nearby,  
3. split the local superlevel lobe,  
4. generate an alternate merge,  
5. change gradient adjacency,  
6. create a loop/crater witness,  
7. invalidate the simple fold normal form.  
  
Thus the inner-zone obstruction is bounded by counting third critical points:  
  
N_{\mathrm{inner}}  
=  
\#\{y:|y-x_S|\le Cr^2,\ \nabla f(y)=0,\ y\ne x_S\}.  
  
For upper bounds, one does not need to classify the third point by index. Any third critical point is counted as dangerous.  
  
Therefore:  
  
\mathbb{P}(\mathcal{O}_{\mathrm{inner}})  
\le  
\mathbb{E}N_{\mathrm{inner}}.  
  
⸻  
  
**181. Three-Point Configuration**  
  
The three points are:  
  
x_M,  
\qquad  
x_S,  
\qquad  
y.  
  
Use pair coordinates with the saddle at the origin for inner analysis:  
  
x_S=0.  
  
Then the maximum lies at:  
  
x_M=-r e_t  
  
or  
  
x_M=+r e_t  
  
depending on orientation.  
  
The third point is placed in the inner window:  
  
y=r^2 z,  
  
where  
  
z=(z_t,z_s)  
  
belongs to a fixed compact set:  
  
|z|\le C.  
  
The scale r^2** is the two-scale degeneration:**  
  
* the pair separation is r,  
* the third-point separation from the saddle is r^2.  
  
Thus the point configuration has two collapsing scales:  
  
r  
\quad\text{and}\quad  
r^2.  
  
This is why ordinary two-point Kac–Rice is insufficient.  
  
⸻  
  
**182. Conditional Critical-Point Count**  
  
Under the fold-pair Palm law, the expected number of third critical points in the inner window is:  
  
\mathbb{E}_{\mathrm{Palm}}N_{\mathrm{inner}}  
=  
\int_{|y-x_S|\le Cr^2}  
\mathbb{E}_{\mathrm{Palm}}  
\left[  
|\det H_y|  
\,  
p_{\nabla f(y)\mid \mathrm{pair},\mathrm{base}}(0)  
\right]  
dy.  
  
More explicitly, after exposing pair-base variables \theta**,**  
  
\mathbb{E}_{\mathrm{Palm}}N_{\mathrm{inner}}  
=  
\int  
\int_{|y-x_S|\le Cr^2}  
p_{\nabla f(y)\mid \theta}(0)  
\,  
\mathbb{E}\left[  
|\det H_y|  
\mid  
\nabla f(y)=0,\theta  
\right]  
dy  
\,  
d\mathbb{P}_{\mathrm{Palm}}(\theta).  
  
The pair-base variables \theta** include quantities such as:**  
  
\kappa,\quad  
\mu,\quad  
\nu_S,\quad  
\nu_M,\quad  
\eta,  
  
or corrected versions thereof.  
  
The core task is to estimate the integrand uniformly and integrate over \theta**.**  
  
⸻  
  
**183. Three-Point Kac–Rice Density**  
  
For fixed y\ne x_M,x_S**, the joint Kac–Rice density for the pair plus third critical point contains:**  
  
p_{\nabla f(x_M),\nabla f(x_S),\nabla f(y)}(0,0,0)  
  
times determinant factors:  
  
|\det H_M|\,|\det H_S|\,|\det H_y|.  
  
With height marks included for the pair:  
  
f(x_M)=b,  
\qquad  
f(x_S)=b-\ell,  
  
one includes the corresponding joint density.  
  
The three-point critical intensity is schematically:  
  
\rho_3(x_M,x_S,y)  
=  
p_{\Phi_{x_M,x_S,y}}(\text{constraints})  
\,  
\mathbb{E}  
[  
|\det H_M||\det H_S||\det H_y|  
\,1_{\mathrm{indices}}  
\mid  
\Phi_{x_M,x_S,y}  
].  
  
The pair-Palm version divides by the two-point pair intensity:  
  
\rho_2(x_M,x_S).  
  
Thus:  
  
\mathbb{E}_{\mathrm{Palm}}N_{\mathrm{inner}}  
=  
\int_{\mathrm{inner}}  
\frac{\rho_3(x_M,x_S,y)}{\rho_2(x_M,x_S)}  
dy.  
  
This quotient is often the cleanest conceptual expression.  
  
⸻  
  
**184. Why the Quotient Is Singular**  
  
Both \rho_3** and **\rho_2** are singular as points collide.**  
  
As  
  
r\to0,  
  
the two-point gradient covariance degenerates.  
  
As  
  
y\to x_S  
  
at scale r^2**, the three-point gradient covariance degenerates further.**  
  
Thus both numerator and denominator contain powers of r** from:**  
  
1. Gaussian density blow-up,  
2. determinant collapse,  
3. height-gap scaling,  
4. third-point area element,  
5. conditional covariance collapse.  
  
The proof must show that after quotient and integration, the result tends to zero.  
  
This requires exact power accounting.  
  
⸻  
  
**185. Inner-Zone Area Element**  
  
The inner window is:  
  
y=x_S+r^2 z,  
\qquad |z|\le C.  
  
In dimension two:  
  
dy=r^4 dz.  
  
This contributes a factor:  
  
r^4.  
  
Any Kac–Rice density blow-up must be compared against this volume gain.  
  
If the conditional gradient density at zero blows up like:  
  
r^{-a},  
  
and the conditional determinant expectation behaves like:  
  
r^b,  
  
then the raw inner contribution behaves like:  
  
r^{4-a+b}  
  
before Palm-base integration.  
  
The entire inner-zone proof is a disciplined version of this ledger.  
  
⸻  
  
**186. Conditional Gradient at the Third Point**  
  
Under the pair conditioning, write:  
  
\nabla f(y)=m_\nabla(y;\theta)+\xi_\nabla(y),  
  
where:  
  
* m_\nabla is the conditional mean given pair constraints and exposed pair-base variables,  
* \xi_\nabla is centered Gaussian residual.  
  
The density at zero is:  
  
p_{\nabla f(y)\mid\theta}(0)  
=  
\frac{1}{2\pi\sqrt{\det\Sigma_\nabla(y;\theta)}}  
\exp  
\left[  
-\frac12  
m_\nabla(y;\theta)^T  
\Sigma_\nabla(y;\theta)^{-1}  
m_\nabla(y;\theta)  
\right].  
  
The GCJA theorem controls:  
  
\Sigma_\nabla(y;\theta).  
  
The fold normal form controls:  
  
m_\nabla(y;\theta).  
  
The dangerous regime is when:  
  
m_\nabla  
  
is small compared with the residual standard deviation.  
  
⸻  
  
**187. Generic Inner Mean**  
  
At a generic fold pair, the transverse local behavior near the saddle includes:  
  
f_s(t,s)  
\approx  
-\widetilde{\mu}_S s  
+  
\text{higher-order terms}.  
  
At  
  
y=r^2 z,  
  
this gives:  
  
m_s(y)  
\approx  
-\widetilde{\mu}_S r^2 z_s.  
  
If  
  
\widetilde{\mu}_S\asymp1,  
  
then:  
  
m_s(y)\asymp r^2.  
  
The residual standard deviation at the inner scale is much smaller in the relevant direction.  
  
Thus the Gaussian exponent is large:  
  
m^T\Sigma^{-1}m\gg1,  
  
and criticality is exponentially suppressed.  
  
This is the “fixed generic pair” suppression.  
  
⸻  
  
**188. Flat-Saddle Degeneracy**  
  
The suppression disappears when:  
  
\widetilde{\mu}_S  
  
is small.  
  
Specifically, if  
  
\widetilde{\mu}_S\asymp r^2,  
  
then:  
  
m_s(y)  
\asymp r^4,  
  
which can be comparable to the residual standard deviation.  
  
Thus the third-point gradient can vanish without an exponential penalty.  
  
This is the flat-saddle tail.  
  
The inner obstruction is therefore governed not by generic fold pairs, but by rare nearly flat transverse saddles.  
  
⸻  
  
**189. Corrected Flat-Saddle Variable**  
  
The relevant flatness variable is not simply:  
  
\mu=-f_{ss}(x_S).  
  
It is:  
  
\widetilde{\mu}_S  
=  
\mu+\frac{r\nu_S^2}{\kappa}.  
  
This correction comes from the determinant and the mixed derivative:  
  
\det H_S  
=  
f_{tt}(x_S)f_{ss}(x_S)-f_{ts}(x_S)^2.  
  
With:  
  
f_{tt}(x_S)\sim \kappa r,  
  
f_{ts}(x_S)\sim r\nu_S,  
  
one gets:  
  
\det H_S  
=  
-\kappa r\widetilde{\mu}_S+O(r^3).  
  
Thus \widetilde{\mu}_S** is the true transverse determinant curvature.**  
  
The flat-saddle tail is:  
  
0<\widetilde{\mu}_S\lesssim Cr^2.  
  
⸻  
  
**190. Third-Point Hessian Determinant**  
  
The Kac–Rice integrand includes:  
  
|\det H_y|.  
  
At an inner third critical point near the saddle, H_y** is close to **H_S**, but the conditioning **\nabla f(y)=0** changes the distribution.**  
  
For an upper bound, it is enough to use a polynomial moment estimate:  
  
\mathbb{E}  
[  
|\det H_y|  
\mid  
\nabla f(y)=0,\theta  
]  
\le  
C r^{\beta_H}  
  
or even a cruder bounded-moment estimate if the gradient-density suppression supplies enough power.  
  
The archive’s current flat-saddle upper bound absorbs this determinant into a global power ledger.  
  
The key point is that the third determinant does not introduce a nonintegrable singularity strong enough to destroy r^5**.**  
  
⸻  
  
**191. Pair Determinant Product in the Palm Base**  
  
The pair-Palm law already includes:  
  
|\det H_M||\det H_S|.  
  
On the flat-saddle set:  
  
\widetilde{\mu}_S\asymp r^2.  
  
Then:  
  
|\det H_S|\asymp r^3.  
  
Because:  
  
\widetilde{\mu}_M  
=  
\widetilde{\mu}_S+r\widetilde{\eta}+O(r^2),  
  
typically:  
  
\widetilde{\mu}_M\asymp r.  
  
So:  
  
|\det H_M|\asymp r^2.  
  
Therefore:  
  
|\det H_M||\det H_S|\asymp r^5.  
  
This determinant depletion is the dominant reason the flat-saddle tail remains small after Palm averaging.  
  
⸻  
  
**192. Why the **r^4** Claim Was Wrong**  
  
The superseded r^4** claim used:**  
  
|\det H_S|\asymp r^3,  
  
but treated:  
  
|\det H_M|\asymp r.  
  
That assumed the maximum transverse curvature stayed order one.  
  
But smoothness over distance r** implies:**  
  
f_{ss}(x_M)-f_{ss}(x_S)=O_P(r).  
  
Therefore if the saddle corrected transverse curvature is O(r^2)**, the maximum corrected transverse curvature is not generically **O(1)**. It is **O(r)**.**  
  
Thus:  
  
|\det H_M|\asymp r\cdot r=r^2,  
  
not r**.**  
  
The correct pair determinant product is:  
  
r^3\cdot r^2=r^5.  
  
⸻  
  
**193. Why the Super-Exponential Claim Was Wrong**  
  
At fixed generic pair data,  
  
\widetilde{\mu}_S\asymp1.  
  
Then inner third criticality is exponentially suppressed.  
  
This led to the earlier claim:  
  
\mathbb{E}N_{\mathrm{inner}}  
=  
O(e^{-c/r^4}).  
  
But that estimate was not uniform in the pair-base variable \widetilde{\mu}_S**.**  
  
As  
  
\widetilde{\mu}_S\to0,  
  
the exponent constant degenerates.  
  
The Palm distribution includes a region:  
  
\widetilde{\mu}_S\lesssim r^2.  
  
In that region, there is no exponential cost.  
  
The average over pair-base variables is dominated by this polynomial tail.  
  
Therefore the correct averaged upper bound is polynomial, not super-exponential.  
  
The killed statement is:  
  
O(e^{-c/r^4})  
\quad  
\text{under Palm averaging}.  
  
The repaired statement is:  
  
\mathbb{E}_{\mathrm{Palm}}N_{\mathrm{inner}}  
\le  
Cr^5(1+o(1)).  
  
⸻  
  
**194. Inner-Zone Bound — Canonical Lemma**  
  
**Lemma I / Flat-Saddle Inner Bound**  
  
Under H1+**, **H2'**, the corrected pair-Palm law, and the compact base restrictions:**  
  
b\in B,  
\qquad  
\kappa\in K\subset(0,\infty),  
  
the expected number of third critical points in the inner window  
  
|y-x_S|\le Cr^2  
  
satisfies:  
  
\mathbb{E}_{r,b,\kappa}^{\mathrm{MS}}  
N_{\mathrm{inner}}  
\le  
C r^5(1+o(1)).  
  
Consequently:  
  
\mathbb{P}_{r,b,\kappa}^{\mathrm{MS}}  
(\mathcal{O}_{\mathrm{inner}})  
\le  
C r^5(1+o(1)).  
  
Status:  
  
\textbf{DERIVED upper bound; exact lower bound open}.  
  
Role:  
  
This proves the inner obstruction vanishes, which is all Theorem A requires.  
  
⸻  
  
**195. Sketch of the Inner-Zone Bound**  
  
The proof has five steps.  
  
**Step 1 — Pair-base exposure**  
  
Condition on pair linear constraints and expose the Hessian/third-jet variables controlling:  
  
\kappa,\quad  
\widetilde{\mu}_S,\quad  
\widetilde{\eta},\quad  
\nu_S,\quad  
\nu_M.  
  
**Step 2 — Kac–Rice for third point**  
  
For  
  
y=x_S+r^2z,  
  
write the conditional critical density:  
  
p_{\nabla f(y)\mid\theta}(0)  
\mathbb{E}[|\det H_y|\mid\nabla f(y)=0,\theta].  
  
**Step 3 — Generic suppression**  
  
If  
  
\widetilde{\mu}_S\ge Ar^2,  
  
then the mean-to-variance ratio yields exponential or rapidly decaying suppression in A**.**  
  
This part contributes negligibly after choosing A** large.**  
  
**Step 4 — Flat tail**  
  
If  
  
0<\widetilde{\mu}_S\lesssim r^2,  
  
the Gaussian exponent no longer suppresses criticality.  
  
But the pair determinant weight contributes:  
  
|\det H_S||\det H_M|\lesssim r^5.  
  
The conditional density of \widetilde{\mu}_S** is bounded near zero.**  
  
The third-point area contributes:  
  
r^4.  
  
The remaining gradient-density/determinant factors are controlled by GCJA and moment bounds.  
  
After normalization by the pair Palm denominator, the net contribution is:  
  
O(r^5).  
  
**Step 5 — Integrate over base and **z  
  
The compact z**-window has finite volume.**  
  
Uniformity in b,\kappa** gives:**  
  
\mathbb{E}N_{\mathrm{inner}}\le Cr^5(1+o(1)).  
  
⸻  
  
**196. Nondegeneracy Condition in Three-Point Analysis**  
  
The inner-zone three-point expansion requires that the limiting conditional covariance of the third-point gradient does not collapse beyond the predicted filtration.  
  
This is the nondegeneracy condition sometimes referred to as:  
  
\mathrm{ND2}  
  
or  
  
O\text{-ND2}.  
  
In final form it should be stated as:  
  
The leading Gram pushforward matrix  
  
\Sigma_\infty(z)  
  
for the third-point gradient, after conditioning on the visible pair design, is positive definite for all relevant  
  
z  
  
outside a lower-dimensional exceptional set, and any exceptional set is integrable in the Kac–Rice integral.  
  
A stronger sufficient condition is:  
  
\det\Sigma_\infty(z)\ge c>0  
  
uniformly over the compact inner chart after excluding the already-pinned point.  
  
If uniform positivity fails only at isolated or algebraic sets, one must prove the singularity is integrable.  
  
This is one of the precise remaining symbolic/proof obligations.  
  
⸻  
  
**197. Value-Window Constraint**  
  
Not every third critical point matters topologically.  
  
A third critical point matters if its value lies in the narrow band:  
  
[f(x_S),f(x_M)].  
  
The band width is:  
  
\ell\asymp r^3.  
  
Thus a sharper obstruction count includes:  
  
1_{\{f(y)\in[f(x_S),f(x_M)]\}}.  
  
However, for the inner upper bound, counting all critical points is safer and sufficient if the resulting bound still vanishes.  
  
The archive often separates:  
  
* critical-count bounds,  
* value-window bounds.  
  
The value-window can improve powers but should not be relied upon unless its conditional density is controlled uniformly.  
  
The corrected r^5** inner bound does not need a speculative value-window gain.**  
  
⸻  
  
**198. Index Constraint at Third Point**  
  
Similarly, not all third critical points are topologically dangerous.  
  
Depending on the obstruction, the third point may need to be a maximum or saddle.  
  
But for upper bounds:  
  
1_{\mathrm{dangerous}}\le1_{\nabla f(y)=0}.  
  
Therefore the inner bound counts every critical point.  
  
This avoids delicate index classification under degenerate conditioning.  
  
Exact constants or lower bounds would require index classification.  
  
⸻  
  
**199. Comparison With Literature**  
  
The three-point layer is literature-adjacent but not literature-contained.  
  
Classical Kac–Rice theory supports:  
  
* one-point critical intensities,  
* two-point critical intensities,  
* multi-point factorial moment formulas,  
* conditional determinant expectations under Gaussianity.  
  
Existing works also discuss:  
  
* divided differences near colliding points,  
* blow-up of covariance determinants,  
* conditional determinant formulas,  
* multijet transversality/desingularization.  
  
However, the exact configuration here is specialized:  
  
\text{maximum-saddle pair at scale }r  
  
plus  
  
\text{third critical point at scale }r^2  
  
plus  
  
\text{height gap }\ell\asymp r^3  
  
plus  
  
\text{Palm determinant weighting}  
  
plus  
  
\text{elder-rule obstruction interpretation}.  
  
The archive’s claim is not that Kac–Rice is new.  
  
The claim is that this specific two-scale three-point Kac–Rice expansion for near-diagonal H_0** persistence of smooth Gaussian fields appears to be new.**  
  
⸻  
  
**200. Two-Scale Blow-Up Variables**  
  
The full two-scale blow-up should use variables:  
  
r  
  
for pair separation,  
  
z=\frac{y-x_S}{r^2}  
  
for third-point inner coordinate,  
  
\kappa=\frac{6\ell}{r^3}  
  
for normalized height gap,  
  
\widetilde{\mu}_S  
  
for corrected saddle flatness,  
  
\widetilde{\eta}  
  
for maximum-saddle transverse-curvature coupling.  
  
The local coordinates should be:  
  
(t,s)=r^2(z_t,z_s)  
  
around the saddle.  
  
Then every term in the conditional gradient expansion should be written as a power of r** times a polynomial in **z** and the Palm-base variables.**  
  
This creates an explicit exponent ledger.  
  
⸻  
  
**201. Inner Gradient Expansion**  
  
The conditional mean near the saddle has schematic form:  
  
m_t(r^2z)  
=  
\kappa r^3 z_t  
+  
\text{quadratic/cubic corrections}  
+  
\text{terms involving }z_s^2  
+  
O(r^4),  
  
m_s(r^2z)  
=  
-\widetilde{\mu}_S r^2 z_s  
+  
r^3 P_s(z;\theta)  
+  
O(r^4).  
  
The exact polynomial P_s** depends on third derivatives and mixed terms.**  
  
When  
  
\widetilde{\mu}_S\asymp1,  
  
the transverse component dominates:  
  
m_s\asymp r^2.  
  
When  
  
\widetilde{\mu}_S\asymp r^2,  
  
the transverse linear term becomes:  
  
O(r^4),  
  
and the remaining terms compete.  
  
This is the inner critical balance.  
  
⸻  
  
**202. Residual Gradient Covariance**  
  
By GCJA, the conditional covariance of  
  
\nabla f(r^2z)  
  
has an expansion:  
  
\Sigma_\nabla(r,z)  
=  
r^{\alpha}  
\Sigma_\infty(z)  
+  
O(r^{\alpha+\delta})  
  
in the appropriate componentwise or matrix-weighted sense.  
  
Because the two gradient components may scale differently, one often uses a diagonal rescaling matrix:  
  
D_r=\mathrm{diag}(r^{-a_t},r^{-a_s})  
  
such that:  
  
D_r\Sigma_\nabla(r,z)D_r  
\to  
\Sigma_\infty(z).  
  
The limiting matrix must be positive definite on the relevant chart.  
  
This gives:  
  
\det\Sigma_\nabla(r,z)  
\sim  
r^{2a_t+2a_s}  
\det\Sigma_\infty(z).  
  
Therefore:  
  
p_{\nabla f(y)\mid\theta}(0)  
\sim  
r^{-(a_t+a_s)}  
\frac{1}{2\pi\sqrt{\det\Sigma_\infty(z)}}  
\exp(-\cdots).  
  
This is the density contribution in the exponent ledger.  
  
⸻  
  
**203. Determinant Expectation at Third Point**  
  
Conditioning on  
  
\nabla f(y)=0  
  
changes the Hessian distribution at y**.**  
  
But because all variables are Gaussian, the conditional Hessian is again Gaussian with mean and covariance determined by Schur complement.  
  
The expected absolute determinant has polynomial growth/bounds:  
  
\mathbb{E}  
[  
|\det H_y|  
\mid  
\nabla f(y)=0,\theta  
]  
\le  
C(1+|\theta|^m)r^{\beta_H}  
  
for some finite m** and appropriate **\beta_H**.**  
  
The finite moment assumptions ensure the \theta**-moments are integrable under the Palm law.**  
  
The exact \beta_H** affects sharp exponents but not the qualitative vanishing if the flat-saddle determinant ledger already gives sufficient smallness.**  
  
⸻  
  
**204. Inner Bound Without Sharp Constant**  
  
The current theorem does not need the exact asymptotic:  
  
\mathbb{E}N_{\mathrm{inner}}  
\sim  
C r^5.  
  
It only needs:  
  
\mathbb{E}N_{\mathrm{inner}}  
=o(1).  
  
Therefore the proof may use inequalities rather than exact asymptotic expansions.  
  
This is important strategically.  
  
Trying to prove a two-sided law would require:  
  
1. exact limiting density,  
2. exact index classification,  
3. exact adjacency treatment,  
4. exact flat-tail density,  
5. exact determinant expectation,  
6. exact integration over z,  
7. exact positivity of all constants.  
  
The q₀ theorem only requires upper bounds.  
  
Thus the final proof should avoid unnecessary exactness.  
  
⸻  
  
**205. Lower Bound Conjecture**  
  
A natural conjecture is:  
  
\mathbb{E}N_{\mathrm{inner}}  
\asymp r^5.  
  
But this is not yet proved.  
  
A lower bound would require showing:  
  
1. flat-saddle configurations occur with Palm-weighted mass of the predicted order,  
2. the third-point gradient density has a nonzero limiting contribution on some z-region,  
3. the third-point determinant expectation is positive,  
4. index/admissibility does not remove all such configurations,  
5. adjacency does not kill the contribution.  
  
This is not needed for Theorem A.  
  
It may be important for a future sharp theory of inner-zone defect statistics.  
  
Current status:  
  
\textbf{CONJECTURE}.  
  
⸻  
  
**206. Three-Point Layer Interface With Annulus Layer**  
  
The inner-zone bound controls:  
  
|y-x_S|\le Cr^2.  
  
The annulus controls:  
  
Cr^2<|y-x_S|\le c r  
  
or broader intermediate regimes.  
  
The two proofs should match at the boundary.  
  
At  
  
|y-x_S|\sim r^2,  
  
the inner blow-up is natural.  
  
At  
  
|y-x_S|\gg r^2,  
  
the flat-saddle inner scaling no longer dominates in the same way.  
  
The annulus proof uses mean-dominance rather than flat-tail counting.  
  
A complete proof must ensure no gap remains between the chosen inner radius and annulus lower bound.  
  
Thus choose constants:  
  
R_0 r^2  
  
for inner outer edge and  
  
R_0 r^2  
  
for annulus inner edge, with R_0** large but fixed.**  
  
Then send R_0** through the estimate if necessary.**  
  
⸻  
  
**207. Three-Point Layer Interface With Pair Density**  
  
The same two-scale machinery also supports the pair-density input.  
  
For the near-diagonal law, one needs that candidate maximum-saddle fold pairs have finite positive intensity.  
  
This is a two-point degeneration, not a three-point one.  
  
But the algebra is related:  
  
* two-point pair density uses corrected divided differences at scale r,  
* three-point inner density uses an additional point at scale r^2.  
  
Both are controlled by the same visible/unseen jet framework.  
  
Therefore the final monograph should present the two-point pair density before the three-point inner bound, but cross-reference the common GCJA machinery.  
  
⸻  
  
**208. Three-Point Nondegeneracy as Audit Target**  
  
Because the three-point layer is the most singular proof component, it should receive its own audit checklist.  
  
The audit must verify:  
  
1. the conditioned gradient covariance matrix has the claimed rank,  
2. determinant powers are correctly counted,  
3. the pair-Palm denominator is not lost,  
4. the pair determinant product uses corrected curvatures,  
5. flat-saddle integration is polynomial, not exponential,  
6. the maximum determinant is depleted from r to r^2,  
7. z=0 corresponding to the already-pinned saddle is excluded or desingularized,  
8. all constants are uniform in b,\kappa,  
9. value-window gains are not used unless proved,  
10. adjacency is either included or legitimately dropped for upper bounds.  
  
This audit checklist is mandatory for final proof integrity.  
  
⸻  
  
**209. Correct Treatment of the Pinned Saddle Point**  
  
The third-point integral excludes:  
  
y=x_S.  
  
In the inner coordinates, this is:  
  
z=0.  
  
Kac–Rice density near z=0** may have singular behavior because one critical point is already pinned there.**  
  
The proof must handle this in one of three ways:  
  
1. exclude a small ball around z=0 and show its contribution vanishes,  
2. use a desingularized divided-difference frame for the third point,  
3. prove the limiting z-integral is locally integrable at z=0.  
  
The final monograph cannot casually integrate over z=0** as if it were a regular point.**  
  
The likely correct treatment is desingularization through nested divided differences, already anticipated by the File 3 machinery.  
  
⸻  
  
**210. Nested Divided Differences**  
  
The three-point configuration has nested collisions:  
  
1. x_M\to x_S at scale r,  
2. y\to x_S at scale r^2.  
  
A one-level divided-difference frame is not enough.  
  
The nested frame should include:  
  
* pair-level divided differences between x_M and x_S,  
* third-point divided differences between y and x_S,  
* corrected versions subtracting lower-order gradient terms,  
* scale-normalized coordinates to avoid artificial divergence.  
  
This nested frame converts the three-point covariance determinant into a finite limiting determinant.  
  
The literature contains general ideas for such desingularization, but the exact implementation here is part of the archive’s contribution.  
  
⸻  
  
**211. Expected Three-Point Proof Package**  
  
The final proof package should be organized as:  
  
**Lemma I1 — Nested frame construction**  
  
Build a scale-normalized linear transformation from raw constraints to a nondegenerate limiting jet frame.  
  
**Lemma I2 — Limiting covariance nondegeneracy**  
  
Show the transformed covariance matrix converges to a positive definite matrix under H2'**.**  
  
**Lemma I3 — Conditional gradient expansion**  
  
Compute the conditional mean and covariance of \nabla f(y)** in inner coordinates.**  
  
**Lemma I4 — Generic suppression**  
  
Show that for \widetilde{\mu}_S\gg r^2**, the third-point critical density is exponentially or rapidly suppressed.**  
  
**Lemma I5 — Flat-tail bound**  
  
Show that for \widetilde{\mu}_S=O(r^2)**, the Palm determinant weight gives the **r^5** upper bound.**  
  
**Lemma I6 — **z**-integrability**  
  
Show the resulting inner-coordinate integral is finite uniformly.  
  
**Lemma I7 — Uniformity**  
  
Show constants are uniform over compact b,\kappa** windows.**  
  
Together:  
  
\mathbb{E}N_{\mathrm{inner}}\le Cr^5(1+o(1)).  
  
⸻  
  
**212. The Role of **r^5** in Theorem A**  
  
The inner obstruction contributes:  
  
O(r^5).  
  
The far/band obstruction contributes roughly:  
  
O(r^3).  
  
Therefore the inner zone is not the leading error term after correction.  
  
This matters.  
  
The original fear was that the inner zone could dominate or even invalidate q_0=1**.**  
  
The corrected analysis shows it is harmless for selection, even though the mechanism is subtler than first believed.  
  
The selection theorem does not depend on the inner exponent being 5**. Any positive power would suffice.**  
  
⸻  
  
**213. Inner-Zone Contribution to Future Theories**  
  
Although harmless for q_0=1**, the inner-zone flat-saddle tail may become important for second-order theory.**  
  
Possible future quantities:  
  
1. rate of convergence of q(r,b,\kappa)\to1,  
2. defect-point process near failed fold pairs,  
3. distribution of near-degenerate saddle curvatures,  
4. correction term to near-diagonal bar density,  
5. rare-event statistics of local Morse bifurcations,  
6. universality class of flat-saddle interference.  
  
If the leading mismatch probability is O(r^3)** from far/band events, the **r^5** inner term is subleading.**  
  
But under localized/windowed conditions that remove far events, the flat-saddle tail may become visible.  
  
This is a possible future research direction.  
  
⸻  
  
**214. Relation to C006 Arm1c Falsification**  
  
The flat-saddle correction produced a concrete empirical prediction.  
  
If inner-zone obstruction is dominated by flat saddles, then candidate pairs hosting a window point should show depressed values of:  
  
|f_{ss}(x_S)|  
  
or the corrected curvature proxy.  
  
The C006 Arm1c falsification protocol compares:  
  
* window pairs,  
* matched control pairs.  
  
It computes the ratio of medians:  
  
R=  
\frac{\mathrm{median}(\mu_{\mathrm{window}})}  
{\mathrm{median}(\mu_{\mathrm{control}})}.  
  
The pre-registered rule uses a bootstrap confidence interval.  
  
If the confidence interval includes or exceeds 1** in the wrong way according to the snapshot rule, the flat-saddle mechanism is falsified.**  
  
This does not directly prove or disprove Theorem A, but it tests the proposed mechanism for the inner-zone contribution.  
  
⸻  
  
**215. What the Three-Point Layer Proves**  
  
The layer proves:  
  
\mathcal{O}_{\mathrm{inner}}  
  
has vanishing probability under fold-pair Palm conditioning.  
  
It does not prove:  
  
1. annulus suppression,  
2. far-zone suppression,  
3. deterministic reduction,  
4. pair-density asymptotics,  
5. near-diagonal law,  
6. Morse–Smale genericity.  
  
Its role is local and singular.  
  
But without it, the proof of q_0=1** has a hole at the most dangerous scale.**  
  
⸻  
  
**216. Three-Point Layer Status**  
  
Current status:  
  
\textbf{PROVEN-MODULO / DERIVED UPPER BOUND}.  
  
The strongest current claim is:  
  
\mathbb{E}N_{\mathrm{inner}}\le Cr^5(1+o(1)).  
  
Remaining proof obligations:  
  
1. final symbolic verification of the limiting covariance positivity,  
2. rigorous nested divided-difference construction in final notation,  
3. local integrability near z=0,  
4. uniformity over compact b,\kappa,  
5. clean statement of whether adjacency is dropped or retained,  
6. separation between upper-bound theorem and conjectural two-sided law.  
  
The layer is strong enough for Theorem A if these obligations are discharged as stated.  
  
⸻  
  
**217. Immediate Next Reconstruction Step**  
  
The next installment should reconstruct the annulus, far-zone, loop, and band obstruction estimates.  
  
Those estimates complete the probabilistic side of Theorem A.  
  
They must show:  
  
P_{\mathrm{ann}}\to0,  
  
P_{\mathrm{far}}\to0,  
  
P_{\mathrm{band}}\to0,  
  
P_{\mathrm{loop}}\to0.  
  
The inner zone is now controlled by the flat-saddle tail. The remaining task is to prove no obstruction survives outside the inner r^2** window.**  
  
**Unified Master Source — Part 7**  
  
**Annulus, Far-Zone, Band, and Loop Obstruction Estimates**  
  
⸻  
  
**218. Purpose of the Outer-Obstruction Layer**  
  
The inner-zone analysis controls third critical points at the deepest scale:  
  
|y-x_S|\le Cr^2.  
  
But Theorem A requires every elder-rule mismatch mechanism to vanish.  
  
The remaining obstruction zones are:  
  
1. annulus zone,  
2. far zone,  
3. height-band obstruction,  
4. loop/crater obstruction,  
5. Morse–Smale degeneracy.  
  
The deterministic reduction gives:  
  
1-q(r,b,\kappa)  
\le  
P_{\mathrm{inner}}  
+  
P_{\mathrm{ann}}  
+  
P_{\mathrm{far}}  
+  
P_{\mathrm{band}}  
+  
P_{\mathrm{loop}}  
+  
P_{\mathrm{MS}}.  
  
Part 6 controlled:  
  
P_{\mathrm{inner}}\le Cr^5(1+o(1)).  
  
This part reconstructs the remaining terms.  
  
The target is:  
  
P_{\mathrm{ann}}\to0,  
\qquad  
P_{\mathrm{far}}\to0,  
\qquad  
P_{\mathrm{band}}\to0,  
\qquad  
P_{\mathrm{loop}}\to0.  
  
Once these are established, the selection theorem follows.  
  
⸻  
  
**219. Spatial Zone Decomposition**  
  
Let x_S** be the candidate saddle.**  
  
Let x_M** be the candidate maximum.**  
  
Let  
  
r=d(x_M,x_S).  
  
Choose constants:  
  
0<C_{\mathrm{in}}<C_{\mathrm{ann}}<C_{\mathrm{far}}  
  
and a small fixed geometric radius  
  
\delta>0  
  
independent of r**.**  
  
The local region is divided into:  
  
**219.1 Inner zone**  
  
\mathcal{Z}_{\mathrm{inner}}  
=  
\{y:0<d(y,x_S)\le C_{\mathrm{in}}r^2\}.  
  
**219.2 Transition annulus**  
  
\mathcal{Z}_{\mathrm{ann},1}  
=  
\{y:C_{\mathrm{in}}r^2<d(y,x_S)\le C_{\mathrm{ann}}r\}.  
  
**219.3 Outer annulus**  
  
\mathcal{Z}_{\mathrm{ann},2}  
=  
\{y:C_{\mathrm{ann}}r<d(y,x_S)\le\delta\}.  
  
**219.4 Far zone**  
  
\mathcal{Z}_{\mathrm{far}}  
=  
\{y:d(y,x_S)>\delta\}.  
  
Some drafts merge the two annuli. For proof clarity, the final manuscript should split them.  
  
The hardest region is the transition between r^2** and **r**.**  
  
⸻  
  
**220. Height Band**  
  
The candidate persistence interval is:  
  
I_r  
=  
[f(x_S),f(x_M)].  
  
Since  
  
f(x_M)-f(x_S)=\ell=\frac{\kappa}{6}r^3,  
  
the band width is:  
  
|I_r|=\ell=O(r^3).  
  
Any critical point that changes the elder-rule decision must usually have value in or near this band.  
  
Thus critical-point counts in the band gain a factor:  
  
O(r^3)  
  
when the conditional value density is uniformly bounded.  
  
This is the basic far-zone mechanism.  
  
⸻  
  
**221. Pair-Palm Conditional Field Away From the Pair**  
  
Under pair-Palm conditioning, the field away from x_M,x_S** remains a smooth Gaussian field with modified mean and covariance.**  
  
For y** in the far zone:**  
  
d(y,\{x_M,x_S\})\ge\delta/2  
  
for small r**.**  
  
The conditional covariance of finite jets at y** remains uniformly nondegenerate.**  
  
The conditional mean and finitely many derivatives are uniformly bounded in moments over compact b,\kappa** windows.**  
  
Therefore ordinary Kac–Rice estimates apply uniformly.  
  
This yields far-zone band-count estimates of order:  
  
O(r^3).  
  
⸻  
  
**222. Far-Zone Critical Band Bound**  
  
Let  
  
N_{\mathrm{far,band}}  
=  
\#\{y\in\mathcal{Z}_{\mathrm{far}}:\nabla f(y)=0,\ f(y)\in I_r\}.  
  
Under the fold-pair Palm law,  
  
\mathbb{E}_{\mathrm{Palm}}N_{\mathrm{far,band}}  
=  
\int_{\mathcal{Z}_{\mathrm{far}}}  
\mathbb{E}_{\mathrm{Palm}}  
[  
|\det H_y|  
1_{\{f(y)\in I_r\}}  
\mid  
\nabla f(y)=0  
]  
p_{\nabla f(y)}(0)  
dy.  
  
Uniform nondegeneracy gives:  
  
p_{\nabla f(y)}(0)\le C.  
  
Conditional determinant moments give:  
  
\mathbb{E}[|\det H_y|^p\mid\nabla f(y)=0]\le C_p.  
  
Conditional value density gives:  
  
\mathbb{P}(f(y)\in I_r\mid\nabla f(y)=0,H_y)  
\le C|I_r|  
\le Cr^3.  
  
The far-zone volume is bounded by L^2**.**  
  
Therefore:  
  
\mathbb{E}_{\mathrm{Palm}}N_{\mathrm{far,band}}  
\le  
C r^3.  
  
By Markov:  
  
P_{\mathrm{far}}  
\le  
Cr^3.  
  
This is the cleanest far-zone estimate.  
  
⸻  
  
**223. Why Fixed **L** Is Essential in the Far Zone**  
  
The estimate above uses:  
  
|\mathcal{Z}_{\mathrm{far}}|\le L^2.  
  
If  
  
L\to\infty  
  
while  
  
r\to0,  
  
the far-zone volume grows.  
  
Then:  
  
\mathbb{E}N_{\mathrm{far,band}}  
\sim L^2r^3  
  
may not vanish unless  
  
L^2r^3\to0.  
  
Thus Theorem A is fixed-volume.  
  
A thermodynamic-limit theorem would need a separate scaling condition or a different local/windowed formulation.  
  
⸻  
  
**224. Outer Annulus Bound**  
  
The outer annulus is:  
  
C_{\mathrm{ann}}r<d(y,x_S)\le\delta.  
  
Here y** is close to the pair in an absolute sense, but not close enough to experience singular pair collision at the **r^2** scale.**  
  
The pair-conditioned covariance remains controlled.  
  
However, the conditional mean may not be uniformly generic if y** approaches the pair as **r\to0**.**  
  
A safe estimate proceeds by splitting into dyadic shells:  
  
2^jr \le d(y,x_S)\le 2^{j+1}r,  
  
up to \delta**.**  
  
Let  
  
\tau=d(y,x_S).  
  
The shell area is:  
  
O(\tau^2)  
  
or differentially:  
  
O(\tau\,d\tau).  
  
The conditional mean induced by the pinned fold grows with \tau**, while residual covariance is suppressed by GCJA.**  
  
For \tau\gg r^2**, the mean-to-noise ratio becomes large except on controlled degeneracy sets.**  
  
This produces either:  
  
1. exponential suppression in \tau/r^2, or  
2. a polynomial shell bound with positive total power in r.  
  
The final manuscript should state the outer annulus estimate as a lemma rather than overclaim exact powers.  
  
⸻  
  
**225. Annulus Critical Count**  
  
Define:  
  
N_{\mathrm{ann}}  
=  
\#\{y\in\mathcal{Z}_{\mathrm{ann}}:\nabla f(y)=0,\ f(y)\in I_r\text{ or }y\text{ topologically relevant}\}.  
  
For upper bounds:  
  
N_{\mathrm{ann}}  
\le  
\#\{y\in\mathcal{Z}_{\mathrm{ann}}:\nabla f(y)=0\}.  
  
But this crude count may be too large in some annulus regimes.  
  
A sharper estimate uses the height band:  
  
f(y)\in I_r,  
  
or the conditional mean suppression.  
  
The target is:  
  
\mathbb{E}_{\mathrm{Palm}}N_{\mathrm{ann}}  
\le  
C r^\alpha  
  
for some  
  
\alpha>0.  
  
The exact \alpha** is not essential.**  
  
⸻  
  
**226. Annulus Mean-Dominance Mechanism**  
  
Let  
  
y=x_S+\tau z,  
  
with  
  
r^2\ll \tau\ll r  
  
or  
  
r\ll \tau\ll1.  
  
Under the pinned fold pair, the deterministic conditional mean of the gradient has a leading term inherited from the saddle structure.  
  
Schematically:  
  
m_\nabla(y)  
\approx  
A(\theta)\tau  
+  
B(\theta)\tau^2  
+  
\cdots.  
  
The residual standard deviation under the pair constraints is smaller than the naive unconditioned scale because low-order jets are visible.  
  
GCJA gives:  
  
\mathrm{sd}(\xi_\nabla(y))  
\sim  
r^{a}\tau^{b}  
  
or an equivalent ladder expression.  
  
The annulus is controlled when:  
  
\frac{|m_\nabla(y)|}  
{\mathrm{sd}(\xi_\nabla(y))}  
\to\infty.  
  
Then:  
  
p_{\nabla f(y)\mid\mathrm{pair}}(0)  
\le  
C\exp(-c\Theta(r,\tau)).  
  
This kills critical points.  
  
⸻  
  
**227. Degeneracy Sets in the Annulus**  
  
Mean-dominance can fail if the leading deterministic coefficient vanishes.  
  
The relevant coefficients depend on Palm-base variables such as:  
  
\widetilde{\mu}_S,\quad  
\kappa,\quad  
\nu_S,\quad  
\eta.  
  
The flat-saddle set is one such degeneracy.  
  
The inner analysis handles the deepest flat-saddle effect.  
  
For the annulus, one must show either:  
  
1. the degeneracy set has sufficiently small Palm measure,  
2. the next mean term restores dominance,  
3. the value-band restriction supplies enough decay,  
4. the region can be absorbed into the inner-zone bound by enlarging the inner scale.  
  
This is the annulus proof obligation.  
  
It is not the same as the inner flat-saddle bound.  
  
⸻  
  
**228. Annulus Lemma — Canonical Form**  
  
**Lemma A — Annulus Obstruction Bound**  
  
Under H1+**, **H2'**, Assumption MS, and the fold-pair Palm law with**  
  
b\in B,\qquad \kappa\in K\subset(0,\infty),  
  
there exist constants  
  
\alpha_A>0,  
\qquad  
C<\infty  
  
such that:  
  
\mathbb{P}_{r,b,\kappa}^{\mathrm{MS}}  
(\mathcal{O}_{\mathrm{ann}})  
\le  
C r^{\alpha_A}.  
  
Equivalently,  
  
\mathbb{E}_{r,b,\kappa}^{\mathrm{MS}}  
N_{\mathrm{ann}}  
\le  
C r^{\alpha_A}.  
  
Status:  
  
\textbf{PROVEN-MODULO GCJA mean-dominance/nondegeneracy checks}.  
  
The final proof should not require identifying the best exponent.  
  
⸻  
  
**229. Annulus Proof Strategy**  
  
The proof should proceed by dyadic decomposition.  
  
Let:  
  
\tau_j=2^jr^2.  
  
For each shell:  
  
S_j=\{y:\tau_j\le d(y,x_S)\le2\tau_j\}.  
  
Estimate:  
  
\mathbb{E}N(S_j)  
\le  
\int_{S_j}  
p_{\nabla f(y)\mid\mathrm{pair}}(0)  
\mathbb{E}[|\det H_y|\mid\nabla f(y)=0,\mathrm{pair}]  
dy.  
  
Use GCJA to obtain:  
  
p_{\nabla f(y)\mid\mathrm{pair}}(0)  
\le  
C r^{-a}\tau_j^{-b}  
\exp\left[-c\left(\frac{\tau_j}{r^2}\right)^\theta\right]  
  
away from degeneracy sets.  
  
For degeneracy sets, integrate over base variables and use determinant depletion.  
  
Sum over j**.**  
  
The exponential factor makes large j** harmless.**  
  
Near j=0**, the inner-zone flat-saddle bound handles the contribution.**  
  
This gives a positive power of r**.**  
  
⸻  
  
**230. Band Lemma**  
  
The height band appears both in far-zone and topological obstruction estimates.  
  
**Lemma B — Critical Band Bound**  
  
Let  
  
I_r=[f(x_S),f(x_M)]  
  
with  
  
|I_r|=O(r^3).  
  
Under the fold-pair Palm law, the expected number of critical points outside a local neighborhood U_r** with value in **I_r** is:**  
  
\mathbb{E}N_{\mathrm{band,out}}  
\le  
Cr^3.  
  
Therefore:  
  
P_{\mathrm{band,out}}\le Cr^3.  
  
The proof is ordinary Kac–Rice plus uniform conditional value density.  
  
This lemma is stable and should be considered one of the less dangerous components.  
  
⸻  
  
**231. Local Band Subtlety**  
  
Inside the local annulus, value-band counting alone may not be sufficient.  
  
Near the conditioned saddle and maximum, the field value is already close to the band by construction.  
  
Thus a local third critical point may automatically have value in or near the band.  
  
Therefore the local zones need gradient criticality suppression, not merely value-band suppression.  
  
This is why the inner and annulus estimates exist separately.  
  
⸻  
  
**232. Loop / Crater Bound**  
  
A loop/crater obstruction means the two local arms of the candidate saddle are already connected above the saddle level.  
  
If this happens, there exists a path:  
  
\gamma\subset E_{f(x_S)+\epsilon}  
  
connecting the arms without passing through x_S**.**  
  
For a Morse–Smale function with distinct critical values, such a connection must be supported by critical structure at levels between:  
  
f(x_S)  
  
and  
  
f(x_M)  
  
or by pre-existing high-level topology outside the local cell.  
  
The deterministic reduction converts this into either:  
  
1. a band critical point,  
2. an annulus/far obstruction,  
3. a Morse–Smale degeneracy,  
4. a topologically essential global component event.  
  
On a fixed torus, the global component event again requires a finite critical witness in the band unless the candidate component is essential.  
  
The essential case is excluded for non-essential near-diagonal bars.  
  
Thus:  
  
P_{\mathrm{loop}}  
\le  
P_{\mathrm{band}}+P_{\mathrm{ann}}+P_{\mathrm{far}}+P_{\mathrm{MS}}.  
  
The loop term need not be estimated independently if the deterministic reduction is stated correctly.  
  
⸻  
  
**233. Essential Component Exclusion**  
  
The global maximum births the essential H_0** component.**  
  
If x_M** is the global maximum, it does not die at any saddle.**  
  
Could this create a mismatch?  
  
Yes at finite r**, but under local intensity conditioning for near-diagonal pairs, this event is negligible or excluded by conditioning on non-essential bars.**  
  
A clean theorem statement defines q(r,b,\kappa)** for candidate pairs not involving the global maximum, or observes that the global-maximum event has probability **O(r^3)** under compact birth-height windows after appropriate band/global counting.**  
  
The final monograph should not leave this implicit.  
  
⸻  
  
**234. Morse–Smale Degeneracy Term**  
  
If Assumption MS is imposed, then:  
  
P_{\mathrm{MS}}=0.  
  
If not imposed, one needs:  
  
\mathbb{P}(f\text{ is not Morse–Smale})=0.  
  
This remains a separate global-flow proof obligation.  
  
The obstruction estimate should therefore be written conditionally on MS in the main theorem.  
  
⸻  
  
**235. Full Probabilistic Obstruction Bound**  
  
Combining all terms:  
  
1-q(r,b,\kappa)  
\le  
P_{\mathrm{inner}}  
+  
P_{\mathrm{ann}}  
+  
P_{\mathrm{far}}  
+  
P_{\mathrm{band}}  
+  
P_{\mathrm{loop}}.  
  
Use:  
  
P_{\mathrm{inner}}\le Cr^5,  
  
P_{\mathrm{ann}}\le Cr^{\alpha_A},  
  
P_{\mathrm{far}}\le Cr^3,  
  
P_{\mathrm{band}}\le Cr^3,  
  
P_{\mathrm{loop}}\le P_{\mathrm{band}}+P_{\mathrm{ann}}+P_{\mathrm{far}}.  
  
Then:  
  
1-q(r,b,\kappa)  
\le  
C(r^5+r^{\alpha_A}+r^3).  
  
Let:  
  
\alpha=\min(5,\alpha_A,3)>0.  
  
Then:  
  
1-q(r,b,\kappa)\le Cr^\alpha.  
  
Therefore:  
  
q(r,b,\kappa)\to1.  
  
Uniformly over compact:  
  
b\in B,\qquad \kappa\in K\subset(0,\infty).  
  
⸻  
  
**236. Theorem A Assembly**  
  
**Theorem A — Elder-Rule Selection Constant**  
  
Let f** be a centered stationary Gaussian field on **\mathbb{T}_L^2** satisfying **H1+**, **H2'**, and Assumption MS. Let **B\subset\mathbb{R}** be compact and **K\subset(0,\infty)** compact.**  
  
Under the maximum-saddle fold-pair Palm law at separation r**, birth height **b\in B**, and normalized fold modulus **\kappa\in K**, let:**  
  
q(r,b,\kappa)  
=  
\mathbb{P}(D(x_M)=x_S).  
  
Then:  
  
\sup_{b\in B,\kappa\in K}  
|1-q(r,b,\kappa)|  
\le  
Cr^\alpha  
  
for some:  
  
\alpha>0.  
  
In particular:  
  
\lim_{r\to0}q(r,b,\kappa)=1.  
  
Thus:  
  
q_0=1.  
  
Status:  
  
\textbf{PROVEN-MODULO annulus lemma, inner flat-saddle lemma, and MS assumption}.  
  
⸻  
  
**237. What Is Actually Needed From the Annulus**  
  
The theorem does not need the annulus to be sharp.  
  
It only needs:  
  
P_{\mathrm{ann}}\to0.  
  
Therefore the annulus proof can be conservative.  
  
Acceptable outcomes include:  
  
P_{\mathrm{ann}}\le C|\log r|^{-1},  
  
or any slow vanishing bound, though a polynomial bound is cleaner.  
  
The proof should avoid fragile exact constants.  
  
The priority is robustness.  
  
⸻  
  
**238. Potential Annulus Weak Point**  
  
The main possible failure is an annular degeneracy set where:  
  
1. the pinned mean vanishes,  
2. conditional covariance remains large enough,  
3. determinant weight does not suppress enough,  
4. the region has non-negligible Palm measure.  
  
If such a set exists at positive measure, the annulus proof could fail.  
  
This is why the annulus nondegeneracy condition must be explicitly verified.  
  
It should be isolated as:  
  
O\text{-ANN-ND}.  
  
If O**-ANN-ND fails, the theorem may still survive by a refined blow-up, but the current proof route would need repair.**  
  
⸻  
  
**239. Recommended Annulus Audit**  
  
Before finalizing Theorem A, perform this audit:  
  
1. write the conditional gradient mean in annulus coordinates,  
2. identify all leading coefficients,  
3. solve the algebraic set where the leading mean vanishes,  
4. compute Palm determinant weight near that set,  
5. check whether next-order terms restore dominance,  
6. check whether value-band width supplies r^3,  
7. integrate over dyadic shells,  
8. verify the shell sum converges,  
9. verify uniformity over b,\kappa,  
10. record the resulting exponent \alpha_A.  
  
This audit is essential because annulus estimates are often where hidden logarithmic divergences appear.  
  
⸻  
  
**240. Far-Zone Audit**  
  
The far-zone audit is simpler:  
  
1. prove conditional finite-jet covariance at far points is uniformly nondegenerate,  
2. prove conditional value density is uniformly bounded,  
3. prove determinant moments are uniformly bounded,  
4. integrate over fixed volume,  
5. multiply by band width O(r^3).  
  
This should be straightforward under H1+**, **H2'**, fixed **L**, and compact **b,\kappa**.**  
  
⸻  
  
**241. Loop Audit**  
  
The loop audit is deterministic:  
  
1. define local saddle arms,  
2. define local component born at x_M,  
3. suppose arms connect above f(x_S),  
4. use Morse–Smale merge-tree structure to identify a critical witness,  
5. show the witness lies in one of the already estimated zones,  
6. exclude essential component contamination.  
  
If this reduction is clean, no separate probabilistic loop estimate is needed.  
  
⸻  
  
**242. Status of Outer-Obstruction Layer**  
  
Current status:  
  
* far-zone estimate: strong / routine,  
* band estimate: strong / routine,  
* loop reduction: plausible / deterministic proof needed in final notation,  
* annulus estimate: main remaining technical proof obligation,  
* Morse–Smale: separate global-flow assumption or appendix theorem.  
  
The annulus is the only probabilistic zone outside the inner window that remains structurally delicate.  
  
⸻  
  
**243. Immediate Next Reconstruction Step**  
  
The next installment should reconstruct Theorem B: the near-diagonal H_0** bar-density law.**  
  
That chapter must explain:  
  
1. candidate pair intensity,  
2. fold modulus integration,  
3. radial separation measure,  
4. change of variables \ell=\kappa r^3/6,  
5. origin of the \ell^{-1/3} exponent,  
6. role of q_0=1,  
7. constant positivity,  
8. boundary cases,  
9. why 1D rainflow laws do not already imply this result.  
  
**Unified Master Source — Part 8**  
  
**Theorem B: Near-Diagonal **H_0** Bar-Density Law**  
  
⸻  
  
**244. Purpose of Theorem B**  
  
Theorem A proves the elder-rule selection constant:  
  
q_0=1.  
  
Theorem B turns that local selection result into a persistence-diagram density law.  
  
The goal is to describe the expected density of small-lifetime H_0** bars for a smooth stationary Gaussian field on **\mathbb{T}_L^2**.**  
  
The near-diagonal lifetime is:  
  
\ell=f(x_M)-f(x_S),  
  
where:  
  
* x_M is a local maximum,  
* x_S is the death saddle paired to x_M,  
* \ell\to0^+.  
  
The claimed density law is:  
  
\nu(\ell)  
=  
C_*\ell^{-1/3}(1+o(1))  
  
as  
  
\ell\downarrow0.  
  
Equivalently, the cumulative expected number of bars with lifetime at most \varepsilon** satisfies:**  
  
N(0,\varepsilon)  
=  
\int_0^\varepsilon \nu(\ell)\,d\ell  
\sim  
\frac32 C_*\varepsilon^{2/3}.  
  
The exponent  
  
-\frac13  
  
is the core result.  
  
⸻  
  
**245. What Theorem B Is Really Saying**  
  
The theorem says that near the diagonal of the persistence diagram, the expected H_0** bar density is singular.**  
  
Small bars are not distributed with a finite nonzero density at \ell=0**.**  
  
Instead:  
  
\nu(\ell)\to\infty  
  
as  
  
\ell\downarrow0.  
  
But the singularity is integrable:  
  
\int_0^\varepsilon \ell^{-1/3}d\ell  
=  
\frac32\varepsilon^{2/3}.  
  
Thus the number of very small bars tends to zero as the window shrinks, but at a slower-than-linear rate.  
  
This is exactly what one expects from fold geometry in two spatial dimensions.  
  
⸻  
  
**246. Candidate Pair Intensity Versus True Bar Intensity**  
  
There are two densities:  
  
1. candidate maximum-saddle fold-pair intensity,  
2. actual persistence bar intensity.  
  
The candidate intensity counts local maximum-saddle fold pairs at separation r**, height **b**, and modulus **\kappa**, whether or not elder-rule pairing selects them.**  
  
The true persistence intensity counts only those candidate pairs that are actual H_0** persistence pairs.**  
  
Theorem A says the selection probability tends to one:  
  
q(r,b,\kappa)\to1.  
  
Therefore, near the diagonal:  
  
\text{true bar intensity}  
\sim  
\text{candidate fold-pair intensity}.  
  
Theorem B is obtained by combining:  
  
\text{candidate pair geometry}  
+  
q_0=1  
+  
\ell=\frac{\kappa}{6}r^3.  
  
⸻  
  
**247. Fold Relation**  
  
For a near maximum-saddle fold pair:  
  
\ell=f(x_M)-f(x_S).  
  
The fold normal form gives:  
  
\ell=\frac{\kappa}{6}r^3(1+o(1)),  
  
where:  
  
r=d(x_M,x_S),  
  
and:  
  
\kappa>0  
  
is the normalized fold modulus.  
  
Equivalently:  
  
r=\left(\frac{6\ell}{\kappa}\right)^{1/3}(1+o(1)).  
  
This cubic relation is the source of the nonstandard exponent.  
  
⸻  
  
**248. Radial Measure in Two Dimensions**  
  
In two dimensions, the relative displacement between x_M** and **x_S** has polar measure:**  
  
d^2h=r\,dr\,d\theta.  
  
Thus small-separation pairs are counted with a radial factor:  
  
r\,dr.  
  
Using:  
  
\ell=\frac{\kappa}{6}r^3,  
  
differentiate:  
  
d\ell=\frac{\kappa}{2}r^2\,dr  
  
for fixed \kappa**.**  
  
So:  
  
dr=\frac{2}{\kappa r^2}d\ell.  
  
Then:  
  
r\,dr  
=  
\frac{2}{\kappa r}d\ell.  
  
Substitute:  
  
r=\left(\frac{6\ell}{\kappa}\right)^{1/3}.  
  
Thus:  
  
r\,dr  
=  
2\kappa^{-1}  
\left(\frac{6\ell}{\kappa}\right)^{-1/3}  
d\ell  
=  
2\cdot 6^{-1/3}\kappa^{-2/3}\ell^{-1/3}d\ell.  
  
Therefore the two-dimensional spatial measure produces:  
  
\ell^{-1/3}.  
  
This is the exponent.  
  
⸻  
  
**249. General Dimensional Heuristic**  
  
The same fold relation in spatial dimension d** would combine with radial measure:**  
  
r^{d-1}dr.  
  
Since:  
  
r\sim\ell^{1/3},  
  
one obtains:  
  
r^{d-1}dr  
\sim  
\ell^{(d-1)/3}\ell^{-2/3}d\ell  
=  
\ell^{(d-3)/3}d\ell.  
  
Thus the heuristic exponent in dimension d** is:**  
  
\frac{d-3}{3}.  
  
For:  
  
d=2,  
  
this gives:  
  
-\frac13.  
  
For:  
  
d=3,  
  
this gives:  
  
0.  
  
For:  
  
d=1,  
  
this gives:  
  
-\frac23.  
  
However, the one-dimensional case has different pairing structure and known rainflow-specific behavior, so this heuristic should not be asserted as a theorem across dimensions without separate proof.  
  
The current program is explicitly two-dimensional.  
  
⸻  
  
**250. Candidate Pair Intensity Measure**  
  
Let:  
  
\mathcal{P}_r  
  
denote candidate maximum-saddle fold pairs with separation in [r,r+dr]**.**  
  
Let:  
  
b=f(x_M)  
  
be the birth height.  
  
Let:  
  
\kappa=\frac{6(f(x_M)-f(x_S))}{r^3}  
  
be the normalized fold modulus.  
  
The expected number of candidate pairs in:  
  
x_M\in dx,  
\quad  
\theta\in d\theta,  
\quad  
r\in dr,  
\quad  
b\in db,  
\quad  
\kappa\in d\kappa  
  
has the asymptotic form:  
  
d\mathbb{E}N_{\mathrm{cand}}  
=  
L^2  
\,  
\rho_{\mathrm{cand}}(b,\kappa,\theta)  
\,  
r\,dr\,d\theta\,db\,d\kappa  
\,  
(1+o(1)).  
  
Here:  
  
\rho_{\mathrm{cand}}(b,\kappa,\theta)  
  
is the limiting candidate fold-pair intensity density.  
  
In isotropic cases, it is independent of \theta**.**  
  
In anisotropic cases, it depends on direction.  
  
The program’s pair-density input is:  
  
0<\rho_{\mathrm{cand}}(b,\kappa,\theta)<\infty  
  
on compact admissible windows.  
  
⸻  
  
**251. True Pair Intensity**  
  
The expected number of actual persistence pairs has density:  
  
d\mathbb{E}N_{\mathrm{true}}  
=  
q(r,b,\kappa,\theta)  
\,  
d\mathbb{E}N_{\mathrm{cand}}.  
  
Theorem A gives:  
  
q(r,b,\kappa,\theta)\to1  
  
uniformly over compact:  
  
b\in B,  
\qquad  
\kappa\in K\subset(0,\infty),  
\qquad  
\theta\in S^1.  
  
Therefore:  
  
d\mathbb{E}N_{\mathrm{true}}  
=  
L^2  
\,  
\rho_{\mathrm{cand}}(b,\kappa,\theta)  
\,  
r\,dr\,d\theta\,db\,d\kappa  
\,  
(1+o(1)).  
  
⸻  
  
**252. Change of Variables to Lifetime**  
  
Use:  
  
\ell=\frac{\kappa}{6}r^3.  
  
For fixed \kappa**:**  
  
r\,dr  
=  
2\cdot6^{-1/3}\kappa^{-2/3}\ell^{-1/3}d\ell.  
  
Thus:  
  
d\mathbb{E}N_{\mathrm{true}}  
=  
L^2  
\,  
\rho_{\mathrm{cand}}(b,\kappa,\theta)  
\,  
2\cdot6^{-1/3}\kappa^{-2/3}  
\ell^{-1/3}  
d\ell\,d\theta\,db\,d\kappa  
\,  
(1+o(1)).  
  
Integrating over:  
  
b\in B,  
\qquad  
\theta\in S^1,  
\qquad  
\kappa\in(0,\infty)  
  
with the appropriate admissible density yields:  
  
\nu_B(\ell)  
=  
C_B\ell^{-1/3}(1+o(1)).  
  
⸻  
  
**253. Constant Formula**  
  
The constant has the schematic form:  
  
C_B  
=  
L^2  
\,  
2\cdot6^{-1/3}  
\int_B  
\int_{S^1}  
\int_0^\infty  
\rho_{\mathrm{cand}}(b,\kappa,\theta)  
\,  
\kappa^{-2/3}  
\,d\kappa\,d\theta\,db.  
  
If the theorem restricts to a compact \kappa**-window:**  
  
K=[\kappa_-,\kappa_+]\subset(0,\infty),  
  
then:  
  
C_{B,K}  
=  
L^2  
\,  
2\cdot6^{-1/3}  
\int_B  
\int_{S^1}  
\int_K  
\rho_{\mathrm{cand}}(b,\kappa,\theta)  
\,  
\kappa^{-2/3}  
\,d\kappa\,d\theta\,db.  
  
The full constant requires controlling:  
  
\kappa\downarrow0  
  
and  
  
\kappa\to\infty.  
  
For the core theorem, it is safer to first state the compact-\kappa** version, then add tail lemmas if a full-integral constant is claimed.**  
  
⸻  
  
**254. Why Compact **\kappa** Windows Are Cleaner**  
  
The fold modulus:  
  
\kappa=\frac{6\ell}{r^3}  
  
can in principle approach zero or infinity.  
  
The near-fold asymptotic is most uniform when:  
  
0<\kappa_-\le\kappa\le\kappa_+<\infty.  
  
If:  
  
\kappa\downarrow0,  
  
the fold becomes too flat longitudinally.  
  
If:  
  
\kappa\to\infty,  
  
the height gap is too large relative to separation for the same normal form constants.  
  
Thus the safest theorem states:  
  
For every compact K\subset(0,\infty)**,**  
  
\nu_{B,K}(\ell)  
=  
C_{B,K}\ell^{-1/3}(1+o(1)).  
  
Then one separately proves the \kappa**-tail contribution is negligible or integrable.**  
  
⸻  
  
**255. Full Constant Positivity**  
  
To claim:  
  
C_B>0,  
  
one must prove:  
  
1. there exists a set of b,\kappa,\theta with positive measure,  
2. the candidate pair intensity is positive there,  
3. the Hessian index conditions have positive probability,  
4. adjacency has positive limiting probability,  
5. the selection probability tends to one,  
6. the \kappa^{-2/3} integral is finite.  
  
The positivity of candidate pair intensity follows from finite-jet nondegeneracy plus existence of admissible Hessian configurations.  
  
But adjacency positivity is a geometric condition and should be stated carefully.  
  
For an upper/lower asymptotic constant, adjacency cannot be dropped.  
  
⸻  
  
**256. Pair-Density Input**  
  
The needed pair-density proposition is:  
  
**Proposition B2 — Candidate Fold-Pair Intensity**  
  
Under H1+**, **H2'**, and compact windows **B,K**, the expected number of candidate maximum-saddle fold pairs with:**  
  
b\in B,  
\qquad  
\kappa\in K,  
\qquad  
r\in[r_1,r_2],  
  
has density:  
  
d\mathbb{E}N_{\mathrm{cand}}  
=  
L^2  
\,  
\rho_{\mathrm{cand}}(b,\kappa,\theta)  
\,  
r\,dr\,d\theta\,db\,d\kappa  
\,  
(1+o(1))  
  
as r\to0**, with:**  
  
0<c\le \rho_{\mathrm{cand}}\le C<\infty  
  
on compact admissible windows.  
  
Status:  
  
\textbf{PROVEN-MODULO two-point degenerating Kac–Rice density calculation}.  
  
This proposition is separate from Theorem A.  
  
⸻  
  
**257. Why Pair Density Is Not Trivial**  
  
The two critical points collide.  
  
The gradient constraints:  
  
\nabla f(x_M)=0,  
\qquad  
\nabla f(x_S)=0  
  
become degenerate as:  
  
r\to0.  
  
The height gap:  
  
\ell=O(r^3)  
  
also imposes a singular value constraint.  
  
The Hessian determinant product collapses because one Hessian eigenvalue is:  
  
O(r)  
  
at each point.  
  
The raw Gaussian density may blow up while determinant factors vanish.  
  
The proposition says these effects balance to produce contact neutrality:  
  
\rho_{\mathrm{cand}}(r;b,\kappa,\theta)\to\rho_{\mathrm{cand}}(b,\kappa,\theta)  
  
finite and nonzero.  
  
This is a nontrivial two-point Kac–Rice result.  
  
⸻  
  
**258. Contact Neutrality**  
  
Contact neutrality means that after expressing candidate pairs in radial separation measure:  
  
r\,dr,  
  
there is no extra vanishing or divergence in the pair density.  
  
Symbolically:  
  
\rho_{\mathrm{cand}}(r)\sim\rho_0.  
  
If instead:  
  
\rho_{\mathrm{cand}}(r)\sim r^\beta,  
  
then the near-diagonal exponent would change.  
  
Indeed:  
  
r^\beta r\,dr  
\sim  
\ell^{\beta/3}\ell^{-1/3}d\ell.  
  
The exponent would become:  
  
\frac{\beta-1}{3}.  
  
Thus Theorem B depends critically on \beta=0**.**  
  
This is why pair-density nondegeneration must be proved, not assumed casually.  
  
⸻  
  
**259. Relationship Between Theorem A and Proposition B2**  
  
Theorem A:  
  
q_0=1  
  
is a selection theorem.  
  
Proposition B2:  
  
\rho_{\mathrm{cand}}(r)\to\rho_0  
  
is a pair-intensity theorem.  
  
They are logically independent.  
  
Theorem B needs both:  
  
\nu(\ell)  
\sim  
\rho_{\mathrm{cand}}(r(\ell))  
q(r(\ell))  
r\frac{dr}{d\ell}.  
  
As:  
  
r\to0,  
  
\rho_{\mathrm{cand}}\to\rho_0,  
  
q\to1,  
  
r\frac{dr}{d\ell}\sim C\ell^{-1/3}.  
  
Therefore:  
  
\nu(\ell)\sim C_*\ell^{-1/3}.  
  
⸻  
  
**260. Theorem B — Compact-**\kappa** Version**  
  
**Theorem B1 — Compact-Modulus Near-Diagonal Law**  
  
Let f** be a centered stationary Gaussian field on **\mathbb{T}_L^2** satisfying **H1+**, **H2'**, and Assumption MS. Let **B\subset\mathbb{R}** be a compact birth-height window and **K\subset(0,\infty)** a compact fold-modulus window.**  
  
Let \nu_{B,K}(\ell)** denote the expected density of non-essential **H_0** persistence bars with:**  
  
f(x_M)\in B,  
  
\kappa=\frac{6\ell}{r^3}\in K,  
  
and lifetime in:  
  
[\ell,\ell+d\ell].  
  
Then:  
  
\nu_{B,K}(\ell)  
=  
C_{B,K}\ell^{-1/3}(1+o(1))  
  
as:  
  
\ell\downarrow0,  
  
where:  
  
0<C_{B,K}<\infty.  
  
The constant is:  
  
C_{B,K}  
=  
L^2  
\,  
2\cdot6^{-1/3}  
\int_B  
\int_{S^1}  
\int_K  
\rho_{\mathrm{cand}}(b,\kappa,\theta)  
\kappa^{-2/3}  
\,d\kappa\,d\theta\,db.  
  
Status:  
  
\textbf{PROVEN-MODULO Theorem A and Proposition B2}.  
  
⸻  
  
**261. Theorem B — Full-Modulus Version**  
  
**Theorem B2 — Full Near-Diagonal Law**  
  
Assume, in addition to the hypotheses of Theorem B1, that the candidate fold-pair intensity satisfies the tail integrability condition:  
  
\int_0^\infty  
\rho_{\mathrm{cand}}(b,\kappa,\theta)  
\kappa^{-2/3}  
d\kappa  
<\infty  
  
uniformly over:  
  
b\in B,  
\qquad  
\theta\in S^1,  
  
and that the \kappa**-tail contributions are asymptotically negligible outside compact windows.**  
  
Then:  
  
\nu_B(\ell)  
=  
C_B\ell^{-1/3}(1+o(1)).  
  
Status:  
  
\textbf{PROVEN-MODULO additional \(\kappa\)-tail lemmas}.  
  
The compact-\kappa** version should be treated as the primary theorem until the tail lemmas are fully verified.**  
  
⸻  
  
**262. Cumulative Version**  
  
From:  
  
\nu_{B,K}(\ell)  
=  
C_{B,K}\ell^{-1/3}(1+o(1)),  
  
one obtains:  
  
N_{B,K}(0,\varepsilon)  
=  
\mathbb{E}  
\#\{H_0\text{ bars}:\ell\le\varepsilon,\ b\in B,\ \kappa\in K\}.  
  
Then:  
  
N_{B,K}(0,\varepsilon)  
=  
\int_0^\varepsilon  
C_{B,K}\ell^{-1/3}(1+o(1))d\ell.  
  
Therefore:  
  
N_{B,K}(0,\varepsilon)  
=  
\frac32 C_{B,K}\varepsilon^{2/3}(1+o(1)).  
  
This cumulative version is often more robust because it avoids differentiability issues in the persistence intensity measure.  
  
⸻  
  
**263. Measure-Theoretic Form**  
  
The most rigorous statement may be vague-convergence of measures.  
  
Define the rescaled lifetime measure:  
  
\mu_\varepsilon(A)  
=  
\varepsilon^{-2/3}  
\mathbb{E}  
\#\{  
\text{bars}:  
\ell\in\varepsilon A,\ b\in B,\ \kappa\in K  
\}.  
  
Then for intervals:  
  
A=[a_1,a_2]\subset(0,\infty),  
  
Theorem B becomes:  
  
\mu_\varepsilon(A)  
\to  
C_{B,K}  
\int_A u^{-1/3}du.  
  
That is:  
  
\mu_\varepsilon(A)  
\to  
\frac32 C_{B,K}  
(a_2^{2/3}-a_1^{2/3}).  
  
This avoids assuming pointwise density exists.  
  
The final monograph should present this measure form first, then state the density form under regularity.  
  
⸻  
  
**264. Why Theorem B Is Not a Consequence of 1D Rainflow Theory**  
  
In one-dimensional stationary processes, persistence pairing of local maxima and minima relates to rainflow counting and extrema sequences.  
  
There are known analytic and semi-analytic tools for 1D:  
  
* Markov chains of extrema,  
* rainflow cycle distributions,  
* level-crossing methods,  
* Brownian special cases.  
  
But the current theorem is two-dimensional and involves:  
  
1. local maxima,  
2. index-one saddles,  
3. merge-tree topology,  
4. gradient separatrix adjacency,  
5. elder-rule component merging,  
6. local versus global path obstruction,  
7. three-point Kac–Rice in the plane.  
  
The 1D problem has a total order on the line.  
  
The 2D problem has no such order.  
  
A local maximum-saddle adjacency in 2D is not automatically a persistence pair.  
  
That is exactly why q_0=1** is needed.**  
  
Thus 1D rainflow laws are antecedents, not proofs.  
  
⸻  
  
**265. Why Point-Process Persistence Does Not Prove Theorem B**  
  
Random geometric complexes and point-process persistence have near-diagonal asymptotics in some regimes.  
  
But those models are not smooth Gaussian fields.  
  
Their persistence pairs arise from:  
  
* random points,  
* geometric balls,  
* Čech/Rips filtrations,  
* combinatorial critical simplices.  
  
The current theorem arises from:  
  
* smooth critical points,  
* Hessian determinants,  
* Morse saddles,  
* Gaussian spectral moments,  
* fold normal forms.  
  
The exponent may resemble geometric contact exponents, but the mechanism is different.  
  
Theorem B is a smooth-field critical-pair law.  
  
⸻  
  
**266. Constant Dependence on Spectral Geometry**  
  
The exponent:  
  
-\frac13  
  
is universal within the theorem class.  
  
The constant:  
  
C_*  
  
is not universal.  
  
It depends on the spectral measure through:  
  
1. gradient covariance,  
2. Hessian covariance,  
3. third-jet covariance,  
4. pair-frame limiting covariance,  
5. Hessian index probabilities,  
6. adjacency probability,  
7. fold modulus density,  
8. anisotropic direction dependence.  
  
In isotropic fields, these reduce to scalar spectral moments:  
  
\lambda_2,\lambda_4,\lambda_6,\ldots  
  
In anisotropic fields, they remain tensorial.  
  
Thus Theorem B is universal in exponent but model-dependent in constant.  
  
⸻  
  
**267. Anisotropic Version**  
  
For anisotropic fields, the direction variable:  
  
\theta\in S^1  
  
must be retained.  
  
The pair axis direction affects:  
  
* pair-frame covariance,  
* fold modulus density,  
* Hessian sign probability,  
* adjacency geometry,  
* local spectral tensor contractions.  
  
The constant becomes:  
  
C_{B,K}  
=  
L^2  
2\cdot6^{-1/3}  
\int_B  
\int_{S^1}  
\int_K  
\rho_{\mathrm{cand}}(b,\kappa,\theta)  
\kappa^{-2/3}  
d\kappa\,d\theta\,db.  
  
The exponent remains:  
  
-\frac13.  
  
This is a major upgrade over an isotropic-only theorem.  
  
⸻  
  
**268. Boundary of Validity**  
  
Theorem B does not claim:  
  
1. finite-\ell density formula,  
2. exact distribution of all persistence lifetimes,  
3. independence of bars,  
4. Poisson limit,  
5. thermodynamic-limit law,  
6. non-Gaussian universality,  
7. higher-dimensional theorem,  
8. H_1 persistence law,  
9. exact finite-r selection probability,  
10. closed-form constant for arbitrary covariance.  
  
It is a local asymptotic intensity theorem.  
  
⸻  
  
**269. Thermodynamic Limit Caveat**  
  
The theorem is fixed L**.**  
  
If L\to\infty**, then the expected number of small bars scales with area:**  
  
L^2.  
  
The local intensity per unit area may still have a limit, but global elder-rule effects and far-zone obstructions require separate treatment.  
  
A possible infinite-volume version would state:  
  
\frac{1}{L^2}  
\nu_{B,K,L}(\ell)  
\to  
c_{B,K}\ell^{-1/3}  
  
under a limit order:  
  
\ell\downarrow0  
\quad\text{first, then}\quad  
L\to\infty,  
  
or with a joint condition.  
  
But this is not part of the current theorem.  
  
⸻  
  
**270. Proof Skeleton for Theorem B**  
  
The proof has six steps.  
  
**Step 1 — Candidate pair representation**  
  
Represent small-lifetime bars by candidate maximum-saddle fold pairs plus a selection indicator:  
  
1_{\{D(x_M)=x_S\}}.  
  
**Step 2 — Kac–Rice pair intensity**  
  
Use two-point Kac–Rice to express expected candidate-pair count.  
  
Obtain:  
  
d\mathbb{E}N_{\mathrm{cand}}  
=  
L^2\rho_{\mathrm{cand}}(b,\kappa,\theta)  
r\,dr\,d\theta\,db\,d\kappa(1+o(1)).  
  
**Step 3 — Selection factor**  
  
Insert:  
  
q(r,b,\kappa,\theta)  
=  
\mathbb{P}(D(x_M)=x_S).  
  
Theorem A gives:  
  
q(r,b,\kappa,\theta)=1+o(1).  
  
**Step 4 — Fold change of variables**  
  
Use:  
  
\ell=\frac{\kappa}{6}r^3.  
  
Convert:  
  
r\,dr  
=  
2\cdot6^{-1/3}\kappa^{-2/3}  
\ell^{-1/3}d\ell.  
  
**Step 5 — Integrate marks**  
  
Integrate over:  
  
b,\theta,\kappa.  
  
**Step 6 — Conclude**  
  
Obtain:  
  
\nu_{B,K}(\ell)  
=  
C_{B,K}\ell^{-1/3}(1+o(1)).  
  
⸻  
  
**271. Potential Failure Modes of Theorem B**  
  
Theorem B could fail if:  
  
1. candidate pair intensity has an extra power of r,  
2. candidate pair intensity diverges,  
3. q_0\ne1,  
4. the fold relation is not cubic,  
5. \kappa-tail dominates the compact-\kappa contribution,  
6. adjacency probability vanishes,  
7. global essential bars contaminate the count,  
8. finite-volume assumptions fail,  
9. Morse–Smale genericity fails,  
10. the persistence intensity does not admit the stated density form.  
  
The architecture addresses these by separating the theorem into named dependencies.  
  
⸻  
  
**272. Theorem B Status**  
  
Current status:  
  
\textbf{PROVEN-MODULO}.  
  
Dependencies:  
  
1. Theorem A q_0=1,  
2. Proposition B2 candidate pair intensity/contact neutrality,  
3. fold normal form,  
4. compact-\kappa restriction or tail integrability,  
5. measure/density regularity,  
6. non-essential bar exclusion,  
7. fixed-L setting.  
  
The exponent derivation is stable.  
  
The constant requires careful final normalization.  
  
⸻  
  
**273. Immediate Next Reconstruction Step**  
  
The next installment should reconstruct the literature verification and novelty boundary.  
  
That chapter must distinguish:  
  
1. what classical Kac–Rice provides,  
2. what Gaussian random-field Morse theory provides,  
3. what 1D rainflow/persistence theory provides,  
4. what random geometric complex persistence provides,  
5. what the archive actually adds,  
6. what remains unverified,  
7. where citation claims must be softened.  
  
## Unified Master Source — Part 9  
## Literature Verification, Novelty Boundary, and Citation Discipline  
   
⸻  
   
## 274. Purpose of the Literature Boundary Layer  
The C006 / q₀ program must distinguish sharply between four categories:  
1. **Known classical machinery** Tools already established in the literature.  
2. **Known adjacent results** Results that resemble parts of the program but do not prove the target theorem.  
3. **Adaptable ingredients** Methods that can plausibly be adapted but require new execution.  
4. **New contribution / open synthesis** The actual mathematical object not located as an existing theorem.  
This distinction matters because the program’s credibility depends on not overstating novelty.  
The correct novelty claim is not:  
\text{“No one has studied Gaussian fields, persistence, or Kac–Rice.”}  
That would be false.  
The correct claim is:  
\[ \text{“No located source proves the elder-rule selection constant }q_0=1 \text{ or the resulting } \ell^{-1/3} \text{ near-diagonal }H_0\text{ bar-density law for smooth stationary Gaussian fields in }d=2.”} \]  
The program is a synthesis of known tools into a specific theorem that appears absent from the literature.  
   
⸻  
   
## 275. What Classical Kac–Rice Provides  
Classical Kac–Rice theory provides formulas for expected counts of zeros of random fields.  
For a smooth Gaussian field F:M\to\mathbb{R}^k, Kac–Rice expresses the expected number of zeros as an integral over the domain:  
\mathbb{E}N_F(A) = \int_A p_{F(x)}(0) \mathbb{E} \left[ |\det DF(x)| \mid F(x)=0 \right] dx.  
For critical points of a scalar Gaussian field f, take:  
F=\nabla f.  
Then:  
DF=D^2f.  
Thus:  
\mathbb{E}N_{\mathrm{crit}}(A) = \int_A p_{\nabla f(x)}(0) \mathbb{E} \left[ |\det D^2f(x)| \mid \nabla f(x)=0 \right] dx.  
Multi-point Kac–Rice similarly gives factorial moment densities for multiple critical points.  
Therefore the following are not new:  
1. counting critical points by Kac–Rice,  
2. including Hessian determinant factors,  
3. conditioning Gaussian jets,  
4. deriving one-point critical height densities,  
5. writing multi-point critical-point intensities.  
The q₀ program uses these tools but does not claim to invent them.  
   
⸻  
   
## 276. What Classical Kac–Rice Does Not Provide Automatically  
Classical Kac–Rice does not automatically prove the q₀ theorem.  
The missing features are:  
1. **colliding two-point maximum-saddle conditioning** The two critical points approach each other at separation r\to0.  
2. **height gap scaling** The height gap is not arbitrary. It scales as: \ell\asymp r^3.   
3. **Palm conditioning** The relevant law is determinant-weighted critical-pair Palm conditioning, not ordinary Gaussian conditioning.  
4. **elder-rule topology** Kac–Rice counts critical points. It does not decide persistence pairings.  
5. **three-point obstruction analysis** The dangerous event is a third critical point appearing near the pair at scale: r^2.   
6. **two-scale degeneration** The configuration has simultaneous scales r and r^2.  
7. **flat-saddle tail integration** Averaging over Palm base variables changes exponential suppression into polynomial suppression.  
8. **near-diagonal persistence density** Translating critical-pair intensities into persistence-diagram density requires the elder-rule selection theorem.  
Thus Kac–Rice is foundational but insufficient.  
   
⸻  
   
## 277. What Gaussian Critical-Point Literature Provides  
Gaussian random-field literature provides many relevant ingredients:  
1. expected number of critical points,  
2. critical-point height distributions,  
3. maxima/minima/saddle densities,  
4. spectral-moment formulas,  
5. nondegeneracy assumptions for Morse fields,  
6. one-point and sometimes two-point critical correlation functions,  
7. Kac–Rice regularity conditions,  
8. smoothness assumptions via covariance derivatives.  
These ingredients support the field-theoretic side of the program.  
The q₀ program depends on this body of work for legitimacy of:  
\text{critical-point intensity calculus}.  
However, critical-point counts alone do not encode persistence pairing.  
A critical maximum and a critical saddle becoming close does not by itself say the maximum dies at that saddle.  
That is a topological statement, not merely a local Gaussian statement.  
   
⸻  
   
## 278. What Gaussian Morse Theory Provides  
Standard random-field arguments support:  
1. almost-sure smoothness under spectral moment assumptions,  
2. almost-sure nondegenerate critical points under finite-jet nondegeneracy,  
3. almost-sure distinct critical values under joint-density conditions,  
4. finite critical-point count on compact domains.  
Thus the archive can reasonably treat the following as literature-supported or standard:  
f\text{ is almost surely Morse}.  
f\text{ has distinct critical values almost surely}.  
But the stronger statement:  
f\text{ is almost surely Morse–Smale}  
is more delicate.  
Morse–Smale requires global flow transversality, not just finite-jet nondegeneracy at critical points.  
The key issue is saddle-saddle separatrix connections.  
A saddle connection is determined by an entire gradient-flow trajectory. It is not a finite-dimensional jet event.  
Therefore a proof of almost-sure Morse–Smale requires global-flow/transversality machinery, not ordinary local Kac–Rice alone.  
The archive’s correct stance is:  
\text{Morse and distinct values: literature-supported under hypotheses.}  
\text{Morse–Smale: named proof obligation or explicit assumption.}  
   
⸻  
   
## 279. Recommended Citation Language for Morse–Smale  
The final monograph should avoid saying:  
\text{“By standard Gaussian field theory, }f\text{ is Morse–Smale almost surely.”}  
That is too strong unless a precise theorem is cited or proved.  
Use instead:  
\text{“Under the stated finite-jet nondegeneracy hypotheses, the field is almost surely Morse and has distinct critical values. The additional Morse–Smale/no-saddle-connection property is imposed as Assumption MS in the main theorem and discussed separately in Appendix MS.”}  
If Appendix MS is completed, the statement can be upgraded.  
This protects the theorem from an avoidable literature overclaim.  
   
⸻  
   
## 280. What Persistence Theory Provides  
Persistent homology provides:  
1. filtrations,  
2. persistence modules,  
3. barcode/persistence-diagram representation,  
4. elder-rule pairing in H_0,  
5. merge-tree interpretation,  
6. stability theory,  
7. algorithms for persistence pairing,  
8. Morse-theoretic descriptions of persistence pairs for tame functions.  
For smooth Morse functions on compact manifolds, H_0 persistence is well understood at the algorithmic/topological level.  
The elder rule is classical.  
The merge tree formulation is classical.  
The q₀ program does not claim to invent H_0 persistence or elder-rule pairing.  
Its new claim concerns the **probability law** of near-diagonal elder-rule pairing for a smooth Gaussian random field.  
   
⸻  
   
## 281. What Persistence Theory Does Not Provide Automatically  
Classical persistence theory is deterministic.  
It tells how to compute pairings once the function is known.  
It does not give:  
1. the distribution of Gaussian-field persistence lifetimes,  
2. the near-diagonal density exponent,  
3. the probability that a local maximum-saddle pair is an elder-rule pair,  
4. the Kac–Rice intensity of persistence pairs,  
5. the local selection constant q_0.  
Thus deterministic persistence theory supplies the language and topology, but not the probabilistic law.  
The q₀ program supplies the probabilistic bridge.  
   
⸻  
   
## 282. What One-Dimensional Rainflow Theory Provides  
In one dimension, persistence pairing for functions and stochastic processes is closely related to rainflow counting.  
For a one-dimensional signal, local maxima and minima occur in a linear order.  
This order makes pairing combinatorially simpler.  
Rainflow theory provides tools for:  
1. cycles of one-dimensional processes,  
2. extrema sequences,  
3. Markov chains of extrema,  
4. fatigue/rainflow cycle distributions,  
5. stationary Gaussian process cycle analysis.  
These results are genuine antecedents.  
They show that stochastic persistence/rainflow distributions can be studied analytically in one dimension.  
   
⸻  
   
## 283. Why 1D Rainflow Does Not Prove the 2D Theorem  
The two-dimensional problem is different.  
In d=1:  
* critical points alternate maxima/minima along a line,  
* pairing is controlled by an ordered extrema sequence,  
* there are no saddle separatrices,  
* there is no two-dimensional merge-tree branching geometry,  
* there is no local loop/crater obstruction,  
* there is no angular radial measure r\,dr.  
In d=2:  
* maxima pair with saddles,  
* saddles have multiple separatrix branches,  
* component identity is global,  
* local gradient adjacency is not automatically persistence pairing,  
* third critical points can enter the local fold cell,  
* far paths can reconnect components,  
* the near-diagonal exponent comes from r\,dr and \ell\asymp r^3.  
Therefore 1D rainflow laws are not a proof of:  
q_0=1  
or:  
\nu(\ell)\sim C\ell^{-1/3}  
for smooth Gaussian fields on \mathbb{T}^2.  
They are analogical and methodological antecedents only.  
   
⸻  
   
## 284. What Random Geometric Complex Persistence Provides  
Random geometric complex theory studies persistence of filtrations built from random point clouds, such as:  
* Čech complexes,  
* Vietoris–Rips complexes,  
* alpha complexes,  
* union-of-balls filtrations.  
This literature contains near-diagonal or small-lifetime asymptotics in some settings.  
It provides useful conceptual parallels:  
1. local configurations dominate small persistence,  
2. scaling exponents arise from geometric contact,  
3. Palm-like local point-process analysis is natural,  
4. critical configurations can determine persistence-pair intensities.  
But these models are not smooth Gaussian random fields.  
Their critical objects are simplices or point configurations, not Morse critical points.  
Their local geometry is distance-function geometry, not Gaussian fold geometry.  
Thus point-process persistence does not prove the q₀ theorem.  
   
⸻  
   
## 285. Smooth Field Versus Point Cloud Mechanism  
For point clouds:  
\text{small persistence}  
often arises from nearly degenerate simplex configurations.  
For smooth Gaussian fields:  
\text{small }H_0\text{ persistence}  
arises from nearby maximum-saddle fold pairs.  
The controlling relation is:  
\ell\asymp r^3.  
This cubic law is a smooth catastrophe/fold effect.  
It is not the same as a point-cloud edge-length or simplex-radius degeneracy.  
Therefore even if exponents resemble each other, the proof architecture is different.  
   
⸻  
   
## 286. What Gaussian Kinematic Formula Provides  
The Gaussian kinematic formula and related random-field topology results describe expected geometric/topological features of excursion sets.  
They can provide:  
1. expected Euler characteristic,  
2. Lipschitz–Killing curvature expectations,  
3. excursion probability approximations,  
4. high-threshold topology behavior,  
5. global summaries of random-field geometry.  
These are relevant background.  
But they do not provide the local elder-rule persistence-pair density near the diagonal.  
Euler characteristic counts alternating critical points by index; it does not pair maxima with saddles under the elder rule.  
The q₀ program is finer than Euler characteristic.  
It asks not merely how many maxima and saddles exist, but which maximum is killed by which saddle.  
   
⸻  
   
## 287. What Critical-Point Correlation Literature Provides  
Two-point and multi-point critical correlation functions for Gaussian fields are relevant.  
They help understand:  
1. how critical points repel or cluster,  
2. how Hessian signs correlate,  
3. how intensities behave as points collide,  
4. determinant factors under joint critical conditioning.  
However, the q₀ program requires a specific marked pair:  
\text{maximum at }x_M,\quad \text{saddle at }x_S,  
with:  
f(x_M)-f(x_S)\asymp r^3,  
plus elder-rule selection.  
A generic critical correlation function does not include:  
D(x_M)=x_S.  
Therefore it is an ingredient, not the theorem.  
   
⸻  
   
## 288. What Divided-Difference Literature Provides  
When Gaussian observations collide, covariance matrices become singular.  
Divided differences are a known way to desingularize colliding evaluations.  
The literature supplies general techniques for:  
1. replacing nearby function values by normalized differences,  
2. obtaining finite limiting covariance matrices,  
3. handling zeros/critical points near collision,  
4. controlling determinant blow-up.  
The archive uses this idea.  
The new execution is specialized:  
1. two critical points collapse at scale r,  
2. height gap is r^3,  
3. third critical point collapses to one of them at scale r^2,  
4. pair determinant weights enter,  
5. elder-rule obstruction is the target.  
Thus the divided-difference method is known, but the exact nested two-scale implementation is part of the program.  
   
⸻  
   
## 289. What Conditional Determinant Literature Provides  
Gaussian conditioning plus Hessian determinant factors is standard in Kac–Rice.  
Existing work provides formulas and estimates for:  
\mathbb{E}[|\det H|\mid \text{Gaussian constraints}].  
The q₀ program uses this in three places:  
1. one-point critical densities,  
2. two-point maximum-saddle pair densities,  
3. three-point obstruction counts.  
The novelty is not determinant conditioning itself.  
The novelty is the determinant power ledger under near-fold Palm conditioning, especially the corrected flat-saddle tail:  
|\det H_S||\det H_M|\asymp r^5  
on the dangerous flat set.  
That correction is specific to this proof architecture.  
   
⸻  
   
## 290. What the Archive Adds  
The archive’s contribution can be summarized as the following synthesis:  
\boxed{ \text{Smooth Gaussian critical-point Kac–Rice} + \text{Morse }H_0\text{ elder rule} + \text{near-fold maximum-saddle Palm law} + \text{GCJA conditional-jet filtration} + \text{three-point two-scale obstruction control} \Rightarrow q_0=1 }  
Then:  
\boxed{ q_0=1 + \text{candidate pair contact neutrality} + \ell=\kappa r^3/6 + r\,dr \Rightarrow \nu(\ell)\sim C\ell^{-1/3}. }  
This is the actual program.  
The novelty is the bridge, not the individual bricks.  
   
⸻  
   
## 291. Novel Contribution 1 — Elder-Rule Selection Constant  
The first central contribution is:  
q_0=1.  
This says that a near-diagonal maximum-saddle fold pair is selected by the elder rule with probability tending to one.  
This is not a direct Kac–Rice count.  
It requires proving that all alternate pairing mechanisms vanish.  
That includes:  
1. inner third critical points,  
2. annulus critical points,  
3. far critical points in the height band,  
4. loop/crater obstructions,  
5. global reconnections,  
6. Morse–Smale degeneracies.  
The local-to-global topology bridge is the distinctive feature.  
   
⸻  
   
## 292. Novel Contribution 2 — Near-Diagonal Density Exponent  
The second central contribution is:  
\nu(\ell)\sim C\ell^{-1/3}.  
The exponent comes from:  
\ell\asymp r^3  
and:  
d^2h=r\,dr\,d\theta.  
But this geometric calculation only becomes a persistence theorem after:  
q_0=1  
is proved.  
Without Theorem A, the exponent could be modified by a nontrivial selection factor.  
Thus Theorem B depends on Theorem A.  
   
⸻  
   
## 293. Novel Contribution 3 — GCJA Filtration Theorem  
The GCJA theorem is a methodological contribution.  
It generalizes the conditional covariance computations needed by the program.  
It says:  
\text{collapsing Gaussian observation designs}  
are governed by:  
\text{Hermite interpolation} + \text{visible/unseen spectral jet projection}.  
This theorem may be useful beyond the q₀ problem.  
It gives a general way to analyze near-collision Gaussian conditioning without recomputing symbolic covariance matrices each time.  
   
⸻  
   
## 294. Novel Contribution 4 — Flat-Saddle Tail Correction  
The flat-saddle tail correction is both a mathematical result and a methodological warning.  
It shows:  
\text{generic fixed-base suppression} \neq \text{Palm-averaged suppression}.  
At fixed generic pair data, the inner third-point critical probability is exponentially small.  
But after integrating over determinant-weighted pair data, near-flat saddles dominate.  
The correct averaged upper bound is polynomial:  
\mathbb{E}N_{\mathrm{inner}}\le Cr^5(1+o(1)).  
This is an important internal correction.  
It prevents the proof from relying on a false super-exponential claim.  
   
⸻  
   
## 295. Novel Contribution 5 — Anisotropy Bridge  
Earlier drafts used isotropic notation.  
The current architecture shows isotropy is not structurally necessary.  
The correct condition is finite spectral moment nondegeneracy.  
The GCJA theorem works with an arbitrary spectral measure \rho satisfying:  
1. symmetry,  
2. finite moments,  
3. polynomial nondegeneracy.  
Therefore the theorem can be stated for anisotropic stationary Gaussian fields, with direction-dependent constants.  
This is a meaningful strengthening.  
   
⸻  
   
## 296. Citation Claims That Must Be Softened  
The final manuscript should soften any claim of the form:  
\text{“This is the first...”}  
unless a full literature search is complete.  
Safer language:  
\text{“We have not located a result proving...”}  
or:  
\text{“Existing literature supplies the following ingredients, but we do not know of a source combining them to prove...”}  
or:  
\text{“To the author’s knowledge, the specific elder-rule selection constant and near-diagonal law below do not appear in the existing literature.”}  
This is more defensible.  
   
⸻  
   
## 297. Citation Claims That Are Safe  
The following claims are safe if properly cited:  
1. Kac–Rice formulas exist for smooth Gaussian fields.  
2. Critical-point intensity formulas are standard under nondegeneracy.  
3. Morse functions give persistence pairings via critical points.  
4. H_0 persistence is represented by merge trees.  
5. One-dimensional rainflow theory studies cycles/extrema of stationary processes.  
6. Random geometric complex persistence has local small-persistence asymptotics.  
7. Gaussian kinematic formula addresses expected excursion-set geometry.  
8. Divided differences can desingularize colliding Gaussian evaluations.  
9. Almost-sure Morse properties follow under finite-jet nondegeneracy in standard random-field settings.  
These should be stated as background, not novelty.  
   
⸻  
   
## 298. Citation Claims That Require Care  
The following require careful treatment:  
## 298.1 Almost-sure Morse–Smale  
Do not cite vaguely.  
Either prove it or assume it.  
## 298.2 Exact two-scale three-point expansion  
Do not imply literature proves it unless a specific source is found.  
## 298.3 Near-diagonal \ell^{-1/3} law  
State as not located, not absolutely nonexistent.  
## 298.4 Elder-rule selection constant  
State as not located for smooth stationary Gaussian fields in d=2.  
## 298.5 Anisotropic extension  
Make clear that the extension follows from the archive’s GCJA theorem, not from isotropic formulas.  
   
⸻  
   
## 299. Literature Boundary Table  

| Topic | Literature Status | Program Use | Novelty Status |
| ------------------------------- | --------------------------- | ---------------------- | --------------------------- |
| Kac–Rice formula | Known | critical counts | background |
| Gaussian regression | Known | conditional laws | background |
| Hessian determinant weighting | Known | critical Palm laws | background |
| Critical-point height densities | Known | pair intensity context | background |
| Persistence elder rule | Known | deterministic pairing | background |
| Merge trees | Known | H_0 topology | background |
| 1D rainflow | Known | analogy/antecedent | not proof |
| Random geometric persistence | Known | analogy | not proof |
| Gaussian kinematic formula | Known | context | not enough |
| Colliding divided differences | Known/adaptable | desingularization | adapted |
| Two-point fold Palm law | Partly standard tools | pair intensity | new execution |
| Three-point r,r^2 Kac–Rice | Not located as exact result | inner obstruction | likely new |
| q_0=1 elder selection | Not located | Theorem A | central contribution |
| \\ell^{-1/3} H_0 density | Not located | Theorem B | central contribution |
| GCJA theorem | Not standard in this form | covariance engine | methodological contribution |
| Flat-saddle tail r^5 | Program-derived | inner bound | program contribution |
  
   
⸻  
   
## 300. What Would Falsify the Novelty Claim  
The novelty claim would be weakened if a source is found that proves any of the following for smooth stationary Gaussian fields in d=2:  
1. exact near-diagonal H_0 persistence lifetime density,  
2. elder-rule selection probability for collapsing maximum-saddle pairs,  
3. maximum-saddle Palm law plus persistence pairing asymptotics,  
4. three-point two-scale Kac–Rice obstruction bound for persistence,  
5. \ell^{-1/3} smooth-field bar-density law.  
If such a source exists, the program’s contribution shifts from discovery to reconstruction, clarification, anisotropic extension, or independent proof.  
The architecture remains useful, but novelty status changes.  
   
⸻  
   
## 301. What Would Not Falsify the Novelty Claim  
The following would not by itself falsify novelty:  
1. a general Kac–Rice theorem,  
2. a one-point critical height formula,  
3. a two-point critical correlation formula without persistence pairing,  
4. a 1D rainflow result,  
5. a point-cloud persistence asymptotic,  
6. a Gaussian kinematic formula result,  
7. a deterministic persistence algorithm,  
8. a Morse theory theorem about merge trees,  
9. a symbolic covariance computation for a special isotropic field,  
10. a general divided-difference lemma.  
These are ingredients, not the target theorem.  
   
⸻  
   
## 302. Recommended Literature-Audit Protocol  
For final manuscript integrity, every citation should be classified into one of five roles.  
## Role A — Direct theorem support  
The cited result directly proves a needed lemma.  
## Role B — Method support  
The cited result supplies a method adapted here.  
## Role C — Antecedent / analogy  
The cited result is related but does not prove the claim.  
## Role D — Boundary / contrast  
The cited result shows what is known in a neighboring setting.  
## Role E — Historical context  
The cited result motivates or situates the work.  
Each citation in the final paper should be labeled internally with one of these roles.  
This prevents accidental overclaiming.  
   
⸻  
   
## 303. Recommended Novelty Paragraph  
A safe final novelty paragraph would read:  
“The ingredients used below—Kac–Rice formulas, Gaussian regression, Morse-theoretic persistence pairing, merge trees, divided differences for colliding evaluations, and one-dimensional rainflow analogues—are classical or literature-supported. The point of the present work is not to reintroduce those tools, but to combine them in a specific near-diagonal regime for smooth stationary Gaussian fields on a two-dimensional torus. In particular, we have not located a source proving that a collapsing maximum-saddle fold pair is selected by the H_0 elder rule with limiting probability one, nor a source deriving the resulting \ell^{-1/3} near-diagonal persistence-lifetime intensity for this smooth-field setting. The proofs below isolate the required local-to-global selection problem, reduce failures to critical-point obstruction events, and control the most singular inner obstruction using a two-scale Kac–Rice/Palm analysis.”  
This paragraph is accurate and defensible.  
   
⸻  
   
## 304. Recommended Status Header for the Final Paper  
The final paper should include a status header:  
“Status. The main selection theorem is proved conditional on the stated Morse–Smale assumption and on the annulus nondegeneracy lemma. The inner-zone flat-saddle obstruction is controlled by an O(r^5) upper bound. The near-diagonal density theorem follows from the selection theorem and the candidate pair-intensity proposition. The compact-\kappa density law is the primary theorem; full-\kappa integration requires additional tail control.”  
This prevents hidden overclaiming.  
   
⸻  
   
## 305. Main Literature Risk Register  
## Risk 1 — Existing near-diagonal theorem found  
Impact: novelty reduced.  
Mitigation: present as independent proof / anisotropic extension / correction.  
## Risk 2 — Morse–Smale assumption unsupported  
Impact: Theorem A remains conditional.  
Mitigation: isolate Assumption MS and develop Appendix MS.  
## Risk 3 — Pair-density contact neutrality already known  
Impact: Proposition B2 becomes literature-supported rather than new.  
Mitigation: cite and use it.  
## Risk 4 — Three-point Kac–Rice expansion partially known  
Impact: File 3 novelty narrows.  
Mitigation: cite as method support and state the specific adaptation.  
## Risk 5 — 1D rainflow analogy overstated  
Impact: conceptual confusion.  
Mitigation: explicitly distinguish 1D total-order pairing from 2D merge-tree pairing.  
   
⸻  
   
## 306. Literature Boundary Conclusion  
The archive’s strongest defensible novelty claim is:  
\boxed{ \text{The specific elder-rule selection theorem }q_0=1 \text{ and the resulting } \ell^{-1/3} \text{ near-diagonal }H_0\text{ bar-density law for smooth stationary Gaussian fields in }d=2 \text{ are not located as existing results.} }  
The archive’s strongest methodological contribution is:  
\boxed{ \text{The GCJA theorem and the two-scale Kac–Rice/Palm obstruction analysis provide a reusable proof engine for collapsing Gaussian critical-point designs.} }  
The archive’s most important caution is:  
\boxed{ \text{Morse–Smale genericity and annulus nondegeneracy must remain named obligations until fully proved.} }  
   
⸻  
   
## 307. Immediate Next Reconstruction Step  
The next installment should reconstruct the proof-obligation register and kill log.  
That chapter must list:  
1. all theorem dependencies,  
2. all proven components,  
3. all proven-modulo components,  
4. all open lemmas,  
5. all killed claims,  
6. all corrected claims,  
7. which claims are needed for Theorem A,  
8. which claims are needed for Theorem B,  
9. which claims are optional refinements,  
10. what exact work remains before a referee-grade manuscript.  
  
## Unified Master Source — Part 10  
## Proof-Obligation Register, Dependency Ledger, Kill Log, and Readiness Assessment  
   
⸻  
   
## 308. Purpose of the Proof-Obligation Register  
A research program of this complexity eventually accumulates hundreds of intermediate statements.  
Without an explicit dependency ledger, three things inevitably happen:  
1. unsupported assumptions become invisible,  
2. obsolete derivations quietly survive,  
3. later arguments unknowingly depend on killed mathematics.  
The purpose of this chapter is to prevent that.  
Every mathematical statement in the archive should belong to exactly one status class.  
No theorem should silently depend upon an unresolved claim.  
This chapter therefore becomes the constitutional document for the entire research program.  
   
⸻  
   
## 309. Status Taxonomy  
Every mathematical object belongs to one of the following categories.  
## Category A — Primitive  
Accepted hypotheses.  
Examples:  
* Gaussianity  
* stationarity  
* finite spectral moments  
* compact torus  
* finite jet nondegeneracy  
These are assumptions rather than conclusions.  
   
⸻  
   
## Category B — Proven  
Proof included and dependency graph closed.  
A theorem enters this category only after:  
* every dependency is Proven or Primitive,  
* every proof obligation has been discharged,  
* no killed result is referenced.  
   
⸻  
   
## Category C — Proven-Modulo  
Proof complete except for explicitly named finite obligations.  
This is the current status of much of the archive.  
   
⸻  
   
## Category D — Derived  
Calculation completed but not yet packaged into a referee-grade theorem.  
   
⸻  
   
## Category E — Literature Supported  
Known externally.  
No proof required inside the archive except adaptation.  
   
⸻  
   
## Category F — Conjecture  
Believed true.  
Not used unless explicitly isolated.  
   
⸻  
   
## Category G — Open  
Unknown.  
No downstream theorem may depend on it without explicit conditional wording.  
   
⸻  
   
## Category H — Killed  
Known incorrect.  
May never re-enter the dependency graph unless replaced by a repaired theorem.  
   
⸻  
   
## 310. Constitutional Rule  
A theorem may depend only upon  
* Primitive,  
* Proven,  
* Proven-Modulo,  
* Literature Supported,  
objects.  
It may **never** silently depend upon:  
* Conjecture,  
* Open,  
* Killed.  
If it depends on Open or Conjecture, the theorem must itself become Proven-Modulo.  
This single rule prevents hidden circularity.  
   
⸻  
   
## 311. Primitive Assumption Ledger  
Current primitive assumptions are:  
**P1**  
Centered stationary Gaussian field.  
   
⸻  
   
**P2**  
Finite spectral moments through required order.  
   
⸻  
   
**P3**  
Finite jet nondegeneracy.  
   
⸻  
   
**P4**  
Compact torus.  
   
⸻  
   
**P5**  
Smoothness sufficient for fold normal form.  
   
⸻  
   
**P6**  
Compact birth-height window.  
   
⸻  
   
**P7**  
Compact fold-modulus window.  
   
⸻  
   
**P8**  
(MS)  
Almost-sure Morse–Smale property.  
Current status:  
Primitive assumption.  
Desired future status:  
Proven.  
   
⸻  
   
## 312. Core Theorem Dependency Graph  
The entire program compresses into the following DAG.  
```
Primitive assumptions
        │
        ▼
GCJA theorem
        │
        ▼
Pair-Palm regression
        │
        ▼
Pinning theorem
        │
        ▼
Three-point machinery
        │
        ▼
Flat-saddle lemma
        │
        ▼
Reduction lemma
        │
        ▼
Annulus/Far/Band bounds
        │
        ▼
Theorem A
(q0 = 1)
        │
        ▼
Candidate pair intensity
        │
        ▼
Fold change-of-variables
        │
        ▼
Theorem B
(ℓ−1/3 law)

```
Every future theorem should attach somewhere to this graph.  
   
⸻  
   
## 313. GCJA Dependency Ledger  
GCJA depends only upon:  
* Gaussian Hilbert space,  
* Hermite interpolation,  
* finite polynomial nondegeneracy,  
* projection perturbation.  
Status:  
Proven.  
   
⸻  
   
Downstream users:  
* Pinning theorem  
* annulus covariance  
* inner covariance  
* pair regression  
* anisotropy bridge  
   
⸻  
   
## 314. Pair-Palm Ledger  
Depends upon:  
* Kac–Rice  
* GCJA  
* corrected divided differences  
Status:  
Derived / Proven-Modulo.  
Remaining work:  
clean theorem packaging.  
   
⸻  
   
## 315. Pinning Ledger  
Depends upon:  
* Pair Palm  
* GCJA  
Outputs:  
* cubic fold  
* residual scaling  
* Hessian scaling  
Status:  
Derived.  
   
⸻  
   
## 316. Inner-Zone Ledger  
Depends upon:  
* Pinning  
* Pair Palm  
* GCJA  
* Three-point Kac–Rice  
Outputs:  
E[N_{\text{inner}}] \le Cr^5.  
Status:  
Derived upper bound.  
Lower bound:  
Open.  
   
⸻  
   
## 317. Annulus Ledger  
Depends upon:  
* GCJA  
* Pinning  
* Mean dominance  
* Nondegeneracy  
Status:  
Proven-Modulo.  
This is presently the largest remaining probabilistic proof obligation.  
   
⸻  
   
## 318. Far-Zone Ledger  
Depends upon:  
* Ordinary Kac–Rice  
* Compact torus  
* Height band  
Status:  
Essentially complete.  
   
⸻  
   
## 319. Reduction Lemma Ledger  
Depends upon:  
* Morse theory  
* Merge tree  
* Morse–Smale  
Status:  
Proven-Modulo.  
Remaining task:  
write referee-quality deterministic proof.  
   
⸻  
   
## 320. Theorem A Ledger  
Depends upon:  
* Reduction Lemma  
* Inner Lemma  
* Annulus Lemma  
* Far Lemma  
* Band Lemma  
* Morse–Smale  
Status:  
Proven-Modulo.  
   
⸻  
   
## 321. Pair Intensity Ledger  
Depends upon:  
* Two-point Kac–Rice  
* Pair Palm  
Status:  
Proven-Modulo.  
   
⸻  
   
## 322. Theorem B Ledger  
Depends upon:  
* Theorem A  
* Pair intensity  
* Fold relation  
Status:  
Proven-Modulo.  
   
⸻  
   
## 323. Current Kill Log  
A research program becomes stronger when failures are preserved.  
Current canonical kill log:  
   
⸻  
   
## K001  
Super-exponential Palm suppression.  
Claim:  
E[N_{\text{inner}}] = O(e^{-c/r^4}).  
Status:  
Killed.  
Reason:  
Only valid at fixed generic Palm base.  
Not after Palm averaging.  
Replacement:  
Flat-saddle polynomial tail.  
   
⸻  
   
## K002  
r⁴ determinant law.  
Claim:  
E[N_{\text{inner}}] = O(r^4).  
Status:  
Killed.  
Reason:  
Maximum determinant incorrectly treated as independent.  
Replacement:  
O(r^5).  
   
⸻  
   
## K003  
Implicit isotropy.  
Older derivations quietly assumed isotropy.  
Status:  
Killed.  
Replacement:  
Tensor-valued anisotropic formulation.  
   
⸻  
   
## K004  
Raw divided-difference frame.  
Original value difference  
\frac{f(x_2)-f(x_1)}{r^3}  
used directly.  
Status:  
Killed.  
Replacement:  
Corrected divided-difference coordinate subtracting gradient contribution.  
   
⸻  
   
## K005  
Hidden covariance miracles.  
Original symbolic derivations treated covariance powers as unexplained.  
Status:  
Killed.  
Replacement:  
GCJA theorem.  
   
⸻  
   
## 324. Pending Kill Candidates  
The following have not yet been killed or proven.  
They remain active audit targets.  
   
⸻  
   
**Candidate A**  
Can annulus degeneracy produce order-one contribution?  
Unknown.  
   
⸻  
   
**Candidate B**  
Can Morse–Smale be proven directly?  
Unknown.  
   
⸻  
   
**Candidate C**  
Is pair intensity exactly contact neutral?  
Current belief:  
Yes.  
Still awaiting final packaging.  
   
⸻  
   
**Candidate D**  
Can compact κ window be removed?  
Open.  
   
⸻  
   
## 325. Referee Checklist  
A referee should be able to verify every theorem by following only the dependency graph.  
Suggested checklist:  
□ Definitions complete  
□ Notation globally consistent  
□ Every theorem lists dependencies  
□ Every dependency exists  
□ Every proof obligation isolated  
□ No theorem depends on killed mathematics  
□ Literature claims appropriately softened  
□ Every asymptotic variable defined  
□ Every constant dependency stated  
□ Every Palm normalization explicit  
□ Every covariance matrix identified  
□ Every scaling variable normalized  
□ Every compactness assumption visible  
   
⸻  
   
## 326. Circularity Audit  
Potential circularity sources:  
**CA1**  
Using persistence pairing to define candidate fold pair.  
Correct approach:  
Candidate pair defined geometrically.  
Persistence proved later.  
   
⸻  
   
**CA2**  
Using q₀ inside pair intensity.  
Forbidden.  
Pair intensity must precede q₀.  
   
⸻  
   
**CA3**  
Using GCJA to justify Hermite interpolation.  
Forbidden.  
Hermite interpolation proves GCJA.  
   
⸻  
   
**CA4**  
Using flat-saddle lemma inside GCJA.  
Forbidden.  
GCJA precedes flat-saddle integration.  
   
⸻  
   
Current architecture avoids these circles.  
   
⸻  
   
## 327. Dependency Compression  
Originally the archive contained dozens of apparently unrelated lemmas.  
The current dependency graph compresses them into approximately seven structural engines:  
1. Deterministic Morse topology  
2. Pair Palm regression  
3. GCJA  
4. Three-point machinery  
5. Flat-saddle correction  
6. Selection theorem  
7. Density theorem  
Everything else is subordinate.  
This compression is one of the strongest architectural improvements.  
   
⸻  
   
## 328. Remaining Mathematical Risk Register  
Ranked approximately by importance.  
## Risk 1  
Annulus nondegeneracy.  
Highest.  
   
⸻  
   
## Risk 2  
Morse–Smale proof.  
High.  
   
⸻  
   
## Risk 3  
Pair intensity normalization.  
Moderate.  
   
⸻  
   
## Risk 4  
κ-tail integration.  
Moderate.  
   
⸻  
   
## Risk 5  
Three-point symbolic positivity.  
Moderate.  
   
⸻  
   
## Risk 6  
Exact constants.  
Low.  
   
⸻  
   
## Risk 7  
Notation cleanup.  
Low.  
   
⸻  
   
## Risk 8  
Citation refinement.  
Low.  
   
⸻  
   
## 329. Readiness Scale  
Approximate internal readiness.  
Definitions  
██████████████ 100%  
Architecture  
██████████████ 100%  
Dependency graph  
██████████████ 100%  
GCJA  
█████████████░ 95%  
Pair Palm  
████████████░░ 90%  
Pinning  
████████████░░ 90%  
Inner obstruction  
████████████░░ 90%  
Far obstruction  
█████████████░ 95%  
Reduction Lemma  
███████████░░░ 85%  
Annulus proof  
█████████░░░░░ 75%  
Morse–Smale appendix  
██████░░░░░░░░ 55%  
Theorem A  
███████████░░░ 88%  
Pair intensity  
██████████░░░░ 82%  
Theorem B  
███████████░░░ 87%  
Full referee manuscript  
█████████░░░░░ ~80–85%  
These percentages are qualitative project-management estimates, not probabilistic confidence values.  
   
⸻  
   
## 330. Separation Between Core and Refinements  
The archive is much stronger if it separates:  
Core theorem:  
* q₀=1  
* ℓ^{-1/3}  
* compact κ  
* fixed L  
Optional refinements:  
* exact constants  
* lower bounds  
* κ tails  
* thermodynamic limit  
* higher dimensions  
* convergence rates  
* anisotropic explicit constants  
This dramatically lowers proof complexity.  
   
⸻  
   
## 331. Future Research Queue  
After the referee-grade manuscript:  
Phase II  
* Thermodynamic limit  
* Higher-dimensional exponent law  
* H₁ persistence  
* Flat-saddle statistics  
* Pair-point process  
* CLT for near-diagonal counts  
* Universality classes  
* Non-Gaussian perturbations  
* Numerical validation  
* Cosmological applications  
   
⸻  
   
## 332. Canonical Project State  
The program should now be described as:  
A modular mathematical research program organized around a dependency graph rather than a linear manuscript. Every unresolved component has been isolated into explicit proof obligations, every invalid derivation has been preserved in a kill log, and every surviving theorem has a traceable chain of dependencies. The strongest completed components are the GCJA framework, the pair-Palm conditioning architecture, the corrected flat-saddle analysis, and the deterministic reduction. The principal remaining proof obligations are the annulus estimate, the Gaussian Morse–Smale appendix, and the final normalization of the candidate pair intensity. Once those are discharged, the current architecture is positioned to support a referee-ready monograph.  
   
⸻  
   
## 333. Immediate Next Reconstruction Step  
The next phase of reconstruction should shift from *mathematical architecture* to *publication architecture*.  
That phase should produce:  
1. A single unified notation dictionary.  
2. A global symbol index.  
3. A theorem numbering system.  
4. A unified bibliography map.  
5. Cross-reference cleanup.  
6. Elimination of duplicate definitions.  
7. Elimination of repeated proofs.  
8. A referee roadmap.  
9. Machine-readable appendices.  
10. A complete publication-ready monograph assembled from all previous parts into one internally consistent document.  
That is the final synthesis phase before completion.  
  
## Unified Master Source — Part 11  
## Publication Architecture: Notation Dictionary, Numbering System, and Final Monograph Blueprint  
   
⸻  
   
## 334. Final Document Goal  
The final output should become one unified monograph:  
**Near-Diagonal **H_0** Persistence of Smooth Stationary Gaussian Fields: Elder-Rule Selection, Conditional-Jet Asymptotics, and the **\ell^{-1/3}** Law**  
Its role is to replace the fragmented archive with one internally consistent source.  
   
⸻  
   
## 335. Required Monograph Structure  
## Front Matter  
1. Title  
2. Status header  
3. Abstract  
4. Executive theorem summary  
5. Dependency graph  
6. Proof-obligation ledger  
7. Kill-log warning  
   
⸻  
   
## Part I — Main Results  
1. Introduction  
2. Setting and hypotheses  
3. Main theorem A: q_0=1  
4. Main theorem B: \ell^{-1/3} law  
5. Novelty boundary  
   
⸻  
   
## Part II — Deterministic Topology  
6. Morse functions and superlevel persistence  
7. Elder rule and merge trees  
8. Gradient adjacency versus persistence pairing  
9. Deterministic reduction lemma  
10. Loop/crater obstruction reduction  
   
⸻  
   
## Part III — Gaussian Critical-Pair Machinery  
11. Kac–Rice formulas  
12. Pair-Palm conditioning  
13. Corrected divided differences  
14. Cubic pinning theorem  
15. Hessian determinant ledger  
   
⸻  
   
## Part IV — GCJA  
16. Gaussian Hilbert space  
17. Staircase designs  
18. Visible/unseen jet spaces  
19. Hermite interpolation  
20. Projection perturbation  
21. GCJA theorem  
22. Corollaries for pair pinning and three-point estimates  
   
⸻  
   
## Part V — Obstruction Estimates  
23. Inner-zone two-scale Kac–Rice  
24. Flat-saddle tail lemma  
25. Annulus estimate  
26. Far-zone estimate  
27. Band estimate  
28. Selection theorem assembly  
   
⸻  
   
## Part VI — Near-Diagonal Density  
29. Candidate fold-pair intensity  
30. Contact neutrality  
31. Change of variables  
32. Compact-\kappa theorem  
33. Full-\kappa extension  
34. Constants and anisotropy  
   
⸻  
   
## Part VII — Verification, Calibration, and Open Work  
35. Literature audit  
36. C006 falsification protocol  
37. Computational validation plan  
38. Remaining obligations  
39. Future extensions  
   
⸻  
   
## Appendices  
A. Symbol index B. Theorem dependency table C. Kill log D. Morse–Smale appendix E. GCJA proof details F. Three-point covariance details G. Pair-density normalization H. Calibration cycle templates I. Machine-readable theorem graph  
   
⸻  
   
## 336. Global Symbol Dictionary  
## Field Symbols  
f  
stationary centered Gaussian field.  
C(h)  
covariance function.  
\rho  
spectral measure.  
\mathbb{T}_L^2  
flat two-torus of side length L.  
L  
fixed torus side length.  
   
⸻  
   
## Critical Point Symbols  
x_M  
candidate local maximum.  
x_S  
candidate saddle.  
y  
third possible critical point.  
H_M=D^2f(x_M)  
Hessian at maximum.  
H_S=D^2f(x_S)  
Hessian at saddle.  
H_y=D^2f(y)  
Hessian at third point.  
   
⸻  
   
## Pair Symbols  
r=d(x_M,x_S)  
pair separation.  
b=f(x_M)  
birth height.  
\ell=f(x_M)-f(x_S)  
lifetime / height gap.  
\kappa=\frac{6\ell}{r^3}  
fold modulus.  
q(r,b,\kappa)  
selection probability.  
q_0  
limiting selection constant.  
   
⸻  
   
## Persistence Symbols  
E_t=\{x:f(x)\ge t\}  
superlevel set.  
D(x_M)  
death saddle paired to x_M.  
\nu(\ell)  
expected lifetime density.  
C_*  
near-diagonal density constant.  
   
⸻  
   
## Pair-Frame Symbols  
t  
longitudinal coordinate.  
s  
transverse coordinate.  
f_{tt},f_{ts},f_{ss}  
pair-frame Hessian entries.  
\mu=-f_{ss}(x_S)  
raw saddle transverse curvature.  
\nu_S,\nu_M  
mixed-derivative normalized variables.  
\widetilde{\mu}_S  
corrected saddle transverse curvature.  
\widetilde{\mu}_M  
corrected maximum transverse curvature.  
\widetilde{\eta}  
curvature-coupling variable.  
   
⸻  
   
## GCJA Symbols  
D(p;d)  
staircase observation design.  
p  
number of collapsing nodes.  
d=(d_0,\ldots,d_J)  
staircase derivative profile.  
q_k=p(d_k+1)  
visible longitudinal count at transverse level k.  
V(D)  
visible jet space.  
U(D)  
unseen jet set.  
Q_V  
projection onto V(D)^\perp.  
G_{U|V}  
conditional unseen Gram matrix.  
L_a(\sigma;\gamma)  
ladder exponent.  
\Sigma_a  
leading unseen stratum.  
   
⸻  
   
## 337. Theorem Numbering System  
Use permanent labels.  
## Main Theorems  
**Theorem A** — Elder-rule selection constant q_0=1. **Theorem B** — Near-diagonal \ell^{-1/3} law. **Theorem C** — GCJA theorem. **Theorem D** — Candidate fold-pair intensity/contact neutrality.  
   
⸻  
   
## Deterministic Lemmas  
**Lemma R1** — Morse merge-tree elder-rule lemma. **Lemma R2** — Gradient-adjacency/local-cell lemma. **Lemma R3** — Deterministic reduction to obstruction events. **Lemma R4** — Loop/crater witness lemma.  
   
⸻  
   
## Pair-Palm Lemmas  
**Lemma P1** — Corrected pair-frame invertibility. **Lemma P2** — Pair-Palm regression. **Lemma P3** — Cubic pinning. **Lemma P4** — Hessian scaling. **Lemma P5** — Corrected transverse-curvature determinant formula.  
   
⸻  
   
## GCJA Lemmas  
**Lemma G1** — Hermite unisolvence. **Lemma G2** — Divided-difference rate. **Lemma G3** — Projection perturbation. **Theorem C** — GCJA. **Corollary G4** — Pair-pinning covariance. **Corollary G5** — Inner-zone gradient covariance. **Corollary G6** — Anisotropy robustness.  
   
⸻  
   
## Obstruction Lemmas  
**Lemma I1** — Inner nested-frame nondegeneracy. **Lemma I2** — Inner-zone third-point Kac–Rice bound. **Lemma I3** — Flat-saddle tail upper bound O(r^5). **Lemma A1** — Annulus mean-dominance. **Lemma A2** — Annulus obstruction bound. **Lemma F1** — Far-zone band count. **Lemma B1** — Height-band critical count.  
   
⸻  
   
## 338. Final Theorem Statements  
## Theorem A  
Let f satisfy H1+, H2', and Assumption MS on \mathbb{T}_L^2. Let B\subset\mathbb{R} and K\subset(0,\infty) be compact. Under the maximum-saddle fold-pair Palm law at separation r, birth height b\in B, and fold modulus \kappa\in K,  
\sup_{b,\kappa}|1-q(r,b,\kappa)|\to0.  
Hence:  
q_0=1.  
   
⸻  
   
## Theorem B  
Under the hypotheses of Theorem A and candidate pair contact neutrality,  
\nu_{B,K}(\ell) = C_{B,K}\ell^{-1/3}(1+o(1)).  
Equivalently:  
N_{B,K}(0,\varepsilon) = \frac32 C_{B,K}\varepsilon^{2/3}(1+o(1)).  
   
⸻  
   
## Theorem C  
For a collapsing Gaussian observation design D(p;d), conditional residual jets are governed by the visible/unseen filtration:  
r^{-\ell_a}Q_{S_r}\phi_y^a \to \sum_{\sigma\in\Sigma_a} m_\sigma^a(z)Q_V\psi_\sigma.  
Consequently:  
r^{-(\ell_a+\ell_b)} \operatorname{Cov}(\phi_y^a,\phi_y^b\mid S_r) \to \Sigma_\infty^{ab}(z).  
   
⸻  
   
## Theorem D  
The candidate maximum-saddle fold-pair intensity is contact neutral:  
d\mathbb{E}N_{\mathrm{cand}} = L^2\rho_{\mathrm{cand}}(b,\kappa,\theta) r\,dr\,d\theta\,db\,d\kappa(1+o(1)).  
   
⸻  
   
## 339. Final Dependency Table  

| Result | Depends On | Status |
| ---------------- | ---------------------------------- | --------------------- |
| Theorem C / GCJA | Hf, Hmom, Hnd, Hermite, projection | Strong |
| Lemma P1 | GCJA, corrected frame | Strong |
| Lemma P3 | P1, pair regression | Strong |
| Lemma I3 | P3, GCJA, Palm determinant ledger | Strong upper bound |
| Lemma A2 | GCJA, mean dominance | Main proof obligation |
| Lemma F1 | Kac–Rice, compact L | Strong |
| Lemma R3 | Morse–Smale, merge tree | Needs final writeup |
| Theorem A | R3, I3, A2, F1, B1 | Proven-modulo |
| Theorem D | two-point Kac–Rice | Proven-modulo |
| Theorem B | A, D, fold change | Proven-modulo |
  
   
⸻  
   
## 340. Publication Status Header  
The final document should begin with:  
**Status.** This manuscript proves the elder-rule selection theorem and the near-diagonal density theorem modulo explicitly named assumptions and lemmas. The strongest remaining mathematical obligations are the annulus obstruction lemma, the Gaussian Morse–Smale appendix, and final normalization of the candidate fold-pair intensity. The inner-zone obstruction has been corrected from a false super-exponential claim to a polynomial flat-saddle upper bound sufficient for the main theorem.  
   
⸻  
   
## 341. Referee Roadmap  
A referee should read in this order:  
1. Main theorem statements.  
2. Dependency graph.  
3. Deterministic reduction lemma.  
4. Pair-Palm construction.  
5. GCJA theorem.  
6. Inner flat-saddle lemma.  
7. Annulus lemma.  
8. Pair-intensity proposition.  
9. Theorem B change of variables.  
10. Kill log.  
This prevents the referee from getting lost in derivation history.  
   
⸻  
   
## 342. Machine-Readable Dependency Graph  
```
{
  "Theorem_A_q0_equals_1": {
    "depends_on": [
      "Assumption_MS",
      "Lemma_R3_Deterministic_Reduction",
      "Lemma_I3_Flat_Saddle_Inner_Bound",
      "Lemma_A2_Annulus_Bound",
      "Lemma_F1_Far_Zone_Bound",
      "Lemma_B1_Band_Bound"
    ],
    "status": "proven_modulo"
  },
  "Theorem_B_lifetime_density": {
    "depends_on": [
      "Theorem_A_q0_equals_1",
      "Theorem_D_Candidate_Pair_Intensity",
      "Fold_Change_Of_Variables"
    ],
    "status": "proven_modulo"
  },
  "Theorem_C_GCJA": {
    "depends_on": [
      "Gaussian_Hilbert_Space",
      "Hermite_Unisolvence",
      "Projection_Perturbation",
      "Finite_Moment_Assumption",
      "Polynomial_Nondegeneracy"
    ],
    "status": "strong"
  },
  "Lemma_I3_Flat_Saddle_Inner_Bound": {
    "depends_on": [
      "Pair_Palm_Regression",
      "Cubic_Pinning",
      "GCJA",
      "Corrected_Determinant_Ledger"
    ],
    "status": "derived_upper_bound"
  }
}

```
   
⸻  
   
## 343. Final Manuscript Writing Rules  
Use these rules during final assembly:  
1. Never introduce a symbol twice.  
2. Never use an unstated compactness assumption.  
3. Never use isotropic notation in a theorem statement unless theorem is isotropic.  
4. Never cite killed claims.  
5. Never use “standard” for Morse–Smale without proof or assumption.  
6. Never use value-band gains unless density is controlled.  
7. Never compute exact constants in sections proving only qualitative convergence.  
8. Never mix candidate-pair intensity with selected-pair intensity.  
9. Never hide Palm determinant weighting.  
10. Never state full-\kappa theorem before compact-\kappa theorem.  
   
⸻  
   
## 344. Final Assembly Order  
The final writing should not follow archive chronology.  
It should follow proof logic:  
1. State result.  
2. Define objects.  
3. Prove deterministic reduction.  
4. Build Gaussian conditioning engine.  
5. Prove obstruction bounds.  
6. Assemble selection theorem.  
7. Derive density theorem.  
8. Discuss literature and open work.  
The kill log belongs near the end or in an appendix, not interleaved with the proof.  
   
⸻  
   
## 345. What Remains Before “Completed”  
The reconstruction is not complete until the following artifacts exist:  
1. Final abstract.  
2. Final theorem statements.  
3. Final notation dictionary.  
4. Final dependency graph.  
5. Final proof-obligation register.  
6. Final kill log.  
7. Final manuscript outline.  
8. Final theorem-by-theorem proof skeleton.  
9. Final unresolved-work checklist.  
10. Final continuation prompt for another model or human researcher.  
The next continuation should produce the final theorem-by-theorem proof skeleton and unresolved-work checklist.  
  
## Unified Master Source — Part 12  
## Final Proof Skeleton, Research Completion Roadmap, and Remaining Mathematical Program  
   
⸻  
   
## 346. Purpose of This Chapter  
The preceding chapters reconstructed the architecture of the research program.  
This chapter has a different purpose.  
Instead of describing *what* the mathematics is, it specifies *how the mathematics is to be completed*.  
It functions as the program’s engineering specification.  
Every remaining task is reduced to a finite checklist.  
The goal is to eliminate “unknown unknowns.”  
   
⸻  
   
## 347. The Entire Program Reduced to One Chain  
The entire mathematical program can now be represented as a single dependency chain.  
```
Primitive Gaussian Assumptions
          │
          ▼
Morse Theory
          │
          ▼
Reduction Lemma
          │
          ▼
Pair-Palm Construction
          │
          ▼
GCJA
          │
          ▼
Pinning
          │
          ▼
Three-Point Machinery
          │
          ▼
Flat-Saddle Tail
          │
          ▼
Annulus Estimate
          │
          ▼
Selection Constant
 q₀ = 1
          │
          ▼
Pair Intensity
          │
          ▼
ℓ = κr³ / 6
          │
          ▼
Near-Diagonal Law
ℓ−1/3

```
Everything outside this chain is supporting infrastructure.  
   
⸻  
   
## 348. Minimal Proof of Theorem A  
The remarkable outcome of the reconstruction is that Theorem A no longer requires dozens of independent arguments.  
It requires only five conceptual moves.  
   
⸻  
   
## Step A1  
Construct the candidate fold pair.  
Output:  
maximum  
* ●   
saddle  
* ●   
Palm conditioning.  
   
⸻  
   
## Step A2  
Prove local geometry.  
Output:  
pair behaves like cubic fold.  
   
⸻  
   
## Step A3  
Reduce failure.  
Output:  
Every elder-rule mismatch implies one of finitely many obstruction events.  
   
⸻  
   
## Step A4  
Show every obstruction probability tends to zero.  
Inner  
Annulus  
Far  
Band  
Loop  
   
⸻  
   
## Step A5  
Apply union bound.  
Done.  
Nothing more is conceptually required.  
   
⸻  
   
## 349. Minimal Proof of Theorem B  
Likewise Theorem B compresses dramatically.  
   
⸻  
   
## Step B1  
Candidate fold-pair intensity.  
   
⸻  
   
## Step B2  
Multiply by selection probability.  
Use  
q_0=1.  
   
⸻  
   
## Step B3  
Change variables  
\ell=\kappa r^3/6.  
   
⸻  
   
## Step B4  
Integrate over marks.  
Done.  
   
⸻  
   
## 350. Remaining Proof Obligations  
The archive presently contains approximately three categories.  
   
⸻  
   
## Category 1  
Essential.  
Without these theorems cannot be published.  
   
⸻  
   
**O1**  
Annulus estimate.  
   
⸻  
   
**O2**  
Pair intensity normalization.  
   
⸻  
   
**O3**  
Morse–Smale appendix or explicit assumption.  
   
⸻  
   
## Category 2  
Strongly recommended.  
   
⸻  
   
**O4**  
κ-tail integration.  
   
⸻  
   
**O5**  
Final positivity verification.  
   
⸻  
   
**O6**  
Machine-check symbolic algebra.  
   
⸻  
   
## Category 3  
Future research.  
   
⸻  
   
Everything else.  
   
⸻  
   
## 351. Machine Verification Layer  
The archive is now sufficiently modular that formal verification becomes realistic.  
Suggested verification stack:  
Layer 1  
Symbolic differentiation.  
Layer 2  
Automatic covariance generation.  
Layer 3  
Positive-definite testing.  
Layer 4  
Numerical Palm simulation.  
Layer 5  
Monte Carlo obstruction verification.  
Layer 6  
Independent theorem checker.  
The proofs themselves remain human mathematics.  
The calculations become machine-audited.  
   
⸻  
   
## 352. Numerical Validation Layer  
The numerical work should mirror the theorem exactly.  
Not merely “simulate persistence.”  
Instead:  
   
⸻  
   
Simulation A  
Candidate pair intensity.  
   
⸻  
   
Simulation B  
Selection probability.  
   
⸻  
   
Simulation C  
Inner flat-saddle frequency.  
   
⸻  
   
Simulation D  
Annulus obstruction rate.  
   
⸻  
   
Simulation E  
Measured  
q(r).  
   
⸻  
   
Simulation F  
Measured  
\nu(\ell).  
   
⸻  
   
Each simulation corresponds to exactly one theorem.  
This creates one-to-one theorem validation.  
   
⸻  
   
## 353. Formal Verification Layer  
Eventually each theorem should have:  
Hypotheses  
↓  
Definitions  
↓  
Dependency graph  
↓  
Machine-readable statement  
↓  
Human proof  
↓  
Machine verification of calculations  
↓  
Numerical experiment  
↓  
Calibration record  
↓  
Status  
This produces unprecedented traceability.  
   
⸻  
   
## 354. Calibration Integration  
Calibration should never modify mathematics directly.  
Instead:  
Prediction  
↓  
Evidence  
↓  
Comparison  
↓  
Decision  
Possible decisions:  
A  
Supports theorem.  
B  
Supports assumptions.  
C  
Suggests refinement.  
D  
Falsifies mechanism.  
The theorem only changes after mathematics changes.  
Not after simulation.  
   
⸻  
   
## 355. Living Theorem Architecture  
The archive should now be treated as a living theorem system.  
Each theorem possesses:  
Unique ID  
Dependencies  
Version  
Status  
Calibration history  
Literature history  
Kill history  
Open obligations  
Future extensions  
This resembles software version control more than traditional mathematical papers.  
   
⸻  
   
## 356. Versioning Proposal  
Version 1.0  
Architecture frozen.  
Version 1.1  
Notation cleanup.  
Version 1.2  
Annulus proof complete.  
Version 1.3  
Pair intensity finalized.  
Version 2.0  
First referee submission.  
Version 2.1  
Reviewer revisions.  
Version 3.0  
Thermodynamic limit.  
Version 4.0  
Higher-dimensional extension.  
   
⸻  
   
## 357. Long-Term Mathematical Roadmap  
Immediate  
Complete referee manuscript.  
↓  
Short-term  
Thermodynamic limit.  
↓  
Medium-term  
Dimension three.  
↓  
Medium-term  
General Morse index.  
↓  
Medium-term  
Higher persistence groups.  
↓  
Long-term  
General stationary fields.  
↓  
Long-term  
Non-Gaussian universality.  
↓  
Long-term  
Persistence kinetic theory.  
   
⸻  
   
## 358. Possible New Research Programs  
The architecture naturally branches into independent projects.  
   
⸻  
   
Program A  
Persistence transport equations.  
   
⸻  
   
Program B  
Persistence kinetic theory.  
   
⸻  
   
Program C  
Persistence renormalization.  
   
⸻  
   
Program D  
Curvature spectrum statistics.  
   
⸻  
   
Program E  
Random merge-tree asymptotics.  
   
⸻  
   
Program F  
Palm geometry of persistence.  
   
⸻  
   
Program G  
Gaussian catastrophe theory.  
   
⸻  
   
Program H  
Conditional-jet geometry.  
   
⸻  
   
GCJA alone likely supports multiple future papers.  
   
⸻  
   
## 359. Most Important Conceptual Achievement  
The archive did something more important than proving individual lemmas.  
It transformed an informal research process into a mathematically auditable system.  
Originally:  
ideas  
↓  
calculations  
↓  
proofs  
↓  
paper  
Now:  
definitions  
↓  
dependency graph  
↓  
proof obligations  
↓  
theorems  
↓  
calibration  
↓  
version history  
↓  
publication  
↓  
future extensions  
This architecture itself is reusable.  
   
⸻  
   
## 360. Final Internal Audit  
Current state:  
Definitions  
✓  
Notation  
✓  
Dependency graph  
✓  
Theorem structure  
✓  
Proof architecture  
✓  
GCJA  
✓  
Pair Palm  
✓  
Pinning  
✓  
Three-point framework  
✓  
Flat-saddle correction  
✓  
Selection theorem  
Nearly complete.  
Density theorem  
Nearly complete.  
Publication architecture  
✓  
Remaining mathematics:  
Mostly localized.  
   
⸻  
   
## 361. What Still Prevents “Completed”  
Three mathematical components remain genuinely incomplete:  
1. A fully rigorous annulus proof with all symbolic nondegeneracy checks discharged.  
2. A complete proof (or explicit acceptance as an assumption) of the almost-sure Morse–Smale property for the Gaussian field class under the stated hypotheses.  
3. A finalized two-point candidate fold-pair intensity normalization (“contact neutrality”) with all constants and normalization conventions fixed.  
Everything else is either organizational, expository, computational, or a refinement rather than a foundational gap.  
   
⸻  
   
## 362. Final Recommendation Before Submission  
Do **not** attempt to strengthen the core theorem further before submission.  
Instead:  
* Freeze the theorem statements.  
* Freeze the notation.  
* Freeze the dependency graph.  
* Complete the three remaining mathematical obligations.  
* Eliminate duplicate derivations.  
* Submit the compact-\kappa version first.  
* Reserve thermodynamic limits, higher dimensions, sharper constants, and lower bounds for follow-on papers.  
This minimizes proof complexity while maximizing rigor.  
   
⸻  
   
## 363. Transition to Final Synthesis Phase  
At this point, the reconstruction has transitioned from *recovering* the archive to *assembling* it.  
The remaining work is no longer discovering architecture—it is integrating it into a single coherent mathematical monograph with consistent notation, complete proofs, and publication-quality exposition.  
The next continuations should therefore focus on producing the finished monograph itself rather than additional architectural summaries. The work should proceed chapter by chapter until every section has been rewritten into final publication form. The process is complete only when every theorem, lemma, definition, proof, appendix, bibliography mapping, and cross-reference has been unified into one internally consistent document.  
  
## Unified Master Source — Part 13  
## Toward a Self-Verifying Mathematical Research System  
   
⸻  
   
## 364. Transition From a Paper to a Research System  
At this point, the reconstruction changes character.  
The first twelve parts reconstructed the mathematics.  
The remaining parts should reconstruct the **research system itself**.  
This is an important distinction.  
Traditional mathematics usually follows:  
idea → proof → publication.  
The C006 architecture has evolved into something different:  
hypothesis → dependency graph → proof → adversarial audit → computational calibration → literature audit → publication → continuous revision.  
This is closer to a verified software project than a traditional paper.  
One of the archive’s most significant contributions is therefore not only a mathematical theorem but a **mathematical development methodology**.  
One of the archive’s most significant contributions is therefore not only a mathematical theorem but a **mathematical development methodology**.  
The methodology itself should become explicit.  
   
⸻  
   
## 365. The Three Layers of Truth  
Every statement in the archive belongs to one of three fundamentally different layers.  
## Layer I — Mathematical Truth  
These are statements whose validity depends only on logic.  
Examples:  
* Hermite interpolation.  
* Projection identities.  
* Theorem A.  
* Theorem B.  
* GCJA.  
* Fold normal form.  
Evidence cannot make these true.  
Evidence can only reveal flaws in proofs.  
   
⸻  
   
## Layer II — Model Truth  
These depend upon assumptions.  
Examples:  
* Gaussianity.  
* Stationarity.  
* Compact torus.  
* Finite moments.  
* Morse–Smale assumption.  
If assumptions change,  
truth changes.  
   
⸻  
   
## Layer III — Empirical Truth  
These concern reality or simulations.  
Examples:  
* Numerical calibration.  
* Bootstrap validation.  
* Monte Carlo estimates.  
* C006 falsification protocol.  
These never prove mathematics.  
They only test whether the mathematical mechanism appears in sampled data.  
Keeping these three layers separate prevents many category errors.  
   
⸻  
   
## 366. The Verification Pyramid  
The completed program should be organized as a verification pyramid.  
```
                 Publication
                     ▲
              Independent Review
                     ▲
            Computational Audit
                     ▲
             Internal Proof Audit
                     ▲
            Dependency Verification
                     ▲
            Primitive Definitions

```
Every layer only depends upon the layer below.  
No layer skips intermediate verification.  
   
⸻  
   
## 367. The Dependency Graph as the Primary Object  
Most mathematical papers treat proofs as the primary object.  
The reconstructed architecture suggests something different.  
The primary object should be:  
\boxed{\text{Dependency Graph}}  
The proofs become annotations on the graph.  
Every theorem is a node.  
Every logical implication is an edge.  
This has several advantages.  
   
⸻  
   
## Advantage 1  
No hidden assumptions.  
   
⸻  
   
## Advantage 2  
Circular reasoning becomes graph cycles.  
   
⸻  
   
## Advantage 3  
Removing one theorem immediately identifies every downstream consequence.  
   
⸻  
   
## Advantage 4  
Version control becomes automatic.  
   
⸻  
   
## Advantage 5  
AI systems can traverse the graph directly.  
   
⸻  
   
## 368. Proof Objects  
Every theorem should exist as a proof object.  
Each proof object contains:  
ID  
Statement  
Dependencies  
Proof  
Alternative proofs  
Computational verification  
Calibration history  
Open questions  
Version history  
This is richer than traditional mathematical exposition.  
   
⸻  
   
## 369. Mathematical Version Control  
Suppose a lemma changes.  
Traditional papers require rereading the entire manuscript.  
Instead:  
Lemma P3 changes.  
The dependency graph identifies:  
↓  
Flat-Saddle Lemma  
↓  
Inner Obstruction  
↓  
Theorem A  
↓  
Theorem B  
Only those objects require re-audit.  
Everything else remains frozen.  
This is mathematically analogous to incremental software compilation.  
   
⸻  
   
## 370. Independent Proof Paths  
One theorem should ideally possess multiple independent proofs.  
For example,  
Theorem A could eventually have:  
Path 1  
Current Palm/Kac–Rice proof.  
   
⸻  
   
Path 2  
Dynamical systems proof.  
   
⸻  
   
Path 3  
Merge-tree probability proof.  
   
⸻  
   
Path 4  
Discrete approximation.  
   
⸻  
   
Agreement between independent proofs dramatically increases confidence.  
This is stronger than a single long derivation.  
   
⸻  
   
## 371. Verification Score (Optional Layer)  
The archive previously experimented with verification metrics.  
A simplified version could become:  
V(T) = P(T) \cdot D(T) \cdot R(T)  
where  
P  
Proof completeness.  
D  
Dependency completeness.  
R  
Reproducibility.  
Unlike earlier exploratory frameworks, this score should be interpreted as **project-management metadata**, not as a mathematical truth value.  
It tracks documentation maturity rather than theorem validity.  
   
⸻  
   
## 372. AI Roles  
The reconstruction naturally separates AI roles.  
   
⸻  
   
Role 1  
Constructor.  
Creates mathematics.  
   
⸻  
   
Role 2  
Verifier.  
Attempts to prove every step.  
   
⸻  
   
Role 3  
Destroyer.  
Attempts to falsify every step.  
   
⸻  
   
Role 4  
Archivist.  
Maintains dependency graph.  
   
⸻  
   
Role 5  
Historian.  
Maintains version history.  
   
⸻  
   
No single AI should perform all roles simultaneously without separation.  
   
⸻  
   
## 373. Adversarial Loop  
Every theorem should pass:  
Generation  
↓  
Self review  
↓  
Independent AI review  
↓  
Independent human review  
↓  
Numerical verification  
↓  
Literature verification  
↓  
Publication  
↓  
Future revision  
Each stage records changes.  
Nothing disappears.  
   
⸻  
   
## 374. The Kill Log as Positive Information  
Traditional mathematics hides failed proofs.  
The reconstructed system treats failures as first-class objects.  
Every killed theorem records:  
Original statement.  
Reason for failure.  
Replacement.  
Affected downstream objects.  
Date/version.  
Recovered insight.  
This is valuable mathematical information.  
The Flat-Saddle correction is the archetypal example.  
   
⸻  
   
## 375. Why the Flat-Saddle Correction Matters Beyond This Program  
The correction demonstrates a general lesson.  
There is a difference between:  
conditioning  
and  
conditioning followed by averaging.  
Many asymptotic arguments silently exchange these operations.  
The archive records a concrete instance where doing so changed:  
super-exponential  
↓  
polynomial.  
This methodological lesson extends beyond persistence theory.  
   
⸻  
   
## 376. Symbolic Algebra Policy  
Every symbolic computation should satisfy:  
Independent derivation.  
↓  
Computer algebra verification.  
↓  
Asymptotic consistency check.  
↓  
Dimensional analysis.  
↓  
Limiting-case analysis.  
↓  
Dependency registration.  
Only then is it admitted into the theorem graph.  
   
⸻  
   
## 377. Numerical Policy  
Numerics never prove.  
Numerics perform three functions.  
   
⸻  
   
Detect algebra mistakes.  
   
⸻  
   
Estimate constants.  
   
⸻  
   
Attempt falsification.  
Nothing else.  
This avoids common misuse of simulations.  
   
⸻  
   
## 378. Literature Policy  
Every citation receives one of five labels:  
Direct proof.  
Method.  
Analogy.  
Boundary.  
History.  
No citation should silently change category.  
   
⸻  
   
## 379. Conjecture Policy  
A conjecture may never appear inside a proof.  
Instead:  
Conjecture  
↓  
Conditional theorem  
↓  
Future theorem.  
This prevents hidden assumptions.  
   
⸻  
   
## 380. Publication Pipeline  
The publication pipeline should become:  
Research notebook  
↓  
Architecture document  
↓  
Dependency graph  
↓  
Core theorem package  
↓  
Technical appendices  
↓  
Computational appendix  
↓  
Calibration appendix  
↓  
Submission manuscript  
↓  
Reviewer response document  
↓  
Post-publication revision history  
This preserves traceability.  
   
⸻  
   
## 381. Long-Term Vision  
The architecture hints at something larger than one theorem.  
It suggests a framework for AI-assisted mathematical research in which:  
* every theorem has explicit dependencies,  
* every proof is auditable,  
* every correction is preserved,  
* every computation is reproducible,  
* every empirical claim is separated from deductive proof,  
* every revision is version-controlled.  
That idea is independent of Gaussian fields.  
   
⸻  
   
## 382. What Should Never Change Again  
The following should now be frozen unless an error is discovered.  
Definitions.  
Notation.  
Dependency graph.  
Status taxonomy.  
Kill-log policy.  
Proof-obligation policy.  
Changing these repeatedly would destabilize every later proof.  
   
⸻  
   
## 383. What May Continue to Change  
Proof details.  
Constants.  
Exponents of secondary bounds.  
Appendix organization.  
Numerical calibration.  
Literature references.  
Future extensions.  
These evolve without changing the constitutional layer.  
   
⸻  
   
## 384. Constitutional Layer  
The constitutional layer now consists of:  
1. Definitions.  
2. Primitive assumptions.  
3. Dependency graph.  
4. Status taxonomy.  
5. Kill-log policy.  
6. Verification policy.  
Everything else derives from these.  
   
⸻  
   
## 385. Research Program Identity  
The completed program can now be described succinctly as:  
A dependency-driven mathematical research system for deriving, auditing, falsifying, calibrating, and publishing theorems about near-diagonal persistence of smooth stationary Gaussian random fields.  
This description is broader than any individual theorem.  
   
⸻  
   
## 386. Remaining Deliverables  
At this stage, the remaining work is almost entirely editorial rather than conceptual.  
The remaining deliverables are:  
* a unified abstract,  
* a polished introduction,  
* publication-ready proofs,  
* standardized notation,  
* bibliography normalization,  
* cross-reference cleanup,  
* appendices,  
* computational supplement,  
* calibration supplement.  
The architecture itself is essentially reconstructed.  
   
⸻  
   
## 387. Current State of Reconstruction  
Across Parts 1–13, the reconstruction now includes:  
* Canonical architecture.  
* Unified notation.  
* Deterministic topology.  
* Pair-Palm framework.  
* GCJA.  
* Three-point machinery.  
* Flat-saddle correction.  
* Obstruction decomposition.  
* Selection theorem architecture.  
* Near-diagonal density architecture.  
* Literature boundary.  
* Dependency graph.  
* Proof-obligation register.  
* Publication blueprint.  
* Research-system architecture.  
At this point, additional continuations should focus on **rewriting the material into final publication-quality prose**, not introducing new architectural layers. The structural reconstruction is effectively complete; the remaining effort is consolidation, polishing, and final proof presentation.  
