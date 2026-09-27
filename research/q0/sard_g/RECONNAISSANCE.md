# Reconnaissance — SARD-G execution repair

Primary external context inspected 2026-09-25:

- AMS survey material on invariant manifolds and local families: https://www.ams.org/bookstore/pspdf/surv-66-prev.pdf . It explicitly discusses extension to parameter-dependent local families.
- Cambridge University Press, Daniel W. Stroock, Gaussian Measures on a Banach Space, Chapter 8 of Probability Theory: An Analytic View: https://www.cambridge.org/core/books/abs/probability-theory-an-analytic-view/gaussian-measures-on-a-banach-space/7728BE474E26B4DE50E936668EFC4C26 . This is background for the Cameron–Martin/Gaussian Banach-space structure.
- The workspace's own specialist review KIMI-AUD-022 and the exact 19,242-byte transversality manuscript.

The repair is intentionally self-contained at the logical interfaces it changes. It uses external sources as standard-theory context, not as evidence that the workspace theorem is automatically correct.

Key design choice: do not solve measurable selection of a first chart; use a countable union of open chart events. Do not insist that endpoint launch variation is a two-jet atom; prove only support localization of the endpoint derivative functional, which is exactly what the RKHS support-separation contradiction needs.

No novelty or priority claim is made for stable-manifold parameter dependence, Gaussian Banach-space decomposition, or countable-union slicing.
