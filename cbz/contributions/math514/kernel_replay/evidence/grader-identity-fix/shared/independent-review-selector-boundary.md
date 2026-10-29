# Selector boundary review

AMEND at exact kernel `d1783e48` and layer-a `6a719ffd`.

F3's raw descriptor/proxy traversal ordering is fixed, with four actual new
guard assertion kills and zero errors. Previous findings and applicable
verification remain separately preserved.

**F4 / P1:** `isinstance(..., (staticmethod, classmethod))` also permits custom
subclasses with an overriding `__get__`. Both runners accept such a descriptor,
which aliases the second declared method to the first before snapshots. The
actual results are SURVIVED/exit 0 with two names and trace
`descriptor, first, first`; the retained second failing body is absent.

Use exact built-in descriptor types for the supported profile, and inspect
constructor/attribute-dispatch members statically so the refusal check itself
cannot invoke a custom descriptor. Exact-source probes and receipts are in
`review-probes/staticmethod-subclass-alias/`. Only these two new normal-mode
probes were executed; no completed check was repeated.
