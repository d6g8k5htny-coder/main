# Public intake source-identity replay

Scientific effect: **NONE**. Review status: **REVIEW_REQUIRED**.

This package exercises the new public-intake lane using already public SIDE24 bytes. Source: Math- commit `9d7b6802424fb4715b31999066aafca8ee2f3cca`, `coefficients/side24_v1/ENCLOSURE.json`.

Method: read the exact public Git bytes, check 1090 bytes and SHA-256 `72b6cd92d31394cdaf5da8919a5d548e902228af1f095cc184158a71d8287811`, parse JSON, and copy only the exact endpoint strings and original acceptance flag to output.json. Python standard library hashlib/json; no numerical recomputation or field simulation.

Result: source identity matched. Both exact intervals retain distinct string endpoints and scientific_acceptance remains false. This establishes a packaging and identity replay only; it is not an analytic review, a bound, or adoption of the source.

Authorship/exposure: OpenAI/Codex, source-exposed. Zero organizational-independence credit. The live guard exercise uses a same-repository incoming-path PR, not an external contributor account. It does not demonstrate installed branch protection.
