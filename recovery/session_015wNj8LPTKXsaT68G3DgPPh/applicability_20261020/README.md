# C8 review-applicability pre-assessments, 20 October about 00:20Z

These are read-only notes by Anthropic Claude, session `session_015wNj8LPTKXsaT68G3DgPPh` (C8). Dylan Roy — delegated AI review. Credit 0. Scientific effect NONE. They were written before any refresh of main#451 or main#452 existed, and they are not verdicts on any new head.

Following Dylan's instruction at about 00:14Z, each one keeps three questions apart:
1. **Byte identity:** mechanical, from `../previews_20261019_2355/delta_readback.sh`.
2. **Review applicability:** C8's judgment of the base changes, the review's dependencies and the interactions with the original findings. Identical files alone do not establish that a composition is compatible.
3. **Hosted checks:** reported separately, at the actual new head.

| File | What it is |
|---|---|
| `PRE_ASSESSMENT_452.md` | main#452 against `main` `b78134c7`. |
| `PRE_ASSESSMENT_451.md` | main#451 (R1) against `main` `b78134c7`. |
| `deps_452.txt`, `deps_451.txt` | The dependency paths passed to the script as `DEPS`. The PR's own changed files are left out, because the raw-diff check covers them. |
| `preview_451_onto_8f7c2a49.txt` | Added at about 00:53Z. A read-only preview of main#451's head `e9733444` (refreshed at 00:39:40Z) composed with `main` `8f7c2a49`, after #452 merged. Tree `dc72423c`; 192/192 reviewed blobs unchanged; R1's dependencies unchanged. It is a simulated local commit, not a readback of any head. |

A release by 014K or Claude017 ends that worker's reservation. It does not clear a permission denial or authorize another session to perform the denied action.
