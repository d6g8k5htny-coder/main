# Research repository conventions: evidence and decisions

30 September 2026 · presentation and engineering audit, not a mathematical review.

The useful benchmark is a reader who can understand the question, find the precise
result, reproduce a stated calculation and cite the source. We retain the existing
federation and Home / Explore / Research / Library experience, with accessible
[Reproduce](site/reproduce.html) and [Cite](site/cite.html) routes. Detailed working
instructions remain in [Workspace and tools](WORKSPACE.md).

## Adopted

- A short public entry with deeper documentation one click away. GUDHI routes
  readers to mathematical tasks, examples and their manuals; scikit-tda addresses
  newcomers and puts installation, examples and citation together.
- One exact calculation before the multi-repository contributor instructions:
  prerequisites, immutable source, commands, observed output and explicit scope.
- Object-specific citation guidance and a local commit-reference helper. Main's
  CFF describes this website/reproduction surface. Proof authorship and imported
  sources retain their own attribution and terms.
- A concise personal profile with its original research-stack essay preserved as
  an explicitly historical, linked source. Profile and site remain distinct objects.

Benchmarks: [GUDHI](https://gudhi.inria.fr/),
[scikit-tda documentation](https://docs.scikit-tda.org/en/latest/),
[GitHub citation support](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files).

## Checked before restructuring

These are dated observations, not a new inventory or scientific-status register.
Main was inspected at `354b697669b59d43563f1b972203fa9ea6d3e6b7`; Math- at
`de54d1da2f6cdde59df3c34bb50ecd85c25ca333`.

| Recommendation received | Observed state and decision |
|---|---|
| Replace approximately 80 claim workflows with four generic workflows | Math- has 86 workflow files, of which 84 use path-filtered PR triggers. The remaining pair is the required downstream caller and its reusable/manual formal workflow. A bounded recent-run sample showed 2–5 workflows per Math PR. Keep the named required-check contracts and exact execution receipts; file count alone does not establish runaway execution. |
| Move Lean pins to the root | Both projects intentionally keep pinned Lean 4.34.1 packages in `formal/`. Main uses core Lean; Math- pins Mathlib separately. Keep the working layout and its documented build command. |
| Remove 453 MB of current artifacts / ignore every PDF | GitHub reports 453,217 KiB of main repository storage, while its current tree has 8,686,480 tracked bytes across 545 blobs. Math-'s current tree has 6,062,389 bytes across 966 blobs. Neither tree contains PDFs. History/other refs were not scanned; these measurements do not diagnose the storage difference. Preserve exact evidence and avoid blanket rules or history rewriting. |
| Put the whole federation under MIT | Main and query- already have licenses; Math- lacks a root license. Imported material needs a component-level rights/attribution inventory before a root grant can accurately describe it. Do not treat main's MIT license as permission for linked sources. |
| Copy an official Lean template verbatim | LeanProject is a customizable blueprint template; Lake supports a configured source directory. Adopt useful reading/formalization links without relocating working source or changing pinned compilers. |
| Transfer/rename repositories, privatize experiments, follow/star accounts | These are separate identity, distribution or access decisions. They do not repair an observed reader or verification defect here. Keep working source URLs and exact pins; do not infer a privacy breach from a repository name. |

Source examples: [Math required caller](https://github.com/d6g8k5htny-coder/Math-/blob/de54d1da2f6cdde59df3c34bb50ecd85c25ca333/.github/workflows/downstream-gate.yml),
[main required caller](https://github.com/d6g8k5htny-coder/main/blob/354b697669b59d43563f1b972203fa9ea6d3e6b7/.github/workflows/workspace-landing.yml),
[main formal package](https://github.com/d6g8k5htny-coder/main/blob/354b697669b59d43563f1b972203fa9ea6d3e6b7/formal/README.md),
[LeanProject](https://github.com/leanprover-community/LeanProject),
[Lake source-directory configuration](https://lean-lang.org/doc/api/Lake/Config/Package.html#Lake.Package.srcDir).

## Citation and licensing corrections

GUDHI is not uniformly MIT across every module and dependency. Its own licensing
explanation and module catalog include distinct GPL/LGPL cases. Copying another
project's top-level license is not a rights inventory.
[GUDHI licensing](https://gudhi.inria.fr/licensing/) ·
[Module catalog](https://gudhi.inria.fr/doc/latest/).

Zenodo does not impose a universal OSI-only deposit rule: access and reuse depend
on the record and stated license. Software Heritage archives public source
without a preliminary license filter; archival presence does not itself grant
reuse rights. Its source archive also does not preserve every extrinsic GitHub
review conversation. For this project, a deposit intended to preserve verification should include
the relevant manifests and review/execution evidence alongside the code.
[Zenodo policies](https://about.zenodo.org/policies/) ·
[Software Heritage FAQ](https://www.softwareheritage.org/software-heritage-faq/).

CITATION.cff enables GitHub's APA/BibTeX interface. A version DOI should identify
an actual deposited version; a concept DOI identifies a version family. Neither
is fabricated to make a repository look complete. Zenodo's GitHub integration
can take license metadata from `.zenodo.json` or CFF ahead of GitHub's detected
license, so those declarations must agree with the rights actually held.
[Zenodo citation guidance](https://support.zenodo.org/help/en-gb/25-citations/234-how-to-share-or-cite-a-zenodo-record) ·
[License ingestion precedence](https://support.zenodo.org/help/en-gb/24-github-integration/149-how-to-specify-a-license-for-a-software-record-on-github).

## Remaining work with a concrete purpose

- Inventory original and imported Math- material before adding accurate licensing
  and citation metadata. Preserve per-object authorship and review identities.
- Add query- citation/support metadata against its actual package, without
  presenting it as a theorem checker or advertising an unpublished package release.
- Prepare an archival release only after its content, licenses and evidence
  package are fixed; publish real archive identifiers after verified deposition.
- Consider PR-only cancellation of superseded aggregate CI runs if measured
  queue/compute costs justify it. One observed overlap was 53 seconds; that does
  not establish a current large bottleneck. Main-push receipts and every required
  current-commit check must remain intact.

The [current workflow](../governance/OP-WORKFLOW-20260930.md) remains the operational
entry. Community files, packaging, citation and green CI do not replace mathematical
review, formal alignment, imported hypotheses or required independence.
