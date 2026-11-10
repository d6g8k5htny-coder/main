# SDGA Task A Component source handoff

Date: 2026-11-10 UTC. Source review only.

## Immutable delivery

- Intended path: `tools/sdga/Component.lean`.
- Repository: `d6g8k5htny-coder/main`.
- Final UTF-8 blob: `ecb422db32b1825b71833e91120de3dbc59fd7a6`.
- Exact connector UTF-8 readback compared equal to the authored source (24127 JavaScript string characters; 605 newline-delimited entries).
- This handoff changes no ref, tree, commit, PR, comment, workflow, control, fixture, test, profile, configuration or proof source. Superseded candidate blobs `3bcaa69eef53073e0a5413933eb008b4b3c2d528` and `f82b6627997bef4922f1afe128cdb8b5e450c33a` must not be published.

## Reviewed inputs and implementation boundary

Read approved design `3487f54476b5e6a283839902797bc05aae2ce623`, plan `9eefcb144247aa3f358b735956e580777e2b8945`, actual Controls `b8ec51fbcee8516fcfeb89a637fa08097d918602`, launcher `d7f72edb1b7dac6458c568e9411edf21f2703476`, Safe `814c0f3169b4e967f5b4940d205a479effebeed1` and current Task 1 `14d4617b523c29007da4bdb95ee6ee4dc588a86c`.

The source provides shared public SDGA request, lossless Name, structural-table, load, owner, observation and inventory helpers expected by the unchanged Controls. It reads actual ordered OLean parts together with readModuleDataParts, retains OLean and IR ModuleData and CompactedRegion values, uses actual importModules with private visibility/loadExts=false/trust=0, and validates actual forward/reverse declaration ownership. It inventories the complete raw final population in raw order, preserves raw tables, observes is_private from private-name spelling or exported-population absence, and calls actual Meta.isProp through MetaM.toIO.

The request parser checks exact keys and rejects duplicate JSON keys rather than relying on core Json.parse's overwrite behavior. Request parsing deliberately defers object existence/order to loadExact so C13/C15 can require the specified load-time causes. Path resolution uses initSearchPath/getBuildDir and findOLean without a child invocation, requires exact requested/resolved object equality, exact siblings and canonical real paths. Digest fields remain controller measurements; Lean repetition does not independently authenticate hashes.

Inventory emits only the original Task 1 header/module/declaration/end fields and no protocol extensions. Its per-declaration axiom arrays and terminal union are computed from actual type/value/transitive dependencies using ConstantInfo.getUsedConstantsAsSet plus explicit inductive/constructor/recursor-rule links. Every reached missing lookup is fatal. There is no hardcoded empty axiom array and no permissive collectAxioms shortcut.

Task B owned-axiom/allowlist/safety/replay-policy controls are not implemented or qualified. The dependency traversal is a minimal honest inventory closure; it must not be promoted to full SDGA use before Task B review and native qualification. Task C replayOwned and runReplay deliberately throw SDGA_ERROR NOT_IMPLEMENTED and cannot report replay success. Task D raw IR data is retained, but IR provenance/residue policy is not qualified.

## Pinned API review and assumptions

Reviewed actual leanprover/lean4 commit `5045d0056413266e57c625dcd7c365b10e377c52` sources, including Environment `33928512dadcfa2e42a307499e9228596da0341a`, Meta.Basic `e207f8f137bda69488be8424896fd82bad6cca8d`, FoldConsts `5429906bfda3805967920ea17f77e68611007fdf`, Declaration `52a32d4fdeae1fe12e46cd46a1635420a3b62195`, InferType `0e3d57afdffbb1d4d62bb423b8847afa4216af13`, and pinned Path, JSON Basic/Parser/Printer, String Basic/Legacy, ByteArray Basic, CompactedRegion, Setup, Array Basic, NameMap Basic, Expr and PrettyPrinter sources.

A fresh scoped reviewer independently confirmed the invoked API shapes and found no further concrete source blocker after the author's correction of raw pair bang-indexing: CompactedRegion derives Nonempty, not Inhabited. The final source uses optional access to raw region pairs. A local binder was conservatively renamed moduleName after review; the public request's module field is unchanged.

Source inspection cannot establish Lean parser/elaborator acceptance, successful native compilation, fixture imports, native exception causality or any control result. Those remain explicit uncertainties for the root-owned hosted native run. No filesystem, local execution, ref/PR/comment mutation or rerun occurred in this implementation task. The author remains responsible for the first three hosted fix rounds.

## Status

No execution or semantic-control claim is made. Initial implementation absence is not semantic-negative or mutation-kill credit. Mathematical/scientific/global credit: NONE. scientific_effect=NONE; qualification=BLOCKED; classification_status=BLOCKED; execution_authenticated=false. Full SDGA, kernel replay, complete IR/native coverage and scientific acceptance remain unestablished.
