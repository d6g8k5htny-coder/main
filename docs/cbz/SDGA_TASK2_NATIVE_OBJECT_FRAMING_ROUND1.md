# SDGA Task A native binary-read family, author fix round 1

Date: 2026-11-10 UTC. Source-only staged repair; fresh native outcome pending.

## Immutable candidate and scope

- Intended file: `tools/sdga/Component.lean`.
- Repository: `d6g8k5htny-coder/main`.
- Candidate UTF-8 blob: `a46ec37234825ead870b8166ef28d8fbb4997e6f`.
- Exact connector UTF-8 readback matched the authored source.
- Baseline: `ecb422db32b1825b71833e91120de3dbc59fd7a6`.
- Delta: bounded preflight helpers immediately before the existing readParts and one `preflightParts parts` call before the unchanged `readModuleDataParts (parts.map (·.path))`. Removing that helper block and call reconstructs the baseline source exactly.
- Superseded draft `6fd06f872bd076880052eeaa697eae455d916a8a` must not be published.

No loading model, request/Name/JSONL contract, inventory row, axiom traversal, judgment, replay stub, fixture, controller, control, test, workflow, profile or proof assertion is changed. No ref, tree, commit, PR, comment, filesystem/local execution or rerun was performed here.

## Actual failure read from primary evidence

I read the primary job logs for run `42745988350`, attempt 1, job `129598081546`, and parsed its retained original command rows. In both modes, only `control-C13-truncated-private` exited 139 with stdout=0 bytes, stderr=0 bytes, timed_out=false, cleanup.confirmed=true, group_absent=true, leader_reaped=true and no cleanup signals. It emitted no OBJECT_READ/subcase record, so its status remains not_run. That crash provides no semantic refusal credit.

The root/publisher observation is six native methods, five passing and C13 failing; 38/39 subcases passed (33 Lean and five Python), while the existing pure suite passed 12/12. The valid fixture inventory and independent raw/core oracle reconciled 15 real CIs, owners, private/Prop/kind fields, no OLean extras and the unchanged target split 3/1/2. Those valid results do not change the failed truncation control.

The truncator changes only the copied genuine private object to the first quarter of its original bytes and updates the measured request digest. Missing-server and reordered-parts already return their correct typed causes. The crash occurs along the actual loadExact/readParts route, before a native expected-refusal row can be produced.

## Pinned format and root-cause mechanism

All source below is from leanprover/lean4 commit `5045d0056413266e57c625dcd7c365b10e377c52`.

Ordinary module parts use v2. There is **no recorded total region-length field in v2**, so this repair does not invent one. The fixed header is marker[5], version[1], flags[1], lean_version[33], githash[40], base_addr[size_t]. On the supported Linux x86_64 LE64 target this is 88 bytes, base_addr starts at byte 80, and compacted data begins with an eight-byte saved-root-address word at byte 88.

The native reader validates the header, maps the physical file length, then trusts the saved root. Its mmap fast path can return that address without walking or checking the root against the physical mapping. Constructor compaction serializes children before the constructor, placing a fresh ModuleData root near the file end. A quarter-file truncation can preserve the header/root word but remove that root object. The fresh private ModuleData differs from prior populations; it cannot be assumed to be a deduplicated earlier root.

This is a source-supported unchecked-root failure mechanism matching the actual truncation-only crash. No native stack trace or binary-header dump was retained, so the precise faulting instruction/address has not been directly observed. The fresh native guard diagnostic must establish the actual refused bounds and original C13 must still run; this report makes no post-fix execution claim.

| Pinned source/blob | Lines | Evidence |
|---|---|---|
| src/Lean/Environment.lean `33928512dadcfa2e42a307499e9228596da0341a` | 1772-1790 | saveModuleDataParts calls CompactedRegion.save with default allowClosures=false; readModuleDataParts calls unsafe CompactedRegion.read with earlier regions. |
| src/library/module.cpp `402e5220e1379d1204beefec0817ce9dda74294a` | 106-144 | Exact header fields, static packed-size assertion and v2/v3 layout distinction; v2 has no data_size/trailer. |
| same module.cpp | 271-273, 316-336 | Writer uses 64KiB logical-base alignment and ordinary v2 [header][data] with machine-word-aligned compactor allocation. |
| same module.cpp | 474-497, 533-535, 580-608 | Reader takes physical stat size, validates only fixed header compatibility, mmaps, and passes physical data range to region_reader. |
| src/runtime/compact.cpp `73c9a4e18798496bc764cdb7e9899021eed3e4b3` | 163-180, 255-264, 297-309, 461-469 | Allocations round to machine words; copied constructor header records its actual size; children precede constructor; saved root slot is allocated first. |
| same compact.cpp | 518-539, 542-547, 648-665 | Dependency translation permits earlier-part sharing; reader.move rounds sizes; mmap fast path returns saved root without validating its physical extent. |
| src/include/lean/lean.h `01a73c4462b89c866e82f81d87a11d6473aeffcc` | 124-156, 764-783 | Compacted small-object header: m_rc=0, m_cs_sz is the 16-bit size at byte 4, m_tag at byte 7; ModuleData's single constructor has tag 0. |
| src/Init/System/IO.lean `64faef771c3b33af437dde294e5472dced69f6a5` | 841-847, 963-971, 1115-1128 | Safe Handle.read, withFile and Metadata.byteSize APIs. |
| src/Init/System/Platform.lean `4ff07f93786f2447f786d6e6d1f4da448da875b1` | target/isLinux/numBits definitions | Explicit target support check for the interpreted ABI. |

## Bounded production guard

Before any selected part is deserialized, readParts now:

1. Refuses unsupported ABI/format through OBJECT_READ: this bounded implementation supports Linux x86_64 LE64 ordinary v2 parts. v3 closure-bearing compacted snapshots are explicit unsupported input and never silently deserialized.
2. Reads the exact source-derived 88-byte header plus eight-byte root word using safe byte IO, handling short reads explicitly.
3. Checks magic, v2, physical machine-word completion, 64KiB logical-base alignment, nonoverflowing Nat address arithmetic and nonoverlapping part ranges.
4. Requires the saved root's header to lie inside the actual current or earlier selected part data range. Earlier-part roots are permitted for genuine compactor deduplication.
5. Reads only that root's eight-byte ordinary header through bounded-memory streaming. It checks the recorded compacted constructor header and requires the full word-rounded declared root size to fit the corresponding physical file.
6. Calls the original readModuleDataParts only after those checks. IO/framing exceptions use the existing single OBJECT_READ error route.

The 64KiB streaming read chunk is a memory bound, not a minimum accepted object size. The only prefix/extent lengths used for acceptance derive from the actual pinned format/runtime layout. In particular, rootSize itself is not assumed word-aligned: the final guard mirrors compactor allocation rounding, avoiding an unnecessary mimalloc assumption.

## Independent review, limits and fresh native obligations

A fresh scoped reviewer read exact candidate `a46ec37234825ead870b8166ef28d8fbb4997e6f` and the pinned native/Lean sources. It found no blocking source finding for the bounded v2 check and confirmed offsets, alignment, earlier-part root sharing, rounded extent arithmetic and the invoked safe Lean APIs.

This is a root-framing guard, not a generic object graph/type validator or a certification of arbitrary corrupted binaries. It does not validate all reachable objects, dependency files outside the selected parts, or concurrent file mutation. Existing controller before/after artifact checks remain necessary. Any unsupported format, signal, crash, missing row or later failure remains nonzero/blocking; none is a substitute for the requested typed truncation refusal.

Root must require fresh native compilation, the unchanged complete valid inventory/oracle reconciliation and all original six methods/39 subcase IDs in both modes. In particular, C13/truncated-private must now emit its own actual pass row carrying a genuine SDGA_ERROR OBJECT_READ from this route. The current candidate has not run, and no 39/39 or crash-fix claim is made.

Scientific/global/mathematical credit: NONE. scientific_effect=NONE; qualification=BLOCKED; classification_status=BLOCKED; execution_authenticated=false. Task B policy, Task C actual replay and Task D full IR qualification remain pending.
