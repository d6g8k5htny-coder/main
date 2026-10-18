# Pinned source inventory: finite read-only DATA

Repository: `d6g8k5htny-coder/main`. Exact immutable cut: commit `6fce9b8c1a9e723dd4983bfc0d2ebeede09c87f8`, tree `5b3447fa17a4f6596ad585da81cd249b3906404d`, observed by Root at 2026-10-18 00:34:52 UTC. This report makes no claim about later Main.

The full recursive native tree response is untruncated and contains 1,307 entries. `TREE.json` lists every entry with its repo-relative path, type, mode, native SHA, and available size. Full GitHub connector responses were saved before parsing or inspecting `isError`; no remote error or denial occurred. All remote operations were GitHub source GETs. No protection, registry, old failed Actions, Actions logs, artifacts, or runtime endpoints were requested.

Workflow `.github/workflows/research-architecture.yml`: 26,511 bytes, 426 source lines, SHA-256 `e9fc7a1bc875144ebbd81574aef3ce80888796fca8901c80b22dd7af47bee9d2`; computed Git blob SHA-1 `8739e26fd6d9f6cbc40d6015edebe872f79e6e53` equals the immutable native tree entry. Its literal source has workdir `/source`, bind target `/source`, and `ARCHITECTURE_SANDBOX=1`. At lines 328 and 330 it specifies:

```text
python -B -S -m unittest discover -s tests -p "test_architecture_*.py" -v
python -B -O -S -m unittest discover -s tests -p "test_architecture_*.py" -v
```

Eight native files match this literal filename selection, all immediate children of `tests`. Each was fetched by the native Git blob SHA, retained as raw connector response plus UTF-8 source bytes, and checked against both native size and computed Git blob SHA-1. Static AST reading derives **246 distinct explicit native method-ID candidates**, from 8 classes with literal base `unittest.TestCase`. It derives **60 distinct explicit IDs** from separately accepted S01 bytes. Neither historical 171 nor historical 231 was used to calculate a result.

| Repo-relative source path | Explicit methods | Native Git blob SHA-1 |
| --- | ---: | --- |
| `tests/test_architecture_artifact_index.py` | 34 | `46fc4f736f6776c458894fea7e27bac8fd43845e` |
| `tests/test_architecture_binding.py` | 30 | `8d4907f7d487ff97d18923a7e2e061ea0e3388ce` |
| `tests/test_architecture_cli_boundaries.py` | 4 | `9b526cae56af608cc2e4e3f4863cb9223676e294` |
| `tests/test_architecture_evidence_adapter.py` | 82 | `927274620307e6e4e983527106b3eda1e723f377` |
| `tests/test_architecture_graph_artifact_readback.py` | 37 | `5ae0aa7e44b97083dc9170125ac8c50cf7a6c3b3` |
| `tests/test_architecture_graph_export.py` | 18 | `429d6d6cf60cee2fe8d0f3b0aa01f7ea6a43b577` |
| `tests/test_architecture_receipt.py` | 36 | `ddb8062b09694d3caf1f83cfa0afab2a66579563` |
| `tests/test_architecture_sandbox.py` | 5 | `ee4a8fd474079c4520d2051d34190367ac5a9d40` |
| `tests/test_architecture_lease_protocol.py` | 60 | `SEPARATE S01; no native blob` |

The accepted S01 source is 179,658 bytes, SHA-256 `5eaa431716e8b28f0df20fb6d3da9a6e875e60fb0c4c1d57ead1654270c810e5`, 2,769 lines. Its literal intended path is `tests/test_architecture_lease_protocol.py`, with `TestLeaseProtocol(unittest.TestCase)` at line 1393. It is a **separate proposed overlay**. The complete native tree has neither `tests/test_architecture_lease_protocol.py` nor `architecture/lease_protocol.py`. No native Main lease product/test or current composition is established. **246 + 60 = 306** is arithmetic over two separate static source inventories, not a current suite or runtime count.

`INVENTORY.json` records all 306 explicit method IDs, repo-relative path, class, method, exact definition line and end line, definition text, decorators, source SHA-256, native blob when applicable, and ordering. Files use repo-relative lexical order; classes and methods use source order. A separate lexical ID ordinal is included. Native candidates and S01 candidates retain separate origin and native-at-snapshot fields. The same static native ID candidates apply to both literal normal and -O invocations; they are not counted twice.

All nine test classes directly name `unittest.TestCase`; no project-defined test base needed further fetches. No duplicate immediate test method names, nested or module-level test function definitions, module `load_tests` definitions, method decorators, or top-level loop/try/with statements were observed in these nine ASTs. The native sandbox class has a `unittest.skipUnless` decorator at line 10 requiring `ARCHITECTURE_SANDBOX=1`; the workflow specifies that literal value. The CLI-boundary module imports the artifact-index and graph-readback modules as fixture modules; both exact source files are included. Source fixture `REQUIRED_IDS` / `REQUIRED_METHODS` strings were excluded from method counts.

These observations do not establish actual unittest discovery completeness. No source was executed or imported. The pinned runtime's standard-library loader and base class were not executed or resolved, nor were imports, package eligibility, dynamic or externally patched bindings, skip disposition, discovery errors, runtime ordering, outcomes, or normal/-O parity observed. The untruncated tree contains `tests/casefold` and `tests/fixtures` and no `__init__.py` under `tests`; recursive loader behavior was not exercised. This is complete for the explicitly stated AST definition rule and fetched literal source selection only.

Only own inert standard-library AST/JSON/hash/text processing was performed. There were no project helper/function calls, eval/import of project sources, Git CLI, Node, Lean, browser, model, containers, tests, dependency installation, non-GitHub network access, remote comments, or remote mutations. Outputs are confined to this new child directory; directory mode 0700 and regular output modes 0444 with single links are checked in `SEAL.json`.

State remains: partition **NOT_DESIGNED**; schedule **UNADMITTED**; harness **NOT_PREPARED**; product **LOCKED**; runtime **NOTRUN**; org **0**; science **NONE**. This finite source inventory supports later design only. Only this read-only source-inventory scope is released after sealing outputs; no product/design/runtime writer claim is taken.
