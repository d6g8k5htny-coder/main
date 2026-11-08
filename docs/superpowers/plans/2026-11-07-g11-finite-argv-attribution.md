# G11 Finite argv Attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the four current G11 static-attribution gaps without crediting dynamic or command-mode Python invocations.

**Architecture:** Extend the existing finite-alternative abstract value so nested conditionals and starred sequence elements retain only explicit list/tuple alternatives. Keep `_command_tool_values` as the fail-closed interpreter-semantic boundary. Update the three optimized child launchers to express the existing `[]`/`-O`/`-OO` behavior as a pure finite value, and give the graph readback CLI one direct real refusal execution outside its `-c` audit launcher.

**Tech Stack:** Python 3 stdlib, `ast`, `unittest`, the existing G11 mutation self-test.

**Spec:** `governance/MASTER_DIRECTIVE_20261029_SWARM.txt`; current disposition evidence is `/workspace/scratch/7abea556a9be/cbz-g11-disposition-20261107/inventory.json`.

## Global Constraints

- Preserve Layer A, especially A5, A7, and A8; do not manufacture static credit.
- Preserve the tools' runtime behavior in normal, `-O`, and `-OO` modes.
- Reject any alternative containing `-`, `-c`, `-m`, unknown interpreter options, a dynamic starred prefix, or a different tool.
- Do not change historical mathematical statements, outcomes, authorship, default refs, hosted PR heads, or canonical global surfaces.
- Work and commit only in `codex/cbz-g11-review-207-209-20261107`; publication requires a later fresh-head compare-and-swap.

## Review Focus

- Nested alternatives must flatten so all concrete branches reach the command validator.
- Starred alternatives must form whole argv alternatives, not concatenate mutually exclusive values into one argv.
- An unsafe branch must remove credit from every branch by intersection.
- Mutable aliases and arbitrary function-returned prefixes must remain unresolved and uncredited.
- The graph audit launcher uses `-c`; only the new direct no-argument refusal run may establish its test attribution.

---

### Task 1: Finite starred argv analysis

**Files:**
- Modify: `tools/cbz/g11_census.py`
- Modify: `tests/test_g11_census.py`
- Modify: `tools/cbz/g11_mutation_selftest.py`

**Interfaces:**
- Consumes: existing `_StaticAlternatives`, `_static_add`, `_python_static_value`, and `_python_sequence_static_value` contracts.
- Produces: flattened finite alternatives and whole-sequence expansion for explicit starred list/tuple alternatives; no new public interface.

- [ ] **Step 1: Write failing positive and adverse tests**

Add real `python_tool_references` behavior tests for explicit `[]`/`-O`/`-OO` nested alternatives and for starred alternatives containing `-c`, mismatched tools, or dynamic values.

- [ ] **Step 2: Run the focused tests to verify RED**

Run: `python3 -B -S -m unittest tests.test_g11_census.ToolChecks -v`

Expected: the explicit finite starred positives fail to receive credit; existing adverse controls remain green.

- [ ] **Step 3: Implement the minimal finite-alternative propagation**

Flatten nested `_StaticAlternatives` in `_static_alternatives`; construct list/tuple values by repeated `_static_add` so starred finite sequences create alternatives instead of unresolved elements. Do not special-case tool paths or interpreter options.

- [ ] **Step 4: Add mutation coverage for each new trust decision**

Add bounded source mutants that remove nested flattening, collapse starred alternatives, or union unsafe branches; update the declared mutation inventory only from the actual generated inventory.

- [ ] **Step 5: Run focused tests and mutation self-test to verify GREEN**

Run: `python3 -B -S -m unittest tests.test_g11_census -v`

Run: `python3 -B -S tools/cbz/g11_mutation_selftest.py`

Expected: all tests and all declared mutants pass with zero skips.

- [ ] **Step 6: Commit**

Commit message: `fix(g11): attribute finite starred Python argv`

### Task 2: Exercise the four active tools

**Files:**
- Modify: `tests/test_architecture_evidence_adapter.py`
- Modify: `tests/test_architecture_artifact_index.py`
- Modify: `tests/test_architecture_graph_artifact_readback.py`
- Modify: `tests/test_atlas_formal_source_bridge.py`

**Interfaces:**
- Consumes: Task 1 finite starred argv analysis.
- Produces: direct, statically attributable, real CLI executions preserving current runtime mode propagation.

- [x] **Step 1: Write the graph readback refusal test and verify RED attribution**

Add a test that runs the actual graph readback CLI with no arguments under the current optimization mode and asserts its nonzero exit plus `ARCHITECTURE_GRAPH_READBACK_REFUSED` marker. The pre-rewrite local reference check left the evidence adapter, artifact index, and graph readback uncredited; the source bridge was already credited by Task 1's now-supported finite boolean mode. The actual graph refusal test passed before its launcher was made statically attributable.

- [x] **Step 2: Express optimization flags as finite pure values**

Use the explicit nested value `['-OO'] if sys.flags.optimize == 2 else (['-O'] if sys.flags.optimize == 1 else [])` in the four test launchers. Preserve `PYTHONOPTIMIZE` removal and every existing subprocess argument.

- [x] **Step 3: Run actual normal and optimized tool suites**

Run each of the four registered test modules under `python3 -B -S`, `python3 -B -S -O`, and `python3 -B -S -OO`.

Expected: all tests pass, zero skips/errors; the graph no-argument refusal test observes the exact refusal marker.

- [ ] **Step 4: Run G11 and verify the exact disposition delta**

Run the repository's documented G11 census command against the worktree revision.

Expected: the four active tools become referenced; the 28 dormant candidates remain uncredited; no dynamic or `-c` audit path gains credit.

- [x] **Step 5: Run full relevant regression and mutation gates**

Run the four tool modules normally and optimized as described above. Run the hosted-isolation boundary module with its documented discovery form: `python3 -B -S -m unittest discover -s tests -p test_architecture_cli_boundaries.py -v`.

Run the boundary discovery command again with `python3 -B -S -O`.

Run `python3 -B -S tools/cbz/g11_mutation_selftest.py` and its `-O` counterpart.

Expected: all selected tests and all declared mutants pass with zero skips.

- [ ] **Step 6: Commit**

Commit message: `test(g11): exercise four active CLI tools`

### Task 3: Local qualification and truthful handoff

**Files:**
- Modify only local ignored execution ledger/evidence unless a later coordinated handoff authorizes repository documentation.

**Interfaces:**
- Consumes: Tasks 1 and 2 commits and actual outputs.
- Produces: exact local commit/tree, commands, exits, counts, limitations, and a fresh disposition inventory; no CBZ-0 claim.

- [ ] **Step 1: Run the repository-wide suite required by the project**

Run the documented full test command in normal and optimized modes; preserve any unrelated failure by name.

- [ ] **Step 2: Run a clean-tree G11 census and adverse mutation qualification**

Expected: exact source is recorded; no cached result substitutes.

- [ ] **Step 3: Self-review the whole local diff**

Review every trust-boundary change against the five Review Focus cases and record the lack of independent reviewer lineage.

- [ ] **Step 4: Commit any evidence-only repository changes, if coordination permits**

Do not push, merge, or overwrite hosted refs. A later publication uses fresh-head compare-and-swap and normal required checks.
