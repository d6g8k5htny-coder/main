## What this adds

A **read-only** dependency-tree and evidence-profile view over `frontiers/downstream_gate_20260925/GRAPH.json`. It is the derived view adopted in [main#275](https://github.com/d6g8k5htny-coder/main/issues/275), with its protocol in [main#276](https://github.com/d6g8k5htny-coder/main/pull/276). Scientific effect **NONE**. It writes nothing, promotes nothing and is not a status record.

```sh
python3 -B -S tools/evidence_profile.py                 # every math.* root as a tree
python3 -B -S tools/evidence_profile.py --node math.rn-region.witness-collision
python3 -B -S tools/evidence_profile.py --json          # profiles; status_authority false
```

For each node, the view shows:
- its recorded classification;
- the author and reviewer providers exactly as the record names them;
- whether a recorded reviewer's provider family differs from the author's (`yes`, `no` or `not recorded`, never inferred);
- the transitive required dependencies that do not satisfy the hard gate.

The graph is loaded and validated with the gate's **own** strict loader and dependency rule: an edge is required unless it says otherwise, and only `PROVED_REVIEWED` satisfies a required premise. Axes the graph does not record (blind reconstruction, adversarial attack, formal evidence, numerical reproduction, novelty, human reading) are listed as outside the graph, not reported as passing or failing.

What it already shows on today's graph: `math.lifetime-remainder` (D2) and `math.side24-coefficient` (D3) are ACCEPT-scoped in STATUS but still `AUTHOR_SIDE_CANDIDATE` in GRAPH. `math.rn-fixed-annulus-window` and `math.rn-mesoscopic-reduction` are held only by `math.rn-count-interface`. `math.d1-component.reconciliation-record` has a same-provider review only (`provider-distinct no`). Executing Math-#167 → #160 → #173 changes exactly these rows; the view itself changes none.

## Current disposition

- Head: `cda3e5d81ca44d77f016e6a585f600769b6e6da8`, one commit over main `08f8686c`. Files:
  - `tools/evidence_profile.py`: blob `0a6dbcd4`, 7163 B
  - `tests/test_evidence_profile.py`: blob `c4efb2d5`, 6349 B
- Author: Anthropic Claude, session `017Mi3hx` (Dylan Roy — delegated AI work). The author will not self-merge.
- Requested: **one nonauthor engineering read**, ideally from a provider other than the author's, then a separate eligible integrator. Completed reviews: none. Unresolved findings: none.

## Verification (run locally at the head)

- `python3 -B -S -m unittest discover -s tests -p test_evidence_profile.py -v`: 7/7 OK. The same in `-O` mode: 7/7 OK.
- Full root suite `python3 -B -S -m unittest discover -s tests -p 'test_*.py'`: 237 tests OK (230 existing plus 7 new).
- Sensitivity: 7 single-line mutants of the tool, run on disposable copies, are each rejected by an assertion failure in both modes. No mutant produced an import or setup error. The mutants were:
  - every edge treated as required;
  - classification ignored when collecting unsatisfied dependencies;
  - provider-distinct hard-wired to `yes`;
  - no revisit guard in the tree;
  - non-`math.*` roots included;
  - `status_authority` flipped to true;
  - an ambiguous provider string resolved to one family.
- The live-graph test hashes GRAPH.json before and after the run to confirm nothing is written. Its assertions are structural, so the test stays valid after the register is executed.
- No workflow, packet, manifest, GRAPH, SELECTOR_REGION, STATUS, PROOF_INDEX or catalog file is touched. The pinned downstream-gate counts (71 gate tests, five regression folders) are unaffected.

## Limits

Provider family is read from free-form record strings by token matching (OpenAI/ChatGPT/Codex, Anthropic/Claude, xAI/Grok, Google/Gemini). A string naming no family, or several, reads `not recorded`. "Provider-distinct" is not organizational independence, which stays 0 under shared accounts. The view reports only what GRAPH records; a node's full evidence lives in its review records.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_017Mi3hxjaxV45x6zo6o1ee3
