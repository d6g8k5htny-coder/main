# Task A pure dependency API contract

This additive test-first slice proposes only the two pure APIs below in
tools/sdga_component.py. Production code remains absent until root observes
these tests fail for explicit implementation absence and authorizes the minimal
implementation. No compile helper, observation helper, inventory combiner,
production CLI or generic command framework is part of this slice.

The ordinary offline suite tests/test_sdga_component_protocol.py declares
12 unittest methods: six header methods containing 31 API call vectors and six
topology methods containing 14 vectors, for 45 calls when every body executes.
It uses only Python's standard library, with no Lean, network, enable flag or
skip path. Each method loads the component after asserting its file exists;
absence must be a method failure, not an import-time error or semantic-adverse
pass. These are authored source-shaped fixtures, not recorded native output.
The existing explicit Task A launcher remains exactly six methods/39 subcases.

## Header API

parse_dependency_headers(payload: str, expected_files: tuple[str, ...])
returns dict[str, tuple[str, ...]], preserving expected_files insertion order
and each row's import occurrence order, including legitimate duplicates such
as implicit runtime/meta Init imports.

The shape comes directly from pinned Lean release
5045d0056413266e57c625dcd7c365b10e377c52:

- src/Lean/Parser/ParseImportsFast.lean, blob
  7ef54dfcae1459216929a50bd4317934c9e37c47: PrintImportsResult has imports rows;
  PrintImportResult has optional result and errors.
- src/Lean/Setup.lean: ModuleHeader has imports and isModule; each Import has
  module, importAll, isExported and isMeta.
- src/Lean/Data/Json/FromToJson/Basic.lean documents that optional fields ending
  with a question mark omit that mark, and omit the field when absent. Thus
  result? becomes result, and failed parsing can omit result completely.

Require exact schema keys, actual JSON booleans for the Boolean fields,
nonempty module strings, a present non-null result, and errors=[] for every
row. Reject duplicate JSON object keys at every level, malformed JSON,
missing/wrong fields, unknown fields, row-count mismatch and duplicate
expected file paths with ValueError beginning DEPENDENCY_HEADERS:.

Rows contain no filename. Binding is positional to expected_files, with exact
count; this API cannot authenticate source-file order or detect a reordered
native invocation from payload alone. Future native acquisition must bind the
actual argv, source identities and original streams. A zero child exit does
not justify accepting a row with errors; the existing native C15 malformed
header control remains necessary and unchanged.

## Topology API

topological_modules(edges: dict[str, tuple[str, ...]],
expected: tuple[str, ...]) returns tuple[str, ...]. Each key is a fixed local
module and each value lists its local dependencies. Every dependency precedes
its users. Among vertices currently ready, choose the earliest position in
expected, including a newly ready vertex before a later ready vertex.
Dictionary insertion order must not choose the result.

Expected vertices must be unique (ValueError prefix DUPLICATE_MODULE:).
Adjacency keys must equal the fixed vertex census, and every dependency must
be in that census (prefix MISSING_LOCAL_EDGE: for missing/extra adjacency
vertices or a dangling local dependency). Self, mutual and disconnected cycles
fail with prefix DEPENDENCY_CYCLE:. Repeated import occurrences are accepted
and may be deduplicated internally for graph traversal.

Both APIs accept the empty pure census, returning {} and () respectively.
This boundary behavior does not authorize an empty production observation:
the future controller must require the fixed nonempty 99-module cohort and
bind/filter known external/core imports before supplying LOCAL edges. Unknown
provided local edges must never be silently ignored.

## Execution and limits

Root will publish the test-only checkpoint and run it normally and optimized,
sequentially, before minimal production implementation. Planned direct runs:

python3 -B -S tests/test_sdga_component_protocol.py -v
python3 -B -O -S tests/test_sdga_component_protocol.py -v

No command was executed to author this slice. Source review and blob readback
do not establish test execution. A missing component yields twelve explicit
method failures before any of the 45 semantic calls; those calls receive no
semantic credit. Native contracts, original failures and future production
qualification remain separately reported. scientific_effect=NONE;
classification_status=BLOCKED; qualification=BLOCKED;
execution_authenticated=false.
