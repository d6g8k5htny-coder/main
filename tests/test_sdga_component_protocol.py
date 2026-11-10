"""Pure Task A dependency contracts; authored fixtures, not native observations.

These tests run in ordinary unittest discovery without Lean, network or flags.
The production component is loaded inside each method: absence is an explicit
test failure, never an import-time error, skip or expected semantic rejection.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
COMPONENT = ROOT / "tools" / "sdga_component.py"

# Source-shaped fixtures from pinned Lean 5045d005 ParseImportsFast and Setup.
# No filename is present in a row; expected_files supplies positional binding.
SOURCE_SHAPED_HEADERS = r"""
{"imports":[
  {"result":{"imports":[
    {"module":"Init","importAll":false,"isExported":true,"isMeta":false},
    {"module":"Init","importAll":false,"isExported":true,"isMeta":true}
  ],"isModule":false},"errors":[]},
  {"result":{"imports":[
    {"module":"Fixture.Safe","importAll":false,"isExported":true,"isMeta":false},
    {"module":"Lean","importAll":true,"isExported":false,"isMeta":true}
  ],"isModule":true},"errors":[]},
  {"result":{"imports":[],"isModule":true},"errors":[]}
]}
"""
ONE_HEADER = (
    '{"imports":[{"result":{"imports":[],"isModule":true},"errors":[]}]}'
)


class DependencyProtocolTests(unittest.TestCase):
    def component(self):
        self.assertTrue(
            COMPONENT.is_file(),
            "IMPLEMENTATION_ABSENCE: tools/sdga_component.py; "
            "no pure API semantic credit",
        )
        spec = importlib.util.spec_from_file_location("sdga_component_protocol", COMPONENT)
        self.assertIsNotNone(spec, "component loader unavailable")
        self.assertIsNotNone(spec.loader, "component loader unavailable")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def rejected(self, prefix, call):
        with self.assertRaisesRegex(ValueError, "^" + prefix + ":"):
            call()

    def test_headers_preserve_source_imports_and_positional_file_order(self):
        component = self.component()
        result = component.parse_dependency_headers(
            SOURCE_SHAPED_HEADERS,
            ("z/Legacy.lean", "a/Imports.lean", "Prelude.lean"),
        )
        self.assertEqual(result, {
            "z/Legacy.lean": ("Init", "Init"),
            "a/Imports.lean": ("Fixture.Safe", "Lean"),
            "Prelude.lean": (),
        })
        self.assertEqual(
            tuple(result), ("z/Legacy.lean", "a/Imports.lean", "Prelude.lean")
        )

    def test_headers_accept_empty_pure_census(self):
        component = self.component()
        self.assertEqual(component.parse_dependency_headers('{"imports":[]}', ()), {})

    def test_headers_refuse_error_rows_even_when_a_result_is_present(self):
        component = self.component()
        for payload in (
            '{"imports":[{"errors":["malformed-header.lean:2:13: expected identifier"]}]}',
            '{"imports":[{"result":{"imports":[],"isModule":true},'
            '"errors":["malformed-header.lean:2:13: expected identifier"]}]}',
        ):
            with self.subTest(payload=payload):
                self.rejected("DEPENDENCY_HEADERS", lambda: (
                    component.parse_dependency_headers(payload, ("Bad.lean",))
                ))

    def test_headers_require_exact_unambiguous_row_census(self):
        component = self.component()
        for payload, files in (
            ('{"imports":[]}', ("A.lean",)),
            (ONE_HEADER, ()),
            (json.dumps({"imports": json.loads(ONE_HEADER)["imports"] * 2}),
             ("A.lean", "A.lean")),
        ):
            with self.subTest(payload=payload, files=files):
                self.rejected("DEPENDENCY_HEADERS", lambda: (
                    component.parse_dependency_headers(payload, files)
                ))

    def test_headers_refuse_duplicate_json_keys_at_every_schema_level(self):
        component = self.component()
        payloads = (
            '{"imports":[],"imports":[]}',
            '{"imports":[{"result":{"imports":[],"isModule":true},'
            '"errors":[],"errors":[]}]}',
            '{"imports":[{"result":{"imports":[],"imports":[],"isModule":true},'
            '"errors":[]}]}',
            '{"imports":[{"result":{"imports":[{"module":"A","module":"B",'
            '"importAll":false,"isExported":true,"isMeta":false}],'
            '"isModule":true},"errors":[]}]}',
        )
        for index, payload in enumerate(payloads):
            with self.subTest(level=index):
                self.rejected("DEPENDENCY_HEADERS", lambda: (
                    component.parse_dependency_headers(
                        payload, () if index == 0 else ("A.lean",)
                    )
                ))

    def test_headers_refuse_malformed_or_non_schema_values(self):
        component = self.component()
        payloads = (
            'not JSON',
            '[]',
            '{"imports":{}}',
            '{"imports":[[]]}',
            '{"imports":[{"result":null,"errors":[]}]}',
            '{"imports":[{"errors":[]}]}',
            '{"imports":[{"result":{"imports":[],"isModule":true}}]}',
            '{"imports":[{"result":{"imports":[],"isModule":true},"errors":""}]}',
            '{"imports":[{"result":{"imports":[]},"errors":[]}]}',
            '{"imports":[{"result":{"imports":[],"isModule":0},"errors":[]}]}',
            '{"imports":[{"result":{"imports":{},"isModule":true},"errors":[]}]}',
            '{"imports":[{"result":{"imports":["A"],"isModule":true},"errors":[]}]}',
            '{"imports":[{"result":{"imports":[{"module":7,"importAll":false,'
            '"isExported":true,"isMeta":false}],"isModule":true},"errors":[]}]}',
            '{"imports":[{"result":{"imports":[{"module":"","importAll":false,'
            '"isExported":true,"isMeta":false}],"isModule":true},"errors":[]}]}',
            '{"imports":[{"result":{"imports":[{"module":"A","importAll":false,'
            '"isExported":true}],"isModule":true},"errors":[]}]}',
            '{"imports":[{"result":{"imports":[{"module":"A","importAll":false,'
            '"isExported":true,"isMeta":1}],"isModule":true},"errors":[]}]}',
            '{"imports":[],"unknown":true}',
            '{"imports":[{"result":{"imports":[],"isModule":true},'
            '"errors":[],"unknown":true}]}',
            '{"imports":[{"result":{"imports":[],"isModule":true,"unknown":true},'
            '"errors":[]}]}',
            '{"imports":[{"result":{"imports":[{"module":"A","importAll":false,'
            '"isExported":true,"isMeta":false,"unknown":true}],'
            '"isModule":true},"errors":[]}]}',
        )
        for index, payload in enumerate(payloads):
            with self.subTest(shape=index):
                self.rejected("DEPENDENCY_HEADERS", lambda: (
                    component.parse_dependency_headers(
                        payload, () if index == 16 else ("A.lean",)
                    )
                ))

    def test_topology_emits_dependencies_before_their_users(self):
        component = self.component()
        self.assertEqual(
            component.topological_modules(
                {"C": (), "A": ("B",), "B": ("C",)}, ("A", "B", "C")
            ),
            ("C", "B", "A"),
        )

    def test_topology_uses_expected_order_among_currently_ready_vertices(self):
        component = self.component()
        cases = (
            (
                {"App": ("Library",), "Library": ("Second",),
                 "First": (), "Second": (), "Isolated": ()},
                ("App", "Library", "First", "Second", "Isolated"),
                ("First", "Second", "Library", "App", "Isolated"),
            ),
            (
                {"Top": ("Left", "Right"), "Right": ("Base",),
                 "Base": (), "Left": ("Base",)},
                ("Top", "Left", "Right", "Base"),
                ("Base", "Left", "Right", "Top"),
            ),
        )
        for edges, expected, want in cases:
            with self.subTest(edges=edges):
                self.assertEqual(component.topological_modules(edges, expected), want)

    def test_topology_handles_isolated_empty_and_repeated_import_occurrences(self):
        component = self.component()
        for edges, expected, want in (
            ({"C": (), "A": (), "B": ()}, ("B", "C", "A"), ("B", "C", "A")),
            ({}, (), ()),
            ({"A": ("B", "B"), "B": ()}, ("A", "B"), ("B", "A")),
        ):
            with self.subTest(edges=edges):
                self.assertEqual(component.topological_modules(edges, expected), want)

    def test_topology_requires_unique_exact_fixed_vertices(self):
        component = self.component()
        for edges, expected, prefix in (
            ({"A": ()}, ("A", "A"), "DUPLICATE_MODULE"),
            ({"A": ()}, ("A", "B"), "MISSING_LOCAL_EDGE"),
            ({"A": (), "B": ()}, ("A",), "MISSING_LOCAL_EDGE"),
        ):
            with self.subTest(edges=edges, expected=expected):
                self.rejected(prefix, lambda: component.topological_modules(edges, expected))

    def test_topology_refuses_local_references_outside_the_fixed_census(self):
        component = self.component()
        for edges, expected in (
            ({"A": ("Missing",)}, ("A",)),
            ({"A": (), "B": ("Missing",)}, ("A", "B")),
        ):
            with self.subTest(edges=edges):
                self.rejected("MISSING_LOCAL_EDGE", lambda: (
                    component.topological_modules(edges, expected)
                ))

    def test_topology_refuses_self_mutual_and_disconnected_cycles(self):
        component = self.component()
        for edges, expected in (
            ({"A": ("A",)}, ("A",)),
            ({"A": ("B",), "B": ("A",)}, ("A", "B")),
            ({"Free": (), "A": ("B",), "B": ("A",)}, ("Free", "A", "B")),
        ):
            with self.subTest(edges=edges):
                self.rejected("DEPENDENCY_CYCLE", lambda: (
                    component.topological_modules(edges, expected)
                ))


if __name__ == "__main__":
    unittest.main()
