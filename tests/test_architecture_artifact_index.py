"""External contract for the inert, explicitly pinned Artifact Index projection.

Every fixture below is synthetic TEST data. It models formattedValue API JSON,
not a real Sheet capture, native revision authentication, or delivered evidence.
"""

import base64
import copy
import hashlib
import http.server
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import urllib.request
import unittest


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "tools" / "artifact_index_export.py"
HEADER = [
    "Artifact ID", "Title", "Org", "Class", "Priority", "Topics / object tags",
    "Status", "Authority / canonical impact", "Dependencies or supersession",
    "Modified UTC", "Source", "Notes",
]
PUBLIC_COLUMNS = [
    "Artifact ID", "Title", "Org", "Class", "Status",
    "Authority / canonical impact", "Dependencies or supersession", "Modified UTC",
]
PIN_FLAGS = {
    "capture_sha256": "--expected-capture-sha256",
    "metadata_sha256": "--expected-metadata-sha256",
    "policy_sha256": "--expected-policy-sha256",
    "bindings_sha256": "--expected-bindings-sha256",
    "graph_nodes_sha256": "--expected-graph-nodes-sha256",
}
ROLES = ["source", "review", "kernel", "computation", "alignment", "teaching-source"]
TITLE = "TEST Artifact Index"
RANGES = [
    "'TEST Artifact Index'!A1:L1",
    "'TEST Artifact Index'!A936:L937",
    "'TEST Artifact Index'!A1001:L1001",
]

# Trusted test launcher, executed only in the external child. Its Python audit
# observer covers the fixture paths; it is not an operating-system sandbox.
AUDIT_LAUNCHER = r'''
import json
import os
import runpy
import sys

cli_path, report_number, fixture_path, secret_path = sys.argv[1:5]
cli_arguments = sys.argv[5:]
report_fd = int(report_number)
fixture_root = os.path.normcase(os.path.abspath(fixture_path))
labels = ("read", "write", "mkdir", "remove", "rename", "rmdir")
counts = dict.fromkeys(labels, 0)
calibration = dict(counts)
guard = False

def fixture_target(path, dir_fd=None):
    if isinstance(path, int):
        # Standard streams and the trusted report descriptor are allowed.
        # Other opaque descriptors conservatively count as fixture attempts.
        return path not in (0, 1, 2, report_fd)
    if path is None:
        path = os.getcwd()
    path = os.fsdecode(path)
    if not os.path.isabs(path) and dir_fd not in (None, -1):
        # A relative dir_fd target cannot safely be assigned to another path
        # without extra I/O, so observe/guard it conservatively.
        return True
    path = os.path.normcase(os.path.abspath(path))
    return path == fixture_root or path.startswith(fixture_root + os.sep)

def audit(event, arguments):
    selected = []
    if event == "open" and fixture_target(arguments[0]):
        flags = arguments[2]
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC):
            selected.append("write")
        if not flags & os.O_WRONLY:
            selected.append("read")
    elif event in ("os.listdir", "os.scandir") and fixture_target(arguments[0]):
        selected.append("read")
    elif event == "os.mkdir" and fixture_target(arguments[0], arguments[2]):
        selected.append("mkdir")
    elif event in ("os.remove", "os.rmdir") and fixture_target(arguments[0], arguments[1]):
        selected.append("remove" if event == "os.remove" else "rmdir")
    elif event == "os.rename" and (
        fixture_target(arguments[0], arguments[2]) or fixture_target(arguments[1], arguments[3])
    ):
        selected.append("rename")
    for label in selected:
        counts[label] += 1
    if selected and guard:
        raise RuntimeError("TEST_FIXTURE_IO_GUARD")

sys.addaudithook(audit)
try:
    # Positive calibration uses trusted paths, never captured code or values.
    with open(secret_path, "rb") as stream:
        stream.read()
    temporary = os.path.join(fixture_root, "TEST-audit-temporary")
    renamed = os.path.join(fixture_root, "TEST-audit-renamed")
    directory = os.path.join(fixture_root, "TEST-audit-directory")
    with open(temporary, "wb") as stream:
        stream.write(b"TEST calibration")
    os.rename(temporary, renamed)
    os.remove(renamed)
    os.mkdir(directory)
    os.rmdir(directory)
    os.listdir(fixture_root)
    calibration = dict(counts)
    counts = dict.fromkeys(labels, 0)
    guard = True
    sys.argv = [cli_path, *cli_arguments]
    sys.path[0] = os.path.dirname(cli_path)
    runpy.run_path(cli_path, run_name="__main__")
finally:
    report = {"calibration": calibration, "attempts": counts, "optimize": sys.flags.optimize}
    encoded = (json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")
    os.write(report_fd, encoded)
    os.close(report_fd)
'''


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def native_bytes(value):
    return (json.dumps(value, separators=(",", ":"), ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def row(values):
    return {"values": [{"formattedValue": value} for value in values]}


def selected_values(artifact_id, label):
    return [
        artifact_id, label + " — α; TEST.node.a is only title text", "TEST Org",
        "dated review record", "PRIVATE_PRIORITY_" + artifact_id,
        "PRIVATE_TOPICS_" + artifact_id, "HISTORICAL / NOT ACCEPTED",
        "independence=0; scope=historical only; as of 2026-10-01",
        "supersedes TEST-OLDER only within the stated scope", "2026-10-01T00:00:00Z",
        "PRIVATE_SOURCE_" + artifact_id, "PRIVATE_NOTES_" + artifact_id,
    ]


def synthetic_native():
    unselected = ["TEST-UNSELECTED"] + ["PRIVATE_UNSELECTED_COL_%02d" % i
                                         for i in range(1, 12)]
    return {
        "spreadsheetId": "TEST-R17-SPREADSHEET",
        "privateAnnotation": "PRIVATE_NATIVE_ANNOTATION",
        "sheets": [{
            "properties": {
                "sheetId": 4242, "title": TITLE,
                "gridProperties": {"rowCount": 1002, "columnCount": 12},
            },
            "data": [
                {"startRow": 0, "startColumn": 0, "rowData": [row(HEADER)]},
                {"startRow": 935, "startColumn": 0, "rowData": [
                    row(selected_values("TEST-ZETA", "Zeta title")),
                    row(selected_values("TEST-ALPHA", "Alpha title")),
                ]},
                {"startRow": 1000, "startColumn": 0, "rowData": [row(unselected)]},
            ],
        }],
    }


def synthetic_packet():
    return {
        "schema_version": 1,
        "capture": {
            "capture_id": "TEST-capture-1", "captured_at": "2026-10-01T12:34:56Z",
            "source": {
                "spreadsheet_id": "TEST-R17-SPREADSHEET", "worksheet_id": 4242,
                "title": TITLE, "ranges": list(RANGES), "cell_fields": "formattedValue",
            },
            "raw_base64": base64.b64encode(native_bytes(synthetic_native())).decode("ascii"),
        },
        "policy": {"schema_version": 1, "artifact_ids": ["TEST-ZETA", "TEST-ALPHA"],
                   "columns": list(PUBLIC_COLUMNS)},
        "bindings": [],
        "graph_nodes": ["TEST.node.z", "TEST.node.a", "TEST.node.m"],
    }


def pins_for(packet):
    capture = packet["capture"]
    raw = base64.b64decode(capture["raw_base64"], validate=True)
    metadata = {key: value for key, value in capture.items() if key != "raw_base64"}
    return {
        "capture_sha256": digest(raw), "metadata_sha256": digest(canonical(metadata)),
        "policy_sha256": digest(canonical(packet["policy"])),
        "bindings_sha256": digest(canonical(packet["bindings"])),
        "graph_nodes_sha256": digest(canonical(packet["graph_nodes"])),
    }


def replace_raw(packet, raw):
    packet["capture"]["raw_base64"] = base64.b64encode(raw).decode("ascii")


def get_native(packet):
    return json.loads(base64.b64decode(packet["capture"]["raw_base64"]))


def replace_native(packet, value):
    replace_raw(packet, native_bytes(value))


class ArtifactIndexProjectionContractTests(unittest.TestCase):
    def setUp(self):
        self.packet = synthetic_packet()
        self.pins = pins_for(self.packet)
        self.temp = tempfile.TemporaryDirectory(prefix="TEST-artifact-index-")
        self.addCleanup(self.temp.cleanup)
        self.workdir = Path(self.temp.name)
        self.private_values = ["PRIVATE_NATIVE_ANNOTATION"]
        for block in get_native(self.packet)["sheets"][0]["data"][1:]:
            for native_row in block["rowData"]:
                values = [cell["formattedValue"] for cell in native_row["values"]]
                if values[0] == "TEST-UNSELECTED":
                    self.private_values.extend(values)
                else:
                    self.private_values.extend(values[index] for index in (4, 5, 10, 11))

    def flag_args(self, pins):
        return [argument for name, flag in PIN_FLAGS.items() for argument in (flag, pins[name])]

    def invoke(self, packet=None, *, expected=None, raw=None, args=None, audit=None):
        # Missing exporters must fail every control before launching a child.
        self.assertTrue(CLI.is_file(), "missing required external CLI: tools/artifact_index_export.py")
        packet = self.packet if packet is None else packet
        expected = pins_for(packet) if expected is None else expected
        raw = canonical(packet) + b"\n" if raw is None else raw
        flags = ["-B", "-S"]
        if sys.flags.optimize:
            flags.append("-" + "O" * sys.flags.optimize)
        env = dict(os.environ)
        env.pop("PYTHONOPTIMIZE", None)
        arguments = self.flag_args(expected) if args is None else args

        def launch(command, pass_fds=()):
            return subprocess.run(
                command, input=raw, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=self.workdir, env=env, timeout=20, check=False, pass_fds=pass_fds,
            )

        if audit is None:
            return launch([sys.executable, *flags, str(CLI), *arguments])
        with tempfile.TemporaryDirectory(prefix="TEST-audit-report-") as report_directory:
            report_root = Path(report_directory)
            launcher = report_root / "trusted_launcher.py"
            report_path = report_root / "counts.json"
            launcher.write_text(AUDIT_LAUNCHER, encoding="utf-8")
            report_fd = os.open(report_path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o600)
            try:
                result = launch([
                    sys.executable, *flags, str(launcher), str(CLI), str(report_fd),
                    str(self.workdir), str(audit["secret"]), *arguments,
                ], pass_fds=(report_fd,))
            finally:
                os.close(report_fd)
            result.audit_report = json.loads(report_path.read_bytes())
            return result

    def assert_private_absent(self, result):
        combined = result.stdout + result.stderr
        for value in self.private_values:
            self.assertNotIn(value.encode("utf-8"), combined)
        self.assertNotIn(self.packet["capture"]["raw_base64"].encode("ascii"), combined)

    def expected_projection(self, packet, expected):
        source = packet["capture"]["source"]
        selected = set(packet["policy"]["artifact_ids"])
        projected_rows = []
        for block in get_native(packet)["sheets"][0]["data"]:
            start = block.get("startRow", 0)
            for offset, native_row in enumerate(block["rowData"]):
                values = [cell.get("formattedValue", "") for cell in native_row["values"]]
                if values[0] in selected:
                    projected_rows.append({
                        "artifact_id": values[0], "source_row": start + offset + 1,
                        "fields": {name: values[HEADER.index(name)] for name in PUBLIC_COLUMNS},
                    })
        bindings = [dict(binding, roles=sorted(binding["roles"])) for binding in packet["bindings"]]
        bindings.sort(key=lambda binding: (binding["node_id"], binding["artifact_id"]))
        bound_nodes = {binding["node_id"] for binding in bindings}
        return {
            "schema_version": 1, "scientific_effect": "NONE", "scientific_status_authority": False,
            "custody": "unknown",
            "source": {
                "capture_id": packet["capture"]["capture_id"],
                "captured_at": packet["capture"]["captured_at"],
                "spreadsheet_id": source["spreadsheet_id"], "worksheet_id": source["worksheet_id"],
                "worksheet_title": source["title"], "ranges": source["ranges"],
                "cell_fields": source["cell_fields"], "representation": "native-api-formattedValue-json",
            },
            "identities": expected, "columns": list(PUBLIC_COLUMNS),
            "rows": sorted(projected_rows, key=lambda value: value["artifact_id"]),
            "bindings": bindings, "unmapped_nodes": sorted(set(packet["graph_nodes"]) - bound_nodes),
        }

    def accept(self, packet=None, *, expected=None, raw=None, audit=None):
        packet = self.packet if packet is None else packet
        expected = pins_for(packet) if expected is None else expected
        result = self.invoke(packet, expected=expected, raw=raw, audit=audit)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, b"")
        self.assertTrue(result.stdout.isascii())
        projected = json.loads(result.stdout)
        self.assertEqual(projected, self.expected_projection(packet, expected))
        self.assertIs(type(projected["schema_version"]), int)
        self.assertIs(projected["scientific_status_authority"], False)
        self.assertIs(type(projected["source"]["worksheet_id"]), int)
        for value in projected["rows"]:
            self.assertIs(type(value["source_row"]), int)
        self.assertEqual(result.stdout, canonical(projected) + b"\n")
        self.assert_private_absent(result)
        return result

    def refuse(self, packet=None, *, expected=None, raw=None, args=None):
        result = self.invoke(packet, expected=expected, raw=raw, args=args)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"")
        self.assertEqual(result.stderr, b"ARTIFACT_INDEX_REFUSED\n")
        self.assert_private_absent(result)
        return result

    def test_exact_public_projection_is_stable_and_retains_original_row_positions(self):
        before = sorted(path.name for path in self.workdir.iterdir())
        first = self.accept()
        self.assertEqual(self.accept().stdout, first.stdout)
        projected = json.loads(first.stdout)
        self.assertEqual([(item["artifact_id"], item["source_row"]) for item in projected["rows"]],
                         [("TEST-ALPHA", 937), ("TEST-ZETA", 936)])
        self.assertEqual(projected["unmapped_nodes"], ["TEST.node.a", "TEST.node.m", "TEST.node.z"])
        self.assertEqual(sorted(path.name for path in self.workdir.iterdir()), before)

    def test_all_49_explicit_graph_nodes_remain_unmapped_without_a_binding(self):
        packet = copy.deepcopy(self.packet)
        packet["graph_nodes"] = ["TEST.node.%02d" % index for index in reversed(range(49))]
        projected = json.loads(self.accept(packet).stdout)
        self.assertEqual(len(projected["unmapped_nodes"]), 49)
        self.assertEqual(projected["bindings"], [])

    def test_public_unicode_empty_cells_and_historical_scope_are_preserved_without_promotion(self):
        packet = copy.deepcopy(self.packet)
        native = get_native(packet)
        values = native["sheets"][0]["data"][1]["rowData"][1]["values"]
        values[1]["formattedValue"] = "Unicode α / résumé / 雪; declared text only"
        values[2] = {"unknownAnnotation": "PRIVATE_EMPTY_CELL_ANNOTATION"}
        values[6]["formattedValue"] = "FAILED; skipped; as of 2026-09-30"
        values[7]["formattedValue"] = "independence=0; historical scope only"
        replace_native(packet, native)
        result = self.accept(packet)
        fields = json.loads(result.stdout)["rows"][0]["fields"]
        self.assertEqual(fields["Org"], "")
        self.assertEqual(fields["Status"], "FAILED; skipped; as of 2026-09-30")
        self.assertNotIn(b"PRIVATE_EMPTY_CELL_ANNOTATION", result.stdout)

    def test_default_zero_offsets_and_unknown_native_annotations_are_private_and_inert(self):
        packet = copy.deepcopy(self.packet)
        native = get_native(packet)
        header = native["sheets"][0]["data"][0]
        del header["startRow"]
        for block in native["sheets"][0]["data"]:
            del block["startColumn"]
            block["annotation"] = {"raw": "PRIVATE_BLOCK_ANNOTATION", "nested": [True, None, 1.5]}
        native["sheets"][0]["properties"]["annotation"] = "PRIVATE_PROPERTIES_ANNOTATION"
        replace_native(packet, native)
        result = self.accept(packet)
        self.assertNotIn(b"PRIVATE_BLOCK_ANNOTATION", result.stdout)
        self.assertNotIn(b"PRIVATE_PROPERTIES_ANNOTATION", result.stdout)

    def test_explicit_six_roles_and_bindings_sort_without_inferred_acceptance(self):
        packet = copy.deepcopy(self.packet)
        packet["bindings"] = [
            {"node_id": "TEST.node.z", "artifact_id": "TEST-ZETA", "roles": list(ROLES)},
            {"node_id": "TEST.node.a", "artifact_id": "TEST-ZETA", "roles": ["teaching-source"]},
            {"node_id": "TEST.node.a", "artifact_id": "TEST-ALPHA", "roles": ["review", "source"]},
        ]
        projected = json.loads(self.accept(packet).stdout)
        self.assertEqual(projected["bindings"][2]["roles"], sorted(ROLES))
        self.assertEqual(projected["unmapped_nodes"], ["TEST.node.m"])
        self.assertEqual(projected["custody"], "unknown")

    def test_policy_array_order_is_pinned_even_when_public_rows_have_the_same_order(self):
        packet = copy.deepcopy(self.packet)
        packet["policy"]["artifact_ids"].reverse()
        original = json.loads(self.accept().stdout)
        reordered = json.loads(self.accept(packet).stdout)
        self.assertEqual(original["rows"], reordered["rows"])
        self.assertNotEqual(original["identities"]["policy_sha256"], reordered["identities"]["policy_sha256"])

    def test_each_of_the_five_independent_expected_pins_must_match(self):
        for name in PIN_FLAGS:
            with self.subTest(pin=name):
                expected = dict(self.pins)
                expected[name] = "0" * 64 if expected[name] != "0" * 64 else "1" * 64
                self.refuse(expected=expected)

    def test_changed_input_components_cannot_replace_independently_supplied_pins(self):
        for component in ("capture", "metadata", "policy", "bindings", "graph_nodes"):
            with self.subTest(component=component):
                packet = copy.deepcopy(self.packet)
                if component == "capture":
                    replace_raw(packet, base64.b64decode(packet["capture"]["raw_base64"]) + b" ")
                elif component == "metadata":
                    packet["capture"]["captured_at"] = "2026-10-02T12:34:56Z"
                elif component == "policy":
                    packet["policy"]["artifact_ids"].reverse()
                elif component == "bindings":
                    packet["bindings"] = [{"node_id": "TEST.node.a", "artifact_id": "TEST-ALPHA", "roles": ["source"]}]
                else:
                    packet["graph_nodes"].reverse()
                self.refuse(packet, expected=self.pins)

    def test_all_cli_flags_are_required_once_and_only_full_names_are_accepted(self):
        arguments = self.flag_args(self.pins)
        for index, (name, flag) in enumerate(PIN_FLAGS.items()):
            with self.subTest(flag=flag, fault="missing"):
                self.refuse(args=arguments[:2 * index] + arguments[2 * index + 2:])
            with self.subTest(flag=flag, fault="duplicate"):
                self.refuse(args=arguments + [flag, self.pins[name]])
            with self.subTest(flag=flag, fault="abbreviated"):
                amended = list(arguments)
                amended[2 * index] = flag[:-1]
                self.refuse(args=amended)

    def test_invalid_cli_digests_and_private_argument_values_are_never_echoed(self):
        for name in PIN_FLAGS:
            for value in ("", "a" * 63, "a" * 65, "A" * 64, "g" * 64, "../PRIVATE_ARG_PATH"):
                with self.subTest(pin=name, value=value):
                    expected = dict(self.pins)
                    expected[name] = value
                    result = self.refuse(expected=expected)
                    self.assertNotIn(b"PRIVATE_ARG_PATH", result.stderr)
        for extra in (["--PRIVATE_UNKNOWN_FLAG", "PRIVATE_ARGUMENT_VALUE"], ["/PRIVATE_POSITIONAL_PATH"]):
            with self.subTest(extra=extra):
                result = self.refuse(args=self.flag_args(self.pins) + extra)
                self.assertNotIn(b"PRIVATE_", result.stderr)

    def test_packet_wrappers_are_closed_and_schema_versions_have_exact_integer_types(self):
        wrapped = copy.deepcopy(self.packet)
        wrapped["bindings"] = [{"node_id": "TEST.node.a", "artifact_id": "TEST-ALPHA", "roles": ["source"]}]

        def remaining_pins(packet):
            # Pin every remaining component, so an unrelated mismatch cannot
            # mask the missing/extra-key control. A wholly absent component has
            # no hash to supply and retains the positive fixture's expectation.
            expected = pins_for(wrapped)
            for name, key in (("policy_sha256", "policy"), ("bindings_sha256", "bindings"),
                              ("graph_nodes_sha256", "graph_nodes")):
                if key in packet:
                    expected[name] = digest(canonical(packet[key]))
            if "capture" in packet:
                capture = packet["capture"]
                expected["metadata_sha256"] = digest(canonical({
                    key: value for key, value in capture.items() if key != "raw_base64"
                }))
                if "raw_base64" in capture:
                    expected["capture_sha256"] = digest(base64.b64decode(capture["raw_base64"], validate=True))
            return expected

        paths = [(), ("capture",), ("capture", "source"), ("policy",), ("bindings", 0)]
        for path in paths:
            target = wrapped
            for key in path:
                target = target[key]
            for key in list(target):
                with self.subTest(path=path, missing=key):
                    packet = copy.deepcopy(wrapped)
                    changed = packet
                    for part in path:
                        changed = changed[part]
                    del changed[key]
                    self.refuse(packet, expected=remaining_pins(packet))
            with self.subTest(path=path, extra=True):
                packet = copy.deepcopy(wrapped)
                changed = packet
                for part in path:
                    changed = changed[part]
                changed["PRIVATE_UNRECOGNIZED_WRAPPER"] = "PRIVATE_WRAPPER_VALUE"
                result = self.refuse(packet, expected=remaining_pins(packet))
                self.assertNotIn(b"PRIVATE_UNRECOGNIZED_WRAPPER", result.stderr)
                self.assertNotIn(b"PRIVATE_WRAPPER_VALUE", result.stderr)
        for path in ((), ("policy",)):
            for value in (True, 1.0, "1", 0, 2, None):
                with self.subTest(path=path, schema=value):
                    packet = copy.deepcopy(self.packet)
                    target = packet if not path else packet["policy"]
                    target["schema_version"] = value
                    self.refuse(packet)

    def test_capture_identity_timestamp_and_declared_source_are_strictly_typed(self):
        cases = {
            "capture_id": ("", "x" * 193, " space", "TEST\nID", "é", True, 1, None),
            "captured_at": ("2026-02-30T12:00:00Z", "2026-10-01T24:00:00Z", "2026-10-01T12:34:56+00:00",
                            "2026-10-01T12:34:56.000Z", "2026-10-01", True, None),
        }
        for key, values in cases.items():
            for value in values:
                with self.subTest(field=key, value=value):
                    packet = copy.deepcopy(self.packet)
                    packet["capture"][key] = value
                    self.refuse(packet)
        source_cases = {
            "spreadsheet_id": ("", "x" * 193, "é", True, None),
            "worksheet_id": (True, 4242.0, -1, "4242", None),
            "title": ("", "λ" * 129, "TEST\nTitle", "TEST'Quote", True, None),
            "cell_fields": ("", "formattedValue,userEnteredValue", ["formattedValue"], None),
        }
        for key, values in source_cases.items():
            for value in values:
                with self.subTest(source_field=key, value=value):
                    packet = copy.deepcopy(self.packet)
                    packet["capture"]["source"][key] = value
                    self.refuse(packet)

    def test_native_spreadsheet_worksheet_and_title_must_match_declared_metadata(self):
        for field, value in (("spreadsheetId", "TEST-OTHER-SHEET"), ("sheetId", 4243),
                             ("sheetId", True), ("sheetId", 4242.0), ("title", "TEST Other Title")):
            with self.subTest(field=field, value=value):
                packet = copy.deepcopy(self.packet)
                native = get_native(packet)
                target = native if field == "spreadsheetId" else native["sheets"][0]["properties"]
                target[field] = value
                replace_native(packet, native)
                self.refuse(packet)

    def test_exact_header_order_names_types_and_single_header_are_required(self):
        for fault in ("rename", "swap", "empty", "nonstring", "second_header"):
            with self.subTest(fault=fault):
                packet = copy.deepcopy(self.packet)
                native = get_native(packet)
                header = native["sheets"][0]["data"][0]["rowData"][0]["values"]
                if fault == "rename":
                    header[10]["formattedValue"] = "Sources"
                elif fault == "swap":
                    header[0], header[1] = header[1], header[0]
                elif fault == "empty":
                    header[11] = {}
                elif fault == "nonstring":
                    header[0]["formattedValue"] = 1
                else:
                    native["sheets"][0]["data"][2]["rowData"][0] = row(HEADER)
                replace_native(packet, native)
                self.refuse(packet)

    def test_ranges_require_exact_title_a_to_l_syntax_order_and_nonoverlap(self):
        cases = [[], RANGES + [RANGES[1]], list(reversed(RANGES)),
                 [RANGES[0], "'TEST Artifact Index'!A936:L937", "'TEST Artifact Index'!A937:L1001"],
                 ["TEST Artifact Index!A1:L1", *RANGES[1:]],
                 ["'TEST Other Title'!A1:L1", *RANGES[1:]],
                 ["'TEST Artifact Index'!A0:L1", *RANGES[1:]],
                 ["'TEST Artifact Index'!A01:L1", *RANGES[1:]],
                 ["'TEST Artifact Index'!B1:L1", *RANGES[1:]],
                 ["'TEST Artifact Index'!A1:K1", *RANGES[1:]],
                 [RANGES[0], "'TEST Artifact Index'!A937:L936", RANGES[2]],
                 [RANGES[0], True, RANGES[2]], "PRIVATE_RANGES_STRING"]
        for value in cases:
            with self.subTest(ranges=value):
                packet = copy.deepcopy(self.packet)
                packet["capture"]["source"]["ranges"] = value
                self.refuse(packet)

    def test_grid_dimensions_are_exact_integers_and_bound_all_declared_blocks(self):
        for key, value in (("rowCount", 0), ("rowCount", -1), ("rowCount", True),
                           ("rowCount", 1002.0), ("rowCount", 1000), ("rowCount", 10000001),
                           ("columnCount", 11), ("columnCount", 13), ("columnCount", 12.0),
                           ("columnCount", True)):
            with self.subTest(field=key, value=value):
                packet = copy.deepcopy(self.packet)
                native = get_native(packet)
                native["sheets"][0]["properties"]["gridProperties"][key] = value
                replace_native(packet, native)
                self.refuse(packet)

    def test_native_offsets_match_range_positions_and_refuse_boolean_float_or_missing_nonzero(self):
        for block_index, key, value in ((0, "startRow", 1), (1, "startRow", 934),
                                        (1, "startRow", True), (1, "startRow", 935.0),
                                        (1, "startRow", -1), (1, "startRow", "935"),
                                        (1, "startRow", None), (2, "startColumn", 1),
                                        (2, "startColumn", False), (2, "startColumn", 0.0)):
            with self.subTest(block=block_index, offset=key, value=value):
                packet = copy.deepcopy(self.packet)
                native = get_native(packet)
                block = native["sheets"][0]["data"][block_index]
                if value is None:
                    del block[key]
                else:
                    block[key] = value
                replace_native(packet, native)
                self.refuse(packet)

    def test_native_sheet_block_row_and_cell_shapes_and_cardinalities_are_enforced(self):
        for fault in ("root_array", "no_sheet", "two_sheets", "no_block", "extra_block", "block_order",
                      "short_rows", "extra_rows", "short_cells", "extra_cells", "cell_not_object",
                      "row_not_object", "block_not_object", "missing_properties", "missing_grid"):
            with self.subTest(fault=fault):
                packet = copy.deepcopy(self.packet)
                native = get_native(packet)
                sheet = native["sheets"][0]
                blocks = sheet["data"]
                if fault == "root_array":
                    native = [native]
                elif fault == "no_sheet":
                    native["sheets"] = []
                elif fault == "two_sheets":
                    native["sheets"].append(copy.deepcopy(sheet))
                elif fault == "no_block":
                    blocks.pop()
                elif fault == "extra_block":
                    blocks.append(copy.deepcopy(blocks[2]))
                elif fault == "block_order":
                    blocks[1], blocks[2] = blocks[2], blocks[1]
                elif fault == "short_rows":
                    blocks[1]["rowData"].pop()
                elif fault == "extra_rows":
                    blocks[1]["rowData"].append(copy.deepcopy(blocks[1]["rowData"][0]))
                elif fault == "short_cells":
                    blocks[2]["rowData"][0]["values"].pop()
                elif fault == "extra_cells":
                    blocks[2]["rowData"][0]["values"].append({})
                elif fault == "cell_not_object":
                    blocks[2]["rowData"][0]["values"][10] = "PRIVATE_BAD_CELL"
                elif fault == "row_not_object":
                    blocks[2]["rowData"][0] = []
                elif fault == "block_not_object":
                    blocks[2] = []
                elif fault == "missing_properties":
                    del sheet["properties"]
                else:
                    del sheet["properties"]["gridProperties"]
                replace_native(packet, native)
                result = self.refuse(packet)
                self.assertNotIn(b"PRIVATE_BAD_CELL", result.stderr)

    def test_all_formatted_cells_are_strings_including_private_and_unselected_cells(self):
        for block_index, row_index, column in ((1, 0, 1), (1, 1, 10), (1, 1, 11), (2, 0, 3)):
            for value in (None, True, 3, 3.0, ["PRIVATE_ARRAY_CELL"], {"PRIVATE_OBJECT_CELL": "value"}):
                with self.subTest(block=block_index, row=row_index, column=column, value=value):
                    packet = copy.deepcopy(self.packet)
                    native = get_native(packet)
                    native["sheets"][0]["data"][block_index]["rowData"][row_index]["values"][column]["formattedValue"] = value
                    replace_native(packet, native)
                    result = self.refuse(packet)
                    self.assertNotIn(b"PRIVATE_ARRAY_CELL", result.stderr)
                    self.assertNotIn(b"PRIVATE_OBJECT_CELL", result.stderr)

    def test_formatted_cell_limit_counts_utf8_bytes_in_public_private_and_unselected_rows(self):
        packet = copy.deepcopy(self.packet)
        native = get_native(packet)
        native["sheets"][0]["data"][1]["rowData"][1]["values"][1]["formattedValue"] = "λ" * 4096
        replace_native(packet, native)
        self.accept(packet)
        for block_index, row_index, column in ((1, 1, 1), (1, 0, 10), (2, 0, 11)):
            with self.subTest(block=block_index, row=row_index, column=column):
                packet = copy.deepcopy(self.packet)
                native = get_native(packet)
                native["sheets"][0]["data"][block_index]["rowData"][row_index]["values"][column]["formattedValue"] = "λ" * 4096 + "x"
                replace_native(packet, native)
                self.refuse(packet)

    def test_forged_column_policies_cannot_publish_source_notes_priority_or_topics(self):
        cases = [PUBLIC_COLUMNS + [name] for name in ("Source", "Notes", "Priority", "Topics / object tags")]
        cases.extend([list(reversed(PUBLIC_COLUMNS)), PUBLIC_COLUMNS[:-1],
                      PUBLIC_COLUMNS[:-1] + ["Source"], PUBLIC_COLUMNS[:-1] + [PUBLIC_COLUMNS[0]],
                      PUBLIC_COLUMNS[:-1] + [True], "PRIVATE_COLUMNS_STRING"])
        for columns in cases:
            with self.subTest(columns=columns):
                packet = copy.deepcopy(self.packet)
                packet["policy"]["columns"] = columns
                self.refuse(packet)

    def test_policy_ids_are_nonempty_unique_valid_and_present_in_native_rows(self):
        packet = copy.deepcopy(self.packet)
        native = get_native(packet)
        boundary_id = "T" + "x" * 191
        native["sheets"][0]["data"][1]["rowData"][1]["values"][0]["formattedValue"] = boundary_id
        packet["policy"]["artifact_ids"] = ["TEST-ZETA", boundary_id]
        replace_native(packet, native)
        self.accept(packet)
        for ids in ([], ["TEST-UNKNOWN"], ["TEST-ALPHA", "TEST-ALPHA"], [""], [True], [1],
                    ["TEST\nID"], [" TEST-ALPHA"], "TEST-ALPHA", None):
            with self.subTest(ids=ids):
                packet = copy.deepcopy(self.packet)
                packet["policy"]["artifact_ids"] = ids
                self.refuse(packet)

    def test_artifact_ids_are_globally_unique_and_valid_even_on_unselected_native_rows(self):
        for value in ("TEST-ALPHA", "TEST-ZETA", "", True, 1, " TEST-UNSELECTED", "TEST\nUNSELECTED",
                      "T" + "x" * 192, "TEST-é", "TEST/UNSELECTED"):
            with self.subTest(unselected_id=value):
                packet = copy.deepcopy(self.packet)
                native = get_native(packet)
                native["sheets"][0]["data"][2]["rowData"][0]["values"][0]["formattedValue"] = value
                replace_native(packet, native)
                self.refuse(packet)

    def test_graph_ids_and_bindings_require_explicit_known_unique_identities(self):
        packet = copy.deepcopy(self.packet)
        packet["graph_nodes"] = ["T" + "x" * 191]
        self.accept(packet)
        for nodes in (["TEST.node.a", "TEST.node.a"], [""], [True], [1], ["TEST\nnode"],
                      [" TEST.node.a"], ["T" + "x" * 192], ["TEST.é"], ["TEST/node"], "TEST.node.a", None):
            with self.subTest(nodes=nodes):
                packet = copy.deepcopy(self.packet)
                packet["graph_nodes"] = nodes
                self.refuse(packet)
        cases = [
            [{"node_id": "TEST.unknown", "artifact_id": "TEST-ALPHA", "roles": ["source"]}],
            [{"node_id": "TEST.node.a", "artifact_id": "TEST-UNSELECTED", "roles": ["source"]}],
            [{"node_id": "TEST.node.a", "artifact_id": "TEST-UNKNOWN", "roles": ["source"]}],
            [{"node_id": True, "artifact_id": "TEST-ALPHA", "roles": ["source"]}],
            [{"node_id": "TEST.node.a", "artifact_id": True, "roles": ["source"]}],
            [{"node_id": "TEST.node.a", "artifact_id": "TEST-ALPHA", "roles": ["source"]},
             {"node_id": "TEST.node.a", "artifact_id": "TEST-ALPHA", "roles": ["review"]}],
            [True], {}, None,
        ]
        for bindings in cases:
            with self.subTest(bindings=bindings):
                packet = copy.deepcopy(self.packet)
                packet["bindings"] = bindings
                self.refuse(packet)

    def test_binding_roles_are_nonempty_unique_strings_from_the_exact_six_role_set(self):
        for roles in ([], ["accepted"], ["Source"], ["source", "source"], ["kernel", True],
                      [1], "source", None):
            with self.subTest(roles=roles):
                packet = copy.deepcopy(self.packet)
                packet["bindings"] = [{"node_id": "TEST.node.a", "artifact_id": "TEST-ALPHA", "roles": roles}]
                self.refuse(packet)

    def test_collection_limits_accept_4096_and_refuse_4097_without_duplicate_shortcuts(self):
        packet = copy.deepcopy(self.packet)
        packet["graph_nodes"] = ["TEST.node.%04d" % index for index in range(4096)]
        packet["bindings"] = [{"node_id": node, "artifact_id": "TEST-ALPHA", "roles": ["source"]}
                              for node in packet["graph_nodes"]]
        self.accept(packet)
        too_many = copy.deepcopy(packet)
        too_many["graph_nodes"].append("TEST.node.4096")
        self.refuse(too_many)
        too_many = copy.deepcopy(packet)
        too_many["bindings"].append({"node_id": "TEST.node.0000", "artifact_id": "TEST-ZETA", "roles": ["review"]})
        self.refuse(too_many)
        for count in (4096, 4097):
            with self.subTest(native_rows_and_policy_ids=count):
                many = copy.deepcopy(self.packet)
                ids = ["TEST-ROW-%04d" % index for index in range(count)]
                native = synthetic_native()
                native["sheets"][0]["properties"]["gridProperties"]["rowCount"] = count + 1
                native["sheets"][0]["data"] = [
                    {"rowData": [row(HEADER)]},
                    {"startRow": 1, "rowData": [row([artifact_id] + ["TEST value"] * 9 +
                                                    ["PRIVATE_MANY_SOURCE", "PRIVATE_MANY_NOTES"])
                                                for artifact_id in ids]},
                ]
                many["capture"]["source"]["ranges"] = [RANGES[0], "'TEST Artifact Index'!A2:L%d" % (count + 1)]
                many["policy"]["artifact_ids"] = ids
                replace_native(many, native)
                result = self.accept(many) if count == 4096 else self.refuse(many)
                self.assertNotIn(b"PRIVATE_MANY_", result.stdout + result.stderr)

    def test_stdin_limit_is_16_mib_of_bytes_including_legal_json_whitespace(self):
        raw = canonical(self.packet)
        bounded = raw + b" " * (16 * 1024 * 1024 - len(raw))
        self.accept(raw=bounded)
        self.refuse(raw=bounded + b" ")

    def test_decoded_capture_limit_is_4_mib_of_original_bytes_not_base64_length(self):
        packet = copy.deepcopy(self.packet)
        raw = base64.b64decode(packet["capture"]["raw_base64"])
        bounded = raw + b" " * (4 * 1024 * 1024 - len(raw))
        replace_raw(packet, bounded)
        self.accept(packet)
        replace_raw(packet, bounded + b" ")
        self.refuse(packet)

    def test_duplicate_keys_refuse_in_packet_wrappers_and_native_private_annotations(self):
        raw = canonical(self.packet)
        mutations = [
            raw[:-1] + b',"schema_version":1}',
            raw.replace(b'"schema_version":1', b'"schema_version":1,"schema_version":1', 1),
            raw.replace(b'"capture_id":"TEST-capture-1"', b'"capture_id":"TEST-capture-1","capture_id":"TEST-capture-1"', 1),
            raw.replace(b'"artifact_ids":[', b'"artifact_ids":[],"artifact_ids":[', 1),
        ]
        for index, changed in enumerate(mutations):
            with self.subTest(packet_duplicate=index):
                self.refuse(raw=changed, expected=self.pins)
        for marker, replacement in (
            (b'"spreadsheetId":"TEST-R17-SPREADSHEET"', b'"spreadsheetId":"TEST-R17-SPREADSHEET","spreadsheetId":"TEST-R17-SPREADSHEET"'),
            (b'"formattedValue":"TEST-ZETA"', b'"formattedValue":"TEST-ZETA","formattedValue":"TEST-ZETA"'),
            (b'"privateAnnotation":"PRIVATE_NATIVE_ANNOTATION"', b'"privateAnnotation":{"x":1,"x":1}'),
        ):
            with self.subTest(native_duplicate=marker):
                packet = copy.deepcopy(self.packet)
                native_raw = base64.b64decode(packet["capture"]["raw_base64"])
                self.assertIn(marker, native_raw)
                replace_raw(packet, native_raw.replace(marker, replacement, 1))
                self.refuse(packet)

    def test_nonfinite_numbers_and_numeric_overflow_refuse_even_in_unknown_native_annotations(self):
        for token in (b"NaN", b"Infinity", b"-Infinity", b"1e9999"):
            with self.subTest(token=token, location="native_annotation"):
                packet = copy.deepcopy(self.packet)
                raw = base64.b64decode(packet["capture"]["raw_base64"])
                raw = raw.replace(b'"privateAnnotation":"PRIVATE_NATIVE_ANNOTATION"', b'"privateAnnotation":' + token, 1)
                replace_raw(packet, raw)
                self.refuse(packet)
            with self.subTest(token=token, location="packet_schema"):
                raw = canonical(self.packet).replace(b'"schema_version":1', b'"schema_version":' + token, 1)
                self.refuse(raw=raw, expected=self.pins)

    def test_json_nesting_is_bounded_in_unknown_native_annotations_without_schema_masking(self):
        packet = copy.deepcopy(self.packet)
        raw = base64.b64decode(packet["capture"]["raw_base64"])
        marker = b'"privateAnnotation":"PRIVATE_NATIVE_ANNOTATION"'
        self.assertIn(marker, raw)
        # The native root object plus 63 nested arrays occupies 64 levels.
        bounded = b"[" * 63 + b"0" + b"]" * 63
        replace_raw(packet, raw.replace(marker, b'"privateAnnotation":' + bounded, 1))
        self.accept(packet)
        # One root object plus 64 arrays is exactly one level over the limit.
        deep = b"[" * 64 + b"0" + b"]" * 64
        replace_raw(packet, raw.replace(marker, b'"privateAnnotation":' + deep, 1))
        self.refuse(packet)

    def test_malformed_utf8_json_and_multiple_documents_refuse_at_both_input_layers(self):
        for raw in (b"", b"{", b"[]", b"null", canonical(self.packet) + b"{}", b"\xff",
                    canonical(self.packet).replace(b"TEST-capture-1", b"TEST-\xff-capture", 1)):
            with self.subTest(packet_raw=raw[:20]):
                self.refuse(raw=raw, expected=self.pins)
        for raw in (b"", b"{", b"null", b"\xff", native_bytes(synthetic_native()) + b"{}",
                    native_bytes(synthetic_native()).replace(b"PRIVATE_NATIVE_ANNOTATION", b"PRIVATE_\xff_NATIVE", 1)):
            with self.subTest(native_raw=raw[:20]):
                packet = copy.deepcopy(self.packet)
                replace_raw(packet, raw)
                self.refuse(packet)

    def test_base64_is_strict_canonical_and_not_merely_decodable_to_the_expected_bytes(self):
        encoded = self.packet["capture"]["raw_base64"]
        for value in ("", "!!PRIVATE_BAD_BASE64!!", encoded + "=", encoded + "\n", " " + encoded,
                      encoded[:-1], True, None):
            with self.subTest(base64_type=type(value).__name__, value_prefix=str(value)[:20]):
                packet = copy.deepcopy(self.packet)
                packet["capture"]["raw_base64"] = value
                result = self.refuse(packet, expected=self.pins)
                self.assertNotIn(b"PRIVATE_BAD_BASE64", result.stderr)
        packet = copy.deepcopy(self.packet)
        raw = base64.b64decode(encoded)
        raw += b" " * ((1 - len(raw)) % 3)
        replace_raw(packet, raw)
        canonical_encoded = packet["capture"]["raw_base64"]
        self.assertTrue(canonical_encoded.endswith("=="))
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
        index = alphabet.index(canonical_encoded[-3])
        alternate = canonical_encoded[:-3] + alphabet[index + 1] + "=="
        self.assertEqual(base64.b64decode(alternate, validate=True), raw)
        expected = pins_for(packet)
        self.accept(packet)
        packet["capture"]["raw_base64"] = alternate
        self.refuse(packet, expected=expected)

    def test_urls_paths_and_code_are_inert_data_without_fetch_execution_or_filesystem_writes(self):
        requests = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                requests.append(self.path)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"PRIVATE_FETCH_RESULT")

            def log_message(self, *args):
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            marker = self.workdir / "PRIVATE_EXECUTION_MARKER"
            secret = self.workdir / "PRIVATE_LOCAL_SOURCE"
            secret.write_text("PRIVATE_LOCAL_FILE_CONTENT", encoding="utf-8")
            packet = copy.deepcopy(self.packet)
            native = get_native(packet)
            code = "__import__('pathlib').Path(%r).write_text('executed')" % str(marker)
            url = "http://127.0.0.1:%d/PRIVATE_URL_PATH" % server.server_port
            probe = "http://127.0.0.1:%d/TEST_OBSERVER_PROBE" % server.server_port
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(probe, timeout=5) as response:
                self.assertEqual(response.read(), b"PRIVATE_FETCH_RESULT")
            self.assertEqual(requests, ["/TEST_OBSERVER_PROBE"])
            requests.clear()
            public = "inert URL %s; inert code %s; path %s" % (url, code, secret)
            values = native["sheets"][0]["data"][1]["rowData"][1]["values"]
            values[1]["formattedValue"] = public
            values[10]["formattedValue"] = url
            values[11]["formattedValue"] = code
            native["privateAnnotation"] = {"file": str(secret), "code": code, "url": url}
            replace_native(packet, native)
            before = sorted(path.name for path in self.workdir.iterdir())
            result = self.accept(packet, audit={"secret": secret})
            report = result.audit_report
            labels = {"read", "write", "mkdir", "remove", "rename", "rmdir"}
            self.assertEqual(set(report), {"calibration", "attempts", "optimize"})
            self.assertIs(type(report["optimize"]), int)
            self.assertEqual(report["optimize"], sys.flags.optimize)
            self.assertEqual(set(report["calibration"]), labels)
            for count in report["calibration"].values():
                self.assertIs(type(count), int)
                self.assertGreater(count, 0)
            self.assertEqual(report["attempts"], dict.fromkeys(labels, 0))
            for count in report["attempts"].values():
                self.assertIs(type(count), int)
            projected = json.loads(result.stdout)
            self.assertEqual(projected["rows"][0]["fields"]["Title"], public)
            self.assertEqual(requests, [])
            self.assertFalse(marker.exists())
            self.assertEqual(secret.read_text(encoding="utf-8"), "PRIVATE_LOCAL_FILE_CONTENT")
            self.assertEqual(sorted(path.name for path in self.workdir.iterdir()), before)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
