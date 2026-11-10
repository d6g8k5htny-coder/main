"""Test-first contract for a private data-only trace analyzer.

The frozen JSON contains genuine selected historical R1 whole lines. Inputs made
below by concatenating a subset are CONTRACT EXCERPTS with local positions; they
are not a replay or a complete old trace. All other trace bytes in this module
are explicitly SYNTHETIC parser-contract fixtures and earn no live credit.
The production analyzer is deliberately absent at this source stage.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

# The targeted runner captures this exact checked repository before/after.
# Load only its fixed private source path, so a PYTHONPATH tools package or a
# cached project module cannot substitute another analyzer. This executes no
# artifact/trace/fixture payload. Honest absence fails module import at R0.
_SUBJECT_SOURCE = Path(__file__).resolve().parent.parent / "tools" / "formal_trace_reanalysis.py"
_SPEC = importlib.util.spec_from_file_location("formal_trace_reanalysis_contract_subject", _SUBJECT_SOURCE)
subject = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = subject
_SPEC.loader.exec_module(subject)


FIXTURE = Path(__file__).parent / "fixtures" / "formal_trace_r1_lines_20261110.json"
FIXTURE_BLOB = "e087f661d8494f7fb780e2f32ce04bee1aeb8f3c"
MIB = 1024 * 1024
TOP = {
    "scope", "status", "input", "indexed_prefix_bytes", "all_lines_considered",
    "partial_reason", "events", "concerns", "counts", "manual_review_required",
}
EVENT = {
    "pid", "syscall", "state", "write_intent", "targets", "anchors",
    "returned_fd", "symlink_content", "flags", "result", "source_lines",
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git_blob(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()


def provenance(raw):
    answer, offset = [], 0
    for number, line in enumerate(raw.splitlines(keepends=True), 1):
        answer.append({
            "line": number, "offset": offset, "bytes": len(line),
            "sha256": sha(line), "raw_utf8": line.decode("utf-8"),
        })
        offset += len(line)
    return answer


def synthetic(payload, pid=1, stamp="1.000000"):
    """SYNTHETIC contract fixture, never original native evidence."""
    return (str(pid) + "  " + stamp + " " + payload + "\n").encode("utf-8")


class FormalTraceReanalysisContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture_raw = FIXTURE.read_bytes()
        cls.fixture = json.loads(cls.fixture_raw)
        cls.rows = {
            (row["mode"], row["original_line_number"]): row
            for row in cls.fixture["lines"]
        }

    def genuine_line(self, mode, number):
        return self.rows[(mode, number)]["raw_utf8"].encode("utf-8")

    def analyze(self, raw):
        result = subject._analyze_trace(raw)
        self.assertIs(type(result), dict)
        self.assertEqual(set(result), TOP)
        self.assertEqual(result["scope"], "TRACE_SYNTAX_DIAGNOSTIC_ONLY")
        self.assertIs(result["manual_review_required"], True)
        self.assertEqual(result["input"], {
            "bytes": len(raw), "sha256": sha(raw),
            "line_count": len(raw.splitlines(keepends=True)),
        })
        self.assertIn(result["status"], (
            "COMPLETE_BOUNDED_DIAGNOSTIC", "HOLD", "PARTIAL_HOLD"))
        self.assertIs(type(result["indexed_prefix_bytes"]), int)
        self.assertTrue(0 <= result["indexed_prefix_bytes"] <= len(raw))
        self.assertIs(type(result["all_lines_considered"]), bool)
        self.assertIs(type(result["events"]), list)
        self.assertIs(type(result["concerns"]), list)
        self.assertEqual(set(result["counts"]), {
            "complete_calls", "read_only_calls", "write_intent_calls", "context_calls",
        })
        for count in result["counts"].values():
            self.assertIs(type(count), int)
            self.assertGreaterEqual(count, 0)
        known = provenance(raw)
        for event in result["events"]:
            self.assertEqual(set(event), EVENT)
            self.assertIn(event["state"], ("COMPLETE", "UNRESOLVED"))
            self.assertTrue(event["write_intent"] is None
                            or type(event["write_intent"]) is bool)
            self.assertTrue(event["pid"] is None or type(event["pid"]) is int)
            self.assertEqual(set(event["result"]), {"value", "errno"})
            self.assertTrue(event["result"]["value"] is None
                            or type(event["result"]["value"]) is int)
            self.assertTrue(event["result"]["errno"] is None
                            or type(event["result"]["errno"]) is str)
            self.assertTrue(event["source_lines"])
            for item in event["source_lines"]:
                self.assertIs(type(item["line"]), int)
                self.assertTrue(1 <= item["line"] <= len(known))
                self.assertEqual(item, known[item["line"] - 1])
                prefix = item["raw_utf8"].split()[0]
                if prefix.isdecimal():
                    self.assertGreater(int(prefix), 0)
                    self.assertEqual(event["pid"], int(prefix))
        for concern in result["concerns"]:
            self.assertEqual(set(concern), {"event_index", "reason"})
            self.assertIs(type(concern["event_index"]), int)
            self.assertTrue(0 <= concern["event_index"] < len(result["events"]))
            self.assertIs(type(concern["reason"]), str)
        encoded = (json.dumps(result, sort_keys=True, ensure_ascii=True,
                              allow_nan=False) + "\n").encode("ascii")
        self.assertLessEqual(len(encoded), 4 * MIB)
        self.assertLessEqual(len(result["events"]), 4096)
        if result["status"] == "COMPLETE_BOUNDED_DIAGNOSTIC":
            self.assertFalse(result["concerns"])
            self.assertTrue(result["all_lines_considered"])
            self.assertEqual(result["indexed_prefix_bytes"], len(raw))
            self.assertIsNone(result["partial_reason"])
        elif result["status"] == "HOLD":
            self.assertTrue(result["concerns"])
        else:
            self.assertFalse(result["all_lines_considered"])
            self.assertIn(result["partial_reason"], (
                "LINE_LIMIT", "RECORD_LIMIT", "SERIALIZED_LIMIT"))
        return result

    def reasons(self, report, event=None):
        return {
            row["reason"] for row in report["concerns"]
            if event is None or report["events"][row["event_index"]] is event
        }

    def one_event(self, raw):
        report = self.analyze(raw)
        self.assertEqual(len(report["events"]), 1)
        return report, report["events"][0]

    def test_01_frozen_genuine_whole_line_identity(self):
        self.assertEqual(git_blob(self.fixture_raw), FIXTURE_BLOB)
        self.assertEqual(self.fixture["kind"], "HISTORICAL_GENUINE_R1_WHOLE_TRACE_LINE_FIXTURES")
        self.assertEqual(self.fixture["original"]["status"], "HOLD_UNCHANGED")
        self.assertEqual(self.fixture["acquisition"]["run"], 42811785759)
        self.assertEqual(self.fixture["acquisition"]["native_log_sha256"],
                         "fa15d5f3a6569d7fa0b47aa0dbfc9fcb5a7968ae49ac0e14f2615dbbab1ae498")
        rows = self.fixture["lines"]
        self.assertEqual(len(rows), 34)
        self.assertEqual(len(self.rows), 34)
        self.assertEqual(sum(row["original_bytes"] for row in rows), 6199)
        self.assertEqual([sum(row["mode"] == mode for row in rows)
                          for mode in ("normal", "optimized")], [19, 15])
        for row in rows:
            with self.subTest(id=row["id"]):
                raw = row["raw_utf8"].encode("utf-8")
                self.assertTrue(row["full_original_line"])
                self.assertTrue(raw.endswith(b"\n"))
                self.assertEqual(len(raw), row["original_bytes"])
                self.assertEqual(sha(raw), row["original_sha256"])
                self.assertIn("@" + str(row["original_byte_offset"]) + ".bin", row["id"])

    def test_02_all_six_genuine_open_roles_and_read_neighbors(self):
        for mode, anchors in (("normal", (323, 350, 383)),
                              ("optimized", (425, 452, 485))):
            for number in anchors:
                with self.subTest(mode=mode, line=number):
                    raw = self.genuine_line(mode, number)
                    report, event = self.one_event(raw)
                    self.assertEqual(report["status"], "COMPLETE_BOUNDED_DIAGNOSTIC")
                    self.assertEqual(report["counts"], {
                        "complete_calls": 1, "read_only_calls": 0,
                        "write_intent_calls": 1, "context_calls": 0,
                    })
                    self.assertEqual(event["syscall"], "openat")
                    self.assertIs(event["write_intent"], True)
                    self.assertEqual(event["flags"], ["O_RDWR"])
                    self.assertEqual(event["targets"], [{
                        "role": "path", "argument": 1, "raw": '"/dev/null"',
                        "path": "/dev/null", "anchor": None,
                    }])
                    self.assertEqual(event["anchors"][0]["role"], "cwd")
                    self.assertEqual(event["anchors"][0]["argument"], 0)
                    self.assertTrue(event["anchors"][0]["path"].endswith("/subjects/" + mode))
                    self.assertEqual(event["returned_fd"], {
                        "fd": 3, "path": "/dev/null",
                        "device": {"kind": "char", "major": 1, "minor": 3},
                        "raw": "3</dev/null<char 1:3>>",
                    })
                    self.assertEqual(event["result"], {"value": 3, "errno": None})
                    self.assertEqual(event["source_lines"], provenance(raw))
                    self.assertEqual(event["source_lines"][0]["line"], 1)
                    self.assertEqual(event["source_lines"][0]["offset"], 0)
                    for neighbor in (number - 1, number + 1):
                        read_report = self.analyze(self.genuine_line(mode, neighbor))
                        self.assertEqual(read_report["events"], [])
                        self.assertEqual(read_report["counts"], {
                            "complete_calls": 1, "read_only_calls": 1,
                            "write_intent_calls": 0, "context_calls": 0,
                        })

    def test_03_genuine_interleaving_pair_provenance_and_excerpt_gaps(self):
        for mode, expected_pairs, required in (
            ("normal", ((2019, 2023), (2020, 2022), (2021, 2028), (2024, 2029)),
             {"ORPHAN_RESUME", "NATIVE_FAILURE"}),
            ("optimized", ((2109, 2111), (2110, 2112)),
             {"UNFINISHED_ENTRY", "NATIVE_FAILURE"}),
        ):
            rows = sorted((row for row in self.fixture["lines"] if row["mode"] == mode),
                          key=lambda row: row["original_line_number"])
            raw = b"".join(row["raw_utf8"].encode("utf-8") for row in rows)
            report = self.analyze(raw)
            self.assertEqual(report["status"], "HOLD")
            self.assertTrue(report["all_lines_considered"])
            self.assertTrue(required <= self.reasons(report))
            self.assertEqual(report["counts"], {
                "complete_calls": 14 if mode == "normal" else 12,
                "read_only_calls": 6,
                "write_intent_calls": 9 if mode == "normal" else 7,
                "context_calls": 0,
            })
            by_sources = {
                tuple(line["sha256"] for line in event["source_lines"]): event
                for event in report["events"]
            }
            self.assertEqual(len(by_sources), len(report["events"]))
            standalone = 2018 if mode == "normal" else 2108
            single_key = (sha(self.genuine_line(mode, standalone)),)
            self.assertIn(single_key, by_sources)
            failed = by_sources[single_key]
            self.assertEqual(failed["state"], "COMPLETE")
            self.assertEqual(failed["result"], {"value": -1, "errno": "ENOENT"})
            self.assertIn("NATIVE_FAILURE", self.reasons(report, failed))
            gap_number, gap_reason = ((2027, "ORPHAN_RESUME") if mode == "normal"
                                      else (2113, "UNFINISHED_ENTRY"))
            gap_key = (sha(self.genuine_line(mode, gap_number)),)
            self.assertIn(gap_key, by_sources)
            gap = by_sources[gap_key]
            self.assertEqual(gap["state"], "UNRESOLVED")
            self.assertIn(gap_reason, self.reasons(report, gap))
            if mode == "normal":
                self.assertEqual(gap["result"], {"value": -1, "errno": "ENOENT"})
                self.assertIn("NATIVE_FAILURE", self.reasons(report, gap))
            else:
                self.assertEqual(gap["result"], {"value": None, "errno": None})
            for first, second in expected_pairs:
                key = tuple(sha(self.genuine_line(mode, n)) for n in (first, second))
                self.assertIn(key, by_sources)
                event = by_sources[key]
                self.assertEqual(event["state"], "COMPLETE")
                if event["syscall"] == "unlink":
                    self.assertEqual(event["result"], {"value": -1, "errno": "ENOENT"})
                    self.assertIn("NATIVE_FAILURE", self.reasons(report, event))
                else:
                    self.assertEqual(event["syscall"], "mkdir")
                    self.assertEqual(event["result"], {"value": 0, "errno": None})
            self.assertTrue(all(item["line"] <= len(rows)
                                for event in report["events"] for item in event["source_lines"]))

    def test_04_synthetic_flags_are_read_only_from_the_flags_operand(self):
        # SYNTHETIC: token-looking filename/context text is not a flags operand.
        readonly = synthetic('openat(AT_FDCWD</cwd/O_CREAT>, "/tmp/O_WRONLY,name", '
                             'O_RDONLY|O_CLOEXEC) = 3</tmp/O_WRONLY,name> <0.000001>')
        self.assertEqual(self.analyze(readonly)["events"], [])
        for payload, target in (
            ('open("/tmp/a", O_WRONLY|O_CREAT, 0600) = 3</tmp/a> <0.000001>', "/tmp/a"),
            ('open("/tmp/a", O_RDONLY|O_CREAT, 0600) = 3</tmp/a> <0.000001>', "/tmp/a"),
            ('open("/tmp/a", O_RDONLY|O_TRUNC) = 3</tmp/a> <0.000001>', "/tmp/a"),
            ('open("/tmp/a", O_RDONLY|O_APPEND) = 3</tmp/a> <0.000001>', "/tmp/a"),
            ('openat(4</tmp>, "a", O_RDWR|O_APPEND) = 3</tmp/a> <0.000001>', "a"),
            ('openat2(AT_FDCWD</tmp>, "a", {flags=O_WRONLY|O_CREAT, mode=0600, '
             'resolve=RESOLVE_BENEATH}, 24) = 3</tmp/a> <0.000001>', "a"),
        ):
            with self.subTest(payload=payload):
                _, event = self.one_event(synthetic(payload))
                self.assertIs(event["write_intent"], True)
                self.assertEqual(event["targets"][0]["path"], target)
                self.assertTrue(set(event["flags"]) & {
                    "O_WRONLY", "O_RDWR", "O_CREAT", "O_TRUNC", "O_APPEND"})
        unknown, event = self.one_event(synthetic(
            'open("/tmp/a", O_FUTURE_UNKNOWN) = -1 EINVAL (Invalid argument) <0.000001>'))
        self.assertIn("UNSUPPORTED_FLAGS", self.reasons(unknown, event))

    def test_05_synthetic_dual_path_and_symlink_content_roles(self):
        # SYNTHETIC: source/destination both retained; symlink content is separate.
        for payload in (
            'rename("/a", "/b") = 0 <0.000001>',
            'link("/a", "/b") = 0 <0.000001>',
            'renameat(4</base>, "a", 5</other>, "b") = 0 <0.000001>',
            'linkat(4</base>, "a", 5</other>, "b", 0) = 0 <0.000001>',
        ):
            report, event = self.one_event(synthetic(payload))
            self.assertEqual([target["role"] for target in event["targets"]],
                             ["source", "destination"])
            self.assertEqual(len(event["targets"]), 2)
            self.assertEqual(report["status"], "COMPLETE_BOUNDED_DIAGNOSTIC")
        _, link = self.one_event(synthetic(
            'symlink("/outside/target", "/tmp/link") = 0 <0.000001>'))
        self.assertEqual(link["symlink_content"], "/outside/target")
        self.assertEqual([target["path"] for target in link["targets"]], ["/tmp/link"])
        self.assertEqual(link["targets"][0]["role"], "destination")
        _, linkat = self.one_event(synthetic(
            'symlinkat("../target", 4</tmp>, "link") = 0 <0.000001>'))
        self.assertEqual(linkat["symlink_content"], "../target")
        self.assertEqual(linkat["targets"][0]["anchor"], "/tmp")

    def test_06_synthetic_relative_fd_and_cwd_context_is_explicit(self):
        # SYNTHETIC: this is syntax/context extraction, never path admission.
        _, relative = self.one_event(synthetic(
            'openat(4</base>, "child", O_WRONLY) = 3</base/child> <0.000001>'))
        self.assertEqual(relative["targets"][0]["path"], "child")
        self.assertEqual(relative["targets"][0]["anchor"], "/base")
        _, fd = self.one_event(synthetic('ftruncate(4</base/file>, 0) = 0 <0.000001>'))
        self.assertEqual(fd["targets"][0]["role"], "fd")
        self.assertEqual(fd["targets"][0]["path"], "/base/file")
        for payload, reason in (
            ('openat(4, "child", O_WRONLY) = -1 EBADF (Bad file descriptor) <0.000001>',
             "MISSING_CONTEXT"),
            ('ftruncate(4, 0) = -1 EBADF (Bad file descriptor) <0.000001>', "MISSING_CONTEXT"),
            ('ftruncate(4</base/file (deleted)>, 0) = 0 <0.000001>', "AMBIGUOUS_FD_CONTEXT"),
        ):
            report, event = self.one_event(synthetic(payload))
            self.assertIn(reason, self.reasons(report, event))
        changed = (
            synthetic('chdir("/first") = 0 <0.000001>')
            + synthetic('chdir("/missing") = -1 ENOENT (No such file or directory) <0.000001>')
            + synthetic('open("child", O_WRONLY) = 3 <0.000001>')
        )
        report = self.analyze(changed)
        opened = [event for event in report["events"] if event["syscall"] == "open"]
        self.assertEqual(opened[0]["targets"][0]["anchor"], "/first")
        self.assertNotIn("MISSING_CONTEXT", self.reasons(report, opened[0]))
        self.assertIn("NATIVE_FAILURE", self.reasons(report))
        unknown, event = self.one_event(synthetic(
            'open("child", O_WRONLY) = -1 EACCES (Permission denied) <0.000001>', pid=2))
        self.assertIn("MISSING_CONTEXT", self.reasons(unknown, event))

    def test_07_synthetic_pairs_never_borrow_or_reuse_a_half(self):
        # SYNTHETIC split calls; both input lines must survive exact provenance.
        start = synthetic('unlink("/tmp/a" <unfinished ...>', pid=10)
        end = synthetic('<... unlink resumed>) = -1 ENOENT (No such file or directory) '
                        '<0.000001>', pid=10, stamp="1.000002")
        valid, event = self.one_event(start + end)
        self.assertEqual(event["source_lines"], provenance(start + end))
        self.assertEqual(event["result"], {"value": -1, "errno": "ENOENT"})
        self.assertIn("NATIVE_FAILURE", self.reasons(valid, event))
        for raw, expected in (
            (start, {"UNFINISHED_ENTRY"}),
            (end, {"ORPHAN_RESUME"}),
            (start + synthetic('<... unlink resumed>) = 0 <0.000001>', pid=11),
             {"UNFINISHED_ENTRY", "ORPHAN_RESUME"}),
            (start + synthetic('<... mkdir resumed>) = 0 <0.000001>', pid=10),
             {"AMBIGUOUS_PAIR"}),
            (start + start + end, {"AMBIGUOUS_PAIR"}),
            (start + end + end, {"ORPHAN_RESUME"}),
        ):
            report = self.analyze(raw)
            self.assertEqual(report["status"], "HOLD")
            self.assertTrue(expected <= self.reasons(report))

    def test_08_synthetic_escaping_brackets_and_malformed_syntax(self):
        # SYNTHETIC supported C escapes are explicit, not Python literal eval.
        _, escaped = self.one_event(synthetic(
            r'unlink("/tmp/a,b\"c\\d\040e<literal>") = 0 <0.000001>'))
        self.assertEqual(escaped["targets"][0]["path"], '/tmp/a,b"c\\d e<literal>')
        _, literal = self.one_event(synthetic(
            'unlink("/tmp/name<char 1:3>") = 0 <0.000001>'))
        self.assertEqual(literal["targets"][0]["path"], "/tmp/name<char 1:3>")
        for payload in (
            r'unlink("/tmp/a\q") = 0 <0.000001>',
            'unlink("/tmp/truncated"...',
            'openat2(AT_FDCWD</tmp>, "a", {flags=O_RDWR, flags=O_RDONLY}, 24) = 3</tmp/a>',
            'not_a_supported_syscall("/tmp/a") = 0 <0.000001>',
        ):
            report = self.analyze(synthetic(payload))
            self.assertEqual(report["status"], "HOLD")
            self.assertTrue(self.reasons(report) & {"UNSUPPORTED_SYNTAX", "UNSUPPORTED_ESCAPE"})
        for payload in (
            'openat(AT_FDCWD</cwd>, "/dev/nullish", O_RDWR) = 3</dev/nullish> <0.000001>',
            'openat(AT_FDCWD</cwd>, "/outside/file", O_RDWR) = 3</outside/file> <0.000001>',
        ):
            _, event = self.one_event(synthetic(payload))
            self.assertIs(event["write_intent"], True)
            self.assertEqual(len(event["targets"]), 1)
            self.assertNotEqual(event["targets"][0]["path"], "/dev/null")

    def test_09_synthetic_native_failed_mutations_never_disappear(self):
        # SYNTHETIC failures are concerns even when their path grammar is clear.
        for operation in ('unlink("/tmp/a")', 'mkdir("/tmp/a", 0777)',
                          'rename("/tmp/a", "/tmp/b")', 'fchmod(4</tmp/a>, 0600)',
                          'open("/tmp/a", O_WRONLY)'):
            report, event = self.one_event(synthetic(
                operation + ' = -1 ENOENT (No such file or directory) <0.000001>'))
            self.assertEqual(event["result"], {"value": -1, "errno": "ENOENT"})
            self.assertIn("NATIVE_FAILURE", self.reasons(report, event))
            self.assertEqual(report["status"], "HOLD")
            self.assertTrue(report["all_lines_considered"])

    def test_10_synthetic_local_provenance_and_input_refusals(self):
        raw = synthetic('mkdir("/tmp/a", 0777) = 0 <0.000001>', pid=8)
        report, event = self.one_event(raw)
        self.assertEqual(event["source_lines"], provenance(raw))
        self.assertEqual(report["input"]["sha256"], sha(raw))
        for bad in (None, "", bytearray(raw), {"raw": raw}):
            with self.subTest(kind=type(bad).__name__):
                with self.assertRaisesRegex(ValueError, "^INVALID_INPUT$"):
                    subject._analyze_trace(bad)
        with self.assertRaisesRegex(ValueError, "^INVALID_UTF8$"):
            subject._analyze_trace(b"\xff\n")
        unfinished, event = self.one_event(raw[:-1])
        self.assertIn("INCOMPLETE_LINE", self.reasons(unfinished, event))

    def test_11_synthetic_source_and_record_ceilings_are_honest(self):
        with self.assertRaisesRegex(ValueError, "^TRACE_LIMIT$"):
            subject._analyze_trace(b"x" * (128 * MIB + 1))
        line = synthetic('mkdir("/x", 0) = 0 <0.000001>')
        for raw, reason in ((b"x" * 16384 + b"\n", "LINE_LIMIT"),
                            (line * 4097, "RECORD_LIMIT")):
            report = self.analyze(raw)
            self.assertEqual(report["status"], "PARTIAL_HOLD")
            self.assertEqual(report["partial_reason"], reason)
            self.assertLess(report["indexed_prefix_bytes"], len(raw))
            self.assertFalse(report["all_lines_considered"])

    def test_12_synthetic_serialized_and_pending_growth_ceiling(self):
        # SYNTHETIC: valid long operands hit serialized cap before record cap.
        name = "/" + "x" * 3000
        line = synthetic('rename("' + name + 'a", "' + name + 'b") = 0 <0.000001>')
        report = self.analyze(line * 1000)
        self.assertEqual(report["status"], "PARTIAL_HOLD")
        self.assertEqual(report["partial_reason"], "SERIALIZED_LIMIT")
        self.assertLess(len(report["events"]), 4096)
        self.assertFalse(report["all_lines_considered"])
        # Live pending entries consume the same unchanged cap/record budget.
        pending = b"".join(
            synthetic('mkdir("/x", 0 <unfinished ...>', pid=number + 1)
            for number in range(4097))
        held = self.analyze(pending)
        self.assertEqual(held["status"], "PARTIAL_HOLD")
        self.assertIn(held["partial_reason"], ("RECORD_LIMIT", "SERIALIZED_LIMIT"))
        self.assertFalse(held["all_lines_considered"])
        self.assertIn("UNFINISHED_ENTRY", self.reasons(held))


if __name__ == "__main__":
    unittest.main()
