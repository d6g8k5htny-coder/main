"""Prospective binary capture controls; run only on the hosted source-bound lane."""
from contextlib import redirect_stdout
from importlib import import_module
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SiteGraphCaptureContract(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT / "tools/site_graph_capture.py").is_file(),
                        "bounded capture implementation is missing")
        self.api = import_module("tools.site_graph_capture")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "captured"

    def run_child(self, source, **options):
        return self.api.run_capture([sys.executable, "-B", "-S", "-c", source],
                                    self.output, context={"scope": "SYNTHETIC_ONLY"}, **options)

    def test_binary_success_retains_exact_original_streams_and_closed_inventory(self):
        result = self.run_child("import os;os.write(1,b'a\\x00b');os.write(2,b'c\\xffd')")
        self.assertEqual(result["controller_exit"], 0)
        self.assertEqual(result["child_returncode"], 0)
        self.assertTrue(result["original_streams_complete"])
        self.assertEqual((self.output / "stdout.bin").read_bytes(), b"a\x00b")
        self.assertEqual((self.output / "stderr.bin").read_bytes(), b"c\xffd")
        self.assertEqual({x.name for x in self.output.iterdir()},
                         {"stdout.bin", "stderr.bin", "context.json", "result.json"})
        self.assertLessEqual(sum(x.stat().st_size for x in self.output.iterdir()), 1024 * 1024)
        self.assertEqual(json.loads((self.output / "result.json").read_bytes()), result)

    def test_nonzero_child_status_is_propagated_and_recorded(self):
        result = self.run_child("raise SystemExit(7)")
        self.assertEqual(result["child_returncode"], 7)
        self.assertEqual(result["controller_exit"], 7)
        self.assertEqual(result["controller_outcome"], "CHILD_FAILED")

    def test_exact_stream_and_aggregate_limits_remain_successful(self):
        result = self.run_child("import os;os.write(1,b'x'*32);os.write(2,b'y'*16)",
                                file_limit=32, aggregate_limit=48)
        self.assertEqual(result["controller_exit"], 0)
        self.assertEqual(result["retained_total_bytes"], 48)
        self.assertFalse(result["overflow"])
        self.assertTrue(result["original_streams_complete"])

    def test_per_stream_overflow_preserves_prefix_and_refuses_success(self):
        result = self.run_child("import os;os.write(1,b'x'*65)",
                                file_limit=32, aggregate_limit=64)
        self.assertEqual(result["controller_exit"], 74)
        self.assertTrue(result["overflow"])
        self.assertFalse(result["original_streams_complete"])
        self.assertEqual((self.output / "stdout.bin").read_bytes(), b"x" * 32)
        self.assertEqual(result["streams"]["stdout"]["retained_bytes"], 32)
        self.assertGreater(result["streams"]["stdout"]["observed_bytes"], 32)

    def test_aggregate_overflow_is_distinct_from_individual_file_limits(self):
        result = self.run_child("import os;os.write(1,b'x'*40);os.write(2,b'y'*40)",
                                file_limit=64, aggregate_limit=48)
        self.assertEqual(result["controller_exit"], 74)
        self.assertTrue(result["overflow"])
        self.assertEqual(result["retained_total_bytes"], 48)
        for row in result["streams"].values():
            self.assertLessEqual(row["retained_bytes"], 64)

    def test_timeout_records_actual_child_and_controller_outcomes(self):
        result = self.run_child("import time;time.sleep(5)", wall_seconds=0.05,
                                terminate_seconds=0.02, drain_seconds=0.05)
        self.assertEqual(result["controller_exit"], 124)
        self.assertTrue(result["timed_out"])
        self.assertFalse(result["original_streams_complete"])
        self.assertIsInstance(result["child_returncode"], int)

    def test_exited_parent_with_held_descendant_pipe_is_incomplete(self):
        source = ("import os,time\n"
                  "if os.fork()==0:\n time.sleep(5)\n os._exit(0)\n"
                  "os._exit(0)\n")
        result = self.run_child(source, wall_seconds=2, terminate_seconds=0.02,
                                drain_seconds=0.05)
        self.assertEqual(result["child_returncode"], 0)
        self.assertEqual(result["controller_exit"], 75)
        self.assertTrue(result["incomplete"])
        self.assertFalse(result["original_streams_complete"])

    def test_existing_destination_is_never_overwritten(self):
        self.output.mkdir()
        marker = self.output / "prior"
        marker.write_bytes(b"unchanged")
        with self.assertRaises((ValueError, FileExistsError)):
            self.run_child("pass")
        self.assertEqual(marker.read_bytes(), b"unchanged")

    def test_invalid_or_expanded_limits_are_refused_before_child(self):
        for options in ({"file_limit": 0}, {"file_limit": 262145},
                        {"aggregate_limit": 524289}, {"wall_seconds": 91},
                        {"drain_seconds": 4}, {"terminate_seconds": 3}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.run_child("pass", **options)
        self.assertFalse(self.output.exists())

    def test_runner_command_text_is_retained_without_raw_replay(self):
        stream = StringIO()
        with redirect_stdout(stream):
            result = self.run_child("import os;os.write(1,b'::error::synthetic\\n')")
        self.assertEqual(stream.getvalue(), "")
        self.assertEqual((self.output / "stdout.bin").read_bytes(), b"::error::synthetic\n")
        summary = self.api.inert_summary(result)
        self.assertNotIn("::", summary)
        self.assertNotIn("synthetic", summary)
        self.assertEqual(result["controller_exit"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
