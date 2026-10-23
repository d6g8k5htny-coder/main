"""Prospective binary capture controls; run only on the hosted source-bound lane."""
from contextlib import redirect_stdout
from importlib import import_module
from io import StringIO
import json
import os
import signal
import time
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


    def _resistant_descendant_program(self, ready, mode):
        # The direct child owns the captured pipes. Its same-group descendant
        # closes both streams, ignores TERM, announces its exact Linux identity,
        # and self-expires even if an assertion or controller cleanup fails.
        return (
            "import json,os,signal,time\n"
            "from pathlib import Path\n"
            "ready=Path(" + repr(str(ready)) + ")\n"
            "mode=" + repr(mode) + "\n"
            "leader=os.getpid()\n"
            "signal.signal(signal.SIGTERM,lambda *_:os._exit(0))\n"
            "signal.alarm(12)\n"
            "read_fd,write_fd=os.pipe()\n"
            "descendant=os.fork()\n"
            "if descendant==0:\n"
            " os.close(read_fd)\n"
            " signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
            " signal.signal(signal.SIGALRM,signal.SIG_DFL)\n"
            " signal.alarm(12)\n"
            " os.close(1)\n"
            " os.close(2)\n"
            " fields=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()\n"
            " record={'pid':os.getpid(),'group':os.getpgrp(),'leader':leader,'start':fields[19]}\n"
            " ready.write_text(json.dumps(record))\n"
            " os.write(write_fd,b'R')\n"
            " os.close(write_fd)\n"
            " while True: signal.pause()\n"
            "os.close(write_fd)\n"
            "token=os.read(read_fd,1)\n"
            "os.close(read_fd)\n"
            "if token!=b'R': os._exit(90)\n"
            "if mode=='overflow': os.write(1,b'x'*65)\n"
            "if mode=='post-loop':\n"
            " os.close(1)\n"
            " os.close(2)\n"
            " Path(str(ready)+'.parent-closed').write_bytes(b'closed')\n"
            "while True: signal.pause()\n"
        )

    def _fixture_record(self, ready):
        record = json.loads(ready.read_bytes())
        self.assertEqual(set(record), {"pid", "group", "leader", "start"})
        for key in ("pid", "group", "leader"):
            self.assertIs(type(record[key]), int)
            self.assertGreater(record[key], 1)
        self.assertIs(type(record["start"]), str)
        self.assertTrue(record["start"].isascii() and record["start"].isdigit())
        self.assertEqual(record["group"], record["leader"])
        self.assertNotEqual(record["pid"], record["leader"])
        self.assertNotEqual(record["group"], os.getpgrp())
        return record

    def _fixture_running(self, record):
        try:
            raw = Path("/proc/" + str(record["pid"]) + "/stat").read_text()
        except (FileNotFoundError, ProcessLookupError):
            return False
        fields = raw.rsplit(")", 1)[1].split()
        # A reused PID is not this fixture; an unreaped zombie is not running.
        return fields[19] == record["start"] and fields[0] not in ("Z", "X")

    def _cleanup_resistant_fixture(self, ready):
        if not ready.is_file():
            return
        record = self._fixture_record(ready)
        if self._fixture_running(record):
            self.assertEqual(os.getpgid(record["pid"]), record["group"])
            try:
                os.killpg(record["group"], signal.SIGKILL)
            except ProcessLookupError:
                pass
            end = time.monotonic() + 0.5
            while self._fixture_running(record) and time.monotonic() < end:
                time.sleep(0.005)
            self.assertFalse(self._fixture_running(record),
                             "test finally cleanup left its identified descendant running")

    def _run_group_cleanup_case(self, mode, expected_outcome, expected_exit):
        ready = Path(self.temp.name) / "descendant-ready.json"
        try:
            result = self.run_child(
                self._resistant_descendant_program(ready, mode),
                wall_seconds=1, terminate_seconds=0.15, drain_seconds=0.25,
                file_limit=32, aggregate_limit=64,
            )
            self.assertTrue(ready.is_file(), "descendant readiness was never established")
            record = self._fixture_record(ready)
            self.assertEqual(result["controller_outcome"], expected_outcome)
            self.assertEqual(result["controller_exit"], expected_exit)
            self.assertEqual(result["child_returncode"], 0)
            self.assertFalse(result["original_streams_complete"])
            self.assertLess(result["elapsed_seconds"], 3)
            if mode == "post-loop":
                self.assertEqual(Path(str(ready) + ".parent-closed").read_bytes(), b"closed")
                self.assertEqual(result["streams"]["stdout"]["observed_bytes"], 0)
                self.assertEqual(result["streams"]["stderr"]["observed_bytes"], 0)
            end = time.monotonic() + 0.5
            while self._fixture_running(record) and time.monotonic() < end:
                time.sleep(0.005)
            self.assertFalse(
                self._fixture_running(record),
                "identified TERM-resistant same-group descendant survived capture return",
            )
            return result
        finally:
            self._cleanup_resistant_fixture(ready)

    @unittest.skipUnless(sys.platform == "linux" and hasattr(os, "fork"),
                         "requires Linux POSIX process groups and /proc identity")
    def test_timeout_finishes_group_cleanup_after_term_closes_pipes(self):
        # Catches skipping group escalation when TERM exits the direct child
        # and selector EOF occurs before the termination grace expires.
        result = self._run_group_cleanup_case("timeout", "CHILD_TIMEOUT", 124)
        self.assertTrue(result["timed_out"])
        self.assertFalse(result["overflow"])

    @unittest.skipUnless(sys.platform == "linux" and hasattr(os, "fork"),
                         "requires Linux POSIX process groups and /proc identity")
    def test_overflow_finishes_group_cleanup_after_term_closes_pipes(self):
        # Catches treating an exited direct child and drained pipes as complete
        # cleanup while its TERM-resistant descendant remains in the group.
        result = self._run_group_cleanup_case("overflow", "OUTPUT_OVERFLOW", 74)
        self.assertTrue(result["overflow"])
        self.assertEqual((self.output / "stdout.bin").read_bytes(), b"x" * 32)

    @unittest.skipUnless(sys.platform == "linux" and hasattr(os, "fork"),
                         "requires Linux POSIX process groups and /proc identity")
    def test_post_loop_wait_timeout_finishes_process_group_cleanup(self):
        # Catches the distinct timeout after both captured pipes were already
        # closed: waiting for the direct child must still finish group cleanup.
        result = self._run_group_cleanup_case("post-loop", "CHILD_TIMEOUT", 124)
        self.assertTrue(result["timed_out"])
        self.assertFalse(result["overflow"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
