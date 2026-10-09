"""External TEST ledger controls; execute only in the hosted isolated runner.

Every request, token observation, account, accounting record, clock and native
identity is synthetic. The trusted controller alone records a fixed fake
operation. These fixtures do not authenticate a provider or admit a real call.
All 42 methods guard product existence before any fixture is created. Their
normal, -O and -OO children assert behavior with unittest, never Python assert.

Source-review correction follows the 81c3b1f9 test checkpoint and the accepted
32b384e0 contract checkpoint. No runtime RED/GREEN is inferred. The physical
64 MiB ENOSPC control is prepared; exhaustion of the 16 MiB DB/journal and
1024-task/4096-event limits is unexercised and remains NOT_RUN.
"""

from concurrent.futures import ThreadPoolExecutor
import copy
import errno
import hashlib
import importlib
import json
import os
from pathlib import Path
import selectors
import signal
import socket
import sqlite3
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "architecture/routing_broker.py"
CLI = ROOT / "tools/architecture_routing_broker.py"
FIXTURE_MOUNT = Path("/routing-fixtures")
REFUSAL = b"ARCHITECTURE_ROUTING_BROKER_REFUSED\n"
MIB = 1024 * 1024
MAX_INT = 2 ** 63 - 1
NOW = 1700000000000
INPUT = b"TEST routing input one\n"
SECOND_INPUT = b"TEST routing input two\n"
FIXED_RESULT = b"TEST fake response alpha\n"
PRIVATE_MARKER = b"PRIVATE_SYNTHETIC_ROUTING_MARKER"
SOURCE_BYTES = b"TEST original final accounting source\n"
TOKENIZER = "e" * 64
BUDGET = "b" * 32
_IN_CHILD = False
_CHILD_BASE = None
_API = None
_GUARD = threading.local()
_COUNTS = {"network": 0, "process": 0, "sdk": 0, "fake_io": 0}
_CALIBRATION = dict(_COUNTS)
_FAKE_FILES = set()

VIEW_COLUMNS = {
    "test_broker_accounts": "budget_id account_id account_limit held spent admission_status invariant_status version".split(),
    "test_broker_tasks": "task_id request_sha256 input_sha256 canonical_request canonical_policy canonical_price canonical_token_admission canonical_native_context registry_object_sha256 state attempt_id reservation_upper_bound reservation_held reservation_released spend_applied billing_status actual_charge completion_sha256 economic_sha256 result_blob latest_receipt".split(),
    "test_broker_attempts": "attempt_id task_id request_sha256 claim_event_sequence capacity_slot_held native_provider_request_id".split(),
    "test_broker_completions": "attempt_id completion_sha256 canonical_packet original_receipt".split(),
    "test_broker_accounting": "attempt_id economic_sha256 canonical_usage accounting_source_sha256 actual_charge applied_event_sequence".split(),
    "test_broker_settlements": "settlement_id attempt_id settlement_sha256 canonical_packet accounting_evidence_sha256 applied_event_sequence original_receipt".split(),
    "test_broker_events": "event_sequence task_id attempt_id kind request_sha256 completion_sha256 economic_sha256 settlement_sha256 charge_nano_usd_decimal aggregate_spent_nano_usd_decimal".split(),
    "test_broker_results": "task_id request_sha256 result_sha256 result_bytes result_blob".split(),
}
VIEW_BLOB_COLUMNS = {
    "test_broker_tasks": set("canonical_request canonical_policy canonical_price canonical_token_admission canonical_native_context result_blob latest_receipt".split()),
    "test_broker_completions": {"canonical_packet", "original_receipt"},
    "test_broker_accounting": {"canonical_usage"},
    "test_broker_settlements": {"canonical_packet", "original_receipt"},
    "test_broker_results": {"result_blob"},
}
VIEW_INTEGER_COLUMNS = {
    "test_broker_accounts": {"account_limit", "held", "spent", "version"},
    "test_broker_tasks": set("reservation_upper_bound reservation_held reservation_released spend_applied actual_charge".split()),
    "test_broker_attempts": {"claim_event_sequence", "capacity_slot_held"},
    "test_broker_accounting": {"actual_charge", "applied_event_sequence"},
    "test_broker_settlements": {"applied_event_sequence"},
    "test_broker_events": {"event_sequence"},
    "test_broker_results": {"result_bytes"},
}
VIEW_NULLABLE_COLUMNS = {
    "test_broker_accounts": {"account_limit", "spent"},
    "test_broker_tasks": set("state attempt_id reservation_upper_bound actual_charge completion_sha256 economic_sha256 result_blob".split()),
    "test_broker_attempts": {"native_provider_request_id"},
    "test_broker_events": set("task_id attempt_id request_sha256 completion_sha256 economic_sha256 settlement_sha256 charge_nano_usd_decimal aggregate_spent_nano_usd_decimal".split()),
}
PREDICATE_COLUMNS = {"reservation_held", "reservation_released", "spend_applied", "capacity_slot_held"}
EVENT_KINDS = {"RESERVATION_DENIED", "RESERVED", "DISPATCH_CLAIMED", "UNKNOWN_MARKED",
               "NONDISPATCH_CONFIRMED", "COMPLETION_RECORDED", "ACCOUNTING_APPLIED", "ACCOUNT_FROZEN"}
RECEIPT_FIELDS = {
    "request": "task_id input_sha256 request_sha256 disclosure_scope operation required_capabilities account_id provider_id deployment_id model_id policy_sha256 price_sha256 budget_id deadline_unix_ms context_tokens max_output_tokens max_cost_nano_usd".split(),
    "admission": "status tier reason policy_sha256 price_sha256".split(),
    "delivery": "state attempt_id transport_kind transport_attempts native_provider_request_id outcome".split(),
    "budget": "currency scale account_limit reservation_upper_bound held spent actual_charge billing_status invariant_status".split(),
    "result": "sha256 bytes acceptance".split(),
    "evidence": "repository checked_commit run_id run_attempt policy_source_sha256 price_source_sha256 adapter_sha256 input_token_admission_sha256 completion_sha256 settlement_id settlement_sha256 accounting_evidence_sha256 ledger_schema_version event_sequence".split(),
}


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def ident(number):
    return "%032x" % number


def registry():
    price = {
        "currency": "USD", "scale": 9, "input_rate_per_million": 1000000,
        "cached_input_rate_per_million": 500000,
        "cache_write_rate_per_million": 1000000,
        "output_rate_per_million": 2000000, "fixed_request_charge": 3,
        "reasoning_included_in_output": True, "tools_disabled": True,
        "valid_from_unix_ms": 1, "valid_until_unix_ms": MAX_INT,
    }
    card = digest(canonical(price))
    deployments = {}
    routes = {}
    for tier, operation in enumerate(("navigation", "coding_review", "mathematical_audit"), 1):
        name = "TEST-deployment-%d" % tier
        routes[operation] = [name]
        deployments[name] = {
            "account_id": "TEST-account", "provider_id": "TEST-provider",
            "model_id": "TEST-model-%d" % tier, "tier": tier,
            "operations": [operation], "capabilities": ["source_read"],
            "status": "ADMITTED", "adapter_id": "TEST",
            "api_identity_sha256": digest(b"TEST API identity\n"),
            "capacity_status": "ADMITTED", "max_parallel_attempts": 32,
            "max_context_tokens": 1000, "max_output_tokens": 32,
            "price_sha256": card, "tokenizer_sha256": TOKENIZER,
        }
    counts = {}
    for payload, count in ((INPUT, 7), (SECOND_INPUT, 9)):
        counts[digest(payload)] = {
            name: {TOKENIZER: {"input_tokens": count, "payload_bytes": len(payload),
                              "count_evidence_sha256": digest(b"TEST count observation " + payload)}}
            for name in deployments
        }
    return {
        "schema_version": 1, "kind": "TEST_ROUTING_REGISTRY",
        "native_context": {"repository": "TEST-owner/TEST-repository",
                           "checked_commit": "a" * 40, "run_id": "12345", "run_attempt": "1"},
        "policy": {"version": 1, "valid_from_unix_ms": 1,
                   "valid_until_unix_ms": MAX_INT, "routes": routes,
                   "redis_status": "NOT_USED", "storage_status": "ADMITTED"},
        "price_cards": {card: price},
        "accounts": {BUDGET: {"account_id": "TEST-account", "currency": "USD", "scale": 9,
                              "scope": "BROKER_TRAFFIC", "limit_nano_usd": 1024,
                              "status": "ADMITTED", "version": 1}},
        "deployments": deployments,
        "disclosure_admissions": {key: ["local_private", "public", "synthetic"] for key in counts},
        "input_token_admissions": counts,
        "accounting_record_admissions": {},
        "limits": {"max_tasks": 1024, "max_events": 4096, "max_result_bytes": 65536,
                   "db_bytes": 16 * MIB, "retained_fixture_bytes": 32 * MIB},
    }


def replace_price(reg, **changes):
    old = reg["deployments"]["TEST-deployment-1"]["price_sha256"]
    price = copy.deepcopy(reg["price_cards"][old])
    price.update(changes)
    new = digest(canonical(price))
    reg["price_cards"][new] = price
    for deployment in reg["deployments"].values():
        deployment["price_sha256"] = new
    return new


def _audit(event, arguments):
    active = getattr(_GUARD, "active", False)
    label = None
    if event in ("socket.connect", "socket.getaddrinfo", "urllib.Request"):
        label = "network"
    elif event in ("subprocess.Popen", "os.system", "os.posix_spawn") and active:
        label = "process"
    elif event == "import" and active and str(arguments[0]).split(".")[0] in {
        "litellm", "openai", "anthropic", "ollama", "requests", "httpx",
    }:
        label = "sdk"
    elif event == "open" and active and not isinstance(arguments[0], int):
        target = os.path.abspath(os.fsdecode(arguments[0]))
        if target in _FAKE_FILES:
            label = "fake_io"
    if label is not None:
        if active:
            _COUNTS[label] += 1
        raise RuntimeError("TEST_EXTERNAL_ACTIVITY_GUARD")


def _guarded(function, *arguments, **keywords):
    old = getattr(_GUARD, "active", False)
    _GUARD.active = True
    try:
        return function(*arguments, **keywords)
    finally:
        _GUARD.active = old


def _load_api():
    global _API
    if _API is None:
        sys.path.insert(0, str(ROOT))
        _API = _guarded(importlib.import_module, "architecture.routing_broker")
    return _API


def bounded_process(command, payload=b"", *, pass_fds=(), new_session=False):
    """Bound pipes and wall time before retaining untrusted child output."""
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, pass_fds=pass_fds,
                               start_new_session=new_session,
                               env={"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8"})
    selected = selectors.DefaultSelector()
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    pending = memoryview(payload)
    for stream, name in ((process.stdout, "stdout"), (process.stderr, "stderr")):
        os.set_blocking(stream.fileno(), False)
        selected.register(stream, selectors.EVENT_READ, name)
    if pending:
        os.set_blocking(process.stdin.fileno(), False)
        selected.register(process.stdin, selectors.EVENT_WRITE, "stdin")
    else:
        process.stdin.close()
    deadline = time.monotonic() + 20
    try:
        while selected.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError("TEST child exceeded its 20-second cap")
            for key, _ in selected.select(min(remaining, 0.2)):
                stream, name = key.fileobj, key.data
                if name == "stdin":
                    try:
                        amount = os.write(stream.fileno(), pending[:65536])
                        pending = pending[amount:]
                    except BrokenPipeError:
                        pending = pending[:0]
                    if not pending:
                        selected.unregister(stream)
                        stream.close()
                else:
                    block = os.read(stream.fileno(), 65536)
                    if not block:
                        selected.unregister(stream)
                        stream.close()
                    else:
                        if sum(map(len, buffers.values())) + len(block) > 4 * MIB:
                            raise AssertionError("TEST child output exceeded 4 MiB")
                        buffers[name].extend(block)
        status = process.wait(timeout=max(0.01, deadline - time.monotonic()))
        return status, bytes(buffers["stdout"]), bytes(buffers["stderr"])
    finally:
        if process.poll() is None:
            if new_session:
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait(timeout=2)
        selected.close()
        for stream in (process.stdin, process.stdout, process.stderr):
            if not stream.closed:
                stream.close()


CHILD_LAUNCHER = r'''
import importlib.util, pathlib, sys, unittest
source, method, base = sys.argv[1:4]
spec = importlib.util.spec_from_file_location("routing_external_TEST", source)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
module._IN_CHILD = True
module._CHILD_BASE = pathlib.Path(base)
sys.addaudithook(module._audit)
suite = unittest.TestSuite([module.ArchitectureRoutingBrokerContract(method)])
result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
'''

CLI_LAUNCHER = r'''
import json, os, runpy, socket, sys
cli, report_number, fake = sys.argv[1:4]
arguments = sys.argv[4:]
report_fd = int(report_number)
counts = {"network": 0, "process": 0, "sdk": 0, "fake_io": 0}
fake = os.path.abspath(fake)
def audit(event, values):
    label = None
    if event in ("socket.connect", "socket.getaddrinfo", "urllib.Request"):
        label = "network"
    elif event in ("subprocess.Popen", "os.system", "os.posix_spawn"):
        label = "process"
    elif event == "import" and str(values[0]).split(".")[0] in (
        "litellm", "openai", "anthropic", "ollama", "requests", "httpx"
    ):
        label = "sdk"
    elif event == "open" and not isinstance(values[0], int) and os.path.abspath(os.fsdecode(values[0])) == fake:
        label = "fake_io"
    if label is not None:
        counts[label] += 1
        raise RuntimeError("TEST_EXTERNAL_ACTIVITY_GUARD")
sys.addaudithook(audit)
try:
    try:
        with socket.socket() as connection:
            connection.connect(("127.0.0.1", 9))
    except RuntimeError:
        pass
    try:
        open(fake, "rb")
    except RuntimeError:
        pass
    calibration = dict(counts)
    counts = dict.fromkeys(counts, 0)
    sys.argv = [cli, *arguments]
    sys.path[0] = os.path.dirname(cli)
    runpy.run_path(cli, run_name="__main__")
finally:
    raw = (json.dumps({"calibration": calibration, "attempts": counts,
                      "optimize": sys.flags.optimize}, sort_keys=True,
                     separators=(",", ":")) + "\n").encode("ascii")
    os.write(report_fd, raw)
    os.close(report_fd)
'''

CONTROLLER_LAUNCHER = r'''
import importlib.util, json, os, pathlib, sys, unittest
source, ledger, registry_file, fake_file, phase, clock = sys.argv[1:7]
spec = importlib.util.spec_from_file_location("routing_crash_TEST", source)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
module._FAKE_FILES.add(os.path.abspath(fake_file))
sys.addaudithook(module._audit)
check = unittest.TestCase()
registry = json.loads(pathlib.Path(registry_file).read_bytes())
request = json.loads(sys.stdin.buffer.read(65537))
api = module._load_api()
broker = module._guarded(api.Broker, pathlib.Path(ledger), registry, clock=lambda: int(clock))
module._guarded(broker.reserve, request)
if phase == "before-claim":
    check.assertEqual(module._COUNTS, dict.fromkeys(module._COUNTS, 0))
    os._exit(17)
claim = json.loads(module._guarded(broker.claim_dispatch, request["task_id"], module.digest(module.canonical(request))))
if phase == "after-claim":
    check.assertEqual(module._COUNTS, dict.fromkeys(module._COUNTS, 0))
    os._exit(18)
check.assertEqual(phase, "after-fake")
event = {"schema_version": 1, "kind": "TEST_FAKE_ATTEMPT", "event": "ACCEPTED",
         "task_id": request["task_id"], "attempt_id": claim["delivery"]["attempt_id"],
         "request_sha256": module.digest(module.canonical(request)),
         "result_sha256": module.digest(module.FIXED_RESULT), "result_bytes": len(module.FIXED_RESULT)}
descriptor = os.open(fake_file, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
encoded = module.canonical(event)
check.assertEqual(os.write(descriptor, encoded), len(encoded))
os.fsync(descriptor)
os.close(descriptor)
check.assertEqual(module._COUNTS, dict.fromkeys(module._COUNTS, 0))
os._exit(19)
'''


class Fixture:
    def __init__(self, check, reg=None):
        self.check = check
        self.base = Path(tempfile.mkdtemp(prefix="case-", dir=_CHILD_BASE))
        self.ledger = self.base / "ledger"
        self.ledger.mkdir(mode=0o700)
        self.controller = self.base / "fixture-controller"
        self.controller.mkdir(mode=0o700)
        self.fake_file = self.controller / "attempts.jsonl"
        _FAKE_FILES.add(str(self.fake_file))
        self.registry_file = self.base / "registry.json"
        self.reg = copy.deepcopy(reg if reg is not None else registry())
        self.clock = [NOW]
        self.api = _load_api()
        self.broker = self.reopen()
        self.requests = {}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if self.broker is not None:
            _guarded(self.broker.close)
        self.check.assertEqual(_COUNTS, dict.fromkeys(_COUNTS, 0))
        total = 0
        for path in self.base.rglob("*"):
            info = path.lstat()
            if stat.S_ISREG(info.st_mode):
                total += info.st_size
                if path.name in ("routing-broker.sqlite3", "routing-broker.sqlite3-journal"):
                    self.check.assertLessEqual(info.st_size, 16 * MIB)
        self.check.assertLessEqual(total, 32 * MIB)

    def reopen(self, reg=None):
        chosen = self.reg if reg is None else reg
        return _guarded(self.api.Broker, self.ledger, chosen, clock=lambda: self.clock[0])

    def call(self, name, *arguments, broker=None):
        return _guarded(getattr(broker or self.broker, name), *arguments)

    def request(self, task=1, *, operation="navigation", payload=INPUT, **changes):
        deployment = self.reg["policy"]["routes"][operation][0]
        record = self.reg["deployments"][deployment]
        value = {
            "task_id": ident(task), "input_sha256": digest(payload),
            "disclosure_scope": "synthetic", "operation": operation,
            "required_capabilities": ["source_read"], "account_id": record["account_id"],
            "provider_id": record["provider_id"], "deployment_id": deployment,
            "model_id": record["model_id"], "policy_sha256": digest(canonical(self.reg["policy"])),
            "price_sha256": record["price_sha256"], "budget_id": BUDGET,
            "max_cost_nano_usd": MAX_INT, "deadline_unix_ms": MAX_INT,
            "context_tokens": 10 if payload == INPUT else 12, "max_output_tokens": 3,
        }
        value.update(changes)
        self.requests[value["task_id"]] = value
        return value

    def receipt(self, raw):
        self.check.assertIs(type(raw), bytes)
        self.check.assertLessEqual(len(raw), 256 * 1024)
        value = json.loads(raw)
        self.check.assertEqual(raw, canonical(value))
        self.check.assertEqual(set(value), {"schema_version", "scientific_effect",
                                           "scientific_status_authority", "custody", "kind",
                                           *RECEIPT_FIELDS})
        self.check.assertIs(type(value["schema_version"]), int)
        self.check.assertEqual(value["schema_version"], 1)
        self.check.assertEqual(value["scientific_effect"], "NONE")
        self.check.assertIs(value["scientific_status_authority"], False)
        self.check.assertEqual(value["custody"], "unknown")
        self.check.assertEqual(value["kind"], "TEST_BROKER_RECEIPT")
        for name, fields in RECEIPT_FIELDS.items():
            self.check.assertEqual(set(value[name]), set(fields))
        for name in ("deadline_unix_ms", "context_tokens", "max_output_tokens", "max_cost_nano_usd"):
            self.check.assertIs(type(value["request"][name]), int)
        self.check.assertEqual((value["budget"]["currency"], value["budget"]["scale"]), ("USD", 9))
        self.check.assertIs(type(value["budget"]["scale"]), int)
        for name in ("account_limit", "reservation_upper_bound", "held", "spent", "actual_charge"):
            amount = value["budget"][name]
            if amount is not None:
                self.check.assertIs(type(amount), int)
                self.check.assertGreaterEqual(amount, 0)
                self.check.assertLessEqual(amount, MAX_INT)
        self.check.assertIs(type(value["evidence"]["ledger_schema_version"]), int)
        self.check.assertEqual(value["evidence"]["ledger_schema_version"], 1)
        if value["evidence"]["event_sequence"] is not None:
            self.check.assertIs(type(value["evidence"]["event_sequence"]), int)
            self.check.assertGreaterEqual(value["evidence"]["event_sequence"], 0)
        if value["result"]["bytes"] is not None:
            self.check.assertIs(type(value["result"]["bytes"]), int)
            self.check.assertGreaterEqual(value["result"]["bytes"], 0)
        attempts = value["delivery"]["transport_attempts"]
        state = value["delivery"]["state"]
        if state in (None, "reserved", "confirmed_not_dispatched"):
            self.check.assertIs(type(attempts), int)
            self.check.assertEqual(attempts, 0)
        elif state in ("dispatching", "unknown_delivery"):
            self.check.assertIsNone(attempts)
        elif state == "completed":
            self.check.assertIs(type(attempts), int)
            self.check.assertEqual(attempts, 1)
        self.check.assertEqual(value["result"]["acceptance"], "NOT_ASSESSED")
        self.check.assertNotIn(PRIVATE_MARKER, raw)
        self.check.assertNotIn(FIXED_RESULT.rstrip(b"\n"), raw)
        return value

    def reserve(self, request=None, **changes):
        request = self.request(**changes) if request is None else request
        raw = self.call("reserve", request)
        return raw, self.receipt(raw)

    def claim(self, request):
        raw = self.call("claim_dispatch", request["task_id"], digest(canonical(request)))
        value = self.receipt(raw)
        self.check.assertEqual(value["delivery"]["state"], "dispatching")
        attempt = value["delivery"]["attempt_id"]
        self.check.assertRegex(attempt, r"^[0-9a-f]{32}$")
        return raw, value

    def read(self, request):
        return self.receipt(self.call("read", request["task_id"], digest(canonical(request))))

    def fake(self, request, claim):
        event = {"schema_version": 1, "kind": "TEST_FAKE_ATTEMPT", "event": "ACCEPTED",
                 "task_id": request["task_id"], "attempt_id": claim["delivery"]["attempt_id"],
                 "request_sha256": digest(canonical(request)),
                 "result_sha256": digest(FIXED_RESULT), "result_bytes": len(FIXED_RESULT)}
        descriptor = os.open(self.fake_file, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            encoded = canonical(event)
            amount = os.write(descriptor, encoded)
            self.check.assertEqual(amount, len(encoded))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self.check.assertEqual(self.fake_file.stat().st_mode & 0o777, 0o600)
        self.check.assertLessEqual(self.fake_file.stat().st_size, 64 * 1024)
        return {"status": "RESULT", "result_utf8": FIXED_RESULT.decode("utf-8"),
                "native_provider_request_id": "TEST-request-" + request["task_id"],
                "error_code": None}

    def fake_records(self):
        if not self.fake_file.exists():
            return []
        return [json.loads(line) for line in self.fake_file.read_bytes().splitlines()]

    def usage(self, **changes):
        value = {"input_tokens": 7, "output_tokens": 2, "cached_input_tokens": 2,
                 "cache_write_tokens": 1, "reasoning_tokens": 1,
                 "actual_charge_nano_usd": 13, "accounting_source_sha256": digest(SOURCE_BYTES)}
        value.update(changes)
        return value

    def finish(self, *, task=1, known=True, request=None):
        request = self.request(task=task) if request is None else request
        self.reserve(request)
        _, claim = self.claim(request)
        outcome = self.fake(request, claim)
        usage = self.usage() if known else None
        raw = self.call("complete", claim["delivery"]["attempt_id"], outcome, usage)
        return request, claim, outcome, usage, raw, self.receipt(raw)

    def evidence(self, request, attempt, usage, *, completed=True):
        state = self.read(request)
        original = b"TEST original final accounting observation " + attempt.encode("ascii") + b"\n"
        return {
            "schema_version": 1, "kind": "TEST_ACCOUNTING_EVIDENCE", "attempt_id": attempt,
            "account_id": request["account_id"], "provider_id": request["provider_id"],
            "deployment_id": request["deployment_id"], "model_id": request["model_id"],
            "input_sha256": request["input_sha256"], "request_sha256": digest(canonical(request)),
            "policy_sha256": request["policy_sha256"], "price_sha256": request["price_sha256"],
            "tokenizer_sha256": TOKENIZER,
            "completion_sha256": state["evidence"]["completion_sha256"] if completed else None,
            "native_provider_request_id": state["delivery"]["native_provider_request_id"],
            "usage_sha256": digest(canonical(usage)),
            "accounting_source_sha256": usage["accounting_source_sha256"],
            "accounting_record_sha256": digest(original),
        }

    def admit(self, evidence):
        self.reg["accounting_record_admissions"][evidence["accounting_record_sha256"]] = copy.deepcopy(evidence)
        self.broker.close()
        self.broker = self.reopen()

    def rows(self, view):
        self.check.assertIn(view, VIEW_COLUMNS)
        database = self.ledger / "routing-broker.sqlite3"
        connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
        try:
            connection.execute("PRAGMA query_only=ON")
            definition = connection.execute("SELECT type,sql FROM sqlite_master WHERE name=?", (view,)).fetchone()
            self.check.assertIsNotNone(definition)
            self.check.assertEqual(definition[0], "view")
            admitted = connection.execute("SELECT name,type FROM sqlite_master WHERE type IN ('view','trigger') ORDER BY name,type").fetchall()
            self.check.assertEqual(admitted, [(name, "view") for name in sorted(VIEW_COLUMNS)])
            columns = VIEW_COLUMNS[view]
            cursor = connection.execute("SELECT *," + ",".join("typeof(" + name + ")" for name in columns) + " FROM " + view)
            self.check.assertEqual([column[0] for column in cursor.description[:len(columns)]], columns)
            values = []
            for original in cursor.fetchall():
                row = dict(zip(columns, original[:len(columns)]))
                self._validate_row(view, row, original[len(columns):])
                values.append(row)
            self.check.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 1)
        finally:
            connection.close()
        if view == "test_broker_events":
            values.sort(key=lambda row: row["event_sequence"])
        else:
            values.sort(key=lambda row: str(row[VIEW_COLUMNS[view][0]]))
        return values

    def _validate_row(self, view, row, sql_types):
        # typeof() checks actual SQL values; Python equality would accept 13.0 as 13.
        for column, sql_type in zip(VIEW_COLUMNS[view], sql_types):
            value = row[column]
            if value is None:
                self.check.assertIn(column, VIEW_NULLABLE_COLUMNS.get(view, set()))
                self.check.assertEqual(sql_type, "null")
                continue
            if column in VIEW_BLOB_COLUMNS.get(view, set()):
                self.check.assertEqual(sql_type, "blob")
                self.check.assertIs(type(value), bytes)
                if column != "result_blob":
                    self.check.assertEqual(value, canonical(json.loads(value)))
            elif column in VIEW_INTEGER_COLUMNS.get(view, set()):
                self.check.assertEqual(sql_type, "integer")
                self.check.assertIs(type(value), int)
                self.check.assertGreaterEqual(value, 0)
                self.check.assertLessEqual(value, MAX_INT)
                if column in PREDICATE_COLUMNS:
                    self.check.assertIn(value, (0, 1))
                elif column in {"version", "event_sequence", "claim_event_sequence", "applied_event_sequence"}:
                    self.check.assertGreater(value, 0)
                elif column == "result_bytes":
                    self.check.assertLessEqual(value, 65536)
            else:
                self.check.assertEqual(sql_type, "text")
                self.check.assertIs(type(value), str)
                if column.endswith("_sha256"):
                    self.check.assertRegex(value, r"^[0-9a-f]{64}$")
                elif column in {"task_id", "attempt_id", "settlement_id", "budget_id"}:
                    self.check.assertRegex(value, r"^[0-9a-f]{32}$")
                elif column.endswith("_nano_usd_decimal"):
                    self.check.assertRegex(value, r"^(0|[1-9][0-9]*)$")
                elif column == "native_provider_request_id":
                    self.check.assertRegex(value, r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$")
        if view == "test_broker_accounts":
            self.check.assertIn(row["admission_status"], {"ADMITTED", "UNKNOWN", "DISABLED", "FROZEN"})
            self.check.assertIn(row["invariant_status"], {"WITHIN_LIMIT", "VIOLATED", "UNKNOWN"})
            if row["account_limit"] is None:
                self.check.assertEqual(row["invariant_status"], "UNKNOWN")
            if row["spent"] is None:
                self.check.assertIn(row["invariant_status"], {"VIOLATED", "UNKNOWN"})
        elif view == "test_broker_tasks":
            self.check.assertIn(row["state"], {None, "reserved", "dispatching", "completed", "confirmed_not_dispatched", "unknown_delivery"})
            self.check.assertIn(row["billing_status"], {"KNOWN", "UNKNOWN"})
            self.check.assertEqual(row["actual_charge"] is None, row["billing_status"] == "UNKNOWN")
            if row["state"] is None:
                denial = json.loads(row["latest_receipt"])
                self.check.assertEqual(denial["admission"]["status"], "DENIED")
                self.check.assertIsNone(denial["delivery"]["state"])
                for name in ("attempt_id", "actual_charge", "completion_sha256", "economic_sha256", "result_blob"):
                    self.check.assertIsNone(row[name])
                for name in ("reservation_held", "reservation_released", "spend_applied"):
                    self.check.assertEqual(row[name], 0)
            elif row["state"] in {"reserved", "confirmed_not_dispatched"}:
                self.check.assertIsNone(row["attempt_id"])
                for name in ("actual_charge", "completion_sha256", "economic_sha256", "result_blob"):
                    self.check.assertIsNone(row[name])
            else:
                self.check.assertIsNotNone(row["attempt_id"])
                self.check.assertIsNotNone(row["reservation_upper_bound"])
                if row["state"] == "completed":
                    self.check.assertIsNotNone(row["completion_sha256"])
                else:
                    self.check.assertIsNone(row["completion_sha256"])
                    self.check.assertIsNone(row["result_blob"])
            if row["state"] is not None:
                self.check.assertEqual(json.loads(row["latest_receipt"])["admission"]["status"], "ADMITTED")
                self.check.assertIsNotNone(row["reservation_upper_bound"])
            self.check.assertEqual(row["economic_sha256"] is None, row["billing_status"] == "UNKNOWN")
        elif view == "test_broker_events":
            self.check.assertIn(row["kind"], EVENT_KINDS)
            if row["kind"] == "ACCOUNTING_APPLIED":
                for name in ("attempt_id", "economic_sha256", "charge_nano_usd_decimal", "aggregate_spent_nano_usd_decimal"):
                    self.check.assertIsNotNone(row[name])
            elif row["kind"] in {"RESERVATION_DENIED", "RESERVED", "NONDISPATCH_CONFIRMED"}:
                for name in ("task_id", "request_sha256"):
                    self.check.assertIsNotNone(row[name])
                for name in ("attempt_id", "completion_sha256", "economic_sha256", "settlement_sha256",
                             "charge_nano_usd_decimal", "aggregate_spent_nano_usd_decimal"):
                    self.check.assertIsNone(row[name])

    def snapshot(self):
        return tuple((name, tuple(sorted(repr(row) for row in self.rows(name))))
                     for name in VIEW_COLUMNS)

    def account(self):
        rows = self.rows("test_broker_accounts")
        self.check.assertEqual(len(rows), 1)
        return rows[0]

    def refuse(self, name, *arguments, broker=None):
        with self.check.assertRaises(self.api.RoutingBrokerRefused) as caught:
            self.call(name, *arguments, broker=broker)
        self.check.assertEqual(str(caught.exception), REFUSAL.decode("ascii").rstrip("\n"))

    def refuse_fresh_reserve(self, request):
        # Namespace admission must be tested on a new constructor, never a closed instance.
        self.check.assertIsNone(self.broker)
        try:
            self.broker = self.reopen()
        except self.api.RoutingBrokerRefused as error:
            self.check.assertEqual(str(error), REFUSAL.decode("ascii").rstrip("\n"))
        else:
            self.refuse("reserve", request, broker=self.broker)

    def cli(self, action, packet, *, raw=None, arguments=None):
        self.registry_file.write_bytes(canonical(self.reg))
        self.registry_file.chmod(0o600)
        flags = ["--test-ledger-root", str(self.ledger), "--test-registry", str(self.registry_file),
                 "--action", action] if arguments is None else arguments
        read_fd, write_fd = os.pipe()
        try:
            command = [sys.executable, "-I", "-B", "-S", *(["-" + "O" * sys.flags.optimize] if sys.flags.optimize else []),
                       "-c", CLI_LAUNCHER, str(CLI), str(write_fd), str(self.fake_file), *flags]
            result = bounded_process(command, canonical(packet) if raw is None else raw, pass_fds=(write_fd,))
            os.close(write_fd)
            write_fd = None
            report = json.loads(os.read(read_fd, 16384))
            self.check.assertGreaterEqual(report["calibration"]["network"], 1)
            self.check.assertGreaterEqual(report["calibration"]["fake_io"], 1)
            self.check.assertEqual(report["attempts"], dict.fromkeys(_COUNTS, 0))
            self.check.assertEqual(report["optimize"], sys.flags.optimize)
            self.check.assertNotIn(PRIVATE_MARKER, result[1] + result[2])
            return result
        finally:
            os.close(read_fd)
            if write_fd is not None:
                os.close(write_fd)

    def cli_success(self, action, packet):
        status, output, error = self.cli(action, packet)
        self.check.assertEqual(status, 0)
        self.check.assertEqual(error, b"")
        return output, self.receipt(output)

    def cli_refusal(self, action, packet, **changes):
        status, output, error = self.cli(action, packet, **changes)
        self.check.assertEqual((status, output, error), (1, b"", REFUSAL))

    def crash_controller(self, phase, request):
        self.registry_file.write_bytes(canonical(self.reg))
        self.registry_file.chmod(0o600)
        flags = [] if sys.flags.optimize == 0 else ["-" + "O" * sys.flags.optimize]
        command = [sys.executable, "-I", "-B", "-S", *flags, "-c", CONTROLLER_LAUNCHER,
                   str(Path(__file__).resolve()), str(self.ledger), str(self.registry_file),
                   str(self.fake_file), phase, str(self.clock[0])]
        result = bounded_process(command, canonical(request))
        self.check.assertEqual(result, ({"before-claim": 17, "after-claim": 18, "after-fake": 19}[phase], b"", b""))


class ArchitectureRoutingBrokerContract(unittest.TestCase):
    def _require_product(self):
        missing = [str(path.relative_to(ROOT)) for path in (CLI, MODULE) if not path.is_file()]
        self.assertEqual(missing, [], "TEST routing CLI/module must exist before fixtures: " + ", ".join(missing))

    def _outer(self):
        if _IN_CHILD:
            return False
        self.assertTrue(FIXTURE_MOUNT.is_dir(), "trusted hosted /routing-fixtures mount is required")
        filesystem = os.statvfs(FIXTURE_MOUNT)
        self.assertGreaterEqual(filesystem.f_blocks * filesystem.f_frsize, 32 * MIB)
        self.assertLessEqual(filesystem.f_blocks * filesystem.f_frsize, 64 * MIB)
        mounts = [line.split(" - ", 1) for line in Path("/proc/self/mountinfo").read_text().splitlines()]
        matches = [left.split()[5].split(",") for left, _ in mounts if left.split()[4] == str(FIXTURE_MOUNT)]
        self.assertEqual(len(matches), 1)
        self.assertTrue({"rw", "nodev", "nosuid", "noexec"}.issubset(matches[0]))
        for mode in (0, 1, 2):
            with self.subTest(active_child_optimize=mode):
                with tempfile.TemporaryDirectory(prefix="routing-controller-", dir=FIXTURE_MOUNT) as base:
                    flags = [] if mode == 0 else ["-" + "O" * mode]
                    command = [sys.executable, "-I", "-B", "-S", *flags, "-c", CHILD_LAUNCHER,
                               str(Path(__file__).resolve()), self._testMethodName, base]
                    status, output, error = bounded_process(command, new_session=True)
                    self.assertEqual(status, 0, error.decode("utf-8", errors="replace"))
                    self.assertEqual(output, b"")
                    total = sum(path.lstat().st_size for path in Path(base).rglob("*")
                                if stat.S_ISREG(path.lstat().st_mode))
                    self.assertLessEqual(total, 32 * MIB)
        return True

    def fixture(self, reg=None):
        return Fixture(self, reg)

    def race(self, function, count=16):
        gate = threading.Barrier(count)
        def once(index):
            gate.wait(timeout=3)
            try:
                return ("returned", function(index))
            except _load_api().RoutingBrokerRefused:
                return ("refused", None)
        with ThreadPoolExecutor(max_workers=count) as pool:
            futures = [pool.submit(once, index) for index in range(count)]
            return [future.result(timeout=8) for future in futures]

    def test_reserve_claim_complete(self):
        self._require_product()
        if self._outer():
            return
        with self.fixture() as f:
            request = f.request()
            original, reserved = f.reserve(request)
            self.assertEqual(reserved["admission"]["status"], "ADMITTED")
            self.assertEqual((f.account()["held"], f.account()["spent"]), (16, 0))
            task = f.rows("test_broker_tasks")[0]
            self.assertEqual(task["canonical_request"], canonical(request))
            self.assertEqual(task["canonical_native_context"], canonical(f.reg["native_context"]))
            self.assertEqual(task["registry_object_sha256"], digest(canonical(f.reg)))
            self.assertEqual(task["canonical_policy"], canonical(f.reg["policy"]))
            self.assertEqual(task["canonical_price"], canonical(f.reg["price_cards"][request["price_sha256"]]))
            self.assertEqual(f.call("reserve", request), original)
            _, claim = f.claim(request)
            self.assertIsNone(claim["delivery"]["transport_attempts"])
            outcome = f.fake(request, claim)
            self.assertEqual(len(f.fake_records()), 1)
            usage = f.usage()
            completed_raw = f.call("complete", claim["delivery"]["attempt_id"], outcome, usage)
            completed = f.receipt(completed_raw)
            self.assertEqual(completed["delivery"]["state"], "completed")
            self.assertEqual((f.account()["held"], f.account()["spent"]), (0, 13))
            self.assertEqual(f.call("read_result", request["task_id"], digest(canonical(request))), FIXED_RESULT)
            result = f.rows("test_broker_results")[0]
            self.assertEqual((result["result_blob"], result["result_sha256"], result["result_bytes"]),
                             (FIXED_RESULT, digest(FIXED_RESULT), len(FIXED_RESULT)))
            self.assertEqual(len(f.rows("test_broker_accounting")), 1)
            self.assertEqual(len([row for row in f.rows("test_broker_events") if row["kind"] == "ACCOUNTING_APPLIED"]), 1)
            for view in VIEW_COLUMNS:
                f.rows(view)

    def test_integer_cost_ceiling(self):
        self._require_product()
        if self._outer():
            return
        # Literal expectations were calculated from rational subtotals, not the broker.
        cases = [(1, 1, 1, 1, 0, 2, 2),
                 (1000001, 1000001, 1000001, 1000001, 0, 7, 7),
                 (0, 0, 0, 0, 0, 0, 0)]
        for ordinary, cached, write, output, fixed, upper, charge in cases:
            reg = registry()
            replace_price(reg, input_rate_per_million=ordinary, cached_input_rate_per_million=cached,
                          cache_write_rate_per_million=write, output_rate_per_million=output,
                          fixed_request_charge=fixed)
            reg["input_token_admissions"][digest(INPUT)]["TEST-deployment-1"][TOKENIZER]["input_tokens"] = 3
            with self.fixture(reg) as f:
                request = f.request(context_tokens=5, max_output_tokens=2, max_cost_nano_usd=upper)
                _, value = f.reserve(request)
                self.assertEqual(value["budget"]["reservation_upper_bound"], upper)
                _, claim = f.claim(request)
                outcome = f.fake(request, claim)
                usage = f.usage(input_tokens=3, cached_input_tokens=1, cache_write_tokens=1,
                                output_tokens=2, reasoning_tokens=2, actual_charge_nano_usd=charge)
                f.receipt(f.call("complete", claim["delivery"]["attempt_id"], outcome, usage))
                self.assertEqual((f.account()["held"], f.account()["spent"]), (0, charge))

    def test_concurrent_budget_boundary(self):
        self._require_product()
        if self._outer():
            return
        reg = registry()
        reg["accounts"][BUDGET]["limit_nano_usd"] = 80
        with self.fixture(reg) as f:
            requests = [f.request(task=index + 1) for index in range(16)]
            result = self.race(lambda index: f.call("reserve", requests[index]))
            self.assertTrue(all(kind == "returned" for kind, _ in result))
            receipts = [f.receipt(raw) for _, raw in result]
            self.assertEqual(sum(item["admission"]["status"] == "ADMITTED" for item in receipts), 5)
            self.assertEqual(sum(item["admission"]["reason"] == "BUDGET_EXHAUSTED" for item in receipts), 11)
            self.assertEqual((f.account()["held"], f.account()["spent"]), (80, 0))
            self.assertEqual(sum(row["reservation_held"] for row in f.rows("test_broker_tasks")), 5)
            self.assertEqual(f.rows("test_broker_attempts"), [])
            self.assertEqual(f.fake_records(), [])

    def test_concurrent_duplicate_reservation(self):
        self._require_product()
        if self._outer():
            return
        with self.fixture() as f:
            request = f.request()
            result = self.race(lambda _: f.call("reserve", request))
            self.assertTrue(all(kind == "returned" for kind, _ in result))
            self.assertEqual(len({raw for _, raw in result}), 1)
            f.receipt(result[0][1])
            self.assertEqual((f.account()["held"], f.account()["spent"]), (16, 0))
            self.assertEqual(len(f.rows("test_broker_tasks")), 1)
            self.assertEqual(len(f.rows("test_broker_events")), 1)

    def test_single_dispatch_winner(self):
        self._require_product()
        if self._outer():
            return
        with self.fixture() as f:
            request = f.request()
            f.reserve(request)
            result = self.race(lambda _: f.call("claim_dispatch", request["task_id"], digest(canonical(request))))
            winners = [f.receipt(raw) for kind, raw in result if kind == "returned"]
            self.assertEqual(len(winners), 1)
            self.assertEqual(sum(kind == "refused" for kind, _ in result), 15)
            f.fake(request, winners[0])
            self.assertEqual(len(f.fake_records()), 1)
            self.assertEqual(len(f.rows("test_broker_attempts")), 1)
            self.assertEqual(sum(row["capacity_slot_held"] for row in f.rows("test_broker_attempts")), 1)
            fresh = f.reopen()
            f.refuse("claim_dispatch", request["task_id"], digest(canonical(request)), broker=fresh)
            fresh.close()
            self.assertEqual(len(f.fake_records()), 1)

    def test_identical_replay(self):
        self._require_product()
        if self._outer():
            return
        with self.fixture() as f:
            request = f.request()
            first, _ = f.reserve(request)
            before = f.snapshot()
            for _ in range(3):
                self.assertEqual(f.call("reserve", copy.deepcopy(request)), first)
            self.assertEqual(f.snapshot(), before)
            denied = f.request(task=2, max_cost_nano_usd=15)
            denial, value = f.reserve(denied)
            self.assertEqual(value["admission"]["reason"], "BUDGET_EXHAUSTED")
            before = f.snapshot()
            self.assertEqual(f.call("reserve", denied), denial)
            self.assertEqual(f.call("read", denied["task_id"], digest(canonical(denied))), denial)
            self.assertEqual(f.snapshot(), before)

    def test_input_hash_collision(self):
        self._require_product()
        if self._outer():
            return
        with self.fixture() as f:
            request = f.request()
            f.reserve(request)
            before = f.snapshot()
            changed = dict(request, input_sha256=digest(SECOND_INPUT), context_tokens=12)
            f.refuse("reserve", changed)
            self.assertEqual(f.snapshot(), before)
            self.assertEqual(f.fake_records(), [])

    def test_changed_controls_same_input(self):
        self._require_product()
        if self._outer():
            return
        with self.fixture() as f:
            request = f.request()
            f.reserve(request)
            before = f.snapshot()
            changed = {"context_tokens": 11, "max_output_tokens": 2, "max_cost_nano_usd": 100,
                       "deadline_unix_ms": MAX_INT - 1, "disclosure_scope": "public",
                       "model_id": "TEST-model-2", "policy_sha256": "f" * 64,
                       "price_sha256": "f" * 64, "budget_id": "c" * 32,
                       "account_id": "TEST-other", "provider_id": "TEST-other",
                       "deployment_id": "TEST-deployment-2", "operation": "coding_review"}
            for field, value in changed.items():
                f.refuse("reserve", dict(request, **{field: value}))
                self.assertEqual(f.snapshot(), before)

    def test_restart_accounting(self):
        self._require_product()
        if self._outer():
            return
        with self.fixture() as f:
            request, _, _, _, raw, _ = f.finish()
            pending = f.request(task=2)
            f.reserve(pending)
            self.assertEqual((f.account()["held"], f.account()["spent"]), (16, 13))
            before = f.snapshot()
            f.reg["accounts"][BUDGET]["limit_nano_usd"] = 1
            f.reg["native_context"]["run_id"] = "99999"
            replay, value = f.cli_success("read", {"task_id": request["task_id"], "request_sha256": digest(canonical(request))})
            self.assertEqual(replay, raw)
            self.assertEqual(value["evidence"]["run_id"], "12345")
            self.assertEqual((f.account()["account_limit"], f.account()["held"], f.account()["spent"]), (1024, 16, 13))
            self.assertEqual(f.snapshot(), before)

    def test_crash_before_claim(self):
        self._require_product()
        if self._outer():
            return
        with self.fixture() as f:
            request = f.request()
            f.crash_controller("before-claim", request)
            value = f.read(request)
            self.assertEqual(value["delivery"]["state"], "reserved")
            self.assertEqual(f.rows("test_broker_attempts"), [])
            self.assertEqual(f.fake_records(), [])
            self.assertEqual(f.account()["held"], 16)

    def test_crash_after_claim(self):
        self._require_product()
        if self._outer():
            return
        with self.fixture() as f:
            request = f.request()
            f.crash_controller("after-claim", request)
            value = f.read(request)
            attempt = value["delivery"]["attempt_id"]
            self.assertEqual(value["delivery"]["state"], "dispatching")
            self.assertEqual(f.fake_records(), [])
            f.receipt(f.call("mark_unknown", attempt, "PROCESS_RECOVERY"))
            f.refuse("claim_dispatch", request["task_id"], digest(canonical(request)))
            self.assertEqual((f.account()["held"], f.account()["spent"]), (16, 0))
            self.assertEqual(f.rows("test_broker_attempts")[0]["capacity_slot_held"], 1)

    def test_crash_after_fake_acceptance(self):
        self._require_product()
        if self._outer():
            return
        with self.fixture() as f:
            request = f.request()
            f.crash_controller("after-fake", request)
            value = f.read(request)
            event = f.fake_records()[0]
            self.assertEqual((len(f.fake_records()), event["result_sha256"], event["result_bytes"]),
                             (1, digest(FIXED_RESULT), len(FIXED_RESULT)))
            self.assertEqual(f.rows("test_broker_completions"), [])
            f.receipt(f.call("mark_unknown", value["delivery"]["attempt_id"], "COMPLETION_WRITE_FAILED"))
            f.refuse("claim_dispatch", request["task_id"], digest(canonical(request)))
            self.assertEqual((f.account()["held"], f.account()["spent"]), (16, 0))
            self.assertEqual(len(f.fake_records()), 1)

    def test_confirmed_nondispatch_release(self):
        self._require_product()
        if self._outer():
            return
        with self.fixture() as f:
            request = f.request(deadline_unix_ms=NOW + 1)
            f.reserve(request)
            f.clock[0] = NOW + 1
            raw = f.call("confirm_not_dispatched", request["task_id"], digest(canonical(request)), "EXPIRED")
            value = f.receipt(raw)
            self.assertEqual(value["delivery"]["state"], "confirmed_not_dispatched")
            self.assertEqual((f.account()["held"], f.account()["spent"]), (0, 0))
            task = f.rows("test_broker_tasks")[0]
            self.assertEqual((task["reservation_held"], task["reservation_released"]), (0, 1))
            self.assertEqual(f.fake_records(), [])
            f.refuse("claim_dispatch", request["task_id"], digest(canonical(request)))

    def test_release_after_attempt_refused(self):
        self._require_product()
        if self._outer():
            return
        with self.fixture() as f:
            request = f.request(deadline_unix_ms=NOW + 1)
            f.reserve(request)
            _, claim = f.claim(request)
            f.fake(request, claim)
            f.receipt(f.call("mark_unknown", claim["delivery"]["attempt_id"], "CONNECTION_LOST"))
            f.clock[0] = NOW + 1
            before = f.snapshot()
            f.refuse("confirm_not_dispatched", request["task_id"], digest(canonical(request)), "EXPIRED")
            self.assertEqual(f.snapshot(), before)
            self.assertEqual(f.account()["held"], 16)
            self.assertEqual(len(f.fake_records()), 1)

    def test_unknown_delivery_after_expiry(self):
        self._require_product()
        if self._outer():
            return
        with self.fixture() as f:
            request = f.request(deadline_unix_ms=NOW + 1)
            f.reserve(request)
            _, claim = f.claim(request)
            f.receipt(f.call("mark_unknown", claim["delivery"]["attempt_id"], "TIMEOUT"))
            f.clock[0] = NOW + 100
            fresh = f.reopen()
            value = f.receipt(f.call("read", request["task_id"], digest(canonical(request)), broker=fresh))
            self.assertEqual((value["delivery"]["state"], value["budget"]["held"]), ("unknown_delivery", 16))
            f.refuse("claim_dispatch", request["task_id"], digest(canonical(request)), broker=fresh)
            f.refuse("confirm_not_dispatched", request["task_id"], digest(canonical(request)), "EXPIRED", broker=fresh)
            fresh.close()
            self.assertEqual(f.rows("test_broker_attempts")[0]["capacity_slot_held"], 1)
            self.assertEqual(f.fake_records(), [])

    def test_completed_unknown_billing(self):
        self._require_product()
        if self._outer():
            return
        with self.fixture() as f:
            request, _, _, _, raw, value = f.finish(known=False)
            self.assertEqual((value["delivery"]["state"], value["budget"]["billing_status"]), ("completed", "UNKNOWN"))
            self.assertIsNone(value["budget"]["actual_charge"])
            self.assertEqual((f.account()["held"], f.account()["spent"]), (16, 0))
            self.assertEqual(f.rows("test_broker_accounting"), [])
            self.assertEqual(f.call("read_result", request["task_id"], digest(canonical(request))), FIXED_RESULT)
            self.assertEqual(len(f.fake_records()), 1)
            replay, _ = f.cli_success("read", {"task_id": request["task_id"], "request_sha256": digest(canonical(request))})
            self.assertEqual(replay, raw)

    def test_known_zero_charge(self):
        self._require_product()
        if self._outer():
            return
        reg = registry()
        replace_price(reg, input_rate_per_million=0, cached_input_rate_per_million=0,
                      cache_write_rate_per_million=0, output_rate_per_million=0, fixed_request_charge=0)
        reg["accounts"][BUDGET]["limit_nano_usd"] = 0
        with self.fixture(reg) as f:
            request = f.request(max_cost_nano_usd=0)
            _, reserved = f.reserve(request)
            self.assertEqual(reserved["budget"]["reservation_upper_bound"], 0)
            _, claim = f.claim(request)
            outcome = f.fake(request, claim)
            observed = f.usage(input_tokens=0, cached_input_tokens=0, cache_write_tokens=0,
                               output_tokens=0, reasoning_tokens=0, actual_charge_nano_usd=0)
            value = f.receipt(f.call("complete", claim["delivery"]["attempt_id"], outcome, observed))
            self.assertEqual((value["budget"]["billing_status"], value["budget"]["actual_charge"]), ("KNOWN", 0))
            self.assertEqual(json.loads(f.rows("test_broker_accounting")[0]["canonical_usage"]), observed)
            self.assertEqual((f.account()["held"], f.account()["spent"]), (0, 0))
            self.assertEqual(len(f.fake_records()), 1)

    def test_over_reservation_charge(self):
        self._require_product()
        if self._outer():
            return
        with self.fixture() as f:
            request = f.request()
            f.reserve(request)
            _, claim = f.claim(request)
            outcome = f.fake(request, claim)
            observed = f.usage(actual_charge_nano_usd=17)
            value = f.receipt(f.call("complete", claim["delivery"]["attempt_id"], outcome, observed))
            self.assertEqual((value["budget"]["actual_charge"], value["budget"]["invariant_status"]), (17, "VIOLATED"))
            self.assertEqual((f.account()["held"], f.account()["spent"], f.account()["admission_status"]), (0, 17, "FROZEN"))
            charge = f.rows("test_broker_accounting")[0]
            self.assertEqual(charge["actual_charge"], 17)
            event = [row for row in f.rows("test_broker_events") if row["kind"] == "ACCOUNTING_APPLIED"][0]
            self.assertEqual((event["charge_nano_usd_decimal"], event["aggregate_spent_nano_usd_decimal"]), ("17", "17"))
            _, denial = f.reserve(f.request(task=2))
            self.assertEqual(denial["admission"]["reason"], "UNKNOWN_ACCOUNT")
            self.assertEqual(f.account()["spent"], 17)
            self.assertEqual(len(f.fake_records()), 1)

    def test_invalid_usage_types(self):
        self._require_product()
        if self._outer():
            return
        with self.fixture() as f:
            request = f.request()
            f.reserve(request)
            _, claim = f.claim(request)
            outcome = f.fake(request, claim)
            attempt = claim["delivery"]["attempt_id"]
            before = f.snapshot()
            for field in f.usage():
                for value in (True, False, None, "0", 0.5, -1):
                    f.refuse("complete", attempt, outcome, f.usage(**{field: value}))
                    self.assertEqual(f.snapshot(), before)
                missing = f.usage()
                del missing[field]
                f.refuse("complete", attempt, outcome, missing)
            f.refuse("complete", attempt, outcome, f.usage(cached_input_tokens=7, cache_write_tokens=1))
            f.refuse("complete", attempt, outcome, f.usage(reasoning_tokens=3))
            f.refuse("complete", attempt, outcome, {})
            self.assertEqual(f.snapshot(), before)
            f.receipt(f.call("complete", attempt, outcome, f.usage()))
            self.assertEqual(f.account()["spent"], 13)

    def test_unknown_admission_records(self):
        self._require_product()
        if self._outer():
            return
        for category, reason in (("account", "UNKNOWN_ACCOUNT"), ("deployment", "UNKNOWN_CAPABILITY"),
                                 ("capacity", "CAPACITY_UNAVAILABLE"), ("price", "UNKNOWN_PRICE")):
            reg = registry()
            if category == "account":
                reg["accounts"][BUDGET]["status"] = "UNKNOWN"
            elif category == "deployment":
                reg["deployments"]["TEST-deployment-1"]["status"] = "UNKNOWN"
            elif category == "capacity":
                reg["deployments"]["TEST-deployment-1"]["capacity_status"] = "UNKNOWN"
            else:
                replace_price(reg, cached_input_rate_per_million=None)
            with self.fixture(reg) as f:
                _, value = f.reserve()
                self.assertEqual((value["admission"]["status"], value["admission"]["reason"]), ("DENIED", reason))
                self.assertEqual((f.account()["held"], f.account()["spent"]), (0, 0))
                self.assertEqual(f.rows("test_broker_attempts"), [])
                self.assertEqual(f.fake_records(), [])
        with self.fixture() as f:
            before = f.snapshot()
            for field in ("policy_sha256", "price_sha256", "budget_id"):
                f.refuse("reserve", f.request(**{field: "f" * (32 if field == "budget_id" else 64)}))
                self.assertEqual(f.snapshot(), before)

    def test_route_model_substitution(self):
        self._require_product()
        if self._outer():
            return
        with self.fixture() as f:
            before = f.snapshot()
            for field in ("account_id", "provider_id", "deployment_id", "model_id"):
                f.refuse("reserve", f.request(**{field: "UNBOUND-" + PRIVATE_MARKER.decode("ascii")}))
                self.assertEqual(f.snapshot(), before)
            for index, operation in enumerate(("navigation", "coding_review", "mathematical_audit"), 1):
                request = f.request(task=index, operation=operation)
                _, value = f.reserve(request)
                self.assertEqual(value["admission"]["tier"], index)
                self.assertEqual(value["request"]["model_id"], "TEST-model-%d" % index)
            self.assertEqual(f.fake_records(), [])

    def test_disclosure_without_hash_admission(self):
        self._require_product()
        if self._outer():
            return
        reg = registry()
        reg["disclosure_admissions"][digest(INPUT)] = ["local_private"]
        with self.fixture(reg) as f:
            request = f.request(disclosure_scope="public")
            _, value = f.reserve(request)
            self.assertEqual(value["admission"]["reason"], "DISCLOSURE_DENIED")
            self.assertEqual((f.account()["held"], f.account()["spent"]), (0, 0))
            before = f.snapshot()
            f.refuse("reserve", f.request(task=2, input_sha256=digest(PRIVATE_MARKER)))
            self.assertEqual(f.snapshot(), before)
            _, admitted = f.reserve(f.request(task=3, disclosure_scope="local_private"))
            self.assertEqual(admitted["admission"]["status"], "ADMITTED")
            self.assertEqual(f.fake_records(), [])

    def test_expired_revoked_admission(self):
        self._require_product()
        if self._outer():
            return
        with self.fixture() as f:
            _, expired = f.reserve(f.request(deadline_unix_ms=NOW))
            self.assertEqual(expired["admission"]["reason"], "EXPIRED")
            request = f.request(task=2, deadline_unix_ms=NOW + 1)
            f.reserve(request)
            before = f.snapshot()
            f.clock[0] = NOW + 1
            f.refuse("claim_dispatch", request["task_id"], digest(canonical(request)))
            self.assertEqual(f.snapshot(), before)
            self.assertEqual(f.account()["held"], 16)
        with self.fixture() as f:
            request = f.request()
            f.reserve(request)
            f.reg["accounts"][BUDGET]["status"] = "DISABLED"
            fresh = f.reopen()
            before = f.snapshot()
            f.refuse("claim_dispatch", request["task_id"], digest(canonical(request)), broker=fresh)
            self.assertEqual(f.snapshot(), before)
            self.assertEqual(f.fake_records(), [])
            fresh.close()

        def locked_time_change(f, method, arguments):
            # Only the external controller owns the clock and competing writer.
            # A post-release clock sample independently probes the actual SQL
            # write lock; an expiry refusal from SQLITE_BUSY gets no credit.
            samples = []
            sample_lock = threading.Lock()
            sampled = threading.Event()
            ready = threading.Event()
            released = threading.Event()

            def probe_writer():
                probe = sqlite3.connect((f.ledger / "routing-broker.sqlite3").as_uri() + "?mode=rw",
                                        uri=True, timeout=0, isolation_level=None)
                try:
                    try:
                        probe.execute("BEGIN IMMEDIATE")
                    except sqlite3.OperationalError as error:
                        code = getattr(error, "sqlite_errorcode", None)
                        return ("busy" if code == sqlite3.SQLITE_BUSY else "error", code)
                    else:
                        probe.rollback()
                        return ("available", None)
                finally:
                    probe.close()

            def observed_clock():
                with sample_lock:
                    value = f.clock[0]
                    after_release = released.is_set()
                # Before release, the controller owns the observed lock. After
                # release, a busy probe establishes the broker's BEGIN succeeded.
                observation = probe_writer() if after_release else None
                with sample_lock:
                    samples.append((value, after_release, observation))
                    sampled.set()
                return value

            _guarded(f.broker.close)
            f.broker = _guarded(f.api.Broker, f.ledger, f.reg, clock=observed_clock)
            writer = sqlite3.connect((f.ledger / "routing-broker.sqlite3").as_uri() + "?mode=rw",
                                     uri=True, timeout=0, isolation_level=None)
            try:
                writer.execute("BEGIN IMMEDIATE")
                locked_at = time.monotonic()
                self.assertEqual(probe_writer(), ("busy", sqlite3.SQLITE_BUSY))

                def contender():
                    ready.set()
                    try:
                        return ("returned", f.call(method, *arguments))
                    except f.api.RoutingBrokerRefused:
                        return ("refused", None)

                with ThreadPoolExecutor(max_workers=1) as pool:
                    pending = pool.submit(contender)
                    try:
                        self.assertTrue(ready.wait(timeout=0.25), "TEST contender must start within its cap")
                        # Old code samples before BEGIN; corrected code samples
                        # only after acquisition. Neither path needs an unbounded
                        # handshake or a product failpoint to release this lock.
                        sampled.wait(timeout=0.1)
                        with sample_lock:
                            f.clock[0] = NOW + 2
                            writer.rollback()
                            released.set()
                        self.assertLess(time.monotonic() - locked_at, 0.75,
                                        "TEST writer must release before the 1000 ms busy timeout")
                    finally:
                        writer.rollback()
                        released.set()
                    outcome = pending.result(timeout=3)
            finally:
                writer.close()
            return outcome, samples

        for boundary in ("deadline", "policy", "price"):
            for method in ("reserve", "claim_dispatch", "confirm_not_dispatched"):
                for valid_after_release in (False, True):
                    with self.subTest(lock_expiry=boundary, transition=method,
                                      valid_after_release=valid_after_release):
                        reg = registry()
                        end = NOW + (3 if valid_after_release else 1)
                        if boundary == "policy":
                            reg["policy"]["valid_until_unix_ms"] = end
                        elif boundary == "price":
                            replace_price(reg, valid_until_unix_ms=end)
                        with self.fixture(reg) as f:
                            changes = {"deadline_unix_ms": end} if boundary == "deadline" else {}
                            request = f.request(**changes)
                            identity = (request["task_id"], digest(canonical(request)))
                            if method != "reserve":
                                f.reserve(request)
                            before = f.snapshot()
                            arguments = ((request,) if method == "reserve" else
                                         (*identity, "EXPIRED") if method == "confirm_not_dispatched" else identity)
                            (kind, raw), samples = locked_time_change(f, method, arguments)

                            if method == "reserve":
                                self.assertEqual(kind, "returned")
                                receipt = f.receipt(raw)
                                if valid_after_release:
                                    self.assertEqual(receipt["admission"]["status"], "ADMITTED")
                                    self.assertEqual(receipt["delivery"]["state"], "reserved")
                                    self.assertEqual((f.account()["held"], f.account()["spent"]), (16, 0))
                                else:
                                    self.assertEqual(receipt["admission"]["status"], "DENIED")
                                    self.assertEqual(receipt["admission"]["reason"], "EXPIRED")
                                    self.assertIsNone(receipt["delivery"]["state"])
                                    self.assertEqual((f.account()["held"], f.account()["spent"]), (0, 0))
                                tasks = f.rows("test_broker_tasks")
                                self.assertEqual(len(tasks), 1)
                                self.assertEqual(tasks[0]["canonical_request"], canonical(request))
                                self.assertEqual(tasks[0]["latest_receipt"], raw)
                                self.assertEqual(tasks[0]["state"], "reserved" if valid_after_release else None)
                                self.assertEqual(tasks[0]["reservation_held"], int(valid_after_release))
                                self.assertEqual([row["kind"] for row in f.rows("test_broker_events")],
                                                 ["RESERVED" if valid_after_release else "RESERVATION_DENIED"])
                                self.assertEqual(f.rows("test_broker_attempts"), [])
                            elif method == "claim_dispatch":
                                if valid_after_release:
                                    self.assertEqual(kind, "returned")
                                    receipt = f.receipt(raw)
                                    self.assertEqual(receipt["delivery"]["state"], "dispatching")
                                    attempts = f.rows("test_broker_attempts")
                                    self.assertEqual(len(attempts), 1)
                                    self.assertEqual(attempts[0]["attempt_id"], receipt["delivery"]["attempt_id"])
                                    self.assertEqual(attempts[0]["capacity_slot_held"], 1)
                                    self.assertEqual(f.rows("test_broker_tasks")[0]["latest_receipt"], raw)
                                    self.assertEqual([row["kind"] for row in f.rows("test_broker_events")],
                                                     ["RESERVED", "DISPATCH_CLAIMED"])
                                else:
                                    self.assertEqual(kind, "refused")
                                    self.assertEqual(f.snapshot(), before)
                                    self.assertEqual(f.rows("test_broker_attempts"), [])
                                self.assertEqual((f.account()["held"], f.account()["spent"]), (16, 0))
                            else:
                                if valid_after_release:
                                    # EXPIRED cannot release a still-valid request.
                                    self.assertEqual(kind, "refused")
                                    self.assertEqual(f.snapshot(), before)
                                    self.assertEqual((f.account()["held"], f.account()["spent"]), (16, 0))
                                else:
                                    self.assertEqual(kind, "returned")
                                    receipt = f.receipt(raw)
                                    self.assertEqual(receipt["delivery"]["state"], "confirmed_not_dispatched")
                                    self.assertEqual((f.account()["held"], f.account()["spent"]), (0, 0))
                                    task = f.rows("test_broker_tasks")[0]
                                    self.assertEqual((task["reservation_held"], task["reservation_released"]), (0, 1))
                                    self.assertEqual(task["latest_receipt"], raw)
                                    self.assertEqual([row["kind"] for row in f.rows("test_broker_events")],
                                                     ["RESERVED", "NONDISPATCH_CONFIRMED"])
                                self.assertEqual(f.rows("test_broker_attempts"), [])
                            self.assertEqual(f.fake_records(), [])
                            acquired = [sample for sample in samples if sample[1]]
                            self.assertTrue(acquired, "TEST must observe a post-release clock sample")
                            self.assertTrue(all(sample == (NOW + 2, True, ("busy", sqlite3.SQLITE_BUSY))
                                                for sample in acquired),
                                            "TEST must independently observe the broker's successful BEGIN after release")

    def test_unbounded_billable_components(self):
        self._require_product()
        if self._outer():
            return
        for field in ("input_rate_per_million", "cached_input_rate_per_million",
                      "cache_write_rate_per_million", "output_rate_per_million", "fixed_request_charge"):
            reg = registry()
            replace_price(reg, **{field: None})
            with self.fixture(reg) as f:
                _, denied = f.reserve()
                self.assertEqual(denied["admission"]["reason"], "UNKNOWN_PRICE")
                self.assertEqual((f.account()["held"], f.account()["spent"]), (0, 0))
                self.assertEqual(f.fake_records(), [])
        api = _load_api()
        for field in ("reasoning_included_in_output", "tools_disabled"):
            reg = registry()
            replace_price(reg, **{field: False})
            with self.assertRaises(api.RoutingBrokerRefused):
                self.fixture(reg)

    def test_invalid_integer_budget(self):
        self._require_product()
        if self._outer():
            return
        with self.fixture() as f:
            request = f.request()
            f.reserve(request)
            before = f.snapshot()
            for field in ("max_cost_nano_usd", "deadline_unix_ms", "context_tokens", "max_output_tokens"):
                for value in (-1, True, False, 1.5, "1", None, MAX_INT + 1):
                    f.refuse("reserve", f.request(task=2, **{field: value}))
                    self.assertEqual(f.snapshot(), before)
            for field in ("deadline_unix_ms", "context_tokens", "max_output_tokens"):
                f.refuse("reserve", f.request(task=2, **{field: 0}))
                self.assertEqual(f.snapshot(), before)
        api = _load_api()
        for value in (-1, True, "1", 0.5, MAX_INT + 1):
            reg = registry()
            reg["accounts"][BUDGET]["limit_nano_usd"] = value
            with self.assertRaises(api.RoutingBrokerRefused):
                self.fixture(reg)

    def test_unavailable_database(self):
        self._require_product()
        if self._outer():
            return
        with self.fixture() as f:
            request = f.request()
            _guarded(f.broker.close)
            f.broker = None
            f.broker = f.reopen()
            _, admitted = f.reserve(request)
            self.assertEqual(admitted["admission"]["status"], "ADMITTED")
            self.assertEqual((f.account()["held"], f.account()["spent"]), (16, 0))
            self.assertEqual(f.fake_records(), [])
        with self.fixture() as f:
            request = f.request()
            before = f.snapshot()
            writer = sqlite3.connect(f.ledger / "routing-broker.sqlite3", timeout=0)
            try:
                writer.execute("BEGIN IMMEDIATE")
                f.refuse("reserve", request)
            finally:
                writer.rollback()
                writer.close()
            self.assertEqual(f.snapshot(), before)
            self.assertEqual(f.fake_records(), [])
        for fault in ("corrupt", "version", "sidecar"):
            with self.fixture() as f:
                database = f.ledger / "routing-broker.sqlite3"
                _guarded(f.broker.close)
                f.broker = None
                if fault == "corrupt":
                    database.write_bytes(b"TEST deliberately corrupt SQLite\n")
                elif fault == "version":
                    connection = sqlite3.connect(database)
                    connection.execute("PRAGMA user_version=2")
                    connection.close()
                else:
                    (f.ledger / "routing-broker.sqlite3-wal").write_bytes(b"TEST unadmitted WAL\n")
                f.refuse_fresh_reserve(f.request())
                self.assertEqual(f.fake_records(), [])

    def _storage_pressure(self, function):
        # Transient public zero padding fills only the dedicated <=64 MiB mount.
        # It is removed after the one refusal, never retained as fixture evidence.
        padding = _CHILD_BASE / "TEST-transient-capacity-pressure"
        descriptor = os.open(padding, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        total = 0
        exhausted = False
        try:
            while total <= 64 * MIB:
                try:
                    amount = os.write(descriptor, b"\0" * 65536)
                    total += amount
                except OSError as error:
                    self.assertEqual(error.errno, errno.ENOSPC)
                    exhausted = True
                    break
            self.assertTrue(exhausted, "dedicated fixture mount must enforce its physical capacity")
            function()
        finally:
            os.close(descriptor)
            padding.unlink()

    def test_storage_commit_failure(self):
        self._require_product()
        if self._outer():
            return
        with self.fixture() as f:
            request = f.request()
            before = f.snapshot()
            self._storage_pressure(lambda: f.refuse("reserve", request))
            self.assertEqual(f.snapshot(), before)
            self.assertEqual(f.fake_records(), [])
        with self.fixture() as f:
            request, claim, _, _, _, _ = f.finish(known=False)
            usage = f.usage()
            attempt = claim["delivery"]["attempt_id"]
            evidence = f.evidence(request, attempt, usage)
            f.admit(evidence)
            before = f.snapshot()
            self._storage_pressure(lambda: f.refuse("settle", attempt, ident(50), usage, evidence))
            self.assertEqual(f.snapshot(), before)
            self.assertEqual((f.account()["held"], f.account()["spent"]), (16, 0))
            self.assertEqual(len(f.fake_records()), 1)

    def test_unsafe_ledger_namespace(self):
        self._require_product()
        if self._outer():
            return
        with self.fixture() as f:
            request = f.request()
            _guarded(f.broker.close)
            f.broker = None
            f.broker = f.reopen()
            _, admitted = f.reserve(request)
            self.assertEqual(admitted["admission"]["status"], "ADMITTED")
            self.assertEqual((f.account()["held"], f.account()["spent"]), (16, 0))
            self.assertEqual(f.fake_records(), [])
        for fault in ("symlink-root", "hardlink-db", "wrong-mode", "extra", "fifo-db"):
            with self.fixture() as f:
                database = f.ledger / "routing-broker.sqlite3"
                _guarded(f.broker.close)
                f.broker = None
                if fault == "symlink-root":
                    backup = f.base / "TEST-backup-ledger"
                    f.ledger.rename(backup)
                    f.ledger.symlink_to(backup, target_is_directory=True)
                elif fault == "hardlink-db":
                    os.link(database, f.base / "TEST-second-link")
                elif fault == "wrong-mode":
                    database.chmod(0o644)
                elif fault == "extra":
                    (f.ledger / "TEST-foreign-entry").write_bytes(b"TEST foreign entry\n")
                else:
                    database.unlink()
                    os.mkfifo(database, mode=0o600)
                f.refuse_fresh_reserve(f.request())
                self.assertEqual(f.fake_records(), [])
        with self.fixture() as f:
            self.assertNotEqual(FIXTURE_MOUNT.stat().st_uid, os.geteuid())
            with self.assertRaises(f.api.RoutingBrokerRefused):
                _guarded(f.api.Broker, FIXTURE_MOUNT, f.reg, clock=lambda: NOW)

    def test_strict_input_and_cli(self):
        self._require_product()
        if self._outer():
            return
        with self.fixture() as f:
            request = f.request()
            first, value = f.cli_success("reserve", {"request": request})
            self.assertEqual(value["admission"]["status"], "ADMITTED")
            before = f.snapshot()
            flags = ["--test-ledger-root", str(f.ledger), "--test-registry", str(f.registry_file), "--action", "reserve"]
            for index in (0, 2, 4):
                f.cli_refusal("reserve", {"request": request}, arguments=flags[:index] + flags[index + 2:])
                f.cli_refusal("reserve", {"request": request}, arguments=flags + flags[index:index + 2])
                changed = list(flags)
                changed[index] = changed[index][:-1]
                f.cli_refusal("reserve", {"request": request}, arguments=changed)
            for extra in (["--help"], ["--live"], ["--endpoint", PRIVATE_MARKER.decode("ascii")],
                          ["--clock", "1"], ["--action", "read-result"]):
                f.cli_refusal("reserve", {"request": request}, arguments=flags + extra)
            body = canonical({"request": request})
            invalid = [b"", b"\xff", b"{}", b"[]", b"null", body + b"{}",
                       b'{"request":NaN}', b'{"request":1e309}', b'{"request":"\\ud800"}',
                       b'{"request":' + b"[" * 17 + b"0" + b"]" * 17 + b"}",
                       b'{"request":' + canonical(request).rstrip() + b',"request":' + canonical(request).rstrip() + b"}",
                       canonical({"request": request, "private": PRIVATE_MARKER.decode("ascii")}),
                       body + b" " * (64 * 1024 + 1 - len(body))]
            for raw in invalid:
                f.cli_refusal("reserve", {}, raw=raw)
            duplicate = body.replace(b'"task_id":', b'"task_id":"' + ident(99).encode() + b'","task_id":', 1)
            f.cli_refusal("reserve", {}, raw=duplicate)
            for field in request:
                changed = dict(request)
                del changed[field]
                f.cli_refusal("reserve", {"request": changed})
            self.assertEqual(f.snapshot(), before)
            replay, _ = f.cli_success("reserve", {"request": request})
            self.assertEqual(replay, first)
            denied = f.request(task=2, max_cost_nano_usd=15)
            _, denial = f.cli_success("reserve", {"request": denied})
            self.assertEqual(denial["admission"]["reason"], "BUDGET_EXHAUSTED")
            self.assertEqual(f.account()["held"], 16)
            def closed_packet(action, packet):
                snapshot = f.snapshot()
                f.cli_refusal(action, dict(packet, caller_supplied=PRIVATE_MARKER.decode("ascii")))
                for field in packet:
                    missing = dict(packet)
                    del missing[field]
                    f.cli_refusal(action, missing)
                self.assertEqual(f.snapshot(), snapshot)
            identity = {"task_id": request["task_id"], "request_sha256": digest(canonical(request))}
            closed_packet("claim-dispatch", identity)
            _, claim = f.cli_success("claim-dispatch", identity)
            outcome = f.fake(request, claim)
            attempt = claim["delivery"]["attempt_id"]
            uncertain = {"attempt_id": attempt, "reason": "COMPLETION_WRITE_FAILED"}
            closed_packet("mark-unknown", uncertain)
            f.cli_success("mark-unknown", uncertain)
            completion = {"attempt_id": attempt, "outcome": outcome, "usage": f.usage()}
            closed_packet("complete", completion)
            f.cli_success("complete", completion)
            evidence = f.evidence(request, attempt, f.usage())
            f.admit(evidence)
            settlement = {"attempt_id": attempt, "settlement_id": ident(100),
                          "usage": f.usage(), "accounting_evidence": evidence}
            closed_packet("settle", settlement)
            f.cli_refusal("settle", dict(settlement, accounting_record_admissions={evidence["accounting_record_sha256"]: evidence}))
            f.cli_success("settle", settlement)
            closed_packet("read", identity)
            f.cli_success("read", identity)
            third = f.request(task=3)
            f.reserve(third)
            f.reg["accounts"][BUDGET]["status"] = "DISABLED"
            nondispatch = {"task_id": third["task_id"], "request_sha256": digest(canonical(third)),
                          "reason": "UNKNOWN_ACCOUNT"}
            closed_packet("confirm-not-dispatched", nondispatch)
            _, terminal = f.cli_success("confirm-not-dispatched", nondispatch)
            self.assertEqual(terminal["delivery"]["state"], "confirmed_not_dispatched")
            self.assertEqual((f.account()["held"], f.account()["spent"]), (0, 13))
            self.assertEqual(len(f.fake_records()), 1)

    def test_private_result_retention(self):
        self._require_product()
        if self._outer():
            return
        reg = registry()
        private_input = digest(PRIVATE_MARKER)
        reg["disclosure_admissions"][private_input] = ["local_private"]
        reg["input_token_admissions"][private_input] = copy.deepcopy(reg["input_token_admissions"][digest(INPUT)])
        for values in reg["input_token_admissions"][private_input].values():
            values[TOKENIZER]["payload_bytes"] = len(PRIVATE_MARKER)
            values[TOKENIZER]["count_evidence_sha256"] = digest(b"TEST private count observation\n")
        with self.fixture(reg) as f:
            request = f.request(payload=PRIVATE_MARKER, disclosure_scope="local_private", context_tokens=10)
            _, _, _, _, raw, _ = f.finish(request=request)
            self.assertNotIn(PRIVATE_MARKER, raw)
            self.assertNotIn(FIXED_RESULT.rstrip(b"\n"), raw)
            self.assertEqual(f.call("read_result", request["task_id"], digest(canonical(request))), FIXED_RESULT)
            f.refuse("read_result", request["task_id"], "f" * 64)
            f.cli_refusal("read-result", {"task_id": request["task_id"], "request_sha256": digest(canonical(request))})
            row = f.rows("test_broker_results")[0]
            self.assertEqual((row["result_blob"], row["result_bytes"], row["result_sha256"]),
                             (FIXED_RESULT, len(FIXED_RESULT), digest(FIXED_RESULT)))
            before = f.snapshot()
            for bad in (dict(request, prompt=PRIVATE_MARKER.decode()), dict(request, credential=PRIVATE_MARKER.decode())):
                f.cli_refusal("reserve", {"request": bad})
                self.assertEqual(f.snapshot(), before)

    def test_zero_external_activity(self):
        self._require_product()
        if self._outer():
            return
        with self.fixture() as calibration:
            request = calibration.request()
            calibration.reserve(request)
            _, claim = calibration.claim(request)
            calibration.fake(request, claim)
            self.assertEqual(len(calibration.fake_records()), 1)
            probes = [lambda: _guarded(open, calibration.fake_file, "rb"),
                      lambda: _guarded(urllib.request.urlopen, "http://127.0.0.1:9/TEST"),
                      lambda: _guarded(subprocess.Popen, [sys.executable, "-c", "pass"]),
                      lambda: _guarded(__import__, "litellm")]
            for label, probe in zip(("fake_io", "network", "process", "sdk"), probes):
                with self.assertRaises(RuntimeError):
                    probe()
                self.assertGreaterEqual(_COUNTS[label], 1)
                _CALIBRATION[label] = _COUNTS[label]
                _COUNTS.update(dict.fromkeys(_COUNTS, 0))
            self.assertTrue(all(number >= 1 for number in _CALIBRATION.values()))
        with self.fixture() as blocked:
            request = blocked.request(max_cost_nano_usd=15)
            _, denial = blocked.reserve(request)
            self.assertEqual(denial["admission"]["status"], "DENIED")
            blocked.refuse("claim_dispatch", request["task_id"], digest(canonical(request)))
            self.assertEqual(blocked.fake_records(), [])
            self.assertEqual(_COUNTS, dict.fromkeys(_COUNTS, 0))
        with self.fixture() as f:
            f.finish()
            self.assertEqual(len(f.fake_records()), 1)
            self.assertEqual(_COUNTS, dict.fromkeys(_COUNTS, 0))

    def test_replay_context_and_optimization(self):
        self._require_product()
        if self._outer():
            return
        self.assertIn(sys.flags.optimize, (0, 1, 2))
        reg = registry()
        reg["native_context"]["run_attempt"] = "10"
        with self.fixture(reg) as f:
            request, claim, outcome, usage, raw, value = f.finish()
            original_context = f.rows("test_broker_tasks")[0]["canonical_native_context"]
            f.reg["native_context"] = {"repository": "TEST-new/TEST-new", "checked_commit": "b" * 40,
                                        "run_id": "99999", "run_attempt": "2"}
            replay, observed = f.cli_success("complete", {"attempt_id": claim["delivery"]["attempt_id"],
                                                          "outcome": outcome, "usage": usage})
            self.assertEqual(replay, raw)
            self.assertEqual(observed["evidence"]["run_attempt"], "10")
            self.assertEqual(observed["evidence"]["repository"], "TEST-owner/TEST-repository")
            self.assertEqual(f.rows("test_broker_tasks")[0]["canonical_native_context"], original_context)
            self.assertEqual(f.account()["spent"], 13)
            self.assertEqual(len(f.fake_records()), 1)

    def test_trusted_input_token_counts(self):
        self._require_product()
        if self._outer():
            return
        with self.fixture() as f:
            first = f.request()
            second = f.request(task=2, payload=SECOND_INPUT)
            _, seven = f.reserve(first)
            _, nine = f.reserve(second)
            self.assertEqual((seven["budget"]["reservation_upper_bound"], nine["budget"]["reservation_upper_bound"]), (16, 18))
            self.assertEqual(f.account()["held"], 34)
            self.assertNotEqual(seven["evidence"]["input_token_admission_sha256"], nine["evidence"]["input_token_admission_sha256"])
            _, too_small = f.reserve(f.request(task=3, context_tokens=9))
            self.assertEqual(too_small["admission"]["reason"], "TOKEN_ADMISSION_MISMATCH")
            self.assertEqual(f.account()["held"], 34)
            self.assertEqual(f.rows("test_broker_attempts"), [])
        with self.fixture() as f:
            original = copy.deepcopy(f.reg["input_token_admissions"][digest(INPUT)]["TEST-deployment-1"][TOKENIZER])
            f.reg["input_token_admissions"][digest(INPUT)]["TEST-deployment-1"][TOKENIZER]["input_tokens"] = 100
            _, value = f.reserve()
            self.assertEqual(value["budget"]["reservation_upper_bound"], 16)
            self.assertEqual(f.rows("test_broker_tasks")[0]["canonical_token_admission"], canonical(original))

    def test_untrusted_or_mismatched_token_admission(self):
        self._require_product()
        if self._outer():
            return
        with self.fixture() as f:
            before = f.snapshot()
            for changes in ({"input_tokens": 7}, {"count_evidence_sha256": "f" * 64}, {"payload_bytes": len(INPUT)}):
                f.refuse("reserve", f.request(**changes))
                self.assertEqual(f.snapshot(), before)
            missing = copy.deepcopy(f.reg)
            del missing["input_token_admissions"][digest(INPUT)]["TEST-deployment-1"]
            fresh = f.reopen(missing)
            f.refuse("reserve", f.request(), broker=fresh)
            fresh.close()
            self.assertEqual(f.snapshot(), before)
            wrong = copy.deepcopy(f.reg)
            wrong["deployments"]["TEST-deployment-1"]["tokenizer_sha256"] = "f" * 64
            with self.assertRaises(f.api.RoutingBrokerRefused):
                f.reopen(wrong)
            self.assertEqual(f.snapshot(), before)
            request = f.request()
            f.reserve(request)
            changed = copy.deepcopy(f.reg)
            changed["input_token_admissions"][digest(INPUT)]["TEST-deployment-1"][TOKENIZER]["input_tokens"] = 8
            fresh = f.reopen(changed)
            before = f.snapshot()
            f.refuse("claim_dispatch", request["task_id"], digest(canonical(request)), broker=fresh)
            fresh.close()
            self.assertEqual(f.snapshot(), before)
            self.assertEqual((f.account()["held"], f.account()["spent"]), (16, 0))
            self.assertEqual(f.fake_records(), [])

    def test_cache_write_rate_upper_bound(self):
        self._require_product()
        if self._outer():
            return
        # N=7/M=3; partition 4 ordinary + 2 cached + 1 write; output=2 inclusive.
        for cached, write, upper, charge in ((3000000, 4000000, 44, 29),
                                           (5000000, 1000000, 51, 30)):
            reg = registry()
            replace_price(reg, input_rate_per_million=2000000, cached_input_rate_per_million=cached,
                          cache_write_rate_per_million=write, output_rate_per_million=5000000,
                          fixed_request_charge=1)
            with self.fixture(reg) as f:
                request = f.request(max_cost_nano_usd=upper)
                _, value = f.reserve(request)
                self.assertEqual(value["budget"]["reservation_upper_bound"], upper)
                _, claim = f.claim(request)
                outcome = f.fake(request, claim)
                usage = f.usage(reasoning_tokens=2, actual_charge_nano_usd=charge)
                f.receipt(f.call("complete", claim["delivery"]["attempt_id"], outcome, usage))
                self.assertEqual((f.account()["held"], f.account()["spent"]), (0, charge))
                self.assertEqual(f.rows("test_broker_accounting")[0]["actual_charge"], charge)
        reg = registry()
        replace_price(reg, input_rate_per_million=7, cached_input_rate_per_million=3,
                      cache_write_rate_per_million=11, output_rate_per_million=13, fixed_request_charge=0)
        reg["input_token_admissions"][digest(INPUT)]["TEST-deployment-1"][TOKENIZER]["input_tokens"] = 3
        with self.fixture(reg) as f:
            request = f.request(max_output_tokens=2, context_tokens=5, max_cost_nano_usd=2)
            _, value = f.reserve(request)
            self.assertEqual(value["budget"]["reservation_upper_bound"], 2)
            _, claim = f.claim(request)
            outcome = f.fake(request, claim)
            usage = f.usage(input_tokens=3, cached_input_tokens=1, cache_write_tokens=1,
                            output_tokens=1, reasoning_tokens=1, actual_charge_nano_usd=2)
            f.receipt(f.call("complete", claim["delivery"]["attempt_id"], outcome, usage))
            self.assertEqual(f.account()["spent"], 2)  # Separate input rounding would charge 4.

    def test_completion_replay_and_conflict(self):
        self._require_product()
        if self._outer():
            return
        with self.fixture() as f:
            request = f.request()
            f.reserve(request)
            _, claim = f.claim(request)
            outcome = f.fake(request, claim)
            usage = f.usage()
            attempt = claim["delivery"]["attempt_id"]
            result = self.race(lambda _: f.call("complete", attempt, outcome, usage))
            self.assertTrue(all(kind == "returned" for kind, _ in result))
            self.assertEqual(len({raw for _, raw in result}), 1)
            original = result[0][1]
            f.receipt(original)
            self.assertEqual((f.account()["held"], f.account()["spent"]), (0, 13))
            self.assertEqual(len(f.rows("test_broker_completions")), 1)
            self.assertEqual(len(f.rows("test_broker_accounting")), 1)
            before = f.snapshot()
            for changes in ({"result_utf8": "TEST changed result"},
                            {"native_provider_request_id": "TEST-different-request"},
                            {"status": "ERROR", "result_utf8": None, "error_code": "OTHER"}):
                f.refuse("complete", attempt, dict(outcome, **changes), usage)
                self.assertEqual(f.snapshot(), before)
            f.refuse("complete", attempt, outcome, f.usage(actual_charge_nano_usd=14))
            self.assertEqual(f.snapshot(), before)
            replay, _ = f.cli_success("complete", {"attempt_id": attempt, "outcome": outcome, "usage": usage})
            self.assertEqual(replay, original)
            self.assertEqual(f.snapshot(), before)
            self.assertEqual(len(f.fake_records()), 1)

    def test_concurrent_settlement_once(self):
        self._require_product()
        if self._outer():
            return
        with self.fixture() as f:
            request, claim, _, _, _, _ = f.finish(known=False)
            attempt = claim["delivery"]["attempt_id"]
            usage = f.usage()
            evidence = f.evidence(request, attempt, usage)
            f.admit(evidence)
            result = self.race(lambda _: f.call("settle", attempt, ident(100), usage, evidence))
            self.assertTrue(all(kind == "returned" for kind, _ in result))
            self.assertEqual(len({raw for _, raw in result}), 1)
            f.receipt(result[0][1])
            self.assertEqual((f.account()["held"], f.account()["spent"]), (0, 13))
            events = f.rows("test_broker_events")
            self.assertEqual(len([row for row in events if row["kind"] == "ACCOUNTING_APPLIED"]), 1)
            self.assertEqual(len(f.rows("test_broker_accounting")), 1)
            self.assertEqual(len(f.rows("test_broker_settlements")), 1)
            result_before = f.rows("test_broker_results")
            f.receipt(f.call("settle", attempt, ident(101), usage, evidence))
            self.assertEqual(f.rows("test_broker_events"), events)
            self.assertEqual(len(f.rows("test_broker_settlements")), 2)
            self.assertEqual(len({row["applied_event_sequence"] for row in f.rows("test_broker_settlements")}), 1)
            self.assertEqual(f.rows("test_broker_results"), result_before)
            self.assertEqual(f.account()["spent"], 13)

    def test_settlement_restart_replay_and_conflict(self):
        self._require_product()
        if self._outer():
            return
        with self.fixture() as f:
            request, claim, _, _, _, _ = f.finish(known=False)
            attempt = claim["delivery"]["attempt_id"]
            usage = f.usage()
            evidence = f.evidence(request, attempt, usage)
            f.admit(evidence)
            original = f.call("settle", attempt, ident(100), usage, evidence)
            f.receipt(original)
            before = f.snapshot()
            packet = {"attempt_id": attempt, "settlement_id": ident(100), "usage": usage, "accounting_evidence": evidence}
            replay, _ = f.cli_success("settle", packet)
            self.assertEqual(replay, original)
            self.assertEqual(f.snapshot(), before)
            changed = dict(evidence, accounting_record_sha256=digest(b"TEST unadmitted changed record\n"))
            f.refuse("settle", attempt, ident(100), usage, changed)
            self.assertEqual(f.snapshot(), before)
            contradictory = f.usage(actual_charge_nano_usd=14)
            other = dict(evidence, usage_sha256=digest(canonical(contradictory)),
                         accounting_record_sha256=digest(b"TEST alternate contradictory final observation\n"))
            f.admit(other)
            saved = (f.rows("test_broker_accounting"), f.rows("test_broker_completions"), f.rows("test_broker_results"))
            f.refuse("settle", attempt, ident(102), contradictory, other)
            self.assertEqual((f.rows("test_broker_accounting"), f.rows("test_broker_completions"), f.rows("test_broker_results")), saved)
            self.assertEqual((f.account()["held"], f.account()["spent"], f.account()["admission_status"]), (0, 13, "FROZEN"))
            self.assertEqual(len(f.rows("test_broker_settlements")), 1)
            self.assertEqual(len([row for row in f.rows("test_broker_events") if row["kind"] == "ACCOUNTING_APPLIED"]), 1)

    def test_completed_unknown_to_known_billing(self):
        self._require_product()
        if self._outer():
            return
        with self.fixture() as f:
            request, claim, outcome, _, original_complete, _ = f.finish(known=False)
            unrelated = f.request(task=2)
            f.reserve(unrelated)
            self.assertEqual((f.account()["held"], f.account()["spent"]), (32, 0))
            result_before = f.rows("test_broker_results")
            completion_before = f.rows("test_broker_completions")
            task_before = f.rows("test_broker_tasks")[0]
            attempt = claim["delivery"]["attempt_id"]
            usage = f.usage()
            evidence = f.evidence(request, attempt, usage)
            f.admit(evidence)
            value = f.receipt(f.call("settle", attempt, ident(100), usage, evidence))
            self.assertEqual(value["budget"]["billing_status"], "KNOWN")
            self.assertEqual((f.account()["held"], f.account()["spent"]), (16, 13))
            self.assertEqual(f.rows("test_broker_results"), result_before)
            self.assertEqual(f.rows("test_broker_completions"), completion_before)
            task_after = next(row for row in f.rows("test_broker_tasks") if row["task_id"] == request["task_id"])
            for field in ("canonical_request", "canonical_policy", "canonical_price", "canonical_token_admission",
                          "canonical_native_context", "registry_object_sha256", "completion_sha256", "result_blob"):
                self.assertEqual(task_after[field], task_before[field])
            untouched = next(row for row in f.rows("test_broker_tasks") if row["task_id"] == unrelated["task_id"])
            self.assertEqual((untouched["state"], untouched["reservation_held"], untouched["reservation_released"]), ("reserved", 1, 0))
            self.assertEqual(f.call("complete", attempt, outcome, None), original_complete)
            f.refuse("complete", attempt, outcome, usage)
            self.assertEqual(f.account()["spent"], 13)

    def test_unknown_delivery_billing_hold(self):
        self._require_product()
        if self._outer():
            return
        reg = registry()
        reg["deployments"]["TEST-deployment-1"]["max_parallel_attempts"] = 1
        with self.fixture(reg) as f:
            request = f.request(deadline_unix_ms=NOW + 1)
            other = f.request(task=2)
            f.reserve(request)
            f.reserve(other)
            _, claim = f.claim(request)
            outcome = f.fake(request, claim)
            attempt = claim["delivery"]["attempt_id"]
            f.receipt(f.call("mark_unknown", attempt, "COMPLETION_WRITE_FAILED"))
            usage = f.usage()
            evidence = f.evidence(request, attempt, usage, completed=False)
            f.admit(evidence)
            value = f.receipt(f.call("settle", attempt, ident(100), usage, evidence))
            self.assertEqual((value["delivery"]["state"], value["budget"]["billing_status"]), ("unknown_delivery", "KNOWN"))
            self.assertEqual((f.account()["held"], f.account()["spent"]), (32, 13))
            self.assertEqual(f.rows("test_broker_attempts")[0]["capacity_slot_held"], 1)
            f.clock[0] = NOW + 100
            fresh = f.reopen()
            f.refuse("claim_dispatch", other["task_id"], digest(canonical(other)), broker=fresh)
            f.refuse("claim_dispatch", request["task_id"], digest(canonical(request)), broker=fresh)
            f.refuse("confirm_not_dispatched", request["task_id"], digest(canonical(request)), "EXPIRED", broker=fresh)
            fresh.close()
            f.cli_success("read", {"task_id": request["task_id"], "request_sha256": digest(canonical(request))})
            self.assertEqual((f.account()["held"], f.account()["spent"]), (32, 13))
            terminal = f.call("complete", attempt, outcome, None)
            f.receipt(terminal)
            self.assertEqual((f.account()["held"], f.account()["spent"]), (16, 13))
            self.assertEqual(f.call("complete", attempt, outcome, None), terminal)
            self.assertEqual(len([row for row in f.rows("test_broker_events") if row["kind"] == "ACCOUNTING_APPLIED"]), 1)
            self.assertEqual(f.rows("test_broker_attempts")[0]["capacity_slot_held"], 0)
            f.claim(other)
            self.assertEqual(len(f.fake_records()), 1)

    def test_settlement_evidence_binding(self):
        self._require_product()
        if self._outer():
            return
        with self.fixture() as f:
            request, claim, _, _, _, _ = f.finish(known=False)
            attempt = claim["delivery"]["attempt_id"]
            usage = f.usage()
            evidence = f.evidence(request, attempt, usage)
            self.assertEqual(len(evidence), 17)
            f.admit(evidence)
            before = f.snapshot()
            for field in evidence:
                if field == "schema_version":
                    wrong_value = 2
                elif field == "kind":
                    wrong_value = "OTHER_ACCOUNTING_EVIDENCE"
                elif field == "attempt_id":
                    wrong_value = ident(999)
                elif field.endswith("sha256"):
                    wrong_value = "f" * 64 if evidence[field] != "f" * 64 else "0" * 64
                else:
                    wrong_value = "TEST-other-identity"
                wrong = dict(evidence, **{field: wrong_value})
                f.refuse("settle", attempt, ident(100), usage, wrong)
                self.assertEqual(f.snapshot(), before)
                if field in ("schema_version", "kind", "accounting_record_sha256"):
                    continue
                # The trusted entry itself matches wrong evidence. Immutable task/
                # usage/source/provider bindings must still refuse it independently.
                trusted_wrong = copy.deepcopy(f.reg)
                trusted_wrong["accounting_record_admissions"][wrong["accounting_record_sha256"]] = wrong
                fresh = None
                try:
                    with self.assertRaises(f.api.RoutingBrokerRefused):
                        fresh = f.reopen(trusted_wrong)
                        f.call("settle", attempt, ident(101), usage, wrong, broker=fresh)
                finally:
                    if fresh is not None:
                        fresh.close()
                self.assertEqual(f.snapshot(), before)
            without = copy.deepcopy(f.reg)
            without["accounting_record_admissions"] = {}
            fresh = f.reopen(without)
            f.refuse("settle", attempt, ident(102), usage, evidence, broker=fresh)
            fresh.close()
            self.assertEqual(f.snapshot(), before)
            original_price = f.rows("test_broker_tasks")[0]["canonical_price"]
            replace_price(f.reg, input_rate_per_million=9000000, cached_input_rate_per_million=9000000,
                          cache_write_rate_per_million=9000000, output_rate_per_million=9000000)
            f.admit(evidence)
            f.receipt(f.call("settle", attempt, ident(103), usage, evidence))
            self.assertEqual((f.account()["held"], f.account()["spent"]), (0, 13))
            self.assertEqual(f.rows("test_broker_tasks")[0]["canonical_price"], original_price)
            row = f.rows("test_broker_settlements")[0]
            self.assertEqual(row["accounting_evidence_sha256"], digest(canonical(evidence)))

    def test_settlement_integer_and_charge_bounds(self):
        self._require_product()
        if self._outer():
            return
        with self.fixture() as f:
            request, claim, _, _, _, _ = f.finish(known=False)
            attempt = claim["delivery"]["attempt_id"]
            for field in ("input_tokens", "output_tokens", "cached_input_tokens", "cache_write_tokens",
                          "reasoning_tokens", "actual_charge_nano_usd"):
                for value in (-1, True, False, 0.5, "0", None, MAX_INT + 1):
                    usage = f.usage(**{field: value})
                    evidence = f.evidence(request, attempt, usage)
                    f.admit(evidence)
                    before = f.snapshot()
                    f.refuse("settle", attempt, ident(100), usage, evidence)
                    self.assertEqual(f.snapshot(), before)
            for bad in ({}, dict(f.usage(), refund_nano_usd=1)):
                evidence = f.evidence(request, attempt, bad) if bad else f.evidence(request, attempt, f.usage())
                if bad:
                    f.admit(evidence)
                before = f.snapshot()
                f.refuse("settle", attempt, ident(100), bad, evidence)
                self.assertEqual(f.snapshot(), before)
        reg = registry()
        reg["accounts"][BUDGET]["limit_nano_usd"] = MAX_INT
        with self.fixture(reg) as f:
            first, first_claim, _, _, _, _ = f.finish(known=False)
            second, second_claim, _, _, _, _ = f.finish(task=2, known=False)
            self.assertEqual(f.account()["held"], 32)
            for request, claim, charge, settlement in ((first, first_claim, MAX_INT, 100),
                                                       (second, second_claim, 1, 101)):
                attempt = claim["delivery"]["attempt_id"]
                usage = f.usage(actual_charge_nano_usd=charge)
                evidence = f.evidence(request, attempt, usage)
                f.admit(evidence)
                value = f.receipt(f.call("settle", attempt, ident(settlement), usage, evidence))
                self.assertEqual(value["budget"]["actual_charge"], charge)
                self.assertEqual(value["budget"]["invariant_status"], "VIOLATED")
            account = f.account()
            self.assertEqual((account["held"], account["admission_status"], account["invariant_status"]), (0, "FROZEN", "VIOLATED"))
            self.assertIsNone(account["spent"])
            applied = [row for row in f.rows("test_broker_events") if row["kind"] == "ACCOUNTING_APPLIED"]
            self.assertEqual(len(applied), 2)
            self.assertEqual(applied[-1]["aggregate_spent_nano_usd_decimal"], "9223372036854775808")
            self.assertEqual([row["charge_nano_usd_decimal"] for row in applied], ["9223372036854775807", "1"])
            self.assertEqual(sorted(row["actual_charge"] for row in f.rows("test_broker_accounting")), [1, MAX_INT])
            self.assertEqual(len(f.fake_records()), 2)
        with self.fixture() as f:
            request, claim, _, _, _, _ = f.finish(known=False)
            attempt = claim["delivery"]["attempt_id"]
            usage = f.usage()
            evidence = f.evidence(request, attempt, usage)
            f.admit(evidence)
            before = f.snapshot()
            self._storage_pressure(lambda: f.refuse("settle", attempt, ident(100), usage, evidence))
            self.assertEqual(f.snapshot(), before)
            self.assertEqual((f.account()["held"], f.account()["spent"]), (16, 0))


if __name__ == "__main__":
    unittest.main()
