"""Private, durable TEST routing authority. No transport or scientific authority.

The caller is a trusted fixture controller, not a model or public endpoint.
Every operation owns a short SQLite transaction and closes its connection.
Registry declarations can authorize only synthetic TEST state, never a real call.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import uuid

__all__ = ["Broker", "RoutingBrokerRefused"]

REFUSAL = "ARCHITECTURE_ROUTING_BROKER_REFUSED"
MAX_INT = (1 << 63) - 1
DB_LIMIT = 16 * 1024 * 1024
REQUEST_LIMIT = 32 * 1024
REGISTRY_LIMIT = 1024 * 1024
HEX32 = re.compile(r"[0-9a-f]{32}")
HEX64 = re.compile(r"[0-9a-f]{64}")
IDENTITY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}")
CAPABILITY = re.compile(r"[a-z][a-z0-9_]{0,63}")
OPERATIONS = {"navigation": 1, "coding_review": 2, "mathematical_audit": 3}
SCOPES = {"public", "synthetic", "local_private"}
STATUSES = {"ADMITTED", "UNKNOWN", "DISABLED"}
DENIALS = {
    "INVALID_INPUT", "REQUEST_CONFLICT", "UNKNOWN_POLICY", "UNKNOWN_ACCOUNT",
    "UNKNOWN_PRICE", "UNKNOWN_CAPABILITY", "UNKNOWN_TOKEN_ADMISSION",
    "TOKEN_ADMISSION_MISMATCH", "DISCLOSURE_DENIED", "ROUTE_MISMATCH", "EXPIRED",
    "COST_UNBOUNDED", "BUDGET_EXHAUSTED", "AUTHORITY_UNAVAILABLE",
    "STORAGE_UNAVAILABLE", "CAPACITY_UNAVAILABLE",
}
UNKNOWN_REASONS = {"TIMEOUT", "CONNECTION_LOST", "PROCESS_RECOVERY", "COMPLETION_WRITE_FAILED"}
ERROR_CODES = {"AUTHENTICATION", "RATE_LIMIT", "TIMEOUT", "CONNECTION_LOST",
               "INVALID_RESPONSE", "CANCELLED", "OTHER"}
REQUEST_FIELDS = set("task_id input_sha256 disclosure_scope operation required_capabilities account_id provider_id deployment_id model_id policy_sha256 price_sha256 budget_id max_cost_nano_usd deadline_unix_ms context_tokens max_output_tokens".split())
USAGE_FIELDS = set("input_tokens output_tokens cached_input_tokens cache_write_tokens reasoning_tokens actual_charge_nano_usd accounting_source_sha256".split())
EVIDENCE_FIELDS = set("schema_version kind attempt_id account_id provider_id deployment_id model_id input_sha256 request_sha256 policy_sha256 price_sha256 tokenizer_sha256 completion_sha256 native_provider_request_id usage_sha256 accounting_source_sha256 accounting_record_sha256".split())
RATE_FIELDS = ("input_rate_per_million", "cached_input_rate_per_million",
               "cache_write_rate_per_million", "output_rate_per_million", "fixed_request_charge")
LIMITS = {"max_tasks": 1024, "max_events": 4096, "max_result_bytes": 65536,
          "db_bytes": DB_LIMIT, "retained_fixture_bytes": 32 * 1024 * 1024}


class RoutingBrokerRefused(Exception):
    """Opaque refusal, including when a private storage operation fails."""

    def __init__(self):
        super().__init__(REFUSAL)


def require(condition):
    if not condition:
        raise RoutingBrokerRefused()


def opaque(function):
    @wraps(function)
    def guarded(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception:
            raise RoutingBrokerRefused() from None
    return guarded


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def integer(value, low=0, high=MAX_INT):
    require(type(value) is int and low <= value <= high)


def string(value, pattern):
    require(type(value) is str and pattern.fullmatch(value) is not None)


def fields(value, expected):
    require(type(value) is dict and set(value) == set(expected))


def tree(value, depth=0):
    """Enforce the same bounded, exact JSON domain on module and CLI inputs."""
    if type(value) in (dict, list):
        require(depth < 16)
        if type(value) is dict:
            for key, item in value.items():
                require(type(key) is str)
                key.encode("utf-8", errors="strict")
                tree(item, depth + 1)
        else:
            for item in value:
                tree(item, depth + 1)
    elif type(value) is str:
        value.encode("utf-8", errors="strict")
    elif type(value) is int:
        require(-MAX_INT <= value <= MAX_INT)
    else:
        require(value is None or type(value) is bool)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result)
        result[key] = value
    return result


def reject_number(_):
    raise RoutingBrokerRefused()


def decode(raw, limit):
    require(type(raw) is bytes and len(raw) <= limit)
    value = json.loads(raw.decode("utf-8", errors="strict"),
                       object_pairs_hook=unique_object, parse_float=reject_number,
                       parse_constant=reject_number)
    tree(value)
    return value


def snapshot(value, limit):
    tree(value)
    raw = canonical(value)
    require(len(raw) <= limit)
    return decode(raw, limit), raw


def sorted_list(value, allowed=None, pattern=None, minimum=0, maximum=1024):
    require(type(value) is list and minimum <= len(value) <= maximum)
    require(all(type(item) is str for item in value))
    require(value == sorted(set(value)))
    for item in value:
        if allowed is not None:
            require(item in allowed)
        if pattern is not None:
            string(item, pattern)


def validate_request(request):
    fields(request, REQUEST_FIELDS)
    for name in ("task_id", "budget_id"):
        string(request[name], HEX32)
    for name in ("input_sha256", "policy_sha256", "price_sha256"):
        string(request[name], HEX64)
    for name in ("account_id", "provider_id", "deployment_id", "model_id"):
        string(request[name], IDENTITY)
    require(request["disclosure_scope"] in SCOPES and request["operation"] in OPERATIONS)
    sorted_list(request["required_capabilities"], pattern=CAPABILITY, minimum=1, maximum=16)
    integer(request["max_cost_nano_usd"])
    integer(request["deadline_unix_ms"], 1)
    integer(request["context_tokens"], 1, 1048576)
    integer(request["max_output_tokens"], 1, 131072)


def validate_usage(usage):
    fields(usage, USAGE_FIELDS)
    for name in USAGE_FIELDS - {"accounting_source_sha256"}:
        integer(usage[name])
    string(usage["accounting_source_sha256"], HEX64)
    require(usage["cached_input_tokens"] + usage["cache_write_tokens"] <= usage["input_tokens"])
    require(usage["reasoning_tokens"] <= usage["output_tokens"])


def validate_evidence(evidence):
    fields(evidence, EVIDENCE_FIELDS)
    integer(evidence["schema_version"], 1, 1)
    require(evidence["kind"] == "TEST_ACCOUNTING_EVIDENCE")
    string(evidence["attempt_id"], HEX32)
    for name in ("account_id", "provider_id", "deployment_id", "model_id"):
        string(evidence[name], IDENTITY)
    for name in EVIDENCE_FIELDS:
        if name.endswith("_sha256") and name != "completion_sha256":
            string(evidence[name], HEX64)
    if evidence["completion_sha256"] is not None:
        string(evidence["completion_sha256"], HEX64)
    if evidence["native_provider_request_id"] is not None:
        string(evidence["native_provider_request_id"], IDENTITY)
    require(len(canonical(evidence)) <= 8192)


def validate_registry(registry):
    fields(registry, {"schema_version", "kind", "native_context", "policy", "price_cards",
                      "accounts", "deployments", "disclosure_admissions", "input_token_admissions",
                      "accounting_record_admissions", "limits"})
    integer(registry["schema_version"], 1, 1)
    require(registry["kind"] == "TEST_ROUTING_REGISTRY")
    native = registry["native_context"]
    fields(native, {"repository", "checked_commit", "run_id", "run_attempt"})
    string(native["repository"], re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+"))
    require(all(part not in (".", "..") for part in native["repository"].split("/")))
    string(native["checked_commit"], re.compile(r"[0-9a-f]{40}"))
    for name in ("run_id", "run_attempt"):
        string(native[name], re.compile(r"[1-9][0-9]*"))
    policy = registry["policy"]
    fields(policy, {"version", "valid_from_unix_ms", "valid_until_unix_ms", "routes",
                    "redis_status", "storage_status"})
    integer(policy["version"], 1)
    integer(policy["valid_from_unix_ms"], 1)
    integer(policy["valid_until_unix_ms"], policy["valid_from_unix_ms"])
    require(policy["redis_status"] == "NOT_USED" and policy["storage_status"] in STATUSES)
    fields(policy["routes"], OPERATIONS)
    for name in ("price_cards", "accounts", "deployments", "disclosure_admissions",
                 "input_token_admissions", "accounting_record_admissions"):
        require(type(registry[name]) is dict)
    for key, card in registry["price_cards"].items():
        string(key, HEX64)
        fields(card, {*RATE_FIELDS, "currency", "scale", "reasoning_included_in_output",
                      "tools_disabled", "valid_from_unix_ms", "valid_until_unix_ms"})
        require(card["currency"] == "USD")
        integer(card["scale"], 9, 9)
        for name in RATE_FIELDS:
            if card[name] is not None:
                integer(card[name])
        require(card["reasoning_included_in_output"] is True and card["tools_disabled"] is True)
        integer(card["valid_from_unix_ms"], 1)
        integer(card["valid_until_unix_ms"], card["valid_from_unix_ms"])
        require(key == digest(canonical(card)))
    for budget, account in registry["accounts"].items():
        string(budget, HEX32)
        fields(account, {"account_id", "currency", "scale", "scope", "limit_nano_usd", "status", "version"})
        string(account["account_id"], IDENTITY)
        require(account["currency"] == "USD" and account["scope"] == "BROKER_TRAFFIC")
        integer(account["scale"], 9, 9)
        require(account["status"] in STATUSES)
        if account["limit_nano_usd"] is None:
            require(account["status"] != "ADMITTED")
        else:
            integer(account["limit_nano_usd"])
        integer(account["version"], 1)
    for deployment, record in registry["deployments"].items():
        string(deployment, IDENTITY)
        fields(record, {"account_id", "provider_id", "model_id", "tier", "operations", "capabilities",
                        "status", "adapter_id", "api_identity_sha256", "capacity_status",
                        "max_parallel_attempts", "max_context_tokens", "max_output_tokens",
                        "price_sha256", "tokenizer_sha256"})
        for name in ("account_id", "provider_id", "model_id"):
            string(record[name], IDENTITY)
        require(record["model_id"].startswith("TEST-") and record["adapter_id"] == "TEST")
        integer(record["tier"], 1, 3)
        sorted_list(record["operations"], allowed=OPERATIONS, minimum=1, maximum=3)
        sorted_list(record["capabilities"], pattern=CAPABILITY, minimum=1)
        require(record["status"] in STATUSES and record["capacity_status"] in STATUSES)
        integer(record["max_parallel_attempts"], 1, 1024)
        integer(record["max_context_tokens"], 1, 1048576)
        integer(record["max_output_tokens"], 1, 131072)
        for name in ("api_identity_sha256", "price_sha256", "tokenizer_sha256"):
            string(record[name], HEX64)
        require(record["price_sha256"] in registry["price_cards"])
        require(any(account["account_id"] == record["account_id"] for account in registry["accounts"].values()))
    for operation, route in policy["routes"].items():
        require(type(route) is list and len(route) <= 1024)
        require(all(type(item) is str for item in route) and len(set(route)) == len(route))
        for deployment in route:
            require(deployment in registry["deployments"])
            require(registry["deployments"][deployment]["tier"] == OPERATIONS[operation])
    for key, scopes in registry["disclosure_admissions"].items():
        string(key, HEX64)
        sorted_list(scopes, allowed=SCOPES, minimum=1, maximum=3)
    for key, deployments in registry["input_token_admissions"].items():
        string(key, HEX64)
        require(type(deployments) is dict)
        for deployment, tokenizers in deployments.items():
            require(deployment in registry["deployments"] and type(tokenizers) is dict)
            for tokenizer, record in tokenizers.items():
                string(tokenizer, HEX64)
                require(tokenizer == registry["deployments"][deployment]["tokenizer_sha256"])
                fields(record, {"input_tokens", "payload_bytes", "count_evidence_sha256"})
                integer(record["input_tokens"], 0, 1048576)
                integer(record["payload_bytes"], 0, 16777216)
                string(record["count_evidence_sha256"], HEX64)
    for key, evidence in registry["accounting_record_admissions"].items():
        string(key, HEX64)
        validate_evidence(evidence)
        require(evidence["accounting_record_sha256"] == key)
    fields(registry["limits"], LIMITS)
    for key, value in LIMITS.items():
        integer(registry["limits"][key], value, value)


# Eight real tables own facts; the eight views project relational columns.
# SQL text is also the schema allowlist checked on every connection.
TABLE_SQL = {
    "accounts": """CREATE TABLE accounts (
budget_id TEXT PRIMARY KEY, account_id TEXT NOT NULL, account_limit INTEGER,
held INTEGER NOT NULL CHECK(held>=0), spent INTEGER, admission_status TEXT NOT NULL,
invariant_status TEXT NOT NULL, version INTEGER NOT NULL CHECK(version>0),
canonical_account BLOB NOT NULL, aggregate_spent_decimal TEXT NOT NULL) STRICT""",
    "tasks": """CREATE TABLE tasks (
task_id TEXT PRIMARY KEY, budget_id TEXT NOT NULL REFERENCES accounts(budget_id),
deployment_id TEXT NOT NULL, request_sha256 TEXT NOT NULL, input_sha256 TEXT NOT NULL,
canonical_request BLOB NOT NULL, canonical_policy BLOB NOT NULL, canonical_price BLOB NOT NULL,
canonical_token_admission BLOB NOT NULL, canonical_native_context BLOB NOT NULL,
canonical_deployment BLOB NOT NULL, registry_object_sha256 TEXT NOT NULL,
state TEXT, attempt_id TEXT UNIQUE, reservation_upper_bound INTEGER,
reservation_held INTEGER NOT NULL CHECK(reservation_held IN (0,1)),
reservation_released INTEGER NOT NULL CHECK(reservation_released IN (0,1)),
spend_applied INTEGER NOT NULL CHECK(spend_applied IN (0,1)), billing_status TEXT NOT NULL,
actual_charge INTEGER, completion_sha256 TEXT, economic_sha256 TEXT,
admission_reason TEXT, outcome TEXT NOT NULL, event_sequence INTEGER NOT NULL,
original_reserve_receipt BLOB NOT NULL, latest_receipt BLOB NOT NULL,
unknown_reason TEXT, unknown_receipt BLOB, nondispatch_reason TEXT) STRICT""",
    "attempts": """CREATE TABLE attempts (
attempt_id TEXT PRIMARY KEY, task_id TEXT NOT NULL UNIQUE REFERENCES tasks(task_id),
request_sha256 TEXT NOT NULL, claim_event_sequence INTEGER NOT NULL REFERENCES events(event_sequence),
capacity_slot_held INTEGER NOT NULL CHECK(capacity_slot_held IN (0,1)),
native_provider_request_id TEXT) STRICT""",
    "completions": """CREATE TABLE completions (
attempt_id TEXT PRIMARY KEY REFERENCES attempts(attempt_id), completion_sha256 TEXT NOT NULL,
canonical_packet BLOB NOT NULL, original_receipt BLOB NOT NULL) STRICT""",
    "accounting": """CREATE TABLE accounting (
attempt_id TEXT PRIMARY KEY REFERENCES attempts(attempt_id), economic_sha256 TEXT NOT NULL,
canonical_usage BLOB NOT NULL, accounting_source_sha256 TEXT NOT NULL, actual_charge INTEGER NOT NULL,
applied_event_sequence INTEGER NOT NULL UNIQUE REFERENCES events(event_sequence)) STRICT""",
    "settlements": """CREATE TABLE settlements (
settlement_id TEXT PRIMARY KEY, attempt_id TEXT NOT NULL REFERENCES accounting(attempt_id),
settlement_sha256 TEXT NOT NULL, canonical_packet BLOB NOT NULL,
accounting_evidence_sha256 TEXT NOT NULL,
applied_event_sequence INTEGER NOT NULL REFERENCES events(event_sequence),
original_receipt BLOB NOT NULL) STRICT""",
    "events": """CREATE TABLE events (
event_sequence INTEGER PRIMARY KEY, task_id TEXT REFERENCES tasks(task_id) DEFERRABLE INITIALLY DEFERRED,
attempt_id TEXT, kind TEXT NOT NULL, request_sha256 TEXT, completion_sha256 TEXT,
economic_sha256 TEXT, settlement_sha256 TEXT, charge_nano_usd_decimal TEXT,
aggregate_spent_nano_usd_decimal TEXT) STRICT""",
    "results": """CREATE TABLE results (
task_id TEXT PRIMARY KEY REFERENCES tasks(task_id), request_sha256 TEXT NOT NULL,
result_sha256 TEXT NOT NULL, result_bytes INTEGER NOT NULL CHECK(result_bytes BETWEEN 0 AND 65536),
result_blob BLOB NOT NULL) STRICT""",
}
VIEW_SQL = {
    "test_broker_accounts": "CREATE VIEW test_broker_accounts AS SELECT budget_id,account_id,account_limit,held,spent,admission_status,invariant_status,version FROM accounts",
    "test_broker_tasks": "CREATE VIEW test_broker_tasks AS SELECT t.task_id,t.request_sha256,t.input_sha256,t.canonical_request,t.canonical_policy,t.canonical_price,t.canonical_token_admission,t.canonical_native_context,t.registry_object_sha256,t.state,t.attempt_id,t.reservation_upper_bound,t.reservation_held,t.reservation_released,t.spend_applied,t.billing_status,t.actual_charge,t.completion_sha256,t.economic_sha256,r.result_blob,t.latest_receipt FROM tasks t LEFT JOIN results r ON r.task_id=t.task_id",
    "test_broker_attempts": "CREATE VIEW test_broker_attempts AS SELECT attempt_id,task_id,request_sha256,claim_event_sequence,capacity_slot_held,native_provider_request_id FROM attempts",
    "test_broker_completions": "CREATE VIEW test_broker_completions AS SELECT attempt_id,completion_sha256,canonical_packet,original_receipt FROM completions",
    "test_broker_accounting": "CREATE VIEW test_broker_accounting AS SELECT attempt_id,economic_sha256,canonical_usage,accounting_source_sha256,actual_charge,applied_event_sequence FROM accounting",
    "test_broker_settlements": "CREATE VIEW test_broker_settlements AS SELECT settlement_id,attempt_id,settlement_sha256,canonical_packet,accounting_evidence_sha256,applied_event_sequence,original_receipt FROM settlements",
    "test_broker_events": "CREATE VIEW test_broker_events AS SELECT event_sequence,task_id,attempt_id,kind,request_sha256,completion_sha256,economic_sha256,settlement_sha256,charge_nano_usd_decimal,aggregate_spent_nano_usd_decimal FROM events",
    "test_broker_results": "CREATE VIEW test_broker_results AS SELECT task_id,request_sha256,result_sha256,result_bytes,result_blob FROM results",
}


def stored(raw, limit=REGISTRY_LIMIT):
    value = decode(raw, limit)
    require(canonical(value) == raw)
    return value


class Broker:
    @opaque
    def __init__(self, ledger_root, registry, *, clock):
        require(isinstance(ledger_root, (str, Path)) and callable(clock))
        self.root = Path(ledger_root)
        require(self.root.is_absolute())
        self.registry, raw = snapshot(registry, REGISTRY_LIMIT)
        validate_registry(self.registry)
        self.registry_sha256 = digest(raw)
        self.clock = clock
        self.closed = False
        self.db = self.root / "routing-broker.sqlite3"
        self._namespace()
        fresh = not self.db.exists()
        if fresh:
            require(not any(self.root.iterdir()))
            fd = os.open(self.db, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            os.close(fd)
        with self._transaction(initialize=fresh) as connection:
            if fresh:
                for sql in TABLE_SQL.values():
                    connection.execute(sql)
                for sql in VIEW_SQL.values():
                    connection.execute(sql)
                connection.execute("PRAGMA user_version=1")
                for budget, account in self.registry["accounts"].items():
                    connection.execute("INSERT INTO accounts VALUES (?,?,?,?,?,?,?,?,?,?)", (
                        budget, account["account_id"], account["limit_nano_usd"], 0, 0,
                        account["status"], "UNKNOWN" if account["limit_nano_usd"] is None else "WITHIN_LIMIT",
                        account["version"], canonical(account), "0"))
        self._namespace()

    def _namespace(self):
        # SQLite opens filenames; the fixture owner supplies exclusive isolation.
        require(".." not in self.root.parts and "." not in self.root.parts)
        for component in (self.root, *self.root.parents):
            info = component.lstat()
            require(stat.S_ISDIR(info.st_mode))
        info = self.root.lstat()
        require(info.st_uid == os.geteuid() and stat.S_IMODE(info.st_mode) == 0o700)
        total = 0
        for entry in self.root.iterdir():
            require(entry.name in {"routing-broker.sqlite3", "routing-broker.sqlite3-journal"})
            info = entry.lstat()
            require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1)
            require(info.st_uid == os.geteuid() and stat.S_IMODE(info.st_mode) == 0o600)
            require(info.st_size <= DB_LIMIT)
            total += info.st_size
        require(total <= LIMITS["retained_fixture_bytes"])

    @contextmanager
    def _transaction(self, *, initialize=False, write=True):
        require(not self.closed)
        self._namespace()
        require(self.db.is_file())
        connection = sqlite3.connect(self.db.as_uri() + "?mode=rw", uri=True,
                                     timeout=1.0, isolation_level=None)
        try:
            connection.row_factory = sqlite3.Row
            require(connection.execute("PRAGMA journal_mode=DELETE").fetchone()[0] == "delete")
            connection.execute("PRAGMA synchronous=EXTRA")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA trusted_schema=OFF")
            connection.execute("PRAGMA read_uncommitted=OFF")
            connection.execute("PRAGMA busy_timeout=1000")
            for pragma, expected in (("synchronous", 3), ("foreign_keys", 1), ("trusted_schema", 0),
                                     ("read_uncommitted", 0), ("busy_timeout", 1000)):
                require(connection.execute("PRAGMA " + pragma).fetchone()[0] == expected)
            page_size = connection.execute("PRAGMA page_size").fetchone()[0]
            require(type(page_size) is int and 512 <= page_size <= 65536)
            # Bound the DB and an active DELETE journal including page headers.
            # Leave two MiB of aggregate headroom for the separately owned
            # bounded registry/fake evidence; their collection belongs to the
            # fixture controller and the broker never opens its attempt file.
            page_limit = (DB_LIMIT - 1024 * 1024) // (page_size + 8)
            require(connection.execute("PRAGMA max_page_count=" + str(page_limit)).fetchone()[0] == page_limit)
            connection.execute("PRAGMA journal_size_limit=16777216")
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            if not initialize:
                self._verify(connection)
            yield connection
            self._namespace()
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _verify(self, connection):
        require(connection.execute("PRAGMA user_version").fetchone()[0] == 1)
        schema = {row["name"]: (row["type"], row["sql"]) for row in connection.execute(
            "SELECT name,type,sql FROM sqlite_master WHERE sql IS NOT NULL")}
        expected = {name: ("table", sql) for name, sql in TABLE_SQL.items()}
        expected.update({name: ("view", sql) for name, sql in VIEW_SQL.items()})
        require(schema == expected)
        require(connection.execute("PRAGMA quick_check").fetchall()[0][0] == "ok")
        require(not connection.execute("PRAGMA foreign_key_check").fetchall())
        tasks = list(connection.execute("SELECT * FROM tasks"))
        require(len(tasks) <= LIMITS["max_tasks"])
        require(connection.execute("SELECT count(*) FROM events").fetchone()[0] <= LIMITS["max_events"])
        for task in tasks:
            request = stored(task["canonical_request"], REQUEST_LIMIT)
            validate_request(request)
            require(task["task_id"] == request["task_id"] and task["budget_id"] == request["budget_id"])
            require(task["request_sha256"] == digest(task["canonical_request"]))
            require(task["input_sha256"] == request["input_sha256"] and task["deployment_id"] == request["deployment_id"])
            require(digest(task["canonical_policy"]) == request["policy_sha256"])
            require(digest(task["canonical_price"]) == request["price_sha256"])
            for name in ("canonical_policy", "canonical_price", "canonical_token_admission",
                         "canonical_native_context", "canonical_deployment"):
                stored(task[name])
            string(task["registry_object_sha256"], HEX64)
            state = task["state"]
            require(state in (None, "reserved", "dispatching", "completed", "confirmed_not_dispatched", "unknown_delivery"))
            attempt = connection.execute("SELECT * FROM attempts WHERE task_id=?", (task["task_id"],)).fetchone()
            require((attempt is not None) == (state in ("dispatching", "completed", "unknown_delivery")))
            if attempt is not None:
                require(attempt["attempt_id"] == task["attempt_id"] and attempt["request_sha256"] == task["request_sha256"])
                require(attempt["capacity_slot_held"] == int(state in ("dispatching", "unknown_delivery")))
            else:
                require(task["attempt_id"] is None)
            accounting = connection.execute("SELECT * FROM accounting WHERE attempt_id=?", (task["attempt_id"],)).fetchone()
            completion = connection.execute("SELECT * FROM completions WHERE attempt_id=?", (task["attempt_id"],)).fetchone()
            require((completion is not None) == (state == "completed"))
            require(task["spend_applied"] == int(accounting is not None))
            require(task["billing_status"] == ("KNOWN" if accounting is not None else "UNKNOWN"))
            if accounting is None:
                require(task["actual_charge"] is None and task["economic_sha256"] is None)
            else:
                usage = stored(accounting["canonical_usage"])
                validate_usage(usage)
                require(accounting["economic_sha256"] == digest(canonical({"attempt_id": task["attempt_id"], "usage": usage})))
                require(task["economic_sha256"] == accounting["economic_sha256"])
                require(task["actual_charge"] == accounting["actual_charge"] == usage["actual_charge_nano_usd"])
                require(accounting["accounting_source_sha256"] == usage["accounting_source_sha256"])
                event = connection.execute("SELECT * FROM events WHERE event_sequence=?", (accounting["applied_event_sequence"],)).fetchone()
                require(event["kind"] == "ACCOUNTING_APPLIED" and event["attempt_id"] == task["attempt_id"])
                require(event["economic_sha256"] == task["economic_sha256"] and event["charge_nano_usd_decimal"] == str(task["actual_charge"]))
            if completion is None:
                require(task["completion_sha256"] is None)
            else:
                packet = stored(completion["canonical_packet"], 512 * 1024)
                fields(packet, {"attempt_id", "outcome", "usage"})
                require(packet["attempt_id"] == task["attempt_id"])
                require(task["completion_sha256"] == completion["completion_sha256"] == digest(completion["canonical_packet"]))
                stored(completion["original_receipt"])
                fields(packet["outcome"], {"status", "result_utf8", "native_provider_request_id", "error_code"})
                require(packet["outcome"]["status"] == task["outcome"])
                require(packet["outcome"]["native_provider_request_id"] == attempt["native_provider_request_id"])
                if packet["usage"] is not None:
                    validate_usage(packet["usage"])
                    require(accounting is not None and canonical(packet["usage"]) == accounting["canonical_usage"])
            held = int(state in ("reserved", "dispatching", "unknown_delivery") or (state == "completed" and accounting is None))
            released = int(state == "confirmed_not_dispatched" or (state == "completed" and accounting is not None))
            require(task["reservation_held"] == held and task["reservation_released"] == released)
            if state is None:
                require(task["admission_reason"] in DENIALS)
            else:
                integer(task["reservation_upper_bound"])
                require(task["admission_reason"] is None)
            for name in ("original_reserve_receipt", "latest_receipt"):
                receipt = stored(task[name], 256 * 1024)
                require(receipt["request"]["request_sha256"] == task["request_sha256"])
                require(receipt["scientific_effect"] == "NONE" and receipt["scientific_status_authority"] is False)
            result = connection.execute("SELECT * FROM results WHERE task_id=?", (task["task_id"],)).fetchone()
            if completion is not None:
                expected_result = packet["outcome"]["result_utf8"]
                require((result is not None) == (expected_result is not None))
                if result is not None:
                    require(type(expected_result) is str and result["result_blob"] == expected_result.encode("utf-8"))
            if result is not None:
                require(state == "completed" and result["request_sha256"] == task["request_sha256"])
                require(result["result_bytes"] == len(result["result_blob"]) <= LIMITS["max_result_bytes"])
                require(result["result_sha256"] == digest(result["result_blob"]))
                result["result_blob"].decode("utf-8", errors="strict")
        for account in connection.execute("SELECT * FROM accounts"):
            original = stored(account["canonical_account"])
            require(original["account_id"] == account["account_id"] and original["limit_nano_usd"] == account["account_limit"])
            total_held = sum(task["reservation_upper_bound"] for task in tasks
                             if task["budget_id"] == account["budget_id"] and task["reservation_held"])
            total_spent = sum(row[0] for row in connection.execute(
                "SELECT a.actual_charge FROM accounting a JOIN attempts p ON p.attempt_id=a.attempt_id JOIN tasks t ON t.task_id=p.task_id WHERE t.budget_id=?", (account["budget_id"],)))
            require(account["held"] == total_held and total_held <= MAX_INT)
            require(account["aggregate_spent_decimal"] == str(total_spent))
            require(account["spent"] == (total_spent if total_spent <= MAX_INT else None))
            integer(account["version"], original["version"])
            require(account["admission_status"] in STATUSES | {"FROZEN"})
            require(account["invariant_status"] in {"WITHIN_LIMIT", "VIOLATED", "UNKNOWN"})
            if total_spent > MAX_INT or (account["account_limit"] is not None and total_spent + total_held > account["account_limit"]):
                require(account["invariant_status"] == "VIOLATED" and account["admission_status"] == "FROZEN")
            observed_aggregate = 0
            for fact in connection.execute("SELECT a.actual_charge,a.economic_sha256,a.applied_event_sequence,p.attempt_id,e.* FROM accounting a JOIN attempts p ON p.attempt_id=a.attempt_id JOIN tasks t ON t.task_id=p.task_id JOIN events e ON e.event_sequence=a.applied_event_sequence WHERE t.budget_id=? ORDER BY a.applied_event_sequence", (account["budget_id"],)):
                observed_aggregate += fact["actual_charge"]
                require(fact["aggregate_spent_nano_usd_decimal"] == str(observed_aggregate))
        by_task = {task["task_id"]: task for task in tasks}
        for event in connection.execute("SELECT * FROM events ORDER BY event_sequence"):
            integer(event["event_sequence"], 1)
            require(event["task_id"] in by_task and event["request_sha256"] == by_task[event["task_id"]]["request_sha256"])
            require(event["kind"] in {"RESERVATION_DENIED", "RESERVED", "DISPATCH_CLAIMED", "UNKNOWN_MARKED",
                                      "NONDISPATCH_CONFIRMED", "COMPLETION_RECORDED", "ACCOUNTING_APPLIED", "ACCOUNT_FROZEN"})
            if event["kind"] in {"RESERVATION_DENIED", "RESERVED", "NONDISPATCH_CONFIRMED"}:
                require(all(event[name] is None for name in ("attempt_id", "completion_sha256", "economic_sha256", "settlement_sha256", "charge_nano_usd_decimal", "aggregate_spent_nano_usd_decimal")))
            if event["attempt_id"] is not None:
                require(event["attempt_id"] == by_task[event["task_id"]]["attempt_id"])
            if event["kind"] == "ACCOUNTING_APPLIED":
                fact = connection.execute("SELECT * FROM accounting WHERE applied_event_sequence=?", (event["event_sequence"],)).fetchone()
                require(fact is not None and fact["attempt_id"] == event["attempt_id"])
                require(fact["economic_sha256"] == event["economic_sha256"])
        for settlement in connection.execute("SELECT * FROM settlements"):
            packet = stored(settlement["canonical_packet"])
            require(packet["settlement_id"] == settlement["settlement_id"] and packet["attempt_id"] == settlement["attempt_id"])
            require(digest(settlement["canonical_packet"]) == settlement["settlement_sha256"])
            require(digest(canonical(packet["accounting_evidence"])) == settlement["accounting_evidence_sha256"])
            accounting = connection.execute("SELECT * FROM accounting WHERE attempt_id=?", (settlement["attempt_id"],)).fetchone()
            require(accounting["canonical_usage"] == canonical(packet["usage"]))
            require(accounting["applied_event_sequence"] == settlement["applied_event_sequence"])
            stored(settlement["original_receipt"])

    def _now(self):
        now = self.clock()
        integer(now, 1)
        return now

    def _binding(self, request):
        registry = self.registry
        require(request["policy_sha256"] == digest(canonical(registry["policy"])))
        require(request["budget_id"] in registry["accounts"])
        account = registry["accounts"][request["budget_id"]]
        require(request["account_id"] == account["account_id"])
        require(request["deployment_id"] in registry["deployments"])
        deployment = registry["deployments"][request["deployment_id"]]
        for name in ("account_id", "provider_id", "model_id", "price_sha256"):
            require(request[name] == deployment[name])
        require(request["price_sha256"] in registry["price_cards"])
        require(request["input_sha256"] in registry["disclosure_admissions"])
        reviewed_capabilities = {capability for record in registry["deployments"].values()
                                 for capability in record["capabilities"]}
        require(set(request["required_capabilities"]).issubset(reviewed_capabilities))
        admitted = registry["input_token_admissions"].get(request["input_sha256"], {})
        require(request["deployment_id"] in admitted)
        tokens = admitted[request["deployment_id"]]
        require(deployment["tokenizer_sha256"] in tokens)
        return deployment, registry["price_cards"][request["price_sha256"]], tokens[deployment["tokenizer_sha256"]]

    @staticmethod
    def _upper(request, card, tokens):
        if any(card[name] is None for name in RATE_FIELDS):
            return None
        return ((tokens["input_tokens"] * max(card[name] for name in RATE_FIELDS[:3]) + 999999) // 1000000
                + (request["max_output_tokens"] * card["output_rate_per_million"] + 999999) // 1000000
                + card["fixed_request_charge"])

    def _admission(self, connection, request, deployment, card, tokens, now, own_hold=0):
        upper = self._upper(request, card, tokens)
        policy = self.registry["policy"]
        account = connection.execute("SELECT * FROM accounts WHERE budget_id=?", (request["budget_id"],)).fetchone()
        require(account is not None and account["account_id"] == request["account_id"])
        if (not policy["valid_from_unix_ms"] <= now < policy["valid_until_unix_ms"]
                or not card["valid_from_unix_ms"] <= now < card["valid_until_unix_ms"]
                or now >= request["deadline_unix_ms"]):
            return "EXPIRED", upper
        if policy["storage_status"] != "ADMITTED":
            return "AUTHORITY_UNAVAILABLE", upper
        current_account = self.registry["accounts"][request["budget_id"]]
        if current_account["status"] != "ADMITTED" or account["admission_status"] != "ADMITTED":
            return "UNKNOWN_ACCOUNT", upper
        original_account = stored(account["canonical_account"])
        if any(current_account[name] != original_account[name] for name in ("account_id", "currency", "scale", "scope", "limit_nano_usd", "version")):
            return "AUTHORITY_UNAVAILABLE", upper
        if (deployment["status"] != "ADMITTED" or deployment["tier"] != OPERATIONS[request["operation"]]
                or request["operation"] not in deployment["operations"]
                or not set(request["required_capabilities"]).issubset(deployment["capabilities"])):
            return "UNKNOWN_CAPABILITY", upper
        if request["disclosure_scope"] not in self.registry["disclosure_admissions"][request["input_sha256"]]:
            return "DISCLOSURE_DENIED", upper
        if (tokens["input_tokens"] + request["max_output_tokens"] > min(request["context_tokens"], deployment["max_context_tokens"])
                or request["max_output_tokens"] > deployment["max_output_tokens"]):
            return "TOKEN_ADMISSION_MISMATCH", upper
        if upper is None:
            return "UNKNOWN_PRICE", None
        if upper > MAX_INT:
            return "COST_UNBOUNDED", None
        if (upper > request["max_cost_nano_usd"] or account["spent"] is None or account["account_limit"] is None
                or account["spent"] + account["held"] - own_hold + upper > account["account_limit"]):
            return "BUDGET_EXHAUSTED", upper
        active = connection.execute("SELECT count(*) FROM attempts p JOIN tasks t ON p.task_id=t.task_id WHERE t.deployment_id=? AND p.capacity_slot_held=1", (request["deployment_id"],)).fetchone()[0]
        if deployment["capacity_status"] != "ADMITTED" or active >= deployment["max_parallel_attempts"]:
            return "CAPACITY_UNAVAILABLE", upper
        return None, upper

    def _selected(self, connection, request, now, own_hold=0):
        for name in self.registry["policy"]["routes"][request["operation"]]:
            deployment = self.registry["deployments"][name]
            candidate = dict(request, deployment_id=name, account_id=deployment["account_id"],
                             provider_id=deployment["provider_id"], model_id=deployment["model_id"],
                             price_sha256=deployment["price_sha256"])
            for budget, account in self.registry["accounts"].items():
                if account["account_id"] != deployment["account_id"]:
                    continue
                candidate["budget_id"] = budget
                try:
                    bound = self._binding(candidate)
                except RoutingBrokerRefused:
                    continue
                credit = own_hold if budget == request["budget_id"] and name == request["deployment_id"] else 0
                reason, _ = self._admission(connection, candidate, *bound, now, credit)
                if reason is None:
                    return (budget, name)
        return None

    def _event(self, connection, task, kind, *, completion=None, economic=None, settlement=None,
               charge=None, aggregate=None, attempt=None):
        count = connection.execute("SELECT count(*) FROM events").fetchone()[0]
        require(count < LIMITS["max_events"])
        cursor = connection.execute("INSERT INTO events (task_id,attempt_id,kind,request_sha256,completion_sha256,economic_sha256,settlement_sha256,charge_nano_usd_decimal,aggregate_spent_nano_usd_decimal) VALUES (?,?,?,?,?,?,?,?,?)", (
            task["task_id"], attempt if attempt is not None else task.get("attempt_id"), kind,
            task["request_sha256"], completion, economic, settlement,
            None if charge is None else str(charge), None if aggregate is None else str(aggregate)))
        return cursor.lastrowid

    def _task(self, connection, task_id, request_sha256):
        string(task_id, HEX32)
        string(request_sha256, HEX64)
        task = connection.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        require(task is not None and task["request_sha256"] == request_sha256)
        return dict(task)

    def _attempt_task(self, connection, attempt_id):
        string(attempt_id, HEX32)
        task = connection.execute("SELECT t.* FROM tasks t JOIN attempts p ON t.task_id=p.task_id WHERE p.attempt_id=?", (attempt_id,)).fetchone()
        require(task is not None)
        return dict(task)

    def _receipt(self, connection, task, *, settlement=None):
        request = stored(task["canonical_request"], REQUEST_LIMIT)
        request["request_sha256"] = task["request_sha256"]
        account = connection.execute("SELECT * FROM accounts WHERE budget_id=?", (task["budget_id"],)).fetchone()
        attempt = connection.execute("SELECT * FROM attempts WHERE attempt_id=?", (task["attempt_id"],)).fetchone()
        result = connection.execute("SELECT * FROM results WHERE task_id=?", (task["task_id"],)).fetchone()
        state = task["state"]
        evidence = stored(task["canonical_native_context"])
        evidence.update({"policy_source_sha256": digest(task["canonical_policy"]),
                         "price_source_sha256": digest(task["canonical_price"]), "adapter_sha256": None,
                         "input_token_admission_sha256": digest(task["canonical_token_admission"]),
                         "completion_sha256": task["completion_sha256"], "settlement_id": None,
                         "settlement_sha256": None, "accounting_evidence_sha256": None,
                         "ledger_schema_version": 1, "event_sequence": task["event_sequence"]})
        if settlement is not None:
            evidence.update(settlement)
        elif task["latest_receipt"]:
            previous = stored(task["latest_receipt"], 256 * 1024)["evidence"]
            for name in ("settlement_id", "settlement_sha256", "accounting_evidence_sha256"):
                evidence[name] = previous[name]
        receipt = {
            "schema_version": 1, "scientific_effect": "NONE", "scientific_status_authority": False,
            "custody": "unknown", "kind": "TEST_BROKER_RECEIPT", "request": request,
            "admission": {"status": "DENIED" if state is None else "ADMITTED",
                          "tier": OPERATIONS[request["operation"]], "reason": task["admission_reason"],
                          "policy_sha256": request["policy_sha256"], "price_sha256": request["price_sha256"]},
            "delivery": {"state": state, "attempt_id": task["attempt_id"], "transport_kind": "TEST",
                         "transport_attempts": None if state in ("dispatching", "unknown_delivery") else int(state == "completed"),
                         "native_provider_request_id": None if attempt is None else attempt["native_provider_request_id"],
                         "outcome": task["outcome"]},
            "budget": {"currency": "USD", "scale": 9, "account_limit": account["account_limit"],
                       "reservation_upper_bound": task["reservation_upper_bound"], "held": account["held"],
                       "spent": account["spent"], "actual_charge": task["actual_charge"],
                       "billing_status": task["billing_status"], "invariant_status": account["invariant_status"]},
            "result": {"sha256": None if result is None else result["result_sha256"],
                       "bytes": None if result is None else result["result_bytes"], "acceptance": "NOT_ASSESSED"},
            "evidence": evidence,
        }
        raw = canonical(receipt)
        require(len(raw) <= 256 * 1024)
        return raw

    def _save_receipt(self, connection, task, *, settlement=None):
        raw = self._receipt(connection, task, settlement=settlement)
        connection.execute("UPDATE tasks SET latest_receipt=? WHERE task_id=?", (raw, task["task_id"]))
        return raw

    @opaque
    def reserve(self, request):
        request, raw = snapshot(request, REQUEST_LIMIT)
        validate_request(request)
        with self._transaction() as connection:
            original = connection.execute("SELECT * FROM tasks WHERE task_id=?", (request["task_id"],)).fetchone()
            if original is not None:
                require(original["canonical_request"] == raw)
                return original["original_reserve_receipt"]
            deployment, card, tokens = self._binding(request)
            require(connection.execute("SELECT count(*) FROM tasks").fetchone()[0] < LIMITS["max_tasks"])
            now = self._now()
            reason, upper = self._admission(connection, request, deployment, card, tokens, now)
            selected = self._selected(connection, request, now)
            if selected is not None and selected != (request["budget_id"], request["deployment_id"]):
                reason = "ROUTE_MISMATCH"
            elif selected is None and reason is None:
                reason = "ROUTE_MISMATCH"
            task = {
                "task_id": request["task_id"], "budget_id": request["budget_id"], "deployment_id": request["deployment_id"],
                "request_sha256": digest(raw), "input_sha256": request["input_sha256"], "canonical_request": raw,
                "canonical_policy": canonical(self.registry["policy"]), "canonical_price": canonical(card),
                "canonical_token_admission": canonical(tokens), "canonical_native_context": canonical(self.registry["native_context"]),
                "canonical_deployment": canonical(deployment), "registry_object_sha256": self.registry_sha256,
                "state": None if reason is not None else "reserved", "attempt_id": None,
                "reservation_upper_bound": upper, "reservation_held": int(reason is None), "reservation_released": 0,
                "spend_applied": 0, "billing_status": "UNKNOWN", "actual_charge": None, "completion_sha256": None,
                "economic_sha256": None, "admission_reason": reason, "outcome": "ERROR" if reason is not None else "PENDING",
                "event_sequence": 0, "original_reserve_receipt": b"", "latest_receipt": b"",
                "unknown_reason": None, "unknown_receipt": None, "nondispatch_reason": None,
            }
            if reason is None:
                connection.execute("UPDATE accounts SET held=held+?,version=version+1 WHERE budget_id=?", (upper, task["budget_id"]))
            task["event_sequence"] = self._event(connection, task, "RESERVATION_DENIED" if reason is not None else "RESERVED")
            receipt = self._receipt(connection, task)
            task["original_reserve_receipt"] = task["latest_receipt"] = receipt
            columns = tuple(task)
            connection.execute("INSERT INTO tasks (" + ",".join(columns) + ") VALUES (" + ",".join("?" for _ in columns) + ")", tuple(task[name] for name in columns))
            return receipt

    def _dispatch_check(self, connection, task, now):
        request = stored(task["canonical_request"], REQUEST_LIMIT)
        deployment, card, tokens = self._binding(request)
        require(task["canonical_policy"] == canonical(self.registry["policy"]))
        require(task["canonical_price"] == canonical(card) and task["canonical_token_admission"] == canonical(tokens))
        require(task["canonical_deployment"] == canonical(deployment))
        reason, upper = self._admission(connection, request, deployment, card, tokens, now, task["reservation_upper_bound"])
        require(upper == task["reservation_upper_bound"])
        selected = self._selected(connection, request, now, task["reservation_upper_bound"])
        if reason is None and selected != (task["budget_id"], task["deployment_id"]):
            reason = "ROUTE_MISMATCH"
        return reason

    @opaque
    def claim_dispatch(self, task_id, request_sha256):
        with self._transaction() as connection:
            task = self._task(connection, task_id, request_sha256)
            require(task["state"] == "reserved" and task["attempt_id"] is None)
            now = self._now()
            require(self._dispatch_check(connection, task, now) is None)
            attempt = uuid.uuid4().hex
            task["attempt_id"] = attempt
            task["state"] = "dispatching"
            task["event_sequence"] = self._event(connection, task, "DISPATCH_CLAIMED")
            connection.execute("INSERT INTO attempts VALUES (?,?,?,?,?,?)", (attempt, task_id, request_sha256, task["event_sequence"], 1, None))
            updated = connection.execute("UPDATE tasks SET state='dispatching',attempt_id=?,event_sequence=? WHERE task_id=? AND state='reserved' AND attempt_id IS NULL", (attempt, task["event_sequence"], task_id))
            require(updated.rowcount == 1)
            return self._save_receipt(connection, task)

    def _release(self, connection, task):
        require(task["reservation_held"] == 1 and task["reservation_released"] == 0)
        changed = connection.execute("UPDATE accounts SET held=held-?,version=version+1 WHERE budget_id=? AND held>=?", (task["reservation_upper_bound"], task["budget_id"], task["reservation_upper_bound"]))
        require(changed.rowcount == 1)
        task["reservation_held"], task["reservation_released"] = 0, 1
        connection.execute("UPDATE tasks SET reservation_held=0,reservation_released=1 WHERE task_id=?", (task["task_id"],))

    def _nondispatch_denial(self, connection, task, now):
        # Revocation may remove a binding that dispatch rightly refuses to
        # echo. The trusted zero-claim controller can still release this exact
        # original reservation with the applicable closed revocation reason.
        request = stored(task["canonical_request"], REQUEST_LIMIT)
        if digest(canonical(self.registry["policy"])) != request["policy_sha256"]:
            return "UNKNOWN_POLICY"
        current_account = self.registry["accounts"].get(task["budget_id"])
        if current_account is None or current_account["account_id"] != request["account_id"]:
            return "UNKNOWN_ACCOUNT"
        current = self.registry["deployments"].get(task["deployment_id"])
        if current is None:
            return "UNKNOWN_CAPABILITY"
        if current["price_sha256"] != request["price_sha256"]:
            return "UNKNOWN_PRICE"
        original = stored(task["canonical_deployment"])
        if current["tokenizer_sha256"] != original["tokenizer_sha256"]:
            return "TOKEN_ADMISSION_MISMATCH"
        admitted = self.registry["input_token_admissions"].get(request["input_sha256"], {})
        token = admitted.get(task["deployment_id"], {}).get(current["tokenizer_sha256"])
        if token is None:
            return "UNKNOWN_TOKEN_ADMISSION"
        if canonical(token) != task["canonical_token_admission"]:
            return "TOKEN_ADMISSION_MISMATCH"
        if request["input_sha256"] not in self.registry["disclosure_admissions"]:
            return "DISCLOSURE_DENIED"
        if canonical(current) != task["canonical_deployment"]:
            if any(current[name] != original[name] for name in ("capacity_status", "max_parallel_attempts")):
                return "CAPACITY_UNAVAILABLE"
            return "UNKNOWN_CAPABILITY"
        return self._dispatch_check(connection, task, now)

    @opaque
    def confirm_not_dispatched(self, task_id, request_sha256, reason):
        require(type(reason) is str and reason in DENIALS - {"INVALID_INPUT", "REQUEST_CONFLICT"})
        with self._transaction() as connection:
            task = self._task(connection, task_id, request_sha256)
            if task["state"] == "confirmed_not_dispatched":
                require(task["nondispatch_reason"] == reason)
                return task["latest_receipt"]
            require(task["state"] == "reserved" and task["attempt_id"] is None)
            now = self._now()
            require(self._nondispatch_denial(connection, task, now) == reason)
            self._release(connection, task)
            task["state"] = "confirmed_not_dispatched"
            task["outcome"] = "ERROR"
            task["event_sequence"] = self._event(connection, task, "NONDISPATCH_CONFIRMED")
            connection.execute("UPDATE tasks SET state=?,outcome=?,event_sequence=?,nondispatch_reason=? WHERE task_id=?", (task["state"], task["outcome"], task["event_sequence"], reason, task_id))
            return self._save_receipt(connection, task)

    @opaque
    def mark_unknown(self, attempt_id, reason):
        require(type(reason) is str and reason in UNKNOWN_REASONS)
        with self._transaction() as connection:
            task = self._attempt_task(connection, attempt_id)
            if task["state"] == "unknown_delivery":
                require(task["unknown_reason"] == reason)
                return task["unknown_receipt"]
            require(task["state"] == "dispatching")
            task["state"], task["outcome"] = "unknown_delivery", "UNKNOWN"
            task["event_sequence"] = self._event(connection, task, "UNKNOWN_MARKED")
            connection.execute("UPDATE tasks SET state=?,outcome=?,event_sequence=?,unknown_reason=? WHERE task_id=?", (task["state"], task["outcome"], task["event_sequence"], reason, task["task_id"]))
            raw = self._save_receipt(connection, task)
            connection.execute("UPDATE tasks SET unknown_receipt=? WHERE task_id=?", (raw, task["task_id"]))
            return raw

    def _freeze(self, connection, task, *, violated=False):
        account = connection.execute("SELECT * FROM accounts WHERE budget_id=?", (task["budget_id"],)).fetchone()
        invariant = "VIOLATED" if violated else account["invariant_status"]
        if account["admission_status"] != "FROZEN" or account["invariant_status"] != invariant:
            sequence = self._event(connection, task, "ACCOUNT_FROZEN", economic=task["economic_sha256"])
            connection.execute("UPDATE accounts SET admission_status='FROZEN',invariant_status=?,version=version+1 WHERE budget_id=?", (invariant, task["budget_id"]))
            task["event_sequence"] = sequence
            connection.execute("UPDATE tasks SET event_sequence=? WHERE task_id=?", (sequence, task["task_id"]))

    def _apply_accounting(self, connection, task, usage, *, settlement=None):
        economic = digest(canonical({"attempt_id": task["attempt_id"], "usage": usage}))
        previous = connection.execute("SELECT * FROM accounting WHERE attempt_id=?", (task["attempt_id"],)).fetchone()
        if previous is not None:
            require(previous["economic_sha256"] == economic and previous["canonical_usage"] == canonical(usage))
            return previous["applied_event_sequence"]
        require(task["spend_applied"] == 0)
        card = stored(task["canonical_price"])
        tokens = stored(task["canonical_token_admission"])
        request = stored(task["canonical_request"], REQUEST_LIMIT)
        n, c, w, o = (usage[name] for name in ("input_tokens", "cached_input_tokens", "cache_write_tokens", "output_tokens"))
        charge = usage["actual_charge_nano_usd"]
        calculated = (((n-c-w) * card["input_rate_per_million"] + c * card["cached_input_rate_per_million"]
                      + w * card["cache_write_rate_per_million"] + 999999) // 1000000
                      + (o * card["output_rate_per_million"] + 999999) // 1000000 + card["fixed_request_charge"])
        account = connection.execute("SELECT * FROM accounts WHERE budget_id=?", (task["budget_id"],)).fetchone()
        aggregate = int(account["aggregate_spent_decimal"]) + charge
        task["economic_sha256"], task["actual_charge"] = economic, charge
        task["spend_applied"], task["billing_status"] = 1, "KNOWN"
        sequence = self._event(connection, task, "ACCOUNTING_APPLIED", economic=economic,
                               completion=task["completion_sha256"], settlement=settlement,
                               charge=charge, aggregate=aggregate)
        connection.execute("INSERT INTO accounting VALUES (?,?,?,?,?,?)", (task["attempt_id"], economic,
                           canonical(usage), usage["accounting_source_sha256"], charge, sequence))
        connection.execute("UPDATE accounts SET spent=?,aggregate_spent_decimal=?,version=version+1 WHERE budget_id=?", (aggregate if aggregate <= MAX_INT else None, str(aggregate), task["budget_id"]))
        connection.execute("UPDATE tasks SET economic_sha256=?,actual_charge=?,spend_applied=1,billing_status='KNOWN' WHERE task_id=?", (economic, charge, task["task_id"]))
        if task["state"] == "completed":
            self._release(connection, task)
        remaining_held = connection.execute("SELECT held FROM accounts WHERE budget_id=?", (task["budget_id"],)).fetchone()[0]
        violated = (n > tokens["input_tokens"] or o > request["max_output_tokens"] or charge != calculated
                    or charge > task["reservation_upper_bound"] or aggregate > MAX_INT
                    or account["account_limit"] is None or aggregate + remaining_held > account["account_limit"])
        task["event_sequence"] = sequence
        connection.execute("UPDATE tasks SET event_sequence=? WHERE task_id=?", (sequence, task["task_id"]))
        if violated:
            self._freeze(connection, task, violated=True)
        return sequence

    @opaque
    def complete(self, attempt_id, outcome, usage):
        packet, raw = snapshot({"attempt_id": attempt_id, "outcome": outcome, "usage": usage}, 512 * 1024)
        string(attempt_id, HEX32)
        outcome, usage = packet["outcome"], packet["usage"]
        fields(outcome, {"status", "result_utf8", "native_provider_request_id", "error_code"})
        require(outcome["status"] in {"RESULT", "PROVIDER_REJECTED", "ERROR"})
        result = outcome["result_utf8"]
        if result is not None:
            require(type(result) is str)
            result = result.encode("utf-8", errors="strict")
            require(len(result) <= LIMITS["max_result_bytes"])
        if outcome["native_provider_request_id"] is not None:
            string(outcome["native_provider_request_id"], IDENTITY)
        require(outcome["error_code"] is None or outcome["error_code"] in ERROR_CODES)
        require((outcome["status"] == "RESULT" and result is not None and outcome["error_code"] is None)
                or (outcome["status"] != "RESULT" and result is None and outcome["error_code"] is not None))
        if usage is not None:
            validate_usage(usage)
        with self._transaction() as connection:
            task = self._attempt_task(connection, attempt_id)
            existing = connection.execute("SELECT * FROM completions WHERE attempt_id=?", (attempt_id,)).fetchone()
            if existing is not None:
                require(existing["canonical_packet"] == raw)
                return existing["original_receipt"]
            require(task["state"] in ("dispatching", "unknown_delivery"))
            attempt = connection.execute("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
            if attempt["native_provider_request_id"] is not None:
                require(attempt["native_provider_request_id"] == outcome["native_provider_request_id"])
            accounting = connection.execute("SELECT * FROM accounting WHERE attempt_id=?", (attempt_id,)).fetchone()
            if accounting is not None and usage is not None:
                require(accounting["canonical_usage"] == canonical(usage))
            task["state"], task["outcome"], task["completion_sha256"] = "completed", outcome["status"], digest(raw)
            task["event_sequence"] = self._event(connection, task, "COMPLETION_RECORDED", completion=task["completion_sha256"])
            connection.execute("UPDATE tasks SET state='completed',outcome=?,completion_sha256=?,event_sequence=? WHERE task_id=?", (task["outcome"], task["completion_sha256"], task["event_sequence"], task["task_id"]))
            connection.execute("UPDATE attempts SET capacity_slot_held=0,native_provider_request_id=? WHERE attempt_id=?", (outcome["native_provider_request_id"], attempt_id))
            if result is not None:
                connection.execute("INSERT INTO results VALUES (?,?,?,?,?)", (task["task_id"], task["request_sha256"], digest(result), len(result), result))
            if accounting is not None:
                self._release(connection, task)
            elif usage is not None:
                self._apply_accounting(connection, task, usage)
            receipt = self._save_receipt(connection, task)
            connection.execute("INSERT INTO completions VALUES (?,?,?,?)", (attempt_id, task["completion_sha256"], raw, receipt))
            return receipt

    @opaque
    def settle(self, attempt_id, settlement_id, usage, accounting_evidence):
        packet, raw = snapshot({"attempt_id": attempt_id, "settlement_id": settlement_id,
                                "usage": usage, "accounting_evidence": accounting_evidence}, 64 * 1024)
        string(attempt_id, HEX32)
        string(settlement_id, HEX32)
        usage, accounting_evidence = packet["usage"], packet["accounting_evidence"]
        validate_usage(usage)
        validate_evidence(accounting_evidence)
        conflict = False
        with self._transaction() as connection:
            existing = connection.execute("SELECT * FROM settlements WHERE settlement_id=?", (settlement_id,)).fetchone()
            if existing is not None:
                require(existing["canonical_packet"] == raw)
                return existing["original_receipt"]
            task = self._attempt_task(connection, attempt_id)
            require(task["state"] in ("completed", "unknown_delivery"))
            admitted = self.registry["accounting_record_admissions"].get(accounting_evidence["accounting_record_sha256"])
            require(admitted is not None and canonical(admitted) == canonical(accounting_evidence))
            request = stored(task["canonical_request"], REQUEST_LIMIT)
            deployment = stored(task["canonical_deployment"])
            attempt = connection.execute("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
            expected = {name: request[name] for name in ("account_id", "provider_id", "deployment_id", "model_id", "input_sha256", "policy_sha256", "price_sha256")}
            expected.update({"attempt_id": attempt_id, "request_sha256": task["request_sha256"],
                             "tokenizer_sha256": deployment["tokenizer_sha256"],
                             "completion_sha256": task["completion_sha256"],
                             "native_provider_request_id": attempt["native_provider_request_id"],
                             "usage_sha256": digest(canonical(usage)),
                             "accounting_source_sha256": usage["accounting_source_sha256"]})
            require(all(accounting_evidence[name] == value and type(accounting_evidence[name]) is type(value)
                        for name, value in expected.items()))
            economic = digest(canonical({"attempt_id": attempt_id, "usage": usage}))
            previous = connection.execute("SELECT * FROM accounting WHERE attempt_id=?", (attempt_id,)).fetchone()
            if previous is not None and previous["economic_sha256"] != economic:
                self._freeze(connection, task)
                self._save_receipt(connection, task)
                conflict = True
            else:
                settlement_sha256 = digest(raw)
                applied = self._apply_accounting(connection, task, usage, settlement=settlement_sha256)
                receipt = self._save_receipt(connection, task, settlement={
                    "settlement_id": settlement_id, "settlement_sha256": settlement_sha256,
                    "accounting_evidence_sha256": digest(canonical(accounting_evidence))})
                connection.execute("INSERT INTO settlements VALUES (?,?,?,?,?,?,?)", (settlement_id, attempt_id,
                                   settlement_sha256, raw, digest(canonical(accounting_evidence)), applied, receipt))
        if conflict:
            raise RoutingBrokerRefused()
        return receipt

    @opaque
    def read(self, task_id, request_sha256):
        with self._transaction(write=False) as connection:
            return self._task(connection, task_id, request_sha256)["latest_receipt"]

    @opaque
    def read_result(self, task_id, request_sha256):
        with self._transaction(write=False) as connection:
            self._task(connection, task_id, request_sha256)
            result = connection.execute("SELECT result_blob FROM results WHERE task_id=?", (task_id,)).fetchone()
            return None if result is None else result[0]

    @opaque
    def close(self):
        # Connections are operation-local; closing has no recovery/financial effect.
        self.closed = True
