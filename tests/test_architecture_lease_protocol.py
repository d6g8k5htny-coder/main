"""External lease controls; source authoring is not a runtime disposition.

Contract: 252 lines, SHA256 852a5dcf27cf229cb425d535bf85af64c6231a21f0ffc1305a4f3d253726149e.
Every method checks the absent product FIRST. Root must subsequently prepare,
review and bind the closed consumer ABI below. No fallback namespace, controller,
seed, dependency resolution or production/native-provider capability exists here.

RUNTIME ABI (canonical JSON, <=131072 bytes, schema_version=1): exactly
schema_version,mode,image_reference,image_id,source_sha256,contract_sha256,
dependency_tar_sha256,installed_files,namespace,resources,channels,seed_variants,
intent_bindings.
installed_files: the original 171 regular members, each exactly path,size,sha256;
path is relative to /deps. namespace: exactly source,deps,control,fixture_root.
resources: exactly cpu,memory_bytes,pids,tmpfs_bytes,main_capacity_bytes,
full_capacity_bytes,target_capacity_bytes,publication_capacity_bytes,
max_live_children,max_race_workers,method_deadline_s. These are root's admitted
native bindings; tests also observe their own runtime, cgroup and mounted work.
channels: exactly target,controller; fixed /work/lease-fixtures/<mode>/*.sock,
UNIX sockets owned/enrolled by root, never a request-selected URL/DB/callback.
seed_variants: exactly fence_max_minus_one,revision_max_minus_one,
audit_max_minus_one. Each descriptor: exactly path,sha256,continuity_base64url;
root prepares the readonly original DB with valid fixture history/floors before
construction. This source neither creates nor claims those preparations.
intent_bindings maps all exact 60 method IDs to {path,sha256}; path is exactly
/control/intents/<mode>/<method-id>.json. That readonly canonical <=262144-byte
file has exactly schema_version,method_id,source_sha256,intents; intents is a
nonempty <=256 list of base64url original canonical Intent bytes, sorted by
their exact SHA and unique. Root's fixture/controller enrolls those originals
independently before construction, never from product request packets. The
Publisher retains this closed immutable map at construction, joins native
intent_sha256 to its original source/payload/target/expected precondition, and
refuses missing/different bindings without opening the broker ledger. Ordinary
fixtures need their planned a/other intent variants; C59 needs its exact 200
cycle identities, alternating public payloads and independently known heads.
C15 also needs its independently enrolled duplicate-event intent: new intent
and request IDs, payload B, but the original a event ID, so actual membership
can refuse a second append before any send rather than changing that identity.

Trusted UNIX commands use canonical {schema_version,command,method_id,arguments}
and responses {schema_version,ok,result,error}. Controller commands are reset,
calibrate,snapshot,set_head,pause,barrier,barrier_state,release,cancel,join,callback_entry,
race_barrier,dispose. Target commands are send,observe,precondition. They are closed below;
no arbitrary SQL/path/source, broker ledger query, force, retry or product
failpoint command exists. Root implements them in its separate private target
namespace. Product construction receives only the enrolled Publisher object;
it receives no controller handle, target DB path or ledger-aware callback.
pause/barrier/cancel act on actual external callback/target workers only.
snapshot result is exactly attempts,terminal,refs,events,observations,
callback_entries,native_dispatches,workers,capacity_bytes,disk_bytes,preconditions. Rows join
the actual contract target tables; request_bytes and payload_bytes are base64url.
Event rows additionally report stream_id from the actual retained target request;
worker rows report actual pid/alive/exitcode. Counters are by kind, cumulative
since last reset and independently corroborated by retained original rows.
calibrate returns exactly attempts,effects,readback_payload_sha256,worker_pid,
worker_exitcode. barrier_state returns exactly phase,entered,pid,attempt_id and
waits for that actual external dependency entry; release is one-shot. join
returns exactly pid,alive,exitcode,attempt_id from actual process wait/status.
reset/dispose must retain original private DB/observations before fresh reset,
return exactly disposed,retained_original_sha256, and never erase failed work.
These consumer requirements are NOT_PREPARED until root's harness is reviewed.
callback_entry arguments are exactly kind,attempt_id,phase,native_request_sha256;
phase prepare/send, append-only response exactly recorded:true,sequence positive
int. It observes callback entry only; native_dispatches separately report actual
transport invocation, not merely this entry. barrier adds actual kind and
request_base64url (sign has kind None and public signing message); cancellation
adds actual kind/request_base64url/broker_pid, independently observed by the
external controller. Cancelling a never-dispatched attempt requires the actual
broker dead/completed and exact original prepared binding; zero rows alone is
not sufficient. No callback command supplies broker ledger/path/SQL authority.
The precondition barrier phase blocks the external write callback only after
its actual readonly target response and original bound result are retained.

precondition accepts only ref_update/event_append and exact kind,attempt_id,
request_base64url. The separately prepared target verifies independently
enrolled intent/request/target bytes, opens its fixed native DB mode=ro with
query_onlyON, and returns the closed current head/event-membership observation
below. It creates no write attempt/effect/terminal record. Snapshot preconditions
retain exact query_id,query_pid,observation_base64url,connection_mode,query_only,
statements,scientific_effect,scientific_status_authority from that actual reader;
these are external read records, not inserts on its readonly connection. The
private target/controller implementation remains NOT_PREPARED. Publisher has
no ledger path/connection; native Observer times join the callback's actual
query interval to the second BEGIN/resource read before any send-marker COMMIT.

All records are public TEST data, scientific NONE/false, org independence 0.
Production D1/auth/exclusive Git/Work Events/enrollment remains NOT_DEPLOYED.

Retained 2026-10-09 review history: lease_abi_review REQUEST_CHANGES on the full
2206-line/139295-byte predecessor SHA256
1cef94a68ae540f94971a1601eef19818ce6843b5b7a28cfad7b9f993e774ba5.
Two P2 source findings: BUSY classification lacked an exactly-one actual BEGIN
IMMEDIATE/native timeout join; RESOURCE_HELD race losers were counted as
predicate refusals without each operation's native transaction/read trace.
Root assigned this test-file-only correction. The native trace record below now
joins original diagnostics by operation_id/PID/diagnostic SHA, snapshots actual
connection state and reads PRAGMA busy_timeout from that actual connection
before cleanup. Observer reads are separately retained and excluded from the
product operation trace. Classified race refusals retain every joined original;
SELECT reads of the registered resource row need no UPDATE and are never CAS0
by inference. This is source correction/NOT_RUN, not a hosted RED or acceptance.
The accepted 234/171-line contracts, 60 method IDs and first guards stay fixed.

Retained subsequent REQUEST_CHANGES: adapter_negative_log_tests reviewed the
2433-line/152451-byte successor SHA256
4d14992c22e79edc6c8eb1455c048ba8a189fc0ee2a62b09ec5dc89d61e3915b.
Its P2 identified the missing trusted current-target observation required by
accepted contract 234/88867 SHA6ce31363110afdab824a4e75b9c8009bbbbdf8b0a2effc6eca83b981ef56c7ef
and C14: prepare had no observation and observe joined post-attempt records.
Root assigned lease_precondition_abi_author the narrow contract/control ABI
correction. The historical accepted234 identity above remains preserved; the
current amended contract is separately pinned below. The construction-enrolled
precondition method and closed readonly target command return exact current
state before marker, with native timing/binding evidence; later target CAS can
still reject a head move after that read. Prior BEGIN/BUSY/predicate corrections,
60 IDs/first guards, original171, bounds and product absence are preserved.
This is SOURCE_CORRECTION_PREPARED/NOT_RUN, not a runtime RED or acceptance.

Root relayed a subsequent distinct fresh REQUEST_CHANGES on the complete2700
lines/174257 bytes SHA8ec0f1790f99d3ba2ab168cdf3fe2cf74867dabad16dfc2f7c7776aa347068a5
and248-line/96298-byte contract SHA7b4d9b7cf7ed9567bd064ab34a74e9869d6b037175d570ee6a5c1710b8103287:
the precondition callback could block after an earlier live clock sample. The
amended gate freshly samples/evaluates live time after its correctly bound
return, immediately before marker. C30 keeps the original prepare-pause and
recovery literal, then pauses the external callback after genuine native query
and retained result, crosses exact expiry before return and requires EXPIRED,
unchanged original marker/reservation/attempt and zero send/dispatch/effect.
Only the existing trusted barrier gains a precondition phase; the private
controller/runtime remains NOT_PREPARED. This is source correction/NOT_RUN.
"""

import base64
import dataclasses
import hashlib
import importlib
import json
import os
import pathlib
import re
import shutil
import signal
import socket
import sqlite3
import stat
import subprocess
import sys
import time
import types
import unittest


SOURCE = pathlib.Path('/source/tests/test_architecture_lease_protocol.py')
PRODUCT = pathlib.Path('/source/architecture/lease_protocol.py')
ROOT = pathlib.Path('/work/lease-fixtures')
CONTROL = pathlib.Path('/control/lease-fixture-runtime.json')
CONTRACT_SHA = '852a5dcf27cf229cb425d535bf85af64c6231a21f0ffc1305a4f3d253726149e'
IMAGE = 'python:3.11.14-slim-bookworm@sha256:83f339c1be6340ae1096010fdccf6552ac932d8f410d45d206014916bdf37e48'
IMAGE_ID = 'sha256:44d8f4434bcb025b8cb5c3151be689a906223e47cc3146971feccda936d2e82b'
DEPS_TAR_SHA = '4f7da12d310e943c352a131e4e08afb83468d830afb08e3d6eb0f7de63c900be'
I64 = 9223372036854775807
ZERO = '0' * 64
ISSUER = 'test-lease-issuer'
EPOCH = '1' * 32
INSTANCE = '2' * 32
NOW = 1700000000
DOMAIN = b'research-lease/v1\x00'
PAYLOADS = { 'A': b'lease-fixture-payload-A/v1\n', 'B': b'lease-fixture-payload-B/v1\n' }
PAYLOAD_SHAS = {'A': '2866e56346baeb8fce958fd35e35589f5ba47f9e63f3a8cac317658905f5e682',
                'B': 'c5587ba226353afb3e34c7630061d4fd2872ce4a06bd4061ba08c52c38d204c9'}
SCIENCE = {'scientific_effect': 'NONE', 'scientific_status_authority': False}
VECTORS = (
    ('9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60',
     'd75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a', '',
     'e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555f5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b'),
    ('4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb',
     '3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c', '72',
     '92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00'),
    ('c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7',
     'fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025', 'af82',
     '6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac18ff9b538d16f290aeeddd245b8e19db3b0a2e2b64521ddad2adcc558554ec00'),
    ('833fe62409237b9d62ec77587520911e9a759cec1d19755b7da901b96dca3d42',
     'ec172b93ad5e563bf4932c70e1245034c35467ef2efd4d64ebf819683467e2bf',
     'ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f',
     'dc2a4459e7369633a52b1bf277839a00201009a3efbf3ecb69bea2186c26b58909351fc9ac90b3ecfdfbc7c66431e0303dca179c138ac17ad9bef1177331a704'),
)
TABLES = {
    'lease_meta': ('schema_version','db_instance_id','issuer_id','issuer_epoch','registry_sha256','authority_revision','restore_state'),
    'lease_resources': ('resource_id','scope_id','state','last_fence','current_generation','current_revision','current_intent_id','time_highwater_s','legacy_hold','hold_reason','held_from_state'),
    'lease_intents': ('intent_id','resource_id','intent_sha256','expected_native_sha256','intent_bytes'),
    'lease_grants': ('generation','revision','resource_id','intent_id','key_id','principal','session','request_id','request_sha256','issued_at_s','expires_at_s','envelope_bytes'),
    'lease_idempotency': ('principal','request_id','resource_id','operation','request_sha256','result_status','result_bytes'),
    'lease_authority': ('authority_revision','bundle_revision','bundle_bytes','policy_bytes','policy_sha256','legacy_holds_bytes','update_id','update_sha256','result_bytes'),
    'lease_keys': ('authority_revision','key_id','public_key','activated_at_s','retired_at_s','revoked'),
    'lease_reservations': ('reservation_id','resource_id','generation','revision','intent_id','publisher_id','delivery_attempt_id','native_request_sha256','native_request_bytes','send_started','state','held_from_state','hold_reason','result_bytes'),
    'lease_outcomes': ('outcome_id','reservation_id','delivery_attempt_id','classification','evidence_sha256','evidence_bytes','result_bytes'),
    'lease_observations': ('evidence_id','reservation_id','event_id','delivery_attempt_id','kind','classification','evidence_sha256','evidence_bytes'),
    'lease_audit': ('sequence','event_id','resource_id','transition','event_sha256','event_bytes'),
    'lease_outbox': ('event_id','sequence','target_bytes','payload_sha256','reference_sha256','state','delivery_attempt_id','send_started','evidence_sha256','evidence_bytes','result_bytes'),
}
GRANT_FIELDS = frozenset('schema_version issuer_id issuer_epoch key_id resource_id scope_id holder_principal holder_session claim_id source_sha256 generation fence revision request_id request_sha256 intent_id intent_sha256 expected_native_sha256 issued_at_s expires_at_s authorization_policy_sha256 scientific_effect scientific_status_authority'.split())
OBS_FIELDS = frozenset('schema_version operation_id pid stage sqlite_errorcode sqlite_errorname in_transaction update_calls commit_calls changed_rows busy_timeout_ms retries elapsed_ns'.split())
NATIVE_TRACE_FIELDS = frozenset('schema_version operation_id pid diagnostic_sha256 statements statement_times_ns observed_at_ns begin_immediate_indexes other_begin_indexes native_in_transaction native_changed_rows native_busy_timeout_ms observer_reads scientific_effect scientific_status_authority'.split())
NATIVE_PRECONDITION_FIELDS = frozenset('schema_version query_id kind attempt_id request_sha256 target observed_precondition scientific_effect scientific_status_authority'.split())
PRECONDITION_FIELDS = frozenset('schema_version query_id kind attempt_id reservation_id publisher_id native_request_sha256 target observed_precondition scientific_effect scientific_status_authority'.split())
PRECONDITION_RECORD_FIELDS = frozenset('schema_version pid query_started_ns query_finished_ns prepared_call_sha256 native_observation_base64url observed_precondition_base64url returned_precondition_base64url scientific_effect scientific_status_authority'.split())
_NETWORK_ENROLLED = None


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')


def sha(value):
    return hashlib.sha256(value).hexdigest()


def b64(value):
    return base64.urlsafe_b64encode(value).rstrip(b'=').decode('ascii')


def unb64(value):
    if not isinstance(value, str) or '=' in value or re.fullmatch('[A-Za-z0-9_-]*', value) is None:
        raise ValueError('noncanonical public bytes')
    result = base64.b64decode(value + '=' * (-len(value) % 4), altchars=b'-_', validate=True)
    if b64(result) != value:
        raise ValueError('noncanonical public bytes')
    return result


def packet(value):
    if type(value) is not bytes:
        raise AssertionError('result must be immutable bytes')
    parsed = json.loads(value)
    if canonical(parsed) != value:
        raise AssertionError('result must be exact canonical ASCII JSON without LF')
    return parsed


def require(condition, label='RUNTIME_NOT_PREPARED'):
    if not condition:
        raise AssertionError(label)


def sql_tokens(statement):
    """Lexical observation only; never execute, normalize or replace product SQL."""
    require(type(statement) is str and len(statement) <= 262144, 'unbounded native SQL trace')
    tokens = []
    cursor = 0
    while cursor < len(statement):
        character = statement[cursor]
        if character.isspace():
            cursor += 1
        elif statement.startswith('--',cursor):
            stop = statement.find('\n',cursor+2)
            cursor = len(statement) if stop == -1 else stop+1
        elif statement.startswith('/*',cursor):
            stop = statement.find('*/',cursor+2)
            require(stop != -1, 'incomplete native SQL comment')
            cursor = stop+2
        elif character in "'\"`[":
            closing = ']' if character == '[' else character
            kind = 'string' if character == "'" else 'identifier'
            cursor += 1
            value = ''
            while cursor < len(statement):
                if statement[cursor] == closing:
                    if closing != ']' and cursor+1 < len(statement) and statement[cursor+1] == closing:
                        value += closing; cursor += 2
                    else:
                        cursor += 1
                        break
                else:
                    value += statement[cursor]; cursor += 1
            else:
                raise AssertionError('incomplete native SQL literal')
            tokens.append((kind,value.lower() if kind == 'identifier' else value))
        elif character.isalpha() or character == '_':
            stop = cursor+1
            while stop < len(statement) and (statement[stop].isalnum() or statement[stop] in '_$'):
                stop += 1
            tokens.append(('identifier',statement[cursor:stop].lower())); cursor = stop
        elif character.isdigit():
            stop = cursor+1
            while stop < len(statement) and statement[stop].isdigit():
                stop += 1
            tokens.append(('number',statement[cursor:stop])); cursor = stop
        else:
            tokens.append(('symbol',character)); cursor += 1
    if tokens and tokens[-1] == ('symbol',';'):
        tokens.pop()
    return tokens


def begin_indexes(statements):
    immediate = []; other = []
    for index,statement in enumerate(statements):
        tokens = sql_tokens(statement)
        if tokens and tokens[0] == ('identifier','begin'):
            if tokens in ([('identifier','begin'),('identifier','immediate')],
                          [('identifier','begin'),('identifier','immediate'),('identifier','transaction')]):
                immediate.append(index)
            else:
                other.append(index)
    return immediate,other


def resource_predicate_indexes(statements, resource_id):
    """Accept observable single-row SELECT syntax, independent of private code.

    Case/comments/identifier quoting/main qualification/AS alias/column order
    may vary. A real SELECT must read lease_resources, current predicate fields
    (or its row wildcard), and select this exact resource_id; optional LIMIT 1
    is admitted. Literal/comment mentions, COUNT(*), unrelated/filtered rows,
    hidden debug helpers or a product error code do not establish this read.
    """
    matches = []
    current = {'state','last_fence','current_generation','current_revision','current_intent_id','legacy_hold','hold_reason','held_from_state'}
    for index,statement in enumerate(statements):
        tokens = sql_tokens(statement)
        if not tokens or tokens[0] != ('identifier','select'):
            continue
        try:
            from_index = tokens.index(('identifier','from'))
            where_index = tokens.index(('identifier','where'),from_index+1)
        except ValueError:
            continue
        projection = tokens[1:from_index]
        table = tokens[from_index+1:where_index]
        if table[:2] == [('identifier','main'),('symbol','.')]:
            table = table[2:]
        if not table or table[0] != ('identifier','lease_resources'):
            continue
        tail = table[1:]
        alias = 'lease_resources'
        if tail[:1] == [('identifier','as')]:
            tail = tail[1:]
        if tail:
            if len(tail) != 1 or tail[0][0] != 'identifier':
                continue
            alias = tail[0][1]
        columns = []; column = []
        for item in projection+[('symbol',',')]:
            if item == ('symbol',','):
                columns.append(column); column = []
            else:
                column.append(item)
        current_read = False
        for column in columns:
            if ('identifier','as') in column:
                column = column[:column.index(('identifier','as'))]
            if column == [('symbol','*')]:
                current_read = True
            elif len(column)==1 and column[0][0]=='identifier' and column[0][1] in current:
                current_read = True
            elif len(column)==3 and column[0] in (('identifier',alias),('identifier','lease_resources')) and column[1]==('symbol','.'):
                if column[2]==('symbol','*') or (column[2][0]=='identifier' and column[2][1] in current):
                    current_read = True
        if not current_read:
            continue
        predicate = tokens[where_index+1:]
        if predicate[-2:] == [('identifier','limit'),('number','1')]:
            predicate = predicate[:-2]
        predicate = [item for item in predicate if item not in (('symbol','('),('symbol',')'))]
        columns = ([('identifier','resource_id')],
                   [('identifier',alias),('symbol','.'),('identifier','resource_id')],
                   [('identifier','lease_resources'),('symbol','.'),('identifier','resource_id')])
        literal = [('string',resource_id)]
        if any(predicate == column+[('symbol','=')]+literal or predicate == literal+[('symbol','=')]+column for column in columns):
            matches.append(index)
    return matches


def missing_module_first(case):
    try:
        identity = PRODUCT.lstat()
    except OSError:
        case.fail('LEASE_PROTOCOL_REQUIRED: architecture/lease_protocol.py is absent')
    if identity.st_mode & 0o170000 != 0o100000 or identity.st_nlink != 1:
        case.fail('LEASE_PROTOCOL_REQUIRED: architecture/lease_protocol.py is absent')


def regular_bytes(path, limit):
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and 0 <= before.st_size <= limit)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(descriptor)
        require((before.st_dev, before.st_ino, before.st_size) == (opened.st_dev, opened.st_ino, opened.st_size))
        with os.fdopen(descriptor, 'rb', closefd=False) as reader:
            content = reader.read(limit + 1)
        after = os.fstat(descriptor)
        require(len(content) == before.st_size and after.st_nlink == 1 and
                (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns))
        return content
    finally:
        os.close(descriptor)


def prepared_runtime():
    try:
        return _prepared_runtime()
    except (OSError,ValueError,KeyError,TypeError,ModuleNotFoundError,AssertionError) as failure:
        if isinstance(failure,AssertionError) and str(failure).startswith('RUNTIME_NOT_PREPARED'):
            raise
        raise AssertionError('RUNTIME_NOT_PREPARED') from None


def _prepared_runtime():
    global _NETWORK_ENROLLED
    mode = os.environ.get('LEASE_TEST_MODE')
    require(mode in ('normal', 'optimized') and os.environ.get('LEASE_FIXTURE_ROOT') == str(ROOT))
    require(sys.flags.isolated == 1 and sys.flags.no_site == 1 and sys.dont_write_bytecode and
            sys.flags.optimize == (1 if mode == 'optimized' else 0))
    require(sys.version_info[:3] == (3, 11, 14) and sqlite3.sqlite_version == '3.40.1')
    require(sys.platform == 'linux' and os.uname().machine == 'x86_64' and os.getuid() == os.getgid() == 65532)
    metadata = packet(regular_bytes(CONTROL, 131072))
    require(set(metadata) == set('schema_version mode image_reference image_id source_sha256 contract_sha256 dependency_tar_sha256 installed_files namespace resources channels seed_variants intent_bindings'.split()))
    require(type(metadata['schema_version']) is int and metadata['schema_version'] == 1 and metadata['mode'] == mode)
    require(metadata['image_reference'] == IMAGE and metadata['image_id'] == IMAGE_ID and
            metadata['contract_sha256'] == CONTRACT_SHA and metadata['dependency_tar_sha256'] == DEPS_TAR_SHA)
    require(metadata['namespace'] == {'source':'/source','deps':'/deps','control':'/control','fixture_root':str(ROOT)})
    source_bytes = regular_bytes(SOURCE, 262144)
    require(metadata['source_sha256'] == sha(source_bytes))
    limits = {'cpu':2,'memory_bytes':2**30,'pids':128,'tmpfs_bytes':64*2**20,
              'main_capacity_bytes':60*2**20,'full_capacity_bytes':4*2**20,
              'target_capacity_bytes':16*2**20,'publication_capacity_bytes':64*2**20,
              'max_live_children':64,'max_race_workers':16,'method_deadline_s':60}
    require(set(metadata['resources']) == set(limits))
    for key, maximum in limits.items():
        actual = metadata['resources'][key]
        require(type(actual) is int and 0 < actual <= maximum)
    require(pathlib.Path('/sys/fs/cgroup/memory.max').read_text().strip() == str(2**30))
    require(pathlib.Path('/sys/fs/cgroup/pids.max').read_text().strip() == '128')
    require(pathlib.Path('/sys/fs/cgroup/cpu.max').read_text().split() == ['200000', '100000'])
    require(os.path.ismount(ROOT))
    fs = os.statvfs(ROOT)
    require(0 < fs.f_blocks * fs.f_frsize <= 60*2**20)
    require(set(metadata['channels']) == {'target','controller'})
    for kind, path in metadata['channels'].items():
        expected = ROOT / mode / (kind + '.sock')
        require(path == str(expected) and stat.S_ISSOCK(expected.lstat().st_mode))
    enrolled = frozenset(metadata['channels'].values())
    if _NETWORK_ENROLLED is None:
        def audit_network(event, arguments):
            if event in ('socket.getaddrinfo','urllib.Request'):
                raise AssertionError('external network activity is forbidden')
            if event == 'socket.__new__' and arguments[1] != socket.AF_UNIX:
                raise AssertionError('only enrolled local UNIX transport is permitted')
            if event == 'socket.connect' and arguments[1] not in enrolled:
                raise AssertionError('unenrolled local channel')
        sys.addaudithook(audit_network)
        _NETWORK_ENROLLED = enrolled
    require(_NETWORK_ENROLLED == enrolled)
    require(set(metadata['seed_variants']) == {'fence_max_minus_one','revision_max_minus_one','audit_max_minus_one'})
    for name, entry in metadata['seed_variants'].items():
        require(set(entry) == {'path','sha256','continuity_base64url'})
        require(entry['path'] == '/control/seeds/' + mode + '/' + name + '.sqlite')
        require(re.fullmatch('[0-9a-f]{64}', entry['sha256']) is not None)
        unb64(entry['continuity_base64url'])
    inventory = {name for name in dir(TestLeaseProtocol) if name.startswith('test_l')}
    require(set(metadata['intent_bindings']) == inventory and len(inventory) == 60)
    for method_id, entry in metadata['intent_bindings'].items():
        require(set(entry) == {'path','sha256'} and entry['path'] == '/control/intents/'+mode+'/'+method_id+'.json')
        require(type(entry['sha256']) is str and re.fullmatch('[0-9a-f]{64}',entry['sha256']) is not None)
    installed = metadata['installed_files']
    require(type(installed) is list and len(installed) == 171)
    seen = set()
    for item in installed:
        require(set(item) == {'path','size','sha256'} and type(item['size']) is int and 0 <= item['size'] <= 16*2**20)
        relative = pathlib.PurePosixPath(item['path'])
        require(not relative.is_absolute() and '..' not in relative.parts and str(relative) not in seen)
        seen.add(str(relative))
        data = regular_bytes(pathlib.Path('/deps') / relative, 16*2**20)
        require(len(data) == item['size'] and sha(data) == item['sha256'])
    if '/deps' not in sys.path:
        sys.path.insert(0, '/deps')
    crypto = importlib.import_module('cryptography')
    cffi = importlib.import_module('cffi')
    parser = importlib.import_module('pycparser')
    ed = importlib.import_module('cryptography.hazmat.primitives.asymmetric.ed25519')
    serialization = importlib.import_module('cryptography.hazmat.primitives.serialization')
    backend = importlib.import_module('cryptography.hazmat.backends.openssl.backend').backend
    require((crypto.__version__,cffi.__version__,parser.__version__) == ('50.0.2','2.1.1','3.01'))
    for module in (crypto,cffi,parser,ed):
        require(pathlib.Path(module.__file__).is_relative_to('/deps'))
    rust = importlib.import_module('cryptography.hazmat.bindings._rust')
    native_cffi = importlib.import_module('_cffi_backend')
    require(sha(regular_bytes(pathlib.Path(rust.__file__),16*2**20)) == 'a9d379995757480a31a7eebbfb2228af93becdedd205e3af0f60ba9648650404')
    require(sha(regular_bytes(pathlib.Path(native_cffi.__file__),2*2**20)) == '192828af4429c83d5cd92c40275ffd3c71459c3dc7a58e43cf4b5c346d78dc8e')
    require(backend.openssl_version_number() == 1073741872 and backend.openssl_version_text().startswith('OpenSSL 4.0.3'))
    if '/source' not in sys.path:
        sys.path.append('/source')
    product = importlib.import_module('architecture.lease_protocol')
    require(pathlib.Path(product.__file__) == PRODUCT)
    return metadata, product, ed, serialization


@dataclasses.dataclass(frozen=True)
class AuthFacts:
    principal: str
    session: str
    roles: frozenset
    allowed_claims: frozenset


class Authorization:
    def __init__(self):
        self.handles = {name: AuthFacts(name,name+'-session',frozenset({role}),frozenset({'test-claim'}))
                        for name,role in (('alice','holder'),('bob','holder'),('publisher','publisher'),('delivery','delivery'),('recovery','recovery'))}

    def resolve(self, context):
        if type(context) is not str or context not in self.handles:
            raise ValueError('unknown public fixture handle')
        return self.handles[context]

    def allow(self, facts, resource_id, operation, claim_id, policy_bytes):
        if claim_id not in facts.allowed_claims:
            return False
        policy = packet(policy_bytes)
        role = {'begin_write':'publisher','record_outcome':'publisher','deliver_audit':'delivery',
                'reconcile':'recovery','replace_authority':'recovery','reconcile_delivery':'recovery'}.get(operation,'holder')
        if operation == 'status':
            role = next(iter(facts.roles)) if len(facts.roles) == 1 else ''
        return role in facts.roles and any(e['principal'] == facts.principal and e['session'] == facts.session
                    and e['resource_id'] == resource_id and e['claim_id'] == claim_id and operation in e['operations']
                    for e in policy['entries'])


class Clock:
    def __init__(self, path):
        self.path = path

    def now_s(self):
        return json.loads(regular_bytes(self.path, 128))['now_s']


class Channel:
    ARGUMENTS = {
        'reset':frozenset(), 'calibrate':frozenset({'source_sha256','payload_sha256','payload_base64url'}),
        'snapshot':frozenset(), 'set_head':frozenset({'target','head_sha256'}),
        'pause':frozenset({'phase'}), 'barrier':frozenset({'phase','attempt_id','kind','request_base64url'}),
        'barrier_state':frozenset({'phase'}), 'release':frozenset({'phase'}),
        'cancel':frozenset({'attempt_id','kind','request_base64url','broker_pid'}), 'join':frozenset({'attempt_id'}),
        'callback_entry':frozenset({'kind','attempt_id','phase','native_request_sha256'}),
        'race_barrier':frozenset({'round_id','parties'}), 'dispose':frozenset(),
        'send':frozenset({'kind','attempt_id','request_base64url'}),
        'observe':frozenset({'kind','attempt_id','observer_record_sha256'}),
        'precondition':frozenset({'kind','attempt_id','request_base64url'}),
    }
    PHASES = frozenset({'sign','prepare','precondition','send','transport','target_before_effect','target_after_effect'})

    def __init__(self, metadata, method_id, controller=False):
        self.path = metadata['channels']['controller' if controller else 'target']
        self.method_id = method_id
        self.controller = controller

    def call(self, command, **arguments):
        require(command in self.ARGUMENTS and set(arguments) == self.ARGUMENTS[command], 'invalid trusted controller command')
        require((command in ('send','observe','precondition')) != self.controller, 'wrong enrolled channel')
        if command == 'precondition':
            require(arguments['kind'] in ('ref_update','event_append'), 'write-only precondition command')
        if 'phase' in arguments:
            require(arguments['phase'] in self.PHASES, 'invalid external phase')
            allowed = {'callback_entry':{'prepare','send'},'barrier':{'sign','prepare','precondition','transport'},
                       'pause':{'target_before_effect','target_after_effect'}}.get(command,self.PHASES)
            require(arguments['phase'] in allowed,'wrong external command phase')
            if command == 'barrier' and arguments['phase'] == 'precondition':
                require(arguments['kind'] in ('ref_update','event_append'), 'write-only precondition barrier')
        if 'kind' in arguments:
            require(arguments['kind'] in ('ref_update','event_append','audit_append') or
                    (command=='barrier' and arguments['phase']=='sign' and arguments['kind'] is None))
        if 'request_base64url' in arguments:
            require(len(unb64(arguments['request_base64url'])) <= 65536)
        if 'broker_pid' in arguments:
            require(type(arguments['broker_pid']) is int and 0 < arguments['broker_pid'] < 2**31)
        if 'parties' in arguments:
            require(type(arguments['parties']) is int and arguments['parties']==16)
        body = canonical({'schema_version':1,'command':command,'method_id':self.method_id,'arguments':arguments})
        require(len(body) <= 65536)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(55)
            connection.connect(self.path)
            connection.sendall(len(body).to_bytes(4,'big') + body)
            header = self._read(connection,4)
            length = int.from_bytes(header,'big')
            require(0 < length <= 4*2**20)
            response = packet(self._read(connection,length))
        require(set(response) == {'schema_version','ok','result','error'} and type(response['ok']) is bool and response['schema_version'] == 1)
        require(response['ok'] and response['error'] is None, 'RUNTIME_NOT_PREPARED: controller response refused')
        if command=='callback_entry':
            require(set(response['result']) == {'recorded','sequence'} and response['result']['recorded'] is True
                    and type(response['result']['sequence']) is int and 0 < response['result']['sequence'] <= 4096)
        return response['result']

    @staticmethod
    def _read(connection, count):
        result = b''
        while len(result) < count:
            part = connection.recv(count-len(result))
            require(bool(part), 'RUNTIME_NOT_PREPARED: truncated controller response')
            result += part
        return result


@dataclasses.dataclass(frozen=True)
class PreparedCall:
    kind: str
    attempt_id: str
    reservation_id: object
    native_request_bytes: bytes
    native_request_sha256: str
    publisher_id: str


class Signer:
    def __init__(self, ed, controller, key='rfc-test1', fault=None, pause=False):
        self.key_id = key
        self.ed = ed
        self.controller = controller
        self.fault = fault
        self.pause = pause
        self.calls = []
        self.timings = []

    def sign(self, message):
        self.calls.append(message)
        if self.pause:
            self.controller.call('barrier',phase='sign',attempt_id=self.key_id,kind=None,request_base64url=b64(message))
        if self.fault == 'raise':
            raise ValueError('public signer failure')
        if self.fault == 'short':
            return b'x'
        index = 1 if self.key_id == 'rfc-test2' else 0
        if self.fault == 'wrong_key':
            index = 2
        started = time.monotonic_ns()
        result = self.ed.Ed25519PrivateKey.from_private_bytes(bytes.fromhex(VECTORS[index][0])).sign(message)
        self.timings.append(time.monotonic_ns()-started)
        return result


class Observer:
    def __init__(self, directory):
        self.directory = directory
        self.active = {}
        self.records = []
        self.statement_count = 0

    def attach(self, connection, operation_id):
        require(isinstance(connection,sqlite3.Connection),'observer requires actual native SQLite connection')
        entry = {'connection':connection,'sql':[],'statement_times_ns':[],'initial_changes':connection.total_changes}
        require(operation_id not in self.active, 'duplicate connection operation ID')
        self.active[operation_id] = entry
        def trace(statement):
            entry['statement_times_ns'].append(time.monotonic_ns())
            entry['sql'].append(statement)
        sqlite3.Connection.set_trace_callback(connection,trace)

    def finish(self, operation_id, record_bytes):
        record = packet(record_bytes)
        require(set(record) == OBS_FIELDS and len(record_bytes) <= 4096, 'invalid observation shape')
        entry = self.active.pop(operation_id)
        statements = list(entry['sql'])
        statement_times = list(entry['statement_times_ns'])
        observed_at_ns = time.monotonic_ns()
        self.statement_count += len(statements)
        conn = entry['connection']
        native_transaction = sqlite3.Connection.in_transaction.__get__(conn)
        native_changes = sqlite3.Connection.total_changes.__get__(conn)-entry['initial_changes']
        cursor = sqlite3.Connection.execute(conn,'PRAGMA busy_timeout')
        timeout_rows = sqlite3.Cursor.fetchall(cursor)
        require(len(timeout_rows)==1 and len(timeout_rows[0])==1 and type(timeout_rows[0][0]) is int,
                'missing native busy_timeout observation')
        native_timeout = timeout_rows[0][0]
        immediate,other = begin_indexes(statements)
        native = {'schema_version':1,'operation_id':operation_id,'pid':os.getpid(),'diagnostic_sha256':sha(record_bytes),
                  'statements':statements,'statement_times_ns':statement_times,'observed_at_ns':observed_at_ns,
                  'begin_immediate_indexes':immediate,'other_begin_indexes':other,
                  'native_in_transaction':native_transaction,'native_changed_rows':native_changes,'native_busy_timeout_ms':native_timeout,
                  'observer_reads':entry['sql'][len(statements):],**SCIENCE}
        require(set(native)==NATIVE_TRACE_FIELDS,'native observation shape changed')
        # Preserve originals before assertions; a failed join cannot erase its evidence.
        with (self.directory / ('observer-'+str(os.getpid())+'.jsonl')).open('ab') as writer:
            writer.write(record_bytes+b'\n')
        with (self.directory / ('trace-'+str(os.getpid())+'.jsonl')).open('ab') as writer:
            writer.write(canonical(native)+b'\n')
        require(record['operation_id'] == operation_id and record['pid'] == os.getpid() and record['schema_version'] == 1)
        require(type(record['in_transaction']) is bool and record['in_transaction'] == native_transaction)
        if record['stage']=='BEGIN' and record['sqlite_errorcode'] is not None and record['sqlite_errorcode'] & 255 == sqlite3.SQLITE_BUSY:
            require(bool(immediate),'claimed BEGIN BUSY without an actual native BEGIN IMMEDIATE')
        require(type(native_timeout) is int and native_timeout==1000 and native['observer_reads']==['PRAGMA busy_timeout'],
                'native SQLite settings observation mismatch')
        updates = sum(s.lstrip().upper().startswith('UPDATE ') for s in statements)
        commits = sum(s.strip().upper() in ('COMMIT','END','END TRANSACTION') for s in statements)
        require(record['update_calls'] == updates and record['commit_calls'] == commits and record['retries'] == 0)
        for field in ('update_calls','commit_calls'):
            require(type(record[field]) is int and 0 <= record[field] <= 256)
        require(record['changed_rows'] is None or (type(record['changed_rows']) is int and 0 <= record['changed_rows'] <= 256))
        require(record['busy_timeout_ms'] is None or record['busy_timeout_ms'] == 1000)
        require(type(record['elapsed_ns']) is int and 0 <= record['elapsed_ns'] <= 60000000000)
        self.records.append(record)


class Publisher:
    publisher_id = 'fixture-publisher'

    def __init__(self, metadata, method_id, source_sha, controller, *, precondition_path):
        self.target = Channel(metadata,method_id)
        self.controller = controller
        require(type(precondition_path) is pathlib.PosixPath and precondition_path.parent.parent == ROOT/metadata['mode']/method_id
                and precondition_path.parent.name.startswith('fixture')
                and precondition_path.name == 'precondition-'+str(os.getpid())+'.jsonl', 'unenrolled public precondition output')
        self.precondition_path = precondition_path
        self.source_sha = source_sha
        self.calls = []
        self.last_result = None
        self.prepared_fault = None
        self.precondition_fault = None
        self.result_fault = None
        self.pause_prepare = False
        self.pause_precondition = False
        self.pause_transport = False
        self._issued_calls = {}
        descriptor = metadata['intent_bindings'][method_id]
        original = regular_bytes(pathlib.Path(descriptor['path']),262144)
        require(sha(original) == descriptor['sha256'])
        enrolled = packet(original)
        require(set(enrolled) == {'schema_version','method_id','source_sha256','intents'} and enrolled['schema_version'] == 1
                and enrolled['method_id'] == method_id and enrolled['source_sha256'] == source_sha)
        require(type(enrolled['intents']) is list and 1 <= len(enrolled['intents']) <= 256)
        bindings = {}
        hashes = []
        for encoded in enrolled['intents']:
            raw = unb64(encoded); require(len(raw) <= 65536)
            intent = packet(raw)
            require(set(intent) == set('schema_version intent_id resource_id scope_id operation_kind native_target expected_native payload_sha256 source_sha256 claim_id'.split()))
            require(intent['source_sha256'] == source_sha and intent['payload_sha256'] in PAYLOAD_SHAS.values())
            require(intent['claim_id'] == 'test-claim' and intent['schema_version'] == 1 and intent['intent_id'].startswith(method_id+':intent:'))
            require(intent['resource_id'] in ('fixture/ref','fixture/ref-other','fixture/events') and intent['scope_id'] == intent['resource_id']+'-scope')
            require(intent['native_target'] == Fixture.target(intent['resource_id']))
            require(intent['operation_kind'] == ('event_append' if intent['resource_id']=='fixture/events' else 'ref_update'))
            if intent['operation_kind']=='ref_update':
                require(set(intent['expected_native']) == {'expected_head_sha256'} and intent['expected_native']['expected_head_sha256'] in (ZERO,*PAYLOAD_SHAS.values()))
            else:
                require(set(intent['expected_native']) == {'append_contract','event_id'} and intent['expected_native']['append_contract']=='exclusive-v1')
            identity = sha(raw); require(identity not in bindings)
            bindings[identity] = raw; hashes.append(identity)
        require(hashes == sorted(hashes))
        self.intent_bindings = types.MappingProxyType(bindings)

    def prepare(self, kind, attempt_id, native_request_bytes, *, reservation_id):
        request = packet(native_request_bytes)
        require(kind in ('ref_update','event_append','audit_append') and request['delivery_attempt_id'] == attempt_id)
        require((kind == 'audit_append' and reservation_id is None) or
                (kind != 'audit_append' and isinstance(reservation_id,str) and bool(reservation_id)))
        payload = unb64(request['payload_base64url'])
        require(sha(payload) == request['payload_sha256'])
        if kind != 'audit_append':
            require(request['intent_sha256'] in self.intent_bindings, 'unenrolled original intent')
            intent = packet(self.intent_bindings[request['intent_sha256']])
            require(intent['operation_kind'] == kind and intent['source_sha256'] == self.source_sha and
                    intent['payload_sha256'] == request['payload_sha256'] and intent['native_target'] == request['target'])
            require(payload in PAYLOADS.values())
            if kind == 'ref_update':
                require(set(request) == set('schema_version delivery_attempt_id intent_sha256 target expected_head_sha256 payload_sha256 payload_base64url'.split()))
                require(request['expected_head_sha256'] == intent['expected_native']['expected_head_sha256'])
            else:
                require(set(request) == set('schema_version delivery_attempt_id intent_sha256 target event_id append_contract payload_sha256 payload_base64url reference_sha256 reference_base64url'.split()))
                require(request['event_id'] == intent['expected_native']['event_id'] and request['append_contract']=='exclusive-v1')
                reference = canonical({'kind':'TEST_LEASE_SOURCE','source_sha256':self.source_sha})
                require(unb64(request['reference_base64url']) == reference and request['reference_sha256'] == sha(reference))
        else:
            require(set(request) == set('schema_version delivery_attempt_id target event_id append_contract payload_sha256 payload_base64url reference_sha256 reference_base64url'.split()))
            reference = unb64(request['reference_base64url'])
            require(sha(reference) == request['reference_sha256'] and request['append_contract']=='exclusive-v1')
        call = PreparedCall(kind,attempt_id,reservation_id,native_request_bytes,sha(native_request_bytes),self.publisher_id)
        require(attempt_id not in self._issued_calls, 'duplicate prepared attempt')
        self._issued_calls[attempt_id] = call
        self.calls.append(('prepare',call))
        self.controller.call('callback_entry',kind=kind,attempt_id=attempt_id,phase='prepare',native_request_sha256=sha(native_request_bytes))
        if self.pause_prepare:
            self.controller.call('barrier',phase='prepare',attempt_id=attempt_id,kind=kind,request_base64url=b64(native_request_bytes))
        field_values = {'kind':'audit_append','reservation_id':None,'wrong_reservation':'other-reservation',
                        'attempt_id':'other-attempt','native_request_sha256':ZERO,'publisher_id':'other-publisher'}
        if self.prepared_fault:
            name = 'reservation_id' if self.prepared_fault == 'wrong_reservation' else self.prepared_fault
            call = dataclasses.replace(call,**{name:field_values[self.prepared_fault]})
        return call

    @staticmethod
    def call_bytes(call):
        return canonical({'kind':call.kind,'attempt_id':call.attempt_id,'reservation_id':call.reservation_id,
                          'native_request_base64url':b64(call.native_request_bytes),
                          'native_request_sha256':call.native_request_sha256,'publisher_id':call.publisher_id})

    def require_issued_call(self, call):
        require(type(call) is PreparedCall and self._issued_calls.get(call.attempt_id) is call,
                'unissued or changed prepared call')
        require(call.publisher_id == self.publisher_id and call.native_request_sha256 == sha(call.native_request_bytes),
                'prepared call identity changed')

    def precondition(self, call):
        self.require_issued_call(call)
        require(call.kind in ('ref_update','event_append') and type(call.reservation_id) is str and bool(call.reservation_id),
                'write-only precondition binding')
        request = packet(call.native_request_bytes)
        started = time.monotonic_ns()
        observation = self.target.call('precondition',kind=call.kind,attempt_id=call.attempt_id,
                                       request_base64url=b64(call.native_request_bytes))
        finished = time.monotonic_ns()
        original = canonical(observation)
        require(len(original) <= 4096 and set(observation) == NATIVE_PRECONDITION_FIELDS,
                'invalid native precondition observation shape')
        require(type(observation['schema_version']) is int and observation['schema_version'] == 1)
        require(type(observation['query_id']) is str and re.fullmatch('[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}',observation['query_id']) is not None)
        require((observation['kind'],observation['attempt_id'],observation['request_sha256'],observation['target']) ==
                (call.kind,call.attempt_id,call.native_request_sha256,request['target']), 'native precondition binding mismatch')
        require(observation['scientific_effect'] == 'NONE' and observation['scientific_status_authority'] is False)
        current = observation['observed_precondition']
        if call.kind == 'ref_update':
            require(set(current) == {'kind','current_head_sha256'} and current['kind'] == 'ref_head' and
                    type(current['current_head_sha256']) is str and re.fullmatch('[0-9a-f]{64}',current['current_head_sha256']) is not None)
        else:
            require(set(current) == {'kind','append_contract','event_id','event_present'} and current['kind'] == 'exclusive_append'
                    and current['append_contract'] == request['append_contract'] == 'exclusive-v1'
                    and current['event_id'] == request['event_id'] and type(current['event_present']) is bool)
        result = {'schema_version':1,'query_id':observation['query_id'],'kind':call.kind,'attempt_id':call.attempt_id,
                  'reservation_id':call.reservation_id,'publisher_id':call.publisher_id,
                  'native_request_sha256':call.native_request_sha256,'target':request['target'],
                  'observed_precondition':current,**SCIENCE}
        require(set(result) == PRECONDITION_FIELDS)
        actual_result = canonical(result)
        require(len(actual_result) <= 4096)
        if self.precondition_fault:
            values = {'kind':'audit_append','attempt_id':'other-attempt','reservation_id':'other-reservation',
                      'publisher_id':'other-publisher','native_request_sha256':ZERO,'target':Fixture.target('fixture/ref-other'),
                      'observed_precondition':{'kind':'unknown'}}
            result[self.precondition_fault] = values[self.precondition_fault]
        returned = canonical(result)
        require(len(returned) <= 4096)
        record = {'schema_version':1,'pid':os.getpid(),'query_started_ns':started,'query_finished_ns':finished,
                  'prepared_call_sha256':sha(self.call_bytes(call)),'native_observation_base64url':b64(original),
                  'observed_precondition_base64url':b64(actual_result),'returned_precondition_base64url':b64(returned),**SCIENCE}
        require(set(record) == PRECONDITION_RECORD_FIELDS)
        with self.precondition_path.open('ab') as writer:
            writer.write(canonical(record)+b'\n')
        self.calls.append(('precondition',call))
        if self.pause_precondition:
            self.controller.call('barrier',phase='precondition',attempt_id=call.attempt_id,kind=call.kind,
                                 request_base64url=b64(call.native_request_bytes))
        return returned

    def send(self, call):
        self.require_issued_call(call)
        self.calls.append(('send',call))
        self.controller.call('callback_entry',kind=call.kind,attempt_id=call.attempt_id,phase='send',native_request_sha256=call.native_request_sha256)
        if self.pause_transport:
            self.controller.call('barrier',phase='transport',attempt_id=call.attempt_id,kind=call.kind,request_base64url=b64(call.native_request_bytes))
        observation = self.target.call('send',kind=call.kind,attempt_id=call.attempt_id,request_base64url=b64(call.native_request_bytes))
        result = self.evidence(call,observation)
        if self.result_fault == 'variant':
            result = self.evidence(dataclasses.replace(call,kind='ref_update',reservation_id='wrong-reservation')
                                   if call.kind == 'audit_append' else dataclasses.replace(call,kind='audit_append',reservation_id=None),observation)
        elif self.result_fault:
            result[self.result_fault] = ZERO if self.result_fault.endswith('sha256') else 'wrong-public-binding'
        self.last_result = canonical(result)
        return self.last_result

    def observe(self, kind, attempt_id, observer_record_sha256):
        observation = self.target.call('observe',kind=kind,attempt_id=attempt_id,observer_record_sha256=observer_record_sha256)
        content = canonical(observation)
        if observer_record_sha256 is not None:
            require(sha(content) == observer_record_sha256, 'wrong native observer identity')
        return content

    @staticmethod
    def evidence(call, observation):
        request = packet(call.native_request_bytes)
        metadata = observation['native_metadata']
        readback = observation['readback']
        terminal = observation['terminal']
        quiescent = observation['old_attempt_quiescent']
        if call.kind == 'audit_append':
            return {'schema_version':1,'event_id':request['event_id'],'delivery_attempt_id':call.attempt_id,
                    'native_operation_id':None if metadata is None else metadata['operation_id'],
                    'terminal':terminal,'old_attempt_quiescent':quiescent,
                    'native_metadata_sha256':None if metadata is None else sha(canonical(metadata)),
                    'readback_event_id':None if readback is None else readback['event_id'],
                    'readback_payload_sha256':None if readback is None else readback['payload_sha256'],
                    'readback_reference_sha256':None if readback is None else readback['reference_sha256'],
                    'observer_record_sha256':sha(canonical(observation)),**SCIENCE}
        classification = 'UNKNOWN' if not terminal or not quiescent or metadata is None else (
            'APPLIED' if metadata['classification'] == 'APPLIED' else 'DEFINITELY_NOT_APPLIED')
        joined = None if readback is None or metadata is None else {
            'target':request['target'],'operation_id':metadata['operation_id'],'request_sha256':sha(call.native_request_bytes),
            'payload_sha256':request['payload_sha256'],'effect_sha256':readback['effect_sha256'],'event_id':metadata['event_id']}
        return {'schema_version':1,'outcome_id':'outcome:'+call.attempt_id,'reservation_id':call.reservation_id,
                'delivery_attempt_id':call.attempt_id,'publisher_id':call.publisher_id,
                'native_request_sha256':sha(call.native_request_bytes),'classification':classification,
                'native_operation_id':None if metadata is None else metadata['operation_id'],
                'terminal':terminal,'old_attempt_quiescent':quiescent,'readback':joined,
                'payload_sha256':request['payload_sha256'],**SCIENCE}


class Fixture:
    def __init__(self, case, *, suffix='', seed=None, existing=False, full=False, directory=None):
        self.case = case
        self.metadata,self.product,self.ed,self.serialization = prepared_runtime()
        self.method_id = case._testMethodName
        self.base = ROOT / self.metadata['mode'] / self.method_id
        self.base.mkdir(mode=0o700,exist_ok=True)
        self.directory = self.base / ('fixture'+suffix) if directory is None else pathlib.Path(directory)
        require(self.directory.parent == self.base and self.directory.name.startswith('fixture'))
        if not existing:
            self.directory.mkdir(mode=0o700)
        require(self.directory.is_dir() and not self.directory.is_symlink())
        self.db = (self.base/'full-ledger'/'ledger.sqlite') if full else (self.directory/'ledger.sqlite')
        if full:
            require(os.path.ismount(self.db.parent))
            fs = os.statvfs(self.db.parent)
            require(0 < fs.f_blocks*fs.f_frsize <= 4*2**20)
        self.clock_path = self.directory/'clock.json'
        if not existing:
            self.set_time(NOW)
        self.auth = Authorization()
        self.controller = Channel(self.metadata,self.method_id,controller=True)
        self.publisher = Publisher(self.metadata,self.method_id,self.metadata['source_sha256'],self.controller,
                                   precondition_path=self.directory/('precondition-'+str(os.getpid())+'.jsonl'))
        self.signer = Signer(self.ed,self.controller)
        self.observer = Observer(self.directory)
        self.registry = self.make_registry()
        self.policy = self.make_policy()
        self.bundle = self.make_bundle()
        self.continuity = self.make_continuity()
        if seed:
            entry = self.metadata['seed_variants'][seed]
            content = regular_bytes(pathlib.Path(entry['path']),4*2**20)
            require(sha(content) == entry['sha256'])
            self.db.write_bytes(content)
            self.continuity = unb64(entry['continuity_base64url'])
        self.children = []
        self._calibrated = False
        self.cumulative_children = 0
        self.peak_children = 0
        self.start_ns = time.monotonic_ns()
        self.protocol = self.construct()
        self.cold_constructor_ns = time.monotonic_ns()-self.start_ns

    def __enter__(self):
        self.calibrate()
        return self

    def __exit__(self, kind, value, traceback):
        retained = {'schema_version':1,'method_id':self.method_id,'mode':self.metadata['mode'],
                    'elapsed_ns':time.monotonic_ns()-self.start_ns,'broker_cumulative_children':self.cumulative_children,
                    'broker_peak_live_children':self.peak_children,'scientific_effect':'NONE','scientific_status_authority':False,
                    'org_independence':0,'production_scope':'NOT_DEPLOYED'}
        try:
            retained['target'] = self.controller.call('snapshot')
            retained['ledger'] = self.snapshot()
        finally:
            (self.directory/'retained-final.json').write_bytes(canonical(retained))
            for child in self.children:
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=5)
            self.controller.call('dispose')
        self.case.assertLessEqual(retained['elapsed_ns'],60000000000)

    def construct(self, **overrides):
        kwargs = {'db_path':self.db,'registry_bytes':self.registry,'authorization':self.auth,'clock':Clock(self.clock_path),
                  'signer':self.signer,'root_public_key':bytes.fromhex(VECTORS[2][1]),'initial_bundle_bytes':self.bundle,
                  'initial_policy_bytes':self.policy,'publisher':self.publisher,'continuity_bytes':self.continuity,'observer':self.observer}
        kwargs.update(overrides)
        return self.product.LeaseProtocol(**kwargs)

    def set_time(self, value):
        self.clock_path.write_bytes(canonical({'now_s':value}))

    @staticmethod
    def target(resource='fixture/ref'):
        if resource == 'fixture/events':
            return {'kind':'fixture_events','stream_id':'TEST_STREAM'}
        return {'kind':'fixture_ref','repository_id':'TEST_REPOSITORY','ref':'refs/heads/fixture'+('-other' if resource.endswith('-other') else '')}

    def make_registry(self):
        resources = []
        for resource in ('fixture/ref','fixture/ref-other','fixture/events'):
            resources.append({'resource_id':resource,'scope_id':resource+'-scope',
                'aliases':['fixture/ref','FIXTURE/ref','fixture/subtree','refs/heads/fixture'] if resource == 'fixture/ref' else [resource],
                'operation_kind':'event_append' if resource == 'fixture/events' else 'ref_update','native_target':self.target(resource),
                'publisher_id':'fixture-publisher','delivery_target':{'kind':'fixture_events','stream_id':'TEST_AUDIT_STREAM'}})
        return canonical({'schema_version':1,'issuer_id':ISSUER,'issuer_epoch':EPOCH,'resources':resources,**SCIENCE})

    @staticmethod
    def make_policy(revision=1):
        entries = []
        roles = {'alice':['acquire','renew','release','status'],'bob':['acquire','renew','release','status'],
                 'publisher':['begin_write','record_outcome','status'],'delivery':['deliver_audit','status'],
                 'recovery':['reconcile','replace_authority','reconcile_delivery','status']}
        for principal,operations in roles.items():
            for resource in ('fixture/ref','fixture/ref-other','fixture/events'):
                entries.append({'principal':principal,'session':principal+'-session','resource_id':resource,'claim_id':'test-claim','operations':sorted(operations)})
        entries.sort(key=lambda e:(e['principal'],e['session'],e['resource_id'],e['claim_id']))
        return canonical({'schema_version':1,'issuer_id':ISSUER,'issuer_epoch':EPOCH,'policy_revision':str(revision),'entries':entries,**SCIENCE})

    def root_signed(self, payload, domain):
        private = self.ed.Ed25519PrivateKey.from_private_bytes(bytes.fromhex(VECTORS[2][0]))
        return canonical({'payload':payload,'signature_base64url':b64(private.sign(domain+canonical(payload)))})

    def make_bundle(self, revision=1, *, rotate=False, revoked=False, retired=False):
        keys = [{'key_id':'rfc-test1','public_key_base64url':b64(bytes.fromhex(VECTORS[0][1])),
                 'activated_at_s':0,'retired_at_s':NOW if retired else None,'revoked':revoked}]
        if rotate:
            keys.append({'key_id':'rfc-test2','public_key_base64url':b64(bytes.fromhex(VECTORS[1][1])),
                         'activated_at_s':0,'retired_at_s':None,'revoked':False})
        return self.root_signed({'schema_version':1,'issuer_id':ISSUER,'issuer_epoch':EPOCH,'bundle_revision':str(revision),'keys':keys,**SCIENCE},b'research-lease-trust/v1\x00')

    def make_continuity(self, floors=None, authority=0, pending=()):
        return self.root_signed({'schema_version':1,'db_instance_id':INSTANCE,'issuer_id':ISSUER,'issuer_epoch':EPOCH,
            'authority_revision_floor':str(authority),'resource_fence_floors':floors or {r:'0' for r in ('fixture/ref','fixture/ref-other','fixture/events')},
            'pending_delivery_ids':sorted(pending),**SCIENCE},b'research-lease-continuity/v1\x00')

    def request(self, suffix='a', *, resource='fixture/ref', alias=None, payload='A', ttl=120, head=ZERO):
        intent = {'schema_version':1,'intent_id':self.method_id+':intent:'+suffix,'resource_id':resource,'scope_id':resource+'-scope',
                  'operation_kind':'event_append' if resource == 'fixture/events' else 'ref_update','native_target':self.target(resource),
                  'expected_native':{'append_contract':'exclusive-v1','event_id':self.method_id+':event:'+suffix} if resource == 'fixture/events' else {'expected_head_sha256':head},
                  'payload_sha256':PAYLOAD_SHAS[payload],'source_sha256':self.metadata['source_sha256'],'claim_id':'test-claim'}
        return canonical({'schema_version':1,'request_id':self.method_id+':'+suffix,'operation':'acquire',
                          'resource_alias':alias or resource,'ttl_s':ttl,'intent':intent})

    def acquire(self, suffix='a', context='alice', **kwargs):
        request = self.request(suffix,**kwargs)
        grant = self.protocol.acquire(request,context)
        return grant,request

    def renew(self, grant, suffix='renew', ttl=120, context='alice'):
        request = canonical({'schema_version':1,'request_id':self.method_id+':'+suffix,'operation':'renew','ttl_s':ttl})
        return self.protocol.renew(grant,request,context),request

    def release(self, grant, suffix='release', context='alice'):
        request = canonical({'schema_version':1,'request_id':self.method_id+':'+suffix,'operation':'release'})
        return self.protocol.release(grant,request,context),request

    def update(self, suffix='authority', *, bundle=None, policy=None, legacy=False):
        update = canonical({'schema_version':1,'update_id':self.method_id+':'+suffix,
                            'bundle':packet(bundle or self.make_bundle(2)), 'policy':packet(policy or self.make_policy(2)),
                            'legacy_holds':{r:(legacy and r == 'fixture/ref') for r in ('fixture/ref','fixture/ref-other','fixture/events')}})
        return self.protocol.replace_authority(update,'recovery'),update

    def rows(self, table):
        require(table in TABLES, 'unregistered observation table')
        with sqlite3.connect('file:'+str(self.db)+'?mode=ro',uri=True) as connection:
            connection.execute('PRAGMA query_only=ON')
            connection.row_factory = sqlite3.Row
            return [dict(row) for row in connection.execute('SELECT '+','.join(TABLES[table])+' FROM '+table)]

    def snapshot(self):
        return {table:[{k:(b64(v) if type(v) is bytes else v) for k,v in row.items()} for row in self.rows(table)] for table in TABLES}

    def resource(self, name='fixture/ref'):
        return next(row for row in self.rows('lease_resources') if row['resource_id'] == name)

    def integrity(self):
        with sqlite3.connect('file:'+str(self.db)+'?mode=ro',uri=True) as connection:
            connection.execute('PRAGMA query_only=ON')
            self.case.assertEqual(connection.execute('PRAGMA integrity_check').fetchall(),[('ok',)])
        self.case.assertFalse(pathlib.Path(str(self.db)+'-journal').exists())

    def grant_event(self, grant):
        matches = [row for row in self.rows('lease_audit') if packet(row['event_bytes'])['grant_sha256'] == sha(grant)
                   and row['transition'] in ('ACQUIRE','RENEW')]
        self.case.assertEqual(len(matches),1)
        return matches[0]['event_id']

    def ack(self, grant):
        event = self.grant_event(grant)
        result = self.protocol.deliver_audit(event,'delivery')
        self.case.assertEqual(packet(result)['state'],'ACKNOWLEDGED')
        return event,result

    def begin(self, grant, request):
        return self.protocol.begin_write(grant,canonical(packet(request)['intent']),'publisher')

    def terminal(self, reservation):
        self.case.assertIsNotNone(self.publisher.last_result)
        return self.protocol.record_outcome(packet(reservation)['reservation_id'],self.publisher.last_result,'publisher')

    def evidence(self, reservation, suffix='reconcile'):
        reserved = packet(reservation)
        row = next(r for r in self.rows('lease_reservations') if r['reservation_id'] == reserved['reservation_id'])
        call = PreparedCall(packet(row['native_request_bytes'])['target']['kind'] == 'fixture_ref' and 'ref_update' or 'event_append',
                            row['delivery_attempt_id'],row['reservation_id'],row['native_request_bytes'],row['native_request_sha256'],row['publisher_id'])
        observation = packet(self.publisher.observe(call.kind,call.attempt_id,None))
        outcome = Publisher.evidence(call,observation)
        return canonical({'schema_version':1,'evidence_id':self.method_id+':'+suffix,'native_outcome':outcome,'observer_record_sha256':sha(canonical(observation))})

    def target_snapshot(self):
        result = self.controller.call('snapshot')
        require(set(result) == {'attempts','terminal','refs','events','observations','callback_entries','native_dispatches','workers','capacity_bytes','disk_bytes','preconditions'})
        require(type(result['capacity_bytes']) is int and 0 < result['capacity_bytes'] <= 16*2**20)
        require(type(result['disk_bytes']) is int and 0 <= result['disk_bytes'] <= result['capacity_bytes'])
        return result

    def calibrate(self):
        if self._calibrated:
            return self.calibration_observation
        result = self.controller.call('calibrate',source_sha256=self.metadata['source_sha256'],payload_sha256=PAYLOAD_SHAS['A'],payload_base64url=b64(PAYLOADS['A']))
        self.case.assertEqual(set(result),{'attempts','effects','readback_payload_sha256','worker_pid','worker_exitcode'})
        self.case.assertEqual(result['attempts'],1)
        self.case.assertEqual(result['effects'],1)
        self.case.assertEqual(result['readback_payload_sha256'],PAYLOAD_SHAS['A'])
        self.case.assertGreater(result['worker_pid'],0)
        self.case.assertEqual(result['worker_exitcode'],0)
        self.controller.call('reset')
        self.assert_no_write()
        self.calibration_observation = result
        self._calibrated = True
        (self.directory/'calibration.json').write_bytes(canonical(result))
        return result

    def assert_no_write(self):
        observed = self.target_snapshot()
        self.case.assertEqual([r for r in observed['attempts'] if r['kind'] in ('ref_update','event_append')],[])
        self.case.assertEqual([r for r in observed['events'] if r['stream_id'] == 'TEST_STREAM'],[])
        self.case.assertEqual(observed['native_dispatches'].get('ref_update',0),0)
        self.case.assertEqual(observed['native_dispatches'].get('event_append',0),0)

    def error(self, code, callable_, *args, **kwargs):
        with self.case.assertRaises(self.product.LeaseError) as caught:
            callable_(*args,**kwargs)
        failure = caught.exception
        self.case.assertEqual(failure.code,code)
        self.case.assertEqual(failure.args,(failure.code,failure.request_id))
        self.case.assertIsInstance(failure,ValueError)
        self.case.assertLessEqual(set(vars(failure)),{'code','request_id'})
        self.case.assertNotIn('/target/',str(failure))
        self.case.assertNotIn('Traceback',str(failure))
        return failure

    def status(self, resource='fixture/ref', **selectors):
        query = {'schema_version':1,'resource_id':resource,'request_id':None,'reservation_id':None,'event_id':None}
        query.update(selectors)
        return self.protocol.status(canonical(query),'recovery')

    def start_worker(self, operation, *, request=None, grant=None, reservation=None, phase=None, round_id=None):
        identity = self.method_id+':child:'+str(self.cumulative_children)
        config = {'schema_version':1,'method_id':self.method_id,'directory':str(self.directory),
                  'operation':operation,'request':None if request is None else b64(request),
                  'grant':None if grant is None else b64(grant),'reservation':reservation,
                  'phase':phase,'round_id':round_id,'identity':identity}
        path = self.directory/(identity.rsplit(':',1)[-1]+'.child.json')
        path.write_bytes(canonical(config))
        output = self.directory/(path.stem+'.result.json')
        argv = [sys.executable,'-I','-B','-S'] + (['-O'] if self.metadata['mode'] == 'optimized' else []) + [str(SOURCE),'--lease-worker',str(path),str(output)]
        child = subprocess.Popen(argv,stdout=subprocess.DEVNULL,stderr=(self.directory/(path.stem+'.stderr')).open('wb'),env={'LEASE_FIXTURE_ROOT':str(ROOT),'LEASE_TEST_MODE':self.metadata['mode']})
        self.children.append(child)
        self.cumulative_children += 1
        self.peak_children = max(self.peak_children,sum(p.poll() is None for p in self.children))
        self.case.assertLessEqual(self.peak_children,64)
        return child,output

    def join_worker(self, worker):
        child,output = worker
        child.wait(timeout=55)
        self.case.assertEqual(child.returncode,0)
        result = packet(regular_bytes(output,131072))
        self.case.assertEqual(result['pid'],child.pid)
        self.case.assertEqual(result['optimize'],sys.flags.optimize)
        return result

    def wait_phase(self, phase):
        observed = self.controller.call('barrier_state',phase=phase)
        self.case.assertTrue(observed['entered'])
        self.case.assertGreater(observed['pid'],0)
        return observed

    def kill_worker(self, worker):
        child,_ = worker
        child.send_signal(signal.SIGKILL)
        child.wait(timeout=5)
        self.case.assertEqual(child.returncode,-signal.SIGKILL)

    def wait_committed_response_gap(self, worker):
        child,output = worker
        marker = pathlib.Path(str(output)+'.committed')
        deadline = time.monotonic()+55
        while not marker.exists() and child.poll() is None and time.monotonic()<deadline:
            time.sleep(0.005)
        self.case.assertTrue(marker.exists())
        identity = packet(regular_bytes(marker,4096))
        self.case.assertEqual(identity['pid'],child.pid)
        self.case.assertFalse(output.exists())
        return identity

    def assert_begin_busy(self, result):
        pairs = self.operation_observations(result)
        failures = [(diagnostic,native) for diagnostic,native in pairs if diagnostic['sqlite_errorcode'] is not None]
        self.case.assertEqual(len(failures),1)
        observed,native = failures[0]
        self.case.assertEqual(observed['sqlite_errorcode'] & 255,sqlite3.SQLITE_BUSY)
        self.case.assertEqual((observed['stage'],observed['in_transaction'],observed['update_calls'],observed['commit_calls'],
                               observed['changed_rows'],observed['busy_timeout_ms'],observed['retries']),('BEGIN',False,0,0,None,1000,0))
        self.case.assertEqual(len(native['begin_immediate_indexes']),1)
        self.case.assertEqual(native['other_begin_indexes'],[])
        self.case.assertIs(native['native_in_transaction'],False)
        self.case.assertEqual((native['native_busy_timeout_ms'],native['native_changed_rows']),(1000,0))
        readonly_settings = [[('identifier','pragma'),('identifier',name)] for name in ('busy_timeout','foreign_keys','journal_mode','synchronous')]
        actions = [(index,sql_tokens(statement)) for index,statement in enumerate(native['statements'])
                   if sql_tokens(statement) not in readonly_settings]
        self.case.assertTrue(actions)
        self.case.assertEqual(actions[-1][0],native['begin_immediate_indexes'][0])
        return {'diagnostic':observed,'native_trace':native}

    def operation_observations(self, result):
        pid = result['pid']
        self.case.assertIs(type(pid),int)
        self.case.assertIs(type(result['operation_ids']),list)
        self.case.assertGreater(len(result['operation_ids']),0)
        self.case.assertEqual(len(set(result['operation_ids'])),len(result['operation_ids']))
        diagnostics = [packet(line) for line in regular_bytes(self.directory/('observer-'+str(pid)+'.jsonl'),2*2**20).splitlines()]
        traces = [packet(line) for line in regular_bytes(self.directory/('trace-'+str(pid)+'.jsonl'),4*2**20).splitlines()]
        pairs = []
        for operation_id in result['operation_ids']:
            selected = [r for r in diagnostics if r['operation_id']==operation_id]
            native = [r for r in traces if r['operation_id']==operation_id]
            self.case.assertEqual((len(selected),len(native)),(1,1))
            diagnostic,trace = selected[0],native[0]
            self.case.assertEqual(set(diagnostic),OBS_FIELDS)
            self.case.assertEqual(set(trace),NATIVE_TRACE_FIELDS)
            self.case.assertEqual((diagnostic['pid'],trace['pid']),(pid,pid))
            self.case.assertEqual(trace['diagnostic_sha256'],sha(canonical(diagnostic)))
            self.case.assertIs(type(trace['native_in_transaction']),bool)
            self.case.assertIs(type(trace['native_changed_rows']),int)
            self.case.assertIs(type(trace['native_busy_timeout_ms']),int)
            self.case.assertEqual(trace['native_busy_timeout_ms'],1000)
            self.case.assertEqual(trace['observer_reads'],['PRAGMA busy_timeout'])
            self.case.assertIs(type(trace['statement_times_ns']),list)
            self.case.assertEqual(len(trace['statement_times_ns']),len(trace['statements']))
            self.case.assertTrue(all(type(value) is int and value > 0 for value in trace['statement_times_ns']))
            self.case.assertEqual(trace['statement_times_ns'],sorted(trace['statement_times_ns']))
            self.case.assertIs(type(trace['observed_at_ns']),int)
            self.case.assertTrue(all(value <= trace['observed_at_ns'] for value in trace['statement_times_ns']))
            immediate,other = begin_indexes(trace['statements'])
            self.case.assertEqual((trace['begin_immediate_indexes'],trace['other_begin_indexes']),(immediate,other))
            self.case.assertEqual(diagnostic['in_transaction'],trace['native_in_transaction'])
            self.case.assertEqual({k:trace[k] for k in SCIENCE},SCIENCE)
            pairs.append((diagnostic,trace))
        return pairs

    def assert_precondition_in_final_transaction(self, result, row, expected_current):
        """Join an actual readonly target response to the second native transaction.

        The Publisher never receives a ledger handle. Its original query times
        and identity join the independently retained native SQLite trace here;
        no trace string, response or pre-read is send-time target atomicity.
        """
        request = packet(row['native_request_bytes'])
        kind = 'ref_update' if request['target']['kind'] == 'fixture_ref' else 'event_append'
        call = PreparedCall(kind,row['delivery_attempt_id'],row['reservation_id'],row['native_request_bytes'],
                            row['native_request_sha256'],row['publisher_id'])
        records = [packet(line) for line in regular_bytes(self.directory/('precondition-'+str(result['pid'])+'.jsonl'),2*2**20).splitlines()]
        selected = [record for record in records if packet(unb64(record['observed_precondition_base64url']))['attempt_id'] == call.attempt_id]
        self.case.assertEqual(len(selected),1)
        retained = selected[0]
        self.case.assertEqual(set(retained),PRECONDITION_RECORD_FIELDS)
        self.case.assertEqual((retained['schema_version'],retained['pid'],retained['prepared_call_sha256']),
                             (1,result['pid'],sha(Publisher.call_bytes(call))))
        self.case.assertEqual({k:retained[k] for k in SCIENCE},SCIENCE)
        self.case.assertIs(type(retained['query_started_ns']),int)
        self.case.assertIs(type(retained['query_finished_ns']),int)
        self.case.assertGreater(retained['query_started_ns'],0)
        self.case.assertGreaterEqual(retained['query_finished_ns'],retained['query_started_ns'])
        actual = packet(unb64(retained['observed_precondition_base64url']))
        native_observation = packet(unb64(retained['native_observation_base64url']))
        self.case.assertEqual(set(actual),PRECONDITION_FIELDS)
        self.case.assertEqual(set(native_observation),NATIVE_PRECONDITION_FIELDS)
        self.case.assertEqual(actual,{'schema_version':1,'query_id':native_observation['query_id'],'kind':kind,
            'attempt_id':call.attempt_id,'reservation_id':call.reservation_id,'publisher_id':call.publisher_id,
            'native_request_sha256':call.native_request_sha256,'target':request['target'],
            'observed_precondition':expected_current,**SCIENCE})
        self.case.assertEqual(native_observation,{'schema_version':1,'query_id':actual['query_id'],'kind':kind,
            'attempt_id':call.attempt_id,'request_sha256':call.native_request_sha256,'target':request['target'],
            'observed_precondition':expected_current,**SCIENCE})
        target_reads = [record for record in self.target_snapshot()['preconditions'] if record['query_id'] == actual['query_id']]
        self.case.assertEqual(len(target_reads),1)
        target_read = target_reads[0]
        self.case.assertEqual(set(target_read),set('query_id query_pid observation_base64url connection_mode query_only statements scientific_effect scientific_status_authority'.split()))
        self.case.assertEqual(target_read['observation_base64url'],retained['native_observation_base64url'])
        self.case.assertIs(type(target_read['query_pid']),int)
        self.case.assertGreater(target_read['query_pid'],0)
        self.case.assertNotEqual(target_read['query_pid'],result['pid'])
        self.case.assertEqual(target_read['connection_mode'],'ro')
        self.case.assertIs(type(target_read['query_only']),int)
        self.case.assertEqual(target_read['query_only'],1)
        self.case.assertEqual({k:target_read[k] for k in SCIENCE},SCIENCE)
        expected_query = ('SELECT head_sha256 FROM target_refs WHERE target_id = '+repr(sha(canonical(request['target'])))
                          if kind == 'ref_update' else 'SELECT event_id FROM target_events WHERE event_id = '+repr(request['event_id']))
        query_tokens = sql_tokens(expected_query)
        statements = target_read['statements']
        self.case.assertIs(type(statements),list)
        self.case.assertEqual(sum(sql_tokens(statement) == query_tokens for statement in statements),1)
        allowed = [query_tokens,sql_tokens('PRAGMA query_only=ON'),sql_tokens('PRAGMA query_only'),
                   sql_tokens('BEGIN'),sql_tokens('COMMIT'),sql_tokens('ROLLBACK')]
        self.case.assertTrue(all(sql_tokens(statement) in allowed for statement in statements))
        joined = []
        for diagnostic,native in self.operation_observations(result):
            times = native['statement_times_ns']
            preceding = [index for index in native['begin_immediate_indexes'] if times[index] <= retained['query_started_ns']]
            if not preceding or retained['query_finished_ns'] > native['observed_at_ns']:
                continue
            begin_index = preceding[-1]
            reads = [index for index in resource_predicate_indexes(native['statements'],'fixture/events' if kind == 'event_append' else 'fixture/ref')
                     if begin_index < index and times[index] <= retained['query_started_ns']]
            if not reads:
                continue
            boundaries = [index for index,statement in enumerate(native['statements'])
                          if sql_tokens(statement) in (sql_tokens('COMMIT'),sql_tokens('END'),sql_tokens('END TRANSACTION'),sql_tokens('ROLLBACK'))
                          and begin_index < index and times[index] <= retained['query_finished_ns']]
            self.case.assertEqual(boundaries,[])
            self.case.assertEqual(native['other_begin_indexes'],[])
            joined.append({'diagnostic':diagnostic,'native_trace':native,'begin_index':begin_index,'resource_read_indexes':reads})
        self.case.assertEqual(len(joined),1)
        return {'callback_record':retained,'target_read':target_read,'broker_transaction':joined[0]}

    def assert_resource_predicate_refusal(self, result, resource_id='fixture/ref'):
        self.case.assertEqual(result['code'],'RESOURCE_HELD')
        matches = []
        for diagnostic,native in self.operation_observations(result):
            reads = resource_predicate_indexes(native['statements'],resource_id)
            if reads:
                matches.append((diagnostic,native,reads))
        self.case.assertEqual(len(matches),1)
        diagnostic,native,reads = matches[0]
        self.case.assertEqual(diagnostic['stage'],'PREDICATE')
        self.case.assertIsNone(diagnostic['sqlite_errorcode'])
        self.case.assertIsNone(diagnostic['sqlite_errorname'])
        self.case.assertEqual((len(native['begin_immediate_indexes']),native['other_begin_indexes']),(1,[]))
        self.case.assertTrue(all(index>native['begin_immediate_indexes'][0] for index in reads))
        self.case.assertIs(native['native_in_transaction'],True)
        self.case.assertEqual((diagnostic['commit_calls'],diagnostic['retries'],native['native_changed_rows']),(0,0,0))
        self.case.assertIn(diagnostic['changed_rows'],(None,0))
        self.case.assertEqual(self.resource(resource_id)['state'],'LEASED')
        # An UPDATE is neither required nor inferred; this proves the native read.
        return {'diagnostic':diagnostic,'native_trace':native,'resource_predicate_read_indexes':reads}

    def cancel_attempt(self, row, broker_pid):
        request = packet(row['native_request_bytes'])
        kind = 'ref_update' if request['target']['kind']=='fixture_ref' else 'event_append'
        return self.controller.call('cancel',attempt_id=row['delivery_attempt_id'],kind=kind,
                                    request_base64url=b64(row['native_request_bytes']),broker_pid=broker_pid)


def worker_main(config_path, output_path):
    config = packet(regular_bytes(pathlib.Path(config_path),65536))
    case = TestLeaseProtocol(methodName=config['method_id'])
    missing_module_first(case)
    fixture = Fixture(case,existing=True,directory=config['directory'])
    fixture.signer.pause = config['phase'] == 'sign'
    fixture.publisher.pause_prepare = config['phase'] == 'prepare'
    fixture.publisher.pause_precondition = config['phase'] == 'precondition'
    fixture.publisher.pause_transport = config['phase'] == 'transport'
    if config['round_id']:
        fixture.controller.call('race_barrier',round_id=config['round_id'],parties=16)
    result = {'pid':os.getpid(),'optimize':sys.flags.optimize,'result':None,'code':None}
    before_operation = len(fixture.observer.records)
    operation_started_ns = time.monotonic_ns()
    try:
        operation = config['operation']
        request = None if config['request'] is None else unb64(config['request'])
        grant = None if config['grant'] is None else unb64(config['grant'])
        if operation == 'acquire':
            value = fixture.protocol.acquire(request,'alice')
        elif operation == 'begin_write':
            value = fixture.protocol.begin_write(grant,request,'publisher')
        elif operation == 'deliver_audit':
            value = fixture.protocol.deliver_audit(config['reservation'],'delivery')
        elif operation == 'status':
            value = fixture.status()
        else:
            raise AssertionError('unknown external worker operation')
        if config['phase'] == 'response':
            pathlib.Path(str(output_path)+'.committed').write_bytes(canonical({'pid':os.getpid(),'result_sha256':sha(value)}))
            os.kill(os.getpid(),signal.SIGSTOP)
        result['result'] = b64(value)
    except fixture.product.LeaseError as failure:
        result['code'] = failure.code
    result['operation_ids'] = [r['operation_id'] for r in fixture.observer.records[before_operation:]]
    result['elapsed_ns'] = time.monotonic_ns()-operation_started_ns
    pathlib.Path(output_path).write_bytes(canonical(result))


class TestLeaseProtocol(unittest.TestCase):
    # Each first statement remains deliberately visible to source review.
    def test_l01_end_to_end_calibrated_write_and_next_fence(self):
        missing_module_first(self)
        with Fixture(self) as f:
            f.calibrate()
            grant,request = f.acquire()
            payload = packet(grant)['payload']
            self.assertEqual((payload['fence'],payload['revision']),('1','1'))
            f.ack(grant)
            reservation = f.begin(grant,request)
            admitted = packet(reservation)
            self.assertEqual(set(admitted),set('schema_version reservation_id resource_id generation fence revision intent_id intent_sha256 publisher_id delivery_attempt_id native_request_sha256 state send_started scientific_effect scientific_status_authority'.split()))
            self.assertEqual((admitted['state'],admitted['send_started']),('WRITE_INFLIGHT',False))
            call = [v for kind,v in f.publisher.calls if kind == 'prepare' and v.kind == 'ref_update'][0]
            self.assertEqual(call.reservation_id,admitted['reservation_id'])
            self.assertEqual(packet(f.publisher.last_result)['reservation_id'],call.reservation_id)
            terminal = packet(f.terminal(reservation))
            self.assertEqual(set(terminal),set('schema_version reservation_id delivery_attempt_id outcome_id classification resource_id state native_operation_id evidence_sha256 scientific_effect scientific_status_authority'.split()))
            self.assertEqual((terminal['classification'],terminal['state']),('APPLIED','AVAILABLE'))
            target = f.target_snapshot()
            self.assertEqual(len([r for r in target['attempts'] if r['kind'] == 'ref_update']),1)
            self.assertEqual(next(r for r in target['refs'] if r['target_id'] == sha(canonical(f.target())))['head_sha256'],PAYLOAD_SHAS['A'])
            next_grant,_ = f.acquire('next',payload='B',head=PAYLOAD_SHAS['A'])
            self.assertEqual(packet(next_grant)['payload']['fence'],'2')
            self.assertNotEqual(packet(next_grant)['payload']['generation'],payload['generation'])
            for record in (packet(grant)['payload'],admitted,terminal):
                self.assertEqual({k:record[k] for k in SCIENCE},SCIENCE)
            self.assertEqual(len(f.rows('lease_outcomes')),1)
            f.integrity()

    def test_l02_rfc_empty_short_and_long_ordinary_ed25519(self):
        missing_module_first(self)
        _,_,ed,serialization = prepared_runtime()
        invalid = importlib.import_module('cryptography.exceptions').InvalidSignature
        for index,(seed,public,message,signature) in enumerate(VECTORS):
            with self.subTest(vector=index+1):
                private = ed.Ed25519PrivateKey.from_private_bytes(bytes.fromhex(seed))
                self.assertEqual(private.public_key().public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw),bytes.fromhex(public))
                self.assertEqual(private.sign(bytes.fromhex(message)),bytes.fromhex(signature))
                private.public_key().verify(bytes.fromhex(signature),bytes.fromhex(message))
                for wrong in (bytes.fromhex(message)+b'x',DOMAIN+bytes.fromhex(message)):
                    with self.assertRaises(invalid):
                        private.public_key().verify(bytes.fromhex(signature),wrong)
                with self.assertRaises(invalid):
                    ed.Ed25519PublicKey.from_public_bytes(bytes.fromhex(VECTORS[(index+1)%4][1])).verify(bytes.fromhex(signature),bytes.fromhex(message))
                bad = bytearray(bytes.fromhex(signature)); bad[0] ^= 1
                with self.assertRaises(invalid):
                    private.public_key().verify(bytes(bad),bytes.fromhex(message))

    def test_l03_canonical_grant_bytes_and_domain(self):
        missing_module_first(self)
        with Fixture(self) as f:
            request = f.request()
            reordered = json.dumps(packet(request),sort_keys=False,indent=2).encode('ascii')
            grant = f.protocol.acquire(reordered,'alice')
            observed = packet(grant)['payload']
            expected = {'schema_version':1,'issuer_id':ISSUER,'issuer_epoch':EPOCH,'key_id':'rfc-test1',
                'resource_id':'fixture/ref','scope_id':'fixture/ref-scope','holder_principal':'alice','holder_session':'alice-session',
                'claim_id':'test-claim','source_sha256':f.metadata['source_sha256'],'generation':observed['generation'],'fence':'1','revision':'1',
                'request_id':self._testMethodName+':a','request_sha256':sha(request),'intent_id':self._testMethodName+':intent:a',
                'intent_sha256':sha(canonical(packet(request)['intent'])),'expected_native_sha256':sha(canonical({'expected_head_sha256':ZERO})),
                'issued_at_s':NOW,'expires_at_s':NOW+120,'authorization_policy_sha256':sha(f.policy),**SCIENCE}
            private = f.ed.Ed25519PrivateKey.from_private_bytes(bytes.fromhex(VECTORS[0][0]))
            self.assertRegex(observed['generation'],'^[0-9a-f]{32}$')
            self.assertEqual(grant,canonical({'payload':expected,'signature_base64url':b64(private.sign(DOMAIN+canonical(expected)))}))
            self.assertEqual(f.signer.calls,[DOMAIN+canonical(expected)])
            for wrong_domain in (b'',b'research-lease/v1',DOMAIN+b'\n'):
                altered = canonical({'payload':expected,'signature_base64url':b64(private.sign(wrong_domain+canonical(expected)))})
                f.error('SIGNATURE_INVALID',f.protocol.verify_signature,altered,f.bundle)

    def test_l04_strict_request_duplicate_nonfinite_and_types(self):
        missing_module_first(self)
        with Fixture(self) as f:
            f.calibrate()
            request = f.request()
            invalid = [b'{"schema_version":1,"schema_version":1}',
                request.replace(b'"ttl_s":120',b'"ttl_s":NaN'),request.replace(b'"ttl_s":120',b'"ttl_s":Infinity'),
                request.replace(b'"ttl_s":120',b'"ttl_s":120.0'),request.replace(b'"ttl_s":120',b'"ttl_s":true'),
                request.replace(b'"ttl_s":120',b'"ttl_s":null'),request+b' false',
                request.replace(b'"expected_head_sha256":',b'"x":1,"x":2,"expected_head_sha256":')]
            for field in ('schema_version','operation','intent'):
                value = packet(request); value.pop(field); invalid.append(canonical(value))
            value = packet(request); value['unknown'] = 1; invalid.append(canonical(value))
            for index,value in enumerate(invalid):
                with self.subTest(case=index):
                    before = f.snapshot()
                    f.error('REQUEST_INVALID',f.protocol.acquire,value,'alice')
                    self.assertEqual(f.snapshot(),before)
                    self.assertEqual(f.signer.calls,[])
            for wrong_type in (request.decode('ascii'),bytearray(request),None):
                f.error('REQUEST_INVALID',f.protocol.acquire,wrong_type,'alice')
            grant = f.protocol.acquire(request,'alice')
            self.assertEqual(packet(grant)['payload']['fence'],'1')
            f.assert_no_write()

    def test_l05_request_byte_depth_and_identifier_bounds(self):
        missing_module_first(self)
        with Fixture(self) as f:
            legal = f.request()
            padded = legal+b' '*(65536-len(legal))
            grant = f.protocol.acquire(padded,'alice')
            self.assertEqual(packet(grant)['payload']['request_sha256'],sha(legal))
            invalid = [padded+b' ',b'\xff',b'['*16+b'0'+b']'*16,b'['*17+b'0'+b']'*17]
            for field,value in (('request_id','x'*129),('request_id','é'),('request_id','bad\x00id'),('request_id','\ud800')):
                raw = packet(f.request('bad')); raw[field] = value; invalid.append(canonical(raw))
            raw = packet(f.request('bad')); raw['intent']['payload_sha256'] = PAYLOAD_SHAS['A'].upper(); invalid.append(canonical(raw))
            for index,value in enumerate(invalid):
                with self.subTest(boundary=index):
                    f.error('REQUEST_INVALID',f.protocol.acquire,value,'alice')
            self.assertEqual(len(f.rows('lease_grants')),1)

    def test_l06_untrusted_identity_time_path_sql_and_url(self):
        missing_module_first(self)
        with Fixture(self) as f:
            f.calibrate()
            for field,value in {'issuer_id':'caller','holder_principal':'alice','holder_session':'alice-session','key_id':'rfc-test1',
                                'now_s':NOW,'db_path':'/tmp/ledger','url':'https://invalid.example','sql':'UPDATE lease_resources',
                                'import':'os','force':True,'callback':'send','precondition':{'current_head_sha256':ZERO}}.items():
                with self.subTest(field=field):
                    raw = packet(f.request()); raw[field] = value
                    f.error('REQUEST_INVALID',f.protocol.acquire,canonical(raw),'alice')
                    self.assertEqual(f.rows('lease_grants'),[])
                    self.assertEqual(f.signer.calls,[])
            f.assert_no_write()
            grant,_ = f.acquire()
            self.assertEqual(packet(grant)['payload']['holder_principal'],'alice')

    def test_l07_authentication_claim_and_scoped_policy(self):
        missing_module_first(self)
        with Fixture(self) as f:
            f.calibrate()
            for context in (None,{},'unknown','publisher','delivery','recovery'):
                with self.subTest(context=str(context)):
                    f.error('AUTH_DENIED',f.protocol.acquire,f.request(),context)
            for axis in ('principal','session','roles','allowed_claims'):
                facts = f.auth.handles['alice']
                replacement = {'principal':'unlisted','session':'wrong-session','roles':frozenset({'publisher'}),'allowed_claims':frozenset()}[axis]
                f.auth.handles['axis'] = dataclasses.replace(facts,**{axis:replacement})
                f.error('AUTH_DENIED',f.protocol.acquire,f.request(),'axis')
            request = packet(f.request()); request['intent']['claim_id'] = 'unlisted-claim'
            f.error('AUTH_DENIED',f.protocol.acquire,canonical(request),'alice')
            f.assert_no_write()
            grant,_ = f.acquire()
            self.assertEqual(packet(grant)['payload']['claim_id'],'test-claim')
            changed = packet(f.make_policy(2))
            changed['entries'] = [e for e in changed['entries'] if not (e['principal']=='alice' and e['resource_id']=='fixture/ref-other')]
            f.update(policy=canonical(changed))
            f.error('AUTH_DENIED',f.protocol.acquire,f.request('wrong-scope',resource='fixture/ref-other'),'alice')

    def test_l08_alias_overlap_and_disjoint_resources(self):
        missing_module_first(self)
        with Fixture(self) as f:
            first,request = f.acquire()
            for index,alias in enumerate(('FIXTURE/ref','fixture/subtree','refs/heads/fixture')):
                f.error('RESOURCE_HELD',f.protocol.acquire,f.request('alias'+str(index),alias=alias),'bob')
            other,other_request = f.acquire('other',resource='fixture/ref-other',context='bob')
            f.ack(other); result = f.begin(other,other_request); f.terminal(result)
            self.assertEqual(packet(first)['payload']['resource_id'],'fixture/ref')
            self.assertEqual(packet(other)['payload']['resource_id'],'fixture/ref-other')
            changed = packet(request); changed['resource_alias'] = 'FIXTURE/ref'
            f.error('IDEMPOTENCY_CONFLICT',f.protocol.acquire,canonical(changed),'alice')
            bad = packet(f.registry); bad['resources'][1]['aliases'].append('fixture/ref')
            f.error('REQUEST_INVALID',f.construct,registry_bytes=canonical(bad),db_path=f.directory/'bad-registry.sqlite')

    def test_l09_grant_closed_fields_and_scientific_authority(self):
        missing_module_first(self)
        with Fixture(self) as f:
            grant,_ = f.acquire()
            payload = packet(grant)['payload']
            self.assertEqual(set(payload),GRANT_FIELDS)
            self.assertEqual(payload['scientific_effect'],'NONE')
            self.assertIs(payload['scientific_status_authority'],False)
            for field in ('fence','revision'):
                self.assertIs(type(payload[field]),str)
                self.assertRegex(payload[field],'^[1-9][0-9]*$')
            for field,value in (('scientific_status_authority',True),('scientific_effect','SUPPORTED'),('schema_version',True),('fence',1),('fence','01'),('revision','-1'),('fence',str(I64+1))):
                raw = packet(grant); raw['payload'][field] = value
                f.error('REQUEST_INVALID',f.protocol.verify_signature,canonical(raw),f.bundle)
            for remove in ('source_sha256','intent_sha256'):
                raw = packet(grant); raw['payload'].pop(remove)
                f.error('REQUEST_INVALID',f.protocol.verify_signature,canonical(raw),f.bundle)
            raw = packet(grant); raw['payload']['extra'] = None
            f.error('REQUEST_INVALID',f.protocol.verify_signature,canonical(raw),f.bundle)

    def test_l10_signature_length_encoding_and_scalar(self):
        missing_module_first(self)
        with Fixture(self) as f:
            grant,_ = f.acquire()
            self.assertTrue(packet(f.protocol.verify_signature(grant,f.bundle))['cryptographically_valid'])
            for length in (0,1,63,65):
                raw = packet(grant); raw['signature_base64url'] = b64(b'\0'*length)
                f.error('REQUEST_INVALID',f.protocol.verify_signature,canonical(raw),f.bundle)
            for suffix in ('=','+','/'):
                raw = packet(grant); raw['signature_base64url'] += suffix
                f.error('REQUEST_INVALID',f.protocol.verify_signature,canonical(raw),f.bundle)
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_'
            raw = packet(grant); text = raw['signature_base64url']
            raw['signature_base64url'] = text[:-1]+alphabet[(alphabet.index(text[-1]) & 48)+1]
            f.error('REQUEST_INVALID',f.protocol.verify_signature,canonical(raw),f.bundle)
            raw = packet(grant); signature = unb64(raw['signature_base64url'])
            raw['signature_base64url'] = b64(signature[:32]+b'\xff'*32)
            f.error('SIGNATURE_INVALID',f.protocol.verify_signature,canonical(raw),f.bundle)
            raw = packet(grant); raw['signature_base64url'] = b64(bytes.fromhex(VECTORS[1][3]))
            f.error('SIGNATURE_INVALID',f.protocol.verify_signature,canonical(raw),f.bundle)

    def test_l11_wrong_issuer_epoch_and_untrusted_bundle(self):
        missing_module_first(self)
        with Fixture(self) as f:
            grant,_ = f.acquire()
            self.assertTrue(packet(f.product.verify_signature(grant,f.bundle,root_public_key=bytes.fromhex(VECTORS[2][1])))['cryptographically_valid'])
            for field,value in (('issuer_id','other-issuer'),('issuer_epoch','3'*32),('key_id','unknown-key')):
                raw = packet(grant); raw['payload'][field] = value
                f.error('BUNDLE_INVALID',f.protocol.verify_signature,canonical(raw),f.bundle)
            f.error('BUNDLE_INVALID',f.product.verify_signature,grant,f.bundle,root_public_key=bytes.fromhex(VECTORS[0][1]))
            raw = packet(f.bundle)
            raw['signature_base64url'] = b64(f.ed.Ed25519PrivateKey.from_private_bytes(bytes.fromhex(VECTORS[0][0])).sign(b'research-lease-trust/v1\0'+canonical(raw['payload'])))
            f.error('BUNDLE_INVALID',f.protocol.verify_signature,grant,canonical(raw))

    def test_l12_holder_session_and_resource_live_join(self):
        missing_module_first(self)
        with Fixture(self) as f:
            grant,request = f.acquire()
            renew = canonical({'schema_version':1,'request_id':self._testMethodName+':renew','operation':'renew','ttl_s':120})
            f.error('AUTH_DENIED',f.protocol.renew,grant,renew,'bob')
            f.auth.handles['alice-other-session'] = dataclasses.replace(f.auth.handles['alice'],session='other-session')
            f.error('AUTH_DENIED',f.protocol.renew,grant,renew,'alice-other-session')
            f.ack(grant)
            for field,value in (('resource_id','fixture/ref-other'),('scope_id','fixture/ref-other-scope'),('generation','4'*32)):
                raw = packet(grant); raw['payload'][field] = value
                raw['signature_base64url'] = b64(f.ed.Ed25519PrivateKey.from_private_bytes(bytes.fromhex(VECTORS[0][0])).sign(DOMAIN+canonical(raw['payload'])))
                f.error('NOT_CURRENT',f.protocol.begin_write,canonical(raw),canonical(packet(request)['intent']),'publisher')
            self.assertEqual(len(f.rows('lease_reservations')),0)
            f.assert_no_write()

    def test_l13_immutable_intent_and_exact_native_payload(self):
        missing_module_first(self)
        for fault in ('kind','reservation_id','wrong_reservation','attempt_id','native_request_sha256','publisher_id'):
            with self.subTest(prepared=fault),Fixture(self,suffix='-'+fault) as f:
                f.calibrate(); grant,request = f.acquire(); f.ack(grant)
                f.publisher.prepared_fault = fault
                f.error('EVIDENCE_INCOMPLETE',f.begin,grant,request)
                self.assertEqual(f.rows('lease_reservations')[0]['send_started'],0)
                self.assertEqual([c for c in f.publisher.calls if c[0] == 'send' and c[1].kind != 'audit_append'],[])
                f.assert_no_write()
        for fault in ('kind','attempt_id','reservation_id','publisher_id','native_request_sha256','target','observed_precondition'):
            with self.subTest(precondition=fault),Fixture(self,suffix='-precondition-'+fault) as f:
                grant,request = f.acquire(); f.ack(grant)
                f.publisher.precondition_fault = fault
                f.error('EVIDENCE_INCOMPLETE',f.begin,grant,request)
                row = f.rows('lease_reservations')[0]
                self.assertEqual((row['state'],row['send_started']),('WRITE_INFLIGHT',0))
                self.assertEqual([c for c in f.publisher.calls if c[0] == 'send' and c[1].kind != 'audit_append'],[])
                records = [packet(line) for line in regular_bytes(f.publisher.precondition_path,2*2**20).splitlines()]
                selected = [record for record in records if packet(unb64(record['observed_precondition_base64url']))['attempt_id'] == row['delivery_attempt_id']]
                self.assertEqual(len(selected),1)
                self.assertNotEqual(selected[0]['observed_precondition_base64url'],selected[0]['returned_precondition_base64url'])
                self.assertEqual(packet(unb64(selected[0]['native_observation_base64url']))['observed_precondition'],
                                 {'kind':'ref_head','current_head_sha256':ZERO})
                f.assert_no_write()
        with Fixture(self,suffix='-intent') as f:
            grant,request = f.acquire(); f.ack(grant)
            for field,value in (('payload_sha256',PAYLOAD_SHAS['B']),('source_sha256',ZERO),('claim_id','other-claim')):
                raw = packet(request)['intent']; raw[field] = value
                f.error('INTENT_CONFLICT',f.protocol.begin_write,grant,canonical(raw),'publisher')
            raw = packet(request)['intent']; raw['native_target']['ref'] = 'refs/heads/other'
            f.error('INTENT_CONFLICT',f.protocol.begin_write,grant,canonical(raw),'publisher')
            raw = packet(request)['intent']; raw['expected_native']['expected_head_sha256'] = PAYLOAD_SHAS['B']
            f.error('INTENT_CONFLICT',f.protocol.begin_write,grant,canonical(raw),'publisher')
            f.assert_no_write()

    def test_l14_expected_head_is_actual_target_cas(self):
        missing_module_first(self)
        with Fixture(self) as f:
            f.calibrate(); grant,request = f.acquire(); f.ack(grant)
            f.publisher.pause_prepare = True
            child = f.start_worker('begin_write',request=canonical(packet(request)['intent']),grant=grant,phase='prepare')
            f.wait_phase('prepare')
            before_gate = f.rows('lease_reservations')[0]
            self.assertEqual((before_gate['state'],before_gate['send_started']),('WRITE_INFLIGHT',0))
            f.controller.call('set_head',target=f.target(),head_sha256=PAYLOAD_SHAS['B'])
            f.controller.call('release',phase='prepare')
            result = f.join_worker(child)
            self.assertEqual(result['code'],'NATIVE_PRECONDITION')
            self.assertEqual(f.resource()['state'],'WRITE_INFLIGHT')
            row = f.rows('lease_reservations')[0]
            self.assertEqual((row['reservation_id'],row['delivery_attempt_id'],row['native_request_bytes']),
                             (before_gate['reservation_id'],before_gate['delivery_attempt_id'],before_gate['native_request_bytes']))
            self.assertEqual(row['send_started'],0)
            proof = f.assert_precondition_in_final_transaction(result,row,{'kind':'ref_head','current_head_sha256':PAYLOAD_SHAS['B']})
            self.assertEqual(proof['callback_record']['observed_precondition_base64url'],proof['callback_record']['returned_precondition_base64url'])
            self.assertEqual(proof['broker_transaction']['diagnostic']['stage'],'PREDICATE')
            self.assertIs(proof['broker_transaction']['native_trace']['native_in_transaction'],True)
            final_trace = proof['broker_transaction']['native_trace']['statements'][proof['broker_transaction']['begin_index']+1:]
            self.assertFalse(any(sql_tokens(statement) in (sql_tokens('COMMIT'),sql_tokens('END'),sql_tokens('END TRANSACTION'))
                                 for statement in final_trace))
            target = f.target_snapshot()
            self.assertEqual(next(r for r in target['refs'] if r['target_id'] == sha(canonical(f.target())))['head_sha256'],PAYLOAD_SHAS['B'])
            self.assertFalse(any(entry['kind'] == 'ref_update' and entry['phase'] == 'send' for entry in target['callback_entries']))
            (f.directory/'precondition-gate-refusal.json').write_bytes(canonical(proof))
            f.assert_no_write()
        # A genuine native CAS refusal has one dispatch, not zero attempts.
        with Fixture(self,suffix='-native') as f:
            grant,request = f.acquire(); f.ack(grant)
            child = f.start_worker('begin_write',request=canonical(packet(request)['intent']),grant=grant,phase='transport')
            f.wait_phase('transport')
            before_dispatch = f.rows('lease_reservations')[0]
            self.assertEqual(before_dispatch['send_started'],1)
            f.controller.call('set_head',target=f.target(),head_sha256=PAYLOAD_SHAS['B'])
            f.controller.call('release',phase='transport')
            result = f.join_worker(child)
            self.assertIsNone(result['code'])
            reservation = unb64(result['result'])
            row = f.rows('lease_reservations')[0]
            self.assertEqual(row['reservation_id'],before_dispatch['reservation_id'])
            self.assertEqual(row['send_started'],1)
            proof = f.assert_precondition_in_final_transaction(result,row,{'kind':'ref_head','current_head_sha256':ZERO})
            self.assertEqual(proof['callback_record']['observed_precondition_base64url'],proof['callback_record']['returned_precondition_base64url'])
            (f.directory/'precondition-then-native-cas-rejection.json').write_bytes(canonical(proof))
            evidence = f.evidence(reservation)
            self.assertEqual(packet(evidence)['native_outcome']['classification'],'DEFINITELY_NOT_APPLIED')
            terminal = f.protocol.reconcile(packet(reservation)['reservation_id'],evidence,'recovery')
            self.assertEqual(packet(terminal)['state'],'AVAILABLE')
            self.assertEqual(len([r for r in f.target_snapshot()['attempts'] if r['kind']=='ref_update']),1)
            target = f.target_snapshot()
            self.assertEqual(target['native_dispatches'].get('ref_update',0),1)
            self.assertEqual(next(r for r in target['refs'] if r['target_id'] == sha(canonical(f.target())))['head_sha256'],PAYLOAD_SHAS['B'])
            terminal_rows = [entry for entry in target['terminal'] if entry['attempt_id'] == row['delivery_attempt_id']]
            self.assertEqual(len(terminal_rows),1)
            self.assertEqual(terminal_rows[0]['classification'],'REJECTED')

    def test_l15_audit_append_intent_and_event_identity(self):
        missing_module_first(self)
        with Fixture(self) as f:
            grant,request = f.acquire(resource='fixture/events'); f.ack(grant)
            reservation = f.begin(grant,request)
            call = [v for kind,v in f.publisher.calls if kind == 'prepare' and v.kind == 'event_append'][0]
            self.assertEqual(call.reservation_id,packet(reservation)['reservation_id'])
            evidence = packet(f.publisher.last_result)
            self.assertEqual(evidence['reservation_id'],call.reservation_id)
            native = packet(call.native_request_bytes)
            self.assertEqual(native['append_contract'],'exclusive-v1')
            self.assertEqual(native['reference_sha256'],sha(canonical({'kind':'TEST_LEASE_SOURCE','source_sha256':f.metadata['source_sha256']})))
            records = [packet(line) for line in regular_bytes(f.publisher.precondition_path,2*2**20).splitlines()]
            current = [packet(unb64(record['observed_precondition_base64url'])) for record in records
                       if packet(unb64(record['observed_precondition_base64url']))['attempt_id'] == call.attempt_id]
            self.assertEqual(len(current),1)
            self.assertEqual(current[0]['observed_precondition'],{'kind':'exclusive_append','append_contract':'exclusive-v1',
                             'event_id':native['event_id'],'event_present':False})
            terminal = f.terminal(reservation)
            self.assertEqual(packet(terminal)['classification'],'APPLIED')
            effects = [row for row in f.target_snapshot()['events'] if row['stream_id']=='TEST_STREAM']
            self.assertEqual(len(effects),1)
            self.assertEqual(effects[0]['event_id'],packet(request)['intent']['expected_native']['event_id'])
            self.assertEqual(effects[0]['payload_sha256'],PAYLOAD_SHAS['A'])
        with Fixture(self,suffix='-wrong-event') as f:
            grant,request = f.acquire(resource='fixture/events'); f.ack(grant)
            altered = packet(request)['intent']; altered['expected_native']['event_id'] = 'wrong-event'
            f.error('INTENT_CONFLICT',f.protocol.begin_write,grant,canonical(altered),'publisher')
            f.assert_no_write()
            f.publisher.result_fault = 'variant'
            f.error('NATIVE_UNKNOWN',f.begin,grant,request)
            self.assertEqual(f.resource('fixture/events')['state'],'WRITE_UNKNOWN')
            self.assertEqual(len(f.rows('lease_outcomes')),0)
        with Fixture(self,suffix='-present') as f:
            first,first_request = f.acquire(resource='fixture/events'); f.ack(first)
            reservation = f.begin(first,first_request); f.terminal(reservation)
            event_id = packet(first_request)['intent']['expected_native']['event_id']
            duplicate = packet(f.request('duplicate-event',resource='fixture/events',payload='B'))
            duplicate['intent']['expected_native']['event_id'] = event_id
            request = canonical(duplicate)
            grant = f.protocol.acquire(request,'alice'); f.ack(grant)
            before = f.target_snapshot()
            prior_records = len(f.observer.records)
            f.error('NATIVE_PRECONDITION',f.begin,grant,request)
            row = next(row for row in f.rows('lease_reservations') if row['intent_id'] == duplicate['intent']['intent_id'])
            self.assertEqual((row['state'],row['send_started']),('WRITE_INFLIGHT',0))
            result = {'pid':os.getpid(),'operation_ids':[record['operation_id'] for record in f.observer.records[prior_records:]]}
            proof = f.assert_precondition_in_final_transaction(result,row,{'kind':'exclusive_append','append_contract':'exclusive-v1',
                                                                          'event_id':event_id,'event_present':True})
            self.assertEqual(proof['callback_record']['observed_precondition_base64url'],proof['callback_record']['returned_precondition_base64url'])
            after = f.target_snapshot()
            self.assertEqual(after['attempts'],before['attempts'])
            self.assertEqual(after['events'],before['events'])
            self.assertEqual(after['native_dispatches'],before['native_dispatches'])
            self.assertEqual(f.resource('fixture/events')['state'],'WRITE_INFLIGHT')
            (f.directory/'event-present-precondition-refusal.json').write_bytes(canonical(proof))

    def test_l16_acquire_duplicate_and_conflicting_request_id(self):
        missing_module_first(self)
        with Fixture(self) as f:
            grant,request = f.acquire()
            before = f.snapshot(); calls = len(f.signer.calls)
            self.assertEqual(f.protocol.acquire(request,'alice'),grant)
            self.assertEqual(f.snapshot(),before)
            self.assertEqual(len(f.signer.calls),calls)
            for field,value in (('ttl_s',30),('resource_alias','FIXTURE/ref')):
                raw = packet(request); raw[field] = value
                f.error('IDEMPOTENCY_CONFLICT',f.protocol.acquire,canonical(raw),'alice')
            release = canonical({'schema_version':1,'request_id':packet(request)['request_id'],'operation':'release'})
            f.error('IDEMPOTENCY_CONFLICT',f.protocol.release,grant,release,'alice')
            self.assertEqual(f.rows('lease_idempotency')[0]['result_bytes'],grant)

    def test_l17_concurrent_duplicate_commits_one_result(self):
        missing_module_first(self)
        with Fixture(self) as f:
            request = f.request()
            workers = [f.start_worker('acquire',request=request,round_id=self._testMethodName+':same') for _ in range(16)]
            results = [f.join_worker(worker) for worker in workers]
            successes = [unb64(r['result']) for r in results if r['code'] is None]
            self.assertGreaterEqual(len(successes),1)
            self.assertEqual(len(set(successes)),1)
            self.assertEqual(len(f.rows('lease_grants')),1)
            self.assertEqual(len(f.rows('lease_idempotency')),1)
            self.assertEqual(len(f.rows('lease_audit')),1)
            for result in results:
                if result['code']:
                    self.assertEqual(result['code'],'LEDGER_BUSY')
                    f.assert_begin_busy(result)
            self.assertEqual(len({r['pid'] for r in results}),16)
            f.integrity()

    def test_l18_replay_after_expiry_release_and_key_rotation(self):
        missing_module_first(self)
        with Fixture(self) as f:
            grant,request = f.acquire()
            released,release_request = f.release(grant)
            f.update(bundle=f.make_bundle(2,rotate=True,revoked=True))
            f.set_time(NOW+1000)
            before = f.snapshot()
            self.assertEqual(f.protocol.acquire(request,'alice'),grant)
            self.assertEqual(f.protocol.release(grant,release_request,'alice'),released)
            self.assertEqual(f.snapshot(),before)
            self.assertEqual(f.resource()['last_fence'],1)
            f.assert_no_write()

    def test_l19_renew_revision_intent_and_idempotence(self):
        missing_module_first(self)
        with Fixture(self) as f:
            original,request = f.acquire()
            f.set_time(NOW+10)
            renewed,renew_request = f.renew(original,ttl=30)
            old,new = packet(original)['payload'],packet(renewed)['payload']
            self.assertEqual([new[k] for k in ('generation','fence','intent_id','intent_sha256')],[old[k] for k in ('generation','fence','intent_id','intent_sha256')])
            self.assertEqual((new['revision'],new['issued_at_s'],new['expires_at_s']),('2',NOW+10,NOW+40))
            before = f.snapshot()
            self.assertEqual(f.protocol.renew(original,renew_request,'alice'),renewed)
            self.assertEqual(f.snapshot(),before)
            changed = packet(renew_request); changed['ttl_s'] = 120
            f.error('IDEMPOTENCY_CONFLICT',f.protocol.renew,original,canonical(changed),'alice')
            f.ack(renewed)
            f.error('NOT_CURRENT',f.begin,original,request)

    def test_l20_release_and_stale_renew_release_refusal(self):
        missing_module_first(self)
        with Fixture(self) as f:
            original,_ = f.acquire()
            renewed,_ = f.renew(original)
            f.error('NOT_CURRENT',f.release,original,'stale-release')
            released,release_request = f.release(renewed)
            self.assertEqual(packet(released)['state'],'AVAILABLE')
            self.assertEqual(f.resource()['last_fence'],1)
            next_grant,_ = f.acquire('next',context='bob')
            f.error('NOT_CURRENT',f.renew,renewed,'stale-renew')
            self.assertEqual(f.protocol.release(renewed,release_request,'alice'),released)
            f.set_time(packet(next_grant)['payload']['expires_at_s'])
            f.error('EXPIRED',f.release,next_grant,'expired',context='bob')

    def test_l21_eight_rounds_sixteen_process_acquire_cas(self):
        missing_module_first(self)
        method_started = time.monotonic_ns()
        cumulative = 0
        rounds = []
        for round_number in range(8):
            with Fixture(self,suffix='-round'+str(round_number)) as f:
                workers = [f.start_worker('acquire',request=f.request('r'+str(round_number)+'-p'+str(index)),
                                         round_id=self._testMethodName+':round:'+str(round_number)) for index in range(16)]
                results = [f.join_worker(worker) for worker in workers]
                winners = [r for r in results if r['code'] is None]
                self.assertEqual(len(winners),1)
                losers = [r for r in results if r['code'] == 'RESOURCE_HELD']
                busy = [r for r in results if r['code'] == 'LEDGER_BUSY']
                self.assertEqual(len(losers)+len(busy),15)
                self.assertEqual(len({r['pid'] for r in results}),16)
                refusals = []
                for refused in busy:
                    proof = f.assert_begin_busy(refused)
                    refusals.append({'pid':refused['pid'],'classification':'BEGIN_BUSY','proof':proof})
                for refused in losers:
                    proof = f.assert_resource_predicate_refusal(refused)
                    refusals.append({'pid':refused['pid'],'classification':'RESOURCE_PREDICATE_REFUSAL','proof':proof})
                self.assertEqual(len(refusals),15)
                (f.directory/'race-refusals.json').write_bytes(canonical({'round':round_number+1,'refusals':refusals,**SCIENCE}))
                self.assertEqual(len(f.rows('lease_grants')),1)
                self.assertEqual(len(f.rows('lease_idempotency')),1)
                self.assertEqual(len(f.rows('lease_intents')),1)
                self.assertEqual(len(f.rows('lease_audit')),1)
                self.assertEqual(len(f.rows('lease_outbox')),1)
                self.assertEqual(f.rows('lease_grants')[0]['envelope_bytes'],unb64(winners[0]['result']))
                self.assertEqual(f.resource()['last_fence'],1)
                f.integrity()
                cumulative += f.cumulative_children
                rounds.append({'round':round_number+1,'winner_pid':winners[0]['pid'],'predicate_refusals':len(losers),'begin_busy':len(busy),'pids':sorted(r['pid'] for r in results)})
                (f.directory/'race-rounds.json').write_bytes(canonical({'rounds':rounds,'race_children':cumulative,'peak_race_workers':16,'all_process_counts':'ROOT_CONTROLLER_NATIVE_JOIN_REQUIRED'}))
        self.assertEqual(cumulative,128)
        self.assertEqual(len(rounds),8)
        self.assertLessEqual(time.monotonic_ns()-method_started,60000000000)

    def test_l22_busy_timeout_single_attempt_and_rollback(self):
        missing_module_first(self)
        with Fixture(self) as f:
            f.calibrate()
            before = f.snapshot()
            lock = sqlite3.connect(f.db,isolation_level=None)
            try:
                lock.execute('PRAGMA busy_timeout=1000')
                lock.execute('BEGIN IMMEDIATE')
                worker = f.start_worker('acquire',request=f.request('locked'))
                result = f.join_worker(worker)
                self.assertEqual(result['code'],'LEDGER_BUSY')
                proof = f.assert_begin_busy(result)
                failure = proof['diagnostic']
                self.assertGreaterEqual(failure['elapsed_ns'],800000000)
                self.assertLessEqual(failure['elapsed_ns'],2000000000)
                self.assertGreaterEqual(result['elapsed_ns'],800000000)
                self.assertLessEqual(result['elapsed_ns'],2000000000)
                (f.directory/'busy-refusal.json').write_bytes(canonical({'lock_owner_pid':os.getpid(),'worker_elapsed_ns':result['elapsed_ns'],
                    'proof':proof,'experiment':'ONE_ATTEMPT_HELD_EXTERNAL_LOCK',**SCIENCE}))
                self.assertEqual(f.snapshot(),before)
                f.assert_no_write()
            finally:
                lock.rollback(); lock.close()
            fresh,_ = f.acquire('new-experiment')
            self.assertEqual(packet(fresh)['payload']['fence'],'1')

    def test_l23_fence_i64_exhaustion_no_wrap(self):
        missing_module_first(self)
        with Fixture(self,seed='fence_max_minus_one') as f:
            self.assertEqual(f.resource()['last_fence'],I64-1)
            grant,_ = f.acquire()
            self.assertEqual(packet(grant)['payload']['fence'],str(I64))
            f.release(grant)
            before = f.snapshot(); signatures = len(f.signer.calls)
            f.error('COUNTER_EXHAUSTED',f.protocol.acquire,f.request('overflow'),'alice')
            self.assertEqual(f.resource()['last_fence'],I64)
            self.assertEqual(f.snapshot(),before)
            self.assertEqual(len(f.signer.calls),signatures)
            f.assert_no_write()

    def test_l24_revision_and_time_i64_bounds(self):
        missing_module_first(self)
        with Fixture(self,seed='revision_max_minus_one') as f:
            rows = f.rows('lease_grants')
            current = next(r['envelope_bytes'] for r in rows if r['revision'] == I64-1)
            grant,_ = f.renew(current)
            self.assertEqual(packet(grant)['payload']['revision'],str(I64))
            before = f.snapshot()
            f.error('COUNTER_EXHAUSTED',f.renew,grant,'revision-overflow')
            self.assertEqual(f.snapshot(),before)
        with Fixture(self,suffix='-time') as f:
            f.set_time(I64-30)
            grant,_ = f.acquire(ttl=30)
            self.assertEqual(packet(grant)['payload']['expires_at_s'],I64)
            f.set_time(I64-29)
            before = f.snapshot()
            f.error('COUNTER_EXHAUSTED',f.renew,grant,ttl=30)
            self.assertEqual(f.snapshot(),before)
            for invalid in (-1,True,1.0,I64+1):
                f.set_time(invalid)
                f.error('REQUEST_INVALID',f.renew,grant,'invalid-time')
        with Fixture(self,suffix='-audit',seed='audit_max_minus_one') as f:
            grant,_ = f.acquire()
            self.assertEqual(max(r['sequence'] for r in f.rows('lease_audit')),I64)
            before = f.snapshot()
            f.error('COUNTER_EXHAUSTED',f.release,grant)
            self.assertEqual(f.snapshot(),before)

    def test_l25_ttl_range_and_exact_expiration(self):
        missing_module_first(self)
        for ttl in (30,120,300):
            with self.subTest(ttl=ttl),Fixture(self,suffix='-'+str(ttl)) as f:
                grant,_ = f.acquire(ttl=ttl)
                self.assertEqual(packet(grant)['payload']['expires_at_s'],NOW+ttl)
                f.set_time(NOW+ttl-1)
                renewed,_ = f.renew(grant)
                self.assertEqual(packet(renewed)['payload']['issued_at_s'],NOW+ttl-1)
                f.set_time(packet(renewed)['payload']['expires_at_s'])
                f.error('EXPIRED',f.release,renewed)
        with Fixture(self,suffix='-invalid') as f:
            for ttl in (29,301,True,120.0,None,'120'):
                f.error('REQUEST_INVALID',f.protocol.acquire,f.request(ttl=ttl),'alice')
            self.assertEqual(f.rows('lease_grants'),[])

    def test_l26_backward_clock_persists_highwater_hold(self):
        missing_module_first(self)
        with Fixture(self) as f:
            f.calibrate(); grant,_ = f.acquire()
            same,_ = f.renew(grant,'equal')
            f.set_time(NOW+10); current,_ = f.renew(same,'forward')
            self.assertEqual(f.resource()['time_highwater_s'],NOW+10)
            f.set_time(NOW+9)
            f.error('CLOCK_ROLLBACK',f.release,current)
            held = f.resource()
            self.assertEqual((held['state'],held['hold_reason'],held['time_highwater_s']),('RECONCILIATION_HOLD','CLOCK_ROLLBACK',NOW+10))
            worker = f.start_worker('status')
            observed = packet(unb64(f.join_worker(worker)['result']))
            self.assertEqual(observed['resource']['state'],'RECONCILIATION_HOLD')
            f.set_time(NOW+10000)
            f.error('RESOURCE_HELD',f.protocol.acquire,f.request('takeover'),'bob')
            self.assertEqual(f.resource()['current_generation'],held['current_generation'])
            f.assert_no_write()

    def test_l27_forward_jump_expires_only_unreserved(self):
        missing_module_first(self)
        with Fixture(self) as f:
            old,_ = f.acquire()
            f.set_time(NOW+10000)
            new,_ = f.acquire('new',context='bob')
            self.assertEqual(packet(new)['payload']['fence'],'2')
            self.assertNotEqual(packet(old)['payload']['generation'],packet(new)['payload']['generation'])
            transitions = [r['transition'] for r in f.rows('lease_audit')]
            self.assertIn('EXPIRE',transitions)
        with Fixture(self,suffix='-reserved') as f:
            grant,request = f.acquire(); f.ack(grant)
            reservation = f.begin(grant,request)
            f.set_time(NOW+10000)
            f.error('RESOURCE_HELD',f.protocol.acquire,f.request('takeover',payload='B',head=PAYLOAD_SHAS['A']),'bob')
            calls = len(f.publisher.calls)
            self.assertEqual(f.begin(grant,request),reservation)
            self.assertEqual(len(f.publisher.calls),calls)

    def test_l28_legacy_owner_hold_is_not_expiry_takeover(self):
        missing_module_first(self)
        with Fixture(self) as f:
            f.calibrate(); f.update(legacy=True)
            before = f.resource()
            f.set_time(NOW+1000000)
            f.error('RESOURCE_HELD',f.protocol.acquire,f.request(),'alice')
            self.assertEqual(f.resource()['legacy_hold'],1)
            self.assertEqual(f.resource()['last_fence'],before['last_fence'])
            f.update('clear',bundle=f.make_bundle(3),policy=f.make_policy(3),legacy=False)
            grant,_ = f.acquire()
            self.assertEqual(packet(grant)['payload']['fence'],'1')
            self.assertEqual(len([r for r in f.rows('lease_audit') if r['transition']=='AUTHORITY_REPLACED']),2)
            f.assert_no_write()

    def test_l29_no_renew_release_or_overlap_with_reservation(self):
        missing_module_first(self)
        with Fixture(self) as f:
            grant,request = f.acquire(); f.ack(grant)
            reservation = f.begin(grant,request)
            count = len(f.publisher.calls)
            self.assertEqual(f.begin(grant,request),reservation)
            self.assertEqual(len(f.publisher.calls),count)
            f.error('RESOURCE_HELD',f.renew,grant)
            f.error('RESOURCE_HELD',f.release,grant)
            f.error('RESOURCE_HELD',f.protocol.acquire,f.request('other'),'bob')
            self.assertEqual(len(f.rows('lease_reservations')),1)
            self.assertEqual(f.rows('lease_reservations')[0]['result_bytes'],reservation)

    def test_l30_pause_before_final_gate_past_expiry(self):
        missing_module_first(self)
        with Fixture(self) as f:
            f.calibrate(); grant,request = f.acquire(); f.ack(grant)
            worker = f.start_worker('begin_write',request=canonical(packet(request)['intent']),grant=grant,phase='prepare')
            f.wait_phase('prepare')
            row = f.rows('lease_reservations')[0]
            self.assertEqual((row['state'],row['send_started']),('WRITE_INFLIGHT',0))
            f.set_time(NOW+120)
            f.controller.call('release',phase='prepare')
            self.assertEqual(f.join_worker(worker)['code'],'EXPIRED')
            self.assertEqual(f.rows('lease_reservations')[0]['delivery_attempt_id'],row['delivery_attempt_id'])
            f.assert_no_write()
            f.error('RESOURCE_HELD',f.protocol.acquire,f.request('takeover'),'bob')
            f.cancel_attempt(row,worker[0].pid)
            f.controller.call('join',attempt_id=row['delivery_attempt_id'])
            result = f.protocol.reconcile(row['reservation_id'],f.evidence(row['result_bytes']),'recovery')
            self.assertEqual(packet(result)['classification'],'DEFINITELY_NOT_APPLIED')

        with Fixture(self,suffix='-precondition-expiry') as f:
            f.calibrate(); grant,request = f.acquire(); f.ack(grant)
            expires = packet(grant)['payload']['expires_at_s']
            self.assertEqual(expires,NOW+120)
            worker = f.start_worker('begin_write',request=canonical(packet(request)['intent']),grant=grant,phase='precondition')
            entered = f.wait_phase('precondition')
            self.assertEqual(entered['pid'],worker[0].pid)
            self.assertFalse(worker[1].exists())
            before = f.rows('lease_reservations')[0]
            self.assertEqual((before['state'],before['send_started']),('WRITE_INFLIGHT',0))
            self.assertEqual(entered['attempt_id'],before['delivery_attempt_id'])
            retained = [packet(line) for line in regular_bytes(f.directory/('precondition-'+str(worker[0].pid)+'.jsonl'),2*2**20).splitlines()]
            self.assertEqual(len(retained),1)
            observed = packet(unb64(retained[0]['observed_precondition_base64url']))
            self.assertEqual(retained[0]['observed_precondition_base64url'],retained[0]['returned_precondition_base64url'])
            self.assertEqual((observed['kind'],observed['attempt_id'],observed['reservation_id'],observed['publisher_id'],
                              observed['native_request_sha256'],observed['target']),
                             ('ref_update',before['delivery_attempt_id'],before['reservation_id'],before['publisher_id'],
                              before['native_request_sha256'],f.target()))
            self.assertEqual(observed['observed_precondition'],{'kind':'ref_head','current_head_sha256':ZERO})
            native = packet(unb64(retained[0]['native_observation_base64url']))
            reads = [row for row in f.target_snapshot()['preconditions'] if row['query_id'] == native['query_id']]
            self.assertEqual(len(reads),1)
            self.assertEqual(reads[0]['observation_base64url'],retained[0]['native_observation_base64url'])
            self.assertEqual(Clock(f.clock_path).now_s(),NOW)
            f.assert_no_write()
            f.set_time(expires)
            clock_at_release = Clock(f.clock_path).now_s()
            self.assertEqual(clock_at_release,expires)
            f.controller.call('release',phase='precondition')
            result = f.join_worker(worker)
            self.assertEqual(result['code'],'EXPIRED')
            after = f.rows('lease_reservations')[0]
            for field in ('reservation_id','delivery_attempt_id','native_request_bytes','native_request_sha256','publisher_id','result_bytes'):
                self.assertEqual(after[field],before[field])
            self.assertEqual(after['send_started'],0)
            self.assertNotIn(after['state'],('APPLIED','DEFINITELY_NOT_APPLIED'))
            self.assertNotEqual(f.resource()['state'],'AVAILABLE')
            proof = f.assert_precondition_in_final_transaction(result,after,{'kind':'ref_head','current_head_sha256':ZERO})
            self.assertEqual(proof['callback_record']['observed_precondition_base64url'],proof['callback_record']['returned_precondition_base64url'])
            snapshot = f.target_snapshot()
            self.assertFalse(any(entry['kind'] == 'ref_update' and entry['phase'] == 'send' for entry in snapshot['callback_entries']))
            self.assertEqual(next(row for row in snapshot['refs'] if row['target_id'] == sha(canonical(f.target())))['head_sha256'],ZERO)
            f.assert_no_write()
            f.error('RESOURCE_HELD',f.protocol.acquire,f.request('takeover'),'bob')
            (f.directory/'post-precondition-expiry-refusal.json').write_bytes(canonical({'expires_at_s':expires,
                'clock_before_barrier_release_s':clock_at_release,'barrier':entered,'proof':proof,**SCIENCE}))

    def test_l42_key_rotation_keeps_historical_verification(self):
        missing_module_first(self)
        with Fixture(self) as f:
            original,_ = f.acquire()
            f.release(original)
            rotated = f.make_bundle(2,rotate=True,retired=True)
            f.update(bundle=rotated)
            f.signer.key_id = 'rfc-test2'
            new,_ = f.acquire('new')
            self.assertEqual(packet(new)['payload']['key_id'],'rfc-test2')
            self.assertTrue(packet(f.protocol.verify_signature(original,rotated))['cryptographically_valid'])
            self.assertTrue(packet(f.protocol.verify_signature(original,f.bundle))['cryptographically_valid'])
            f.release(new)
            f.signer.key_id = 'rfc-test1'
            before = f.snapshot()
            f.error('KEY_INACTIVE',f.protocol.acquire,f.request('retired'),'alice')
            self.assertEqual(f.snapshot(),before)

    def test_l43_key_update_and_begin_write_committed_order(self):
        missing_module_first(self)
        for order,phase in (('update-first','prepare'),('marker-first','transport')):
            with self.subTest(order=order),Fixture(self,suffix='-'+order) as f:
                grant,request = f.acquire(); f.ack(grant)
                worker = f.start_worker('begin_write',request=canonical(packet(request)['intent']),grant=grant,phase=phase)
                observed = f.wait_phase(phase)
                self.assertNotEqual(observed['pid'],os.getpid())
                row = f.rows('lease_reservations')[0]
                self.assertEqual(row['send_started'],0 if order == 'update-first' else 1)
                f.update(bundle=f.make_bundle(2,rotate=True,revoked=True))
                f.controller.call('release',phase=phase)
                result = f.join_worker(worker)
                if order == 'update-first':
                    self.assertEqual(result['code'],'KEY_INACTIVE')
                    f.assert_no_write()
                else:
                    self.assertIsNone(result['code'])
                    self.assertNotEqual(f.resource()['state'],'AVAILABLE')
                    terminal = f.protocol.reconcile(row['reservation_id'],f.evidence(row['result_bytes']),'recovery')
                    self.assertEqual(packet(terminal)['classification'],'APPLIED')
                    self.assertEqual(len([r for r in f.target_snapshot()['attempts'] if r['kind']=='ref_update']),1)

    def test_l44_bundle_rollback_and_removed_key_refusal(self):
        missing_module_first(self)
        with Fixture(self) as f:
            result,update = f.update(bundle=f.make_bundle(2,rotate=True))
            before = f.snapshot()
            self.assertEqual(f.protocol.replace_authority(update,'recovery'),result)
            self.assertEqual(f.snapshot(),before)
            for name,bundle in (('lower',f.make_bundle(1)),('same-conflict',f.make_bundle(2,rotate=True,revoked=True))):
                raw = packet(update); raw['update_id'] = self._testMethodName+':'+name; raw['bundle'] = packet(bundle)
                f.error('BUNDLE_ROLLBACK',f.protocol.replace_authority,canonical(raw),'recovery')
            raw_bundle = packet(f.make_bundle(3,rotate=True))['payload']; raw_bundle['keys'] = [raw_bundle['keys'][1]]
            raw = packet(update); raw['update_id'] = self._testMethodName+':removed'; raw['bundle'] = packet(f.root_signed(raw_bundle,b'research-lease-trust/v1\0')); raw['policy'] = packet(f.make_policy(3))
            f.error('BUNDLE_INVALID',f.protocol.replace_authority,canonical(raw),'recovery')
            self.assertEqual(f.rows('lease_meta')[0]['authority_revision'],2)
            self.assertEqual(f.snapshot(),before)

    def test_l45_policy_change_blocks_old_grant_and_races(self):
        missing_module_first(self)
        with Fixture(self) as f:
            grant,request = f.acquire(); f.ack(grant)
            changed = packet(f.make_policy(2))
            for entry in changed['entries']:
                if entry['principal']=='bob':
                    entry['operations'] = ['status']
            f.update(policy=canonical(changed))
            f.error('POLICY_CHANGED',f.begin,grant,request)
            f.error('AUTH_DENIED',f.protocol.acquire,f.request('denied',resource='fixture/ref-other'),'bob')
            self.assertNotEqual(f.rows('lease_authority')[-1]['policy_sha256'],packet(grant)['payload']['authorization_policy_sha256'])
            f.assert_no_write()
        with Fixture(self,suffix='-marker') as f:
            grant,request = f.acquire(); f.ack(grant)
            worker = f.start_worker('begin_write',request=canonical(packet(request)['intent']),grant=grant,phase='transport')
            f.wait_phase('transport'); row = f.rows('lease_reservations')[0]
            f.update(policy=f.make_policy(2))
            f.controller.call('release',phase='transport')
            self.assertIsNone(f.join_worker(worker)['code'])
            self.assertEqual(f.rows('lease_reservations')[0]['delivery_attempt_id'],row['delivery_attempt_id'])
            self.assertNotEqual(f.resource()['state'],'AVAILABLE')

    def test_l46_signer_exception_and_wrong_signature_rollback(self):
        missing_module_first(self)
        for fault in ('raise','short','wrong_key'):
            with self.subTest(fault=fault),Fixture(self,suffix='-'+fault) as f:
                f.calibrate(); before = f.snapshot()
                f.signer.fault = fault
                f.error('SIGNING_FAILED',f.protocol.acquire,f.request(),'alice')
                self.assertEqual(f.snapshot(),before)
                self.assertEqual(f.resource()['last_fence'],0)
                f.assert_no_write()
                f.signer.fault = None
                grant,_ = f.acquire('new-positive')
                self.assertEqual(packet(grant)['payload']['fence'],'1')

    def test_l47_intent_and_request_collision_no_success_sideeffects(self):
        missing_module_first(self)
        with Fixture(self) as f:
            original,request = f.acquire()
            retained = f.rows('lease_intents')[0]['intent_bytes']
            before = f.snapshot()
            conflict = packet(f.request('other',resource='fixture/ref-other'))
            conflict['intent']['intent_id'] = packet(request)['intent']['intent_id']
            f.error('INTENT_CONFLICT',f.protocol.acquire,canonical(conflict),'alice')
            changed = packet(request); changed['intent']['payload_sha256'] = PAYLOAD_SHAS['B']
            f.error('IDEMPOTENCY_CONFLICT',f.protocol.acquire,canonical(changed),'alice')
            self.assertEqual(f.snapshot(),before)
            self.assertEqual(f.rows('lease_intents')[0]['intent_bytes'],retained)
            self.assertEqual(f.rows('lease_idempotency')[0]['result_bytes'],original)
            with sqlite3.connect('file:'+str(f.db)+'?mode=ro',uri=True) as connection:
                self.assertTrue(any(row[2] for row in connection.execute('PRAGMA index_list(lease_idempotency)')))
                self.assertTrue(any(row[2] for row in connection.execute('PRAGMA index_list(lease_intents)')))

    def test_l48_actual_full_filesystem_refuses_commit(self):
        missing_module_first(self)
        with Fixture(self,full=True) as f:
            f.calibrate(); before = f.snapshot()
            filler = f.db.parent/'public-full-device-filler'
            observed_errno = None
            try:
                with filler.open('wb',buffering=0) as writer:
                    block = b'F'*65536
                    for _ in range(128):
                        writer.write(block)
            except OSError as failure:
                observed_errno = failure.errno
            self.assertEqual(observed_errno,28)
            fs = os.statvfs(f.db.parent)
            self.assertEqual(fs.f_bavail,0)
            f.error('LEDGER_FULL',f.protocol.acquire,f.request(),'alice')
            records = [r for r in f.observer.records if r['sqlite_errorcode'] is not None]
            self.assertTrue(any(r['sqlite_errorcode'] & 255 == sqlite3.SQLITE_FULL for r in records))
            self.assertEqual(f.snapshot(),before)
            self.assertEqual(f.rows('lease_grants'),[])
            f.assert_no_write()
            (f.directory/'full-filesystem-observation.json').write_bytes(canonical({'errno':observed_errno,'device':f.db.parent.stat().st_dev,
                'capacity_bytes':fs.f_blocks*fs.f_frsize,'available_blocks':fs.f_bavail,'filler_size':filler.stat().st_size,'filler_archive_member':False}))
            filler.unlink()
            f.integrity()

    def test_l49_integrity_failure_blocks_admission(self):
        missing_module_first(self)
        with Fixture(self) as f:
            grant,_ = f.acquire(); f.integrity()
            self.assertEqual(f.construct().verify_signature(grant,f.bundle),f.protocol.verify_signature(grant,f.bundle))
            corrupt = f.directory/'corrupt-owned-copy.sqlite'
            corrupt.write_bytes(regular_bytes(f.db,4*2**20))
            with corrupt.open('r+b') as writer:
                writer.write(b'BROKEN SQLITE!!!')
            calls = len(f.signer.calls)
            f.error('INTEGRITY_HOLD',f.construct,db_path=corrupt)
            self.assertEqual(len(f.signer.calls),calls)
            self.assertEqual(f.rows('lease_grants')[0]['envelope_bytes'],grant)
            f.assert_no_write()

    def test_l50_audit_outbox_atomic_order_and_readback_ack(self):
        missing_module_first(self)
        with Fixture(self) as f:
            grant,acquire_request = f.acquire()
            renewed,renew_request = f.renew(grant)
            _,release_request = f.release(renewed)
            audit = sorted(f.rows('lease_audit'),key=lambda r:r['sequence'])
            outbox = sorted(f.rows('lease_outbox'),key=lambda r:r['sequence'])
            self.assertEqual([r['sequence'] for r in audit],[1,2,3])
            self.assertEqual([r['transition'] for r in audit],['ACQUIRE','RENEW','RELEASE'])
            self.assertEqual([r['event_id'] for r in audit],[r['event_id'] for r in outbox])
            self.assertEqual([r['state'] for r in outbox],['PENDING']*3)
            request_packets = [packet(acquire_request),packet(renew_request),packet(release_request)]
            fingerprints = [sha(acquire_request),sha(canonical({'grant_sha256':sha(grant),'request':packet(renew_request)})),
                            sha(canonical({'grant_sha256':sha(renewed),'request':packet(release_request)}))]
            for index,(event,box) in enumerate(zip(audit,outbox)):
                payload = packet(event['event_bytes']); without_id = dict(payload); without_id.pop('event_id')
                self.assertEqual(set(payload),set('schema_version sequence resource_id transition previous_state state hold_reason request_id request_sha256 intent_id intent_sha256 grant_sha256 reservation_id outcome_id source_sha256 scientific_effect scientific_status_authority event_id'.split()))
                self.assertEqual({k:payload[k] for k in SCIENCE},SCIENCE)
                expected = {'schema_version':1,'sequence':str(index+1),'resource_id':'fixture/ref',
                    'transition':('ACQUIRE','RENEW','RELEASE')[index], 'previous_state':('AVAILABLE','LEASED','LEASED')[index],
                    'state':('LEASED','LEASED','AVAILABLE')[index],'hold_reason':None,
                    'request_id':request_packets[index]['request_id'],'request_sha256':fingerprints[index],
                    'intent_id':packet(acquire_request)['intent']['intent_id'],'intent_sha256':sha(canonical(packet(acquire_request)['intent'])),
                    'grant_sha256':sha(grant if index==0 else renewed),'reservation_id':None,'outcome_id':None,
                    'source_sha256':f.metadata['source_sha256'],**SCIENCE}
                self.assertEqual(without_id,expected)
                self.assertEqual(sha(canonical(without_id)),event['event_id'])
                self.assertEqual(event['event_sha256'],sha(event['event_bytes']))
                self.assertEqual(box['payload_sha256'],event['event_sha256'])
                result = f.protocol.deliver_audit(event['event_id'],'delivery')
                prepared = [v for kind,v in f.publisher.calls if kind=='prepare' and v.kind=='audit_append'][-1]
                self.assertIsNone(prepared.reservation_id)
                evidence = packet(f.publisher.last_result)
                self.assertNotIn('reservation_id',evidence)
                self.assertEqual(evidence['event_id'],event['event_id'])
                self.assertEqual((evidence['readback_payload_sha256'],evidence['readback_reference_sha256']),(box['payload_sha256'],box['reference_sha256']))
                self.assertEqual(packet(result)['state'],'ACKNOWLEDGED')
                calls = len(f.publisher.calls); before = f.snapshot()
                self.assertEqual(f.protocol.deliver_audit(event['event_id'],'delivery'),result)
                self.assertEqual(len(f.publisher.calls),calls)
                self.assertEqual(f.snapshot(),before)
            self.assertEqual(len(f.rows('lease_outbox')),3)
            self.assertEqual(len([r for r in f.target_snapshot()['attempts'] if r['kind']=='audit_append']),3)

    def test_l51_transport_success_is_not_delivery_ack(self):
        missing_module_first(self)
        for fault in ('readback_payload_sha256','readback_reference_sha256','readback_event_id','native_metadata_sha256','variant'):
            with self.subTest(fault=fault),Fixture(self,suffix='-'+fault) as f:
                grant,_ = f.acquire(); event = f.grant_event(grant)
                f.publisher.result_fault = fault
                f.error('NATIVE_UNKNOWN',f.protocol.deliver_audit,event,'delivery')
                row = f.rows('lease_outbox')[0]
                self.assertEqual((row['state'],row['send_started']),('UNKNOWN',1))
                self.assertIsNone(row['result_bytes'])
                calls = len(f.publisher.calls)
                f.error('RESOURCE_HELD',f.protocol.deliver_audit,event,'delivery')
                self.assertEqual(len(f.publisher.calls),calls)
        with Fixture(self,suffix='-prepared') as f:
            grant,_ = f.acquire(); event = f.grant_event(grant)
            f.publisher.prepared_fault = 'wrong_reservation'
            f.error('EVIDENCE_INCOMPLETE',f.protocol.deliver_audit,event,'delivery')
            self.assertEqual([c for c in f.publisher.calls if c[0]=='send'],[])
            self.assertEqual(len(f.target_snapshot()['attempts']),0)

    def test_l52_unknown_audit_append_no_resend_or_clear(self):
        missing_module_first(self)
        with Fixture(self) as f:
            grant,_ = f.acquire(); event = f.grant_event(grant)
            f.controller.call('pause',phase='target_before_effect')
            worker = f.start_worker('deliver_audit',reservation=event)
            f.wait_phase('target_before_effect')
            row = f.rows('lease_outbox')[0]
            self.assertEqual(row['send_started'],1)
            self.assertEqual(f.target_snapshot()['events'],[])
            f.error('RESOURCE_HELD',f.protocol.deliver_audit,event,'delivery')
            f.error('AUTH_DENIED',f.protocol.deliver_audit,event,'alice')
            f.kill_worker(worker)
            self.assertIsNone(f.join_worker(f.start_worker('status'))['code'])
            f.error('RESOURCE_HELD',f.protocol.deliver_audit,event,'delivery')
            f.controller.call('release',phase='target_before_effect')
            f.controller.call('join',attempt_id=row['delivery_attempt_id'])
            observation = packet(f.publisher.observe('audit_append',row['delivery_attempt_id'],None))
            request = next(r['request_bytes'] for r in f.target_snapshot()['attempts'] if r['attempt_id']==row['delivery_attempt_id'])
            call = PreparedCall('audit_append',row['delivery_attempt_id'],None,unb64(request),sha(unb64(request)),'fixture-publisher')
            evidence = canonical(Publisher.evidence(call,observation))
            result = f.protocol.reconcile_delivery(event,evidence,'recovery')
            self.assertEqual(packet(result)['state'],'ACKNOWLEDGED')
            self.assertEqual(len(f.target_snapshot()['events']),1)

    def test_l53_audit_sender_crash_and_recovery(self):
        missing_module_first(self)
        with Fixture(self) as f:
            grant,_ = f.acquire(); event = f.grant_event(grant)
            f.controller.call('pause',phase='target_after_effect')
            worker = f.start_worker('deliver_audit',reservation=event)
            f.wait_phase('target_after_effect')
            row = f.rows('lease_outbox')[0]
            self.assertEqual(len(f.target_snapshot()['events']),1)
            f.kill_worker(worker)
            self.assertIsNone(f.join_worker(f.start_worker('status'))['code'])
            self.assertNotEqual(f.rows('lease_outbox')[0]['state'],'ACKNOWLEDGED')
            f.controller.call('release',phase='target_after_effect')
            f.controller.call('join',attempt_id=row['delivery_attempt_id'])
            actual = next(r for r in f.target_snapshot()['attempts'] if r['attempt_id']==row['delivery_attempt_id'])
            request = unb64(actual['request_bytes'])
            call = PreparedCall('audit_append',row['delivery_attempt_id'],None,request,sha(request),'fixture-publisher')
            observation = packet(f.publisher.observe('audit_append',row['delivery_attempt_id'],None))
            evidence = canonical(Publisher.evidence(call,observation))
            result = f.protocol.reconcile_delivery(event,evidence,'recovery')
            self.assertEqual(packet(result)['state'],'ACKNOWLEDGED')
            self.assertEqual(len(f.target_snapshot()['events']),1)

    def test_l54_delivery_evidence_replay_and_conflict(self):
        missing_module_first(self)
        with Fixture(self) as f:
            grant,_ = f.acquire(); event,result = f.ack(grant)
            evidence = f.publisher.last_result
            before = f.snapshot()
            self.assertEqual(f.protocol.reconcile_delivery(event,evidence,'recovery'),result)
            self.assertEqual(f.snapshot(),before)
            for field in ('native_metadata_sha256','readback_payload_sha256','readback_reference_sha256','delivery_attempt_id'):
                changed = packet(evidence); changed[field] = ZERO if field.endswith('sha256') else 'wrong-attempt'
                f.error('EVIDENCE_CONFLICT',f.protocol.reconcile_delivery,event,canonical(changed),'recovery')
            self.assertEqual(f.rows('lease_outbox')[0]['evidence_bytes'],evidence)
            self.assertEqual(f.rows('lease_outbox')[0]['result_bytes'],result)

    def test_l55_required_delivery_failure_holds_fixture_write(self):
        missing_module_first(self)
        with Fixture(self) as f:
            f.calibrate(); grant,request = f.acquire()
            f.error('RESOURCE_HELD',f.begin,grant,request)
            self.assertEqual(f.rows('lease_reservations'),[])
            f.assert_no_write()
            f.ack(grant)
            renewed,_ = f.renew(grant)
            f.error('RESOURCE_HELD',f.begin,renewed,request)
            f.publisher.result_fault = 'readback_reference_sha256'
            f.error('NATIVE_UNKNOWN',f.protocol.deliver_audit,f.grant_event(renewed),'delivery')
            f.error('RESOURCE_HELD',f.begin,renewed,request)
            self.assertEqual(f.rows('lease_reservations'),[])
            f.assert_no_write()

    def test_l56_backup_restore_below_known_floor_holds(self):
        missing_module_first(self)
        with Fixture(self) as f:
            earlier = f.directory/'earlier.sqlite'
            with sqlite3.connect('file:'+str(f.db)+'?mode=ro',uri=True) as source:
                with sqlite3.connect(earlier) as destination:
                    source.backup(destination)
            grant,_ = f.acquire(); event = f.grant_event(grant)
            floors = {'fixture/ref':'1','fixture/ref-other':'0','fixture/events':'0'}
            witness = f.make_continuity(floors,authority=1,pending=(event,))
            self.assertIsNotNone(f.construct(continuity_bytes=witness))
            f.error('RESTORE_HOLD',f.construct,db_path=earlier,continuity_bytes=witness)
            self.assertEqual(f.rows('lease_grants')[0]['envelope_bytes'],grant)
            changed = packet(f.registry); changed['issuer_epoch'] = '3'*32
            f.error('RESTORE_HOLD',f.construct,registry_bytes=canonical(changed))
            f.assert_no_write()

    def test_l57_restart_reopens_tombstones_keys_and_outbox(self):
        missing_module_first(self)
        with Fixture(self) as f:
            grant,request = f.acquire(); f.ack(grant)
            reservation = f.begin(grant,request)
            f.update(bundle=f.make_bundle(2,rotate=True,revoked=True))
            before = f.snapshot()
            restarted = f.start_worker('status')
            result = f.join_worker(restarted)
            self.assertNotEqual(result['pid'],os.getpid())
            self.assertEqual(f.snapshot(),before)
            self.assertEqual(f.protocol.acquire(request,'alice'),grant)
            self.assertEqual(f.begin(grant,request),reservation)
            self.assertEqual(f.rows('lease_reservations')[0]['result_bytes'],reservation)
            self.assertEqual(len(f.rows('lease_intents')),1)
            self.assertGreaterEqual(len(f.rows('lease_keys')),1)
            self.assertEqual(f.rows('lease_meta')[0]['authority_revision'],2)
            self.assertEqual({r['key_id'] for r in f.rows('lease_keys') if r['authority_revision']==2},{'rfc-test1','rfc-test2'})
            f.integrity()

    def test_l58_status_is_readonly_and_errors_are_closed(self):
        missing_module_first(self)
        with Fixture(self) as f:
            grant,_ = f.acquire()
            before = f.snapshot(); raw_before = regular_bytes(f.db,4*2**20)
            result = packet(f.status(request_id='absent',reservation_id='absent',event_id='absent'))
            self.assertEqual((result['idempotency'],result['reservation'],result['outcome'],result['delivery']),(None,None,None,None))
            f.protocol.verify_signature(grant,f.bundle)
            self.assertEqual(f.snapshot(),before)
            self.assertEqual(regular_bytes(f.db,4*2**20),raw_before)
            public_marker = 'PRIVATE_PAYLOAD_MARKER'
            raw = packet(f.request()); raw['secret_marker'] = public_marker
            failure = f.error('REQUEST_INVALID',f.protocol.acquire,canonical(raw),'alice')
            self.assertNotIn(public_marker,str(failure))
            self.assertNotIn(VECTORS[0][0],str(failure))
            self.assertLessEqual(set(vars(failure)),{'code','request_id'})

    def test_l59_sequential_thousand_operations_measurement(self):
        missing_module_first(self)
        with Fixture(self) as f:
            samples = []; operations = 0; failures = []; verify_samples = []
            constructor_start = time.monotonic_ns(); f.construct()
            reopen_ns = time.monotonic_ns()-constructor_start
            target_start = f.target_snapshot()['disk_bytes']
            def measured(name, action):
                nonlocal operations
                operations += 1
                started = time.monotonic_ns()
                try:
                    return action()
                except BaseException as failure:
                    failures.append({'operation':name,'ordinal':operations,'type':type(failure).__name__,
                                     'code':getattr(failure,'code',None)})
                    raise
                finally:
                    samples.append({'operation':name,'ordinal':operations,'elapsed_ns':time.monotonic_ns()-started})
                    (f.directory/'measurement.json').write_bytes(canonical({'operations':operations,'failures':failures,'samples':samples,
                        'cold_constructor_ns':f.cold_constructor_ns,'reopen_ns':reopen_ns,'backend_sign_ns':f.signer.timings,
                        'backend_verify_ns':verify_samples,'rss_status':pathlib.Path('/proc/self/status').read_text(),
                        'ledger_bytes':f.db.stat().st_size,'statement_count':f.observer.statement_count,
                        'target_initial_bytes':target_start,'elapsed_ns':time.monotonic_ns()-f.start_ns,
                        'baseline':'NOT_RUN','promotion':'NONE'}))
            head = ZERO
            for cycle in range(200):
                payload = 'A' if cycle%2 == 0 else 'B'
                grant,request = measured('acquire',lambda:f.acquire('cycle-'+str(cycle),payload=payload,head=head))
                renewed,_ = measured('renew',lambda:f.renew(grant,'renew-'+str(cycle)))
                measured('deliver_audit',lambda:f.ack(renewed))
                reservation = measured('begin_write',lambda:f.begin(renewed,request))
                terminal = measured('record_outcome',lambda:f.terminal(reservation))
                verify_start = time.monotonic_ns()
                f.ed.Ed25519PublicKey.from_public_bytes(bytes.fromhex(VECTORS[0][1])).verify(unb64(packet(renewed)['signature_base64url']),DOMAIN+canonical(packet(renewed)['payload']))
                verify_samples.append(time.monotonic_ns()-verify_start)
                self.assertEqual(packet(terminal)['classification'],'APPLIED')
                head = PAYLOAD_SHAS[payload]
                self.assertEqual(packet(renewed)['payload']['fence'],str(cycle+1))
            self.assertEqual(operations,1000)
            self.assertEqual(len(f.rows('lease_outcomes')),200)
            self.assertEqual(f.resource()['last_fence'],200)
            self.assertEqual(f.resource()['state'],'AVAILABLE')
            self.assertEqual(len([r for r in f.target_snapshot()['attempts'] if r['kind']=='ref_update']),200)
            def percentiles(name):
                values = sorted(r['elapsed_ns'] for r in samples if r['operation']==name)
                return {'p50_ns':values[(len(values)-1)//2],'p95_ns':values[(95*len(values)-1)//100],'max_ns':values[-1]}
            (f.directory/'measurement.json').write_bytes(canonical({'operations':operations,'failures':failures,'samples':samples,
                'rss_status':pathlib.Path('/proc/self/status').read_text(),'ledger_bytes':f.db.stat().st_size,
                'observations':len(f.observer.records),'statement_count':f.observer.statement_count,
                'cold_constructor_ns':f.cold_constructor_ns,'reopen_ns':reopen_ns,'backend_sign_ns':f.signer.timings,
                'backend_verify_ns':verify_samples,'acquire':percentiles('acquire'),'begin_write':percentiles('begin_write'),
                'target_initial_bytes':target_start,'target_final_bytes':f.target_snapshot()['disk_bytes'],
                'elapsed_ns':time.monotonic_ns()-f.start_ns,'baseline':'NOT_RUN','promotion':'NONE'}))
            f.integrity()

    def test_l60_runtime_inventory_and_test_scope(self):
        missing_module_first(self)
        with Fixture(self) as f:
            inventory = sorted(name for name in dir(TestLeaseProtocol) if name.startswith('test_l'))
            self.assertEqual(len(inventory),60)
            self.assertEqual([int(name.split('_')[1][1:]) for name in inventory],list(range(1,61)))
            result = f.join_worker(f.start_worker('status'))
            self.assertEqual(result['optimize'],sys.flags.optimize)
            self.assertEqual(f.metadata['mode'],'optimized' if sys.flags.optimize else 'normal')
            f.integrity(); f.assert_no_write()
            (f.directory/'scope.json').write_bytes(canonical({'schema_version':1,'method_inventory':inventory,'mode':f.metadata['mode'],
                'source_sha256':f.metadata['source_sha256'],'contract_sha256':CONTRACT_SHA,'image_reference':IMAGE,'image_id':IMAGE_ID,
                'scientific_effect':'NONE','scientific_status_authority':False,'org_independence':0,
                'crypto_scope':'PUBLIC_RFC8032_ORDINARY_ED25519','production_scope':'NOT_DEPLOYED','native_custody':'ROOT_EXTERNAL_JOIN_REQUIRED'}))
        with Fixture(self,suffix='-sink') as f:
            class BrokenAttach:
                def attach(self, connection, operation_id):
                    raise OSError('public observation sink is unavailable')
                def finish(self, operation_id, record_bytes):
                    raise OSError('public observation sink is unavailable')
            before = f.snapshot()
            f.error('INTEGRITY_HOLD',lambda:f.construct(observer=BrokenAttach()).acquire(f.request(),'alice'))
            self.assertEqual(f.snapshot(),before)
        with Fixture(self,suffix='-committed-sink') as f:
            class BrokenAfterCommit(Observer):
                def finish(self, operation_id, record_bytes):
                    super().finish(operation_id,record_bytes)
                    if packet(record_bytes)['commit_calls']:
                        raise OSError('public observation sink lost after actual commit')
            sink = BrokenAfterCommit(f.directory)
            candidate = f.construct(observer=sink)
            request = f.request()
            f.error('COMMIT_UNKNOWN',candidate.acquire,request,'alice')
            self.assertEqual(len(f.rows('lease_grants')),1)
            self.assertEqual(len(f.rows('lease_idempotency')),1)
            self.assertEqual(f.rows('lease_grants')[0]['envelope_bytes'],f.rows('lease_idempotency')[0]['result_bytes'])
            self.assertEqual(candidate.acquire(request,'alice'),f.rows('lease_idempotency')[0]['result_bytes'])


    def test_l31_revocation_before_final_gate_blocks_send(self):
        missing_module_first(self)
        with Fixture(self) as f:
            f.calibrate(); grant,request = f.acquire(); f.ack(grant)
            worker = f.start_worker('begin_write',request=canonical(packet(request)['intent']),grant=grant,phase='prepare')
            f.wait_phase('prepare')
            f.update(bundle=f.make_bundle(2,rotate=True,revoked=True))
            f.controller.call('release',phase='prepare')
            self.assertEqual(f.join_worker(worker)['code'],'KEY_INACTIVE')
            self.assertTrue(packet(f.protocol.verify_signature(grant,f.bundle))['cryptographically_valid'])
            self.assertEqual(f.rows('lease_reservations')[0]['send_started'],0)
            f.assert_no_write()
            f.error('RESOURCE_HELD',f.protocol.acquire,f.request('takeover'),'bob')

    def test_l32_crash_before_commit_actual_rollback(self):
        missing_module_first(self)
        with Fixture(self) as f:
            f.calibrate(); before = f.snapshot()
            worker = f.start_worker('acquire',request=f.request('killed'),phase='sign')
            observed = f.wait_phase('sign')
            self.assertEqual(observed['pid'],worker[0].pid)
            f.kill_worker(worker)
            restarted = f.start_worker('status')
            self.assertNotEqual(restarted[0].pid,worker[0].pid)
            self.assertIsNone(f.join_worker(restarted)['code'])
            self.assertEqual(f.snapshot(),before)
            f.integrity(); f.assert_no_write()
            grant,_ = f.acquire('fresh-request')
            self.assertEqual(packet(grant)['payload']['fence'],'1')

    def test_l33_commit_lost_response_replay_after_restart(self):
        missing_module_first(self)
        with Fixture(self) as f:
            request = f.request()
            worker = f.start_worker('acquire',request=request,phase='response')
            marker = f.wait_committed_response_gap(worker)
            original = f.rows('lease_grants')[0]['envelope_bytes']
            self.assertEqual(marker['result_sha256'],sha(original))
            self.assertEqual(f.rows('lease_idempotency')[0]['result_bytes'],original)
            f.kill_worker(worker)
            before = f.snapshot()
            replay = f.start_worker('acquire',request=request)
            recovered = f.join_worker(replay)
            self.assertNotEqual(recovered['pid'],worker[0].pid)
            self.assertEqual(unb64(recovered['result']),original)
            self.assertEqual(f.snapshot(),before)
            self.assertEqual(packet(original)['payload']['expires_at_s'],NOW+120)

    def test_l34_crash_after_reservation_before_send(self):
        missing_module_first(self)
        with Fixture(self) as f:
            f.calibrate(); grant,request = f.acquire(); f.ack(grant)
            worker = f.start_worker('begin_write',request=canonical(packet(request)['intent']),grant=grant,phase='prepare')
            f.wait_phase('prepare'); row = f.rows('lease_reservations')[0]
            f.kill_worker(worker)
            restart = f.start_worker('status'); self.assertIsNone(f.join_worker(restart)['code'])
            self.assertNotEqual(restart[0].pid,worker[0].pid)
            self.assertEqual(f.rows('lease_reservations')[0]['native_request_bytes'],row['native_request_bytes'])
            self.assertEqual(f.begin(grant,request),row['result_bytes'])
            self.assertEqual(f.rows('lease_reservations')[0]['send_started'],0)
            f.assert_no_write()
            f.set_time(NOW+10000)
            f.error('RESOURCE_HELD',f.protocol.acquire,f.request('takeover'),'bob')

    def test_l35_crash_during_delayed_native_send(self):
        missing_module_first(self)
        with Fixture(self) as f:
            grant,request = f.acquire(); f.ack(grant)
            f.controller.call('pause',phase='target_before_effect')
            worker = f.start_worker('begin_write',request=canonical(packet(request)['intent']),grant=grant)
            observed = f.wait_phase('target_before_effect')
            row = f.rows('lease_reservations')[0]
            self.assertEqual(row['send_started'],1)
            self.assertGreater(observed['pid'],0)
            f.kill_worker(worker)
            restart = f.start_worker('status'); self.assertIsNone(f.join_worker(restart)['code'])
            f.set_time(NOW+10000)
            f.error('RESOURCE_HELD',f.protocol.acquire,f.request('takeover'),'bob')
            snapshot = f.target_snapshot()
            self.assertEqual(len([r for r in snapshot['attempts'] if r['kind']=='ref_update']),1)
            self.assertEqual(next(r for r in snapshot['refs'] if r['target_id']==sha(canonical(f.target())))['head_sha256'],ZERO)
            f.controller.call('release',phase='target_before_effect')
            f.controller.call('join',attempt_id=row['delivery_attempt_id'])
            result = f.protocol.reconcile(row['reservation_id'],f.evidence(row['result_bytes']),'recovery')
            self.assertEqual(packet(result)['classification'],'APPLIED')
            self.assertEqual(len([r for r in f.target_snapshot()['attempts'] if r['kind']=='ref_update']),1)

    def test_l36_crash_after_effect_before_outcome(self):
        missing_module_first(self)
        with Fixture(self) as f:
            grant,request = f.acquire(); f.ack(grant)
            f.controller.call('pause',phase='target_after_effect')
            worker = f.start_worker('begin_write',request=canonical(packet(request)['intent']),grant=grant)
            f.wait_phase('target_after_effect')
            row = f.rows('lease_reservations')[0]
            target = f.target_snapshot()
            self.assertEqual(next(r for r in target['refs'] if r['target_id']==sha(canonical(f.target())))['head_sha256'],PAYLOAD_SHAS['A'])
            self.assertEqual(len(f.rows('lease_outcomes')),0)
            f.kill_worker(worker)
            self.assertIsNone(f.join_worker(f.start_worker('status'))['code'])
            self.assertNotEqual(f.resource()['state'],'AVAILABLE')
            f.controller.call('release',phase='target_after_effect')
            f.controller.call('join',attempt_id=row['delivery_attempt_id'])
            terminal = f.protocol.reconcile(row['reservation_id'],f.evidence(row['result_bytes']),'recovery')
            self.assertEqual(packet(terminal)['state'],'AVAILABLE')
            self.assertEqual(len([r for r in f.target_snapshot()['attempts'] if r['kind']=='ref_update']),1)

    def test_l37_unchanged_target_with_live_old_request_holds(self):
        missing_module_first(self)
        with Fixture(self) as f:
            grant,request = f.acquire(); f.ack(grant)
            f.controller.call('pause',phase='target_before_effect')
            worker = f.start_worker('begin_write',request=canonical(packet(request)['intent']),grant=grant)
            f.wait_phase('target_before_effect'); row = f.rows('lease_reservations')[0]
            pending = f.evidence(row['result_bytes'],'pending')
            self.assertEqual(packet(pending)['native_outcome']['classification'],'UNKNOWN')
            f.error('EVIDENCE_INCOMPLETE',f.protocol.reconcile,row['reservation_id'],pending,'recovery')
            self.assertEqual(f.resource()['state'],'RECONCILIATION_HOLD')
            self.assertTrue(any(r['alive'] for r in f.target_snapshot()['workers']))
            f.set_time(NOW+10000)
            f.error('RESOURCE_HELD',f.protocol.acquire,f.request('takeover'),'bob')
            f.controller.call('release',phase='target_before_effect')
            f.join_worker(worker); f.controller.call('join',attempt_id=row['delivery_attempt_id'])
            terminal = f.protocol.reconcile(row['reservation_id'],f.evidence(row['result_bytes'],'terminal'),'recovery')
            self.assertEqual(packet(terminal)['classification'],'APPLIED')

    def test_l38_terminal_nonapplication_requires_quiescence(self):
        missing_module_first(self)
        with Fixture(self) as f:
            grant,request = f.acquire(); f.ack(grant)
            f.controller.call('pause',phase='target_before_effect')
            worker = f.start_worker('begin_write',request=canonical(packet(request)['intent']),grant=grant)
            f.wait_phase('target_before_effect'); row = f.rows('lease_reservations')[0]
            raw = packet(f.evidence(row['result_bytes'],'claimed-cancel'))
            raw['native_outcome'].update(classification='DEFINITELY_NOT_APPLIED',terminal=True,old_attempt_quiescent=True,native_operation_id='claimed-cancel')
            f.error('EVIDENCE_INCOMPLETE',f.protocol.reconcile,row['reservation_id'],canonical(raw),'recovery')
            self.assertNotEqual(f.resource()['state'],'AVAILABLE')
            f.cancel_attempt(row,worker[0].pid)
            joined = f.controller.call('join',attempt_id=row['delivery_attempt_id'])
            self.assertFalse(joined['alive']); self.assertIsNotNone(joined['exitcode'])
            f.join_worker(worker)
            terminal = f.protocol.reconcile(row['reservation_id'],f.evidence(row['result_bytes'],'cancelled'),'recovery')
            self.assertEqual(packet(terminal)['classification'],'DEFINITELY_NOT_APPLIED')
            self.assertEqual(next(r for r in f.target_snapshot()['refs'] if r['target_id']==sha(canonical(f.target())))['head_sha256'],ZERO)

    def test_l39_terminal_outcome_exact_replay_and_conflict(self):
        missing_module_first(self)
        with Fixture(self) as f:
            grant,request = f.acquire(); f.ack(grant)
            reservation = f.begin(grant,request)
            outcome = f.publisher.last_result
            terminal = f.terminal(reservation)
            before = f.snapshot()
            self.assertEqual(f.protocol.record_outcome(packet(reservation)['reservation_id'],outcome,'publisher'),terminal)
            self.assertEqual(f.snapshot(),before)
            for field,value in (('classification','DEFINITELY_NOT_APPLIED'),('payload_sha256',ZERO),('delivery_attempt_id','other-attempt')):
                raw = packet(outcome); raw[field] = value
                f.error('EVIDENCE_CONFLICT',f.protocol.record_outcome,packet(reservation)['reservation_id'],canonical(raw),'publisher')
            self.assertEqual(f.rows('lease_outcomes')[0]['evidence_bytes'],outcome)
            self.assertEqual(f.rows('lease_outcomes')[0]['result_bytes'],terminal)

    def test_l40_recovery_scope_and_evidence_identity(self):
        missing_module_first(self)
        with Fixture(self) as f:
            grant,request = f.acquire(); f.ack(grant)
            reservation = f.begin(grant,request)
            identifier = packet(reservation)['reservation_id']
            evidence = f.evidence(reservation)
            f.error('AUTH_DENIED',f.protocol.reconcile,identifier,evidence,'alice')
            for field,value in (('observer_record_sha256',ZERO),):
                raw = packet(evidence); raw[field] = value; raw['evidence_id'] += ':wrong-observer'
                f.error('EVIDENCE_INCOMPLETE',f.protocol.reconcile,identifier,canonical(raw),'recovery')
            for field,value in (('reservation_id','wrong-reservation'),('reservation_id',None),('publisher_id','wrong-publisher')):
                raw = packet(evidence); raw['evidence_id'] += ':'+field+str(value); raw['native_outcome'][field] = value
                expected = 'REQUEST_INVALID' if value is None else 'EVIDENCE_INCOMPLETE'
                f.error(expected,f.protocol.reconcile,identifier,canonical(raw),'recovery')
            result = f.protocol.reconcile(identifier,evidence,'recovery')
            self.assertEqual(packet(result)['classification'],'APPLIED')

    def test_l41_send_marker_pause_is_unknown_not_takeover(self):
        missing_module_first(self)
        with Fixture(self) as f:
            f.calibrate(); grant,request = f.acquire(); f.ack(grant)
            worker = f.start_worker('begin_write',request=canonical(packet(request)['intent']),grant=grant,phase='transport')
            f.wait_phase('transport'); row = f.rows('lease_reservations')[0]
            self.assertEqual(row['send_started'],1)
            f.assert_no_write()
            f.kill_worker(worker)
            self.assertIsNone(f.join_worker(f.start_worker('status'))['code'])
            self.assertEqual(f.begin(grant,request),row['result_bytes'])
            f.set_time(NOW+10000)
            f.error('RESOURCE_HELD',f.protocol.acquire,f.request('takeover'),'bob')
            f.assert_no_write()
            f.cancel_attempt(row,worker[0].pid)
            f.controller.call('join',attempt_id=row['delivery_attempt_id'])
            result = f.protocol.reconcile(row['reservation_id'],f.evidence(row['result_bytes']),'recovery')
            self.assertEqual(packet(result)['classification'],'DEFINITELY_NOT_APPLIED')


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--lease-worker':
        worker_main(sys.argv[2],sys.argv[3])
    else:
        unittest.main()
