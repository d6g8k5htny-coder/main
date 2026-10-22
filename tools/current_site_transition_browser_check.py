#!/usr/bin/env python3
"""Controlled Chromium old-HTML/new-payload checks; never publication evidence.

Run only after the exact candidate loader passed its unchanged 30-case suite.
This standalone companion leaves the existing 35-case harness untouched.
Source authoring/review does not execute this program or establish browser RED.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from hashlib import sha256
import hashlib
from html.parser import HTMLParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import threading
import time
import traceback
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
PAGES = ('cite.html', 'dependencies.html', 'explore.html', 'formal.html',
         'index.html', 'measure.html', 'museum.html', 'reproduce.html',
         'research.html', 'workspace.html')
START, END = b'<!-- CURRENT-SITE:START -->', b'<!-- CURRENT-SITE:END -->'
MAX_INPUT = 128 * 1024 * 1024
MAX_OUTPUT = 16 * 1024 * 1024
MAX_MEMBER = 2 * 1024 * 1024
REPORT_RESERVE = MAX_MEMBER
MANIFEST_RESERVE = 256 * 1024
PAYLOAD_LIMIT = MAX_OUTPUT - REPORT_RESERVE - MANIFEST_RESERVE
HEX40, HEX64 = re.compile(r'[0-9a-f]{40}\Z'), re.compile(r'[0-9a-f]{64}\Z')
SYNTHETIC_SCOPE = ('Synthetic observations and loopback responses only. Commit, '
                   'source and graph identities in these fixtures are not project evidence. '
                   'Scientific effect NONE. No public runtime, cache-layer or spoken AT claim.')


class SetupFailure(Exception):
    """Source/fixture/loader preparation failed; cannot qualify a UI regression."""


class UIRegression(AssertionError):
    """A verified observation reached the actual page but its UI is incoherent."""


class BudgetRegression(AssertionError):
    """The actual entrypoint did not abort/refuse within its outer budget."""


def require(condition, message, kind=SetupFailure):
    if not condition:
        raise kind(message)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')


def identity(raw):
    return {'bytes': len(raw), 'sha256': sha256(raw).hexdigest()}


def strict_json(raw):
    def pairs(rows):
        value = {}
        for key, item in rows:
            if key in value:
                raise SetupFailure('Duplicate JSON key')
            value[key] = item
        return value
    def constant(value):
        raise SetupFailure('Nonfinite JSON constant: ' + value)
    return json.loads(raw.decode('utf-8'), object_pairs_hook=pairs,
                      parse_constant=constant)


def regular_bytes(path, maximum=MAX_INPUT):
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1,
            'Non-regular/symlinked/multilink input: ' + path.name)
    require(before.st_size <= maximum, 'Input byte bound exceeded: ' + path.name)
    def geometry(s):
        return (s.st_dev, s.st_ino, s.st_mode, s.st_uid, s.st_nlink,
                s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as source:
        require(geometry(before) == geometry(os.fstat(source.fileno())), 'Input changed before read')
        raw = source.read(maximum + 1)
        require(geometry(before) == geometry(os.fstat(source.fileno())) == geometry(path.lstat()),
                'Input changed during read')
    require(len(raw) == before.st_size, 'Incomplete input read')
    return raw


def load_inputs(manifest_path, expected_manifest_sha, context_path, expected_context_sha):
    """Read before project imports; freeze served bytes in memory, not mutable paths."""
    raw = regular_bytes(manifest_path, MAX_MEMBER)
    require(sha256(raw).hexdigest() == expected_manifest_sha, 'Source manifest identity mismatch')
    manifest = strict_json(raw)
    require(set(manifest) == {'schema_version', 'files'} and manifest['schema_version'] == 1
            and isinstance(manifest['files'], list) and len(manifest['files']) <= 20000,
            'Invalid source manifest')
    contents, total = {}, 0
    for row in manifest['files']:
        require(isinstance(row, dict) and set(row) == {'path', 'bytes', 'sha256'}, 'Invalid source row')
        name = row['path']
        require(isinstance(name, str) and name and '\\' not in name and ':' not in name
                and not name.startswith('/') and all(p not in ('', '.', '..') for p in name.split('/')),
                'Unsafe source path')
        require(name not in contents and type(row['bytes']) is int and row['bytes'] >= 0
                and isinstance(row['sha256'], str) and HEX64.fullmatch(row['sha256']),
                'Duplicate or invalid source identity')
        path = ROOT.joinpath(*PurePosixPath(name).parts)
        require(not any(p.is_symlink() for p in (path, *path.parents) if p != ROOT.parent),
                'Symlinked source component')
        body = regular_bytes(path)
        require(identity(body) == {'bytes': row['bytes'], 'sha256': row['sha256']},
                'Source byte identity mismatch: ' + name)
        contents[name] = body
        total += len(body)
        require(total <= MAX_INPUT, 'Cumulative source byte bound exceeded')
    # Pin all project Python modules before importing the actual producer. Do not
    # substitute test doubles for its acquisition/date/history dependencies.
    python_files = {p.relative_to(ROOT).as_posix() for p in (ROOT / 'tools').rglob('*.py')}
    require(python_files <= set(contents), 'Unpinned project Python dependency')
    require({f'docs/site/{name}' for name in PAGES} <= set(contents), 'Missing maintained page source')
    for name in ('docs/site/current.mjs', 'docs/site/current-model.mjs',
                 'docs/site/dependency-model.mjs', 'tests/browser-requirements.txt'):
        require(name in contents, 'Missing runtime source: ' + name)
    actual_pages = {p.name for p in (ROOT / 'docs/site').glob('*.html')}
    require(actual_pages == set(PAGES), 'Maintained ten-page membership changed')
    context_raw = regular_bytes(context_path, MAX_MEMBER)
    require(sha256(context_raw).hexdigest() == expected_context_sha, 'Run context identity mismatch')
    context = strict_json(context_raw)
    require(isinstance(context, dict) and context.get('schema_version') == 1, 'Invalid run context')
    for key in ('head_sha', 'base_sha', 'tested_merge_sha'):
        require(isinstance(context.get(key), str) and HEX40.fullmatch(context[key]), 'Missing native Git identity')
    gate = context.get('loader_qualification', {})
    require(isinstance(gate, dict) and gate.get('disposition') == '30_OF_30_PASS'
            and gate.get('model_sha256') == sha256(contents['docs/site/current-model.mjs']).hexdigest()
            and isinstance(gate.get('native_receipt'), str)
            and re.fullmatch(r'https://github\.com/d6g8k5htny-coder/main/(?:pull|issues)/[1-9][0-9]*#issuecomment-[1-9][0-9]*',
                             gate['native_receipt']), 'Qualified working-loader receipt absent or mismatched')
    return contents, {'source_manifest': identity(raw), 'context_identity': identity(context_raw),
                      'context': context, 'source_files': len(contents), 'source_bytes': total,
                      'scope': 'Receipt supplied by separately qualified route; this program does not authenticate GitHub offline.'}


def verify_project_import_sources(contents):
    require(not any((ROOT / 'tools').rglob('*.pyc')), 'Preexisting project bytecode refused')
    paths = {p.relative_to(ROOT).as_posix() for p in (ROOT / 'tools').rglob('*.py')}
    require(paths <= set(contents), 'Project Python membership changed before/after import')
    for name in sorted(paths):
        require(regular_bytes(ROOT / name) == contents[name], 'Project Python bytes changed around import: ' + name)


class Output:
    def __init__(self, path):
        require(not path.exists(), 'Output directory must be new')
        path.mkdir(mode=0o700)
        self.path, self.total, self.payload_total, self.members, self.attempts = path, 0, 0, [], []

    def write(self, name, raw):
        require(re.fullmatch(r'[a-z0-9][a-z0-9_.-]*', name) and len(name) <= 150, 'Unsafe output name')
        require(len(self.attempts) < 128, 'Finite output member bound exceeded')
        terminal = name in ('report.json', 'manifest.json')
        limit = MANIFEST_RESERVE if name == 'manifest.json' else MAX_MEMBER
        require(len(raw) <= limit and self.total + len(raw) <= MAX_OUTPUT,
                'Finite output byte bound exceeded')
        require(terminal or self.payload_total + len(raw) <= PAYLOAD_LIMIT,
                'Payload exhausted its budget; terminal evidence space is reserved')
        # Charge before I/O and never refund on failure: a partially retained
        # file cannot become invisible to the aggregate capacity calculation.
        self.total += len(raw)
        if not terminal:
            self.payload_total += len(raw)
        attempt = {'path': name, 'bytes_reserved': len(raw), 'complete': False}
        self.attempts.append(attempt)
        try:
            with (self.path / name).open('xb') as stream:
                require(stream.write(raw) == len(raw), 'Incomplete output write')
            os.chmod(self.path / name, 0o600)
            attempt['complete'] = True
        except Exception as error:
            attempt['error'] = str(error)[:1000]
            raise
        row = {'path': name, **identity(raw)}
        self.members.append(row)
        return row


def synthetic_fixture_pair():
    """Return synthetic Source fixtures, not acquired repository/file originals.

    Source-authored only: future harness must call the actual qualified Python
    validator and actual current_section producer before any browser observation.
    json and hashlib are supplied by the containing standalone helper.
    """
    main_repository = "d6g8k5htny-coder/main"
    math_repository = "d6g8k5htny-coder/Math-"

    def encoded(value):
        return (json.dumps(value, sort_keys=True, ensure_ascii=True,
                           separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")

    def identity(repository, commit, path, raw):
        return {
            "repository": repository,
            "commit": commit,
            "path": path,
            "blob": hashlib.sha1(b"blob " + str(len(raw)).encode("ascii")
                                  + b"\0" + raw).hexdigest(),
            "bytes": len(raw),
        }

    # Construction inputs are separate from the literal expectation table.
    specifications = (
        ("old", "OLD", "2026-10-21T00:00:00Z", "1" * 40, "2" * 40,
         "a" * 40, "b" * 40, "fixture.old"),
        ("new", "NEW", "2026-10-22T00:00:00Z", "3" * 40, "4" * 40,
         "c" * 40, "d" * 40, "fixture.new"),
    )
    # Literal test oracles, never derived from target HTML, loaded snapshots,
    # or generated payload fields. These URLs are explicitly synthetic pins.
    expected_by_phase = {
        "new": {
            "edge_count": 0,
            "graph_node_id": "fixture.new",
            "graph_url": "https://github.com/d6g8k5htny-coder/Math-/blob/4444444444444444444444444444444444444444/claims/graph.json",
            "main_commit": "3333333333333333333333333333333333333333",
            "main_repository_url": "https://github.com/d6g8k5htny-coder/main/tree/3333333333333333333333333333333333333333",
            "main_tree": "cccccccccccccccccccccccccccccccccccccccc",
            "math_commit": "4444444444444444444444444444444444444444",
            "math_repository_url": "https://github.com/d6g8k5htny-coder/Math-/tree/4444444444444444444444444444444444444444",
            "math_tree": "dddddddddddddddddddddddddddddddddddddddd",
            "node_count": 1,
            "node_marker": "Synthetic NEW transition fixture; scientific effect NONE.",
            "observed_at": "2026-10-22T00:00:00Z",
            "reference_labels": [
                "Synthetic NEW STATUS; scientific effect NONE.",
                "Synthetic NEW research index; scientific effect NONE.",
                "Synthetic NEW proof index; scientific effect NONE."
            ],
            "reference_urls": [
                "https://github.com/d6g8k5htny-coder/main/blob/3333333333333333333333333333333333333333/STATUS.md",
                "https://github.com/d6g8k5htny-coder/main/blob/3333333333333333333333333333333333333333/docs/RESEARCH_INDEX.md",
                "https://github.com/d6g8k5htny-coder/Math-/blob/4444444444444444444444444444444444444444/PROOF_INDEX.md"
            ],
            "scientific_effect": "NONE",
            "source_count": 4
        },
        "old": {
            "edge_count": 0,
            "graph_node_id": "fixture.old",
            "graph_url": "https://github.com/d6g8k5htny-coder/Math-/blob/2222222222222222222222222222222222222222/claims/graph.json",
            "main_commit": "1111111111111111111111111111111111111111",
            "main_repository_url": "https://github.com/d6g8k5htny-coder/main/tree/1111111111111111111111111111111111111111",
            "main_tree": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "math_commit": "2222222222222222222222222222222222222222",
            "math_repository_url": "https://github.com/d6g8k5htny-coder/Math-/tree/2222222222222222222222222222222222222222",
            "math_tree": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "node_count": 1,
            "node_marker": "Synthetic OLD transition fixture; scientific effect NONE.",
            "observed_at": "2026-10-21T00:00:00Z",
            "reference_labels": [
                "Synthetic OLD STATUS; scientific effect NONE.",
                "Synthetic OLD research index; scientific effect NONE.",
                "Synthetic OLD proof index; scientific effect NONE."
            ],
            "reference_urls": [
                "https://github.com/d6g8k5htny-coder/main/blob/1111111111111111111111111111111111111111/STATUS.md",
                "https://github.com/d6g8k5htny-coder/main/blob/1111111111111111111111111111111111111111/docs/RESEARCH_INDEX.md",
                "https://github.com/d6g8k5htny-coder/Math-/blob/2222222222222222222222222222222222222222/PROOF_INDEX.md"
            ],
            "scientific_effect": "NONE",
            "source_count": 4
        }
    }
    pair = {}
    for phase, label, observed_at, main_commit, math_commit, main_tree, math_tree, node_id in specifications:
        graph_data = {
            "nodes": {
                node_id: {
                    "kind": "candidate",
                    "classification": "OPEN_ACTIVE",
                    "source": "PROOF_INDEX.md",
                    "label": "Synthetic " + label + " transition fixture; scientific effect NONE.",
                    "scientific_effect": "NONE",
                },
            },
            "edges": [],
        }
        graph_raw = encoded(graph_data)
        documents = (
            (main_repository, main_commit, "STATUS.md",
             ("# Synthetic " + label + " STATUS\n\nScientific effect: NONE.\n").encode("utf-8"),
             "Synthetic " + label + " STATUS; scientific effect NONE."),
            (main_repository, main_commit, "docs/RESEARCH_INDEX.md",
             ("# Synthetic " + label + " research index\n\nScientific effect: NONE.\n").encode("utf-8"),
             "Synthetic " + label + " research index; scientific effect NONE."),
            (math_repository, math_commit, "PROOF_INDEX.md",
             ("# Synthetic " + label + " proof index\n\nScientific effect: NONE.\n").encode("utf-8"),
             "Synthetic " + label + " proof index; scientific effect NONE."),
        )
        sources = []
        references = []
        source_contents = []
        for repository, commit, path, document_raw, reference_label in documents:
            whole = identity(repository, commit, path, document_raw)
            sources.append(dict(whole))
            references.append({
                **whole,
                "sha256": hashlib.sha256(document_raw).hexdigest(),
                "label": reference_label,
            })
            source_contents.append({
                "repository": repository, "commit": commit, "path": path,
                "raw": document_raw, "synthetic": True,
            })
        graph_source = identity(math_repository, math_commit, "claims/graph.json", graph_raw)
        sources.append(dict(graph_source))
        source_contents.append({
            "repository": math_repository, "commit": math_commit,
            "path": "claims/graph.json", "raw": graph_raw, "synthetic": True,
        })
        data = {
            "schema_version": 1,
            "observed_at": observed_at,
            "scientific_effect": "NONE",
            "repositories": [
                {"repository": main_repository, "commit": main_commit, "tree": main_tree},
                {"repository": math_repository, "commit": math_commit, "tree": math_tree},
            ],
            "sources": sources,
            "graph": {
                "identity": {**graph_source, "sha256": hashlib.sha256(graph_raw).hexdigest()},
                "data": graph_data,
            },
            "equations": [],
            "references": references,
        }
        raw = encoded(data)
        # Exactly the three admitted legacy metadata fields; no invented archive.
        meta = {"sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw), "observed_at": observed_at}
        pair[phase] = {
            "data": data,
            "raw": raw,
            "meta": meta,
            "raw_meta": encoded(meta),
            "expected": expected_by_phase[phase],
            "source_contents": source_contents,
            "synthetic": True,
        }
    return pair


def build_documents(contents, pair, current_section, validate_payload):
    docs, modes = {}, {}
    for version in ('old', 'new'):
        validate_payload(pair[version]['data'])
    for page in PAGES:
        original = contents['docs/site/' + page]
        require(original.count(START) == original.count(END) == 1
                and original.index(START) < original.index(END), 'Malformed maintained page markers')
        script_free = page in ('research.html', 'formal.html') or bool(re.search(
            rb"script-src\s+(?:&#39;|'|&apos;)none(?:&#39;|'|&apos;)", original))
        has_current = bool(re.search(rb'<script\b[^>]*\bsrc=[\"\x27]current\.mjs(?:\?[^\"\x27]*)?[\"\x27]', original))
        require(script_free or has_current, 'Page has no qualified current.mjs entrypoint: ' + page)
        modes[page] = 'static' if script_free else 'interactive'
        left = original.index(START) + len(START)
        right = original.index(END)
        for version in ('old', 'new'):
            fixture = pair[version]
            section = current_section(fixture['data'], fixture['meta'], search=not script_free).encode('utf-8')
            body = original[:left] + b'\n' + section + b'\n' + original[right:]
            require(body[:left] == original[:left] and body[-len(original[right:]):] == original[right:],
                    'Fixture changed bytes outside generated current section')
            docs[(page, version)] = body
    require(any(m == 'interactive' for m in modes.values()) and any(m == 'static' for m in modes.values()),
            'Dynamic/static page controls absent')
    return docs, modes


# Passive boundary observations. Preserve the native fetch, its response and the
# caller's actual signal; never replace AbortController, timers or Date/performance.
PROBE = r"""(() => {
  const nativeFetch = window.fetch, ids = new WeakMap(); let next = 0;
  const records = []; window.__transitionProbe = records;
  window.fetch = function(input, init) {
    const url=new URL(typeof input === 'string' ? input : input.url, location.href);
    const current=new URL('current.json', location.href), meta=new URL('current-meta.json', location.href);
    if(url.origin!==current.origin || ![current.pathname,meta.pathname].includes(url.pathname))
      return Reflect.apply(nativeFetch, this, [input,init]);
    if (records.length >= 32) throw new Error('Transition request evidence bound exceeded');
    if(url.href.length>2048) throw new Error('Transition request URL evidence bound exceeded');
    const signal = init?.signal;
    if (signal && !ids.has(signal)) ids.set(signal, ++next);
    const row = {url:url.href, method:init?.method ?? input?.method ?? 'GET',
      started:performance.now(), cache:init?.cache ?? null, credentials:init?.credentials ?? null,
      signal:signal ? ids.get(signal) : null, abortedAt:null};
    records.push(row);
    if(signal) signal.addEventListener('abort', () => {row.abortedAt=performance.now();}, {once:true});
    const promise = Reflect.apply(nativeFetch, this, [input, init]);
    promise.then(response => {row.status=response.status;}, error => {
      row.errorName=error.name;row.errorAt=performance.now();});
    return promise;
  };
})()"""


@contextmanager
def serve(contents, document, page_name, pair, version, stall):
    """Only loopback; synthetic endpoints override immutable, source-pinned assets."""
    requests, lock, release = [], threading.Lock(), threading.Event()
    new, old = pair['new'], pair['old']
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass
        def do_GET(self):
            target = urlsplit(self.path)
            path, query = target.path, target.query
            row = {'target': self.path, 'started_monotonic': time.monotonic()}
            with lock:
                if len(requests) >= 128 or len(self.path) > 2048:
                    self.send_error(429); return
                requests.append(row)
            body, media, status = None, 'application/octet-stream', 200
            if path == '/site/' + page_name and not query:
                body, media = document, 'text/html; charset=utf-8'
            elif path == '/site/current-meta.json' and not query:
                row['endpoint'] = 'descriptor'
                if stall == 'metadata':
                    release.wait(24)
                body, media = new['raw_meta'], 'application/json'
            elif path == '/site/current.json':
                row['endpoint'] = 'payload'
                allowed = ('', 'observation=' + old['meta']['sha256'],
                           'observation=' + new['meta']['sha256'])
                if query not in allowed:
                    status, body = 404, b'Unexpected observation key'
                else:
                    if stall and query in ('', 'observation=' + old['meta']['sha256']):
                        release.wait(3)
                    if stall == 'successor' and query == 'observation=' + new['meta']['sha256']:
                        release.wait(24)
                    body, media = pair[version]['raw'], 'application/json'
            else:
                # Some real pages also import /public-math assets. Serve only
                # explicitly pinned docs bytes, never arbitrary checkout paths.
                name = path.removeprefix('/')
                key = 'docs/' + name
                if path.startswith('/') and key in contents and '..' not in name.split('/'):
                    body = contents[key]
                    media = mimetypes.guess_type(name)[0] or 'application/octet-stream'
                    if name.endswith('.mjs'):
                        media = 'text/javascript'
                else:
                    status, body = 404, b'Not a qualified site asset'
            row.update({'status': status, 'body': identity(body)})
            try:
                self.send_response(status)
                self.send_header('Content-Type', media)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(body)
                row['sent_monotonic'] = time.monotonic()
            except (BrokenPipeError, ConnectionResetError):
                row['client_closed'] = True
    class Server(ThreadingHTTPServer):
        daemon_threads, block_on_close, request_queue_size = True, False, 32
        def __init__(self, *args):
            self.slots = threading.BoundedSemaphore(32)
            self.workers, self.worker_lock = set(), threading.Lock()
            super().__init__(*args)
        def get_request(self):
            connection, address = super().get_request()
            connection.settimeout(5)
            return connection, address
        def process_request(self, request, address):
            if not self.slots.acquire(blocking=False):
                self.shutdown_request(request); return
            worker = threading.Thread(target=self.process_request_thread, args=(request, address), daemon=True)
            # Register pending starts as well as running handlers. shutdown()
            # drains the accept loop before cleanup snapshots this complete set.
            with self.worker_lock:
                self.workers.add(worker)
            try:
                worker.start()
            except Exception:
                with self.worker_lock:
                    self.workers.discard(worker)
                self.slots.release(); raise
        def process_request_thread(self, request, address):
            worker = threading.current_thread()
            try:
                super().process_request_thread(request, address)
            finally:
                with self.worker_lock:
                    self.workers.discard(worker)
                self.slots.release()
    server = Server(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.05), daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/site/', requests
    finally:
        release.set(); server.shutdown(); server.server_close(); thread.join(timeout=2)
        require(not thread.is_alive(), 'Loopback server did not terminate')
        deadline = time.monotonic() + 6
        with server.worker_lock:
            workers = list(server.workers)
        for worker in workers:
            worker.join(timeout=max(0, deadline - time.monotonic()))
        require(not any(worker.is_alive() for worker in workers), 'Loopback worker cleanup deadline exceeded')


def probe_rows(page):
    return page.evaluate('window.__transitionProbe || []')


def request_contract(page, origin, paths):
    rows = probe_rows(page)
    require([r['url'] for r in rows] == [origin + p for p in paths],
            'Request order/key/count differs from the bounded successor contract')
    require(all(r['method'] == 'GET' and r['cache'] == 'no-store' and r['credentials'] == 'omit' and r['signal'] == 1 for r in rows),
            'Request options or shared actual AbortSignal differ')
    return rows


def loaded_snapshot(page):
    """Use the same already-started module URL, never a fresh import cache key."""
    return page.evaluate("""async () => {
      const script=[...document.querySelectorAll('script[type=module][src]')]
        .find(s=>new URL(s.src).pathname.endsWith('/current.mjs'));
      if(!script) throw Error('Actual current.mjs script absent');
      const observe=async()=>{const module=await import(script.src);return await module.currentSnapshot;};
      let observerTimer;
      const bounded=new Promise((_,reject)=>{observerTimer=setTimeout(()=>reject(Error('Snapshot observer deadline; no product result inferred')),5000);});
      let data;
      try{data=await Promise.race([observe(),bounded]);}finally{clearTimeout(observerTimer);}
      return data && {observed_at:data.observed_at, repositories:data.repositories,
        references:data.references, graph:data.graph.identity, scientific_effect:data.scientific_effect};
    }""")


def assert_surface(page, expected, kind, *, verified):
    section = page.locator('#current-research')
    require(section.count() == 1, 'Current section absent', kind)
    dates = section.locator('#current-date-explanation')
    require(dates.count() == 1, 'Observation/source date explanation absent', kind)
    if dates.get_attribute('open') is None:
        dates.locator('summary').click()
    observed = section.evaluate("""section => {
      const visible=e=>!!e.getClientRects().length && getComputedStyle(e).visibility!=='hidden';
      const dates=section.querySelector('#current-date-explanation');
      const state=section.querySelector('.current-state');
      return {state:state?.textContent ?? null,state_visible:!!state && visible(state),
        links:[...section.querySelectorAll('a[href]')].map(a=>({href:a.getAttribute('href'),
          text:a.textContent,visible:visible(a),inventory:!!a.closest('.current-results,.current-acquisition-notice'),
          dates:!!a.closest('#current-date-explanation')})),
        dates_text:dates?.textContent ?? null,
        dates_observations:[...dates.querySelectorAll('time')].map(t=>t.getAttribute('datetime'))};
    }""")
    # Observe every field before asserting, so an old repository pin cannot mask
    # the independent current-reference/date/graph-source observations.
    problems = []
    state = observed['state'] or ''
    if not (observed['state_visible'] and expected['observed_at'] in state and
            (not verified or state.startswith('Verified source observation:'))):
        problems.append('visible observation timestamp differs')
    for name in ('main', 'math'):
        matches = [a for a in observed['links'] if a['href'] == expected[name + '_repository_url']
                   and not a['inventory'] and not a['dates']]
        if not (len(matches) == 1 and matches[0]['visible'] and expected[name + '_commit'] in matches[0]['text']):
            problems.append('visible repository pin differs: ' + name)
    for url in expected['reference_urls']:
        if not any(a['href'] == url and a['visible'] and not a['inventory'] and not a['dates']
                   for a in observed['links']):
            problems.append('visible current reference differs: ' + url)
    repo_rows = [a for a in observed['links'] if not a['inventory'] and not a['dates'] and
                 re.match(r'https://github\.com/d6g8k5htny-coder/(?:main|Math-)/tree/', a['href'])]
    if sorted(a['href'] for a in repo_rows) != sorted(expected[n + '_repository_url'] for n in ('main', 'math')):
        problems.append('contradictory or extra current repository pins')
    reference_rows = [a for a in observed['links'] if not a['inventory'] and not a['dates'] and
                      re.match(r'https://github\.com/d6g8k5htny-coder/(?:main|Math-)/blob/', a['href'])]
    if sorted((a['href'], a['text']) for a in reference_rows) != sorted(zip(expected['reference_urls'], expected['reference_labels'])):
        problems.append('current reference URL/label census differs')
    graph_rows = [a for a in observed['links'] if a['dates'] and
                  re.match(r'https://github\.com/d6g8k5htny-coder/Math-/blob/', a['href'])]
    if len(graph_rows) != 1 or graph_rows[0]['href'] != expected['graph_url'] or not graph_rows[0]['visible']:
        problems.append('visible current graph source URL differs')
    if expected['math_commit'] not in (observed['dates_text'] or ''):
        problems.append('visible graph commit differs')
    if observed['dates_observations'] != [expected['observed_at']]:
        problems.append('date explanation names a different current observation')
    observed['problems'] = problems
    if problems:
        error = kind('; '.join(problems))
        error.observation = observed
        raise error
    return observed


def historical_links(page):
    return page.locator('#current-research a[href*="snapshots/"]').evaluate_all(
        '(links)=>links.map(a=>({href:a.getAttribute("href"),text:a.textContent}))')


def served_historical_links(document):
    class Links(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.inside, self.active, self.rows = False, None, []
        def handle_starttag(self, tag, attrs):
            attributes = dict(attrs)
            if tag == 'section' and attributes.get('id') == 'current-research':
                self.inside = True
            if self.inside and tag == 'a' and 'snapshots/' in attributes.get('href', ''):
                self.active = {'href': attributes['href'], 'text': ''}
        def handle_data(self, data):
            if self.active is not None:
                self.active['text'] += data
        def handle_endtag(self, tag):
            if tag == 'a' and self.active is not None:
                self.rows.append(self.active); self.active = None
            if tag == 'section':
                self.inside = False
    parser = Links()
    parser.feed(document.decode('utf-8'))
    require(len(parser.rows) == 2, 'Controlled historical link census differs')
    return parser.rows


def exercise(browser, contents, docs, modes, pair, output, cases, timeout_error):
    # Every maintained page has positive old/old + new/new controls. Static
    # documents retain old coherent pins in the mixed case and perform no fetch.
    for page_name in PAGES:
        mode = modes[page_name]
        scenarios = [('old-positive', 'old', 'old', None),
                     ('new-positive', 'new', 'new', None),
                     ('old-html-new-payload', 'old', 'new', None)]
        if mode == 'interactive':
            scenarios += [('outer-budget-metadata', 'old', 'new', 'metadata'),
                          ('outer-budget-successor', 'old', 'new', 'successor')]
        controls_ok = True
        for scenario, html_version, body_version, stall in scenarios:
            case_id = page_name.removesuffix('.html') + '--' + scenario
            case = {'case': case_id, 'page': page_name, 'mode': mode, 'passed': False,
                    'disposition': 'NOT_COMPLETED', 'page_errors': [], 'requests': [],
                    'scope': SYNTHETIC_SCOPE, 'browser_requests': [], 'page_error_count': 0}
            cases.append(case)
            context, page = None, None
            server_rows = []
            try:
                context = browser.new_context(viewport={'width': 1200, 'height': 1000},
                                              reduced_motion='reduce', service_workers='block')
                if mode == 'interactive':
                    context.add_init_script(PROBE)
                page = context.new_page()
                page.set_default_timeout(5000)
                def page_error(error, row=case):
                    row['page_error_count'] += 1
                    if len(row['page_errors']) < 16:
                        row['page_errors'].append(str(error)[:2000])
                def request_seen(request, row=case):
                    if len(row['browser_requests']) < 128:
                        row['browser_requests'].append({'url': request.url[:2048],
                                                        'method': request.method,
                                                        'resource_type': request.resource_type})
                    else:
                        row['request_capture_overflow'] = True
                page.on('pageerror', page_error)
                page.on('request', request_seen)
                require(controls_ok or scenario.endswith('positive'), 'Positive page control failed')
                document = docs[(page_name, html_version)]
                case['served_html'] = identity(document)
                with serve(contents, document, page_name, pair, body_version, stall) as (origin, server_rows):
                    start = time.monotonic()
                    page.goto(origin + page_name, wait_until='domcontentloaded', timeout=10000)
                    initial_history = served_historical_links(document)
                    if mode == 'static':
                        page.wait_for_load_state('networkidle', timeout=5000)
                        case['surface'] = assert_surface(page, pair[html_version]['expected'], SetupFailure, verified=False)
                        case['static_dom'] = page.evaluate("""() => ({
                          policies:[...document.querySelectorAll('meta[http-equiv]')].filter(m=>m.httpEquiv.toLowerCase()==='content-security-policy').map(m=>m.content),
                          scripts:[...document.querySelectorAll('script')].map(s=>({type:s.type,src:s.getAttribute('src')}))})""")
                        require(any(re.search(r"script-src\s+'none'", policy) for policy in case['static_dom']['policies']),
                                'Static mode has no actual DOM script-none policy')
                        require(not any(r['resource_type'] in ('fetch', 'xhr') for r in case['browser_requests']),
                                'Script-free page performed an observed data request')
                        require(historical_links(page) == initial_history, 'Static historical link mismatch')
                        case['disposition'] = 'PASS_STATIC_COHERENT_OBSERVATION_NO_READER'
                    elif stall:
                        paths = ['current.json', 'current.json?observation=' + pair['old']['meta']['sha256'],
                                 'current-meta.json']
                        if stall == 'successor':
                            paths += ['current.json?observation=' + pair['new']['meta']['sha256']]
                        try:
                            page.wait_for_function("document.querySelector('#current-research .current-state')?.textContent.startsWith('Unavailable:')", timeout=21000)
                        except timeout_error as error:
                            case['timeout_probe'] = probe_rows(page)
                            request_contract(page, origin, paths)
                            require(any(r['target'] == '/site/' + paths[-1] and 'endpoint' in r
                                        for r in server_rows), 'Intended stalled request never reached loopback server')
                            raise BudgetRegression('Reached actual intended stalled request but no bounded program refusal') from error
                        rows = request_contract(page, origin, paths)
                        aborted = [r for r in rows if r.get('abortedAt') is not None]
                        require(len(aborted) == len(rows), 'Actual caller signal did not abort every shared request', BudgetRegression)
                        require(rows[-1].get('errorName') == 'AbortError',
                                'Stalled native fetch did not reject with AbortError', BudgetRegression)
                        budget_ms = rows[-1]['abortedAt'] - rows[0]['started']
                        case['observed_outer_abort_ms'] = budget_ms
                        require(14000 <= budget_ms <= 19000,
                                'Actual signal abort outside 15s scheduling envelope', BudgetRegression)
                        require(loaded_snapshot(page) is None, 'Aborted entrypoint admitted a snapshot', BudgetRegression)
                        require(time.monotonic() - start <= 21.5, 'Entrypoint did not settle after its abort', BudgetRegression)
                        search = page.locator('#current-research .current-search')
                        require(search.count() == 1 and search.is_disabled(), 'Aborted source search enabled', BudgetRegression)
                        case['disposition'] = 'PASS_ACTUAL_OUTER_15S_ABORT'
                    else:
                        page.wait_for_function("document.querySelector('#current-research .current-state')?.textContent.match(/^(Verified source observation:|Unavailable:)/)", timeout=10000)
                        admitted = loaded_snapshot(page)
                        expected = pair[body_version]['expected']
                        def source_url(source):
                            return ('https://github.com/' + source['repository'] + '/blob/' +
                                    source['commit'] + '/' + source['path'])
                        require(admitted is not None and admitted['observed_at'] == expected['observed_at']
                                and admitted['scientific_effect'] == 'NONE'
                                and {r['repository']: (r['commit'], r['tree']) for r in admitted['repositories']} == {
                                    'd6g8k5htny-coder/main': (expected['main_commit'], expected['main_tree']),
                                    'd6g8k5htny-coder/Math-': (expected['math_commit'], expected['math_tree'])}
                                and source_url(admitted['graph']) == expected['graph_url']
                                and [source_url(r) for r in admitted['references']] == expected['reference_urls'],
                                'Working-loader admission prerequisite failed; not UI RED')
                        case['verified_admission'] = admitted
                        paths = ['current.json']
                        if html_version != body_version:
                            paths += ['current.json?observation=' + pair['old']['meta']['sha256'],
                                      'current-meta.json',
                                      'current.json?observation=' + pair['new']['meta']['sha256']]
                        case['requests'] = request_contract(page, origin, paths)
                        kind = UIRegression if scenario == 'old-html-new-payload' else SetupFailure
                        case['surface'] = assert_surface(page, expected, kind, verified=True)
                        require(historical_links(page) == initial_history, 'Historical links were rewritten', kind)
                        case['disposition'] = 'PASS_VISIBLE_CURRENT_COHERENCE'
                    require(not case['page_errors'], 'Application page error; not qualified UI regression')
                    case['elapsed_seconds'] = round(time.monotonic() - start, 6)
                    case['requests'] = probe_rows(page)
                    case['server_requests'] = [dict(r) for r in server_rows]
                    case['passed'] = True
            except UIRegression as error:
                case.update(disposition='UI_COHERENCE_ASSERTION', error=str(error),
                            surface=getattr(error, 'observation', None))
            except BudgetRegression as error:
                case.update(disposition='OUTER_BUDGET_ASSERTION', error=str(error))
            except Exception as error:
                case.update(disposition='SETUP_OR_INCOMPLETE', error=str(error),
                            traceback=traceback.format_exc()[-6000:],
                            surface=getattr(error, 'observation', None))
                if scenario.endswith('positive'):
                    controls_ok = False
            finally:
                try:
                    case['server_requests'] = [dict(r) for r in server_rows]
                    if page is not None:
                        case['requests'] = probe_rows(page)
                        image = page.locator('#current-research').screenshot(timeout=3000)
                        case['screenshot'] = output.write(case_id + '.png', image)
                except Exception as error:
                    case['capture_error'] = str(error)[:2000]
                    case['passed'], case['disposition'] = False, 'SETUP_OR_INCOMPLETE'
                if context is not None:
                    try:
                        context.close()
                    except Exception as error:
                        case['close_error'] = str(error)[:2000]
                        case['passed'], case['disposition'] = False, 'SETUP_OR_INCOMPLETE'
                # Late page errors must also demote earlier behavior assertions.
                # An unrelated JS exception can never earn a qualified UI RED.
                if case['page_error_count'] or case.get('request_capture_overflow'):
                    case['passed'], case['disposition'] = False, 'SETUP_OR_INCOMPLETE'
                if scenario.endswith('positive') and not case['passed']:
                    controls_ok = False
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest', required=True, type=Path)
    parser.add_argument('--source-manifest-sha256', required=True)
    parser.add_argument('--context', required=True, type=Path)
    parser.add_argument('--context-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = Output(args.output)
    report = {'schema_version': 1, 'scientific_effect': 'NONE', 'scope': SYNTHETIC_SCOPE,
              'cases': [], 'disposition': 'SETUP_OR_INCOMPLETE',
              'limitations': ['Controlled local Chromium, not public byte custody.',
                              'No spoken AT/VoiceOver observation.',
                              'Script-free pages have no dynamic transition or abort claim.',
                              '15s is observed fetch abort/refusal, not a hard CPU/render deadline.']}
    code = 2
    try:
        contents, context = load_inputs(args.source_manifest, args.source_manifest_sha256,
                                        args.context, args.context_sha256)
        report['inputs'] = context
        # These are the real pinned producer/optional browser modules. No test
        # substitutes and no project execution during Source authoring/review.
        require(sys.dont_write_bytecode, 'Mandatory execution interface: python -B (clean checkout, no preexisting pyc)')
        verify_project_import_sources(contents)
        require('current_site_build' not in sys.modules, 'Producer was already imported before admission')
        from current_site_build import current_section, validate_payload
        require(Path(sys.modules['current_site_build'].__file__).resolve() == ROOT / 'tools/current_site_build.py',
                'Producer imported from a different path')
        verify_project_import_sources(contents)
        from playwright.sync_api import sync_playwright, TimeoutError as BrowserTimeout
        pair = synthetic_fixture_pair()
        docs, modes = build_documents(contents, pair, current_section, validate_payload)
        report['page_modes'] = modes
        report['fixtures'] = {}
        for version in ('old', 'new'):
            fixture = pair[version]
            report['fixtures'][version] = {'payload': output.write(version + '-current.json', fixture['raw']),
                                           'descriptor': output.write(version + '-current-meta.json', fixture['raw_meta']),
                                           'expected': fixture['expected']}
        report['served_documents'] = [output.write(page.removesuffix('.html') + '--' + version + '.html', body)
                                      for (page, version), body in sorted(docs.items())]
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel='chrome', chromium_sandbox=True)
            try:
                report['browser_version'] = browser.version
                exercise(browser, contents, docs, modes, pair, output, report['cases'], BrowserTimeout)
            finally:
                browser.close()
        cases = report['cases']
        expected_count = len(PAGES) * 3 + 2 * sum(m == 'interactive' for m in modes.values())
        require(len(cases) == expected_count and len({r['case'] for r in cases}) == expected_count,
                'Incomplete or duplicate case census')
        setup = [r['case'] for r in cases if r['disposition'] == 'SETUP_OR_INCOMPLETE']
        failures = [r['case'] for r in cases if not r['passed']]
        report['counts'] = {'expected': expected_count, 'observed': len(cases),
                            'passed': sum(r['passed'] for r in cases), 'failed': len(failures),
                            'setup_or_incomplete': len(setup), 'skipped': 0}
        if setup:
            code, report['disposition'] = 2, 'SETUP_OR_INCOMPLETE'
        elif failures:
            code, report['disposition'] = 1, 'QUALIFIED_BEHAVIOR_FAILURE'
        else:
            code, report['disposition'] = 0, 'CONTROLLED_BROWSER_CASES_PASS'
    except Exception as error:
        report.update(error=str(error), traceback=traceback.format_exc()[-6000:])
    finally:
        # The caller retains exit1; never expectedFailure/continue-on-error.
        report['exit_code'] = code
        try:
            raw_report = json_bytes(report)
            if len(raw_report) > REPORT_RESERVE:
                # Preserve observed case/control outcomes, but make the missing
                # detailed evidence an explicit non-qualifying failure.
                code = 2
                report = {'schema_version': 1, 'scientific_effect': 'NONE',
                          'disposition': 'SETUP_OR_INCOMPLETE', 'exit_code': 2,
                          'error': 'Detailed report exceeded the finite report reserve',
                          'unretained_detailed_report': identity(raw_report),
                          'cases': [{k: row[k] for k in ('case', 'page', 'mode', 'passed', 'disposition')}
                                    for row in report['cases']]}
                raw_report = json_bytes(report)
            output.write('report.json', raw_report)
            output.write('manifest.json', json_bytes({'schema_version': 1, 'members': list(output.members),
                                                      'write_attempts': list(output.attempts),
                                                      'manifest_self_excluded': True,
                                                      'bytes_before_manifest': output.total,
                                                      'maximum_bytes_including_manifest': MAX_OUTPUT}))
        except Exception as error:
            # Missing/partial footer is a transport failure, never a RED/PASS.
            print(json.dumps({'disposition': 'TERMINAL_EVIDENCE_INCOMPLETE',
                              'exit_code': 2, 'error': str(error)[:1000]}), file=sys.stderr)
            return 2
    print(json.dumps({'disposition': report['disposition'], 'exit_code': code,
                      'cases': len(report['cases'])}, sort_keys=True))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
