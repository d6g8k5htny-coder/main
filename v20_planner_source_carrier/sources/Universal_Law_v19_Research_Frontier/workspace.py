#!/usr/bin/env python3
"""Local v18 research workspace; canonical proof engine remains an independent axis."""
from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit, parse_qs, quote

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'engine'))
from conductor import Coordinator, Handler, Server
from flow import Intake
from library import ResearchLibrary
from formal_queue import FormalQueue
from research_catalog import ResearchCatalog, UnknownDocument, ResearchCatalogError
from toolchain_setup import default_toolchain


class WorkspaceServer(Server):
    def __init__(self, address, handler, workdir, *, watch=False, source_root=None,
                 toolchain=None, prefetch=False):
        from intelligence import ResearchDirector
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        coord = Coordinator(self.workdir / 'engine')
        super().__init__(address, handler, coord)
        self.director = ResearchDirector(self.workdir / 'director', coordinator=coord)
        self.director.seed_plan()
        self.director.seed_milestones()
        self.library = ResearchLibrary(ROOT / 'literature/catalog.json', self.workdir / 'library')
        self.toolchain = toolchain if toolchain is not None else default_toolchain()
        self.research = ResearchCatalog(ROOT)
        self.lineage = json.loads((ROOT / 'LINEAGE_FINAL.json').read_text())
        self.formal_queue = FormalQueue(self.workdir, toolchain=self.toolchain, max_workers=1)
        self.stopping = threading.Event()
        self.intake_thread = None
        self.formal_inventory = {'state': 'blocked', 'reason': 'formal lane is unavailable'}
        try:
            import formal_lane
            self.formal_inventory = formal_lane.inventory(self.toolchain)
            self.formal_profiles = formal_lane.profiles()
        except ImportError:
            self.formal_profiles = []
        self.formal_lock = threading.Lock()
        self.cursor_lock = threading.Lock()
        self.director_cursor_mtime = None
        self.lease_deadlines = []
        self.prefetch_enabled = bool(prefetch)
        self.prefetch_thread = threading.Thread(target=self._prefetch_loop, daemon=True) if prefetch else None
        if watch and source_root:
            config = json.loads((ROOT / 'watch.json').read_text())
            self.intake = Intake(coord, source_root, config)
            self.intake_thread = threading.Thread(target=self.intake.watch,
                args=(self.stopping, 2), daemon=True)

    def start_services(self):
        self.coord.start()
        self.formal_queue.start()
        if self.intake_thread:
            self.intake_thread.start()
        if self.prefetch_thread:
            self.prefetch_thread.start()

    def stop_services(self):
        self.stopping.set()
        self.formal_queue.stop()
        if self.intake_thread and self.intake_thread.is_alive():
            self.intake_thread.join(timeout=20)
        if self.prefetch_thread and self.prefetch_thread.is_alive():
            self.prefetch_thread.join(timeout=20)
        self.coord.stop()

    @staticmethod
    def need_targets(director):
        result = set(director.get('target_record_ids', []))
        # Explicit vocabulary alignment, not a semantic inference or theorem
        # dependency: literature tags name research needs, canonical IDs name
        # preserved mathematical records.
        literature_needs = {
            'd5-second-factorial-upper': ['D5-second-factorial', 'multipoint-collisions',
                'height-window', 'conditional-count-moments'],
            'marked-kac-rice-elder-alignment': ['Borel-KR', 'AAL-CONTINUOUS-WEIGHT',
                'AAL-FINITE-MEASURES', 'ELDER-PERSISTENCE', 'persistence-measurability', 'C2-KERNEL'],
            'normalizer-floor-weighted-transfer': ['conditional-count-moments', 'MOMENT-JOINT'],
        }
        for task in director.get('tasks', []):
            if task.get('kind') == 'project_milestone' and not task.get('stale'):
                result.update(task.get('project_targets', []))
                result.update(task.get('bound_record_ids', []))
                result.update(literature_needs.get(task.get('milestone_id'), []))
        return sorted(result)

    def _prefetch_loop(self):
        # Proactive acquisition is opt-in; offline operation is the default.
        # Each service run has a hard attempt/byte budget, even if plans churn.
        previous, attempts, consumed = None, 0, 0
        while not self.stopping.is_set() and attempts < 12 and consumed < 24 * 1024 * 1024:
            director = self.director.snapshot()
            sequence = director['last_sequence']
            if sequence != previous:
                previous = sequence
                plan = self.library.plan_prefetch(self.need_targets(director),
                                                  max_items=min(3,12-attempts))
                if plan['source_ids']:
                    remaining=24*1024*1024-consumed
                    if remaining<1:
                        break
                    result = self.library.prefetch(plan['source_ids'],
                        max_items=min(3,12-attempts), max_total_bytes=remaining)
                    attempts += len(plan['source_ids'])
                    consumed += result['network_bytes_read']
            self.stopping.wait(60)

    def formal_state(self):
        receipts = []
        for parent in [ROOT / 'examples/formal', self.workdir / 'formal']:
            for path in sorted(parent.rglob('receipt.json')) if parent.exists() else []:
                try:
                    receipt = json.loads(path.read_text())
                    if not isinstance(receipt, dict):
                        raise ValueError('retained receipt must be an object')
                    receipts.append({**receipt, 'receipt_path': str(path.relative_to(ROOT))
                        if path.is_relative_to(ROOT) else 'runtime/formal/' + path.relative_to(self.workdir / 'formal').as_posix(),
                        'display_context': 'retained execution receipt; current verification requires replay'})
                except (ValueError, OSError):
                    receipts.append({'status': 'invalid_receipt', 'reason': 'unreadable retained receipt'})
        queued = self.formal_queue.snapshot()
        return {'inventory': self.formal_inventory,
                'inventory_context': 'Startup inventory only; every actual execution independently validates the current pinned toolchain.',
                'profiles': self.formal_profiles,
                'jobs': queued['jobs'], 'queue': queued,
                'receipts': receipts, 'full_theorem_certificates': 0,
                'scope': 'Reviewed arithmetic profiles only; canonical theorem gate remains independent.'}

    def cursor(self):
        # Cheap change detection: do not replay the mathematical state four
        # times per second for every connected browser.
        paths = [self.coord.journal.path, self.director.journal.path, self.library.journal.path,
                 self.formal_queue.journal]
        paths += list((self.workdir / 'formal').rglob('receipt.json'))
        stamps = [str(p.stat().st_mtime_ns) if p.exists() else '0' for p in paths]
        with self.cursor_lock:
            if stamps[1] != self.director_cursor_mtime:
                self.lease_deadlines = [c['expires_at'] for c in self.director.snapshot()['claims']
                                       if c.get('status') == 'active']
                self.director_cursor_mtime = stamps[1]
            expired = sum(time.time() >= deadline for deadline in self.lease_deadlines)
        return ':'.join(stamps) + ':expired=' + str(expired)

    def workspace_snapshot(self):
        director = self.director.snapshot()
        targets = self.need_targets(director)
        return {'version': '18.0', 'engine': self.coord.snapshot(), 'director': director,
                'literature': self.library.snapshot(targets), 'formal': self.formal_state(),
                'research': self.research.snapshot(), 'lineage': self.lineage,
                'capabilities': {'adaptive_policy': 'bounded registered strategies with checked artifacts',
                    'learning': 'durable contextual strategy history; no trained general model claimed',
                    'contradictions': 'typed exact proposition and scope; prose conflicts need explicit registration',
                    'prefetch': 'explicit known-source IDs; at most 4 sources and 32 MiB per request',
                    'formal_scope': '31 reviewed arithmetic propositions; no full UL certificate',
                    'continuous_intake': self.intake_thread is not None,
                    'proactive_prefetch': self.prefetch_enabled,
                    'hosting': 'local process; environment lifetime only'}}


class WorkspaceHandler(Handler):
    def do_GET(self):
        if not self._loopback_request():
            self.write_json({'error': 'loopback origin required'}, 403)
            return
        parsed = urlsplit(self.path)
        path = parsed.path
        if path in ('/', '/workspace.html'):
            source = ROOT / 'workspace.html'
            if not source.exists():
                self.write_json({'error': 'workspace interface unavailable'}, 503)
                return
            raw = source.read_text().replace('%%LOCAL_TOKEN%%', self.server.token)
            raw = raw.replace('%%BOOTSTRAP%%', json.dumps(self.server.workspace_snapshot()).replace('</', '<\\/'))
            self._asset(raw.encode(), 'text/html; charset=utf-8')
        elif path in ('/workspace.js', '/workspace.css', '/assets/workspace.js', '/assets/workspace.css'):
            source = ROOT / path.split('/')[-1]
            self._asset(source.read_bytes(), mimetypes.guess_type(source.name)[0] or 'application/octet-stream')
        elif path == '/api/workspace':
            self.write_json(self.server.workspace_snapshot())
        elif path == '/api/workspace/events':
            self._events()
        elif path == '/api/research/document':
            identity=parse_qs(parsed.query).get('id',[''])[0]
            try:
                descriptor, raw=self.server.research.read_document(identity)
                media=descriptor.get('media_type','text/plain')
                if parse_qs(parsed.query).get('download')==['1']:
                    self.send_response(200)
                    self.send_header('Content-Type',media)
                    self.send_header('Content-Disposition','attachment; filename="'+Path(descriptor['path']).name.replace('"','')+'"')
                    self.send_header('Content-Length',str(len(raw)))
                    self.send_header('X-Content-Type-Options','nosniff')
                    self.end_headers();self.wfile.write(raw)
                elif media.startswith('text/') or media in {'application/json','application/mermaid'}:
                    self.write_json({'document':descriptor,'content':raw.decode('utf8')})
                else:
                    self.write_json({'document':descriptor,'content':'Binary artifact; use the verified download.',
                        'download_url':'/api/research/document?id='+quote(identity)+'&download=1'})
            except UnknownDocument:
                self.write_json({'error':'unknown research document'},404)
            except (ResearchCatalogError,OSError,UnicodeError) as error:
                self.write_json({'error':str(error)},409)
        else:
            super().do_GET()

    def _asset(self, raw, content_type):
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(raw)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        try:
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _events(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('X-Accel-Buffering', 'no')
        self.end_headers()
        previous, ping = None, time.monotonic()
        try:
            while not self.server.stopping.is_set():
                cursor = self.server.cursor()
                if cursor != previous:
                    payload = json.dumps(self.server.workspace_snapshot(), separators=(',', ':'))
                    self.wfile.write(('id: ' + cursor + '\nevent: workspace\ndata: ' + payload + '\n\n').encode())
                    self.wfile.flush()
                    previous, ping = cursor, time.monotonic()
                elif time.monotonic() - ping > 10:
                    self.wfile.write(b': keep-alive\n\n')
                    self.wfile.flush()
                    ping = time.monotonic()
                self.server.stopping.wait(.25)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            return

    def do_POST(self):
        path = urlsplit(self.path).path
        if not path.startswith(('/api/director/', '/api/library/', '/api/formal/')):
            return super().do_POST()
        if not self._loopback_request() or self.headers.get('X-UL-Token') != self.server.token:
            self.write_json({'error': 'local write token and loopback origin required'}, 403)
            return
        try:
            n = int(self.headers.get('Content-Length', '0'))
            if not 1 <= n <= 30000:
                raise ValueError('request body must be between 1 and 30000 bytes')
            data = json.loads(self.rfile.read(n))
            if not isinstance(data, dict):
                raise ValueError('request must be a JSON object')
            director = self.server.director
            if path == '/api/director/claim':
                result = director.claim(**data)
            elif path == '/api/director/attempt':
                result = director.record_attempt(**data)
            elif path == '/api/director/assertion':
                result = director.add_assertion(**data)
            elif path == '/api/director/resolve':
                data['actor'] = 'human:local-workspace'
                result = director.resolve_conflict(**data)
            elif path == '/api/library/prefetch':
                result = self.server.library.prefetch(**data)
            elif path == '/api/formal/run':
                if set(data)-{'profile_id','event_id'} or 'profile_id' not in data:
                    raise ValueError('only a reviewed profile ID may be submitted')
                result = self.server.formal_queue.enqueue(**data)
            elif path == '/api/formal/retry':
                if set(data)!={'job_id'}:
                    raise ValueError('retry requires only a job ID')
                result = self.server.formal_queue.retry(data['job_id'])
            else:
                self.write_json({'error': 'not found'}, 404)
                return
            self.write_json(result)
        except (ValueError, KeyError, TypeError, OSError) as error:
            self.write_json({'error': str(error)}, 400)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workdir', type=Path, default=ROOT / 'runtime')
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--lean-toolchain', type=Path, default=os.environ.get('UL_LEAN_TOOLCHAIN'))
    parser.add_argument('--port', type=int, default=8772)
    parser.add_argument('--prefetch', action='store_true', help='acquire top missing primary sources on plan changes; bounded 24 MiB per service run')
    args = parser.parse_args()
    server = WorkspaceServer(('127.0.0.1', args.port), WorkspaceHandler, args.workdir,
        watch=bool(args.source_root), source_root=args.source_root, toolchain=args.lean_toolchain,
        prefetch=args.prefetch)
    server.start_services()
    print(json.dumps({'dashboard': 'http://127.0.0.1:' + str(server.server_port),
                      'workdir': str(args.workdir), 'continuous_intake': bool(args.source_root)}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        server.stop_services()


if __name__ == '__main__':
    main()
