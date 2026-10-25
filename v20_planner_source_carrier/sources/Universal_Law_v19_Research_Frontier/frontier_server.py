#!/usr/bin/env python3
"""Local v19 control surface; native proof authority remains independently checked."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import signal
import threading
import time
from urllib.parse import urlsplit

from workspace import WorkspaceServer, WorkspaceHandler, ROOT
from frontier_runtime import FrontierRuntime
from frontier_gpu import stream_positions
from frontier_ir import bounded_json

ASSETS={'frontier.html':'text/html; charset=utf-8', 'frontier.js':'text/javascript; charset=utf-8',
        'frontier.css':'text/css; charset=utf-8','frontier_gpu.js':'text/javascript; charset=utf-8',
        'frontier_worker.js':'text/javascript; charset=utf-8'}


def exact(data, required=(), optional=()):
    if not isinstance(data,dict) or set(required)-set(data) or set(data)-set(required)-set(optional):
        raise ValueError('unexpected or missing action fields')
    return data


class FrontierServer(WorkspaceServer):
    def __init__(self,address,handler,workdir,**options):
        super().__init__(address,handler,workdir,**options)
        self.frontier=FrontierRuntime(workdir,self.coord,toolchain=self.toolchain)

    def start_services(self):
        super().start_services()
        self.frontier.start()

    def stop_services(self):
        self.stopping.set()
        if hasattr(self,'frontier'):self.frontier.stop()
        super().stop_services()

    def workspace_snapshot(self):
        state=super().workspace_snapshot()
        state['distribution']='Universal Law v19 Research Frontier'
        state['frontier_url']='/frontier'
        return state


class FrontierHandler(WorkspaceHandler):
    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy','same-origin')
        self.send_header('Cross-Origin-Embedder-Policy','require-corp')
        self.send_header('Cross-Origin-Resource-Policy','same-origin')
        super().end_headers()

    def do_GET(self):
        path=urlsplit(self.path).path
        if not (path.startswith('/api/frontier/') or path in ('/frontier','/frontier/') or path.removeprefix('/') in ASSETS):
            return super().do_GET()
        if not self._loopback_request():
            self.write_json({'error':'loopback origin required'},403);return
        try:
            if path in ('/frontier','/frontier/','/frontier.html'):
                raw=(ROOT/'frontier.html').read_text().replace('%%LOCAL_TOKEN%%',self.server.token)
                if '%%BOOTSTRAP%%' in raw:
                    raw=raw.replace('%%BOOTSTRAP%%',json.dumps(self.server.frontier.snapshot()).replace('</','<\\/'))
                self._asset(raw.encode(),ASSETS['frontier.html'])
            elif path.removeprefix('/') in ASSETS:
                name=path[1:]
                self._asset((ROOT/name).read_bytes(),ASSETS[name])
            elif path=='/api/frontier/state':
                self.write_json(self.server.frontier.snapshot())
            elif path=='/api/frontier/events':
                self._frontier_events()
            elif path=='/api/frontier/positions':
                self._asset(self.server.frontier.positions(),'application/octet-stream')
            elif path=='/api/frontier/stream':
                stream_positions(self,self.server.frontier.positions,self.server.stopping)
            else:self.write_json({'error':'not found'},404)
        except (ValueError,TypeError,KeyError,OSError) as error:
            self.write_json({'error':str(error)},409)

    def _frontier_events(self):
        self.send_response(200)
        self.send_header('Content-Type','text/event-stream')
        self.send_header('Cache-Control','no-cache')
        self.send_header('X-Accel-Buffering','no')
        self.end_headers()
        previous,ping=None,time.monotonic()
        try:
            while not self.server.stopping.is_set():
                current=self.server.frontier.cursor()
                if current!=previous:
                    state=self.server.frontier.snapshot()
                    payload=json.dumps(state,separators=(',',':'),allow_nan=False)
                    self.wfile.write(('id: '+current+'\nevent: frontier\ndata: '+payload+'\n\n').encode())
                    self.wfile.flush();previous,ping=current,time.monotonic()
                elif time.monotonic()-ping>10:
                    self.wfile.write(b': keep-alive\n\n');self.wfile.flush();ping=time.monotonic()
                self.server.stopping.wait(.25)
        except (BrokenPipeError,ConnectionResetError,TimeoutError):pass

    def do_POST(self):
        path=urlsplit(self.path).path
        if not path.startswith('/api/frontier/'):
            return super().do_POST()
        if not self._loopback_request() or self.headers.get('X-UL-Token')!=self.server.token:
            self.write_json({'error':'local write token and loopback origin required'},403);return
        try:
            if len(self.headers.get_all('Content-Length',[]))!=1:
                raise ValueError('one Content-Length required')
            n=int(self.headers['Content-Length'])
            if not 1<=n<=65536:raise ValueError('action body must contain 1 to 65536 bytes')
            data=json.loads(self.rfile.read(n))
            if not isinstance(data,dict):raise ValueError('action must be a JSON object')
            data=bounded_json(data,'action',65536)
            rt=self.server.frontier
            if path=='/api/frontier/plan':
                exact(data,('target_ids',),('budget',))
                result=rt.conductor.blueprint(**data)
            elif path=='/api/frontier/claim':
                exact(data,('task_id',),('agent_id','strategy','lease_seconds'))
                if data.get('agent_id','local-owner')!='local-owner':raise ValueError('owner UI claims use local-owner identity')
                result=rt.conductor.claim(**dict(data,agent_id='local-owner'))
            elif path=='/api/frontier/failure':
                exact(data,('claim_id','diagnosis'),('evidence',))
                result=rt.conductor.submit_failure(**data)
            elif path=='/api/frontier/belief':
                exact(data,('proposition','scope','polarity','source'),('evidence','input_fingerprint','record_id'))
                result=rt.beliefs.assert_claim(**data)
            elif path=='/api/frontier/run-math':
                exact(data,(),('event_id',))
                result=rt.queue_math(**data)
            elif path=='/api/frontier/redteam':
                exact(data,(),('power','claimed_upper_bound'))
                from counterexample_probe import probe_bound,verify_counterexample
                from moment_certificates import make_certificate
                certificate=make_certificate()
                probe=probe_bound(certificate,**data)
                verification=verify_counterexample(probe,certificate)
                if verification['valid']:
                    rt.beliefs.assert_claim(probe['question']+' Power '+str(probe['power'])+'; bound '+str(probe['claimed_upper_bound']),
                        {'kind':'generic_moment_family','law':certificate['law'],'domain':certificate['domain']},
                        'refute','local:exact-moment-probe',evidence={'probe':probe,'verification':verification},
                        input_fingerprint=certificate['certificate_sha256'])
                result={'probe':probe,'verification':verification,'scope':'Generic law only; no canonical record changed.'}
            elif path=='/api/frontier/learning':
                exact(data,(),('epochs','seed'))
                result=rt.learn(**data)
            elif path=='/api/frontier/branch':
                exact(data,('name','edits'),('base_hash',))
                result=rt.evolution.create_branch(**dict(data,actor='local-owner'))
            elif path=='/api/frontier/branch/stage':
                exact(data,('branch_id',))
                result=rt.evolution.stage(data['branch_id'],actor='local-owner')
            elif path=='/api/frontier/source-update':
                exact(data,('source_id','old_sha256','new_sha256','url'))
                result=rt.evolution.propose_source_update(**dict(data,actor='local-owner'))
            elif path=='/api/frontier/annotations':
                exact(data,('annotation_id',),('record_id','text','tag','action'))
                change=dict(data)
                change.setdefault('action','set')
                result=rt.annotations.apply(change,actor='local-owner')
            elif path=='/api/frontier/annotations/merge':
                # This only merges the annotation CRDT. It cannot contain graph,
                # acceptance, source pin, job, checker or certificate mutations.
                exact(data,('state',))
                result=rt.annotations.merge(data['state'])
            else:self.write_json({'error':'not found'},404);return
            self.write_json(result)
        except (ValueError,KeyError,TypeError,OSError,OverflowError,RecursionError) as error:
            self.write_json({'error':str(error)},400)


def main():
    from setup_final import setup
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8776)
    parser.add_argument('--workdir',type=Path,default=ROOT/'runtime')
    parser.add_argument('--lean-toolchain',type=Path)
    parser.add_argument('--source-root',type=Path)
    parser.add_argument('--prefetch',action='store_true')
    args=parser.parse_args()
    prepared=setup(args.lean_toolchain)
    server=FrontierServer(('127.0.0.1',args.port),FrontierHandler,args.workdir,
        toolchain=args.lean_toolchain,watch=bool(args.source_root),source_root=args.source_root,prefetch=args.prefetch)
    def interrupted(signum,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    server.start_services()
    print(json.dumps({'dashboard':'http://127.0.0.1:'+str(server.server_port)+'/frontier',
        'existing_workspace':'http://127.0.0.1:'+str(server.server_port)+'/',
        'formal_setup':prepared['formal']['state'],'full_theorem_certificates':0}),flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.stop_services();server.server_close()


if __name__=='__main__':main()
