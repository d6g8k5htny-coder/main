#!/usr/bin/env python3
"""Deterministic hosted controls only; never imports project/browser helpers.

Source authored only until separately observed hosted execution. Loads exactly
one hash-pinned staged supervisor from outside the immutable project checkout.
"""
import argparse
import base64
import hashlib
import os
from pathlib import Path
import runpy
import sys
import re
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supervisor', type=Path, required=True)
    parser.add_argument('--supervisor-sha256', required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args()
    root = args.evidence
    root.mkdir(mode=0o700)
    # Independent pending record exists even if source validation/loading fails.
    import json
    pending = {'schema_version':1,'capture_complete':False,'disposition':'HOLD_INCOMPLETE',
               'scientific_effect':'NONE'}
    (root/'route.json').write_text(json.dumps(pending,sort_keys=True)+'\n')
    raw = args.supervisor.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.supervisor_sha256:
        raise SystemExit('supervisor source hash mismatch; HOLD')
    s = runpy.run_path(str(args.supervisor), run_name='qualified_control_subject')
    packed, require = s['packed'], s['require']
    write_new, replace = s['write_new'], s['replace_json']
    rows = []
    env = {'PATH':'/usr/bin:/bin','LANG':'C.UTF-8','LC_ALL':'C.UTF-8',
           'PYTHONDONTWRITEBYTECODE':'1'}
    command = lambda text: [sys.executable,'-B','-I','-S','-c',text]
    cases = [
      ('exit-zero', command("import os;os.write(1,b'zero\\n');os.write(2,b'zero-err\\n')"),
       None, 0, True, b'zero\n', b'zero-err\n'),
      ('exit-seven', command("import os;os.write(1,b'seven\\n');raise SystemExit(7)"),
       None, 7, True, b'seven\n', b''),
      ('stdout-cap', command("import os;os.write(1,b'A'*8192)"),
       'STDOUT_CAP', None, False, b'A'*4096, None),
      ('stderr-cap', command("import os;os.write(2,b'B'*8192)"),
       'STDERR_CAP', None, False, None, b'B'*4096),
      ('pending-visible', command("from pathlib import Path;import json;"
       "p=Path('pending-visible');assert json.loads((p/'pending.json').read_text())['disposition']=='HOLD_INCOMPLETE';"
       "assert (p/'stdout.bin').is_file() and (p/'stderr.bin').is_file();print('pending-observed')"),
       None, 0, True, b'pending-observed\n', b''),
      ('launch-missing', [str((root/'intentionally-absent-child').resolve())],
       'SUPERVISOR_OR_LAUNCH_ERROR', None, False, b'', b''),
    ]
    child = "import signal,time,os;signal.signal(signal.SIGTERM,signal.SIG_IGN);print('DESC',os.getpid(),'START',open('/proc/self/stat').read().rsplit(')',1)[1].split()[19],flush=True);time.sleep(60)"
    # Both ordinary process-group and escaped-session descendants must disappear.
    for name, escaped in [('hang-group',False),('hang-escaped-session',True)]:
        text = ("import subprocess,sys,signal,time,os;signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                "p=subprocess.Popen([sys.executable,'-B','-I','-S','-c',"+repr(child)+"],start_new_session="+repr(escaped)+",stdout=subprocess.PIPE);"
                "print(p.stdout.readline().decode().strip(),flush=True);"
                "print('LEADER',os.getpid(),'DESCENDANT',p.pid,flush=True);time.sleep(60)")
        cases.append((name,command(text),'WALL_TIMEOUT',None,False,None,b''))
    text = ("import subprocess,sys;p=subprocess.Popen([sys.executable,'-B','-I','-S','-c',"+repr(child)+"],"
            "start_new_session=True,stdout=subprocess.PIPE);print(p.stdout.readline().decode().strip(),flush=True);print('leader-left',flush=True)")
    cases.append(('leader-exit-orphan',command(text),'DESCENDANTS_AFTER_LEADER_EXIT',0,False,None,b''))
    closed_child = "import signal,time,os;signal.signal(signal.SIGTERM,signal.SIG_IGN);print('READY',os.getpid(),'START',open('/proc/self/stat').read().rsplit(')',1)[1].split()[19],flush=True);os.close(1);os.close(2);time.sleep(60)"
    text = ("import subprocess,sys;p=subprocess.Popen([sys.executable,'-B','-I','-S','-c',"+repr(closed_child)+"],"
            "start_new_session=True,stdout=subprocess.PIPE);print(p.stdout.readline().decode().strip(),flush=True)")
    cases.append(('leader-exit-closed-pipes',command(text),'DESCENDANTS_AFTER_LEADER_EXIT',0,False,None,b''))
    all_ok = True
    for name, cmd, reason, rc, complete, expected_out, expected_err in cases:
        try:
            observed_start = time.monotonic()
            status = s['supervise'](cmd,root,root/name,env,deadline=3.0,pipe_cap=4096)
            observed_elapsed = time.monotonic()-observed_start
            out, err = (root/name/'stdout.bin').read_bytes(), (root/name/'stderr.bin').read_bytes()
            require(status['stop_reason'] == reason and status['complete'] is complete, 'classification')
            if rc is not None:
                require(status['child_returncode'] == rc, 'exit code')
            if name != 'launch-missing':
                require(status['cleanup_complete'], 'tree cleanup absent')
            if expected_out is not None:
                require(out == expected_out, 'stdout bytes')
            if expected_err is not None:
                require(err == expected_err, 'stderr bytes')
            if name.startswith('hang-') or name.startswith('leader-exit-'):
                require(len(status['tracked_processes']) >= 2, 'descendant not observed')
                require('KILL' in status['signals'], 'SIGTERM-ignore path not exercised')
            if name.startswith('hang-') or name.startswith('leader-exit-'):
                # Independent child-authored PID/start witness; do not use the
                # subject's process inventory or its elapsed-time assertion.
                markers = re.findall(rb'(?:DESC|READY) ([1-9][0-9]*) START ([1-9][0-9]*)',out)
                require(len(markers)==1, 'independent descendant witness missing/duplicate')
                for pid_raw, ticks_raw in markers:
                    pid, ticks = int(pid_raw), int(ticks_raw)
                    try:
                        raw = Path('/proc')/str(pid)/'stat'
                        body = raw.read_bytes()
                        require(len(body)<=4096,'independent proc witness cap')
                        actual_ticks=int(body.rsplit(b')',1)[1].split()[19])
                        require(actual_ticks != ticks,'independently witnessed descendant survives')
                    except FileNotFoundError:
                        pass
                lower, upper = (7.8,14.0) if name.startswith('hang-') else (4.8,12.0)
                require(lower<=observed_elapsed<=upper,'independent deadline/cleanup elapsed envelope')
            rows.append({'control':name,'passed':True,'expected_stop_reason':reason,
                'independent_elapsed_seconds':round(observed_elapsed,6),
                'status':s['identity']((root/name/'status.json').read_bytes())})
        except Exception as error:
            all_ok = False
            rows.append({'control':name,'passed':False,'error':type(error).__name__+': '+str(error)[:1500]})
    # Fixed binary members cover empty, chunk boundary, non-UTF8 and two chunks.
    members = {'empty.bin':b'', 'exact.bin':bytes(range(256))*12,
               'plus-one.bin':bytes(range(256))*12+b'\xff'}
    try:
        wire = s['carrier_bytes'](members,'qualification-control',True)
        begin, recovered = s['decode_carrier'](wire)
        require(begin['payload_complete'] and recovered == members, 'complete carrier roundtrip')
        write_new(root/'carrier-control.bin',wire)
        corruptions = {
          'missing-footer':wire[:wire.rfind(s['PREFIX'])],
          'truncated-byte':wire[:-1],
          'extra-frame':wire+wire.splitlines(keepends=True)[-1],
          'duplicate-member':wire.replace(b'"name":"exact.bin"',b'"name":"empty.bin"'),
          'invalid-base64':wire.replace(base64.b64encode(members['exact.bin']),b'!'+base64.b64encode(members['exact.bin'])[1:],1),
          'foreign-line':b'not-a-frame\n'+wire,
          'wrong-chunk-order':wire.replace(b'"n":0',b'"n":1',1),
          'wrong-hash':None,
        }
        marker=b'"sha256":"'
        at=wire.index(marker)+len(marker)
        nibble=b'1' if wire[at:at+1]==b'0' else b'0'
        corruptions['wrong-hash']=wire[:at]+nibble+wire[at+1:]
        # Rebind the outer digest on semantic corruptions so the decoder must
        # actually reject Base64/schema/order, rather than only a stale footer.
        def rebind_footer(candidate):
            lines = candidate.splitlines(keepends=True)
            raw_lines = [line[len(s['PREFIX']):] for line in lines]
            footer = s['strict'](raw_lines[-1])
            footer['stream_sha256'] = hashlib.sha256(b''.join(raw_lines[:-1])).hexdigest()
            return b''.join(lines[:-1])+s['PREFIX']+packed(footer)
        corruptions['boolean-index'] = wire.replace(b'"i":0',b'"i":false',1)
        for name in ('duplicate-member','invalid-base64','wrong-chunk-order','wrong-hash','boolean-index'):
            corruptions[name] = rebind_footer(corruptions[name])
        for name, corrupt in corruptions.items():
            require(corrupt != wire, 'corruption did not modify bytes: '+name)
            rejected = False
            try:
                s['decode_carrier'](corrupt)
            except Exception:
                rejected = True
            require(rejected,'invalid carrier accepted: '+name)
        # Incomplete payload has a complete transport but never adoption credit.
        incomplete = s['carrier_bytes']({'fallback.json':packed({'complete':False})},'fallback-control',False)
        require(s['decode_carrier'](incomplete)[0]['payload_complete'] is False, 'incomplete flag lost')
        rows.append({'control':'strict-complete-carrier','passed':True,
                     'negative_count':len(corruptions),'binary_members':len(members)})
    except Exception as error:
        all_ok = False
        rows.append({'control':'strict-complete-carrier','passed':False,'error':str(error)[:1500]})
    try:
        import contextlib
        import io
        subject = root/'oversize-carrier-control'
        subject.mkdir(mode=0o700)
        write_new(subject/'route.json',packed({'capture_complete':False}))
        s['precreate'](subject/'process')
        with (subject/'process/stdout.bin').open('ab') as raw:
            raw.write(b'finite-raw-fallback\n')
        oversized = subject/'oversized-control.bin'
        with oversized.open('xb') as raw:
            raw.truncate(s['RAW_MAX']+1)
        buffer = io.BytesIO()
        stream = io.TextIOWrapper(buffer,encoding='ascii')
        try:
            with contextlib.redirect_stdout(stream):
                result = s['emit'](subject)
            stream.flush()
            wire = buffer.getvalue()
        finally:
            stream.detach()
            # This synthetic sparse cap fixture is not browser evidence.
            oversized.unlink()
        begin, members = s['decode_carrier'](wire)
        require(result == 2 and begin['label'] == 'site-route-fallback' and
                begin['payload_complete'] is False, 'oversize did not remain incomplete')
        require(members['process/stdout.bin'] == b'finite-raw-fallback\n' and
                'process/pending.json' in members and 'process/status.json' in members,
                'oversize fallback lost finite raw/pending/status')
        require(s['strict'](members['fallback.json'])['omitted_original_outputs'] is True,
                'omitted bytes not explicit')
        write_new(root/'fallback-control.bin',wire)
        rows.append({'control':'oversize-carrier-fallback','passed':True,
                     'retained_raw_bytes':len(members['process/stdout.bin'])})
    except Exception as error:
        all_ok = False
        rows.append({'control':'oversize-carrier-fallback','passed':False,'error':str(error)[:1500]})
    try:
        # Pure Source-structure fixtures, never a browser admission context.
        # This real preexisting URL tests grammar only; no supervisor/native
        # qualification relation is asserted by reusing it in this fixture.
        grammar_url='https://github.com/d6g8k5htny-coder/main/pull/501#issuecomment-6352945290'
        context = {'schema_version':1,'head_sha':'1'*40,'base_sha':'2'*40,
            'tested_merge_sha':'3'*40,'source_manifest_sha256':'4'*64,
            'helper_sha256':'5'*64,'requirements_sha256':'6'*64,
            'loader_qualification':{'disposition':'30_OF_30_PASS','model_sha256':'7'*64,
                'native_receipt':grammar_url},
            'supervisor_qualification':{'disposition':'ROOT_ADOPTED_CONTROLS_PASS',
                'source_sha256':'8'*64,'native_receipt':grammar_url}}
        manifest_id, context_id = s['identity'](b'x'), s['identity'](b'y')
        inputs={'context':context,'source_manifest':manifest_id,'context_identity':context_id,
            'source_files':1,'source_bytes':1,
            'scope':'Receipt supplied by separately qualified route; this program does not authenticate GitHub offline.'}
        s['retained_inputs'](inputs,context,manifest_id,context_id,1,1)
        require(not s['same_json']({'schema_version':True},{'schema_version':1}),
                'canonical comparison collapsed boolean/integer')
        mutations=[]
        for value in (True,1.0):
            row=json.loads(json.dumps(inputs));row['context']['schema_version']=value
            mutations.append(('retained-schema-'+type(value).__name__,row))
        for name in ('source_manifest','context_identity'):
            for value in (True,1.0):
                row=json.loads(json.dumps(inputs));row[name]['bytes']=value
                mutations.append((name+'-bytes-'+type(value).__name__,row))
        for name in ('source_files','source_bytes'):
            row=json.loads(json.dumps(inputs));row[name]=True
            mutations.append((name+'-boolean',row))
        checks=[]
        for name,row in mutations:
            rejected=False
            try:
                s['retained_inputs'](row,context,manifest_id,context_id,1,1)
            except Exception:
                rejected=True
            require(rejected,'typed retained input accepted: '+name)
            checks.append({'case':name,'rejected':True})
        baseline={'page_errors':[],'page_error_count':0,'passed':True}
        s['clean_case'](baseline)
        patches=[('page-errors',{'page_errors':['late page error']}),
            ('page-error-count',{'page_error_count':1}),
            ('boolean-page-error-count',{'page_error_count':False}),
            ('request-overflow',{'request_capture_overflow':True}),
            ('present-false-overflow',{'request_capture_overflow':False}),
            ('empty-capture-error',{'capture_error':''}),('empty-close-error',{'close_error':''}),
            ('setup-traceback',{'traceback':''})]
        for disposition,passed in [('PASS_VISIBLE_CURRENT_COHERENCE',True),
                ('UI_COHERENCE_ASSERTION',False),('OUTER_BUDGET_ASSERTION',False)]:
            semantic_baseline={**baseline,'disposition':disposition,'passed':passed}
            if not passed: semantic_baseline['error']='Synthetic assertion for Source guard control only.'
            s['clean_case'](semantic_baseline)
            for name,patch in patches:
                row={**semantic_baseline,**patch}
                rejected=False
                try:
                    s['clean_case'](row)
                except Exception:
                    rejected=True
                require(rejected,'rebound label suppressed error evidence: '+name+'/'+disposition)
                checks.append({'case':name+'/'+disposition,'rejected':True})
        for name,row in [
            ('passed-with-error',{**baseline,'error':''}),
            ('assertion-without-error',{**baseline,'passed':False}),
            ('assertion-empty-error',{**baseline,'passed':False,'error':''}),
            ('assertion-nonstring-error',{**baseline,'passed':False,'error':False})]:
            rejected=False
            try:
                s['clean_case'](row)
            except Exception:
                rejected=True
            require(rejected,'contradictory assertion error field: '+name)
            checks.append({'case':name,'rejected':True})
        rows.append({'control':'typed-retained-inputs-and-error-evidence','passed':True,
            'checks':checks,'negative_count':len(checks),
            'scope':'Source-structure fixtures only; no native receipt relation or browser result qualified.'})
    except Exception as error:
        all_ok=False
        rows.append({'control':'typed-retained-inputs-and-error-evidence','passed':False,
                     'error':type(error).__name__+': '+str(error)[:1500]})
    report = {'schema_version':1,'controls':rows,'passed':all_ok,'count':len(rows),
        'supervisor_sha256':args.supervisor_sha256,'scientific_effect':'NONE',
        'runtime':{k:os.environ.get(k) for k in ('GITHUB_RUN_ID','GITHUB_RUN_ATTEMPT','GITHUB_SHA',
                   'GITHUB_REPOSITORY','SITE_HEAD_SHA','SITE_BASE_SHA')},
        'scope':'Supervisor/transport controls only; no browser or project helper executed.'}
    write_new(root/'controls-report.json',packed(report))
    replace(root/'route.json',{'schema_version':1,'capture_complete':True,
        'disposition':'SUPERVISOR_CONTROLS_PASS' if all_ok else 'SUPERVISOR_CONTROLS_FAIL',
        'exit_code':0 if all_ok else 1,'scientific_effect':'NONE'})
    return 0 if all_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
