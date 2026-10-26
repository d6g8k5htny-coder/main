"""Source-only closed inline-harness watchdog; Linux containment only, NOT_RUN.

Darwin intentionally refuses before launch: PS identity-then-kill is not atomic.
No arbitrary command API. Root must freeze/review all contract/Source first.
"""
import argparse
import ast
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import shlex
import signal
import stat
import sys
import time

CAP=1048576
TRACE_CAP=65536
BASELINE='InlineCaptureTests.test_missing_tap_does_not_skip_source_recheck_or_context_attempt'
FIXTURE_ROOT='tests/fixtures/site_transition_capture_regression_20261025'
FROZEN_MANIFEST='ROOT_FROZEN_FINAL_CANONICAL_COMPOSITION'
TRACE_ROW_CAP=4096
MEMBER_PATHS=tuple(sorted([
    'tools/site_transition_capture_regression.py',
    'tools/site_transition_capture_regression_supervisor.py',
    'tools/site_transition_capture_regression_launcher.py',
    *[FIXTURE_ROOT+'/inputs/member-%02d.txt'%number for number in range(16)],
    *[FIXTURE_ROOT+'/children/'+name+'.py' for name in ('capturedirectory','capturefifo','capturehardlink','capturesymlink','missingcapture','nonzero','outputcap','sourcechange','sourcefifo','success','timeout')],
    *[FIXTURE_ROOT+'/shadow/'+name+'.py' for name in ('base64','hashlib','json','pathlib','selectors')],
    FIXTURE_ROOT+'/ledger.json',FIXTURE_ROOT+'/isolation_probe.py',
    FIXTURE_ROOT+'/baseline-current-site-transition-candidate.yml.txt',
]))


def packed(value):return (json.dumps(value,sort_keys=True,indent=2)+'\n').encode('ascii')


def compact(value):return (json.dumps(value,sort_keys=True,separators=(',',':'))+'\n').encode('ascii')


def unique_object(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('Duplicate JSON key')
        result[key]=value
    return result


def json_value(raw):return json.loads(raw,object_pairs_hook=unique_object)


def exact_keys(value,keys,label):
    if type(value) is not dict or set(value)!=set(keys):raise ValueError(label+' schema')


def text_value(value,maximum=512):
    return type(value) is str and len(value)<=maximum


def integer(value,minimum=0):return type(value) is int and value>=minimum


def finite_number(value):
    return type(value) in (int,float) and math.isfinite(value) and value>=0


def digest(value,length=64):
    return type(value) is str and len(value)==length and all(c in '0123456789abcdef' for c in value)


def physical_root(contract):
    value=contract['cwd']
    if type(value) is not str:raise ValueError('Physical frozen cwd type')
    root=Path(value)
    if not root.is_absolute() or root.resolve()!=root or not root.is_dir():
        raise ValueError('Physical frozen cwd required')
    return root


def validate_launch_cwd(contract,observed_cwd=None):
    """Check observed execution cwd; this function does not change directories."""
    expected=str(physical_root(contract))
    observed=os.getcwd() if observed_cwd is None else observed_cwd
    if type(observed) is not str or observed!=expected:
        raise ValueError('Observed launch cwd mismatch')
    if str(Path(observed).resolve())!=observed:raise ValueError('Observed cwd is not physical')
    return observed


def manifest_row(row,root):
    exact_keys(row,('path','bytes','sha256','git_blob','role'),'Manifest row')
    relative=Path(row['path']) if type(row['path']) is str else None
    if relative is None or relative.is_absolute() or not relative.parts or '..' in relative.parts or str(relative)!=row['path']:
        raise ValueError('Manifest relative path')
    if not integer(row['bytes']) or not digest(row['sha256']) or not digest(row['git_blob'],40) or not text_value(row['role'],128):
        raise ValueError('Pending or malformed manifest row')
    path=root/relative
    if path.resolve()!=path:raise ValueError('Nonphysical manifest member')
    return {'path':str(path),'bytes':row['bytes'],'sha256':row['sha256']}


def closed_sources(contract):
    """Authenticate only the explicit durable manifest, never a recursive tree."""
    root=physical_root(contract);fixture=root/FIXTURE_ROOT
    if contract['regression_root']!=str(fixture):raise ValueError('Physical regression root')
    expected_paths={
        'harness':root/'tools/site_transition_capture_regression.py',
        'supervisor_source':root/'tools/site_transition_capture_regression_supervisor.py',
        'launcher_source':root/'tools/site_transition_capture_regression_launcher.py',
        'source_manifest':fixture/'source-fixture-manifest.json',
    }
    for label,path in expected_paths.items():
        if contract[label]['path']!=str(path):raise ValueError('Durable Source layout')
    if contract['workflows']['baseline']['path']!=str(fixture/'baseline-current-site-transition-candidate.yml.txt') or contract['workflows']['candidate']['path']!=str(root/'.github/workflows/current-site-transition-candidate.yml'):
        raise ValueError('Durable workflow layout')
    manifest=json_value(checked(contract['source_manifest']))
    exact_keys(manifest,('schema_version','status','fixture_root','members','external_inputs','selectors'),'Manifest')
    if type(manifest['schema_version']) is not int or manifest['schema_version']!=1 or manifest['status']!=FROZEN_MANIFEST:
        raise ValueError('PENDING Source manifest')
    if manifest['fixture_root']!=FIXTURE_ROOT or manifest['selectors']!=contract['selectors']:
        raise ValueError('Manifest root/selector join')
    if type(manifest['members']) is not list or len(manifest['members'])!=38:
        raise ValueError('Exact 38 Source members')
    if sorted(row['path'] for row in manifest['members'])!=list(MEMBER_PATHS):
        raise ValueError('Exact durable member path census')
    external=manifest['external_inputs']
    exact_keys(external,('guard','final_freeze'),'External Source inputs')
    rows=[*manifest['members'],external['guard'],external['final_freeze']]
    entries=[manifest_row(row,root) for row in rows]
    entries.extend([contract['source_manifest'],contract['workflows']['candidate']])
    if len({entry['path'] for entry in entries})!=42:raise ValueError('Duplicate durable closure')
    closure=contract['closure']
    if type(closure) is not list or len(closure)!=42:raise ValueError('Exact durable closure size')
    for entry in closure:exact_keys(entry,('path','bytes','sha256'),'Closure descriptor')
    actual={entry['path']:entry for entry in closure}
    if len(actual)!=42 or actual!={entry['path']:entry for entry in entries}:
        raise ValueError('Exact manifest/closure join')
    for label in ('harness','supervisor_source','launcher_source'):
        if actual.get(contract[label]['path'])!=contract[label]:raise ValueError('Executable Source closure join')
    for value in contract['workflows'].values():
        if actual.get(value['path'])!=value:raise ValueError('Workflow Source closure join')
    if contract['source_seal']['path'] in actual:raise ValueError('Source seal must be external')
    captured={path:checked(entry) for path,entry in actual.items()}
    for row,entry in zip(rows,entries):
        raw=captured[entry['path']]
        if hashlib.sha1(b'blob '+str(len(raw)).encode('ascii')+b'\0'+raw).hexdigest()!=row['git_blob']:
            raise ValueError('Manifest Git blob join')
    return captured


def workflow_entries(raw):
    """Select the four actual code-under-test entries from their named job."""
    text=raw.decode('utf-8')
    jobs=list(re.finditer(r'^  frozen-reader-transition-candidate:\n(.*?)(?=^  [A-Za-z0-9_-]+:|\Z)',text,re.M|re.S))
    if len(jobs)!=1:raise ValueError('Unique actual frozen core workflow job')
    blocks=re.findall(r"^          (python3[^\n]+) <<'(PY[A-Z]+)'\n",jobs[0].group(1),re.M)
    if len(blocks)!=4 or {name for _,name in blocks}!={'PYCONTEXT','PYGUARD','PYNODE','PYEMIT'}:
        raise ValueError('Authenticated workflow stdin entries')
    return {name:shlex.split(command) for command,name in blocks}


def derive_allowed_child_argv(contract,captured):
    """Derive the closed argv census from authenticated child/workflow bytes."""
    fixture=Path(contract['regression_root']);commands=[]
    for name in ('capturedirectory','capturefifo','capturehardlink','capturesymlink','missingcapture','nonzero','outputcap','sourcechange','sourcefifo','success','timeout'):
        raw=captured[str(fixture/'children'/(name+'.py'))]
        body=raw.decode('utf-8').replace('\r\n','\n').replace('\r','\n')
        commands.append([contract['interpreter']['path'],'-I','-B','-S','-c',body])
    for descriptor in contract['workflows'].values():
        commands.extend(workflow_entries(captured[descriptor['path']]).values())
    result={hashlib.sha256(compact(command)).hexdigest():command for command in commands}
    if len(result)!=13:raise ValueError('Exact derived 13 child argv')
    return [result[sha] for sha in sorted(result)]


def child_argv(contract,captured):
    result={hashlib.sha256(compact(command)).hexdigest():command for command in derive_allowed_child_argv(contract,captured)}
    if type(contract['allowed_child_argv_sha256']) is not list or contract['allowed_child_argv_sha256']!=sorted(result):
        raise ValueError('Derived allowed child argv join')
    return result


def derive_expected_children(contract,captured):
    """Closed Source census: run_case children plus the two stdin controls."""
    tree=ast.parse(captured[contract['harness']['path']]);methods={}
    for cls in tree.body:
        if isinstance(cls,ast.ClassDef) and cls.name=='InlineCaptureTests':
            methods={cls.name+'.'+method.name:method for method in cls.body if isinstance(method,ast.FunctionDef) and method.name.startswith('test_')}
    if len(methods)!=20 or contract['selectors']['candidate']!=sorted(methods) or contract['selectors']['baseline']!=[BASELINE]:
        raise ValueError('Exact Source selectors for child census')
    counts={}
    for name,method in methods.items():
        calls=[node for node in ast.walk(method) if isinstance(node,ast.Call)]
        cases=[node for node in calls if isinstance(node.func,ast.Attribute) and node.func.attr=='run_case']
        runs=[node for node in calls if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='subprocess' and node.func.attr=='run']
        if cases:
            if len(cases)!=1 or len(cases[0].args)!=1 or not isinstance(cases[0].args[0],ast.Constant) or type(cases[0].args[0].value) is not str or runs:
                raise ValueError('Closed run_case child census')
            counts[name]=0 if cases[0].args[0].value=='setupfailure' else 1
        elif name=='InlineCaptureTests.test_real_stdin_entrypoints_exclude_cwd_and_pythonpath_shadow' and len(runs)==1:
            counts[name]=len(workflow_entries(captured[contract['workflows']['candidate']['path']]))
        elif name=='InlineCaptureTests.test_actual_baseline_emitter_imports_shadow_and_candidate_emitter_holds_without_it' and len(runs)==2:
            counts[name]=len(runs)
        else:raise ValueError('Unqualified child-producing case Source')
    result={'baseline':counts[BASELINE],'candidate':sum(counts.values())}
    if result!={'baseline':1,'candidate':23}:raise ValueError('Exact derived durable child census')
    return result


def checked(entry, maximum=2097152):
    path=Path(entry['path'])
    if not path.is_absolute() or path.resolve()!=path:raise ValueError('Physical absolute path required')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        a=os.fstat(fd)
        if not stat.S_ISREG(a.st_mode) or a.st_nlink!=1 or not 0<=a.st_size<=maximum:
            raise ValueError('Nonregular/oversized Source')
        parts=[];size=0
        while size<=maximum:
            part=os.read(fd,min(65536,maximum+1-size))
            if not part:break
            parts.append(part);size+=len(part)
        b=os.fstat(fd);raw=b''.join(parts)
        shape=lambda s:(s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        if shape(a)!=shape(b) or len(raw)!=entry['bytes'] or hashlib.sha256(raw).hexdigest()!=entry['sha256']:
            raise ValueError('Source identity mismatch')
        return raw
    finally:os.close(fd)


def file_pin(path,maximum):
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        s=os.fstat(fd)
        if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1 or s.st_size>maximum:raise ValueError('Capture shape')
        raw=os.read(fd,maximum+1)
        if len(raw)!=s.st_size:raise ValueError('Capture short read')
        return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()},raw
    finally:os.close(fd)


def own_children():
    # This enumerates only this dedicated subreaper's direct children, never
    # foreign process commandlines. Do not reap until pidfd ownership is held.
    path='/proc/self/task/%d/children'%os.getpid()
    with open(path,'rb') as stream:raw=stream.read(16385)
    if len(raw)>16384:raise ValueError('Owned child list cap')
    values=[int(x) for x in raw.split()]
    if len(values)>128:raise ValueError('Owned child count cap')
    return values


def proc_identity(pid):
    with open('/proc/%d/stat'%pid,'rb') as stream:raw=stream.read(4097)
    if len(raw)>4096:raise ValueError('Owned process identity cap')
    fields=raw.rsplit(b')',1)[1].split()
    return {'pid':pid,'ppid':int(fields[1]),'pgid':int(fields[2]),
            'start_ticks':int(fields[19]),'state':fields[0].decode('ascii')}


def cleanup_owned(seconds=10):
    began=time.monotonic();deadline=began+seconds;observed={};errors=[]
    while time.monotonic()<deadline:
        pids=own_children()
        if not pids:return {'confirmed':True,'observed':list(observed.values()),'errors':errors,'elapsed_seconds':round(time.monotonic()-began,6)}
        for pid in pids:
            if time.monotonic()>=deadline:break
            fd=None
            try:
                # Parent is this single-threaded process, SIGCHLD is default,
                # no concurrent wait/reaper exists: unreaped PID cannot reuse.
                fd=os.pidfd_open(pid,0)
                identity=proc_identity(pid)
                if identity['ppid']!=os.getpid():raise ValueError('Direct-child ownership changed')
                signal.pidfd_send_signal(fd,signal.SIGKILL,None,0)
                waited,status=os.waitpid(pid,os.WNOHANG)
                observed[pid]={'pid':pid,'pidfd_signal':'SIGKILL','reaped':waited==pid,
                               'wait_status':status if waited==pid else None,'identity':identity}
                if len(observed)>256:raise ValueError('Owned cleanup trace cap')
            except ProcessLookupError:
                try:os.waitpid(pid,os.WNOHANG)
                except ChildProcessError:pass
            except (OSError,ChildProcessError) as error:
                if len(errors)>=32:raise ValueError('Cleanup error capacity')
                errors.append({'pid':pid,'error':type(error).__name__})
            finally:
                if fd is not None:os.close(fd)
        time.sleep(0.02)
    return {'confirmed':not own_children(),'observed':list(observed.values()),'errors':errors,'elapsed_seconds':round(time.monotonic()-began,6)}


def validate(contract,selection):
    if type(contract) is not dict or contract.get('status')!='ROOT_FROZEN_EXECUTION_CONTRACT':raise ValueError('PENDING execution contract')
    if contract['platform']!='linux' or not sys.platform.startswith('linux'):
        raise ValueError('SOURCE_HOLD_UNSUPPORTED_DARWIN_CONTAINMENT')
    if not hasattr(os,'pidfd_open') or not hasattr(signal,'pidfd_send_signal'):
        raise ValueError('SOURCE_HOLD_PIDFD_UNAVAILABLE')
    captured=closed_sources(contract)  # Reject pending Source manifest before any fork.
    if list(sys.version_info[:3])!=contract['interpreter']['version'] or str(Path(sys.executable).resolve())!=contract['interpreter']['path']:
        raise ValueError('Interpreter identity/version')
    checked(contract['interpreter'],268435456)
    ceiling=120 if selection=='baseline' else 180
    if type(contract['wall_seconds'][selection]) is not int or not 1<=contract['wall_seconds'][selection]<=ceiling:
        raise ValueError('Wall ceiling')
    if contract['selectors']['baseline']!=[BASELINE]:raise ValueError('Baseline selector')
    source=captured[contract['harness']['path']]
    tree=ast.parse(source)
    names=[c.name+'.'+n.name for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='InlineCaptureTests'
           for n in c.body if isinstance(n,ast.FunctionDef) and n.name.startswith('test_')]
    if len(names)!=20 or len(set(names))!=20 or contract['selectors']['candidate']!=sorted(names):
        raise ValueError('Exact 20 sorted candidate selectors')
    if contract['expected_children']!=derive_expected_children(contract,captured) or any(type(value) is not int for value in contract['expected_children'].values()):
        raise ValueError('Actual durable child census')
    child_argv(contract,captured)
    descriptor=os.pidfd_open(os.getpid(),0)
    try:signal.pidfd_send_signal(descriptor,0,None,0)
    finally:os.close(descriptor)
    return source


def validate_trace(body,contract,selection,integer_exit,leader_pid,supervisor_pid,leader_identity):
    """Validate evidence only. Trace fields never authorize process signals."""
    if type(body) is not bytes or not body or len(body)>=TRACE_CAP or not body.endswith(b'\n'):
        raise ValueError('Partial/oversized trace')
    lines=body.splitlines()
    if any(not line or len(line)+1>TRACE_ROW_CAP for line in lines):raise ValueError('Trace row bound')
    events=[json_value(line) for line in lines]
    if any(compact(row)!=line+b'\n' for row,line in zip(events,lines)):
        raise ValueError('Noncanonical trace encoding')
    base={'kind','sequence','monotonic'}
    identity={'pid','ppid','pgid','start_ticks'}
    schemas={
        'LAUNCHER_BEGIN':identity|{'observed_cwd'},
        'SPAWN_BEGIN':{'token','argv','argv_sha256'},
        'SPAWNED':identity|{'token','argv','argv_sha256'},
        'SPAWN_FAILED':{'token','error'},
        'CASE_BEGIN':{'case_id','runtime_id'},
        'CASE_END':{'case_id','runtime_id','status','exception_type','message','traceback_bytes','traceback_sha256'},
        'CASE_SUBTEST':{'case_id','subtest_id','status','exception_type','message'},
        'CASE_WITNESS':{'case_id','scenario','caught_type','caught_exit','pre_source_opens','post_source_opens','total_source_opens','save_attempts','stored','capture_errors','cleanup_errors','stdout_missing','child_count','elapsed_seconds','context_read_error'},
        'HARNESS_RESULT':{'tests_run','passed','failed','errors','skipped','expected_failures','unexpected_successes','ordered_cases','subtests'},
        'HARNESS_END':{'exit_code'},
    }
    expected=contract['selectors'][selection]
    captured={entry['path']:checked(entry) for entry in contract['closure']}
    allowed=child_argv(contract,captured)
    tree=ast.parse(captured[contract['harness']['path']])
    scenarios={}
    for cls in tree.body:
        if not isinstance(cls,ast.ClassDef) or cls.name!='InlineCaptureTests':continue
        for method in cls.body:
            if not isinstance(method,ast.FunctionDef):continue
            calls=[node for node in ast.walk(method) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='run_case']
            if calls:
                if len(calls)!=1 or len(calls[0].args)!=1 or not isinstance(calls[0].args[0],ast.Constant) or type(calls[0].args[0].value) is not str:
                    raise ValueError('Closed run_case scenario Source')
                scenarios[cls.name+'.'+method.name]=calls[0].args[0].value
    if len(scenarios)!=18:raise ValueError('Exact run_case scenario Source census')
    cases=[];completed=[];current=None;pending=None;tokens=set();identities=set();spawn_count=0
    case_spawns={};witnesses=set();statuses=[];subtests=0;subtest_labels=[];result=None;previous=-1.0;launcher=None
    isolation='InlineCaptureTests.test_real_stdin_entrypoints_exclude_cwd_and_pythonpath_shadow'
    nullable_string=lambda value: value is None or text_value(value)
    def check_identity(row):
        if any(not integer(row[key],1 if key!='start_ticks' else 0) for key in identity):raise ValueError('Trace process identity type')
    def check_errors(rows):
        if type(rows) is not list or len(rows)>8:raise ValueError('Witness error bound')
        for row in rows:
            exact_keys(row,('operation','error'),'Witness error')
            if not all(text_value(row[key],128) for key in row):raise ValueError('Witness error type')
    def check_state(state):
        if state is None:return
        exact_keys(state,('node_phase','capture_status','immutable_source_recheck','node_returncode','capture_cleanup_diagnostics'),'Witness state')
        for key in ('node_phase','capture_status','immutable_source_recheck','capture_cleanup_diagnostics'):
            if not nullable_string(state[key]):raise ValueError('Witness state type')
        if state['node_returncode'] is not None and type(state['node_returncode']) is not int:raise ValueError('Witness returncode type')
    for index,row in enumerate(events):
        if type(row) is not dict or type(row.get('kind')) is not str or row['kind'] not in schemas:
            raise ValueError('Unknown trace kind')
        kind=row['kind'];exact_keys(row,base|schemas[kind],kind)
        if type(row['sequence']) is not int or row['sequence']!=index:raise ValueError('Consecutive trace sequence')
        if not finite_number(row['monotonic']) or row['monotonic']<previous:raise ValueError('Monotonic trace timing')
        previous=row['monotonic']
        if kind=='LAUNCHER_BEGIN':
            if index!=0 or launcher is not None:raise ValueError('Unique first launcher endpoint')
            check_identity(row)
            if row['pid']!=leader_pid or row['ppid']!=supervisor_pid or row['pgid']!=leader_pid:
                raise ValueError('Observed launcher identity join')
            if type(leader_identity) is not dict or leader_identity.get('pid')!=leader_pid or leader_identity.get('ppid')!=supervisor_pid or row['start_ticks']!=leader_identity.get('start_ticks'):
                raise ValueError('Held leader process lifetime join')
            validate_launch_cwd(contract,row['observed_cwd']);launcher=row
            continue
        if launcher is None:raise ValueError('Missing launcher endpoint')
        if kind=='HARNESS_END':
            if index!=len(events)-1 or result is None or current is not None or pending is not None:
                raise ValueError('Unique last harness endpoint')
            if type(row['exit_code']) is not int or type(integer_exit) is not int or row['exit_code']!=integer_exit:
                raise ValueError('Trace/actual exit mismatch')
            expected_exit=1 if result['failed']+result['errors']+result['unexpected_successes'] else 0
            if integer_exit!=expected_exit:raise ValueError('Actual unittest result/exit join')
            continue
        if index==len(events)-1 or result is not None:raise ValueError('Missing or displaced final endpoint')
        if kind=='CASE_BEGIN':
            if current is not None or pending is not None or len(cases)>=len(expected) or row['case_id']!=expected[len(cases)]:
                raise ValueError('Canonical actual case order')
            if row['runtime_id']!='__main__.'+row['case_id']:raise ValueError('Case runtime identity')
            current=row['case_id'];cases.append(current);case_spawns[current]=0
        elif kind in ('SPAWN_BEGIN','SPAWNED','SPAWN_FAILED'):
            if current is None or not integer(row['token']):raise ValueError('Spawn outside actual case')
            if kind=='SPAWN_BEGIN':
                if pending is not None or row['token'] in tokens or row['token']!=index:raise ValueError('Unique ordered spawn token')
                argv=row['argv']
                if type(argv) is not list or not 1<=len(argv)<=16 or any(not text_value(value,2048) for value in argv):raise ValueError('Actual argv shape')
                sha=hashlib.sha256(compact(argv)).hexdigest()
                if row['argv_sha256']!=sha or allowed.get(sha)!=argv:raise ValueError('Actual argv/hash/Source join')
                tokens.add(row['token']);pending=row
            else:
                if pending is None or row['token']!=pending['token']:raise ValueError('Spawn must follow its unique begin')
                if kind=='SPAWN_FAILED':
                    if not text_value(row['error'],128):raise ValueError('Spawn failure type')
                    raise ValueError('Failed child spawn evidence')
                check_identity(row)
                if row['ppid']!=launcher['pid'] or row['argv']!=pending['argv'] or row['argv_sha256']!=pending['argv_sha256']:
                    raise ValueError('Spawned identity/actual argv join')
                expected_pgid=row['pid'] if len(row['argv'])==6 and row['argv'][4]=='-c' else launcher['pgid']
                if row['pid'] in (launcher['pid'],supervisor_pid) or row['pgid']!=expected_pgid or row['start_ticks']<launcher['start_ticks']:
                    raise ValueError('Spawned process lifetime/group join')
                process=(row['pid'],row['start_ticks'])
                if process in identities:raise ValueError('Duplicate spawned process lifetime')
                identities.add(process);pending=None;spawn_count+=1;case_spawns[current]+=1
        elif kind=='CASE_WITNESS':
            if current is None or pending is not None or row['case_id']!=current or current in witnesses or scenarios.get(current)!=row['scenario']:
                raise ValueError('Unique current-case witness/scenario join')
            if not nullable_string(row['caught_type']) or row['caught_exit'] is not None and type(row['caught_exit']) is not int:
                raise ValueError('Witness caught exception type')
            for key in ('pre_source_opens','post_source_opens','total_source_opens','child_count'):
                if not integer(row[key]):raise ValueError('Witness counter type')
            if row['pre_source_opens']+row['post_source_opens']!=row['total_source_opens'] or row['child_count']!=case_spawns[current]:
                raise ValueError('Witness actual counter join')
            if type(row['save_attempts']) is not list or len(row['save_attempts'])>2:raise ValueError('Witness save bound')
            for state in row['save_attempts']:check_state(state)
            check_state(row['stored']);check_errors(row['capture_errors']);check_errors(row['cleanup_errors'])
            if type(row['stdout_missing']) is not bool or not finite_number(row['elapsed_seconds']) or not nullable_string(row['context_read_error']):
                raise ValueError('Witness physical result types')
            witnesses.add(current)
        elif kind=='CASE_SUBTEST':
            if current is None or pending is not None or row['case_id']!=current or not text_value(row['subtest_id'],512) or row['status'] not in ('PASS','FAIL','ERROR') or not nullable_string(row['exception_type']) or not text_value(row['message']):
                raise ValueError('Current-case actual subtest schema')
            labels=re.findall(r"\(block='(PYCONTEXT|PYGUARD|PYNODE|PYEMIT)'\)$",row['subtest_id'])
            if current!=isolation or len(labels)!=1 or labels[0] in subtest_labels:
                raise ValueError('Unique actual stdin subtest label')
            if row['status']=='PASS' and (row['exception_type'] is not None or row['message']):raise ValueError('PASS subtest exception contradiction')
            if row['status']!='PASS' and not row['exception_type']:raise ValueError('Missing actual subtest exception')
            subtest_labels.extend(labels)
            subtests+=1
        elif kind=='CASE_END':
            if current is None or pending is not None or row['case_id']!=current or row['runtime_id']!='__main__.'+current:
                raise ValueError('Ordered unique case endpoint')
            status=row['status']
            if status not in ('PASS','FAIL','ERROR','SKIP','EXPECTED_FAILURE','UNEXPECTED_SUCCESS') or not nullable_string(row['exception_type']) or not text_value(row['message']) or not integer(row['traceback_bytes']):
                raise ValueError('Actual case result schema')
            if row['traceback_sha256'] is not None and not digest(row['traceback_sha256']):raise ValueError('Actual traceback hash')
            if (row['traceback_bytes']==0)!=(row['traceback_sha256'] is None):raise ValueError('Traceback length/hash join')
            if status=='PASS' and (row['exception_type'] is not None or row['message'] or row['traceback_bytes']):raise ValueError('PASS exception contradiction')
            if status in ('FAIL','ERROR','EXPECTED_FAILURE') and (not text_value(row['exception_type'],128) or not row['exception_type'] or not row['traceback_bytes']):raise ValueError('Missing actual failure exception')
            if current in scenarios and current not in witnesses:raise ValueError('Missing actual case witness')
            if current==isolation and subtest_labels!=['PYCONTEXT','PYGUARD','PYNODE','PYEMIT']:
                raise ValueError('Complete ordered actual stdin subtests')
            statuses.append(status);completed.append(current);current=None
        elif kind=='HARNESS_RESULT':
            if current is not None or pending is not None or cases!=expected or completed!=expected or row['ordered_cases']!=cases:
                raise ValueError('Complete actual unittest case census')
            keys={'passed':'PASS','failed':'FAIL','errors':'ERROR','skipped':'SKIP','expected_failures':'EXPECTED_FAILURE','unexpected_successes':'UNEXPECTED_SUCCESS'}
            if not integer(row['tests_run']) or row['tests_run']!=len(expected) or not integer(row['subtests']) or row['subtests']!=subtests:
                raise ValueError('Actual unittest result census')
            for key,status in keys.items():
                if not integer(row[key]) or row[key]!=statuses.count(status):raise ValueError('Actual unittest status counts')
            result=row
    if len(events)<4 or events[-1]['kind']!='HARNESS_END' or spawn_count!=contract['expected_children'][selection] or witnesses!={case for case in expected if case in scenarios}:
        raise ValueError('Incomplete trace/child/witness census')
    return {'verified_observed_cwd':launcher['observed_cwd'],'launcher_identity':{key:launcher[key] for key in identity},
            'spawned_children':spawn_count,'ordered_cases':cases,'harness_result':result,
            'case_results':[row for row in events if row['kind']=='CASE_END'],
            'witnesses':[row for row in events if row['kind']=='CASE_WITNESS'],
            'subtests':[row for row in events if row['kind']=='CASE_SUBTEST'],'events':events}


def validate_regression_outcome(summary,selection):
    """Classify actual validated observations separately from capture completion.

    This is an admission predicate over observed rows. It never manufactures a
    missing witness, result, exception or expected value in the evidence.
    """
    rows=summary['events'];result=summary['harness_result']
    ends=[row for row in rows if row['kind']=='CASE_END']
    witnesses=[row for row in rows if row['kind']=='CASE_WITNESS']
    subtests=[row for row in rows if row['kind']=='CASE_SUBTEST']
    starting={'node_phase':'starting','capture_status':'NOT_RUN','immutable_source_recheck':'pending',
              'node_returncode':None,'capture_cleanup_diagnostics':None}
    if selection=='baseline':
        if len(ends)!=1 or len(witnesses)!=1 or subtests or result['tests_run']!=1 or result['failed']!=1 or any(result[key] for key in ('passed','errors','skipped','expected_failures','unexpected_successes')):
            return 'HOLD'
        end=ends[0];witness=witnesses[0]
        if end['case_id']!=BASELINE or end['status']!='FAIL' or end['exception_type']!='AssertionError' or 'Actual post-child source read was skipped' not in end['message']:
            return 'HOLD'
        if witness['scenario']!='missingcapture' or witness['caught_type']!='FileNotFoundError' or witness['caught_exit'] is not None or (witness['pre_source_opens'],witness['post_source_opens'],witness['total_source_opens'])!=(16,0,16):
            return 'HOLD'
        if witness['save_attempts']!=[starting] or witness['stored']!=starting or witness['stdout_missing'] is not True or witness['child_count']!=1 or witness['context_read_error'] is not None or witness['capture_errors'] or witness['cleanup_errors']:
            return 'HOLD'
        return 'BASELINE_CAUSAL_RED'
    if selection!='candidate' or len(ends)!=20 or len(witnesses)!=18 or result['tests_run']!=20 or result['passed']!=20 or any(result[key] for key in ('failed','errors','skipped','expected_failures','unexpected_successes')):
        return 'HOLD'
    isolation='InlineCaptureTests.test_real_stdin_entrypoints_exclude_cwd_and_pythonpath_shadow'
    if len(subtests)!=4 or any(row['case_id']!=isolation or row['status']!='PASS' or row['exception_type'] is not None or row['message'] for row in subtests) or len({row['subtest_id'] for row in subtests})!=4:
        return 'HOLD'
    statuses={'success':'COMPLETED','nonzero':'COMPLETED','timeout':'TIMEOUT','outputcap':'REFUSED_OUTPUT_LIMIT',
        'missingcapture':'REFUSED_CAPTURE','unreadablecapture':'REFUSED_CAPTURE','exitwritefailure':'REFUSED_CAPTURE',
        'killfailure':'REFUSED_CLEANUP','reapfailure':'REFUSED_REAP','cleanupinterrupt':'REFUSED_CLEANUP',
        'sourcechange':'REFUSED_SOURCE_CHANGED','sourcefifo':'REFUSED_SOURCE_CHANGED','setupfailure':'REFUSED_SETUP',
        'contextsavefailure':'COMPLETED','capturefifo':'REFUSED_CAPTURE','capturedirectory':'REFUSED_CAPTURE',
        'capturesymlink':'REFUSED_CAPTURE','capturehardlink':'REFUSED_CAPTURE'}
    if {row['scenario'] for row in witnesses}!=set(statuses):return 'HOLD'
    required_capture={
        'missingcapture':('capture:tap.stdout.tap','FileNotFoundError'),
        'unreadablecapture':('capture:tap.stderr.txt','PermissionError'),
        'exitwritefailure':('exit_write','PermissionError'),
        'capturefifo':('capture:tap.stdout.tap','ValueError'),
        'capturedirectory':('capture:tap.stdout.tap','ValueError'),
        'capturesymlink':('capture:tap.stdout.tap','OSError'),
        'capturehardlink':('capture:tap.stdout.tap','ValueError'),
        'setupfailure':('capture','OSError')}
    required_cleanup={'killfailure':('killpg','PermissionError'),'reapfailure':('wait','TimeoutExpired')}
    for witness in witnesses:
        scenario=witness['scenario']
        if witness['pre_source_opens']!=16 or witness['post_source_opens']<1 or witness['total_source_opens']<17 or len(witness['save_attempts'])!=2 or witness['context_read_error'] is not None or witness['save_attempts'][0]!=starting:
            return 'HOLD'
        final=witness['save_attempts'][1]
        if final is None or final['capture_status']!=statuses[scenario] or final['node_phase']!=('not_started' if scenario=='setupfailure' else 'completed') or final['immutable_source_recheck']!=('failed' if scenario in ('sourcechange','sourcefifo') else 'passed') or final['capture_cleanup_diagnostics']!=('incomplete' if scenario=='cleanupinterrupt' else 'completed'):
            return 'HOLD'
        if witness['child_count']!=(0 if scenario=='setupfailure' else 1) or witness['stdout_missing']!=(scenario=='missingcapture'):
            return 'HOLD'
        if scenario=='contextsavefailure':
            if witness['stored']!=starting or witness['caught_type']!='PermissionError' or witness['caught_exit'] is not None:return 'HOLD'
        else:
            if witness['stored']!=final:return 'HOLD'
            if scenario=='cleanupinterrupt':
                if witness['caught_type']!='KeyboardInterrupt' or witness['caught_exit'] is not None:return 'HOLD'
            else:
                code=0 if scenario=='success' else 7 if scenario=='nonzero' else 1
                if witness['caught_type']!='SystemExit' or witness['caught_exit']!=code:return 'HOLD'
        if scenario in ('cleanupinterrupt','setupfailure'):
            if final['node_returncode'] is not None:return 'HOLD'
        elif type(final['node_returncode']) is not int:return 'HOLD'
        if scenario in ('success','nonzero') and final['node_returncode']!=(0 if scenario=='success' else 7):return 'HOLD'
        capture={(row['operation'],row['error']) for row in witness['capture_errors']}
        cleanup={(row['operation'],row['error']) for row in witness['cleanup_errors']}
        if scenario in required_capture:
            required=required_capture[scenario]
            allowed={required}|({('capture','FileNotFoundError')} if scenario=='missingcapture' else set())
            if required not in capture or not capture<=allowed:return 'HOLD'
        elif capture:return 'HOLD'
        if scenario in required_cleanup:
            if cleanup!={required_cleanup[scenario]}:return 'HOLD'
        elif cleanup:return 'HOLD'
    return 'CANDIDATE_20_GREEN'


def main():
    started=time.monotonic()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract',required=True,type=Path)
    parser.add_argument('--contract-sha256',required=True)
    parser.add_argument('--selection',required=True,choices=['baseline','candidate'])
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    root=args.output
    root.mkdir(mode=0o700)
    if root.resolve()!=root:raise ValueError('Physical owned output path')
    receipt={'schema_version':1,'disposition':'HOLD_NOT_RUN','selection':args.selection,
             'harness_status':'NOT_RUN','integer_exit':None,'timeout':False,'cap':False,
             'cleanup':{'confirmed':False},'source_before':{},'source_after':{},'scientific_effect':'NONE',
             'verified_observed_cwd':None,'regression_status':'HOLD'}
    leader_pid=None;leader_fd=None;enabled=False
    try:
        _,raw=file_pin(args.contract,65536)
        if len(raw)>65536 or hashlib.sha256(raw).hexdigest()!=args.contract_sha256:raise ValueError('Contract hash')
        receipt['contract_before']={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
        contract=json_value(raw);validate(contract,args.selection)
        for label in ['supervisor_source','launcher_source','source_seal']:checked(contract[label])
        _,executed_source=file_pin(Path(__file__),2097152)
        if checked(contract['supervisor_source'])!=executed_source:raise ValueError('Supervisor executed Source')
        deadline=started+contract['wall_seconds'][args.selection]
        for entry in [*contract['closure'],contract['source_seal']]:
            checked(entry);receipt['source_before'][entry['path']]={'bytes':entry['bytes'],'sha256':entry['sha256']}
            if time.monotonic()>=deadline:raise TimeoutError('Whole invocation setup wall')
        for name in ['home','tmp']:(root/name).mkdir(mode=0o700)
        staged=root/'launcher.py';launcher_bytes=checked(contract['launcher_source'])
        staged.write_bytes(launcher_bytes);staged.chmod(0o400)
        staged_contract=root/'execution-contract.json';staged_contract.write_bytes(raw);staged_contract.chmod(0o400)
        trace=root/'spawn-trace.jsonl';trace.write_bytes(b'')
        paths=[root/'stdout.bin',root/'stderr.bin']
        libc=ctypes.CDLL(None,use_errno=True)
        signal.signal(signal.SIGCHLD,signal.SIG_DFL)
        if libc.prctl(36,1,0,0,0)!=0:raise OSError(ctypes.get_errno(),'subreaper unavailable')
        enabled=True
        if own_children():raise ValueError('Dedicated supervisor already has children')
        interpreter=contract['interpreter']['path']
        # The physical binary may be named python3.12. The owned PATH exposes
        # only an explicit python3 symlink to that exact authenticated target.
        binary_dir=root/'bin';binary_dir.mkdir(mode=0o700)
        (binary_dir/'python3').symlink_to(interpreter)
        receipt['python3_alias']={'path':str(binary_dir/'python3'),'target':interpreter}
        env={'PATH':str(binary_dir),'HOME':str(root/'home'),
             'TMPDIR':str(root/'tmp'),'LANG':'C.UTF-8','LC_ALL':'C.UTF-8','TZ':'UTC',
             'PYTHONDONTWRITEBYTECODE':'1','INLINE_WRAPPER_WORKFLOW':contract['workflows'][args.selection]['path']}
        env['SITE_CAPTURE_SELECTION']=args.selection
        command=[interpreter,'-I','-B','-S',str(staged),'--contract',str(staged_contract),
                 '--contract-sha256',args.contract_sha256,'--selection',args.selection,'--trace',str(trace)]
        receipt.update(command=command,expected_cwd=contract['cwd'],environment=env,wall_seconds=contract['wall_seconds'][args.selection],stream_cap=CAP)
        if time.monotonic()>=deadline:
            receipt['timeout']=True
            raise TimeoutError('Whole invocation setup wall')
        with paths[0].open('xb') as out,paths[1].open('xb') as err:
            receipt['launch_attempted']=True
            # Parent returns from fork without waiting on a Popen preexec/
            # error-pipe handshake; watchdog covers child setup/exec as well.
            leader_pid=os.fork()
            if leader_pid==0:
                try:
                    os.setsid()
                    os.chdir(contract['cwd'])
                    validate_launch_cwd(contract)
                    resource.setrlimit(resource.RLIMIT_FSIZE,(CAP,CAP))
                    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
                    null=os.open(os.devnull,os.O_RDONLY)
                    os.dup2(null,0);os.dup2(out.fileno(),1);os.dup2(err.fileno(),2)
                    if null>2:os.close(null)
                    os.execve(interpreter,command,env)
                except BaseException as error:
                    try:os.write(2,('FIXED_LAUNCH_FAILED '+type(error).__name__+'\n').encode('ascii'))
                    finally:os._exit(126)
            leader_fd=os.pidfd_open(leader_pid,0)
            receipt['leader_identity_at_pidfd']=proc_identity(leader_pid)
            if receipt['leader_identity_at_pidfd']['ppid']!=os.getpid():raise ValueError('Original leader ownership changed')
            receipt['harness_status']='STARTED'
            receipt['leader_pid']=leader_pid
            while True:
                if any(p.stat().st_size>=CAP for p in paths) or trace.stat().st_size>=TRACE_CAP:
                    receipt['cap']=True;break
                waited,status=os.waitpid(leader_pid,os.WNOHANG)
                if waited==leader_pid:
                    receipt['integer_exit']=os.waitstatus_to_exitcode(status);break
                if time.monotonic()>=deadline:receipt['timeout']=True;break
                time.sleep(0.02)
    except BaseException as error:
        receipt['error']=type(error).__name__+': '+str(error)[:1000]
    finally:
        if leader_fd is not None:
            # This is the original child even after PID reuse. Signal/close
            # diagnostics cannot bypass the separate adoption cleanup below.
            try:signal.pidfd_send_signal(leader_fd,signal.SIGKILL,None,0)
            except ProcessLookupError:pass
            except BaseException as error:receipt['leader_signal_error']=type(error).__name__
            finally:
                try:os.close(leader_fd)
                except BaseException as error:receipt['leader_close_error']=type(error).__name__
        if enabled and receipt.get('launch_attempted'):
            try:receipt['cleanup']=cleanup_owned(10)
            except BaseException as error:receipt['cleanup']={'confirmed':False,'error':type(error).__name__}
        if leader_pid is not None and receipt['integer_exit'] is None:
            for row in receipt['cleanup'].get('observed',[]):
                if row['pid']==leader_pid and row['reaped']:
                    receipt['integer_exit']=os.waitstatus_to_exitcode(row['wait_status'])
            # No Popen fallback can invent exit0 after independent waitpid.
        try:
            if 'contract' in locals() and contract.get('status')=='ROOT_FROZEN_EXECUTION_CONTRACT':
                after_contract,_=file_pin(args.contract,65536)
                if after_contract!=receipt['contract_before']:raise ValueError('Execution contract changed')
                receipt['contract_after']=after_contract
                if 'staged' in locals():
                    after_staged,_=file_pin(staged,2097152)
                    if after_staged!={'bytes':len(launcher_bytes),'sha256':hashlib.sha256(launcher_bytes).hexdigest()}:raise ValueError('Staged launcher changed')
                    staged_contract_pin,_=file_pin(staged_contract,65536)
                    if staged_contract_pin!=receipt['contract_before']:raise ValueError('Staged contract changed')
                    receipt['staged_after']={'launcher':after_staged,'contract':staged_contract_pin}
                checked(contract['interpreter'],268435456)
                receipt['interpreter_after']={'path':contract['interpreter']['path'],
                    'sha256':contract['interpreter']['sha256'],'bytes':contract['interpreter']['bytes']}
                for entry in [*contract['closure'],contract['source_seal']]:
                    checked(entry);receipt['source_after'][entry['path']]={'bytes':entry['bytes'],'sha256':entry['sha256']}
        except BaseException as error:receipt['source_after_error']=type(error).__name__
        try:
            if leader_pid is not None:
                receipt['captures']={}
                for name,maximum in [('stdout.bin',CAP),('stderr.bin',CAP),('spawn-trace.jsonl',TRACE_CAP)]:
                    identity,body=file_pin(root/name,maximum);receipt['captures'][name]=identity
                    if identity['bytes']>=maximum:receipt['cap']=True
                    if name=='spawn-trace.jsonl':
                        summary=validate_trace(body,contract,args.selection,receipt['integer_exit'],leader_pid,os.getpid(),receipt['leader_identity_at_pidfd'])
                        receipt['verified_observed_cwd']=summary['verified_observed_cwd']
                        receipt['trace_validation']={key:summary[key] for key in ('verified_observed_cwd','launcher_identity','spawned_children','ordered_cases')}
                        receipt['regression_evidence']={key:summary[key] for key in ('harness_result','case_results','witnesses','subtests')}
                        receipt['regression_status']=validate_regression_outcome(summary,args.selection)
                        receipt['trace_complete']=True
        except BaseException as error:receipt['capture_or_trace_error']=type(error).__name__+': '+str(error)[:500]
        receipt['elapsed_seconds']=round(time.monotonic()-started,6)
        good=(leader_pid is not None and type(receipt['integer_exit']) is int and not receipt['timeout'] and not receipt['cap']
              and receipt['cleanup']['confirmed'] and not receipt['cleanup'].get('errors') and receipt.get('trace_complete') is True
              and receipt.get('verified_observed_cwd')==contract['cwd']
              and receipt['source_before']==receipt['source_after'] and not any(k in receipt for k in ['error','source_after_error','capture_or_trace_error','leader_signal_error','leader_close_error']))
        receipt['disposition']='CAPTURE_COMPLETE_REQUIRES_HARNESS_REVIEW' if good else 'HOLD_INCOMPLETE_OR_NOT_RUN'
        result=packed(receipt)
        if len(result)>262144:
            receipt={'schema_version':1,'disposition':'HOLD_RECEIPT_CAP',
                     'omitted_receipt_bytes':len(result),'omitted_receipt_sha256':hashlib.sha256(result).hexdigest(),
                     'harness_status':'UNKNOWN_REVIEW_ORIGINAL_CAPTURES','scientific_effect':'NONE'}
            result=packed(receipt)
        (root/'receipt.json').write_bytes(result)
    print(receipt['disposition'])
    return 0 if receipt['disposition']=='CAPTURE_COMPLETE_REQUIRES_HARNESS_REVIEW' else 2


if __name__=='__main__':raise SystemExit(main())
