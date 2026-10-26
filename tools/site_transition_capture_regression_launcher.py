"""Source-only closed-harness launcher. NOT_RUN; requires a Root-frozen contract."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
import types

REAL_POPEN = subprocess.Popen
REAL_MONOTONIC = time.monotonic
TRACE_LIMIT = 65536
TRACE_ROW_CAP = 4096


def packed(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'))+'\n').encode('ascii')


def checked(entry, maximum=2097152):
    path=Path(entry['path'])
    if not path.is_absolute() or path.resolve()!=path:
        raise ValueError('Physical absolute Source path required')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        a=os.fstat(fd)
        if not stat.S_ISREG(a.st_mode) or a.st_nlink!=1 or not 0<=a.st_size<=maximum:
            raise ValueError('Unsafe Source')
        raw=b''
        while len(raw)<=maximum:
            part=os.read(fd,min(65536,maximum+1-len(raw)))
            if not part:break
            raw+=part
        b=os.fstat(fd)
        shape=lambda s:(s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        if shape(a)!=shape(b) or len(raw)!=entry['bytes'] or hashlib.sha256(raw).hexdigest()!=entry['sha256']:
            raise ValueError('Source identity mismatch')
        return raw
    finally:os.close(fd)


def proc_identity(pid):
    with open('/proc/%d/stat'%pid,'rb') as stream:raw=stream.read(4097)
    if len(raw)>4096:raise ValueError('Process identity cap')
    fields=raw.rsplit(b')',1)[1].split()
    return {'pid':pid,'ppid':int(fields[1]),'pgid':int(fields[2]),'start_ticks':int(fields[19])}


def validate_launch_cwd(contract,observed_cwd=None):
    """Record an actual physical execution cwd, never the contract label alone."""
    expected=contract['cwd']
    if type(expected) is not str:raise ValueError('Frozen cwd type')
    path=Path(expected)
    if not path.is_absolute() or path.resolve()!=path or not path.is_dir():
        raise ValueError('Physical frozen cwd required')
    observed=os.getcwd() if observed_cwd is None else observed_cwd
    if type(observed) is not str or observed!=expected or str(Path(observed).resolve())!=observed:
        raise ValueError('Observed launch cwd mismatch')
    return observed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract',required=True,type=Path)
    parser.add_argument('--contract-sha256',required=True)
    parser.add_argument('--selection',required=True,choices=['baseline','candidate'])
    parser.add_argument('--trace',required=True,type=Path)
    args=parser.parse_args()
    descriptor=os.open(args.contract,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        info=os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1 or info.st_size>65536:
            raise ValueError('Contract shape')
        raw=os.read(descriptor,65537)
        if len(raw)!=info.st_size:raise ValueError('Contract short read')
    finally:os.close(descriptor)
    if len(raw)>65536 or hashlib.sha256(raw).hexdigest()!=args.contract_sha256:
        raise ValueError('Contract identity')
    contract=json.loads(raw)
    if contract['status']!='ROOT_FROZEN_EXECUTION_CONTRACT' or sys.platform!='linux':
        raise ValueError('PENDING or unsupported platform')
    observed_cwd=validate_launch_cwd(contract)
    signal.signal(signal.SIGCHLD,signal.SIG_DFL)
    captured={entry['path']:checked(entry) for entry in contract['closure']}
    manifest=json.loads(captured[contract['source_manifest']['path']])
    if manifest.get('status')!='ROOT_FROZEN_FINAL_CANONICAL_COMPOSITION':
        raise ValueError('PENDING Source manifest')
    harness=captured[contract['harness']['path']]
    workflow=contract['workflows'][args.selection]
    if captured[workflow['path']]!=checked(workflow):raise ValueError('Workflow capture join')
    checked(contract['source_seal'])
    trace=os.open(args.trace,os.O_WRONLY|os.O_APPEND|os.O_NOFOLLOW|os.O_NONBLOCK)
    info=os.fstat(trace)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1:raise ValueError('Unsafe trace')
    serial=[0]
    def event(kind,**values):
        row=packed({'kind':kind,'sequence':serial[0],'monotonic':REAL_MONOTONIC(),**values})
        serial[0]+=1
        if len(row)>TRACE_ROW_CAP or os.fstat(trace).st_size+len(row)>TRACE_LIMIT:
            raise ValueError('Trace cap')
        if os.write(trace,row)!=len(row):raise ValueError('Partial trace')
    allowed=set(contract['allowed_child_argv_sha256'])
    class ObservedPopen(REAL_POPEN):
        # Class inheritance preserves Popen.wait(child,...), context-manager,
        # communicate/poll and intentional instance-level wait fault adapters.
        def __init__(self,*values,**kwargs):
            command=values[0] if values else kwargs.get('args')
            if not isinstance(command,(list,tuple)) or not all(type(x) is str for x in command):
                raise ValueError('Closed argv required')
            argv_sha=hashlib.sha256(packed(list(command))).hexdigest()
            if argv_sha not in allowed or kwargs.get('shell',False) or kwargs.get('executable') is not None:
                raise ValueError('Unqualified child argv')
            token=serial[0]
            actual_argv=list(command)
            event('SPAWN_BEGIN',token=token,argv=actual_argv,argv_sha256=argv_sha)
            try:
                super().__init__(*values,**kwargs)
                event('SPAWNED',token=token,argv=actual_argv,argv_sha256=argv_sha,**proc_identity(self.pid))
            except BaseException as error:
                event('SPAWN_FAILED',token=token,error=type(error).__name__)
                raise
    # Frozen executable bodies must not be hash-then-path-reread. The closed
    # harness reads workflow/child/probe/shadow bodies only through these two
    # Path methods. Mutable owned case files continue through real methods.
    real_read_bytes=Path.read_bytes
    real_read_text=Path.read_text
    def captured_bytes(path):
        body=captured.get(str(path))
        return body if body is not None else real_read_bytes(path)
    def captured_text(path,encoding=None,errors=None):
        body=captured.get(str(path))
        if body is None:return real_read_text(path,encoding=encoding,errors=errors)
        with io.TextIOWrapper(io.BytesIO(body),encoding=encoding,errors=errors) as stream:
            return stream.read()
    Path.read_bytes=captured_bytes
    Path.read_text=captured_text
    subprocess.Popen=ObservedPopen
    event('LAUNCHER_BEGIN',observed_cwd=observed_cwd,**proc_identity(os.getpid()))
    os.environ.pop('PYTHONPATH',None)
    os.environ['INLINE_WRAPPER_WORKFLOW']=workflow['path']
    os.environ['SITE_CAPTURE_SELECTION']=args.selection
    selectors=contract['selectors'][args.selection]
    sys.argv=[contract['harness']['path'],*selectors]
    module=types.ModuleType('__main__')
    module.__dict__.update(__file__=contract['harness']['path'],__package__=None,__spec__=None,
                           __builtins__=__builtins__,__capture_observer__=event,
                           __capture_contract__=contract)
    # unittest.main discovers sys.modules['__main__']; an anonymous exec dict
    # would silently discover zero harness tests.
    sys.modules['__main__']=module
    namespace=module.__dict__
    code=2
    try:
        # Authenticated captured bytes, not a later mutable-path runpy import.
        exec(compile(harness,contract['harness']['path'],'exec'),namespace)
        code=0
    except SystemExit as error:
        code=error.code if type(error.code) is int else 0 if error.code is None else 1
    finally:
        try:event('HARNESS_END',exit_code=code)
        finally:os.close(trace)
    return code


if __name__=='__main__':raise SystemExit(main())
