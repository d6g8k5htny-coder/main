"""Execution/custody controls for the frozen PR380 one-dimensional coefficient."""
import importlib.util, json, os, subprocess, sys, tempfile, time, unittest
from unittest import mock
from pathlib import Path

DRIVER=Path(__file__).resolve().parents[1]/'tools/lm006_oned_coefficient_replay.py'

class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(DRIVER.is_file(),'replay implementation missing')
        spec=importlib.util.spec_from_file_location('oned_replay',DRIVER)
        self.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.m)
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'repo';self.root.mkdir();self.logs=Path(self.tmp.name)/'logs';self.logs.mkdir()

    def fixture(self):
        pins={}
        for name in self.m.PINS:
            p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(('fixture:'+name+'\n').encode())
            pins[name]=self.m.git_blob(p.read_bytes())
        return pins

    def child(self, code, expected=b'ok\n', timeout=3, name='child'):
        return self.m.run_step([sys.executable,'-B','-S','-c',code],self.root,self.logs,name,expected,'exact',timeout)

    def test_snapshot_and_exact_packet_membership(self):
        pins=self.fixture();ids=self.m.validate_snapshot(self.root,pins)
        self.assertEqual(set(ids),set(pins))
        extra=self.root/self.m.PACKET/'extra';extra.write_text('x')
        with self.assertRaisesRegex(self.m.ReplayError,'packet_membership'): self.m.validate_snapshot(self.root,pins)

    def test_changed_missing_and_symlink_sources_rejected(self):
        pins=self.fixture();name=next(iter(pins));p=self.root/name;old=p.read_bytes()
        p.write_bytes(old+b'x')
        with self.assertRaisesRegex(self.m.ReplayError,'source_identity'): self.m.validate_snapshot(self.root,pins)
        p.unlink()
        with self.assertRaisesRegex(self.m.ReplayError,'source_missing'): self.m.validate_snapshot(self.root,pins)
        p.write_bytes(old); outside=Path(self.tmp.name)/'outside';outside.write_bytes(old);p.unlink();p.symlink_to(outside)
        with self.assertRaisesRegex(self.m.ReplayError,'source_symlink'): self.m.validate_snapshot(self.root,pins)

    def test_path_validation(self):
        for name in ('../x','/x','a/../b','a//b','a\\b','a/./b',''):
            with self.assertRaises(self.m.ReplayError): self.m.source_file(self.root,name)

    def test_success_process_record(self):
        row=self.child("print('ok')");self.assertEqual(row['status'],'PASS');self.assertFalse(row['timed_out'])
        self.assertEqual((self.logs/'child.stdout').read_bytes(),b'ok\n')

    def test_wrong_exit_output_and_stderr_rejected(self):
        cases=[("raise RuntimeError('x')",'unexpected_exit'),('pass','stdout_mismatch'),("print('bad')",'stdout_mismatch'),("import sys;print('ok');print('w',file=sys.stderr)",'unexpected_stderr')]
        for i,(code,reason) in enumerate(cases):
            with self.assertRaisesRegex(self.m.ReplayError,reason): self.child(code,name='bad'+str(i))

    def test_timeout_retains_partial_evidence(self):
        with self.assertRaisesRegex(self.m.ReplayError,'timeout'):
            self.child("import sys,time;print('s',flush=True);print('e',file=sys.stderr,flush=True);time.sleep(10)",timeout=.2)
        row=json.loads((self.logs/'child.process.json').read_text());self.assertTrue(row['timed_out']);self.assertEqual(row['failure'],'timeout')


    def test_secondary_timeout_still_writes_process_record(self):
        class Stuck:
            pid=424242
            def __init__(self,*args,**kwargs):
                kwargs['stdout'].write(b'started\\n');kwargs['stdout'].flush()
                kwargs['stderr'].write(b'partial\\n');kwargs['stderr'].flush()
            def wait(self,timeout=None):
                raise subprocess.TimeoutExpired(['fake'],timeout)
        with mock.patch.object(self.m.subprocess,'Popen',Stuck), mock.patch.object(self.m.os,'killpg') as kill:
            with self.assertRaisesRegex(self.m.ReplayError,'timeout'):
                self.m.run_step(['fake'],self.root,self.logs,'stuck',b'','exact',.1)
        kill.assert_called_once_with(424242,self.m.signal.SIGKILL)
        row=json.loads((self.logs/'stuck.process.json').read_text())
        self.assertEqual((row['status'],row['failure'],row['timed_out'],row['returncode']),('FAIL','timeout',True,None))
        self.assertIn('did not finish',row['detail'])
        self.assertEqual((self.logs/'stuck.stdout').read_bytes(),b'started\\n')
        self.assertEqual((self.logs/'stuck.stderr').read_bytes(),b'partial\\n')
        self.assertRegex(row['stdout_sha256'],r'^[0-9a-f]{64}$')
        self.assertRegex(row['stderr_sha256'],r'^[0-9a-f]{64}$')

    def test_existing_evidence_not_overwritten(self):
        self.child("print('ok')");before={p.name:p.read_bytes() for p in self.logs.iterdir()}
        with self.assertRaisesRegex(self.m.ReplayError,'step_evidence_exists'): self.child("print('changed')")
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.logs.iterdir()})

    def test_test_report_contract_requires_all_named_passes(self):
        good=('\n'.join(n+' (test_oned.OneDimensionalTests.'+n+') ... ok' for n in self.m.TEST_NAMES)+'\n\n'+'-'*70+'\nRan 15 tests in 0.001s\n\nOK\n').encode()
        self.m.check_output(b'',good,None,'tests')
        for bad in (good.replace(b' ... ok',b' ... FAIL',1),good+b'noise',good.replace(b'Ran 15',b'Ran 0')):
            with self.assertRaisesRegex(self.m.ReplayError,'test_report_mismatch'): self.m.check_output(b'',bad,None,'tests')

    def test_exact_output_contract_is_byte_exact(self):
        want=b'{"panels":1024,"independently_reviewed":false}\n';self.m.check_output(want,b'',want,'exact')
        for bad in (b'',want+b'\n',want.replace(b'false',b'true')):
            with self.assertRaisesRegex(self.m.ReplayError,'stdout_mismatch'): self.m.check_output(bad,b'',want,'exact')

    def test_host_identity(self):
        h='1'*40;env={'GITHUB_ACTIONS':'true','GITHUB_REPOSITORY':self.m.REPOSITORY,'GITHUB_SHA':h,'GITHUB_RUN_ID':'1','GITHUB_RUN_ATTEMPT':'2'}
        self.m.host_identity(h,env)
        env['GITHUB_SHA']='2'*40
        with self.assertRaisesRegex(self.m.ReplayError,'host_identity'): self.m.host_identity(h,env)

    def test_invalid_cli(self):
        for args in ([],['--mode','bad','--output','x'],['--mode','normal','--output','x','--panels','2048']):
            p=subprocess.run([sys.executable,'-B','-S',str(DRIVER),*args],capture_output=True);self.assertEqual(p.returncode,2)

    def test_workflow_contract(self):
        wf=DRIVER.parents[1]/'.github/workflows/lm006-oned-coefficient.yml';self.assertTrue(wf.is_file(),'workflow missing')
        text=wf.read_text()
        for phrase in ('permissions:\n  contents: read','fail-fast: false','mode: [normal, optimized]','if: always()',"python-version: '3.13.5'",'persist-credentials: false','if-no-files-found: error'):
            self.assertIn(phrase,text)
        self.assertNotIn('continue-on-error:',text);self.assertNotIn('pull_request_target:',text)


    def synthetic_project(self, bad_test=False, output='synthetic', mutate=False):
        import shutil
        previous=os.environ.get('GITHUB_ACTIONS')
        self.addCleanup(lambda: os.environ.pop('GITHUB_ACTIONS',None) if previous is None else os.environ.__setitem__('GITHUB_ACTIONS',previous))
        os.environ['GITHUB_ACTIONS']='false'
        pins=self.fixture(); packet=self.root/self.m.PACKET
        (packet/'RESULTS.json').write_text('synthetic\n')
        methods='\n'.join('    def '+n+'(self): self.assertTrue('+('False' if bad_test and i==0 else 'True')+')' for i,n in enumerate(self.m.TEST_NAMES))
        (packet/'test_oned.py').write_text('import unittest\nclass OneDimensionalTests(unittest.TestCase):\n'+methods+'\n')
        suffix="\nfrom pathlib import Path;Path('NOTE.md').write_text('changed')" if mutate else ''
        (packet/'oned.py').write_text("import sys\nassert sys.argv[1:]==['--panels','1024']\nprint("+repr(output)+")"+suffix+'\n')
        for name in self.m.EXECUTION_FILES:
            d=self.root/name;d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(DRIVER.parents[1]/name,d)
        subprocess.run(['git','init','-q'],cwd=self.root,check=True);subprocess.run(['git','add','.'],cwd=self.root,check=True)
        subprocess.run(['git','-c','user.name=Fixture','-c','user.email=f@example.invalid','-c','commit.gpgsign=false','commit','-qm','fixture'],cwd=self.root,check=True)
        spec=importlib.util.spec_from_file_location('fixture_driver',self.root/self.m.EXECUTION_FILES[0]);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        mod.PINS={n:mod.git_blob((self.root/n).read_bytes()) for n in pins}
        return mod

    def test_complete_orchestration_real_subprocesses(self):
        mod=self.synthetic_project();out=Path(self.tmp.name)/'ok';receipt=mod.execute(self.root,out,'optimized')
        self.assertEqual(receipt['status'],'PASS');self.assertEqual(set(receipt['steps']),{'tests','coefficient'});self.assertEqual((out/'coefficient.stdout').read_bytes(),b'synthetic\n')

    def test_failed_tests_stop_before_coefficient(self):
        mod=self.synthetic_project(bad_test=True);out=Path(self.tmp.name)/'badtests'
        with self.assertRaisesRegex(mod.ReplayError,'unexpected_exit'): mod.execute(self.root,out,'normal')
        self.assertFalse((out/'coefficient.stdout').exists());self.assertEqual(json.loads((out/'receipt.json').read_text())['status'],'FAIL')

    def test_wrong_output_cannot_pass(self):
        mod=self.synthetic_project(output='wrong');out=Path(self.tmp.name)/'wrong'
        with self.assertRaisesRegex(mod.ReplayError,'stdout_mismatch'): mod.execute(self.root,out,'normal')
        row=json.loads((out/'receipt.json').read_text());self.assertEqual(row['steps']['tests']['record']['status'],'PASS');self.assertEqual(row['steps']['coefficient']['record']['status'],'FAIL')

    def test_postflight_source_drift_rejected(self):
        mod=self.synthetic_project(mutate=True);out=Path(self.tmp.name)/'drift'
        with self.assertRaisesRegex(mod.ReplayError,'source_identity'): mod.execute(self.root,out,'normal')
        self.assertEqual(json.loads((out/'receipt.json').read_text())['status'],'FAIL')

if __name__=='__main__': unittest.main()
