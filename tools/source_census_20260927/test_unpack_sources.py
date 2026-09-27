"""Inert fixture controls for exact-source extraction, never executing inputs."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

class ExtractionTests(unittest.TestCase):
    def tool(self):
        path=Path(__file__).with_name('unpack_sources.py')
        self.assertTrue(path.is_file(), 'source unpacker not implemented')
        spec=importlib.util.spec_from_file_location('unpacker_test',path)
        m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
    def manifest(self,name,data):
        return {name:{'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}}
    def marked(self,name,payload,sha=None):
        h=sha or hashlib.sha256(payload).hexdigest()
        return (f'<!-- BEGIN SOURCE: {name} | sha256:{h[:12]} | status:historical -->\n\n'.encode()+payload+b'\n\n'+f'<!-- END SOURCE: {name} -->'.encode())
    def test_exact_marker_boundary_is_preserved(self):
        m=self.tool();data=b'# proof\n  line\n';r,e=m.extract_marked(self.marked('a.md',data),self.manifest('a.md',data))
        self.assertEqual(len(r),1);self.assertEqual(r[0]['payload'],data);self.assertEqual(e,[])
    def test_changed_content_not_accepted(self):
        m=self.tool();data=b'proof\n';r,e=m.extract_marked(self.marked('a.md',b'wrong\n'),self.manifest('a.md',data));self.assertEqual(r,[]);self.assertTrue(e)
    def test_hash_prefix_alone_not_sufficient(self):
        m=self.tool();data=b'proof';r,e=m.extract_marked(self.marked('a',data),{'a':{'sha256':hashlib.sha256(data).hexdigest()[:12],'bytes':5}});self.assertEqual(r,[])
    def test_missing_end_is_not_silent(self):
        m=self.tool();d=self.marked('a',b'proof').replace(b'END SOURCE: a',b'END SOURCE: wrong');r,e=m.extract_marked(d,self.manifest('a',b'proof'));self.assertEqual(r,[]);self.assertTrue(e)
    def test_size_is_checked_even_when_hash_matches(self):
        m=self.tool();mf=self.manifest('a',b'proof');mf['a']['bytes']=6;r,e=m.extract_marked(self.marked('a',b'proof'),mf);self.assertEqual(r,[])
    def test_no_path_traversal_names(self):
        m=self.tool();r,e=m.extract_marked(self.marked('../bad',b'proof'),self.manifest('../bad',b'proof'));self.assertEqual(r,[])
    def test_python_literals_decoded_without_execution(self):
        m=self.tool();data=b'print("inert")\n';h=hashlib.sha256(data).hexdigest()
        with tempfile.TemporaryDirectory() as d:
            marker=Path(d)/'ran'
            s=f'open({str(marker)!r},"w").write("bad")\n_S64={{}}\nHASHES={{}}\n_S64["a.py"]={base64.b64encode(data).decode()!r}\nHASHES["a.py"]={h!r}\n'.encode()
            r,e=m.extract_python(s,self.manifest('a.py',data));self.assertFalse(marker.exists());self.assertEqual(r[0]['payload'],data)
    def test_internal_prefix_requires_external_full_manifest(self):
        m=self.tool();data=b'print("inert")\n';h=hashlib.sha256(data).hexdigest()
        src=f'_S64={{"a.py":{base64.b64encode(data).decode()!r}}}; HASHES={{"a.py":{h[:16]!r}}}'.encode()
        records,issues=m.extract_python(src,self.manifest('a.py',data))
        self.assertEqual(len(records),1)
        self.assertEqual(records[0]['payload'],data)
        self.assertEqual(records[0]['identity_scope'],'original_manifest')
        self.assertEqual(issues,[])
    def test_internal_prefix_without_full_manifest_is_rejected(self):
        m=self.tool();data=b'inert';h=hashlib.sha256(data).hexdigest()
        src=f'_S64={{"a.py":{base64.b64encode(data).decode()!r}}}; HASHES={{"a.py":{h[:16]!r}}}'.encode()
        records,issues=m.extract_python(src,{})
        self.assertEqual(records,[]);self.assertTrue(issues)
    def test_wrong_internal_prefix_is_rejected_with_correct_manifest(self):
        m=self.tool();data=b'inert'
        src=f'_S64={{"a.py":{base64.b64encode(data).decode()!r}}}; HASHES={{"a.py":"0000000000000000"}}'.encode()
        records,issues=m.extract_python(src,self.manifest('a.py',data))
        self.assertEqual(records,[]);self.assertTrue(issues)
    def test_dynamic_assignment_is_rejected(self):
        m=self.tool();r,e=m.extract_python(b'_S64={}; HASHES={}; _S64["a.py"]=str(123)',{});self.assertEqual(r,[]);self.assertTrue(e)
    def test_corrupt_base64_is_rejected(self):
        m=self.tool();r,e=m.extract_python(b'_S64={"a.py":"!!!!"}; HASHES={}',self.manifest('a.py',b'1'));self.assertEqual(r,[]);self.assertTrue(e)
    def test_json_reencoding_requires_full_hash(self):
        m=self.tool();obj={'a':[1,2]};data=(json.dumps(obj,indent=2)+'\n').encode();container={'manifest':self.manifest('a.json',data),'json_artifacts':{'a.json':obj}}
        r,e=m.extract_machine(json.dumps(container).encode());self.assertEqual(r[0]['payload'],data)
    def test_semantic_json_match_does_not_equal_byte_match(self):
        m=self.tool();obj={'a':[1,2]};container={'manifest':self.manifest('a.json',b'{"a" : [1 ,2]}'),'json_artifacts':{'a.json':obj}}
        r,e=m.extract_machine(json.dumps(container).encode());self.assertEqual(r,[]);self.assertTrue(e)
    def test_original_crlf_csv_preserved(self):
        m=self.tool();data=b'a,b\r\n1,2\r\n';obj={'manifest':self.manifest('a.csv',data),'csv_artifacts':{'a.csv':data.decode()}};r,e=m.extract_machine(json.dumps(obj).encode());self.assertEqual(r[0]['payload'],data)
    def test_png_base64_name_does_not_alter_bytes(self):
        m=self.tool();data=b'\xff\xd8\xffimage';obj={'manifest':self.manifest('misnamed.png',data),'png_artifacts_base64':{'misnamed.png':base64.b64encode(data).decode()}};r,e=m.extract_machine(json.dumps(obj).encode());self.assertEqual(r[0]['payload'],data)
    def test_pdf_text_marker_is_reported_not_promoted(self):
        m=self.tool();data=b'<!-- BEGIN SOURCE: old.pdf | original-sha256:123456789abc | status:bundle-text-extracted -->\nOCR text\n<!-- END SOURCE: old.pdf -->'
        records,issues=m.extract_marked(data,{})
        self.assertEqual(records,[])
        self.assertEqual(len(issues),1)
        self.assertEqual(issues[0]['reason'],'derived_text_not_original_file_bytes')
    def test_cli_refuses_symlink_input_before_output(self):
        with tempfile.TemporaryDirectory() as folder:
            d=Path(folder);source=d/'source';source.write_text('{}');link=d/'alias';link.symlink_to(source)
            command=[sys.executable,'-B','-S',str(Path(__file__).with_name('unpack_sources.py'))]
            for flag in ('master','ledger','machine','scripts'):command += ['--'+flag,str(link)]
            command += ['--out',str(d/'out')]
            result=subprocess.run(command,capture_output=True,timeout=10)
            self.assertNotEqual(result.returncode,0);self.assertIn(b'invalid or oversized input',result.stderr)
            self.assertFalse((d/'out').exists())
    def test_cli_never_overwrites_existing_output(self):
        with tempfile.TemporaryDirectory() as folder:
            d=Path(folder);out=d/'out';out.mkdir();sentinel=out/'keep';sentinel.write_text('unchanged')
            command=[sys.executable,'-B','-S',str(Path(__file__).with_name('unpack_sources.py'))]
            for flag in ('master','ledger','machine','scripts'):command += ['--'+flag,str(d/'missing')]
            command += ['--out',str(out)]
            result=subprocess.run(command,capture_output=True,timeout=10)
            self.assertNotEqual(result.returncode,0);self.assertIn(b'output must be a new directory',result.stderr)
            self.assertEqual(sentinel.read_text(),'unchanged')
    def test_duplicate_json_keys_rejected(self):
        m=self.tool()
        with self.assertRaises(ValueError):m.extract_machine(b'{"manifest":{},"manifest":{}}')

if __name__=='__main__':unittest.main()
