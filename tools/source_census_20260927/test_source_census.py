"""Behavior controls for offline source inspection; fixtures never execute."""
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
import zipfile

class CensusTests(unittest.TestCase):
    def scanner(self, **kwargs):
        path = Path(__file__).with_name('source_census.py')
        self.assertTrue(path.is_file(), 'source census implementation is missing')
        spec = importlib.util.spec_from_file_location('census_under_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.Scanner(**kwargs)

    def zipped(self, members):
        out = io.BytesIO()
        with zipfile.ZipFile(out, 'w') as z:
            for name, value in members:
                z.writestr(name, value)
        return out.getvalue()

    def test_extensionless_utf16_text(self):
        s = self.scanner(); s.inspect_bytes('opaque', 'needle alpha'.encode('utf-16'))
        self.assertEqual(s.records[0]['kind'], 'text')
        self.assertIn('needle', s.texts[s.records[0]['sha256']])

    def test_zip_signature_not_filename(self):
        s = self.scanner(); s.inspect_bytes('misnamed.bin', self.zipped([('proof.odd','TARGET')]))
        self.assertEqual(len(s.records), 2)
        self.assertEqual(s.records[0]['kind'], 'zip')
        self.assertIn('TARGET', s.texts[s.records[1]['sha256']])

    def test_nested_zip_keeps_provenance(self):
        data = self.zipped([('inside.zip', self.zipped([('leaf.txt','x')]))])
        s = self.scanner(); s.inspect_bytes('outer.zip', data)
        self.assertEqual(s.records[-1]['path'], 'outer.zip!/inside.zip!/leaf.txt')

    def test_duplicate_content_parsed_once_not_dropped(self):
        s = self.scanner(); s.inspect_bytes('a.txt',b'alpha'); s.inspect_bytes('b.txt',b'alpha')
        self.assertEqual(len(s.records), 2); self.assertEqual(s.parse_count, 1)
        self.assertTrue(s.records[1]['cache_hit'])

    def test_unsafe_archive_paths_not_opened(self):
        s = self.scanner(); s.inspect_bytes('a.zip', self.zipped([('../evil','x'),('/evil','x'),('C:/evil','x')]))
        self.assertEqual(len(s.records), 1)
        self.assertEqual(len(s.issues), 3)
        self.assertTrue(all(x['reason']=='unsafe_path' for x in s.issues))

    def test_duplicate_member_names_are_not_ambiguous_sources(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore'); data=self.zipped([('a','first'),('a','second')])
        s=self.scanner();s.inspect_bytes('a.zip',data)
        self.assertEqual(len(s.records),1)
        self.assertTrue(any(x['reason']=='duplicate_member_name' for x in s.issues))

    def test_per_member_budget(self):
        s=self.scanner(max_member_bytes=8);s.inspect_bytes('a.zip',self.zipped([('a',b'0123456789')]))
        self.assertEqual(len(s.records),1)
        self.assertEqual(s.issues[0]['reason'],'member_size_limit')

    def test_depth_budget(self):
        s=self.scanner(max_depth=0);s.inspect_bytes('a.zip',self.zipped([('a','x')]))
        self.assertEqual(len(s.records),1)
        self.assertEqual(s.issues[0]['reason'],'depth_limit')

    def test_unknown_binary_is_not_claimed_read(self):
        s=self.scanner();s.inspect_bytes('unknown',b'\x00\xff\x00\x99'*20)
        self.assertEqual(s.records[0]['status'],'unsupported_binary')

    def test_python_is_text_not_executed(self):
        with tempfile.TemporaryDirectory() as d:
            marker=Path(d)/'ran'
            data=f"open({str(marker)!r},'w').write('bad')".encode()
            s=self.scanner();s.inspect_bytes('danger.py',data)
            self.assertFalse(marker.exists());self.assertEqual(s.records[0]['kind'],'text')

    def test_docx_text_and_math_xml(self):
        data=self.zipped([('[Content_Types].xml','<Types/>'),('word/document.xml','<w:document xmlns:w="w" xmlns:m="m"><w:p><w:t>Proof</w:t><m:t>x^2</m:t></w:p></w:document>')])
        s=self.scanner();s.inspect_bytes('doc.unknown',data)
        r=s.records[0];self.assertEqual(r['kind'],'ooxml');self.assertIn('x^2',s.texts[r['sha256']])

    def test_tar_regular_not_symlink(self):
        b=io.BytesIO()
        with tarfile.open(fileobj=b,mode='w') as t:
            i=tarfile.TarInfo('x.txt');i.size=3;t.addfile(i,io.BytesIO(b'abc'))
            i=tarfile.TarInfo('link');i.type=tarfile.SYMTYPE;i.linkname='/tmp/any';t.addfile(i)
        s=self.scanner();s.inspect_bytes('opaque',b.getvalue())
        self.assertEqual(s.records[-1]['path'],'opaque!/x.txt')
        self.assertTrue(any(x['reason']=='nonregular_member' for x in s.issues))

    def test_binary_plist_is_inert_structured_data(self):
        import plistlib
        data=plistlib.dumps({'title':'research notes','numbers':[1,2]},fmt=plistlib.FMT_BINARY)
        s=self.scanner();s.inspect_bytes('opaque.nib',data)
        r=s.records[0];self.assertEqual(r['kind'],'plist')
        self.assertIn('research notes',s.texts[r['sha256']])

    def test_entity_declaration_is_not_expanded(self):
        data=self.zipped([('[Content_Types].xml','<Types/>'),('word/document.xml','<!DOCTYPE x [<!ENTITY e "hello">]><x>&e;</x>')])
        s=self.scanner();s.inspect_bytes('x.docx',data)
        self.assertEqual(s.records[0]['status'],'ooxml_partial')

    def test_ooxml_embedded_file_is_inspected(self):
        data=self.zipped([('[Content_Types].xml','<Types/>'),('word/document.xml','<p>Text</p>'),('word/embeddings/evidence.bin',self.zipped([('proof.txt','EMBEDDED')] ))])
        s=self.scanner();s.inspect_bytes('x.docx',data)
        self.assertTrue(any(r['path'].endswith('!/word/embeddings/evidence.bin!/proof.txt') for r in s.records))

    def test_persistent_cache(self):
        with tempfile.TemporaryDirectory() as d:
            a=self.scanner(cache_dir=d);a.inspect_bytes('x',b'text')
            b=self.scanner(cache_dir=d);b.inspect_bytes('y',b'text')
            self.assertEqual(b.parse_count,0);self.assertTrue(b.records[0]['cache_hit'])

    def test_symlink_root_not_followed(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'real').write_text('abc');(p/'link').symlink_to(p/'real')
            s=self.scanner();s.scan_root('input',p)
            self.assertEqual(len(s.records),1);self.assertTrue(any(x['reason']=='symlink' for x in s.issues))

    def test_utf16_entity_declaration_is_not_expanded(self):
        xml='<!DOCTYPE x [<!ENTITY e "expanded">]><x>&e;</x>'.encode('utf-16')
        data=self.zipped([('[Content_Types].xml','<Types/>'),('word/document.xml',xml)])
        s=self.scanner();s.inspect_bytes('x.docx',data)
        self.assertEqual(s.records[0]['status'],'ooxml_partial')

    def test_duplicate_office_xml_is_blocked(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            data=self.zipped([('[Content_Types].xml','<Types/>'),('word/document.xml','<p>one</p>'),('word/document.xml','<p>two</p>')])
        s=self.scanner();s.inspect_bytes('x.docx',data)
        self.assertEqual(s.records[0]['status'],'ooxml_partial')
        self.assertNotIn('two',s.texts[s.records[0]['sha256']])

    def test_cli_refuses_cache_inside_input(self):
        import subprocess,sys
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'input';root.mkdir();(root/'one').write_text('alpha')
            result=subprocess.run([sys.executable,'-B','-S',str(Path(__file__).with_name('source_census.py')),
                '--root','fixture='+str(root),'--cache',str(root/'cache'),'--out',str(Path(d)/'out')],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('cache must be outside scanned roots',result.stderr)
            self.assertFalse((root/'cache').exists())

if __name__ == '__main__':
    unittest.main()
