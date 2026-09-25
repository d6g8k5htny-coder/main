"""Synthetic navigation controls. The public readback tests use mocks, not the network."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('nav',Path(__file__).resolve().parents[1]/'tools/navigation_check.py')
n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n)


class NavigationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);(self.root/'docs').mkdir()
        (self.root/'README.md').write_text('# Home\n[Guide](docs/guide.md#result)\n')
        (self.root/'docs/guide.md').write_text('# Guide\n## Result\n[Home](../README.md)\n')
        self.data={'version':1,'pages':['README.md','docs/guide.md'],'public_targets':[]}
        self.save()
    def save(self): (self.root/'docs/NAVIGATION.json').write_text(json.dumps(self.data))
    def test_valid(self): self.assertEqual(n.check(self.root)['problems'],[])
    def test_counts(self):
        r=n.check(self.root);self.assertEqual(r['local_links'],2);self.assertEqual(r['fragment_links'],1)
    def test_missing_page(self):
        (self.root/'docs/guide.md').unlink()
        with self.assertRaises(OSError): n.check(self.root)
    def test_missing_fragment(self):
        (self.root/'docs/guide.md').write_text('# Renamed\n')
        self.assertIn('fragment',n.check(self.root)['problems'][0])
    def test_duplicate_headings(self): self.assertEqual(n.anchors('# A\n## A\n## A'),{'a','a-1','a-2'})
    def test_formatting_in_heading(self): self.assertIn('read-this-now',n.anchors('# Read **this** now!'))
    def test_custom_anchor(self): self.assertIn('exact',n.anchors('<a name="exact"></a>'))
    def test_fenced_examples_ignored(self):
        (self.root/'README.md').write_text('```md\n[Fake](no.md)\n```\n# Home\n')
        self.assertEqual(n.check(self.root)['problems'],[])
    def test_root_escape(self):
        (self.root/'README.md').write_text('[Bad](../../bad.md)')
        self.assertTrue(n.check(self.root)['problems'])
    def test_symlink_rejected(self):
        (self.root/'sym.md').symlink_to(self.root/'docs/guide.md')
        (self.root/'README.md').write_text('[Bad](sym.md)')
        self.assertIn('symlink',n.check(self.root)['problems'][0])
    def test_unsupported_scheme(self):
        (self.root/'README.md').write_text('[Bad](javascript:alert)')
        self.assertTrue(n.check(self.root)['problems'])
    def test_percent_encoded_anchor(self):
        (self.root/'README.md').write_text('[Good](docs/guide.md#%72esult)')
        self.assertEqual(n.check(self.root)['problems'],[])
    def test_duplicate_manifest_key(self):
        (self.root/'docs/NAVIGATION.json').write_text('{"version":1,"version":1}')
        with self.assertRaisesRegex(ValueError,'duplicate'): n.load(self.root)
    def test_duplicate_page(self):
        self.data['pages'].append('README.md');self.save()
        with self.assertRaises(ValueError): n.load(self.root)
    def target(self):
        row={'repository':'Math-','ref':'main','path':'proof.md','bytes':2,'sha256':hashlib.sha256(b'x\n').hexdigest()}
        self.data['public_targets']=[row];self.save()
        (self.root/'README.md').write_text('[Proof](https://github.com/d6g8k5htny-coder/Math-/blob/main/proof.md)')
        return row
    def test_private_target_refused(self):
        row=self.target();row['repository']='sandbox';self.save()
        with self.assertRaisesRegex(ValueError,'nonpublic'): n.load(self.root)
    def test_unlinked_target_refused(self):
        self.target();(self.root/'README.md').write_text('# Home')
        self.assertIn('no direct navigation link',n.check(self.root)['problems'][0])
    def test_public_readback_exact(self):
        self.target()
        with patch.object(n.urllib.request,'urlopen',return_value=io.BytesIO(b'x\n')):
            r=n.check(self.root,True)
        self.assertEqual(r['problems'],[]);self.assertEqual(len(r['public_sources_verified']),1)
    def test_public_changed_bytes(self):
        self.target()
        with patch.object(n.urllib.request,'urlopen',return_value=io.BytesIO(b'y\n')):
            self.assertIn('changed',n.check(self.root,True)['problems'][0])
    def test_no_implicit_network(self):
        self.target()
        with patch.object(n.urllib.request,'urlopen',side_effect=AssertionError('network called')):
            self.assertEqual(n.check(self.root)['public_sources_verified'],[])
    def test_invalid_public_path(self):
        row=self.target();row['path']='../sandbox/secret';self.save()
        with self.assertRaises(ValueError): n.load(self.root)
    def test_invalid_boolean_size(self):
        row=self.target();row['bytes']=True;self.save()
        with self.assertRaises(ValueError): n.load(self.root)
    def test_public_transport_error(self):
        self.target()
        with patch.object(n.urllib.request,'urlopen',side_effect=OSError('offline')):
            self.assertIn('offline',n.check(self.root,True)['problems'])
    def test_no_writes(self):
        before={str(p):p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        n.check(self.root)
        self.assertEqual(before,{str(p):p.read_bytes() for p in self.root.rglob('*') if p.is_file()})

    def test_repository_navigation_pages_resolve(self):
        root=Path(__file__).resolve().parents[1]
        result=n.check(root)
        self.assertEqual(result['problems'],[],result)

    def test_research_map_and_execution_pages_are_declared_and_resolve(self):
        root=Path(__file__).resolve().parents[1]
        data=n.load(root)
        for page in ('docs/RESEARCH_MAP.md','docs/RESEARCH_EXECUTION.md'):
            self.assertIn(page,data['pages'])
            self.assertTrue((root/page).is_file(),page)
        result=n.check(root)
        self.assertEqual(result['problems'],[],result)

    def test_rn_side24_prep_triad_pages_are_declared_and_resolve(self):
        root=Path(__file__).resolve().parents[1]
        data=n.load(root)
        for page in ('docs/RN_SIDE24.md','docs/RN_SIDE24_DENSITY.md','docs/RN_SIDE24_CELL.md'):
            self.assertIn(page,data['pages'])
            self.assertTrue((root/page).is_file(),page)
        result=n.check(root)
        self.assertEqual(result['problems'],[],result)

    def test_downstream_rn_crosswalk_page_is_declared_and_resolve(self):
        root=Path(__file__).resolve().parents[1]
        data=n.load(root)
        page='docs/DOWNSTREAM_RN_CROSSWALK_20260925.md'
        self.assertIn(page,data['pages'])
        self.assertTrue((root/page).is_file(),page)
        # Closed math_status packet must not absorb this navigation note.
        self.assertFalse((root/'docs/math_status/DOWNSTREAM_CROSSWALK_20260925.md').exists())
        self.assertFalse((root/'docs/math_status/DOWNSTREAM_RN_CROSSWALK_20260925.md').exists())
        text=(root/page).read_text()
        # D0 ABSENT carriers must stay named as historical-route blockers.
        for token in ('rnu_env.py','CL_ANTHROPIC_BUNDLE_2026-09-17_v5.zip','allcell_fdz_enclosures.json',
                      'hist.rnu_env.py','BLOCKED_ABSENT','OPEN_HISTORICAL','OPEN_ACTIVE'):
            self.assertIn(token,text,token)
        # Companion executable gate; scientific effect remains none.
        self.assertIn('Math-/pull/8',text)
        self.assertIn('scientific effect NONE',text)
        result=n.check(root)
        self.assertEqual(result['problems'],[],result)

    def test_mesoscopic_challenge_page_is_declared_and_bound(self):
        root=Path(__file__).resolve().parents[1]
        data=n.load(root)
        page='docs/RN_MESOSCOPIC_REDUCTION_CHALLENGE_20260925.md'
        self.assertIn(page,data['pages'])
        text=(root/page).read_text()
        lowered=text.lower()
        self.assertIn('b7ef84cd1e5946c766e125e1ccff6bf611214254d35dc2e0778f019b8d48dcb9',text)
        self.assertIn('r17 review record',lowered)
        self.assertIn('independence credit is **0**',lowered)
        self.assertIn('scientific effect: none',lowered)
        self.assertIn('not a gate movement',lowered)
        self.assertNotIn('lemma_closed=true',lowered)
        result=n.check(root)
        self.assertEqual(result['problems'],[],result)

    def test_navigation_pages_do_not_flip_claim_flags(self):
        root=Path(__file__).resolve().parents[1]
        data=n.load(root)
        banned=(
            'lemma_closed: true','lemma_closed": true','lemma_closed=true',
            'prizes_solved: true','prizes_solved": true','prizes_solved=true',
            'discharges_obl_h5_jetmod: true','discharges_obl_h5_jetmod": true',
            'discharges_obl_h5_jetmod=true',
            'certified_c_h: true','certified_c_h": true','certified_c_h=true',
            'inventable_attempt_accepted: true','inventable_attempt_accepted": true',
            'inventable_attempt_accepted=true',
        )
        for page in data['pages']:
            text=(root/page).read_text().lower()
            for token in banned:
                self.assertNotIn(token,text,page+' contains '+token)


if __name__=='__main__':unittest.main()
