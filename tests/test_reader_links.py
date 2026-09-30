"""Actual public HTML route checks, including deep links after the home split."""
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit
import unittest
import importlib.util
import sys
from unittest.mock import patch
from tempfile import TemporaryDirectory
import shutil

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / 'docs' / 'site'

class Page(HTMLParser):
    def __init__(self, text):
        super().__init__(); self.ids = []; self.links = []; self.nav = []; self.primary = False
        self.feed(text)
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if 'id' in a: self.ids.append(a['id'])
        if tag == 'nav' and a.get('aria-label') == 'Primary': self.primary = True
        if tag == 'a' and self.primary: self.nav.append(a.get('href'))
        for key in ('href', 'src'):
            if key in a: self.links.append(a[key])
    def handle_endtag(self, tag):
        if tag == 'nav': self.primary = False

class PublicRoutes(unittest.TestCase):
    def test_all_local_html_links_assets_and_fragments_resolve(self):
        pages={path:Page(path.read_text()) for path in SITE.glob('*.html')}
        # These exact claim IDs are rendered only after museum source verification.
        # The actual rendering and deferred fragment navigation are tested separately.
        dynamic={SITE/'museum.html': [c['id'] for c in json.loads((SITE/'museum.json').read_text())['claims']]}
        self.assertGreaterEqual(len(pages),6)
        for path,page in pages.items():
            with self.subTest(page=path.name):
                self.assertEqual(len(page.ids),len(set(page.ids)), 'duplicate ID')
                for link in page.links:
                    u=urlsplit(link)
                    if u.scheme or u.netloc: continue
                    target=(path.parent/unquote(u.path)).resolve() if u.path else path
                    self.assertTrue(target.is_relative_to(ROOT),link)
                    self.assertTrue(target.is_file(),f'{path.name}: {link}')
                    if u.fragment and target.suffix=='.html':
                        self.assertIn(unquote(u.fragment),pages[target].ids + dynamic.get(target, []),f'{path.name}: {link}')
    def test_consistent_navigation_and_accessible_entry(self):
        expected=['index.html','explore.html','research.html','workspace.html#inventory']
        for name in ('index','explore','research','museum','formal','workspace','reproduce','cite'):
            text=(SITE/f'{name}.html').read_text(); page=Page(text)
            # Workspace's local inventory anchor is the same destination.
            nav=['workspace.html#inventory' if x=='#inventory' else x for x in page.nav]
            self.assertEqual(nav,expected,name)
            self.assertIn('main-content',page.ids)
            self.assertIn('href="reproduce.html"',text)
            self.assertIn('href="cite.html"',text)
            self.assertIn('href="#main-content"',text)
            self.assertNotIn("'unsafe-inline'",text)
            self.assertNotIn("'unsafe-eval'",text)

class ReproductionGuide(unittest.TestCase):
    def test_technical_guide_keeps_exact_pins_and_rejects_substitution(self):
        sys.path.insert(0,str(ROOT/'tools'))
        try:
            spec=importlib.util.spec_from_file_location('museum_reader_check',ROOT/'tools/museum_check.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            module.check_local_identity()
            with TemporaryDirectory() as directory:
                root=Path(directory);site=root/'docs/site';site.mkdir(parents=True)
                for name in ('README.md','config.json','museum.json'): shutil.copyfile(SITE/name,site/name)
                original=(site/'README.md').read_text()
                for repo in ('Math-','query-'):
                    (site/'README.md').write_text(original.replace(f'git -C {repo} checkout --detach ',f'git -C {repo} checkout --detach wrong-'))
                    with patch.object(module,'ROOT',root),self.assertRaisesRegex(ValueError,'checkouts differ'):
                        module.check_local_identity()
        finally: sys.path.pop(0)

if __name__=='__main__': unittest.main()
