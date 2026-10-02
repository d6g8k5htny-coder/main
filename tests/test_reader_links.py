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


class LatestPublicWork(unittest.TestCase):
    MATH_CUT = '07320089a9c690c154d2fa70e4ebc12f36f2422c'
    def section(self):
        text=(SITE/'research.html').read_text()
        self.assertIn('id="latest-work"',text)
        return text.split('id="latest-work"',1)[1].split('<section id="lifetimes"',1)[0]

    def test_latest_work_is_reachable_without_javascript(self):
        for name in ('index','workspace'):
            self.assertIn('href="research.html#latest-work"',(SITE/f'{name}.html').read_text())
        section=self.section()
        self.assertIn('Latest public work',section)
        self.assertIn('datetime="2026-10-02T19:50:57Z"',section)
        self.assertIn('curated reading cut',section)
        self.assertIn('not an exhaustive artifact or status catalog',section)
        self.assertIn('does not update automatically',section)

    def test_landed_chain_has_pinned_proofs_and_separate_review_routes(self):
        section=self.section()
        for number,name in ((242,'soft_rejected_pairs'),(243,'soft_fold_limit'),(244,'soft_closed_form')):
            self.assertIn(f'https://github.com/d6g8k5htny-coder/Math-/blob/{self.MATH_CUT}/frontiers/{name}_20261002/PROOF.md',section)
            self.assertIn(f'https://github.com/d6g8k5htny-coder/Math-/pull/{number}',section)
        for limit in ('Conjecture 7 remains open','uncertified','scientific acceptance'):
            self.assertIn(limit,section)

    def test_open_formal_candidate_and_issue_proofs_remain_distinct(self):
        section=self.section()
        for source in ('pull/246','pullrequestreview-5396110365','issuecomment-5957815904','issuecomment-5957885844','issuecomment-5959920397','issuecomment-5959988102'):
            self.assertIn(source,section)
        for limit in ('Unmerged at this cut','I1–I4','persistence-module bookkeeping','Issue-only model notes','finite-r','same-provider'):
            self.assertIn(limit,section)

    def test_historical_library_and_deliberately_mutable_navigation_are_labeled(self):
        section=self.section()
        self.assertIn('Check newer work',section)
        self.assertIn('Mutable upstream navigation',section)
        self.assertIn('2026-09-26',section)
        self.assertIn('2,138',section)
        self.assertIn('/Math-/tree/main/frontiers',section)
        self.assertIn('/meta-framework/blob/main/registry.json',section)
        self.assertIn('/query-',section)
        self.assertNotIn('docs.google.com',section)
        self.assertNotIn('drive.google.com',section)
        self.assertNotIn('dropbox',section.lower())

    def test_latest_reading_sources_allow_only_observed_public_hosts(self):
        section=self.section()
        for link in Page('<section '+section).links:
            parsed=urlsplit(link)
            if parsed.scheme:
                self.assertEqual(parsed.scheme,'https')
                self.assertEqual(parsed.netloc,'github.com')
                self.assertTrue(parsed.path.startswith('/d6g8k5htny-coder/'))

    def test_latest_work_browser_route_has_keyboard_and_no_fetch_checks(self):
        harness=(ROOT/'tools/public_shop_browser_check.py').read_text()
        for token in ('check_latest_work_flow','Latest public work','latest-identities','latest-source-requests','latest-entry'):
            self.assertIn(token,harness)



class PinnedReadingLinks(HTMLParser):
    def __init__(self, text):
        super().__init__();self.links=[];self.feed(text)
    def handle_starttag(self, tag, attrs):
        values=dict(attrs)
        if tag=='a' and values.get('data-source-kind')=='pinned':
            self.links.append(values['href'])

def validate_reading_pins(text):
    expected={
        'Math-':{'07320089a9c690c154d2fa70e4ebc12f36f2422c','2fd72745896402fe50c54ebaeb132ead17f19ff8'},
        'main':{'1e1c9a1cdafd4b2c1a71639516e2233b168d9e05'},
        'meta-framework':{'f063d9dcab51302aaaef6666245cac9cf2307548'},
        'query-':{'aeffebc0ab984ff218b6f07d6f2a999ef4c6ca96'},
    }
    pins=PinnedReadingLinks(text).links
    if len(pins)!=12: raise ValueError('reading source count changed')
    for link in pins:
        u=urlsplit(link);parts=u.path.split('/')
        if u.scheme!='https' or u.netloc!='github.com' or len(parts)<5 or parts[1]!='d6g8k5htny-coder':
            raise ValueError('unapproved public source')
        if parts[2] not in expected or parts[3] not in ('blob','tree') or parts[4] not in expected[parts[2]] or (parts[3]=='blob' and len(parts)<6):
            raise ValueError('reading source commit drift')
    for digest in ('ee2930c1434bb765d11da0690a2abf3ab330d544ce7ede1ae076a4c29e206c75','f972f46d6b7348a4ff5d2eea022694364895a5273265818533dd01204b4c06a0','dd9b436a58d5d58f706ca59ddcf3eb31519866ea9e0a854e168036a8c21e1eaf'):
        if digest not in text: raise ValueError('proof SHA-256 drift')
    return pins

class LatestSourceControls(unittest.TestCase):
    def test_pinned_citations_are_exact_and_hashes_retain_the_checked_identity(self):
        self.assertEqual(len(validate_reading_pins((SITE/'research.html').read_text())),12)
    def test_mutable_private_or_stale_pinned_source_substitutions_fail_closed(self):
        original=(SITE/'research.html').read_text()
        for replacement in ('main','0'*40):
            with self.assertRaisesRegex(ValueError,'commit drift'):
                validate_reading_pins(original.replace(LatestPublicWork.MATH_CUT,replacement))
        with self.assertRaisesRegex(ValueError,'unapproved public source'):
            validate_reading_pins(original.replace('https://github.com/d6g8k5htny-coder/Math-/blob/'+LatestPublicWork.MATH_CUT,'https://drive.google.com/file/d/private-source'))
        with self.assertRaisesRegex(ValueError,'SHA-256 drift'):
            validate_reading_pins(original.replace('ee2930c1434bb765d11da0690a2abf3ab330d544ce7ede1ae076a4c29e206c75','0'*64))

if __name__=='__main__': unittest.main()
