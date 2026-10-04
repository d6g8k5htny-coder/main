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
from hashlib import sha256

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
    def test_remaining_teaching_exports_keep_readable_static_limits_and_disabled_actions(self):
        text=(SITE/'explore.html').read_text()
        class ExportControls(HTMLParser):
            def __init__(self):
                super().__init__();self.buttons={};self.textareas={}
            def handle_starttag(self,tag,attrs):
                a=dict(attrs)
                if tag=='button':self.buttons[a.get('id')]=a
                if tag=='textarea':self.textareas[a.get('id')]=a
        page=ExportControls();page.feed(text)
        for prefix in ('pin','palette'):
            for suffix in ('export-svg','copy-citation'):
                identity=f'{prefix}-{suffix}'
                self.assertIn(identity,page.buttons,'Each teaching figure needs an explicit export control')
                self.assertIn('disabled',page.buttons[identity],'No-JavaScript export must not download a blank diagram')
                self.assertEqual(page.buttons[identity]['type'],'button')
            self.assertIn(f'{prefix}-figure-citation',Page(text).ids)
            self.assertIn(f'{prefix}-export-status',Page(text).ids)
            self.assertIn('readonly',page.textareas[f'{prefix}-citation-text'])
        self.assertIn('selected scales are teaching choices',text)
        self.assertIn('not a valid assignment',text)
        self.assertIn('source verification is not performed by this export',text)

    def test_curvature_exports_start_disabled_and_keep_the_pinned_teaching_scope(self):
        text=(SITE/'explore.html').read_text()
        class Controls(HTMLParser):
            def __init__(self):
                super().__init__();self.buttons={}
            def handle_starttag(self,tag,attrs):
                a=dict(attrs)
                if tag=='button':self.buttons[a.get('id')]=a
        page=Controls();page.feed(text)
        for identity in ('curvature-export-svg','curvature-copy-citation'):
            self.assertIn(identity,page.buttons,'No-JavaScript export control missing')
            self.assertIn('disabled',page.buttons[identity],'Blank SVG or unsupported clipboard must not be enabled by static HTML')
            self.assertEqual(page.buttons[identity]['type'],'button')
        self.assertIn('curvature-figure-citation',Page(text).ids)
        self.assertIn('curvature-export-status',Page(text).ids)
        self.assertIn('9d7b6802424fb4715b31999066aafca8ee2f3cca/coefficients/side24_v1/PROOF.md#1-a-nonperiodic-reference-coefficient-with-exact-cone-moments',text)

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

    def test_research_page_routes_to_bounded_reader_tools(self):
        text=(SITE/'research.html').read_text()
        self.assertIn('href="measure.html"',text)
        self.assertIn('href="dependencies.html"',text)
        self.assertIn('synthetic counts-to-density',text.lower())
        self.assertIn('dated claim-dependency snapshot',text.lower())


class FormalCoverage(unittest.TestCase):
    """The public scope view must agree with the pinned source, not a summary."""
    CUT = 'cc2989c1280f4f227d0c6aa30c8841d6ba01e46e'
    DIGEST = '350732d7a501fd37d15632b13bd2ac30b8a6d87259b9a3b0931639041a1cfb84'

    class Table(HTMLParser):
        def __init__(self, text):
            super().__init__(); self.inside = False; self.cell = None; self.rows = []; self.row = []
            self.feed(text)
        def handle_starttag(self, tag, attrs):
            if tag == 'table': self.inside = dict(attrs).get('id') == 'formal-coverage-table'
            if self.inside and tag == 'tr': self.row = []
            if self.inside and tag in ('td', 'th'): self.cell = []
        def handle_data(self, data):
            if self.cell is not None: self.cell.append(data)
        def handle_endtag(self, tag):
            if self.inside and tag in ('td', 'th') and self.cell is not None:
                self.row.append(' '.join(''.join(self.cell).split())); self.cell = None
            if self.inside and tag == 'tr': self.rows.append(self.row)
            if tag == 'table': self.inside = False

    def test_coverage_and_exclusions_match_all_pinned_source_rows(self):
        source = (ROOT/'tests/fixtures/formal_scope_cc2989.md').read_bytes()
        self.assertEqual(len(source),3832)
        self.assertEqual(sha256(source).hexdigest(),self.DIGEST)
        expected = [[cell.strip() for cell in row.strip('|').split('|')]
                    for row in source.decode().splitlines() if row.startswith('|') and not row.startswith('|---')]
        actual = self.Table((SITE/'formal.html').read_text()).rows
        self.assertEqual(actual,expected)
        self.assertEqual(len(actual)-1,9)
        self.assertEqual(sum(len(row[0].split(', ')) for row in actual[1:]),13)

    def test_static_disclosure_retains_historical_source_and_alignment_boundary(self):
        text = (SITE/'formal.html').read_text()
        page = Page(text)
        self.assertIn('formal-coverage',page.ids)
        self.assertIn('href="#formal-coverage"',text)
        self.assertIn('<summary>Inspect all 13 scalar declarations and their limits</summary>',text)
        self.assertIn('tabindex="0" role="region" aria-label="Exact scalar companion scope table"',text)
        self.assertIn(f'/Math-/blob/{self.CUT}/formal/SCOPE.md',text)
        self.assertIn(self.DIGEST,text)
        self.assertIn('not a current formalization inventory',text)
        self.assertIn('Independent statement-alignment review at this source snapshot: PENDING',text)
        self.assertIn('Scientific effect: NONE',text)
        self.assertNotIn('<script',text)

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
        for name,fragment in (('index','pair-endpoint-rate'),('workspace','latest-work')):
            self.assertIn(f'href="research.html#{fragment}"',(SITE/f'{name}.html').read_text())
        section=self.section()
        self.assertIn('Latest public work',section)
        self.assertIn('datetime="2026-10-03T18:00:00Z"',section)
        self.assertIn('2026-10-02T21:18:12Z',section)
        self.assertIn('curated reading cut',section)
        self.assertIn('not an exhaustive artifact or status catalog',section)
        self.assertIn('does not update automatically',section)

    def test_actual_bar_additions_retain_conditional_population_and_open_consumer(self):
        section=self.section()
        for term in ('Count each actual bar once', 'Two candidates can describe one bar', 'P, CAP, E1, E2, REC, CUB and ELDER', 'ordinary finite', 'essential', 'global elder partner', 'positive-width', 'h^5', 'C_U(1) &lt; C_R', 'not uniform over collapsing bands', 'Math #234', 'integration in progress at this cut'):
            self.assertIn(term,section)
        for review in ('pullrequestreview-5382933608','pullrequestreview-5401559344'):
            self.assertIn(review,section)

    def test_landed_chain_has_pinned_proofs_and_separate_review_routes(self):
        section=self.section()
        for number,name in ((242,'soft_rejected_pairs'),(243,'soft_fold_limit'),(244,'soft_closed_form')):
            self.assertIn(f'https://github.com/d6g8k5htny-coder/Math-/blob/{self.MATH_CUT}/frontiers/{name}_20261002/PROOF.md',section)
            self.assertIn(f'https://github.com/d6g8k5htny-coder/Math-/pull/{number}',section)
        for limit in ('Conjecture 7 remains open','uncertified','scientific acceptance'):
            self.assertIn(limit,section)

    def test_landed_formal_packet_and_issue_proofs_remain_distinct(self):
        section=self.section()
        for source in ('pull/246','pullrequestreview-5396110365','issuecomment-5957815904','issuecomment-5957885844','issuecomment-5959920397','issuecomment-5959988102'):
            self.assertIn(source,section)
        for limit in ('Landed at this cut','I1–I4','persistence-module bookkeeping','Source-bound issue proofs','finite-r','same-provider'):
            self.assertIn(limit,section)

    def test_issue_refresh_preserves_actual_field_and_model_distinctions(self):
        section=self.section()
        for source in ('issuecomment-5960195345','issuecomment-5960343316','issuecomment-5960630207','issuecomment-5960777483'):
            self.assertIn(source,section)
        for boundary in ('C81–C82 concern the soft model','C83–C84 control actual/contact field quantities','same weighted observable','unequal elder marks','EH','Region A/B'):
            self.assertIn(boundary,section)

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
        super().__init__();self.links=[];self.hash_bindings=[];self.items=[];self.in_code=False;self.feed(text)
    def handle_starttag(self, tag, attrs):
        values=dict(attrs)
        if tag=='li': self.items.append({'links':[],'code':[]})
        if tag=='code': self.in_code=True
        if tag=='a' and values.get('data-source-kind')=='pinned':
            self.links.append(values['href'])
            if self.items: self.items[-1]['links'].append(values['href'])
    def handle_data(self, data):
        if self.in_code and self.items: self.items[-1]['code'].append(data)
    def handle_endtag(self, tag):
        if tag=='code': self.in_code=False
        if tag=='li':
            item=self.items.pop()
            if item['code']:
                self.hash_bindings.extend((url,''.join(item['code'])) for url in item['links'])

SOURCE_ROOT='https://github.com/d6g8k5htny-coder/'
MATH_PREFIX=SOURCE_ROOT+'Math-/'
MATH_REF='07320089a9c690c154d2fa70e4ebc12f36f2422c'
MAIN_PREFIX=SOURCE_ROOT+'main/'
MAIN_REF='1e1c9a1cdafd4b2c1a71639516e2233b168d9e05'
PROOF_IDENTITIES={
    'soft_rejected_pairs':'ee2930c1434bb765d11da0690a2abf3ab330d544ce7ede1ae076a4c29e206c75',
    'soft_fold_limit':'f972f46d6b7348a4ff5d2eea022694364895a5273265818533dd01204b4c06a0',
    'soft_closed_form':'dd9b436a58d5d58f706ca59ddcf3eb31519866ea9e0a854e168036a8c21e1eaf',
}
EXPECTED_READING_URLS=[
    *(MATH_PREFIX+'blob/'+MATH_REF+'/frontiers/'+name+'_20261002/PROOF.md' for name in PROOF_IDENTITIES),
    *(MATH_PREFIX+'tree/'+MATH_REF+'/frontiers/'+name+'_20261002' for name in PROOF_IDENTITIES),
    MATH_PREFIX+'blob/0fda855b8ee0c26d597bb033e0b4cdfb6d07e5e6/frontiers/cap_first_exit_lean_20261002/ALIGNMENT.md',
    MAIN_PREFIX+'blob/'+MAIN_REF+'/experiments/periodic_h0/README.md',
    MAIN_PREFIX+'tree/'+MAIN_REF+'/experiments/periodic_h0',
    MAIN_PREFIX+'blob/'+MAIN_REF+'/docs/RESEARCH_INDEX.md',
    SOURCE_ROOT+'meta-framework/blob/f063d9dcab51302aaaef6666245cac9cf2307548/registry.json',
    SOURCE_ROOT+'query-/tree/aeffebc0ab984ff218b6f07d6f2a999ef4c6ca96',
]
ACTUAL_BAR_PINS=[
    MATH_PREFIX+'blob/e9ff8c6165ef0276e1c84d03ee4fb2273bc8278b/frontiers/unique_replacement_bar_intensity_20261001/'+name for name in ('PROOF.md','ROOTS.md','SOURCES.json')
]+[
    MATH_PREFIX+'blob/e05b8303aa7648cf321d16a4c626d522b9dda3db/frontiers/strict_unique_replacement_coefficient_20261003/'+name for name in ('PROOF.md','SOURCE_FILES.json')
]
EXPECTED_READING_URLS += ACTUAL_BAR_PINS
READING_ADDENDUM_PINS = [
    MATH_PREFIX+'blob/c2836c2ae5df1dfdf72549fd9480b83b4621113e/frontiers/replacement_bar_occurrence_20261001/PROOF.md',
    *(MATH_PREFIX+'blob/e0bed3ed394f1e208fc5ff28a694e22f25c751f2/frontiers/planar_soft_layer_chain_20261003/'+name
      for name in ('C99/PROOF.md','C99/REVIEW.md','C103/PROOF.md','C103/REVIEW.md')),
]
EXPECTED_READING_URLS += READING_ADDENDUM_PINS
C124_PREFIX = MATH_PREFIX+'blob/bbe85e270f2c8b747f2d5d9477c86e86e323fe15/frontiers/planar_soft_layer_chain_20261003/'
C124_SOURCE_IDENTITIES = {
    'C124/PROOF.md': '90148657397dcce31e8039afa9015c022f74b2ed0d41e75f2f2339074a8a520f',
    'C124/REVIEW.md': '19c405c68a93c3f5506629f0ebf1ffa78d5c25ad92f17fc4fd04ef0066deb6cf',
    'C124/REVIEW_CLAUDE.md': '1ebe3e0faa30d1e2f81a4a6f292bf953811ca9ea8efa4b12f2a9ac6f4a2d5f2b',
    'C124/SOURCE_IDENTITIES.json': 'a0371bd1985358820250e0d75936d6ec5112f165824df7a9f6e29e25e3f46f6d',
}
C124_PINS = [C124_PREFIX+path for path in C124_SOURCE_IDENTITIES]+[C124_PREFIX+'SOURCES.json']
EXPECTED_READING_URLS += C124_PINS
EXPECTED_HASH_BINDINGS=[(MATH_PREFIX+'tree/'+MATH_REF+'/frontiers/'+name+'_20261002',digest) for name,digest in PROOF_IDENTITIES.items()]
EXPECTED_HASH_BINDINGS += [(C124_PREFIX+path,digest) for path,digest in C124_SOURCE_IDENTITIES.items()]

def validate_reading_pins(text):
    parsed=PinnedReadingLinks(text)
    if len(parsed.links)!=len(EXPECTED_READING_URLS): raise ValueError('reading source count changed')
    for link in parsed.links:
        u=urlsplit(link)
        if u.scheme!='https' or u.netloc!='github.com' or not u.path.startswith('/d6g8k5htny-coder/'):
            raise ValueError('unapproved public source')
    # Full URL includes repository, kind, immutable ref, exact path and no added
    # query/fragment. Sorting retains multiplicity, so duplication cannot omit a pin.
    if sorted(parsed.links)!=sorted(EXPECTED_READING_URLS):
        raise ValueError('reading source identity drift')
    # Displayed proof hashes bind to their own reproduction directory, rather than
    # merely appearing elsewhere on the page or alongside another source.
    if sorted(parsed.hash_bindings)!=sorted(EXPECTED_HASH_BINDINGS):
        raise ValueError('proof SHA-256 association drift')
    return parsed.links

class LatestSourceControls(unittest.TestCase):
    def test_pinned_citations_are_exact_and_hashes_retain_the_checked_identity(self):
        self.assertEqual(len(validate_reading_pins((SITE/'research.html').read_text())),27)

    def test_endpoint_addendum_rejects_wrong_sources_or_swapped_review_hashes(self):
        original=(SITE/'research.html').read_text()
        for link in C124_PINS:
            for replacement in (link.replace('bbe85e270f2c8b747f2d5d9477c86e86e323fe15','main'),
                                link.rsplit('/',1)[0]+'/MISSING.md'):
                with self.subTest(link=link,replacement=replacement),self.assertRaisesRegex(ValueError,'identity drift'):
                    validate_reading_pins(original.replace(link,replacement))
        left=C124_SOURCE_IDENTITIES['C124/REVIEW.md']
        right=C124_SOURCE_IDENTITIES['C124/REVIEW_CLAUDE.md']
        with self.assertRaisesRegex(ValueError,'SHA-256.*drift'):
            validate_reading_pins(original.replace(left,'TEMP_HASH').replace(right,left).replace('TEMP_HASH',right))

    def test_addendum_cannot_replace_a_pinned_proof_with_a_branch_or_different_object(self):
        original=(SITE/'research.html').read_text()
        for link in READING_ADDENDUM_PINS:
            ref=urlsplit(link).path.split('/')[4]
            for replacement in (link.replace(ref,'main'), link.replace(ref,'0'*40),
                                link.rsplit('/',1)[0]+'/MISSING.md'):
                with self.subTest(link=link,replacement=replacement),self.assertRaisesRegex(ValueError,'identity drift'):
                    validate_reading_pins(original.replace(link,replacement))
        # Five valid-looking links cannot hide one missing source by duplication.
        with self.assertRaisesRegex(ValueError,'identity drift'):
            validate_reading_pins(original.replace(READING_ADDENDUM_PINS[0],READING_ADDENDUM_PINS[1]))
    def test_mutable_private_or_stale_pinned_source_substitutions_fail_closed(self):
        original=(SITE/'research.html').read_text()
        for replacement in ('main','0'*40):
            with self.assertRaisesRegex(ValueError,'identity drift'):
                validate_reading_pins(original.replace(LatestPublicWork.MATH_CUT,replacement))
        with self.assertRaisesRegex(ValueError,'unapproved public source'):
            validate_reading_pins(original.replace('https://github.com/d6g8k5htny-coder/Math-/blob/'+LatestPublicWork.MATH_CUT,'https://drive.google.com/file/d/private-source'))
        with self.assertRaisesRegex(ValueError,'SHA-256.*drift'):
            validate_reading_pins(original.replace('ee2930c1434bb765d11da0690a2abf3ab330d544ce7ede1ae076a4c29e206c75','0'*64))



class LatestReviewRegressionControls(unittest.TestCase):
    def test_changed_path_and_misattributed_hash_do_not_pass_exact_pin_check(self):
        original=(SITE/'research.html').read_text()
        wrong=original.replace('frontiers/soft_rejected_pairs_20261002">Soft rejected-pair code','frontiers/does-not-exist">Soft rejected-pair code')
        with self.assertRaises(ValueError): validate_reading_pins(wrong)
        left='ee2930c1434bb765d11da0690a2abf3ab330d544ce7ede1ae076a4c29e206c75'
        right='f972f46d6b7348a4ff5d2eea022694364895a5273265818533dd01204b4c06a0'
        swapped=original.replace(left,'TEMP_HASH').replace(right,left).replace('TEMP_HASH',right)
        with self.assertRaises(ValueError): validate_reading_pins(swapped)

    def test_fragment_landmarks_have_authored_visible_focus(self):
        css=(SITE/'home.css').read_text()
        self.assertIn('.latest-work:focus-visible, .latest-upstream:focus-visible',css)
        self.assertIn('outline: 3px solid var(--ul-focus)',css)
        harness=(ROOT/'tools/public_shop_browser_check.py').read_text()
        self.assertIn('fragment_focus_indicator',harness)

if __name__=='__main__': unittest.main()
