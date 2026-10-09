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
import re
import ast

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
    def test_status_currency_pointer_matches_pinned_snapshot(self):
        page=(SITE/'workspace.html').read_text(); status=json.loads((SITE/'status.json').read_text())
        pointer=page.split('id="status-currency"',1)[1].split('</p>',1)[0]
        self.assertIn(f'<code>{status["source"]["commit"][:7]}</code>',pointer)
        self.assertIn('href="https://github.com/d6g8k5htny-coder/main/blob/main/STATUS.md"',pointer)
        self.assertIn('side branch <code>chatgpt/drive-github-hardening-20260919</code>, not into main',pointer)

    def test_status_rendered_from_line_matches_payload_bytes(self):
        page=(SITE/'workspace.html').read_text(); config=json.loads((SITE/'config.json').read_text())
        line=page.split('id="status-rendered-from"',1)[1].split('</p>',1)[0]
        payload=(SITE/'status.json').read_bytes(); digest=sha256(payload).hexdigest()
        self.assertEqual(digest,config['status_json']['sha256']); self.assertEqual(len(payload),config['status_json']['bytes'])
        for value in (digest,f"{config['status_json']['bytes']:,} bytes",config['status']['commit'],config['status']['sha256'],f"{config['status']['bytes']:,} bytes"):
            self.assertIn(value,line)
        # the static proof link names the pinned proof that the script also assigns
        href=page.split('<a id="proof-link" href="',1)[1].split('"',1)[0]
        self.assertEqual(href,f"https://github.com/{config['proof']['repository']}/blob/{config['proof']['commit']}/{config['proof']['path']}")
        for pinned in ('blob/f2e432ea5c86742e480c66624775bc9103343314/STATUS.md','blob/9d7b6802424fb4715b31999066aafca8ee2f3cca/coefficients/side24_v1/ENCLOSURE.json','tree/f2e432ea5c86742e480c66624775bc9103343314/docs/public-math'):
            self.assertIn(pinned,page.split('<noscript>',1)[1] if pinned.startswith('blob/f2e') else page)
        self.assertEqual(page.count('<noscript>'),3)

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
        for name in ('index','explore','research','museum','formal','workspace','reproduce','cite','measure','dependencies'):
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

    # The site's own policy for pages that fetch nothing remote, copied verbatim into docs/404.html.
    NOT_FOUND_CSP=("default-src 'self'; connect-src 'self'; style-src 'self'; script-src 'self'; "
                   "img-src 'self' data:; object-src 'none'; base-uri 'none'; form-action 'none'")

    # Allowlists for the static 404 page: anything else is refused rather than interpreted.
    NOT_FOUND_HEAD_TAGS=frozenset({'meta','title','link'})
    NOT_FOUND_BODY_TAGS=frozenset({'a','span','nav','header','main','section','h1','p','ul','li','strong','footer'})
    NOT_FOUND_ATTRIBUTES={'html':{'lang'},'meta':{'charset','name','content','http-equiv','media'},'link':{'rel','type','href'},
                          'a':{'href','class'},'nav':{'aria-label','class'},'main':{'id','tabindex'},'section':{'class'}}
    NOT_FOUND_CLASSES={'a':{'skip-link','brand'},'section':{'hero'},'nav':{'site-map'}}
    NOT_FOUND_ASSET_SUFFIX={'stylesheet':'.css','icon':'.svg'}
    # Start tags before which a browser closes an open p (an implied end tag), so they never appear inside a p here.
    NOT_FOUND_CLOSES_P=frozenset({'nav','header','main','section','h1','p','ul','li','footer'})
    # The bytes after the doctype must split into these tokens only: comments that end at their first '-->', end tags without
    # attributes, start tags with lowercase names and double-quoted values free of '"', '<', '>' and '&', and text without '<'
    # or '&'. Browsers and every HTMLParser version split such bytes at the same places, and no character reference is decoded.
    # Other bytes can split or decode differently: CPython 3.11 and 3.12 read '<!-->' as opening a comment, and every version
    # reads '<![CDATA[' as hiding everything up to ']]>', while a browser ends both at the first '>' and runs the script after
    # it; and html.unescape turns '&#11;' and other control or noncharacter references into nothing, while a browser keeps the
    # character, which closes head when it comes before the CSP.
    NOT_FOUND_TOKEN=re.compile(r'<!--(?!-?>)(?:(?!--)[^<>])*-->|</[a-z][a-z0-9]*>|<[a-z][a-z0-9]*(?:[\t\n\f\r ]+[a-z][a-z-]*="[^"<>&]*")*>|[^<&]+')
    # Head, pinned element by element with every attribute; only the title text and the description's content (None) are free.
    # It opens with the one encoding declaration, so a browser decodes the bytes as this test does; it links exactly the site's two
    # stylesheets and its icon (without an icon link a browser fetches /favicon.ico; with another type it skips a stylesheet); and
    # it keeps the viewport, so a phone does not zoom every link out of sight.
    NOT_FOUND_HEAD=[('meta',[('charset','utf-8')]),
                    ('meta',[('content','width=device-width, initial-scale=1'),('name','viewport')]),
                    ('meta',[('content',None),('name','description')]),
                    ('meta',[('content',NOT_FOUND_CSP),('http-equiv','Content-Security-Policy')]),
                    ('title',[]),
                    ('link',[('href','/main/site/style.css'),('rel','stylesheet')]),
                    ('link',[('href','/main/site/brand.css'),('rel','stylesheet')]),
                    ('link',[('href','/main/site/brand/mark.svg'),('rel','icon'),('type','image/svg+xml')]),
                    ('meta',[('content','dark light'),('name','color-scheme')]),
                    ('meta',[('content','#f6f8fc'),('media','(prefers-color-scheme: light)'),('name','theme-color')]),
                    ('meta',[('content','#07111f'),('media','(prefers-color-scheme: dark)'),('name','theme-color')])]
    NOT_FOUND_SITE_URL=re.compile(r'/main/site/(?:[a-z0-9-]+/)?[a-z0-9-]+\.(?:html|css|svg)(?:#[a-z0-9-]+)?')
    NOT_FOUND_EXTERNAL_URL=re.compile(r'https://github\.com/[A-Za-z0-9-]+(?:/[A-Za-z0-9._-]+)*')

    def assert_static_not_found_page(self, text):
        """docs/404.html stays a static page under exactly the site's CSP and reaches every site page.

        The page is checked against allowlists, not denylists, in three layers, and anything outside them is refused rather
        than interpreted. Tokens: the bytes must split into the strict tokens of NOT_FOUND_TOKEN, so this parser and a browser
        cut them at the same places. Structure: one document shape only, html, then head with its meta, title and link
        elements, then body; every body element is closed explicitly, innermost first, and never by an implied end tag; then
        </body></html>. So the browser's tree is this parser's tree; any other end tag is refused because in head a browser
        reads </body>, </html> and </br> as closing head, which moves a later CSP meta into body. Elements: each tag carries
        only its own attributes and classes, with no duplicate (browsers keep the first, a dict the last); text appears only in
        the title and the body; head is pinned element by element (NOT_FOUND_HEAD); the CSP is the one http-equiv, in head
        before anything that loads; the skip link opens body with exactly its own words, and the brand link opens the one
        header. URLs keep their element: an href is allowed only on a and link; the link elements are the pinned stylesheets
        and icon; a elements carry a
        fragment, a top-level /main/site/ page or a github.com page, show an ASCII letter or digit, and only plain a elements
        without a class count as destinations, which must be every site page."""
        self.assertTrue(text.startswith('<!doctype html>\n'))
        self.assertEqual([c for c in text if (ord(c)<0x20 and c not in '\t\n') or ord(c)==0x7f],[])  # no CR either: a browser turns it into LF, this parser keeps it
        position=len('<!doctype html>\n')
        while position<len(text):
            token=self.NOT_FOUND_TOKEN.match(text,position)
            self.assertIsNotNone(token,('token',text[position:position+40]))
            position=token.end()
        head_tags,body_tags,closes_p=self.NOT_FOUND_HEAD_TAGS,self.NOT_FOUND_BODY_TAGS,self.NOT_FOUND_CLOSES_P
        allowed,classes=self.NOT_FOUND_ATTRIBUTES,self.NOT_FOUND_CLASSES
        class Static(HTMLParser):
            def __init__(self):
                super().__init__(); self.problems=[]; self.part='start'; self.open=[]; self.in_title=False; self.seen=[]
                self.policies=[]; self.head=[]; self.assets=[]; self.anchors=[]; self.anchor=None; self.landmarks=[]
            def handle_startendtag(self, tag, attrs):  # unreachable after the token check; a browser ignores the slash and keeps the element open
                self.problems.append(('self-closing',tag)); self.handle_starttag(tag,attrs)
            def handle_starttag(self, tag, attrs):
                names=[name for name,_ in attrs]
                self.problems+=[('duplicate attribute',tag,name) for name in sorted(set(names)) if names.count(name)>1]
                self.problems+=[('attribute',tag,name) for name in names if name not in allowed.get(tag,())]
                a={name:(value or '') for name,value in attrs}
                if 'class' in a and a['class'] not in classes.get(tag,()): self.problems.append(('class',tag,a['class']))
                if self.in_title: self.problems.append(('element in title',tag))  # CPython 3.11-3.13 read title as text (RCDATA), as browsers do; this and the end-tag twin guard older parsers
                if tag=='html' and self.part=='start' and not self.seen: pass
                elif tag=='head' and self.part=='start' and self.seen==['html']: self.part='head'
                elif tag=='body' and self.part=='between': self.part='body'
                elif self.part=='head' and tag in head_tags:
                    self.head.append((tag,sorted((name,None if (tag,name,a.get('name'))==('meta','content','description') else value) for name,value in attrs)))
                elif self.part=='body' and tag in body_tags:
                    # Landmarks are children of body, once each. The skip link is body's first element and the brand link is the
                    # header's first: a second .brand's absolutely positioned mark could otherwise paint over a plain link.
                    if tag in ('header','main','footer'):
                        if self.open or tag in self.landmarks: self.problems.append(('landmark',self.open[-1:],tag))
                        self.landmarks.append(tag)
                    if a.get('class')=='skip-link' and (self.seen[-1]!='body' or a.get('href')!='#main-content'): self.problems.append(('skip link',self.seen[-1]))
                    if a.get('class')=='brand' and (self.seen[-1]!='header' or self.open!=['header']): self.problems.append(('brand link',self.seen[-1],self.open))
                    # A browser first closes an open p before a block or heading, an a before an a, an h1 before an h1, and an li
                    # before an li; refused, so that each element keeps the content this parser gives it.
                    if ('p' in self.open and tag in closes_p) or (tag in ('a','h1') and tag in self.open) or (tag=='li' and self.open[-1:]!=['ul']):
                        self.problems.append(('implied end tag',self.open[-1:],tag))
                    self.open.append(tag)
                    # The page nests 6 deep in body; a browser stops nesting at 512 open elements and puts deeper ones beside them.
                    if len(self.open)>16: self.problems.append(('depth',tag))
                else: self.problems.append(('element',self.part,tag))
                if tag=='title': self.in_title=True
                if tag=='link':
                    if a.get('rel') not in ('stylesheet','icon'): self.problems.append(('link rel',a.get('rel')))
                    else: self.assets.append((a['rel'],a.get('href','')))
                if 'a' in self.open[:-1]: self.anchors[-1][3].append(tag)  # an element inside a link
                if tag=='a': self.anchor=[]; self.anchors.append((a.get('href'),a.get('class'),self.anchor,[]))
                if tag=='meta' and 'http-equiv' in a:
                    if a['http-equiv'].lower()!='content-security-policy': self.problems.append(('pragma',a['http-equiv']))
                    else:
                        # Effective only in head and before anything that loads: a meta CSP governs only later fetches.
                        self.policies.append((self.part=='head' and set(self.seen)<={'html','head','meta'},a.get('content','')))
                self.seen.append(tag)
            def handle_endtag(self, tag):
                if self.in_title and tag=='title': self.in_title=False
                elif self.in_title: self.problems.append(('end tag in title',tag))
                elif tag=='head' and self.part=='head': self.part='between'
                elif self.part=='body' and self.open[-1:]==[tag]:
                    self.open.pop()
                    if tag=='a': self.anchor=None
                elif tag=='body' and self.part=='body' and not self.open: self.part='after body'
                elif tag=='html' and self.part=='after body': self.part='end'
                else: self.problems.append(('end tag',self.part,tag))
            def handle_data(self, data):
                if self.anchor is not None: self.anchor.append(data)
                if data.strip('\t\n\f\r ') and not self.in_title and self.part!='body':
                    self.problems.append(('text outside body',self.part,data.strip()[:30]))
        static=Static(); static.feed(text); static.close()
        self.assertEqual(static.problems,[])
        self.assertEqual((static.part,static.open,static.in_title),('end',[],False))  # the document ends with </body></html>, every element closed
        self.assertEqual(static.policies,[(True,self.NOT_FOUND_CSP)])  # exactly one effective policy: never moved, removed, emptied, weakened or doubled
        self.assertEqual(static.head,self.NOT_FOUND_HEAD)
        page=Page(text); pages={path.name:Page(path.read_text()) for path in SITE.glob('*.html')}
        for rel,link in static.assets:  # stylesheets and the icon: existing /main/site/ files of the matching type, never pages
            self.assertIsNotNone(self.NOT_FOUND_SITE_URL.fullmatch(link),link)  # already pinned by NOT_FOUND_HEAD; kept so a renamed asset fails here too
            target=SITE/link[len('/main/site/'):]
            self.assertEqual(('#' in link,target.suffix),(False,self.NOT_FOUND_ASSET_SUFFIX[rel]),link)
            self.assertTrue(target.is_file(),link)
        reached=set()
        for link,css_class,words,children in static.anchors:  # only a elements navigate; an href on any other element was refused above
            self.assertIsNotNone(link)
            if css_class=='skip-link':  # positioned above the page with an opaque background: any larger content would cover the links
                self.assertEqual((''.join(words),children),('Skip to content',[]))
            self.assertRegex(''.join(words),r'[A-Za-z0-9]',link)  # every link shows an ASCII letter or digit: not only spaces or a filler such as U+3164
            if link.startswith('#'):
                self.assertRegex(link,r'\A#[a-z0-9-]+\Z'); self.assertIn(link[1:],page.ids,link); continue
            if self.NOT_FOUND_EXTERNAL_URL.fullmatch(link):
                self.assertEqual(urlsplit(link).netloc,'github.com',link); continue
            self.assertIsNotNone(self.NOT_FOUND_SITE_URL.fullmatch(link),link)
            u=urlsplit(link); target=SITE/u.path[len('/main/site/'):]
            self.assertEqual((target.parent,target.suffix),(SITE,'.html'),link)  # a destination is a site page, never a stylesheet, an image or a file in a subfolder
            self.assertTrue(target.is_file(),link)
            if u.fragment: self.assertIn(u.fragment,pages[target.name].ids,link)
            if css_class is None: reached.add(target.name)  # the skip link sits offscreen until focused, and the brand repeats Home
        self.assertEqual(reached,set(pages))  # every page of the site is a destination of a plain, visible link

    def test_custom_not_found_page_is_static_and_routes_into_the_site(self):
        # GitHub Pages serves docs/404.html for any missing path under /main/; it must stay script-free, keep the CSP and reach every real page.
        self.assertIn(f'content="{self.NOT_FOUND_CSP}"',(SITE/'index.html').read_text())  # the 404 copies the site's own policy
        self.assert_static_not_found_page((ROOT/'docs/404.html').read_text())

    def test_custom_not_found_guard_rejects_weakened_copies(self):
        # Negative controls: each copy breaks one promise of the 404 page, or reads differently in a browser than in this parser, and must
        # be refused by the rule its name states. A control is one (old, new) replacement or a list of them, each applied to exactly one place.
        text=(ROOT/'docs/404.html').read_text()
        meta=f'<meta http-equiv="Content-Security-Policy" content="{self.NOT_FOUND_CSP}">'
        measure='<a href="/main/site/measure.html">From counts to density</a>'
        site_map='<nav class="site-map" aria-label="All pages">'
        icon='<link rel="icon" type="image/svg+xml" href="/main/site/brand/mark.svg">'
        mutants={
            'CSP content emptied': (meta,'<meta http-equiv="Content-Security-Policy" content="">'),
            'CSP weakened': ("script-src 'self';","script-src 'self' 'unsafe-inline';"),
            'CSP removed': (meta,''),
            'CSP doubled': (meta,meta+'<meta http-equiv="content-security-policy" content="default-src *">'),
            'uppercase script element': ('</main>','<SCRIPT SRC="/main/site/app.js"></SCRIPT></main>'),
            'lowercase script element': ('</main>','<script src="/main/site/app.js"></script></main>'),
            'mixed-case inline script': ('</main>','<ScRiPt>document.title="x"</ScRiPt></main>'),
            'inline event handler': ('<body>','<body onload="document.title=1">'),
            'javascript: URL': ('</main>','<a href="javascript:void(0)">x</a></main>'),
            'tab inside the scheme': ('</main>','<a href="java\tscript:void(0)">x</a></main>'),
            'newline inside the scheme': ('</main>','<a href="java\nscript:void(0)">x</a></main>'),
            'character reference in an attribute value': ('</main>','<a href="java&#9;script:void(0)">x</a></main>'),
            'data: URL': ('</main>','<a href="data:text/html,x">x</a></main>'),
            'CSP content duplicated, empty first': (meta,f'<meta http-equiv="Content-Security-Policy" content="" content="{self.NOT_FOUND_CSP}">'),
            'CSP http-equiv duplicated, refresh first': (meta,f'<meta http-equiv="refresh" http-equiv="Content-Security-Policy" content="{self.NOT_FOUND_CSP}">'),
            'CSP moved out of head': [(meta,''),('<main id="main-content" tabindex="-1">','<main id="main-content" tabindex="-1">'+meta)],
            'duplicate href': ('<a href="/main/site/cite.html">','<a href="/main/site/cite.html" href="/main/site/no-such-page.html">'),
            'base element': ('</head>','<base href="https://example.com/"></head>'),
            'refresh pragma': ('</head>','<meta http-equiv="refresh" content="0; url=https://example.com/"></head>'),
            'iframe': ('</main>','<iframe srcdoc="x"></iframe></main>'),
            # Placement and head: a meta CSP governs only later fetches, text or a non-head element in head closes head early, and the
            # rest of head is pinned.
            'non-breaking space in head before the CSP': (meta,' '+meta),
            'control-character reference in head before the CSP': (meta,'&#11;'+meta),
            'doctype replaced by a character reference': ('<!doctype html>\n','&#11;<!--    -->'),
            'CSP after the stylesheets and the icon': [(meta,''),(icon,icon+meta)],
            'CSP wrapped in noscript': (meta,'<noscript>'+meta+'</noscript>'),
            'CSP wrapped in template': (meta,'<template>'+meta+'</template>'),
            'CSP inside the title': [(meta,''),('<title>','<title>'+meta)],
            'external stylesheet before the CSP': (meta,'<link rel="stylesheet" href="https://github.com/x.css">'+meta),
            'prefetch link': ('</head>','<link rel="prefetch" href="/main/site/index.html"></head>'),
            'replacement-encoding charset': ('<meta charset="utf-8">','<meta charset="iso-2022-kr">'),
            'icon link removed': (icon,''),
            'stylesheet with a type a browser does not load': ('<link rel="stylesheet" href="/main/site/style.css">','<link rel="stylesheet" type="text/plain" href="/main/site/style.css">'),
            'viewport zoomed out on phones': ('content="width=device-width, initial-scale=1"','content="width=4000, initial-scale=0.1, minimum-scale=0.1, maximum-scale=0.1, user-scalable=no"'),
            # URL shapes a browser resolves somewhere other than the file the test would check.
            'dot segments': ('/main/site/measure.html','/main/site/../../docs/site/measure.html'),
            'percent-encoded dot segments': ('/main/site/measure.html','/main/site/%2e%2e/site/measure.html'),
            'backslash': ('/main/site/measure.html','/main/site/brand\\..\\measure.html'),
            'trailing slash': ('/main/site/measure.html','/main/site/measure.html/'),
            'protocol-relative host': ('<a class="brand" href="/main/site/index.html">','<a class="brand" href="//example.com/main/site/index.html">'),
            'userinfo before the host': ('href="https://github.com/d6g8k5htny-coder/main">d6g8k5htny-coder','href="https://github.com@example.com/d6g8k5htny-coder/main">d6g8k5htny-coder'),
            'https without authority': ('<a class="brand" href="/main/site/index.html">','<a class="brand" href="https:/main/site/no-such-page.html">'),
            'uppercase scheme': ('</main>','<a href="JAVASCRIPT:void(0)">x</a></main>'),
            'query before a fragment': ('href="/main/site/workspace.html#board"','href="/main/site/workspace.html?#board"'),
            'link to a stylesheet': ('<a href="/main/site/cite.html">Cite</a>','<a href="/main/site/cite.html">Cite</a> · <a href="/main/site/brand.css">Style</a>'),
            'link without an href': ('<a href="/main/site/cite.html">Cite</a>','<a href="/main/site/cite.html">Cite</a> · <a>Cite</a>'),
            'one page dropped': (' · '+measure,''),
            'missing page': ('/main/site/measure.html','/main/site/no-such-page.html'),
            'missing fragment': ('#coefficient"','#no-such-fragment"'),
            'release key added': ('/main/site/brand.css"','/main/site/brand.css?site-release=0"'),
            # Content a browser hides, disables or treats as links without an href attribute.
            'SVG link with xlink:href': ('</main>','<svg><a xlink:href="data:text/html,x"><text>x</text></a></svg></main>'),
            'SVG link with href': ('</main>','<svg><a href="data:text/html,x"><text>x</text></a></svg></main>'),
            'site map in a template': (site_map,'<template>'+site_map),
            'inert site map': (site_map,'<nav class="site-map" aria-label="All pages" inert="">'),
            'site map given the button class': (site_map,'<nav class="button" aria-label="All pages">'),
            'download instead of navigation': ('<a href="/main/site/measure.html">','<a href="/main/site/measure.html" download="">'),
            'attribute without a value': ('<a href="/main/site/measure.html">','<a href="/main/site/measure.html" download>'),
            'ping attribute': ('<a href="/main/site/cite.html">','<a href="/main/site/cite.html" ping="https://github.com/">'),
            # Structure (C188-310-01): in head a browser reads these end tags as closing head, and the CSP after them lands in body.
            'end tag of body before the CSP': (meta,'</body>'+meta),
            'end tag of html before the CSP': (meta,'</html>'+meta),
            'end tag of br before the CSP': (meta,'</br>'+meta),
            'page not closed': ('</body></html>',''),
            # Structure: start tags a browser answers by first closing an open element, and nesting past the browser's depth limit.
            'nested link empties the Measure link': (measure,'<a href="/main/site/measure.html"><a href="/main/site/explore.html">From counts to density</a></a>'),
            'site map inside an open paragraph': [('</p>\n'+site_map,'\n'+site_map),('</ul></nav>\n</section>','</ul></nav></p>\n</section>')],
            'heading inside a heading': ('<h1>Page not found</h1>','<h1>Page <h1>not</h1> found</h1>'),
            'list item inside a list item': ('<li><strong>Research</strong>','<li>x<li>y</li></li><li><strong>Research</strong>'),
            'Measure link nested past the browser depth limit': (measure,'<span>'*505+'<a href="/main/site/measure.html"><span>From counts to density</span></a>'+'</span>'*505),
            'control character in the heading': ('<h1>Page not found</h1>','<h1>Page\x0bnot found</h1>'),
            # Tokens this parser and a browser split differently: the browser ends each at the first '>' and runs the script.
            'comment opened and closed by <!-->': ('</main>','<!--><script src="/main/site/cite.mjs"></script><!-- --></main>'),
            'comment opened and closed by <!--->': ('</main>','<!---><script src="/main/site/cite.mjs"></script><!-- --></main>'),
            'CDATA section around a script': ('</main>','<![CDATA[ > <script src="/main/site/cite.mjs"></script> ]]></main>'),
            # Element identity (C188-310-02): only a visible, plain a element is a destination.
            'Measure link as a span': (measure,'<span href="/main/site/measure.html">From counts to density</span>'),
            'icon link pointed at the Measure page': [(' · '+measure,''),('href="/main/site/brand/mark.svg"','href="/main/site/measure.html"')],
            'Measure reached only through the brand link': [(' · '+measure,''),('<a class="brand" href="/main/site/index.html">','<a class="brand" href="/main/site/measure.html">')],
            'empty Measure link': (measure,'<a href="/main/site/measure.html"></a>From counts to density'),
            'Measure link out of the tab order': ('<a href="/main/site/measure.html">','<a href="/main/site/measure.html" tabindex="-1">'),
            'Measure link text only a non-breaking space': ('>From counts to density</a>','> </a>From counts to density'),
            'Measure link text only a Hangul filler': ('>From counts to density</a>','>ㅤ</a>From counts to density'),
            # Classed links have fixed places and content: the skip link sits above the page with an opaque background, and the brand
            # link's mark is positioned over its own padding.
            'skip-link class on the Measure link, offscreen until focused': ('<a href="/main/site/measure.html">','<a class="skip-link" href="/main/site/measure.html">'),
            'second skip link before the site map': (site_map,'<a class="skip-link" href="#main-content">Skip</a>'+site_map),
            'skip link text grown over the page': ('href="#main-content">Skip to content</a>','href="#main-content">'+'Skip to content '*250+'</a>'),
            'skip link with block content over the page': ('href="#main-content">Skip to content</a>','href="#main-content"><h1>Skip to content</h1>'+'<section></section>'*6+'</a>'),
            'second brand mark over the Measure link': [(' · '+measure,''),(site_map,'<ul><li><a class="brand" href="/main/site/index.html">U</a></li>'
                                                        '<li>   <a href="/main/site/measure.html">M</a></li></ul>'+site_map)],
            'second header around the site map': [(site_map,'<header>'+site_map),('</ul></nav>\n</section>','</ul></nav></header>\n</section>')],
            # Over-strict by design: a browser still keeps the CSP in head here, but the guard accepts one document structure only.
            'early end tag of head before the CSP': (meta,'</head>'+meta),
            'ignored end tag of p in head before the CSP': (meta,'</p>'+meta),
        }
        for name,edits in mutants.items():
            with self.subTest(name):
                copy=text
                for old,new in (edits if isinstance(edits,list) else [edits]):
                    self.assertEqual(copy.count(old),1,(name,old)); copy=copy.replace(old,new)
                with self.assertRaises(AssertionError): self.assert_static_not_found_page(copy)
        # Positive control: a comment before the CSP leaves head open, and a browser enforces the policy.
        self.assert_static_not_found_page(text.replace(meta,'<!-- policy -->'+meta))

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
        for name,fragment in (('index','reading-cut-20261009'),('workspace','reading-cut-20261009'),('formal','cap-on-torus')):
            page=(SITE/f'{name}.html').read_text()
            self.assertIn(f'href="research.html#{fragment}"',page)
            if name in ('index','workspace'):
                # the dated sibling beside the entry link must name the newest cut's timestamp
                self.assertIn('datetime="2026-10-09T00:05:00Z"',page)
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
SHRINKING_BIN_PINS = {
    MAIN_PREFIX+'blob/5a7a41daac2e04ea513b97d5fae10a7334f422da/experiments/periodic_h0/EXACT_SAMPLE_SHRINKING_BIN.md':
        '3d938d2223d99f7a8dbaa005b9eb3769813a6f8099a7780200ef048049637c82',
    MAIN_PREFIX+'blob/ad7ea9156138911f44b91bcc0e9497a4ba53f181/experiments/periodic_h0/spectral_shrinking_bins/PROOF.md':
        'fa6d443fd5eeeb21c0d311de4d11eb5b406758f3c5ff09a222d02ba54bd0752c',
    MAIN_PREFIX+'blob/ad7ea9156138911f44b91bcc0e9497a4ba53f181/experiments/periodic_h0/spectral_shrinking_bins/SOURCES.json':
        '783319599514159bd8e4ca6a4dd02021063d6c547e2c80cb3a8ae1ec1212ccb3',
}
EXPECTED_READING_URLS += list(SHRINKING_BIN_PINS)
LIFETIME_PREFIX = MATH_PREFIX+'blob/7d2f62500ad6effba2bbdf6f3b89b0826408dc4a/frontiers/lifetime_two_thirds_chain_20261007/'
CAP_PREFIX = MATH_PREFIX+'blob/54ebcedee137e40d22dbc14a25ba51623c073a25/frontiers/cap_first_exit_lean_20261002/'
OCT7_PINS = {
    LIFETIME_PREFIX+'README.md': '0377126475c0bfc77782b1c747581698f93f5223fb7c6a9f3f7fbefd993e70ca',
    LIFETIME_PREFIX+'C3/PROOF.md': '0764d004f2f36d2fda79f7bea3b161fb1876c918b6a2dfc72110bcb89f623968',
    LIFETIME_PREFIX+'LU/PROOF.md': '373710a50e981e41472e51f49be4d50bcf5da70c01b518f833cb49b1e0680dac',
    LIFETIME_PREFIX+'LU/AUTHOR_SUCCESSOR_TEXT_AND_READBACK_SUMMONS.md': '5c9d403e0b38bc6acc738760ffdbd8cba3c0722641c90be0a69088a39fc03a73',
    LIFETIME_PREFIX+'SOURCES.json': '31f97e51d60402ab4303de974947df8c59917f531f762133ad0fe0f67408afe9',
    CAP_PREFIX+'ALIGNMENT.md': '91b086c04d083d2220821c8b6900e2bd070ad11f4be03bf2255bdbd83d677f21',
    CAP_PREFIX+'README.md': '826c0d67f25119f242575f65c29e70c2621266980b96527a428bbba6aa04a2ab',
    MAIN_PREFIX+'blob/7c4cef8c6a2ca4b6f984e5e45108037c2caec4e1/governance/OP-CLOSURE-EVIDENCE-20261006.md': '3fd6c95d1585c1226038237db3e81e68daed943b1a10d7d5737231218b0603bc',
    MATH_PREFIX+'blob/0793dc26bc3979e3381c6b59d378876080c48563/tools/evidence_profile.py': '941ab165fe5e8f3f4b8b8e326afde279ee8a1311dc51a9c0eac065d9de644a02',
    MATH_PREFIX+'blob/34618d0d032f4361c1ec163f3b4cfd6ad01ab814/reviews/retrofit_20261006/CONTRACT.md': '7a1d03fcc1f87342e848126c02c674f78e4342bd827e974203195078ee248cba',
}
EXPECTED_READING_URLS += list(OCT7_PINS)
# 9 October cut: every link is pinned at main 0be54aa2 (main #329), where the cut was checked.
# The program, PROOF.md and REVIEW.md have the same bytes there as at 2c84170712c5 (main #326,
# where the proof landed); the suite guide changed in #329 and is pinned at its 0be54aa2 bytes.
OCTNEW_REF = '0be54aa227d7015b0abda00feb5e847a3fbdbcdb'
OCTNEW_PREFIX = MAIN_PREFIX+'blob/'+OCTNEW_REF+'/'
OCTNEW_FACE_PINS = [OCTNEW_PREFIX+'experiments/universality/finite_h0_lifetime/PROOF.md']
OCTNEW_PINS = {
    OCTNEW_PREFIX+'docs/PERSISTENCE_UNIVERSALITY_PROGRAM.md': '18fb926d3474e9c02a7b5d957278514558fabedd45685f5d2027996744a08ef9',
    OCTNEW_PREFIX+'experiments/universality/README.md': 'd592cba372a731266ebe030f9b7daa526d5cdde8fab4420870b82d7941416e5d',
    OCTNEW_PREFIX+'experiments/universality/finite_h0_lifetime/PROOF.md': '5ea91111fa5f3d78d7c4188ebe84b3bc2c452b05dfdfc62c74b03123cb2696d1',
    OCTNEW_PREFIX+'experiments/universality/finite_h0_lifetime/REVIEW.md': '81f0067dd3bd768c3db2c3c6e1a35d60b07a2ffe98fa6cafe6e48b5dc6021c75',
}
EXPECTED_READING_URLS += OCTNEW_FACE_PINS + list(OCTNEW_PINS)
EXPECTED_HASH_BINDINGS=[(MATH_PREFIX+'tree/'+MATH_REF+'/frontiers/'+name+'_20261002',digest) for name,digest in PROOF_IDENTITIES.items()]
EXPECTED_HASH_BINDINGS += [(C124_PREFIX+path,digest) for path,digest in C124_SOURCE_IDENTITIES.items()]
EXPECTED_HASH_BINDINGS += list(SHRINKING_BIN_PINS.items())
EXPECTED_HASH_BINDINGS += list(OCT7_PINS.items())
EXPECTED_HASH_BINDINGS += list(OCTNEW_PINS.items())

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
        self.assertEqual(len(validate_reading_pins((SITE/'research.html').read_text())),45)

    def test_sampling_sources_reject_mutable_paths_duplicates_and_unbound_hashes(self):
        original=(SITE/'research.html').read_text()
        for link,digest in SHRINKING_BIN_PINS.items():
            ref=urlsplit(link).path.split('/')[4]
            for replacement in (link.replace(ref,'main'),link.rsplit('/',1)[0]+'/MISSING.md'):
                with self.subTest(link=link,replacement=replacement),self.assertRaisesRegex(ValueError,'identity drift'):
                    validate_reading_pins(original.replace(link,replacement))
            with self.subTest(digest=digest),self.assertRaisesRegex(ValueError,'SHA-256.*drift'):
                validate_reading_pins(original.replace(digest,'0'*64))
        left,right=list(SHRINKING_BIN_PINS)[:2]
        with self.assertRaisesRegex(ValueError,'identity drift'):
            validate_reading_pins(original.replace(left,right))

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


class ShrinkingBinReading(unittest.TestCase):
    def section(self):
        text=(SITE/'research.html').read_text()
        self.assertIn('id="shrinking-bin-sampling"',text)
        return text.split('<section id="shrinking-bin-sampling"',1)[1].split('<section id="pair-endpoint-rate"',1)[0]

    def test_static_entry_and_disclosure_keep_exact_sample_and_spectral_scopes_separate(self):
        text=(SITE/'research.html').read_text();section=self.section()
        self.assertIn('href="#reading-cut-20261009">Latest public work',text)
        self.assertIn('href="#shrinking-bin-sampling">06:20 UTC grid-sampling reading cut below',text)
        self.assertIn('tabindex="-1" aria-labelledby="sampling-heading"',section)
        self.assertIn('datetime="2026-10-04T06:20:32Z"',section)
        self.assertIn('<details class="latest-identities" id="sampling-details">',section)
        self.assertNotIn('<script',text)
        for term in ('C131 · EXACT VERTEX SAMPLES','C132 · CONDITIONAL SPECTRAL EXTENSION',
                     'fixed side-24 planar torus','unconditioned normalized Gaussian field',
                     'finite ordinary superlevel H0 bars','multiplicity','essential class is excluded',
                     'no longest finite bar is discarded','[λτ, μτ)','0 &lt; λ &lt; μ',
                     'deterministic','h²√log(1/h) = o(τ_h)',
                     'expected absolute count discrepancy','expectation ratio tends to one',
                     'M4–M5','same Gaussian coordinates','matching seed labels',
                     'd_n = o(τ_n)','n²b_n + √b_n = o(τ_n^(2/3))',
                     'small failure probability alone is insufficient',
                     'unconditioned total-critical-count second moment','C131 does not consume C6',
                     'retains that source’s author-side disposition','determinant-tilted Palm window count'):
            with self.subTest(term=term):self.assertIn(term,section)

    def test_limits_and_full_sources_remain_available_without_status_promotion(self):
        section=self.section()
        for term in ('not a certificate for the current numerical sampler','FFT','roundoff',
                     'persistence-library','finite-grid','confidence intervals','post hoc',
                     'zero-endpoint cumulative bins','evaluated constants','Lean',
                     'global theorem','scientific status','PASS_TECHNICAL_SCOPED',
                     'Organizational-independence credit is zero',
                     'does not fetch, execute or verify',
                     'issuecomment-5976299711','issuecomment-5976527971',
                     'pullrequestreview-5404446570'):
            with self.subTest(term=term):self.assertIn(term,section)
        parsed=PinnedReadingLinks('<section '+section)
        self.assertEqual(parsed.links,list(SHRINKING_BIN_PINS))
        self.assertEqual(parsed.hash_bindings,list(SHRINKING_BIN_PINS.items()))

    def test_preceding_reading_cuts_remain_byte_identical(self):
        text=(SITE/'research.html').read_text()
        # The 7 October cut stopped being newest on 9 October; it is pinned whole, its cut-pointer included.
        october=text[text.index('<section id="reading-cut-20261007"'):text.index('<section id="shrinking-bin-sampling"')]
        self.assertEqual(sha256(october.encode()).hexdigest(),
                         'ba7094b93c93598321958dc72ea5967c360b5e03a2d273f0b23f4db684668eb8')
        sampling=text[text.index('<section id="shrinking-bin-sampling"'):text.index('<section id="pair-endpoint-rate"')]
        self.assertEqual(sha256(sampling.encode()).hexdigest(),
                         '6787bf96f0bbb12c5d465d05e4829bf3dbc628629cedd0d3dcb82fdd37ea47c0')
        cuts=text[text.index('<section id="pair-endpoint-rate"'):text.index('<section id="lifetimes"')]
        # Only labeled lineage notes (two U1 reviewer notes, two source-identity notes) are additive inside the dated cuts; every original byte stays pinned.
        notes=LINEAGE_NOTE.findall(cuts)
        self.assertEqual(len(notes),4)
        self.assertEqual(sha256(LINEAGE_NOTE.sub('',cuts).encode()).hexdigest(),
                         'ab38d6b603d5ae22009cc13e1975aad61eabb36d29c63511091547afd8b67af2')
        # The reading paths carry no lineage notes and are pinned whole; only the Further-reading landmark and the footer (navigation chrome) are outside both pins.
        paths=text[text.index('<section id="lifetimes"'):text.index('<section id="further-reading"')]
        self.assertEqual(LINEAGE_NOTE.findall(paths),[])
        self.assertEqual(sha256(paths.encode()).hexdigest(),
                         '13bbdcb553f13316c3cf0ef9bea7cfb1a021668463319322b1850df0977c9db7')

    def test_lineage_notes_sit_beside_their_reviews_without_new_sources(self):
        text=(SITE/'research.html').read_text()
        for card,review in (('actual-bars','pullrequestreview-5382933608'),('strict-bar-coefficient','pullrequestreview-5401559344')):
            body=text.split(f'id="{card}"',1)[1].split('</article>',1)[0]
            notes=LINEAGE_NOTE.findall(body)
            self.assertEqual(len(notes),1,card)
            note=notes[0]
            self.assertLess(body.index(review),body.index(note))
            for term in ('added after this cut','OpenAI/Codex','same provider','zero organizational-independence credit','does not mean independent'):
                self.assertIn(term,note)
            self.assertNotIn('<a ',note)

    def test_existing_browser_cases_cover_new_route_and_javascript_off_access(self):
        harness=(ROOT/'tools/public_shop_browser_check.py').read_text()
        for token in ('#shrinking-bin-sampling','#sampling-details','sampling-sources',
                      '00:18 UTC designated-pair reading cut below',
                      'C131/C132 sampling scope','java_script_enabled=False'):
            self.assertIn(token,harness)

    def test_historical_scope_assertion_resolves_one_actual_boundary(self):
        class Boundaries(HTMLParser):
            def __init__(self,text):
                super().__init__();self.sections=[];self.current=None;self.rows=[];self.feed(text)
            def handle_starttag(self,tag,attrs):
                values=dict(attrs)
                if tag=='section':self.sections.append(values.get('id'))
                if tag=='p' and 'latest-boundary' in values.get('class','').split():
                    self.current=[self.sections[-1],[]]
            def handle_data(self,data):
                if self.current:self.current[1].append(data)
            def handle_endtag(self,tag):
                if tag=='p' and self.current:
                    self.rows.append((self.current[0],''.join(self.current[1])));self.current=None
                if tag=='section':self.sections.pop()
        rows=Boundaries((SITE/'research.html').read_text()).rows
        self.assertEqual(len(rows),4,'The 9 October scope, the 7 October scope, the sampling scope and the original boundary must remain visible')
        class ActualBoundaryPage:
            def locator(self,selector):
                region=selector.split(' ',1)[0][1:] if selector.startswith('#') else None
                return [text for identity,text in rows if region is None or identity==region]
        class StrictSingleExpectation:
            def __init__(self,matches):self.matches=matches
            def to_contain_text(self,expected):
                if len(self.matches)!=1:raise AssertionError(f'Historical boundary assertion matches {len(self.matches)} elements')
                if expected not in self.matches[0]:raise AssertionError('Wrong scientific scope was selected')
        module=ast.parse((ROOT/'tools/public_shop_browser_check.py').read_text())
        flow=next(node for node in module.body if isinstance(node,ast.FunctionDef) and node.name=='check_latest_work_flow')
        statement=next(node for node in ast.walk(flow) if isinstance(node,ast.Expr)
                       and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute)
                       and node.value.func.attr=='to_contain_text' and node.value.args
                       and isinstance(node.value.args[0],ast.Constant)
                       and node.value.args[0].value=='Conjecture 7 remains open')
        exec(compile(ast.Module(body=[statement],type_ignores=[]),'<historical-boundary>','exec'),
             {'page':ActualBoundaryPage(),'expect':StrictSingleExpectation})

LINEAGE_NOTE = re.compile(r'<p class="lineage-note">.*?</p>', re.S)


class October7Reading(unittest.TestCase):
    def section(self):
        text=(SITE/'research.html').read_text()
        self.assertIn('id="reading-cut-20261007"',text)
        return text.split('<section id="reading-cut-20261007"',1)[1].split('<section id="shrinking-bin-sampling"',1)[0]

    def test_new_cut_is_dated_first_and_links_back_and_forward(self):
        section=self.section()
        self.assertIn('tabindex="-1" aria-labelledby="oct7-heading"',section)
        self.assertIn('datetime="2026-10-07T18:37:14Z"',section)
        self.assertIn('href="#shrinking-bin-sampling"',section)
        self.assertIn('href="#newer-work"',section)
        self.assertIn('href="#latest-work"',section)

    def test_landed_and_candidate_work_stay_distinct(self):
        section=self.section()
        for term in ('Math #390','scientific effect is NONE','its sign at <code>L = 24</code> is not proved','Conjecture 7 remains open',
                     'No rate or certified number','xAI Grok 4.7','zero','none is an owner reading',
                     'open proposals at this cut','does not present them as landed',
                     'pull/388','pull/387','pull/384','pull/392','pull/393',
                     'v2.6','152','(I1)','(I2)','(I4)','A kernel build or merge is not an alignment review',
                     'Math #193','does not read that closure as a completed proof',
                     'main #276','“not recorded”, never a pass','change no scientific status','not a status record',
                     'not scientific acceptance','3 October supplement','27 September'):
            with self.subTest(term=term):self.assertIn(term,section)
        parsed=PinnedReadingLinks('<section '+section)
        self.assertEqual(parsed.links,list(OCT7_PINS))
        self.assertEqual(parsed.hash_bindings,list(OCT7_PINS.items()))

    def test_cap_card_keeps_the_elder_death_hypotheses(self):
        # PRES284-01: torus_elder_death_level_blocks also assumes L > 0 and f continuous
        # (Math 54ebcede ALIGNMENT.md row 114); the cap clause must not claim it.
        card=self.section().split('id="cap-on-torus"',1)[1].split('</article>',1)[0]
        paragraphs=[p for p in card.split('<p>') if 'elder death level' in p]
        self.assertEqual(len(paragraphs),1)
        paragraph=paragraphs[0]
        self.assertNotIn('elder death level',paragraph.split('four continuous derivatives',1)[0])
        clause=paragraph.split('elder death level',1)[0].rsplit('. ',1)[-1]
        for term in ('positive side <code>L</code>','continuous on the whole torus'):
            with self.subTest(term=term):self.assertIn(term,clause)

    def test_open_proposals_are_never_called_landed(self):
        section=self.section()
        for number in (384,387,388,392,393):
            card=section.split(f'pull/{number}"',1)[0].rsplit('<p>',1)[1]
            self.assertIn('Still candidates',card)


class FiniteH0LifetimeReading(unittest.TestCase):
    """The 9 October cut: one landed finite Gaussian H0 lifetime source, quoted, never promoted."""
    CUT_ID = 'reading-cut-20261009'
    CUT_ISO = '2026-10-09T00:05:00Z'
    LANDED_REF = '2c84170712c59d9de580c172815bd30bac5d93cd'
    LANDED_README = ('c44553c00c86fad5045ac8c9e6c90f70b1e347173d2bad6f59804b35c888e5a7', 10559)
    ALLOWED_CLASSES = {'latest-work','section-heading','eyebrow','section-intro','latest-grid','latest-card',
                       'card-kicker','display-formula','card-status','latest-boundary','latest-identities'}
    FORBIDDEN = (
        # status or acceptance words this cut never uses
        'ACCEPT','AMEND','PENDING','NOT_VERIFIED','PROVED_REVIEWED','AUTHOR_SIDE_CANDIDATE',
        'universality class','universal law','proves universality','accepted','established','independently',
        'peer-review','independent review','independent read','independent check','validated','confirmed','admitted','kernel-checked','Lean-verified','machine-checked',
        'almost sure','held-out test','ℓ^(1/3)','side24','SIDE24','continuous density','exponent 1/3',
        # stale counts and ordering from before main #329
        'seven proof sources','all seven','five merges','last of those proofs','latest proof','newest proof',
        'The four files have the same bytes','last changed in main #326',
        # main #329 mathematics, scope or files, which this cut does not restate or link
        'Poisson','independent of dimension','any dimension','every dimension','higher dimension',
        'five interfaces','all five','image-sum','infinite image','d>=2','d≥2','iid','independent copies',
        'expanding-domain','cluster avoidance','counterexample','r dr',
        'fold_structural_universality','periodized_gaussian_h0','iid_short_bar_process',
        'Regular cubic folds and the structural H0 lifetime exponent',
        'Actual H0 lifetime intensity for the exact periodized Gaussian law',
        'Marked iid-copy Poisson limit for actual short H0 bars',
        'blob/main','tree/main',
    )
    QUOTES = {
        'program': ('a reusable theorem connecting a local singularity, the probability law transverse to its discriminant, and the intensity of actual persistence bars',
                    'a research design and execution route, not a new scientific-status register or an acceptance vote',
                    'a precisely stated transfer target','does not verify those five hypotheses for any new field',
                    'Keep the full objective open until its explicit proof, computation, publication and uptake requirements have their actual evidence.'),
        'readme': ('constructs a Borel selector and counts every actual finite superlevel H0 bar exactly once',
                   'The selected density is not asserted continuous',
                   'The sampler and catalogue admissibility labels retain their original meaning.',
                   'generates explicitly exploratory fields',
                   'does not estimate a slope or compare against reference coefficient digits',
                   'No held-out seeds have been consumed by this suite.',
                   'regular-fold structural theorem','exact periodized Gaussian H0 adapter','marked short-bar process theorem'),
        'proof': ('No human, blind or provider-distinct acceptance, formal Lean proof, sampler execution, seed consumption or scientific promotion is claimed.',
                  'the particular finite Borel representative constructed in Section 5',
                  'no O(1) near-term remainder is claimed',
                  'No finite-word or quantized sampler is identified with the continuous ideal Gaussian coefficient law.',
                  'conventional analytic source'),
        'review': ('a conventional analytic theorem with explicit imports',
                   'All contributors/readers are source and route exposed OpenAI/Codex.',
                   'No human, blind reconstruction or provider-distinct acceptance is claimed.',
                   'Fresh full exact-file nonauthor reviews of the incorporated proof: PASS.',
                   'H0 consumes C/CG/ET/FG',
                   'Both actual-source mathematical verdicts and proof bytes are unchanged.',
                   'The coefficient is positive and field-dependent',
                   'Finite-family uniformity is not twenty empirical confirmations.',
                   'Positivity-boundary spectra, K1, non-Gaussian coefficients, infinite-field laws, numerical coupling, quantitative usable windows, higher homology, factorial moments, process limits, formal source alignment, blind fifty-law validation and external uptake remain outside the result.'),
    }
    SOURCE_PATHS = {'program':'docs/PERSISTENCE_UNIVERSALITY_PROGRAM.md','readme':'experiments/universality/README.md',
                    'proof':'experiments/universality/finite_h0_lifetime/PROOF.md','review':'experiments/universality/finite_h0_lifetime/REVIEW.md'}
    EARLIER_NOTE_DIGESTS = ('b49f2ea5975dc05270f51fea4a3574093f726258cdab94888742507c2caa275f',
                            '55eb6c33105190da4194e63d90ceeb40f3434b14b92803700ab840df67e9295e',
                            '3f1d0e1f1af51a20aa49681d0255d09d728e5d62c5f00cbca90e0671798265ed',
                            '652e66f8645eb1a34c28a85900b05405d424bb0e282b86a7d5597f10b8f9c8e3',
                            '52dc91e3bc68b8f023f11e9ec37b1423853d721dc9188046246e96b4131e205f',
                            'a6fb3d0397e622d8a1061ff6e83d05f66e99d53fbc2dd35c1b8ee82d076a4458')
    PROSE_NUMBERS = ('8 314 326 8 0 326 0 329 23:58 0 0 7 318 324 326 0 0 0 1/3 2/3 5 1 1 526 0 326 8 2026 22:42 314 319 321 323 326 '
                     '27,064 329 8 23:58 0 326 8 22:42 314 329 10,559 256 326 256 256 256 0 256 0 321 323 326 319 321 323 329 3 18:00 '
                     '107 3 20:01 8 2026 256 99 99 103 103 4 00:18 8 2026 256 7 18:37 8 2026 00:58 193 18:45 7 18:46 4 7 18:37')
    # Each quote keeps the source the page names for it.
    ATTRIBUTIONS = ('In the suite guide’s words, the proof “constructs a Borel selector',
                    'The review record says “The coefficient is positive and field-dependent”',
                    'The proof says its pointwise law concerns “the particular finite Borel representative',
                    'and “no O(1) near-term remainder is claimed”',
                    'In the suite guide’s words, “The selected density is not asserted continuous”',
                    'The review record says “Finite-family uniformity is not twenty empirical confirmations.”',
                    'The proof says “No finite-word or quantized sampler',
                    'In the review record’s words, “Positivity-boundary spectra',
                    'the record says “All contributors/readers are source and route exposed OpenAI/Codex.”',
                    'The proof’s own declaration: “No human, blind or provider-distinct acceptance',
                    'The review record calls the proof “a conventional analytic theorem with explicit imports”',
                    'The program says to “Keep the full objective open',
                    'In the suite guide’s words, “The sampler and catalogue admissibility labels',
                    'Its record states “H0 consumes C/CG/ET/FG”',
                    'After the first correction the record says “Both actual-source mathematical verdicts',
                    'In the suite guide’s words, its exploratory runner “generates explicitly exploratory fields”',
                    'The program describes itself as “a research design and execution route',
                    'which counts actual finite H0 bars for fixed ideal finite Gaussian laws; it is not a universality theorem.')
    OTHER_PAGES_ROW = ('<dt>Words used on other pages</dt><dd>ACCEPT and AMEND / open are the proof index’s and STATUS.md’s wording at their pinned snapshots, shown on <a href="museum.html#claims">Source records</a> and the <a href="workspace.html#board">Library board</a>. PROVED_REVIEWED, AUTHOR_SIDE_CANDIDATE and the other gate classifications are the downstream gate’s states, shown on the <a href="dependencies.html">dated claim-dependency snapshot</a>. Each page quotes its own source; none of ACCEPT, AMEND, PROVED_REVIEWED or AUTHOR_SIDE_CANDIDATE appears in this page’s cuts.</dd>')

    class Tree(HTMLParser):
        """Elements of one section with class, attributes, direct text and inner text."""
        VOID = {'br','img','meta','link','hr','input'}
        def __init__(self, text):
            super().__init__();self.elements=[];self.stack=[];self.feed(text)
        def handle_starttag(self, tag, attrs):
            node={'tag':tag,'attrs':dict(attrs),'classes':set((dict(attrs).get('class') or '').split()),'text':[],'parents':[n['tag'] for n in self.stack],'parent_classes':set().union(*[n['classes'] for n in self.stack]) if self.stack else set()}
            self.elements.append(node)
            if tag not in self.VOID: self.stack.append(node)
        def handle_endtag(self, tag):
            while self.stack:
                node=self.stack.pop()
                if node['tag']==tag: break
        def handle_data(self, data):
            for node in self.stack: node['text'].append(data)
        def find(self, tag=None, cls=None):
            return [n for n in self.elements if (tag is None or n['tag']==tag) and (cls is None or cls in n['classes'])]

    @staticmethod
    def text_of(node): return ''.join(node['text'])

    def page(self): return (SITE/'research.html').read_text()

    def section(self):
        text=self.page()
        start=text.index(f'<section id="{self.CUT_ID}"');end=text.index('<section id="reading-cut-20261007"')
        return text[start:end]

    def card(self):
        section=self.section()
        return section[section.index('<article class="latest-card" id="finite-h0-lifetimes"'):section.index('</article>')+len('</article>')]

    def source_bytes(self, ref, path):
        # Pinned sources are read from git objects (CI checks out with full history); a tree without them cannot check identities.
        import subprocess
        try:
            return subprocess.run(['git','cat-file','blob',f'{ref}:{path}'],cwd=ROOT,check=True,capture_output=True).stdout
        except (OSError,subprocess.CalledProcessError):
            self.skipTest(f'git object {ref[:8]}:{path} unavailable in this checkout')

    def test_cut_is_newest_entry_and_links_to_the_7_october_cut_and_upstream(self):
        text=self.page();section=self.section()
        self.assertTrue(section.endswith('</section>\n'),'the new cut sits immediately above the 7 October cut')
        self.assertLess(text.index('id="status-words"'),text.index(f'<section id="{self.CUT_ID}"'))
        self.assertEqual(text.index('class="latest-work"'),text.index(f'<section id="{self.CUT_ID}"')+len(f'<section id="{self.CUT_ID}" '),'first dated cut on the page')
        self.assertIn(f'<section id="{self.CUT_ID}" class="latest-work" tabindex="-1" aria-labelledby="octnew-heading">',section)
        self.assertIn('<h2 id="octnew-heading">How common are short-lived bars in a fixed finite Gaussian field?</h2>',section)
        self.assertEqual(re.findall(r'<time datetime="([^"]+)">',section),[self.CUT_ISO])
        self.assertIn(f'<time datetime="{self.CUT_ISO}">9 October 2026 · 00:05 UTC</time>',section)
        self.assertIn('<a href="#reading-cut-20261007">7 October 18:37 UTC reading cut below</a>',section)
        self.assertIn('<a href="#newer-work">current branches and discussions</a>',section)
        self.assertIn(f'<a class="button secondary" href="#{self.CUT_ID}">Latest public work</a>',text)
        self.assertIn(f'Newest reading cut: <time datetime="{self.CUT_ISO}">9 October 2026, 00:05 UTC</time>.',text)
        first_row=text.split('<nav class="cut-list"',1)[1].split('<li>',2)[1]
        self.assertEqual(first_row,f'<a href="#{self.CUT_ID}">Reading cut · 9 October 2026, 00:05 UTC</a> — How common are short-lived bars in a fixed finite Gaussian field? <span class="muted">Full sources: open “Exact sources, review record and the program”.</span></li>\n')
        self.assertIn(f'<details class="latest-identities" id="{self.CUT_ID}-details"><summary>Exact sources, review record and the program</summary>',section)
        names=[name for name in re.findall(r'<a [^>]*>(.*?)</a>',text,re.S)]
        for name in ('7 October 18:37 UTC reading cut below','current branches and discussions','Read the finite H0 lifetime proof',
                     'Proof bytes checked for this cut','Read the persistence universality program',
                     'Read the suite guide, which links all ten proof sources and their reviews',
                     'Read the H0 lifetime review record, with both verdicts and the exclusions'):
            with self.subTest(name=name):self.assertEqual(names.count(name),1)

    def test_quoted_phrases_are_verbatim_in_their_pinned_sources(self):
        # The PASS line is quoted in the #status-words row that names this cut; every other quote sits in the cut.
        words=self.page().split('<details id="status-words"',1)[1].split('</details>',1)[0]
        section=self.section()+words
        for key,quotes in self.QUOTES.items():
            source=self.source_bytes(OCTNEW_REF,self.SOURCE_PATHS[key]).decode()
            flat=' '.join(source.replace('`','').replace('**','').split())
            for quote in quotes:
                with self.subTest(source=key,quote=quote):
                    self.assertIn(quote,section)
                    self.assertIn(quote,flat)
        for attribution in self.ATTRIBUTIONS:
            with self.subTest(attribution=attribution):self.assertEqual(section.count(attribution),1)
        self.assertNotIn(', and the record says “Both actual-source',section)

    def test_forbidden_words_and_later_source_details_are_absent(self):
        section=self.section()
        for word in self.FORBIDDEN:
            with self.subTest(word=word):
                self.assertNotIn(word,section)
                # lower-case phrases are also refused at a sentence start or in any other case
                if word==word.lower():self.assertNotIn(word,section.lower())
        self.assertEqual(section.count('periodized'),1)
        self.assertEqual(section.count('exact periodized Gaussian H0 adapter'),1)
        for phrase in ('not a universality theorem.','not a universality theorem for a wider class of fields'):
            self.assertIn(phrase,section)
        self.assertNotIn('universality',section.split('<h2',1)[1].split('</h2>',1)[0])
        for link,name in re.findall(r'<a [^>]*href="([^"]+)"[^>]*>(.*?)</a>',section):
            if 'universality' in name: self.assertEqual(name,'Read the persistence universality program')
        lead=section.split('</div>',1)[1].split('<div class="latest-grid">',1)[0]
        self.assertEqual(section.count('#324'),1);self.assertIn('#324',lead)
        card=self.card()
        # The card, its status rows included, does not name main #329 or its sources; the lead and the disclosure do.
        self.assertEqual(card.count('#329'),0)
        for token in ('#324','regular-fold','short-bar process','periodized','structural theorem','adapter'):
            with self.subTest(card=token):self.assertNotIn(token,card)
        eyebrow=section.split('<p class="eyebrow">',1)[1].split('</p>',1)[0]
        self.assertEqual(eyebrow,'Landed 8 October · main #314–#326 · fixed finite Gaussian scope')
        hero=self.page().split('<section id="reading-cut-20261009"',1)[0].split('<section class="hero">',1)[1]
        for token in ('#329','#324','regular-fold','short-bar','periodized','universality'):
            with self.subTest(hero=token):self.assertNotIn(token,hero)

    def test_two_estimands_keep_their_exponents_and_no_coefficient_value(self):
        section=self.section();card=self.card()
        self.assertIn('The first is for a specified density <code>ν_bar</code> of the expected bar-lifetime measure per unit area.',card)
        self.assertIn('The second is for the expected number <code>E N_bar((0, t])</code> of finite bars with lifetime at most <code>t</code>, per unit area',card)
        formulas=re.findall(r'<p class="display-formula"><code>(.*?)</code></p>',section)
        self.assertEqual(formulas,['ν_bar(ℓ) ~ C_loc ℓ^(−1/3), as ℓ decreases to zero',
                                   'E N_bar((0, t]) / V ~ (3/2) C_loc t^(2/3), as t decreases to zero'])
        self.assertIn('The density exponent is −1/3; the cumulative exponent is 2/3.',card)
        self.assertNotIn('t^(−1/3)',section);self.assertNotIn('ℓ^(2/3)',section)
        # no second statement of either exponent anywhere in the cut, in either minus sign
        self.assertEqual((section.count('exponent'),section.count('cumulative'),section.count('−1/3'),section.count('2/3')),(2,1,3,2))
        self.assertNotIn('-1/3',section)
        plain=re.sub(r'<[^>]+>','',card)
        self.assertIsNone(re.search(r'\d[.,]\d{1,2}(?!\d)|\d\.\d',plain),'no coefficient digits on the card')
        self.assertIsNone(re.search(r'C_loc\s*(?:[=≈≃<>]|\(?\d)',plain))
        self.assertIn('this page gives no value',card)
        # the promise is page-wide: no decimal anywhere in the cut's prose, and no digit in any sentence naming C_loc
        prose=re.sub(r'<[^>]+>','',re.sub(r'<p class="display-formula">.*?</p>','',section))
        self.assertIsNone(re.search(r'\d[.,]\d{1,2}(?!\d)|\d\.\d',prose))
        for sentence in re.split(r'(?<=[.;:])\s+',prose):
            if 'C_loc' in sentence:
                with self.subTest(sentence=sentence[:60]):self.assertNotRegex(sentence,r'\d')
        self.assertIn('the same local coefficient as in this line’s candidate-lifetime theorem.',card)
        self.assertNotIn('candidate-pair',section)
        # The cut does not change once it lands: its prose numbers (outside code, display formulas and the time) are pinned in order.
        numbers=re.sub(r'<time[^>]*>.*?</time>','',re.sub(r'<code>.*?</code>','',re.sub(r'<p class="display-formula">.*?</p>','',section)))
        self.assertEqual(' '.join(re.findall(r'\d[\d,.:–−/]*\d|\d',re.sub(r'<[^>]+>','',numbers))),self.PROSE_NUMBERS)
        # The #status-words rows that name this cut give no coefficient value either.
        words=self.page().split('<details id="status-words"',1)[1].split('</details>',1)[0]
        for row in re.findall(r'<dd>(.*?)</dd>',words,re.S):
            if '9 October 2026 cut' not in row: continue
            with self.subTest(row=row[:60]):
                self.assertNotIn('C_loc',row);self.assertIsNone(re.search(r'\d[.,]\d|[=≈≃]',re.sub(r'<[^>]+>','',row)))

    def test_one_card_one_face_pin_three_status_rows_and_neutral_classes(self):
        section=self.section();tree=self.Tree(section)
        cards=tree.find(cls='latest-card')
        self.assertEqual([c['attrs'].get('id') for c in cards],['finite-h0-lifetimes'])
        self.assertEqual(len(tree.find('p','display-formula')),2)
        self.assertEqual(len(tree.find('p','latest-boundary')),1)
        statuses=tree.find('dl','card-status');self.assertEqual(len(statuses),1)
        self.assertEqual([self.text_of(n) for n in tree.find('dt')],['Landed?','Read by','Scientific effect'])
        self.assertEqual(len(tree.find('dd')),3)
        self.assertTrue(all('card-status' in n['parent_classes'] and 'latest-card' in n['parent_classes'] for n in tree.find('dt')+tree.find('dd')))
        rows=dict(zip([self.text_of(n) for n in tree.find('dt')],[self.text_of(n) for n in tree.find('dd')]))
        self.assertTrue(rows['Landed?'].startswith('Landed at main 2c841707 (main #326, merged 8 October 2026, 22:42 UTC).'))
        self.assertEqual(rows['Read by'],'OpenAI/Codex · “PASS”, quoted from the linked review record · the landed proof’s exact 27,064 bytes; organizational-independence credit zero.')
        self.assertEqual(rows['Scientific effect'],'The proof’s own declaration: “No human, blind or provider-distinct acceptance, formal Lean proof, sampler execution, seed consumption or scientific promotion is claimed.”')
        used=set().union(*[n['classes'] for n in tree.elements])
        self.assertLessEqual(used,self.ALLOWED_CLASSES,used-self.ALLOWED_CLASSES)
        for bad in ('accept','amend','gold','latest-chain','lineage-note','cut-pointer'):
            self.assertNotIn(bad,used)
        card=PinnedReadingLinks('<section '+self.card())
        self.assertEqual(card.links,OCTNEW_FACE_PINS)
        self.assertIn('<strong>Who read it:</strong>',self.card())

    def test_pinned_items_carry_only_their_digest_and_the_digests_match_the_pinned_bytes(self):
        section=self.section()
        parsed=PinnedReadingLinks('<section '+section)
        self.assertEqual(parsed.links,OCTNEW_FACE_PINS+list(OCTNEW_PINS))
        self.assertEqual(parsed.hash_bindings,list(OCTNEW_PINS.items()))
        for item in re.findall(r'<li>(.*?)</li>',section,re.S):
            if 'data-source-kind="pinned"' not in item: continue
            codes=re.findall(r'<code>(.*?)</code>',item)
            with self.subTest(item=item[:80]):
                self.assertEqual(len(codes),1);self.assertRegex(codes[0],r'\A[0-9a-f]{64}\Z')
        for href in re.findall(r'href="(https://[^"]+)"',section):
            with self.subTest(href=href):
                self.assertTrue(href.startswith(OCTNEW_PREFIX));self.assertEqual(urlsplit(href).query+urlsplit(href).fragment,'')
        self.assertNotIn(self.LANDED_REF+'/',section)
        self.assertEqual(section.count(self.LANDED_REF),1)
        for url,digest in OCTNEW_PINS.items():
            data=self.source_bytes(OCTNEW_REF,url[len(OCTNEW_PREFIX):])
            with self.subTest(url=url):self.assertEqual(sha256(data).hexdigest(),digest)
            if not url.endswith('/README.md'):
                self.assertEqual(sha256(self.source_bytes(self.LANDED_REF,url[len(OCTNEW_PREFIX):])).hexdigest(),digest,'same bytes where the proof landed')
        proof=self.source_bytes(OCTNEW_REF,'experiments/universality/finite_h0_lifetime/PROOF.md')
        self.assertEqual((len(proof),proof.count(b'\n')),(27064,526))
        self.assertIn('27,064 bytes',section);self.assertIn('all 526 lines',section)
        landed_readme=self.source_bytes(self.LANDED_REF,'experiments/universality/README.md')
        self.assertEqual((sha256(landed_readme).hexdigest(),len(landed_readme)),self.LANDED_README)
        self.assertIn(f'earlier 10,559-byte suite guide its readers read (SHA-256 <code>{self.LANDED_README[0]}</code>)',section)

    def test_earlier_notes_are_restated_with_their_qualifiers_and_without_links(self):
        text=self.page();section=self.section()
        notes=section.split('<p>Notes added after earlier cuts',1)[1].split('<p>The source identities above were checked',1)[0]
        self.assertNotIn('<a ',notes);self.assertNotIn('lineage-note',notes)
        for digest in self.EARLIER_NOTE_DIGESTS:
            with self.subTest(digest=digest):self.assertIn(digest,notes)
        for card in ('actual-bars','strict-bar-coefficient'):
            original=LINEAGE_NOTE.findall(text.split(f'id="{card}"',1)[1].split('</article>',1)[0])[0]
            body=original.split('</strong> ',1)[1][:-len('</p>')].replace('“nonauthor” here','“nonauthor” there')
            with self.subTest(card=card):self.assertIn(body,notes)
        self.assertIn('<code>PASS_TECHNICAL</code>',notes)
        self.assertIn('The first comment after the reopening, at 18:46 UTC, is headed “Tracking correction verified — issue reopened; scientific records unchanged”.',notes)
        for paragraph in re.findall(r'<(?:p|dd)\b[^>]*>(.*?)</(?:p|dd)>',section,re.S):
            if 'nonauthor' in paragraph:
                with self.subTest(paragraph=paragraph[:60]):self.assertIn('does not mean independent',paragraph)
        self.assertEqual(section.count('does not mean independent'),3)
        self.assertIn('Organizational-independence credit is zero',self.card())

    def test_status_words_quote_pass_and_the_source_label_without_rewording_other_rows(self):
        text=self.page()
        words=text.split('<details id="status-words" class="reading-glossary">',1)[1].split('</details>',1)[0]
        self.assertIn('<summary>Status words on this page and where each is quoted from (as read 9 October 2026)</summary>',words)
        terms=re.findall(r'<dt>(.*?)</dt>',words)
        self.assertEqual(terms[2:5],['aligned · kernel-checked · read · slice read · readback','PASS','conventional analytic source'])
        self.assertEqual(terms[5],'NOT READY')
        self.assertIn('linked in the 9 October 2026 cut (“Fresh full exact-file nonauthor reviews of the incorporated proof: PASS.”)',words)
        self.assertIn('It is that record’s own word, not PASS_TECHNICAL.',words)
        self.assertIn('In that record “nonauthor” does not mean independent: organizational-independence credit is zero.',words)
        self.assertIn('Its review record calls it “a conventional analytic theorem with explicit imports”.',words)
        self.assertIn(self.OTHER_PAGES_ROW,words)
        for section in re.findall(r'<section id="[^"]+" class="latest-work".*?</section>',text,re.S):
            for token in ('ACCEPT','AMEND','PROVED_REVIEWED','AUTHOR_SIDE_CANDIDATE'):
                with self.subTest(section=section[:40],token=token):self.assertNotIn(token,section)

    def test_card_status_rows_use_the_glossary_type_not_monospace(self):
        css=(SITE/'home.css').read_text()
        self.assertIn('.reading-glossary dt, .latest-card .card-status dt { font-weight: 650; margin-top: 12px; }',css)
        rule=re.search(r'\.reading-glossary dd, \.latest-card \.card-status dd \{([^}]*)\}',css)
        self.assertIsNotNone(rule);self.assertIn('font-family: inherit',rule.group(1));self.assertIn('font-size: 15px',rule.group(1))
        self.assertIn('.latest-card .card-status dd { max-width: 78ch; }',css)
        self.assertIn('@media print { .latest-card:has(> .card-status) { break-inside: auto; } .latest-card .card-status, p.display-formula { break-inside: avoid; } }',css)
        self.assertIn('.latest-identities code, .latest-card code { overflow-wrap: anywhere; }',css)
        # agent paths in the card are breakable code, so they cannot push past the card at 320 px with WCAG 1.4.12 spacing
        for name in ('/root/universality_review','/root/benchmark_formal_audit','/root/final_contract_review'):
            with self.subTest(name=name):self.assertIn(f'<code>{name}</code>',self.card())
        for sheet in ('home.css','brand.css'):
            for selector,body in re.findall(r'([^{}]*card-status[^{}]*)\{([^}]*)\}',(SITE/sheet).read_text()):
                with self.subTest(sheet=sheet,selector=selector.strip()):self.assertNotIn('monospace',body)

    def test_browser_flows_cover_the_new_cut(self):
        harness=(ROOT/'tools/public_shop_browser_check.py').read_text()
        for token in (f'#{self.CUT_ID}',f'#{self.CUT_ID}-details',self.CUT_ISO,'7 October 18:37 UTC reading cut below',
                      '#finite-h0-lifetimes .card-status dd','Organizational-independence credit is zero'):
            with self.subTest(token=token):self.assertIn(token,harness)


if __name__=='__main__': unittest.main()
