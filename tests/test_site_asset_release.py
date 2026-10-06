"""Release URLs must select fresh entry and transitive assets, without editing data."""
from html.parser import HTMLParser
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit, parse_qs
import subprocess
import importlib.util
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / 'tools/site_asset_release.py'

class Assets(HTMLParser):
    def __init__(self, text):
        super().__init__(); self.urls = []; self.feed(text)
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'script' and 'src' in a: self.urls.append(a['src'])
        if tag == 'link' and a.get('rel') in ('stylesheet', 'icon'): self.urls.append(a['href'])

spec = importlib.util.spec_from_file_location('asset_release', TOOL)
asset_release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(asset_release)

class AssetRelease(unittest.TestCase):
    def stage(self, root):
        subprocess.run(['git', '-C', str(root), 'add', '-A'], check=True,
                       capture_output=True)

    def run_tool(self, site, *args):
        return subprocess.run([sys.executable, str(TOOL), '--site', str(site), *args], text=True, capture_output=True)

    def fixture(self, root):
        subprocess.run(['git', 'init', '-q', str(root)], check=True, capture_output=True)
        site = root/'site'; site.mkdir()
        (site/'index.html').write_text('<script type="module" src="app.mjs"></script><link rel="stylesheet" href="style.css"><a href="other.html#reading">Read</a>')
        (site/'app.mjs').write_text("import {x} from './core.mjs'; const later=()=>import('./lazy.mjs'); const external=()=>import('https://example.test/tool.mjs');")
        (site/'core.mjs').write_text('export const x=1;')
        (site/'lazy.mjs').write_text("import './core.mjs';")
        (site/'style.css').write_text('body{background:url("mark.svg")}')
        (site/'mark.svg').write_text('<svg/>')
        (site/'config.json').write_bytes(b'{"pin":"unchanged"}\n')
        (site/'proof.md').write_bytes(b'Exact research source.\n')
        self.stage(root)
        return site

    def test_current_entry_assets_select_a_release(self):
        for path in (ROOT/'docs/site').glob('*.html'):
            for asset in Assets(path.read_text()).urls:
                if urlsplit(asset).scheme: continue
                with self.subTest(page=path.name, asset=asset):
                    release = parse_qs(urlsplit(asset).query).get('site-release', [])
                    self.assertEqual(len(release), 1, 'Unversioned entry can reuse an older script/style')
                    self.assertRegex(release[0], r'^[0-9a-f]{64}$')

    def test_transitive_imports_and_css_share_the_release(self):
        with TemporaryDirectory() as tmp:
            site = self.fixture(Path(tmp)); result = self.run_tool(site)
            self.assertEqual(result.returncode, 0, result.stderr)
            release = result.stdout.strip()
            self.assertRegex(release, r'^[0-9a-f]{64}$')
            for file, target in [('app.mjs','./core.mjs'),('app.mjs','./lazy.mjs'),('lazy.mjs','./core.mjs'),('style.css','mark.svg')]:
                self.assertIn(target+'?site-release='+release, (site/file).read_text())
            self.assertIn("import('https://example.test/tool.mjs')", (site/'app.mjs').read_text())
            self.assertIn('href="other.html#reading"', (site/'index.html').read_text())
            self.assertEqual((site/'config.json').read_bytes(), b'{"pin":"unchanged"}\n')
            self.assertEqual((site/'proof.md').read_bytes(), b'Exact research source.\n')

    def test_changed_transitive_asset_and_config_invalidate_every_entry(self):
        with TemporaryDirectory() as tmp:
            site = self.fixture(Path(tmp)); first = self.run_tool(site)
            self.assertEqual(first.returncode, 0, first.stderr)
            (site/'core.mjs').write_text('export const x=2;')
            stale = self.run_tool(site, '--check'); self.assertNotEqual(stale.returncode, 0)
            second = self.run_tool(site); self.assertEqual(second.returncode, 0, second.stderr)
            self.assertNotEqual(first.stdout, second.stdout)
            (site/'config.json').write_text('{"pin":"new"}\n')
            third = self.run_tool(site); self.assertEqual(third.returncode, 0, third.stderr)
            self.assertNotEqual(second.stdout, third.stdout)

    def test_quoted_css_import_is_versioned_and_missing_import_is_refused(self):
        with TemporaryDirectory() as tmp:
            site = self.fixture(Path(tmp))
            (site/'style.css').write_text('@import "theme.css" screen;\nbody{color:navy}')
            (site/'theme.css').write_text('h1{color:gold}')
            self.stage(Path(tmp))
            result = self.run_tool(site)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('theme.css?site-release='+result.stdout.strip(), (site/'style.css').read_text())
            (site/'theme.css').unlink()
            missing = self.run_tool(site, '--check')
            self.assertNotEqual(missing.returncode, 0)
            self.assertIn('missing/outside local asset', missing.stderr)

    def test_generation_is_idempotent_and_check_does_not_repair(self):
        with TemporaryDirectory() as tmp:
            site = self.fixture(Path(tmp)); first = self.run_tool(site)
            self.assertEqual(first.returncode, 0, first.stderr)
            before = {p.name:p.read_bytes() for p in site.iterdir()}
            second = self.run_tool(site); self.assertEqual(second.stdout, first.stdout)
            self.assertEqual(before, {p.name:p.read_bytes() for p in site.iterdir()})
            (site/'app.mjs').write_text("import './core.mjs';")
            broken = {p.name:p.read_bytes() for p in site.iterdir()}
            self.assertNotEqual(self.run_tool(site, '--check').returncode, 0)
            self.assertEqual(broken, {p.name:p.read_bytes() for p in site.iterdir()})


    def test_query_order_fragment_and_html_entity_survive(self):
        old = 'a' * 64
        with TemporaryDirectory() as tmp:
            root = Path(tmp); site = self.fixture(root)
            (site/'index.html').write_text(
                '<link href="style.css?site-release='+old+'&amp;media=screen&amp;x=1&amp;x=2#part">')
            result = self.run_tool(site)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('media=screen&amp;x=1&amp;x=2&amp;site-release='+result.stdout.strip()+'#part',
                          (site/'index.html').read_text())
            self.assertEqual(self.run_tool(site, '--check').returncode, 0)
            self.assertEqual(asset_release.without_release(
                'style.css?media=screen&site-release='+old+'&x=1&x=2#part'),
                'style.css?media=screen&x=1&x=2#part')
            self.assertEqual(asset_release.without_release(
                'style.css?site-release='+old+'&media=screen'),
                'style.css?media=screen')

    def test_untracked_noise_ignored_and_tracked_input_changes_identity(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); site = self.fixture(root)
            first = self.run_tool(site)
            self.assertEqual(first.returncode, 0, first.stderr)
            (site/'editor.tmp').write_text('irrelevant')
            (site/'extra.json').write_text('{}')
            (site/'ghost.mjs').write_text('not tracked')
            self.assertEqual(self.run_tool(site).stdout, first.stdout)
            self.stage(root)
            second = self.run_tool(site)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertNotEqual(first.stdout, second.stdout)
            (site/'extra.json').unlink()
            self.stage(root)
            third = self.run_tool(site)
            self.assertEqual(third.returncode, 0, third.stderr)
            self.assertNotEqual(second.stdout, third.stdout)

    def test_inventory_has_explicit_posix_order_and_two_root_order(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); site = self.fixture(root)
            (site/'a_dir').mkdir(); (site/'a_dir'/'z.json').write_text('{}')
            (site/'a_file.json').write_text('{}')
            catalog = root/'public-math'; catalog.mkdir()
            (catalog/'catalog.json').write_text('{}')
            self.stage(root)
            names = [p.relative_to(root).as_posix() for p in asset_release.inventory(site)]
            wanted = sorted([p for p in names if p.startswith('site/')])
            wanted += ['public-math/catalog.json']
            self.assertEqual(names, wanted)

    def test_inactive_examples_do_not_resolve_missing_assets(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); site = self.fixture(root)
            (site/'index.html').write_text(
                '<!-- <script src="ghost.js"></script> -->'
                '<script>const s = \'<link href="ghost.css">\';</script>'
                '<textarea><script src="ghost.js"></script></textarea>'
                '<script data-example=\'src="ghost.js"\' src="app.mjs"></script>')
            (site/'style.css').write_text(
                '/* url(ghost.svg); @import "ghost.css"; */ '
                'a::before{content:\'url(ghost.svg)\'} body{background:url(mark.svg)}')
            (site/'app.mjs').write_text(
                'const s = "import(\'ghost.js\')"; /* from "ghost.js" */ '
                'const t = '+chr(96)+'import("ghost.js")'+chr(96)+'; '
                'const r = /import("ghost.js")/; '
                'if (ok) /import("ghost.js")/.test(s); import "./core.mjs";')
            result = self.run_tool(site)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('src="ghost.js"', (site/'index.html').read_text())
            self.assertIn('url(ghost.svg)', (site/'style.css').read_text())
            self.assertIn('import("ghost.js")', (site/'app.mjs').read_text())
            self.assertIn('./core.mjs?site-release='+result.stdout.strip(),
                          (site/'app.mjs').read_text())
            self.assertEqual(self.run_tool(site, '--check').returncode, 0)

    def test_reexports_and_dynamic_options_but_not_computed_imports(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); site = self.fixture(root)
            (site/'app.mjs').write_text(
                'export * from "./core.mjs"; export {x} from "./core.mjs"; '
                'export * as ns from "./core.mjs"; '
                'import /*c*/ ("./lazy.mjs", {}); obj.import("ghost.js"); '
                'obj?.import("ghost.js"); import("./dir/" + name);')
            result = self.run_tool(site)
            self.assertEqual(result.returncode, 0, result.stderr)
            text = (site/'app.mjs').read_text()
            self.assertEqual(text.count('./core.mjs?site-release='+result.stdout.strip()), 3)
            self.assertIn('./lazy.mjs?site-release='+result.stdout.strip(), text)
            self.assertIn('import("./dir/" + name)', text)
            self.assertIn('obj.import("ghost.js")', text)

    def test_unicode_and_property_keywords_do_not_become_imports_or_regex(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); site = self.fixture(root)
            (site/'app.mjs').write_text(
                'αimport("./core.mjs"); obj.if(ok) / scale / import("./lazy.mjs"); '
                'obj?.if(ok) / scale / import("./lazy.mjs"); '
                'obj.return / scale / import("./lazy.mjs");')
            (site/'style.css').write_text(
                'a{madeup:éurl(ghost.svg);other:💠url(mark.svg);'
                'combining:́url(mark.svg);background:url(mark.svg)}')
            result = self.run_tool(site)
            self.assertEqual(result.returncode, 0, result.stderr)
            text = (site/'app.mjs').read_text()
            self.assertIn('αimport("./core.mjs")', text)
            self.assertNotIn('αimport("./core.mjs?site-release=', text)
            self.assertEqual(text.count('./lazy.mjs?site-release='+result.stdout.strip()), 3)
            css = (site/'style.css').read_text()
            self.assertIn('éurl(ghost.svg)', css)
            self.assertIn('💠url(mark.svg)', css)
            self.assertIn('́url(mark.svg)', css)
            self.assertEqual(css.count('mark.svg?site-release='), 1)
            self.assertIn('mark.svg?site-release='+result.stdout.strip(),
                          (site/'style.css').read_text())

    def test_unsupported_contexts_fail_before_any_writes(self):
        examples = [
            'const t = '+chr(96)+chr(36)+'{import("./lazy.mjs")}'+chr(96)+';',
            'function f() {} /import("ghost.js")/.test(s);',
            'import("./core\\u002emjs");',
            'obj.if() / import(".\\/core.mjs") / 2;',
            'obj?.if() / import(".\\/core.mjs") / 2;',

        ]
        for example in examples:
            with self.subTest(example=example), TemporaryDirectory() as tmp:
                root = Path(tmp); site = self.fixture(root)
                (site/'app.mjs').write_text(example)
                before = {p.name:p.read_bytes() for p in site.iterdir()}
                result = self.run_tool(site)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(before, {p.name:p.read_bytes() for p in site.iterdir()})

    def test_untracked_dependency_and_non_git_site_are_refused(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); site = self.fixture(root)
            (site/'new.mjs').write_text('export const fresh=1;')
            (site/'app.mjs').write_text('import "./new.mjs";')
            result = self.run_tool(site)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('must be tracked', result.stderr)
        with TemporaryDirectory() as tmp:
            site = Path(tmp)/'site'; site.mkdir()
            (site/'index.html').write_text('<p>Non-Git input</p>')
            result = self.run_tool(site)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Git worktree', result.stderr)

    def test_symlink_and_deleted_tracked_file_are_refused(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); site = self.fixture(root)
            (site/'mark.svg').unlink()
            result = self.run_tool(site)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('missing/outside local asset', result.stderr)
        with TemporaryDirectory() as tmp:
            root = Path(tmp); site = self.fixture(root)
            (site/'mark.svg').unlink()
            (site/'mark.svg').symlink_to(site/'proof.md')
            result = self.run_tool(site)
            self.assertNotEqual(result.returncode, 0)

    def test_crlf_survives_and_check_does_not_write(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); site = self.fixture(root)
            (site/'app.mjs').write_bytes(b'import "./core.mjs";\r\nexport const a=1;\r\n')
            result = self.run_tool(site)
            self.assertEqual(result.returncode, 0, result.stderr)
            text = (site/'app.mjs').read_bytes()
            self.assertEqual(text.count(b'\r\n'), 2)
            self.assertEqual(text.count(b'\n'), 2)
            self.assertEqual(self.run_tool(site, '--check').returncode, 0)


if __name__ == '__main__': unittest.main()
