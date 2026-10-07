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
    PRIVATE_NAMES = (
        'await', 'break', 'case', 'catch', 'class', 'const', 'continue',
        'debugger', 'default', 'delete', 'do', 'else', 'enum', 'export',
        'extends', 'false', 'finally', 'for', 'function', 'if', 'import',
        'in', 'instanceof', 'new', 'null', 'return', 'super', 'switch',
        'this', 'throw', 'true', 'try', 'typeof', 'var', 'void', 'while',
        'with', 'yield', 'let', 'static', 'implements', 'interface',
        'package', 'private', 'protected', 'public', 'as', 'from', 'of',
        'using', 'async', 'get', 'set', 'x', 'α', '$x', '_x',
    )
    LEXICAL_SLASH_CASES = (
        ('commonjs', 'var await = 1; await / import("core.mjs") / 2;'),
        ('commonjs', 'var yield = 1; yield / import("core.mjs") / 2;'),
        ('commonjs', 'for (var await = 1; await / import("core.mjs") / 2; ) {}'),
        ('commonjs', 'for (var yield = 1; yield / import("core.mjs") / 2; ) {}'),
        ('commonjs', 'function f() { var await = 1; await / import("core.mjs") / 2; }'),
        ('commonjs', 'function f() { var yield = 1; yield / import("core.mjs") / 2; }'),
        ('commonjs', 'var await = 1; await /= import("core.mjs");'),
        ('commonjs', 'var yield = 1; yield /= import("core.mjs");'),
        ('module', 'async function f() { return await /import("core.mjs")/; }'),
        ('module', 'function* f() { yield /import("core.mjs")/; }'),
        ('module', 'async function f() { return await /*gap*/ /import("core.mjs")/; }'),
        ('module', 'function* f() { yield /*gap*/ /import("core.mjs")/; }'),
        ('module', 'async function f() { return await //gap\n /import("core.mjs")/; }'),
        ('module', 'function* f() { yield //gap\n /import("core.mjs")/.test("x"); }'),
        ('module', 'async function f() { return `${await /import("core.mjs")/}`; }'),
        ('module', 'function* f() { return `${yield /import("core.mjs")/}`; }'),
    )

    def stage(self, root):
        subprocess.run(['git', '-C', str(root), 'add', '-A'], check=True,
                       capture_output=True)

    def run_tool(self, site, *args):
        flags = ['-B', '-S'] + (['-O'] if sys.flags.optimize else [])
        return subprocess.run([sys.executable, *flags, str(TOOL), '--site', str(site), *args], text=True, capture_output=True)

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

    def catalog_fixture(self, root):
        site = self.fixture(root)
        catalog = root/'public-math'; catalog.mkdir()
        (catalog/'catalog.json').write_bytes(b'{"catalog":"tracked"}\n')
        self.stage(root)
        return site, catalog

    def snapshot(self, root):
        files = {p.relative_to(root).as_posix(): p.read_bytes()
                 for p in root.rglob('*')
                 if '.git' not in p.relative_to(root).parts and p.is_file()}
        index = subprocess.run(
            ['git', '-C', str(root), 'ls-files', '--stage', '-z'],
            check=True, capture_output=True).stdout
        return files, index

    def test_inventory_refuses_missing_indexed_catalog_root(self):
        for replacement in (False, True):
            with self.subTest(replacement=replacement), TemporaryDirectory() as tmp:
                root = Path(tmp); site, catalog = self.catalog_fixture(root)
                (catalog/'catalog.json').unlink(); catalog.rmdir()
                if replacement: catalog.write_text('not a catalog directory')
                with self.assertRaisesRegex(ValueError, 'missing/outside local asset'):
                    asset_release.inventory(site)

    def test_missing_indexed_catalog_root_refuses_cli_before_writes(self):
        for replacement in (False, True):
            with self.subTest(replacement=replacement), TemporaryDirectory() as tmp:
                root = Path(tmp); site, catalog = self.catalog_fixture(root)
                baseline = self.run_tool(site)
                self.assertEqual(baseline.returncode, 0, baseline.stderr)
                (catalog/'catalog.json').unlink(); catalog.rmdir()
                if replacement: catalog.write_text('not a catalog directory')
                before = self.snapshot(root)
                for args in [('--check',), (), ('--check',)]:
                    with self.subTest(args=args):
                        result = self.run_tool(site, *args)
                        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                        self.assertEqual(result.stdout, '')
                        self.assertIn('missing/outside local asset in tracked inventory',
                                      result.stderr)
                        self.assertEqual(self.snapshot(root), before)

    def test_staged_catalog_removal_is_valid_and_changes_release(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); site, catalog = self.catalog_fixture(root)
            first = self.run_tool(site)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertIn(catalog/'catalog.json', asset_release.inventory(site))
            self.stage(root)
            subprocess.run(
                ['git', '-C', str(root), '-c', 'user.name=Asset release fixture',
                 '-c', 'user.email=fixture@example.invalid', 'commit', '--no-gpg-sign',
                 '-qm', 'Intact catalog baseline'], check=True, capture_output=True)
            (catalog/'catalog.json').unlink(); catalog.rmdir(); self.stage(root)
            before = self.snapshot(root)
            stale = self.run_tool(site, '--check')
            self.assertEqual(stale.returncode, 1, stale.stderr)
            self.assertEqual(stale.stderr, '')
            self.assertEqual(self.snapshot(root), before)
            second = self.run_tool(site)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertNotEqual(second.stdout, first.stdout)
            self.assertTrue(all(p.is_relative_to(site) for p in asset_release.inventory(site)))
            self.assertEqual(self.run_tool(site, '--check').returncode, 0)

    def test_optional_catalog_siblings_preserve_inventory_and_release(self):
        for kind in ('absent', 'empty', 'untracked-file', 'tracked-file', 'broken-symlink'):
            with self.subTest(kind=kind), TemporaryDirectory() as tmp:
                root = Path(tmp); site = self.fixture(root)
                first = self.run_tool(site)
                self.assertEqual(first.returncode, 0, first.stderr)
                files = asset_release.inventory(site)
                catalog = root/'public-math'
                if kind == 'empty': catalog.mkdir()
                if kind in ('untracked-file', 'tracked-file'): catalog.write_text('ordinary sibling')
                if kind == 'broken-symlink': catalog.symlink_to(root/'missing', target_is_directory=True)
                if kind in ('tracked-file', 'broken-symlink'): self.stage(root)
                self.assertEqual(asset_release.inventory(site), files)
                before = self.snapshot(root)
                for args in [(), ('--check',)]:
                    result = self.run_tool(site, *args)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout, first.stdout)
                    self.assertEqual(self.snapshot(root), before)

    def test_git_root_site_preserves_outside_catalog_boundary(self):
        for kind in ('absent', 'file', 'directory'):
            with self.subTest(kind=kind), TemporaryDirectory() as tmp:
                outer = Path(tmp); repo = outer/'repo'; repo.mkdir()
                site = self.fixture(repo)
                for path in site.iterdir(): path.rename(repo/path.name)
                site.rmdir(); self.stage(repo)
                catalog = outer/'public-math'
                if kind == 'file': catalog.write_text('ordinary sibling')
                if kind == 'directory': catalog.mkdir()
                before = self.snapshot(repo)
                result = self.run_tool(repo)
                if kind == 'directory':
                    self.assertEqual(result.returncode, 2)
                    self.assertIn('site/catalog must remain inside', result.stderr)
                    self.assertEqual(self.snapshot(repo), before)
                else:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(self.run_tool(repo, '--check').returncode, 0)

    def test_catalog_existing_missing_file_and_directory_symlink_refusals(self):
        for kind in ('missing-file', 'inside-symlink', 'outside-symlink'):
            with self.subTest(kind=kind), TemporaryDirectory() as tmp:
                outer = Path(tmp); root = outer/'repo'; root.mkdir()
                site, catalog = self.catalog_fixture(root)
                if kind == 'missing-file':
                    (catalog/'catalog.json').unlink()
                else:
                    target = (root if kind == 'inside-symlink' else outer)/'target'
                    catalog.rename(target); catalog.symlink_to(target, target_is_directory=True)
                    self.stage(root)
                before = self.snapshot(root)
                for args in [('--check',), ()]:
                    result = self.run_tool(site, *args)
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertIn('missing/outside local asset', result.stderr)
                    self.assertEqual(self.snapshot(root), before)

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

    def test_iteration_header_regex_has_no_module_spans(self):
        pattern = '/import("ghost.mjs?site-release=keep")/'
        examples = [
            f'for (const x of {pattern}) {{}}',
            f'for (let x of {pattern}) {{}}',
            f'for (var x of {pattern}) {{}}',
            f'for (x of {pattern}) {{}}',
            f'for (of of {pattern}) {{}}',
            f'for (const of of {pattern}) {{}}',
            f'for (const [x, {{y}}] of {pattern}) {{}}',
            f'for ({{x}} of {pattern}) {{}}',
            f'for (obj.of of {pattern}) {{}}',
            f'for (obj.in of {pattern}) {{}}',
            f'for (obj.for of {pattern}) {{}}',
            f'for (obj.return of {pattern}) {{}}',
            f'for (obj.let of {pattern}) {{}}',
            f'for (obj[key] of {pattern}) {{}}',
            f'for ((x) of {pattern}) {{}}',
            f'for /*a*/ (const x /*b*/ of //c\n {pattern}) {{}}',
            f'for await (const x of {pattern}) {{}}',
            f'for /*a*/ await /*b*/ (const x of /*c*/ {pattern}) {{}}',
            f'for (const x of xs) for (const y of {pattern}) {{}}',
            f'for (let x = (() => {{ for (const y of {pattern}) {{}} }})();; ) {{}}',
            f'for (const x of ({pattern})) {{}}',
            'for (const x of /[(){}]import("ghost.mjs")/g) {}',
        ]
        for source in examples:
            with self.subTest(source=source):
                self.assertEqual(asset_release.source_spans(source, '.mjs'), [])
                self.assertEqual(asset_release.transform(
                    source, '.mjs', asset_release.without_release), source)

    def test_iteration_regex_cli_preserves_tracked_and_missing_pattern_targets(self):
        for header in ('for', 'for await'):
            for tracked in (False, True):
                with self.subTest(header=header, tracked=tracked), TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    subprocess.run(['git', 'init', '-q', str(root)], check=True,
                                   capture_output=True)
                    site = root/'site'; site.mkdir()
                    source = f'{header} (const x of /import("core.mjs")/) {{}}\n'
                    (site/'app.mjs').write_text(source)
                    if tracked: (site/'core.mjs').write_text('export const x = 1;\n')
                    self.stage(root)
                    before = self.snapshot(root)
                    token = None
                    for args in [('--check',), (), ('--check',), ()]:
                        result = self.run_tool(site, *args)
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        self.assertEqual(result.stderr, '')
                        self.assertRegex(result.stdout, r'^[0-9a-f]{64}\n$')
                        if token is None: token = result.stdout
                        self.assertEqual(result.stdout, token)
                        self.assertEqual(self.snapshot(root), before)

    def test_identifier_of_division_still_discovers_real_imports(self):
        live = 'import("./core.mjs")'
        examples = [
            f'const of = 2; of / {live} / 2;',
            f'obj.of / {live} / 2; obj?.of / {live} / 2;',
            f'for (of / {live} / 2; ; ) {{}}',
            f'for (let x = of / {live} / 2; ; ) {{}}',
            f'for (let of = 2; of / {live} / 2; of / {live} / 2) {{}}',
            f'for (x in of / {live} / 2) {{}}',
            f'for (x of of / {live} / 2) {{}}',
            f'for await (x of of / {live} / 2) {{}}',
            f'for (x of obj.of / {live} / 2) {{}}',
            f'for (x of obj.in / {live} / 2) {{}}',
            f'for (x of obj?.of / {live} / 2) {{}}',
            f'for (x of obj[of / {live} / 2]) {{}}',
            f'for (let x = {{of: of / {live} / 2}}; ; ) {{}}',
            f'for (let x = (of / {live} / 2); ; ) {{}}',
            f'for (let x = [of / {live} / 2]; ; ) {{}}',
            f'for (let x = of; x; x += of / {live} / 2) {{}}',
            f'for (let x = typeof of / {live} / 2; ; ) {{}}',
            f'obj.for(of / {live} / 2); obj?.for(of / {live} / 2);',
            f'obj.for(ok) / scale / {live}; obj?.for(ok) / scale / {live};',
            f'for (x of xs) {{}} of / {live} / 2;',
            f'for (obj[of / {live} / 2] of values) {{}}',
            f'for (const [x = of / {live} / 2] of values) {{}}',
        ]
        for source in examples:
            with self.subTest(source=source):
                spans = asset_release.source_spans(source, '.mjs')
                self.assertEqual([source[a:b] for a, b in spans],
                                 ['./core.mjs'] * source.count(live))
                self.assertEqual(asset_release.transform(
                    source, '.mjs', lambda value: value + '?site-release=new'),
                    source.replace('./core.mjs', './core.mjs?site-release=new'))

    def test_iteration_regex_and_real_imports_have_distinct_cli_behavior(self):
        pattern = '/import("ghost.mjs?site-release=keep")/'
        source = (f'for (const x of {pattern}) {{ import("./core.mjs"); }}\n'
                  'for (x of of / import("./core.mjs") / 2) {}\n')
        with TemporaryDirectory() as tmp:
            root = Path(tmp); site = self.fixture(root)
            (site/'app.mjs').write_text(source)
            first = self.run_tool(site)
            self.assertEqual(first.returncode, 0, first.stderr)
            expected = source.replace('./core.mjs', './core.mjs?site-release=' + first.stdout.strip())
            self.assertEqual((site/'app.mjs').read_text(), expected)
            before = self.snapshot(root)
            for args in [('--check',), ()]:
                result = self.run_tool(site, *args)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, first.stdout)
                self.assertEqual(self.snapshot(root), before)

    def test_missing_iteration_division_import_refuses_before_writes(self):
        source = ('for (const x of /import("core.mjs")/) {}\n'
                  'for (x of of / import("./missing.mjs") / 2) {}\n')
        with TemporaryDirectory() as tmp:
            root = Path(tmp); site = self.fixture(root)
            (site/'app.mjs').write_text(source)
            before = self.snapshot(root)
            for args in [('--check',), ()]:
                result = self.run_tool(site, *args)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertEqual(result.stdout, '')
                self.assertIn('missing/outside local asset ./missing.mjs', result.stderr)
                self.assertEqual(self.snapshot(root), before)

    def test_iteration_regex_refusals_preserve_all_files_and_index(self):
        examples = [
            ('for (const x of /import("core.mjs")\n) {}',
             'Unsupported or unterminated regex'),
            ('for await (const x of /import("core.mjs")\n) {}',
             'Unsupported or unterminated regex'),
            ('for (const \\u0078 of /import("core.mjs")/) {}',
             'Unsupported JavaScript identifier syntax'),
            ('for (const x of xs) {} /import("core.mjs")/.test(s);',
             'Ambiguous regex/division containing module syntax'),
            ('for (obj[of / import("./core.mjs") / 2] of /import("ghost.mjs")/) {}',
             'Ambiguous regex/division containing module syntax'),
            ('for (const [x = of / import("./core.mjs") / 2] of /import("ghost.mjs")/) {}',
             'Ambiguous regex/division containing module syntax'),
        ]
        for source, message in examples:
            with self.subTest(source=source), TemporaryDirectory() as tmp:
                root = Path(tmp); site = self.fixture(root)
                (site/'app.mjs').write_text('import "./core.mjs";\n' + source)
                before = self.snapshot(root)
                for args in [('--check',), ()]:
                    result = self.run_tool(site, *args)
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertEqual(result.stdout, '')
                    self.assertIn(message, result.stderr)
                    self.assertEqual(self.snapshot(root), before)

    def test_private_keyword_targets_preserve_iteration_regex(self):
        pattern = '/import("ghost.mjs?site-release=keep")/'
        for name in self.PRIVATE_NAMES:
            for header in ('for', 'for /*a*/ await /*b*/'):
                for target in (f'this.#{name}', f'(this.#{name})',
                               f'this /*c*/ . /*d*/ #{name}',
                               f'this[#{name} in this]'):
                    source = (f'class C {{ #{name}; async run() {{ '
                              f'{header} ({target} of {pattern}) {{}} }} }}')
                    with self.subTest(name=name, header=header, target=target):
                        try:
                            spans = asset_release.source_spans(source, '.mjs')
                        except ValueError as error:
                            self.fail(f'Private target was not recognized: {error}')
                        self.assertEqual(spans, [])
                        self.assertEqual(asset_release.transform(
                            source, '.mjs', asset_release.without_release), source)

    def test_private_keyword_divisions_discover_genuine_imports(self):
        live = 'import("core.mjs")'
        for name in self.PRIVATE_NAMES:
            source = (f'class C {{ #{name}; run(obj) {{\n'
                      f'this.#{name} / {live} / 2;\n'
                      f'this?.#{name} / {live} / 2;\n'
                      f'for (let x = this.#{name} / {live} / 2; ; ) {{}}\n'
                      f'for (x of this.#{name} / {live} / 2) {{}}\n'
                      f'for (x in this.#{name} / {live} / 2) {{}}\n'
                      f'for (let x = (#{name} in obj); x; ) {{ {live}; }}\n'
                      '} }')
            with self.subTest(name=name):
                try:
                    spans = asset_release.source_spans(source, '.mjs')
                except ValueError as error:
                    self.fail(f'Private division was not recognized: {error}')
                self.assertEqual([source[a:b] for a, b in spans], ['core.mjs'] * 6)
                self.assertEqual(asset_release.transform(
                    source, '.mjs', lambda value: value + '?site-release=new'),
                    source.replace('core.mjs', 'core.mjs?site-release=new'))

    def test_private_import_calls_are_not_module_syntax(self):
        source = ('class C { #import() {} #if() {} run() {\n'
                  'this.#import("ghost.mjs"); this?.#import("ghost.mjs");\n'
                  'const t = `${this.#import("ghost.mjs")}`;\n'
                  'this.#if(ok) / scale / import("core.mjs");\n'
                  'import("core.mjs");\n'
                  '} }')
        try:
            spans = asset_release.source_spans(source, '.mjs')
        except ValueError as error:
            self.fail(f'Private call was treated as a module import: {error}')
        self.assertEqual([source[a:b] for a, b in spans], ['core.mjs'] * 2)
        self.assertEqual(asset_release.transform(
            source, '.mjs', lambda value: value + '?site-release=new'),
            source.replace('core.mjs', 'core.mjs?site-release=new'))
        genuine_template = source.replace('${this.#import(', '${import(')
        with self.assertRaisesRegex(ValueError, 'Import in template expression'):
            asset_release.source_spans(genuine_template, '.mjs')

    def test_private_regex_cli_preserves_pattern_bytes_and_raw_index(self):
        for name in ('in', 'return', 'const', 'of', 'x'):
            for tracked in (False, True):
                with self.subTest(name=name, tracked=tracked), TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    subprocess.run(['git', 'init', '-q', str(root)], check=True,
                                   capture_output=True)
                    site = root/'site'; site.mkdir()
                    source = (f'class C {{ #{name}; run() {{ for (this.#{name} '
                              'of /import("core.mjs")/) {} } }\n')
                    (site/'app.mjs').write_text(source)
                    if tracked: (site/'core.mjs').write_text('export const x = 1;\n')
                    self.stage(root)
                    before = self.snapshot(root)
                    index = (root/'.git/index').read_bytes()
                    token = None
                    for args in [('--check',), (), ('--check',), ()]:
                        result = self.run_tool(site, *args)
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        self.assertEqual(result.stderr, '')
                        self.assertRegex(result.stdout, r'^[0-9a-f]{64}\n$')
                        if token is None: token = result.stdout
                        self.assertEqual(result.stdout, token)
                        self.assertEqual(self.snapshot(root), before)
                        self.assertEqual((root/'.git/index').read_bytes(), index)

    def test_private_calls_and_division_have_distinct_cli_dependencies(self):
        for missing in (False, True):
            with self.subTest(missing=missing), TemporaryDirectory() as tmp:
                root = Path(tmp); site = self.fixture(root)
                target = 'missing.mjs' if missing else 'core.mjs'
                source = ('class C { #import() {} #return; run() {\n'
                          'this.#import("ghost.mjs");\n'
                          f'this.#return / import("{target}") / 2;\n'
                          '} }\n')
                (site/'app.mjs').write_text(source)
                before = self.snapshot(root)
                index = (root/'.git/index').read_bytes()
                result = self.run_tool(site)
                if missing:
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertEqual(result.stdout, '')
                    self.assertIn('missing/outside local asset missing.mjs', result.stderr)
                    self.assertEqual(self.snapshot(root), before)
                else:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    expected = source.replace('core.mjs', 'core.mjs?site-release=' + result.stdout.strip())
                    self.assertEqual((site/'app.mjs').read_text(), expected)
                    after = self.snapshot(root)
                    for args in [('--check',), ()]:
                        again = self.run_tool(site, *args)
                        self.assertEqual(again.returncode, 0, again.stderr)
                        self.assertEqual(again.stdout, result.stdout)
                        self.assertEqual(self.snapshot(root), after)
                self.assertEqual((root/'.git/index').read_bytes(), index)

    def test_unsupported_private_syntax_refuses_before_any_writes(self):
        examples = [
            'class C { #\\u0069n; run() { return this.#\\u0069n; } }',
            'class C { #; }',
            'class C { #/*comment*/in; }',
            '#!/usr/bin/env node\nimport("./core.mjs");',
        ]
        for source in examples:
            with self.subTest(source=source), TemporaryDirectory() as tmp:
                root = Path(tmp); site = self.fixture(root)
                (site/'app.mjs').write_text(source)
                before = self.snapshot(root)
                index = (root/'.git/index').read_bytes()
                for args in [('--check',), ()]:
                    result = self.run_tool(site, *args)
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertEqual(result.stdout, '')
                    self.assertIn('Unsupported JavaScript identifier syntax', result.stderr)
                    self.assertEqual(self.snapshot(root), before)
                    self.assertEqual((root/'.git/index').read_bytes(), index)

    def test_declaration_position_preserves_contextual_binding_regex(self):
        pattern = '/import("ghost.mjs?site-release=keep")/.source'
        module_bindings = [f'{decl} {name}' for decl in ('var', 'let', 'const')
                           for name in ('using', 'of', 'x', 'async')]
        module_bindings += ['using using', 'using x', 'await using using',
                            'await using of', 'await using x', 'using']
        cases = [('.mjs', f'{header} ({binding} of {pattern}) {{}}')
                 for header in ('for', 'for /*a*/ await /*b*/')
                 for binding in module_bindings]
        cases += [('.js', f'for ({decl} {name} of {pattern}) {{}}')
                  for decl in ('var', 'let', 'const') for name in ('await', 'yield')]
        cases += [('.js', f'for (var let of {pattern}) {{}}'),
                  ('.js', f'for ((await) of {pattern}) {{}}'),
                  ('.js', f'for ((yield) of {pattern}) {{}}'),
                  ('.mjs', f'for (const /*c*/ using /*d*/ of {pattern}) {{}}'),
                  ('.mjs', f'for (const\nusing of\n{pattern}) {{}}'),
                  ('.mjs', f'for (const x of xs) for (const using of {pattern}) {{}}'),
                  ('.mjs', f'for (const [using] of {pattern}) {{}}'),
                  ('.mjs', f'for (const {{using}} of {pattern}) {{}}')]
        for suffix, source in cases:
            with self.subTest(suffix=suffix, source=source):
                self.assertEqual(asset_release.source_spans(source, suffix), [])
                self.assertEqual(asset_release.transform(
                    source, suffix, asset_release.without_release), source)

    def test_declaration_prefix_does_not_hide_genuine_division_imports(self):
        live = 'import("core.mjs")'
        cases = [
            ('.mjs', f'for (const using = 1; using / {live} / 2; ) {{}}'),
            ('.mjs', f'for (let using = of / {live} / 2; false; ) {{}}'),
            ('.mjs', f'for (let x = using, of = 2; of / {live} / 2; ) {{}}'),
            ('.mjs', f'for (let x = (of / {live} / 2); false; ) {{}}'),
            ('.mjs', f'for (let x = {{using: of / {live} / 2}}; false; ) {{}}'),
            ('.mjs', f'for (using / {live} / 2; false; ) {{}}'),
            ('.mjs', f'for (using of of / {live} / 2) {{}}'),
            ('.mjs', f'for (using using of of / {live} / 2) {{}}'),
            ('.mjs', f'for (await using of of of / {live} / 2) {{}}'),
            ('.mjs', f'for (const using of {live}) {{}}'),
            ('.mjs', f'for (await using of of {live}) {{}}'),
            ('.mjs', f'async function f() {{ for (let x = await of / {live} / 2; false; ) {{}} }}'),
            ('.mjs', f'function* f() {{ for (let x = yield of / {live} / 2; false; ) {{}} }}'),
            ('.mjs', f'async function f() {{ for (let x = await using / {live} / 2; false; ) {{}} }}'),
            ('.js', f'for (var await = of / {live} / 2; false; ) {{}}'),
            ('.js', f'for (var yield = of / {live} / 2; false; ) {{}}'),
        ]
        for suffix, source in cases:
            with self.subTest(suffix=suffix, source=source):
                spans = asset_release.source_spans(source, suffix)
                self.assertEqual([source[a:b] for a, b in spans], ['core.mjs'])
                self.assertEqual(asset_release.transform(
                    source, suffix, lambda value: value + '?site-release=new'),
                    source.replace('core.mjs', 'core.mjs?site-release=new'))

    def test_bare_await_yield_headers_refuse_both_valid_interpretations(self):
        cases = [
            ('.js', 'for (await of /import("core.mjs")/) {}'),
            ('.js', 'for (yield of /import("core.mjs")/) {}'),
            ('.js', 'for (await of values) {}'),
            ('.js', 'for (yield of values) {}'),
            ('.mjs', 'async function f() { for (await of / import("core.mjs") / 2; false; ) {} }'),
            ('.mjs', 'function* f() { for (yield of / import("core.mjs") / 2; false; ) {} }'),
        ]
        for suffix, source in cases:
            with self.subTest(suffix=suffix, source=source):
                with self.assertRaisesRegex(ValueError, 'Ambiguous await/yield for header'):
                    asset_release.source_spans(source, suffix)

    def test_binding_position_cli_preserves_tracked_and_missing_regex_targets(self):
        cases = [
            ('.mjs', 'for (const using of /import("core.mjs")/) {}\n'),
            ('.mjs', 'for await (let using of /import("core.mjs")/) {}\n'),
            ('.mjs', 'for (using of /import("core.mjs")/) {}\n'),
            ('.mjs', 'for (await using of of /import("core.mjs")/) {}\n'),
            ('.js', 'for (var await of /import("core.mjs")/) {}\n'),
        ]
        for suffix, source in cases:
            for tracked in (False, True):
                with self.subTest(source=source, tracked=tracked), TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    subprocess.run(['git', 'init', '-q', str(root)], check=True,
                                   capture_output=True)
                    site = root/'site'; site.mkdir()
                    (site/('app' + suffix)).write_text(source)
                    if tracked: (site/'core.mjs').write_text('export const x = 1;\n')
                    self.stage(root)
                    before = self.snapshot(root); index = (root/'.git/index').read_bytes()
                    token = None
                    for args in [('--check',), (), ('--check',), ()]:
                        result = self.run_tool(site, *args)
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        self.assertEqual(result.stderr, '')
                        self.assertRegex(result.stdout, r'^[0-9a-f]{64}\n$')
                        if token is None: token = result.stdout
                        self.assertEqual(result.stdout, token)
                        self.assertEqual(self.snapshot(root), before)
                        self.assertEqual((root/'.git/index').read_bytes(), index)

    def test_ambiguous_binding_cli_refuses_before_all_file_and_index_writes(self):
        cases = [
            ('.js', 'for (await of /import("core.mjs")/) {}'),
            ('.js', 'for (yield of /import("missing.mjs")/) {}'),
            ('.mjs', 'async function f() { for (await of / import("core.mjs") / 2; false; ) {} }'),
            ('.mjs', 'function* f() { for (yield of / import("core.mjs") / 2; false; ) {} }'),
        ]
        for suffix, source in cases:
            with self.subTest(source=source), TemporaryDirectory() as tmp:
                root = Path(tmp); site = self.fixture(root)
                # Earlier files have genuine pending updates when the final file refuses.
                (site/('z-ambiguous' + suffix)).write_text(source)
                self.stage(root)
                before = self.snapshot(root); index = (root/'.git/index').read_bytes()
                for args in [('--check',), ()]:
                    result = self.run_tool(site, *args)
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertEqual(result.stdout, '')
                    self.assertIn('Ambiguous await/yield for header', result.stderr)
                    self.assertEqual(self.snapshot(root), before)
                    self.assertEqual((root/'.git/index').read_bytes(), index)

    def test_declared_binding_regex_and_real_import_cli_remain_distinct(self):
        for missing in (False, True):
            with self.subTest(missing=missing), TemporaryDirectory() as tmp:
                root = Path(tmp); site = self.fixture(root)
                target = 'missing.mjs' if missing else 'core.mjs'
                source = ('for (const using of /import("ghost.mjs")/.source) { '
                          f'import("{target}"); }}\n')
                (site/'app.mjs').write_text(source)
                before = self.snapshot(root); index = (root/'.git/index').read_bytes()
                result = self.run_tool(site)
                if missing:
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertEqual(result.stdout, '')
                    self.assertIn('missing/outside local asset missing.mjs', result.stderr)
                    self.assertEqual(self.snapshot(root), before)
                else:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual((site/'app.mjs').read_text(), source.replace(
                        'core.mjs', 'core.mjs?site-release=' + result.stdout.strip()))
                    after = self.snapshot(root)
                    for args in [('--check',), ()]:
                        again = self.run_tool(site, *args)
                        self.assertEqual(again.returncode, 0, again.stderr)
                        self.assertEqual(again.stdout, result.stdout)
                        self.assertEqual(self.snapshot(root), after)
                self.assertEqual((root/'.git/index').read_bytes(), index)

    def test_unqualified_await_yield_slash_refuses_without_mode_inference(self):
        for mode, source in self.LEXICAL_SLASH_CASES:
            # The checked fixture mode is evidence, not an input to the scanner.
            for suffix in ('.js', '.mjs'):
                with self.subTest(mode=mode, suffix=suffix, source=source):
                    with self.assertRaisesRegex(ValueError, 'Ambiguous await/yield regex/division'):
                        asset_release.source_spans(source, suffix)

    def test_lexical_goal_refusal_preserves_pending_files_and_raw_index(self):
        for mode, source in self.LEXICAL_SLASH_CASES:
            with self.subTest(mode=mode, source=source), TemporaryDirectory() as tmp:
                root = Path(tmp); site = self.fixture(root)
                suffix = '.js' if mode == 'commonjs' else '.mjs'
                (site/('z-ambiguous' + suffix)).write_text(source)
                self.stage(root)
                before = self.snapshot(root); index = (root/'.git/index').read_bytes()
                for args in [('--check',), ()]:
                    result = self.run_tool(site, *args)
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertEqual(result.stdout, '')
                    self.assertIn('Ambiguous await/yield regex/division', result.stderr)
                    self.assertEqual(self.snapshot(root), before)
                    self.assertEqual((root/'.git/index').read_bytes(), index)

    def test_unambiguous_await_yield_and_opaque_words_keep_module_spans(self):
        cases = [
            ('async function f() { return await import("core.mjs"); }', ['core.mjs']),
            ('function* f() { yield import("core.mjs"); }', ['core.mjs']),
            ('async function f() { return await (/import("ghost.mjs")/); }', []),
            ('function* f() { yield (/import("ghost.mjs")/); yield* /import("ghost.mjs")/.source; }', []),
            ('async function f() { return `${await (/import("ghost.mjs")/)}`; }', []),
            ('obj.await / import("core.mjs") / 2;\nobj?.yield / import("core.mjs") / 2;',
             ['core.mjs', 'core.mjs']),
            ('class C { #await; #yield; f() {\nthis.#await / import("core.mjs") / 2;\n'
             'this?.#yield / import("core.mjs") / 2;\n} }', ['core.mjs', 'core.mjs']),
            ('// await / import("ghost.mjs") /\n/* yield / import("ghost.mjs") / */\n'
             'const a = \'await / import("ghost.mjs") /\';\n'
             'const b = `yield / import("ghost.mjs") /`;\n'
             'const c = /await[/]import("ghost.mjs")/; import("core.mjs");', ['core.mjs']),
        ]
        for source, expected in cases:
            with self.subTest(source=source):
                spans = asset_release.source_spans(source, '.mjs')
                self.assertEqual([source[a:b] for a, b in spans], expected)

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
