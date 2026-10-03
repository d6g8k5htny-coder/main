"""Release URLs must select fresh entry and transitive assets, without editing data."""
from html.parser import HTMLParser
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit, parse_qs
import subprocess
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

class AssetRelease(unittest.TestCase):
    def run_tool(self, site, *args):
        return subprocess.run([sys.executable, str(TOOL), '--site', str(site), *args], text=True, capture_output=True)

    def fixture(self, root):
        site = root/'site'; site.mkdir()
        (site/'index.html').write_text('<script type="module" src="app.mjs"></script><link rel="stylesheet" href="style.css"><a href="other.html#reading">Read</a>')
        (site/'app.mjs').write_text("import {x} from './core.mjs'; const later=()=>import('./lazy.mjs'); const external=()=>import('https://example.test/tool.mjs');")
        (site/'core.mjs').write_text('export const x=1;')
        (site/'lazy.mjs').write_text("import './core.mjs';")
        (site/'style.css').write_text('body{background:url("mark.svg")}')
        (site/'mark.svg').write_text('<svg/>')
        (site/'config.json').write_bytes(b'{"pin":"unchanged"}\n')
        (site/'proof.md').write_bytes(b'Exact research source.\n')
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

if __name__ == '__main__': unittest.main()
