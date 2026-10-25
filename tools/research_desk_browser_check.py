#!/usr/bin/env python3
"""Exercise Library discovery and the reader-owned list in a real browser.

Local mode reuses the committed hash-verified catalog shards; unavailable
unrelated remote sources stay explicit. No mathematical or spoken-AT verdict.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import partial
from hashlib import sha256
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import threading
import traceback
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
KEY = 'universal-law-reading-list/v1'

@contextmanager
def serve():
    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *args): pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Quiet, directory=str(ROOT / 'docs')))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try: yield f'http://127.0.0.1:{server.server_port}/site/'
    finally: server.shutdown(); server.server_close(); thread.join()

def run(origin, browser, out, report, local):
    from playwright.sync_api import expect
    config = json.loads((ROOT / 'docs/site/config.json').read_text())
    fixtures = {config['inventory']['url']: ROOT / 'docs/public-math/sources.json',
                config['status']['url']: ROOT / 'tests/fixtures/status_f2e432e.md'}
    fixtures.update({pin['url']: ROOT / pin['path'] for pin in config['inventory']['pages']})
    def fixture(route):
        path = fixtures.get(route.request.url)
        route.fulfill(status=200 if path else 503,
                      headers={'Access-Control-Allow-Origin': '*'},
                      body=path.read_bytes() if path else b'')
    def context(**kw):
        c = browser.new_context(accept_downloads=True, reduced_motion='reduce', **kw)
        if local: c.route('https://raw.githubusercontent.com/**', fixture)
        return c
    def page(c, suffix='workspace.html'):
        p = c.new_page(); p.set_default_timeout(20000)
        p.on('pageerror', lambda e: report['page_errors'].append(str(e)))
        p.goto(origin + suffix)
        return p
    def close(c):
        c.unroute_all(behavior='wait')
        c.close()
    def case(name, fn):
        row = {'case': name}
        report['cases'].append(row)
        try: fn(); row['passed'] = True
        except Exception:
            row['passed'] = False; row['error'] = traceback.format_exc(); raise

    for width, scheme in [(1440, 'light'), (390, 'light'), (320, 'dark')]:
        def journey():
            c = context(viewport={'width': width, 'height': 900}, color_scheme=scheme)
            p = page(c)
            expect(p.locator('#library-current-link')).to_be_visible()
            p.locator('#library-current-link').focus(); p.keyboard.press('Enter')
            expect(p.locator('.current-search')).to_be_visible()
            expect(p.locator('.current-search')).to_be_enabled()
            p.locator('.current-search').fill('PROOF.md')
            expect(p.locator('.current-results > li').first).to_be_visible()
            expect(p.locator('.desk-save-source').first).to_be_visible()
            source = p.locator('.current-results > li > a').first.get_attribute('href')
            p.locator('.desk-save-source').first.click()
            p.locator('#reading-list > summary').click()
            expect(p.locator('#desk-entries > li')).to_have_count(1)
            assert p.locator('#desk-entries a').first.get_attribute('href') == source
            p.locator('#desk-save-current').click()
            expect(p.locator('#desk-entries > li')).to_have_count(2)
            p.reload()
            expect(p.locator('.current-search')).to_be_visible()
            expect(p.locator('.current-search')).to_have_value('PROOF.md')
            p.locator('#library-current-link').focus(); p.keyboard.press('Enter')
            expect(p.locator('.current-search')).to_be_focused()
            expect(p.locator('#desk-count')).to_have_text('2 saved items')
            p.locator('#reading-list > summary').click()
            with p.expect_download() as pending: p.locator('#desk-export').click()
            download = pending.value; target = out / f'list-{width}-{scheme}.json'
            download.save_as(target)
            data = json.loads(target.read_text()); assert len(data['entries']) == 2
            assert data['schema'] == KEY
            p.locator('#desk-clear').click()
            expect(p.locator('#desk-entries > li')).to_have_count(0)
            p.locator('#desk-import').set_input_files(target)
            expect(p.locator('#desk-entries > li')).to_have_count(2)
            p.locator('#desk-import').set_input_files(target)
            expect(p.locator('#desk-entries > li')).to_have_count(2)
            p.locator('#desk-entries button').first.click()
            expect(p.locator('#desk-entries > li')).to_have_count(1)
            assert p.evaluate('document.documentElement.scrollWidth <= innerWidth')
            for item in p.locator('.library-collections a, #reading-list button').all():
                if item.is_visible():
                    b = item.bounding_box(); assert b['width'] >= 44 and b['height'] >= 44, b
            p.locator('#library-start').scroll_into_view_if_needed()
            p.screenshot(path=str(out / f'library-{width}-{scheme}.png'))
            p.locator('#reading-list').screenshot(path=str(out / f'list-{width}-{scheme}.png'))
            close(c)
        case(f'current-search-save-persist-export-import-{width}-{scheme}', journey)

    def catalog():
        c = context(viewport={'width': 390, 'height': 844})
        p = page(c, 'workspace.html?q=PROOF.md&repository=Math-&path=coefficients%2Fside24_v1%2F#inventory')
        expect(p.locator('#catalog > tr')).to_have_count(1)
        p.locator('#desk-save-catalog').click()
        p.locator('#reading-list > summary').click()
        href = p.locator('#desk-entries a').first.get_attribute('href')
        assert 'q=PROOF.md' in href and 'repository=Math-' in href and 'source-q' not in href
        expect(p.locator('#desk-entries')).to_contain_text('Dated catalog search')
        p.locator('#catalog .desk-save-source').click()
        expect(p.locator('#desk-entries > li')).to_have_count(2)
        close(c)
    case('dated-catalog-isolated-query-and-source', catalog)

    def invalid_import():
        c = context(); p = page(c)
        p.locator('#reading-list > summary').click()
        before = p.evaluate('(key)=>localStorage.getItem(key)', KEY)
        for name, content in [('malformed', b'{'), ('oversize', b' ' * 131073),
                              ('unsafe', json.dumps({'schema':KEY,'entries':[{'kind':'source','url':'javascript:alert(1)'}]}).encode())]:
            p.locator('#desk-import').set_input_files({'name':name+'.json','mimeType':'application/json','buffer':content})
            expect(p.locator('#desk-status')).to_contain_text('Import refused')
            assert p.evaluate('(key)=>localStorage.getItem(key)', KEY) == before
        close(c)
    case('invalid-imports-atomic', invalid_import)

    def storage_failure():
        c = context()
        c.add_init_script("Object.defineProperty(window, 'localStorage', {get(){throw new DOMException('Blocked','SecurityError')}})")
        p = page(c, 'workspace.html?source-q=PROOF.md#library-current')
        expect(p.locator('#desk-save-current')).to_be_enabled()
        p.locator('#desk-save-current').click()
        expect(p.locator('#desk-count')).to_have_text('1 saved item')
        expect(p.locator('#desk-status')).to_contain_text('this tab only')
        p.locator('#reading-list > summary').click()
        expect(p.locator('#desk-export')).to_be_enabled(); close(c)
    case('unavailable-storage-session-and-export', storage_failure)

    def refused():
        c = context()
        c.route('**/current.json', lambda route: route.fulfill(status=503, body=''))
        p = page(c, 'workspace.html?source-q=PROOF.md#library-current')
        expect(p.locator('.current-search')).to_be_visible()
        expect(p.locator('.current-state')).to_contain_text('Unavailable:')
        expect(p.locator('#desk-save-current')).to_be_disabled()
        assert p.locator('.current-results .desk-save-source').count() == 0
        close(c)
    case('refused-current-data-never-save-static-rows', refused)

    def no_js():
        c = context(java_script_enabled=False, viewport={'width':390,'height':844})
        p = page(c)
        expect(p.locator('#library-static-sources')).to_be_visible()
        assert p.locator('#library-static-sources').get_attribute('href') == 'nodes/index.html#sources'
        expect(p.locator('#desk-save-current')).to_be_disabled(); close(c)
    case('no-javascript-static-source-path', no_js)

    def focus_readiness():
        c=context(); held=[]
        c.route('**/current.json',lambda route:held.append(route))
        p=page(c)
        p.locator('#library-current-link').click()
        p.locator('#reading-list > summary').focus()
        assert held
        held[0].continue_()
        expect(p.locator('.current-search')).to_be_enabled()
        expect(p.locator('#reading-list > summary')).to_be_focused()
        close(c)
    case('late-readiness-preserves-newer-focus',focus_readiness)

    def storage_modes():
        for failure,script in [
            ('corrupt',"localStorage.setItem('universal-law-reading-list/v1','{corrupt')"),
            ('quota',"Storage.prototype.setItem=function(){throw new DOMException('Full','QuotaExceededError')}"),
            ('lock-refusal',"navigator.locks.request=function(){return Promise.reject(new DOMException('Denied','SecurityError'))}")]:
            c=context();c.add_init_script(script)
            p=page(c,'workspace.html?source-q=PROOF.md#library-current')
            expect(p.locator('#desk-save-current')).to_be_enabled()
            p.locator('#desk-save-current').click()
            expect(p.locator('#desk-count')).to_have_text('1 saved item')
            expect(p.locator('#desk-status')).to_contain_text('this tab only')
            if failure=='corrupt':assert p.evaluate('(key)=>localStorage.getItem(key)',KEY)=='{corrupt'
            p.locator('#reading-list > summary').click()
            expect(p.locator('#desk-export')).to_be_enabled();close(c)
    case('corrupt-quota-and-lock-refusal-preserve-session-use',storage_modes)

    def two_tabs():
        c=context();a=page(c,'workspace.html?source-q=PROOF.md#library-current');b=page(c,'workspace.html?source-q=README.md#library-current')
        expect(a.locator('#desk-save-current')).to_be_enabled();expect(b.locator('#desk-save-current')).to_be_enabled()
        a.locator('#desk-save-current').click();b.locator('#desk-save-current').click()
        expect(a.locator('#desk-count')).to_have_text('2 saved items');expect(b.locator('#desk-count')).to_have_text('2 saved items')
        stored=json.loads(a.evaluate('(key)=>localStorage.getItem(key)',KEY))
        assert {x['query'] for x in stored['entries']}=={'PROOF.md','README.md'}
        close(c)
    case('two-tabs-preserve-both-explicit-saves',two_tabs)

    def query_limit():
        c=context();p=page(c,'workspace.html#library-current')
        expect(p.locator('#desk-save-current')).to_be_enabled();p.locator('.current-search').fill('x'*501)
        p.locator('#desk-save-current').click()
        expect(p.locator('#desk-status')).to_contain_text('Search could not be saved')
        expect(p.locator('#desk-count')).to_have_text('0 saved items');close(c)
    case('oversize-query-refused-without-page-error',query_limit)

    def primary_library():
        c=context(viewport={'width':390,'height':844});p=page(c,'index.html')
        p.locator('nav[aria-label="Primary"] a',has_text='Library').click()
        expect(p.locator('#library-current-link')).to_be_in_viewport()
        assert urlsplit(p.url).fragment!='inventory'
        p.go_back();expect(p).to_have_url(origin+'index.html');close(c)
    case('home-library-opens-collection-choice',primary_library)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--browser', type=Path, default=Path('/opt/google/chrome/chrome'))
    parser.add_argument('--origin', help='Optional actual published site origin ending /site/; no local fixtures')
    args=parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    report={'started_utc':datetime.now(timezone.utc).isoformat(), 'cases':[], 'page_errors':[],
            'source':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'sandbox':True, 'browser_path':str(args.browser), 'browser_sha256':sha256(args.browser.read_bytes()).hexdigest(),
            'limits':'Headless Chromium/Linux; viewport emulation, not physical touch, Safari, spoken AT or scientific verification.'}
    from playwright.sync_api import sync_playwright
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(executable_path=str(args.browser),chromium_sandbox=True)
            report['browser_version']=browser.version
            if args.origin: report['origin']=args.origin; run(args.origin,browser,args.output,report,False)
            else:
                with serve() as origin: report['origin']=origin; run(origin,browser,args.output,report,True)
            browser.close()
        assert not report['page_errors'], report['page_errors']
        report['passed']=True
    except Exception: report['passed']=False; report['error']=traceback.format_exc()
    finally:
        report['finished_utc']=datetime.now(timezone.utc).isoformat()
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':report['passed'],'cases':len(report['cases']),'output':str(args.output)}))
    return 0 if report['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
