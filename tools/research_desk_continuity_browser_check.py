#!/usr/bin/env python3
"""Real two-tab reading continuity; no mathematical or spoken-AT verdict."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import re
from pathlib import Path
import subprocess
import traceback
from research_desk_browser_check import ROOT, KEY, serve


def run(origin, browser, out, report, local):
    from playwright.sync_api import expect
    config = json.loads((ROOT / 'docs/site/config.json').read_text())
    fixtures = {config['inventory']['url']: ROOT / 'docs/public-math/sources.json',
                config['status']['url']: ROOT / 'tests/fixtures/status_f2e432e.md'}
    fixtures.update({pin['url']: ROOT / pin['path'] for pin in config['inventory']['pages']})

    def focused(control):
        # Data/count assertions above synchronize the synchronous render.
        assert control.evaluate('(e)=>e===document.activeElement'), 'Focused control was lost'

    def case(name, fn):
        row = {'case': name}; report['cases'].append(row)
        try: fn(); row['passed'] = True
        except Exception: row['passed'] = False; row['error'] = traceback.format_exc()
        print(json.dumps({'case': name, 'passed': row['passed']}), flush=True)

    for width, scheme in [(1440, 'light'), (390, 'dark'), (320, 'light')]:
        c = browser.new_context(viewport={'width': width, 'height': 900},
                                color_scheme=scheme, reduced_motion='reduce')
        if local:
            def fixture(route):
                path = fixtures.get(route.request.url)
                route.fulfill(status=200 if path else 503,
                              headers={'Access-Control-Allow-Origin': '*'},
                              body=path.read_bytes() if path else b'')
            c.route('https://raw.githubusercontent.com/**', fixture)
        a, b = c.new_page(), c.new_page()
        for p in (a, b):
            p.set_default_timeout(10000)
            p.on('pageerror', lambda error: report['page_errors'].append(str(error)))
        a.goto(origin + 'workspace.html?source-q=PROOF.md#library-current')
        expect(a.locator('#desk-save-current')).to_be_enabled()
        a.locator('#desk-save-current').click()  # Earlier query precedes target source.
        a.locator('.current-results .desk-save-source').first.click()
        expect(a.locator('#desk-count')).to_have_text('2 saved items')
        a.locator('#reading-list > summary').click()
        b.goto(origin + 'workspace.html?source-q=README.md#library-current')
        expect(b.locator('#desk-save-current')).to_be_enabled()
        b.locator('#reading-list > summary').click()
        source = a.locator('#desk-entries > li').nth(1)

        def add_peer():
            count = int(a.locator('#desk-count').inner_text().split()[0])
            b.locator('.current-search').fill('continuity ' + str(count))
            b.locator('#desk-save-current').click()
            expect(a.locator('#desk-count')).to_have_text(f'{count + 1} saved item' + ('s' if count else ''))

        for role, selector in [('source', ':scope > div > a:first-of-type'),
                               ('reference', '.desk-cite-source'),
                               ('identity', 'details > summary'), ('remove', '.desk-remove')]:
            def peer_add(selector=selector):
                detail = source.locator('details')
                if not detail.evaluate('(e)=>e.open'): detail.locator('summary').click()
                control = source.locator(selector); control.focus()
                add_peer()
                focused(control)
                assert detail.evaluate('(e)=>e.open'), 'Open identity collapsed'
            case(f'peer-add-retains-{role}-{width}-{scheme}', peer_add)

        def saved_search():
            control = a.locator('#desk-entries > li').first.locator('a').first
            control.focus(); add_peer(); focused(control)
        case(f'peer-add-retains-search-{width}-{scheme}', saved_search)

        def closed():
            detail = source.locator('details')
            if detail.evaluate('(e)=>e.open'): detail.locator('summary').click()
            control = source.locator('.desk-cite-source'); control.focus(); add_peer()
            focused(control)
            assert not detail.evaluate('(e)=>e.open'), 'Closed identity reopened'
        case(f'closed-identity-stays-closed-{width}-{scheme}', closed)

        def outside():
            control = a.locator('#library-current-link'); control.focus()
            before = a.evaluate('scrollY'); add_peer(); focused(control)
            assert a.evaluate('scrollY') == before, 'Background update moved scroll'
        case(f'outside-focus-and-scroll-{width}-{scheme}', outside)

        def earlier_removal():
            control = source.locator('.desk-cite-source'); control.focus()
            b.locator('#desk-entries > li').first.locator('.desk-remove').click()
            focused(a.locator('#desk-entries > li').first.locator('.desk-cite-source'))
        case(f'earlier-removal-retains-literal-source-{width}-{scheme}', earlier_removal)
        source = a.locator('#desk-entries > li').first

        def enrichment():
            detail = source.locator('details')
            if not detail.evaluate('(e)=>e.open'): detail.locator('summary').click()
            control = source.locator('.desk-cite-source'); control.focus()
            entries = json.loads(b.evaluate('(key)=>localStorage.getItem(key)', KEY))
            entry = next(e for e in entries['entries'] if e['kind'] == 'source')
            entry['sha256'] = 'a' * 64  # Fixture metadata, never authenticated source bytes.
            payload = json.dumps({'schema': KEY, 'entries': [entry]})
            b.locator('#desk-import').set_input_files({'name': 'enrich.json',
                'mimeType': 'application/json', 'buffer': payload.encode()})
            expect(control).to_have_attribute('href', re.compile('sha256=' + 'a' * 64))
            focused(control)
            expect(detail).to_contain_text('SHA-256')
            expect(detail).to_contain_text('a' * 64)
            assert detail.evaluate('(e)=>e.open')
            a.locator('#reading-list').screenshot(path=str(out / f'identity-{width}-{scheme}.png'))
        case(f'validated-enrichment-refreshes-reference-{width}-{scheme}', enrichment)

        def duplicate_import():
            control = source.locator('.desk-cite-source'); control.focus()
            payload = a.evaluate('(key)=>localStorage.getItem(key)', KEY)
            a.locator('#desk-import').set_input_files({'name': 'duplicate.json',
                'mimeType': 'application/json', 'buffer': payload.encode()})
            expect(a.locator('#desk-message')).to_contain_text('Reading list imported')
            focused(control)
            assert source.locator('details').evaluate('(e)=>e.open')
        case(f'duplicate-import-keeps-reading-state-{width}-{scheme}', duplicate_import)

        def removed():
            source.locator('.desk-cite-source').focus()
            b.locator('#desk-entries > li').first.locator('.desk-remove').click()
            expect(a.locator('#desk-entries .desk-cite-source')).to_have_count(0)
            focused(a.locator('#reading-list > summary'))
        case(f'peer-removal-focus-fallback-{width}-{scheme}', removed)

        def peer_clear():
            a.locator('#desk-entries > li').first.locator('a').first.focus()
            b.locator('#desk-clear').click()
            expect(a.locator('#desk-count')).to_have_text('0 saved items')
            focused(a.locator('#reading-list > summary'))
        case(f'peer-clear-focus-fallback-{width}-{scheme}', peer_clear)

        def local_remove():
            a.locator('#desk-save-current').click()
            expect(a.locator('#desk-count')).to_have_text('1 saved item')
            a.locator('#desk-entries .desk-remove').focus(); a.keyboard.press('Enter')
            expect(a.locator('#desk-count')).to_have_text('0 saved items')
            focused(a.locator('#reading-list > summary'))
        case(f'local-last-remove-fallback-{width}-{scheme}', local_remove)

        def local_clear():
            a.locator('#desk-save-current').click()
            expect(a.locator('#desk-count')).to_have_text('1 saved item')
            a.locator('#desk-clear').focus(); a.keyboard.press('Enter')
            expect(a.locator('#desk-count')).to_have_text('0 saved items')
            focused(a.locator('#reading-list > summary'))
        case(f'local-clear-fallback-{width}-{scheme}', local_clear)

        def queued_remove():
            a.locator('#desk-save-current').click()
            expect(a.locator('#desk-count')).to_have_text('1 saved item')
            a.evaluate('''key=>{navigator.locks.request(key,()=>new Promise(resolve=>{
                window.releaseContinuityLock=resolve}));}''', KEY)
            a.wait_for_function("typeof window.releaseContinuityLock==='function'")
            a.locator('#desk-entries .desk-remove').click()
            outside = a.locator('#library-current-link'); outside.focus()
            a.evaluate('window.releaseContinuityLock()')
            expect(a.locator('#desk-count')).to_have_text('0 saved items')
            focused(outside)
        case(f'queued-removal-keeps-newer-focus-{width}-{scheme}', queued_remove)

        def closed_list():
            a.locator('#reading-list > summary').click()
            control = a.locator('#reading-list > summary'); control.focus()
            add_peer(); focused(control)
            assert not a.locator('#reading-list').evaluate('(e)=>e.open')
            control.click()
        case(f'closed-list-background-update-{width}-{scheme}', closed_list)
        assert a.evaluate('document.documentElement.scrollWidth <= innerWidth')
        a.locator('#reading-list').screenshot(path=str(out / f'list-{width}-{scheme}.png'))
        c.unroute_all(behavior='wait'); c.close()

    def local_only():
        c = browser.new_context(viewport={'width': 390, 'height': 900}, color_scheme='dark')
        c.add_init_script("Object.defineProperty(window,'localStorage',{get(){throw new Error('fixture storage unavailable')}})")
        if local: c.route('https://raw.githubusercontent.com/**', fixture)
        p = c.new_page(); p.on('pageerror', lambda error: report['page_errors'].append(str(error)))
        p.goto(origin + 'workspace.html?source-q=PROOF.md#library-current')
        expect(p.locator('.current-results .desk-save-source').first).to_be_enabled()
        p.locator('.current-results .desk-save-source').first.click()
        p.locator('#reading-list > summary').click()
        detail = p.locator('#desk-entries details'); detail.locator('summary').click()
        control = p.locator('#desk-entries .desk-cite-source'); control.focus()
        # A result-attachment update uses the same renderer even in tab-only mode.
        p.locator('.current-search').evaluate("e=>{e.value='README.md';e.dispatchEvent(new Event('input',{bubbles:true}))}")
        expect(p.locator('.current-results > li > a').first).to_contain_text('README.md')
        # MutationObserver attachment finishes before the next browser task.
        p.wait_for_function("document.querySelector('.current-results .desk-save-source')!==null")
        focused(control); assert detail.evaluate('(e)=>e.open')
        expect(p.locator('#desk-status')).to_contain_text('this tab only')
        control = p.locator('#desk-entries .desk-remove'); control.focus(); p.keyboard.press('Enter')
        expect(p.locator('#desk-count')).to_have_text('0 saved items')
        focused(p.locator('#reading-list > summary'))
        c.unroute_all(behavior='wait'); c.close()
    case('tab-only-result-update-and-last-removal', local_only)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--browser', type=Path, default=Path('/opt/google/chrome/chrome'))
    parser.add_argument('--origin', help='Actual site origin ending /site/; no local fixtures')
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    try: commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError): commit = 'No Git metadata'
    report = {'started_utc': datetime.now(timezone.utc).isoformat(), 'git_head': commit,
              'local_product_sha256': sha256((ROOT / 'docs/site/research-desk.mjs').read_bytes()).hexdigest(),
              'sandbox': True, 'browser_sha256': sha256(args.browser.read_bytes()).hexdigest(),
              'cases': [], 'page_errors': [],
              'limits': 'Headless Chromium/Linux, emulated viewport; no physical touch, Safari, spoken AT or scientific qualification.'}
    from playwright.sync_api import sync_playwright
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=str(args.browser), chromium_sandbox=True)
            report['browser_version'] = browser.version
            if args.origin: report['origin'] = args.origin; run(args.origin, browser, args.output, report, False)
            else:
                with serve() as origin: report['origin'] = origin; run(origin, browser, args.output, report, True)
            browser.close()
    except Exception: report['error'] = traceback.format_exc()
    report['passed'] = (not report.get('error') and not report['page_errors'] and
                        bool(report['cases']) and all(row['passed'] for row in report['cases']))
    report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'passed': report['passed'], 'cases': len(report['cases']), 'output': str(args.output)}))
    return 0 if report['passed'] else 1


if __name__ == '__main__': raise SystemExit(main())
