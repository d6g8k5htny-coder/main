#!/usr/bin/env python3
"""Real-browser Library removal recovery; fixture identities are reading notes only."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import traceback
from research_desk_browser_check import ROOT, KEY, serve


def source(index=1, length=0):
    return {'kind': 'source', 'collection': 'catalog',
            'repository': 'd6g8k5htny-coder/main', 'commit': f'{index:040x}',
            'path': f'fixture-{index}/' + 'x' * length + 'PROOF.md',
            'blob': f'{index + 100:040x}', 'sha256': f'{index + 200:064x}',
            'bytes': index, 'savedAt': '2026-10-26T00:00:00Z',
            'observedAt': '2026-10-25T00:00:00Z'}


def search(index=1):
    return {'kind': 'search', 'collection': 'catalog', 'query': f'fixture {index}',
            'repository': 'main', 'path': 'proofs/', 'savedAt': '2026-10-26T00:00:00Z'}


def run(origin, browser, out, report, local, baseline):
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

    def open_page(context):
        p = context.new_page(); p.set_default_timeout(10000)
        p.on('pageerror', lambda error: report['page_errors'].append(str(error)))
        p.goto(origin + 'workspace.html#reading-list')
        expect(p.locator('#desk-import')).to_be_enabled()
        if not p.locator('#reading-list').evaluate('(e)=>e.open'):
            p.locator('#reading-list > summary').click()
        return p

    def count(p, n):
        expect(p.locator('#desk-count')).to_have_text(f'{n} saved item' + ('' if n == 1 else 's'))

    def imported(p, entries):
        payload = json.dumps({'schema': KEY, 'entries': entries}).encode()
        p.locator('#desk-import').set_input_files({'name': 'synthetic-reading-notes.json',
                    'mimeType': 'application/json', 'buffer': payload})
        expect(p.locator('#desk-message')).to_contain_text('Reading list imported')

    def saved(p):
        return json.loads(p.evaluate('(key)=>localStorage.getItem(key)', KEY))['entries']

    def same(actual, expected):
        assert sorted(json.dumps(x, sort_keys=True) for x in actual) == sorted(
            json.dumps(x, sort_keys=True) for x in expected), 'Exact saved identities changed'

    def undo(p):
        control = p.locator('#desk-undo')
        assert control.count() == 1, 'Removal has no Undo control'
        expect(control).to_be_visible(); expect(control).to_be_enabled()
        control.focus(); p.keyboard.press('Enter')

    views = [(1440, 'light'), (390, 'dark'), (320, 'light')]
    if not baseline: views += [(1440, 'dark'), (390, 'light')]
    for width, scheme in views:
        def case(name, scenario, init=None):
            row = {'case': f'{name}-{width}-{scheme}'}; report['cases'].append(row)
            c = browser.new_context(viewport={'width': width, 'height': 900},
                                    color_scheme=scheme, reduced_motion='reduce')
            if init: c.add_init_script(init)
            if local: c.route('https://raw.githubusercontent.com/**', fixture)
            try:
                p = open_page(c); scenario(c, p); row['passed'] = True
            except Exception:
                row['passed'] = False; row['error'] = traceback.format_exc()
            finally:
                c.unroute_all(behavior='wait'); c.close()
            print(json.dumps({'case': row['case'], 'passed': row['passed']}), flush=True)

        def remove(_c, p):
            original = [source(), search()]; imported(p, original); count(p, 2)
            p.locator('#desk-entries .desk-remove').first.focus(); p.keyboard.press('Enter')
            count(p, 1); undo(p); count(p, 2); same(saved(p), original)
            expect(p.locator('#desk-undo')).to_be_hidden()
            expect(p.locator('#reading-list > summary')).to_be_focused()
            expect(p.locator('#desk-message')).to_contain_text('restored')
            assert p.evaluate('document.documentElement.scrollWidth<=innerWidth')
        case('keyboard-remove-exact-restoration', remove)

        def clear(c, p):
            original = [source(), search()]; imported(p, original)
            peer = open_page(c); count(peer, 2)
            p.locator('#desk-clear').focus(); p.keyboard.press('Enter')
            count(p, 0); count(peer, 0)
            expect(p.locator('#reading-list > summary')).to_be_focused()
            undo(p); count(p, 2); count(peer, 2); same(saved(p), original)
            p.locator('#reading-list').screenshot(path=str(out / f'undo-{width}-{scheme}.png'))
        case('keyboard-clear-and-peer-restoration', clear)
        if baseline: continue

        def peer_add(c, p):
            original = [source(), search()]; imported(p, original)
            peer = open_page(c); p.locator('#desk-clear').click(); count(p, 0); count(peer, 0)
            imported(peer, [search(99)]); count(p, 1)
            undo(p); count(p, 3); count(peer, 3); same(saved(p), original + [search(99)])
        case('undo-preserves-peer-additions', peer_add)

        def duplicate(c, p):
            original = source(); imported(p, [original]); peer = open_page(c)
            p.locator('#desk-entries .desk-remove').click(); count(peer, 0)
            imported(peer, [original, search(99)]); count(p, 2)
            undo(p); count(p, 2); same(saved(p), [original, search(99)])
            expect(p.locator('#desk-undo')).to_be_hidden()
        case('peer-readded-source-is-not-duplicated', duplicate)

        def conflict(c, p):
            original = source(); imported(p, [original]); peer = open_page(c)
            p.locator('#desk-entries .desk-remove').click(); count(peer, 0)
            changed = {**original, 'sha256': 'f' * 64}
            imported(peer, [changed, search(99)]); count(p, 2)
            before = saved(p); undo(p)
            expect(p.locator('#desk-message')).to_contain_text('Undo refused:')
            same(saved(p), before); expect(p.locator('#desk-undo')).to_be_enabled()
            peer.locator('#desk-entries .desk-remove').first.click(); count(p, 1)
            undo(p); count(p, 2); same(saved(p), [original, search(99)])
        case('conflicting-source-refused-and-retryable', conflict)

        def capacity(c, p):
            original = [source(), search()]; imported(p, original); peer = open_page(c)
            p.locator('#desk-clear').click(); count(peer, 0)
            imported(peer, [search(i) for i in range(100, 150)]); count(p, 50)
            before = saved(p); undo(p)
            expect(p.locator('#desk-message')).to_contain_text('Undo refused:')
            same(saved(p), before); expect(p.locator('#desk-undo')).to_be_enabled()
            for n in [49, 48]: peer.locator('#desk-entries .desk-remove').first.click(); count(p, n)
            remaining = saved(p); undo(p); count(p, 50)
            same(saved(p), remaining + original)
        case('fifty-entry-refusal-keeps-recovery', capacity)

        def bytes_limit(c, p):
            original = [source(1, 40000), source(2, 40000)]
            imported(p, original); peer = open_page(c); p.locator('#desk-clear').click(); count(peer, 0)
            imported(peer, [source(3, 40000), source(4, 40000)]); count(p, 2)
            before = saved(p); undo(p)
            expect(p.locator('#desk-message')).to_contain_text('128 KiB')
            same(saved(p), before); expect(p.locator('#desk-undo')).to_be_enabled()
            peer.locator('#desk-entries .desk-remove').first.click(); count(p, 1)
            remaining = saved(p); undo(p); count(p, 3); same(saved(p), remaining + original)
        case('byte-limit-refusal-keeps-recovery', bytes_limit)

        def latest(_c, p):
            original = [source(), search(1), search(2)]; imported(p, original)
            p.locator('#desk-entries .desk-remove').first.click(); count(p, 2)
            p.locator('#desk-entries .desk-remove').first.click(); count(p, 1)
            undo(p); count(p, 2); same(saved(p), original[1:])
        case('only-latest-actual-removal-is-restored', latest)

        def hold(p):
            p.evaluate('''key=>{navigator.locks.request(key,()=>new Promise(resolve=>{
                window.releaseUndoLock=resolve}));}''', KEY)
            p.wait_for_function("()=>typeof window.releaseUndoLock==='function'")

        def no_op(_c, p):
            original = [source(), search()]; imported(p, original); hold(p)
            button = p.locator('#desk-entries .desk-remove').first
            button.click(); button.click()
            p.evaluate('window.releaseUndoLock()'); count(p, 1)
            expect(p.locator('#desk-message')).to_contain_text('No further items')
            undo(p); count(p, 2); same(saved(p), original)
        case('queued-no-op-does-not-erase-recovery', no_op)

        def late(_c, p):
            original = [source(), search()]; imported(p, original)
            p.locator('#desk-clear').click(); count(p, 0); hold(p)
            undo(p); outside = p.locator('#library-current-link'); outside.focus()
            p.evaluate('window.releaseUndoLock()'); count(p, 2); same(saved(p), original)
            expect(outside).to_be_focused()
        case('queued-undo-keeps-newer-outside-focus', late)

        def peer_only(c, p):
            imported(p, [source()]); peer = open_page(c)
            peer.locator('#desk-clear').click(); count(p, 0)
            expect(p.locator('#desk-undo')).to_be_hidden()
            undo(peer); count(p, 1)
        case('peer-removal-does-not-invent-local-recovery', peer_only)

        def reload(_c, p):
            imported(p, [source()]); p.locator('#desk-clear').click(); count(p, 0)
            expect(p.locator('#desk-undo')).to_be_visible(); p.reload()
            expect(p.locator('#desk-import')).to_be_enabled(); count(p, 0)
            expect(p.locator('#desk-undo')).to_be_hidden()
        case('recovery-is-not-persisted-across-reload', reload)

        def tab_only(_c, p):
            imported(p, [source(), search()]); count(p, 2)
            p.locator('#desk-clear').click(); count(p, 0)
            undo(p); count(p, 2)
            expect(p.locator('#desk-message')).to_contain_text('this tab only')
        case('tab-only-recovery', tab_only,
             "Object.defineProperty(window,'localStorage',{get(){throw Error('fixture unavailable')}})")
        case('refused-lock-remains-tab-only', tab_only,
             "navigator.locks.request=()=>Promise.reject(Error('fixture lock refusal'))")
        case('refused-write-remains-tab-only', tab_only,
             "Storage.prototype.setItem=function(){throw Error('fixture write refusal')}")

        def unreadable(c, p):
            imported(p, [source(), search()]); peer = open_page(c)
            p.locator('#desk-clear').click(); count(peer, 0)
            peer.evaluate('(key)=>localStorage.setItem(key,"fixture invalid JSON")', KEY)
            expect(p.locator('#desk-message')).to_contain_text('unreadable')
            undo(p); count(p, 2)
            assert p.evaluate('(key)=>localStorage.getItem(key)', KEY) == 'fixture invalid JSON'
            expect(p.locator('#desk-message')).to_contain_text('this tab only')
        case('unreadable-peer-storage-is-preserved', unreadable)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--browser', type=Path, default=Path(shutil.which('chromium') or '/opt/google/chrome/chrome'))
    parser.add_argument('--origin', help='Actual /site/ origin; disables local source fixtures')
    parser.add_argument('--baseline', action='store_true', help='Only the two core missing-recovery regressions')
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    try: commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError): commit = 'No Git metadata'
    report = {'started_utc': datetime.now(timezone.utc).isoformat(), 'git_head': commit,
              'local_product_sha256': sha256((ROOT / 'docs/site/research-desk.mjs').read_bytes()).hexdigest(),
              'browser_path': str(args.browser), 'sandbox': True, 'cases': [], 'page_errors': [],
              'limits': 'Synthetic imported reading notes, headless Chromium/Linux, emulated widths. No source authentication, physical touch, Safari, spoken AT or scientific qualification.'}
    from playwright.sync_api import sync_playwright
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=str(args.browser), chromium_sandbox=True)
            report['browser_version'] = browser.version
            if args.origin: report['origin'] = args.origin; run(args.origin, browser, args.output, report, False, args.baseline)
            else:
                with serve() as origin:
                    report['origin'] = origin; run(origin, browser, args.output, report, True, args.baseline)
            browser.close()
    except Exception: report['error'] = traceback.format_exc()
    report['passed'] = (not report.get('error') and not report['page_errors'] and bool(report['cases'])
                        and all(row['passed'] for row in report['cases']))
    report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'passed': report['passed'], 'cases': len(report['cases']), 'output': str(args.output)}))
    return 0 if report['passed'] else 1


if __name__ == '__main__': raise SystemExit(main())
