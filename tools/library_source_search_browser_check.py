#!/usr/bin/env python3
"""Exercise Library search recovery against real admitted local source bytes."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import traceback
from urllib.parse import urlsplit, parse_qs
from research_desk_browser_check import ROOT, serve


def run(origin, browser, output, report, baseline=False):
    from playwright.sync_api import expect
    def case(name, operation):
        row = {'case': name}; report['cases'].append(row)
        try:
            operation(); row['passed'] = True
        except Exception:
            row['passed'] = False; row['error'] = traceback.format_exc()

    for width, scheme in [(1440, 'light'), (1440, 'dark'), (390, 'light'), (390, 'dark'), (320, 'light')]:
        context = browser.new_context(viewport={'width': width, 'height': 960},
                                      color_scheme=scheme, reduced_motion='reduce')
        context.route('https://**', lambda route: route.abort())
        page = context.new_page(); page.set_default_timeout(5000)
        page.on('pageerror', lambda error: report['page_errors'].append(str(error)))
        def open_page(query=''):
            page.goto(origin+'workspace.html?keep=literal%23note&source-q='+query+'#library-current')
            expect(page.locator('.current-search')).to_be_enabled()
        def empty():
            open_page('zz-no-source-20261026')
            expect(page.locator('.current-results > li')).to_have_count(0)
            expect(page.locator('#library-source-empty')).to_be_visible()
            expect(page.locator('#library-source-empty')).to_contain_text('Try a shorter path')
            expect(page.locator('#library-source-clear')).to_be_enabled()
            expect(page.locator('#library-source-link')).to_be_visible()
            assert 'results are not frozen' in page.locator('#library-source-query-note').inner_text()
            assert page.locator('#library-source-empty').get_attribute('role') == 'status'
            page.locator('#library-source-clear').focus(); page.keyboard.press('Enter')
            expect(page.locator('.current-search')).to_have_value('')
            expect(page.locator('.current-search')).to_be_focused()
            expect(page.locator('.current-results > li')).to_have_count(50)
            expect(page.locator('#library-source-empty')).to_be_hidden()
            expect(page.locator('#library-source-clear')).to_be_disabled()
            parsed = parse_qs(urlsplit(page.url).query)
            assert 'source-q' not in parsed and parsed['keep'] == ['literal#note']
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            page.locator('#library-source-tools').screenshot(path=str(output/f'tools-{width}-{scheme}.png'))
        case(f'empty-recovery-keyboard-{width}-{scheme}', empty)
        if not baseline:
            def literal():
                open_page()
                for query in ['Δ + a&b#c?d%20', 'proof', '<img src=x onerror=alert(1)>', '  coefficients  ']:
                    page.locator('.current-search').fill(query)
                    href = page.locator('#library-source-link').get_attribute('href')
                    parsed = urlsplit(href)
                    assert parse_qs(parsed.query)['source-q'] == [query]
                    assert parse_qs(parsed.query)['keep'] == ['literal#note']
                    assert parsed.fragment == 'library-current'
                    assert not page.locator('#library-source-tools img').count()
                    fresh = context.new_page(); fresh.goto(href)
                    expect(fresh.locator('.current-search')).to_be_enabled()
                    expect(fresh.locator('.current-search')).to_have_value(query)
                    fresh.close()
                page.reload(); expect(page.locator('.current-search')).to_have_value('  coefficients  ')
                expect(page.locator('#library-source-link')).to_have_attribute('href', page.url)
            case(f'literal-query-native-link-reload-{width}-{scheme}', literal)
            def page_window():
                open_page('proof')
                expect(page.locator('.current-results > li')).to_have_count(50)
                page.locator('.current-more').click()
                expect(page.locator('.current-results > li')).to_have_count(100)
                page.locator('#library-source-clear').click()
                expect(page.locator('.current-results > li')).to_have_count(50)
                # Native fragment history carries the existing query snapshots.
                page.locator('.current-search').fill('coefficients')
                # replaceState is synchronous in the renderer; the automation
                # client's cached frame URL may arrive on the next protocol turn.
                first = page.evaluate('location.href')
                report.setdefault('history_observations',[]).append({'stage':'before-fragment','width':width,'scheme':scheme,'client_url':first,'renderer_url':page.evaluate('location.href')})
                page.evaluate("location.hash='reading-list'")
                page.locator('.current-search').fill('zz-no-source-20261026')
                second = page.evaluate('location.href')
                report['history_observations'].append({'stage':'before-back','width':width,'scheme':scheme,'client_url':second,'renderer_url':page.evaluate('location.href')})
                page.go_back(); expect(page.locator('.current-search')).to_have_value('coefficients')
                report['history_observations'].append({'stage':'after-back','width':width,'scheme':scheme,'client_url':page.url,'renderer_url':page.evaluate('location.href'),'expected':first})
                expect(page).to_have_url(first)
                expect(page.locator('#library-source-link')).to_have_attribute('href', first)
                page.go_forward(); expect(page.locator('.current-search')).to_have_value('zz-no-source-20261026')
                report['history_observations'].append({'stage':'after-forward','width':width,'scheme':scheme,'client_url':page.url,'renderer_url':page.evaluate('location.href'),'expected':second})
                expect(page).to_have_url(second)
                expect(page.locator('#library-source-empty')).to_be_visible()
                # A background result replacement must not steal newer focus.
                page.locator('#reading-list > summary').focus()
                page.evaluate("document.querySelector('.current-results').append(document.createElement('li'))")
                expect(page.locator('#reading-list > summary')).to_be_focused()
            case(f'pagination-history-focus-{width}-{scheme}', page_window)
        context.close()
    if not baseline:
        def unavailable():
            context = browser.new_context(viewport={'width':390,'height':844})
            context.route('https://**', lambda route: route.abort())
            context.route('**/current.json', lambda route: route.fulfill(status=503, body='Unavailable'))
            page = context.new_page(); page.on('pageerror', lambda error: report['page_errors'].append(str(error)))
            page.goto(origin+'workspace.html#library-current')
            expect(page.locator('.current-state')).to_contain_text('Unavailable')
            expect(page.locator('.current-search')).to_be_disabled()
            assert page.locator('#library-source-tools').count() == 0
            expect(page.locator('#desk-save-current')).to_be_disabled()
            context.close()
        case('unavailable-snapshot-no-enabled-tools', unavailable)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--browser', default=shutil.which('chromium') or shutil.which('google-chrome'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--origin')
    parser.add_argument('--baseline', action='store_true')
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    report = {'observed_at':datetime.now(timezone.utc).isoformat(), 'cases':[], 'page_errors':[],
              'scope':'Engineering only; no source acquisition, mathematical review or spoken AT claim.'}
    from playwright.sync_api import sync_playwright
    def execute(origin):
        with sync_playwright() as pw:
            browser = pw.chromium.launch(executable_path=args.browser, chromium_sandbox=True)
            report['browser'] = browser.version
            run(origin, browser, args.output, report, args.baseline); browser.close()
    if args.origin: execute(args.origin)
    else:
        with serve() as origin: execute(origin)
    report['passed'] = sum(row['passed'] for row in report['cases'])
    report['failed'] = len(report['cases'])-report['passed']
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ['browser','passed','failed','page_errors']}))
    return 1 if report['failed'] or report['page_errors'] else 0


if __name__ == '__main__': raise SystemExit(main())
