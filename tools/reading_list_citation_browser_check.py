#!/usr/bin/env python3
"""Exercise saved-source → existing Cite navigation without acquiring sources."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import traceback
from urllib.parse import urlsplit
from research_desk_browser_check import ROOT, KEY, serve


def run(origin, browser, out, report, local=True):
    from playwright.sync_api import expect
    site=urlsplit(origin);site_origin=(site.scheme,site.netloc)
    config=json.loads((ROOT/'docs/site/config.json').read_text())
    fixtures={config['inventory']['url']:ROOT/'docs/public-math/sources.json',
              config['status']['url']:ROOT/'tests/fixtures/status_f2e432e.md'}
    fixtures.update({pin['url']:ROOT/pin['path'] for pin in config['inventory']['pages']})
    def context(width,scheme):
        c=browser.new_context(viewport={'width':width,'height':900},color_scheme=scheme,
                              accept_downloads=True,reduced_motion='reduce')
        def fixture(route):
            path=fixtures.get(route.request.url)
            route.fulfill(status=200 if path else 503,headers={'Access-Control-Allow-Origin':'*'},
                          body=path.read_bytes() if path else b'')
        if local:c.route('https://raw.githubusercontent.com/**',fixture)
        return c
    def case(name,fn):
        row={'case':name};report['cases'].append(row)
        try: fn();row['passed']=True
        except Exception:
            row['passed']=False;row['error']=traceback.format_exc();raise
    def page(c):
        p=c.new_page();p.set_default_timeout(15000)
        p.on('pageerror',lambda error:report['page_errors'].append(str(error)))
        external=[]
        def track_external(request):
            url=urlsplit(request.url)
            if url.scheme in ('http','https') and (url.scheme,url.netloc)!=site_origin:external.append(request.url)
        p.on('request',track_external)
        return p,external
    def inspect_reference(p,entry):
        expect(p.locator('#reference-builder')).to_be_visible()
        expect(p.locator('#reference-repository')).to_have_value(entry['repository'].split('/')[1])
        expect(p.locator('#reference-commit')).to_have_value(entry['commit'])
        expect(p.locator('#reference-path')).to_have_value(entry['path'])
        expect(p.locator('#reference-sha256')).to_have_value(entry.get('sha256',''))
        details=p.get_by_text('Inspect structured source identity (JSON)',exact=True)
        details.focus();p.keyboard.press('Enter')
        p.locator('#reference-json').focus();expect(p.locator('#reference-json')).to_be_focused()
        data=json.loads(p.locator('#reference-json').inner_text())
        assert data['repository']==entry['repository'] and data['commit']==entry['commit'] and data['path']==entry['path']
        assert data['sha256']==entry.get('sha256') and data['verification']=='not_performed'
        assert 'blob' not in data and 'author' not in data and 'review' not in data
        expect(p.locator('#reference-status')).to_contain_text('not checked')

    for width,scheme in [(390,'light'),(1440,'dark')]:
        def journey():
            c=context(width,scheme);p,external=page(c)
            p.goto(origin+'index.html')
            p.locator('nav[aria-label="Primary"] a',has_text='Library').click()
            p.locator('#library-current-link').focus();p.keyboard.press('Enter')
            expect(p.locator('.current-search')).to_be_enabled()
            p.locator('.current-search').fill('PROOF.md')
            p.locator('.current-results .desk-save-source').first.click()
            expect(p.locator('#desk-count')).to_have_text('1 saved item')
            p.locator('#reading-list > summary').click()
            before=p.evaluate('(key)=>localStorage.getItem(key)',KEY)
            entry=json.loads(before)['entries'][0]
            p.wait_for_load_state('networkidle');requests_before=list(external)
            cite=p.locator('#desk-entries .desk-cite-source');cite.focus();p.keyboard.press('Enter')
            inspect_reference(p,entry)
            p.wait_for_load_state('networkidle')
            assert external==requests_before,external[len(requests_before):]
            assert p.url.startswith(origin+'cite.html?') and p.url.endswith('#reference-builder')
            assert p.evaluate('document.documentElement.scrollWidth<=innerWidth')
            p.locator('#reference-builder').screenshot(path=str(out/f'cite-{width}-{scheme}.png'))
            p.go_back();expect(p.locator('#desk-count')).to_have_text('1 saved item')
            assert p.evaluate('(key)=>localStorage.getItem(key)',KEY)==before
            if not p.locator('#reading-list').evaluate('(element)=>element.open'):p.locator('#reading-list > summary').click()
            expect(p.locator('#desk-entries .desk-cite-source')).to_be_visible()
            assert p.evaluate('document.documentElement.scrollWidth<=innerWidth')
            p.locator('#reading-list').screenshot(path=str(out/f'list-{width}-{scheme}.png'))
            # Back may reload the ordinary Library observation. Cite itself did
            # not request any document or external source bytes.
            report.setdefault('journeys',[]).append({'width':width,'scheme':scheme,'saved':entry,
                'requests_before_cite':requests_before,'external_requests':external})
            c.unroute_all(behavior='wait');c.close()
        case(f'home-library-save-keyboard-reference-back-{width}-{scheme}',journey)

    def imported():
        c=context(390,'light');p,external=page(c);p.goto(origin+'workspace.html')
        p.locator('#reading-list > summary').click()
        base={'kind':'source','collection':'catalog','repository':'d6g8k5htny-coder/main',
              'commit':'a'*40,'savedAt':'2026-10-26T00:00:00Z'}
        entries=[{**base,'path':"docs/Δ + a&b#c?d{}!'().md",'sha256':'b'*64},
                 {**base,'path':'docs/no-sha.md'},
                 {**base,'path':'docs/blob-only.md','blob':'c'*40},
                 *[{**base,'path':path} for path in ['docs/literal%20.md','docs/\u202eproof.md','a'*501]],
                 {'kind':'search','collection':'current','query':'PROOF.md','repository':'','path':'','savedAt':base['savedAt']}]
        payload=json.dumps({'schema':KEY,'entries':entries}).encode()
        p.locator('#desk-import').set_input_files({'name':'literal-paths.json','mimeType':'application/json','buffer':payload})
        expect(p.locator('#desk-entries > li')).to_have_count(7)
        expect(p.locator('#desk-entries .desk-cite-source')).to_have_count(3)
        expect(p.locator('.desk-citation-unavailable')).to_have_count(3)
        expect(p.locator('.desk-citation-unavailable').first).to_have_text('Reference builder unavailable for this literal path.')
        stored=p.evaluate('(key)=>localStorage.getItem(key)',KEY)
        for index in range(3):
            p.wait_for_load_state('networkidle');count=len(external)
            p.locator('#desk-entries > li').nth(index).locator('.desk-cite-source').focus();p.keyboard.press('Enter')
            inspect_reference(p,entries[index]);p.wait_for_load_state('networkidle')
            assert len(external)==count,external[count:]
            p.go_back();expect(p.locator('#desk-count')).to_have_text('7 saved items')
            if not p.locator('#reading-list').evaluate('(element)=>element.open'):p.locator('#reading-list > summary').click()
            assert p.evaluate('(key)=>localStorage.getItem(key)',KEY)==stored
        # Unrepresentable entries retain their exact source link and export.
        for index in range(3,6):
            row=p.locator('#desk-entries > li').nth(index)
            assert row.locator('a').count()==1 and row.locator('a').get_attribute('href').startswith('https://github.com/')
            expect(row.locator('button')).to_be_enabled()
        assert p.locator('#desk-entries > li').nth(6).locator('.desk-cite-source').count()==0
        with p.expect_download() as download:p.locator('#desk-export').click()
        target=out/'imported-roundtrip.json';download.value.save_as(target)
        assert json.loads(target.read_text())['entries']==entries
        # Removing a refused source still restores the next Remove button.
        remove=p.locator('#desk-entries > li').nth(3).locator('button');remove.focus();p.keyboard.press('Enter')
        expect(p.locator('#desk-count')).to_have_text('6 saved items')
        expect(p.locator('#desk-entries > li').nth(3).locator('button')).to_be_focused()
        p.locator('#reading-list').screenshot(path=str(out/'literal-path-refusals-mobile-light.png'))
        report['imported_external_requests']=external
        c.unroute_all(behavior='wait');c.close()
    case('imported-literal-paths-optional-sha-blob-refusals-export-remove',imported)


def parse_arguments(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--browser',type=Path,default=Path('/opt/google/chrome/chrome'))
    parser.add_argument('--origin',help='Published site origin ending /site/; uses served assets and no local source fixtures')
    return parser.parse_args(argv)


def main():
    args=parse_arguments();args.output.mkdir(parents=True,exist_ok=True)
    report={'started_utc':datetime.now(timezone.utc).isoformat(),'cases':[],'page_errors':[],
        'local_product_sha256':{name:sha256((ROOT/'docs/site'/name).read_bytes()).hexdigest() for name in ['reading-list-citation.mjs','research-desk.mjs','source-reference.mjs']},
        'local_product_sha256_scope':'Local checkout comparison context only; these hashes do not establish actual served asset identity.',
        'browser_sha256':sha256(args.browser.read_bytes()).hexdigest(),'sandbox':True,
        'limits':'Headless Chromium/Linux viewport emulation; no physical-device, Safari, spoken AT or scientific verdict.'}
    from playwright.sync_api import sync_playwright
    try:
        with sync_playwright() as playwright:
            browser=playwright.chromium.launch(executable_path=str(args.browser),chromium_sandbox=True)
            report['browser_version']=browser.version
            if args.origin:
                origin=args.origin.rstrip('/')+'/';report['origin']=origin;run(origin,browser,args.output,report,False)
            else:
                with serve() as origin:report['origin']=origin;run(origin,browser,args.output,report)
            browser.close()
        assert not report['page_errors'],report['page_errors'];report['passed']=True
    except Exception:report['passed']=False;report['error']=traceback.format_exc()
    report['finished_utc']=datetime.now(timezone.utc).isoformat()
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':report['passed'],'cases':len(report['cases']),'output':str(args.output)}))
    return 0 if report['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
