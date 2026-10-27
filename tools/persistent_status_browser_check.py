#!/usr/bin/env python3
"""Current status presentation, exact source and fallback checks; not speech certification."""
import argparse
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from playwright.sync_api import expect, sync_playwright


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--site', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--browser', default='/usr/bin/chromium')
    ap.add_argument('--observe', action='store_true')
    ap.add_argument('--base-url', help='Replay a publication using the supplied local source identity')
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    raw = (args.site/'current.json').read_bytes(); data = json.loads(raw)
    meta = json.loads((args.site/'current-meta.json').read_bytes())
    assert hashlib.sha256(raw).hexdigest() == meta['sha256'] and len(raw) == meta['bytes']
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *unused): pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Handler, directory=str(args.site)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = (args.base_url or f'http://127.0.0.1:{server.server_port}').rstrip('/')
    report = {'payload_sha256':meta['sha256'], 'payload_bytes':len(raw), 'repositories':data['repositories'],
              'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'cases':[]}
    def save(): (args.output/'persistent-browser.json').write_text(json.dumps(report, indent=2)+'\n')
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(executable_path=args.browser, chromium_sandbox=True,
                                         args=['--enable-unsafe-swiftshader'])
            report['browser'] = browser.version
            scenarios = [(f'{width}-{scheme}',width,scheme,True,'dependencies.html',False)
                         for width in (1200,390) for scheme in ('light','dark')]
            scenarios += [('nojs-'+str(w),w,'light',False,'dependencies.html',False) for w in (1200,390)]
            scenarios += [('static-index',1200,'light',True,'nodes/index.html',False),
                          ('static-node',390,'dark',True,'nodes/math.side24-coefficient.html',False),
                          ('script-free-formal',390,'light',True,'formal.html',False),
                          ('unavailable',390,'dark',True,'dependencies.html',True)]
            for name,width,scheme,js,path,unavailable in scenarios:
                row={'case':name,'failures':[],'errors':[]};report['cases'].append(row)
                context=browser.new_context(viewport={'width':width,'height':900},color_scheme=scheme,
                                            java_script_enabled=js,reduced_motion='reduce')
                page=context.new_page();page.on('pageerror',lambda e,r=row:r['errors'].append(str(e)))
                def check(ok,label):
                    if not ok:row['failures'].append(label)
                def paragraph(locator,label):
                    check(locator.evaluate('(e)=>e.tagName==="P" && !e.hasAttribute("role") && e.parentElement.getAttribute("role")==="status"'),label+' paragraph in status region')
                try:
                    if unavailable:
                        page.route('**/current.json*',lambda route:route.fulfill(status=200,content_type='application/json',body='{}'))
                    t=time.perf_counter();page.goto(origin+'/'+path,wait_until='networkidle')
                    status=page.locator('#current-research .current-state')
                    if unavailable:expect(status).to_contain_text('Unavailable:')
                    elif js and path=='dependencies.html':expect(status).to_contain_text('Verified source observation:')
                    else:expect(status).to_contain_text('Pinned source observation:')
                    row['ready_ms']=(time.perf_counter()-t)*1000
                    paragraph(status,'source observation');expect(status).to_be_visible()
                    if not unavailable:expect(status).to_contain_text(data['observed_at'])
                    row['text']=status.inner_text();row['geometry']=status.bounding_box()
                    row['style']=status.evaluate('(e)=>{const s=getComputedStyle(e);return {font:s.fontSize,line:s.lineHeight,top:s.marginTop,bottom:s.marginBottom};}')
                    if width==1200:check(row['geometry']['width']<800,'desktop prose keeps its existing reading measure')
                    if path=='dependencies.html':
                        check(row['style']=={'font':'13px','line':'20.8px','top':'0px','bottom':'0px'},'compact paragraph keeps exact typography/margins')
                        legacy=page.locator('#load-status');paragraph(legacy,'preserved viewer')
                        if js:expect(legacy).to_contain_text('Pinned Math commit')
                        else:expect(legacy).to_contain_text('Loading the local source snapshot')
                        row['legacy_text']=legacy.inner_text();row['legacy_geometry']=legacy.bounding_box()
                        status.scroll_into_view_if_needed();page.locator('#current-research').screenshot(path=str(args.output/(name+'-source.png')))
                        page.locator('.atlas-archive-notices').screenshot(path=str(args.output/(name+'-preserved.png')))
                        if not unavailable:
                            details=page.locator('#current-research .current-source-details');summary=details.locator(':scope > summary')
                            summary.focus();page.keyboard.press('Enter');expect(details).to_have_attribute('open','')
                            check(summary.evaluate('(e)=>document.activeElement===e'),'disclosure keeps keyboard focus')
                            for repo in data['repositories']:
                                href=f'https://github.com/{repo["repository"]}/tree/{repo["commit"]}'
                                expect(details.locator(f'a[href="{href}"]')).to_have_count(1)
                            page.keyboard.press('Space');expect(details).not_to_have_attribute('open','')
                        if not js:expect(page.locator('#atlas-view')).to_be_disabled()
                        if unavailable:expect(page.locator('#atlas-view')).to_be_disabled()
                    else:
                        for repo in data['repositories']:
                            expect(page.locator(f'#current-research a[href="https://github.com/{repo["repository"]}/tree/{repo["commit"]}"]')).to_have_count(1)
                    check(page.evaluate('() => document.documentElement.scrollWidth<=innerWidth'),'no horizontal document overflow')
                    check(not row['errors'],'no page errors')
                except Exception as e:row['failures'].append(str(e))
                row['passed']=not row['failures'];context.close();save()
            browser.close()
    finally:server.shutdown();server.server_close()
    print(json.dumps({'passed':sum(r['passed'] for r in report['cases']),'total':len(report['cases']),'report':str(args.output/'persistent-browser.json')}))
    if not args.observe and any(not r['passed'] for r in report['cases']):raise SystemExit(1)


if __name__=='__main__':main()
