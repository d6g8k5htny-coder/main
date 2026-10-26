#!/usr/bin/env python3
"""Matched source-connection browser acceptance; actual source identities remain separate.

Run with --site PATH --output PATH --payload frozen/current.json.
Uses system Chromium with its sandbox and the site's original CSP. No interception.
"""
from __future__ import annotations

import argparse
from collections import Counter
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import subprocess
import threading
import time
import traceback
from urllib.parse import urlencode, urlsplit, parse_qs

from playwright.sync_api import sync_playwright, expect

NODE='math.side24-coefficient'

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def identity(path):
    raw=path.read_bytes()
    return {'bytes':len(raw),'sha256':digest(raw),'git_blob':hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()}

GEOMETRY='''() => {
 const panel=document.querySelector('#atlas-inspector'),base=panel.getBoundingClientRect();
 const box=e=>{if(!e)return null; const r=e.getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height,top_relative_inspector:r.top-base.top,left_relative_inspector:r.left-base.left,viewport_intersects:r.bottom>0&&r.top<innerHeight&&r.right>0&&r.left<innerWidth};};
 const source=[...panel.querySelectorAll('a')].find(a=>a.textContent.includes('Immutable source reference'));
 return {inspector:box(panel),reader:box(document.querySelector('#atlas-reader')),source:box(source),scope:box(document.querySelector('#atlas-claim-summary > p')),formal:box(document.querySelector('#atlas-formal-evidence')),formal_details:box(document.querySelector('#atlas-formal-details')),formal_summary:box(document.querySelector('#atlas-formal-details > summary')),scroll:{x:scrollX,y:scrollY},document_width:document.documentElement.scrollWidth,viewport_width:innerWidth};
}'''

PERFORMANCE='''() => ({navigation:performance.getEntriesByType('navigation').map(e=>e.toJSON()),resources:performance.getEntriesByType('resource').map(e=>({name:e.name,initiatorType:e.initiatorType,startTime:e.startTime,duration:e.duration,transferSize:e.transferSize,encodedBodySize:e.encodedBodySize,decodedBodySize:e.decodedBodySize,responseEnd:e.responseEnd}))})'''

def record_check(row, name, value, detail=None):
    row['checks'].append({'name':name,'passed':bool(value),'detail':detail})
    if not value: row['failures'].append(name)

def inspect_signature(page):
    return {'selected':page.locator('#atlas-inspector h3').inner_text(),
            'source':page.locator('#atlas-claim-summary a').first.get_attribute('href'),
            'scope':page.locator('#atlas-claim-summary > p').first.inner_text(),
            'pins':{k:parse_qs(urlsplit(page.url).query).get(k) for k in ['snapshot','main-snapshot','atlas-node']}}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--site',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--payload',type=Path,required=True)
    p.add_argument('--browser',default='/usr/bin/chromium')
    args=p.parse_args()
    args.site=args.site.resolve(); args.output=args.output.resolve(); args.payload=args.payload.resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    raw=args.payload.read_bytes(); data=json.loads(raw)
    if (args.site/'current.json').read_bytes()!=raw: raise SystemExit('Site current.json differs from frozen payload')
    meta=json.loads((args.site/'current-meta.json').read_bytes())
    if meta['sha256']!=digest(raw) or meta['bytes']!=len(raw): raise SystemExit('Site metadata does not match frozen payload')
    mainpin=next(r['commit'] for r in data['repositories'] if r['repository'].endswith('/main'))
    mathpin=data['graph']['identity']['commit']; node=data['graph']['data']['nodes'][NODE]
    modules={p.name:identity(p) for p in sorted(args.site.iterdir()) if p.suffix in ['.mjs','.css'] or p.name=='dependencies.html'}
    requests=[]
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self,fmt,*values):
            requests.append({'path':self.path,'message':fmt % values})
    server=ThreadingHTTPServer(('127.0.0.1',0),partial(Handler,directory=str(args.site)))
    threading.Thread(target=server.serve_forever,daemon=True).start()
    origin=f'http://127.0.0.1:{server.server_port}'
    query={'atlas-node':NODE,'atlas-view':'graph','snapshot':mathpin,'main-snapshot':mainpin}
    start_url=origin+'/dependencies.html?'+urlencode(query)+'#current-atlas'
    report={'mode':'local-formal-source-connection-acceptance','identity':{'observer':'Actual operator must be identified in its execution receipt; this script does not infer provider/model.','review_credit':False,'scientific_credit':False},
            'site':str(args.site),'frozen_payload':str(args.payload),'payload':identity(args.payload),
            'payload_repositories':data['repositories'],'graph_identity':data['graph']['identity'],
            'observed_at':data['observed_at'],'selected':NODE,'node':node,
            'tested_site_modules':modules,'driver':{'path':str(Path(__file__).resolve()),**identity(Path(__file__))},
            'browser_executable':{'path':args.browser,**identity(Path(args.browser))},
            'browser_command_version':subprocess.check_output([args.browser,'--version'],text=True).strip(),
            'platform':platform.platform(),'playwright':importlib.metadata.version('playwright'),
            'chromium_sandbox':True,'reduced_motion':'reduce','network':'loopback only application journeys; inherited normal proxy/CA retained',
            'csp':'unmodified site policy; no bypass_csp, routes, security-disable flags, or injected application code',
            'scope':'Four matched formal-source reader journeys, real links/keyboard/history and refused selection; no AT or scientific acceptance.',
            'started_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'cases':[]}
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(executable_path=args.browser,chromium_sandbox=True,
                                      args=['--disable-background-networking'])
            report['browser_version']=browser.version
            for width,height in [(1200,900),(390,844)]:
                for scheme in ['light','dark']:
                    row={'case':f'{width}x{height}-{scheme}','viewport':{'width':width,'height':height},
                         'color_scheme':scheme,'checks':[],'failures':[],'pageerrors':[],'console_errors':[],
                         'failed_requests':[],'http_errors':[],'screenshots':{}}
                    report['cases'].append(row)
                    context=browser.new_context(viewport=row['viewport'],color_scheme=scheme,reduced_motion='reduce')
                    page=context.new_page(); page.set_default_timeout(15000)
                    page.on('pageerror',lambda err,r=row:r['pageerrors'].append(str(err)))
                    page.on('console',lambda msg,r=row:r['console_errors'].append(msg.text) if msg.type=='error' else None)
                    page.on('requestfailed',lambda req,r=row:r['failed_requests'].append({'url':req.url,'failure':req.failure}))
                    page.on('response',lambda res,r=row:r['http_errors'].append({'url':res.url,'status':res.status}) if res.status>=400 else None)
                    try:
                        started=time.perf_counter()
                        response=page.goto(start_url,wait_until='load')
                        expect(page.locator('#atlas-inspector h3')).to_have_text(NODE)
                        expect(page.locator('#atlas-formal-evidence')).to_be_attached()
                        row['ready_ms']=(time.perf_counter()-started)*1000
                        row['initial_url']=page.url
                        row['navigation_status']=response.status
                        row['performance_initial']=page.evaluate(PERFORMANCE)
                        nav=row['performance_initial']['navigation'][0]
                        resources=row['performance_initial']['resources']
                        row['load_metrics']={'load_event_ms':nav['loadEventEnd'],'dom_content_loaded_ms':nav['domContentLoadedEventEnd'],
                                             'resource_count':len(resources),
                                             'resource_transfer_bytes':sum(r['transferSize'] for r in resources),
                                             'resource_encoded_body_bytes':sum(r['encodedBodySize'] for r in resources),
                                             'resource_decoded_body_bytes':sum(r['decodedBodySize'] for r in resources),
                                             'document_transfer_bytes':nav['transferSize'],'document_encoded_body_bytes':nav['encodedBodySize']}
                        if width<721:
                            t=time.perf_counter(); page.locator('#atlas-open-reader').click()
                            expect(page.locator('#atlas-inspector')).to_be_visible()
                            row['open_reader_ms']=(time.perf_counter()-t)*1000
                        inspector=page.locator('#atlas-inspector')
                        source=inspector.locator('a',has_text='Immutable source reference').first
                        expect(source).to_be_visible()
                        row['source']={'href':source.get_attribute('href'),'text':source.inner_text(),'visible':source.is_visible()}
                        row['scope']={'text':page.locator('#atlas-claim-summary > p').first.inner_text(),
                                      'visible':page.locator('#atlas-claim-summary > p').first.is_visible()}
                        row['formal_gap_text']=page.locator('#atlas-formal-evidence > p').first.inner_text()
                        signature=inspect_signature(page); row['initial_signature']=signature
                        record_check(row,'full literal scope visible',row['scope']['visible'] and row['scope']['text']==node['scope'])
                        expected_source=f'https://github.com/{data["graph"]["identity"]["repository"]}/blob/{mathpin}/{node["source"]}'
                        record_check(row,'immutable selected proof source visible',row['source']['visible'] and row['source']['href']==expected_source)
                        formal=page.locator('#atlas-formal-details')
                        record_check(row,'native formal disclosure initially closed',not formal.evaluate('(e)=>e.open'))
                        row['geometry_before']=page.evaluate(GEOMETRY)
                        inspector.evaluate('(e)=>e.scrollIntoView({block:"start",behavior:"auto"})')
                        row['geometry_reader_viewport']=page.evaluate(GEOMETRY)
                        for kind,locator in [('reader',None),('full-inspector',inspector)]:
                            name=f'{kind}-{row["case"]}.png'; target=args.output/name
                            if locator is None: page.screenshot(path=str(target))
                            else: locator.screenshot(path=str(target))
                            row['screenshots'][kind]=name
                        summary=formal.locator(':scope > summary')
                        summary.focus(); expect(summary).to_be_focused()
                        row['formal_focus_before_enter']=summary.evaluate('(e)=>({tag:e.tagName,focused:document.activeElement===e,outlineWidth:getComputedStyle(e).outlineWidth,outlineStyle:getComputedStyle(e).outlineStyle})')
                        row['geometry_focused_formal']=page.evaluate(GEOMETRY)
                        started=time.perf_counter()
                        page.keyboard.press('Enter'); expect(formal).to_have_attribute('open','')
                        row['formal_enter_to_open_ms']=(time.perf_counter()-started)*1000
                        row['formal_fields']=formal.locator(':scope > dl > dt').evaluate_all('(els)=>els.map(e=>({name:e.textContent,value:e.nextElementSibling.textContent}))')
                        row['formal_disclosure_text']=formal.inner_text()
                        row['formal_disclosure_links']=formal.locator('a').evaluate_all('(els)=>els.map(e=>({text:e.textContent,href:e.href}))')
                        connection=page.locator('#atlas-formal-connection')
                        expect(connection).to_be_visible(timeout=1000)
                        expect(connection).to_contain_text('UniversalLaw.Side24.pairing_sum_six')
                        expect(connection).to_contain_text('theorem pairing_sum_six : pairingSum 6 = 76')
                        expect(formal).to_contain_text('PENDING_INDEPENDENT_REVIEW')
                        expect(formal).to_contain_text('REJECTED')
                        expect(page.locator('#atlas-formal-evidence > p').first).to_contain_text('does not prove the coefficient or parent theorem')
                        row['connection_name']=connection.locator('p').first.inner_text()
                        record_check(row,'adverse package review remains explicit', 'REJECTED' in formal.inner_text())
                        page.screenshot(path=str(args.output/f'formal-first-viewport-{row["case"]}.png'))
                        link=connection.get_by_role('link',name='Original execution receipt (JSON)',exact=True)
                        receipt_page=context.new_page();receipt_response=receipt_page.goto(link.get_attribute('href'))
                        receipt_bytes=receipt_response.body();receipt_page.close()
                        row['receipt_sha256']=digest(receipt_bytes)
                        record_check(row,'actual original receipt link returns exact bytes',row['receipt_sha256']=='1ae8341c86a043ba917ec1fe68bf7e66ec528cd94b15170528487009f090c7e0')
                        type_detail=page.locator('#atlas-formal-type-details');type_detail.locator('summary').focus();page.keyboard.press('Enter')
                        expect(type_detail).to_have_attribute('open','')
                        expect(type_detail).to_contain_text('@Eq Nat')
                        record_check(row,'original elaborated type is keyboard readable',type_detail.locator('pre').is_visible())
                        row['geometry_expanded_formal']=page.evaluate(GEOMETRY)
                        name=f'expanded-formal-{row["case"]}.png'; formal.screenshot(path=str(args.output/name));row['screenshots']['expanded-formal']=name
                        name=f'expanded-formal-viewport-{row["case"]}.png'; page.screenshot(path=str(args.output/name));row['screenshots']['expanded-formal-viewport']=name
                        row['inspector_text']=inspector.inner_text()
                        # Matched behavior checks, intentionally independent of future formal mapping content.
                        if width<721: page.locator('#atlas-browse-link').click()
                        page.locator('#atlas-view').select_option('list')
                        record_check(row,'Graph/List same claim source scope and pins',inspect_signature(page)==signature)
                        record_check(row,'same-claim layout retains formal disclosure',formal.evaluate('(e)=>e.open'))
                        row['share_url']=page.locator('#atlas-share').get_attribute('href')
                        shared=parse_qs(urlsplit(row['share_url']).query)
                        record_check(row,'share link retains selected source pins',all(shared.get(k)==[v] for k,v in [('atlas-node',NODE),('snapshot',mathpin),('main-snapshot',mainpin)]))
                        # Selection creates native history; restoring must recover the original source.
                        alternate=page.locator('#atlas-list button').filter(has_not_text=NODE).first
                        if alternate.count():
                            alternate_id=alternate.evaluate('(e)=>e.firstChild.textContent')
                            alternate.click(); expect(page.locator('#atlas-inspector h3')).to_have_text(alternate_id)
                            record_check(row,'changed selection resets formal disclosure',not formal.evaluate('(e)=>e.open'))
                            record_check(row,'changed claim removes old arithmetic connection',page.locator('#atlas-formal-connection').count()==0)
                            page.go_back(); expect(page.locator('#atlas-inspector h3')).to_have_text(NODE)
                            record_check(row,'native Back restores same claim source and pins',inspect_signature(page)==signature)
                            page.go_forward(); expect(page.locator('#atlas-inspector h3')).to_have_text(alternate_id)
                            page.go_back(); expect(page.locator('#atlas-inspector h3')).to_have_text(NODE)
                            row['history']={'alternate_node':alternate_id,'back_forward_back':'completed'}
                        else: row['history']={'skipped':'No alternate node in loaded neighborhood'}
                        page.goto(row['share_url']); expect(page.locator('#atlas-inspector h3')).to_have_text(NODE)
                        record_check(row,'share navigation restores exact selection',inspect_signature(page)==signature)
                        page.reload(); expect(page.locator('#atlas-inspector h3')).to_have_text(NODE)
                        record_check(row,'reload restores exact selection',inspect_signature(page)==signature)
                        record_check(row,'reload starts formal disclosure closed',not formal.evaluate('(e)=>e.open'))
                        row['final_geometry']=page.evaluate(GEOMETRY)
                        record_check(row,'no document overflow',page.evaluate('document.documentElement.scrollWidth<=innerWidth'))
                        row['final_url']=page.url
                        refused_query={**query,'main-snapshot':'0'*40}
                        page.goto(origin+'/dependencies.html?'+urlencode(refused_query))
                        expect(page.locator('#atlas-error')).to_be_visible()
                        record_check(row,'refused source removes arithmetic connection',page.locator('#atlas-formal-connection').count()==0)
                    except Exception as err:
                        row['failures'].append(str(err));row['traceback']=traceback.format_exc()
                        try: page.screenshot(path=str(args.output/f'failure-{row["case"]}.png'))
                        except Exception: pass
                    finally:
                        record_check(row,'no pageerrors',not row['pageerrors'])
                        record_check(row,'no failed application requests',not row['failed_requests'])
                        record_check(row,'no HTTP errors',not row['http_errors'])
                        row['passed']=not row['failures']; context.close()
                        (args.output/'formal-browser.json').write_text(json.dumps(report,indent=2)+'\n')
            browser.close()
    finally:
        server.shutdown(); server.server_close()
        (args.output/'server-requests.json').write_text(json.dumps(requests,indent=2)+'\n')
        report['finished_at_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        report['passed']=sum(c.get('passed',False) for c in report['cases']);report['total']=len(report['cases'])
        report['metric_summary']={}
        for key in ['ready_ms','formal_enter_to_open_ms']:
            values=[c[key] for c in report['cases'] if key in c]
            if values: report['metric_summary'][key]={'values':values,'median':statistics.median(values),'min':min(values),'max':max(values)}
        report['external_application_requests']=[r for c in report['cases'] for r in c['failed_requests'] if not r['url'].startswith(origin)]
        (args.output/'formal-browser.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['passed','total','metric_summary']}))
    return 0 if report['passed']==report['total']==4 else 1

if __name__=='__main__':
    raise SystemExit(main())
