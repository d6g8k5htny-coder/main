#!/usr/bin/env python3
"""Check compact source-bound relationships in a real browser."""
import argparse
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import platform
import subprocess
import threading
import time
import traceback

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--site', type=Path, default=ROOT / 'docs/site')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--browser', default='/usr/bin/chromium')
    ap.add_argument('--base-url', help='Actual public site origin instead of local fixture')
    ap.add_argument('--observe', action='store_true')
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *args): pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Handler, directory=str(args.site.parent)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = (args.base_url.rstrip('/') if args.base_url else f'http://127.0.0.1:{server.server_port}/{args.site.name}') + '/dependencies.html'
    raw = (args.site / 'current.json').read_bytes(); data = json.loads(raw)
    graph = data['graph']['data']; selected = 'math.uniform-matrix-cap-lifetime'
    report = {'mode':'observation' if args.observe else 'acceptance', 'origin':origin,
              'payload_sha256':hashlib.sha256(raw).hexdigest(), 'repositories':data['repositories'],
              'checked_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'environment':{'platform':platform.platform(),'browser':args.browser,'sandbox':True,'reduced_motion':True},'cases':[]}
    try:
        with sync_playwright() as p:
            launch={'executable_path':args.browser,'chromium_sandbox':True,
                    'args':['--enable-unsafe-swiftshader','--disable-background-networking']}
            proxy=os.environ.get('HTTPS_PROXY') or os.environ.get('HTTP_PROXY')
            if args.base_url and proxy:launch['proxy']={'server':proxy}
            browser = p.chromium.launch(**launch)
            report['environment']['browser_version'] = browser.version
            for width, height in [(1200,900),(390,844)]:
                for scheme in ['light','dark']:
                    row={'case':'compact-relationships','width':width,'scheme':scheme,'passed':False,'errors':[]};report['cases'].append(row)
                    context=browser.new_context(viewport={'width':width,'height':height},color_scheme=scheme,reduced_motion='reduce')
                    page=context.new_page();page.set_default_timeout(10000)
                    page.on('pageerror',lambda e:row['errors'].append(str(e)))
                    try:
                        start=time.perf_counter();page.goto(origin+'?atlas-node='+selected+'&atlas-view=list')
                        expect(page.locator('#atlas-inspector h3')).to_have_text(selected)
                        row['readiness_ms']=(time.perf_counter()-start)*1000
                        if width<721:page.locator('#atlas-open-reader').click()
                        inspector=page.locator('#atlas-inspector');expect(inspector).to_contain_text(graph['nodes'][selected]['scope'])
                        row['source']=inspector.get_by_role('link',name='Immutable source reference',exact=False).get_attribute('href')
                        row['closed_inspector_height']=inspector.bounding_box()['height']
                        inspector.screenshot(path=str(args.output/f'inspector-{width}-{scheme}.png'))
                        # Baseline records geometry before the new presentation criterion.
                        assert page.locator('#atlas-dependencies').count()==1,'Compact source-bound dependency group missing'
                        assert page.locator('#atlas-dependents').count()==1,'Compact used-by group missing'
                        for group, direction in [('dependencies','from'),('dependents','to')]:
                            edges=[e for e in graph['edges'] if e[direction]==selected]
                            section=page.locator('#atlas-'+group);cards=section.locator('.atlas-relation-card')
                            assert cards.count()==len(edges),'Recorded connection lost or duplicated'
                            assert str(len(edges)) in section.locator('h4').inner_text()
                            for index,edge in enumerate(edges):
                                card=cards.nth(index);target=edge['to'] if direction=='from' else edge['from']
                                button=card.locator('button');assert button.evaluate('(e)=>e.firstChild.textContent')==target
                                assert 'Downstream nodes' not in button.inner_text(),'Closed navigation repeats reach metadata'
                                expect(card.locator('.atlas-relation-kind')).to_contain_text('Required' if edge['required'] else 'Context')
                                detail=card.locator('details');assert not detail.evaluate('(e)=>e.open')
                                detail.locator('summary').focus();page.keyboard.press('Enter');expect(detail).to_have_attribute('open','')
                                expect(detail).to_contain_text(edge['relation']);expect(detail).to_contain_text(edge['from']+' → '+edge['to'])
                                expect(detail).to_contain_text('Downstream nodes')
                                if index==0 and group=='dependencies':
                                    box=detail.locator('summary').bounding_box();assert box['height']>=44,'Disclosure touch target too short'
                                    style=detail.locator('summary').evaluate('(e)=>({style:getComputedStyle(e).outlineStyle,width:parseFloat(getComputedStyle(e).outlineWidth)})')
                                    assert style['style']!='none' and style['width']>=2,'Keyboard focus ring missing'
                                detail.locator('summary').press('Enter');expect(detail).not_to_have_attribute('open','')
                        detail=page.locator('#atlas-dependencies details').first;detail.locator('summary').press('Enter')
                        # Same-source layout rerender retains only this selected node's disclosure.
                        if width<721:page.locator('#atlas-browse-link').click()
                        page.locator('#atlas-view').select_option('graph')
                        expect(page.locator('#atlas-dependencies details').first).to_have_attribute('open','')
                        if width<721:page.locator('#atlas-open-reader').click()
                        assert row['source']==inspector.get_by_role('link',name='Immutable source reference',exact=False).get_attribute('href')
                        first=page.locator('#atlas-dependencies button').first;target=first.evaluate('(e)=>e.firstChild.textContent')
                        first.focus();page.keyboard.press('Enter');expect(inspector.locator('h3')).to_have_text(target);expect(inspector).to_be_focused()
                        assert not inspector.locator('.atlas-relations details[open]').count(),'Changed node inherited relationship disclosure'
                        page.go_back();expect(inspector.locator('h3')).to_have_text(selected)
                        assert not inspector.locator('.atlas-relations details[open]').count(),'History inherited another node disclosure'
                        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),'Document overflow'
                        assert not row['errors'],row['errors'];row['passed']=True
                    except Exception as e:
                        row['failure']=str(e);row['traceback']=traceback.format_exc()
                    finally:context.close()
            for width in [1200,390]:
                context=browser.new_context(viewport={'width':width,'height':900},reduced_motion='reduce')
                page=context.new_page();row={'case':'dense-selection-visible','width':width,'passed':False};report['cases'].append(row)
                try:
                    dense='math.d5-pin-neighborhood-first-moment'
                    page.goto(origin+'?atlas-node='+dense+'&atlas-view=list&atlas-panel=evidence')
                    inspector=page.locator('#atlas-inspector');expect(inspector.locator('h3')).to_have_text(dense)
                    group=page.locator('#atlas-dependencies ul') if page.locator('#atlas-dependencies').count() else inspector.locator(':scope > ul').first
                    row['closed_relationship_height']=group.bounding_box()['height']
                    inspector.screenshot(path=str(args.output/f'dense-{width}.png'))
                    button=group.locator('button').last;target=button.evaluate('(e)=>e.firstChild.textContent')
                    button.focus();page.keyboard.press('Enter');expect(inspector.locator('h3')).to_have_text(target);expect(inspector).to_be_focused()
                    box=inspector.locator('h3').bounding_box();row['selected_heading_box']=box
                    assert 0<=box['y'] and box['y']+box['height']<=900,'Selected heading focused outside viewport'
                    page.keyboard.press('Tab');expect(inspector.get_by_role('link',name='Immutable source reference',exact=False)).to_be_focused()
                    row['passed']=True
                except Exception as e:row['failure']=str(e);row['traceback']=traceback.format_exc()
                context.close()
            context=browser.new_context(viewport={'width':1200,'height':900});page=context.new_page()
            row={'case':'visible-selection-keeps-scroll','passed':False};report['cases'].append(row)
            try:
                page.goto(origin+'?atlas-node='+selected+'&atlas-view=list')
                inspector=page.locator('#atlas-inspector');expect(inspector.locator('h3')).to_have_text(selected)
                inspector.locator('h3').evaluate('(e)=>e.scrollIntoView({block:"center",behavior:"instant"})')
                before=page.evaluate('scrollY');page.locator('#atlas-list button').last.evaluate('(e)=>e.click()')
                expect(inspector).to_be_focused();row['before_scroll']=before;row['after_scroll']=page.evaluate('scrollY')
                assert abs(row['after_scroll']-before)<=1,'Already visible heading caused an unnecessary scroll'
                row['passed']=True
            except Exception as e:row['failure']=str(e)
            context.close()
            context=browser.new_context();page=context.new_page()
            row={'case':'refused-source-clears-relationships','passed':False};report['cases'].append(row)
            try:
                page.goto(origin+'?snapshot='+'0'*40);expect(page.locator('#atlas-error')).to_be_visible()
                assert page.locator('.atlas-relations').count()==0,'Refused bookmark substituted current relationships'
                row['passed']=True
            except Exception as e:row['failure']=str(e)
            context.close();browser.close()
    finally:
        server.shutdown();report['passed']=sum(c['passed'] for c in report['cases']);report['total']=len(report['cases'])
        (args.output/'relationships-browser.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':report['passed'],'total':report['total'],'mode':report['mode']}))
    return 0 if args.observe or report['passed']==report['total'] else 1


if __name__=='__main__':raise SystemExit(main())
