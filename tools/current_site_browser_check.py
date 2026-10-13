#!/usr/bin/env python3
"""Exercise the current mathematical atlas; write actual browser evidence."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from functools import partial
from hashlib import sha256
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
import json
import re
from pathlib import Path
import subprocess
import threading
import traceback

ROOT=Path(__file__).resolve().parents[1]
@contextmanager
def serve():
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),partial(Handler,directory=str(ROOT/'docs')))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:yield f'http://127.0.0.1:{server.server_port}/site/'
    finally:server.shutdown();server.server_close();thread.join()
def require(condition,message):
    if not condition:raise ValueError(message)
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    raw=(ROOT/'docs/site/current.json').read_bytes();data=json.loads(raw)
    report={'scientific_effect':'NONE','checked_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_payload_sha256':sha256(raw).hexdigest(),'source_repositories':data['repositories'],'cases':[],'limitations':['Local rendered Chromium evidence; physical mobile and complete screen-reader certification not established.']}
    try:
        from playwright.sync_api import sync_playwright,expect
        with serve() as origin,sync_playwright() as p:
            browser=p.chromium.launch(channel='chrome',chromium_sandbox=True);report['browser']=browser.version
            for width in [1200,320]:
                for scheme in ['light','dark']:
                    case={'width':width,'color_scheme':scheme,'passed':False};report['cases'].append(case)
                    context=browser.new_context(viewport={'width':width,'height':900},color_scheme=scheme,reduced_motion='reduce');page=context.new_page();page.set_default_timeout(20000);errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
                    try:
                        page.goto(origin+'dependencies.html');expect(page.locator('#atlas-status')).to_contain_text('Source snapshot');expect(page.locator('.current-state')).to_contain_text('Verified source observation')
                        page.locator('#current-research summary').click();page.locator('.current-search').fill('universality');expect(page.locator('.current-results')).to_contain_text('experiments/universality');require('source-q=universality' in page.url,'Source search state missing')
                        initial=page.locator('#atlas-canvas .atlas-node').count();page.locator('#atlas-expand').click();expanded=page.locator('#atlas-canvas .atlas-node').count();require(expanded>=initial,'Expansion lost nodes')
                        graph_ids=page.locator('#atlas-canvas .atlas-node').evaluate_all('(nodes)=>nodes.map(n=>n.firstChild.textContent)');page.locator('#atlas-view').select_option('list');list_ids=page.locator('#atlas-list button').evaluate_all('(nodes)=>nodes.map(n=>n.firstChild.textContent)');require(graph_ids==list_ids,'Graph/List mismatch')
                        page.locator('#atlas-list button').last.click();expect(page.locator('#atlas-inspector')).to_contain_text('Kernel metadata not recorded');expect(page.locator('#atlas-inspector')).to_contain_text('No explicit graph-to-artifact binding')
                        node=page.locator('#atlas-inspector h3').inner_text();page.reload();expect(page.locator('#atlas-inspector h3')).to_have_text(node)
                        eq=data['equations'][0];page.locator('#equation-select').select_option(eq['id']);expect(page.locator('#equation-content pre')).to_have_text(eq['text']);require('equation=' in page.url,'Equation URL state absent');page.reload();expect(page.locator('#equation-content pre')).to_have_text(eq['text'])
                        page.locator('#atlas-search').fill('witness-collision');expect(page).to_have_url(re.compile(r'.*atlas-q=witness-collision.*'));page.reload();expect(page.locator('#atlas-search')).to_have_value('witness-collision');page.locator('#atlas-collapse').click();page.locator('#atlas-view').select_option('graph');page.locator('#atlas-zoom-in').focus();page.keyboard.press('Enter');require(page.locator('#atlas-canvas').evaluate('(e)=>e.style.width')!='','Keyboard zoom failed')
                        dimensions=page.evaluate('({width:innerWidth,document:document.documentElement.scrollWidth})');require(dimensions['document']<=dimensions['width'],f'Horizontal document overflow: {dimensions}')
                        shot=args.output/f'{width}-{scheme}.png';page.locator('#current-atlas').screenshot(path=str(shot));case['screenshot']={'path':shot.name,'sha256':sha256(shot.read_bytes()).hexdigest()};case['loaded_nodes']=expanded;case['page_errors']=errors;require(not errors,f'Page errors: {errors}');case['passed']=True
                    except Exception:case['error']=traceback.format_exc()
                    finally:context.close()
            case={'case':'no-javascript-static-parity','passed':False};report['cases'].append(case);context=browser.new_context(java_script_enabled=False);page=context.new_page()
            try:
                page.goto(origin+'nodes/index.html');links=page.locator('main a[href]').all();require(len(links)>=len(data['graph']['data']['nodes']),'Static index omitted nodes');page.locator('main a[href="'+next(iter(data['graph']['data']['nodes']))+'.html"]').click();expect(page.locator('main')).to_contain_text('kernel');expect(page.locator('main')).to_contain_text('not_recorded');case['passed']=True
            except Exception:case['error']=traceback.format_exc()
            finally:context.close()
            case={'case':'stale-bookmark-refusal','passed':False};report['cases'].append(case);context=browser.new_context();page=context.new_page()
            try:
                from urllib.parse import urlencode
                page.goto(origin+'dependencies.html?'+urlencode({'main-snapshot':'c'*40,'equation':data['equations'][0]['id']}));expect(page.locator('#atlas-error')).to_contain_text('another snapshot');require(page.locator('#atlas-canvas .atlas-node').count()==0,'Stale bookmark substituted current graph');require(page.locator('#equation-content pre').count()==0,'Stale bookmark substituted current equation');case['passed']=True
            except Exception:case['error']=traceback.format_exc()
            finally:context.close()
            case={'case':'unknown-prototype-node-refusal','passed':False};report['cases'].append(case);context=browser.new_context();page=context.new_page()
            try:
                for node in ['constructor','toString','__proto__']:
                    page.goto(origin+'dependencies.html?atlas-node='+node);expect(page.locator('#atlas-error')).to_contain_text('Unknown node');require(page.locator('#atlas-canvas .atlas-node').count()==0,'Fabricated inherited node rendered')
                case['passed']=True
            except Exception:case['error']=traceback.format_exc()
            finally:context.close()
            case={'case':'static-mobile-reading','passed':False};report['cases'].append(case);context=browser.new_context(java_script_enabled=False,viewport={'width':320,'height':900});page=context.new_page()
            try:
                for path in ['nodes/index.html','nodes/math.d1-component.section9-replacement-v1_1.html','nodes/math.uniform-matrix-cap-lifetime.html']:
                    page.goto(origin+path);dimensions=page.evaluate('({width:innerWidth,document:document.documentElement.scrollWidth})');require(dimensions['document']<=dimensions['width'],f'Static mobile overflow {path}: {dimensions}')
                case['passed']=True
            except Exception:case['error']=traceback.format_exc()
            finally:context.close()
            case={'case':'digest-refusal','passed':False};report['cases'].append(case);context=browser.new_context();page=context.new_page()
            try:
                page.route('**/current.json',lambda route:route.fulfill(status=200,content_type='application/json',body='{}'));page.goto(origin+'dependencies.html');expect(page.locator('.current-state')).to_contain_text('Unavailable');expect(page.locator('#atlas-status')).to_contain_text('Unavailable');require(page.locator('#atlas-canvas .atlas-node').count()==0,'Refusal rendered graph');case['passed']=True
            except Exception:case['error']=traceback.format_exc()
            finally:context.close();browser.close()
    except Exception:report['error']=traceback.format_exc()
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    passed=not report.get('error') and all(c['passed'] for c in report['cases']);print(f"Current-site browser: {'PASS' if passed else 'FAIL'} ({sum(c['passed'] for c in report['cases'])}/{len(report['cases'])} cases); scientific effect NONE.")
    return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
