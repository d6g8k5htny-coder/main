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
import time
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
    report={'scientific_effect':'NONE','checked_commit':None,'source_payload_sha256':None,'source_repositories':[],'cases':[],'limitations':['Local rendered Chromium evidence; physical mobile and complete screen-reader certification not established.']}
    try:
        report['checked_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        raw=(ROOT/'docs/site/current.json').read_bytes();data=json.loads(raw)
        report['source_payload_sha256']=sha256(raw).hexdigest();report['source_repositories']=data['repositories']
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
            case={'case':'current-reachability-and-equation-search','passed':False};report['cases'].append(case);context=browser.new_context();page=context.new_page();page.set_default_timeout(3000)
            try:
                page.goto(origin+'dependencies.html?atlas-node=math.p15-price-boundary');expect(page.locator('#atlas-reach')).to_contain_text('Downstream nodes · all edges: 1');expect(page.locator('#atlas-reach')).to_contain_text('Downstream nodes · required edges: 0');expect(page.locator('#atlas-inspector')).to_contain_text('REFUTED');expect(page.locator('#atlas-inspector')).to_contain_text('boundary_counterexample_not_required_premise')
                page.locator('#equation-search').fill('ell');expect(page.locator('#equation-count')).to_contain_text('matching');require(page.locator('#equation-select option').count()>1,'Equation text search omitted matching excerpts');expect(page).to_have_url(re.compile(r'.*equation-q=ell.*'));page.reload();expect(page.locator('#equation-search')).to_have_value('ell');case['passed']=True
            except Exception:case['error']=traceback.format_exc()
            finally:context.close()
            case={'case':'keyboard-and-accessibility-tree-reading','passed':False};report['cases'].append(case);context=browser.new_context(reduced_motion='reduce');page=context.new_page();page.set_default_timeout(15000)
            try:
                page.goto(origin+'dependencies.html?atlas-node=math.uniform-matrix-cap-lifetime&atlas-view=list');expect(page.locator('#atlas-inspector h3')).to_have_text('math.uniform-matrix-cap-lifetime')
                page.locator('#atlas-search').focus();page.keyboard.type('CH-LIFT');require(page.locator('#atlas-list button').count()>=1,'Graph search omitted recorded target');page.locator('#atlas-list button').filter(has_text=re.compile(r'^hist\.CH-LIFT')).focus();page.keyboard.press('Enter');expect(page.locator('#atlas-inspector h3')).to_have_text('hist.CH-LIFT');require(page.evaluate('document.activeElement.id')=='atlas-inspector','Keyboard selection did not move reading focus');expect(page.locator('#atlas-reach')).to_contain_text('all edges: 3');expect(page.locator('#atlas-reach')).to_contain_text('required edges: 0')
                page.locator('#atlas-back').focus();page.keyboard.press('Enter');expect(page.locator('#atlas-inspector h3')).to_have_text('math.uniform-matrix-cap-lifetime');page.locator('#atlas-forward').focus();page.keyboard.press('Enter');expect(page.locator('#atlas-inspector h3')).to_have_text('hist.CH-LIFT')
                page.locator('#atlas-reset').focus();page.keyboard.press('Enter');page.locator('#atlas-expand').focus();page.keyboard.press('Enter');page.locator('#atlas-collapse').focus();page.keyboard.press('Enter');page.locator('#atlas-toggle-view').focus();page.keyboard.press('Enter');require(page.locator('#atlas-view').input_value()=='graph','Keyboard Graph switch failed');page.keyboard.press('Enter');require(page.locator('#atlas-view').input_value()=='list','Keyboard List switch failed');page.locator('#atlas-focus').focus();page.keyboard.press('Enter');require(page.evaluate('document.activeElement.id')=='atlas-inspector','Return-to-focus did not focus inspector')
                disclosure=page.locator('#atlas-inspector summary').last;disclosure.focus();page.keyboard.press('Enter');require(page.locator('#atlas-inspector details').last.evaluate('(e)=>e.open'),'Keyboard source disclosure failed');page.locator('#equation-search').focus();page.keyboard.type('ell');page.locator('#equation-next').focus();page.keyboard.press('Enter');expect(page.locator('#equation-content pre')).to_have_count(1)
                diagnostic=page.locator('details').filter(has=page.locator('#atlas-diagnostics'));diagnostic.locator('summary').focus();page.keyboard.press('Enter');diagnostic.locator('button').first.focus();page.keyboard.press('Enter');expect(page.locator('#atlas-inspector h3')).to_have_text(next(iter(data['graph']['data']['nodes'])))
                focus=page.locator('#atlas-focus');focus.focus();style=focus.evaluate('(e)=>({outline:getComputedStyle(e).outlineStyle,width:parseFloat(getComputedStyle(e).outlineWidth),height:e.getBoundingClientRect().height,background:getComputedStyle(e).backgroundColor,color:getComputedStyle(e).color})');require(style['outline']!='none' and style['width']>=2,'Keyboard focus is not visibly outlined');require(style['height']>=44,'Primary keyboard control is below touch target');case['focus_style']=style
                tree=page.locator('#atlas-inspector').aria_snapshot();require('Kernel metadata not recorded' in tree and 'source' in tree.lower(),'Structured reading alternative omitted source/evidence');path=args.output/'inspector-accessibility-tree.txt';path.write_text(tree);case['accessibility_tree']={'path':path.name,'sha256':sha256(path.read_bytes()).hexdigest()};case['limitation']='Chromium accessibility tree and keyboard reading observed; spoken assistive-technology session not observed.';case['passed']=True
            except Exception:case['error']=traceback.format_exc()
            finally:context.close()
            case={'case':'share-after-browser-back-and-forward','passed':False,'checkpoints':[],'page_errors':[]};report['cases'].append(case);context=browser.new_context(reduced_motion='reduce');page=context.new_page();page.set_default_timeout(15000);page.on('pageerror',lambda e:case['page_errors'].append(str(e)))
            try:
                from urllib.parse import urlsplit,parse_qs,urljoin
                pins={'snapshot':[data['graph']['identity']['commit']],'main-snapshot':[next(r['commit'] for r in data['repositories'] if r['repository'].endswith('/main'))]}
                first_equation=data['equations'][0]
                next_equation=next(e for e in data['equations'] if e['id']!=first_equation['id'] and 'ell' in e['text'].lower())
                def settled_selection(tab,label,node,view,graph_query,source_query,equation_query,equation,expected_url=None):
                    # Wait on restored visible controls and exact excerpt before taking
                    # a fresh Share locator reading after two browser paint frames.
                    expect(tab.locator('#atlas-inspector h3')).to_have_text(node)
                    expect(tab.locator('#atlas-view')).to_have_value(view)
                    expect(tab.locator('#atlas-search')).to_have_value(graph_query)
                    expect(tab.locator('.current-search')).to_have_value(source_query)
                    expect(tab.locator('#equation-search')).to_have_value(equation_query)
                    expect(tab.locator('#equation-select')).to_have_value(equation['id'])
                    expect(tab.locator('#equation-content pre')).to_have_text(equation['text'])
                    expect(tab.locator('#atlas-inspector')).to_contain_text(pins['snapshot'][0])
                    expect(tab.locator('#equation-content')).to_contain_text(equation['commit'])
                    tab.evaluate('new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))')
                    address=tab.url;shared=urljoin(address,tab.locator('#atlas-share').get_attribute('href'))
                    actual=parse_qs(urlsplit(address).query);share_state=parse_qs(urlsplit(shared).query)
                    checkpoint={'step':label,'inspector_node':tab.locator('#atlas-inspector h3').inner_text(),'address_url':address,'share_url':shared,'address_state':actual,'share_state':share_state,'source_payload_sha256':report['source_payload_sha256']};case['checkpoints'].append(checkpoint)
                    expected={'atlas-node':[node],'atlas-view':[view],'atlas-q':[graph_query],'source-q':[source_query],'equation-q':[equation_query],'equation':[equation['id']],**pins}
                    for key,value in expected.items():require(actual.get(key)==value,f'{label}: address {key} differs from settled reading: {actual.get(key)!r} != {value!r}')
                    if expected_url is not None:require(actual==parse_qs(urlsplit(expected_url).query),f'{label}: browser history did not restore the complete saved address state')
                    require(urlsplit(shared)._replace(query='')==urlsplit(address)._replace(query=''),f'{label}: Share changed the current page or fragment')
                    require(share_state==actual,f'{label}: fresh Share state disagrees with settled inspector/address: {share_state!r} != {actual!r}')
                    require(tab.locator('#current-research').get_attribute('data-current-digest')==report['source_payload_sha256'],f'{label}: current source identity changed')
                    return address,shared
                page.goto(origin+'dependencies.html?atlas-node=math.uniform-matrix-cap-lifetime&atlas-view=list');expect(page.locator('#atlas-inspector h3')).to_have_text('math.uniform-matrix-cap-lifetime')
                page.locator('#current-research summary').click();page.locator('.current-search').fill('universality')
                page.locator('#equation-search').fill(first_equation['path']);page.locator('#equation-select').select_option(first_equation['id'])
                page.locator('#atlas-search').fill('CH-LIFT');expect(page.locator('#atlas-list button').filter(has_text=re.compile(r'^hist\.CH-LIFT'))).to_have_count(1)
                before,initial_share=settled_selection(page,'before-select','math.uniform-matrix-cap-lifetime','list','CH-LIFT','universality',first_equation['path'],first_equation)
                # Actual node activation creates the second browser history entry.
                page.locator('#atlas-list button').filter(has_text=re.compile(r'^hist\.CH-LIFT')).click();expect(page.locator('#atlas-inspector h3')).to_have_text('hist.CH-LIFT')
                page.locator('#atlas-view').select_option('graph');page.locator('#atlas-search').fill('witness-collision');page.locator('.current-search').fill('frontiers')
                page.locator('#equation-search').fill('ell');page.locator('#equation-select').select_option(next_equation['id'])
                after,selected_share=settled_selection(page,'after-select','hist.CH-LIFT','graph','witness-collision','frontiers','ell',next_equation)
                page.locator('#atlas-back').click()
                restored,back_share=settled_selection(page,'after-back','math.uniform-matrix-cap-lifetime','list','CH-LIFT','universality',first_equation['path'],first_equation,before)
                page.locator('#atlas-forward').click()
                restored,forward_share=settled_selection(page,'after-forward','hist.CH-LIFT','graph','witness-collision','frontiers','ell',next_equation,after)
                for label,shared,node,view,graph_query,source_query,equation_query,equation in [
                    ('reopened-back-share',back_share,'math.uniform-matrix-cap-lifetime','list','CH-LIFT','universality',first_equation['path'],first_equation),
                    ('reopened-forward-share',forward_share,'hist.CH-LIFT','graph','witness-collision','frontiers','ell',next_equation)]:
                    reopened=context.new_page();reopened.on('pageerror',lambda e:case['page_errors'].append(str(e)))
                    try:
                        with reopened.expect_response(lambda response:urlsplit(response.url).path.endswith('/current.json')) as acquired:reopened.goto(shared)
                        acquired_raw=acquired.value.body();require(sha256(acquired_raw).hexdigest()==report['source_payload_sha256'],f'{label}: reopening substituted another source payload');require(json.loads(acquired_raw)['repositories']==data['repositories'],f'{label}: reopening changed exact repository pins')
                        settled_selection(reopened,label,node,view,graph_query,source_query,equation_query,equation,shared)
                    finally:reopened.close()
                require(not case['page_errors'],f'Page errors: {case["page_errors"]}');case['passed']=True
            except Exception:case['error']=traceback.format_exc()
            finally:context.close()
            case={'case':'share-history-refusal-and-legacy-guards','passed':False,'guards':[],'page_errors':[]};report['cases'].append(case)
            try:
                from urllib.parse import urlsplit,parse_qs,urljoin,urlencode
                current_node='math.uniform-matrix-cap-lifetime';equation=data['equations'][0]
                pins={'snapshot':[data['graph']['identity']['commit']],'main-snapshot':[next(r['commit'] for r in data['repositories'] if r['repository'].endswith('/main'))]}
                def history_checkpoint(tab,label):
                    tab.evaluate('new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))')
                    address=tab.url;shared=urljoin(address,tab.locator('#atlas-share').get_attribute('href'))
                    return {'step':label,'address_url':address,'share_url':shared,'address_state':parse_qs(urlsplit(address).query),'share_state':parse_qs(urlsplit(shared).query),'history_length':tab.evaluate('history.length')}
                def valid_share(checkpoint,node):
                    for key,value in {'atlas-node':[node],**pins}.items():require(checkpoint['share_state'].get(key)==value,f'{checkpoint["step"]}: Share changed restored {key}')
                # A legacy address is an ordinary initial navigation; selecting a
                # visible recorded neighbor supplies the only new history entry.
                guard={'bookmark':'legacy-filters-and-fragment','passed':False,'checkpoints':[]};case['guards'].append(guard);context=browser.new_context(reduced_motion='reduce');page=context.new_page();page.set_default_timeout(15000);page.on('pageerror',lambda e:case['page_errors'].append(str(e)))
                try:
                    legacy=origin+'dependencies.html?q=math&classification=PROVED_REVIEWED#search-heading';page.goto(legacy);expect(page.locator('#atlas-inspector h3')).to_have_text(current_node)
                    initial_length=page.evaluate('history.length');neighbor=next(e['to'] for e in data['graph']['data']['edges'] if e['from']==current_node)
                    page.locator('#atlas-inspector button').filter(has_text=re.compile('^'+re.escape(neighbor))).first.click();expect(page.locator('#atlas-inspector h3')).to_have_text(neighbor)
                    selected=history_checkpoint(page,'legacy-selected-neighbor');guard['checkpoints'].append(selected);valid_share(selected,neighbor);require(selected['history_length']==initial_length+1,'Legacy neighbor selection did not create exactly one history entry')
                    page.locator('#atlas-back').click();expect(page.locator('#atlas-inspector h3')).to_have_text(current_node)
                    restored=history_checkpoint(page,'legacy-after-back');guard['checkpoints'].append(restored);require(restored['address_url']==legacy,'Back changed exact historical address or fragment');require(restored['address_state']=={'q':['math'],'classification':['PROVED_REVIEWED']},'Back changed historical filter semantics');valid_share(restored,current_node)
                    require(restored['share_state'].get('q')==['math'] and restored['share_state'].get('classification')==['PROVED_REVIEWED'],'Share dropped restored historical filters');require(urlsplit(restored['share_url']).fragment=='current-atlas','Legacy Share omitted its valid atlas selection');require(restored['history_length']==selected['history_length'],'Legacy Back restoration added a history entry')
                    page.locator('#atlas-forward').click();expect(page.locator('#atlas-inspector h3')).to_have_text(neighbor);forward=history_checkpoint(page,'legacy-after-forward');guard['checkpoints'].append(forward);require(forward['address_url']==selected['address_url'],'Legacy Forward did not restore the explicit neighbor entry');valid_share(forward,neighbor);require(forward['history_length']==selected['history_length'],'Legacy Forward restoration added a history entry');guard['passed']=True
                finally:context.close()
                failures=[]
                for label,query,error_text in [
                    ('wrong-main-snapshot',{'atlas-node':current_node,'main-snapshot':'c'*40},'another snapshot'),
                    ('unknown-node',{'atlas-node':'missing.bookmark'},'Unknown node'),
                    ('prototype-constructor',{'atlas-node':'constructor'},'Unknown node'),
                    ('prototype-__proto__',{'atlas-node':'__proto__'},'Unknown node')]:
                    guard={'bookmark':label,'passed':False,'checkpoints':[]};case['guards'].append(guard);context=browser.new_context(reduced_motion='reduce');page=context.new_page();page.set_default_timeout(15000);page.on('pageerror',lambda e:case['page_errors'].append(str(e)))
                    try:
                        refused=origin+'dependencies.html?'+urlencode({**query,'atlas-view':'list','equation':equation['id']})+'#current-atlas';page.goto(refused);expect(page.locator('#atlas-error')).to_contain_text(error_text);expect(page.locator('#atlas-canvas .atlas-node')).to_have_count(0);expect(page.locator('#equation-content pre')).to_have_count(0)
                        diagnostic=page.locator('details').filter(has=page.locator('#atlas-diagnostics'));diagnostic.locator('summary').click();diagnostic.locator('button').filter(has_text=re.compile('^'+re.escape(current_node)+':')).click();expect(page.locator('#atlas-inspector h3')).to_have_text(current_node)
                        # Explicitly choosing this current excerpt is also user
                        # action; the refused bookmark must not supply it silently.
                        page.locator('#equation-select').select_option(equation['id']);expect(page.locator('#equation-content pre')).to_have_text(equation['text']);selected=history_checkpoint(page,label+'-explicit-current');guard['checkpoints'].append(selected);valid_share(selected,current_node)
                        page.locator('#atlas-back').click();expect(page.locator('#atlas-error')).to_contain_text(error_text);expect(page.locator('#atlas-inspector h3')).to_have_count(0);expect(page.locator('#atlas-canvas .atlas-node')).to_have_count(0);expect(page.locator('#atlas-list button')).to_have_count(0);expect(page.locator('#equation-content pre')).to_have_count(0);expect(page.locator('#equation-content')).to_contain_text('Source-bound bookmark refused')
                        restored=history_checkpoint(page,label+'-after-back');guard['checkpoints'].append(restored);require(restored['address_url']==refused,f'{label}: Back changed the exact refused identity');require(restored['history_length']==selected['history_length'],f'{label}: Back added a history entry')
                        guard['refused_share_matches']=restored['share_url']==refused
                        if not guard['refused_share_matches']:failures.append(f'{label}: fresh Share after Back substituted the explicit current selection for the exact refused URL')
                        page.locator('#atlas-forward').click();expect(page.locator('#atlas-error')).to_be_hidden();expect(page.locator('#atlas-inspector h3')).to_have_text(current_node);expect(page.locator('#equation-content pre')).to_have_text(equation['text']);forward=history_checkpoint(page,label+'-after-forward');guard['checkpoints'].append(forward);require(forward['address_url']==selected['address_url'],f'{label}: Forward did not restore the explicitly selected current entry');valid_share(forward,current_node);require(forward['share_state'].get('equation')==[equation['id']],f'{label}: Forward Share omitted the explicit current excerpt');require(forward['history_length']==selected['history_length'],f'{label}: Forward added a history entry');guard['passed']=guard['refused_share_matches']
                    except Exception:guard['error']=traceback.format_exc();failures.append(label+': browser guard failed')
                    finally:context.close()
                case['guard_failures']=failures;require(not case['page_errors'],f'Page errors: {case["page_errors"]}');require(not failures,'; '.join(failures));case['passed']=True
            except Exception:case['error']=traceback.format_exc()
            case={'case':'enlarged-text-and-monochrome','passed':False};report['cases'].append(case)
            try:
                for width in [320,1200]:
                    context=browser.new_context(viewport={'width':width,'height':900},forced_colors='active',reduced_motion='reduce');page=context.new_page();page.goto(origin+'dependencies.html?atlas-node=math.p15-full-price&atlas-view=list');expect(page.locator('#atlas-inspector h3')).to_have_text('math.p15-full-price');page.evaluate("(()=>{const entries=[document.documentElement,document.body,...document.querySelectorAll('body *')].map(e=>[e,parseFloat(getComputedStyle(e).fontSize)]);for(const [e,n] of entries)e.style.setProperty('font-size',n*2+'px','important');})()");page.locator('#atlas-reset').focus();page.keyboard.press('Enter');dimensions=page.evaluate('({width:innerWidth,document:document.documentElement.scrollWidth,font:parseFloat(getComputedStyle(document.body).fontSize)})');require(dimensions['font']>=30,'Text did not enlarge to 200 percent');require(dimensions['document']<=dimensions['width'],f'Enlarged text document overflow: {dimensions}');require(page.locator('#atlas-next').is_visible() and page.locator('#equation-search').is_visible(),'Essential controls lost after enlargement');page.locator('#atlas-view').select_option('graph');require(page.locator('.atlas-edge.context').first.evaluate('(e)=>getComputedStyle(e).strokeDasharray') not in ['none',''],'Context edge loses monochrome pattern');expect(page.locator('#atlas-count')).to_contain_text('Solid arrows: required; dashed arrows: context');path=args.output/f'enlarged-{width}.png';page.locator('#current-atlas').screenshot(path=str(path));context.close()
                case['passed']=True
            except Exception:case['error']=traceback.format_exc()
            case={'case':'synthetic-large-render-window','passed':False,'scope':'Synthetic display fixture only; does not describe the research corpus or source acquisition.'};report['cases'].append(case);context=browser.new_context();page=context.new_page()
            try:
                synthetic=json.loads(json.dumps(data));nodes={'fixture.root':{'kind':'candidate','classification':'OPEN_ACTIVE','layer':'D1'}};nodes.update({f'fixture.node{i:04d}':{'kind':'candidate','classification':'OPEN_ACTIVE','layer':'D1'} for i in range(1000)});synthetic['graph']['data']['nodes']=nodes;synthetic['graph']['data']['edges']=[{'from':'fixture.root','to':node,'required':True,'relation':'synthetic_required_fixture'} for node in nodes if node!='fixture.root'];fixture_raw=json.dumps(synthetic,ensure_ascii=False).encode();fixture_sha=sha256(fixture_raw).hexdigest();html=(ROOT/'docs/site/dependencies.html').read_text().replace(report['source_payload_sha256'],fixture_sha).replace(f'data-current-bytes="{len(raw)}"',f'data-current-bytes="{len(fixture_raw)}"');page.route('**/dependencies.html?*',lambda route:route.fulfill(status=200,content_type='text/html',body=html));page.route('**/current.json',lambda route:route.fulfill(status=200,content_type='application/json',body=fixture_raw));started=time.perf_counter();page.goto(origin+'dependencies.html?atlas-node=fixture.root&atlas-view=list');expect(page.locator('#atlas-list button')).to_have_count(150);expect(page.locator('#atlas-count')).to_contain_text('1001 nodes in the loaded neighborhood');expect(page.locator('#atlas-count')).to_contain_text('851 outside this page');first=page.locator('#atlas-list button').evaluate_all('(nodes)=>nodes.map(n=>n.firstChild.textContent)');page.locator('#atlas-next').click();expect(page.locator('#atlas-count')).to_contain_text('851 outside this page');second=page.locator('#atlas-list button').evaluate_all('(nodes)=>nodes.map(n=>n.firstChild.textContent)');require(not set(first)&set(second),'Paged large fixture duplicated the first window');page.locator('#atlas-previous').click();require(first==page.locator('#atlas-list button').evaluate_all('(nodes)=>nodes.map(n=>n.firstChild.textContent)'),'Previous window is not deterministic');case['load_and_page_seconds']=round(time.perf_counter()-started,3);session=context.new_cdp_session(page);session.send('Performance.enable');case['js_heap_bytes']=next((m['value'] for m in session.send('Performance.getMetrics')['metrics'] if m['name']=='JSHeapUsedSize'),None);case['fixture_bytes']=len(fixture_raw);case['fixture_sha256']=fixture_sha;case['indexed_nodes']=len(nodes);case['rendered_nodes']=150;case['passed']=True
            except Exception:case['error']=traceback.format_exc()
            finally:context.close()
            case={'case':'no-javascript-static-parity','passed':False};report['cases'].append(case);context=browser.new_context(java_script_enabled=False);page=context.new_page()
            try:
                page.goto(origin+'nodes/index.html');links=page.locator('main a[href]').all();require(len(links)>=len(data['graph']['data']['nodes']),'Static index omitted nodes');node='math.uniform-matrix-cap-lifetime';page.locator('main a[href="'+node+'.html"]').click();expect(page.locator('main')).to_contain_text('kernel');expect(page.locator('main')).to_contain_text('not_recorded');edge=next(e for e in data['graph']['data']['edges'] if e['from']==node);expect(page.locator('main')).to_contain_text(edge['relation']);page.locator('main a[href="'+edge['to']+'.html"]').first.click();require(page.url==origin+'nodes/'+edge['to']+'.html','Current static neighbor left the selected current observation');expect(page.locator('main h1')).to_have_text(edge['to']);tree=page.locator('main').aria_snapshot();require('Typed neighbors' in tree and 'not_recorded' in tree,'Static accessible reading omitted typed relationships or evidence');path=args.output/'static-accessibility-tree.txt';path.write_text(tree);case['accessibility_tree']={'path':path.name,'sha256':sha256(path.read_bytes()).hexdigest()};case['passed']=True
            except Exception:case['error']=traceback.format_exc()
            finally:context.close()
            case={'case':'stale-bookmark-refusal','passed':False};report['cases'].append(case);context=browser.new_context();page=context.new_page()
            try:
                from urllib.parse import urlencode
                page.goto(origin+'dependencies.html?'+urlencode({'main-snapshot':'c'*40,'equation':data['equations'][0]['id']}));expect(page.locator('#atlas-error')).to_contain_text('another snapshot');require(page.locator('#atlas-canvas .atlas-node').count()==0,'Stale bookmark substituted current graph');require(page.locator('#equation-content pre').count()==0,'Stale bookmark substituted current equation');case['passed']=True
            except Exception:case['error']=traceback.format_exc()
            finally:context.close()
            case={'case':'legacy-bookmark-preservation','passed':False};report['cases'].append(case);context=browser.new_context();page=context.new_page()
            try:
                from urllib.parse import urlsplit,parse_qs
                page.goto(origin+'dependencies.html?q=math&classification=PROVED_REVIEWED#search-heading');expect(page.locator('#atlas-status')).to_contain_text('Source snapshot');require(parse_qs(urlsplit(page.url).query)=={'q':['math'],'classification':['PROVED_REVIEWED']},'Atlas startup changed historical viewer URL filters');require(urlsplit(page.url).fragment=='search-heading','Atlas startup stole historical fragment');case['passed']=True
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
