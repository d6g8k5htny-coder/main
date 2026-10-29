#!/usr/bin/env python3
"""Finite F973 browser accessibility audit. Explicit immutable input; fresh output.
Tool injection and text-spacing stylesheet are DEV-only same-origin audit routes.
No input/product writes, source adoption, spoken AT or conformance certification.
"""
import argparse, datetime, hashlib, importlib.metadata, json, os, platform
import threading, traceback, time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from playwright.sync_api import sync_playwright, expect

NOW=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
SELECTED='math.uniform-matrix-cap-lifetime'
SPACING='*{line-height:1.5!important;letter-spacing:.12em!important;word-spacing:.16em!important}p{margin-bottom:2em!important}'
FOCUS='''()=>{const e=document.activeElement,r=e.getBoundingClientRect(),s=getComputedStyle(e),mid=document.elementFromPoint(Math.max(0,Math.min(innerWidth-1,r.x+r.width/2)),Math.max(0,Math.min(innerHeight-1,r.y+r.height/2)));return {tag:e.tagName,id:e.id,text:(e.getAttribute('aria-label')||e.textContent||'').slice(0,180),connected:e.isConnected,focus_visible:e.matches(':focus-visible'),outline:{style:s.outlineStyle,width:s.outlineWidth,color:s.outlineColor,offset:s.outlineOffset},shadow:s.boxShadow,rect:{x:r.x,y:r.y,width:r.width,height:r.height},viewport:{width:innerWidth,height:innerHeight},center_hit:mid?.tagName,center_unobscured:!!mid&&(mid===e||e.contains(mid)),fully_inside:r.x>=0&&r.y>=0&&r.right<=innerWidth&&r.bottom<=innerHeight,outerHTML:e.outerHTML.slice(0,600)};}'''

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True);ap.add_argument('--binding',type=Path,required=True)
    ap.add_argument('--axe',type=Path,required=True);ap.add_argument('--toolchain',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--only',default='')
    ap.add_argument('--expected-observation-main',default='f973ead67ec139dcfef9b97f1758523639c835b3',help='Explicit full40hex main pin for the actual data observation; defaults to this F973 audit')
    ap.add_argument('--target-label',default='F973 frozen public baseline, NOT938/current candidate',help='Explicit product target label; candidate runs require product_source in their own new input binding')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    binding=json.loads(args.binding.read_bytes());assert len(args.expected_observation_main)==40 and all(c in '0123456789abcdef' for c in args.expected_observation_main),'Expected observation pin must be full40hex';assert binding['main']==args.expected_observation_main,'Input observation does not match explicitly selected pin'
    if args.target_label!='F973 frozen public baseline, NOT938/current candidate':assert binding.get('product_source',{}).get('revision'),'A nonbaseline product needs its own explicit product_source.revision binding'
    for row in binding['files']:
        raw=(args.input/row['path']).read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],row['path']
    payload_raw=(args.input/'current.json').read_bytes();assert hashlib.sha256(payload_raw).hexdigest()==binding['payload_sha256'] and len(payload_raw)==binding['payload_bytes'],'Payload identity metadata mismatch'
    payload=json.loads(payload_raw);assert next(r['commit'] for r in payload['repositories'] if r['repository'].endswith('/main'))==args.expected_observation_main,'Payload repository pin mismatch';assert SELECTED in payload['graph']['data']['nodes'],'Audit selected-node contract is absent'
    tc=json.loads(args.toolchain.read_bytes());assert hashlib.sha256(args.axe.read_bytes()).hexdigest()==tc['script_sha256']
    requests=[]
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self,*a):pass
        def do_GET(self):
            requests.append({'path':self.path,'at':NOW()})
            path=urlsplit(self.path).path
            if path in ['/__audit__/axe.min.js','/__audit__/spacing.css']:
                body=args.axe.read_bytes() if path.endswith('.js') else SPACING.encode()
                self.send_response(200);self.send_header('Content-Type','text/javascript' if path.endswith('.js') else 'text/css');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body);return
            return super().do_GET()
    server=ThreadingHTTPServer(('127.0.0.1',0),partial(Handler,directory=str(args.input)))
    threading.Thread(target=server.serve_forever,daemon=True).start()
    origin=f'http://127.0.0.1:{server.server_port}'
    report={'started_at':NOW(),'audit_target':'F973 frozen public baseline, NOT938/current candidate','main':binding['main'],'tree':binding['tree'],'payload_sha256':binding['payload_sha256'],'payload_bytes':binding['payload_bytes'],'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'toolchain':tc,'environment':{'platform':platform.platform(),'python':platform.python_version(),'playwright':importlib.metadata.version('playwright'),'browser_executable':'/usr/bin/chromium','sandbox':True,'renderer':'software SwiftShader','origin':origin,'local_only':True,'reduced_motion':'reduce'},'cases':[],'limits':['Scoped automated and operator browser evidence, not full WCAG conformance certification.','No physical device, spoken AT, assistive technology user or new scientific review.','axe incomplete results need human assessment; automated passes do not prove every criterion.','Prior source/reviewer exposure, same GPT-6 family/provider/shared principal, org0/scienceNONE.','The frozen capture contains one Atlas page and exact captured assets; missing other page/favicon/history captures are fixture limits, not live-site defects.']}
    def save():
        (args.out/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    report['audit_target']=args.target_label
    report['input_binding_sha256']=hashlib.sha256(args.binding.read_bytes()).hexdigest()
    report['observation_source']={'main':binding['main'],'tree':binding['tree'],'payload_sha256':binding['payload_sha256']}
    report['product_source']=binding.get('product_source',{'revision':binding['main'],'tree':binding['tree'],'meaning':'F973 captured public product byte tree; all input members individually hash-bound'})
    def check(row,name,ok,detail=None,criterion=None):
        row.setdefault('manual',[]).append({'check':name,'result':'PASS' if ok else 'FAIL','detail':detail,'criterion':criterion})
    def snapshot(p,row,label):
        name=row['case']+'-'+label+'.png';p.screenshot(path=str(args.out/name));row.setdefault('screenshots',[]).append(name)
    def axe(p,row):
        p.add_script_tag(url=origin+'/__audit__/axe.min.js')
        result=p.evaluate('''async()=>await axe.run(document,{runOnly:{type:'tag',values:['wcag2a','wcag2aa','wcag21a','wcag21aa','wcag22aa','best-practice']},resultTypes:['violations','incomplete','passes','inapplicable']})''')
        name=row['case']+'-axe.json';(args.out/name).write_text(json.dumps(result,indent=2)+'\n')
        row['axe']={'report':name,'version':result['testEngine']['version'],'violations':[{'id':x['id'],'impact':x['impact'],'nodes':len(x['nodes']),'tags':x['tags']}for x in result['violations']],'incomplete':[{'id':x['id'],'nodes':len(x['nodes']),'tags':x['tags']}for x in result['incomplete']],'passes':[{'id':x['id'],'tags':x['tags'],'nodes':len(x['nodes'])}for x in result['passes']],'inapplicable':[x['id'] for x in result['inapplicable']]}
    cases=[]
    for width in [1200,390]:
        for scheme in ['light','dark']:
            for mode in ['graph','reader','3d']:
                cases.append({'case':f'{mode}-{width}-{scheme}','width':width,'scheme':scheme,'mode':mode})
    for width in [320,390]:
        for scheme in ['light','dark']:
            cases.append({'case':f'spacing-reader-{width}-{scheme}','width':width,'scheme':scheme,'mode':'reader','spacing':True})
    for scheme in ['light','dark']:
        cases.extend([{'case':f'nojs-320-{scheme}','width':320,'scheme':scheme,'mode':'nojs'},
                      {'case':f'refused-390-{scheme}','width':390,'scheme':scheme,'mode':'refused'}])
    cases.extend([{'case':'fetch-failure-320-dark','width':320,'scheme':'dark','mode':'fetch-failure'},
                  {'case':'webgl-failure-390-light','width':390,'scheme':'light','mode':'webgl-failure'},
                  {'case':'forced-colors-320','width':320,'scheme':'light','mode':'reader','forced':True}])
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(executable_path='/usr/bin/chromium',chromium_sandbox=True,args=['--enable-unsafe-swiftshader','--disable-background-networking'])
            report['environment']['browser_version']=browser.version
            for spec in cases:
                if args.only and args.only not in spec['case']:continue
                row={**spec,'completed':False,'manual':[],'page_errors':[],'blocked_external':[],'http_failures':[]};report['cases'].append(row)
                context=browser.new_context(viewport={'width':spec['width'],'height':900},color_scheme=spec['scheme'],reduced_motion='reduce',java_script_enabled=spec['mode']!='nojs')
                def route(r):
                    if urlsplit(r.request.url).hostname=='127.0.0.1':r.continue_()
                    else:row['blocked_external'].append(r.request.url);r.abort()
                context.route('**/*',route)
                if spec['mode']=='fetch-failure':context.route('**/current.json',lambda r:r.abort())
                if spec['mode']=='webgl-failure':context.add_init_script("const original=HTMLCanvasElement.prototype.getContext;HTMLCanvasElement.prototype.getContext=function(type,...args){return String(type).includes('webgl')?null:original.call(this,type,...args)}")
                p=context.new_page();p.set_default_timeout(9000)
                p.on('pageerror',lambda e:row['page_errors'].append(str(e)))
                p.on('response',lambda response:row['http_failures'].append({'url':response.url,'status':response.status}) if response.status>=400 else None)
                try:
                    url=origin+'/dependencies.html?atlas-node='+SELECTED+'&atlas-view='+('graph' if spec['mode'] in ['graph','3d','webgl-failure'] else 'list')
                    if spec['mode']=='refused':url+='&snapshot='+'0'*40+'&atlas-panel=evidence'
                    elif spec['mode']=='reader':url+='&atlas-panel=evidence'
                    p.goto(url)
                    if spec['mode']=='nojs':
                        row['axe']={'not_run':'product JavaScript disabled; no axe execution in this state'}
                        check(row,'static source observation readable',p.locator('.current-state').count()>0,p.locator('.current-state').first.inner_text() if p.locator('.current-state').count() else None,'1.3.1')
                        check(row,'product controller not executed',p.locator('#atlas-reader').count()==0)
                        p.locator('#current-research').scroll_into_view_if_needed();snapshot(p,row,'static-source')
                    elif spec['mode']=='fetch-failure':
                        expect(p.locator('#atlas-status')).to_contain_text('Unavailable')
                        check(row,'source failure visible and status semantics',p.locator('#atlas-status').is_visible() and p.locator('#atlas-status').evaluate('(e)=>!!e.closest("[role=status],[aria-live]")'),p.locator('#atlas-status').inner_text(),'4.1.3')
                        check(row,'failure does not substitute inspector relationships',p.locator('.atlas-relations').count()==0)
                        snapshot(p,row,'unavailable');axe(p,row)
                    elif spec['mode']=='refused':
                        expect(p.locator('#atlas-error')).to_be_visible()
                        check(row,'refusal alert semantics',p.locator('#atlas-error').get_attribute('role')=='alert',p.locator('#atlas-error').inner_text(),'4.1.3')
                        check(row,'refusal does not substitute relationships',p.locator('.atlas-relations').count()==0)
                        p.locator('#atlas-error').scroll_into_view_if_needed();snapshot(p,row,'refused');axe(p,row)
                    else:
                        expect(p.locator('#atlas-inspector h3')).to_have_text(SELECTED)
                        row['source_status']=p.locator('#atlas-status').inner_text()
                        if spec.get('forced'):p.emulate_media(forced_colors='active')
                        if spec.get('spacing'):
                            p.add_style_tag(url=origin+'/__audit__/spacing.css')
                            row['text_spacing_css']=SPACING
                        if spec['mode'] in ['3d','webgl-failure']:
                            p.locator('#atlas-space-toggle').click()
                            if spec['mode']=='webgl-failure':
                                expect(p.locator('#atlas-space-status')).to_contain_text('unavailable')
                                check(row,'WebGL failure has named status and 2D/list fallback',p.locator('#atlas-space-status').evaluate('(e)=>!!e.closest("[role=status]")') and p.locator('#atlas-stage').is_visible(),p.locator('#atlas-space-status').inner_text(),'4.1.3')
                            else:
                                canvas=p.locator('#atlas-space-host canvas');expect(canvas).to_have_count(1)
                                p.locator('#atlas-space-host').scroll_into_view_if_needed()
                                expect(p.locator('#atlas-space-host')).to_have_attribute('data-ready','true')
                                row['3d']={'canvas':True,'nodes':p.locator('#atlas-space-host').get_attribute('data-nodes'),'edges':p.locator('#atlas-space-host').get_attribute('data-edges'),'accessible_nodes':p.locator('.atlas-space > details > ul').first.locator('button').count()}
                                canvas.focus();before=canvas.screenshot();p.keyboard.press('ArrowRight');p.wait_for_timeout(150);after=canvas.screenshot()
                                check(row,'3D keyboard orbit changes real canvas',before!=after,{'before_sha256':hashlib.sha256(before).hexdigest(),'after_sha256':hashlib.sha256(after).hexdigest()},'2.1.1')
                                row['canvas_focus']=p.evaluate(FOCUS)
                                p.keyboard.press('Tab');check(row,'keyboard can leave 3D canvas',p.evaluate('document.activeElement.tagName')!='CANVAS',p.evaluate(FOCUS),'2.1.2')
                        elif spec['mode']=='reader':
                            # Native disclosure activation, not a direct open-attribute edit.
                            for sel in ['#atlas-source-details > summary','#atlas-formal-details > summary','#atlas-formal-packages > summary','#atlas-dependencies details > summary']:
                                loc=p.locator(sel).first
                                if loc.count():
                                    loc.focus();p.keyboard.press('Enter');check(row,'native disclosure keyboard '+sel,loc.evaluate('(e)=>e.parentElement.open'),p.evaluate(FOCUS),'2.1.1')
                            p.locator('#atlas-formal-details > summary').focus();row['disclosure_focus']=p.evaluate(FOCUS)
                            snapshot(p,row,'formal-disclosure-focus')
                            # Selection via real keyboard, reading focus and next-source link.
                            button=p.locator('#atlas-dependencies button').first;target=button.evaluate('(e)=>e.firstChild.textContent');button.focus();p.keyboard.press('Space')
                            expect(p.locator('#atlas-inspector h3')).to_have_text(target)
                            check(row,'selection focus lands on admitted inspector',p.evaluate('document.activeElement.id')=='atlas-inspector',p.evaluate(FOCUS),'2.4.3')
                            p.keyboard.press('Tab');row['source_link_focus']=p.evaluate(FOCUS)
                            check(row,'next Tab reaches immutable source',p.get_by_role('link',name='Immutable source reference',exact=False).evaluate('(e)=>e===document.activeElement'),row['source_link_focus'],'2.4.3')
                            snapshot(p,row,'source-focus')
                        # Common document semantics and reflow signals; clipping decisions reviewed later.
                        row['document']=p.evaluate('''()=>({width:innerWidth,scrollWidth:document.documentElement.scrollWidth,lang:document.documentElement.lang,title:document.title,main:document.querySelectorAll('main').length,h1:document.querySelectorAll('h1').length,headings:[...document.querySelectorAll('h1,h2,h3,h4,h5,h6')].filter(e=>e.getClientRects().length).map(e=>({tag:e.tagName,text:e.textContent})),navs:[...document.querySelectorAll('nav')].map(e=>({name:e.getAttribute('aria-label'),visible:!!e.getClientRects().length})),duplicates:[...document.querySelectorAll('[id]')].map(e=>e.id).filter((x,i,a)=>a.indexOf(x)!==i),reducedMotion:matchMedia('(prefers-reduced-motion:reduce)').matches})''')
                        check(row,'document does not require horizontal page scroll',row['document']['scrollWidth']<=row['document']['width'],{'width':row['document']['width'],'scrollWidth':row['document']['scrollWidth']},'1.4.10')
                        check(row,'main, title, language and unique IDs',row['document']['main']==1 and bool(row['document']['title']) and row['document']['lang']=='en' and not row['document']['duplicates'],row['document'],'1.3.1/2.4.2/3.1.1')
                        if spec['mode']=='graph':
                            # Measure live DOM messages after ordinary typing; no speech claim.
                            p.evaluate('''()=>{window.auditLive=[];window.auditMO=new MutationObserver(records=>{for(const rec of records){const e=rec.target.nodeType===1?rec.target:rec.target.parentElement;const live=e?.closest('[role=status],[role=alert],[aria-live]');if(live)auditLive.push({id:live.id,role:live.getAttribute('role'),live:live.getAttribute('aria-live'),text:live.textContent});}});auditMO.observe(document.body,{subtree:true,childList:true,characterData:true});}''')
                            p.locator('#atlas-search').fill('NO-MATCH-ACCESSIBILITY-PROBE');p.wait_for_timeout(120)
                            row['search_live_changes']=p.evaluate('auditLive');row['search_result_count']=p.locator('#atlas-count').inner_text()
                            check(row,'search result count has a live equivalent',any('0 displayed' in x['text'] for x in row['search_live_changes']),row['search_live_changes'],'4.1.3')
                            p.locator('#atlas-search').fill('');p.locator('#equation-search').fill('coefficient');expect(p.locator('#equation-announcement')).to_contain_text('matching excerpts',timeout=3000)
                            row['equation_announcement']=p.locator('#equation-announcement').inner_text();check(row,'equation count text reaches status region',p.locator('#equation-announcement').evaluate('(e)=>!!e.closest("[role=status],[aria-live]")') and bool(row['equation_announcement']),row['equation_announcement'],'4.1.3')
                            # First Tab/Enter skip link uses actual browser keyboard.
                            p.goto(url);expect(p.locator('#atlas-inspector h3')).to_have_text(SELECTED);p.keyboard.press('Tab');row['first_tab']=p.evaluate(FOCUS);p.keyboard.press('Enter')
                            check(row,'skip link sends keyboard focus to main',p.evaluate('document.activeElement.id')=='main-content',p.evaluate(FOCUS),'2.4.1')
                        p.locator('#atlas-heading').scroll_into_view_if_needed() if spec['mode'] in ['graph','3d','webgl-failure'] else None
                        snapshot(p,row,'viewport');axe(p,row)
                        if spec['mode']=='reader':
                            # Accessibility-tree text for source/formal/relationships; not AT speech.
                            (args.out/(row['case']+'-inspector-ax.txt')).write_text(p.locator('#atlas-inspector').aria_snapshot())
                    row['completed']=True
                except Exception as e:
                    row['setup_or_execution_error']=str(e);row['traceback']=traceback.format_exc()
                    try:snapshot(p,row,'error')
                    except Exception:pass
                finally:
                    context.close();save();print(json.dumps({'case':row['case'],'completed':row['completed'],'axe_violations':len(row.get('axe',{}).get('violations',[])),'manual_failures':[x['check'] for x in row['manual'] if x['result']=='FAIL'],'error':row.get('setup_or_execution_error')}),flush=True)
            browser.close()
    finally:
        server.shutdown();server.server_close();report['finished_at']=NOW();report['completed_cases']=sum(x['completed'] for x in report['cases']);report['total_cases']=len(report['cases']);report['requests']=requests
        # Verify immutable input again after every browser/server has closed.
        unchanged=True
        for x in binding['files']:
            unchanged &= hashlib.sha256((args.input/x['path']).read_bytes()).hexdigest()==x['sha256']
        report['input_unchanged_after']=unchanged;save()
    return 0 if report['completed_cases']==report['total_cases'] and report['input_unchanged_after'] else 1

if __name__=='__main__':raise SystemExit(main())
