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

def open_ancestor_disclosures(target):
    """Reach a control by activating each closed parent summary, outermost first."""
    target.wait_for(state='attached')
    for disclosure in target.locator('xpath=ancestor::details').all():
        if target.is_visible():break
        if disclosure.get_attribute('open') is None:
            summary=disclosure.locator(':scope > summary')
            summary.focus();summary.press('Enter')
            require(disclosure.get_attribute('open') is not None,'Keyboard summary activation did not open its disclosure')

# Measurements read activeElement before a later locator can auto-scroll it.
FOCUS_MEASUREMENT = "()=>{const e=document.activeElement,s=getComputedStyle(e),r=e.getBoundingClientRect();let bg=[0,0,0,0];const rgba=x=>(x.match(/[\\d.]+/g)||[]).map(Number);for(let n=e.parentElement;n;n=n.parentElement){let c=rgba(getComputedStyle(n).backgroundColor);if(c.length===3)c.push(1);let a=bg[3]+c[3]*(1-bg[3]);bg=[...bg.slice(0,3).map((v,i)=>a?(v*bg[3]+c[i]*c[3]*(1-bg[3]))/a:0),a];if(a===1)break;}if(bg[3]<1)bg=[...bg.slice(0,3).map(v=>v*bg[3]+255*(1-bg[3])),1];const lum=c=>c.slice(0,3).map(v=>v/255).map(v=>v<=.04045?v/12.92:((v+.055)/1.055)**2.4).reduce((v,c,i)=>v+c*[.2126,.7152,.0722][i],0),fg=rgba(s.outlineColor),opaque=fg.slice(0,3).map((v,i)=>v*(fg[3]??1)+bg[i]*(1-(fg[3]??1))),a=lum(opaque),b=lum(bg);return {id:e.id,selected:e.getAttribute('aria-selected'),outline_color:s.outlineColor,outline_style:s.outlineStyle,outline_width:s.outlineWidth,outline_offset:s.outlineOffset,focus_visible:e.matches(':focus-visible'),background:bg,contrast:(Math.max(a,b)+.05)/(Math.min(a,b)+.05),rect:{x:r.x,y:r.y,top:r.top,bottom:r.bottom,left:r.left,right:r.right,width:r.width,height:r.height},viewport:{width:innerWidth,height:innerHeight},scroll:{x:scrollX,y:scrollY},in_viewport:r.width>0&&r.height>0&&r.top>=0&&r.left>=0&&r.bottom<=innerHeight&&r.right<=innerWidth,url:location.href,history_length:history.length,writes:{...(window.__shellHistoryWrites||{})}};}"
def focused_geometry(page):
    return page.evaluate(FOCUS_MEASUREMENT)
def require_browse_focus(page):
    row=focused_geometry(page)
    require(row['id']=='atlas-open-reader' and row['in_viewport'],f'Browse focus outside viewport: {row}')
    return row
def exercise_focus_ring(page):
    rows=[]
    for forced in ['none','active']:
        page.emulate_media(forced_colors=forced)
        page.locator('#atlas-tab-evidence').focus();page.keyboard.press('Tab');page.keyboard.press('Shift+Tab')
        for action in [None,'ArrowRight','Enter']:
            if action:page.keyboard.press(action)
            row=focused_geometry(page);row['forced_colors']=forced;rows.append(row)
            require(row['focus_visible'] and row['outline_style']!='none' and float(row['outline_width'].removesuffix('px'))>=2 and row['contrast']>=3,f'Tab focus ring lacks 3:1 contrast: {row}')
            if forced=='active':
                highlight=page.evaluate("(()=>{const x=document.createElement('span');x.style.color='Highlight';document.body.append(x);const c=getComputedStyle(x).color;x.remove();return c;})()")
                require(row['outline_color']==highlight,f'Tab focus ring overrides system Highlight: {row}')
        page.keyboard.press('Home');page.keyboard.press('Enter')
    page.emulate_media(forced_colors='none')
    return rows

def exercise_atlas_shell(browser,origin,output,report,data,expect):
    """Declared shell acceptance; results exist only when this runner executes."""
    from urllib.parse import urlsplit,parse_qs,urlencode
    def address(tab):return tab.evaluate('({url:location.href,length:history.length,writes:{...window.__shellHistoryWrites}})')
    watch_history="""window.__shellHistoryWrites={push:0,replace:0};for(const [method,key] of [['pushState','push'],['replaceState','replace']]){const original=history[method].bind(history);history[method]=(...args)=>{window.__shellHistoryWrites[key]++;return original(...args);};}"""
    for width in [320,720,721,1200]:
        case={'case':'atlas-shell-tabs-reading-and-history','width':width,'passed':False,'checkpoints':[],'page_errors':[]};report['cases'].append(case)
        context=browser.new_context(viewport={'width':width,'height':900},reduced_motion='reduce');context.add_init_script(watch_history);page=context.new_page();page.set_default_timeout(15000);page.on('pageerror',lambda e:case['page_errors'].append(str(e)))
        try:
            legacy=origin+'dependencies.html?q=math&classification=PROVED_REVIEWED#search-heading';page.goto(legacy);expect(page.locator('#atlas-reader')).to_have_count(1);expect(page.locator('#atlas-inspector h3')).to_have_count(1)
            initial=address(page);case['checkpoints'].append(initial);require(initial['url']==legacy,'Shell startup rewrote historical filters or fragment');require(initial['writes']=={'push':0,'replace':0},'Shell startup wrote history');require(page.locator('#atlas-view').input_value()==('list' if width<=720 else 'graph'),'Viewport default view is wrong')
            # Explicit Graph must survive entering and leaving narrow reading.
            page.locator('#atlas-view').select_option('graph')
            if width<=720:
                expect(page.locator('#atlas-reader')).to_be_hidden();page.locator('#atlas-open-reader').click();expect(page.locator('#atlas-reader')).to_be_visible();require(page.evaluate('document.activeElement.id')=='atlas-inspector','Narrow reading entry did not focus actual Evidence')
            evidence=page.locator('#atlas-tab-evidence');diagnostics=page.locator('#atlas-tab-diagnostics');evidence.focus();before=address(page);page.keyboard.press('ArrowRight')
            require(page.evaluate('document.activeElement.id')=='atlas-tab-diagnostics','Right arrow did not focus Diagnostics');expect(evidence).to_have_attribute('aria-selected','true');expect(diagnostics).to_have_attribute('aria-selected','false');require(address(page)==before,'Arrow focus wrote state')
            page.keyboard.press('Enter');expect(diagnostics).to_have_attribute('aria-selected','true');expect(page.locator('#atlas-panel-evidence')).to_be_hidden();expect(page.locator('#atlas-panel-diagnostics')).to_be_visible();after=address(page);require(after['length']==before['length']+1 and after['writes']['push']==before['writes']['push']+1,'Tab activation did not create one ordinary entry')
            page.keyboard.press('Home');require(page.evaluate('document.activeElement.id')=='atlas-tab-evidence','Home did not focus Evidence');page.keyboard.press('Space');expect(evidence).to_have_attribute('aria-selected','true');page.keyboard.press('Tab');require(page.evaluate('document.activeElement.id')=='atlas-panel-evidence','Tab did not enter the active panel')
            diagnostics.click();detail=page.locator('#atlas-panel-diagnostics details');detail.locator('summary').click();before_node=address(page);detail.locator('button').first.click();expect(page.locator('#atlas-inspector h3')).to_have_text(next(iter(data['graph']['data']['nodes'])));expect(evidence).to_have_attribute('aria-selected','true');require(page.evaluate('document.activeElement.id')=='atlas-inspector','Diagnostic selection focused a hidden/detached inspector');expect(page.locator('#atlas-inspector .atlas-axes dt')).to_have_count(5)
            selected=address(page);case['checkpoints'].extend([before_node,selected]);require(selected['length']==before_node['length']+1,'Node selection created more than one entry')
            page.go_back();expect(diagnostics).to_have_attribute('aria-selected','true');restored=address(page);require(restored['url']==before_node['url'] and restored['writes']==selected['writes'],'Back changed the saved address or wrote history')
            page.go_forward();expect(evidence).to_have_attribute('aria-selected','true');forward=address(page);require(forward['url']==selected['url'] and forward['writes']==selected['writes'],'Forward changed selection or wrote history')
            if width<=720:
                metrics=page.locator('#atlas-reader').evaluate('(e)=>({height:e.getBoundingClientRect().height,viewport:innerHeight,position:getComputedStyle(e).position,width:document.documentElement.scrollWidth,viewportWidth:innerWidth})');require(metrics['height']>=metrics['viewport'] and metrics['position'] not in ['fixed','absolute'],'Reader is not full-height normal-flow content');require(metrics['width']<=metrics['viewportWidth'],'Reader overflows narrow document');case['reader_metrics']=metrics
                for selector in ['#atlas-search','#atlas-view','#atlas-stage','#atlas-next']:expect(page.locator(selector)).to_be_hidden()
                # Long source metadata remains page-scrollable and the source
                # identity/refusal/status elements stay in the document.
                expect(page.locator('#atlas-status')).to_contain_text('Source snapshot');expect(page.locator('#atlas-share')).to_be_visible();require(page.locator('#atlas-reader').evaluate('(e)=>getComputedStyle(e).overflowY') not in ['hidden','clip'],'Reader clips long content')
                target=page.locator('#atlas-browse-link').get_attribute('href');page.locator('#atlas-browse-link').click();expect(page.locator('#atlas-reader')).to_be_hidden();require(page.url==target,'Browse link changed fields beyond layout');require(page.evaluate('document.activeElement.id')=='atlas-open-reader','Browse exit left focus in hidden content');case['browse_focus_geometry']=require_browse_focus(page);expect(page.locator('#atlas-stage')).to_be_visible();require(page.locator('#atlas-view').input_value()=='graph','Reading lost the explicit Graph choice')
                heights=page.locator('#atlas-canvas .atlas-node').evaluate_all('(nodes)=>nodes.map(n=>n.getBoundingClientRect().height)');require(heights and min(heights)>=124,'Graph geometry was not measured after unhide');require(len(heights)<=150,'Shell bypassed the render boundary')
                deep=context.new_page();deep.goto(selected['url']);expect(deep.locator('#atlas-reader')).to_be_visible();require(address(deep)['writes']=={'push':0,'replace':0},'Direct reader URL wrote history');deep.locator('#atlas-browse-link').click();expect(deep.locator('#atlas-open-reader')).to_be_visible();require('atlas-panel' not in parse_qs(urlsplit(deep.url).query),'Direct reader link has no browse exit');deep.close()
            # Resize an admitted reader, including focus on controls which hide
            # at desktop width, and on a source child replaced by render().
            page.goto(selected['url']);expect(page.locator('#atlas-reader')).to_be_visible();page.set_viewport_size({'width':320,'height':900});expect(page.locator('#atlas-search')).to_be_hidden();page.locator('#atlas-reader-back').focus();before_resize=address(page);page.set_viewport_size({'width':1200,'height':900});expect(page.locator('#atlas-stage')).to_be_visible();expect(page.locator('#atlas-tab-evidence')).to_be_focused();require(address(page)==before_resize,'Resize wrote presentation history');require(page.locator('#atlas-canvas .atlas-node').count()<=150,'Resize bypassed render boundary')
            page.locator('#atlas-inspector a').first.focus();page.set_viewport_size({'width':320,'height':900});expect(page.locator('#atlas-inspector')).to_be_focused();require(page.evaluate('document.activeElement.isConnected'),'Resize left focus detached')
            page.evaluate("(()=>{const entries=[document.documentElement,document.body,...document.querySelectorAll('body *')].map(e=>[e,parseFloat(getComputedStyle(e).fontSize)]);for(const [e,n]of entries)e.style.setProperty('font-size',n*2+'px','important');})()");enlarged=page.evaluate('({width:innerWidth,document:document.documentElement.scrollWidth})');require(enlarged['document']<=enlarged['width'],'Enlarged reader overflows document');expect(page.locator('#atlas-browse-link')).to_be_visible();expect(page.locator('#atlas-tab-diagnostics')).to_be_visible()
            page.emulate_media(media='print');require(page.locator('#atlas-reader').evaluate('(e)=>getComputedStyle(e).minHeight')=='0px','Print retained narrow full-height spacer');expect(page.locator('#atlas-panel-evidence')).to_be_visible();expect(page.locator('#atlas-panel-diagnostics')).to_be_visible();page.emulate_media(media='screen')
            shot=output/f'atlas-shell-{width}.png';page.locator('#current-atlas').screenshot(path=str(shot));case['screenshot']={'path':shot.name,'sha256':sha256(shot.read_bytes()).hexdigest()};require(not case['page_errors'],f'Shell page errors: {case["page_errors"]}');case['passed']=True
        except Exception:case['error']=traceback.format_exc()
        finally:context.close()
    for width in [320,1200]:
        case={'case':'atlas-shell-refused-layout-preservation','width':width,'passed':False};report['cases'].append(case);context=browser.new_context(viewport={'width':width,'height':900});context.add_init_script(watch_history);page=context.new_page();page.set_default_timeout(15000)
        try:
            refused=origin+'dependencies.html?'+urlencode({'atlas-node':'constructor','main-snapshot':'c'*40,'atlas-panel':'diagnostics','atlas-view':'graph','q':'old','source-q':'source','equation-q':'ell','equation':data['equations'][0]['id']})+'#search-heading';page.goto(refused);expect(page.locator('#atlas-error')).to_be_visible();expect(page.locator('#atlas-tab-diagnostics')).to_have_attribute('aria-selected','true');before=address(page);require(before['url']==refused and before['writes']=={'push':0,'replace':0},'Refused startup was rewritten')
            page.locator('#atlas-tab-evidence').click();expect(page.locator('#atlas-error')).to_be_visible();actual=parse_qs(urlsplit(page.url).query);expected=parse_qs(urlsplit(refused).query);expected['atlas-panel']=['evidence'];require(actual==expected and urlsplit(page.url).fragment=='search-heading','Layout action replaced refused identity or filters');expect(page.locator('#atlas-canvas .atlas-node')).to_have_count(0);expect(page.locator('#equation-content pre')).to_have_count(0);require(page.locator('#atlas-share').get_attribute('href')==page.url,'Layout Share differs from refused source URL')
            changed=address(page);page.go_back();expect(page.locator('#atlas-tab-diagnostics')).to_have_attribute('aria-selected','true');require(page.url==refused and address(page)['writes']==changed['writes'],'Refused Back wrote state or substituted current source');case['passed']=True
        except Exception:case['error']=traceback.format_exc()
        finally:context.close()
    case={'case':'atlas-shell-responsive-view-preferences','passed':False,'scenarios':[],'page_errors':[]};report['cases'].append(case)
    scenarios=[
        ('wide-implicit',1200,'?q=math&classification=PROVED_REVIEWED#search-heading',None),
        ('narrow-implicit',320,'?q=math#search-heading',None),
        ('wide-implicit-stage-focus',1200,'?q=math#search-heading','stage-focus'),
        ('wide-implicit-outside-focus',1200,'?q=math#search-heading','outside-focus'),
        ('wide-unknown',1200,'?atlas-view=unknown&q=math#search-heading',None),
        ('narrow-unknown',320,'?atlas-view=unknown&q=math#search-heading',None),
        ('explicit-graph-url',320,'?atlas-view=graph&q=math#search-heading',None),
        ('explicit-list-url',1200,'?atlas-view=list&q=math#search-heading',None),
        ('explicit-graph-control',320,'?q=math#search-heading','graph'),
        ('explicit-list-control',1200,'?q=math#search-heading','list'),
        ('implicit-after-selection',1200,'?q=math#search-heading','select-node'),
        ('implicit-after-search',1200,'?q=math#search-heading','search'),
        ('refused-implicit',1200,'?atlas-node=constructor&main-snapshot='+('c'*40)+'&q=old#search-heading',None),
        ('refused-explicit-graph',320,'?atlas-node=constructor&main-snapshot='+('c'*40)+'&atlas-view=graph&q=old#search-heading',None),
    ]
    for label,start_width,query,action in scenarios:
        item={'scenario':label,'passed':False,'checkpoints':[]};case['scenarios'].append(item)
        context=browser.new_context(viewport={'width':start_width,'height':900},reduced_motion='reduce');context.add_init_script(watch_history);page=context.new_page();page.set_default_timeout(15000);page.on('pageerror',lambda e:case['page_errors'].append(str(e)))
        try:
            page.goto(origin+'dependencies.html'+query);expect(page.locator('#atlas-reader')).to_have_count(1)
            requested=parse_qs(urlsplit(page.url).query).get('atlas-view',[None])[0]
            explicit=requested if requested in ('graph','list') else None
            if action in ('graph','list'):
                page.locator('#atlas-view').select_option(action);explicit=action
            elif action=='select-node':
                page.locator('#atlas-canvas .atlas-node').first.click()
                require('atlas-view' not in parse_qs(urlsplit(page.url).query),'Implicit node selection became an explicit view preference')
            elif action=='search':
                page.locator('#atlas-search').fill('cap')
                require('atlas-view' not in parse_qs(urlsplit(page.url).query),'Implicit search became an explicit view preference')
            elif action=='stage-focus':page.locator('#atlas-stage').focus()
            elif action=='outside-focus':page.locator('body > header a').first.focus()
            outside_focus=page.evaluate('document.activeElement.outerHTML') if action=='outside-focus' else None
            for width in [start_width,320 if start_width>720 else 1200,start_width]:
                before=address(page);page.set_viewport_size({'width':width,'height':900})
                # Native media-change delivery settles before reading the next
                # effective view. No delay, DOM mutation or history write is used.
                page.evaluate('new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))')
                want=explicit or ('list' if width<=720 else 'graph')
                expect(page.locator('#atlas-view')).to_have_value(want)
                after=address(page);require(after==before,'Viewport-only adaptation changed URL, history length or writes')
                if action in ('stage-focus','outside-focus'):
                    focused=page.evaluate('({id:document.activeElement.id,connected:document.activeElement.isConnected,visible:!!document.activeElement.getClientRects().length,html:document.activeElement.outerHTML})')
                    require(focused['connected'] and focused['visible'] and focused['id']!='','Resize left hidden or body focus') if action=='stage-focus' else require(focused['html']==outside_focus,'Resize stole outside focus')
                    item['focus']=focused
                    if focused['id']=='atlas-open-reader':item['focus_geometry']=require_browse_focus(page)
                if label.startswith('refused'):
                    expect(page.locator('#atlas-error')).to_be_visible();expect(page.locator('#atlas-canvas .atlas-node')).to_have_count(0);expect(page.locator('#equation-content pre')).to_have_count(0)
                item['checkpoints'].append({'width':width,'view':want,'address':after})
            item['passed']=True
        except Exception:item['error']=traceback.format_exc()
        finally:context.close()
    failures=[item['scenario'] for item in case['scenarios'] if not item['passed']]
    case['failures']=failures;case['passed']=not failures and not case['page_errors']

def exercise_hosted_amendments(browser,origin,output,report,data,expect):
    """Real native navigation and inspector controls for the three hosted P2s."""
    from urllib.parse import urlsplit,parse_qs
    import sys
    watch="""window.__amendWrites={push:0,replace:0};for(const [method,key]of [['pushState','push'],['replaceState','replace']]){const original=history[method].bind(history);history[method]=(...args)=>{window.__amendWrites[key]++;return original(...args);};}"""
    def address(page):return page.evaluate('({url:location.href,length:history.length,writes:{...window.__amendWrites}})')
    def ready(page,query='?atlas-view=list&atlas-panel=evidence'):
        page.goto(origin+'dependencies.html'+query);expect(page.locator('#atlas-reader')).to_have_count(1);expect(page.locator('#atlas-inspector h3')).to_have_count(1)
    for label in ['same-state-focus-history','modified-browse-current-source-filter','middle-browse-current-source-filter','inspector-panel-disclosures','inspector-breakpoint-disclosures','genuine-node-and-source-change']:
        case={'case':'hosted-amendment-'+label,'passed':False,'observations':[],'page_errors':[]};report['cases'].append(case)
        context=browser.new_context(viewport={'width':1200,'height':900},reduced_motion='reduce');context.add_init_script(watch);page=context.new_page();page.set_default_timeout(15000);page.on('pageerror',lambda e:case['page_errors'].append(str(e)))
        try:
            ready(page)
            if label=='same-state-focus-history':
                first=page.locator('#atlas-inspector h3').inner_text();nodes=list(data['graph']['data']['nodes']);other=next(n for n in nodes if n!=first)
                page.locator('#atlas-search').fill(other);page.locator('#atlas-list button').filter(has_text=other).first.click();expect(page.locator('#atlas-inspector h3')).to_have_text(other)
                open_ancestor_disclosures(page.locator('#atlas-focus'))
                before=address(page)
                for _ in range(3):page.locator('#atlas-focus').click()
                after=address(page);case['observations'].extend([before,after]);require(after==before,'Same-state Return to focus added history or rewrote the address');expect(page.locator('#atlas-inspector')).to_be_focused()
                page.go_back();expect(page.locator('#atlas-inspector h3')).to_have_text(first)
                page.go_forward();expect(page.locator('#atlas-inspector h3')).to_have_text(other)
                page.locator('#atlas-tab-diagnostics').click();diagnostics=address(page);page.locator('#atlas-focus').click();evidence=address(page);case['observations'].extend([diagnostics,evidence]);require(evidence['length']==diagnostics['length']+1 and evidence['writes']['push']==diagnostics['writes']['push']+1,'Real Diagnostics to Evidence focus transition lost its one history entry');page.go_back();expect(page.locator('#atlas-tab-diagnostics')).to_have_attribute('aria-selected','true')
            elif label in ['modified-browse-current-source-filter','middle-browse-current-source-filter']:
                # current.mjs owns this real search and its independent replaceState.
                page.set_viewport_size({'width':320,'height':900})
                ready(page,'?atlas-view=list&atlas-panel=evidence&q=math&classification=OPEN&source-q=old#search-heading')
                open_ancestor_disclosures(page.get_by_text('Search the current public source inventory', exact=True))
                page.get_by_text('Search the current public source inventory', exact=True).click();page.locator('.current-search').fill('universality');expect(page.locator('.current-results')).to_contain_text('experiments/universality')
                before=address(page);expected=parse_qs(urlsplit(before['url']).query);expected.pop('atlas-panel',None);require(expected.get('source-q')==['universality'],'Independent current.mjs search did not set the new source filter')
                with context.expect_page() as popup_event:
                    if label.startswith('modified'):page.locator('#atlas-browse-link').click(modifiers=['Meta' if sys.platform=='darwin' else 'Control'])
                    else:page.locator('#atlas-browse-link').click(button='middle')
                popup=popup_event.value;popup.wait_for_load_state('domcontentloaded');expect(popup.locator('#atlas-reader')).to_have_count(1)
                actual=parse_qs(urlsplit(popup.url).query);case['observations'].append({'original':before,'opened':popup.url,'after':address(page)})
                require(actual==expected and urlsplit(popup.url).fragment==urlsplit(before['url']).fragment,'Native Browse opened stale source-q, pins, query or fragment');require(address(page)==before,'Native Browse mutated the original page history');expect(popup.locator('.current-search')).to_have_value('universality');require(popup.locator('#current-atlas').get_attribute('data-atlas-panel')=='browse','Native Browse opened a reader panel');popup.close()
            elif label in ['inspector-panel-disclosures','inspector-breakpoint-disclosures']:
                summaries=['Required-edge routes to unresolved research targets','Complete node metadata (inert text)']
                for summary in summaries:page.locator('#atlas-inspector details').filter(has=page.get_by_text(summary,exact=True)).locator('summary').click()
                before=address(page);selected=page.locator('#atlas-inspector h3').inner_text();text=page.locator('#atlas-inspector').inner_text()
                if label=='inspector-panel-disclosures':page.locator('#atlas-tab-diagnostics').click();page.locator('#atlas-tab-evidence').click()
                else:
                    for width in [320,1200,720,721]:
                        page.set_viewport_size({'width':width,'height':900});page.evaluate('new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))')
                        require(page.locator('#atlas-inspector details[open]').count()==2,'Breakpoint rebuild closed same-node route/metadata disclosures')
                    require(address(page)==before,'Breakpoint disclosure preservation wrote history')
                opened=page.locator('#atlas-inspector details').evaluate_all('(nodes)=>nodes.map(n=>({summary:n.querySelector("summary").textContent,open:n.open}))');case['observations'].append({'before':before,'after':address(page),'disclosures':opened,'selected':selected})
                require(len(opened)==2 and all(n['open'] for n in opened),'Panel change closed same-node route/metadata disclosures');require(page.locator('#atlas-inspector h3').inner_text()==selected and page.locator('#atlas-inspector').inner_text()==text,'Shell-only change replaced selected evidence')
            else:
                first=page.locator('#atlas-inspector h3').inner_text();other=next(n for n in data['graph']['data']['nodes'] if n!=first)
                page.locator('#atlas-inspector details summary').last.click();page.locator('#atlas-search').fill(other);page.locator('#atlas-list button').filter(has_text=other).first.click();expect(page.locator('#atlas-inspector h3')).to_have_text(other)
                metadata=json.loads(page.locator('#atlas-inspector details pre').text_content());require(metadata==data['graph']['data']['nodes'][other],'Changed node retained old metadata');require(page.locator('#atlas-inspector details[open]').count()==0,'Changed node inherited prior disclosure state')
                # A true source identity change through native history must refuse,
                # then restore the admitted selection through ordinary Forward.
                refused=origin+'dependencies.html?atlas-node='+other+'&snapshot='+('c'*40)+'&atlas-panel=evidence'
                page.evaluate('(url)=>{history.pushState(null,"",url);dispatchEvent(new PopStateEvent("popstate"));}',refused)
                expect(page.locator('#atlas-error')).to_be_visible();expect(page.locator('#atlas-inspector h3')).to_have_count(0);expect(page.locator('#atlas-inspector details')).to_have_count(0);expect(page.locator('#equation-content pre')).to_have_count(0)
                page.go_back();expect(page.locator('#atlas-inspector h3')).to_have_text(other);require(json.loads(page.locator('#atlas-inspector details pre').text_content())==data['graph']['data']['nodes'][other],'Restored admitted source retained refusal or stale evidence');case['observations'].append({'first':first,'changed':other,'refused_url':refused,'restored':address(page)})
            require(not case['page_errors'],f'Page errors: {case["page_errors"]}');case['passed']=True
        except Exception:case['error']=traceback.format_exc()
        finally:context.close()


def assert_resume_readings(interactive,static,data,node,status_records,now):
    """Check each consumer against its source inputs, including shared links.

    The atlas has an optional operational overlay; static pages intentionally do
    not. Expected rows come from the recorded source/model, never from observed
    DOM values, and no field is dropped to make the two consumers look equal.
    """
    from current_site_history import resume_projection
    def expected(records=None,clock=None):
        view=resume_projection(data,node,records,clock)
        groups=[view['state']]
        if view['review']['fields']:groups.append(view['review']['fields'])
        groups.append(view['operational'])
        rows=[];links=[]
        for group in groups:
            row=[]
            for item in group:
                row.extend([item.get('label',item['field']),item['text']+(' '+item['note'] if item.get('note') else '')])
                if item.get('url'):links.append(item['url'])
            rows.append(row)
        if view['continuation']:links.extend([view['continuation']['url'],view['continuation']['identified_in']])
        return {'rows':rows,'links':links}
    expected_live=expected(status_records,now);expected_static=expected()
    require(interactive==expected_live,f'Interactive Resume block differs from its source-bound operational projection for {node}')
    require(static==expected_static,f'Static Resume block differs from its snapshot-only projection for {node}')
    require(interactive['rows'][:-1]==static['rows'][:-1],f'Shared graph/review Resume rows differ for {node}')
    return {'expected_interactive':expected_live,'expected_static':expected_static}


def exercise_resume_parity(browser,origin,report,data,expect):
    # Both show the same graph/review/continuation sources; only the atlas receives operational records.
    from urllib.parse import urlencode,quote
    def reading(block):return block.evaluate("(e)=>({rows:[...e.querySelectorAll('dl.resume-fields')].map(dl=>[...dl.children].map(c=>c.textContent.trim())),links:[...e.querySelectorAll('a[href]')].map(a=>a.href)})")
    pin='math.d5-component.punctured-pin-proof'
    status_records=json.loads((ROOT/'docs/site/node-status.json').read_bytes())
    for node in [pin,'math.lifetime-remainder','math.uniform-matrix-cap-lifetime','hist.CH-LIFT']:
        case={'case':'resume-node-interactive-static-parity','node':node,'passed':False,'page_errors':[]};report['cases'].append(case)
        render_clock=int(time.time()*1000);case['render_clock_ms']=render_clock
        context=browser.new_context(reduced_motion='reduce');context.add_init_script(f'Date.now=()=>{render_clock};')
        page=context.new_page();page.set_default_timeout(15000);page.on('pageerror',lambda e:case['page_errors'].append(str(e)))
        try:
            page.goto(origin+'dependencies.html?'+urlencode({'atlas-node':node,'atlas-view':'list'}));expect(page.locator('#atlas-inspector h3')).to_have_text(node)
            live=page.locator('#atlas-resume');expect(live).to_contain_text('Resume this node');expect(live.locator('.resume-warning')).to_contain_text(data['observed_at']);expect(live.locator('.resume-warning')).to_contain_text('before any ownership- or head-dependent action')
            interactive=reading(live);case['freshness']=live.locator('.resume-warning').get_attribute('data-freshness');require(case['freshness'] in ('current','stale','clock disagreement'),'Interactive block omitted its freshness state')
            page.goto(origin+'nodes/'+quote(node)+'.html');block=page.locator('#resume-node');expect(block).to_contain_text('this static page cannot evaluate freshness');static=reading(block)
            case['interactive']=interactive;case['static']=static
            case.update(assert_resume_readings(interactive,static,data,node,status_records,render_clock))
            require(not any(re.fullmatch(r'https://github\.com/d6g8k5htny-coder/(?:main|Math-)/issues/[0-9]+',href) for href in interactive['links']),'A bare review issue number was turned into a link')
            require(not case['page_errors'],f'Page errors: {case["page_errors"]}');case['passed']=True
        except Exception:case['error']=traceback.format_exc()
        finally:context.close()
def exercise_observation_coherence(browser,origin,output,report,data,expect):
    """Controlled old/new responses; no claim about the serving cache layer."""
    raw=(ROOT/'docs/site/current.json').read_bytes()
    previous=re.sub(rb'("observed_at"\s*:\s*)'+re.escape(json.dumps(data['observed_at']).encode()),lambda m:m.group(1)+b'"2000-01-01T00:00:00Z"',raw,count=1)
    require(json.loads(previous)['observed_at']=='2000-01-01T00:00:00Z' and len(previous)==len(raw),'Mixed-observation fixture identity failed')
    digest=sha256(raw).hexdigest()
    for persistent in [False,True]:
        case={'case':'observation-persistent-mismatch-refusal' if persistent else 'observation-integrity-recovery','passed':False,
              'scope':'Controlled local old/new response fixture; no public cache-layer inference','requests':[],'page_errors':[],
              'expected_payload':{'bytes':len(raw),'sha256':digest},'old_payload':{'bytes':len(previous),'sha256':sha256(previous).hexdigest()}}
        report['cases'].append(case)
        context=browser.new_context(viewport={'width':1200,'height':900},reduced_motion='reduce')
        page=context.new_page();page.on('pageerror',lambda e:case['page_errors'].append(str(e)))
        def respond(route):
            url=route.request.url;case['requests'].append(url)
            body=previous if persistent or url==origin+'current.json' else raw
            route.fulfill(status=200,content_type='application/json',body=body)
        try:
            page.route('**/current.json*',respond);page.goto(origin+'dependencies.html')
            if persistent:
                expect(page.locator('.current-state')).to_contain_text('Unavailable')
                expect(page.locator('#atlas-status')).to_contain_text('Unavailable')
                require(page.locator('#atlas-inspector h3').count()==0 and page.locator('#atlas-canvas .atlas-node').count()==0 and page.locator('#equation-content pre').count()==0,'Persistent mismatch admitted current content')
                require(1<=len(case['requests'])<=2,'Persistent mismatch exceeded the bounded request budget')
            else:
                expect(page.locator('#atlas-inspector h3')).to_have_count(1)
                expect(page.locator('.current-state')).to_contain_text('Verified source observation: '+data['observed_at'])
                require(case['requests']==[origin+'current.json',origin+'current.json?observation='+digest],'Recovery did not bind its only retry to the advertised digest')
                require(page.locator('#atlas-canvas .atlas-node').count()>0,'Recovered observation did not render its actual graph')
            require(not case['page_errors'],'Observation fixture emitted page errors')
            case['passed']=True
        except Exception:case['error']=traceback.format_exc()
        finally:context.close()

def exercise_static_reading_destinations(browser,origin,report,data,expect):
    """Activate real inspector links with independently served static generations.

    The admitted atlas and material archive use the existing tracked fixture.
    Only the mutable static response receives controlled older/newer date and
    pin substitutions. These fixtures say nothing about a public cache layer.
    """
    from urllib.parse import quote,urlencode
    html=(ROOT/'docs/site/dependencies.html').read_text()
    section=re.search(r'(<section\b[^>]*\bid="current-research"[^>]*>)(.*?)(</section>)',html,re.S)
    require(section is not None,'Static reading fixture has no generated current section')
    hrefs=re.findall(r'\bhref="([^"]*)"',section.group(2))
    archives={match.group(1) for href in hrefs if (match:=re.fullmatch(r'snapshots/([0-9a-f]{64})/nodes/index\.html',href))}
    require(len(archives)==1,'Static reading fixture has no unique advertised material archive')
    archive=next(iter(archives));material_root=ROOT/'docs/site/snapshots'/archive
    material_raw=(material_root/'current.json').read_bytes();material=json.loads(material_raw)
    node='math.uniform-matrix-cap-lifetime';encoded=quote(node,safe='')
    require(node in data['graph']['data']['nodes'] and node in material['graph']['data']['nodes'],'Selected static fixture node is missing')
    edge=next(e for e in data['graph']['data']['edges'] if e['from']==node)
    current_path='nodes/'+encoded+'.html';preserved_path='snapshots/'+archive+'/nodes/'+encoded+'.html'
    current_body=(ROOT/'docs/site'/current_path).read_text()
    preserved_body=(ROOT/'docs/site'/preserved_path).read_bytes()
    def identity(snapshot):
        return {'observed_at':snapshot['observed_at'],'pins':{r['repository']:r['commit'] for r in snapshot['repositories']}}
    atlas_identity=identity(data);material_identity=identity(material)
    def static_reading(page):
        # Read the actual visible source fields, not the synthetic response plan.
        header=page.locator('#current-research');time_node=header.locator('.current-state time')
        expect(time_node).to_be_visible();pins={};pin_links={}
        for repository in ['d6g8k5htny-coder/main','d6g8k5htny-coder/Math-']:
            link=header.locator('a[href^="https://github.com/'+repository+'/tree/"]')
            expect(link).to_have_count(1);expect(link).to_be_visible()
            pins[repository]=link.locator('code').inner_text();pin_links[repository]=link.get_attribute('href')
            require(pin_links[repository]=='https://github.com/'+repository+'/tree/'+pins[repository],'Visible static pin disagrees with its source URL')
        return {'url':page.url,'node':page.locator('main > h1').inner_text(),'observed_at':time_node.inner_text(),'datetime':time_node.get_attribute('datetime'),'pins':pins,'pin_links':pin_links,'script_count':page.locator('script').count(),'content_security_policy':page.locator('meta[http-equiv="Content-Security-Policy"]').get_attribute('content')}
    def check_reading(reading,want,path,selected_node=node):
        require(reading['url']==origin+path,'Activated static destination changed its route')
        require(reading['node']==selected_node,'Activated static destination changed the selected node')
        require(reading['observed_at']==want['observed_at'] and reading['datetime']==want['observed_at'],'Activated static destination changed its visible capture date')
        require(reading['pins']==want['pins'],'Activated static destination changed its visible repository pins')
        require(reading['script_count']==0 and "script-src 'none'" in reading['content_security_policy'],'Static reading destination is not script-free')
    def announced(page):
        expect(page.locator('#atlas-inspector h3')).to_have_text(node)
        preserved=page.locator('#atlas-inspector').get_by_role('link',name='Read the preserved mathematical record',exact=True)
        current=page.locator('#atlas-inspector').get_by_role('link',name='Read the static node record',exact=True)
        expect(preserved).to_be_visible();expect(current).to_be_visible()
        expect(preserved.locator('..')).to_contain_text('Opens the preserved mathematical record with its original source revisions and capture date.')
        expect(current.locator('..')).to_contain_text('The capture date and source revisions on that page may differ from this view.')
        require(preserved.get_attribute('href')==preserved_path and current.get_attribute('href')==current_path,'Inspector reading routes do not match the advertised archive and selected node')
        return {'url':page.url,'node':node,**atlas_identity,'preserved_href':preserved.get_attribute('href'),'current_href':current.get_attribute('href'),'preserved_explanation':preserved.locator('..').inner_text(),'current_explanation':current.locator('..').inner_text()}
    atlas_url=origin+'dependencies.html?'+urlencode({'atlas-node':node,'atlas-view':'list','atlas-panel':'evidence'})
    for generation,date,main_pin,math_pin in [('older','2000-01-01T00:00:00Z','1'*40,'2'*40),('newer','2099-01-01T00:00:00Z','3'*40,'4'*40)]:
        wanted={'observed_at':date,'pins':{'d6g8k5htny-coder/main':main_pin,'d6g8k5htny-coder/Math-':math_pin}}
        def current_generation(source):
            body=source.replace(data['observed_at'],date)
            for repository,pin in wanted['pins'].items():body=body.replace(atlas_identity['pins'][repository],pin)
            return body
        body=current_generation(current_body)
        case={'case':'static-reading-material-and-current-mixed-generations','current_static_generation':generation,'passed':False,'page_errors':[],
              'scope':'Controlled local fixture with actual inspector-link activation; historical tracked fixture is not live project Source; no public cache or AT inference',
              'advertised_material_archive':archive,'atlas_identity':atlas_identity,'material_identity':material_identity,'current_static_identity':wanted,
              'fixture_bodies':{'atlas_html':{'bytes':len(html.encode()),'sha256':sha256(html.encode()).hexdigest()},'material_payload':{'bytes':len(material_raw),'sha256':sha256(material_raw).hexdigest()},'material_node':{'bytes':len(preserved_body),'sha256':sha256(preserved_body).hexdigest()},'current_node':{'bytes':len(body.encode()),'sha256':sha256(body.encode()).hexdigest()}},'requests':[]}
        report['cases'].append(case);context=browser.new_context(viewport={'width':1200,'height':900},reduced_motion='reduce');page=context.new_page();page.set_default_timeout(15000);page.on('pageerror',lambda e:case['page_errors'].append(str(e)))
        def serve_atlas(route):
            case['requests'].append(route.request.url);route.fulfill(status=200,content_type='text/html',body=html)
        def serve_material(route):
            case['requests'].append(route.request.url);route.fulfill(status=200,content_type='text/html',body=preserved_body)
        def serve_current(route):
            case['requests'].append(route.request.url)
            paths={origin+'nodes/'+quote(key,safe='')+'.html':ROOT/'docs/site/nodes'/(quote(key,safe='')+'.html') for key in data['graph']['data']['nodes']}
            source=paths.get(route.request.url)
            if source is None:route.continue_();return
            response=current_generation(source.read_text())
            case.setdefault('served_current_nodes',[]).append({'url':route.request.url,'bytes':len(response.encode()),'sha256':sha256(response.encode()).hexdigest()})
            route.fulfill(status=200,content_type='text/html',body=response)
        try:
            require((date< data['observed_at']) if generation=='older' else (date>data['observed_at']),'Mixed-generation fixture is not older/newer than the admitted atlas')
            require(wanted['pins']!=atlas_identity['pins'],'Mixed-generation static fixture did not change repository pins')
            page.route(origin+'dependencies.html*',serve_atlas);page.route(origin+preserved_path,serve_material);page.route(origin+'nodes/*.html',serve_current)
            page.goto(atlas_url);case['before_preserved_activation']=announced(page)
            preserved=page.locator('#atlas-inspector').get_by_role('link',name='Read the preserved mathematical record',exact=True)
            preserved.focus();page.keyboard.press('Enter');expect(page.locator('main > h1')).to_have_text(node)
            case['preserved_destination']=static_reading(page);check_reading(case['preserved_destination'],material_identity,preserved_path)
            neighbor=edge['to'];neighbor_name=quote(neighbor,safe='')+'.html'
            page.locator('main a[href="'+neighbor_name+'"]').first.click();expect(page.locator('main > h1')).to_have_text(neighbor)
            case['preserved_neighbor']=static_reading(page);check_reading(case['preserved_neighbor'],material_identity,'snapshots/'+archive+'/nodes/'+neighbor_name,neighbor)
            page.goto(atlas_url);case['before_current_activation']=announced(page)
            page.locator('#atlas-inspector').get_by_role('link',name='Read the static node record',exact=True).click();expect(page.locator('main > h1')).to_have_text(node)
            case['current_destination']=static_reading(page);check_reading(case['current_destination'],wanted,current_path)
            # Retain the existing generated relative-neighbor behavior at the
            # actual activated alias, without replacing the destination by goto.
            neighbor=edge['to'];page.locator('main a[href="'+quote(neighbor,safe='')+'.html"]').first.click()
            expect(page.locator('main > h1')).to_have_text(neighbor)
            case['static_neighbor']=static_reading(page);check_reading(case['static_neighbor'],wanted,'nodes/'+quote(neighbor,safe='')+'.html',neighbor)
            require(not case['page_errors'],'Mixed-generation reading emitted page errors');case['passed']=True
        except Exception:case['error']=traceback.format_exc()
        finally:context.close()
    for pointer_state in ['missing','ambiguous']:
        contents=section.group(2)
        if pointer_state=='missing':
            for href in hrefs:
                if re.fullmatch(r'snapshots/[0-9a-f]{64}/nodes/index\.html',href):contents=contents.replace('href="'+href+'"','href="#unavailable-fixture"')
        else:
            other='b'*64 if archive!='b'*64 else 'c'*64
            contents+='<a href="snapshots/'+other+'/nodes/index.html">Controlled second material pointer</a>'
        unavailable_html=html[:section.start(2)]+contents+html[section.end(2):]
        case={'case':'static-reading-preserved-unavailable','pointer_state':pointer_state,'passed':False,'page_errors':[],
              'scope':'Controlled generated-page pointer fixture; no inferred archive replacement','fixture_html_sha256':sha256(unavailable_html.encode()).hexdigest()}
        report['cases'].append(case);context=browser.new_context(viewport={'width':1200,'height':900},reduced_motion='reduce');page=context.new_page();page.set_default_timeout(15000);page.on('pageerror',lambda e:case['page_errors'].append(str(e)))
        try:
            page.route(origin+'dependencies.html*',lambda route:route.fulfill(status=200,content_type='text/html',body=unavailable_html));page.goto(atlas_url)
            expect(page.locator('#atlas-inspector h3')).to_have_text(node)
            expect(page.locator('#atlas-inspector').get_by_role('link',name='Read the preserved mathematical record',exact=True)).to_have_count(0)
            expect(page.locator('#atlas-inspector')).to_contain_text('Preserved mathematical record unavailable from this observation.')
            current=page.locator('#atlas-inspector').get_by_role('link',name='Read the static node record',exact=True)
            expect(current).to_be_visible();expect(current.locator('..')).to_contain_text('The capture date and source revisions on that page may differ from this view.')
            require(current.get_attribute('href')==current_path,'Unavailable archive changed the current static destination')
            case['before_activation']={'url':page.url,'node':node,'current_href':current.get_attribute('href'),'explanation':current.locator('..').inner_text()}
            current.focus();page.keyboard.press('Enter');expect(page.locator('main > h1')).to_have_text(node)
            case['current_destination']=static_reading(page);check_reading(case['current_destination'],atlas_identity,current_path)
            require(not case['page_errors'],'Unavailable-preserved reading emitted page errors');case['passed']=True
        except Exception:case['error']=traceback.format_exc()
        finally:context.close()

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
            exercise_atlas_shell(browser,origin,args.output,report,data,expect)
            exercise_hosted_amendments(browser,origin,args.output,report,data,expect)
            exercise_resume_parity(browser,origin,report,data,expect)
            exercise_static_reading_destinations(browser,origin,report,data,expect)
            for width in [1200,320]:
                for scheme in ['light','dark']:
                    case={'width':width,'color_scheme':scheme,'passed':False};report['cases'].append(case)
                    context=browser.new_context(viewport={'width':width,'height':900},color_scheme=scheme,reduced_motion='reduce');page=context.new_page();page.set_default_timeout(20000);errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
                    try:
                        page.goto(origin+'dependencies.html');expect(page.locator('#atlas-status')).to_contain_text('Source snapshot');expect(page.locator('.current-state')).to_contain_text('Verified source observation')
                        open_ancestor_disclosures(page.get_by_text('Search the current public source inventory', exact=True))
                        page.get_by_text('Search the current public source inventory', exact=True).click();page.locator('.current-search').fill('universality');expect(page.locator('.current-results')).to_contain_text('experiments/universality');require('source-q=universality' in page.url,'Source search state missing');source_matches=sum('universality' in (s['repository']+' '+s['path']).lower() for s in data['sources']);expect(page.locator('.current-more')).to_be_hidden() if source_matches<=50 else expect(page.locator('.current-more')).to_be_visible()
                        initial=page.locator('#atlas-canvas .atlas-node').count();page.locator('#atlas-expand').click();expanded=page.locator('#atlas-canvas .atlas-node').count();require(expanded>=initial,'Expansion lost nodes')
                        graph_ids=page.locator('#atlas-canvas .atlas-node').evaluate_all('(nodes)=>nodes.map(n=>n.firstChild.textContent)');page.locator('#atlas-view').select_option('list');list_ids=page.locator('#atlas-list button').evaluate_all('(nodes)=>nodes.map(n=>n.firstChild.textContent)');require(graph_ids==list_ids,'Graph/List mismatch')
                        page.locator('#atlas-list button').last.click();expect(page.locator('#atlas-inspector')).to_contain_text('Kernel metadata not recorded');expect(page.locator('#atlas-inspector')).to_contain_text('No explicit graph-to-artifact binding')
                        node=page.locator('#atlas-inspector h3').inner_text();page.reload();expect(page.locator('#atlas-inspector h3')).to_have_text(node);case['focus_ring_measurements']=exercise_focus_ring(page)
                        eq=data['equations'][0];page.locator('#equation-select').select_option(eq['id']);expect(page.locator('#equation-content pre')).to_have_text(eq['text']);require('equation=' in page.url,'Equation URL state absent');page.reload();expect(page.locator('#equation-content pre')).to_have_text(eq['text'])
                        if width<=720:page.locator('#atlas-browse-link').click()
                        page.locator('#atlas-search').fill('witness-collision');expect(page).to_have_url(re.compile(r'.*atlas-q=witness-collision.*'));page.reload();expect(page.locator('#atlas-search')).to_have_value('witness-collision');page.locator('#atlas-collapse').click();page.locator('#atlas-view').select_option('graph');open_ancestor_disclosures(page.locator('#atlas-zoom-in'));page.locator('#atlas-zoom-in').focus();page.keyboard.press('Enter');require(page.locator('#atlas-canvas').evaluate('(e)=>e.style.width')!='','Keyboard zoom failed')
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
                open_ancestor_disclosures(page.locator('#atlas-back'))
                page.locator('#atlas-back').focus();page.keyboard.press('Enter');expect(page.locator('#atlas-inspector h3')).to_have_text('math.uniform-matrix-cap-lifetime');page.locator('#atlas-forward').focus();page.keyboard.press('Enter');expect(page.locator('#atlas-inspector h3')).to_have_text('hist.CH-LIFT')
                page.locator('#atlas-reset').focus();page.keyboard.press('Enter');page.locator('#atlas-expand').focus();page.keyboard.press('Enter');page.locator('#atlas-collapse').focus();page.keyboard.press('Enter');page.locator('#atlas-view').focus();page.keyboard.press('Home');page.keyboard.press('Enter');require(page.locator('#atlas-view').input_value()=='graph','Keyboard Graph switch failed');page.keyboard.press('End');page.keyboard.press('Enter');require(page.locator('#atlas-view').input_value()=='list','Keyboard List switch failed');page.locator('#atlas-focus').focus();page.keyboard.press('Enter');require(page.evaluate('document.activeElement.id')=='atlas-inspector','Return-to-focus did not focus inspector')
                disclosure=page.locator('#atlas-inspector summary').last;disclosure.focus();page.keyboard.press('Enter');require(page.locator('#atlas-inspector details').last.evaluate('(e)=>e.open'),'Keyboard source disclosure failed');page.locator('#equation-search').focus();page.keyboard.type('ell');page.locator('#equation-next').focus();page.keyboard.press('Enter');expect(page.locator('#equation-content pre')).to_have_count(1)
                page.locator('#atlas-tab-diagnostics').click();diagnostic=page.locator('details').filter(has=page.locator('#atlas-diagnostics'));diagnostic.locator('summary').focus();page.keyboard.press('Enter');diagnostic.locator('button').first.focus();page.keyboard.press('Enter');expect(page.locator('#atlas-inspector h3')).to_have_text(next(iter(data['graph']['data']['nodes'])))
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
                open_ancestor_disclosures(page.get_by_text('Search the current public source inventory', exact=True))
                page.get_by_text('Search the current public source inventory', exact=True).click();page.locator('.current-search').fill('universality')
                page.locator('#equation-search').fill(first_equation['path']);page.locator('#equation-select').select_option(first_equation['id'])
                page.locator('#atlas-search').fill('CH-LIFT');expect(page.locator('#atlas-list button').filter(has_text=re.compile(r'^hist\.CH-LIFT'))).to_have_count(1)
                before,initial_share=settled_selection(page,'before-select','math.uniform-matrix-cap-lifetime','list','CH-LIFT','universality',first_equation['path'],first_equation)
                # Actual node activation creates the second browser history entry.
                page.locator('#atlas-list button').filter(has_text=re.compile(r'^hist\.CH-LIFT')).click();expect(page.locator('#atlas-inspector h3')).to_have_text('hist.CH-LIFT')
                page.locator('#atlas-view').select_option('graph');page.locator('#atlas-search').fill('witness-collision');page.locator('.current-search').fill('frontiers')
                page.locator('#equation-search').fill('ell');page.locator('#equation-select').select_option(next_equation['id'])
                after,selected_share=settled_selection(page,'after-select','hist.CH-LIFT','graph','witness-collision','frontiers','ell',next_equation)
                open_ancestor_disclosures(page.locator('#atlas-back'));page.locator('#atlas-back').click()
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
                    open_ancestor_disclosures(page.locator('#atlas-back'));page.locator('#atlas-back').click();expect(page.locator('#atlas-inspector h3')).to_have_text(current_node)
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
                        refused=origin+'dependencies.html?'+urlencode({**query,'atlas-panel':'diagnostics','atlas-view':'list','equation':equation['id']})+'#current-atlas';page.goto(refused);expect(page.locator('#atlas-error')).to_contain_text(error_text);expect(page.locator('#atlas-canvas .atlas-node')).to_have_count(0);expect(page.locator('#equation-content pre')).to_have_count(0)
                        diagnostic=page.locator('details').filter(has=page.locator('#atlas-diagnostics'));diagnostic.locator('summary').click();diagnostic.locator('button').filter(has_text=re.compile('^'+re.escape(current_node)+':')).click();expect(page.locator('#atlas-inspector h3')).to_have_text(current_node)
                        # Explicitly choosing this current excerpt is also user
                        # action; the refused bookmark must not supply it silently.
                        page.locator('#equation-select').select_option(equation['id']);expect(page.locator('#equation-content pre')).to_have_text(equation['text']);selected=history_checkpoint(page,label+'-explicit-current');guard['checkpoints'].append(selected);valid_share(selected,current_node)
                        open_ancestor_disclosures(page.locator('#atlas-back'));page.locator('#atlas-back').click();expect(page.locator('#atlas-error')).to_contain_text(error_text);expect(page.locator('#atlas-inspector h3')).to_have_count(0);expect(page.locator('#atlas-canvas .atlas-node')).to_have_count(0);expect(page.locator('#atlas-list button')).to_have_count(0);expect(page.locator('#equation-content pre')).to_have_count(0);expect(page.locator('#equation-content')).to_contain_text('Source-bound bookmark refused')
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
                    context=browser.new_context(viewport={'width':width,'height':900},forced_colors='active',reduced_motion='reduce');page=context.new_page();page.goto(origin+'dependencies.html?atlas-node=math.p15-full-price&atlas-view=list');expect(page.locator('#atlas-inspector h3')).to_have_text('math.p15-full-price');page.evaluate("(()=>{const entries=[document.documentElement,document.body,...document.querySelectorAll('body *')].map(e=>[e,parseFloat(getComputedStyle(e).fontSize)]);for(const [e,n] of entries)e.style.setProperty('font-size',n*2+'px','important');})()");open_ancestor_disclosures(page.locator('#atlas-reset'));page.locator('#atlas-reset').focus();page.keyboard.press('Enter');dimensions=page.evaluate('({width:innerWidth,document:document.documentElement.scrollWidth,font:parseFloat(getComputedStyle(document.body).fontSize)})');require(dimensions['font']>=30,'Text did not enlarge to 200 percent');require(dimensions['document']<=dimensions['width'],f'Enlarged text document overflow: {dimensions}');require(page.locator('#atlas-next').is_visible() and page.locator('#equation-search').is_visible(),'Essential controls lost after enlargement');page.locator('#atlas-view').select_option('graph');require(page.locator('.atlas-edge.context').first.evaluate('(e)=>getComputedStyle(e).strokeDasharray') not in ['none',''],'Context edge loses monochrome pattern');open_ancestor_disclosures(page.locator('#atlas-count'));expect(page.locator('#atlas-count')).to_contain_text('Solid arrows: required; dashed arrows: context');path=args.output/f'enlarged-{width}.png';page.locator('#current-atlas').screenshot(path=str(path));context.close()
                case['passed']=True
            except Exception:case['error']=traceback.format_exc()
            case={'case':'synthetic-large-render-window','passed':False,'scope':'Synthetic display fixture only; does not describe the research corpus or source acquisition.'};report['cases'].append(case);context=browser.new_context();page=context.new_page()
            try:
                synthetic=json.loads(json.dumps(data));nodes={'fixture.root':{'kind':'candidate','classification':'OPEN_ACTIVE','layer':'D1'}};nodes.update({f'fixture.node{i:04d}':{'kind':'candidate','classification':'OPEN_ACTIVE','layer':'D1'} for i in range(1000)});synthetic['graph']['data']['nodes']=nodes;synthetic['graph']['data']['edges']=[{'from':'fixture.root','to':node,'required':True,'relation':'synthetic_required_fixture'} for node in nodes if node!='fixture.root'];fixture_raw=json.dumps(synthetic,ensure_ascii=False).encode();fixture_sha=sha256(fixture_raw).hexdigest();html=(ROOT/'docs/site/dependencies.html').read_text().replace(report['source_payload_sha256'],fixture_sha).replace(f'data-current-bytes="{len(raw)}"',f'data-current-bytes="{len(fixture_raw)}"');page.route('**/dependencies.html?*',lambda route:route.fulfill(status=200,content_type='text/html',body=html));page.route('**/current.json',lambda route:route.fulfill(status=200,content_type='application/json',body=fixture_raw));started=time.perf_counter();page.goto(origin+'dependencies.html?atlas-node=fixture.root&atlas-view=list');expect(page.locator('#atlas-list button')).to_have_count(150);open_ancestor_disclosures(page.locator('#atlas-count'));expect(page.locator('#atlas-count')).to_contain_text('1001 nodes in the loaded neighborhood');expect(page.locator('#atlas-count')).to_contain_text('851 outside this page');first=page.locator('#atlas-list button').evaluate_all('(nodes)=>nodes.map(n=>n.firstChild.textContent)');open_ancestor_disclosures(page.locator('#atlas-next'));page.locator('#atlas-next').click();expect(page.locator('#atlas-count')).to_contain_text('851 outside this page');second=page.locator('#atlas-list button').evaluate_all('(nodes)=>nodes.map(n=>n.firstChild.textContent)');require(not set(first)&set(second),'Paged large fixture duplicated the first window');page.locator('#atlas-previous').click();require(first==page.locator('#atlas-list button').evaluate_all('(nodes)=>nodes.map(n=>n.firstChild.textContent)'),'Previous window is not deterministic');case['load_and_page_seconds']=round(time.perf_counter()-started,3);session=context.new_cdp_session(page);session.send('Performance.enable');case['js_heap_bytes']=next((m['value'] for m in session.send('Performance.getMetrics')['metrics'] if m['name']=='JSHeapUsedSize'),None);case['fixture_bytes']=len(fixture_raw);case['fixture_sha256']=fixture_sha;case['indexed_nodes']=len(nodes);case['rendered_nodes']=150;case['passed']=True
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
                page.route('**/current.json*',lambda route:route.fulfill(status=200,content_type='application/json',body='{}'));page.goto(origin+'dependencies.html');expect(page.locator('.current-state')).to_contain_text('Unavailable');expect(page.locator('#atlas-status')).to_contain_text('Unavailable');require(page.locator('#atlas-canvas .atlas-node').count()==0,'Refusal rendered graph');case['passed']=True
            except Exception:case['error']=traceback.format_exc()
            finally:context.close()
            try:exercise_observation_coherence(browser,origin,args.output,report,data,expect)
            finally:browser.close()
    except Exception:report['error']=traceback.format_exc()
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    passed=not report.get('error') and all(c['passed'] for c in report['cases']);print(f"Current-site browser: {'PASS' if passed else 'FAIL'} ({sum(c['passed'] for c in report['cases'])}/{len(report['cases'])} cases); scientific effect NONE.")
    return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
