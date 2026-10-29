"""Bounded real-renderer admission; the caller owns the browser and source server.

Only stdlib and the caller's existing Playwright objects are used. Sphere centers
come from the shipped spatial projection and camera reset contract. Selection is
then proved by a real mouse event hitting CANVAS, never a label/list click.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import time
import traceback
from urllib.parse import parse_qs, unquote, urlencode, urlsplit


def choose_seed(graph):
    """Choose a small recorded connected neighborhood, without proof-name hints."""
    nodes = graph['nodes']
    neighbors = {node: set() for node in nodes}
    for edge in graph['edges']:
        a, b = edge['from'], edge['to']
        if a in nodes and b in nodes and a != b:
            neighbors[a].add(b)
            neighbors[b].add(a)
    candidates = [node for node in nodes if neighbors[node]]
    if not candidates:
        raise ValueError('Spatial admission needs a recorded connected neighbor')
    return min(candidates, key=lambda node: (
        not bool(nodes[node].get('source')), not 2 <= len(neighbors[node]) <= 8,
        len(neighbors[node]) > 8, -min(len(neighbors[node]), 8), node))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


_GEOMETRY = """() => {
 const cards=[...document.querySelectorAll('#atlas-canvas .atlas-node')].map(e=>{
  const r=e.getBoundingClientRect();return {id:e.firstChild.textContent,
   x:r.x,y:r.y,width:r.width,height:r.height};});const overlaps=[];
 for(let i=0;i<cards.length;i++)for(let j=i+1;j<cards.length;j++){
  const a=cards[i],b=cards[j],w=Math.min(a.x+a.width,b.x+b.width)-Math.max(a.x,b.x),
   h=Math.min(a.y+a.height,b.y+b.height)-Math.max(a.y,b.y);
  if(w>1&&h>1)overlaps.push({a:a.id,b:b.id,width:w,height:h});
 }return {cards,overlaps};
}"""

# Read the shipped projection; reconstruct the documented initial/reset camera.
# No renderer internals, onPick callback, or DOM activation is invoked here.
_PROJECTED_SPHERES = """async graph => {
 const resources=performance.getEntriesByType('resource').map(r=>r.name);
 const modelURL=resources.find(u=>new URL(u).pathname.endsWith('/atlas-spatial-model.mjs'));
 const threeURL=resources.find(u=>new URL(u).pathname.endsWith('/three.module.js'));
 if(!modelURL||!threeURL)throw Error('Shipped scene modules were not loaded');
 const [{spatialProjection},T]=await Promise.all([import(modelURL),import(threeURL)]);
 const selected=document.querySelector('#atlas-inspector h3').textContent;
 const ids=[...document.querySelectorAll('#atlas-list button')].map(e=>e.firstChild.textContent);
 const view=spatialProjection(graph,ids,selected,false),bounds=new T.Box3();
 for(const p of Object.values(view.positions))bounds.expandByPoint(new T.Vector3(...p));
 const radius=Math.max(22,bounds.getSize(new T.Vector3()).length()*1.1),phi=1.1,theta=.12;
 const canvas=document.querySelector('#atlas-space-host canvas'),r=canvas.getBoundingClientRect();
 const camera=new T.PerspectiveCamera(50,r.width/r.height,.1,3000);
 camera.position.set(radius*Math.sin(phi)*Math.sin(theta),radius*Math.cos(phi),radius*Math.sin(phi)*Math.cos(theta));
 camera.lookAt(new T.Vector3());camera.updateProjectionMatrix();camera.updateMatrixWorld();
 const points=view.ids.filter(id=>id!==selected).map(id=>{
  const p=new T.Vector3(...view.positions[id]).project(camera),x=r.x+(p.x+1)*r.width/2,y=r.y+(1-p.y)*r.height/2;
  return {id,x,y,z:p.z,canvasHit:document.elementFromPoint(x,y)===canvas};
 }).filter(p=>p.z>-1&&p.z<1&&p.x>r.left+3&&p.x<r.right-3&&p.y>r.top+3&&p.y<r.bottom-3&&p.canvasHit);
 return {points,ids,selected,resetRadius:radius,modules:{modelURL,threeURL}};
}"""


def _screenshot(page, output, row, suffix, selector=None):
    path = Path(output) / (row['case'] + '-' + suffix + '.png')
    if selector:
        page.locator(selector).screenshot(path=str(path))
    else:
        page.screenshot(path=str(path))
    row['observations'].setdefault('screenshots', []).append({
        'path': path.name, 'bytes': path.stat().st_size,
        'sha256': sha256(path.read_bytes()).hexdigest()})


def _inspect(page, data, node, expect):
    graph = data['graph']['data']
    expect(page.locator('#atlas-inspector h3')).to_have_text(node)
    metadata = json.loads(page.locator('#atlas-node-details > pre').text_content())
    _require(metadata == graph['nodes'][node], 'Inspector metadata differs from admitted source node')
    groups = {}
    for group, direction, other in [('dependencies', 'from', 'to'), ('dependents', 'to', 'from')]:
        edges = [edge for edge in graph['edges'] if edge[direction] == node]
        expected = [edge[other] for edge in edges]
        actual = page.locator('#atlas-' + group + ' button').evaluate_all(
            '(buttons)=>buttons.map(b=>b.firstChild.textContent)')
        _require(actual == expected, 'Inspector directed relationship IDs differ from source: ' + group)
        cards = page.locator('#atlas-' + group + ' .atlas-relation-card')
        for index, edge in enumerate(edges):
            card = cards.nth(index)
            _require(card.locator('.atlas-relation-kind').text_content() ==
                     ('Required relationship' if edge['required'] else 'Context relationship'),
                     'Relationship requirement label differs from source')
            details = card.locator('details > p').all_text_contents()
            _require(details[:2] == ['Recorded type: ' + edge['relation'],
                     edge['from'] + ' → ' + edge['to']], 'Relationship type/direction differs from source')
        groups[group] = actual
    source = next((s for s in data['sources'] if s['repository'] == data['graph']['identity']['repository']
                   and s['path'] == metadata.get('source')), None)
    link = page.locator('#atlas-claim-summary a', has_text='Immutable source reference')
    _require(link.count() == int(source is not None), 'Inspector source-link presence differs from admitted source')
    href = link.get_attribute('href') if source else None
    if source:
        parsed = urlsplit(href)
        _require(parsed.scheme == 'https' and parsed.netloc == 'github.com'
                 and unquote(parsed.path) == '/' + source['repository'] + '/blob/'
                 + source['commit'] + '/' + source['path'], 'Inspector source link changed repository/revision/path')
    return {'node': node, 'relationships': groups, 'source_href': href,
            'metadata_sha256': sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()}


def _source_state(page, data):
    snapshot = page.locator('#atlas-snapshot-record').text_content()
    _require(data['graph']['identity']['commit'][:8] in snapshot, 'Atlas snapshot text lacks admitted graph revision')
    return {'snapshot_text': snapshot, 'repositories': data['repositories'],
            'url': page.url, 'history_length': page.evaluate('history.length')}


def _pins(url, data, node=None):
    query = parse_qs(urlsplit(url).query)
    _require(query.get('snapshot') == [data['graph']['identity']['commit']], 'Share/history changed graph source pin')
    main = next(r['commit'] for r in data['repositories'] if r['repository'].endswith('/main'))
    _require(query.get('main-snapshot') == [main], 'Share/history changed Main source pin')
    if node is not None:
        _require(query.get('atlas-node') == [node], 'Share/history changed selected exact node')


def _browse(page, width, expect):
    if width <= 720 and page.locator('#current-atlas').get_attribute('data-atlas-panel') != 'browse':
        page.locator('#atlas-browse-link').click()
    expect(page.locator('#atlas-space-toggle')).to_be_visible()


def _activate(page, row, expect):
    started = time.perf_counter()
    page.locator('#atlas-space-toggle').click()
    # Visibility drives the on-demand renderer. Scroll the attached host before
    # waiting for frames; a failed/hidden host simply remains a failed positive.
    page.locator('#atlas-space-host').evaluate(
        "e=>e.scrollIntoView({block:'center',inline:'nearest',behavior:'instant'})")
    page.wait_for_function("()=>document.querySelector('#atlas-space-host').dataset.ready==='true'||"
                           "/unavailable|context lost|refused/.test(document.querySelector('#atlas-space-status').textContent)")
    status = page.locator('#atlas-space-status').text_content()
    row['observations']['activation_status'] = status
    _require(page.locator('#atlas-space-host').get_attribute('data-ready') == 'true'
             and page.locator('#atlas-space-host canvas').count() == 1,
             'Positive 3D admission requires rendering, not fallback: ' + status)
    page.locator('#atlas-space-host').scroll_into_view_if_needed()
    expect(page.locator('#atlas-space-host')).to_have_attribute('data-ready', 'true')
    expect(page.locator('#atlas-space-status')).to_contain_text('3D ·')
    expect(page.locator('#atlas-space-host canvas')).to_have_count(1)
    frame = page.locator('#atlas-space-host').get_attribute('data-frames')
    _require(int(frame or 0) > 0, 'Positive admission requires actual rendered frames')
    gl = page.locator('#atlas-space-host canvas').evaluate("""e=>{
     const gl=e.getContext('webgl2')||e.getContext('webgl');if(!gl||gl.isContextLost())return null;
     const ext=gl.getExtension('WEBGL_debug_renderer_info');return {
      version:gl.getParameter(gl.VERSION),vendor:gl.getParameter(gl.VENDOR),renderer:gl.getParameter(gl.RENDERER),
      unmasked_vendor:ext?gl.getParameter(ext.UNMASKED_VENDOR_WEBGL):null,
      unmasked_renderer:ext?gl.getParameter(ext.UNMASKED_RENDERER_WEBGL):null,
      width:gl.drawingBufferWidth,height:gl.drawingBufferHeight,error:gl.getError()};} """)
    _require(gl and gl['width'] > 0 and gl['height'] > 0 and gl['error'] == 0,
             'Positive admission requires a usable, error-free actual GL context')
    row['observations']['activation_ms'] = (time.perf_counter() - started) * 1000
    row['observations']['gl'] = gl
    row['observations']['frames'] = int(frame)
    row['observations']['render_ms'] = page.locator('#atlas-space-host').get_attribute('data-render-ms')


def _restore_geometry_and_pick(page, row, data, width, expect):
    _browse(page, width, expect)
    page.locator('#atlas-view').select_option('graph')
    expect(page.locator('#atlas-stage')).to_be_visible()
    geometry = page.evaluate(_GEOMETRY)
    _require(len(geometry['cards']) >= 2, 'Useful fallback requires at least two admitted graph cards')
    _require(not geometry['overlaps'], 'Restored graph cards overlap: ' + str(geometry['overlaps']))
    target = max(geometry['cards'], key=lambda card: (card['height'], len(card['id']), card['id']))
    index = next(i for i, card in enumerate(geometry['cards']) if card['id'] == target['id'])
    card = page.locator('#atlas-canvas .atlas-node').nth(index)
    card.evaluate("e=>e.scrollIntoView({block:'center',inline:'center',behavior:'instant'})")
    point = card.locator('small').last.evaluate("""e=>{
     const range=document.createRange();range.selectNodeContents(e);const boxes=[...range.getClientRects()];
     const r=boxes[boxes.length-1],x=r.x+r.width/2,y=r.y+r.height/2;
     return {x,y,hit:document.elementFromPoint(x,y)?.closest('.atlas-node')?.firstChild.textContent};} """)
    _require(point['hit'] == target['id'], 'Restored metadata hit target belongs to another node')
    _screenshot(page, output=row['_output'], row=row, suffix='restored', selector='#atlas-stage')
    page.mouse.click(point['x'], point['y'])
    expect(page.locator('#atlas-inspector h3')).to_have_text(target['id'])
    _pins(page.url, data, target['id'])
    row['observations']['fallback_geometry'] = geometry
    row['observations']['metadata_mouse_pick'] = {'expected': target['id'], **point}
    row['observations']['restored_inspector'] = _inspect(page, data, target['id'], expect)


def _positive(page, row, data, width, expect):
    graph = data['graph']['data']
    seed = row['observations']['seed']
    _require(page.evaluate("matchMedia('(prefers-reduced-motion:reduce)').matches"), 'Reduced-motion context missing')
    _require(not page.evaluate("performance.getEntriesByType('resource').some(r=>r.name.includes('three.module'))"),
             'Three loaded before optional activation')
    _activate(page, row, expect)
    host = page.locator('#atlas-space-host')
    _require(1 < int(host.get_attribute('data-nodes')) <= 150, 'Rendered node count is missing or unbounded')
    _screenshot(page, row['_output'], row, 'canvas', '#atlas-space-host')
    candidates = page.evaluate(_PROJECTED_SPHERES, graph)
    points = sorted(candidates['points'], key=lambda p: (p['z'], p['id']))
    _require(points, 'No source-derived unselected sphere has an uncovered CANVAS hit target')
    point = points[0]
    canvas = page.locator('#atlas-space-host canvas')
    box = canvas.bounding_box()
    image = canvas.screenshot()
    path = Path(row['_output']) / (row['case'] + '-sphere.png')
    path.write_bytes(image)
    # Inspect the actual composited PNG at the uncovered sphere center. A label
    # or a transparent/blank canvas cannot satisfy this raster witness.
    raster = page.evaluate("""async ({png,x,y})=>{
     const image=await createImageBitmap(new Blob([new Uint8Array(png)],{type:'image/png'}));
     const c=new OffscreenCanvas(image.width,image.height),ctx=c.getContext('2d');ctx.drawImage(image,0,0);
     const px=Math.max(1,Math.min(image.width-2,Math.round(x))),py=Math.max(1,Math.min(image.height-2,Math.round(y)));
     const data=ctx.getImageData(px-1,py-1,3,3).data,colors=[];let spherePixels=0;
     for(let i=0;i<data.length;i+=4){const rgb=[data[i],data[i+1],data[i+2]];colors.push(rgb);
      if(Math.abs(rgb[0]-114)<=8&&Math.abs(rgb[1]-189)<=8&&Math.abs(rgb[2]-237)<=8)spherePixels++;}
     image.close();return {width:c.width,height:c.height,x:px,y:py,spherePixels,colors};} """,
        {'png': list(image), 'x': point['x'] - box['x'], 'y': point['y'] - box['y']})
    row['observations']['sphere_raster'] = {**raster, 'path': path.name,
        'sha256': sha256(image).hexdigest(), 'bytes': len(image)}
    _require(raster['spherePixels'] > 0, 'Projected CANVAS target has no actually rendered sphere pixels')
    before = page.url
    _require(page.evaluate("p=>document.elementFromPoint(p.x,p.y)==="
                           "document.querySelector('#atlas-space-host canvas')", point),
             'Actual raycast mouse point no longer hits CANVAS')
    started = time.perf_counter()
    page.mouse.click(point['x'], point['y'])
    expect(page.locator('#atlas-inspector h3')).to_have_text(point['id'])
    row['observations']['sphere_pick'] = {**point, 'hit_target': 'CANVAS',
        'ms': (time.perf_counter() - started) * 1000, 'projection_modules': candidates['modules']}
    _require(point['id'] != seed, 'Sphere picking did not change exact node selection')
    row['observations']['picked_inspector'] = _inspect(page, data, point['id'], expect)
    share = page.locator('#atlas-share').get_attribute('href')
    _pins(share, data, point['id'])
    _require(parse_qs(urlsplit(share).query).get('atlas-space') == ['3d'], 'Share lost active spatial intent')
    _require(share == page.url and page.url != before, 'Raycast selection did not update the shared address')
    page.reload()
    _inspect(page, data, point['id'], expect)
    page.go_back()
    expect(page.locator('#atlas-inspector h3')).to_have_text(seed)
    _require(page.url == before, 'Back did not restore the original spatial selection address')
    _pins(page.url, data, seed)
    page.go_forward()
    expect(page.locator('#atlas-inspector h3')).to_have_text(point['id'])
    _require(page.url == share, 'Forward did not restore the exact shared spatial address')
    _pins(page.url, data, point['id'])
    _browse(page, width, expect)
    host.scroll_into_view_if_needed()
    expect(host).to_have_attribute('data-ready', 'true')
    # A real reduced-motion keyboard change draws, then the idle renderer stops.
    canvas = page.locator('#atlas-space-host canvas')
    canvas.focus()
    old_frames = int(host.get_attribute('data-frames'))
    page.keyboard.press('ArrowLeft')
    page.wait_for_function("n=>Number(document.querySelector('#atlas-space-host').dataset.frames)>n", arg=old_frames)
    page.wait_for_timeout(100)
    settled = host.get_attribute('data-frames')
    page.wait_for_timeout(120)
    _require(host.get_attribute('data-frames') == settled, 'Reduced-motion renderer continues an idle frame loop')
    row['observations']['keyboard_frames'] = [old_frames, int(settled)]
    if width <= 720:
        page.locator('#atlas-open-reader').click()
    summary = page.locator('#atlas-node-details > summary')
    summary.focus()
    page.keyboard.press('Enter')
    expect(page.locator('#atlas-node-details')).to_have_attribute('open', '')
    original = summary.element_handle()
    _require(original.evaluate('(e)=>e.isConnected&&document.activeElement===e&&e.parentElement.open'),
             'Metadata summary was not open and actually focused before loss')
    # Desktop PRE is a scroll container. Narrow reading deliberately expands
    # PRE (current.css), so preserve its existing native Tab successor instead.
    page.keyboard.press('Tab')
    successor = page.evaluate_handle('document.activeElement')
    successor_state = successor.evaluate("e=>({tag:e.tagName,id:e.id,"
                                         "withinMetadata:!!e.closest('#atlas-node-details')})")
    if width > 720:
        _require(successor_state['tag'] == 'PRE' and successor_state['withinMetadata'],
                 'Desktop baseline Tab does not reach metadata PRE')
    else:
        _require(successor_state == {'tag': 'A', 'id': 'atlas-share', 'withinMetadata': False},
                 'Expanded phone metadata has an unexpected native Tab successor')
    page.keyboard.press('Shift+Tab')
    _require(original.evaluate('(e)=>e.isConnected&&document.activeElement===e&&e.parentElement.open'),
             'Baseline Tab/Shift+Tab did not return to the ORIGINAL metadata SUMMARY')
    row['observations']['baseline_tab_successor'] = successor_state
    state = _source_state(page, data)
    started = time.perf_counter()
    canvas.evaluate("e=>{const gl=e.getContext('webgl2')||e.getContext('webgl');"
                    "const ext=gl&&gl.getExtension('WEBGL_lose_context');"
                    "if(!ext)throw Error('Actual context-loss extension unavailable');ext.loseContext();}")
    expect(page.locator('#atlas-space-status')).to_contain_text('context lost')
    # Do not reacquire by ID: a replacement that restores the same ID must fail.
    _require(original.evaluate('(e)=>e.isConnected&&document.activeElement===e&&e.parentElement.open'),
             'Context loss detached, blurred or closed the ORIGINAL metadata SUMMARY')
    page.keyboard.press('Tab')
    next_focus = page.evaluate("()=>({tag:document.activeElement.tagName,"
                              "withinMetadata:!!document.activeElement.closest('#atlas-node-details')})")
    row['observations']['focus_after_loss_tab'] = next_focus
    _require(successor.evaluate('(e)=>e.isConnected&&document.activeElement===e'),
             'Next Tab did not retain the ORIGINAL native metadata successor')
    if width > 720:
        _require(next_focus == {'tag': 'PRE', 'withinMetadata': True},
                 'Next Tab did not continue into metadata PRE')
    _require(_source_state(page, data) == state, 'Context loss changed source, URL or history')
    expect(page.locator('#atlas-space-host canvas')).to_have_count(0)
    row['observations']['context_loss'] = {'ms': (time.perf_counter() - started) * 1000,
        'original_summary_retained': True, 'next_focus': next_focus, 'source_state': state}
    _restore_geometry_and_pick(page, row, data, width, expect)


def _fallback(page, row, data, width, expect, failure):
    before = _source_state(page, data)
    page.locator('#atlas-space-toggle').click()
    expect(page.locator('#atlas-space-status')).to_contain_text('unavailable')
    expect(page.locator('#atlas-space-host canvas')).to_have_count(0)
    after = _source_state(page, data)
    _require(after['snapshot_text'] == before['snapshot_text'], 'Fallback substituted the source observation')
    _require(after['history_length'] == before['history_length'] + 1, 'Failure added or removed activation history')
    _pins(page.url, data, row['observations']['seed'])
    row['observations']['failure'] = {'kind': failure,
        'status': page.locator('#atlas-space-status').text_content(), 'source': after}
    if failure == 'webgl-list':
        expect(page.locator('#atlas-list')).to_be_visible()
        _require(page.locator('#atlas-list button').count() >= 2,
                 'List fallback did not retain useful admitted nodes')
        _inspect(page, data, row['observations']['seed'], expect)
    _restore_geometry_and_pick(page, row, data, width, expect)


def exercise_spatial_admission(browser, origin, output, report, data, expect):
    """Append six bounded cases, preserving the caller's browser/report lifecycle."""
    output = Path(output)
    specs = [('render-pick-recovery-1200', 1200, 'dark', 'positive'),
             ('render-pick-recovery-390', 390, 'light', 'positive'),
             ('module-fallback-390', 390, 'light', 'module'),
             ('no-webgl-fallback-1200', 1200, 'dark', 'webgl'),
             ('no-webgl-list-fallback-390', 390, 'dark', 'webgl-list'),
             ('refused-source-390', 390, 'light', 'refused')]
    for name, width, theme, kind in specs:
        row = {'case': 'hosted-spatial-' + name, 'passed': False, 'page_errors': [],
               'observations': {'width': width, 'theme': theme, 'reduced_motion': 'reduce'},
               '_output': output}
        report['cases'].append(row)
        context = None
        started = time.perf_counter()
        try:
            seed = choose_seed(data['graph']['data'])
            row['observations']['seed'] = seed
            context = browser.new_context(viewport={'width': width, 'height': 900},
                                          color_scheme=theme, reduced_motion='reduce')
            if kind == 'module':
                context.route('**/atlas-spatial.mjs*', lambda route: route.abort())
            if kind in ('webgl', 'webgl-list'):
                context.add_init_script("const original=HTMLCanvasElement.prototype.getContext;"
                    "HTMLCanvasElement.prototype.getContext=function(type,...args){return "
                    "String(type).includes('webgl')?null:original.call(this,type,...args)}")
            page = context.new_page()
            page.set_default_timeout(10000)
            page.on('pageerror', lambda error, row=row: row['page_errors'].append(str(error)))
            main = next(r['commit'] for r in data['repositories'] if r['repository'].endswith('/main'))
            query = {'atlas-node': seed, 'atlas-view': 'list' if kind == 'webgl-list' else 'graph', 'atlas-panel': 'browse',
                     'snapshot': '0' * 40 if kind == 'refused' else data['graph']['identity']['commit'],
                     'main-snapshot': main}
            page.goto(origin.rstrip('/') + '/dependencies.html?' + urlencode(query))
            expect(page.locator('#atlas-space-toggle')).to_be_visible()
            if kind == 'refused':
                expect(page.locator('#atlas-error')).to_be_visible()
                old_url = page.url
                page.locator('#atlas-space-toggle').click()
                expect(page.locator('#atlas-space-status')).to_contain_text('refused')
                expect(page.locator('#atlas-canvas .atlas-node')).to_have_count(0)
                expect(page.locator('#atlas-space-host canvas')).to_have_count(0)
                _require(parse_qs(urlsplit(page.url).query)['snapshot'] == ['0' * 40], 'Refusal adopted current source pins')
                _require(not page.locator('#atlas-inspector h3').count(), 'Refusal substituted a current inspector')
                row['observations']['refusal'] = {'before': old_url, 'after': page.url,
                    'message': page.locator('#atlas-error').text_content()}
            else:
                row['observations']['initial_inspector'] = _inspect(page, data, seed, expect)
                row['observations']['source'] = _source_state(page, data)
                if kind == 'positive':
                    _positive(page, row, data, width, expect)
                else:
                    _fallback(page, row, data, width, expect, kind)
            _require(not row['page_errors'], 'Uncaught page errors: ' + repr(row['page_errors']))
            row['passed'] = True
        except Exception:
            row['error'] = traceback.format_exc()
        finally:
            if context is not None:
                try:
                    context.close()
                except Exception:
                    row['passed'] = False
                    row['error'] = row.get('error', '') + traceback.format_exc()
            if row['page_errors']:
                row['passed'] = False
                row.setdefault('error', 'Uncaught page errors: ' + repr(row['page_errors']))
            row.pop('_output')
            row['observations']['elapsed_ms'] = (time.perf_counter() - started) * 1000
