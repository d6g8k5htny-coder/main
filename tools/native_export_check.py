#!/usr/bin/env python3
"""Check exported container structure against bounded native topology/formulas.

Native formula strings are preserved separately. Textual transformations are
not asserted equivalent; native behavior, comments, suggestions and history
are not backed up by these checks. DOCX body text is not compared character
by character with the native document. No scientific result is established.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
import gzip
import hashlib
import io
import json
from pathlib import Path
import posixpath
import sys
import xml.etree.ElementTree as ET
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.drive_reconcile import DEFAULT,require,sha
from tools.drive_coverage import DEFAULT as BASELINE
from tools.manifest_integrity_check import local_file

NS={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
    'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}


def a1(row,column):
    prefix=''
    while column:
        column,digit=divmod(column-1,26);prefix=chr(65+digit)+prefix
    return prefix+str(row)


def inspect_container(data,row):
    out={k:row[k] for k in ('id','export_mime_type')}
    out.update(bytes=len(data),sha256=sha(data))
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        require(len(z.namelist())==len(set(z.namelist())),'duplicate container member')
        require(z.testzip() is None,'container CRC failure')
        if 'word/document.xml' in z.namelist():
            body=ET.fromstring(z.read('word/document.xml')).find('w:body',NS)
            require(body is not None,'missing DOCX body')
            text=''.join(t.text or '' for t in body.iter('{'+NS['w']+'}t'))
            out.update(kind='doc',document_body=True,body_text_characters=len(text),
                       body_text_sha256=sha(text.encode()),paragraphs=len(list(body.iter('{'+NS['w']+'}p'))))
        else:
            book=ET.fromstring(z.read('xl/workbook.xml'))
            rels={t.attrib['Id']:t.attrib['Target'] for t in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
            sheets=[]
            for t in book.find('s:sheets',NS):
                path=rels[t.attrib['{'+NS['r']+'}id']]
                path=path.lstrip('/') if path.startswith('/') else posixpath.normpath('xl/'+path)
                xml=ET.fromstring(z.read(path));formulas=[];cells=0
                for cell in xml.findall('.//s:sheetData/s:row/s:c',NS):
                    cells+=1;value=cell.find('s:f',NS)
                    if value is not None:
                        formulas.append({'cell':cell.attrib['r'],'formula':'='+(value.text or ''),'attributes':dict(value.attrib)})
                sheets.append({'title':t.attrib['name'],'state':t.attrib.get('state','visible'),
                               'path':path,'cells':cells,'formulas':formulas})
            out.update(kind='sheet',sheets=sheets)
    out['container_valid']=True
    return out


class Reader:
    """Reuse ZIP directory handles; verify chunks and reconstructed identities."""
    def __init__(self,root,stack):
        self.root=root;self.base=root/BASELINE;self.stack=stack;self.packs={}
        self.index=json.loads((self.base/'objects.json').read_text())

    def read(self,row):
        storage=row.get('storage')
        if storage is None:data=local_file(self.root,row['path']).read_bytes()
        elif 'file' in storage:data=local_file(self.root,storage['file']).read_bytes()
        else:
            parts=[]
            for digest in self.index['objects'][storage['object']]['chunks']:
                rec=self.index['chunks'][digest];pack=rec['pack']
                if pack not in self.packs:
                    path=local_file(self.base,pack)
                    require(sha(path.read_bytes())==path.stem,'pack SHA mismatch')
                    self.packs[pack]=self.stack.enter_context(zipfile.ZipFile(path))
                chunk=self.packs[pack].read(digest)
                require(len(chunk)==rec['bytes'] and sha(chunk)==digest,'chunk identity mismatch')
                parts.append(chunk)
            data=b''.join(parts)
        require(len(data)==row['bytes'] and sha(data)==row['sha256'],'export identity mismatch')
        return data


def compare(native,exports):
    require(len({r['id'] for r in native})==len(native),'duplicate native identity')
    require(len({r['id'] for r in exports})==len(exports),'duplicate export identity')
    by_id={r['id']:r for r in exports}
    require(set(by_id)=={r['id'] for r in native},'native/export identity sets differ')
    counts=Counter();names=[];syntax=[]
    for row in native:
        ex=by_id[row['id']]
        require(isinstance(row.get('observed_at'),str) and row['observed_at'],'missing native observation time')
        require(ex.get('container_valid') is True,'invalid export container')
        require(row['kind']==ex['kind'],'native/export type mismatch')
        if row['kind']=='doc':
            require(isinstance(row.get('revision_id'),str) and row['revision_id'],'missing native document revision observation')
            require(len(row['tabs'])==1,'multi-tab Doc requires native content supplement')
            require(row['tabs'][0].get('tabId'),'native tab identity missing')
            require(ex.get('document_body') is True,'missing DOCX document body')
            counts['docs_single_tab']+=1
        else:
            counts['workbooks']+=1
            expected_ranges=["'"+s['properties']['title'].replace("'","''")+"'!A1:"+
                             a1(s['properties']['gridProperties']['rowCount'],s['properties']['gridProperties']['columnCount'])
                             for s in row['sheets']]
            require(row.get('requested_ranges')==expected_ranges,'formula read did not cover every native grid')
            require(len(row['sheets'])==len(ex['sheets']),'missing exported workbook tab')
            require(len({s['properties']['sheetId'] for s in row['sheets']})==len(row['sheets']),'duplicate native sheet')
            for position,(ns,es) in enumerate(zip(row['sheets'],ex['sheets'])):
                props=ns['properties']
                require(props['index']==position,'native sheet order omitted or changed')
                counts['sheets']+=1
                if props['title']!=es['title']:
                    names.append({'id':row['id'],'index':position,'native':props['title'],'export':es['title']})
                nf={a1(f['row'],f['column']):f['formula'] for f in ns['formulas']}
                ef={f['cell']:f for f in es['formulas']}
                require(len(nf)==len(ns['formulas']) and len(ef)==len(es['formulas']),'duplicate formula coordinates')
                require(set(nf)==set(ef),'missing or added formula cell')
                counts['formula_cells']+=len(nf)
                masters={f['attributes'].get('si') for f in ef.values() if f['attributes'].get('t')=='shared' and f['formula']!='='}
                for cell,formula in nf.items():
                    require(isinstance(formula,str) and formula.startswith('='),'missing native formula string')
                    item=ef[cell]
                    if item['attributes'].get('t')=='shared':
                        require(item['attributes'].get('si') in masters,'unresolved shared-formula master')
                        counts['shared_formula_cells']+=1
                    elif item['formula']==formula:counts['literal_equal_nonshared_formulas']+=1
                    else:
                        counts['transformed_nonshared_formulas']+=1
                        syntax.append({'id':row['id'],'sheet_id':props['sheetId'],'cell':cell,'native':formula,'export':item['formula']})
    return {'counts':dict(sorted(counts.items())),'tab_title_transformations':names,
            'nonshared_formula_transformation_examples':syntax[:20],
            'all_nonshared_transformations_sha256':sha(json.dumps(syntax,sort_keys=True,separators=(',',':')).encode()),
            'all_native_formula_strings_preserved':True,
            'scope':'Tab topology and formula-cell coverage only; shared formula representations are counted without semantic equivalence claims. No native behavior, full document text comparison, comments, suggestions or revision-history completeness.'}


def check(root=ROOT,bundle=DEFAULT,verify_containers=False):
    base=local_file(root,bundle+'/native-fidelity.json').parent
    spec=json.loads((base/'native-fidelity.json').read_text())
    native=json.loads(gzip.decompress((base/'native-topology-formulas.json.gz').read_bytes()))
    exports=json.loads(gzip.decompress((base/'export-structure.json.gz').read_bytes()))
    rows={r['id']:r for r in [json.loads(x) for x in (root/BASELINE/'coverage.jsonl').read_text().splitlines()] if r['status']=='NATIVE_EXPORT'}
    refresh=json.loads((base/'reconciliation.json').read_text())['payloads']
    rows.update({r['id']:r for r in refresh if r['status']=='NATIVE_EXPORT'})
    require(set(rows)=={r['id'] for r in native},'not every native export is audited')
    for ex in exports:
        row=rows[ex['id']]
        require(all(ex[k]==row[k] for k in ('bytes','sha256','export_mime_type')),'structure refers to different export bytes')
    if verify_containers:
        with ExitStack() as stack:
            reader=Reader(root,stack)
            for ex in exports:
                row=rows[ex['id']]
                require(inspect_container(reader.read(row),row)==ex,'container structure observation mismatch')
    result=compare(native,exports)
    require(result==spec['comparison'],'native fidelity report does not reproduce')
    require(spec.get('historical_revisions_complete') is False and spec.get('native_behavior_complete') is False,'native fidelity overclaim')
    return {'native_exports':len(rows),**{k:v for k,v in result.items() if k!='nonshared_formula_transformation_examples'},
            'container_bytes_reparsed':verify_containers}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--bundle',default=DEFAULT);p.add_argument('--verify-containers',action='store_true');a=p.parse_args()
    try:print(json.dumps(check(a.root,a.bundle,a.verify_containers),sort_keys=True))
    except (ValueError,KeyError,OSError) as e:raise SystemExit('Native export check failed: '+str(e))


if __name__=='__main__':main()
