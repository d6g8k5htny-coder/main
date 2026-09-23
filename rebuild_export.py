#!/usr/bin/env python3
"""Lossless, hash-pinned byte transport; not register migration or mathematical evidence."""
from pathlib import Path, PurePosixPath
import argparse, base64, hashlib, io, json, lzma, re, zipfile, zlib

PAYLOAD_SHA256 = 'dc226db04ff7d2d195498ed92c0ec55e0fb15b6adecb269e9e59efd484e6fac0'
SOURCE_SHA256 = 'c3229ecefc642f3e32f23cb8320fb66236d11affb83f67bd135153d068a3f460'
TARGET_SHA256 = '6cec54cefa3ee8f9885dcfd763407a2e0274d2887d73bdc23ccc7793cd721469'
TARGET_BYTES = 1979318
CELL_START = re.compile(rb'<c\s[^>]*>')
STYLE = re.compile(rb'\bs="(\d+)"')
VALUE = re.compile(rb'(<v>)(\d+)(</v>)')

def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def un64(value):
    return base64.b64decode(value, validate=True)

def apply_delta(source, operations):
    pieces=[]
    for op in operations:
        if isinstance(op, str):
            pieces.append(op.encode('latin1'))
        else:
            require(isinstance(op,list) and len(op)==2, 'Bad copy operation')
            start,length=op
            require(type(start) is int and type(length) is int and 0<=start<=start+length<=len(source), 'Copy outside source')
            pieces.append(source[start:start+length])
    return b''.join(pieces)

def shared_items(data):
    items=[]
    for m in re.finditer(rb'<si(?:\s|>)',data):
        end=data.find(b'</si>',m.start())
        require(end>=m.end(),'Unterminated shared string')
        items.append(data[m.start():end+5])
    return items

def worksheet_ids(data):
    values=set()
    for m in CELL_START.finditer(data):
        tag=m.group(0)
        if b't="s"' not in tag or tag.endswith(b'/>'):
            continue
        end=data.find(b'</c>',m.end())
        require(end>=m.end(),'Unterminated cell')
        v=re.search(rb'<v>(\d+)</v>',data[m.end():end])
        if v:
            values.add(int(v.group(1)))
    return sorted(values)

def rebuild(source_bytes, payload):
    require(sha(source_bytes)==SOURCE_SHA256, 'Source hash mismatch')
    require(sha(payload)==PAYLOAD_SHA256, 'Payload hash mismatch')
    recipe=json.loads(lzma.decompress(payload, memlimit=268435456))
    require(recipe['version']==5 and recipe['literal_encoding']=='latin1','Unsupported recipe')
    require(recipe['source_sha256']==SOURCE_SHA256 and recipe['target_sha256']==TARGET_SHA256,'Recipe identity mismatch')
    require(recipe['target_bytes']==TARGET_BYTES and recipe['source_bytes']==len(source_bytes),'Recipe size mismatch')
    with zipfile.ZipFile(io.BytesIO(source_bytes)) as archive:
        require(archive.testzip() is None,'Source CRC failure')
        original={i.filename:archive.read(i) for i in archive.infolist()}
    old_sst=original['xl/sharedStrings.xml'];old_items=shared_items(old_sst)
    sst=recipe['sst'];stream=apply_delta(old_sst,sst['literal_ops']);literal_items=[];pos=0
    for length in sst['literal_lengths']:
        require(type(length) is int and length>=0,'Invalid literal length')
        literal_items.append(stream[pos:pos+length]);pos+=length
    require(pos==len(stream),'Literal length mismatch')
    require(sst['queue_order']=='sorted','Unsupported queue order')
    queues=[worksheet_ids(original[f'xl/worksheets/sheet{n}.xml']) for n in range(1,45)]
    pointers=[-1]*44;plan=un64(sst['sequence_plan']);position=0;next_literal=0;index_map={};new_items=[]
    def read_unsigned():
        nonlocal position
        value=0;shift=0
        while True:
            require(position<len(plan) and shift<=28,'Invalid sequence integer')
            byte=plan[position];position+=1;value|=(byte&127)<<shift
            if byte<128:
                return value
            shift+=7
    for target_index in range(sst['sequence_count']):
        require(position<len(plan),'Truncated sequence')
        token=plan[position];position+=1
        if token==88:
            require(next_literal<len(literal_items),'Literal sequence overflow')
            new_items.append(literal_items[next_literal]);next_literal+=1
            continue
        if token==89:
            old_index=read_unsigned()
        else:
            require(0<=token<88,'Unknown sequence token')
            queue=token%44
            if token<44:
                delta=1
            else:
                encoded=read_unsigned();delta=encoded//2 if encoded%2==0 else -(encoded//2)-1
            pointers[queue]+=delta
            require(0<=pointers[queue]<len(queues[queue]),'Queue outside source')
            old_index=queues[queue][pointers[queue]]
        require(0<=old_index<len(old_items) and old_index not in index_map,'Invalid repeated shared string')
        index_map[old_index]=target_index;new_items.append(old_items[old_index])
    require(position==len(plan) and next_literal==len(literal_items),'Sequence trailing data')
    new_sst=un64(sst['prefix'])+b''.join(new_items)+un64(sst['suffix'])
    style_map=recipe['style_map']
    def normalize_sheet(data):
        chunks=[];position=0
        for match in CELL_START.finditer(data):
            tag=match.group(0)
            end=match.end() if tag.endswith(b'/>') else data.find(b'</c>',match.end())+4
            require(end>=match.end(),'Unterminated source cell')
            cell=data[match.start():end]
            cell=STYLE.sub(lambda m:b's="'+str(style_map.get(m.group(1).decode(),int(m.group(1)))).encode()+b'"',cell)
            if b't="s"' in tag and not tag.endswith(b'/>'):
                cell=VALUE.sub(lambda m:m.group(1)+str(index_map.get(int(m.group(2)),int(m.group(2)))).encode()+m.group(3),cell)
            chunks.append(data[position:match.start()]);chunks.append(cell);position=end
        return b''.join(chunks)+data[position:]
    pieces=[un64(recipe['prefix'])]
    for entry in recipe['entries']:
        require(not PurePosixPath(entry['name']).is_absolute() and '..' not in PurePosixPath(entry['name']).parts,'Unsafe member path')
        data=original.get(entry['source'],b'')
        if entry['kind']=='worksheet_delta':
            data=normalize_sheet(data)
        elif entry['kind']=='shared_strings':
            data=new_sst
        else:
            require(entry['kind']=='copy_delta','Unknown member kind')
        data=apply_delta(data,entry['ops'])
        if entry['method']==8:
            compressor=zlib.compressobj(entry['level'],zlib.DEFLATED,-15)
            compressed=compressor.compress(data)+compressor.flush()
        else:
            require(entry['method']==0,'Unsupported compression')
            compressed=data
        pieces.extend([un64(entry['header']),compressed,un64(entry['post'])])
    pieces.append(un64(recipe['tail']));result=b''.join(pieces)
    require(len(result)==TARGET_BYTES and sha(result)==TARGET_SHA256,'Target reconstruction mismatch; no output written')
    with zipfile.ZipFile(io.BytesIO(result)) as archive:
        require(archive.testzip() is None and len(archive.infolist())==153,'Target ZIP mismatch')
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--payload',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=rebuild(args.source.read_bytes(),args.payload.read_bytes())
    if args.output.exists():
        require(args.output.read_bytes()==result,'Refusing to replace different existing output')
    else:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('xb') as out:
            out.write(result)
    print(json.dumps({'bytes':len(result),'sha256':sha(result),'zip_crc_pass':True,'zip_members':153,'source_unchanged':True,'scientific_status_changed':False,'canonical_import_completed':False,'zlib':zlib.ZLIB_RUNTIME_VERSION}))

if __name__=='__main__':
    main()
