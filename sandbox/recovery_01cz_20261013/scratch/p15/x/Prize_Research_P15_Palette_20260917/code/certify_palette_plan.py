#!/usr/bin/env python3
"""Exact, bounded-explicit palette certificate checker (not a discovery solver)."""
from __future__ import annotations
import argparse, hashlib, json, sys
from fractions import Fraction as F
from pathlib import Path
from math import ceil
from exact_core import GuardFailure, need, rational, mask, minimal, localize, scalar_certificate, validate_palettes

MAX_EXPLICIT_LOCAL_VERTICES = 16

def validate(data: dict) -> dict:
    need(isinstance(data,dict),'json-object-required')
    n=data.get('n');K=data.get('K')
    need(isinstance(n,int) and not isinstance(n,bool) and 0<n<=10000,'explicit-ground-set-range')
    need(isinstance(K,int) and not isinstance(K,bool) and K>=1,'global-palette-positive-integer')
    rawedges=data.get('minimal_witnesses'); rawblocks=data.get('blocks')
    need(isinstance(rawedges,list) and isinstance(rawblocks,list) and rawblocks,'explicit-lists-required')
    edges=[]
    for e in rawedges:
        need(isinstance(e,list) and e and all(isinstance(v,int) and not isinstance(v,bool) and 0<=v<n for v in e),'explicit-witness-range')
        need(len(e)==len(set(e)),'duplicate-witness-coordinate')
        edges.append(mask(e))
    need(len(edges)==len(set(edges)) and tuple(sorted(edges,key=lambda e:(e.bit_count(),e)))==minimal(edges),'source-must-be-minimal-antichain')
    expected=data.get('source_witness_sha256')
    canonical=json.dumps(sorted([sorted(e) for e in rawedges]),separators=(',',':')).encode()
    binding=hashlib.sha256(canonical).hexdigest()
    if expected is not None:need(expected==binding,'source-witness-identity')
    blocks=[];palettes=[];demands=[];local_receipts=[]
    for block in rawblocks:
        need(isinstance(block,dict),'block-object')
        verts=block.get('vertices');pal=block.get('palette');certificate=block.get('local_certificate')
        need(isinstance(verts,list) and verts and isinstance(pal,list) and isinstance(certificate,dict),'block-fields')
        need(all(isinstance(v,int) and not isinstance(v,bool) and 0<=v<n for v in verts),'block-vertex-range')
        need(len(verts)==len(set(verts)),'block-duplicate-coordinate')
        need(all(isinstance(c,int) and not isinstance(c,bool) for c in pal),'integer-palette-labels')
        es=localize(edges,verts)
        kind=certificate.get('kind')
        if kind=='proper_coloring':
            col=certificate.get('colors')
            need(isinstance(col,list) and len(col)==len(verts) and all(isinstance(c,int) and not isinstance(c,bool) and c>=0 for c in col),'proper-color-list')
            for e in es:
                colors={col[i] for i in range(len(verts)) if e>>i&1}
                need(len(colors)>=2,'local-monochromatic-witness')
            demand=max(1,len(set(col)))
            local_receipts.append({'kind':kind,'vertices':len(verts),'demand':demand,'all_local_obstructions_empty':True})
        elif kind=='scalar_sandwich':
            need(len(verts)<=MAX_EXPLICIT_LOCAL_VERTICES,'scalar-exhaustive-implementation-cap')
            vals=certificate.get('weights')
            need(isinstance(vals,list) and len(vals)==len(verts),'scalar-weight-list')
            a=[rational(v) for v in vals]; kappa=rational(certificate.get('kappa'))
            scalar_certificate(len(verts),es,a,kappa)
            demand=ceil(408*kappa)
            local_receipts.append({'kind':kind,'vertices':len(verts),'demand':demand,'kappa':str(kappa),'all_subsets_checked':1<<len(verts)})
        else:raise GuardFailure('unsupported-local-certificate')
        blocks.append(verts);palettes.append(pal);demands.append(demand)
    validate_palettes(n,edges,blocks,palettes,demands,K)
    return {'status':'PASS','n':n,'K':K,'blocks':len(blocks),'minimal_witnesses':len(edges),'source_witness_sha256':binding,'local_certificates':local_receipts,'conclusion':'conditional on the documented P14 scalar theorem where used, product-price cover cost <= min(1,-log mu(D)); if all local certificates are proper colorings, the complete ground set is properly K-colorable','external_review_credit':0,'implementation_scalar_cap':MAX_EXPLICIT_LOCAL_VERTICES}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('input',type=Path);args=ap.parse_args()
    try:
        data=json.loads(args.input.read_text())
        result=validate(data)
    except (GuardFailure,ValueError,TypeError,KeyError,IndexError) as e:
        print(json.dumps({'status':'REJECT','reason':str(e)},sort_keys=True));return 1
    print(json.dumps(result,sort_keys=True));return 0
if __name__=='__main__':sys.exit(main())
