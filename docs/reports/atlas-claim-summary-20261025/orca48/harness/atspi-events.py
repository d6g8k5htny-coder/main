#!/usr/bin/python3
"""Observe real AT-SPI events; never invoke Orca internals or synthesize speech."""
import json, time
import pyatspi

def event(e):
    try:
        source = e.source
        record = dict(time=time.time(), type=e.type, detail1=e.detail1,
                      detail2=e.detail2, role=source.getRoleName(),
                      name=source.name, app=getattr(source.getApplication(),'name',None),
                      attributes=source.getAttributes(),
                      data=str(e.any_data))
        print(json.dumps(record, ensure_ascii=False), flush=True)
    except Exception as exc:
        print(json.dumps(dict(time=time.time(), error=str(exc))), flush=True)

pyatspi.Registry.registerEventListener(event, 'object:text-changed',
    'object:property-change:accessible-name', 'object:state-changed:focused',
    'document:load-complete', 'object:announcement', 'object:children-changed')
print(json.dumps(dict(time=time.time(), observer='ready')), flush=True)
pyatspi.Registry.start()
