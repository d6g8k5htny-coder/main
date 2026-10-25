#!/usr/bin/env python3
"""Step-by-step comparison of this replication's Orca speech with the executor's report.
usage: python3 -I compare.py EVIDENCE_ROOT SCENARIO [SCENARIO...]
The executor's per-step claims (non-Say-All, non-outer-status items, plus counts) are
transcribed below from the report handed to the replicator."""
import json, re, sys
from pathlib import Path

PL, W, F = 'page-load', 'widget', 'focus'
REPORTED = {
 'S1': [
  {'items': [(PL, 'Finished loading c220 start.')], 'say_all': 1},
  {'items': [(PL, 'Loading.  Please wait.'), (PL, 'Finished loading Source records & exhibits · Universal Law.'),
             (PL, 'Page has 8 landmarks, 11 headings, 65 unvisited links.'),
             (W, '2D fallback: the optional 3D library or WebGL could not load. All pin coordinates and six data labels remain visible.')],
   'outer': 8, 'say_all': 51},
  {'items': [], 'say_all': 1}],
 'S2': [
  {'items': [(PL, 'Finished loading c220 start.')], 'say_all': 1},
  {'items': [(PL, 'Finished loading Source records & exhibits · Universal Law.'), (PL, 'Page has 8 landmarks, 7 headings, 59 unvisited links.'),
             (W, '2D fallback: the optional 3D library or WebGL could not load. All pin coordinates and six data labels remain visible.'),
             (W, '2D fallback: the optional 3D library or WebGL could not load. All pin coordinates and six data labels remain visible.')],
   'outer': 8, 'say_all': 48},
  {'items': [], 'say_all': 0}],
 'S3': [
  {'items': [(PL, 'Finished loading c220 start.')], 'say_all': 1},
  {'items': [(PL, 'Loading.  Please wait.'), (PL, 'Finished loading Source records & exhibits · Universal Law.'),
             (PL, 'Page has 9 landmarks, 67 headings, 1 table, 157 unvisited links.'),
             (W, '2D coordinate diagram. Checking optional 3D view…'),
             (W, '3D coordinate plane: drag with mouse or touch, or focus the canvas and use arrow keys. Orbit is a camera control only.')],
   'outer': 8, 'say_all': 15},
  {'items': [(F, 'Show 2D diagram push button.')], 'say_all': 11},
  {'items': [(W, '2D coordinate diagram. All pin coordinates and six data labels remain visible.')], 'say_all': 12},
  {'items': [], 'say_all': 7},
  {'items': [(W, '3D coordinate plane: drag with mouse or touch, or focus the canvas and use arrow keys. Orbit is a camera control only.')], 'say_all': 33},
  {'items': [], 'say_all': 0}],
 'S4': [
  {'items': [(PL, 'Finished loading c220 start.')], 'say_all': 1},
  {'items': [(PL, 'Finished loading Source records & exhibits · Universal Law.'), (PL, 'Page has 8 landmarks, 17 headings, 75 unvisited links.'),
             (W, 'Selected 3 of 3: the full triple is not two-decomposable.')], 'outer': 8, 'say_all': 27},
  {'items': [(F, 'Toggle x1 toggle button pressed.')], 'say_all': 9},
  {'items': [(W, 'Selected 2 of 3: this proper subset is two-decomposable.')], 'say_all': 9},
  {'items': [], 'say_all': 7},
  {'items': [(W, 'Selected 3 of 3: the full triple is not two-decomposable.')], 'say_all': 9},
  {'items': [(F, 'Swap palette colors push button.')], 'say_all': 6},
  {'items': [], 'say_all': 25},
  {'items': [], 'say_all': 0}],
 'S5': [
  {'items': [(PL, 'Finished loading c220 start.')], 'say_all': 1},
  {'items': [(PL, 'Loading.  Please wait.'), (PL, 'Finished loading Source records & exhibits · Universal Law.'),
             (PL, 'Page has 8 landmarks, 11 headings, 65 unvisited links.'),
             (W, '3D coordinate plane: drag with mouse or touch, or focus the canvas and use arrow keys. Orbit is a camera control only.')],
   'outer': 8, 'say_all': 22},
  {'items': [(W, '2D fallback: the graphics context became unavailable. The source-defined geometry remains visible.')], 'say_all': 36},
  {'items': [], 'say_all': 2}],
 'S3F': [
  {'items': [(PL, 'Finished loading c220 start.')], 'say_all': 1},
  {'items': [(PL, 'Finished loading Source records & exhibits · Universal Law.'), (PL, 'Page has 8 landmarks, 11 headings, 65 unvisited links.'),
             (W, '3D coordinate plane: drag with mouse or touch, or focus the canvas and use arrow keys. Orbit is a camera control only.')], 'outer': 8, 'say_all': 0},
  {'items': [(F, 'Show 2D diagram push button.')], 'say_all': 0},
  {'items': [(F, 'Show 3D view'), (W, '2D coordinate diagram. All pin coordinates and six data labels remain visible.')], 'say_all': 0},
  {'items': [], 'say_all': 0},
  {'items': [(F, 'Show 2D diagram'), (W, '3D coordinate plane: drag with mouse or touch, or focus the canvas and use arrow keys. Orbit is a camera control only.')], 'say_all': 0},
  {'items': [], 'say_all': 0}],
 'S4F': [
  {'items': [(PL, 'Finished loading c220 start.')], 'say_all': 1},
  {'items': [(PL, 'Finished loading Source records & exhibits · Universal Law.'), (PL, 'Page has 8 landmarks, 7 headings, 59 unvisited links.'),
             (W, 'Selected 3 of 3: the full triple is not two-decomposable.')], 'outer': 7, 'say_all': 0},
  {'items': [(F, 'Choose a subset of the three coordinates panel.'), (F, 'Toggle x1 toggle button pressed.')], 'say_all': 0},
  {'items': [(F, 'not pressed'), (W, 'Selected 2 of 3: this proper subset is two-decomposable.')], 'say_all': 0},
  {'items': [], 'say_all': 0},
  {'items': [(F, 'pressed'), (W, 'Selected 3 of 3: the full triple is not two-decomposable.')], 'say_all': 0},
  {'items': [(F, 'Swap palette colors push button.')], 'say_all': 0},
  {'items': [], 'say_all': 0},
  {'items': [], 'say_all': 0}],
}

def mine_items(step):
    items, outer, harness = [], 0, []
    for d in step['spoken']:
        k = d['kind']
        if k.startswith('outer museum status'):
            outer += 1
        elif k == 'Orca startup/shutdown' or (step['step'] == 0 and 'window:activate' in k):
            harness.append(d['text'])
        elif k == 'Orca page-load message':
            items.append((PL, d['text'], d))
        elif k == 'WIDGET live region':
            items.append((W, d['text'], d))
        else:
            items.append((F, d['text'], d))
    return items, outer, harness

root = Path(sys.argv[1])
report = {}
for sc in sys.argv[2:]:
    steps = json.loads(root.joinpath(sc, 'steps.json').read_text())
    rep = REPORTED[re.sub(r'r\d+$', '', sc)]
    print(f'===== {sc}')
    sc_out = []
    for i, st in enumerate(steps):
        items, outer, harness = mine_items(st)
        r = rep[i] if i < len(rep) else {'items': [], 'say_all': None}
        mine_t = [(k, t) for k, t, _ in items]
        theirs = [tuple(x) for x in r['items']]
        same = mine_t == theirs
        diffs = []
        if not same:
            for x in theirs:
                if x not in mine_t: diffs.append(f'reported only: {x[1]!r}')
            for x in mine_t:
                if x not in theirs: diffs.append(f'replicated only: {x[1]!r}')
            if not diffs: diffs.append('same items, different order or multiplicity')
        if r.get('outer') is not None and r['outer'] != outer:
            diffs.append(f'outer status count reported {r["outer"]}, replicated {outer}')
        print(f'-- step {st["step"]}: {st["action"]}  [{"AGREE" if same else "DIFFER"}]')
        for k, t, d in items:
            print(f'     sent {d["sent"]} played {d["played"]} [{d["kind"]}] {t}')
        print(f'     outer status msgs: {outer} (reported {r.get("outer")}); Say All chunks: {st["say_all_chunks"]} (reported {r.get("say_all")}); harness-only: {harness}')
        for x in diffs: print('     DIFF', x)
        sc_out.append({'step': st['step'], 'action': st['action'], 'agree': same, 'diffs': diffs,
                       'spoken': [f'sent {d["sent"]}, played {d["played"]} [{d["kind"]}] {t}' for k, t, d in items],
                       'outer': outer, 'say_all': st['say_all_chunks'], 'harness': harness})
    report[sc] = sc_out
root.joinpath('comparison.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
