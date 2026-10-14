"""Mutation controls for main#399 current-model.mjs (reviewer-private until delivery).

usage: python3 -I mutate.py WORKTREE LOG
Each mutation must match exactly once; the file is restored after every run and
its blob is re-checked at the end.
"""
import hashlib
import subprocess
import sys
import time

WT, LOG = sys.argv[1], sys.argv[2]
PATH = WT + "/docs/site/current-model.mjs"
MUTANTS = [
    ("M01 tie-break drops relation", "a.to.localeCompare(b.to)||a.relation.localeCompare(b.relation)", "a.to.localeCompare(b.to)"),
    ("M02 targets ordered by id only", "targets.sort((a,b)=>a.length-b.length||a.id.localeCompare(b.id))", "targets.sort((a,b)=>a.id.localeCompare(b.id))"),
    ("M03 incoming edge off by one", "start+j?route.edges[start+j-1]:null", "start+j?route.edges[start+j]:null"),
    ("M04 binary search admits empty boundary segment", "analysis.prefixSums[mid+1]<=offset", "analysis.prefixSums[mid+1]<offset"),
    ("M05 page not clamped", "page=Math.min(requested,pages)", "page=requested"),
    ("M06 required-only reach ignores required flag", "if((requiredOnly&&edge.required!==true)||visited.has(edge.from))continue;", "if(visited.has(edge.from))continue;"),
    ("M07 snapshot index not cached", "snapshotIndexes.set(graph,index);", ""),
    ("M08 selected-root analysis not cached", "if(index.analysis?.root===id)return index.analysis;", ""),
    ("M09 route page accepts 0", "Number(rawRoutePage)>0", "Number(rawRoutePage)>=0"),
    ("M10 relation rows re-sorted", "rows=(direction==='outgoing'?index.outgoing:index.incoming).get(id)||[]", "rows=[...((direction==='outgoing'?index.outgoing:index.incoming).get(id)||[])].sort((a,b)=>a.relation.localeCompare(b.relation))"),
    ("M11 last same-depth predecessor wins", "traversedEdges++;if(depths.has(edge.to))continue;", "traversedEdges++;if(depths.has(edge.to)){if(depths.get(edge.to)===depth+1)predecessors.set(edge.to,edge);continue;}"),
    ("M12 BFS cutoff at 5000 dequeued nodes", "for(let cursor=0;cursor<queue.length;cursor++){", "for(let cursor=0;cursor<Math.min(queue.length,5000);cursor++){"),
    ("M13 window limit not clamped to 150", "const limit=Number.isInteger(selected.limit)&&selected.limit>0?Math.min(selected.limit,MAX_WINDOW_NODES):MAX_WINDOW_NODES;", "const limit=Number.isInteger(selected.limit)&&selected.limit>0?selected.limit:MAX_WINDOW_NODES;"),
    ("M14 window materializes all preceding routes", "for(let i=lo;i<analysis.targetCount", "for(let i=0;i<analysis.targetCount"),
    ("M15 route length excludes root", "targets.push({id:current,length:depth+1})", "targets.push({id:current,length:depth})"),
    ("M16 root not excluded as target", "if(current!==id&&unresolved(graph.nodes[current]))", "if(unresolved(graph.nodes[current]))"),
    ("M17 reach walks outgoing edges", "for(const edge of index.incoming.get(queue[cursor])){", "for(const edge of index.outgoing.get(queue[cursor])){"),
    ("M18 index keeps unknown-endpoint edges", "if(!hasNode(graph,edge.from)||!hasNode(graph,edge.to))continue;", "if(!hasNode(graph,edge.from))continue;"),
    ("M19 route page skips safe-integer check", "Number.isSafeInteger(Number(rawRoutePage))&&", ""),
]


def blob(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


with open(PATH, "rb") as f:
    original = f.read()
text = original.decode()
out = ["original blob %s" % blob(original)]
for name, old, new in MUTANTS:
    n = text.count(old)
    if n != 1:
        out.append("%s: SETUP ERROR pattern count %d" % (name, n))
        continue
    with open(PATH, "w") as f:
        f.write(text.replace(old, new))
    t0 = time.time()
    try:
        p = subprocess.run(["node", "--test", "tests/test_current_site.mjs"], cwd=WT, capture_output=True, text=True, timeout=600)
        res = p.stdout
        fail = [l for l in res.splitlines() if l.startswith("not ok")]
        summary = {k: next((l.split()[-1] for l in res.splitlines() if l.startswith("# " + k + " ")), "?") for k in ("pass", "fail")}
        verdict = "CAUGHT" if p.returncode != 0 else "SURVIVED"
        out.append("%s: %s (rc %d, pass %s fail %s, %.1fs) %s" % (name, verdict, p.returncode, summary["pass"], summary["fail"], time.time() - t0, "; ".join(x[:110] for x in fail)))
    except subprocess.TimeoutExpired:
        out.append("%s: CAUGHT-BY-TIMEOUT (600s)" % name)
    finally:
        with open(PATH, "wb") as f:
            f.write(original)
    with open(LOG, "w") as f:
        f.write("\n".join(out) + "\n")
with open(PATH, "rb") as f:
    restored = f.read()
out.append("restored blob %s %s" % (blob(restored), "OK" if restored == original else "MISMATCH"))
with open(LOG, "w") as f:
    f.write("\n".join(out) + "\n")
print("\n".join(out))
