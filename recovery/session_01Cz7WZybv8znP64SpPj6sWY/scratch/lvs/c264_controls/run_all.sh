#!/usr/bin/env bash
# Full re-run of the C264 controls evidence in this lane directory only (the worktree is read, never written).
set -u
D=<SCRATCH>/lvs/c264_controls
C=<SCRATCH>/repos/math_c263reg/companions/torus_filtration_20261011
G=$D/gate
busy() { n=$(pgrep -c -x lean); if [ "${n:-0}" -ge 2 ]; then echo "ABORT: $n lean processes running"; exit 3; fi; }
cd $D
sha256sum $D/OrdinaryH0Controls.lean $C/OrdinaryH0.lean $C/ClosedSuperlevel.lean
# 1. module compile (task command; LEAN_PATH also names this lane directory, where the dependency .oleans are)
busy; ./compile.sh $D/OrdinaryH0Controls.lean $D/build-final.log; echo "--- build-final.log:"; cat $D/build-final.log
# 2. .olean for the audit
rm -f $D/OrdinaryH0Controls.olean $D/OrdinaryH0Controls.ilean
busy; ./compile.sh $D/OrdinaryH0Controls.lean $D/build-olean.log -o $D/OrdinaryH0Controls.olean; cat $D/build-olean.log
# 3. axioms and census
python3 -B -I - "$C/check.py" "$D/OrdinaryH0Controls.lean" > $D/targets.txt <<'PY'
import importlib.util, re, sys
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('torus_check', sys.argv[1])
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
c.MODULES = c.MODULES + ('OrdinaryH0Controls',)
c.PRELUDE['import'] = re.compile(r'import (?:Mathlib(?:\.[A-Za-z0-9_]+)*|' + '|'.join(c.MODULES) + ')')
scan = c.scan_module(open(sys.argv[2], encoding='utf-8').read(), 'OrdinaryH0Controls')
for k in ('theorem', 'abbrev', 'def', 'instance'):
    for n in scan[k]:
        print('TorusFiltration.' + n)
PY
(echo "import OrdinaryH0Controls"; sed 's/^/#print axioms /' $D/targets.txt) > $D/AxiomsControls.lean
busy; ./compile.sh $D/AxiomsControls.lean $D/axioms.log
busy; ./compile.sh $D/CensusControls.lean $D/census.log
# 4. emulated registered side directory: the four companion modules from unmodified worktree bytes, this module,
#    and a candidate root TorusFiltration = ROOT_SOURCE + 'import OrdinaryH0Controls'
rm -f $G/side/* $G/logs/*
for M in TorusTopology ClosedSuperlevel SuperlevelBar OrdinaryH0; do
  busy; $G/gatelean.sh $G/logs/build-$M.log $G/logs/build-$M.status -DwarningAsError=true --root=$C -o $G/side/$M.olean $C/$M.lean
done
busy; $G/gatelean.sh $G/logs/build-OrdinaryH0Controls.log $G/logs/build-OrdinaryH0Controls.status -DwarningAsError=true --root=$D -o $G/side/OrdinaryH0Controls.olean $D/OrdinaryH0Controls.lean
busy; $G/gatelean.sh $G/logs/build-TorusFiltration.log $G/logs/build-TorusFiltration.status -DwarningAsError=true --root=$G/rootsrc -o $G/side/TorusFiltration.olean $G/rootsrc/TorusFiltration.lean
echo "--- build logs (bytes):"; wc -c $G/logs/build-*.log | tail -n +1
# 5. the gate's own emit(), then the false controls exactly as replay.sh runs them
rm -rf $G/out $G/discrim; mkdir -p $G/out $G/discrim
python3 -B -I $G/gate_harness.py emit $G/out
python3 -B -I $G/gate_harness.py discrim $G/discrim $G/discrim.json
ALL="RejectSuperlevelSignFlip RejectSuperlevelBelowThreshold RejectConstantNotPeriodic RejectEssentialDeathBound RejectReducedPointClass RejectMergerKillsPointClass RejectSeparationWithoutNonzeroCoeff RejectNonmergerKillsDifference"
NEW="RejectMergerKillsPointClass RejectSeparationWithoutNonzeroCoeff RejectNonmergerKillsDifference"
for control in $ALL; do busy; $G/gatelean.sh $G/out/$control.log $G/out/$control.status -DwarningAsError=true --root=$G/out $G/out/$control.lean; done
for control in $NEW; do busy; $G/gatelean.sh $G/discrim/$control.log $G/discrim/$control.status -DwarningAsError=true --root=$G/discrim $G/discrim/$control.lean; done
args=""; for c in $ALL; do args="$args $c:$G/out/$c.log:$G/out/$c.status"; done
for c in $NEW; do args="$args $c:$G/discrim/$c.log:$G/discrim/$c.status"; done
python3 -B -I $G/gate_harness.py negative $args
# 6. kernel replay of this module's declarations against the prebuilt dependency .oleans
busy; cd <SCRATCH>/repos/math_c263reg/formal
code=0; PATH="$HOME/.elan/bin:$PATH" LEAN_NUM_THREADS=2 timeout 1200 lake env bash -c 'export LEAN_PATH="'"$G/side"':$LEAN_PATH"; exec leanchecker OrdinaryH0Controls' > $G/logs/leanchecker.log 2>&1 || code=$?
printf '%s\n' "$code" > $G/logs/leanchecker.status; echo "leanchecker OrdinaryH0Controls exit=$code log_bytes=$(wc -c < $G/logs/leanchecker.log)"
cd $D
# 7. gate audits
python3 -B -I $G/gate_harness.py axioms $D/axioms.log $D/targets.txt
python3 -B -I $G/gate_harness.py scan $D/OrdinaryH0Controls.lean
python3 -B -I $D/scan_harness.py $C/check.py $D/OrdinaryH0Controls.lean | head -1
python3 -I -c "
t=set(open('$D/targets.txt').read().split())
lines=[l.split() for l in open('$D/census.log') if l.strip() and not l.startswith('CENSUS_END')]
ur={n for n,a,b in lines if a=='user' and b=='range'}
print('census user-range == inventory:', ur==t, len(ur), '; other:', [(n,a,b) for n,a,b in lines if (a,b)!=('user','range')])"
cmp $D/OrdinaryH0Controls.olean $G/side/OrdinaryH0Controls.olean && echo "lane and gate/side OrdinaryH0Controls.olean byte-identical"
echo "statuses: $(for f in $G/logs/*.status; do printf '%s=%s ' $(basename $f .status) $(cat $f); done)"
