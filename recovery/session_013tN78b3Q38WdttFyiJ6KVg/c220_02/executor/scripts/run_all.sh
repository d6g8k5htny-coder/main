#!/bin/bash
# Runs the five C220-02 scenarios one stack at a time, then extracts speech per scenario.
C=/tmp/claude-0/-home-user-main/d5c985ca-97e5-58d8-a9d4-1e0a19065a22/scratchpad/c220
B=http://127.0.0.1:8765/museum.html
SW="--use-angle=swiftshader --enable-unsafe-swiftshader"
for spec in "S1|$B?view=ec014|fail|$SW" "S2|$B?view=ec014|fulfill|--disable-webgl --disable-3d-apis" "S3|$B?view=ec014|fulfill|$SW" "S4|$B?view=p15|fulfill|$SW" "S5|$B?view=ec014|fulfill|$SW"; do
  IFS='|' read -r id url mode flags <<< "$spec"
  [ -n "$ONLY" ] && [[ " $ONLY " != *" $id "* ]] && continue
  echo "=== $id $(date -u +%H:%M:%S)"
  $C/scripts/run_scenario.sh "$id" "$url" "$mode" $flags
  python3 -I $C/scripts/extract.py "$C/evidence/$id" > /dev/null && echo "extracted $id"
done
echo "ALL DONE $(date -u +%H:%M:%S)"
