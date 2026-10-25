#!/bin/bash
# Supplementary variants (same driver): X = keys sent as real X events via AT-SPI/XTest so Orca sees them;
# F = the site's own exhibit link form with #active-exhibit (Orca skips Say All on a fragment).
C=/tmp/claude-0/-home-user-main/d5c985ca-97e5-58d8-a9d4-1e0a19065a22/scratchpad/c220
B=http://127.0.0.1:8765/museum.html
SW="--use-angle=swiftshader --enable-unsafe-swiftshader"
for spec in "S3X|$B?view=ec014|fulfill|xtest" "S4X|$B?view=p15|fulfill|xtest" "S3F|$B?view=ec014#active-exhibit|fulfill|cdp" "S4F|$B?view=p15#active-exhibit|fulfill|cdp"; do
  IFS='|' read -r id url mode keymode <<< "$spec"
  [ -n "$ONLY" ] && [[ " $ONLY " != *" $id "* ]] && continue
  echo "=== $id $(date -u +%H:%M:%S)"
  C220_KEYMODE=$keymode $C/scripts/run_scenario.sh "$id" "$url" "$mode" $SW
  python3 -I $C/scripts/extract.py "$C/evidence/$id" > /dev/null && echo "extracted $id"
done
echo "VARIANTS DONE $(date -u +%H:%M:%S)"
