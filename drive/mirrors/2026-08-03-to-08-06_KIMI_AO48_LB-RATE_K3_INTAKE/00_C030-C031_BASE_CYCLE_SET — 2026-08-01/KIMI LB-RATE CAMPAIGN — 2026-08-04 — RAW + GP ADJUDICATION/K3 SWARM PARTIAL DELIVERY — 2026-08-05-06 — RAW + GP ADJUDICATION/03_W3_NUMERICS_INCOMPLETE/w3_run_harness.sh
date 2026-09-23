#!/bin/sh
# W3 harness runner: normal + -O byte-identity + 3 mutation tests, with receipts.
cd /mnt/agents/output/K3_SIDE24_LB/W3_numerics
R=receipts
mkdir -p $R
run() {
    name="$1"; shift
    "$@" > "$R/$name.out" 2> "$R/$name.err"
    echo "$?" > "$R/$name.exit"
    sha256sum "$R/$name.out" | awk '{print $1}' > "$R/$name.out.sha256"
}
run harness_normal python3 w3_harness.py
run harness_O python3 -O w3_harness.py
run harness_mut_sqrt env W3_MUTATE=sqrt python3 w3_harness.py
run harness_mut_pin env W3_MUTATE=pin python3 w3_harness.py
run harness_mut_zone env W3_MUTATE=zone python3 w3_harness.py
# byte-identity check
if cmp -s "$R/harness_normal.out" "$R/harness_O.out"; then
    echo "BYTEIDENT PASS" > "$R/byteident.txt"
else
    echo "BYTEIDENT FAIL" > "$R/byteident.txt"
fi
# mutation fail-closed check: each mutation must exit nonzero
ok=1
for m in sqrt pin zone; do
    ec=$(cat "$R/harness_mut_$m.exit")
    if [ "$ec" = "0" ]; then ok=0; fi
done
n=$(cat "$R/harness_normal.exit")
if [ "$n" != "0" ]; then ok=0; fi
o=$(cat "$R/harness_O.exit")
if [ "$o" != "0" ]; then ok=0; fi
if [ "$ok" = "1" ]; then
    echo "FAILCLOSED PASS" > "$R/failclosed.txt"
else
    echo "FAILCLOSED FAIL" > "$R/failclosed.txt"
fi
echo DONE
