#!/usr/bin/env bash
# Sequential scratch compiles: wait until no other Lean process runs, then one lake env lean at a time.
set -u
export PATH=$HOME/.elan/bin:$PATH
S=<SCRATCH>
PLAN="$1"; OUTDIR="$2"
cd "$S/repos/math_c254/formal" || exit 99
wait_free() {
  local n=0
  while pgrep -f 'toolchains/[^ ]*/bin/lean ' >/dev/null 2>&1; do
    sleep 3; n=$((n+1))
    if [ "$n" -gt 600 ]; then echo "WAIT_TIMEOUT"; return 1; fi
  done
}
while IFS=$'\t' read -r label flags path; do
  [ -z "$label" ] && continue
  wait_free || exit 98
  echo "START $label $(date -u +%T)"
  code=0
  # shellcheck disable=SC2086
  lake env lean $flags "$path" > "$OUTDIR/$label.log" 2>&1 || code=$?
  printf '%s\n' "$code" > "$OUTDIR/$label.status"
  echo "END $label exit=$code bytes=$(wc -c < "$OUTDIR/$label.log") $(date -u +%T)"
done < "$PLAN"
echo "PLAN_DONE"
