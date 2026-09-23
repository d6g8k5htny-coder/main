#!/usr/bin/env bash
set -euo pipefail

out="SIDE24_TECHNICAL_APPENDIX_EXACT.md"
tmp_root="$(mktemp -d)"
trap 'rm -rf "$tmp_root"' EXIT

extract_range() {
  local source="$1"
  local start="$2"
  local stop="$3"
  local target="$4"
  awk -v start="$start" -v stop="$stop" '
    {sub(/\r$/, "")}
    $0 == start {take=1}
    $0 == stop {take=0}
    take {print}
  ' "$source" > "$target"
}

append_module() {
  local label="$1"
  local title="$2"
  local source="$3"
  local start="$4"
  local stop="$5"
  local target="$tmp_root/$label.txt"
  extract_range "$source" "$start" "$stop" "$target"
  local bytes
  local digest
  bytes="$(wc -c < "$target" | tr -d ' ')"
  digest="$(sha256sum "$target" | cut -d' ' -f1)"
  printf '\n---\n\n## Module %s — %s\n\n' "$label" "$title" >> "$out"
  printf '*Exact extracted text: %s bytes; SHA-256 `%s`.*\n\n' "$bytes" "$digest" >> "$out"
  printf '```text\n' >> "$out"
  sed 's/```/` ` `/g' "$target" >> "$out"
  printf '```\n' >> "$out"
}

append_module \
  "A" \
  "Corrected pins, dimension cancellation, and lifetime pushforward" \
  "ls_der_046.txt" \
  "2. ABSTRACT LOCAL SETUP" \
  "7. CORRECTION OF THE OLD EXPONENT TRIANGLE"

append_module \
  "B" \
  "Planar three-dimensional collar and singular-near selection charts" \
  "ls_der_053.txt" \
  "2. LIMITING PAIR PIN LAW" \
  "8. PERIODIZED AND THERMODYNAMIC CONSEQUENCES"

append_module \
  "C" \
  "Exact finite-side coefficient reduction" \
  "ls_der_056.txt" \
  "2. DIRECTIONAL CONTACT FRAME" \
  "11. EXECUTION PROTOCOL"

append_module \
  "D" \
  "Coefficient-map Hessian bounds and side-24 analytic remainder" \
  "ls_der_064.txt" \
  "2. COEFFICIENT MAP" \
  "9. SCOPE"

append_module \
  "E" \
  "Structural side-24 transfer of the local charts" \
  "ls_der_065.txt" \
  "3. PAIR PIN STABILITY" \
  "7. REMAINING INTERFACES AND MARKS"

append_module \
  "F" \
  "Full-spectrum Gram positivity and the corrected pair frame" \
  "ls_der_067.txt" \
  "2. SPECTRAL GRAM IDENTITY" \
  "5. UNIFORM FIXED-DISTANCE KAC--RICE CONTROL"

append_module \
  "G" \
  "Exact nearest-image jet remainder" \
  "ls_der_068.txt" \
  "2. ONE-DIMENSIONAL PERIODIZED MOMENTS" \
  "7. STATUS"

append_module \
  "H" \
  "Current corrections, exact first variation, and fixed-field interface ledger" \
  "ls_der_071.txt" \
  "2. TWO NONSTRUCTURAL CORRECTIONS" \
  "8. REVIEW FIREWALL"

append_module \
  "I" \
  "Palm-normalized witness counts and selection recombination" \
  "ls_der_072.txt" \
  "1. PURPOSE AND REVIEW SCOPE" \
  "7. MEASURABILITY AND SCOPE"

append_module \
  "J" \
  "Controlling energy-adapted matrix capture and escape repair" \
  "ls_der_073_v1_3.txt" \
  "1. PURPOSE AND REVIEW SCOPE" \
  "10. SIDE-24 CONSEQUENCE AND SCOPE FIREWALL"

append_module \
  "K" \
  "Exact absolute-normalization factorization" \
  "ls_der_075.txt" \
  "2. CUBIC-GAP CHANGE OF VARIABLES" \
  "7. SCOPE AND STATUS"

append_module \
  "L" \
  "Ordered-pair Jacobian, directed sphere, and H0 unit multiplicity" \
  "ls_der_076.txt" \
  "2. ORDERED TYPED PAIR SPACE" \
  "7. REVIEWED-SCOPE CONSEQUENCE"
