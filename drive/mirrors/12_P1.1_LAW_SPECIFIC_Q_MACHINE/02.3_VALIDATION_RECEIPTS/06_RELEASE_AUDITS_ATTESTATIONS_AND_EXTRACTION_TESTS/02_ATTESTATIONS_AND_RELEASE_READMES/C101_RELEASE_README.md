# Q0-C101 Qualitative Rate Release — Reproduction Guide

## Theorem boundary

This release establishes, at internal program/verification grade, that for the
exact normalized periodized Bargmann–Fock field on \(\mathbb T_{24}^2\), at
\(b=6/5\), under the typed maximum–saddle pair-Palm law, there exists a finite
constant \(C_{Q0}\) such that

\[
0\le1-q(r,6/5)\le C_{Q0}r^3,
\qquad
0<r\le0.025,
\]

and therefore

\[
q(r,6/5)\to1.
\]

No numerical value of \(C_{Q0}\) is claimed.

## Principal roots

```text
Q0_CUBIC_RATE_EXISTENCE_C101
f92e4aae4c5b6cf7a3a34c41a43642f14f129543d03eceb82d73a4e77f49100f

Q0_LIMIT_C101
5ed20eafac7cffa923518fce06cae56822c0a2a62b2d8061e5cd90163ca47996
```

## Deterministic proof instruments

```bash
python3 C099_CUBIC_TYPE_NOGO.py
python3 C099_SCALED_FRAME_CERTIFICATE.py
python3 C100_COLLAR_SCALED_FRAME.py
python3 C101_WINDOW_SADDLE_FRAME.py
python3 C101_INDEPENDENT_CHECK.py
python3 gate_kernel_v2_0.py
python3 validate_gate_kernel_v2_0.py
```

## Frozen diagnostics

```bash
python3 C098_GAMMA_MAXCOUNT_BENCHMARK.py
python3 C099_NEAR_ENDPOINT_SCAN.py
python3 C100_COLLAR_PROFILE_SCAN.py
python3 C101_WINDOW_SADDLE_SCAN.py
```

The diagnostic scripts use fixed seeds and exact high-precision covariance
assembly. They are not theorem-grade dependencies. The release deep audit
re-executes the compact C098 anchor and validates the hashes/schema/invariants
of the larger C099--C101 diagnostic outputs. The three larger scans are
optional full reproductions because they are frozen falsification instruments,
not inputs to the theorem grade.

## Immutable-root protocol

The release ZIP contains a self-excluded manifest.

1. Extract into a new empty immutable directory.
2. Run `verify_q0_c101_release.py`.
3. Copy the directory to a disposable execution directory.
4. Run `audit_q0_c101_release.py` in the disposable copy.
5. Re-run the verifier in the untouched immutable directory.

Writer-based checks must never execute in the immutable verification root.

## Not claimed

- No \(4.35\), \(4.3\), or other numerical upper coefficient.
- No finite numerical lower coefficient.
- No sharpened C089 upper or lower theorem.
- No thermodynamic-limit or critical-height theorem.
- No external specialist acceptance of SARD-G.
- No publication-grade interval certificate for the existence constants.
