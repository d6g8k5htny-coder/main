#!/usr/bin/env python3
"""Verify the stored QS A3-A3.6 records and replay their published author controls (standard library only).

Every file in this packet except replay.py, README.md and SOURCES.json is one of two things:
  - an exact copy of a public comment body on main#229 (a note, claim, controls comment, review, readback,
    erratum, successor text or author response);
  - a control script or its stdout, extracted verbatim from a stored controls comment by the extraction rule that
    comment states (the rule of main#229 comment 5971055189).
SOURCES.json pins each file by byte count and SHA-256 and lists each replay. This script performs four checks.

  1. The packet tree equals SOURCES.json's file list, with no symlinks, and every stored file has its pinned
     identity.
  2. Every extracted script and stdout equals the fenced payload inside its stored controls comment, under the
     extraction rule.
  3. Every control script runs under the current interpreter's -O/-S flags and reproduces its published stdout byte
     for byte (none prints an interpreter version), with empty stderr. Every listed mutant or invalid invocation
     must reproduce its exact source-bound rejection on BOTH captured streams, not merely exit 1 or 2.
  4. Negative controls must be rejected:
     - a one-byte change to a stored note (A3_6/PROOF.md);
     - a control script with one changed constant (A3.6's hard-gap integral coefficient 5/4 replaced by 1/4).

Reproducibility only. The analytic notes and their reads carry the mathematics. These finite checks support
algebra, as the reads say. Nothing here reads or grades a proof.

Usage: python3 -B -S frontiers/qs_d3_soft_layer_chain_20261005/replay.py   (exit 0 iff every check passes)
"""
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
OWN = {'README.md', 'SOURCES.json', 'replay.py'}

# Exact rejection fingerprints from the UNCHANGED source packet at b157df1b57da41d2d75d6515b5e795b234ee7d34.
# Each key is (script path, complete argv tail); each value is (exit, stdout SHA256,
# stderr SHA256, readable reason). References are frozen independently of this run.
# All 48 mutants and 14 invalid invocations were executed and their named reasons
# checked against their source. Different checkers intentionally use different
# streams/formats. Exact fingerprints enforce those full contracts, including
# JSON structure/types, labels, counts, usage text, and empty/nonempty streams.
# No current failing output, generic exit code or broad regex creates a reference.
# Any source/runtime output drift fails closed and requires a reviewed amendment.
REJECTION_CONTRACTS = {
    ('A3/author_controls.py', ('--mutant', 'Z2')):
        (1, 'e04b2610f0b94857f65265e92fc2ef03f5d26c90e9bc76c87b0e60c3fe73bbe0', 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'Z2-gain'),
    ('A3/author_controls.py', ('--mutant', 'Z3')):
        (1, '15364c942daeecd42584fc0cd28568908542950e96d8905c94a8fe387080d22f', 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'Z3-drop'),
    ('A3/author_controls.py', ('--mutant', 'Z4')):
        (1, '5425d23fdf3ae54af74001f1d56f06b4cf9942d4d9258f9b26f84fcc52b2cfb3', 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'Z4'),
    ('A3/author_controls.py', ('--bogus',)):
        (2, '034489f97f641cc1afed3ce1cbcd4e992ee86bd1ecb35605d27149c59e7f37c9', 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', '{"error": "unknown arguments"}'),
    ('A3_1/author_controls.py', ('--mutant', 'N1')):
        (1, 'bdae332b611455aadc8065ec8651cb09bb49f06a60eab73839052e6fcea5d757', 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'N1'),
    ('A3_1/author_controls.py', ('--mutant', 'F1')):
        (1, '8f637a031d7176b4859ff503e98c661c0ed11c344637d4b9f043b3d75ca9148e', 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'F1-linear'),
    ('A3_1/author_controls.py', ('--mutant', 'R1')):
        (1, '94f45519c4c9b5ed2f891377e7b4780e6fdacddf16544251b0a426baf2538d15', 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'R1'),
    ('A3_1/author_controls.py', ('--mutant', 'C1')):
        (1, 'ab8c4393467bd87b70446711b78122408a74cf8954c3e3dd16b3f33fefab95b9', 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'C1'),
    ('A3_1/author_controls.py', ('--mutant', 'H1')):
        (1, '15d71c1992177a1f8334ac678726414f9fe8bf50851f42f2b547e0935415aa20', 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'H1; H1-global'),
    ('A3_1/author_controls.py', ('--bogus',)):
        (2, '034489f97f641cc1afed3ce1cbcd4e992ee86bd1ecb35605d27149c59e7f37c9', 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', '{"error": "unknown arguments"}'),
    ('A3_2/author_controls.py', ('--mutant', 'W1')):
        (1, '62a9beb4f697c0f2d86151c91a93c8f1de3056ea2bb22c178b30c69672ddd1e3', 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'W1-det'),
    ('A3_2/author_controls.py', ('--mutant', 'W3')):
        (1, 'fbea0daa327d6d93570ef86cea6365cd3a6ff27130c5eae4307c9d54ecc83f9b', 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'W3-psi-c'),
    ('A3_2/author_controls.py', ('--mutant', 'W4')):
        (1, '5dd21eff08c0b14b4f413c743f7ba3ef8e44e85ae496550d3271f39df6fedff8', 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'W4-hard'),
    ('A3_2/author_controls.py', ('--mutant', 'W4r')):
        (1, '3d62fdc97b4db41e301926538b642d03131dd2a7097a731a4c53fb02023f6da9', 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'W4-extreme'),
    ('A3_2/author_controls.py', ('--bogus',)):
        (2, '034489f97f641cc1afed3ce1cbcd4e992ee86bd1ecb35605d27149c59e7f37c9', 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', '{"error": "unknown arguments"}'),
    ('A3_3/author_controls.py', ('--mutant', 'M1')):
        (1, 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', '1d40f29decb2fb7edc239f430921c1826267f40b1ea902ec4569d6b2526761e7', 'K1 BD(a) violated on the explicit instance (ratio 13225/6664)'),
    ('A3_3/author_controls.py', ('--mutant', 'M2')):
        (1, 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'ab5b2910401dd46dcd01f5b8217bb506cdcf47956c01b4785ae9f02ce3d05a49', 'K2 BD(b) violated: n1=1 j=2 lhs=1 |c|=1'),
    ('A3_3/author_controls.py', ('--mutant', 'M3')):
        (1, 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', '43a8d1f30540704886973e7bd777cafcbc0b8ca21e013f939eb0e026ecf2ff1d', 'K4 s^1 part is not the mixed block at entry (0,2)'),
    ('A3_3/author_controls.py', ('--mutant', 'M4')):
        (1, 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', '0f33e6d22f652eb410cf4dfc6328cd166fa3de09babbf28e127bb86a587e29f6', 'K5 W3 bound violated: r=1/100 lam2=1 ratio=1380122500/60162641'),
    ('A3_3/author_controls.py', ('--mutant', 'M5')):
        (1, 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', '3ae2328a110d8870008d319e30b02ed13bcd3602f027c5ba1c04bfa0f24a2645', 'K3 index additivity fails: (2,1) j=1'),
    ('A3_3/author_controls.py', ('--bogus',)):
        (2, 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', '727a88582d0e4605f5fafcfc233dc55111bde6af3c12aaf17e1a9f65f697cf4a', 'usage: a33_exact.py [--mutant M1|M2|M3|M4|M5]'),
    ('A3_3/addendum_control_k5o.py', ('author_controls.py', '--mutant', 'W1')):
        (1, '0f36d669163dbaf123b5b862b84258e6c03385e1b5f05cda49948a409d1c7ca3', '5ad81e75571085c09eed3eac956968afe62463ca2352b4f2e3d8580e7c20e7bb', 'O2: W_r/r^4; O2: pin Hessians'),
    ('A3_3/addendum_control_k5o.py', ('author_controls.py', '--mutant', 'W2')):
        (1, '0f36d669163dbaf123b5b862b84258e6c03385e1b5f05cda49948a409d1c7ca3', 'cba682aa71052ef54aec90f0f7571579b0dd273762c2cf231fcf0982b08ad5d9', 'O2: W_r/r^4; O2: not ordered with lambda2 < 0; O2: pin Hessians'),
    ('A3_3/addendum_control_k5o.py', ('author_controls.py', '--mutant', 'W3')):
        (2, 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', '26eac1668d0a95fd54a36c92aeff13007a4a145dc3a3667f6ec836bec0cc6cd1', 'usage: a33_k5o.py PATH/a33_exact.py [--mutant W1|W2]'),
    ('A3_3/addendum_control_k5o.py', ()):
        (2, 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', '26eac1668d0a95fd54a36c92aeff13007a4a145dc3a3667f6ec836bec0cc6cd1', 'usage: a33_k5o.py PATH/a33_exact.py [--mutant W1|W2]'),
    ('A3_4/author_controls.py', ('--mutant', 'N1')):
        (1, 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', '8acd4b89cd7fac49fea480c701c5d79c6291bdea9558e66955ed5996c8eb8a1d', 'S1 spectral Jacobian fails: mu1=0 mu2=14/11 tau=-2/13'),
    ('A3_4/author_controls.py', ('--mutant', 'N2')):
        (1, 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', '0d1941b2b1e8f1fcfd676bda8df6cf7bebb065c0b7dc94decc20fa6d516dfac6', 'S2 the monomial matrix at the lattice points is singular'),
    ('A3_4/author_controls.py', ('--mutant', 'N3')):
        (1, 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'b811a904b49d6d2ddc91698c011856f53acd1b41b97f36df308f8d8d5079cbab', 'S3 eigenframe jet formulas fail'),
    ('A3_4/author_controls.py', ('--mutant', 'N4')):
        (1, 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', '66faa6cea9d23e2908f8f8bbbf4eda251cf8eb1c949d1edf5e152cdc4d3d9da4', 'S4 a_M + a_S != 12 lt'),
    ('A3_4/author_controls.py', ('--mutant', 'N5')):
        (1, 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', '42645b1d864dde35dbed7465deea3ee17db948fd64e8dca32d882d89c2215258', 'S5 the strip integral is not of order delta^2: Lam=1/2 delta=1/2 ratio=397/144'),
    ('A3_4/author_controls.py', ('--mutant', 'N6')):
        (1, 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', '75e7a9077ee5beed85ef3eec5a6b998f08a5596525a95dbf7a60a2e0b3ed3d4c', 'S2 the 20 contact functionals are not the distinct multi-indices of order <= 3'),
    ('A3_4/author_controls.py', ('--bogus',)):
        (2, 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'ff172c1663e868f7cbd25928d384be3d46dd2f959d924bac2192c13f67b6aac2', 'usage: a34_exact.py [--mutant N1|N2|N3|N4|N5|N6]'),
    ('A3_4/author_controls.py', ('--mutant', 'N7')):
        (2, 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'ff172c1663e868f7cbd25928d384be3d46dd2f959d924bac2192c13f67b6aac2', 'usage: a34_exact.py [--mutant N1|N2|N3|N4|N5|N6]'),
    ('A3_4/author_controls.py', ('--mutant',)):
        (2, 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'ff172c1663e868f7cbd25928d384be3d46dd2f959d924bac2192c13f67b6aac2', 'usage: a34_exact.py [--mutant N1|N2|N3|N4|N5|N6]'),
    ('A3_5/author_controls.py', ('--mutant', 'M1')):
        (1, '2c6ab7d931ca3a040117e5359616b63defd5ff7e3e95b2336fcc881c400962dc', 'fb7bdb03d16f56f9d947d8380e3d2416c56cb23f5f84a7cd52fbca5f3d732e6a', 'F2_FW2'),
    ('A3_5/author_controls.py', ('--mutant', 'M2')):
        (1, '44e0e82e0d69a2d424b1ca74a13dcc3647a1b0f3a4f6be5320ebe98eddce8bb7', '6c6ecda028a985c503ae735ff1520195c2efbec40573b4922117840e81594216', 'F2_FW1'),
    ('A3_5/author_controls.py', ('--mutant', 'M3')):
        (1, '925de071af2eeb04f9f62ccd73ca83b13c5bc31a3b8ff573b8d66b45115f3f9d', '6067508640a48e55fe45b1686c0a3053b4df05049beac6bb341729e3c2447cc9', 'D1_prop_D3'),
    ('A3_5/author_controls.py', ('--mutant', 'M4')):
        (1, '1a7ca14805737057103daa5c0045ee36459ddb40837b029fce6102d187b25115', 'cc4c2e10e6d1992b1f4c6245f225282029e3e8bcc8ce17e4933cb25d48d611e6', 'T1_frobenius'),
    ('A3_5/author_controls.py', ('--mutant', 'M5')):
        (1, '1a7ca14805737057103daa5c0045ee36459ddb40837b029fce6102d187b25115', 'f511c4c70a9c52e6f84d2a931907dd71927eb20e5aaf9972153b08b9c03334fe', 'G1a_inner_bound'),
    ('A3_5/author_controls.py', ('--mutant', 'M6')):
        (1, '1a7ca14805737057103daa5c0045ee36459ddb40837b029fce6102d187b25115', 'ccf2fe89fcfad2d6886fc309b99c3ff4017ed077ce144542222daabaaca4e29d', 'G2_exponents'),
    ('A3_5/author_controls.py', ('--mutant', 'M7')):
        (1, '1a7ca14805737057103daa5c0045ee36459ddb40837b029fce6102d187b25115', '36bd6fdab1e967a26ade00260cd111fdc9acaeb6ab6581eb66fbb01b8cdef927', 'D1_prop_D3'),
    ('A3_5/author_controls.py', ('--mutant', 'M8')):
        (1, 'c54b30aef280a5ee39d3979927d909cdb5d7a9598c458bdcfc8585f09d1a15cb', 'b5937f8dde1e7ff7e5df16e757977a71f845319df30d2ec3373db3ac75e2b753', 'F2_FW1'),
    ('A3_5/author_controls.py', ('--mutant', 'M9')):
        (1, '132abd915b426b3381bd0d45860ccbba3e73897b83b3fca26cdcac95ab4afafa', '42a8a5c652e734ed7ef37b73fb04b98f43c9a1d07b1980b565f3c60bffff5799', 'F2_FW2'),
    ('A3_5/author_controls.py', ('--mutant', 'M10')):
        (1, '9f82daa09adda5d24cf9c0adde4e22a35f2aedd710d7e82cb18aa7a1b3bc070e', 'c17e81d9ac6f1140c1e4a5464621ed1f38e89b318b33eb1cc4707f5b301d1f4b', 'F2_FW4'),
    ('A3_5/author_controls.py', ('--bogus',)):
        (2, 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', '65b18625f68d8aa7b28667b23ea338d3b162126d20ea6477cc662f9246ba5cfe', 'usage: a35_exact.py [--mutant M1|...|M10]'),
    ('A3_5/author_controls.py', ('--mutant', 'M11')):
        (2, 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', '65b18625f68d8aa7b28667b23ea338d3b162126d20ea6477cc662f9246ba5cfe', 'usage: a35_exact.py [--mutant M1|...|M10]'),
    ('A3_6/author_controls.py', ('--mutant', 'M1')):
        (1, 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'ac6b3ae6d76bd82dec6636b71bfcef44906c92a43b3a94f8a2800cb783b6a888', 'FAILED: X1_chart, X12_witness, X16_endpoint_chain'),
    ('A3_6/author_controls.py', ('--mutant', 'M2')):
        (1, 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', '31dae24b868f836f9a9f4ddac102788d320a57594cde796c2af692a2acf7b26c', 'FAILED: X2_jacobian'),
    ('A3_6/author_controls.py', ('--mutant', 'M3')):
        (1, 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', '137f71e7a1be90f405a9becb4d5b4ec0e67db661e93c781392096831a89c1470', 'FAILED: X5_elder_floor'),
    ('A3_6/author_controls.py', ('--mutant', 'M4')):
        (1, 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', '12cbede6726c438a06b28c9db6b4fbee2099c74ae6d6d7bece81770c3b6fd717', 'FAILED: X7_radius'),
    ('A3_6/author_controls.py', ('--mutant', 'M5')):
        (1, 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', '754f0674ace72d8f126243f08ec5c4069efe11ea751ee585269ec9add9198dfe', 'FAILED: X9_hardgap'),
    ('A3_6/author_controls.py', ('--mutant', 'M6')):
        (1, 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'd090a99859fbe57008b1a9fdc4952fc410da58163a6151ec2ab27840ddf48301', 'FAILED: X8_levelband'),
    ('A3_6/author_controls.py', ('--mutant', 'M7')):
        (1, 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'b9d4a83f3b0f80aaa6f4cd9c1a3d6fae07dd7cfbc96f36a16ec83e97f4cc158f', 'FAILED: X11_ledgers'),
    ('A3_6/author_controls.py', ('--mutant', 'M8')):
        (1, 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', '44b7327fd1d6686f1577e04be5e4df2f6b54e7b6ab7eab5fa93b4324a70cde28', 'FAILED: X13_chord'),
    ('A3_6/author_controls.py', ('--mutant', 'M9')):
        (1, 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'db57033261a452390cc359f24e523fe3b61834241b5cb0aad4132a3a4eb571c5', 'FAILED: X6_endpoint_floor'),
    ('A3_6/author_controls.py', ('--mutant', 'M10')):
        (1, 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', '7d637e942a5e2ca81f824a5a0adfb7c16983d563a5b405b0030cf8d059a7e058', 'FAILED: X14_coverage'),
    ('A3_6/author_controls.py', ('--mutant', 'M11')):
        (1, 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', '42b5323bf1b7a86bde8aff93beb03692b6ab8bb3ac499dd9a08b8dca6f3268ba', 'FAILED: X15_G3minus'),
    ('A3_6/author_controls.py', ('--mutant', 'M12')):
        (1, 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', '8865b9662578908c6efbd521ebc7cf81a95f437ae4c3787908f9ead73a15345c', 'FAILED: X17_inclusions'),
    ('A3_6/author_controls.py', ('--mutant', 'M13')):
        (1, 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', '678e7e25a484af0176e0cdf3481bc96d72710fcb92680662e4740f79a6952dca', 'FAILED: X13_chord'),
    ('A3_6/author_controls.py', ('--bogus',)):
        (2, 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', '230b5ecba1d2582ee0d266ad94e94441c000ff845246535c4c88289d8c454610', 'usage: a36_exact.py [--mutant M1..M13]'),
    ('A3_6/author_controls.py', ('--mutant', 'M14')):
        (2, 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', '230b5ecba1d2582ee0d266ad94e94441c000ff845246535c4c88289d8c454610', 'usage: a36_exact.py [--mutant M1..M13]'),
    ('A3_6/author_controls.py', ('--mutant',)):
        (2, 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', '230b5ecba1d2582ee0d266ad94e94441c000ff845246535c4c88289d8c454610', 'usage: a36_exact.py [--mutant M1..M13]'),
}

# Replacing A3.6's 5/4 by 1/4 with MUT=None omits the usual mutant-description line.
# This separately executed altered-source control must fail specifically at X9.
NEGATIVE_HARDGAP = (1, hashlib.sha256(b'').hexdigest(),
                    hashlib.sha256(b'FAILED: X9_hardgap\n').hexdigest(), 'X9_hardgap')


def check_rejection(result, expected):
    """Compare both full captured streams, not a permissive failure-message pattern."""
    code, out, err = result
    failures = []
    if type(code) is not int or code != expected[0]:
        failures.append('wrong exit code')
    if type(out) is not bytes or hashlib.sha256(out).hexdigest() != expected[1]:
        failures.append('stdout differs from intended diagnostic')
    if type(err) is not bytes or hashlib.sha256(err).hexdigest() != expected[2]:
        failures.append('stderr differs from intended diagnostic')
    return failures


def rejection_inventory_failures(manifest):
    listed = {}
    for obj in manifest['objects']:
        for replay in obj['replays']:
            for kind, code in (('mutants', 1), ('invalid', 2)):
                for args in replay[kind]:
                    key = replay['script'], tuple(args)
                    if key in listed:
                        return ['duplicate rejection invocation: ' + repr(key)]
                    listed[key] = code
    if set(listed) != set(REJECTION_CONTRACTS):
        return ['rejection invocation inventory differs from frozen contracts']
    if any(REJECTION_CONTRACTS[key][0] != code for key, code in listed.items()):
        return ['rejection invocation category differs from frozen contracts']
    return []


def report_control(script, args, result, reason):
    code, out, err = result
    print(json.dumps(dict(script=script, args=list(args), returncode=code,
                          stdout_sha256=hashlib.sha256(out).hexdigest(),
                          stderr_sha256=hashlib.sha256(err).hexdigest(),
                          expected_reason=reason), sort_keys=True))


def identity(data):
    return len(data), hashlib.sha256(data).hexdigest()


def interpreter_flags():
    flags = ['-B']
    if sys.flags.optimize:
        flags.append('-O')
    if sys.flags.no_site:
        flags.append('-S')
    return flags


def tree_failures(root, manifest):
    failures = []
    listed = {}
    for obj in manifest['objects']:
        for f in obj['stored']:
            listed[f['path']] = (f['bytes'], f['sha256'])
    present = set()
    for path in sorted(root.rglob('*')):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            failures.append('symlink: ' + rel)
        elif path.is_file():
            present.add(rel)
    for rel in sorted(present ^ (set(listed) | OWN)):
        failures.append(('unlisted file: ' if rel in present else 'missing file: ') + rel)
    for rel, ident in sorted(listed.items()):
        if rel in present and identity((root / rel).read_bytes()) != ident:
            failures.append('identity mismatch: ' + rel)
    return failures


def extraction_failures(root, manifest):
    """Re-extract every script and stdout from its stored controls comment and compare with the stored payloads."""
    failures = []
    for obj in manifest['objects']:
        extracted = [f for f in obj['stored'] if 'extracted_from' in f]
        for script in [f for f in extracted if f['path'].endswith('.py')]:
            stdout = [f for f in extracted if f['extracted_from'] == script['extracted_from']
                      and f['heading'] == script['heading'] and f['path'].endswith('.json')]
            if len(stdout) != 1:
                failures.append('%s: no single stdout record for %s' % (obj['tag'], script['path']))
                continue
            stdout = stdout[0]
            body = (root / script['extracted_from']).read_text(encoding='utf-8')
            if body.count(script['heading']) < 1:
                failures.append('%s: heading %r not found in %s' % (obj['tag'], script['heading'], script['extracted_from']))
                continue
            section = body[body.index(script['heading']):]
            m1 = re.search(r'```python\n(.*?)\n```\n', section, re.S)
            m2 = re.search(r'```json\n(.*?)\n```\n', section[m1.end():], re.S) if m1 else None
            if not (m1 and m2):
                failures.append('%s: fences not found after %r' % (obj['tag'], script['heading']))
                continue
            if (m1.group(1) + '\n').encode('utf-8') != (root / script['path']).read_bytes():
                failures.append('%s: %s differs from the payload in %s' % (obj['tag'], script['path'], script['extracted_from']))
            if (m2.group(1) + '\n').encode('utf-8') != (root / stdout['path']).read_bytes():
                failures.append('%s: %s differs from the payload in %s' % (obj['tag'], stdout['path'], stdout['extracted_from']))
    return failures


def run(script, *args):
    proc = subprocess.run([sys.executable, *interpreter_flags(), str(script), *args], capture_output=True,
                          timeout=1800, cwd=str(script.parent))
    return proc.returncode, proc.stdout, proc.stderr


def replay_failures(root, manifest):
    failures = rejection_inventory_failures(manifest)
    if failures:
        return failures  # Do not execute an incomplete/changed control inventory.
    for obj in manifest['objects']:
        for replay in obj['replays']:
            label = '%s %s' % (obj['tag'], replay['name'])
            script = root / replay['script']
            code, out, err = run(script, *replay['args'])
            baseline_failures = []
            if code != 0:
                baseline_failures.append('%s: exit code %s' % (label, code))
            if out != (root / replay['stdout']).read_bytes():
                baseline_failures.append('%s: stdout differs from the published bytes' % label)
            if err:
                baseline_failures.append('%s: unexpected baseline stderr' % label)
            if baseline_failures:
                failures.extend(baseline_failures)
                continue
            report_control(replay['script'], replay['args'], (code, out, err), 'exact published baseline')
            for kind in ('mutants', 'invalid'):
                for args in replay[kind]:
                    expected = REJECTION_CONTRACTS[replay['script'], tuple(args)]
                    result = run(script, *args)
                    mismatches = check_rejection(result, expected)
                    failures.extend('%s %r: %s' % (label, args, why) for why in mismatches)
                    if not mismatches:
                        report_control(replay['script'], args, result, expected[3])
    return failures


def negative_control_failures(root, manifest):
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        copy = pathlib.Path(tmp) / 'packet'
        shutil.copytree(root, copy, symlinks=True)
        target = copy / 'A3_6' / 'PROOF.md'
        data = bytearray(target.read_bytes())
        data[100] ^= 0x01
        target.write_bytes(bytes(data))
        if 'identity mismatch: A3_6/PROOF.md' not in tree_failures(copy, manifest):
            failures.append('negative control: a one-byte change to A3_6/PROOF.md was not detected')
        script = copy / 'A3_6' / 'author_controls.py'
        text = script.read_text(encoding='utf-8')
        old = 'coef = F(1, 4) if MUT == "M5" else F(5, 4)'
        if text.count(old) != 1:
            failures.append('negative control: the hard-gap coefficient line was not found exactly once')
        else:
            script.write_text(text.replace(old, 'coef = F(1, 4) if MUT == "M5" else F(1, 4)'), encoding='utf-8')
            result = run(script)
            mismatches = check_rejection(result, NEGATIVE_HARDGAP)
            failures.extend('negative control: changed hard-gap constant: ' + why for why in mismatches)
            if not mismatches:
                report_control('A3_6/author_controls.py', [], result, 'changed 5/4 to 1/4: X9_hardgap')
    return failures


def main():
    manifest = json.loads((HERE / 'SOURCES.json').read_text(encoding='utf-8'))
    failures = tree_failures(HERE, manifest)
    if not failures:
        failures = extraction_failures(HERE, manifest)
    if not failures:
        failures = replay_failures(HERE, manifest)
    if not failures:
        failures = negative_control_failures(HERE, manifest)
    n_scripts = sum(len(o['replays']) for o in manifest['objects'])
    n_mut = sum(len(r['mutants']) for o in manifest['objects'] for r in o['replays'])
    n_inv = sum(len(r['invalid']) for o in manifest['objects'] for r in o['replays'])
    n_files = sum(len(o['stored']) for o in manifest['objects'])
    mode = ' '.join(interpreter_flags())
    if failures:
        for f in failures:
            print('FAIL ' + f)
        print('QS d = 3 chain (A3-A3.6) incorporation replay (%s): %d failure(s)' % (mode, len(failures)))
        return 1
    print('QS d = 3 chain (A3-A3.6) incorporation replay (%s): %d stored identities, %d extractions, %d checker stdouts, '
          '%d mutants, %d invalid invocations and 2 negative controls PASS' % (mode, n_files, 2 * n_scripts, n_scripts, n_mut, n_inv))
    return 0


if __name__ == '__main__':
    sys.exit(main())
