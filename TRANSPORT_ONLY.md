# Isolated binary-transport workspace

This branch is a one-off operational transport aid for the R1 register handoff. It is not a research branch, governing source, proof, review, or proposal to merge into main. The default-branch ancestor's historical content is not used as evidence.

The first job has read-only repository permissions, checks out an exact research commit, validates the existing Sept 18 workbook's byte count, digest and ZIP CRC, and exposes that already-public file through the supported Actions artifact download interface. No Drive credentials or temporary bearer URLs are published. No source or research branch is changed by this job.

Owner authorization: OP-AUTONOMY-20260923-v1.0, already distributed and merged through PR25. The goal is to solve a tool binary-transfer mismatch rather than leave the register import blocked. No mathematical status or independence credit changes. Preserve the original export and distinguish exact export bytes from the native Sheet representation.
